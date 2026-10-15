"""VektorFIL lumped filter designer: interactive window with live updates.

Run:  python -m vektorfil.gui.filter_window
"""
from __future__ import annotations

import datetime
import math
import re
import sys
import traceback
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QFontDatabase, QGuiApplication, QIcon
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from vektorfil import __version__, app_paths
from vektorfil.analysis import timedomain
from vektorfil.filters import (SERIES_NAMES, SHUNT_FIRST, TYPES, FilterSpec, auto_freqs, canonical_subtype,
                              design, estimate_orders, project, resonator_unit, subtypes)
from vektorfil.filters.design import SINGLE, format_mask, parse_mask
from vektorfil.gui.montecarlo_tab import MonteCarloTab
from vektorfil.gui.plotstyle import C_EXTRA, C_IDEAL, C_S11, C_S21, faded, new_plot
from vektorfil.gui.schematic import SchematicView, eng
from vektorfil.gui.smith import SmithChart

APPROXIMATIONS = ("Butterworth", "Chebyshev", "Bessel", "Elliptic")
SINGLE_OK = ("Butterworth", "Chebyshev", "Elliptic")      # approximations with a singly terminated form
ELLIPTIC_BY = ("Amin", "Fs", "Mask")                      # FilterSpec.elliptic_spec per list entry
MAX_MARKERS = 6

GITHUB_URL = "https://github.com/ibelinp/vektorfil"

SUFFIX = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "K": 1e3, "m": 1e-3, "u": 1e-6, "µ": 1e-6,
          "n": 1e-9, "p": 1e-12, "f": 1e-15}


def mono_font(size: float) -> QFont:
    """The system's fixed-width font (Consolas, Menlo, DejaVu Sans Mono ...) at size pt."""
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    f.setPointSizeF(size)
    return f


def parse_value(text: str, unit_mult: float = 1.0):
    """'70', '70M', '1.5G', '2.2k', '1e8' -> float.  A bare number is in the field's unit."""
    t = text.strip().replace(",", ".")
    if not t:
        return None
    m = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([TGMkKmuµnpf]?)\s*(?:Hz|hz|Ω|ohm|dB)?", t)
    if not m:
        raise ValueError(f"cannot read '{text}'")
    v = float(m.group(1))
    return v * SUFFIX[m.group(2)] if m.group(2) else v * unit_mult


class EngSpinBox(QAbstractSpinBox):
    """Value box with up/down arrows that still accepts typed engineering values.

    Each step multiplies the value by `ratio` (e.g. 1.02 = 2 % per click), which suits
    inductors, capacitors and frequencies alike; Ctrl+arrow or PageUp/PageDown takes ten
    steps.  The text is in the field's unit (`unit_mult`, e.g. 1e-9 for nH), but suffixes
    such as '27n' or '1.5G' are understood.  An empty box means "automatic"; the first
    arrow click then starts from `auto_hint`, the value the last design actually used.
    """
    textChanged = Signal(str)

    def __init__(self, text="", unit_mult=1.0, ratio=1.01, parent=None):
        super().__init__(parent)
        self.unit_mult = unit_mult
        self.ratio = ratio
        self.auto_hint = None
        self.limits = None          # (lo, hi) in SI units: arrows stop at the ends
        self.lineEdit().setText(text)
        self.lineEdit().textChanged.connect(self.textChanged)
        self.setAccelerated(True)

    # QLineEdit-like convenience API, so the box is a drop-in replacement
    def text(self):
        return self.lineEdit().text()

    def setText(self, t):
        self.lineEdit().setText(t)

    def setPlaceholderText(self, t):
        self.lineEdit().setPlaceholderText(t)

    def value(self):
        """Current value in SI units, or None when empty."""
        return parse_value(self.text(), self.unit_mult)

    def _current(self):
        try:
            v = self.value()
        except ValueError:
            return None
        return self.auto_hint if v is None else v

    def stepEnabled(self):
        """Qt hook: grey out an arrow at the end of the usable range."""
        flags = QAbstractSpinBox.StepUpEnabled | QAbstractSpinBox.StepDownEnabled
        v = self._current()
        if self.limits and v:
            lo, hi = self.limits
            if v >= hi * (1 - 1e-4):          # text is rounded to 5 digits
                flags &= ~QAbstractSpinBox.StepUpEnabled
            if v <= lo * (1 + 1e-4):
                flags &= ~QAbstractSpinBox.StepDownEnabled
        return flags

    def stepBy(self, steps):
        """Qt hook: multiply the value by ratio**steps, kept inside the limits."""
        v = self._current()
        if not v or v <= 0:
            return
        v *= self.ratio ** steps
        if self.limits:
            lo, hi = self.limits
            v = min(max(v, lo), hi)
        self.setText(f"{v / self.unit_mult:.5g}")


# --------------------------------------------------------------------------- plot cursor

class PlotCursor:
    """Hover cursor for one plot: a dashed vertical line, a dot where it crosses each
    curve (in the curve's colour), and the values in a small box beside the line."""
    def __init__(self, plot, fg, bg):
        self.vb = plot.getPlotItem().vb
        self.line = pg.InfiniteLine(angle=90, movable=False,
                                    pen=pg.mkPen(fg, width=0.8, style=Qt.DashLine))
        self.dots = pg.ScatterPlotItem(pxMode=True)
        box = pg.mkColor(bg)
        box.setAlpha(215)
        edge = pg.mkColor(fg)
        edge.setAlpha(90)
        self.label = pg.TextItem(color=fg, fill=pg.mkBrush(box), border=pg.mkPen(edge))
        self.dots.setZValue(20)
        self.label.setZValue(30)
        for item in (self.line, self.dots, self.label):
            plot.addItem(item, ignoreBounds=True)
        self.fg = fg
        self.hide()

    def hide(self):
        for item in (self.line, self.dots, self.label):
            item.setVisible(False)

    def contains(self, pos):
        return self.vb.sceneBoundingRect().contains(pos)

    def x_at(self, pos):
        return self.vb.mapSceneToView(pos).x()

    def show(self, x, text, points):
        """points: (y, colour) pairs; values outside the view (or -inf at a notch) get no dot."""
        (x0, x1), (y0, y1) = self.vb.viewRange()
        spots = [{"pos": (x, y), "size": 9, "symbol": "o", "brush": pg.mkBrush(c), "pen": pg.mkPen(self.fg, width=1)}
                 for y, c in points if np.isfinite(y) and y0 <= y <= y1]
        self.dots.setData(spots=spots)
        self.line.setPos(x)
        self.label.setText(text)
        # beside the line near the top; on the left side when close to the right edge
        right = x < x0 + 0.72 * (x1 - x0)
        dx = 0.012 * (x1 - x0)
        self.label.setAnchor((0, 0) if right else (1, 0))
        self.label.setPos(x + dx if right else x - dx, y1 - 0.03 * (y1 - y0))
        for item in (self.line, self.dots, self.label):
            item.setVisible(True)


# --------------------------------------------------------------------------- worker

class _Signals(QObject):
    """Signals of a background job: done(generation, result) or failed(generation, message)."""
    done = Signal(int, object)
    failed = Signal(int, str)


class DesignJob(QRunnable):
    """Runs design(spec) in the worker thread and reports through _Signals; gen tells a
    current request from a stale one."""
    def __init__(self, gen, spec):
        super().__init__()
        self.gen, self.spec, self.signals = gen, spec, _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.gen, design(self.spec))
        except Exception as e:  # report every failure in the status bar
            msg = str(e) or e.__class__.__name__
            if not isinstance(e, (ValueError, ArithmeticError)):
                msg += "\n" + traceback.format_exc(limit=3)
            self.signals.failed.emit(self.gen, msg)


# --------------------------------------------------------------------------- plot range

class PlotRangeDialog(QDialog):
    """View -> Plot range: automatic or fixed sweep (also used by Export S2P)."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Plot range")
        lay = QFormLayout(self)
        self.chk_auto = QCheckBox("Automatic (follows the design)")
        self.chk_auto.setChecked(True)
        self.e_f1 = QLineEdit("0")
        self.e_f2 = QLineEdit("300")
        self.s_pts = QSpinBox()
        self.s_pts.setRange(51, 20001)
        self.s_pts.setValue(1001)
        lay.addRow(self.chk_auto)
        lay.addRow("Start (MHz)", self.e_f1)
        lay.addRow("Stop (MHz)", self.e_f2)
        lay.addRow("Points", self.s_pts)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.hide)
        lay.addRow(bb)
        for e in (self.e_f1, self.e_f2):
            # typing a range by hand switches the automatic range off
            e.textEdited.connect(lambda *_: self.chk_auto.setChecked(False))


# --------------------------------------------------------------------------- main window

class FilterWindow(QMainWindow):
    """Main window: the specification, step by step, on the left; plots over the
    schematic on the right.

    Every edit restarts a short timer; when it fires, the spec is read and designed in a
    worker thread (DesignJob).  Results carry a generation number so a stale result from
    an older edit is ignored.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VektorFIL — L-C filter synthesis")
        self.resize(1320, 820)
        # one design at a time: the high-precision math (mpmath) is not thread-safe, and
        # holding an arrow down would otherwise start overlapping jobs
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.gen = 0
        self.result = None
        self.path = None              # design file shown in the title bar
        self.remember_last = False    # main() sets this: save the design on close for the next start
        self.timer = QTimer(self, singleShot=True, interval=120)
        self.timer.timeout.connect(self._launch)

        left = self._build_left()
        right = self._build_right()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 960])
        self.setCentralWidget(split)
        self._build_menu()
        self.statusBar().showMessage("Ready")
        self._sync_visibility()
        self.schedule()

    # ------------------------------------------------------------------ left panel
    def _heading(self, num, title):
        """Section heading of the specification panel: the step number in a round badge
        (highlight colour) followed by the title."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 10, 0, 2)
        row.setSpacing(8)
        badge = QLabel(str(num))
        badge.setFixedSize(20, 20)
        badge.setAlignment(Qt.AlignCenter)
        hl = self.palette().highlight().color().name()
        badge.setStyleSheet(f"QLabel {{ background: {hl}; color: white; border-radius: 10px; "
                            f"font-weight: bold; font-size: 9pt; }}")
        text = QLabel(f"<b>{title}</b>")
        row.addWidget(badge)
        row.addWidget(text)
        row.addStretch(1)
        return box

    def _build_left(self):
        """Left panel: the specification in five steps (response, approximation,
        circuit, specification, components), then the result summary."""
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setSpacing(4)

        # 1  Response: a row of toggle buttons
        col.addWidget(self._heading(1, "Response"))
        row = QHBoxLayout()
        row.setSpacing(2)
        self.type_group = QButtonGroup(self)
        for i, t in enumerate(TYPES):
            b = QPushButton(t)
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.type_group.addButton(b, i)
            row.addWidget(b)
        col.addLayout(row)

        # 2  Approximation
        col.addWidget(self._heading(2, "Approximation"))
        f2 = QFormLayout()
        self.c_approx = QComboBox()
        self.c_approx.addItems(APPROXIMATIONS)
        self.c_approx.setCurrentText("Chebyshev")
        f2.addRow(self.c_approx)
        self.chk_single = QCheckBox("Singly terminated (far end open or shorted)")
        self.chk_single.setToolTip("The building block of diplexers and multiplexers: the source "
                                   "resistance on one side only.")
        f2.addRow(self.chk_single)
        self.c_even = QComboBox()
        self.c_even.addItems(["Type C: equal terminations", "Type B: unequal terminations"])
        self.l_even = QLabel("Even order")
        f2.addRow(self.l_even, self.c_even)
        col.addLayout(f2)

        # 3  Circuit
        col.addWidget(self._heading(3, "Circuit"))
        self.c_circuit = QComboBox()
        self.c_circuit.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.c_circuit.setMinimumContentsLength(20)
        self.sub_names = []
        self._rebuild_subtypes("Lowpass", "Chebyshev")
        col.addWidget(self.c_circuit)

        # 4  Specification
        col.addWidget(self._heading(4, "Specification"))
        form = QFormLayout()
        self.e_rs = QLineEdit("50")
        self.e_fc = EngSpinBox("100", 1e6, ratio=1.01)       # 1 % per click
        self.l_fc = QLabel("Cutoff frequency (MHz)")
        self.e_bw = EngSpinBox("10", 1e6, ratio=1.01)
        self.l_bw = QLabel("Bandwidth (MHz)")
        # shared resonator L (or C): 2 % per click shifts the skirt balance left/right
        self.e_res = EngSpinBox("", 1e-9, ratio=1.02)
        self.e_res.setPlaceholderText("auto")
        self.l_res = QLabel("Resonator L (nH)")
        self.s_order = QSpinBox()
        self.s_order.setRange(1, 25)
        self.s_order.setValue(5)
        # live order estimate: the order that reaches an attenuation at a frequency
        self.e_est_f = QLineEdit("150")
        self.e_est_a = QLineEdit("50")
        for e in (self.e_est_f, self.e_est_a):
            e.setMaximumWidth(56)
        est = QHBoxLayout()
        est.setContentsMargins(0, 0, 0, 0)
        est.addWidget(QLabel("for"))
        est.addWidget(self.e_est_a)
        est.addWidget(QLabel("dB at"))
        est.addWidget(self.e_est_f)
        est.addWidget(QLabel("MHz"))
        est.addStretch(1)
        self.l_est = QLabel()
        self.l_est.setTextFormat(Qt.RichText)
        self.l_est.setWordWrap(True)
        self.e_ripple = QDoubleSpinBox()
        self.e_ripple.setDecimals(4)
        self.e_ripple.setRange(0.0001, 3.0)
        self.e_ripple.setSingleStep(0.01)
        self.e_ripple.setValue(0.1)
        self.e_aa = QLineEdit("")
        self.e_aa.setPlaceholderText("default")
        self.c_espec = QComboBox()
        self.c_espec.addItems(["Minimum attenuation", "Stopband frequency", "Stepped mask"])
        self.e_amin = QDoubleSpinBox()
        self.e_amin.setDecimals(2)
        self.e_amin.setRange(1.0, 200.0)
        self.e_amin.setSingleStep(1.0)
        self.e_amin.setValue(60.0)
        self.e_fs = QLineEdit("150")
        # stepped stopband mask: start-stop:dB segments in MHz (the zeros are placed to meet it)
        self.e_mask = QLineEdit("125-200:40, 200-:60")
        self.e_mask.setToolTip("Segments start-stop:attenuation in MHz and dB, separated by commas; an empty "
                               "stop means 'and above' (e.g. 125-200:40, 200-:60). The notches are placed "
                               "so that every step has the same margin.")
        self.l_mask = QLabel("Mask (MHz:dB)")
        self.l_amin = QLabel("Stopband Amin (dB)")
        self.l_fs = QLabel("Stopband Fs (MHz)")
        self.l_espec = QLabel("Stopband set by")
        self.l_ripple = QLabel("Passband ripple (dB)")
        self.l_aa = QLabel("Edge attenuation (dB)")
        form.addRow("Impedance R (Ω)", self.e_rs)
        form.addRow(self.l_fc, self.e_fc)
        form.addRow(self.l_bw, self.e_bw)
        form.addRow("Order", self.s_order)
        form.addRow("Order needed", est)
        form.addRow(self.l_est)
        form.addRow(self.l_ripple, self.e_ripple)
        form.addRow(self.l_aa, self.e_aa)
        form.addRow(self.l_espec, self.c_espec)
        form.addRow(self.l_amin, self.e_amin)
        form.addRow(self.l_fs, self.e_fs)
        form.addRow(self.l_mask, self.e_mask)
        form.addRow(self.l_res, self.e_res)
        # coaxial resonators: line length at the centre frequency
        self.e_theta = QDoubleSpinBox()
        self.e_theta.setRange(5.0, 85.0)
        self.e_theta.setDecimals(1)
        self.e_theta.setSingleStep(1.0)
        self.e_theta.setValue(45.0)
        self.e_theta.setSuffix(" °")
        self.l_theta = QLabel("Line length at f0")
        form.addRow(self.l_theta, self.e_theta)
        self.chk_edges = QCheckBox("Exact band edges (re-tune coupled filter)")
        self.chk_edges.setChecked(True)
        self.chk_edges.setToolTip("Off: plain narrow-band formulas -- the edges then "
                                  "move when you change the resonator value.")
        form.addRow(self.chk_edges)
        # tubular / tumbular: a series coil (tumbular: capacitor) at each port lifts the
        # upper limit of the resonator inductor (lower limit of the tumbular capacitor)
        self.chk_coils = QCheckBox("End coils (extend the resonator range)")
        self.chk_coils.setToolTip("Adds a series element at each port once the resonator value goes past the "
                                  "point where the end capacitor to ground vanishes; below that point "
                                  "nothing changes.")
        form.addRow(self.chk_coils)
        col.addLayout(form)

        # 5  Components
        col.addWidget(self._heading(5, "Components"))
        f5 = QFormLayout()
        self.c_std = QComboBox()
        self.c_std.addItems(SERIES_NAMES)
        self.chk_retune = QCheckBox("Re-tune others while snapping")
        self.chk_retune.setChecked(True)
        self.e_ql = QLineEdit("")
        self.e_qc = QLineEdit("")
        self.e_ql.setPlaceholderText("lossless")
        self.e_qc.setPlaceholderText("lossless")
        f5.addRow("Values", self.c_std)
        f5.addRow(self.chk_retune)
        f5.addRow("Inductor Q", self.e_ql)
        f5.addRow("Capacitor Q", self.e_qc)
        col.addLayout(f5)

        # result summary
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        col.addSpacing(6)
        col.addWidget(line)
        self.info = QLabel()
        self.info.setTextFormat(Qt.RichText)
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        col.addWidget(self.info)
        col.addStretch(1)

        # the plot range lives in a dialog (View -> Plot range)
        self.range_dlg = PlotRangeDialog(self)
        self.chk_auto = self.range_dlg.chk_auto
        self.e_f1, self.e_f2, self.s_pts = self.range_dlg.e_f1, self.range_dlg.e_f2, self.range_dlg.s_pts

        # signals: selections change which inputs apply, every input triggers a redesign
        self.est_timer = QTimer(self, singleShot=True, interval=150)
        self.est_timer.timeout.connect(self._update_estimate)
        reselect = lambda *_: (self._sync_visibility(), self.schedule())  # noqa: E731
        self.type_group.idClicked.connect(reselect)
        for c in (self.c_approx, self.c_circuit, self.c_even, self.c_espec):
            c.currentIndexChanged.connect(reselect)
        self.chk_single.toggled.connect(reselect)
        for e in (self.e_ripple, self.e_amin, self.e_theta):
            e.valueChanged.connect(self.schedule)
        for e in (self.e_rs, self.e_fc, self.e_bw, self.e_res, self.e_aa, self.e_fs, self.e_mask,
                  self.e_ql, self.e_qc):
            e.textChanged.connect(self.schedule)
        self.s_order.valueChanged.connect(self.schedule)
        for c in (self.chk_edges, self.chk_coils, self.chk_retune):
            c.toggled.connect(self.schedule)
        self.c_std.currentIndexChanged.connect(self.schedule)
        for e in (self.e_est_f, self.e_est_a):
            e.textChanged.connect(self.est_timer.start)
        for e in (self.e_f1, self.e_f2):
            e.textChanged.connect(self._replot)
        self.chk_auto.toggled.connect(self._replot)
        self.s_pts.valueChanged.connect(self._replot)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)
        return scroll

    # ------------------------------------------------------------------ right panel
    def _build_right(self):
        """Right panel: the Response / Group delay / Element values tabs over
        the schematic (in a scroll area)."""
        pg.setConfigOptions(antialias=True)
        self.schematic = SchematicView()
        pal = self.palette()
        bg = pal.base().color()
        fg = pal.text().color()

        self.plot_resp = pg.PlotWidget(background=bg)
        self.plot_gd = pg.PlotWidget(background=bg)
        for pw, ylab in ((self.plot_resp, "dB"), (self.plot_gd, "Group delay (ns)")):
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True, alpha=0.25)
            pi.setLabel("bottom", "Frequency", units="Hz")
            pi.setLabel("left", ylab)
            for ax in ("left", "bottom"):
                pi.getAxis(ax).setTextPen(fg)
                pi.getAxis(ax).setPen(fg)
            pi.addLegend(offset=(-10, 10))
        self.c_ideal = self.plot_resp.plot(name="S21 ideal", pen=pg.mkPen(C_IDEAL, width=1.2, style=Qt.DashLine))
        self.c_s21 = self.plot_resp.plot(name="S21", pen=pg.mkPen(C_S21, width=2))
        self.c_s11 = self.plot_resp.plot(name="S11", pen=pg.mkPen(C_S11, width=1.5))
        self.c_gd = self.plot_gd.plot(name="Group delay", pen=pg.mkPen(C_S21, width=2))
        self.c_mask = self.plot_resp.plot(pen=pg.mkPen(C_EXTRA, width=1.5, style=Qt.DashLine),  # no legend entry
                                          connect="finite")
        self.plot_resp.setYRange(-80, 2)
        # markers: click the response plot to place one, click it again to remove it
        self.markers = []
        self.mark_dots = pg.ScatterPlotItem(pxMode=True)
        self.mark_dots.setZValue(25)
        self.plot_resp.addItem(self.mark_dots, ignoreBounds=True)
        self.mark_labels = []
        box = pg.mkColor(bg)
        box.setAlpha(215)
        self.mark_box = pg.TextItem(color=fg, fill=pg.mkBrush(box), border=pg.mkPen(faded(fg, 90)), anchor=(0, 1))
        self.mark_box.setZValue(30)
        self.plot_resp.addItem(self.mark_box, ignoreBounds=True)
        self.plot_resp.scene().sigMouseClicked.connect(self._click)
        self._fg = fg
        # hover cursors (each plot has its own scene)
        self.cur_resp = PlotCursor(self.plot_resp, fg, bg)
        self.cur_gd = PlotCursor(self.plot_gd, fg, bg)
        self.plot_resp.scene().sigMouseMoved.connect(lambda pos: self._mouse(self.cur_resp, pos))
        self.plot_gd.scene().sigMouseMoved.connect(lambda pos: self._mouse(self.cur_gd, pos))

        self.elements = QPlainTextEdit(readOnly=True)
        self.elements.setFont(mono_font(10))

        self.smith = SmithChart(fg, bg)
        self.plot_td = new_plot(bg, fg, left="Step / impulse", bottom="Time", bottom_units="s")
        self.c_step = self.plot_td.plot(name="Step response", pen=pg.mkPen(C_S21, width=2))
        self.c_imp = self.plot_td.plot(name="Impulse response (peak = 1)", pen=pg.mkPen(C_EXTRA, width=1.5))
        self.mc_tab = MonteCarloTab(self, fg, bg)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.plot_resp, "Response")
        self.tabs.addTab(self.plot_gd, "Group delay")
        self.tabs.addTab(self.smith, "Smith chart")
        self.tabs.addTab(self.plot_td, "Time domain")
        self.tabs.addTab(self.mc_tab, "Monte Carlo")
        self.tabs.addTab(self.elements, "Element values")
        self.tabs.currentChanged.connect(lambda *_: self._update_time())

        # the schematic keeps a readable minimum size; long ladders scroll sideways
        scroll = QScrollArea()
        scroll.setWidget(self.schematic)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(scroll)
        split.setSizes([520, 300])
        self.right_split = split
        return split

    def _build_menu(self):
        """File: designs, exports, copy, quit.  View: plot range, markers.  Tools: diplexer,
        coil.  Help."""
        m = self.menuBar().addMenu("&File")
        for text, key, slot in (("Open design…", "Ctrl+O", self.open_design), ("Save design", "Ctrl+S", self.save_design),
                                ("Save design as…", "Ctrl+Shift+S", self.save_design_as)):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Export report (PDF)…", self)
        a.triggered.connect(self.export_report)
        m.addAction(a)
        a = QAction("Save picture (PNG)…", self)
        a.triggered.connect(self.save_picture)
        m.addAction(a)
        a = QAction("Export S2P (Touchstone)…", self)
        a.triggered.connect(self.export_s2p)
        m.addAction(a)
        a = QAction("Export SPICE netlist…", self)
        a.triggered.connect(self.export_spice)
        m.addAction(a)
        a = QAction("Copy element values", self)
        a.triggered.connect(lambda: QGuiApplication.clipboard().setText(self.elements.toPlainText()))
        m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        v = self.menuBar().addMenu("&View")
        a = QAction("Plot range…", self)
        a.triggered.connect(lambda: (self.range_dlg.show(), self.range_dlg.raise_()))
        v.addAction(a)
        a = QAction("Clear markers", self)
        a.triggered.connect(self.clear_markers)
        v.addAction(a)
        t = self.menuBar().addMenu("&Tools")
        a = QAction("Diplexer…", self)
        a.triggered.connect(self.open_diplexer)
        t.addAction(a)
        a = QAction("Air-core coil…", self)
        a.triggered.connect(self.open_coil)
        t.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("User guide", self)
        a.setShortcut("F1")
        a.triggered.connect(self.open_guide)
        h.addAction(a)
        a = QAction("VektorFIL on GitHub", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL)))
        h.addAction(a)
        a = QAction("Show log folder", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(app_paths.data_dir()))))
        h.addAction(a)
        h.addSeparator()
        a = QAction("Licenses…", self)
        a.triggered.connect(self.show_licenses)
        h.addAction(a)
        a = QAction("About VektorFIL…", self)
        a.triggered.connect(self.show_about)
        h.addAction(a)

    # ------------------------------------------------------------------ help
    def open_guide(self):
        """Open the bundled user guide (docs/index.html) in the default browser."""
        p = app_paths.resource("docs", "index.html")
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        else:
            QMessageBox.information(self, "User guide", "The user guide was not found next to the program.")

    def show_about(self):
        """Version, author, licence and links."""
        QMessageBox.about(
            self, "About VektorFIL",
            f"<h3>VektorFIL {__version__}</h3>"
            "<p>Interactive L-C filter and diplexer synthesis.</p>"
            "<p>by <b>Pieter Ibelings</b><br>Copyright © 2026 Pieter Ibelings</p>"
            "<p>Created from scratch with the help of Claude (Anthropic), written from published "
            "filter-synthesis theory. It contains no code or data from any other program.</p>"
            "<p>This program is free software: you can redistribute it and/or modify it under the "
            "terms of the GNU General Public License as published by the Free Software Foundation, "
            "either version 3 of the License, or (at your option) any later version. It comes with "
            "<b>absolutely no warranty</b>.</p>"
            f"<p><a href='{GITHUB_URL}'>{GITHUB_URL}</a></p>"
            "<p>Built with Python, Qt for Python (PySide6, LGPLv3), numpy, scipy, mpmath and "
            "pyqtgraph — see Help → Licenses.</p>")

    def show_licenses(self):
        """Show the program licence and the third-party notices."""
        parts = []
        for name in ("LICENSE", "THIRD_PARTY_LICENSES.txt"):
            p = app_paths.resource(name)
            if p.exists():
                parts.append(f"===== {name} =====\n\n" + p.read_text(encoding="utf-8", errors="replace"))
        dlg = QDialog(self)
        dlg.setWindowTitle("Licenses")
        dlg.resize(760, 600)
        lay = QVBoxLayout(dlg)
        txt = QPlainTextEdit(readOnly=True)
        txt.setFont(mono_font(9))
        txt.setPlainText("\n\n".join(parts) or "Licence files not found.")
        lay.addWidget(txt)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec()

    # ------------------------------------------------------------------ spec
    def _rebuild_subtypes(self, ftype, shape):
        """Refill the Circuit list for the current response/approximation, keeping the
        selection when it is still offered."""
        names = subtypes(ftype, shape)
        if names == self.sub_names:
            return
        cur = self.current_subtype() if self.sub_names else SHUNT_FIRST
        self.c_circuit.blockSignals(True)
        self.c_circuit.clear()
        self.c_circuit.addItems(names)
        self.c_circuit.setCurrentIndex(names.index(cur) if cur in names else 0)
        self.c_circuit.blockSignals(False)
        self.sub_names = names

    def current_subtype(self):
        i = self.c_circuit.currentIndex()
        return self.sub_names[i] if 0 <= i < len(self.sub_names) else SHUNT_FIRST

    def current_shape(self):
        """Approximation name as FilterSpec uses it ('Chebyshev, singly terminated')."""
        base = self.c_approx.currentText()
        single = self.chk_single.isChecked() and base in SINGLE_OK
        return base + SINGLE if single else base

    def select(self, ftype=None, approximation=None, circuit=None, single=None):
        """Set the selections in one go (used by scripts and tests)."""
        if ftype is not None:
            self.type_group.button(TYPES.index(ftype)).setChecked(True)
        if approximation is not None:
            if approximation.endswith(SINGLE):
                approximation, single = approximation[:-len(SINGLE)], True
            self.c_approx.setCurrentText(approximation)
        if single is not None:
            self.chk_single.setChecked(single)
        self._sync_visibility()
        if circuit is not None:
            self.c_circuit.setCurrentIndex(self.sub_names.index(circuit))
            self._sync_visibility()

    def _sync_visibility(self):
        """Show only the inputs that apply to the selected response, approximation and circuit."""
        ftype = TYPES[self.type_group.checkedId()]
        base_shape = self.c_approx.currentText()
        self.chk_single.setEnabled(base_shape in SINGLE_OK)
        self._rebuild_subtypes(ftype, self.current_shape())
        band = ftype in ("Bandpass", "Bandstop")
        self.l_fc.setText("Center frequency (MHz)" if band else "Cutoff frequency (MHz)")
        self.l_bw.setVisible(band)
        self.e_bw.setVisible(band)
        sub = self.current_subtype() if ftype == "Bandpass" else ""
        ru = resonator_unit(sub) if sub else None
        self.l_res.setVisible(ru is not None)
        self.e_res.setVisible(ru is not None)
        self.chk_edges.setVisible(ru is not None)
        self.chk_coils.setVisible(sub in ("Tubular", "Tumbular"))
        self.chk_coils.setText("End capacitors (extend the resonator range)" if sub == "Tumbular"
                               else "End coils (extend the resonator range)")
        self.l_theta.setVisible(ru == "Z")
        self.e_theta.setVisible(ru == "Z")
        self.l_res.setText({"C": "Resonator C (pF)", "Z": "Resonator line Z0 (Ω)"}.get(ru, "Resonator L (nH)"))
        unit = {"C": 1e-12, "Z": 1.0}.get(ru, 1e-9)
        if unit != self.e_res.unit_mult:
            # switching between a shared L and a shared C: the old number means nothing now
            self.e_res.unit_mult = unit
            self.e_res.auto_hint = None
            self.e_res.setText("")
        ell = base_shape == "Elliptic"
        by = ELLIPTIC_BY[self.c_espec.currentIndex()]
        by_amin = by == "Amin"
        for w in (self.l_espec, self.c_espec):
            w.setVisible(ell)
        self.l_amin.setVisible(ell and by_amin)
        self.e_amin.setVisible(ell and by_amin)
        self.l_fs.setVisible(ell and by == "Fs")
        self.l_mask.setVisible(ell and by == "Mask")
        self.e_mask.setVisible(ell and by == "Mask")
        self.l_fs.setText("Stopband Fs (MHz)" + (" (upper)" if ftype == "Bandpass" else
                                                  " (lower)" if ftype == "Bandstop" else ""))
        self.e_fs.setVisible(ell and by == "Fs")
        self.l_even.setVisible(ell)
        self.c_even.setVisible(ell)
        rip = base_shape in ("Chebyshev", "Elliptic")
        self.l_ripple.setVisible(rip)
        self.e_ripple.setVisible(rip)
        self.l_aa.setVisible(not ell)
        self.e_aa.setVisible(not ell)
        self.e_aa.setPlaceholderText("3.0103 (default)" if base_shape in ("Butterworth", "Bessel")
                                     else "= ripple (default)")
        self.est_timer.start()

    def spec(self) -> FilterSpec:
        """Read all inputs into a FilterSpec (raises ValueError on unreadable input)."""
        def opt(e):
            v = parse_value(e.text())
            return v if v not in (None, 0) else None
        sub = self.current_subtype()
        ru = resonator_unit(sub)
        res_val = None
        if ru and self.e_res.text().strip():
            res_val = self.e_res.value()
        return FilterSpec(
            ftype=TYPES[self.type_group.checkedId()],
            shape=self.current_shape(),
            subtype=sub,
            bw=parse_value(self.e_bw.text(), 1e6) or 0,
            resonator=res_val,
            edge_correct=self.chk_edges.isChecked(),
            end_coils=self.chk_coils.isChecked(),
            tem_theta_deg=self.e_theta.value(),
            order=self.s_order.value(),
            fc=parse_value(self.e_fc.text(), 1e6),
            r_source=parse_value(self.e_rs.text()),
            ripple_db=self.e_ripple.value(),
            aa_db=parse_value(self.e_aa.text()),
            elliptic_spec=ELLIPTIC_BY[self.c_espec.currentIndex()],
            mask=(parse_mask(self.e_mask.text())
                  if ELLIPTIC_BY[self.c_espec.currentIndex()] == "Mask" and self.c_approx.currentText() == "Elliptic"
                  else ()),
            amin_db=self.e_amin.value(),
            fs=parse_value(self.e_fs.text(), 1e6) or 0,
            even_type="CB"[self.c_even.currentIndex()],
            q_l=opt(self.e_ql),
            q_c=opt(self.e_qc),
            std_series=self.c_std.currentText(),
            std_retune=self.chk_retune.isChecked(),
        )

    def _update_estimate(self):
        """Live order estimate: the lowest order of each approximation that gives the
        requested attenuation at the requested frequency."""
        try:
            sp = self.spec()
            fs = parse_value(self.e_est_f.text(), 1e6)
            as_db = parse_value(self.e_est_a.text())
            if not fs or not as_db or sp.fc is None:
                raise ValueError("enter an attenuation and a frequency")
            res = estimate_orders(sp.ftype, sp.fc, fs, sp.ripple_db, as_db, bw=sp.bw)
        except Exception as e:
            self.l_est.setText(f"<i>{str(e)[:60]}</i>")
            return
        if all(v is None for v in res.values()):
            side = {"Lowpass": "above the cutoff", "Highpass": "below the cutoff",
                    "Bandpass": "outside the band", "Bandstop": "inside the stopband"}[sp.ftype]
            self.l_est.setText(f"<i>choose a frequency {side}</i>")
            return
        self.l_est.setText(" · ".join(f"{k} <b>{'—' if v is None else v}</b>" for k, v in res.items()))

    # ------------------------------------------------------------------ compute
    def schedule(self, *_):
        self.timer.start()
        self.est_timer.start()

    def _launch(self):
        """Start a design job for the current inputs (called by the debounce timer)."""
        try:
            spec = self.spec()
            if spec.fc is None or spec.r_source is None:
                raise ValueError("enter cutoff frequency and impedance")
        except Exception as e:
            self.statusBar().showMessage(f"Input: {e}")
            return
        self.gen += 1
        job = DesignJob(self.gen, spec)
        job.signals.done.connect(self._done)
        job.signals.failed.connect(self._failed)
        self.statusBar().showMessage("Matching standard values…" if spec.std_series != "Ideal" else "Computing…")
        self.pool.start(job)

    def _failed(self, gen, msg):
        """Show a design error, unless a newer request has been made since."""
        if gen != self.gen:
            return
        self.statusBar().showMessage(f"Cannot design: {msg.splitlines()[0]}")
        self.info.setText(f"<span style='color:#c0392b'>{msg.splitlines()[0]}</span>")

    def _done(self, gen, res):
        """Show a finished design: schematic, element list, summary, plots."""
        if gen != self.gen:
            return  # a newer request is on its way
        self.result = res
        lad = res.ladder
        self.schematic.set_ladder(lad)
        self.mc_tab.clear()          # a spread belongs to the design it was computed for
        txt = lad.title + "\n\n" + lad.table()
        if res.ideal_ladder is not None:
            txt += "\n\nIdeal (unrounded) values:\n" + res.ideal_ladder.table()
        self.elements.setPlainText(txt)
        # info
        inf = res.info
        def fl(v):
            return " / ".join(eng(x, "Hz") for x in v) if isinstance(v, (tuple, list)) else eng(v, "Hz")
        st = res.ladder.single_terminated
        rows = [f"Output: <b>{st}</b> (singly terminated; S21 = voltage transfer into the open end)"
                if st == "open" else
                f"Output: <b>{st}</b> (singly terminated; S21 = current into the short)"
                if st == "short" else f"Output R: <b>{inf['r_load']:.6g} Ω</b>"]
        if "edges" in inf:
            rows.append(f"Band edges ({inf['aa_db']:.4g} dB): <b>{fl(inf['edges'])}</b>")
            act = inf.get("edges_actual")
            if act and any(abs(a / e - 1) > 1e-5 for a, e in zip(act, inf["edges"])):
                rows.append(f"<span style='color:#c0392b'>Actual edges: <b>{fl(act)}</b></span>")
        if "skirts" in inf:
            f_lo, a_lo, f_hi, a_hi = inf["skirts"]
            rows.append(f"Skirts: <b>{a_lo:.1f} dB</b> at {eng(f_lo, 'Hz')}, <b>{a_hi:.1f} dB</b> at {eng(f_hi, 'Hz')}")
        if "f3db" in inf:
            rows.append(f"3 dB frequenc{'ies' if isinstance(inf['f3db'], tuple) else 'y'}: <b>{fl(inf['f3db'])}</b>")
        if "fs" in inf:
            rows.append(f"Stopband: Amin <b>{inf['amin_db']:.3f} dB</b> at <b>{fl(inf['fs'])}</b>")
        if "resonator" in inf:
            self.e_res.auto_hint = inf["resonator"]   # arrows start from the value in use
            self.e_res.limits = inf.get("resonator_range")  # arrows stop at the range ends
            ru = resonator_unit(res.spec.subtype)
            u = {"L": "H", "C": "F", "Z": "Ω"}[ru]
            what = {"L": "Resonator L", "C": "Resonator C", "Z": "Line Z0"}[ru]
            lo_r, hi_r = inf.get("resonator_range", (inf["resonator_limit"],) * 2)
            rows.append(f"{what}: <b>{eng(inf['resonator'], u)}</b> "
                        f"(usable {eng(lo_r, u)} … {eng(hi_r, u)})")
        if "mask_margin_db" in inf:
            m = inf["mask_margin_db"]
            colour = "" if m >= -0.01 else " style='color:#c0392b'"
            rows.append(f"<span{colour}>Mask margin: <b>{m:+.2f} dB</b></span>")
        if inf.get("zeros"):
            rows.append("Transmission zeros: " + ", ".join(eng(z, "Hz") for z in inf["zeros"]))
        if "std" in inf:
            st = inf["std"]
            rows.append(f"Standard values {st['series']} ({st['method']}): passband within "
                        f"<b>{st['pass_dev_db']:.3f} dB</b> of ideal"
                        + (f", stopband short by <b>{st['stop_short_db']:.2f} dB</b>" if st['stop_short_db'] > 0.01 else ""))
        rows += [f"<span style='color:#c0392b'>{w}</span>" for w in res.warnings]
        self.info.setText("<br>".join(rows))
        self.statusBar().showMessage(f"{lad.title} — designed in {res.seconds * 1000:.0f} ms")
        self._replot()

    def _freqs(self):
        """Frequency sweep: the range from View -> Plot range, or an automatic one."""
        res = self.result
        n = self.s_pts.value()
        if not self.chk_auto.isChecked():
            try:
                f1, f2 = parse_value(self.e_f1.text(), 1e6), parse_value(self.e_f2.text(), 1e6)
                if f2 > f1 >= 0:
                    return np.linspace(max(f1, f2 * 1e-6), f2, n)
            except Exception:
                pass
        return auto_freqs(res, n)

    def _replot(self, *_):
        """Recompute and draw the response curves for the current design."""
        if self.result is None:
            return
        f = self._freqs()
        r = self.result.ladder.response(f)
        self.c_s21.setData(f, r["s21_db"])
        if self.result.ideal_ladder is not None:
            ri = self.result.ideal_ladder.response(f)
            self.c_ideal.setData(f, ri["s21_db"])
        else:
            self.c_ideal.setData([], [])
        self.c_s11.setData(f, r["s11_db"])
        self.c_gd.setData(f, r["gd"] * 1e9)
        lo = max(-120.0, float(np.nanmin(r["s21_db"])) - 5)
        self.plot_resp.setYRange(lo, 2)
        self.plot_resp.setXRange(f[0], f[-1], padding=0)
        self.plot_gd.setXRange(f[0], f[-1], padding=0)
        self._resp_cache = (f, r, ri["s21_db"] if self.result.ideal_ladder is not None else None)
        self.cur_resp.hide()
        self.cur_gd.hide()
        self.smith.set_data(f, self.result.ladder.sparams(f)[0], self.result.ladder.r_source)
        self._draw_mask(f)
        # markers outside the plotted range (left from another design) are dropped
        self.markers = [m for m in self.markers if f[0] <= m <= f[-1]]
        self._draw_markers()
        self._update_time()
        if self.chk_auto.isChecked():
            # show the automatic range in Start/Stop (this is also what Export S2P uses)
            for e, v in ((self.e_f1, f[0]), (self.e_f2, f[-1])):
                e.blockSignals(True)
                e.setText(f"{v / 1e6:.6g}")
                e.blockSignals(False)

    def _mouse(self, cur, pos):
        """Hover readout on the response or delay plot: the values at the nearest
        computed frequency, with dots on the curves."""
        if self.result is None or not hasattr(self, "_resp_cache") or not cur.contains(pos):
            cur.hide()
            return
        f, r, ideal = self._resp_cache
        x = cur.x_at(pos)
        i = int(np.clip(np.searchsorted(f, x), 0, len(f) - 1))
        if i > 0 and abs(f[i - 1] - x) < abs(f[i] - x):
            i -= 1
        s21, s11 = r["s21_db"][i], r["s11_db"][i]
        if cur is self.cur_gd:
            gd = r["gd"][i] * 1e9
            cur.show(f[i], f"{eng(f[i], 'Hz')}\nDelay {gd:.4g} ns\nS21 {s21:.3f} dB", [(gd, C_S21)])
            return
        self.smith.mark(i)
        lines = [eng(f[i], "Hz"), f"S21 {s21:.3f} dB", f"S11 {s11:.2f} dB"]
        points = [(s21, C_S21), (s11, C_S11)]
        if ideal is not None:
            lines.append(f"ideal {ideal[i]:.3f} dB")
            points.append((ideal[i], C_IDEAL))
        cur.show(f[i], "\n".join(lines), points)

    def _draw_mask(self, f):
        """The stepped stopband mask as dashed steps at -A dB (elliptic 'Stepped mask')."""
        sp = self.result.spec
        if sp.elliptic_spec != "Mask" or not sp.mask or not sp.shape.startswith("Elliptic"):
            self.c_mask.setData([], [])
            return
        xs, ys = [], []
        for a, b, db in sp.mask:
            b = min(b, f[-1]) if math.isinf(b) else b
            xs += [a, b, np.nan]
            ys += [-db, -db, np.nan]
        self.c_mask.setData(xs, ys)

    # ------------------------------------------------------------------ markers
    def _click(self, ev):
        """Left click on the response plot: place a marker, or remove the one clicked."""
        if ev.button() != Qt.LeftButton or self.result is None:
            return
        vb = self.plot_resp.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = vb.mapSceneToView(ev.scenePos()).x()
        (x0, x1), _ = vb.viewRange()
        near = [m for m in self.markers if abs(m - x) < 0.01 * (x1 - x0)]
        if near:
            self.markers.remove(near[0])
        elif len(self.markers) < MAX_MARKERS and x > 0:
            self.markers.append(x)
        self._draw_markers()

    def clear_markers(self):
        self.markers = []
        self._draw_markers()

    def _draw_markers(self):
        """A numbered triangle on S21 and on S11 at each marker frequency, and a small
        table of the values in the lower left corner."""
        for lab in self.mark_labels:
            self.plot_resp.removeItem(lab)
        self.mark_labels = []
        if not self.markers or self.result is None:
            self.mark_dots.setData([])
            self.mark_box.setVisible(False)
            return
        fm = np.array(self.markers)
        s11, s21 = self.result.ladder.sparams(fm)
        d21 = 20 * np.log10(np.abs(s21) + 1e-300)
        d11 = 20 * np.log10(np.abs(s11) + 1e-300)
        (x0, x1), (y0, y1) = self.plot_resp.getPlotItem().vb.viewRange()
        spots = []
        for k, f in enumerate(fm, 1):
            for y, colour in ((d21[k - 1], C_S21), (d11[k - 1], C_S11)):
                if not (np.isfinite(y) and y0 <= y <= y1):
                    continue
                spots.append({"pos": (f, y), "symbol": "t", "size": 12, "brush": pg.mkBrush(colour),
                              "pen": pg.mkPen(self._fg, width=1)})
                lab = pg.TextItem(str(k), color=self._fg, anchor=(0.5, 1.25))
                lab.setPos(f, y)
                lab.setZValue(26)
                self.plot_resp.addItem(lab, ignoreBounds=True)
                self.mark_labels.append(lab)
        self.mark_dots.setData(spots=spots)
        fg = pg.mkColor(self._fg).name()
        cell = "style='padding:0px 10px 0px 0px'"
        rows = [f"<tr><th {cell}></th><th {cell} align=left>Frequency</th>"
                f"<th {cell} align=right style='color:{C_S21}'>S21</th>"
                f"<th {cell} align=right style='color:{C_S11}'>S11</th></tr>"]
        for k, (f, a, b) in enumerate(zip(fm, d21, d11), 1):
            rows.append(f"<tr><td {cell}><b>{k}</b></td><td {cell}>{eng(f, 'Hz')}</td>"
                        f"<td {cell} align=right>{a:.3f} dB</td><td {cell} align=right>{b:.2f} dB</td></tr>")
        self.mark_box.setHtml(f"<table style='color:{fg}' cellspacing=4>{''.join(rows)}</table>")
        self.mark_box.setPos(x0 + 0.01 * (x1 - x0), y0 + 0.03 * (y1 - y0))
        self.mark_box.setVisible(True)

    # ------------------------------------------------------------------ time domain
    def _update_time(self):
        """Step and impulse response; computed only while the Time domain tab is shown."""
        if self.result is None or self.tabs.currentWidget() is not self.plot_td:
            return
        f_max, t_end = timedomain.auto_window(self.result.spec, self.result.info)
        tr = timedomain.time_response(self.result.ladder, f_max)
        sel = tr.t <= t_end
        imp = tr.impulse[sel]
        peak = float(np.max(np.abs(imp))) or 1.0
        self.c_step.setData(tr.t[sel], tr.step[sel])
        self.c_imp.setData(tr.t[sel], imp / peak)
        self.plot_td.setXRange(0, t_end, padding=0)

    # ------------------------------------------------------------------ design files
    def view_state(self) -> dict:
        """Window settings stored in a design file next to the specification."""
        return {"order_estimate": [self.e_est_a.text(), self.e_est_f.text()],
                "plot_range": {"auto": self.chk_auto.isChecked(), "start": self.e_f1.text(),
                               "stop": self.e_f2.text(), "points": self.s_pts.value()},
                "markers": list(self.markers), "monte_carlo": self.mc_tab.settings()}

    def apply_spec(self, spec: FilterSpec, view: dict | None = None):
        """Set every input from a FilterSpec (and window settings), then redesign."""
        spec = replace(spec, subtype=canonical_subtype(spec.ftype, spec.subtype))
        base, single = (spec.shape[:-len(SINGLE)], True) if spec.shape.endswith(SINGLE) else (spec.shape, False)
        self.select(ftype=spec.ftype, approximation=base, single=single)
        if spec.subtype in self.sub_names:
            self.c_circuit.setCurrentIndex(self.sub_names.index(spec.subtype))
        self.c_espec.setCurrentIndex(ELLIPTIC_BY.index(spec.elliptic_spec) if spec.elliptic_spec in ELLIPTIC_BY else 0)
        self.c_even.setCurrentIndex(1 if spec.even_type == "B" else 0)
        self._sync_visibility()
        self.e_rs.setText(f"{spec.r_source:.6g}")
        self.e_fc.setText(f"{spec.fc / 1e6:.6g}")
        self.e_bw.setText(f"{spec.bw / 1e6:.6g}")
        self.s_order.setValue(int(spec.order))
        self.e_ripple.setValue(spec.ripple_db)
        self.e_aa.setText("" if spec.aa_db is None else f"{spec.aa_db:.6g}")
        self.e_amin.setValue(spec.amin_db)
        self.e_fs.setText(f"{spec.fs / 1e6:.6g}")
        if spec.mask:
            self.e_mask.setText(format_mask(spec.mask))
        self.e_theta.setValue(spec.tem_theta_deg)
        self.chk_edges.setChecked(spec.edge_correct)
        self.chk_coils.setChecked(spec.end_coils)
        self.e_ql.setText("" if spec.q_l is None else f"{spec.q_l:g}")
        self.e_qc.setText("" if spec.q_c is None else f"{spec.q_c:g}")
        self.c_std.setCurrentText(spec.std_series)
        self.chk_retune.setChecked(spec.std_retune)
        self.e_res.setText("" if spec.resonator is None else f"{spec.resonator / self.e_res.unit_mult:.6g}")
        view = view or {}
        est = view.get("order_estimate")
        if est and len(est) == 2:
            self.e_est_a.setText(est[0])
            self.e_est_f.setText(est[1])
        pr = view.get("plot_range")
        if pr:
            self.chk_auto.setChecked(bool(pr.get("auto", True)))
            self.e_f1.setText(str(pr.get("start", "0")))
            self.e_f2.setText(str(pr.get("stop", "300")))
            self.s_pts.setValue(int(pr.get("points", 1001)))
        self.markers = [float(m) for m in view.get("markers", [])][:MAX_MARKERS]
        if "monte_carlo" in view:
            self.mc_tab.apply_settings(view["monte_carlo"])
        self._sync_visibility()
        self.schedule()

    def _set_path(self, path):
        """Remember the design file and show its name in the title bar."""
        self.path = path
        name = Path(path).name if path else None
        self.setWindowTitle(f"{name} — VektorFIL" if name else "VektorFIL — L-C filter synthesis")

    def open_design(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open design", "", "VektorFIL design (*.vfil)")
        if path:
            self.load_design(path)

    def load_design(self, path, remember=True):
        """Read a design file into the window."""
        try:
            spec, view = project.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Open design", f"Cannot read {path}:\n{e}")
            return
        self.apply_spec(spec, view)
        if remember:
            self._set_path(path)

    def save_design(self):
        if getattr(self, "path", None):
            self._write_design(self.path)
        else:
            self.save_design_as()

    def save_design_as(self):
        """Ask for a file name and save the design there."""
        sp = self.result.spec if self.result else None
        name = (f"{sp.shape.split(',')[0]}_{sp.ftype}_{sp.order}".lower() if sp else "design") + project.SUFFIX
        path, _ = QFileDialog.getSaveFileName(self, "Save design", name, "VektorFIL design (*.vfil)")
        if path:
            self._write_design(path)
            self._set_path(path)

    def _write_design(self, path):
        try:
            project.save(path, self.spec(), self.view_state())
        except Exception as e:
            QMessageBox.warning(self, "Save design", str(e))
            return
        self.statusBar().showMessage(f"Saved {path}")

    def closeEvent(self, ev):
        """Remember the current design for the next start."""
        if not self.remember_last:
            super().closeEvent(ev)
            return
        try:
            project.save(app_paths.data_dir() / "last.vfil", self.spec(), self.view_state())
        except Exception:
            pass
        super().closeEvent(ev)

    # ------------------------------------------------------------------ pictures and reports
    def _grab_response(self):
        """The response plot as a picture (switching to its tab briefly if needed)."""
        cur = self.tabs.currentIndex()
        self.tabs.setCurrentIndex(0)
        QApplication.processEvents()
        pm = self.plot_resp.grab()
        self.tabs.setCurrentIndex(cur)
        return pm

    def export_report(self):
        """File -> Export report: summary, response, schematic and element list as PDF."""
        if self.result is None:
            return
        from vektorfil.gui.report import write_report
        path, _ = QFileDialog.getSaveFileName(self, "Export report", "filter_report.pdf", "PDF (*.pdf)")
        if not path:
            return
        write_report(path, self.result.ladder.title, self.info.text(),
                     [self._grab_response(), self.schematic.grab()], self.elements.toPlainText())
        self.statusBar().showMessage(f"Saved {path}")

    def save_picture(self):
        """File -> Save picture: the plots and the schematic as a PNG."""
        path, _ = QFileDialog.getSaveFileName(self, "Save picture", "filter.png", "PNG (*.png)")
        if path:
            self.right_split.grab().save(path)
            self.statusBar().showMessage(f"Saved {path}")

    def open_coil(self):
        """Tools -> Air-core coil, with the inductors of the current design to choose from."""
        from vektorfil.gui.coil_dialog import CoilDialog
        inductors = []
        if self.result is not None:
            lad = self.result.ladder
            inductors = [(ln, b.L) for b, (ln, _) in zip(lad.branches, lad.designators())
                         if b.L is not None and b.kind != "TEM"]
        dlg = CoilDialog(self, inductors)
        dlg.show()
        self._coil_dlg = dlg

    # ------------------------------------------------------------------ actions
    def open_diplexer(self):
        """Open (or bring forward) the diplexer designer."""
        from vektorfil.gui.diplexer_window import DiplexerWindow
        if getattr(self, "_diplexer", None) is None:
            self._diplexer = DiplexerWindow()
        self._diplexer.show()
        self._diplexer.raise_()
        self._diplexer.activateWindow()

    def export_s2p(self):
        """Save S-parameters as a Touchstone .s2p file (dB/angle) over the plot range
        (View -> Plot range), referenced to an impedance asked for here."""
        if self.result is None:
            return
        try:
            f1 = parse_value(self.e_f1.text(), 1e6)
            f2 = parse_value(self.e_f2.text(), 1e6)
            if f1 is None or f2 is None or not f2 > f1 >= 0:
                raise ValueError("the plot range needs 0 <= Start < Stop (View -> Plot range)")
        except ValueError as e:
            QMessageBox.warning(self, "Export S2P", str(e))
            return
        z0, ok = QInputDialog.getDouble(self, "Export S2P", "Reference impedance (Ω)",
                                        self.result.ladder.r_source, 0.001, 1e6, 3)
        if not ok:
            return
        n = self.s_pts.value()
        f = np.linspace(f1, f2, n)
        if f[0] == 0:
            f[0] = 1.0   # the lumped model is singular exactly at DC: use 1 Hz
        sp = self.result.spec
        name = f"{sp.shape}_{sp.ftype}_{sp.order}".lower() + ".s2p"
        path, _ = QFileDialog.getSaveFileName(self, "Export S2P (Touchstone)", name, "Touchstone (*.s2p)")
        if not path:
            return
        lad = self.result.ladder
        note = (f"Filter terminations: {lad.r_source:.6g} / {lad.r_load:.6g} ohm; "
                f"S-parameters referenced to {z0:g} ohm at both ports")
        if lad.q_l or lad.q_c:
            note += f"\nComponent Q: L {lad.q_l or 'lossless'}, C {lad.q_c or 'lossless'}"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(lad.touchstone(f, z0, comment=note))
        self.statusBar().showMessage(f"Saved {path} ({n} points, {eng(f[0], 'Hz')} to {eng(f[-1], 'Hz')})")

    def export_spice(self):
        """Ask for a file name and save the design as a SPICE netlist."""
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export SPICE netlist", "filter.cir", "SPICE (*.cir *.sp)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(to_spice(self.result.ladder))
        self.statusBar().showMessage(f"Saved {path}")


def to_spice(lad) -> str:
    """SPICE netlist of a ladder: AC source with RS, the elements (named as in the
    schematic: L1.., C1..), and RL.  Node 'in' is the source, 0 is ground."""
    lines = [f"* {lad.title} - VektorFIL", "V1 in 0 AC 1", f"RS in 1 {lad.r_source:.9g}"]
    node, nxt = 1, 2
    for b, (ln, cn) in zip(lad.branches, lad.designators()):
        if b.pos == "series":
            a, c = node, nxt
            nxt += 1
            if b.kind == "L":
                lines.append(f"{ln} {a} {c} {b.L:.9g}")
            elif b.kind == "C":
                lines.append(f"{cn} {a} {c} {b.C:.9g}")
            elif b.kind == "LC_par":
                lines += [f"{ln} {a} {c} {b.L:.9g}", f"{cn} {a} {c} {b.C:.9g}"]
            else:  # series resonator: L then C through an internal node
                m = nxt
                nxt += 1
                lines += [f"{ln} {a} {m} {b.L:.9g}", f"{cn} {m} {c} {b.C:.9g}"]
            node = c
        else:
            if b.kind == "L":
                lines.append(f"{ln} {node} 0 {b.L:.9g}")
            elif b.kind == "C":
                lines.append(f"{cn} {node} 0 {b.C:.9g}")
            elif b.kind == "LC_ser":  # series L-C to ground through an internal node
                m = nxt
                nxt += 1
                lines += [f"{ln} {node} {m} {b.L:.9g}", f"{cn} {m} 0 {b.C:.9g}"]
            else:
                lines += [f"{ln} {node} 0 {b.L:.9g}", f"{cn} {node} 0 {b.C:.9g}"]
    st = lad.single_terminated
    if st == "open":     # singly terminated: a very large / very small resistor stands in
        lines.append(f"RL {node} 0 1e12 ; output open (singly terminated)")
    elif st == "short":
        lines.append(f"RL {node} 0 1e-6 ; output shorted (singly terminated)")
    else:
        lines.append(f"RL {node} 0 {lad.r_load:.9g}")
    lines += [".end", ""]
    return "\n".join(lines)


LOG_FILE = str(app_paths.log_file())


def _install_crash_guard(window_ref):
    """Keep the program alive on unexpected errors.

    Started without a console there is nowhere for an unhandled error to go (it may even
    take the process down).  Errors are appended to vektorfil.log in the user's
    application-data folder (see app_paths) and shown in the status bar instead."""
    if sys.stderr is None:                       # pythonw: give Python somewhere to write
        sys.stderr = open(LOG_FILE, "a", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = sys.stderr

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(f"\n--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{text}")
        except OSError:
            pass
        w = window_ref()
        if w is not None:
            w.statusBar().showMessage(f"Internal error ({etype.__name__}: {value}) — details in "
                                      f"vektorfil.log (Help → Show log folder)")
    sys.excepthook = hook


def app_icon():
    """The program icon (assets/vektorfil.ico), or an empty icon if it is missing."""
    p = app_paths.resource("assets", "vektorfil.ico")
    return QIcon(str(p)) if p.exists() else QIcon()


def selftest() -> int:
    """Design a few filters and a diplexer without showing a window; exit code 0 if all
    is well.  Used to check a compiled build:  VektorFIL.exe --selftest"""
    import numpy as np
    from vektorfil.filters.diplexer import DiplexSpec, design_diplexer
    out = []
    checks = [
        (FilterSpec(shape="Chebyshev", order=7, fc=70e6, ripple_db=0.25), 70e6, -0.25),
        (FilterSpec(shape="Elliptic", order=6, fc=100e6, amin_db=60), 100e6, -0.1),
        (FilterSpec(ftype="Bandpass", shape="Chebyshev", subtype="Shunt resonators, capacitive coupling", order=5,
                    fc=100e6, bw=5e6), None, None),
    ]
    ok = True
    for spec, f, want in checks:
        r = design(spec)
        msg = f"{r.ladder.title}: {len(r.ladder.branches)} branches, {r.seconds * 1000:.0f} ms"
        if f is not None:
            got = 20 * np.log10(abs(r.ladder.sparams(np.array([f]))[1][0]))
            good = abs(got - want) < 1e-3
            ok &= good
            msg += f", S21({f / 1e6:g} MHz) = {got:.4f} dB {'OK' if good else 'FAIL'}"
        out.append(msg)
    d = design_diplexer(DiplexSpec(lp_order=5, hp_order=5))
    good = d.info["worst_return_loss_db"] > 100
    ok &= good
    out.append(f"Butterworth diplexer: return loss {d.info['worst_return_loss_db']:.0f} dB "
               f"{'OK' if good else 'FAIL'}")
    # exercise the GUI off-screen as well
    app = QApplication.instance() or QApplication(sys.argv)
    w = FilterWindow()
    t0 = datetime.datetime.now()
    while w.result is None and (datetime.datetime.now() - t0).total_seconds() < 60:
        app.processEvents()
    good = w.result is not None
    ok &= good
    out.append(f"Window design: {'OK' if good else 'FAIL'}")
    out.append("SELFTEST " + ("PASSED" if ok else "FAILED"))
    text = "\n".join(out)
    if sys.stdout is not None:               # a windowed build has no console
        print(text)
    try:
        (app_paths.data_dir() / "selftest.txt").write_text(text, encoding="utf-8")
    except OSError:
        pass
    return 0 if ok else 1


def main():
    """Start the application and show the filter window (or run --selftest)."""
    if "--selftest" in sys.argv:
        return selftest()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VektorFIL")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    w = FilterWindow()
    _install_crash_guard(weakref.ref(w))
    w.remember_last = True
    last = app_paths.data_dir() / "last.vfil"
    if last.exists():                      # carry on where the last session ended
        try:
            spec, view = project.load(last)
            w.apply_spec(spec, view)
        except Exception:
            pass
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
