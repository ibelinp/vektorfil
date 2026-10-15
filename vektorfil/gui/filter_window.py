"""VektorFIL lumped filter designer: interactive window with live updates.

Run:  python -m vektorfil.gui.filter_window
"""
from __future__ import annotations

import datetime
import math
import os
import re
import sys
import traceback
import weakref
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QGuiApplication, QIcon
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from vektorfil import __version__, app_paths
from vektorfil.filters import (SERIES_NAMES, SHUNT_FIRST, TYPES, FilterSpec, auto_freqs, design,
                              estimate_orders, resonator_unit, subtypes)
from vektorfil.filters.design import SINGLE

APPROXIMATIONS = ("Butterworth", "Chebyshev", "Bessel", "Elliptic")
SINGLE_OK = ("Butterworth", "Chebyshev", "Elliptic")      # approximations with a singly terminated form
from vektorfil.gui.schematic import SchematicView, eng

GITHUB_URL = "https://github.com/ibelinp/vektorfil"

SUFFIX = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "K": 1e3, "m": 1e-3, "u": 1e-6, "µ": 1e-6,
          "n": 1e-9, "p": 1e-12, "f": 1e-15}


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
        v = self._current()
        if not v or v <= 0:
            return
        v *= self.ratio ** steps
        if self.limits:
            lo, hi = self.limits
            v = min(max(v, lo), hi)
        self.setText(f"{v / self.unit_mult:.5g}")


# --------------------------------------------------------------------------- worker

class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class DesignJob(QRunnable):
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
    @staticmethod
    def _heading(num, title):
        """Section heading of the specification panel: a step number and a title."""
        lab = QLabel(f"<span style='font-size:13pt'>{num}</span>&nbsp; <b>{title}</b>")
        lab.setContentsMargins(0, 10, 0, 2)
        return lab

    def _build_left(self):
        """Left panel: the specification in five steps (response, approximation,
        circuit, specification, components), then the result summary."""
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setSpacing(4)

        # 1  Response: a row of toggle buttons
        col.addWidget(self._heading("①", "Response"))
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
        col.addWidget(self._heading("②", "Approximation"))
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
        col.addWidget(self._heading("③", "Circuit"))
        self.c_circuit = QComboBox()
        self.c_circuit.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.c_circuit.setMinimumContentsLength(20)
        self.sub_names = []
        self._rebuild_subtypes("Lowpass", "Chebyshev")
        col.addWidget(self.c_circuit)

        # 4  Specification
        col.addWidget(self._heading("④", "Specification"))
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
        self.c_espec.addItems(["Minimum attenuation", "Stopband frequency"])
        self.e_amin = QDoubleSpinBox()
        self.e_amin.setDecimals(2)
        self.e_amin.setRange(1.0, 200.0)
        self.e_amin.setSingleStep(1.0)
        self.e_amin.setValue(60.0)
        self.e_fs = QLineEdit("150")
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
        col.addWidget(self._heading("⑤", "Components"))
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
        for e in (self.e_rs, self.e_fc, self.e_bw, self.e_res, self.e_aa, self.e_fs, self.e_ql, self.e_qc):
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
        self.c_ideal = self.plot_resp.plot(name="S21 ideal", pen=pg.mkPen((140, 140, 140), width=1.2, style=Qt.DashLine))
        self.c_s21 = self.plot_resp.plot(name="S21", pen=pg.mkPen("#2f7ed8", width=2))
        self.c_s11 = self.plot_resp.plot(name="S11", pen=pg.mkPen("#e4572e", width=1.5))
        self.c_gd = self.plot_gd.plot(name="Group delay", pen=pg.mkPen("#2f7ed8", width=2))
        self.plot_resp.setYRange(-80, 2)
        self.readout = pg.TextItem(anchor=(0, 0), color=fg)
        self.plot_resp.addItem(self.readout, ignoreBounds=True)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(fg, width=0.6, style=Qt.DashLine))
        self.plot_resp.addItem(self.vline, ignoreBounds=True)
        self.plot_resp.scene().sigMouseMoved.connect(self._mouse)

        self.elements = QPlainTextEdit(readOnly=True)
        self.elements.setFont(QFont("Consolas", 10))

        self.tabs = QTabWidget()
        self.tabs.addTab(self.plot_resp, "Response")
        self.tabs.addTab(self.plot_gd, "Group delay")
        self.tabs.addTab(self.elements, "Element values")

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
        return split

    def _build_menu(self):
        """File: exports, copy, quit.  View: plot range.  Tools: diplexer.  Help."""
        m = self.menuBar().addMenu("&File")
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
        t = self.menuBar().addMenu("&Tools")
        a = QAction("Diplexer…", self)
        a.triggered.connect(self.open_diplexer)
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
        txt.setFont(QFont("Consolas", 9))
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
        by_amin = self.c_espec.currentIndex() == 0
        for w in (self.l_espec, self.c_espec):
            w.setVisible(ell)
        self.l_amin.setVisible(ell and by_amin)
        self.e_amin.setVisible(ell and by_amin)
        self.l_fs.setVisible(ell and not by_amin)
        self.l_fs.setText("Stopband Fs (MHz)" + (" (upper)" if ftype == "Bandpass" else
                                                  " (lower)" if ftype == "Bandstop" else ""))
        self.e_fs.setVisible(ell and not by_amin)
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
            elliptic_spec=("Amin", "Fs")[self.c_espec.currentIndex()],
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
        self._resp_cache = (f, r)
        if self.chk_auto.isChecked():
            # show the automatic range in Start/Stop (this is also what Export S2P uses)
            for e, v in ((self.e_f1, f[0]), (self.e_f2, f[-1])):
                e.blockSignals(True)
                e.setText(f"{v / 1e6:.6g}")
                e.blockSignals(False)

    def _mouse(self, pos):
        """Hover readout: frequency, S21 and S11 at the mouse position."""
        if self.result is None or not hasattr(self, "_resp_cache"):
            return
        vb = self.plot_resp.getPlotItem().vb
        if not self.plot_resp.sceneBoundingRect().contains(pos):
            return
        x = vb.mapSceneToView(pos).x()
        f, r = self._resp_cache
        i = int(np.clip(np.searchsorted(f, x), 0, len(f) - 1))
        self.vline.setPos(f[i])
        self.readout.setText(f"{eng(f[i], 'Hz')}\nS21 {r['s21_db'][i]:.3f} dB\nS11 {r['s11_db'][i]:.2f} dB")
        (x0, x1), (y0, y1) = vb.viewRange()
        self.readout.setPos(x0 + 0.02 * (x1 - x0), y1 - 0.02 * (y1 - y0))

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
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
