"""Diplexer designer window: a lowpass and a highpass channel joined at a common port.

Run on its own:  python -m vektorfil.gui.diplexer_window
or from the filter window: Tools -> Diplexer...

Both channels are singly terminated filters whose common-port end is a series element
(series L for the lowpass, series C for the highpass), so in each passband the other
channel presents a high impedance at the junction.  Each channel is specified by its
3 dB frequency; with a common crossover the two conductances add up to about 1/R and
the common port stays matched (exactly so for equal-order Butterworth channels).
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGroupBox, QLabel, QLineEdit, QMainWindow,
                               QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from vektorfil.filters.diplexer import CHANNEL_SHAPES, DiplexSpec, design_diplexer, touchstone_s3p
from vektorfil.gui.filter_window import EngSpinBox, form_layout, parse_value
from vektorfil.gui.schematic import SchematicView, eng


class _Signals(QObject):
    """Signals of a background job: done(generation, result) or failed(generation, message)."""
    done = Signal(int, object)
    failed = Signal(int, str)


class _Job(QRunnable):
    """Runs design_diplexer(spec) in the background."""
    def __init__(self, gen, spec):
        super().__init__()
        self.gen, self.spec, self.signals = gen, spec, _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.gen, design_diplexer(self.spec))
        except Exception as e:   # shown in the status bar
            msg = str(e) or e.__class__.__name__
            if not isinstance(e, (ValueError, ArithmeticError)):
                msg += "\n" + traceback.format_exc(limit=3)
            self.signals.failed.emit(self.gen, msg)


class _Channel(QGroupBox):
    """Inputs for one channel: shape, order, ripple and (optionally) its own 3 dB point."""

    def __init__(self, title, on_change):
        super().__init__(title)
        form = form_layout(self)
        self.shape = QComboBox()
        self.shape.addItems(CHANNEL_SHAPES)
        self.order = QSpinBox()
        self.order.setRange(1, 15)
        self.order.setValue(5)
        self.ripple = QDoubleSpinBox()
        self.ripple.setDecimals(4)
        self.ripple.setRange(0.0001, 3.0)
        self.ripple.setSingleStep(0.01)
        self.ripple.setValue(0.1)
        self.l_ripple = QLabel("Passband ripple (dB)")
        self.amin = QDoubleSpinBox()
        self.amin.setDecimals(1)
        self.amin.setRange(10.0, 150.0)
        self.amin.setSingleStep(1.0)
        self.amin.setValue(50.0)
        self.l_amin = QLabel("Stopband Amin (dB)")
        self.f3 = EngSpinBox("100", 1e6, ratio=1.01)
        self.l_f3 = QLabel("3 dB frequency (MHz)")
        form.addRow("Approximation", self.shape)
        form.addRow("Order", self.order)
        form.addRow(self.l_ripple, self.ripple)
        form.addRow(self.l_amin, self.amin)
        form.addRow(self.l_f3, self.f3)
        self.shape.currentIndexChanged.connect(lambda *_: (self.sync(), on_change()))
        self.order.valueChanged.connect(on_change)
        self.ripple.valueChanged.connect(on_change)
        self.amin.valueChanged.connect(on_change)
        self.f3.textChanged.connect(on_change)
        self.sync()

    def sync(self):
        """Show the ripple and Amin inputs only for the approximations that use them."""
        shape = self.shape.currentText()
        rip = shape in ("Chebyshev", "Elliptic")
        self.l_ripple.setVisible(rip)
        self.ripple.setVisible(rip)
        self.l_amin.setVisible(shape == "Elliptic")
        self.amin.setVisible(shape == "Elliptic")


class DiplexerWindow(QMainWindow):
    """Lowpass/highpass diplexer: inputs on the left, channel schematics and the 3-port
    response on the right.  Recomputes live in a worker thread, like the filter window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VektorFIL — Diplexer")
        self.resize(1320, 860)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.gen = 0
        self.result = None
        self.timer = QTimer(self, singleShot=True, interval=150)
        self.timer.timeout.connect(self._launch)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([360, 960])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Ready")
        self._sync()
        self.schedule()

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        """Inputs: impedance, crossover, and one box per channel."""
        w = QWidget()
        lay = QVBoxLayout(w)
        top = form_layout()
        self.e_r0 = QLineEdit("50")
        self.e_fx = EngSpinBox("100", 1e6, ratio=1.01)
        self.chk_same = QCheckBox("Common crossover (both channels 3 dB down there)")
        self.chk_same.setChecked(True)
        top.addRow("Impedance R0 (Ω)", self.e_r0)
        top.addRow("Crossover frequency (MHz)", self.e_fx)
        top.addRow(self.chk_same)
        lay.addLayout(top)
        self.lp = _Channel("Lowpass channel (port 2)", self.schedule)
        self.hp = _Channel("Highpass channel (port 3)", self.schedule)
        lay.addWidget(self.lp)
        lay.addWidget(self.hp)
        self.info = QLabel()
        self.info.setTextFormat(Qt.RichText)
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.info)
        btn = QPushButton("Export S3P…")
        btn.setToolTip("3-port Touchstone file (dB/angle): 1 = common, 2 = lowpass, 3 = highpass")
        btn.clicked.connect(self.export_s3p)
        lay.addWidget(btn)
        lay.addStretch(1)
        for e in (self.e_r0,):
            e.textChanged.connect(self.schedule)
        self.e_fx.textChanged.connect(self.schedule)
        self.chk_same.toggled.connect(lambda *_: (self._sync(), self.schedule()))
        return w

    def _build_right(self):
        """Both channel schematics over the 3-port response plot."""
        pal = self.palette()
        bg, fg = pal.base().color(), pal.text().color()
        self.sch_lp = SchematicView()
        self.sch_hp = SchematicView()
        sch_box = QWidget()
        v = QVBoxLayout(sch_box)
        for title, view in (("Lowpass channel: common port (1) → port 2", self.sch_lp),
                            ("Highpass channel: common port (1) → port 3", self.sch_hp)):
            lab = QLabel(f"<b>{title}</b>")
            v.addWidget(lab)
            sc = QScrollArea()
            sc.setWidget(view)
            sc.setWidgetResizable(True)
            sc.setFrameShape(QScrollArea.NoFrame)
            sc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setMinimumHeight(270)
            v.addWidget(sc)
        self.plot = pg.PlotWidget(background=bg)
        pi = self.plot.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.25)
        pi.setLabel("bottom", "Frequency", units="Hz")
        pi.setLabel("left", "dB")
        for ax in ("left", "bottom"):
            pi.getAxis(ax).setTextPen(fg)
            pi.getAxis(ax).setPen(fg)
        pi.addLegend(offset=(-10, 10))
        self.curves = {
            "S21": self.plot.plot(name="S21 common → lowpass", pen=pg.mkPen("#2f7ed8", width=2)),
            "S31": self.plot.plot(name="S31 common → highpass", pen=pg.mkPen("#2ca02c", width=2)),
            "S11": self.plot.plot(name="S11 common port", pen=pg.mkPen("#e4572e", width=1.5)),
            "S22": self.plot.plot(name="S22 lowpass port", pen=pg.mkPen("#9b59b6", width=1.5)),
            "S33": self.plot.plot(name="S33 highpass port", pen=pg.mkPen("#f1c40f", width=1.5)),
            "S32": self.plot.plot(name="S32 isolation", pen=pg.mkPen((150, 150, 150), width=1.2,
                                                                      style=Qt.DashLine)),
        }
        self.plot.setYRange(-80, 2)
        tabs = QTabWidget()
        tabs.addTab(self.plot, "Response")
        split = QSplitter(Qt.Vertical)
        split.addWidget(sch_box)
        split.addWidget(tabs)
        split.setSizes([500, 400])
        return split

    # ------------------------------------------------------------------ spec
    def _sync(self):
        """With a common crossover the per-channel 3 dB fields follow the crossover."""
        same = self.chk_same.isChecked()
        for ch in (self.lp, self.hp):
            ch.l_f3.setVisible(not same)
            ch.f3.setVisible(not same)

    def spec(self) -> DiplexSpec:
        """Read the inputs into a DiplexSpec."""
        fx = self.e_fx.value()
        same = self.chk_same.isChecked()
        lp_f = fx if same else self.lp.f3.value()
        hp_f = fx if same else self.hp.f3.value()
        if not fx or not lp_f or not hp_f:
            raise ValueError("enter the crossover / 3 dB frequencies")
        return DiplexSpec(r0=parse_value(self.e_r0.text()) or 50.0,
                          lp_shape=self.lp.shape.currentText(), lp_order=self.lp.order.value(),
                          lp_fc=lp_f, lp_ripple=self.lp.ripple.value(), lp_amin=self.lp.amin.value(),
                          hp_shape=self.hp.shape.currentText(), hp_order=self.hp.order.value(),
                          hp_fc=hp_f, hp_ripple=self.hp.ripple.value(), hp_amin=self.hp.amin.value())

    # ------------------------------------------------------------------ compute
    def schedule(self, *_):
        self.timer.start()

    def _launch(self):
        """Start a design job for the current inputs (called by the debounce timer)."""
        try:
            spec = self.spec()
        except Exception as e:
            self.statusBar().showMessage(f"Input: {e}")
            return
        self.gen += 1
        job = _Job(self.gen, spec)
        job.signals.done.connect(self._done)
        job.signals.failed.connect(self._failed)
        self.statusBar().showMessage("Computing…")
        self.pool.start(job)

    def _failed(self, gen, msg):
        if gen == self.gen:
            self.statusBar().showMessage(f"Cannot design: {msg.splitlines()[0]}")
            self.info.setText(f"<span style='color:#c0392b'>{msg.splitlines()[0]}</span>")

    def _freqs(self, n=1201):
        """Plot sweep: up to three times the higher channel frequency."""
        sp = self.result.spec
        top = 3 * max(sp.lp_fc, sp.hp_fc)
        return np.linspace(top * 1e-3, top, n)

    def _done(self, gen, dx):
        """Show a finished diplexer: schematics, curves and the summary."""
        if gen != self.gen:
            return
        self.result = dx
        self.sch_lp.set_ladder(dx.lp)
        self.sch_hp.set_ladder(dx.hp, ports=("1", "3"))
        f = self._freqs()
        sp = dx.sparams(f)
        for k, c in self.curves.items():
            c.setData(f, 20 * np.log10(np.abs(sp[k]) + 1e-300))
        self.plot.setXRange(f[0], f[-1], padding=0)
        inf = dx.info
        kind = ("<b>exact</b> constant-resistance pair (complementary Butterworth)" if inf["exact"]
                else "approximately constant-resistance pair")
        rows = [kind,
                f"Worst common-port return loss in the passbands: <b>{inf['worst_return_loss_db']:.1f} dB</b>",
                f"Crossover: <b>{eng(inf['crossover'], 'Hz')}</b>; isolation there "
                f"{20 * np.log10(abs(dx.sparams([inf['crossover']])['S32'][0])):.1f} dB",
                "Common-port ends: series L (lowpass) and series C (highpass)"]
        for name, lad in (("Lowpass", dx.lp), ("Highpass", dx.hp)):
            ci = lad.notes.get("info", {})
            if ci.get("zeros"):
                rows.append(f"{name} notches: " + ", ".join(eng(z, "Hz") for z in ci["zeros"])
                            + f" (Amin {ci['amin_db']:.1f} dB)")
        self.info.setText("<br>".join(rows))
        self.statusBar().showMessage("Diplexer designed")

    # ------------------------------------------------------------------ actions
    def export_s3p(self):
        """Save the 3-port S-parameters as Touchstone .s3p over 0 .. 3x the crossover."""
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export S3P (Touchstone)", "diplexer.s3p",
                                              "Touchstone (*.s3p)")
        if not path:
            return
        f = self._freqs(1001)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(touchstone_s3p(self.result, f, title="VektorFIL diplexer"))
        self.statusBar().showMessage(f"Saved {path}")


def main():
    """Start the diplexer designer on its own."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VektorFIL")
    w = DiplexerWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
