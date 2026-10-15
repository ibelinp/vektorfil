"""The Monte Carlo tab: tolerance spread of the current design.

Controls (runs, inductor and capacitor tolerance, distribution) and a Run button over a
plot of the nominal S21/S11 with the band between the lowest and highest value of all
runs at each frequency.  The runs are computed in a background thread of their own
(plain numpy, independent of the synthesis thread).  A new design clears the result,
since it belongs to the old one.
"""
from __future__ import annotations

import traceback

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from vektorfil.analysis import montecarlo
from vektorfil.gui.plotstyle import C_S11, C_S21, faded, new_plot


class _Signals(QObject):
    """Signals of a background job: done(generation, result) or failed(generation, message)."""
    done = Signal(int, object)
    failed = Signal(int, str)


class _Job(QRunnable):
    """Runs montecarlo.run() in the tab's own background thread."""
    def __init__(self, gen, lad, f, kw):
        super().__init__()
        self.gen, self.lad, self.f, self.kw, self.signals = gen, lad, f, kw, _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.gen, montecarlo.run(self.lad, self.f, **self.kw))
        except Exception as e:  # shown in the tab
            self.signals.failed.emit(self.gen, f"{e}\n{traceback.format_exc(limit=2)}")


def passband(spec, info, f):
    """Boolean mask of the frequencies f inside the passband of a design."""
    if spec.ftype == "Lowpass":
        return f <= spec.fc
    if spec.ftype == "Highpass":
        return f >= spec.fc
    lo, hi = info.get("edges", (spec.fc - spec.bw / 2, spec.fc + spec.bw / 2))
    if spec.ftype == "Bandpass":
        return (f >= lo) & (f <= hi)
    return (f <= lo) | (f >= hi)


class MonteCarloTab(QWidget):
    """Monte Carlo controls and plot.  `window` supplies .result and ._freqs()."""
    def __init__(self, window, fg, bg):
        super().__init__()
        self.window = window
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.gen = 0
        col = QVBoxLayout(self)
        row = QHBoxLayout()
        self.s_runs = QSpinBox()
        self.s_runs.setRange(10, 20000)
        self.s_runs.setValue(200)
        self.s_runs.setSingleStep(50)
        self.e_tol_l = QDoubleSpinBox()
        self.e_tol_c = QDoubleSpinBox()
        for e in (self.e_tol_l, self.e_tol_c):
            e.setRange(0.0, 50.0)
            e.setDecimals(1)
            e.setSingleStep(0.5)
            e.setValue(5.0)
            e.setSuffix(" %")
        self.c_dist = QComboBox()
        self.c_dist.addItems(["Uniform", "Gaussian (tolerance = 3σ)"])
        self.btn = QPushButton("Run")
        self.btn.clicked.connect(self.run)
        for w in (QLabel("Runs"), self.s_runs, QLabel("  L ±"), self.e_tol_l, QLabel("  C ±"), self.e_tol_c,
                  QLabel("  "), self.c_dist, self.btn):
            row.addWidget(w)
        row.addStretch(1)
        col.addLayout(row)
        self.stats = QLabel("Set the tolerances and press Run.")
        col.addWidget(self.stats)
        self.plot = new_plot(bg, fg)
        self.plot.setYRange(-80, 2)
        self.nom21 = self.plot.plot(name="S21 nominal", pen=pg.mkPen(C_S21, width=2))
        self.nom11 = self.plot.plot(name="S11 nominal", pen=pg.mkPen(C_S11, width=1.5))
        self.bands = []
        for colour in (C_S21, C_S11):
            lo = pg.PlotCurveItem(pen=pg.mkPen(faded(colour, 120), width=0.8))
            hi = pg.PlotCurveItem(pen=pg.mkPen(faded(colour, 120), width=0.8))
            fill = pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(faded(colour, 60)))
            for item in (lo, hi, fill):
                self.plot.addItem(item)
            self.bands.append((lo, hi))
        col.addWidget(self.plot, 1)
        self.result = None

    def clear(self):
        """Forget the spread (the design changed)."""
        self.gen += 1
        for lo, hi in self.bands:
            lo.setData([], [])
            hi.setData([], [])
        for c in (self.nom21, self.nom11):
            c.setData([], [])
        self.result = None
        self.btn.setEnabled(True)
        self.stats.setText("Set the tolerances and press Run.")

    def settings(self):
        """Current settings, for design files."""
        return {"runs": self.s_runs.value(), "tol_l": self.e_tol_l.value(), "tol_c": self.e_tol_c.value(),
                "dist": self.c_dist.currentIndex()}

    def apply_settings(self, d):
        """Restore settings saved by settings()."""
        self.s_runs.setValue(int(d.get("runs", 200)))
        self.e_tol_l.setValue(float(d.get("tol_l", 5.0)))
        self.e_tol_c.setValue(float(d.get("tol_c", 5.0)))
        self.c_dist.setCurrentIndex(int(d.get("dist", 0)))

    def run(self):
        """Start the runs for the current design over the current plot range."""
        res = self.window.result
        if res is None:
            return
        self.gen += 1
        f = self.window._freqs()
        kw = {"runs": self.s_runs.value(), "tol_l": self.e_tol_l.value() / 100,
              "tol_c": self.e_tol_c.value() / 100,
              "dist": "gaussian" if self.c_dist.currentIndex() == 1 else "uniform"}
        job = _Job(self.gen, res.ladder, f, kw)
        job.signals.done.connect(self._done)
        job.signals.failed.connect(self._failed)
        self.btn.setEnabled(False)
        self.stats.setText(f"Running {kw['runs']} variations…")
        self._design = res
        self.pool.start(job)

    def _failed(self, gen, msg):
        if gen == self.gen:
            self.btn.setEnabled(True)
            self.stats.setText(f"<span style='color:#c0392b'>{msg.splitlines()[0]}</span>")

    def _done(self, gen, mc):
        """Draw the spread and the passband statistics."""
        if gen != self.gen:
            return
        self.btn.setEnabled(True)
        self.result = mc
        f = mc.f
        self.nom21.setData(f, mc.s21_nom)
        self.nom11.setData(f, mc.s11_nom)
        for (lo, hi), (a, b) in zip(self.bands, ((mc.s21_min, mc.s21_max), (mc.s11_min, mc.s11_max))):
            lo.setData(f, a)
            hi.setData(f, b)
        self.plot.setXRange(f[0], f[-1], padding=0)
        self.plot.setYRange(max(-120.0, float(np.min(mc.s21_min)) - 5), 2)
        res = self._design
        st = mc.passband_stats(passband(res.spec, res.info, f))
        txt = f"{mc.runs} runs, L ±{self.e_tol_l.value():g} %, C ±{self.e_tol_c.value():g} % " \
              f"({self.c_dist.currentText().split(' ')[0].lower()})."
        if st:
            txt += (f"  In the passband: worst S21 <b>{st['s21_worst']:.2f} dB</b> (nominal {st['s21_nom']:.2f}), "
                    f"worst S11 <b>{st['s11_worst']:.1f} dB</b> (nominal {st['s11_nom']:.1f}).")
        self.stats.setText(txt)
