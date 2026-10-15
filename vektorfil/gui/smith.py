"""Smith chart of the input reflection S11.

The chart is the reflection-coefficient plane (|Gamma| <= 1) with the usual grid:
circles of constant normalised resistance r (centre r/(1+r), radius 1/(1+r)) and arcs
of constant reactance x (centre (1, 1/x), radius 1/|x|).  Hovering shows the nearest
point of the sweep with its frequency, |S11| and the input impedance
Z = Z0 (1 + Gamma) / (1 - Gamma).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from vektorfil.gui.plotstyle import C_S11, faded

R_GRID = (0.2, 0.5, 1.0, 2.0, 5.0)
X_GRID = (0.2, 0.5, 1.0, 2.0, 5.0)


def _eng(x, unit):
    from vektorfil.gui.schematic import eng
    return eng(x, unit)


class SmithChart(pg.PlotWidget):
    """S11 on a Smith chart, with a hover readout."""
    def __init__(self, fg, bg):
        super().__init__(background=bg)
        pi = self.getPlotItem()
        pi.hideAxis("left")
        pi.hideAxis("bottom")
        pi.setMouseEnabled(False, False)
        pi.hideButtons()
        self.setAspectLocked(True)
        grid = pg.mkPen(faded(fg, 70), width=0.8)
        rim = pg.mkPen(faded(fg, 160), width=1.2)
        t = np.linspace(0, 2 * np.pi, 721)
        self.plot(np.cos(t), np.sin(t), pen=rim)
        self.plot([-1, 1], [0, 0], pen=grid)
        for r in R_GRID:
            c, rad = r / (1 + r), 1 / (1 + r)
            self.plot(c + rad * np.cos(t), rad * np.sin(t), pen=grid)
            lab = pg.TextItem(f"{r:g}", color=faded(fg, 150), anchor=(1.1, -0.1))
            lab.setPos(c - rad, 0)
            self.addItem(lab)
        for x in X_GRID:
            for sx in (x, -x):
                gx = 1 + (1 / abs(sx)) * np.cos(t)
                gy = 1 / sx + (1 / abs(sx)) * np.sin(t)
                inside = gx * gx + gy * gy <= 1.0 + 1e-9
                self.plot(np.where(inside, gx, np.nan), np.where(inside, gy, np.nan), pen=grid, connect="finite")
                # label where the arc meets the rim: Gamma = (jx - 1)/(jx + 1)
                g = (1j * sx - 1) / (1j * sx + 1)
                lab = pg.TextItem(f"{'+' if sx > 0 else '−'}j{x:g}", color=faded(fg, 150), anchor=(0.5, 0.5))
                lab.setPos(1.1 * g.real, 1.1 * g.imag)
                self.addItem(lab)
        self.curve = self.plot(pen=pg.mkPen(C_S11, width=2))
        self.dot = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(C_S11), pen=pg.mkPen(fg))
        self.dot.setZValue(20)
        box = pg.mkColor(bg)
        box.setAlpha(215)
        self.label = pg.TextItem(color=fg, fill=pg.mkBrush(box), border=pg.mkPen(faded(fg, 90)), anchor=(0, 0))
        self.label.setZValue(30)
        self.addItem(self.dot)
        self.addItem(self.label)
        self.setRange(xRange=(-1.2, 1.2), yRange=(-1.2, 1.2), padding=0)
        self.f = self.gamma = None
        self.z0 = 50.0
        self.scene().sigMouseMoved.connect(self._hover)
        self._hide()

    def set_data(self, f, s11, z0):
        """Show the S11 sweep (complex) over frequencies f, referenced to z0."""
        self.f, self.gamma, self.z0 = np.asarray(f), np.asarray(s11), z0
        self.curve.setData(self.gamma.real, self.gamma.imag)
        self._hide()

    def _hide(self):
        self.dot.setVisible(False)
        self.label.setVisible(False)

    def _hover(self, pos):
        """Mouse moved: show the nearest sweep point with its frequency and impedance."""
        vb = self.getPlotItem().vb
        if self.gamma is None or not vb.sceneBoundingRect().contains(pos):
            self._hide()
            return
        p = vb.mapSceneToView(pos)
        g0 = complex(p.x(), p.y())
        i = int(np.argmin(np.abs(self.gamma - g0)))
        g = self.gamma[i]
        if abs(g - g0) > 0.15:
            self._hide()
            return
        z = self.z0 * (1 + g) / (1 - g) if abs(1 - g) > 1e-12 else complex(np.inf)
        sign = "+" if z.imag >= 0 else "−"
        self.dot.setData([g.real], [g.imag])
        self.label.setText(f"{_eng(self.f[i], 'Hz')}\nS11 {20 * np.log10(abs(g) + 1e-300):.2f} dB\n"
                           f"Z = {z.real:.4g} {sign} j{abs(z.imag):.4g} Ω")
        self.label.setPos(-1.18, 1.18)
        self.dot.setVisible(True)
        self.label.setVisible(True)

    def mark(self, i):
        """Put the dot on sweep point i (used to follow the response-plot cursor)."""
        if self.gamma is None or not 0 <= i < len(self.gamma):
            return
        g = self.gamma[i]
        self.dot.setData([g.real], [g.imag])
        self.dot.setVisible(True)

