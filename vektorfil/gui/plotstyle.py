"""Shared plot colours and axis styling, so every plot in the program looks alike."""
from __future__ import annotations

import pyqtgraph as pg

C_S21 = "#2f7ed8"         # transmission
C_S11 = "#e4572e"         # reflection
C_IDEAL = (140, 140, 140)  # ideal (unrounded) response, grid-like grey
C_EXTRA = "#f1a33c"       # a third curve (impulse response, mask)
C_MARK = "#f5d33b"        # markers


def new_plot(bg, fg, left="dB", bottom="Frequency", bottom_units="Hz", legend=True):
    """A PlotWidget with grid, labelled axes in the theme colours and a legend."""
    pw = pg.PlotWidget(background=bg)
    pi = pw.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.25)
    pi.setLabel("bottom", bottom, units=bottom_units)
    pi.setLabel("left", left)
    for ax in ("left", "bottom"):
        pi.getAxis(ax).setTextPen(fg)
        pi.getAxis(ax).setPen(fg)
    if legend:
        pi.addLegend(offset=(-10, 10))
    return pw


def faded(colour, alpha):
    """The colour with an alpha (0..255), for fills."""
    c = pg.mkColor(colour)
    c.setAlpha(alpha)
    return c
