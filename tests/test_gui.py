"""Smoke test of the GUI: the window builds and designs a filter (runs off-screen)."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def test_window_designs_a_filter():
    from PySide6.QtCore import QEventLoop

    from vektorfil.gui.filter_window import FilterWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FilterWindow()
    w.show()
    t0 = time.time()
    while w.result is None and time.time() - t0 < 30:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.02)
    assert w.result is not None, w.statusBar().currentMessage()
    # switch to a coupled bandpass with edge correction off and check it redesigns
    w.select(ftype="Bandpass", circuit="Shunt resonators, capacitive coupling")
    w.chk_edges.setChecked(False)
    first = w.result
    w.schedule()
    t0 = time.time()
    while (w.result is first or w.timer.isActive()) and time.time() - t0 < 30:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.02)
    assert w.result.spec.subtype == "Shunt resonators, capacitive coupling"
    assert not w.result.spec.edge_correct
    # the live order estimate is filled in
    assert "Chebyshev" in w.l_est.text()
    w.close()
