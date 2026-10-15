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


def test_window_features(tmp_path):
    """Stepped mask, markers, time domain, Monte Carlo and a design file round trip."""
    from PySide6.QtCore import QEventLoop

    from vektorfil.gui.filter_window import FilterWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FilterWindow()
    w.show()

    def wait(cond, t=60):
        t0 = time.time()
        while not cond() and time.time() - t0 < t:
            app.processEvents(QEventLoop.AllEvents, 50)
            time.sleep(0.02)
        return cond()

    def designed():
        first = w.result
        w.schedule()
        return wait(lambda: w.result is not first and not w.timer.isActive())

    assert wait(lambda: w.result is not None)
    w.select(ftype="Lowpass", approximation="Elliptic")
    w.c_espec.setCurrentIndex(2)                      # stepped mask
    w.s_order.setValue(7)
    w.e_mask.setText("125-200:40, 200-:50")
    assert designed()
    assert w.result.info["mask_margin_db"] > 0
    w.markers = [100e6, 150e6]
    w._draw_markers()
    assert "150 MHz" in w.mark_box.textItem.toPlainText()
    w.tabs.setCurrentWidget(w.plot_td)                # time domain is computed on demand
    app.processEvents()
    assert len(w.c_step.getData()[0]) > 10
    w.mc_tab.s_runs.setValue(20)
    w.mc_tab.run()
    assert wait(lambda: w.mc_tab.result is not None)
    path = tmp_path / "d.vfil"
    w._write_design(path)
    spec = w.spec()
    w.select(ftype="Bandpass", approximation="Chebyshev")
    w.load_design(path)
    assert w.spec() == spec and w.markers == [100e6, 150e6]
    w.close()


def test_markers_outside_the_plot_range_are_dropped():
    from PySide6.QtCore import QEventLoop

    from vektorfil.gui.filter_window import FilterWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FilterWindow()
    t0 = time.time()
    while w.result is None and time.time() - t0 < 30:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.02)
    w.markers = [50e6, 100e6, 5e9]          # the last is far beyond the 100 MHz lowpass plot
    w._replot()
    assert w.markers == [50e6, 100e6]
    assert "5 GHz" not in w.mark_box.textItem.toPlainText()
    w.close()
