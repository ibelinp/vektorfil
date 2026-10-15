"""Regenerate the screenshots used by docs/index.html.

    python docs/make_screenshots.py

Runs the application (on Windows the window shows briefly), sets up each example and saves PNGs to docs/img/.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform != "win32":
    # the native Windows platform renders the real system fonts; off-screen rendering
    # substitutes thin fallback fonts, so it is used only where there is no choice
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vektorfil.filters import SERIES_FIRST, SHUNT_FIRST, SYMMETRIC, ZIGZAG  # noqa: E402
from vektorfil.gui.diplexer_window import DiplexerWindow  # noqa: E402
from vektorfil.gui.filter_window import FilterWindow  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv)


def settle(w, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        app.processEvents(QEventLoop.AllEvents, 30)
        time.sleep(0.01)
        m = w.statusBar().currentMessage()
        if not w.timer.isActive() and ("designed" in m or "Cannot" in m or "Diplexer" in m):
            return m
    return "TIMEOUT"


def setup(w, ftype="Lowpass", shape="Chebyshev", sub=None, order=5, fc="100", bw="10",
          ripple=0.1, tab=0):
    """Select response/approximation/circuit, fill in the values, design, show a result tab."""
    w.select(ftype=ftype, approximation=shape, circuit=sub, single=shape.endswith("terminated"))
    w.s_order.setValue(order)
    w.e_fc.setText(fc)
    w.e_bw.setText(bw)
    w.e_ripple.setValue(ripple)
    w.statusBar().showMessage("")
    w.schedule()
    msg = settle(w)
    w.tabs.setCurrentIndex(tab)
    app.processEvents()
    return msg


def shot(w, name):
    """Save the window; widen it first until no schematic needs a horizontal scrollbar."""
    from PySide6.QtWidgets import QScrollArea
    size = w.size()
    for _ in range(20):
        bars = [sa.horizontalScrollBar() for sa in w.findChildren(QScrollArea)]
        if not any(sb.isVisible() and sb.maximum() > 0 for sb in bars):
            break
        w.resize(w.width() + 80, w.height())
        app.processEvents()
        time.sleep(0.05)
        app.processEvents()
    w.grab().save(str(OUT / f"{name}.png"))
    w.resize(size)
    app.processEvents()
    print("saved", name)


w = FilterWindow()
w.resize(1440, 980)  # widened per shot when a schematic does not fit
w.show()
settle(w)

CAP = "Shunt resonators, capacitive coupling"
COAX = "Coaxial resonators, capacitive coupling"
setup(w, "Lowpass", "Chebyshev", SERIES_FIRST, 7, "70", ripple=0.25)
shot(w, "main_window")
setup(w, "Lowpass", "Elliptic", SHUNT_FIRST, 7, "100", ripple=0.1)
shot(w, "lowpass_elliptic")
setup(w, "Bandpass", "Chebyshev", CAP, 5, "100", "5", ripple=0.1)
shot(w, "bandpass_capacitive")
setup(w, "Bandpass", "Chebyshev", "Tubular", 3, "1000", "100", ripple=0.1)
shot(w, "bandpass_tubular")
setup(w, "Bandpass", "Chebyshev", "Tumbular", 3, "1000", "100", ripple=0.1)
shot(w, "bandpass_tumbular")
setup(w, "Bandpass", "Chebyshev", "Tubular", 5, "100", "20", ripple=0.1)
w.chk_coils.setChecked(True)
w.e_res.setText("800")
w.schedule()
settle(w)
shot(w, "bandpass_endcoils")
w.chk_coils.setChecked(False)
w.e_res.setText("")
w.e_theta.setValue(45)
setup(w, "Bandpass", "Chebyshev", COAX, 4, "900", "40", ripple=0.1)
shot(w, "bandpass_coaxial")
setup(w, "Bandpass", "Elliptic", ZIGZAG, 4, "100", "20", ripple=0.1)
shot(w, "bandpass_zigzag")
setup(w, "Bandpass", "Chebyshev", SYMMETRIC, 3, "100", "40", ripple=0.1)
shot(w, "bandpass_symmetric")
setup(w, "Lowpass", "Chebyshev, singly terminated", SHUNT_FIRST, 5, "100", ripple=0.1)
shot(w, "singly_terminated")
w.select(single=False)
w.c_std.setCurrentText("5% (E24)")
setup(w, "Lowpass", "Elliptic", SHUNT_FIRST, 7, "100", ripple=0.1)
shot(w, "standard_values")
w.c_std.setCurrentText("Ideal")

# stepped stopband mask with markers, then the Smith chart, time domain and Monte Carlo tabs
w.c_espec.setCurrentIndex(2)
w.e_mask.setText("125-200:40, 200-:60")
setup(w, "Lowpass", "Elliptic", SHUNT_FIRST, 8, "100", ripple=0.1)
w.markers = [100e6, 125e6, 200e6]
w._draw_markers()
app.processEvents()
shot(w, "mask_markers")
w.clear_markers()
w.c_espec.setCurrentIndex(0)
setup(w, "Bandpass", "Chebyshev", CAP, 5, "100", "10", ripple=0.1, tab=2)
shot(w, "smith_chart")
setup(w, "Lowpass", "Chebyshev", SHUNT_FIRST, 5, "100", ripple=0.5, tab=3)
shot(w, "time_domain")
setup(w, "Lowpass", "Chebyshev", SHUNT_FIRST, 7, "100", ripple=0.1, tab=4)
w.mc_tab.s_runs.setValue(500)
w.mc_tab.run()
t0 = time.time()
while w.mc_tab.result is None and time.time() - t0 < 60:
    app.processEvents(QEventLoop.AllEvents, 30)
    time.sleep(0.01)
shot(w, "monte_carlo")
w.tabs.setCurrentIndex(0)
w.open_coil()
app.processEvents()
w._coil_dlg.grab().save(str(OUT / "coil.png"))
print("saved coil")
w._coil_dlg.close()

w.range_dlg.show()
app.processEvents()
w.range_dlg.grab().save(str(OUT / "plot_range.png"))
print("saved plot_range")
w.range_dlg.hide()

d = DiplexerWindow()
d.resize(1280, 840)
d.show()
d.lp.shape.setCurrentText("Chebyshev")
d.hp.shape.setCurrentText("Chebyshev")
d.lp.order.setValue(5)
d.hp.order.setValue(5)
d.schedule()
settle(d)
shot(d, "diplexer")
