"""Synthesize a 7th-order elliptic lowpass (0.05 dB ripple, 60 dB Amin, 100 MHz, 50 ohm)
and a Chebyshev of the same order, then plot both from circuit simulation."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vektorfil.synth import Ladder, chebyshev_with_zeros, elliptic, synthesize  # noqa: E402

FC, R0 = 100e6, 50.0
designs = {
    "Elliptic N=7, 0.05 dB, Amin 60 dB": synthesize(elliptic(7, 0.05, amin_db=60)),
    "Chebyshev N=7, 0.05 dB": synthesize(chebyshev_with_zeros(7, 0.05)),
}

f = np.linspace(1e6, 300e6, 3000)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for name, res in designs.items():
    lad = Ladder.from_prototype(res, r0=R0, fc=FC, title=name)
    r = lad.response(f)
    ax1.plot(f / 1e6, r["s21_db"], label=name + " S21")
    ax1.plot(f / 1e6, r["s11_db"], "--", lw=0.8, label=name + " S11")
    ax2.plot(f / 1e6, r["gd"] * 1e9, label=name)
    print(f"\n=== {name}\n{lad.table()}")

ax1.set_ylim(-90, 2)
ax1.set_ylabel("dB")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)
ax2.set_ylim(0, 40)
ax2.set_ylabel("group delay (ns)")
ax2.set_xlabel("MHz")
ax2.grid(True, alpha=0.3)
fig.suptitle("VektorFIL synthesis: 100 MHz lowpass, 50 ohm")
out = Path(__file__).with_suffix(".png")
fig.tight_layout()
fig.savefig(out, dpi=120)
print("\nsaved", out)
