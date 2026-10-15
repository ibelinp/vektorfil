"""Monte Carlo tolerance analysis: how much the response of a design spreads when every
inductor and capacitor is off by a random amount within its tolerance.

Each run multiplies every L by (1 + dL) and every C by (1 + dC), drawn independently
per component:

    uniform    dL in [-tol_L, +tol_L]            (worst case for a given tolerance)
    gaussian   dL ~ N(0, tol_L/3), clipped to +-tol_L   (tolerance taken as 3 sigma)

Transmission-line resonators keep their line; only their loading capacitor varies.
Component Q, if set, is kept.  The envelope (minimum and maximum over the runs at each
frequency) is what the GUI draws; the passband statistics summarise it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from vektorfil.synth.network import Ladder


@dataclass
class MonteCarloResult:
    """Envelopes in dB over all runs, per frequency, and the nominal curves."""
    f: np.ndarray
    runs: int
    s21_nom: np.ndarray
    s11_nom: np.ndarray
    s21_min: np.ndarray
    s21_max: np.ndarray
    s11_min: np.ndarray
    s11_max: np.ndarray
    s21_runs: np.ndarray        # (runs, len(f)), for statistics

    def passband_stats(self, in_pass):
        """Worst S21 and S11 (dB) in the passband over all runs, and for the nominal
        design; in_pass is a boolean mask over f."""
        if not np.any(in_pass):
            return None
        return {"s21_worst": float(np.min(self.s21_min[in_pass])),
                "s21_nom": float(np.min(self.s21_nom[in_pass])),
                "s11_worst": float(np.max(self.s11_max[in_pass])),
                "s11_nom": float(np.max(self.s11_nom[in_pass]))}


def _factors(rng, n, tol, dist):
    """n random multipliers 1 + d with |d| <= tol (uniform, or 3-sigma gaussian)."""
    if tol <= 0:
        return np.ones(n)
    if dist == "gaussian":
        return 1 + np.clip(rng.normal(0, tol / 3, n), -tol, tol)
    return 1 + rng.uniform(-tol, tol, n)


def perturbed(lad: Ladder, rng, tol_l: float, tol_c: float, dist: str = "uniform") -> Ladder:
    """A copy of the ladder with every L and C varied once within its tolerance."""
    fl = iter(_factors(rng, len(lad.branches), tol_l, dist))
    fc = iter(_factors(rng, len(lad.branches), tol_c, dist))
    out = []
    for b in lad.branches:
        kl, kc = next(fl), next(fc)
        L = b.L * kl if (b.L is not None and b.kind != "TEM") else b.L
        C = b.C * kc if b.C is not None else None
        out.append(replace(b, L=L, C=C))
    return Ladder(out, lad.r_source, lad.r_load, lad.title, dict(lad.notes), lad.q_l, lad.q_c)


def run(lad: Ladder, f, runs: int = 200, tol_l: float = 0.05, tol_c: float = 0.05,
        dist: str = "uniform", seed: int | None = None) -> MonteCarloResult:
    """Simulate `runs` random versions of the ladder at frequencies f (Hz)."""
    f = np.asarray(f, float)
    rng = np.random.default_rng(seed)

    def db(x):
        return 20 * np.log10(np.abs(x) + 1e-300)

    s11, s21 = lad.sparams(f)
    s21_nom, s11_nom = db(s21), db(s11)
    a21 = np.empty((runs, len(f)))
    a11 = np.empty((runs, len(f)))
    for k in range(runs):
        r11, r21 = perturbed(lad, rng, tol_l, tol_c, dist).sparams(f)
        a21[k], a11[k] = db(r21), db(r11)
    return MonteCarloResult(f, runs, s21_nom, s11_nom, a21.min(0), a21.max(0), a11.min(0), a11.max(0), a21)
