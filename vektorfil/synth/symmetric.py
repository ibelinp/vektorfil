"""Symmetry-preserving bandpass (symmetric lowpass-to-bandpass transform topology).

A conventional lowpass->bandpass transform keeps geometric symmetry: on a linear
frequency axis the upper skirt is much weaker than the lower one, increasingly so for
wider bandwidths.  Carassa showed that good arithmetic symmetry needs roughly three
times as many transmission zeros at infinity as at DC.  The symmetric topology gets
there by replacing the first and then every other shunt parallel resonator of the
conventional (minimum capacitor) bandpass with a series resonator between two shunt
capacitors acting as impedance inverters -- each inverter capacitor adds a zero at
infinity:

    N=3   conventional  S P S        symmetric  S C S C S
    N=4                 S P S P                 S C S C S P
    N=5                 S P S P S               S C S C S P S
    (S = series resonator, P = shunt parallel resonator, C = shunt inverter capacitor)

No exact equiripple realization exists in this topology (the transfer-function class
does not admit it), and published closed-form methods are approximate.  Here:

  1. initial values from inverter theory: a parallel resonator of susceptance slope b
     equals a series resonator of reactance slope x between two inverters with
     K = sqrt(x/b); the inverters are shunt capacitors C = 1/(w0 K) whose negative
     series reactances are absorbed into the neighbouring series resonators;
  2. refinement: the capacitors (and the inductors of the remaining parallel
     resonators) are fitted by least squares to the *arithmetically* mapped lowpass
     response, Omega = (f - f0)/(BW/2), so the band edges sit at f0 +- BW/2 and the
     skirts are as symmetric as the topology allows.  The series inductors stay as
     designed (all equal for N = 3).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .bandxform import transform
from .ladder import Branch, LadderResult
from .network import Ladder


def topology(N):
    """Branch pattern of the symmetric bandpass, e.g. 'SCSCS' for N = 3."""
    conv = "".join("S" if i % 2 == 0 else "P" for i in range(N))
    out, p = [], 0
    for i, c in enumerate(conv):
        if c == "P":
            p += 1
            if p % 2 == 1:
                # a replaced P at the very end needs no output inverter (N = 2: S C S)
                out.append("CS" if i == N - 1 else "CSC")
            else:
                out.append("P")
        else:
            out.append("S")
    return "".join(out)


def initial(res: LadderResult, r0, f0, bw):
    """Inverter-based initial design from a series-first lowpass prototype (S P S ...).

    res must start with a series element (minimum-capacitor bandpass after transform)."""
    conv = transform(res, r0, f0, bw, "bp")
    w0 = 2 * math.pi * f0
    items = []
    p_count = 0
    for b in conv.branches:
        if b.kind == "LC_ser":
            items.append({"t": "S", "L": b.L, "C": b.C})
        else:
            p_count += 1
            last = b is conv.branches[-1]
            if p_count % 2 == 1 and last:
                # last element: [inverter, series resonator, load R].  The inverter must be
                # K = R_load so the load still appears as R; then x = K^2 b fixes the L.
                K = conv.r_load
                L_s = K * K * b.C
                items += [{"t": "K", "K": K}, {"t": "S", "L": L_s, "C": None}]
            elif p_count % 2 == 1:                  # first and then every other P
                b_slope = w0 * b.C                  # susceptance slope of the parallel resonator
                L_s = items[-1]["L"]                # same L as the preceding series resonator
                K = math.sqrt(w0 * L_s / b_slope)   # inverter that turns x into b
                items += [{"t": "K", "K": K}, {"t": "S", "L": L_s, "C": None}, {"t": "K", "K": K}]
            else:
                items.append({"t": "P", "L": b.L, "C": b.C})
    # each inverter (shunt C) carries a negative series reactance -K on both sides; the
    # neighbouring series resonators absorb it: w0 L - 1/(w0 C) = sum of adjacent K
    for i, it in enumerate(items):
        if it["t"] != "S":
            continue
        xr = sum(items[j]["K"] for j in (i - 1, i + 1)
                 if 0 <= j < len(items) and items[j]["t"] == "K")
        d = w0 * it["L"] - xr
        if d <= 0:
            raise ValueError("bandwidth too wide for the symmetric topology")
        it["C"] = 1 / (w0 * d)
    out = []
    for it in items:
        if it["t"] == "S":
            out.append(Branch("series", "LC_ser", L=it["L"], C=it["C"]))
        elif it["t"] == "P":
            out.append(Branch("shunt", "LC_par", L=it["L"], C=it["C"]))
        else:
            out.append(Branch("shunt", "C", C=1 / (w0 * it["K"])))
    return Ladder(out, r0, conv.r_load)


def refine(lad: Ladder, target_db, f0, bw, max_nfev_per_var=200):
    """Least-squares fit of the free element values to an arithmetic target response.

    target_db(Omega) -> loss in dB of the lowpass prototype at |Omega| (Omega = 1 at the
    band edges).  Free values: every capacitor, and the inductors of parallel resonators."""
    span = bw / 2
    f = np.concatenate([np.linspace(f0 - span, f0 + span, 241),
                        np.linspace(max(f0 - 2.2 * span, f0 * 0.02), f0 - 1.001 * span, 60),
                        np.linspace(f0 + 1.001 * span, f0 + 2.2 * span, 60)])
    om = np.abs(f - f0) / span
    tgt = -np.array([target_db(x) for x in om])
    tgt_c = np.maximum(tgt, -45.0)                   # do not chase deep stopband values
    weight = np.where(om <= 1, 30.0, 1.0)            # the passband matters most
    slots = [(i, "C") for i, b in enumerate(lad.branches) if b.C is not None]
    slots += [(i, "L") for i, b in enumerate(lad.branches) if b.kind == "LC_par"]
    x0 = np.log([getattr(lad.branches[i], a) for i, a in slots])

    def resid(x):
        for (i, a), v in zip(slots, np.exp(x)):
            setattr(lad.branches[i], a, float(v))
        a = 20 * np.log10(np.abs(lad.sparams(f)[1]) + 1e-300)
        return weight * (np.maximum(a, -45.0) - tgt_c)

    r = least_squares(resid, x0, max_nfev=max_nfev_per_var * len(x0), xtol=1e-12, ftol=1e-12)
    resid(r.x)
    return lad


def design_symmetric(res: LadderResult, target_db, r0, f0, bw, title=""):
    """Full symmetric bandpass: initial inverter design + refinement.
    res: series-first lowpass prototype ladder; target_db: its loss vs normalised frequency;
    f0: arithmetic centre; bw: bandwidth between the edges (at the prototype's w = 1)."""
    lad = initial(res, r0, f0, bw)
    refine(lad, target_db, f0, bw)
    if any((b.C is not None and b.C <= 0) or (b.L is not None and b.L <= 0) for b in lad.branches):
        raise ValueError("the symmetric refinement produced a negative element")
    lad.title = title
    lad.notes.update(f0=f0, bw=bw, band="bp", topology=topology(len(res.branches)))
    return lad
