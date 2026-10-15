"""Lowpass prototype -> bandpass / bandstop by exact reactance transformation.

    bandpass:  p = (s^2 + w0^2) / (s B)        bandstop:  p = s B / (s^2 + w0^2)

Every prototype branch maps onto Foster-form branches, emitted as consecutive ladder
branches (consecutive series branches are in series, consecutive shunt branches at a
node are in parallel):

    bandpass  series L      -> series  L-C (series resonator)
              shunt C       -> shunt   L||C
              series L||C   -> two series L||C tanks          (elliptic)
              shunt L-C     -> two shunt L-C traps            (elliptic)
    bandstop  series L      -> series  L||C
              shunt C       -> shunt   L-C
              series L||C   -> two series L||C tanks
              shunt L-C     -> two shunt L-C traps
"""
from __future__ import annotations

import math

from .ladder import Branch, LadderResult
from .network import Ladder


def _split_resonator(g_l, g_c, w0, B, band):
    """A prototype resonator with zero at wz = 1/sqrt(g_l g_c), immittance
    X(p) = p g_l / (1 + p^2 g_l g_c), mapped through the band transform, becomes
    sum_i 2 k_i s / (s^2 + w_i^2).  Returns [(k_i, w_i)]."""
    wz = 1 / math.sqrt(g_l * g_c)
    if band == "bp":
        # p(jw) = +-j wz  <=>  w^2 - w0^2 = -+ wz B w
        r = math.sqrt((wz * B) ** 2 + 4 * w0 * w0)
        ws = [(r - wz * B) / 2, (r + wz * B) / 2]
    else:
        # p(jw) = j w B / (w0^2 - w^2) = +-j wz
        r = math.sqrt(B * B + 4 * (wz * w0) ** 2)
        ws = [(-B + r) / (2 * wz), (B + r) / (2 * wz)]

    def X(s):
        p = (s * s + w0 * w0) / (s * B) if band == "bp" else s * B / (s * s + w0 * w0)
        return p * g_l / (1 + p * p * g_l * g_c)

    out = []
    for wi in ws:
        # residue of X at s = j wi in the form 2 k s/(s^2 + wi^2): k = (s^2+wi^2) X / (2 s)
        vals = []
        for d in (1e-6, -1e-6):
            s = 1j * wi * (1 + d)
            vals.append(((s * s + wi * wi) * X(s) / (2 * s)).real)
        out.append((0.5 * (vals[0] + vals[1]), wi))
    return out


def transform(res: LadderResult, r0: float, f0: float, bw: float, band: str, title="") -> Ladder:
    """res: normalised lowpass prototype (1 ohm, 1 rad/s).  f0: (geometric) centre in Hz,
    bw: bandwidth in Hz that corresponds to prototype w = 1."""
    w0 = 2 * math.pi * f0
    B = 2 * math.pi * bw
    out = []
    for b in res.branches:
        gl = float(b.L) if b.L is not None else None
        gc = float(b.C) if b.C is not None else None
        if b.kind == "L":       # series L in the prototype
            if band == "bp":
                out.append(Branch("series", "LC_ser", L=gl / B, C=B / (gl * w0 * w0)))
            else:
                out.append(Branch("series", "LC_par", L=gl * B / (w0 * w0), C=1 / (gl * B)))
        elif b.kind == "C":     # shunt C in the prototype
            if band == "bp":
                out.append(Branch("shunt", "LC_par", L=B / (gc * w0 * w0), C=gc / B))
            else:
                out.append(Branch("shunt", "LC_ser", L=1 / (gc * B), C=gc * B / (w0 * w0)))
        elif b.kind == "LC_par":  # series-arm tank: Z = p L/(1 + p^2 L C)
            for k, wi in _split_resonator(gl, gc, w0, B, band):
                out.append(Branch("series", "LC_par", L=2 * k / (wi * wi), C=1 / (2 * k)))
        elif b.kind == "LC_ser":  # shunt trap: Y = p C/(1 + p^2 L C)
            for k, wi in _split_resonator(gc, gl, w0, B, band):
                out.append(Branch("shunt", "LC_ser", L=1 / (2 * k), C=2 * k / (wi * wi)))
        else:
            raise ValueError(b.kind)
    # impedance scaling to r0
    for b in out:
        if b.L is not None:
            b.L *= r0
        if b.C is not None:
            b.C /= r0
    lad = Ladder(out, r_source=float(res.r_source) * r0, r_load=float(res.r_load) * r0, title=title)
    lad.notes.update(f0=f0, bw=bw, band=band)
    return lad
