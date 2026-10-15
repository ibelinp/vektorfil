"""Lowpass/highpass diplexer from singly terminated filters.

Method (Szentirmai, S/FILSYN application note 1, "Diplexer and multiplexer design"):

  * Each channel filter is designed *voltage driven*: an ideal voltage source at the
    common end (which faces a series element) and the load R at its output.  Being
    lossless, its input conductance follows its own response:
        G * R = |V2 / E|^2.
  * The two channels are connected in parallel at the common port.  If the responses are
    complementary, |H_L|^2 + |H_H|^2 = 1, the conductances add up to exactly 1/R and --
    for these minimum-susceptance networks -- the susceptances cancel: the common port is
    then perfectly matched at every frequency.  Butterworth lowpass and highpass of equal
    order with a common 3 dB crossover are complementary (an exact diplexer).
  * Any other pair is only approximately constant-resistance.  The channels are specified
    by their 3 dB frequencies: where both are 3 dB down their conductances add up to about
    1/R; a common ripple-edge crossover would overload the common port (G R ~ 1.8).  A shunt series-L-C "susceptance corrector" across
    the common port, resonant near the crossover, cancels most of the leftover
    susceptance; its two values are optimised here for the best common-port match.

Ports: 1 = common, 2 = lowpass output, 3 = highpass output; all terminated in R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from vektorfil.synth.ladder import Branch
from vektorfil.synth.network import Ladder

CHANNEL_SHAPES = ("Butterworth", "Chebyshev", "Elliptic")


@dataclass
class DiplexSpec:
    """What the diplexer window specifies: impedance and, per channel, approximation,
    order, 3 dB frequency, ripple and (elliptic) stopband attenuation."""
    r0: float = 50.0
    lp_shape: str = "Butterworth"
    lp_order: int = 5
    lp_fc: float = 100e6          # lowpass 3 dB frequency (the crossover for a matched pair)
    lp_ripple: float = 0.1
    lp_amin: float = 50.0         # elliptic channels: minimum stopband attenuation
    hp_shape: str = "Butterworth"
    hp_order: int = 5
    hp_fc: float = 100e6          # highpass 3 dB frequency
    hp_ripple: float = 0.1
    hp_amin: float = 50.0
    corrector: bool = False       # add an optimised shunt series-LC across the common port


@dataclass
class Diplexer:
    """A designed diplexer: the two channel ladders, both drawn from the common port,
    and their joint 3-port analysis (sparams, touchstone_s3p)."""
    spec: DiplexSpec
    lp: Ladder                    # common port -> lowpass output (load r0)
    hp: Ladder                    # common port -> highpass output (load r0)
    corr: tuple | None = None     # (L, C) of the shunt series-LC corrector, or None
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------ analysis
    @staticmethod
    def _yin(lad, f, r):
        """Input admittance of a channel ladder (from the common end) loaded by r."""
        M = lad.abcd(f)
        A, B, C, D = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
        return (C * r + D) / (A * r + B), (A, B, C, D)

    def _ycorr(self, f):
        """Admittance of the optional shunt series-LC corrector at the common port."""
        if not self.corr:
            return np.zeros(len(np.atleast_1d(f)), complex)
        L, C = self.corr
        s = 1j * 2 * np.pi * np.atleast_1d(np.asarray(f, float))
        # a tiny series loss keeps the admittance finite exactly at resonance
        return 1 / (s * L + 1 / (s * C) + 1e-9 * self.spec.r0)

    def sparams(self, f):
        """dict of complex S11, S21 (common->LP out), S31 (common->HP out),
        S32 (isolation between the outputs), S22, S33."""
        f = np.atleast_1d(np.asarray(f, float))
        r = self.spec.r0
        y_l, (Al, Bl, Cl, Dl) = self._yin(self.lp, f, r)
        y_h, (Ah, Bh, Ch, Dh) = self._yin(self.hp, f, r)
        y_c = self._ycorr(f)
        # common port driven by E with source r
        yt = y_l + y_h + y_c
        zt = 1 / yt
        s11 = (zt - r) / (zt + r)
        vj = zt / (r + zt)                      # junction voltage per unit E
        v2 = vj / (Al + Bl / r)                 # V2 = Vj / (A + B/R)
        v3 = vj / (Ah + Bh / r)
        s21, s31 = 2 * v2, 2 * v3
        # drive the highpass output (port 3); junction loaded by r || lowpass || corrector
        yj = 1 / r + y_l + y_c
        zj = 1 / yj
        # reversed highpass ladder (reciprocal): ABCD [[D, B], [C, A]]
        zin3 = (Dh * zj + Bh) / (Ch * zj + Ah)
        s33 = (zin3 - r) / (zin3 + r)
        v_in3 = zin3 / (r + zin3)
        vj3 = v_in3 / (Dh + Bh / zj)            # voltage at the junction
        s32 = 2 * vj3 / (Al + Bl / r)           # on to the lowpass output
        yj2 = 1 / r + y_h + y_c
        zj2 = 1 / yj2
        zin2 = (Dl * zj2 + Bl) / (Cl * zj2 + Al)
        s22 = (zin2 - r) / (zin2 + r)
        return {"S11": s11, "S21": s21, "S31": s31, "S32": s32, "S22": s22, "S33": s33}


# --------------------------------------------------------------------------- design

def _channel(ftype, shape, order, fc, ripple, r0, amin=50.0):
    """Singly terminated channel, oriented common end -> output (load r0).
    The common end must face a series element (short-circuited / voltage-driven design):
    of the two circuits, the one whose unterminated end is a short is used."""
    from vektorfil.filters.design import SERIES_FIRST, SHUNT_FIRST, FilterSpec, design
    if shape == "Elliptic" and order < 3:
        raise ValueError("elliptic channels need order 3 or more")
    lad, res = None, None
    # try the fewer-capacitor ladder first (lowpass: series first; highpass: shunt first)
    for sub in ((SERIES_FIRST, SHUNT_FIRST) if ftype == "Lowpass" else (SHUNT_FIRST, SERIES_FIRST)):
        # each channel is specified by its 3 dB point: at the crossover both channels then
        # carry half the power and their conductances add up to about 1/R
        res = design(FilterSpec(ftype=ftype, shape=f"{shape}, singly terminated", subtype=sub,
                                order=order, fc=fc, r_source=r0, ripple_db=ripple,
                                amin_db=amin, aa_db=3.0103))
        if res.ladder.single_terminated == "short":
            lad = res.ladder
            break
    if lad is None:
        raise ValueError("no channel realization with a series element at the common port")
    branches = [Branch(b.pos, b.kind, L=b.L, C=b.C) for b in reversed(lad.branches)]
    # r_source is only for display: in the diplexer the common port sees the source r0
    out = Ladder(branches, r_source=r0, r_load=r0,
                 title=f"{shape} {ftype.lower()}, order {order}, {fc / 1e6:.6g} MHz")
    out.notes["info"] = res.info          # e.g. elliptic stopband edge and zeros
    return out


def _passband_grid(spec, n=400):
    """Frequencies inside both passbands, used to judge the common-port match."""
    lo = np.geomspace(spec.lp_fc * 0.01, spec.lp_fc, n // 2)
    hi = np.geomspace(spec.hp_fc, spec.hp_fc * 10, n // 2)
    return np.concatenate([lo, hi])


def _fit_corrector(dx: Diplexer):
    """Optimise the shunt series-LC corrector (resonant near the crossover) for the best
    common-port match over the crossover region and both passbands."""
    spec = dx.spec
    fx = math.sqrt(spec.lp_fc * spec.hp_fc)
    f = np.concatenate([_passband_grid(spec, 200), np.geomspace(spec.lp_fc, spec.hp_fc, 50)
                        if spec.hp_fc > spec.lp_fc else np.array([fx])])
    wx = 2 * math.pi * fx

    def resid(x):
        L = math.exp(x[0])
        fr = fx * math.exp(x[1])
        C = 1 / ((2 * math.pi * fr) ** 2 * L)
        dx.corr = (L, C)
        return np.abs(dx.sparams(f)["S11"])

    best = None
    for L0 in (spec.r0 / wx * k for k in (0.3, 1.0, 3.0)):     # a few starting points
        r = least_squares(resid, [math.log(L0), 0.0], bounds=([-60, -0.7], [10, 0.7]))
        if best is None or r.cost < best.cost:
            best = r
    L = math.exp(best.x[0])
    fr = fx * math.exp(best.x[1])
    dx.corr = (L, 1 / ((2 * math.pi * fr) ** 2 * L))
    return dx.corr


def design_diplexer(spec: DiplexSpec) -> Diplexer:
    """Build both channels, optionally add the corrector, and summarise the result."""
    if spec.hp_fc < spec.lp_fc * 0.999:
        raise ValueError("the highpass cutoff must not lie below the lowpass cutoff")
    lp = _channel("Lowpass", spec.lp_shape, spec.lp_order, spec.lp_fc, spec.lp_ripple, spec.r0,
                  spec.lp_amin)
    hp = _channel("Highpass", spec.hp_shape, spec.hp_order, spec.hp_fc, spec.hp_ripple, spec.r0,
                  spec.hp_amin)
    dx = Diplexer(spec, lp, hp)
    exact = (spec.lp_shape == spec.hp_shape == "Butterworth" and spec.lp_order == spec.hp_order
             and abs(spec.lp_fc / spec.hp_fc - 1) < 1e-9)
    f = _passband_grid(spec)

    def worst():
        return float((20 * np.log10(np.abs(dx.sparams(f)["S11"]) + 1e-300)).max())

    without = worst()
    dx.info["corrector_note"] = ""
    if spec.corrector and exact:
        dx.info["corrector_note"] = "not needed: complementary pair, already exact"
    elif spec.corrector:
        _fit_corrector(dx)
        gain = without - worst()
        if gain < 0.3:                 # keep it only when it really helps
            dx.corr = None
            dx.info["corrector_note"] = f"not useful here (would improve the match by {gain:.2f} dB)"
        else:
            dx.info["corrector_note"] = f"improves the worst passband match by {gain:.1f} dB"
    s_worst = worst()
    dx.info.update(
        exact=exact,
        worst_s11_db=s_worst,
        worst_return_loss_db=-s_worst,
        crossover=math.sqrt(spec.lp_fc * spec.hp_fc),
    )
    return dx


# --------------------------------------------------------------------------- export

def touchstone_s3p(dx: Diplexer, f, title=""):
    """3-port Touchstone text (dB/angle, reference r0): per frequency three lines
    S11 S12 S13 / S21 S22 S23 / S31 S32 S33 (the network is reciprocal)."""
    f = np.asarray(f, float)
    sp = dx.sparams(f)
    S = [["S11", "S21", "S31"], ["S21", "S22", "S32"], ["S31", "S32", "S33"]]
    lines = [f"! {title or 'Diplexer'}", "! Created by VektorFIL",
             "! Port 1 = common, port 2 = lowpass output, port 3 = highpass output",
             f"# Hz S DB R {dx.spec.r0:g}"]

    def pair(v):
        return f"{20 * np.log10(abs(v) + 1e-300):.6f} {np.degrees(np.angle(v)):.6f}"
    for i, fi in enumerate(f):
        for row, names in enumerate(S):
            vals = " ".join(pair(sp[n][i]) for n in names)
            lines.append((f"{fi:.10g} " if row == 0 else "  ") + vals)
    return "\n".join(lines) + "\n"
