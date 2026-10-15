"""Narrow-band coupled-resonator bandpass filters from an all-pole lowpass prototype.

Design by external Q and coupling coefficients (Matthaei, Young & Jones; Cohn):

    Qe1 = g0 g1 / w,  Qen = g_n g_{n+1} / w,  k_{i,i+1} = w / sqrt(g_i g_{i+1}),  w = BW/f0

Resonators all share the same inductor (top-C, shunt-C) or capacitor (top-L), as in the
classic FILTER topologies.  A coupling element is an admittance (or impedance) inverter
plus a negative element on each side; those negative elements are absorbed into the
resonators, and the end couplings are converted series<->parallel exactly at f0.

    top_c    shunt parallel resonators, series coupling capacitors
    top_c_tem  as top_c, but each resonator is a shorted TEM line (impedance Z0, length
             theta at f0, e.g. a ceramic coaxial resonator) loaded by a capacitor; all lines
             equal, the loading capacitors tune each resonator to f0
    top_l    shunt parallel resonators, series coupling inductors   (equal C)
    shunt_c  series resonators in the series arm, shunt coupling capacitors to ground
"""
from __future__ import annotations

import math

from .ladder import Branch
from .network import Ladder


class CouplingError(ValueError):
    """The coupled-resonator design is not realizable with this resonator value
    (an element would turn negative or an end coupling cannot be made)."""


def qe_k(g, w):
    """g = [g0, g1, ..., gn, g_{n+1}] in Matthaei's convention."""
    n = len(g) - 2
    qe1 = g[0] * g[1] / w
    qen = g[n] * g[n + 1] / w
    k = [w / math.sqrt(g[i] * g[i + 1]) for i in range(1, n)]
    return qe1, qen, k


def tem_slope(z0, theta_deg):
    """Susceptance slope b = (w0/2) dB/dw of a shorted line (Z0, theta at f0) tuned to
    resonance at f0 by a loading capacitor: b = (cot(theta) + theta/sin^2(theta)) / (2 Z0)."""
    th = math.radians(theta_deg)
    return (1 / math.tan(th) + th / math.sin(th) ** 2) / (2 * z0)


def resonator_limit(kind, g, f0, bw, r_s, r_l, theta_deg=45.0):
    """Extreme resonator value for which the end coupling element vanishes
    (an inductor for top_c/shunt_c - minimum resp. maximum; a capacitor for top_l;
    the line impedance Z0 for top_c_tem - minimum)."""
    w = bw / f0
    w0 = 2 * math.pi * f0
    qe1, qen, _ = qe_k(g, w)
    if kind == "top_c_tem":  # b(Z0) = tem_slope(Z0) <= Qe/R  ->  Z0 >= R tem_slope(1)/Qe
        return max(r_s * tem_slope(1.0, theta_deg) / qe1, r_l * tem_slope(1.0, theta_deg) / qen)
    if kind == "top_c":     # need Ge = b/Qe <= 1/R, b = 1/(w0 L)  ->  L >= R/(w0 Qe)
        return max(r_s / (w0 * qe1), r_l / (w0 * qen))
    if kind == "top_l":     # b = w0 C  ->  C <= Qe/(w0 R)
        return min(qe1 / (w0 * r_s), qen / (w0 * r_l))
    if kind == "shunt_c":   # Re = x/Qe <= R, x = w0 L  ->  L <= Qe R / w0
        return min(qe1 * r_s / w0, qen * r_l / w0)
    raise ValueError(kind)


def default_resonator(kind, g, f0, bw, r_s, r_l, theta_deg=45.0):
    """A comfortable default: halfway (geometrically) between the limit and the value
    whose reactance equals the termination."""
    w0 = 2 * math.pi * f0
    lim = resonator_limit(kind, g, f0, bw, r_s, r_l, theta_deg)
    if kind == "top_c_tem":
        return max(math.sqrt(lim * max(r_s / 2, lim * 4)), lim * 1.5)
    if kind == "top_c":
        nat = r_s / w0
        return max(math.sqrt(lim * max(nat, lim * 4)), lim * 1.5)
    if kind == "top_l":
        nat = 1 / (w0 * r_s)
        return min(math.sqrt(lim * min(nat, lim / 4)), lim / 1.5)
    nat = r_s / w0
    return min(math.sqrt(lim * min(nat, lim / 4)), lim / 1.5)


def design_coupled(kind, g, f0, bw, r_s, r_l, resonator=None, title="", theta_deg=45.0,
                   end_coils=False):
    """g: [g0..g_{n+1}] (Matthaei convention).  resonator: inductance (top_c, shunt_c),
    capacitance (top_l) or line impedance Z0 (top_c_tem) shared by all resonators;
    None = default.  theta_deg: TEM line length at f0 (top_c_tem only).
    end_coils (shunt_c only): past the limit, couple each end through a series inductor
    and a shunt capacitor instead of the shunt capacitor alone (see below)."""
    n = len(g) - 2
    w = bw / f0
    w0 = 2 * math.pi * f0
    qe1, qen, k = qe_k(g, w)
    if resonator is None:
        resonator = default_resonator(kind, g, f0, bw, r_s, r_l, theta_deg)
    out = []

    if kind in ("top_c", "top_l", "top_c_tem"):
        tem = kind == "top_c_tem"
        capc = kind in ("top_c", "top_c_tem")     # capacitive couplings
        if kind == "top_c":
            Lr = resonator
            b = 1 / (w0 * Lr)                    # susceptance slope of a parallel LC = w0 C
        elif tem:
            z0 = resonator
            b = tem_slope(z0, theta_deg)
            cot = 1 / math.tan(math.radians(theta_deg))
        else:
            Cr = resonator
            b = w0 * Cr
        bphys = [0.0] * n                        # required shunt susceptance at f0 per node
        series = []                              # coupling reactances X between nodes (f0)
        # ends: series reactance to the termination R gives parallel G_e and B_e
        ends = []
        for R, qe in ((r_s, qe1), (r_l, qen)):
            ge = b / qe
            if ge * R > 1:
                lim = resonator_limit(kind, g, f0, bw, r_s, r_l, theta_deg)
                what = {"top_c": "L", "top_l": "C", "top_c_tem": "line Z0"}[kind]
                raise CouplingError(
                    f"resonator {what} outside the limit ({'<=' if kind == 'top_l' else '>='} {lim:.4g})")
            X = math.sqrt(R / ge - R * R)
            X = -X if capc else X                 # capacitor or inductor
            ends.append(X)
        # inner couplings: J = b k ; |X| = 1/J
        for i in range(n - 1):
            J = b * k[i]
            series.append(-1 / J if capc else 1 / J)
        # node susceptance bookkeeping: physical series X adds -1/X at each adjacent node
        for i in range(n - 1):
            bphys[i] += 1 / series[i]
            bphys[i + 1] += 1 / series[i]
        for idx, (X, R) in zip((0, n - 1), zip(ends, (r_s, r_l))):
            be = -X / (R * R + X * X)            # parallel-equivalent of R + jX
            bphys[idx] -= be
        # build the ladder
        def coupling(X):
            return Branch("series", "C", C=-1 / (w0 * X)) if X < 0 else Branch("series", "L", L=X / w0)
        out.append(coupling(ends[0]))
        for i in range(n):
            if tem:                              # w0 C - cot(theta)/Z0 = bphys
                C = (bphys[i] + cot / z0) / w0
                if C <= 0:
                    raise CouplingError("negative loading capacitor: use a higher line impedance")
                out.append(Branch("shunt", "TEM", C=C, z0=z0, theta=theta_deg, f_ref=f0))
            elif kind == "top_c":                # w0 C - 1/(w0 Lr) = bphys
                C = (bphys[i] + 1 / (w0 * Lr)) / w0
                if C <= 0:
                    raise CouplingError("negative resonator capacitor: use a larger inductor")
                out.append(Branch("shunt", "LC_par", L=Lr, C=C))
            else:                                # w0 Cr - 1/(w0 L) = bphys
                inv = w0 * (w0 * Cr - bphys[i])
                if inv <= 0:
                    raise CouplingError("negative resonator inductor: use a smaller capacitor")
                out.append(Branch("shunt", "LC_par", L=1 / inv, C=Cr))
            if i < n - 1:
                out.append(coupling(series[i]))
        out.append(coupling(ends[1]))

    elif kind == "shunt_c":
        Lr = resonator
        x = w0 * Lr                              # reactance slope of a series LC
        xphys = [0.0] * n
        ends, coils = [], []
        for R, qe in ((r_s, qe1), (r_l, qen)):
            re = x / qe
            G = 1 / R
            if re > R:
                if not end_coils:
                    lim = resonator_limit(kind, g, f0, bw, r_s, r_l)
                    raise CouplingError(f"resonator L outside the limit (<= {lim:.4g})")
                # a shunt capacitor alone can only step R down; to step it up to re, put a
                # series inductor Xs in front of it: R + jXs, then shunt jB.  With
                # Xs = sqrt(R (re - R)) and B = Xs / (R re) the result is exactly re (no
                # reactance left over); both vanish at re = R, where the plain form ends
                xs = math.sqrt(R * (re - R))
                ends.append(xs / (R * re))
                coils.append(xs)
                continue
            Bc = math.sqrt(G / re - G * G)       # shunt capacitor susceptance at the end
            ends.append(Bc)
            coils.append(0.0)
        shunts = [1 / (x * kk) for kk in k]      # K = x k ; shunt susceptance B = 1/K
        for i in range(n - 1):
            xphys[i] += 1 / shunts[i]
            xphys[i + 1] += 1 / shunts[i]
        for idx, (Bc, R, xs) in zip((0, n - 1), zip(ends, (r_s, r_l), coils)):
            if xs:
                continue                         # the end coil cancels the reactance
            G = 1 / R
            xe = -Bc / (G * G + Bc * Bc)         # series-equivalent of G + jB
            xphys[idx] -= xe
        if coils[0]:
            out.append(Branch("series", "L", L=coils[0] / w0))
        out.append(Branch("shunt", "C", C=ends[0] / w0))
        for i in range(n):
            # w0 Lr - 1/(w0 C) = xphys
            d = w0 * Lr - xphys[i]
            if d <= 0:
                raise CouplingError("negative resonator capacitor: use a smaller inductor")
            out.append(Branch("series", "LC_ser", L=Lr, C=1 / (w0 * d)))
            if i < n - 1:
                out.append(Branch("shunt", "C", C=shunts[i] / w0))
        out.append(Branch("shunt", "C", C=ends[1] / w0))
        if coils[1]:
            out.append(Branch("series", "L", L=coils[1] / w0))
    else:
        raise ValueError(kind)

    lad = Ladder(out, r_source=r_s, r_load=r_l, title=title)
    lad.notes.update(f0=f0, bw=bw, band="bp", resonator=resonator,
                     limit=resonator_limit(kind, g, f0, bw, r_s, r_l, theta_deg))
    return lad


def to_tubular(lad: Ladder) -> Ladder:
    """Shunt-C coupled bandpass -> classic tubular form (exact).

    Each series resonator capacitor C_k is split into a left part a_k and a right part
    b_k (1/a_k + 1/b_k = 1/C_k).  The end resonators keep their whole capacitor on the
    inner side, the inner ones are split equally.  Between inductors this leaves a
    capacitor tee (series b_k, shunt C_k,k+1, series a_k+1), converted to the pi
        series  Cb = C1 C2 / S,   shunt  Ca = C1 C3 / S,  Cc = C2 C3 / S,  S = C1 + C2 + C3
    The result starts and ends with a capacitor to ground: shunt C, L, pi, L, pi, ..., L,
    shunt C -- series inductors joined by series capacitors with shunt capacitors to
    ground at every junction."""
    br = list(lad.branches)
    # end coils (design_coupled(end_coils=True)) stay in front of the end capacitors
    head = [br.pop(0)] if br[0].pos == "series" and br[0].kind == "L" else []
    tail = [br.pop()] if br[-1].pos == "series" and br[-1].kind == "L" else []
    assert br[0].pos == "shunt" and br[0].kind == "C", "expects a shunt-C coupled ladder"
    res = [b for b in br if b.kind == "LC_ser"]
    cpl = [b.C for b in br if b.pos == "shunt"]          # end, inner..., end
    n = len(res)
    if n == 1:
        out = [Branch("shunt", "C", C=cpl[0]), Branch("series", "L", L=res[0].L),
               Branch("series", "C", C=res[0].C), Branch("shunt", "C", C=cpl[-1])]
        return Ladder(head + out + tail, lad.r_source, lad.r_load, lad.title, dict(lad.notes), lad.q_l, lad.q_c)
    inf = math.inf
    left = [inf] + [2 * r.C for r in res[1:-1]] + [res[-1].C]
    right = [res[0].C] + [2 * r.C for r in res[1:-1]] + [inf]
    out = [Branch("shunt", "C", C=cpl[0]), Branch("series", "L", L=res[0].L)]
    for k in range(n - 1):
        c1, c3, c2 = right[k], cpl[k + 1], left[k + 1]
        s = c1 + c2 + c3
        out += [Branch("shunt", "C", C=c1 * c3 / s), Branch("series", "C", C=c1 * c2 / s),
                Branch("shunt", "C", C=c2 * c3 / s), Branch("series", "L", L=res[k + 1].L)]
    out.append(Branch("shunt", "C", C=cpl[-1]))
    return Ladder(head + out + tail, lad.r_source, lad.r_load, lad.title, dict(lad.notes), lad.q_l, lad.q_c)


END_COIL_SPAN = 10.0   # resonator range offered with end coils, as a multiple of the limit


def resonator_range(kind, g, f0, bw, r_s, r_l, theta_deg=45.0, end_coils=False):
    """Usable range (lo, hi) of the shared resonator value for a coupled type.

    One end is resonator_limit() (an end coupling element vanishes); the other is where
    a resonator element would turn negative.  Found by bisection on design_coupled(),
    which is cheap (plain floats).  With end_coils (shunt_c) the range continues above
    the limit."""
    lim = resonator_limit(kind, g, f0, bw, r_s, r_l, theta_deg)

    def ok(v):
        try:
            design_coupled(kind, g, f0, bw, r_s, r_l, v, theta_deg=theta_deg, end_coils=end_coils)
            return True
        except (CouplingError, ValueError, ZeroDivisionError):
            return False

    # top_c: valid above the limit; top_l and shunt_c: valid below it
    up = kind in ("top_c", "top_c_tem")
    lo, hi = _walk_range(ok, lim, up)
    if end_coils and kind == "shunt_c" and ok(lim * 1.0001):
        # end coils lift the upper limit and there is no hard limit above it, so offer up
        # to END_COIL_SPAN times the plain limit (the end coil grows as sqrt(L))
        top = lim * END_COIL_SPAN
        if not ok(top):
            good, bad = lim, top
            for _ in range(40):
                mid = math.sqrt(good * bad)
                good, bad = (mid, bad) if ok(mid) else (good, mid)
            top = good
        hi = top
    return lo, hi


def _walk_range(ok, lim, up):
    """From the limit, walk away (up or down) until ok() fails, then bisect the edge."""
    inner = lim * (1.0001 if up else 1 / 1.0001)
    if not ok(inner):
        return (lim, lim)
    far, k = inner, 2.0
    for _ in range(40):                       # walk away from the limit until it fails
        trial = far * k if up else far / k
        if not ok(trial):
            break
        far = trial
    else:
        return (inner, far) if up else (far, inner)
    good, bad = far, (far * k if up else far / k)
    for _ in range(40):                       # bisection (geometric) on the failure point
        mid = math.sqrt(good * bad)
        if ok(mid):
            good = mid
        else:
            bad = mid
    return (inner, good) if up else (good, inner)
