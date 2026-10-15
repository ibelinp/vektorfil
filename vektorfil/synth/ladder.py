"""Darlington ladder synthesis of a lowpass prototype by zero shifting.

The input immittance of the lossless 2-port terminated in R_L, seen from a 1-ohm
source, is Z = (E + F)/(E - F) (or its inverse depending on the sign of F).
Elements are extracted one transmission zero at a time:

  zero at infinity, from Y : full removal of a shunt C        (then continue with Z)
  zero at infinity, from Z : full removal of a series L       (then continue with Y)
  finite zero w,   from Y  : partial removal of shunt C so Y-sC vanishes at jw,
                             then a series-arm parallel L-C tank resonant at w
  finite zero w,   from Z  : partial removal of series L so Z-sL vanishes at jw,
                             then a shunt series L-C trap resonant at w

Whatever is left at the end must be a resistor: the load.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import mpmath as mp

from . import poly as P
from .approx import Prototype


@dataclass
class Branch:
    """One ladder branch: position ('series' arm or 'shunt' to ground) and kind
    ('L', 'C', 'LC_par' parallel tank, 'LC_ser' series resonator, 'TEM' shorted
    transmission-line resonator with loading capacitor C) with its values.
    """
    pos: str        # 'series' or 'shunt'
    kind: str       # 'L', 'C', 'LC_par', 'LC_ser', 'TEM'
    L: object = None
    C: object = None
    z0: float = None      # TEM: line impedance (ohm)
    theta: float = None   # TEM: electrical length in degrees at f_ref
    f_ref: float = None   # TEM: frequency (Hz) at which theta is given

    def __repr__(self):
        v = []
        if self.L is not None:
            v.append(f"L={mp.nstr(self.L, 8)}")
        if self.C is not None:
            v.append(f"C={mp.nstr(self.C, 8)}")
        return f"{self.pos}:{self.kind}({', '.join(v)})"


@dataclass
class LadderResult:
    """A synthesized (normalised) ladder: branches, terminations and the extraction
    sequence that produced it.
    """
    branches: list
    r_source: object
    r_load: object
    sequence: list


class SynthesisError(ArithmeticError):
    """No ladder: an extraction step gave a negative or non-finite element, or the
    remainder is not the expected termination."""


def _eval(num, den, s):
    return P.ev(num, s) / P.ev(den, s)


def _check_positive(x, what):
    """Return x as a real positive number or raise SynthesisError."""
    x = mp.mpf(mp.re(x)) if isinstance(x, mp.mpc) else mp.mpf(x)
    if x <= 0:
        raise SynthesisError(f"{what} is not positive ({mp.nstr(x, 6)})")
    return x


def extract(num, den, start, sequence):
    """Run the extraction. start: 'Y' (shunt element first) or 'Z' (series first).
    sequence: list of mp.inf or finite w. Returns (branches, remaining (num, den), type)."""
    branches = []
    kind = start
    s_poly = [mp.mpf(1), mp.mpf(0)]
    for w in sequence:
        if w == mp.inf:
            # full removal of the pole at infinity
            if P.deg(num) != P.deg(den) + 1:
                raise SynthesisError(f"no pole at infinity to remove from {kind}")
            num, den = P.trim(num), P.trim(den)
            k = _check_positive(num[0] / den[0], "element at infinity")
            num = P.trim(P.sub(num, P.mul(P.scale(s_poly, k), den)))
            if kind == 'Y':
                branches.append(Branch('shunt', 'C', C=k))
            else:
                branches.append(Branch('series', 'L', L=k))
            num, den = den, num
            kind = 'Z' if kind == 'Y' else 'Y'
        else:
            w = mp.mpf(w)
            jw = mp.mpc(0, w)
            # partial removal so that the remainder vanishes at jw
            k = _eval(num, den, jw) / jw
            if abs(mp.im(k)) > 1e-20 * (1 + abs(k)):
                raise SynthesisError("partial removal value is complex")
            k = _check_positive(mp.re(k), "partial removal element")
            num = P.trim(P.sub(num, P.mul(P.scale(s_poly, k), den)))
            if kind == 'Y':
                branches.append(Branch('shunt', 'C', C=k))
            else:
                branches.append(Branch('series', 'L', L=k))
            # remainder has a zero at +-jw: factor it out, invert -> pole at +-jw
            quad = [mp.mpf(1), mp.mpf(0), w * w]
            q = P.exact_div(num, quad, "zero shifting")
            # inverse immittance = den / (quad * q); residue form 2K s / (s^2 + w^2)
            Kres = P.ev(den, jw) / (2 * jw * P.ev(q, jw))
            if abs(mp.im(Kres)) > 1e-20 * (1 + abs(Kres)):
                raise SynthesisError("resonator residue is complex")
            Kres = _check_positive(mp.re(Kres), "resonator residue")
            rest = P.sub(den, P.mul(P.scale(s_poly, 2 * Kres), q))
            rest = P.exact_div(rest, quad, "resonator removal")
            if kind == 'Y':
                # inverse is Z: series arm parallel tank  Z = 2K s/(s^2+w^2)
                branches.append(Branch('series', 'LC_par', L=2 * Kres / (w * w), C=1 / (2 * Kres)))
            else:
                # inverse is Y: shunt series resonator  Y = 2K s/(s^2+w^2)
                branches.append(Branch('shunt', 'LC_ser', L=1 / (2 * Kres), C=2 * Kres / (w * w)))
            # remaining inverse immittance rest/q ; flip back to the original kind
            num, den = q, rest
    return branches, (P.trim(num), P.trim(den)), kind


def _distinct_permutations(items):
    """Distinct permutations of a list with repeated items, generated lazily.
    (itertools.permutations would walk all n! orderings -- 25! for a 25th-order
    all-pole filter, whose zeros are all at infinity.)"""
    counts = {}
    order = []
    for x in items:
        if x not in counts:
            counts[x] = 0
            order.append(x)
        counts[x] += 1
    n = len(items)
    cur = []

    def rec():
        if len(cur) == n:
            yield list(cur)
            return
        for x in order:
            if counts[x]:
                counts[x] -= 1
                cur.append(x)
                yield from rec()
                cur.pop()
                counts[x] += 1
    return rec()


def candidate_sequences(proto: Prototype, limit=720):
    """Distinct extraction orders of the finite zeros and the zeros at infinity, at most
    `limit` of them.  Classical layouts come first: finite zeros in a few natural orders
    with the zeros at infinity at the end; then the remaining distinct orders."""
    fin = sorted(proto.zeros)
    tail = [mp.inf] * proto.n_inf
    natural = [fin[::-1], fin]
    if len(fin) > 2:   # outside-in: highest, lowest, next highest, ... (balanced ladders)
        lo, hi, alt = 0, len(fin) - 1, []
        while lo <= hi:
            alt.append(fin[hi])
            if lo != hi:
                alt.append(fin[lo])
            lo, hi = lo + 1, hi - 1
        natural.append(alt)
    out, seen = [], set()

    def add(seq):
        t = tuple(seq)
        if t not in seen:
            seen.add(t)
            out.append(list(seq))

    for nat in natural:
        add(nat + tail)
    for p in itertools.islice(itertools.permutations(fin), limit):
        add(list(p) + tail)
        if len(out) >= limit // 2:
            break
    for p in _distinct_permutations(fin + tail):
        if len(out) >= limit:
            break
        add(p)
    return out


def synthesize_all(proto: Prototype, first='shunt'):
    """Every realizable ladder (all extraction orders, both F signs)."""
    out = []
    for seq in candidate_sequences(proto):
        try:
            out.append(synthesize(proto, first=first, sequence=seq))
        except SynthesisError:
            pass
    return out


def synthesize_single(proto: Prototype, first='shunt', end=None):
    """Singly terminated ladder: 1-ohm source, far end open or shorted.

    With the output open, V2/E = z21/(1 + z11).  Writing E = m + n (even and odd parts),
    z11 = m/n with z21 = P/n (or z11 = n/m, z21 = P/m) gives V2/E = P/E: the required
    response.  z11 is a reactance function: all-pole filters are its continued-fraction
    (Cauer) expansion from infinity; with finite transmission zeros (elliptic) the same
    zero-shifting extraction as for doubly terminated ladders places the notches.  The
    shorted-output case is the dual (y11 with output current).

    An open end must face a shunt element and a shorted end a series element; `end`
    ('open' / 'short') may be requested, otherwise the first realization found is used.
    Returns a LadderResult with r_load = inf (open) or 0 (short)."""
    n_order = proto.order
    E = P.trim(proto.E)
    deg = len(E) - 1
    m = [c if (deg - i) % 2 == 0 else mp.mpf(0) for i, c in enumerate(E)]
    n = [c if (deg - i) % 2 == 1 else mp.mpf(0) for i, c in enumerate(E)]
    m, n = P.trim(m), P.trim(n)
    start = 'Y' if first == 'shunt' else 'Z'
    if proto.zeros:
        seqs = [list(p) + [mp.inf] * proto.n_inf
                for p in itertools.islice(itertools.permutations(sorted(proto.zeros)), 720)]
    else:
        seqs = [[mp.inf] * n_order]
    ends = [end] if end else ['open', 'short']
    for seq in seqs:
        for num, den in ((m, n), (n, m)):
            if P.deg(num) != P.deg(den) + 1:
                continue                         # need a pole at infinity to start
            try:
                br, (rn, rd), kind = extract(num, den, start, seq)
            except (SynthesisError, ArithmeticError):
                continue
            last_series = br[-1].pos == 'series'
            for e in ends:
                if (e == 'short') != last_series:
                    continue                     # open faces shunt, short faces series
                res = LadderResult(br, mp.mpf(1), mp.inf if e == 'open' else mp.mpf(0), seq)
                if _single_ok(res, proto, e):
                    return res
    raise SynthesisError("no singly terminated realization found")


def _single_ok(res, proto, end):
    """Check a singly terminated ladder against the prototype: same |H| shape."""
    import numpy as np
    from .network import Ladder
    lad = Ladder.from_prototype(res, r0=1.0, fc=1 / (2 * math.pi))   # stays normalised
    w = np.array([0.05, 0.3, 0.7, 1.0, 1.5, 3.0])
    h = np.abs(lad.sparams(w / (2 * math.pi))[1])
    ref = np.array([float(abs(proto.s21(x))) for x in w])
    ratio = h / ref
    return bool(np.all(np.abs(ratio / ratio[0] - 1) < 1e-6))


def synthesize(proto: Prototype, first='shunt', sequence=None, minimize=None):
    """Synthesize the doubly terminated ladder (source 1 ohm).

    first: 'shunt' (minimum-inductor style for lowpass) or 'series'.
    minimize: None, 'L' or 'C' -- prefer the realization with the fewest inductors
    (or capacitors); ties are broken by the most even inductor values.
    sequence: optional explicit order of zeros; default tries all orderings of the
    finite zeros followed by the zeros at infinity and keeps the best feasible one.
    """
    start = 'Y' if first == 'shunt' else 'Z'
    E, F = proto.E, proto.F
    candidates = []
    if sequence is not None:
        seqs = [list(sequence)]
    else:
        seqs = candidate_sequences(proto)
    errors = []
    t0 = time.perf_counter()
    for seq in seqs:
        # bounded search: enough candidates (or a short while after the first) is plenty
        elapsed = time.perf_counter() - t0
        if candidates and (len(candidates) >= 24 or elapsed > 1.5):
            break
        if elapsed > 15.0:
            break
        for sgn in (1, -1):
            Fs = P.scale(F, sgn)
            zn, zd = P.add(E, Fs), P.sub(E, Fs)  # Z = (E+F)/(E-F)
            num, den = (zd, zn) if start == 'Y' else (zn, zd)
            try:
                br, (rn, rd), kind = extract(num, den, start, seq)
            except (SynthesisError, ArithmeticError) as e:
                errors.append(str(e))
                continue
            if P.deg(rn) != 0 or P.deg(rd) != 0:
                errors.append("remainder is not a resistor")
                continue
            val = rn[-1] / rd[-1]
            r_load = val if kind == 'Z' else 1 / val
            if r_load <= 0:
                continue
            candidates.append(LadderResult(br, mp.mpf(1), r_load, seq))
            break
    if not candidates:
        raise SynthesisError("no realizable ladder found: " + "; ".join(sorted(set(errors))[:4]))

    def key(res):
        Ls = [b.L for b in res.branches if b.L is not None]
        Cs = [b.C for b in res.branches if b.C is not None]
        count = len(Ls) if minimize == 'L' else len(Cs) if minimize == 'C' else 0
        spread = max(Ls) / min(Ls) if Ls else 1  # most even inductor values (S/FILSYN-like)
        return (count, spread)

    return min(candidates, key=key)
