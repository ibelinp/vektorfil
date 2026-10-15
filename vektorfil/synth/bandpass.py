"""Direct bandpass synthesis.

The bandpass transfer function is built in the bandpass s-plane (normalised to the
centre frequency, w0 = 1) and a ladder is extracted from it directly, instead of
transforming lowpass elements one by one.  This reaches topologies the element-wise
transform cannot, notably the minimum-inductor ("zig-zag") elliptic bandpass of
Saal & Ulbrich.

Extraction steps (the immittance alternates between Z and Y as in ladder.py):

  'inf'          full removal of the pole at infinity      Y: shunt C   Z: series L
  'dc'           full removal of the pole at DC            Y: shunt L   Z: series C
  'flip'         continue with the inverse immittance (next element in the other arm);
                 without a flip, consecutive elements share the same node / arm
  ('fin', w, 'inf')  partial removal at infinity so the remainder vanishes at jw,
                     then the resonator at w               Y: shunt C + series tank
                                                           Z: series L + shunt trap
  ('fin', w, 'dc')   partial removal at DC, then the resonator
                                                           Y: shunt L + series tank
                                                           Z: series C + shunt trap
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mpmath as mp

from . import poly as P
from .approx import Prototype
from .ladder import Branch, LadderResult, SynthesisError
from .network import Ladder


@dataclass
class BandProto:
    """Bandpass transfer function: E(s), F(s), P(s) normalised to w0 = 1, with the
    location of its transmission zeros (finite, at DC, at infinity).
    """
    E: list
    F: list
    P: list
    B: object                 # fractional bandwidth (w0 = 1)
    zeros: list               # finite transmission zeros, positive, normalised to w0
    n_dc: int                 # transmission zeros at DC
    n_inf: int                # transmission zeros at infinity
    order: int                # degree in s (2N for a geometric transform)
    lp: Prototype = None
    meta: dict = field(default_factory=dict)

    def s21(self, w):
        s = mp.mpc(0, w)
        return P.ev(self.P, s) / P.ev(self.E, s)

    def atten_db(self, w):
        return -20 * mp.log10(abs(self.s21(w)))


def _subst_bp(poly_p, N, B):
    """(sB)^N * poly((s^2 + 1)/(s B)) as a polynomial in s (highest power first)."""
    deg = len(poly_p) - 1
    out = [mp.mpf(0)]
    q = [mp.mpf(1), mp.mpf(0), mp.mpf(1)]      # s^2 + 1
    sB = [mp.mpf(B), mp.mpf(0)]                # s B
    for i, c in enumerate(poly_p):
        k = deg - i                            # term c p^k
        if c == 0:
            continue
        term = [mp.mpf(c)]
        for _ in range(k):
            term = P.mul(term, q)
        for _ in range(N - k):
            term = P.mul(term, sB)
        out = P.add(out, term)
    return P.trim(out)


def from_lowpass(lp: Prototype, B) -> BandProto:
    """Geometrically symmetric bandpass from a lowpass prototype, exactly."""
    B = mp.mpf(B)
    N = lp.order
    E = _subst_bp(lp.E, N, B)
    F = _subst_bp(lp.F, N, B)
    Pp = _subst_bp(lp.P, N, B)
    zs = []
    for z in lp.zeros:
        r = mp.sqrt((z * B) ** 2 + 4)
        zs += [(r - z * B) / 2, (r + z * B) / 2]
    return BandProto(E=E, F=F, P=Pp, B=B, zeros=sorted(zs), n_dc=lp.n_inf, n_inf=lp.n_inf,
                     order=2 * N, lp=lp)


# --------------------------------------------------------------------------- extraction

_S = [mp.mpf(1), mp.mpf(0)]


def _val(num, den, s):
    return P.ev(num, s) / P.ev(den, s)


def _norm(num, den):
    """Zero coefficients that are rounding residue, trim, and cancel common powers of s."""
    def clean(p):
        big = max(abs(c) for c in p) or mp.mpf(1)
        eps = big * mp.mpf(10) ** (-(mp.mp.dps * 2 // 3))
        return P.trim([c if abs(c) > eps else mp.mpf(0) for c in p])
    num, den = clean(num), clean(den)
    while len(num) > 1 and len(den) > 1 and num[-1] == 0 and den[-1] == 0:
        num, den = num[:-1], den[:-1]
    return num, den


def _real_pos(x, what, tol):
    """Check that an extracted element value is real and positive; return it."""
    if abs(mp.im(x)) > tol * (1 + abs(x)):
        raise SynthesisError(f"{what} is complex")
    x = mp.re(x)
    if x <= 0:
        raise SynthesisError(f"{what} not positive")
    return x


def _step(num, den, kind, tok, tol):
    """Apply one extraction token. Returns (new_branches, num, den, kind)."""
    br = []
    num, den = _norm(num, den)
    if tok == "inf":
        if P.deg(num) != P.deg(den) + 1:
            raise SynthesisError("no pole at infinity")
        num, den = P.trim(num), P.trim(den)
        k = _real_pos(num[0] / den[0], "element at infinity", tol)
        num = P.trim(P.sub(num, P.mul(P.scale(_S, k), den)))
        br.append(Branch("shunt", "C", C=k) if kind == "Y" else Branch("series", "L", L=k))
        return br, num, den, kind
    if tok == "flip":
        return br, den, num, ("Z" if kind == "Y" else "Y")
    if tok == "dc":
        if den[-1] != 0:
            raise SynthesisError("no pole at DC")
        d1 = den[:-1]                                   # den = s * d1
        k = _real_pos(P.ev(num, 0) / P.ev(d1, 0), "element at DC", tol)
        n1 = P.sub(num, P.scale(d1, k))                 # (num - k d1)/(s d1)
        n1 = P.exact_div(n1, _S, "DC removal")
        br.append(Branch("shunt", "L", L=1 / k) if kind == "Y" else Branch("series", "C", C=1 / k))
        return br, n1, d1, kind
    if tok[0] == "shift":
        # partial removal at infinity or DC so that the immittance vanishes at jw
        _, w, via = tok
        jw = mp.mpc(0, w)
        if via == "inf":                                # remove k s
            if P.deg(num) != P.deg(den) + 1:
                raise SynthesisError("partial removal needs a pole at infinity")
            k = _real_pos(_val(num, den, jw) / jw, "partial removal", tol)
            if k >= num[0] / den[0]:
                raise SynthesisError("partial removal exceeds the pole at infinity")
            n1 = P.sub(num, P.mul(P.scale(_S, k), den))
            d1 = den
            br.append(Branch("shunt", "C", C=k) if kind == "Y" else Branch("series", "L", L=k))
        else:                                           # remove k / s
            if den[-1] != 0:
                raise SynthesisError("partial removal needs a pole at DC")
            k = _real_pos(_val(num, den, jw) * jw, "partial removal", tol)
            if k >= P.ev(num, 0) / P.ev(den[:-1], 0):
                raise SynthesisError("partial removal exceeds the pole at DC")
            n1 = P.sub(P.mul(_S, num), P.scale(den, k))
            d1 = P.mul(_S, den)
            br.append(Branch("shunt", "L", L=1 / k) if kind == "Y" else Branch("series", "C", C=1 / k))
        n1, d1 = _norm(n1, d1)
        return br, n1, d1, kind
    if tok[0] == "res":
        # the immittance has a pole pair at +-jw: remove it as a resonator
        _, w = tok
        jw = mp.mpc(0, w)
        quad = [mp.mpf(1), mp.mpf(0), w * w]
        big = max(abs(c) for c in den)
        if abs(P.ev(den, jw)) > tol * big * (1 + abs(w)) ** len(den):
            raise SynthesisError("no pole at jw")
        q = P.exact_div(den, quad, "pole at jw")
        Kres = _real_pos(P.ev(num, jw) / (2 * jw * P.ev(q, jw)), "resonator residue", tol)
        rest = P.exact_div(P.sub(num, P.mul(P.scale(_S, 2 * Kres), q)), quad, "resonator removal")
        if kind == "Z":       # Z = 2K s/(s^2+w^2): series-arm parallel tank
            br.append(Branch("series", "LC_par", L=2 * Kres / (w * w), C=1 / (2 * Kres)))
        else:                 # Y = 2K s/(s^2+w^2): shunt series resonator
            br.append(Branch("shunt", "LC_ser", L=1 / (2 * Kres), C=2 * Kres / (w * w)))
        return br, rest, q, kind
    raise ValueError(tok)


def _is_resistor(num, den, tol):
    num, den = _norm(num, den)
    return P.deg(num) == 0 and P.deg(den) == 0


POLICIES = {
    "all": None,
    # fewest inductors: zero-shifting partial removals only with capacitors
    "min_L": {("Y", "inf"), ("Z", "dc")},
}


def enumerate_ladders(bp: BandProto, policy="all", max_results=400, time_limit=20.0):
    """Depth-first search over extraction sequences; every realizable ladder found.

    Tokens: 'inf' / 'dc' full removal, ('shift', w, via) zero shifting by partial
    removal, ('res', w) resonator extraction of a pole pair at jw, 'flip'."""
    tol = mp.mpf(10) ** (-(mp.mp.dps // 3))
    shift_ok = POLICIES.get(policy)
    results = []
    t0 = time.perf_counter()
    fin = list(bp.zeros)
    seen = set()

    def dfs(num, den, kind, done, pending, n_dc, n_inf, branches, seq):
        if len(results) >= max_results or time.perf_counter() - t0 > time_limit:
            return
        if len(done) == len(fin) and n_dc == 0 and n_inf == 0 and not pending:
            if _is_resistor(num, den, tol):
                num, den = _norm(num, den)
                val = num[-1] / den[-1]
                rl = val if kind == "Z" else 1 / val
                key = tuple((b.pos, b.kind) for b in branches)
                if rl > 0 and key not in seen:
                    seen.add(key)
                    results.append(LadderResult(list(branches), mp.mpf(1), rl, list(seq)))
            return
        opts = []
        for i in pending:                       # resonators first: they cannot hurt
            opts.append(("res", fin[i]))
        if n_dc:
            opts.append("dc")
        if n_inf:
            opts.append("inf")
        for i, w in enumerate(fin):
            if i in done or i in pending:
                continue
            for via in ("inf", "dc"):
                if shift_ok is None or (kind, via) in shift_ok:
                    opts.append(("shift", w, via))
        if seq and seq[-1] != "flip":
            opts.append("flip")
        for tok in opts:
            try:
                br, n2, d2, k2 = _step(num, den, kind, tok, tol)
            except (SynthesisError, ArithmeticError):
                continue
            d_new, p_new = done, pending
            if isinstance(tok, tuple):
                i = fin.index(tok[1])
                if tok[0] == "shift":
                    p_new = pending | {i}
                else:
                    p_new = pending - {i}
                    d_new = done | {i}
            dfs(n2, d2, k2, d_new, p_new, n_dc - (tok == "dc"), n_inf - (tok == "inf"),
                branches + br, seq + [tok])

    for start in ("Y", "Z"):
        for sgn in (1, -1):
            Fs = P.scale(bp.F, sgn)
            zn, zd = P.add(bp.E, Fs), P.sub(bp.E, Fs)
            num, den = (zd, zn) if start == "Y" else (zn, zd)
            num, den = _norm(num, den)
            dfs(num, den, start, frozenset(), frozenset(), bp.n_dc, bp.n_inf, [], [])
    return results


def count_inductors(res: LadderResult):
    return sum(1 for b in res.branches if b.L is not None)


def best_ladder(results, prefer="min_L"):
    """Pick a ladder from enumerate_ladders(): fewest inductors (for 'min_L'),
    then fewest branches, then the most even inductor values.
    """
    if not results:
        raise SynthesisError("no realizable bandpass ladder found")

    def key(r):
        Ls = [b.L for b in r.branches if b.L is not None]
        spread = float(max(Ls) / min(Ls)) if Ls else 1.0
        nL = count_inductors(r)
        return (nL if prefer == "min_L" else 0, len(r.branches), spread)
    return min(results, key=key)


def to_ladder(res: LadderResult, r0, f0, title="") -> Ladder:
    """Denormalise a w0 = 1, 1-ohm bandpass ladder to r0 ohms and centre f0 Hz."""
    w0 = 2 * math.pi * f0
    out = []
    for b in res.branches:
        L = float(b.L) * r0 / w0 if b.L is not None else None
        C = float(b.C) / (r0 * w0) if b.C is not None else None
        out.append(Branch(b.pos, b.kind, L=L, C=C))
    lad = Ladder(out, r_source=float(res.r_source) * r0, r_load=float(res.r_load) * r0, title=title)
    lad.notes.update(f0=f0, band="bp")
    return lad


# --------------------------------------------------------------------------- approximation

def equiripple(w1, w2, ripple_db, n_pairs, n_dc, finite_zeros=(), iters=60):
    """Equiripple bandpass with an arbitrary distribution of transmission zeros.

    Passband [w1, w2] (normalised), n_pairs reflection zeros inside it, n_dc zeros at DC,
    finite zeros at +-j*finite_zeros, the rest at infinity (degree n = 2*n_pairs).

    With x = w^2, |P(jw)|^2 = Pd(x) = x^n_dc * prod (x_k - x)^2 and F(jw) = Ft(x), a
    polynomial of degree n_pairs.  The characteristic function |K| = |Ft| / sqrt(Pd) is
    made equiripple over [w1^2, w2^2] by the Remez exchange algorithm (weighted minimax
    approximation of zero by a monic polynomial, weight 1/sqrt(Pd)).  Unequal numbers of
    zeros at DC and infinity give the asymmetric weightings used for arithmetically
    symmetric responses (Carassa: about three times as many zeros at infinity).
    """
    n = 2 * n_pairs
    fin = [mp.mpf(z) for z in finite_zeros]
    n_inf = n - n_dc - 2 * len(fin)
    if n_inf < 0:
        raise ValueError("too many transmission zeros for this degree")
    x1, x2 = mp.mpf(w1) ** 2, mp.mpf(w2) ** 2
    N = n_pairs

    def weight(x):
        pd = x ** n_dc
        for z in fin:
            pd *= (z * z - x) ** 2
        return 1 / mp.sqrt(pd)

    # work on t in [-1, 1]: x = c + h t
    c, h = (x1 + x2) / 2, (x2 - x1) / 2
    ref = [mp.cos(mp.pi * (N - i) / N) for i in range(N + 1)]   # Chebyshev extremal points
    for _ in range(iters):
        # solve  t^N - sum_{j<N} a_j t^j = (-1)^i E / w(x_i)   for a_j and E
        A = mp.matrix(N + 1, N + 1)
        b = mp.matrix(N + 1, 1)
        for i, t in enumerate(ref):
            for j in range(N):
                A[i, j] = t ** j
            A[i, N] = (-1) ** i / weight(c + h * t)
            b[i] = t ** N
        sol = mp.lu_solve(A, b)
        a = [sol[j] for j in range(N)]
        Eh = sol[N]

        def err(t):
            return (t ** N - sum(a[j] * t ** j for j in range(N))) * weight(c + h * t)
        # new reference: extrema of err between its sign changes, plus the end points
        grid = [mp.cos(mp.pi * (1 - mp.mpf(k) / 400)) for k in range(401)]
        vals = [err(t) for t in grid]
        new = [grid[0]]
        best, bestv = grid[0], vals[0]
        for t, v in zip(grid[1:], vals[1:]):
            if mp.sign(v) == mp.sign(bestv):
                if abs(v) > abs(bestv):
                    best, bestv = t, v
            else:
                new.append(best) if best != new[-1] else None
                best, bestv = t, v
        new.append(best) if best != new[-1] else None
        if new[-1] != grid[-1]:
            new[-1] = grid[-1]
        if len(new) != N + 1:
            break
        # refine each interior extremum by golden section on |err|
        ref2 = [new[0]]
        for k in range(1, N):
            lo = (ref2[-1] + new[k]) / 2
            hi = (new[k] + new[k + 1]) / 2
            f = lambda t: -abs(err(t))
            gr = (mp.sqrt(5) - 1) / 2
            p1, p2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
            f1, f2 = f(p1), f(p2)
            for _ in range(80):
                if f1 < f2:
                    hi, p2, f2 = p2, p1, f1
                    p1 = hi - gr * (hi - lo)
                    f1 = f(p1)
                else:
                    lo, p1, f1 = p1, p2, f2
                    p2 = lo + gr * (hi - lo)
                    f2 = f(p2)
            ref2.append((lo + hi) / 2)
        ref2.append(new[-1])
        spread = max(abs(err(t)) for t in ref2) / min(abs(err(t)) for t in ref2) - 1
        ref = ref2
        if spread < mp.mpf(10) ** -20:
            break

    # Ft(x) = h^N * (t^N - sum a_j t^j), t = (x - c)/h   (monic in t -> scale irrelevant)
    tpoly = [mp.mpf(1)] + [-a[j] for j in range(N - 1, -1, -1)]   # highest first, in t
    lin = [1 / h, -c / h]                                          # t as a polynomial in x
    Ft = [mp.mpf(0)]
    for i, coef in enumerate(tpoly):
        k = N - i
        term = [coef]
        for _ in range(k):
            term = P.mul(term, lin)
        Ft = P.add(Ft, term)
    # scale so that max |K| over the passband equals eps
    eps = mp.sqrt(mp.power(10, mp.mpf(ripple_db) / 10) - 1)
    kmax = max(abs(P.ev(Ft, c + h * t)) * weight(c + h * t) for t in ref)
    Ft = P.scale(Ft, eps / kmax)
    # polynomials in s
    xs = P.roots(Ft)                                       # reflection zeros: s^2 = -x
    F = [mp.mpf(Ft[0]) * (-1) ** N]
    for r in xs:
        F = P.mul(F, [mp.mpf(1), mp.mpf(0), mp.re(r)])     # (s^2 + x_r)
    F = P.realify(F, "F(s)")
    Ps = [mp.mpf(1)] + [mp.mpf(0)] * n_dc                  # s^n_dc
    for z in fin:
        Ps = P.mul(Ps, [mp.mpf(1), mp.mpf(0), z * z])
    # |E(jw)|^2 = Pd(x) + Ft(x)^2 as a polynomial in x
    Pdx = [mp.mpf(1)] + [mp.mpf(0)] * n_dc
    for z in fin:
        Pdx = P.mul(Pdx, P.mul([mp.mpf(-1), z * z], [mp.mpf(-1), z * z]))
    D = P.trim(P.add(Pdx, P.mul(Ft, Ft)))
    roots_x = P.roots(D)
    s_roots = []
    for r in roots_x:
        sr = mp.sqrt(-mp.mpc(r))
        s_roots.append(sr if mp.re(sr) < 0 else -sr)
    lead = mp.sqrt(abs(D[0]))
    E = P.realify(P.from_roots(s_roots, lead), "E(s)")
    # |P(jw)| = |Ps(jw)| up to sign conventions; make F, P consistent with E on jw
    bp = BandProto(E=E, F=F, P=Ps, B=(mp.mpf(w2) - mp.mpf(w1)), zeros=sorted(fin), n_dc=n_dc,
                   n_inf=n_inf, order=n)
    bp.meta.update(kind="equiripple", w1=mp.mpf(w1), w2=mp.mpf(w2), ripple_db=mp.mpf(ripple_db))
    return bp
