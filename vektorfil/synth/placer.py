"""PLACER: place lowpass transmission zeros to meet a piecewise stopband mask with the
largest equal safety margin (minimax).

Normalised lowpass: passband edge w = 1, passband shape fixed (equiripple or maximally
flat); only the finite transmission zeros move.  The stopband excess loss

    margin(w) = A(w) - A_req(w)

is evaluated at every local minimum between zeros and at every mask breakpoint; the
zeros are moved until the smallest of these is as large as possible, which at the
optimum makes them all equal (the "safety factors" of S/FILSYN).

Even-degree elliptic types:
    'B'  two zeros at infinity, equiripple, unequal terminations (computed directly here)
    'C'  type B followed by w'^2 = (w^2 - a^2)/(1 - a^2), a = lowest reflection zero,
         which puts a reflection zero at DC: equal terminations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mpmath as mp
import numpy as np
from scipy.optimize import fsolve, minimize, minimize_scalar

from . import approx

LN10_10 = 10 / math.log(10)


# ---------------------------------------------------------------- fast float model

def _log_abs_cosh(S):
    """log|cosh(S)| for complex S without overflow."""
    S = np.where(S.real < 0, -S, S)
    return S.real + np.log(np.abs(1 + np.exp(-2 * S))) - math.log(2)


def log_abs_K(w, zeros, n_inf, shape="cheby", order=None):
    """log |characteristic function| (natural log, before eps scaling) at w (array)."""
    w = np.asarray(w, float)
    if shape == "cheby":
        wc = w.astype(complex)
        S = n_inf * np.arccosh(wc)
        for z in zeros:
            for zz in (z, -z):
                x = (w - 1 / zz) / (1 - w / zz)
                S = S + np.arccosh(x.astype(complex))
        return _log_abs_cosh(S)
    # maximally flat: K = w^N / prod(1 - w^2/z^2), normalised to |K(1)| = 1
    n = order
    lk = n * np.log(np.abs(w) + 1e-300)
    for z in zeros:
        lk = lk - np.log(np.abs(1 - (w / z) ** 2) + 1e-300)
    lk1 = -sum(math.log(abs(1 - 1 / z ** 2)) for z in zeros)
    return lk - lk1


def atten_db(w, zeros, n_inf, eps, shape="cheby", order=None, zs_a2=0.0):
    """Insertion loss (dB) of the lossless prototype.  zs_a2 > 0 applies the type-C map:
    the function is the type-B one evaluated at W = sqrt((1 - a^2) w^2 + a^2)."""
    w = np.asarray(w, float)
    if zs_a2:
        w = np.sqrt((1 - zs_a2) * w * w + zs_a2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lk = log_abs_K(w, zeros, n_inf, shape, order)
        val = LN10_10 * np.logaddexp(0.0, 2 * (math.log(eps) + lk))
    return np.where(np.isnan(val), np.inf, val)


# ---------------------------------------------------------------- mask

@dataclass
class Mask:
    """Piecewise-constant stopband requirement for the normalised lowpass.
    segments: list of (w_start, w_stop, A_dB); w_stop may be math.inf."""
    segments: list

    def req(self, w):
        """Required attenuation (dB) at w; -inf outside the stopband segments."""
        w = np.asarray(w, float)
        r = np.full(w.shape, -np.inf)
        for a, b, A in self.segments:
            r = np.where((w >= a) & (w <= b), np.maximum(r, A), r)
        return r

    def breakpoints(self):
        """Mask corner frequencies."""
        pts = []
        for a, b, _ in self.segments:
            pts.append(a)
            if math.isfinite(b):
                pts.append(b)
        return sorted(set(pts))

    @property
    def first_edge(self):
        return min(a for a, _, _ in self.segments)


def single_edge(ws, amin_db):
    return Mask([(float(ws), math.inf, float(amin_db))])


# ---------------------------------------------------------------- result

@dataclass
class PlacerResult:
    """Result of place(): zero locations, margins at every extremum, and the
    type-B/C bookkeeping needed to rebuild the exact prototype.
    """
    zeros: list                 # finite zeros of the final function (normalised, ascending)
    n_inf: int
    order: int
    ripple_db: float
    shape: str
    zs: str                     # 'A' (none), 'B', or 'C'
    margin_db: float            # equalised worst-case excess loss
    extrema: list               # (w, A, A_req) at each margin point
    zs_a2: float = 0.0
    b_zeros: list = field(default_factory=list)   # type-B zeros before the C map
    history: list = field(default_factory=list)

    def prototype(self):
        """Exact (mpmath) prototype for synthesis."""
        if self.shape == "cheby":
            if self.zs == "C":
                return _type_c_prototype(self.order, self.ripple_db, self.b_zeros, self.n_inf)
            return approx.chebyshev_with_zeros(self.order, self.ripple_db, self.zeros)
        return approx.butterworth_with_zeros(self.order, self.ripple_db, self.zeros)


# ---------------------------------------------------------------- core optimiser

def _extrema(zeros, n_inf, eps, shape, order, mask, a2):
    """Margin evaluation points: mask breakpoints + loss minimum in every interval between
    consecutive singular points (zeros / breakpoints) that lies inside the stopband."""
    zs = sorted(zeros)
    stop_lo = mask.first_edge
    stop_hi = max(b for _, b, _ in mask.segments)
    hi_end = (max(zs + [stop_lo]) * 50) if not math.isfinite(stop_hi) else stop_hi
    sing = sorted(set([stop_lo, hi_end] + [z for z in zs if stop_lo < z < hi_end] +
                      [p for p in mask.breakpoints() if stop_lo <= p <= hi_end]))

    def margin(w):
        return atten_db(w, zs, n_inf, eps, shape, order, a2) - mask.req(w)

    bps = mask.breakpoints()
    pts = list(zip(bps, (float(v) for v in margin(np.array(bps, float)))))
    zset = set(zs)
    ivals = []
    for a, b in zip(sing[:-1], sing[1:]):
        if b <= a * (1 + 1e-12):
            continue
        if not np.isfinite(mask.req(math.sqrt(a * b))):
            continue  # not inside a stopband segment
        ea = a * (1 + 1e-9) if a in zset else a
        eb = b * (1 - 1e-9) if b in zset else b
        ivals.append((math.log(ea), math.log(eb)))
    if not ivals:
        return pts
    # loss is unimodal between two zeros: zoom in on all intervals at once (log w)
    K = 17
    lo = np.array([i[0] for i in ivals])
    hi = np.array([i[1] for i in ivals])
    lo0, hi0 = lo.copy(), hi.copy()
    rows = np.arange(len(ivals))
    for _ in range(14):
        x = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, K)[None, :]
        v = margin(np.exp(x).ravel()).reshape(x.shape)
        k = np.argmin(v, axis=1)
        lo, hi = x[rows, np.maximum(k - 1, 0)], x[rows, np.minimum(k + 1, K - 1)]
    x_min, v_min = x[rows, k], v[rows, k]
    span = hi0 - lo0
    for xm, vm, a, b, sp in zip(x_min, v_min, lo0, hi0, span):
        # interior minima only (an end point minimum is already a breakpoint)
        if a + 1e-7 * sp < xm < b - 1e-7 * sp:
            pts.append((math.exp(xm), float(vm)))
    return pts


def _type_b_a2(zeros, n_inf, order, ripple_db):
    """Lowest positive reflection zero a of the type-B equiripple function -> a^2.
    Float: C(w) = cosh(S(w)) is real in the passband; find its first sign change."""
    def C(w):
        w = np.atleast_1d(np.asarray(w, float)).astype(complex)
        S = n_inf * np.arccosh(w)
        for z in zeros:
            for zz in (z, -z):
                S = S + np.arccosh(((w - 1 / zz) / (1 - w / zz)))
        return np.cosh(S).real
    grid = np.linspace(0.0, 1.0, 4 * order + 1)
    vals = C(grid)
    for i in range(len(grid) - 1):
        if vals[i] == 0:
            return float(grid[i]) ** 2
        if vals[i] * vals[i + 1] < 0:
            from scipy.optimize import brentq
            a = brentq(lambda w: C(w)[0], grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15)
            return a * a
    return _type_b_a2_exact(zeros, n_inf, order, ripple_db)


def _type_b_a2_exact(zeros, n_inf, order, ripple_db):
    """High-precision fallback for _type_b_a2 (polynomial roots at 50 digits)."""
    zl = []
    for z in zeros:
        zl += [mp.mpf(z), -mp.mpf(z)]
    zl += [mp.inf] * n_inf
    U = approx._chebyshev_U(order, zl)
    from . import poly as P
    r = [mp.re(x) for x in P.roots(U) if abs(mp.im(x)) < 1e-20 and mp.re(x) > 0]
    return float(min(r)) ** 2


def _type_c_prototype(order, ripple_db, b_zeros, n_inf):
    """Exact type-C prototype: type-B U(W), Pw(W) with W^2 = (1-a^2) w^2 + a^2."""
    from . import poly as P
    zl = []
    for z in b_zeros:
        zl += [mp.mpf(z), -mp.mpf(z)]
    zl += [mp.inf] * n_inf
    U = approx._chebyshev_U(order, zl)
    Pw = approx._p_omega(zl)
    a2 = mp.mpf(_type_b_a2(b_zeros, n_inf, order, ripple_db))
    # refine a^2 at full precision
    rts = [mp.re(x) for x in P.roots(U) if abs(mp.im(x)) < mp.mpf(10) ** -20 and mp.re(x) > 0]
    a2 = min(rts) ** 2

    def subst(poly_w):
        # poly in W (even) -> poly in w via W^2 = c w^2 + a2
        n = len(poly_w) - 1
        out = [mp.mpf(0)]
        c = 1 - a2
        W2 = [c, mp.mpf(0), a2]
        tiny = max(abs(x) for x in poly_w) * mp.mpf(10) ** (-(mp.mp.dps - 10))
        for k, coef in enumerate(poly_w):
            p = n - k
            if abs(coef) <= tiny:
                continue
            if p % 2:
                raise ArithmeticError("type-C map needs even polynomials")
            term = [mp.mpf(1)]
            for _ in range(p // 2):
                term = P.mul(term, W2)
            out = P.add(out, P.scale(term, coef))
        return P.trim(out)

    Uc, Pc = subst(U), subst(Pw)
    eps = approx.ripple_eps(ripple_db)
    c_zeros = [mp.sqrt((mp.mpf(z) ** 2 - a2) / (1 - a2)) for z in b_zeros]
    proto = approx._finish(order, sorted(c_zeros), n_inf, Uc, Pc, eps,
                           {"kind": "elliptic-C", "ripple_db": mp.mpf(ripple_db), "a2": a2})
    return proto


def _initial_zeros(m, mask):
    """Starting zeros for the optimiser: the Cauer zeros of a comparable odd order."""
    ws = mask.first_edge
    if m == 0:
        return []
    try:  # Cauer zeros of the odd function with the same number of finite zeros
        return [float(z) for z in approx.cauer_zeros(2 * m + 1, max(ws, 1.0005))]
    except Exception:
        return list(np.geomspace(ws * 1.02, ws * 3, m))


def _optimise(m, n_inf, ripple_db, mask, shape, order, fixed, a2, x0=None, tol=1e-9):
    """Minimax optimisation of m movable zeros (log-parametrised).

    Stage 1 (cold starts with a non-classical structure): Nelder-Mead on a smoothed
    minimum (soft-min) of the exact margins.  Stage 2: Remez-style exchange that equalises
    the m+1 smallest extrema with fsolve.  A failed warm start falls back to a cold start.
    """
    eps = math.sqrt(10 ** (ripple_db / 10) - 1)
    ws = mask.first_edge
    warm = x0 is not None
    x0 = np.log(np.asarray(x0 if warm else _initial_zeros(m, mask), float))
    if m == 0:
        pts = _extrema(list(fixed), n_inf, eps, shape, order, mask, a2)
        return [], pts

    def zeros_of(x):
        return sorted(list(np.exp(x)) + list(fixed))

    lo_b = math.log(ws * (1 + 1e-7))

    def margins(x):
        """Exact margins at the extremal points, ordered by frequency (None if degenerate)."""
        if np.any(x <= lo_b):
            return None
        z = zeros_of(x)
        if len(set(np.round(z, 10))) < len(z):
            return None  # coincident zeros
        pts = _extrema(z, n_inf, eps, shape, order, mask, a2)
        return [p[1] for p in sorted(pts, key=lambda p: p[0])]

    def worst(x):
        mv = margins(x)
        return -np.inf if mv is None else min(mv)

    # stage 1: maximise a smoothed minimum (soft-min) of the exact margins with
    # Nelder-Mead, tightening the smoothing: robust on this nonsmooth landscape.
    # Skipped for warm starts and for the classical structure (exactly m+1 extrema,
    # e.g. a single stopband edge), where the exchange step alone converges.
    x = x0
    mv0 = margins(x0)
    need_global = not warm and (mv0 is None or len(mv0) != m + 1)
    for tau in ((3.0, 1.0, 0.3, 0.1) if need_global else ()):
        def obj(v, tau=tau):
            mv = margins(v)
            if mv is None:
                return 1e6
            mv = np.asarray(mv)
            return -(mv.min() - tau * np.log(np.sum(np.exp(-(mv - mv.min()) / tau))))
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 400 * (m + 1)})
        if worst(r.x) > worst(x):
            x = r.x

    # stage 2: Remez-style exchange -- equalise the m+1 smallest extrema exactly
    for _ in range(6):
        mv = margins(x)
        if mv is None or len(mv) < m + 1:
            break
        active = sorted(np.argsort(mv)[:m + 1])
        n_struct = len(mv)

        def eqs(v, active=active, n_struct=n_struct):
            mv2 = margins(v)
            if mv2 is None or len(mv2) != n_struct:
                return np.full(m, 1e3)
            a = [mv2[i] for i in active]
            return np.diff(a)
        try:
            sol, _, ier, _ = fsolve(eqs, x, full_output=True, xtol=tol)
        except Exception:
            break
        if ier != 1 or worst(sol) < worst(x) - 1e-9:
            break
        improved = worst(sol) - worst(x)
        x = sol
        if improved < 1e-9:
            break
    pts = _extrema(zeros_of(x), n_inf, eps, shape, order, mask, a2)
    if warm:
        mv = sorted(p[1] for p in pts)
        if len(mv) < m + 1 or mv[m] - mv[0] > 1e-6:
            # warm start landed in a different structure: redo from the cold start
            return _optimise(m, n_inf, ripple_db, mask, shape, order, fixed, a2, None, tol)
    return sorted(np.exp(x)), pts


def place(order=None, ripple_db=0.1, mask=None, n_inf=1, shape="cheby", zs=None,
          fixed_zeros=(), max_order=31, start_zeros=None):
    """Run PLACER.

    order: total degree; if None, the smallest degree (with this n_inf) meeting the mask.
    n_inf: zeros at infinity (1 for odd elliptic, 2 for even type B/C, more for faster roll-off).
    zs: None/'A' plain, 'B' (same as plain with n_inf=2), 'C' equal-termination even degree.
    start_zeros: warm start (movable zeros; type-B domain for zs='C'), e.g. from a previous run.
    """
    if mask is None:
        raise ValueError("a stopband mask is required")
    zs = (zs or "A").upper()
    if zs in ("B", "C") and n_inf < 2:
        n_inf = 2
    fixed = [float(z) for z in fixed_zeros]

    def run(n):
        m2 = n - n_inf - 2 * len(fixed)
        if m2 < 0 or m2 % 2:
            return None
        m = m2 // 2
        if zs == "C":
            # optimise the type-B function against the mask mapped back to the B domain;
            # a depends on the B zeros, so iterate the map to a fixed point.
            x0 = list(start_zeros) if start_zeros is not None and len(start_zeros) == m else None
            a2 = _type_b_a2(list(x0) + fixed, n_inf, n, ripple_db) if x0 else 0.0
            hist = []
            for _ in range(30):
                zb, pts = _optimise(m, n_inf, ripple_db, mask, shape, n, fixed, a2, x0)
                a2_new = _type_b_a2(list(zb) + fixed, n_inf, n, ripple_db)
                hist.append(a2_new)
                x0 = zb
                if abs(a2_new - a2) < 1e-13:
                    a2 = a2_new
                    break
                a2 = a2_new
            zb, pts = _optimise(m, n_inf, ripple_db, mask, shape, n, fixed, a2, x0)
            zc = [math.sqrt((z * z - a2) / (1 - a2)) for z in sorted(list(zb) + fixed)]
            # the optimisation used W-domain zeros against the w-domain mask through the map
            return PlacerResult(zc, n_inf, n, ripple_db, shape, "C",
                                min(p[1] for p in pts), _annotate(pts, mask, zb, n_inf, ripple_db, shape, n, a2),
                                zs_a2=a2, b_zeros=sorted(list(zb) + fixed), history=hist)
        warm = list(start_zeros) if start_zeros is not None and len(start_zeros) == m else None
        z, pts = _optimise(m, n_inf, ripple_db, mask, shape, n, fixed, 0.0, warm)
        allz = sorted(list(z) + fixed)
        return PlacerResult(allz, n_inf, n, ripple_db, shape, zs if zs == "B" else "A",
                            min(p[1] for p in pts), _annotate(pts, mask, allz, n_inf, ripple_db, shape, n, 0.0))

    if order is not None:
        res = run(order)
        if res is None:
            raise ValueError("order incompatible with n_inf / fixed zeros")
        return res
    n = n_inf + 2 * len(fixed)
    while n <= max_order:
        res = run(n)
        if res is not None and res.margin_db >= 0:
            return res
        n += 1
    raise ValueError("mask not met up to max_order")


def _annotate(pts, mask, zeros, n_inf, ripple_db, shape, order, a2):
    """List (w, attenuation, requirement) at every margin point, for reporting."""
    eps = math.sqrt(10 ** (ripple_db / 10) - 1)
    out = []
    for w, mrg in sorted(pts):
        A = float(atten_db(w, zeros, n_inf, eps, shape, order, a2))
        out.append((w, A, float(mask.req(w))))
    return out
