"""Lowpass approximation: build E(s), F(s), P(s) for a doubly terminated lossless 2-port.

Normalised lowpass, passband edge at w = 1 rad/s.

    |S21(jw)|^2 = |P|^2 / |E|^2,   |S11(jw)|^2 = |F|^2 / |E|^2,
    E E* = P P* + F F*   (Feldtkeller), E Hurwitz.

Characteristic function K = F/P:
  * equiripple passband with arbitrary transmission zeros -> generalized Chebyshev
    function (recursion of R. J. Cameron, IEEE MTT 1999).  All zeros at infinity
    gives the ordinary Chebyshev; Cauer zero positions give the elliptic filter.
  * maximally flat passband with arbitrary zeros -> inverse-Chebyshev-like
    Butterworth: K = eps * w^N / P(w) normalised; all zeros at infinity gives
    Butterworth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mpmath as mp

from . import poly as P

INF = mp.inf


@dataclass
class Prototype:
    """Polynomials in s (highest power first), real coefficients."""
    E: list
    F: list
    P: list
    eps: mp.mpf                       # ripple factor (passband)
    zeros: list                       # finite transmission zeros w_k (rad/s), positive
    n_inf: int                        # transmission zeros at infinity
    order: int
    meta: dict = field(default_factory=dict)

    def s21(self, w):
        s = mp.mpc(0, w)
        return P.ev(self.P, s) / P.ev(self.E, s)

    def s11(self, w):
        s = mp.mpc(0, w)
        return P.ev(self.F, s) / P.ev(self.E, s)

    def atten_db(self, w):
        return -20 * mp.log10(abs(self.s21(w)))


def ripple_eps(ripple_db):
    return mp.sqrt(mp.power(10, mp.mpf(ripple_db) / 10) - 1)


def _chebyshev_U(order, zeros):
    """Numerator U(w) of the generalized Chebyshev function C(w) = U(w)/Pw(w).

    zeros: list of N transmission zero frequencies (mp.inf allowed), with the
    finite ones listed as +w and -w pairs for a symmetric (real) lowpass.
    """
    assert len(zeros) == order
    U = [mp.mpf(1)]
    V = [mp.mpf(0)]
    w2m1 = [mp.mpf(1), mp.mpf(0), mp.mpf(-1)]  # w^2 - 1
    for wn in zeros:
        a = mp.mpf(0) if wn == INF else 1 / mp.mpf(wn)
        b = mp.sqrt(1 - a * a)
        lin = [mp.mpf(1), -a]  # (w - a)
        U_new = P.add(P.mul(lin, U), P.scale(P.mul(w2m1, V), b))
        V_new = P.add(P.mul(lin, V), P.scale(U, b))
        U, V = U_new, V_new
    return U


def _p_omega(zeros):
    """Pw(w) = prod over finite zeros of (1 - w/wn); chosen so that C(1) = 1."""
    p = [mp.mpf(1)]
    for wn in zeros:
        if wn != INF:
            p = P.mul(p, [-1 / mp.mpf(wn), mp.mpf(1)])
    return p


def _finish(order, zeros_pos, n_inf, K_num_w, Pw, eps_scale, meta):
    """Given K(w) = eps_scale * K_num_w(w) / Pw(w), produce E, F, P in s."""
    F_w = P.scale(K_num_w, eps_scale)
    # |E|^2 on the jw axis: D(w) = Pw^2 + F_w^2  (both real polynomials in w)
    D = P.add(P.mul(Pw, Pw), P.mul(F_w, F_w))
    D = P.trim(D)
    # roots in w; s = j w ; keep left half plane
    rts = P.roots(D)
    s_roots = [mp.mpc(0, 1) * r for r in rts]
    lhp = [r for r in s_roots if mp.re(r) < 0]
    if len(lhp) != order:
        raise ArithmeticError(f"expected {order} LHP roots, got {len(lhp)}")
    lead = mp.sqrt(abs(D[0]))  # |e_N|^2 = leading coeff of D (degree 2N)
    E = P.realify(P.from_roots(lhp, lead), "E(s)")
    F = P.s_to_omega_real(P.trim(F_w) if P.deg(F_w) == order else F_w)
    Ps = P.s_to_omega_real(Pw) if len(Pw) > 1 else [mp.mpf(Pw[0])]
    # Pw has degree = number of finite zeros (even); s_to_omega multiplies by j^deg, real.
    return Prototype(E=E, F=F, P=Ps, eps=eps_scale, zeros=list(zeros_pos), n_inf=n_inf,
                     order=order, meta=meta)


def chebyshev_with_zeros(order, ripple_db, finite_zeros=()):
    """Equiripple passband (ripple_db, edge at w=1) with given finite zeros (w>1).

    finite_zeros: positive frequencies; each produces a +-w pair (2 orders).
    Remaining order goes to zeros at infinity.
    """
    finite_zeros = [mp.mpf(z) for z in finite_zeros]
    n_inf = order - 2 * len(finite_zeros)
    if n_inf < 0:
        raise ValueError("too many finite zeros for this order")
    if any(z <= 1 for z in finite_zeros):
        raise ValueError("finite zeros must lie in the stopband (w > 1)")
    zl = []
    for z in finite_zeros:
        zl += [z, -z]
    zl += [INF] * n_inf
    U = _chebyshev_U(order, zl)
    Pw = _p_omega(zl)
    eps = ripple_eps(ripple_db)
    return _finish(order, finite_zeros, n_inf, U, Pw, eps,
                   {"kind": "chebyshev", "ripple_db": mp.mpf(ripple_db)})


def butterworth_with_zeros(order, cutoff_atten_db=mp.mpf("3.0102999566398"), finite_zeros=()):
    """Maximally flat passband at w=0 with optional finite zeros; attenuation
    cutoff_atten_db at w = 1."""
    finite_zeros = [mp.mpf(z) for z in finite_zeros]
    n_inf = order - 2 * len(finite_zeros)
    if n_inf < 0:
        raise ValueError("too many finite zeros for this order")
    zl = []
    for z in finite_zeros:
        zl += [z, -z]
    Pw = _p_omega(zl)  # Pw(1) != 1 in general; normalise K(1) = eps
    Kn = [mp.mpf(1)] + [mp.mpf(0)] * order  # w^N
    k1 = abs(P.ev(Kn, 1) / P.ev(Pw, 1))
    eps = ripple_eps(cutoff_atten_db) / k1
    return _finish(order, finite_zeros, n_inf, Kn, Pw, eps,
                   {"kind": "butterworth", "cutoff_atten_db": mp.mpf(cutoff_atten_db)})


# --------------------------------------------------------------------------- Bessel

def bessel(order, cutoff_atten_db=mp.mpf("3.0102999566398")):
    """Bessel-Thomson (maximally flat delay) all-pole lowpass, doubly terminated,
    frequency-scaled so the loss is cutoff_atten_db at w = 1."""
    n = order
    # reverse Bessel polynomial: theta_n(s) = sum (2n-k)! / (2^(n-k) k! (n-k)!) s^k
    coef = [mp.factorial(2 * n - k) / (mp.mpf(2) ** (n - k) * mp.factorial(k) * mp.factorial(n - k))
            for k in range(n, -1, -1)]  # highest power first
    E = [mp.mpf(c) for c in coef]
    Pc = E[-1]
    # |F(jw)|^2 = |E(jw)|^2 - Pc^2: build D(w) via E(jw) E(-jw)
    def eval_sq_poly(Ep):
        # real polynomial in w of E(jw)E(-jw)
        Ejw = [c * (mp.mpc(0, 1) ** (len(Ep) - 1 - i)) for i, c in enumerate(Ep)]
        Emjw = [c * (mp.mpc(0, -1) ** (len(Ep) - 1 - i)) for i, c in enumerate(Ep)]
        return P.realify(P.mul(Ejw, Emjw), "|E|^2")
    D = P.sub(eval_sq_poly(E), [Pc * Pc])
    D = P.trim(D)
    rts = P.roots(D)
    s_roots = sorted((mp.mpc(0, 1) * r for r in rts), key=lambda z: (mp.re(z), mp.im(z)))
    # choose half the roots (left half plane; split the pairs on the jw axis)
    chosen, axis = [], []
    for r in s_roots:
        if abs(mp.re(r)) < mp.mpf(10) ** -15:
            axis.append(r)
        elif mp.re(r) < 0:
            chosen.append(r)
    axis.sort(key=lambda z: mp.im(z))
    chosen += axis[::2]
    lead = mp.sqrt(abs(D[0]))
    F = P.realify(P.from_roots(chosen, lead), "F(s)")
    proto = Prototype(E=E, F=F, P=[Pc], eps=mp.mpf(1), zeros=[], n_inf=n, order=n,
                      meta={"kind": "bessel"})
    # scale frequency so that the loss equals cutoff_atten_db at w = 1
    lo, hi = mp.mpf("1e-3"), mp.mpf(10)
    for _ in range(90):           # geometric bisection: 90 halvings are far below 1e-20
        mid = mp.sqrt(lo * hi)
        if proto.atten_db(mid) < cutoff_atten_db:
            lo = mid
        else:
            hi = mid
    wa = mp.sqrt(lo * hi)

    def scale(p):  # p(s) -> p(wa s)
        k = len(p) - 1
        return [c * wa ** (k - i) for i, c in enumerate(p)]
    proto.E, proto.F = scale(E), scale(F)
    proto.meta["delay_s_norm"] = wa  # group delay at DC, normalised (1 s for wa = 1)
    return proto


# --------------------------------------------------------------------------- elliptic

def cauer_zeros(order, ws):
    """Transmission zeros of the classical (Cauer) elliptic lowpass of odd order
    with passband edge 1 and stopband edge ws (selectivity xi = ws).

    Elliptic rational function R_n(xi, x): poles at xi / cd((2i-1)K/n, 1/xi).
    For odd n one pole is at infinity; returns the finite positive ones.
    """
    if order % 2 == 0:
        raise ValueError("classical Cauer zeros here are for odd order; use equal_minima for even")
    k = 1 / mp.mpf(ws)
    m = k * k
    K = mp.ellipk(m)
    zs = []
    for i in range(1, (order - 1) // 2 + 1):
        u = (2 * i - 1) * K / order
        cd = mp.ellipfun('cd', u, m=m)
        zs.append(mp.mpf(ws) / cd)
    return sorted(zs)


def stopband_minima(proto, ws):
    """Attenuation (dB) at ws and at each local minimum between successive zeros."""
    zs = sorted(proto.zeros)
    vals = [proto.atten_db(ws)]
    edges = list(zs) + ([None] if proto.n_inf > 0 else [])
    for a, b in zip(edges[:-1], edges[1:]):
        lo = a * (1 + mp.mpf(10) ** -12)
        hi = (b * (1 - mp.mpf(10) ** -12)) if b is not None else a * 1e6
        f = lambda w: proto.atten_db(w)
        # golden-section on log frequency
        x0, x1 = mp.log(lo), mp.log(hi)
        g = (mp.sqrt(5) - 1) / 2
        c, d = x1 - g * (x1 - x0), x0 + g * (x1 - x0)
        fc, fd = f(mp.e ** c), f(mp.e ** d)
        for _ in range(120):
            if fc < fd:
                x1, d, fd = d, c, fc
                c = x1 - g * (x1 - x0)
                fc = f(mp.e ** c)
            else:
                x0, c, fc = c, d, fd
                d = x0 + g * (x1 - x0)
                fd = f(mp.e ** d)
        vals.append(min(fc, fd))
    return vals


def elliptic(order, ripple_db, ws=None, amin_db=None, zs="C"):
    """Cauer-Chebyshev lowpass.  Give ws (stopband edge) or amin_db.

    Odd order: classical closed form (equal terminations).
    Even order: zs='C' (equal terminations, default) or 'B' (unequal), both with two
    zeros at infinity, computed by PLACER as the exact equal-minima optimum.
    """
    if (ws is None) == (amin_db is None):
        raise ValueError("specify exactly one of ws, amin_db")
    if order % 2 == 0:
        from . import placer
        if ws is not None:
            res = placer.place(order, float(ripple_db), placer.single_edge(float(ws), 0.0), n_inf=2, zs=zs)
        else:
            # ws whose equal minima equal amin_db: secant on log ws, PLACER warm-started.
            # Initial guess: the odd-order Cauer of one degree less meets Amin a bit later.
            state = {"zeros": None, "res": None, "lw": None}

            def margin(lw):
                start = None
                if state["zeros"] is not None:  # zeros scale roughly with the stopband edge
                    k = math.exp(lw - state["lw"])
                    start = [max(z * k, math.exp(lw) * 1.0001) for z in state["zeros"]]
                res = placer.place(order, float(ripple_db), placer.single_edge(math.exp(lw), float(amin_db)),
                                   n_inf=2, zs=zs, start_zeros=start)
                state.update(zeros=res.b_zeros if zs == "C" else res.zeros, res=res, lw=lw)
                return res.margin_db

            # bracket: odd order n+1 reaches Amin at a lower ws, odd order n-1 at a higher one
            from scipy.optimize import brentq
            lo = math.log(float(elliptic(order + 1, ripple_db, amin_db=amin_db).meta["ws"]))
            hi = math.log(float(elliptic(order - 1, ripple_db, amin_db=amin_db).meta["ws"])) \
                if order > 2 else lo + 2.0
            while margin(hi) < 0:
                hi += 0.5
            lw = brentq(margin, lo, hi, xtol=1e-12)
            margin(lw)
            res = state["res"]
        proto = res.prototype()
        proto.meta["placer"] = res
        return proto
    if ws is not None:
        return chebyshev_with_zeros(order, ripple_db, cauer_zeros(order, ws))
    # solve ws for the requested Amin (monotonic: larger ws -> larger Amin), in float
    from . import placer
    from scipy.optimize import brentq
    eps = float(ripple_eps(ripple_db))

    def excess(lw):
        w = math.exp(lw)
        zs = [float(z) for z in cauer_zeros(order, w)]
        return float(placer.atten_db(w, zs, 1, eps)) - float(amin_db)
    hi = math.log(2.0)
    while excess(hi) < 0:
        hi *= 2
    lw = brentq(excess, math.log(1.0000001), hi, xtol=1e-14)
    p = chebyshev_with_zeros(order, ripple_db, cauer_zeros(order, math.exp(lw)))
    p.meta["ws"] = math.exp(lw)
    return p
