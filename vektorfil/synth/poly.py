"""High-precision real polynomial helpers.

Polynomials are lists of mpmath numbers, highest power first (numpy order).
Synthesis by repeated partial removal loses digits at every step, so all of
the synthesis core runs at extended precision (S/FILSYN used REAL*16 for the
same reason).
"""
from __future__ import annotations

import mpmath as mp

mp.mp.dps = 50

Poly = list


def trim(p: Poly, rel_tol=None) -> Poly:
    """Drop leading coefficients that are negligible relative to the largest one.

    Every extraction step cancels a leading coefficient and leaves rounding residue that
    grows with the number of steps; half the working digits is a safe dividing line
    between residue and a genuine coefficient (at 50 digits: 1e-25)."""
    if rel_tol is None:
        rel_tol = mp.mpf(10) ** (-(mp.mp.dps // 2))
    if not p:
        return [mp.mpf(0)]
    scale = max(abs(c) for c in p)
    if scale == 0:
        return [mp.mpf(0)]
    i = 0
    while i < len(p) - 1 and abs(p[i]) <= rel_tol * scale:
        i += 1
    return list(p[i:])


def deg(p: Poly) -> int:
    return len(trim(p)) - 1


def add(a: Poly, b: Poly) -> Poly:
    """Sum of two polynomials."""
    n = max(len(a), len(b))
    a = [mp.mpf(0)] * (n - len(a)) + list(a)
    b = [mp.mpf(0)] * (n - len(b)) + list(b)
    return [x + y for x, y in zip(a, b)]


def scale(a: Poly, k) -> Poly:
    return [k * x for x in a]


def sub(a: Poly, b: Poly) -> Poly:
    return add(a, scale(b, -1))


def mul(a: Poly, b: Poly) -> Poly:
    """Product of two polynomials."""
    out = [mp.mpf(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def divmod_(a: Poly, b: Poly):
    """Polynomial long division a = q*b + r."""
    a = list(trim(a))
    b = trim(b)
    if len(a) < len(b):
        return [mp.mpf(0)], a
    q = [mp.mpf(0)] * (len(a) - len(b) + 1)
    for i in range(len(q)):
        c = a[i] / b[0]
        q[i] = c
        for j, y in enumerate(b):
            a[i + j] -= c * y
    r = a[len(q):] or [mp.mpf(0)]
    return q, r


def exact_div(a: Poly, b: Poly, what="division") -> Poly:
    """Divide when b is known to be a factor of a; checks the remainder is tiny."""
    q, r = divmod_(a, b)
    big = max(abs(c) for c in a) or 1
    if max(abs(c) for c in r) > big * mp.mpf(10) ** (-(mp.mp.dps // 2)):
        raise ArithmeticError(f"{what}: remainder not negligible ({mp.nstr(max(abs(c) for c in r), 5)})")
    return q


def ev(p: Poly, x):
    return mp.polyval(p, x)


def from_roots(roots, lead=1) -> Poly:
    """Polynomial lead * prod(s - r) from its roots."""
    p = [mp.mpc(lead)]
    for r in roots:
        p = mul(p, [mp.mpf(1), -r])
    return p


def realify(p: Poly, what="polynomial") -> Poly:
    """Convert a polynomial with (numerically) real coefficients to mpf."""
    big = max(abs(c) for c in p) or 1
    out = []
    for c in p:
        c = mp.mpc(c)
        if abs(c.imag) > big * mp.mpf(10) ** (-(mp.mp.dps // 2)):
            raise ArithmeticError(f"{what} has complex coefficients")
        out.append(mp.mpf(c.real))
    return out


def roots(p: Poly):
    """All roots (complex) of a polynomial, at the working precision."""
    p = trim(p)
    if len(p) <= 1:
        return []
    return mp.polyroots(p, maxsteps=400, extraprec=4 * mp.mp.dps)


def s_to_omega_real(p_omega: Poly) -> Poly:
    """Given a real polynomial U(w) of definite parity N, return the real polynomial F(s)
    with F(jw) = j**N * U(w), i.e. F(s) = j**N * U(s/j)."""
    n = len(p_omega) - 1
    out = []
    for k, c in enumerate(p_omega):  # term c * w**(n-k)
        p = n - k
        # j**n * (s/j)**p = j**(n-p) * s**p
        f = mp.mpc(0, 1) ** (n - p)
        out.append(c * f)
    return realify(out, "F(s)")
