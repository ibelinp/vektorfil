import numpy as np
import pytest

from vektorfil.synth import (Ladder, butterworth_with_zeros, chebyshev_with_zeros, elliptic,
                            synthesize)
from vektorfil.synth.approx import stopband_minima


def values(res):
    return [float(b.L if b.L is not None else b.C) for b in res.branches]


def test_butterworth_g_values():
    # g_k = 2 sin((2k-1) pi / 2N)
    for n in range(2, 9):
        res = synthesize(butterworth_with_zeros(n))
        g = [2 * np.sin((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)]
        assert values(res) == pytest.approx(g, rel=1e-12)
        assert float(res.r_load) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("ripple", [0.01, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 9])
def test_chebyshev_matches_closed_form(n, ripple):
    # Matthaei, Young & Jones closed-form g values
    beta = np.log(1 / np.tanh(ripple * np.log(10) / 40))
    gam = np.sinh(beta / (2 * n))
    a = [np.sin((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)]
    b = [gam ** 2 + np.sin(k * np.pi / n) ** 2 for k in range(1, n + 1)]
    g = [2 * a[0] / gam]
    for k in range(1, n):
        g.append(4 * a[k - 1] * a[k] / (b[k - 1] * g[k - 1]))
    gl = 1.0 if n % 2 else 1 / np.tanh(beta / 4) ** 2
    res = synthesize(chebyshev_with_zeros(n, ripple))
    assert values(res) == pytest.approx(g, rel=1e-9)
    # even order: load is gl or 1/gl depending on which end is the reference
    assert float(res.r_load) == pytest.approx(gl, rel=1e-9) or float(res.r_load) == pytest.approx(1 / gl, rel=1e-9)


@pytest.mark.parametrize("n,ws", [(3, 1.2), (5, 1.5), (7, 1.1), (9, 1.05)])
def test_elliptic_equal_minima_and_realization(n, ws):
    p = elliptic(n, 0.1, ws=ws)
    mins = [float(v) for v in stopband_minima(p, ws)]
    assert max(mins) - min(mins) < 1e-6
    res = synthesize(p)
    assert all(float(v) > 0 for v in values(res))
    assert float(res.r_load) == pytest.approx(1.0, rel=1e-9)


def test_elliptic_amin_spec():
    p = elliptic(5, 0.1, amin_db=60)
    assert float(stopband_minima(p, min(p.zeros) * 0.999)[1]) == pytest.approx(60, abs=1e-6)


def test_ladder_simulation_matches_transfer_function():
    p = elliptic(7, 0.05, ws=1.2)
    res = synthesize(p)
    lad = Ladder.from_prototype(res, r0=50, fc=1 / (2 * np.pi))  # stays normalised in frequency
    w = np.array([0.1, 0.5, 0.9, 1.0, 1.2, 1.5, 3.0, 10.0])
    sim = lad.response(w / (2 * np.pi))["s21_db"]
    ref = [-float(p.atten_db(x)) for x in w]
    assert sim == pytest.approx(ref, abs=1e-7)


def test_series_first_dual():
    p = chebyshev_with_zeros(5, 0.1)
    a = values(synthesize(p, first="shunt"))
    b = values(synthesize(p, first="series"))
    assert a == pytest.approx(b, rel=1e-10)  # odd Chebyshev is self-dual in g values
