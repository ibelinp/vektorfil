"""PLACER and even-order elliptic regression tests.

Reference numbers are from the S/FILSYN DOS manual examples (see
_ref/sfilsyn/DESIGN_NOTES.md section 7: R1, R19a, R19b).
"""
import math

import numpy as np
import pytest

from vektorfil.synth import Ladder, elliptic, synthesize
from vektorfil.synth.ladder import synthesize_all
from vektorfil.synth.placer import Mask, _type_c_prototype, place, single_edge


def minima(res):
    return [a for _, a, _ in res.extrema]


def test_placer_reproduces_cauer_R1():
    # S/FILSYN R1: n=7, 0.1 dB, 1150 Hz edge (fp = 1000 Hz)
    r = place(order=7, ripple_db=0.1, mask=single_edge(1.15, 45), n_inf=1)
    assert np.array(r.zeros) * 1000 == pytest.approx([1163.9114, 1316.1276, 2061.1872], abs=2e-4)
    assert minima(r) == pytest.approx([45.758398] * 4, abs=1e-5)


def test_placer_finds_order():
    r = place(order=None, ripple_db=0.1, mask=single_edge(1.15, 45), n_inf=1)
    assert r.order == 7 and r.margin_db > 0


def test_type_b_equal_minima_R19a_edge():
    # S/FILSYN's type-B zeros are the equal-minima optimum for edge 1.0865716
    r = place(order=6, ripple_db=0.25, mask=single_edge(1.086571635560898, 0), n_inf=2, zs="B")
    assert np.array(r.zeros) * 1000 == pytest.approx([1101.5717, 1312.4044], abs=3e-4)


def test_type_c_ladder_matches_sfilsyn_R19a():
    p = _type_c_prototype(6, 0.25, [1.1015717, 1.3124044], 2)
    ladders = [Ladder.from_prototype(r, r0=100, fc=1000) for r in synthesize_all(p, first="series")]
    want = [19.056306e-3, 2.042130e-6, 18.193207e-3, 759.370561e-9, 1.772412e-6,
            7.200272e-3, 2.822766e-6, 630.436812e-9]
    found = False
    for lad in ladders:
        vals = [v for b in lad.branches for v in (b.L, b.C) if v is not None]
        if len(vals) == len(want) and np.allclose(vals, want, rtol=2e-6):
            found = True
            assert lad.r_load == pytest.approx(100, rel=1e-9)
    assert found


def test_type_c_meets_mask_exactly_R19b():
    r = place(order=10, ripple_db=0.1, mask=single_edge(1.05, 50), n_inf=2, zs="C")
    m = minima(r)
    assert max(m) - min(m) < 1e-4
    # within 0.01 % of S/FILSYN's (approximately corrected) type-C zeros
    assert np.array(r.zeros) * 1000 == pytest.approx([1053.5502, 1089.2531, 1212.4294, 1674.5595], rel=2e-4)
    lad = synthesize(r.prototype())
    assert float(lad.r_load) == pytest.approx(1.0, rel=1e-8)


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("zs", ["B", "C"])
def test_even_elliptic_realizable(n, zs):
    p = elliptic(n, 0.1, ws=1.3, zs=zs)
    res = synthesize(p)
    lad = Ladder.from_prototype(res, r0=1, fc=1 / (2 * math.pi))
    w = np.array([0.2, 0.7, 1.0, 1.3, 2.0, 5.0])
    sim = lad.response(w / (2 * math.pi))["s21_db"]
    ref = [-float(p.atten_db(x)) for x in w]
    assert sim == pytest.approx(ref, abs=1e-7)
    if zs == "C":
        assert float(res.r_load) == pytest.approx(1.0, rel=1e-8)
    else:
        assert abs(float(res.r_load) - 1.0) > 1e-3


def test_piecewise_mask_equalises_margins():
    # two-level requirement: 40 dB from 1.25 to 2, 70 dB above 2
    mask = Mask([(1.25, 2.0, 40.0), (2.0, math.inf, 70.0)])
    r = place(order=9, ripple_db=0.05, mask=mask, n_inf=1)
    margins = [a - q for _, a, q in r.extrema]
    assert min(margins) == pytest.approx(r.margin_db)
    assert r.margin_db > 0
    res = synthesize(r.prototype())
    assert all(float(v) > 0 for b in res.branches for v in (b.L, b.C) if v is not None)
