import math

import numpy as np
import pytest

from vektorfil.filters import (SERIES_FIRST, SHUNT_FIRST, SYMMETRIC, ZIGZAG, FilterSpec, design,
                              estimate_orders)


def s21(res, f):
    return 20 * np.log10(np.abs(res.ladder.sparams(np.atleast_1d(f))[1]))


def count(res):
    b = res.ladder.branches
    return sum(x.L is not None for x in b), sum(x.C is not None for x in b)


def test_chebyshev_lowpass_example():
    # 50 ohm, Chebyshev LP, 0.25 dB, 7th order, 70 MHz, min C
    r = design(FilterSpec(ftype="Lowpass", shape="Chebyshev", subtype=SERIES_FIRST, order=7,
                          fc=70e6, ripple_db=0.25))
    assert r.ladder.r_load == pytest.approx(50)
    assert s21(r, 70e6)[0] == pytest.approx(-0.25, abs=1e-6)
    assert r.ladder.branches[0].pos == "series"            # minimum C starts with series L
    assert count(r) == (4, 3)


@pytest.mark.parametrize("shape,aa", [("Butterworth", 1.0), ("Chebyshev", 3.0), ("Bessel", 6.0)])
@pytest.mark.parametrize("ftype", ["Lowpass", "Highpass"])
def test_attenuation_at_cutoff(shape, aa, ftype):
    r = design(FilterSpec(ftype=ftype, shape=shape, order=5, fc=10e6, ripple_db=0.1, aa_db=aa))
    assert s21(r, 10e6)[0] == pytest.approx(-aa, abs=1e-6)


@pytest.mark.parametrize("ftype", ["Lowpass", "Highpass"])
@pytest.mark.parametrize("order", [5, 6])
@pytest.mark.parametrize("sub", [SHUNT_FIRST, SERIES_FIRST])
def test_elliptic_topologies(ftype, order, sub):
    r = design(FilterSpec(ftype=ftype, shape="Elliptic", order=order, subtype=sub, amin_db=50))
    nl, nc = count(r)
    fewer_l = (sub == SHUNT_FIRST) == (ftype == "Lowpass")   # highpass swaps L and C
    assert (nl < nc) if fewer_l else (nc < nl)
    assert r.info["amin_db"] == pytest.approx(50, abs=1e-6)
    # passband edge at the ripple, stopband edge at Amin
    assert s21(r, 100e6)[0] == pytest.approx(-0.1, abs=1e-6)
    assert s21(r, r.info["fs"])[0] == pytest.approx(-50, abs=1e-5)
    assert r.ladder.r_load == pytest.approx(50, rel=1e-8)


def test_losses_reduce_transmission():
    lossless = design(FilterSpec(shape="Chebyshev", order=5, fc=100e6))
    lossy = design(FilterSpec(shape="Chebyshev", order=5, fc=100e6, q_l=50, q_c=200))
    assert s21(lossy, 50e6)[0] < s21(lossless, 50e6)[0] - 0.1


def test_order_help():
    o = estimate_orders("Lowpass", 1e6, 1.5e6, 0.1, 40)
    assert o["Butterworth"] > o["Chebyshev"] > o["Elliptic"]


# ----------------------------------------------------------------- bandpass / bandstop
from vektorfil.filters import SHAPES, subtypes  # noqa: E402


@pytest.mark.parametrize("ftype", ["Bandpass", "Bandstop"])
@pytest.mark.parametrize("shape", SHAPES)
def test_band_filters_hit_edges(ftype, shape):
    if shape.endswith("singly terminated"):
        return  # tested separately (open/short far end)
    for sub in subtypes(ftype, shape):
        if sub == ZIGZAG or sub == SYMMETRIC:
            continue  # tested separately (even orders only / arithmetic centre)
        r = design(FilterSpec(ftype=ftype, shape=shape, subtype=sub, order=5, fc=100e6, bw=10e6,
                              ripple_db=0.1, amin_db=50))
        lo, hi = r.info["edges"]
        assert math.sqrt(lo * hi) == pytest.approx(100e6, rel=1e-12)
        assert hi - lo == pytest.approx(10e6, rel=1e-12)
        assert s21(r, [lo, hi]) == pytest.approx([-r.info["aa_db"]] * 2, abs=2e-4), sub
        if shape == "Elliptic":
            assert s21(r, list(r.info["fs"])) == pytest.approx([-50, -50], abs=1e-4)
        assert r.ladder.r_load == pytest.approx(50, rel=1e-6)


def test_coupled_equal_resonators_and_limit():
    r = design(FilterSpec(ftype="Bandpass", shape="Chebyshev", subtype="Shunt resonators, capacitive coupling", order=5,
                          fc=100e6, bw=5e6, ripple_db=0.1, resonator=20e-9))
    Ls = [b.L for b in r.ladder.branches if b.kind == "LC_par"]
    assert len(Ls) == 5 and all(L == pytest.approx(20e-9) for L in Ls)
    # below the limit: clamped into the usable range, with a warning
    low = design(FilterSpec(ftype="Bandpass", shape="Chebyshev", subtype="Shunt resonators, capacitive coupling", order=5,
                            fc=100e6, bw=5e6, ripple_db=0.1, resonator=1e-9))
    lo, hi = low.info["resonator_range"]
    assert low.info["resonator"] == pytest.approx(lo) and lo > 1e-9
    assert any("limited" in w for w in low.warnings)
    assert s21(low, list(low.info["edges"])) == pytest.approx([-0.1, -0.1], abs=2e-3)


def test_zigzag_min_inductor():
    full = design(FilterSpec(ftype="Bandpass", shape="Elliptic", subtype=SHUNT_FIRST, order=4,
                             fc=100e6, bw=20e6, ripple_db=0.1, amin_db=50))
    zz = design(FilterSpec(ftype="Bandpass", shape="Elliptic", subtype=ZIGZAG, order=4,
                           fc=100e6, bw=20e6, ripple_db=0.1, amin_db=50))
    assert count(zz)[0] == 4 and count(full)[0] == 5          # zig-zag saves an inductor
    f = list(zz.info["edges"]) + list(zz.info["fs"])
    assert s21(zz, f) == pytest.approx([-0.1, -0.1, -50, -50], abs=1e-4)
    with pytest.raises(ValueError):
        design(FilterSpec(ftype="Bandpass", shape="Elliptic", subtype=ZIGZAG, order=5))


def test_standard_values_improve_on_rounding():
    from vektorfil.filters.stdvalues import SERIES, nearest
    r = design(FilterSpec(shape="Chebyshev", order=7, fc=100e6, ripple_db=0.1, std_series="5% (E24)"))
    vals = [v for b in r.ladder.branches for v in (b.L, b.C) if v is not None]
    assert all(abs(nearest(v, SERIES["5% (E24)"]) / v - 1) < 1e-9 for v in vals)
    assert r.info["std"]["pass_dev_db"] < 0.15
    plain = design(FilterSpec(shape="Chebyshev", order=7, fc=100e6, ripple_db=0.1,
                              std_series="5% (E24)", std_retune=False))
    assert r.info["std"]["pass_dev_db"] < plain.info["std"]["pass_dev_db"]


# ----------------------------------------------------------------- symmetric, S2P
def test_symmetric_bandpass_is_more_symmetric():
    for n, frac in ((3, 0.3), (4, 0.3), (5, 0.3)):
        f0, bw = 100e6, frac * 100e6
        sym = design(FilterSpec(ftype="Bandpass", shape="Chebyshev", subtype=SYMMETRIC,
                                order=n, fc=f0, bw=bw, ripple_db=0.1))
        conv = design(FilterSpec(ftype="Bandpass", shape="Chebyshev", subtype=SERIES_FIRST,
                                 order=n, fc=f0, bw=bw, ripple_db=0.1))
        assert sym.info["edges"] == pytest.approx((f0 - bw / 2, f0 + bw / 2))
        assert s21(sym, list(sym.info["edges"])) == pytest.approx([-0.1, -0.1], abs=0.06)
        pb = s21(sym, np.linspace(f0 - bw / 2, f0 + bw / 2, 300))
        assert pb.min() > -0.16
        a = s21(sym, [f0 - bw, f0 + bw])
        c = s21(conv, [f0 - bw, f0 + bw])
        assert abs(a[0] - a[1]) < 0.35 * abs(c[0] - c[1])     # much better balanced skirts


def test_symmetric_topology():
    from vektorfil.synth.symmetric import topology
    assert [topology(n) for n in (2, 3, 4, 5)] == ["SCS", "SCSCS", "SCSCSP", "SCSCSPS"]
    r = design(FilterSpec(ftype="Bandpass", shape="Chebyshev", subtype=SYMMETRIC,
                          order=3, fc=880e6, bw=50e6, ripple_db=0.5))
    kinds = [(b.pos, b.kind) for b in r.ladder.branches]
    assert kinds == [("series", "LC_ser"), ("shunt", "C"), ("series", "LC_ser"), ("shunt", "C"),
                     ("series", "LC_ser")]
    Ls = [b.L for b in r.ladder.branches if b.L is not None]
    assert max(Ls) / min(Ls) == pytest.approx(1.0)                # equal inductors


def test_touchstone_export():
    r = design(FilterSpec(shape="Chebyshev", order=5, fc=100e6, ripple_db=0.1))
    f = np.linspace(1e6, 300e6, 11)
    text = r.ladder.touchstone(f, 50.0)
    lines = [l for l in text.splitlines() if l and not l.startswith("!")]
    assert lines[0] == "# Hz S DB R 50"
    rows = [list(map(float, l.split())) for l in lines[1:]]
    assert len(rows) == 11 and all(len(x) == 9 for x in rows)
    # S21 column equals the design response (50 ohm terminations) and S12 = S21
    ref = s21(r, f)
    assert [x[3] for x in rows] == pytest.approx(list(ref), abs=1e-5)
    assert [x[5] for x in rows] == pytest.approx([x[3] for x in rows], abs=1e-9)


# ----------------------------------------------------------------- TEM, singly terminated
def test_top_c_tem_resonator():
    for theta in (30, 45, 70):
        r = design(FilterSpec(ftype="Bandpass", shape="Chebyshev", subtype="Coaxial resonators, capacitive coupling",
                              order=5, fc=900e6, bw=40e6, ripple_db=0.1, tem_theta_deg=theta))
        lines = [b for b in r.ladder.branches if b.kind == "TEM"]
        assert len(lines) == 5
        assert len({round(b.z0, 9) for b in lines}) == 1                 # identical lines
        assert all(b.theta == pytest.approx(theta) and b.f_ref == 900e6 for b in lines)
        assert s21(r, list(r.info["edges"])) == pytest.approx([-0.1, -0.1], abs=2e-3)
        lo, hi = r.info["resonator_range"]
        assert lo < r.info["resonator"] < hi


@pytest.mark.parametrize("shape", ["Butterworth, singly terminated", "Chebyshev, singly terminated"])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_singly_terminated(shape, n):
    for sub in (SHUNT_FIRST, SERIES_FIRST):
        r = design(FilterSpec(shape=shape, subtype=sub, order=n, fc=100e6, ripple_db=0.1))
        lad = r.ladder
        assert lad.single_terminated in ("open", "short")
        # an open end faces a shunt C, a shorted end a series L
        last = lad.branches[-1]
        assert (last.pos == "shunt") == (lad.single_terminated == "open")
        # Butterworth: -3.01 dB at fc.  Chebyshev: |H| = 1 at DC; odd orders ripple between
        # 0 and -0.1 dB, even orders between 0 and +0.1 dB (voltage gain into the open end)
        cheb_even = shape.startswith("Cheb") and n % 2 == 0
        edge = -3.0103 if shape.startswith("Butter") else (0.0 if cheb_even else -0.1)
        assert s21(r, [100e6])[0] == pytest.approx(edge, abs=1e-4)
        assert s21(r, [1e6])[0] == pytest.approx(0.0, abs=1e-3)
        pb = s21(r, np.linspace(1e6, 100e6, 400))
        if shape.startswith("Cheb"):
            assert (pb.max() == pytest.approx(0.1, abs=1e-3)) if cheb_even else (pb.min() == pytest.approx(-0.1, abs=1e-3))


def test_singly_terminated_textbook_values():
    from vektorfil.synth import approx
    from vektorfil.synth.ladder import synthesize_single
    res = synthesize_single(approx.butterworth_with_zeros(4), first="series")
    g = [float(b.L if b.L is not None else b.C) for b in res.branches]
    # classic singly terminated Butterworth n=4 (listed from the unterminated end: 1.5307 ...)
    assert g[::-1] == pytest.approx([1.5307, 1.5772, 1.0824, 0.3827], abs=1e-4)


# ----------------------------------------------------------------- diplexer
def test_diplexer_butterworth_is_exact():
    from vektorfil.filters.diplexer import DiplexSpec, design_diplexer
    for n in (3, 4, 5):
        d = design_diplexer(DiplexSpec(lp_order=n, hp_order=n, lp_fc=100e6, hp_fc=100e6))
        f = np.geomspace(1e6, 1e9, 200)
        sp = d.sparams(f)
        assert np.max(np.abs(sp["S11"])) < 1e-8                         # perfectly matched
        power = np.abs(sp["S11"]) ** 2 + np.abs(sp["S21"]) ** 2 + np.abs(sp["S31"]) ** 2
        assert power == pytest.approx(np.ones_like(f), abs=1e-9)        # lossless
        x = d.sparams([100e6])
        assert 20 * np.log10(abs(x["S21"][0])) == pytest.approx(-3.0103, abs=1e-3)
        assert 20 * np.log10(abs(x["S31"][0])) == pytest.approx(-3.0103, abs=1e-3)
        # common-port ends: series L (lowpass) and series C (highpass)
        assert (d.lp.branches[0].pos, d.lp.branches[0].kind) == ("series", "L")
        assert (d.hp.branches[0].pos, d.hp.branches[0].kind) == ("series", "C")


def test_diplexer_chebyshev_matched_and_s3p():
    from vektorfil.filters.diplexer import DiplexSpec, design_diplexer, touchstone_s3p
    d = design_diplexer(DiplexSpec(lp_shape="Chebyshev", hp_shape="Chebyshev", lp_order=5, hp_order=5,
                                   lp_ripple=0.1, hp_ripple=0.1))
    assert d.info["worst_return_loss_db"] > 25
    text = touchstone_s3p(d, np.array([50e6, 150e6]))
    data = [l for l in text.splitlines() if l and not l.startswith(("!", "#"))]
    assert len(data) == 6 and "# Hz S DB R 50" in text


def test_elliptic_singly_terminated_and_diplexer():
    from vektorfil.filters.diplexer import DiplexSpec, design_diplexer
    r = design(FilterSpec(shape="Elliptic, singly terminated", subtype=SERIES_FIRST, order=5,
                          fc=100e6, ripple_db=0.1, amin_db=50, aa_db=3.0103))
    assert s21(r, [100e6])[0] == pytest.approx(-3.0103, abs=1e-4)
    assert s21(r, [r.info["fs"]])[0] == pytest.approx(-50, abs=1e-4)
    d = design_diplexer(DiplexSpec(lp_shape="Elliptic", hp_shape="Elliptic", lp_order=7, hp_order=7,
                                   lp_amin=60, hp_amin=60))
    assert d.info["worst_return_loss_db"] > 25
    assert (d.lp.branches[0].kind, d.hp.branches[0].kind) == ("L", "C")
    sp = d.sparams(np.array([120e6, 80e6]))
    assert 20 * np.log10(abs(sp["S21"][0])) < -40 and 20 * np.log10(abs(sp["S31"][1])) < -40


def test_tumbular_mirrors_tubular():
    # Tumbular = tubular with every L and C swapped about fc: same band edges, the
    # response mirrored geometrically (S21 at f equals the tubular's S21 at fc^2/f)
    kw = dict(ftype="Bandpass", shape="Chebyshev", order=3, fc=100e6, bw=20e6, ripple_db=0.1)
    tub = design(FilterSpec(subtype="Tubular", **kw))
    tum = design(FilterSpec(subtype="Tumbular", **kw))
    br = tum.ladder.branches
    assert (br[0].pos, br[0].kind) == ("shunt", "L") and (br[-1].pos, br[-1].kind) == ("shunt", "L")
    assert sum(b.kind == "C" for b in br) == 3                     # three equal series resonator C
    lo, hi = tum.info["edges_actual"]
    assert lo * hi == pytest.approx(1e16, rel=1e-6)
    assert s21(tum, [lo, hi]) == pytest.approx([-0.1, -0.1], abs=1e-4)
    f = np.array([50e6, 70e6, 130e6, 160e6])
    assert s21(tum, f) == pytest.approx(s21(tub, 1e16 / f), abs=1e-3)
    assert s21(tum, 70e6)[0] < s21(tum, 1e16 / 70e6)[0] - 10       # steeper lower skirt
    # resonator capacitor typed in by the user is used as given
    c = tum.info["resonator"] * 1.1
    tum2 = design(FilterSpec(subtype="Tumbular", resonator=c, **kw))
    assert tum2.info["resonator"] == pytest.approx(c)
    lo_r, hi_r = tum2.info["resonator_range"]
    assert lo_r < c < hi_r


@pytest.mark.parametrize("sub", ["Tubular", "Tumbular"])
def test_end_coils_extend_the_resonator_range(sub):
    # a capacitor-only end coupling can only step the termination down, so the tubular
    # inductor (tumbular capacitor) has a hard limit; end coils (tumbular: end capacitors)
    # carry the design past it with exact band edges and the same component count elsewhere
    kw = dict(ftype="Bandpass", shape="Chebyshev", subtype=sub, order=5, fc=100e6, bw=20e6,
              ripple_db=0.1)
    plain = design(FilterSpec(**kw))
    coils = design(FilterSpec(end_coils=True, **kw))
    p_lo, p_hi = plain.info["resonator_range"]
    c_lo, c_hi = coils.info["resonator_range"]
    tub = sub == "Tubular"
    assert (c_hi > 5 * p_hi) if tub else (c_lo < p_lo / 5)
    # below the limit the option changes nothing
    assert coils.ladder.table() == plain.ladder.table()
    # past it: a series element at each port, exact edges, ripple still close
    v = 2 * p_hi if tub else p_lo / 2
    r = design(FilterSpec(end_coils=True, resonator=v, **kw))
    assert r.info["resonator"] == pytest.approx(v)
    br = r.ladder.branches
    end = "L" if tub else "C"
    assert (br[0].pos, br[0].kind) == ("series", end) and (br[-1].pos, br[-1].kind) == ("series", end)
    lo, hi = r.info["edges_actual"]
    assert s21(r, [lo, hi]) == pytest.approx([-0.1, -0.1], abs=1e-4)
    assert r.info["passband_loss_db"] < 0.2
    # without the option the same value is pulled back into the plain range
    assert design(FilterSpec(resonator=v, **kw)).info["resonator"] != pytest.approx(v, rel=1e-3)


def test_old_circuit_names_still_accepted():
    # version 0.2 names map onto the current ones; "Minimum L" was series-first for a highpass
    from vektorfil.filters import canonical_subtype
    assert canonical_subtype("Lowpass", "Minimum L") == SHUNT_FIRST
    assert canonical_subtype("Highpass", "Minimum L") == SERIES_FIRST
    assert canonical_subtype("Highpass", "Minimum C") == SHUNT_FIRST
    assert canonical_subtype("Bandpass", "Top C coupled") == "Shunt resonators, capacitive coupling"
    for ftype in ("Lowpass", "Highpass"):
        old = design(FilterSpec(ftype=ftype, shape="Elliptic", subtype="Minimum L", order=5))
        new = design(FilterSpec(ftype=ftype, shape="Elliptic", order=5,
                                subtype=SERIES_FIRST if ftype == "Highpass" else SHUNT_FIRST))
        assert old.ladder.table() == new.ladder.table()
        assert sum(b.L is not None for b in old.ladder.branches) < sum(b.C is not None for b in old.ladder.branches)
