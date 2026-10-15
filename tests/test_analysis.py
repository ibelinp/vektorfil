"""Tests of the analysis tools (coil, Monte Carlo, time domain), the stepped stopband
mask and design files."""
import math

import numpy as np
import pytest

from vektorfil.analysis import coil, montecarlo, timedomain
from vektorfil.filters import FilterSpec, design
from vektorfil.filters.design import format_mask, parse_mask


# ----------------------------------------------------------------- coil

def test_coil_matches_classic_wheeler():
    # Wheeler 1928: L(uH) = d^2 n^2 / (18 d + 40 l), inches; 1" diameter, 1" long, 10 turns
    inch = 0.0254
    assert coil.inductance(10, inch / 2, inch) == pytest.approx(1.724e-6, rel=5e-3)


def test_coil_long_solenoid_limit():
    a, b, n = 0.005, 1.0, 1000
    assert coil.inductance(n, a, b) == pytest.approx(coil.MU0 * n * n * math.pi * a * a / b, rel=1e-2)


def test_coil_design_hits_target():
    d = coil.design_coil(100e-9, 6e-3, coil.awg_mm(20) * 1e-3)
    radius = d.mean_diameter / 2
    assert coil.inductance(d.turns, radius, d.length) == pytest.approx(100e-9, rel=1e-9)
    assert d.turns_whole == round(d.turns)
    assert d.pitch == pytest.approx(coil.awg_mm(20) * 1e-3)
    assert coil.awg_mm(36) == pytest.approx(0.127)


# ----------------------------------------------------------------- Monte Carlo

def test_monte_carlo_zero_tolerance_is_nominal():
    r = design(FilterSpec(shape="Chebyshev", order=5, fc=100e6))
    f = np.linspace(1e6, 300e6, 301)
    mc = montecarlo.run(r.ladder, f, runs=5, tol_l=0, tol_c=0, seed=1)
    assert np.allclose(mc.s21_min, mc.s21_nom) and np.allclose(mc.s21_max, mc.s21_nom)


def test_monte_carlo_envelope_and_repeatability():
    r = design(FilterSpec(shape="Chebyshev", order=5, fc=100e6))
    f = np.linspace(1e6, 300e6, 301)
    a = montecarlo.run(r.ladder, f, runs=50, tol_l=0.05, tol_c=0.05, seed=7)
    b = montecarlo.run(r.ladder, f, runs=50, tol_l=0.05, tol_c=0.05, seed=7)
    assert np.array_equal(a.s21_min, b.s21_min)
    assert np.all(a.s21_min <= a.s21_max)
    # a lossless filter cannot beat its 0 dB points, so the runs fall below the nominal
    # there; deep in the stopband the nominal lies inside the spread
    stop = f > 200e6
    assert np.all(a.s21_min[stop] <= a.s21_nom[stop]) and np.all(a.s21_max[stop] >= a.s21_nom[stop])
    st = a.passband_stats(f <= 100e6)
    assert st["s21_worst"] < st["s21_nom"] and st["s11_worst"] > st["s11_nom"]
    g = montecarlo.run(r.ladder, f, runs=50, tol_l=0.05, tol_c=0.05, dist="gaussian", seed=7)
    # 3-sigma gaussian spreads less than uniform at the same tolerance
    assert np.mean(g.s21_max - g.s21_min) < np.mean(a.s21_max - a.s21_min)


# ----------------------------------------------------------------- time domain

def test_first_order_step_response():
    # 1st-order Butterworth lowpass: step = 1 - exp(-2 pi fc t)
    fc = 10e6
    r = design(FilterSpec(shape="Butterworth", order=1, fc=fc))
    tr = timedomain.time_response(r.ladder, 400 * fc)
    t = np.array([0.5, 1, 2, 5]) / (2 * math.pi * fc)
    got = np.interp(t, tr.t, tr.step)
    assert got == pytest.approx(1 - np.exp(-2 * math.pi * fc * t), abs=5e-3)


def test_lowpass_step_settles_at_one():
    r = design(FilterSpec(shape="Chebyshev", order=5, fc=100e6))
    f_max, t_end = timedomain.auto_window(r.spec, r.info)
    tr = timedomain.time_response(r.ladder, f_max)
    late = tr.step[(tr.t > 0.8 * t_end) & (tr.t < t_end)]
    assert late == pytest.approx(1.0, abs=5e-3)


# ----------------------------------------------------------------- stepped mask

def test_mask_parse_round_trip():
    m = parse_mask("125-200:40, 200-:70")
    assert m == ((125e6, 200e6, 40.0), (200e6, math.inf, 70.0))
    assert parse_mask(format_mask(m)) == m
    with pytest.raises(ValueError):
        parse_mask("125:40")


@pytest.mark.parametrize("ftype,mask", [("Lowpass", "125-200:40, 200-:50"), ("Highpass", "0-60:60, 60-80:40")])
def test_mask_design_meets_mask_by_simulation(ftype, mask):
    mk = parse_mask(mask)
    r = design(FilterSpec(ftype=ftype, shape="Elliptic", order=7, fc=100e6, elliptic_spec="Mask", mask=mk))
    worst = []
    for a, b, need in mk:
        f = np.linspace(max(a, 1e5), b if math.isfinite(b) else 1e9, 4000)
        worst.append((-20 * np.log10(np.abs(r.ladder.sparams(f)[1]))).min() - need)
    assert min(worst) == pytest.approx(r.info["mask_margin_db"], abs=0.05)
    assert r.info["mask_margin_db"] > 0


def test_mask_reaching_into_passband_is_refused():
    with pytest.raises(ValueError):
        design(FilterSpec(shape="Elliptic", order=7, fc=100e6, elliptic_spec="Mask", mask=parse_mask("90-200:40")))


# ----------------------------------------------------------------- design files

def test_design_file_round_trip(tmp_path):
    from vektorfil.filters import project
    spec = FilterSpec(ftype="Bandpass", shape="Elliptic", order=6, fc=70e6, bw=12e6, elliptic_spec="Mask",
                      mask=parse_mask("0-50:50, 95-:45"), std_series="5% (E24)", q_l=80.0)
    path = tmp_path / "x.vfil"
    project.save(path, spec, {"markers": [1e8]})
    back, view = project.load(path)
    assert back == spec and view == {"markers": [1e8]}
    path.write_text('{"format": "something else"}', encoding="utf-8")
    with pytest.raises(ValueError):
        project.load(path)
