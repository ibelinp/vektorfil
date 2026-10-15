"""Lumped L-C filter design: an interactive filter specification turned into a ladder.

    spec = FilterSpec(ftype="Lowpass", shape="Chebyshev", order=7, fc=70e6, ripple_db=0.25)
    result = design(spec)
    result.ladder.response(f), result.ladder.table(), result.info ...

For bandpass / bandstop, fc is the (geometric) centre frequency and bw the bandwidth
between the cutoff points (the Aa points; the ripple points for Chebyshev/elliptic).
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field, replace

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from vektorfil.synth import approx
from vektorfil.synth import bandpass as bpsyn
from vektorfil.synth.bandxform import transform
from vektorfil.synth.symmetric import design_symmetric
from vektorfil.synth.coupled import (default_resonator, design_coupled, resonator_limit,
                                    resonator_range, to_tubular)
from vektorfil.synth.ladder import synthesize
from vektorfil.synth.network import Ladder

TYPES = ("Lowpass", "Highpass", "Bandpass", "Bandstop")
SINGLE = ", singly terminated"
SHAPES = ("Butterworth", "Chebyshev", "Bessel", "Elliptic",
          "Butterworth" + SINGLE, "Chebyshev" + SINGLE, "Elliptic" + SINGLE)
BUTTER_3DB = 10 * math.log10(2)
# circuit (realization) names, in the words of the filter literature
SHUNT_FIRST = "Shunt element first"
SERIES_FIRST = "Series element first"
COUPLED = {"Shunt resonators, capacitive coupling": "top_c",
           "Shunt resonators, inductive coupling": "top_l",
           "Series resonators, shunt-C coupling": "shunt_c",
           "Tubular": "tubular", "Tumbular": "tumbular",
           "Coaxial resonators, capacitive coupling": "top_c_tem"}
ZIGZAG = "Zig-zag, fewest inductors (Saal–Ulbrich)"
SYMMETRIC = "Quasi-symmetric (Carassa)"

# names used by version 0.2 and earlier, still accepted
_OLD_NAMES = {"Top C coupled": "Shunt resonators, capacitive coupling",
              "Top L coupled": "Shunt resonators, inductive coupling",
              "Shunt C coupled": "Series resonators, shunt-C coupling",
              "Top C TEM resonator": "Coaxial resonators, capacitive coupling",
              "Symmetry preserving": SYMMETRIC, "Zig-zag (min L)": ZIGZAG}


def canonical_subtype(ftype: str, subtype: str) -> str:
    """Current name of a circuit, translating the old names.  The old 'Minimum L' meant
    the fewer-inductor ladder, which starts with a shunt element for lowpass and band
    filters but with a series element for a highpass."""
    if subtype in ("Minimum L", "Minimum C"):
        return SERIES_FIRST if (subtype == "Minimum L") == (ftype == "Highpass") else SHUNT_FIRST
    return _OLD_NAMES.get(subtype, subtype)


def subtypes(ftype: str, shape: str):
    """Circuit choices offered for a response/approximation combination."""
    base = [SHUNT_FIRST, SERIES_FIRST]
    if shape.endswith(SINGLE):
        return base
    if ftype == "Bandpass" and shape != "Elliptic":
        return base + list(COUPLED) + [SYMMETRIC]
    if ftype == "Bandpass" and shape == "Elliptic":
        return base + [ZIGZAG]
    return base


def _sub_title(subtype: str) -> str:
    """Topology name as used inside a title: 'top C coupled', 'tumbular'."""
    return subtype[0].lower() + subtype[1:]


def resonator_unit(subtype: str):
    """'L' or 'C' for coupled types (the element shared by all resonators), else None."""
    k = COUPLED.get(_OLD_NAMES.get(subtype, subtype))
    if k is None:
        return None
    return {"top_l": "C", "tumbular": "C", "top_c_tem": "Z"}.get(k, "L")   # 'Z': TEM line impedance


@dataclass
class FilterSpec:
    """Everything the user specifies (Topology + Settings + Defaults tabs).
    Frequencies in Hz, resistances in ohm, L in H, C in F.
    """
    ftype: str = "Lowpass"
    shape: str = "Chebyshev"
    subtype: str = SHUNT_FIRST
    order: int = 5
    fc: float = 100e6               # cutoff (LP/HP) or centre frequency (BP/BS), Hz
    bw: float = 10e6                # bandwidth (BP/BS), Hz
    r_source: float = 50.0
    ripple_db: float = 0.1          # passband ripple (Chebyshev, Elliptic)
    aa_db: float | None = None      # attenuation at the cutoff(s); None = default
    elliptic_spec: str = "Amin"     # 'Amin', 'Fs' or 'Mask'
    mask: tuple = ()                # elliptic 'Mask': ((f_start, f_stop, A_dB), ...) in Hz;
                                    # f_stop may be inf (lowpass side) or 0 (highpass side)
    amin_db: float = 60.0           # elliptic minimum stopband attenuation
    fs: float = 150e6               # elliptic stopband frequency (Hz) when elliptic_spec == 'Fs'
    even_type: str = "C"            # even-order elliptic: 'C' equal terminations, 'B' unequal
    resonator: float | None = None  # coupled types: shared resonator L (H) or C (F); None = auto
    edge_correct: bool = True       # coupled types: re-tune so the band edges are exact
    end_coils: bool = False         # tubular/tumbular: series end element, extends the range
    tem_theta_deg: float = 45.0     # Top C TEM resonator: line length at the centre frequency
    q_l: float | None = None
    q_c: float | None = None
    std_series: str = "Ideal"      # 'Ideal', '5% (E24)', '2% (E48)', '1% (E96)'
    std_retune: bool = True         # re-tune the remaining elements while snapping


@dataclass
class DesignResult:
    """Output of design(): the finished ladder plus everything the GUI shows.
    info keys: r_load, aa_db, edges, f3db, fs, amin_db, zeros, resonator, std, sum_g.
    """
    spec: FilterSpec
    ladder: Ladder
    g_values: list                  # normalised prototype element values (lowpass, 1 ohm, 1 rad/s)
    g_labels: list
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    seconds: float = 0.0
    ideal_ladder: Ladder | None = None   # set when standard values were applied


def default_aa(spec: FilterSpec) -> float:
    if spec.shape in ("Butterworth", "Bessel"):
        return BUTTER_3DB
    return spec.ripple_db


# --------------------------------------------------------------------------- frequency maps

def to_proto(spec: FilterSpec, f):
    """Real frequency (Hz) -> normalised lowpass prototype frequency (>0)."""
    t, fc, bw = spec.ftype, spec.fc, spec.bw
    if t == "Lowpass":
        return f / fc
    if t == "Highpass":
        return fc / f
    if t == "Bandpass":
        return abs(f * f - fc * fc) / (f * bw)
    return f * bw / abs(fc * fc - f * f)


def from_proto(spec: FilterSpec, w, bw=None):
    """Normalised prototype frequency -> real frequencies (Hz), ascending tuple."""
    t, fc = spec.ftype, spec.fc
    bw = spec.bw if bw is None else bw
    if t == "Lowpass":
        return (fc * w,)
    if t == "Highpass":
        return (fc / w,)
    if t == "Bandpass":
        r = math.sqrt((w * bw) ** 2 + 4 * fc * fc)
        return ((r - w * bw) / 2, (r + w * bw) / 2)
    r = math.sqrt((bw / w) ** 2 + 4 * fc * fc)
    return ((r - bw / w) / 2, (r + bw / w) / 2)


# --------------------------------------------------------------------------- prototype

def parse_mask(text: str, unit: float = 1e6) -> tuple:
    """'125-200:40, 200-:70' -> ((125e6, 200e6, 40), (200e6, inf, 70)).

    Each segment is start-stop:attenuation, frequencies in `unit` (MHz) and attenuation
    in dB; an empty stop means "and beyond" (for a highpass write e.g. '0-60:50')."""
    segs = []
    for part in re.split(r"[;,\n]+", text):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"([\d.eE+]+)\s*-\s*([\d.eE+]*)\s*:\s*([\d.eE+]+)\s*(?:dB)?", part)
        if not m:
            raise ValueError(f"cannot read mask segment '{part}' (use start-stop:dB)")
        f1 = float(m.group(1)) * unit
        f2 = float(m.group(2)) * unit if m.group(2) else math.inf
        segs.append((f1, f2, float(m.group(3))))
    if not segs:
        raise ValueError("enter at least one mask segment, e.g. 125-200:40")
    return tuple(segs)


def format_mask(mask) -> str:
    """Inverse of parse_mask (MHz)."""
    return ", ".join(f"{a / 1e6:g}-{'' if math.isinf(b) else f'{b / 1e6:g}'}:{d:g}" for a, b, d in mask)


def _mask_prototype(spec: FilterSpec):
    """Elliptic-type prototype whose zeros are placed against a stepped stopband mask
    (PLACER): the order is the user's, the zeros give every mask step the same margin."""
    from vektorfil.synth.placer import Mask, place
    if not spec.mask:
        raise ValueError("enter a stopband mask, e.g. 125-200:40, 200-:70")

    def w_of(f):
        if f == 0:
            return math.inf if spec.ftype in ("Highpass", "Bandpass") else 0.0
        if math.isinf(f):
            return 0.0 if spec.ftype in ("Highpass", "Bandstop") else math.inf
        return float(to_proto(spec, f))

    segs = []
    for f1, f2, a in spec.mask:
        if spec.ftype == "Bandpass" and min(f1, f2) < spec.fc < max(f1, f2):
            raise ValueError("a mask segment may not span the passband")
        w1, w2 = sorted((w_of(f1), w_of(f2)))
        if w1 <= 1.0:
            raise ValueError("the mask reaches into the passband")
        segs.append((w1, w2, a))
    n = int(spec.order)
    odd = n % 2 == 1
    pl = place(order=n, ripple_db=spec.ripple_db, mask=Mask(segs), n_inf=1 if odd else 2,
               zs=None if odd else spec.even_type)
    proto = pl.prototype()
    proto.meta["placer"] = pl
    proto.meta["mask_margin"] = pl.margin_db
    return proto


def _prototype(spec: FilterSpec):
    """The normalised lowpass prototype (1 ohm, passband edge 1 rad/s) for the spec shape."""
    n = int(spec.order)
    if spec.shape == "Butterworth":
        return approx.butterworth_with_zeros(n)
    if spec.shape == "Chebyshev":
        return approx.chebyshev_with_zeros(n, spec.ripple_db)
    if spec.shape == "Bessel":
        return approx.bessel(n)
    if spec.shape == "Elliptic":
        if n < 3:
            raise ValueError("Elliptic filters need order 3 or more")
        if spec.elliptic_spec == "Mask":
            return _mask_prototype(spec)
        if spec.elliptic_spec == "Fs":
            ws = to_proto(spec, spec.fs)
            if ws <= 1:
                raise ValueError("The stopband frequency must lie in the stopband")
            return approx.elliptic(n, spec.ripple_db, ws=ws, zs=spec.even_type)
        return approx.elliptic(n, spec.ripple_db, amin_db=spec.amin_db, zs=spec.even_type)
    raise ValueError(f"unknown shape {spec.shape}")


def _w_at_atten(proto, target_db):
    """Normalised frequency where the prototype loss equals target_db (passband side)."""
    lo, hi = mp.mpf("1e-4"), mp.mpf(1)
    while proto.atten_db(hi) < target_db:
        hi *= 2
        if hi > 1e6:
            raise ValueError("attenuation at cutoff is never reached")
    for _ in range(100):
        mid = mp.sqrt(lo * hi)
        if proto.atten_db(mid) < target_db:
            lo = mid
        else:
            hi = mid
    return float(mp.sqrt(lo * hi))


def _g_matthaei(res):
    """[g0, g1..gn, g_{n+1}] from a shunt-first all-pole ladder (Matthaei convention)."""
    g = [float(res.r_source)] + [float(b.L if b.L is not None else b.C) for b in res.branches]
    last = res.branches[-1]
    g.append(float(res.r_load) if last.kind == "C" else 1 / float(res.r_load))
    return g


def _edges(lad, f_lo_guess, f_hi_guess, level_db):
    """The band edges: outermost frequencies where |S21| = -level_db (bandpass).

    Chebyshev ripple dips touch the same level inside the band, so each edge is found by
    walking inward from well outside the band and taking the first crossing."""
    def h(f):
        return 20 * np.log10(np.abs(lad.sparams(np.atleast_1d(f))[1])) + level_db

    span = f_hi_guess - f_lo_guess
    centre = math.sqrt(f_lo_guess * f_hi_guess)
    out = []
    for start in (max(f_lo_guess - 1.5 * span, f_lo_guess * 0.05), f_hi_guess + 1.5 * span):
        grid = np.linspace(start, centre, 1500)        # outside -> centre
        v = h(grid)
        inside = np.nonzero(v >= 0)[0]
        if len(inside) == 0 or inside[0] == 0:
            raise ValueError("band edge not found")
        i = inside[0]
        a, b = grid[i - 1], grid[i]
        out.append(brentq(lambda f: float(h(f)[0]), min(a, b), max(a, b), xtol=1e-7 * centre))
    return sorted(out)


# --------------------------------------------------------------------------- main entry

MAX_ORDER = {"Butterworth": 25, "Chebyshev": 25, "Bessel": 15, "Elliptic": 25,
             "Butterworth" + SINGLE: 25, "Chebyshev" + SINGLE: 25, "Elliptic" + SINGLE: 25}


# mpmath keeps its working precision in one global context, so two syntheses must never
# run at the same time (e.g. the filter and the diplexer window, each with its own
# worker thread).  Re-entrant: the diplexer designs its channels through design().
_MP_LOCK = threading.RLock()


def design(spec: FilterSpec) -> DesignResult:
    """Turn a FilterSpec into a finished, denormalised ladder.

    Steps: lowpass prototype -> attenuation-at-cutoff scaling -> ladder synthesis (choosing
    minimum L / minimum C) -> lowpass/highpass scaling, band transform, coupled-resonator
    or direct-bandpass realization -> optional standard-value snapping.

    Working precision grows with the order (each extraction step costs digits).  When a
    specification is not realizable -- elements would turn negative, as happens
    when the order is pushed too far -- a ValueError with a plain explanation is raised.
    """
    spec = replace(spec, subtype=canonical_subtype(spec.ftype, spec.subtype))
    if spec.order > MAX_ORDER.get(spec.shape, 25):
        raise ValueError(f"{spec.shape} filters are limited to order {MAX_ORDER[spec.shape]}")
    from vektorfil.synth.ladder import SynthesisError
    with _MP_LOCK, mp.workdps(max(50, 60 + 3 * int(spec.order))):
        try:
            return _design(spec)
        except (SynthesisError, ArithmeticError, ZeroDivisionError) as e:
            raise ValueError(
                f"Not realizable at order {spec.order}: element values would turn negative "
                f"or the numbers break down ({str(e)[:60]}). Lower the order, or relax the "
                f"specification (e.g. a lower Amin).") from e


def _design(spec: FilterSpec) -> DesignResult:
    """design() without the precision/realizability wrapper."""
    import dataclasses
    t0 = time.perf_counter()
    user_spec = spec
    # singly terminated shapes: the base shape's response, a different ladder synthesis
    single = spec.shape.endswith(SINGLE)
    if single:
        if spec.subtype not in (SHUNT_FIRST, SERIES_FIRST):
            raise ValueError("Singly terminated filters are plain ladders: shunt or series element first")
        spec = dataclasses.replace(spec, shape=spec.shape[:-len(SINGLE)])
    if spec.order < 1:
        raise ValueError("order must be at least 1")
    if spec.fc <= 0 or spec.r_source <= 0:
        raise ValueError("frequency and impedance must be positive")
    band = spec.ftype in ("Bandpass", "Bandstop")
    if band and not (0 < spec.bw):
        raise ValueError("bandwidth must be positive")
    proto = _prototype(spec)
    warnings = []

    # cutoff definition: shift the prototype so that the band edge is the Aa point
    aa = default_aa(spec) if spec.aa_db is None else spec.aa_db
    w_scale = 1.0
    if spec.shape in ("Butterworth", "Chebyshev", "Bessel"):
        if spec.shape == "Chebyshev" and aa < spec.ripple_db - 1e-12:
            warnings.append("Attenuation at cutoff below the ripple; using the ripple value")
            aa = spec.ripple_db
        native = BUTTER_3DB if spec.shape in ("Butterworth", "Bessel") else spec.ripple_db
        if abs(aa - native) > 1e-12:
            w_scale = _w_at_atten(proto, aa)
    elif (spec.shape == "Elliptic" and spec.aa_db is not None and spec.elliptic_spec == "Amin"
          and spec.aa_db > spec.ripple_db + 1e-12):
        # an elliptic filter defined by its Aa point (e.g. 3 dB, as diplexer channels are)
        aa = spec.aa_db
        w_scale = _w_at_atten(proto, aa)
    else:
        aa = spec.ripple_db

    coupled = COUPLED.get(spec.subtype) if spec.ftype == "Bandpass" else None
    if coupled and spec.shape == "Elliptic":
        raise ValueError("Coupled-resonator types are all-pole: choose another shape")

    # shunt or series element first: every transform keeps the position of the first
    # element.  In prototype terms shunt-first ladders save inductors (the prototype's
    # shunt elements are capacitors), so that is the element count to minimise.
    first = "series" if spec.subtype == SERIES_FIRST else "shunt"
    if coupled:
        first = "shunt"
    if spec.subtype == SYMMETRIC and spec.ftype == "Bandpass":
        first = "series"  # the symmetric topology starts from S P S ...
    scarce = "L" if first == "shunt" else "C"
    if band:
        scarce = None  # band transforms produce equal L and C counts per resonator
    if single:
        from vektorfil.synth.ladder import synthesize_single
        res = synthesize_single(proto, first=first)
    else:
        res = synthesize(proto, first=first, minimize=scarce)

    title = f"{user_spec.shape} {spec.ftype.lower()}, order {spec.order}" + \
            (f", {_sub_title(spec.subtype)}" if spec.subtype not in (SHUNT_FIRST, SERIES_FIRST) else "")
    info = {}
    if spec.ftype in ("Lowpass", "Highpass"):
        fc_proto = spec.fc / w_scale if spec.ftype == "Lowpass" else spec.fc * w_scale
        lad = Ladder.from_prototype(res, r0=spec.r_source, fc=fc_proto)
        lad.notes["fc"] = fc_proto
        if spec.ftype == "Highpass":
            lad = lad.to_highpass()
    elif coupled:
        bw_p = spec.bw / w_scale
        g = _g_matthaei(res)
        # Tumbular: the tubular with every L and C swapped about the centre frequency
        # (f -> fc^2/f).  The passband and its edges stay put, the skirts trade places.
        tumbular = coupled == "tumbular"
        tubular = coupled in ("tubular", "tumbular")
        coupled = "shunt_c" if tubular else coupled  # tubular = exact tee-pi form of shunt-C
        r_l = spec.r_source  # inverter coupling: both terminations equal
        coils = bool(spec.end_coils and tubular)
        res_val = spec.resonator
        # the tumbular's resonator capacitor C is the tubular's inductor 1/(w0^2 C)
        flip = (lambda v: 1 / ((2 * math.pi * spec.fc) ** 2 * v)) if tumbular else (lambda v: v)
        if res_val is not None:
            res_val = flip(res_val)
        lim = resonator_limit(coupled, g, spec.fc, bw_p, spec.r_source, r_l, spec.tem_theta_deg)
        info["resonator_limit"] = lim
        # keep the resonator inside its usable range (2 % margin: the edge correction
        # shifts the design frequencies slightly)
        lo_r, hi_r = resonator_range(coupled, g, spec.fc, bw_p, spec.r_source, r_l, spec.tem_theta_deg,
                                     end_coils=coils)
        lo_r, hi_r = lo_r * 1.02, hi_r / 1.02
        info["resonator_range"] = (lo_r, hi_r)
        if not lo_r < hi_r:
            raise ValueError("No usable resonator value for this specification "
                             "(try a narrower bandwidth or another topology)")
        if res_val is None:
            # automatic: the textbook default, but never outside the usable range
            d = default_resonator(coupled, g, spec.fc, bw_p, spec.r_source, r_l, spec.tem_theta_deg)
            res_val = d if lo_r <= d <= hi_r else math.sqrt(lo_r * hi_r)
        elif not (lo_r <= res_val <= hi_r):
            clamped = min(max(res_val, lo_r), hi_r)
            warnings.append(f"Resonator value limited to the usable range "
                            f"({flip(res_val):.4g} -> {flip(clamped):.4g})")
            res_val = clamped
        # narrow-band formulas are approximate: re-tune centre and bandwidth from the
        # simulated band edges so the Aa points land exactly on the specification
        f0_d, bw_d = spec.fc, bw_p
        target = from_proto(spec, 1.0)  # edges of the requested band at Aa (geometric)
        lad = None
        # without edge correction: one pass of the narrow-band formulas (classic method),
        # so the edges move when the resonator value changes
        for _ in range(6 if spec.edge_correct else 1):
            lad = design_coupled(coupled, g, f0_d, bw_d, spec.r_source, r_l, res_val, title=title,
                                 # the line length is specified at fc; the edge correction
                                 # designs at f0_d, so scale it to keep the same physical line
                                 theta_deg=spec.tem_theta_deg * f0_d / spec.fc,
                                 end_coils=coils)
            res_val = lad.notes["resonator"]
            if not spec.edge_correct:
                break
            try:
                lo, hi = _edges(lad, target[0], target[1], aa)
            except Exception:
                warnings.append("Band edges not found for re-tuning; narrow-band approximation used")
                break
            err_c = math.sqrt(target[0] * target[1]) / math.sqrt(lo * hi)
            err_b = (target[1] - target[0]) / (hi - lo)
            if abs(err_c - 1) < 1e-7 and abs(err_b - 1) < 1e-7:
                break
            f0_d *= err_c
            bw_d *= err_b
        if tubular:
            lad = to_tubular(lad)
        if tumbular:
            lad.notes["fc"] = spec.fc
            lad = lad.to_highpass()
            info["resonator_limit"] = flip(lim)
            info["resonator_range"] = (flip(hi_r), flip(lo_r))
        for b in lad.branches:     # quote TEM line lengths at the requested centre frequency
            if b.kind == "TEM":
                b.theta, b.f_ref = b.theta * spec.fc / b.f_ref, spec.fc
        info["resonator"] = flip(res_val)
        try:  # the edges as built (exact when corrected, approximate otherwise)
            info["edges_actual"] = tuple(_edges(lad, target[0], target[1], aa))
        except ValueError:
            pass
        else:
            # the narrow-band formulas lose accuracy at wide bandwidths and far-out resonator
            # values (end coils): say so when the passband loss grows well past Aa
            lo_e, hi_e = info["edges_actual"]
            fp = np.linspace(lo_e, hi_e, 401)
            worst = -float(np.min(20 * np.log10(np.abs(lad.sparams(fp)[1]) + 1e-300)))
            info["passband_loss_db"] = worst
            if worst > aa + max(0.1, 0.5 * aa):
                warnings.append(f"Passband loss reaches {worst:.2f} dB (specified {aa:.2f} dB): "
                                "the coupled-resonator formulas are stretched here; try a "
                                "resonator value closer to the default or a narrower bandwidth")
        if spec.bw / spec.fc > 0.3:
            warnings.append("Coupled types are accurate below about 30 % bandwidth")
    elif spec.subtype == SYMMETRIC:
        if spec.shape == "Elliptic":
            raise ValueError("Symmetry preserving is an all-pole type: choose another shape")
        # arithmetic band: edges at fc -+ bw/2 at the Aa level; the prototype's Aa point
        # (w_scale) maps to the band edges
        def target_db(om, proto=proto, k=w_scale):
            return float(proto.atten_db(om * k))
        lad = design_symmetric(res, target_db, spec.r_source, spec.fc, spec.bw, title=title)
        frac = spec.bw / spec.fc
        if not 0.1 <= frac <= 0.5:
            warnings.append(f"Bandwidth of {frac:.0%}: the symmetric transform is meant for 10 % to 50 %")
        info["arithmetic"] = True
    elif spec.subtype == ZIGZAG:
        # direct bandpass synthesis, minimum-inductor (Saal & Ulbrich) realization
        if spec.order % 2:
            raise ValueError("The zig-zag elliptic bandpass needs an even order")
        bp = bpsyn.from_lowpass(proto, spec.bw / spec.fc)
        found = bpsyn.enumerate_ladders(bp, policy="min_L", max_results=40, time_limit=8.0)
        if not found:
            raise ValueError("No zig-zag realization found for this specification")
        best = min(found, key=lambda r: (bpsyn.count_inductors(r), abs(math.log(float(r.r_load))),
                                         len(r.branches)))
        lad = bpsyn.to_ladder(best, spec.r_source, spec.fc, title=title)
        if abs(lad.r_load / spec.r_source - 1) > 1e-6:
            warnings.append(f"Zig-zag output resistance is {lad.r_load:.4g} Ω (impedance-transforming topology)")
    else:
        bw_p = spec.bw / w_scale if spec.ftype == "Bandpass" else spec.bw * w_scale
        lad = transform(res, spec.r_source, spec.fc, bw_p, "bp" if spec.ftype == "Bandpass" else "bs",
                        title=title)
    lad.q_l, lad.q_c = spec.q_l, spec.q_c
    lad.title = title

    g_vals, g_labels = [float(res.r_source)], ["g0 (Rs)"]
    for i, b in enumerate(res.branches, 1):
        if b.kind in ("L", "C"):
            g_vals.append(float(b.L if b.L is not None else b.C))
            g_labels.append(f"g{i} ({b.pos} {b.kind})")
        else:
            g_vals += [float(b.L), float(b.C)]
            g_labels += [f"g{i} L ({b.pos} {b.kind})", f"g{i} C ({b.pos} {b.kind})"]
    g_vals.append(float(res.r_load))
    g_labels.append(f"g{len(res.branches) + 1} (RL)")

    info.update(r_load=lad.r_load, aa_db=aa)
    arith = info.get("arithmetic", False)
    if band:
        info["edges"] = (spec.fc - spec.bw / 2, spec.fc + spec.bw / 2) if arith else from_proto(spec, 1.0)
    try:
        w3 = _w_at_atten(proto, BUTTER_3DB) / w_scale
        if arith:
            f3 = (spec.fc - w3 * spec.bw / 2, spec.fc + w3 * spec.bw / 2)
        else:
            f3 = from_proto(spec, w3)
        info["f3db"] = f3 if band else f3[0]
    except ValueError:
        pass
    if arith:  # report the skirt balance one bandwidth away from the centre
        sk = 20 * np.log10(np.abs(lad.sparams(np.array([spec.fc - spec.bw, spec.fc + spec.bw]))[1]))
        info["skirts"] = (spec.fc - spec.bw, float(sk[0]), spec.fc + spec.bw, float(sk[1]))
    if spec.shape == "Elliptic":
        pl = proto.meta.get("placer")
        if pl is not None:
            ws_n = min(w for w, _, _ in pl.extrema)
            info["amin_db"] = min(a for _, a, _ in pl.extrema)
        else:
            ws_n = proto.meta.get("ws") or to_proto(spec, spec.fs)
            info["amin_db"] = float(proto.atten_db(ws_n))
        fs = from_proto(spec, ws_n / w_scale)
        info["fs"] = fs if band else fs[0]
        zs = []
        for z in sorted(float(z) for z in proto.zeros):
            zs += list(from_proto(spec, z / w_scale))
        info["zeros"] = sorted(zs)
        if "mask_margin" in proto.meta:
            info["mask_margin_db"] = float(proto.meta["mask_margin"])
            if info["mask_margin_db"] < -0.01:
                warnings.append(f"The mask is missed by {-info['mask_margin_db']:.2f} dB: raise the order "
                                "or relax the mask")
    info["sum_g"] = sum(g_vals[1:-1])
    # realism check: element reactance far from the impedance level
    f_ref = spec.fc
    wr = 2 * math.pi * f_ref
    odd = []
    for b, (ln, cn) in zip(lad.branches, lad.designators()):
        for v, name, x in ((b.L, ln, (wr * b.L) if b.L else None),
                           (b.C, cn, (1 / (wr * b.C)) if b.C else None)):
            if x is not None and not (1e-4 <= x / spec.r_source <= 1e4):
                odd.append(name)
    if odd:
        warnings.append("Unrealistic element values (" + ", ".join(odd[:6]) +
                        (", ..." if len(odd) > 6 else "") + "): reactance far from the impedance level")
    result = DesignResult(spec, lad, g_vals, g_labels, info, warnings, 0.0)
    if spec.std_series and spec.std_series != "Ideal":
        from vektorfil.filters.stdvalues import snap
        rep = snap(lad, spec.std_series, auto_freqs(result, 600), aa_db=aa,
                   retune=spec.std_retune, stop_db=info.get("amin_db", 80.0))
        rep.ladder.title = lad.title + f" — {spec.std_series} values"
        result.ideal_ladder, result.ladder = lad, rep.ladder
        info["std"] = {"series": spec.std_series, "method": rep.method,
                       "pass_dev_db": rep.max_pass_dev_db, "stop_short_db": rep.worst_stop_loss_db}
    if single:
        info["single"] = lad.single_terminated or (result.ladder.single_terminated)
    result.spec = user_spec
    result.seconds = time.perf_counter() - t0
    return result


def auto_freqs(res: DesignResult, n=1001):
    """A sensible frequency sweep for plotting / evaluating a design."""
    sp = res.spec
    if sp.ftype in ("Bandpass", "Bandstop"):
        span = max(4 * sp.bw, 2.5 * abs(max(res.info.get("fs", (sp.fc,))) - sp.fc) * 2)
        if sp.ftype == "Bandstop":
            span = max(span, 6 * sp.bw)
        lo, hi = max(sp.fc - span, sp.fc * 1e-3), sp.fc + span
        return np.linspace(lo, hi, n)
    top = 3 * sp.fc
    if sp.shape == "Elliptic" and "fs" in res.info and sp.ftype == "Lowpass":
        top = max(top, 2.5 * max(res.info["fs"], sp.fc))
    return np.linspace(top * 1e-4, top, n)


# --------------------------------------------------------------------------- order help

def estimate_orders(ftype, fc, fs, ripple_db, as_db, aa_db=None, bw=None):
    """Minimum order per approximation to reach as_db at fs (the live order estimate).

    Plain scipy only: it runs in the GUI thread while a design may be running in the
    worker thread, and the high-precision mpmath context is shared, not thread-safe."""
    from scipy import signal
    spec = FilterSpec(ftype=ftype, fc=fc, bw=bw or 1.0)
    ws = to_proto(spec, fs)
    if ws <= 1:
        return {s: None for s in SHAPES}
    out = {"Butterworth": int(signal.buttord(1, ws, BUTTER_3DB if aa_db is None else aa_db, as_db, analog=True)[0]),
           "Chebyshev": int(signal.cheb1ord(1, ws, ripple_db, as_db, analog=True)[0]),
           "Elliptic": max(3, int(signal.ellipord(1, ws, ripple_db, as_db, analog=True)[0]))}
    n_b = None
    for n in range(1, 16):
        # Bessel, cutoff (3 dB) at 1 rad/s like the designed filters
        bb, aa = signal.bessel(n, 1.0, analog=True, norm="mag")
        h = signal.freqs(bb, aa, worN=[ws])[1][0]
        if -20 * math.log10(abs(h)) >= as_db:
            n_b = n
            break
    out["Bessel"] = n_b
    return out
