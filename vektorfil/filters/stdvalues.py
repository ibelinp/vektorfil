"""Standard component values (E-series) and snapping a designed filter to them.

Snapping one element at a time and re-tuning the others -- what one would otherwise do
by hand -- is automated here:

  1. snap the element that is closest to a standard value, keep it fixed;
  2. re-optimise the remaining free elements so the response stays close to the ideal;
  3. repeat until every element is standard;
  4. finally try each element one standard step up or down and keep any improvement.

The result is compared against plain rounding to the nearest value and the better one
is returned.  It is never exact, but usually far closer than plain rounding.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7,
       5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
SERIES = {
    "5% (E24)": E24,
    "2% (E48)": [round(10 ** (i / 48), 2) for i in range(48)],
    "1% (E96)": [round(10 ** (i / 96), 2) for i in range(96)],
}
SERIES_NAMES = ["Ideal"] + list(SERIES)


def _decade_table(series):
    base = np.array(series)
    return np.concatenate([base / 10, base, base * 10])


def nearest(x, series):
    """Nearest standard value (geometric distance) and its neighbours below/above."""
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    tab = _decade_table(series)
    i = int(np.argmin(np.abs(np.log(tab / m))))
    return tab[i] * 10 ** e


def step(x, series, direction):
    """Next standard value above (direction=+1) or below (-1) the standard value x."""
    e = math.floor(math.log10(x) + 1e-12)
    m = x / 10 ** e
    tab = _decade_table(series)
    i = int(np.argmin(np.abs(np.log(tab / m))))
    return tab[i + direction] * 10 ** e


# --------------------------------------------------------------------------- snapping

@dataclass
class SnapReport:
    """Result of snap(): the snapped ladder and how far it is from the ideal design."""
    ladder: object
    method: str
    max_pass_dev_db: float      # worst passband |S21| deviation from the ideal design
    worst_stop_loss_db: float   # worst loss of stopband attenuation vs ideal (dB, >= 0)
    rms_error: float


def _slots(lad):
    """(branch index, 'L'|'C') for every element value."""
    out = []
    for i, b in enumerate(lad.branches):
        if b.L is not None:
            out.append((i, "L"))
        if b.C is not None:
            out.append((i, "C"))
    return out


def _get(lad, slots):
    return np.array([getattr(lad.branches[i], a) for i, a in slots])


def _set(lad, slots, vals):
    for (i, a), v in zip(slots, vals):
        setattr(lad.branches[i], a, float(v))


class _Objective:
    """Passband: stay close to the ideal |S21|.  Stopband: only penalise rejection lost
    below the requirement, i.e. the ideal attenuation capped at stop_db (Amin for elliptic
    filters, whose notches may move freely as long as Amin is kept)."""

    def __init__(self, ideal, f, aa_db, stop_db=80.0):
        self.f = f
        s11, s21 = ideal.sparams(f)
        self.ref = 20 * np.log10(np.abs(s21))
        self.pb = self.ref > -(aa_db + 0.5)                 # passband (incl. edges)
        self.sb = self.ref < -20                            # stopband
        self.floor = -abs(stop_db)

    def vector(self, lad):
        """Weighted error vector minimised by the re-tuning (passband + lost rejection)."""
        s21 = lad.sparams(self.f)[1]
        a = 20 * np.log10(np.abs(s21) + 1e-300)
        e_pb = (a[self.pb] - self.ref[self.pb]) * 10.0      # 0.1 dB passband error ~ 1 unit
        ref_sb = np.maximum(self.ref[self.sb], self.floor)
        e_sb = np.maximum(np.maximum(a[self.sb], self.floor) - ref_sb, 0.0)  # only losing rejection
        return np.concatenate([e_pb, e_sb])

    def report(self, lad, method):
        """Summarise a snapped ladder as a SnapReport."""
        s21 = lad.sparams(self.f)[1]
        a = 20 * np.log10(np.abs(s21) + 1e-300)
        pd = float(np.max(np.abs(a[self.pb] - self.ref[self.pb]))) if self.pb.any() else 0.0
        sl = float(np.max(np.maximum(np.maximum(a[self.sb], self.floor) - np.maximum(self.ref[self.sb], self.floor), 0))) \
            if self.sb.any() else 0.0
        v = self.vector(lad)
        return SnapReport(lad, method, pd, sl, float(np.sqrt(np.mean(v * v))))


def snap(ideal, series_name, f, aa_db=0.1, retune=True, stop_db=80.0):
    """Return the better of plain rounding and sequential snap-and-retune.
    stop_db: stopband requirement (dB of attenuation) -- Amin for elliptic filters."""
    series = SERIES[series_name]
    f = np.asarray(f, float)
    if len(f) > 400:
        f = f[np.linspace(0, len(f) - 1, 400).astype(int)]
    obj = _Objective(ideal, f, aa_db, stop_db)

    # plain rounding
    plain = copy.deepcopy(ideal)
    sl = _slots(plain)
    _set(plain, sl, [nearest(v, series) for v in _get(plain, sl)])
    best = obj.report(plain, "nearest value")
    if not retune:
        return best

    lad = copy.deepcopy(ideal)
    slots = _slots(lad)
    vals = _get(lad, slots).astype(float)
    free = list(range(len(slots)))

    def retune_free(vals, free):
        if not free:
            return vals
        x0 = np.log(vals[free])

        def res(x):
            v = vals.copy()
            v[free] = np.exp(x)
            _set(lad, slots, v)
            return obj.vector(lad)
        r = least_squares(res, x0, bounds=(x0 - 0.4, x0 + 0.4), max_nfev=12 * (len(free) + 1),
                          xtol=1e-8, ftol=1e-8)
        v = vals.copy()
        v[free] = np.exp(r.x)
        return v

    while free:
        # snap the free element that is already closest to a standard value
        dist = [abs(math.log(nearest(vals[k], series) / vals[k])) for k in free]
        k = free[int(np.argmin(dist))]
        vals[k] = nearest(vals[k], series)
        free.remove(k)
        vals = retune_free(vals, free)

    def err(v):
        _set(lad, slots, v)
        e = obj.vector(lad)
        return float(e @ e)

    # discrete polish: one standard step up/down per element
    cur = err(vals)
    for _ in range(3):
        improved = False
        for k in range(len(vals)):
            for d in (-1, 1):
                trial = vals.copy()
                trial[k] = step(vals[k], series, d)
                e = err(trial)
                if e < cur - 1e-12:
                    vals, cur, improved = trial, e, True
        if not improved:
            break
    _set(lad, slots, vals)
    tuned = obj.report(lad, "snap and re-tune")
    return tuned if tuned.rms_error <= best.rms_error else best
