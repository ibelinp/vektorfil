"""Single-layer air-core coils: how many turns for a given inductance.

The inductance of a single-layer solenoid (current-sheet model) is Wheeler's 1982
formula, accurate to about 0.1 % over all length/diameter ratios:

    L = mu0 * N^2 * a * [ ln(1 + pi a / b) + 1 / (2.3 + 1.6 b/a + 0.44 (b/a)^2) ]

with a the coil radius (to the wire centre) and b the winding length.  For a long coil
it becomes the textbook mu0 N^2 pi a^2 / b.  The round-wire correction (a few percent
for widely spaced turns) is ignored, as is the self-capacitance.

    H. A. Wheeler, "Inductance formulas for circular and square coils",
    Proc. IEEE, vol. 70, no. 12, 1982.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MU0 = 4e-7 * math.pi


def awg_mm(awg: int) -> float:
    """Diameter (mm) of American Wire Gauge size awg (bare copper)."""
    return 0.127 * 92 ** ((36 - awg) / 39)


def inductance(turns: float, radius: float, length: float) -> float:
    """Inductance (H) of a single-layer solenoid: radius to the wire centre and
    winding length in metres."""
    a, b = radius, max(length, 1e-12)
    return MU0 * turns ** 2 * a * (math.log(1 + math.pi * a / b) + 1 / (2.3 + 1.6 * b / a + 0.44 * (b / a) ** 2))


@dataclass
class CoilDesign:
    """Result of design_coil(); lengths in metres, inductances in henry."""
    turns: float            # exact (fractional) number of turns for the target
    turns_whole: int        # nearest whole number of turns (at least 1)
    l_whole: float          # inductance with turns_whole turns
    length: float           # winding length for the exact turns
    mean_diameter: float    # coil diameter to the wire centre
    wire_length: float      # wire in the winding (leads not included)
    pitch: float            # turn-to-turn spacing


def design_coil(l_target: float, form_diameter: float, wire_diameter: float,
                pitch: float | None = None) -> CoilDesign:
    """Turns of wire (diameter wire_diameter) on a form of form_diameter that give
    l_target.  pitch: turn spacing; None or 0 = close-wound (pitch = wire diameter).
    Lengths in metres."""
    if l_target <= 0 or form_diameter <= 0 or wire_diameter <= 0:
        raise ValueError("inductance, form and wire diameter must be positive")
    p = pitch if pitch and pitch > 0 else wire_diameter
    if p < wire_diameter:
        raise ValueError("the pitch is smaller than the wire diameter")
    d = form_diameter + wire_diameter
    a = d / 2

    def l_of(n):
        return inductance(n, a, n * p)

    lo, hi = 1e-3, 1.0
    while l_of(hi) < l_target:            # inductance grows monotonically with the turns
        hi *= 2
        if hi > 1e6:
            raise ValueError("needs more than a million turns")
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if l_of(mid) < l_target else (lo, mid)
    n = math.sqrt(lo * hi)
    nw = max(1, round(n))
    return CoilDesign(turns=n, turns_whole=nw, l_whole=l_of(nw), length=n * p, mean_diameter=d,
                      wire_length=n * math.hypot(math.pi * d, p), pitch=p)
