"""The interactive design layer: a FilterSpec in, a finished DesignResult out.

    design     FilterSpec -> design() -> DesignResult; circuit names; order estimate
    stdvalues  E-series standard values with snap-and-retune
    diplexer   lowpass/highpass diplexers from singly terminated channels
    project    design files (.vfil)
"""
from .design import (COUPLED, SERIES_FIRST, SHAPES, SHUNT_FIRST, SYMMETRIC, TYPES, ZIGZAG, DesignResult,
                     FilterSpec, canonical_subtype, design, estimate_orders,
                     resonator_unit, subtypes, auto_freqs)
from .stdvalues import SERIES_NAMES

__all__ = ["COUPLED", "SERIES_FIRST", "SHAPES", "SHUNT_FIRST", "SYMMETRIC", "TYPES", "ZIGZAG", "DesignResult",
           "FilterSpec", "canonical_subtype", "design", "estimate_orders", "resonator_unit", "subtypes",
           "auto_freqs", "SERIES_NAMES"]
