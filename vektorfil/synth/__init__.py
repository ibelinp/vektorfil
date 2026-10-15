"""The synthesis engine: pure mathematics, no user interface.

    poly       high-precision polynomial helpers (mpmath)
    approx     lowpass approximations: the polynomials E, F, P of a transfer function
    placer     minimax placement of transmission zeros against a stopband mask
    ladder     Darlington ladder synthesis by zero shifting (lowpass prototypes)
    bandpass   direct bandpass synthesis (zig-zag and other band ladders)
    bandxform  element-wise lowpass -> bandpass / bandstop transforms
    coupled    narrow-band coupled-resonator bandpass filters
    symmetric  the quasi-symmetric (Carassa) bandpass
    network    the finished ladder: scaling, ABCD analysis, S-parameters, tables
"""
from .approx import Prototype, butterworth_with_zeros, chebyshev_with_zeros, elliptic, cauer_zeros
from .ladder import synthesize, SynthesisError
from .network import Ladder

__all__ = ["Prototype", "butterworth_with_zeros", "chebyshev_with_zeros", "elliptic", "cauer_zeros",
           "synthesize", "SynthesisError", "Ladder"]
