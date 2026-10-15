"""Step and impulse response of a design, from its frequency response.

The transmission S21(f) is sampled from DC to f_max and treated as the spectrum of a
real, causal signal; an inverse FFT gives the impulse response

    h(t_m) ~ sum_k S21(f_k) exp(j 2 pi f_k t_m) df      (both sides of the spectrum)

and the step response is its running integral.  For a doubly terminated lowpass the
step response settles at S21(0) = 1.  f_max should be well into the stopband (the
response is cut off there); the frequency step sets the time window, 1/df, which must
be longer than the response itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vektorfil.synth.network import Ladder


@dataclass
class TimeResponse:
    """Sampled impulse and step response."""
    t: np.ndarray           # s
    impulse: np.ndarray     # 1/s (area = DC transmission)
    step: np.ndarray        # dimensionless


def time_response(lad: Ladder, f_max: float, n: int = 1 << 14) -> TimeResponse:
    """Impulse and step response from n+1 samples of S21 between 0 and f_max."""
    f = np.linspace(0.0, f_max, n + 1)
    df = f[1]
    fe = f.copy()
    fe[0] = df * 1e-6               # the lumped model is singular exactly at DC
    s21 = lad.sparams(fe)[1]
    s21[0] = s21[0].real            # the spectrum of a real signal is real at DC
    # gentle taper over the top 10 % so the cut at f_max does not ring
    k0 = int(0.9 * n)
    taper = np.ones(n + 1)
    taper[k0:] = 0.5 * (1 + np.cos(np.pi * np.arange(n + 1 - k0) / (n - k0)))
    m = 2 * n
    h = np.fft.irfft(s21 * taper, m) * m * df
    dt = 1.0 / (m * df)
    t = np.arange(m) * dt
    step = np.cumsum(h) * dt
    return TimeResponse(t, h, step)


def auto_window(spec, info) -> tuple[float, float]:
    """(f_max, t_end) suited to a design: the spectrum to well beyond the passband, the
    time axis to a few delays and settling times."""
    if spec.ftype in ("Bandpass", "Bandstop"):
        bw = max(spec.bw, 1e-9 * spec.fc)
        f_max = spec.fc + 12 * bw if spec.ftype == "Bandpass" else 3 * spec.fc + 12 * bw
        t_end = 12.0 / bw
    else:
        f_max = 20 * spec.fc
        t_end = 12.0 / spec.fc
    return f_max, t_end
