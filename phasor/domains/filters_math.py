"""Digital filters as poles and zeros (pure NumPy, no drawing).

A filter is the ratio of two polynomials in z; its poles are the roots of the bottom one and
its zeros the roots of the top. It is stable (bounded input gives bounded output) when every
pole is strictly inside the unit circle. Points are complex numbers; the plane is the z-plane.
"""

from __future__ import annotations

import math

import numpy as np

KINDS = ("custom", "lowpass", "highpass", "bandpass", "notch")


def biquad(kind: str, cutoff: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Zeros and poles of the standard second-order section (RBJ cookbook). ``cutoff`` is a
    share of the sampling rate (0 to 0.5, the Nyquist frequency)."""
    w0 = 2 * math.pi * cutoff
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    b = {
        "lowpass": np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2]),
        "highpass": np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]),
        "bandpass": np.array([alpha, 0.0, -alpha]),
        "notch": np.array([1.0, -2 * cw, 1.0]),
    }[kind]
    return np.roots(b), np.roots(a)


def pole_radii(poles) -> np.ndarray:
    return np.abs(np.asarray(poles, dtype=np.complex128).reshape(-1))


def max_pole_radius(poles) -> float:
    r = pole_radii(poles)
    return float(r.max()) if r.size else 0.0


def is_stable(poles) -> bool:
    """Every pole strictly inside the unit circle (no poles is stable)."""
    return max_pole_radius(poles) < 1.0


def response(zeros, poles, gain: float, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies (0 to 0.5 of the sampling rate) and the complex frequency response."""
    f = np.linspace(0.0, 0.5, n)
    z = np.exp(2j * np.pi * f)
    h = np.full(n, gain, dtype=np.complex128)
    for q in np.asarray(zeros, dtype=np.complex128).reshape(-1):
        h = h * (z - q)
    for p in np.asarray(poles, dtype=np.complex128).reshape(-1):
        h = h / (z - p)
    return f, h
