"""Sound numerics for the domain objects (pure NumPy, no drawing): an additive tone whose
partials switch on and off over time. The primitives live in ``core/sound.py``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from phasor.core import sound as snd


def stack(rate: int, times: np.ndarray, hzs: Sequence[float], amps: Sequence[float],
          on: np.ndarray | None = None, label: str = "partials") -> snd.Sound | None:  # fmt: skip
    """Partial ``i`` is a sine at ``hzs[i]`` Hz with weight ``amps[i]``, sounding where
    ``on[:, i]`` (one row per control time, truthy or a 0-1 level) is on; ``on=None`` means
    always. The control grid is interpolated, so a partial that switches on fades in over one
    control step (a few ms) and nothing clicks. The sum is scaled so its loudest instant is
    1 (one factor for the whole sound: partial proportions never change). Partials at or
    above Nyquist, or that are never on, are left out; None if nothing is left."""
    times = np.asarray(times, dtype=np.float64)
    gate = np.ones((len(times), len(hzs))) if on is None else np.asarray(on, dtype=np.float64)
    out: np.ndarray | None = None
    t0 = float(times[0])
    for i, (hz, a) in enumerate(zip(hzs, amps, strict=True)):
        if not (0.0 < hz < rate / 2) or a == 0.0 or not gate[:, i].any():
            continue
        s = snd.follow(rate, times, np.full(len(times), hz), a * gate[:, i], harmonics=(1.0,))
        out = s.samples.astype(np.float64) if out is None else out + s.samples
    if out is None:
        return None
    out /= max(1e-12, float(np.abs(out).max()))
    return snd.Sound(t0, out.astype(np.float32), 1.0, label)


def harmonic_series(kind: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The first ``count`` terms of a Fourier series (as in ``fourier_math.series_partial_sum``):
    their harmonic numbers and amplitudes. square: odd n, 4/(pi n); sawtooth: every n, 2/(pi n);
    triangle: odd n, 8/(pi n)^2 (the alternating sign of the triangle is a phase, not heard)."""
    if kind == "square":
        n = 2 * np.arange(1, count + 1) - 1
        return n, 4 / (np.pi * n)
    if kind == "sawtooth":
        n = np.arange(1, count + 1)
        return n, 2 / (np.pi * n)
    if kind == "triangle":
        n = 2 * np.arange(count) + 1
        return n, 8 / (np.pi * n) ** 2
    raise ValueError(f"unknown series {kind!r}")
