"""The built-in sound effects (SOUND.md): short sounds made from numbers, so a scene needs
no downloads and no licences. Each is a pure function of (name, cue id, seed): the noise
comes from ``rng`` (SPEC contract 3), so two cues sound alike but not identical, and the
same cue sounds the same on every render.

All are mono float32 at 48 kHz, peaking near -6 dBFS; the mixer sets their level from
their loudness, so these levels only fix their shape.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from numpy.random import Generator

from phasor.core.rng import rng
from phasor.core.schema import BUILTIN_SOUNDS

RATE = 48_000


def builtin(name: str, cue_id: str = "", seed: int = 0) -> tuple[np.ndarray, int]:
    """The built-in sound ``name`` for cue ``cue_id``: mono float32 samples (read-only) and
    the rate. ``KeyError`` for a name that is not in ``BUILTIN_SOUNDS``."""
    if name not in BUILTIN_SOUNDS:
        raise KeyError(name)
    return _make(name, cue_id, seed), RATE


@functools.lru_cache(maxsize=64)
def _make(name: str, cue_id: str, seed: int) -> np.ndarray:
    g = rng(f"sound.{name}.{cue_id}", seed)
    x = _SYNTHS[name](g)
    x = x * (0.5 / max(float(np.abs(x).max()), 1e-9))  # peak -6 dBFS
    x = _fade(x, 0.002, 0.01).astype(np.float32)
    x.setflags(write=False)
    return x


def _t(seconds: float) -> np.ndarray:
    return np.arange(round(seconds * RATE)) / RATE


def _fade(x: np.ndarray, fade_in: float, fade_out: float) -> np.ndarray:
    """Raised-cosine ends, so no sound starts or stops with a click."""
    x = np.array(x, dtype=np.float64)
    a, b = min(round(fade_in * RATE), len(x)), min(round(fade_out * RATE), len(x))
    if a > 0:
        x[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    if b > 0:
        x[len(x) - b :] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(b) / b))[::-1]
    return x


def _tick(g: Generator) -> np.ndarray:
    """A small wooden tick: a short bright tone with a breath of noise, 40 ms."""
    t = _t(0.04)
    f = 2400.0 * (1.0 + 0.02 * g.standard_normal())
    tone = np.sin(2 * math.pi * f * t) * np.exp(-t / 0.006)
    noise = g.standard_normal(len(t)) * np.exp(-t / 0.002) * 0.3
    return tone + noise


def _pop(g: Generator) -> np.ndarray:
    """A soft pop: a tone falling from about 700 Hz to 180 Hz, 90 ms."""
    t = _t(0.09)
    f0 = 700.0 * (1.0 + 0.03 * g.standard_normal())
    f = 180.0 + (f0 - 180.0) * np.exp(-t / 0.015)
    phase = 2 * math.pi * np.cumsum(f) / RATE
    return np.sin(phase) * np.exp(-t / 0.025)


def _chime(g: Generator) -> np.ndarray:
    """A small glass bell: inharmonic partials over 880 Hz, each dying away, 1.8 s."""
    t = _t(1.8)
    base = 880.0 * (1.0 + 0.002 * g.standard_normal())
    out = np.zeros_like(t)
    for ratio, amp, decay in ((1.0, 1.0, 0.9), (2.0, 0.5, 0.5), (2.76, 0.35, 0.35),
                              (5.4, 0.15, 0.15), (8.93, 0.06, 0.08)):  # fmt: skip
        phase = g.uniform(0, 2 * math.pi)
        out += amp * np.sin(2 * math.pi * base * ratio * t + phase) * np.exp(-t / decay)
    attack = np.minimum(1.0, t / 0.004)
    return out * attack


def _band_noise(g: Generator, seconds: float, lo: float, hi: float,
                block: float = 0.01) -> np.ndarray:  # fmt: skip
    """Noise through a band-pass whose centre glides from ``lo`` to ``hi`` Hz (log scale),
    filtered block by block with the filter's state carried over."""
    from scipy.signal import butter, sosfilt, sosfilt_zi

    n, hop = round(seconds * RATE), round(block * RATE)
    noise = g.standard_normal(n)
    out = np.empty(n)
    zi = None
    for i in range(0, n, hop):
        frac = i / max(n - 1, 1)
        centre = lo * (hi / lo) ** frac
        sos = butter(2, [centre / 1.5, min(centre * 1.5, 0.45 * RATE)], btype="bandpass",
                     fs=RATE, output="sos")  # fmt: skip
        if zi is None:
            zi = sosfilt_zi(sos) * 0.0
        out[i : i + hop], zi = sosfilt(sos, noise[i : i + hop], zi=zi)
    return out


def _whoosh(g: Generator) -> np.ndarray:
    """Air rushing past: band noise gliding up, swelling and fading, 0.7 s."""
    seconds = 0.7
    x = _band_noise(g, seconds, 300.0, 2600.0)
    t = _t(seconds) / seconds
    return x * np.sin(np.pi * t) ** 2


def _swell(g: Generator) -> np.ndarray:
    """A rising swell: a soft fifth and low noise growing, then let go, 1.6 s."""
    seconds = 1.6
    t = _t(seconds)
    rise = (t / seconds) ** 2.5
    tail = np.exp(-np.maximum(t - 0.9 * seconds, 0.0) / 0.06)
    root = 220.0 * (1.0 + 0.002 * g.standard_normal())
    tone = sum(a * np.sin(2 * math.pi * root * r * t) for r, a in ((1.0, 1.0), (1.5, 0.6),
                                                                    (2.0, 0.3)))  # fmt: skip
    air = _band_noise(g, seconds, 200.0, 1200.0) * 0.4
    return (tone + air) * rise * tail


_SYNTHS = {"chime": _chime, "pop": _pop, "swell": _swell, "tick": _tick, "whoosh": _whoosh}
assert set(_SYNTHS) == set(BUILTIN_SOUNDS)
