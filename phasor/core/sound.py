"""Sound: the primitives an object uses to make noise, and the mixer (SPEC "Sound").

Pure NumPy, like ``core/rng.py``: no audio library, no drawing. A sonifiable object has an
optional ``sounds(ctx)`` method (found with ``getattr``, so it is not part of the object
contract) that returns ``Sound`` events. ``Engine.audio_track(sound=True)`` mixes them with
the narration. Sound is a pure function of (document, t) like everything else: an object
finds *when* by sampling ``ctx.state_at`` and never by being told a clock.

Sound is opt-in: nothing here runs unless a render or an export asks for it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from phasor.core.state import SceneState

RATE = 48000  # Hz, when there is no narration to match


@dataclass(frozen=True)
class Sound:
    """One sound placed on the timeline: ``samples`` (mono float, about +-1) start at ``t``
    seconds, scaled by ``gain``."""

    t: float
    samples: np.ndarray
    gain: float = 1.0
    label: str = ""  # what it is, for tests and the inspector (e.g. "arrival 3")


@dataclass(frozen=True)
class SoundContext:
    """What an object's ``sounds`` method may use."""

    state_at: Callable[[float], SceneState]
    duration: float  # scene seconds
    rate: int  # sample rate to synthesize at
    theme: Mapping[str, float]  # the theme's [sound] table
    _grid: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    def times(self) -> np.ndarray:
        """The control grid: ``[sound] control_hz`` samples a second over the scene. Anything
        that changes smoothly (a pitch, a distance) is read here and interpolated."""
        if "t" not in self._grid:
            n = max(2, int(np.ceil(self.duration * self.theme["control_hz"])) + 1)
            self._grid["t"] = np.linspace(0.0, self.duration, n)
        return self._grid["t"]

    def series(self, fn: Callable[[SceneState], float]) -> np.ndarray:
        """``fn(state_at(t))`` for every time on the control grid."""
        return np.array([float(fn(self.state_at(float(t)))) for t in self.times()])


def semitones(n: float) -> float:
    """A frequency ratio of ``n`` semitones."""
    return 2.0 ** (n / 12.0)


# ------------------------------------------------------------------ finding when


def crossings(times: np.ndarray, values: np.ndarray, level: float, *,
              rising: bool | None = None) -> np.ndarray:  # fmt: skip
    """The times at which ``values`` (sampled at ``times``) cross ``level``, by linear
    interpolation between samples. ``rising`` True: upward only, False: downward only,
    None: both. A sample that sits exactly on the level counts once."""
    v = np.asarray(values, dtype=np.float64) - level
    t = np.asarray(times, dtype=np.float64)
    out = []
    for i in range(len(v) - 1):
        a, b = v[i], v[i + 1]
        if a == b:
            continue
        up = a < 0 <= b
        down = a > 0 >= b
        if (up and rising is not False) or (down and rising is not True):
            out.append(t[i] + (t[i + 1] - t[i]) * (0 - a) / (b - a))
    return np.asarray(out)


# ------------------------------------------------------------------ primitives


def _t(dur: float, rate: int) -> np.ndarray:
    return np.arange(max(1, round(dur * rate))) / rate


def _edges(x: np.ndarray, rate: int, attack: float, release: float) -> np.ndarray:
    """Short linear fades, so no sound starts or ends with a pop."""
    a = max(1, min(len(x) // 2, int(attack * rate)))
    r = max(1, min(len(x) // 2, int(release * rate)))
    x[:a] *= np.linspace(0.0, 1.0, a)
    x[-r:] *= np.linspace(1.0, 0.0, r)
    return x


def _unit(x: np.ndarray) -> np.ndarray:
    return x / max(1e-12, float(np.abs(x).max()))


def click(rate: int, hz: float = 1800.0, tau: float = 0.005, dur: float | None = None
          ) -> np.ndarray:  # fmt: skip
    """A tick: a damped sine that starts at once (the first sample is the onset)."""
    t = _t(dur if dur is not None else 8 * tau, rate)
    x = np.sin(2 * np.pi * hz * t) * np.exp(-t / tau)
    x[0] = 0.0
    n = min(len(x), max(2, int(0.0003 * rate)))
    x[:n] *= np.linspace(0.0, 1.0, n)
    return _unit(x).astype(np.float32)


def tone(rate: int, hz: float, dur: float, harmonics: Sequence[float] = (1.0, 0.3, 0.1),
         attack: float = 0.01, release: float = 0.06) -> np.ndarray:  # fmt: skip
    """A steady note with a few harmonics."""
    t = _t(dur, rate)
    x = sum(
        a * np.sin(2 * np.pi * hz * (k + 1) * t)
        for k, a in enumerate(harmonics)
        if hz * (k + 1) < rate / 2
    )
    return _edges(_unit(np.asarray(x, dtype=np.float64)), rate, attack, release).astype(np.float32)


def pluck(rate: int, hz: float, dur: float = 0.6, tau: float = 0.25) -> np.ndarray:
    """A plucked note: harmonics that die away, the high ones first."""
    t = _t(dur, rate)
    x = np.zeros_like(t)
    for k in range(1, 7):
        if hz * k < rate / 2:
            x += np.sin(2 * np.pi * hz * k * t) * np.exp(-t / (tau / (1 + 0.6 * (k - 1)))) / k
    return _edges(_unit(x), rate, 0.001, 0.01).astype(np.float32)


def bell(rate: int, hz: float, dur: float = 1.2, tau: float = 0.5) -> np.ndarray:
    """An inharmonic strike."""
    t = _t(dur, rate)
    x = np.zeros_like(t)
    for ratio, amp in ((1.0, 1.0), (2.76, 0.6), (5.4, 0.35), (8.93, 0.2)):
        if hz * ratio < rate / 2:
            x += amp * np.sin(2 * np.pi * hz * ratio * t) * np.exp(-t / (tau / (1 + 0.3 * ratio)))
    return _edges(_unit(x), rate, 0.0008, 0.02).astype(np.float32)


def chirp(rate: int, f0: float, f1: float, dur: float, attack: float = 0.01,
          release: float = 0.03) -> np.ndarray:  # fmt: skip
    """A glide from ``f0`` to ``f1`` Hz (geometric: equal steps in pitch)."""
    t = _t(dur, rate)
    f = f0 * (f1 / f0) ** (t / max(dur, 1e-9))
    x = np.sin(2 * np.pi * np.cumsum(f) / rate)
    return _edges(x, rate, attack, release).astype(np.float32)


def partials(rate: int, hzs: Sequence[float], amps: Sequence[float], dur: float,
             attack: float = 0.02, release: float = 0.08) -> np.ndarray:  # fmt: skip
    """Additive synthesis: a sum of sines, the way a Fourier series sounds."""
    t = _t(dur, rate)
    x = np.zeros_like(t)
    for f, a in zip(hzs, amps, strict=True):
        if 0 < f < rate / 2:
            x += a * np.sin(2 * np.pi * f * t)
    return _edges(_unit(x), rate, attack, release).astype(np.float32)


def follow(rate: int, times: np.ndarray, hz: np.ndarray, amp: np.ndarray,
           harmonics: Sequence[float] = (1.0, 0.25), fade: float = 0.01) -> Sound:  # fmt: skip
    """A continuous tone whose pitch and loudness are given on a control grid.

    ``times`` (scene seconds), ``hz`` and ``amp`` have the same length. Frequency is
    *integrated* into phase, which is what keeps a gliding tone smooth. Where ``amp`` is 0
    nothing sounds, so a tone that is not there yet costs no more than silence."""
    t0, t1 = float(times[0]), float(times[-1])
    n = max(2, round((t1 - t0) * rate))
    u = t0 + np.arange(n) / rate
    f = np.interp(u, times, hz)
    a = np.clip(np.interp(u, times, amp), 0.0, None)
    phase = 2 * np.pi * np.cumsum(f) / rate
    x = np.zeros(n)
    for k, h in enumerate(harmonics):
        x += h * np.sin((k + 1) * phase) * ((k + 1) * f < rate / 2)
    x *= a / max(1e-12, float(np.sum(harmonics)))
    return Sound(t0, _edges(x, rate, fade, fade).astype(np.float32), 1.0, "follow")


# ------------------------------------------------------------------ mixing


def mix(sounds: Sequence[Sound], duration: float, rate: int, master: float = 1.0,
        ceiling: float = 0.9) -> np.ndarray:  # fmt: skip
    """Sum every sound into one float track ``duration`` seconds long (cut at the end).
    If the sum would pass ``ceiling`` the whole track is turned down by one common factor,
    so loud and quiet sounds keep their proportions."""
    out = np.zeros(int(np.ceil(duration * rate)), dtype=np.float64)
    for s in sounds:
        i = round(s.t * rate)
        if i >= len(out) or i + len(s.samples) <= 0:
            continue
        lo = max(0, -i)
        seg = s.samples[lo : len(out) - i]
        out[i + lo : i + lo + len(seg)] += s.gain * seg
    out *= master
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > ceiling:
        out *= ceiling / peak
    return out.astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def collect(objects: Sequence[Any], ctx: SoundContext) -> list[Sound]:
    """Every sound the scene's objects make: each object's optional ``sounds(ctx)``."""
    out: list[Sound] = []
    for o in objects:
        fn = getattr(o.impl, "sounds", None)
        if fn is not None:
            out.extend(fn(ctx))
    return out
