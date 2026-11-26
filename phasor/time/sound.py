"""Sound events (SPEC "Sound events"): notes and WAV files on the soundtrack. Pure NumPy/SciPy.

A document's ``sounds`` (core.schema ``Note`` and ``SoundFile``) are resolved to seconds by
``compile_document`` through the TimeMap, like tracks, so retiming the narration moves them.
``soundtrack`` turns them into samples and mixes them with the narration:

- **Placement is sample exact.** An event at ``t`` starts at sample ``round(t * rate)``. The
  rate is the narration's when it has audio, else ``RATE``. A file at another rate is
  resampled (polyphase, ``scipy.signal.resample_poly``).
- **Notes are dry and deterministic.** A voice is a fixed sum of sines (no noise, no reverb)
  under a linear attack of ``ATTACK`` and release of ``RELEASE`` seconds that fit inside
  ``dur``; the result is scaled so that its peak is exactly ``amp``. The first sample of every
  note is 0 (the attack starts there).
- **Headroom.** The narration is never changed. Where narration plus sounds would pass
  ``CEILING`` (-1 dBFS), or the narration's own level where that is louder, only the sounds
  are turned down, by a gain that looks ``LOOKAHEAD`` seconds ahead and ramps over twice
  that, so nothing clips and nothing clicks. Below the ceiling every sample is the plain sum.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phasor.core import files
from phasor.core.schema import Note, SoundFile

RATE = 48000  # Hz, when there is no recorded narration to match
CEILING = 10 ** (-1 / 20)  # -1 dBFS: the loudest the sounds may make the mix
LOOKAHEAD = 0.010  # s: the ducking gain starts this long before a peak and ramps over 2x
ATTACK = 0.005  # s: a note's linear fade-in
RELEASE = 0.030  # s: a note's linear fade-out, at the end of its ``dur``

# Partials of each voice: (ratio to the fundamental, amplitude, decay rate in 1/s; 0 = steady)
VOICES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "sine": ((1.0, 1.0, 0.0),),
    "soft": ((1.0, 1.0, 0.0), (2.0, 0.35, 0.0), (3.0, 0.12, 0.0), (4.0, 0.05, 0.0)),
    "pluck": tuple((float(k), 1.0 / k, 3.0 + 2.5 * k) for k in range(1, 9)),
    "bell": ((1.0, 1.0, 1.5), (2.76, 0.6, 2.5), (5.4, 0.35, 4.0), (8.93, 0.2, 6.0)),
}
NOTE_NAMES = ("c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b")


@dataclass(frozen=True)
class PlacedSound:
    """A sound event with its start resolved to seconds on the scene timeline."""

    id: str
    t: float
    spec: Note | SoundFile


def midi_hz(midi: float) -> float:
    """Equal temperament, A4 (69) = 440 Hz."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def note_name(midi: float) -> str:
    """``a4`` for 69, ``cs5`` for 73 (the nearest semitone; ``s`` for sharp, id-safe)."""
    m = round(midi)
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def synth(note: Note, rate: int) -> np.ndarray:
    """The note as float64 samples, ``round(dur * rate)`` long, peak exactly ``amp``."""
    n = max(1, round(note.dur * rate))
    t = np.arange(n) / rate
    f0 = midi_hz(note.midi)
    x = np.zeros(n)
    for ratio, amp, decay in VOICES[note.voice]:
        if f0 * ratio < rate / 2:  # no partial above Nyquist (it would alias)
            part = amp * np.sin(2.0 * np.pi * f0 * ratio * t)
            x += part * np.exp(-decay * t) if decay else part
    a = max(1, min(n // 2, round(ATTACK * rate)))
    r = max(1, min(n // 2, round(RELEASE * rate)))
    env = np.ones(n)
    env[:a] = np.arange(a) / a  # starts at exactly 0
    env[n - r :] = np.minimum(env[n - r :], np.arange(r, 0, -1) / r - 1 / r)  # ends at 0
    x *= env
    peak = float(np.abs(x).max())
    return x * (note.amp / peak) if peak > 0 else x


def read_samples(base_dir: str | Path, sound: SoundFile, rate: int) -> np.ndarray | None:
    """The file as float64 samples at ``rate``, times ``gain``; None when it is missing or not
    a 16-bit PCM WAV (lint says which). Read every time: an edited file is heard at once."""
    from phasor.time.narration import read_wav

    try:
        samples, src = read_wav(files.resolve(base_dir, sound.path))
    except (OSError, ValueError, EOFError, wave.Error):
        return None
    x = samples.astype(np.float64) / 32768.0
    if src != rate:
        from scipy.signal import resample_poly

        g = math.gcd(rate, src)
        x = resample_poly(x, rate // g, src // g)
    return x * sound.gain


def file_seconds(base_dir: str | Path, sound: SoundFile) -> float | None:
    """How long the file plays, from its header; None when it cannot be read."""
    try:
        with wave.open(str(files.resolve(base_dir, sound.path)), "rb") as w:
            if w.getsampwidth() != 2:
                return None
            return w.getnframes() / w.getframerate()
    except (OSError, ValueError, EOFError, wave.Error):
        return None


def duck_gain(voice: np.ndarray, bus: np.ndarray, rate: int) -> np.ndarray:
    """The gain for ``bus`` so that ``|voice + gain * bus| <= max(CEILING, |voice|)`` at every
    sample, smooth and as close to 1 as that allows.

    At each sample the gains that keep the sum in bounds form an interval [0, need] (gain 0
    leaves the voice alone, which is within bounds). A running minimum of ``need`` over
    2W+1 samples followed by a running mean over the same window never exceeds ``need`` at
    the centre (every value averaged is a minimum over a window holding the centre), so the
    smoothed gain is always safe."""
    limit = np.maximum(CEILING, np.abs(voice))
    over = np.abs(voice + bus) > limit
    if not over.any():
        return np.ones_like(bus)
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    need = np.ones_like(bus)
    b = bus[over]
    need[over] = np.clip((limit[over] - np.sign(b) * voice[over]) / np.abs(b), 0.0, 1.0)
    size = 2 * max(1, round(LOOKAHEAD * rate)) + 1
    smooth = uniform_filter1d(minimum_filter1d(need, size, mode="nearest"), size, mode="nearest")
    return np.minimum(smooth, need)  # float rounding in the mean must never tip it over


def to_int16(x: np.ndarray) -> np.ndarray:
    """Float samples (full scale 1.0 = 32768) to int16, the inverse of reading a WAV."""
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def soundtrack(
    sounds: Sequence[PlacedSound],
    base_dir: str | Path,
    duration: float,
    voice: tuple[np.ndarray, int] | None = None,
) -> tuple[np.ndarray, int] | None:
    """The scene's audio: the narration track ``voice`` (mono int16 and its rate, from
    ``narration.mix_track``) with every sound event mixed in. None when there is neither.
    The track lasts ``duration`` (or the narration, if longer); sounds are cut at its end."""
    if not sounds:
        return voice
    rate = voice[1] if voice is not None else RATE
    n = math.ceil(duration * rate - 1e-9)
    if voice is not None:
        n = max(n, len(voice[0]))
    bus = np.zeros(n)
    for p in sounds:
        spec = p.spec
        x = synth(spec, rate) if isinstance(spec, Note) else read_samples(base_dir, spec, rate)
        i = round(p.t * rate)
        if x is None or i >= n:
            continue
        x = x[: n - i]
        bus[i : i + len(x)] += x
    v = np.zeros(n)
    if voice is not None:
        v[: len(voice[0])] = voice[0].astype(np.float64) / 32768.0
    return to_int16(v + duck_gain(v, bus, rate) * bus), rate
