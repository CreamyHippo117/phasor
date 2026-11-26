"""The sound of a scene: every source mixed on one stereo bus (SOUND.md).

The mix is a pure function of the document and the files it names, like ``state_at``: the
same document gives the same samples, bit for bit. Everything here is NumPy and SciPy, so
the encoder and the desktop app play the same mix.

- Sources are stereo float32 in -1..1; mono is centred (the same samples in both channels).
- The bus runs at the sources' common sample rate, or at 48 kHz when they differ.
- Loudness is measured as ITU-R BS.1770-4 integrated loudness (LUFS) with EBU R128 gating.
- The master is a look-ahead limiter at a ceiling (-1 dBFS by default). It only acts where
  the mix would cross the ceiling; quieter mixes pass through untouched.

Files are read only from inside the scene's folder (``core/files.py``).
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phasor.core import files
from phasor.core.schema import Narration

RATE = 48_000  # the bus rate when the sources disagree
CEILING_DB = -1.0  # where the master limiter holds the peaks


class SoundError(ValueError):
    pass


# ---------------------------------------------------------------- reading


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """A sound file as stereo float32 in -1..1, shape (n, 2), and its sample rate.

    WAV (8, 16, 24, 32-bit or float) needs nothing extra. Other formats (OGG, FLAC, MP3) are
    read through ``soundfile`` when it is installed. Cached per file version; the array is
    read-only."""
    p = Path(path)
    return _read(str(p), os.path.getmtime(p))


@functools.lru_cache(maxsize=32)
def _read(path: str, mtime: float) -> tuple[np.ndarray, int]:
    if path.lower().endswith(".wav"):
        from scipy.io import wavfile

        rate, data = wavfile.read(path)
        if data.dtype.kind == "u":  # 8-bit PCM is unsigned, centred on 128
            x = (data.astype(np.float64) - 128.0) / 128.0
        elif data.dtype.kind == "i":
            x = data.astype(np.float64) / float(-np.iinfo(data.dtype).min)
        else:
            x = data.astype(np.float64)
    else:
        try:
            import soundfile
        except ImportError:
            msg = (f"{Path(path).name}: only WAV can be read without the `soundfile` package "
                   "(uv sync --extra sound); install it or convert the file to WAV")  # fmt: skip
            raise SoundError(msg) from None
        x, rate = soundfile.read(path, dtype="float64", always_2d=True)
    out = _stereo(x, path)
    out.setflags(write=False)
    return out, int(rate)


def _stereo(x: np.ndarray, name: str = "sound") -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] not in (1, 2):
        raise SoundError(f"{name}: only mono or stereo sound is supported")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, rate: int, to: int) -> np.ndarray:
    """``x`` (n, channels) from ``rate`` to ``to`` samples per second (polyphase filter)."""
    if rate == to:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(int(rate), int(to))
    y = resample_poly(np.asarray(x, dtype=np.float64), to // g, rate // g, axis=0)
    return y.astype(np.float32)


# ---------------------------------------------------------------- the bus


@dataclass(frozen=True, eq=False)
class Placed:
    """One clip on the bus: stereo float samples at ``rate``, starting ``start`` seconds into
    the scene, ``gain_db`` louder (negative: quieter)."""

    start: float
    samples: np.ndarray
    rate: int
    gain_db: float = 0.0


def bus_rate(clips: Sequence[Placed]) -> int:
    rates = {c.rate for c in clips}
    return rates.pop() if len(rates) == 1 else RATE


def mix(clips: Sequence[Placed], duration: float) -> tuple[np.ndarray, int] | None:
    """The clips summed on one stereo float32 bus at least ``duration`` seconds long (longer
    if a clip runs past it), and its rate. None with no clips. No limiting: see ``master``."""
    if not clips:
        return None
    rate = bus_rate(clips)
    placed = [(c, resample(_stereo(c.samples), c.rate, rate)) for c in clips]
    end = max(duration, max(c.start + len(s) / rate for c, s in placed))
    bus = np.zeros((math.ceil(end * rate), 2), dtype=np.float32)
    for c, s in placed:
        i = round(c.start * rate)
        if i < 0:  # a clip starting before the scene: keep what falls inside it
            s, i = s[-i:], 0
        s = s[: len(bus) - i]
        bus[i : i + len(s)] += s if c.gain_db == 0 else s * np.float32(db_to_gain(c.gain_db))
    return bus, rate


def master(x: np.ndarray, rate: int, ceiling_db: float = CEILING_DB) -> np.ndarray:
    """``x`` with its peaks held at ``ceiling_db`` dBFS by a smooth look-ahead limiter.

    The gain needed at each sample is held for 50 ms either side and then averaged over the
    same span, so it ramps down before a peak and back up after it, never above what any
    sample in reach needs. A mix that stays under the ceiling comes back unchanged."""
    ceiling = db_to_gain(ceiling_db)
    peak = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    need = np.minimum(1.0, ceiling / np.maximum(peak.astype(np.float64), 1e-12))
    if need.min() >= 1.0:
        return x
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    reach = max(1, round(0.05 * rate))
    held = minimum_filter1d(need, size=2 * reach + 1, mode="nearest")
    gain = uniform_filter1d(held, size=2 * reach + 1, mode="nearest")
    gain = np.minimum(gain, need)  # rounding in the average must not let a peak through
    y = x * (gain[:, None] if x.ndim == 2 else gain).astype(np.float32)
    return np.clip(y, -ceiling, ceiling).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    """Float samples in -1..1 as int16 (rounded; full scale clips)."""
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype(
        np.int16
    )


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


# ---------------------------------------------------------------- measuring


def _k_weighting(rate: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """The two BS.1770 pre-filters (a high shelf for the head, then a high pass) at ``rate``,
    designed so that at 48 kHz they equal the coefficients printed in the standard."""
    k = math.tan(math.pi * 1681.974450955533 / rate)
    q = 0.7071752369554196
    vh = 10.0 ** (3.999843853973347 / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = (np.array([(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0,
                       (vh - vb * k / q + k * k) / a0]),
             np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]))  # fmt: skip
    k = math.tan(math.pi * 38.13547087602444 / rate)
    q = 0.5003270373238773
    a0 = 1.0 + k / q + k * k
    high = (np.array([1.0, -2.0, 1.0]),
            np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]))  # fmt: skip
    return shelf, high


def block_powers(x: np.ndarray, rate: int, block: float = 0.4, step: float = 0.1) -> np.ndarray:
    """The K-weighted power (sum over channels) of each ``block``-second window, windows
    ``step`` seconds apart (BS.1770 gating blocks: 400 ms, 75% overlap)."""
    from scipy.signal import lfilter

    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    for b, a in _k_weighting(rate):
        y = lfilter(b, a, y, axis=0)
    n, hop = round(block * rate), round(step * rate)
    if len(y) < n:
        return np.zeros(0)
    c = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y * y, axis=0)])
    starts = np.arange(0, len(y) - n + 1, hop)
    return ((c[starts + n] - c[starts]) / n).sum(axis=1)


def _lufs(power: float | np.ndarray) -> float | np.ndarray:
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(power)


def loudness(x: np.ndarray, rate: int) -> float:
    """Integrated loudness in LUFS (BS.1770-4: absolute gate -70 LUFS, relative gate -10 LU).
    ``-inf`` for silence or for sound shorter than one 400 ms block."""
    z = block_powers(x, rate)
    z = z[_lufs(z) > -70.0]
    if not len(z):
        return -math.inf
    z = z[_lufs(z) > _lufs(z.mean()) - 10.0]
    return float(_lufs(z.mean()))


def true_peak(x: np.ndarray, rate: int) -> float:
    """The peak in dBTP: the highest sample after 4x oversampling (BS.1770 annex 2)."""
    y = resample(np.asarray(x, dtype=np.float32).reshape(len(x), -1), rate, 4 * rate)
    p = float(np.abs(y).max()) if len(y) else 0.0
    return 20.0 * math.log10(p) if p > 0 else -math.inf


# ---------------------------------------------------------------- the scene's lanes


def voice_lane(narration: Narration | None, base_dir: str | Path) -> list[Placed]:
    """Each narration block's recorded or synthesized clip, placed at the block's start."""
    if narration is None:
        return []
    out = []
    for b in narration.blocks:
        if b.audio:
            samples, rate = read_audio(files.resolve(base_dir, b.audio))
            out.append(Placed(b.start, samples, rate))
    return out


def mixdown(
    narration: Narration | None, base_dir: str | Path, duration: float
) -> tuple[np.ndarray, int] | None:
    """The scene's whole sound as stereo int16, shape (n, 2), at least ``duration`` seconds
    long, and its rate; None when nothing makes a sound (estimated narration timing)."""
    bus = mix(voice_lane(narration, base_dir), duration)
    if bus is None:
        return None
    x, rate = bus
    return to_int16(master(x, rate)), rate
