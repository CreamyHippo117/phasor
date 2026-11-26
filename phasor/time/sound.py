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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core import files
from phasor.core.schema import Narration, Sound

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


@dataclass(frozen=True, eq=False)
class Lane:
    """One source after its level, fades, ducking and pan: ``kind`` is "voice", "bed" or
    "cue"; ``id`` the block index ("voice.3"), bed or cue id. Samples are at the mix's rate."""

    id: str
    kind: str
    start: float
    samples: np.ndarray
    gain_db: float = 0.0  # the level the mixer chose (already applied to ``samples``)


@dataclass(frozen=True, eq=False)
class Mix:
    """The scene's sound, lane by lane, before the master limiter."""

    rate: int
    duration: float
    lanes: tuple[Lane, ...]
    speech: tuple[tuple[float, float], ...]  # spans the voice speaks (where beds duck)
    problems: tuple[tuple[str, str], ...] = ()  # (bed or cue id, why it is silent)

    def bus(self) -> np.ndarray:
        """All lanes summed, before the limiter."""
        return mix([Placed(x.start, x.samples, self.rate) for x in self.lanes], self.duration)[0]

    def lane(self, lane_id: str) -> Lane | None:
        return next((x for x in self.lanes if x.id == lane_id), None)


def momentary_max(x: np.ndarray, rate: int) -> float:
    """The loudest 400 ms of ``x`` in LUFS (EBU R128 momentary loudness); a sound shorter
    than that is measured as if followed by silence. How loud a short effect seems."""
    n = round(0.4 * rate)
    if len(x) < n:
        x = np.concatenate([np.asarray(x, np.float32).reshape(len(x), -1),
                            np.zeros((n - len(x), np.asarray(x).reshape(len(x), -1).shape[1]),
                                     np.float32)])  # fmt: skip
    z = block_powers(x, rate)
    return float(_lufs(z.max())) if len(z) and z.max() > 0 else -math.inf


@functools.lru_cache(maxsize=32)
def _file_loudness(path: str, mtime: float) -> float:
    x, rate = _read(path, mtime)
    return loudness(x, rate)


def _speech(narration: Narration | None) -> tuple[tuple[float, float], ...]:
    """Where the voice speaks: each sentence with audio, from its first word to its last
    (the whole block when it has no word times)."""
    if narration is None:
        return ()
    out = []
    for b in narration.blocks:
        if not (b.audio and b.text.strip()):
            continue
        if b.words:
            out.append((b.start + b.words[0].start, b.start + b.words[-1].end))
        else:
            out.append((b.start, b.start + b.duration))
    return tuple(out)


def duck_envelope(t: np.ndarray, speech: Sequence[tuple[float, float]], depth_db: float,
                  attack: float, release: float) -> np.ndarray:  # fmt: skip
    """The gain (dB, zero or negative) a ducked bed gets at times ``t``: ``-depth_db`` while a
    span is spoken, ramping down over ``attack`` s before it and back over ``release`` s."""
    amount = np.zeros(len(t))
    for a, b in speech:
        up = np.clip((t - (a - attack)) / attack, 0.0, 1.0) if attack > 0 else (t >= a) * 1.0
        down = np.clip(((b + release) - t) / release, 0.0, 1.0) if release > 0 else (t <= b) * 1.0
        amount = np.maximum(amount, np.minimum(up, down))
    return -depth_db * amount


def _raised(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(n, 1))


def _balance(pan: float) -> np.ndarray:
    """Per-channel gains: centre is both at 1; a pan to one side lowers the other."""
    return np.array([min(1.0, 1.0 - pan), min(1.0, 1.0 + pan)], dtype=np.float32)


def build(
    narration: Narration | None,
    sound: Sound | None,
    base_dir: str | Path,
    duration: float,
    *,
    levels: dict[str, float] | None = None,
    resolve: Callable[[Any], float] | None = None,
    seed: int = 0,
) -> Mix | None:
    """Every lane of the scene's sound (SOUND.md). ``levels`` is the theme's ``[sound]``
    table: with it the voice is set to ``voice`` LUFS, beds and cues are placed relative to
    it and beds duck under speech; without it every source plays as recorded (no ducking).
    ``resolve`` turns a time (seconds or a ``MarkerRef``) into seconds. A bed or cue whose
    file is missing or unreadable is silent and listed in ``problems``. None: no sound."""
    from phasor.time.synth import builtin

    resolve = resolve or (lambda t: float(t))
    voice = voice_lane(narration, base_dir)
    sound = sound or Sound()
    problems: list[tuple[str, str]] = []
    beds, cues = [], []  # (item, samples, rate, source loudness)
    for bed in sound.beds:
        try:
            path = files.resolve(base_dir, bed.file)
            x, r = read_audio(path)
            beds.append((bed, x, r, _file_loudness(str(path), os.path.getmtime(path))))
        except (OSError, ValueError) as e:
            problems.append((bed.id, _why(e, bed.file)))
    for cue in sound.cues:
        try:
            if "." in cue.sound or "/" in cue.sound:
                x, r = read_audio(files.resolve(base_dir, cue.sound))
            else:
                x, r = builtin(cue.sound, cue.id, seed)
            cues.append((cue, _stereo(x), r, None))
        except (OSError, ValueError) as e:
            problems.append((cue.id, _why(e, cue.sound)))
    if not (voice or beds or cues):  # silent: only what went wrong, if anything
        return Mix(RATE, duration, (), (), tuple(problems)) if problems else None
    rate = bus_rate([*voice, *(Placed(0, x, r) for _, x, r, _ in (*beds, *cues))])
    lv = levels or {}
    lanes: list[Lane] = []
    voice_db = 0.0
    if voice:
        clips = [Placed(c.start, resample(_stereo(c.samples), c.rate, rate), rate) for c in voice]
        if levels is not None:  # measured as it will be heard: at the bus rate
            measured = loudness(mix(clips, 0.0)[0], rate)
            voice_db = lv["voice"] - measured if math.isfinite(measured) else 0.0
        g = np.float32(db_to_gain(voice_db))
        for k, c in enumerate(clips):
            s = c.samples if voice_db == 0 else c.samples * g
            lanes.append(Lane(f"voice.{k}", "voice", c.start, s, voice_db))
    speech = _speech(narration) if voice else ()
    ref = lv.get("voice", 0.0)
    for bed, x, r, measured in beds:
        t0 = max(0.0, resolve(bed.start))
        t1 = min(duration, resolve(bed.end)) if bed.end is not None else duration
        n = round((t1 - t0) * rate)
        if n <= 0:
            problems.append((bed.id, f"ends ({t1:g} s) before it starts ({t0:g} s)"))
            continue
        s = resample(x, r, rate)
        s = np.tile(s, (math.ceil(n / len(s)), 1))[:n] if bed.loop else s[:n]
        db = 0.0
        if levels is not None and math.isfinite(measured):
            db = ref + (lv["bed"] if voice else 0.0) + bed.gain - measured
        elif levels is None:
            db = bed.gain
        gain = np.full(len(s), db_to_gain(db))
        fade = lv.get("fade", 0.0)
        fade_in = round((fade if bed.fade_in is None else bed.fade_in) * rate)
        fade_out = round((fade if bed.fade_out is None else bed.fade_out) * rate)
        if fade_in > 0:
            m = min(fade_in, len(s))
            gain[:m] *= _raised(m)
        if fade_out > 0:
            m = min(fade_out, len(s))
            gain[len(s) - m :] *= _raised(m)[::-1]
        if bed.duck and speech and levels is not None:
            grid = np.arange(t0, t1 + 0.01, 0.01)
            env = duck_envelope(grid, speech, lv["duck"], lv["attack"], lv["release"])
            gain *= 10.0 ** (np.interp(t0 + np.arange(len(s)) / rate, grid, env) / 20.0)
        lanes.append(Lane(bed.id, "bed", t0, (s * gain[:, None]).astype(np.float32), db))
    for cue, x, r, _ in cues:
        s = resample(x, r, rate)
        db = cue.gain
        if levels is not None:
            measured = momentary_max(s, rate)
            if math.isfinite(measured):
                db = ref + lv["cue"] + cue.gain - measured
        s = (s * np.float32(db_to_gain(db)) * _balance(cue.pan)).astype(np.float32)
        lanes.append(Lane(cue.id, "cue", resolve(cue.at), s, db))
    return Mix(rate, duration, tuple(lanes), speech, tuple(problems))


def file_versions(narration: Narration | None, sound: Sound | None,
                  base_dir: str | Path) -> tuple:  # fmt: skip
    """The modification time of every file the sound reads (None for a missing one), so a
    cached mix can tell when a file changed."""
    names = [b.audio for b in (narration.blocks if narration else ()) if b.audio]
    if sound is not None:
        names += [b.file for b in sound.beds]
        names += [c.sound for c in sound.cues if "." in c.sound or "/" in c.sound]
    out = []
    for name in names:
        try:
            out.append(os.path.getmtime(files.resolve(base_dir, name)))
        except (OSError, ValueError):
            out.append(None)
    return tuple(out)


def _why(e: Exception, name: str) -> str:
    if isinstance(e, FileNotFoundError):
        return f"no file {name!r} in the scene's folder"
    return str(e)


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
    narration: Narration | None,
    base_dir: str | Path,
    duration: float,
    *,
    sound: Sound | None = None,
    levels: dict[str, float] | None = None,
    resolve: Callable[[Any], float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, int] | None:
    """The scene's whole sound as stereo int16, shape (n, 2), at least ``duration`` seconds
    long, and its rate; None when nothing makes a sound. See ``build`` for the arguments."""
    m = build(narration, sound, base_dir, duration, levels=levels, resolve=resolve, seed=seed)
    if m is None or not m.lanes:
        return None
    ceiling = (levels or {}).get("ceiling", CEILING_DB)
    return to_int16(master(m.bus(), m.rate, ceiling)), m.rate
