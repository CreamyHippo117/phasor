"""Speech and pauses in narration audio, and where markers belong on it (Milestone 22).

Pure NumPy, deterministic, no model: nothing here reads a file or decodes a format. The
caller hands in a reader (``read(skip, count) -> mono float32``), so a long recording is
looked at a chunk at a time and the work is bounded by ``MAX_SECONDS``.

- ``detect``: energy-based voice activity. A DC offset and everything under ``HIGH_PASS``
  (80 Hz: mains hum, rumble) are removed first. Frames of 10 ms; a noise floor that follows
  the recording (the 10th percentile of each 1.5 s window, the median of it and its two
  neighbours); a start threshold and a lower stop threshold above it (hysteresis);
  pauses shorter than ``MIN_PAUSE`` closed, speech shorter than ``MIN_SPEECH`` dropped. It
  says how sure it is (``quality`` 0..1) and why not (``flags``: ``silence``, ``constant_noise``, ``low_snr``,
  ``clipping``, ``music``, ``short``, ``cut``; ``unreliable`` when the pauses it found cannot
  be trusted).
- ``align``: which speech belongs to which sentence of a script, by dynamic programming over
  the detected segments: a sentence's length should be in proportion to its words (the
  speaking rate is measured from the whole recording), sentence breaks should fall on long
  pauses and long pauses should not fall inside a sentence. Not equal spacing.
- ``plan_snap``: the marker moves ``Engine.snap_to_narration`` proposes, as data.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

FRAME = 0.01  # seconds per analysis frame
MIN_SPEECH = 0.08  # speech shorter than this is a click, not a word
MIN_PAUSE = 0.15  # silences shorter than this are inside a word or between close words
MAX_SECONDS = 3 * 3600.0  # the most of a recording that is looked at (the mix's own bound)
_BLOCK = 150  # frames per noise-floor window (1.5 s)
_CHUNK = 1 << 18  # samples read at a time
MIN_RANGE_DB = 6.0  # speech must be at least this much louder than the floor
_SILENT_DB = -70.0  # a recording whose loud parts are quieter than this is silence
_FLOOR_DB = -80.0  # the floor is never taken lower (16-bit dither is about -96 dBFS)
_MAX_DIPS = 20000

Reader = Callable[[int, int], np.ndarray]


@dataclass(frozen=True)
class Speech:
    """What ``detect`` found: speech segments and the pauses around them, in seconds from the
    start of the recording."""

    segments: tuple[tuple[float, float], ...]
    pauses: tuple[tuple[float, float], ...]
    dips: tuple[float, ...]  # quiet moments inside segments: weaker places to cut
    seconds: float  # how much was looked at
    quality: float  # 0..1: how far the pauses can be trusted
    flags: tuple[str, ...]
    floor_db: float  # the typical noise floor (dBFS)
    level_db: float  # the typical loud speech (dBFS)

    @property
    def unreliable(self) -> bool:
        return "unreliable" in self.flags

    def to_dict(self) -> dict:
        return {"segments": [list(s) for s in self.segments],
                "pauses": [list(p) for p in self.pauses], "seconds": self.seconds,
                "quality": self.quality, "flags": list(self.flags),
                "floor_db": self.floor_db, "level_db": self.level_db}  # fmt: skip


def detect_samples(samples: np.ndarray, rate: int) -> Speech:
    """``detect`` on samples in memory (mono, or frames x channels: mixed down)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.dtype.kind != "f" or (x.size and np.abs(x).max() > 1.5):  # int samples
        x = x / 32768.0
    return detect(lambda skip, count: x[skip : skip + count], int(x.shape[0]), rate)


def detect(read: Reader, frames: int, rate: int) -> Speech:
    """Speech and pauses in a mono recording of ``frames`` samples at ``rate`` Hz, read a
    chunk at a time through ``read(skip, count)`` (float32 in -1..1). At most
    ``MAX_SECONDS`` are looked at (flag ``cut`` past that)."""
    rate = int(rate)
    if not 1000 <= rate <= 384000:
        raise ValueError(f"a sample rate of {rate} Hz (1000 to 384000)")
    flags: list[str] = []
    frames = max(0, int(frames))
    if frames > MAX_SECONDS * rate:
        frames = int(MAX_SECONDS * rate)
        flags.append("cut")
    hop = max(1, round(rate * FRAME))
    n = frames // hop
    energy, clipped = _energies(read, n, hop, rate)
    seconds = round(n * hop / rate, 4)
    if n < 10:
        return Speech((), (), (), seconds, 0.0, (*flags, "short", "silence"), -120.0, -120.0)
    db = 10 * np.log10(energy + 1e-12)
    floor = np.maximum(_floor(db), _FLOOR_DB)  # (the filter leaves faint traces in silence)
    level = float(np.percentile(db, 95))
    base = float(np.median(floor))
    span = level - base
    if clipped > 1e-3:
        flags.append("clipping")
    if seconds < 0.5:
        flags.append("short")
    if level < _SILENT_DB:
        flags.append("silence")
        return Speech((), ((0.0, seconds),), (), seconds, 0.0, tuple(flags),
                      round(base, 2), round(level, 2))  # fmt: skip
    if span < MIN_RANGE_DB:
        flags += ["constant_noise", "unreliable"]
        return Speech((), ((0.0, seconds),), (), seconds, 0.0, tuple(flags),
                      round(base, 2), round(level, 2))  # fmt: skip
    on = floor + max(MIN_RANGE_DB, 0.45 * span)
    off = floor + max(0.6 * MIN_RANGE_DB, 0.25 * span)
    runs = _hysteresis(db > on, db > off)
    runs = _close(runs, round(MIN_PAUSE / FRAME))
    runs = [(a, b) for a, b in runs if b - a >= round(MIN_SPEECH / FRAME)]
    to_s = hop / rate
    segments = tuple((round(a * to_s, 4), round(b * to_s, 4)) for a, b in runs)
    pauses = _pauses(segments, seconds)
    dips = _dips(db, runs, to_s)
    quality = float(np.clip((span - MIN_RANGE_DB) / 24.0, 0.0, 1.0))
    if span < 15.0:
        flags.append("low_snr")
    if _music(read, db, runs, n, hop, base, rate):
        flags.append("music")
    if segments and len(pauses) <= 2 and segments[-1][1] - segments[0][0] > 20.0:
        flags.append("no_pauses")  # 20 s of sound without a breath: not speech alone
    if "clipping" in flags:
        quality *= 0.7
    if "short" in flags:
        quality *= 0.5
    if {"music", "no_pauses", "low_snr"} & set(flags):
        quality *= 0.5
        flags.append("unreliable")
    if not segments:
        quality = 0.0
    return Speech(segments, pauses, dips, seconds, round(quality, 3), tuple(flags),
                  round(base, 2), round(level, 2))  # fmt: skip


HIGH_PASS = 80.0  # Hz: what is below is not speech (DC, hum, rumble) and is removed
_PAD = 4096  # samples read on either side of a chunk so its filtered edges are clean


def _highpass(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` without its mean and what lies under ``HIGH_PASS`` (a linear step from 60 to
    100 Hz in the spectrum: DC and 50/60 Hz hum gone). Being zero-phase, it rings for up to
    about 20 ms before a sudden onset in digital silence (never in a real room)."""
    if x.size < 2:
        return x - (x.mean() if x.size else 0.0)
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(x.size, 1.0 / rate)
    spec *= np.clip((f - 0.75 * HIGH_PASS) / (0.5 * HIGH_PASS), 0.0, 1.0)
    return np.fft.irfft(spec, x.size)


def _energies(read: Reader, n: int, hop: int, rate: int) -> tuple[np.ndarray, float]:
    """Mean square per frame (after ``_highpass``) and the share of samples at full scale,
    a chunk at a time (each read with ``_PAD`` samples either side, so the filter has no
    seams)."""
    out = np.zeros(n, np.float64)
    per = max(1, _CHUNK // hop)  # frames per chunk
    clipped = 0
    for f0 in range(0, n, per):
        f1 = min(n, f0 + per)
        a = f0 * hop
        lead = min(_PAD, a)
        x = np.asarray(read(a - lead, (f1 - f0) * hop + lead + _PAD), dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        core = x[lead : lead + (f1 - f0) * hop]
        got = len(core) // hop
        if got == 0:
            break
        clipped += int(np.count_nonzero(np.abs(core[: got * hop]) >= 0.999))
        y = _highpass(x, rate)[lead : lead + got * hop]
        out[f0 : f0 + got] = (y.reshape(got, hop) ** 2).mean(axis=1)
    return out, clipped / max(1, n * hop)


def _floor(db: np.ndarray) -> np.ndarray:
    """The noise floor at each frame: the 10th percentile of each window, then the median of
    it and its two neighbours (one window that is all speech does not lift the floor; a
    change of room noise is followed within about two windows), spread over the frames
    between window centres."""
    n = len(db)
    k = max(1, math.ceil(n / _BLOCK))
    pad = np.full(k * _BLOCK, np.nan)
    pad[:n] = db
    per = np.nanpercentile(pad.reshape(k, _BLOCK), 10, axis=1)
    edge = np.concatenate([per[:1], per, per[-1:]])
    low = np.median(np.stack([edge[:-2], edge[1:-1], edge[2:]]), axis=0)
    centres = (np.arange(k) + 0.5) * _BLOCK
    return np.interp(np.arange(n), centres, low)


def _hysteresis(start: np.ndarray, stay: np.ndarray) -> list[tuple[int, int]]:
    """Runs of ``stay`` frames that hold at least one ``start`` frame, as [a, b)."""
    edges = np.diff(np.concatenate([[0], stay.astype(np.int8), [0]]))
    a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if a.size == 0:
        return []
    cum = np.concatenate([[0], np.cumsum(start.astype(np.int64))])
    hits = cum[b] - cum[a]
    return [(int(x), int(y)) for x, y, h in zip(a, b, hits, strict=True) if h > 0]


def _close(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in runs:
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _pauses(segments: Sequence[tuple[float, float]], seconds: float):
    edges = [0.0, *(v for s in segments for v in s), seconds]
    return tuple((round(a, 4), round(b, 4)) for a, b in zip(edges[::2], edges[1::2], strict=True)
                 if b - a > 1e-9)  # fmt: skip


def _dips(db: np.ndarray, runs: list[tuple[int, int]], to_s: float) -> tuple[float, ...]:
    """Quiet moments inside speech (6 dB under the loudest frames near them): where a
    sentence break may fall when the speaker did not pause."""
    if not runs:
        return ()
    k = 5
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    out: list[float] = []
    for a, b in runs:
        if b - a < 30:
            continue
        s = smooth[a:b]
        inner = np.flatnonzero((s[1:-1] < s[:-2]) & (s[1:-1] <= s[2:])) + 1
        for i in inner:
            lo, hi = max(0, i - 15), min(len(s), i + 16)
            if i < 8 or i > len(s) - 8:
                continue
            if min(s[lo:i].max(), s[i + 1 : hi].max()) - s[i] >= 6.0:
                out.append(round((a + i) * to_s, 4))
                if len(out) >= _MAX_DIPS:
                    return tuple(out)
    return tuple(out)


def _music(read: Reader, db: np.ndarray, runs, n: int, hop: int, floor: float, rate: int) -> bool:
    """Something tonal and loud under the speech: the quiet frames between words are not
    quiet (well above digital silence) and their spectrum is peaky (a flatness under 0.2),
    as music is and room noise is not. Looks at up to 64 such frames."""
    if floor < -55.0:
        return False
    quiet = np.ones(n, bool)
    for a, b in runs:
        quiet[a:b] = False
    idx = np.flatnonzero(quiet & (db < floor + 3.0))
    if idx.size < 8:
        return False
    pick = idx[np.linspace(0, idx.size - 1, min(64, idx.size)).astype(int)]
    size = 1 << max(6, math.ceil(math.log2(hop * 4)))
    flat = []
    for f in pick:
        x = np.asarray(read(int(f) * hop, size), dtype=np.float64)
        if x.size < size // 2:
            continue
        spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(x.size))) ** 2
        spec = spec[np.fft.rfftfreq(x.size, 1.0 / rate) >= 1.25 * HIGH_PASS] + 1e-20
        flat.append(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    return bool(flat) and float(np.median(flat)) < 0.2


# ------------------------------------------------------------------ sentences


@dataclass(frozen=True)
class Placed:
    """Where one sentence was found: seconds in the recording, and how sure (0..1)."""

    start: float
    end: float
    confidence: float
    onsets: tuple[float, ...] = ()  # speech onsets inside it (after a pause): piece breaks


MAX_SENTENCES = 2000


def text_length(text: str) -> float:
    """How long a sentence's speech should be, in words (the speaking rate is measured from
    the recording): pauses at punctuation are silence, not speech, so they do not count."""
    return float(max(len(text.split()), 1))


_W_DUR = 2.0  # weight of a sentence's length against its text
_W_BREAK = 1.0  # weight of a sentence break on a short gap
_W_INNER = 1.0  # weight of a long gap inside a sentence
_DIP = 1.0  # extra cost of a sentence break where the speaker did not pause
_DOUBT = math.log(1.8)  # a sentence this many times longer or shorter than its text: doubtful


class Alignment(list):
    """``align``'s sentences (a list of ``Placed``) and how well the text fits them:
    ``rate`` (seconds of speech per unit of text), ``misfit`` (the median of
    ``|log(said / expected)|`` over sentences) and ``worst`` (its largest), ``doubtful``
    (some sentence is ``_DOUBT`` or more off, or the median misfit is large: a sentence
    break was probably put in the wrong place)."""

    rate: float = 0.0
    misfit: float = 0.0
    worst: float = 0.0
    doubtful: bool = False

    def to_dict(self) -> dict:
        return {"rate": round(self.rate, 4), "misfit": round(self.misfit, 4),
                "worst": round(self.worst, 4), "doubtful": self.doubtful}  # fmt: skip


def align(speech: Speech, expected: Sequence[float], pause_before: Sequence[float] = ()
          ) -> Alignment:  # fmt: skip
    """Sentence ``i`` (``expected[i]``: its length from its text, any unit) to the speech that
    says it. ``pause_before[i]``: a pause the script asks for before sentence ``i`` (it
    should fall on a long silence). Every detected segment belongs to some sentence (no
    speech is left out). ``ValueError`` when there is no speech or too many sentences."""
    n = len(expected)
    if n == 0:
        return Alignment()
    if n > MAX_SENTENCES:
        raise ValueError(f"{n} sentences: at most {MAX_SENTENCES} are aligned to a recording")
    segs = [list(s) for s in speech.segments]
    if not segs:
        raise ValueError("no speech was found in the recording")
    exp = np.maximum(np.asarray(expected, np.float64), 1e-3)
    rate0 = sum(b - a for a, b in segs) / float(exp.sum())
    segs = _split_long(segs, speech.dips, 0.8 * rate0 * float(np.median(exp)))
    segs = _split(segs, speech.dips, n)
    segs = _merge(segs, max(64, 8 * n))
    m = len(segs)
    starts = np.array([s[0] for s in segs])
    ends = np.array([s[1] for s in segs])
    lens = ends - starts
    gaps = np.concatenate([[0.0], starts[1:] - ends[:-1]])  # gaps[k]: before segment k
    real = gaps[1:][gaps[1:] > 0]
    med = float(np.median(real)) if real.size else 0.1
    z = gaps / (gaps + max(med, 0.02))  # 0..1: how much of a break the gap before k is
    z[0] = 1.0
    rate = float(lens.sum() / exp.sum())  # seconds of speech per unit of text
    G = np.concatenate([[0.0], np.cumsum(lens)])  # speech before segment k
    Z2 = np.concatenate([[0.0], np.cumsum(z[1:] ** 2)])  # Z2[k]: sum of z^2 for gaps 1..k
    z = np.where(gaps > 0, z, -_DIP)  # a cut at a quiet dip (no pause): a worse break still
    pb = list(pause_before) + [0.0] * (n - len(pause_before))
    # the most segments a sentence may span: enough for the longest sentence three times over
    w = min(m - n + 1, max(48, math.ceil(3 * m * float(exp.max() / exp.sum())) + 16))
    inf = np.inf
    P = np.full(m + 1, inf)  # P[k]: the best cost with the sentences so far on segments 0..k-1
    P[0] = 0.0
    back = np.zeros((n, m + 1), np.int32)
    ks = np.arange(m + 1)
    per = []  # what does not depend on the sentence, per width (segments a..k-1, every end k)
    for width in range(1, w + 1):
        a = ks - width
        ok = a >= 0
        aa = np.where(ok, a, 0)
        first = np.minimum(aa, m - 1)
        brk = np.where(aa > 0, 1.0 - z[first], 0.0)
        inner = _W_INNER * (Z2[np.maximum(ks - 1, 0)] - Z2[aa])  # gaps aa+1 .. k-1
        per.append((width, ok, aa, np.log(G[ks] - G[aa] + 0.1), brk, inner, gaps[first]))
    for i in range(n):
        N = np.full(m + 1, inf)
        lw = math.log(rate * exp[i] + 0.1)
        for width, ok, aa, lsaid, brk, inner, g in per:
            cost = _W_DUR * (lsaid - lw) ** 2 + inner
            if i:
                cost += _W_BREAK * brk
            if pb[i] > 0:
                cost += np.where(aa > 0, np.clip(1.0 - g / (0.5 * pb[i]), 0, 1), 0.0)
            total = np.where(ok, P[aa] + cost, inf)
            better = total < N
            N = np.where(better, total, N)
            back[i][better] = width
        P = N
    k = m
    if not np.isfinite(P[k]):
        raise ValueError("the recording could not be split into the script's sentences")
    out = Alignment()
    misfits = []
    for i in range(n - 1, -1, -1):
        width = int(back[i, k])
        a = k - width
        said = G[k] - G[a]
        off = abs(math.log((said + 0.1) / (rate * exp[i] + 0.1)))
        misfits.append(off)
        fit = math.exp(-(off**2))
        brk = 1.0 if a == 0 or i == 0 else float(z[a])
        onsets = tuple(float(starts[j]) for j in range(a + 1, k) if gaps[j] > 0)
        out.append(Placed(float(starts[a]), float(ends[k - 1]),
                          round(speech.quality * fit * brk, 3), onsets))  # fmt: skip
        k = a
    out.reverse()
    out.rate = rate
    out.misfit = float(np.median(misfits))
    out.worst = float(max(misfits))
    breaks = [float(gaps[j]) for j in _starts_of(back, n, m)[1:]]
    chosen = set(_starts_of(back, n, m))
    inner = [float(gaps[j]) for j in range(1, m) if j not in chosen and gaps[j] > 0]
    out.breaks = breaks
    out.inner = inner
    # sentence breaks no longer than the pauses inside sentences: the pauses say nothing
    blind = (
        bool(breaks)
        and bool(inner)
        and float(np.median(breaks)) < 1.5 * float(np.percentile(inner, 75))
    )
    out.doubtful = bool(out.worst >= _DOUBT or out.misfit >= math.log(1.3) or blind)
    if out.doubtful:  # say it in every sentence's confidence too
        for j, p in enumerate(out):
            out[j] = Placed(p.start, p.end, round(min(p.confidence, 0.3), 3), p.onsets)
    return out


def _starts_of(back: np.ndarray, n: int, m: int) -> list[int]:
    """The first segment of each sentence, from the alignment's back pointers."""
    k, out = m, []
    for i in range(n - 1, -1, -1):
        k -= int(back[i, k])
        out.append(k)
    return out[::-1]


def _split_long(segs: list[list[float]], dips: Sequence[float], longest: float
                ) -> list[list[float]]:  # fmt: skip
    """Segments longer than ``longest`` seconds (the shortest sentence could end inside
    them: a reader who does not pause) cut at every quiet dip inside them, so a sentence break
    can fall there."""
    dips = np.sort(np.asarray(dips, np.float64))
    out: list[list[float]] = []
    for a, b in segs:
        if b - a <= longest or dips.size == 0:
            out.append([a, b])
            continue
        inside = dips[(dips > a + 0.05) & (dips < b - 0.05)]
        edges = [a, *inside.tolist(), b]
        out.extend([x, y] for x, y in itertools.pairwise(edges))
    return out


def _split(segs: list[list[float]], dips: Sequence[float], n: int) -> list[list[float]]:
    """At least ``n`` segments: split the longest at the quiet moment nearest its middle (its
    middle when it has none) until there are."""
    segs = [s[:] for s in segs]
    dips = sorted(dips)
    while len(segs) < n:
        k = max(range(len(segs)), key=lambda j: segs[j][1] - segs[j][0])
        a, b = segs[k]
        mid = (a + b) / 2
        inside = [d for d in dips if a + 0.05 < d < b - 0.05]
        cut = min(inside, key=lambda d: abs(d - mid)) if inside else mid
        if b - a < 0.02:
            break
        segs[k : k + 1] = [[a, cut], [cut, b]]
    return segs


def _merge(segs: list[list[float]], cap: int) -> list[list[float]]:
    """At most ``cap`` segments: the shortest gaps are closed first."""
    if len(segs) <= cap:
        return segs
    gaps = np.array([segs[k][0] - segs[k - 1][1] for k in range(1, len(segs))])
    keep = np.sort(np.argsort(-gaps, kind="stable")[: cap - 1]) + 1  # segment starts kept
    out = []
    first = 0
    for k in [*keep.tolist(), len(segs)]:
        out.append([segs[first][0], segs[k - 1][1]])
        first = k
    return out


def phrases(speech: Speech, pause: float, cap: int = MAX_SENTENCES) -> list[Placed]:
    """Without a script: the speech cut at every pause of at least ``pause`` seconds."""
    out: list[list[float]] = []
    for a, b in speech.segments:
        if out and a - out[-1][1] < pause:
            out[-1][1] = b
        else:
            out.append([a, b])
    out = _merge(out, cap)
    return [Placed(a, b, speech.quality) for a, b in out]


def pieces(placed: Placed, weights: Sequence[float]) -> list[float]:
    """Where each piece of a sentence starts (seconds in the recording): in proportion to its
    text over the sentence, each break moved onto a speech onset inside the sentence when one
    is within a quarter of the sentence's length."""
    w = np.maximum(np.asarray(weights, np.float64), 1e-3)
    span = placed.end - placed.start
    cuts = placed.start + span * np.cumsum(w)[:-1] / w.sum()
    out = [placed.start]
    free = list(placed.onsets)
    for c in cuts:
        near = [o for o in free if abs(o - c) <= span / 4 and o > out[-1]]
        t = min(near, key=lambda o: abs(o - c)) if near else float(c)
        t = max(t, out[-1])
        if near:
            free.remove(t)
        out.append(t)
    return out


# ------------------------------------------------------------------ snapping markers


@dataclass(frozen=True)
class Boundary:
    """A place on the timeline a marker may snap to."""

    t: float
    kind: str  # "sentence" (a sentence starts), "phrase" (speech resumes after a pause)
    label: str
    confidence: float = 1.0
    index: int | None = None  # the narration block it starts


@dataclass(frozen=True)
class Move:
    marker: str
    t0: float  # where it is
    t1: float  # where it would go
    why: str
    confidence: float
    kind: str  # "sentence", "phrase", "between", "owned", "kept"

    @property
    def moves(self) -> bool:
        return abs(self.t1 - self.t0) > 5e-5

    def to_dict(self) -> dict:
        return {"marker": self.marker, "from": self.t0, "to": self.t1, "why": self.why,
                "confidence": self.confidence, "kind": self.kind}  # fmt: skip


_PHRASE_COST = 0.25  # of the reach: a phrase start is a worse place than a sentence start
_PASSES = 6


class SnapPlan(list):
    """``plan_snap``'s moves (a list of ``Move``), with ``fit``: how the scene's clock was read
    against the narration's (``scale``, ``offset``: the narration time of a marker at ``t``
    is about ``scale * t + offset``), how well it fits (``residual``: the median seconds
    from a predicted place to its sentence start), the next best reading's cost
    (``margin``: how much worse it is), and ``ambiguous`` (two readings fit almost equally
    well: nothing was moved)."""

    fit: dict


def plan_snap(
    hand: Mapping[str, float],
    owned: Mapping[str, float],
    boundaries: Sequence[Boundary],
    *,
    visual_end: float,
    narration_end: float,
    reach: float,
    need: Mapping[str, float],
    anchors: Mapping[str, Boundary] | None = None,
    allow_short: bool = False,
) -> SnapPlan:
    """Where each hand-placed marker goes (one ``Move`` per marker of ``hand`` and ``owned``).

    First the scene's clock is read against the narration's: a scale and an offset (the scene
    may have been laid out a little slower or faster than the voice, or with a longer intro)
    are fitted so that as many markers as possible fall on sentence starts (``fit_clock``:
    a vote over offsets for each scale, scored on every marker, a mild preference for no
    change). When two different readings fit almost equally well, the plan is *ambiguous*:
    nothing moves and ``fit["ambiguous"]`` says so (pin markers with ``anchors``).

    Then ``owned`` markers (the narration's) stay and bound the others; ``anchors`` pin a
    marker to a boundary; the rest are matched in order (never two to one, never out of
    order) by dynamic programming on the distance from where the fitted clock puts them, at
    most ``reach`` of the narration's length and two sentence-spacing tolerances. Markers left
    unmatched keep their place between their neighbours, stretched with the span around them.
    No span after a marker becomes shorter than ``need[marker]`` unless ``allow_short``: the
    least sure match that would do so is dropped. Planning is repeated on its own result until
    nothing changes, so a second snap moves nothing. ``visual_end`` is kept for callers; the
    fit does not need it."""
    del visual_end
    nend = max(float(narration_end), 1e-3)
    fit = fit_clock(hand, boundaries, nend, set(anchors or {}))
    current = dict(hand)
    moves: list[Move] = []
    rival = fit.pop("rival", None)
    if rival is not None:  # a close second reading: ambiguous only if it moves markers elsewhere
        one, c1 = _plan_once(current, owned, boundaries, (fit["scale"], fit["offset"]), nend,
                             reach, fit["tolerance"], need, anchors or {}, allow_short)  # fmt: skip
        two, c2 = _plan_once(current, owned, boundaries, rival, nend, reach, fit["tolerance"],
                             need, anchors or {}, allow_short)  # fmt: skip
        differ = any(abs(x.t1 - y.t1) > 1e-4 for x, y in zip(one, two, strict=True))
        # each reading's fit plus how well its chain of steps holds, per marker
        k = max(1, len(hand))
        first, second = c1 / k, fit["margin"] + c2 / k
        if differ and second < first - _AMBIGUOUS:
            fit["scale"], fit["offset"] = round(rival[0], 4), round(rival[1], 4)
        fit["ambiguous"] = differ and abs(first - second) <= _AMBIGUOUS
    if not fit["ambiguous"]:
        clock = (fit["scale"], fit["offset"])
        for _ in range(_PASSES):
            moves, _ = _plan_once(current, owned, boundaries, clock, nend, reach,
                                  fit["tolerance"], need, anchors or {}, allow_short)  # fmt: skip
            new = {m.marker: m.t1 for m in moves}
            if all(abs(new[k] - current[k]) <= 5e-5 for k in current):
                break
            current = new
            clock = (1.0, 0.0)  # the markers are on the voice's clock now
    last = {m.marker: m for m in moves}
    out = SnapPlan()
    out.fit = fit
    for name, t0 in sorted({**hand, **owned}.items(), key=lambda kv: (kv[1], kv[0])):
        if name in owned:
            out.append(Move(name, t0, t0, "owned by the narration (a [[marker]] in its "
                            "script): it is already on the words", 1.0, "owned"))  # fmt: skip
            continue
        if fit["ambiguous"]:
            out.append(Move(name, t0, t0, "kept: the scene's timing fits the narration in more "
                            "than one way (pin markers with anchors)", 0.0, "kept"))  # fmt: skip
            continue
        m, t1 = last[name], round(current[name], 4)
        if abs(t1 - t0) <= 5e-5 and m.kind == "between":
            out.append(Move(name, t0, t0, "kept: no sentence or pause near it", m.confidence,
                            "kept"))  # fmt: skip
        else:
            out.append(Move(name, t0, t1 if abs(t1 - t0) > 5e-5 else t0, m.why, m.confidence,
                            m.kind))  # fmt: skip
    return out


_SCALES = (0.6, 1.6)  # the scene's clock against the voice's: slowest and fastest considered
_PRIOR = 0.1  # weight of the preference for the scene's own clock (scale 1, offset 0)
_AMBIGUOUS = 0.08  # a different reading within this much of the best cost: ambiguous


def fit_clock(hand: Mapping[str, float], boundaries: Sequence[Boundary], narration_end: float,
              pinned: set[str] = frozenset()) -> dict:  # fmt: skip
    """The scale and offset that put the hand markers on sentence starts best.

    Cost of a reading (a, b): the mean over markers of ``min(d, tol) / tol``, ``d`` the
    distance from ``a * t + b`` to the nearest sentence start and ``tol`` a third of the
    typical spacing of sentence starts (a marker that belongs to no sentence costs 1
    whatever the reading), plus ``_PRIOR * (((a - 1) / 0.2)^2 + (b / (0.2 * length))^2)``.
    For each scale on a grid fine enough that the last marker moves less than half a
    tolerance between steps, the offset is the one most markers vote for. The best reading
    wins; a reading that puts some well-placed marker on a *different* sentence and costs
    less than ``_AMBIGUOUS`` more makes the answer ambiguous."""
    names = [k for k in sorted(hand, key=lambda k: (hand[k], k)) if k not in pinned]
    h = np.array([hand[k] for k in names], np.float64)
    c = np.array(sorted(b.t for b in boundaries if b.kind == "sentence")
                 or sorted(b.t for b in boundaries), np.float64)  # fmt: skip
    spacing = float(np.median(np.diff(c))) if len(c) > 1 else max(narration_end, 1.0)
    tol = float(np.clip(spacing / 3, 0.15, 3.0))
    none = {"scale": 1.0, "offset": 0.0, "residual": None, "tolerance": tol, "margin": None,
            "ambiguous": False, "matched": 0, "markers": len(h)}  # fmt: skip
    if len(h) == 0 or len(c) == 0:
        return none
    span = max(narration_end, float(h.max()), 1.0)

    def assign(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        p = a * h + b
        j = np.clip(np.searchsorted(c, p), 1, len(c) - 1) if len(c) > 1 else np.zeros(len(p), int)
        if len(c) > 1:
            j = np.where(np.abs(c[j - 1] - p) <= np.abs(c[j] - p), j - 1, j)
        return j, np.abs(c[j] - p)

    def cost(a: float, b: float) -> float:
        _, d = assign(a, b)
        prior = ((a - 1) / 0.3) ** 2 + (b / (0.2 * span)) ** 2
        # few markers say little about the clock: the preference for no change weighs more
        return float(np.mean(np.minimum(d, tol) / tol)) + _PRIOR * max(1.0, 6 / len(h)) * prior

    hs = h if len(h) <= 64 else h[np.linspace(0, len(h) - 1, 64).astype(int)]
    top = max(float(np.abs(hs).max()), tol)
    step = max(tol / (2 * top), (_SCALES[1] - _SCALES[0]) / 800)
    bmax = 0.25 * span + tol
    readings = {(1.0, 0.0)}
    for a in np.arange(_SCALES[0], _SCALES[1] + 1e-9, step):
        offs = (c[None, :] - a * hs[:, None]).ravel()
        offs = np.sort(offs[np.abs(offs) <= bmax])
        if offs.size == 0:
            continue
        ends = np.searchsorted(offs, offs + tol, side="right")
        votes = ends - np.arange(offs.size)
        best_k = np.flatnonzero(votes == votes.max())  # every offset most markers agree on
        for k in best_k[np.linspace(0, best_k.size - 1, min(8, best_k.size)).astype(int)]:
            readings.add((round(float(a), 6), round(float(np.median(offs[k : ends[k]])), 6)))
    scored = sorted((cost(a, b), a, b) for a, b in readings)
    best, a, b = scored[0]
    j, d = assign(a, b)
    good = d <= tol / 2
    margin, rival = None, None
    for other, a2, b2 in scored[1:]:
        j2, d2 = assign(a2, b2)
        if np.any(good & (d2 <= tol / 2) & (j2 != j)):  # a well-placed marker, another sentence
            margin, rival = other - best, (a2, b2)
            break
    return {"scale": round(a, 4), "offset": round(b, 4),
            "residual": round(float(np.median(d)), 4), "tolerance": round(tol, 4),
            "margin": None if margin is None else round(margin, 4),
            "ambiguous": bool(margin is not None and margin < _AMBIGUOUS),
            "rival": rival if margin is not None and margin < _AMBIGUOUS else None,
            "matched": int(good.sum()), "markers": len(h)}  # fmt: skip


def _plan_once(hand, owned, boundaries, clock, nend, reach, tol, need, anchors,
               allow_short):  # fmt: skip
    names = sorted(hand, key=lambda k: (hand[k], k))
    h = np.array([hand[k] for k in names], np.float64)
    bounds = sorted(boundaries, key=lambda b: b.t)
    c = np.array([b.t for b in bounds], np.float64)
    fixed = sorted(owned.items(), key=lambda kv: kv[1])
    # each hand marker must stay strictly between the owned markers around it
    lo = np.full(len(h), -np.inf)
    hi = np.full(len(h), np.inf)
    for _, t in fixed:
        lo = np.where(h > t, np.maximum(lo, t), lo)
        hi = np.where(h < t, np.minimum(hi, t), hi)
    for k, b in anchors.items():  # a pinned marker bounds the others as an owned one does
        if k in hand:
            lo = np.where(h > hand[k], np.maximum(lo, b.t), lo)
            hi = np.where(h < hand[k], np.minimum(hi, b.t), hi)
    match: dict[int, int] = {}
    chained = 0.0
    pinned = {i for i, k in enumerate(names) if k in anchors}
    scale, offset = clock
    window = max(3 * tol, 0.08 * nend)  # how far from where the clock puts it a match may be
    if len(c) and len(h):
        kinds = np.array([b.kind == "phrase" for b in bounds])
        match, chained = _chain(h, c, kinds, scale, offset, tol, window, lo, hi, pinned,
                       {i: anchors[k].t for i, k in enumerate(names) if k in anchors})  # fmt: skip
    detail: dict[str, Move] = {}
    anchor_pts: list[tuple[float, float, str, float]] = []  # (old, new, name, confidence)
    for i, k in enumerate(names):
        if k in anchors:
            b = anchors[k]
            detail[k] = Move(k, h[i], b.t, f"pinned to {b.label}", b.confidence, b.kind)
            anchor_pts.append((h[i], b.t, k, 2.0))
        elif i in match:
            b = bounds[match[i]]
            off = min(abs(scale * h[i] + offset - b.t) / window, 1.0)
            conf = round(b.confidence * (1.0 - 0.5 * float(off)), 3)
            detail[k] = Move(k, h[i], b.t, f"{b.label}", conf, b.kind)
            anchor_pts.append((h[i], b.t, k, conf))
    while True:
        new = _interpolate(names, h, anchor_pts, fixed)
        drops = _violation(names, h, new, fixed, need, allow_short, anchor_pts)
        if not drops:
            break
        for worst in {min(d, key=lambda p: (p[3], p[2])) for d in drops}:
            anchor_pts.remove(worst)
            detail.pop(worst[2], None)
    out = []
    for i, k in enumerate(names):
        if k in detail:
            out.append(detail[k])
        else:
            near = [p[3] for p in anchor_pts if p[3] < 2.0]
            conf = round(0.5 * min(near), 3) if near else 0.0
            out.append(Move(k, h[i], new[k], "kept its place between its neighbours, "
                            "stretched with the span around it", conf, "between"))  # fmt: skip
    return out, chained


_UNARY = 0.3  # weight of the distance from where the fitted clock puts a marker
_KEEP = 1.0  # cost of leaving a marker unmatched
_CANDIDATES = 32  # sentence starts considered per marker (the nearest)


def _chain(h, c, phrase, scale, offset, tol, window, lo, hi, pinned, pins):
    """Which boundary each marker goes to (``{marker index: boundary index}``, and the path's
    cost), by Viterbi over the markers in order. A marker's candidates are the boundaries
    within ``window`` of where the fitted clock puts it (and between the owned markers
    around it); staying unmatched costs ``_KEEP``. What decides is the *spacing*: going from
    one marker to the next, the step between their boundaries should be the scene's step
    times ``scale`` (truncated at ``tol``), so a scene whose pace drifts against the voice
    (slower in its second half, a longer intro) still lands each marker on its own sentence;
    the distance from the fitted clock counts only a little (``_UNARY``). Boundaries
    strictly increase from one marker to the next; a pinned marker has its one boundary."""
    k = len(h)
    states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (times, indices, unary)
    for i in range(k):
        if i in pinned:
            states.append((np.array([pins[i]]), np.array([-2]), np.zeros(1)))
            continue
        p = scale * h[i] + offset
        j = np.flatnonzero((np.abs(c - p) <= window) & (c > lo[i]) & (c < hi[i]))
        if j.size > _CANDIDATES:
            j = j[np.argsort(np.abs(c[j] - p), kind="stable")[:_CANDIDATES]]
            j.sort()
        u = _UNARY * np.abs(c[j] - p) / window + np.where(phrase[j], _PHRASE_COST, 0.0)
        states.append((np.concatenate([c[j], [np.nan]]), np.concatenate([j, [-1]]),
                       np.concatenate([u, [_KEEP]])))  # fmt: skip
    cost = states[0][2].copy()
    back: list[np.ndarray] = []
    for i in range(1, k):
        t0, _, _ = states[i - 1]
        t1, _, u1 = states[i]
        want = scale * (h[i] - h[i - 1])
        step = t1[None, :] - t0[:, None]
        trans = np.minimum(np.abs(step - want), tol) / tol
        trans = np.where(np.isnan(step), 0.5, np.where(step <= 0, np.inf, trans))
        total = cost[:, None] + trans
        arg = np.argmin(total, axis=0)
        back.append(arg)
        cost = total[arg, np.arange(len(t1))] + u1
    s = int(np.argmin(cost))
    total_cost = float(cost[s])
    out: dict[int, int] = {}
    for i in range(k - 1, -1, -1):
        j = int(states[i][1][s])
        if j >= 0:
            out[i] = j
        if i:
            s = int(back[i - 1][s])
    return out, total_cost


def _interpolate(names, h, anchor_pts, fixed) -> dict[str, float]:
    """Every hand marker's new time: anchors where they go; the rest piecewise linear between
    the anchors and owned markers around them (0 stays 0; after the last, shifted with it)."""
    pts = sorted({(float(o), float(n)) for o, n, *_ in anchor_pts} |
                 {(float(t), float(t)) for _, t in fixed})  # fmt: skip
    t = np.asarray(h, np.float64)
    if not pts:
        new = t.copy()
    else:
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if xs[0] > 0.0:
            xs, ys = np.concatenate([[0.0], xs]), np.concatenate([[0.0], ys])
        new = np.interp(t, xs, ys)
        new = np.where(t >= xs[-1], t + ys[-1] - xs[-1], new)
        new = np.where(t < xs[0], t + ys[0] - xs[0], new)
    at = {p[2]: p[1] for p in anchor_pts}
    return {k: float(at[k]) if k in at else float(v) for k, v in zip(names, new, strict=True)}


def _violation(names, h, new, fixed, need, allow_short, anchor_pts) -> list[list]:
    """For each span that runs backwards or (unless ``allow_short``) became shorter than the
    motions anchored at its first marker need (and than it was): the matched anchors that
    place its two ends (the end itself, or the nearest anchor beyond it; an owned or pinned
    one cannot be dropped). Spans run between markers that start a scene: hand-placed ones,
    and owned ones that motions are anchored to (``need`` names them); an owned marker that
    is only a word bounds the order alone."""
    seq = sorted([(float(h[i]), k, new[k]) for i, k in enumerate(names)] +
                 [(t, k, t) for k, t in fixed])  # fmt: skip
    hand = set(names)
    index = {x: i for i, x in enumerate(seq)}
    starts = [x for x in seq if x[1] in hand or x[1] in need]
    placed = {p[2]: p for p in anchor_pts}
    owned = {k for k, _ in fixed}

    def placer(i: int, step: int):
        while 0 <= i < len(seq):
            name = seq[i][1]
            if name in owned:
                return None
            if name in placed:
                return placed[name] if placed[name][3] < 2.0 else None
            i += step
        return None

    bad = [(x, y) for x, y in itertools.pairwise(seq) if y[2] < x[2] - 1e-9]
    if not allow_short:
        bad += [(x, y) for x, y in itertools.pairwise(starts)
                if y[2] - x[2] < min(y[0] - x[0], need.get(x[1], 0.0)) - 1e-6]  # fmt: skip
    out = []
    for x, y in bad:
        ends = [p for p in (placer(index[x], -1), placer(index[y], 1)) if p is not None]
        if ends:
            out.append(ends)
    return out
