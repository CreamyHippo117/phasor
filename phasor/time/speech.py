"""Speech and pauses in narration audio, and where markers belong on it (Milestone 22).

Pure NumPy, deterministic, no model: nothing here reads a file or decodes a format. The
caller hands in a reader (``read(skip, count) -> mono float32``), so a long recording is
looked at a chunk at a time and the work is bounded by ``MAX_SECONDS``.

- ``detect``: energy-based voice activity. Frames of 10 ms; a noise floor that follows the
  recording (the 10th percentile of each 2.5 s window, the lowest of its neighbours); a
  start threshold and a lower stop threshold above it (hysteresis); pauses shorter than
  ``MIN_PAUSE`` closed, speech shorter than ``MIN_SPEECH`` dropped. It says how sure it is
  (``quality`` 0..1) and why not (``flags``: ``silence``, ``constant_noise``, ``low_snr``,
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
    energy, clipped = _energies(read, n, hop)
    seconds = round(n * hop / rate, 4)
    if n < 10:
        return Speech((), (), (), seconds, 0.0, (*flags, "short", "silence"), -120.0, -120.0)
    db = 10 * np.log10(energy + 1e-12)
    floor = _floor(db)
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
    if _music(read, db, runs, n, hop, base):
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


def _energies(read: Reader, n: int, hop: int) -> tuple[np.ndarray, float]:
    """Mean square per frame and the share of samples at full scale, a chunk at a time."""
    out = np.zeros(n, np.float64)
    per = max(1, _CHUNK // hop)  # frames per chunk
    clipped = 0
    for f0 in range(0, n, per):
        f1 = min(n, f0 + per)
        x = np.asarray(read(f0 * hop, (f1 - f0) * hop), dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        got = len(x) // hop
        if got == 0:
            break
        x = x[: got * hop].astype(np.float64)
        clipped += int(np.count_nonzero(np.abs(x) >= 0.999))
        out[f0 : f0 + got] = (x.reshape(got, hop) ** 2).mean(axis=1)
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


def _music(read: Reader, db: np.ndarray, runs, n: int, hop: int, floor: float) -> bool:
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
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2 + 1e-20
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
_MAX_WIDTH = 48  # segments one sentence may span
_SKIP = 2.0  # cost of leaving a segment at either end out (a breath, a click)


def align(speech: Speech, expected: Sequence[float], pause_before: Sequence[float] = ()
          ) -> list[Placed]:  # fmt: skip
    """Sentence ``i`` (``expected[i]``: its length from its text, any unit) to the speech that
    says it. ``pause_before[i]``: a pause the script asks for before sentence ``i`` (it
    should fall on a long silence). ``ValueError`` when there is no speech or too many
    sentences."""
    n = len(expected)
    if n == 0:
        return []
    if n > MAX_SENTENCES:
        raise ValueError(f"{n} sentences: at most {MAX_SENTENCES} are aligned to a recording")
    segs = [list(s) for s in speech.segments]
    if not segs:
        raise ValueError("no speech was found in the recording")
    segs = _split(segs, speech.dips, n)
    segs = _merge(segs, max(64, 8 * n))
    m = len(segs)
    starts = np.array([s[0] for s in segs])
    ends = np.array([s[1] for s in segs])
    lens = ends - starts
    gaps = np.concatenate([[0.0], starts[1:] - ends[:-1]])  # gaps[k]: before segment k
    med = float(np.median(gaps[1:])) if m > 1 else 0.1
    z = gaps / (gaps + max(med, 0.02))  # 0..1: how much of a break the gap before k is
    z[0] = 1.0
    exp = np.maximum(np.asarray(expected, np.float64), 1e-3)
    rate = float(lens.sum() / exp.sum())  # seconds of speech per unit of text
    G = np.concatenate([[0.0], np.cumsum(lens)])  # speech before segment k
    Z2 = np.concatenate([[0.0], np.cumsum(z[1:] ** 2)])  # Z2[k]: sum of z^2 for gaps 1..k
    pb = list(pause_before) + [0.0] * (n - len(pause_before))
    w = min(_MAX_WIDTH, m)
    inf = np.inf
    # P[k]: the best cost with the sentences so far covering segments 0..k-1
    P = np.full(m + 1, inf)
    P[0] = 0.0
    for k in range(1, m + 1):  # leading segments may be left out
        P[k] = _SKIP * k
    back = np.zeros((n, m + 1), np.int16)
    ks = np.arange(m + 1)
    # what does not depend on the sentence, per width: segments a..k-1 for every end k
    per = []
    for width in range(1, w + 1):
        a = ks - width
        ok = a >= 0
        aa = np.where(ok, a, 0)
        first = np.minimum(aa, m - 1)
        brk = np.where(aa > 0, 1.0 - z[first], 0.0)
        inner = 1.5 * (Z2[np.maximum(ks - 1, 0)] - Z2[aa])  # gaps aa+1 .. k-1
        per.append((width, ok, aa, np.log(G[ks] - G[aa] + 0.1), brk, inner, gaps[first]))
    for i in range(n):
        N = np.full(m + 1, inf)
        lw = math.log(rate * exp[i] + 0.1)
        for width, ok, aa, lsaid, brk, inner, g in per:
            cost = (lsaid - lw) ** 2 + inner
            if i:
                cost += 1.5 * brk
            if pb[i] > 0:
                cost += np.where(aa > 0, np.clip(1.0 - g / (0.5 * pb[i]), 0, 1), 0.0)
            total = np.where(ok, P[aa] + cost, inf)
            better = total < N
            N = np.where(better, total, N)
            back[i][better] = width
        P = N
    trail = P + _SKIP * (m - ks)
    k = int(np.argmin(trail))
    if not np.isfinite(trail[k]):
        raise ValueError("the recording could not be split into the script's sentences")
    out: list[Placed] = []
    for i in range(n - 1, -1, -1):
        width = int(back[i, k])
        a = k - width
        said = G[k] - G[a]
        fit = math.exp(-(math.log((said + 0.1) / (rate * exp[i] + 0.1)) ** 2))
        brk = 1.0 if a == 0 or i == 0 else float(z[a])
        onsets = tuple(float(starts[j]) for j in range(a + 1, k))
        out.append(Placed(float(starts[a]), float(ends[k - 1]),
                          round(speech.quality * fit * brk, 3), onsets))  # fmt: skip
        k = a
    return out[::-1]


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
) -> list[Move]:
    """Where each hand-placed marker goes (one ``Move`` per marker of ``hand`` and ``owned``).

    ``owned`` markers belong to the narration and never move; they bound the others.
    ``anchors`` pins a marker to a boundary. The rest are matched to ``boundaries`` in order
    (never two to one, never out of order) by dynamic programming on their place in the
    scene (``t / visual_end``) against the boundary's in the narration (``t /
    narration_end``), or on plain distance when that is smaller (a marker already on or near
    the narration); a match costs at most ``reach`` (a share of the narration's length).
    Markers left unmatched keep their place between their neighbours, stretched with the
    span around them. No span after a marker becomes shorter than ``need[marker]`` (the
    motions anchored at it) unless ``allow_short``: the least sure match that would do so is
    dropped. Planning is repeated on its own result until nothing changes, so a second snap
    moves nothing."""
    current = dict(hand)
    vend = max(float(visual_end), 1e-3)
    moves: list[Move] = []
    for _ in range(_PASSES):
        moves = _plan_once(current, owned, boundaries, vend, max(narration_end, 1e-3), reach,
                           need, anchors or {}, allow_short)  # fmt: skip
        new = {m.marker: m.t1 for m in moves}
        if all(abs(new[k] - current[k]) <= 5e-5 for k in current):
            break
        vend = _remap(vend, current, new)
        current = new
    last = {m.marker: m for m in moves}
    out = []
    for name, t0 in sorted({**hand, **owned}.items(), key=lambda kv: (kv[1], kv[0])):
        if name in owned:
            out.append(Move(name, t0, t0, "owned by the narration (a [[marker]] in its "
                            "script): it is already on the words", 1.0, "owned"))  # fmt: skip
            continue
        m, t1 = last[name], round(current[name], 4)
        if abs(t1 - t0) <= 5e-5 and m.kind == "between":
            out.append(Move(name, t0, t0, "kept: no sentence or pause near it", m.confidence,
                            "kept"))  # fmt: skip
        else:
            out.append(Move(name, t0, t1 if abs(t1 - t0) > 5e-5 else t0, m.why, m.confidence,
                            m.kind))  # fmt: skip
    return out


def _remap(t: float, old: Mapping[str, float], new: Mapping[str, float]) -> float:
    pts = sorted((old[k], new[k]) for k in old)
    if not pts:
        return t
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if t >= xs[-1]:
        return t + ys[-1] - xs[-1]
    return float(np.interp(t, [0.0, *xs], [0.0, *ys]))


def _plan_once(hand, owned, boundaries, vend, nend, reach, need, anchors, allow_short):
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
    pinned = {i for i, k in enumerate(names) if k in anchors}
    if len(c) and len(h):
        extra = np.array([_PHRASE_COST * reach if b.kind == "phrase" else 0.0 for b in bounds])

        def cost(i: int) -> np.ndarray:  # a row at a time: memory is one row, not K x N
            if i in pinned:  # a pinned marker matches its boundary only
                return np.full(len(c), np.inf)
            row = np.minimum(np.abs(h[i] / vend - c / nend), np.abs(h[i] - c) / nend) + extra
            row[(c <= lo[i]) | (c >= hi[i]) | (row > reach)] = np.inf
            return row

        match = _dp(cost, len(h), len(c), reach, pinned)
    detail: dict[str, Move] = {}
    anchor_pts: list[tuple[float, float, str, float]] = []  # (old, new, name, confidence)
    for i, k in enumerate(names):
        if k in anchors:
            b = anchors[k]
            detail[k] = Move(k, h[i], b.t, f"pinned to {b.label}", b.confidence, b.kind)
            anchor_pts.append((h[i], b.t, k, 2.0))
        elif i in match:
            b = bounds[match[i]]
            off = min(abs(h[i] - b.t) / nend, abs(h[i] / vend - b.t / nend))
            conf = round(b.confidence * max(0.0, 1.0 - float(off) / reach), 3)
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
    return out


def _dp(cost: Callable[[int], np.ndarray], k: int, n: int, skip: float,
        pinned: set[int]) -> dict[int, int]:  # fmt: skip
    """An order-keeping matching of ``k`` rows (markers) to ``n`` columns (boundaries), each
    used once, minimising the matched costs (``cost(i)``: row ``i``) plus ``skip`` per
    unmatched row."""
    F = np.zeros(n + 1)  # F[j]: best cost so far using columns < j
    choice = np.zeros((k, n + 1), np.int8)  # 0 skip row, 1 match (i, j-1), 2 carry from j-1
    for i in range(k):
        pen = 0.0 if i in pinned else skip
        G = F + pen
        row = cost(i)
        G[1:] = np.minimum(G[1:], F[:-1] + row)
        took = np.zeros(n + 1, np.int8)
        took[1:] = np.where(F[:-1] + row < F[1:] + pen, 1, 0)
        H = np.minimum.accumulate(G)
        carry = np.zeros(n + 1, np.int8)
        carry[1:] = H[1:] < G[1:]
        choice[i] = np.where(carry == 1, 2, took)
        F = H
    out: dict[int, int] = {}
    j = n
    for i in range(k - 1, -1, -1):
        while choice[i, j] == 2:
            j -= 1
        if choice[i, j] == 1:
            out[i] = j - 1
            j -= 1
    return out


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
