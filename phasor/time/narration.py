"""Narration as time (SPEC "Timeline, state and narration sync"). Pure: no audio libraries.

A script is plain text with markers:

    We start with a function that has two poles [[poles_appear]].
    Draw a small loop around [[loop_draw]] the one at i.
    [[pause 1.5]]
    Only that pole contributes [[residue_glow]].

Each sentence is a *block* (a macroblock). Markers cut a block into *pieces* (microblocks);
a marker's time is where the piece after it starts. ``[[pause s]]`` is a silent block.

How long each piece lasts comes from a voice (``phasor.voice``): synthesized piece by
piece (exact), aligned from a recording, or estimated from the text. This module turns
those durations into a ``Narration`` (core.schema) whose markers become the TimeMap.
"""

from __future__ import annotations

import re
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phasor.core.ids import is_valid_name
from phasor.core.schema import Narration, NarrationBlock, Word

_TAG = re.compile(r"\[\[\s*([^\]]*?)\s*\]\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PAUSE = re.compile(r"pause\s+(\d+(?:\.\d+)?)")
_WORD = re.compile(r"\w")


class ScriptError(ValueError):
    pass


@dataclass(frozen=True)
class Block:
    """A sentence split at its markers, or a pause (``pause`` set, no pieces)."""

    pieces: tuple[str, ...] = ()
    markers: Mapping[str, int] = field(default_factory=dict)  # name -> piece index it starts
    pause: float | None = None

    @property
    def text(self) -> str:
        return " ".join(self.pieces)


def _split_block(sentence: str, seen: set[str]) -> Block:
    pieces: list[str] = []
    markers: dict[str, int] = {}
    pos = 0
    for m in _TAG.finditer(sentence):
        pieces.append(sentence[pos : m.start()])
        name = m.group(1)
        if not is_valid_name(name):
            raise ScriptError(f"bad marker [[{name}]]: use a name like loop_draw, or [[pause 1.5]]")
        if name in seen:
            raise ScriptError(f"marker [[{name}]] is used twice")
        seen.add(name)
        markers[name] = len(pieces)
        pos = m.end()
    pieces.append(sentence[pos:])
    # A piece with no words (". " after a final marker) joins the piece before it; markers
    # that pointed at it now mean "the end of the piece before".
    out: list[str] = []
    for piece in pieces:
        piece = " ".join(piece.split())
        if _WORD.search(piece) or not out:
            out.append(piece)
            continue
        out[-1] = (out[-1] + piece).strip()
        for name, k in markers.items():
            if k > len(out):
                markers[name] = k - 1
    if not _WORD.search(out[0]):  # nothing spoken before the first marker
        if len(out) > 1:
            out.pop(0)
            markers = {n: max(k - 1, 0) for n, k in markers.items()}
        else:
            raise ScriptError(f"block {sentence.strip()!r} has no words")
    return Block(tuple(out), markers)


def parse_script(script: str) -> list[Block]:
    """Blocks in order. Sentences end at . ! ? or a line break; pauses stand alone."""
    blocks: list[Block] = []
    seen: set[str] = set()
    for line in script.splitlines():
        # A pause tag splits the line; text on either side is spoken normally.
        for part in re.split(r"(\[\[\s*pause\b[^\]]*\]\])", line):
            part = part.strip()
            if not part:
                continue
            tag = _TAG.fullmatch(part)
            if tag and tag.group(1).startswith("pause"):
                m = _PAUSE.fullmatch(tag.group(1))
                if m is None:
                    raise ScriptError(f"bad pause {part}: write [[pause 1.5]]")
                blocks.append(Block(pause=float(m.group(1))))
                continue
            for sentence in _SENTENCE_END.split(part):
                if sentence.strip():
                    blocks.append(_split_block(sentence, seen))
    if not blocks:
        raise ScriptError("the script is empty")
    return blocks


# ------------------------------------------------------------------ timing


def estimate_words(text: str, duration: float, offset: float = 0.0) -> tuple[Word, ...]:
    """Spread the words of ``text`` over ``duration`` in proportion to their length."""
    words = text.split()
    if not words:
        return ()
    weights = np.array([len(w) + 1 for w in words], dtype=np.float64)
    edges = offset + duration * np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
    return tuple(Word(text=w, start=round(float(a), 4), end=round(float(b), 4))
                 for w, a, b in zip(words, edges[:-1], edges[1:], strict=True))  # fmt: skip


def estimate_duration(text: str, words_per_second: float) -> float:
    """Speaking time for ``text``: words at a steady rate plus short stops at punctuation."""
    n = len(text.split())
    stops = 0.15 * len(re.findall(r"[,;:]", text)) + 0.3 * len(re.findall(r"[.!?]", text))
    return n / words_per_second + stops


@dataclass(frozen=True)
class Spoken:
    """What a voice produced for one block: piece durations, optional audio and words."""

    durations: tuple[float, ...]
    audio: str | None = None
    words: tuple[Word, ...] = ()


def build(
    script: str,
    voice: str,
    blocks: Sequence[Block],
    spoken: Sequence[Spoken | None],
    *,
    lead: float,
    gap: float,
    extra: Sequence[float] = (),
) -> Narration:
    """Lay blocks end to end: ``lead`` seconds of silence first, ``gap`` between blocks,
    plus ``extra[k]`` after block k (pacing). ``spoken[k]`` is None for pauses."""
    out: list[NarrationBlock] = []
    t = lead
    for k, (block, said) in enumerate(zip(blocks, spoken, strict=True)):
        if block.pause is not None:
            out.append(NarrationBlock(text="", start=round(t, 4), duration=block.pause))
            t += block.pause + (extra[k] if k < len(extra) else 0.0)
            continue
        assert said is not None and len(said.durations) == len(block.pieces)
        starts = np.concatenate([[0.0], np.cumsum(said.durations)])
        markers = {name: round(float(starts[i]), 4) for name, i in block.markers.items()}
        duration = float(starts[-1])
        words = said.words
        if not words:
            words = tuple(w for piece, a, d in zip(block.pieces, starts[:-1], said.durations,
                                                   strict=True)
                          for w in estimate_words(piece, d, a))  # fmt: skip
        out.append(NarrationBlock(text=block.text, start=round(t, 4), duration=round(duration, 4),
                                  audio=said.audio, words=words, markers=markers))  # fmt: skip
        t += duration + gap + (extra[k] if k < len(extra) else 0.0)
    return Narration(script=script, voice=voice, blocks=tuple(out))


def repace(narration: Narration, extra: Sequence[float], gap: float, lead: float) -> Narration:
    """Move blocks so that block k is followed by ``gap + extra[k]`` (pauses: ``extra[k]``).
    Durations and audio stay; only start times, and so the markers, change."""
    out, t = [], lead
    for k, b in enumerate(narration.blocks):
        out.append(NarrationBlock.model_validate({**b.model_dump(), "start": round(t, 4)}))
        t += b.duration + (0.0 if not b.text else gap) + (extra[k] if k < len(extra) else 0.0)
    return Narration(script=narration.script, voice=narration.voice, blocks=tuple(out))


def block_of(narration: Narration, t: float) -> int | None:
    """Index of the block that is playing, or last played, at time ``t``."""
    k = None
    for i, b in enumerate(narration.blocks):
        if b.start <= t + 1e-9:
            k = i
    return k


# ------------------------------------------------------------------ audio track


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono int16 samples and the sample rate (narration: any rate). Stereo is mixed down.
    ``ValueError`` for a file that is not 16-bit PCM WAV (``parse_wav`` says why)."""
    info = parse_wav(path)
    if info.width != 2:
        raise ValueError(f"{path}: only 16-bit WAV is supported")
    with open(path, "rb") as f:
        f.seek(info.offset)
        raw = f.read(info.frames * info.channels * 2)
    data = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if info.channels > 1:
        data = data.reshape(-1, info.channels).mean(axis=1).astype(np.int16)
    return data, info.rate


def write_wav(path: str | Path, samples: np.ndarray, rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


# Bounds on sound clips (``media.Sound``): a clip read from a file the document names is
# never allowed to exhaust memory or time, whatever its header says.
SOUND_RATES = (8000, 192000)  # sample rates a clip may have
MAX_CHANNELS = 16
MAX_CLIP_FRAMES = 1 << 26  # frames of one clip that are read (23 minutes at 48 kHz)
MAX_TRACK_SECONDS = 3 * 3600.0  # the mixed track never covers more than this
SOUND_RATE = 48000  # the track's rate when only sound clips (no narration audio) play
_CHUNK = 1 << 18  # frames converted at a time


@dataclass(frozen=True)
class WavInfo:
    rate: int
    channels: int
    width: int  # bytes per sample: 1 (unsigned), 2, 3 or 4 (signed PCM)
    frames: int  # the frames the file really holds (never more than its header says)
    offset: int = 0  # where the sample data starts in the file

    @property
    def seconds(self) -> float:
        return self.frames / self.rate


_WAVE_PCM, _WAVE_EXTENSIBLE = 1, 0xFFFE
_PCM_GUID = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq"
_MAX_CHUNKS = 1024  # chunks walked before giving up on finding the data


def parse_wav(path: str | Path) -> WavInfo:
    """The one WAV reader's header walk (narration's ``read_wav`` and sound clips'
    ``wav_info``/``read_sound`` share it): PCM (or extensible PCM, as Python's ``wave``)
    format, any rate. Every size is checked against the file before anything is read, so a
    lying or hostile header is a ``ValueError`` with the reason, never another exception and
    never a large read. ``frames`` is what the file really holds. A RIFF size smaller than the
    data it holds is refused (it could be cut anywhere); 0 and 0xFFFFFFFF (a stream whose
    writer never came back to fill it in) mean "to the end of the file"."""
    try:
        return _parse_wav(Path(path))
    except ValueError:
        raise
    except Exception as e:  # anything else a hostile file provokes: said, not raised
        raise ValueError(f"{type(e).__name__}: {e}") from None


def _parse_wav(path: Path) -> WavInfo:
    import os
    import struct

    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:] != b"WAVE":
            raise ValueError("not a RIFF WAVE file")
        (riff,) = struct.unpack("<I", head[4:8])
        known = riff not in (0, 0xFFFFFFFF)
        fmt: tuple[int, int, int] | None = None
        pos = 12
        for _ in range(_MAX_CHUNKS):
            if pos + 8 > size:
                break
            f.seek(pos)
            name, n = struct.unpack("<4sI", f.read(8))
            body = pos + 8
            if name == b"fmt ":
                if n < 16 or body + n > size:
                    raise ValueError(f"its fmt chunk says {n} bytes; the file has "
                                     f"{size - body} after it")  # fmt: skip
                fmt = _fmt_chunk(f.read(min(n, 40)))
            elif name == b"data":
                if fmt is None:
                    raise ValueError("a data chunk before the fmt chunk")
                rate, channels, width = fmt
                held = size - body if n == 0xFFFFFFFF else min(n, size - body)
                if known and body + held > riff + 8 + 1:  # (+1: a pad byte miscounted)
                    raise ValueError(f"its RIFF header says the file is {riff + 8} bytes, but "
                                     f"its data runs to byte {body + held}")  # fmt: skip
                return WavInfo(rate, channels, width, held // (width * channels), body)
            pos = body + n + (n & 1)  # chunks are padded to an even size
        raise ValueError("no fmt chunk and data chunk" if fmt is None else "no data chunk")


def _fmt_chunk(raw: bytes) -> tuple[int, int, int]:
    """(rate, channels, bytes per sample) of a ``fmt `` chunk; PCM only."""
    import struct

    tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", raw[:16])
    if tag == _WAVE_EXTENSIBLE:
        if len(raw) < 40 or raw[24:40] != _PCM_GUID:
            raise ValueError("an extensible format that is not PCM")
    elif tag != _WAVE_PCM:
        raise ValueError(f"format {tag} (compressed): only PCM is read")
    width = (bits + 7) // 8
    if not width or not channels or not rate:
        raise ValueError(f"{channels} channels of {bits}-bit samples at {rate} Hz")
    return rate, channels, width


def wav_info(path: str | Path) -> WavInfo:
    """A WAV file's header (``parse_wav``), checked against the bounds above. ``ValueError``
    (with the reason) for anything that is not PCM WAV phasor can mix: not a WAV, compressed,
    a lying header, a sample rate, channel count or sample width out of range."""
    try:
        info = parse_wav(path)
    except ValueError as e:
        raise ValueError(f"not a PCM WAV file ({e})") from None
    if not SOUND_RATES[0] <= info.rate <= SOUND_RATES[1]:
        raise ValueError(f"a sample rate of {info.rate} Hz (from {SOUND_RATES[0]} to "
                         f"{SOUND_RATES[1]})")  # fmt: skip
    if not 1 <= info.channels <= MAX_CHANNELS:
        raise ValueError(f"{info.channels} channels (1 to {MAX_CHANNELS})")
    if info.width not in (1, 2, 3, 4):
        raise ValueError(f"{info.width * 8}-bit samples (8, 16, 24 or 32-bit PCM)")
    return info


def read_sound(path: str | Path, max_frames: int = MAX_CLIP_FRAMES,
               skip: int = 0) -> tuple[np.ndarray, int]:  # fmt: skip
    """A WAV clip as mono float32 in -1..1 and its rate: at most ``max_frames`` frames of the
    first ``MAX_CLIP_FRAMES`` after the first ``skip``, read a chunk at a time and never more
    than the file holds, so a lying header or a huge file costs no more than what is read.
    ``ValueError`` for a file it cannot read."""
    info = wav_info(path)
    return _frames(path, info, max(0, int(skip)), max(0, int(max_frames))), info.rate


def _frames(path: str | Path, info: WavInfo, skip: int, count: int) -> np.ndarray:
    """Frames ``skip`` to ``skip + count`` of the clip (cut at ``MAX_CLIP_FRAMES`` and at what
    the file holds) as mono float32."""
    want = max(0, min(count, info.frames - skip, MAX_CLIP_FRAMES - skip))
    frame = info.width * info.channels
    chunks: list[np.ndarray] = []
    try:
        with open(path, "rb") as f:
            f.seek(info.offset + skip * frame)
            got = 0
            while got < want:
                raw = f.read(min(_CHUNK, want - got) * frame)
                n = len(raw) // frame
                if n == 0:
                    break
                chunks.append(_pcm(raw[: n * frame], info))
                got += n
    except OSError as e:
        raise ValueError(f"cannot be read ({e})") from None
    if not chunks:
        return np.zeros(0, np.float32)
    out = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return out.astype(np.float32, copy=False)


def _pcm(raw: bytes, info: WavInfo) -> np.ndarray:
    """Interleaved PCM bytes as mono float32 in -1..1 (channels averaged)."""
    if info.width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif info.width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif info.width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / float(1 << 23)
    else:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    return x.reshape(-1, info.channels).mean(axis=1)


def resample(x: np.ndarray, rate: int, to: int, limit: int | None = None) -> np.ndarray:
    """``x`` at ``rate`` as samples at ``to`` (linear interpolation; enough for a sound
    effect under narration), at most ``limit`` of them."""
    n = int(len(x) * to / rate) if rate != to else len(x)
    if limit is not None:
        n = min(n, max(0, int(limit)))
    if rate == to or not len(x):
        return x[:n]
    return np.interp(np.arange(n) * (rate / to), np.arange(len(x)), x).astype(np.float32)


@dataclass(frozen=True)
class Clip:
    """A sound clip to mix in (``media.Sound``): a WAV file in the scene's folder, from
    ``start`` seconds, ``gain`` dB louder."""

    path: str
    start: float
    gain: float = 0.0


def mix_track(
    narration: Narration | None,
    base_dir: str | Path,
    duration: float,
    clips: Sequence[Clip] = (),
    problems: list[str] | None = None,
    start: float = 0.0,
    end: float | None = None,
) -> tuple[np.ndarray, int] | None:
    """The narration and any sound clips as one mono int16 track, each block's audio placed
    at its start, each clip at its ``start`` and cut where the scene (``duration``) ends.
    The track covers ``start`` to ``end`` (by default the scene, or the narration if it runs
    longer), never more than ``MAX_TRACK_SECONDS``: only that window is read and mixed. None
    when nothing has audio. A clip that is missing, outside the folder or unreadable is left
    out and said in ``problems`` (lint names it too: ``missing_file``, ``bad_file``);
    narration audio that cannot be read raises, as it always has."""
    from phasor.core import files

    root = Path(base_dir).resolve()
    spoken = []
    for b in narration.blocks if narration is not None else ():
        if not b.audio:
            continue
        path = (root / b.audio).resolve()
        if root not in path.parents:  # a document must not read files outside its folder
            raise ValueError(f"narration audio {b.audio!r} is outside the scene's folder")
        spoken.append((b.start, read_wav(path)))
    rates = {rate for _, (_, rate) in spoken}
    if len(rates) > 1:
        raise ValueError(f"narration clips have different sample rates: {sorted(rates)}")
    rate = rates.pop() if rates else SOUND_RATE
    scene_end = max(float(duration), 0.0)
    lo = max(float(start), 0.0)
    hi = max([scene_end] + [t + len(s) / rate for t, (s, _) in spoken]) if end is None else end
    hi = min(max(float(hi), lo), lo + MAX_TRACK_SECONDS)
    clip_end = min(scene_end, hi)  # clips stop where the scene (or the window) ends
    sounds = []
    for c in clips:
        if not (max(c.start, lo) < clip_end):
            continue
        try:
            file = files.regular_file(root, c.path)
            info = wav_info(file)
            skip = int((max(lo - c.start, 0.0)) * info.rate)
            span = clip_end - max(c.start, lo)
            x, r = read_sound(file, int(np.ceil(span * info.rate)) + 1, skip)
        except Exception as e:  # whatever a hostile file provokes: left out, said
            if problems is not None:
                problems.append(f"sound {c.path!r} left out: {e}")
            continue
        y = resample(x, r, rate, int(np.ceil(span * rate)) + 1) * float(10 ** (c.gain / 20))
        if len(y):  # a clip that ended before the window adds nothing
            sounds.append((max(c.start, lo), y))
    if not spoken and not sounds:
        return None
    track = np.zeros(int(np.ceil((hi - lo) * rate)), dtype=np.float32)
    stop = int(np.ceil((clip_end - lo) * rate))
    for t, (samples, _) in spoken:
        _place(track, samples, round((t - lo) * rate), len(track))
    for t, x in sounds:
        _place(track, x * 32768.0, round((t - lo) * rate), stop)
    return np.clip(np.round(track), -32768, 32767).astype(np.int16), rate


def _place(track: np.ndarray, x: np.ndarray, i: int, stop: int) -> None:
    """Add ``x`` into ``track`` from index ``i`` (negative: its start is cut), up to ``stop``."""
    if i < 0:
        x, i = x[-i:], 0
    x = x[: max(0, min(stop, len(track)) - i)]
    track[i : i + len(x)] += x
