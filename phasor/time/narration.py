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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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


PRE_ROLL = 0.03  # seconds of a recording kept before a sentence's first sound (as TTS clips)
POST_ROLL = 0.05  # ... and after its last
PHRASE = "\u2026"  # the text of a phrase whose words are not known (a recording, no script)


def place_recorded(
    script: str,
    voice: str,
    audio: str,
    blocks: Sequence[Block],
    found: Sequence[tuple[float, float, Sequence[float]]],
    seconds: float,
) -> Narration:
    """A narration cut from one recording (``audio``, ``seconds`` long), each spoken block
    where the recording says it (``found[i]``: the start and end of the ``i``-th spoken
    block's speech and where each of its pieces starts, seconds into the recording). Each
    block plays its part of the file (``audio_span``: ``PRE_ROLL`` before the first sound to
    ``POST_ROLL`` after the last, never overlapping its neighbours) at the time it has in the
    recording, so the scene's timeline is the recording's. A marker at a block's first piece
    is the block's start; one at a later piece is that piece's first sound, less the pre-roll.
    A ``[[pause]]`` between two blocks lasts the silence the recording has there."""
    spoken = [b for b in blocks if b.pause is None]
    if len(spoken) != len(found):
        raise ScriptError(f"{len(spoken)} sentences but {len(found)} were found in the audio")
    spans: list[tuple[float, float]] = []
    for i, (a, b, _) in enumerate(found):
        lo = max(0.0, a - PRE_ROLL, spans[-1][1] if spans else 0.0)
        nxt = found[i + 1][0] if i + 1 < len(found) else seconds
        hi = min(max(b, lo), b + POST_ROLL, max(b, (b + nxt) / 2), seconds)
        spans.append((round(lo, 4), round(max(hi, lo), 4)))
    out: list[NarrationBlock] = []
    k = 0
    t = 0.0
    for j, block in enumerate(blocks):
        if block.pause is not None:
            later = [i for i in range(j + 1, len(blocks)) if blocks[i].pause is None]
            end = spans[k][0] if later else t + block.pause
            out.append(NarrationBlock(text="", start=round(t, 4), duration=round(max(0.0,
                                      end - t), 4)))  # fmt: skip
            t = max(t, end)
            continue
        lo, hi = spans[k]
        piece_starts = list(found[k][2]) or [found[k][0]]
        offs = [0.0] + [min(max(0.0, p - PRE_ROLL - lo), hi - lo) for p in piece_starts[1:]]
        for i in range(1, len(offs)):
            offs[i] = max(offs[i], offs[i - 1])
        ends = [*offs[1:], hi - lo]
        words = tuple(w for piece, a, e in zip(block.pieces, offs, ends, strict=False)
                      for w in estimate_words(piece, max(e - a, 0.0), a))  # fmt: skip
        markers = {name: round(offs[i], 4) for name, i in block.markers.items()}
        out.append(NarrationBlock(text=block.text, start=lo, duration=round(hi - lo, 4),
                                  audio=audio, words=words, markers=markers,
                                  audio_span=(lo, hi)))  # fmt: skip
        t = hi
        k += 1
    return Narration(script=script, voice=voice, blocks=tuple(out))


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
    if info.width != 2 or info.floating:
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
    floating: bool = False  # IEEE float samples (format 3), 4 or 8 bytes, in -1..1 (M22)

    @property
    def seconds(self) -> float:
        return self.frames / self.rate


_WAVE_PCM, _WAVE_FLOAT, _WAVE_EXTENSIBLE = 1, 3, 0xFFFE
_PCM_GUID = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq"
_FLOAT_GUID = b"\x03" + _PCM_GUID[1:]
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
                rate, channels, width, floating = fmt
                held = size - body if n == 0xFFFFFFFF else min(n, size - body)
                if known and body + held > riff + 8 + 1:  # (+1: a pad byte miscounted)
                    raise ValueError(f"its RIFF header says the file is {riff + 8} bytes, but "
                                     f"its data runs to byte {body + held}")  # fmt: skip
                return WavInfo(rate, channels, width, held // (width * channels), body,
                               floating)  # fmt: skip
            pos = body + n + (n & 1)  # chunks are padded to an even size
        raise ValueError("no fmt chunk and data chunk" if fmt is None else "no data chunk")


def _fmt_chunk(raw: bytes) -> tuple[int, int, int, bool]:
    """(rate, channels, bytes per sample, float) of a ``fmt `` chunk: PCM, or IEEE float
    (32 or 64-bit; plain or extensible)."""
    import struct

    tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", raw[:16])
    if tag == _WAVE_EXTENSIBLE:
        if len(raw) >= 40 and raw[24:40] == _FLOAT_GUID:
            tag = _WAVE_FLOAT
        elif len(raw) < 40 or raw[24:40] != _PCM_GUID:
            raise ValueError("an extensible format that is neither PCM nor float")
    elif tag not in (_WAVE_PCM, _WAVE_FLOAT):
        raise ValueError(f"format {tag} (compressed): only PCM and float are read")
    width = (bits + 7) // 8
    if not width or not channels or not rate:
        raise ValueError(f"{channels} channels of {bits}-bit samples at {rate} Hz")
    if tag == _WAVE_FLOAT and bits not in (32, 64):
        raise ValueError(f"{bits}-bit float samples (32 or 64-bit float)")
    return rate, channels, width, tag == _WAVE_FLOAT


def wav_info(path: str | Path) -> WavInfo:
    """A WAV file's header (``parse_wav``), checked against the bounds above. ``ValueError``
    (with the reason) for anything that is not PCM (or float, M22) WAV phasor can mix: not a
    WAV, compressed,
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
    if info.width not in ((4, 8) if info.floating else (1, 2, 3, 4)):
        raise ValueError(f"{info.width * 8}-bit samples (8, 16, 24 or 32-bit PCM, or 32 or "
                         "64-bit float)")  # fmt: skip
    return info


def read_sound(path: str | Path, max_frames: int = MAX_CLIP_FRAMES,
               skip: int = 0) -> tuple[np.ndarray, int]:  # fmt: skip
    """A WAV clip as mono float32 in -1..1 and its rate: at most ``max_frames`` frames of the
    first ``MAX_CLIP_FRAMES`` after the first ``skip``, read a chunk at a time and never more
    than the file holds, so a lying header or a huge file costs no more than what is read.
    ``ValueError`` for a file it cannot read."""
    info = wav_info(path)
    return _frames(path, info, max(0, int(skip)), max(0, int(max_frames))), info.rate


def _frames(path: str | Path, info: WavInfo, skip: int, count: int,
            limit: int = MAX_CLIP_FRAMES) -> np.ndarray:  # fmt: skip
    """Frames ``skip`` to ``skip + count`` of the clip (cut at ``limit``, by default
    ``MAX_CLIP_FRAMES``, and at what the file holds) as mono float32."""
    want = max(0, min(count, info.frames - skip, limit - skip))
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
    """Interleaved PCM (or float) bytes as mono float32 in -1..1 (channels averaged)."""
    if info.floating:
        x = np.frombuffer(raw, "<f4" if info.width == 4 else "<f8").astype(np.float32)
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    elif info.width == 1:
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


class Source(Protocol):
    """Mono samples a clip plays: a WAV file (``WavSource``), or anything else that can say
    its rate and length and hand out float32 samples in -1..1 a window at a time (Milestone
    23's synthesizer). ``frames`` is already bounded; ``read`` never returns more than asked
    and may return fewer (the end; a file cut short on disk raises ``ValueError``)."""

    @property
    def rate(self) -> int: ...

    @property
    def frames(self) -> int: ...

    def read(self, skip: int, count: int) -> np.ndarray: ...


@dataclass(frozen=True)
class WavSource:
    """A PCM WAV file as a ``Source``: read a chunk at a time (``_frames``), at most ``limit``
    frames from its start."""

    path: Path
    info: WavInfo
    limit: int = MAX_CLIP_FRAMES

    @property
    def rate(self) -> int:
        return self.info.rate

    @property
    def frames(self) -> int:
        return max(0, min(self.info.frames, self.limit))

    def read(self, skip: int, count: int) -> np.ndarray:
        return _frames(self.path, self.info, skip, count, self.limit)


@dataclass(frozen=True)
class Clip:
    """A sound clip to mix in (``media.Sound``): a file in the scene's folder (``path``), from
    ``start`` seconds, ``gain`` dB louder. ``source``: samples already opened (a type that
    makes its own sound); None reads the file at ``path`` (a WAV, or through the mix's
    ``resolve``: another format decoded to a WAV). ``marker``: a marker the clip starts at,
    ``start`` seconds after it (``Engine.audio_track`` resolves it before mixing; the mix
    itself reads only ``start``)."""

    path: str
    start: float
    gain: float = 0.0
    source: Source | None = None
    marker: str | None = None


Resolve = Callable[[Path], Path]  # a file in the scene's folder -> a PCM WAV of its sound


def mix_track(
    narration: Narration | None,
    base_dir: str | Path,
    duration: float,
    clips: Sequence[Clip] = (),
    problems: list[str] | None = None,
    start: float = 0.0,
    end: float | None = None,
    resolve: Resolve | None = None,
) -> tuple[np.ndarray, int] | None:
    """The narration and any sound clips as one mono int16 track, each block's audio placed
    at its start, each clip at its ``start`` and cut where the scene (``duration``) ends.
    The track covers ``start`` to ``end`` (by default the scene, or the narration if it runs
    longer), never more than ``MAX_TRACK_SECONDS``: only that window is read and mixed, on the
    scene's own sample grid, so a window is the same samples as that slice of the whole mix.
    It is mixed ``_OUT_CHUNK`` samples at a time, each clip read only for the chunk it is in:
    memory is the track and one chunk, whatever the number or length of the clips. At most
    ``MAX_SOUNDS`` clips and ``MAX_MIX_SAMPLES`` clip samples in all are mixed. None when
    nothing has audio. A clip that is missing, outside the folder, unreadable or past those
    bounds is left out and said in ``problems`` (lint names it too: ``missing_file``,
    ``bad_file``, ``sound_limit``); narration audio that cannot be read raises, as it always
    has. A narration block with an ``audio_span`` (a recording cut into sentences, M22) plays
    that part of its file, read a chunk at a time like a clip (its samples count towards
    ``MAX_MIX_SAMPLES``); one that cannot be read is left out and said. ``resolve`` turns a
    file that is not a WAV (an mp4's sound) into one (``domains.audio``: this layer decodes
    nothing)."""
    from phasor.core import files

    root = Path(base_dir).resolve()
    spoken = []
    cut: list[NarrationBlock] = []
    for b in narration.blocks if narration is not None else ():
        if not b.audio:
            continue
        path = (root / b.audio).resolve()
        if root not in path.parents:  # a document must not read files outside its folder
            raise ValueError(f"narration audio {b.audio!r} is outside the scene's folder")
        if b.audio_span is None:
            spoken.append((b.start, read_wav(path)))
        else:
            cut.append(b)
    opened = _open_spans(root, files, cut, resolve, problems)
    rates = {rate for _, (_, rate) in spoken} | {src.rate for src in opened.values()}
    if len(rates) > 1:
        raise ValueError(f"narration clips have different sample rates: {sorted(rates)}")
    rate = rates.pop() if rates else SOUND_RATE
    scene_end = max(float(duration), 0.0)
    lo = max(float(start), 0.0)
    ends = [t + len(s) / rate for t, (s, _) in spoken]
    # a recording's blocks lengthen the track at most SPAN_MARGIN past the scene (a block
    # placed far beyond it is a hostile or broken document: lint says so, nothing is made)
    late = [b for b in cut if b.start > scene_end + SPAN_MARGIN]
    if late and problems is not None:
        problems.append(f"{len(late)} narration block(s) from {late[0].start:g} s left out: "
                        f"past the scene's end ({scene_end:g} s) by more than "
                        f"{SPAN_MARGIN:g} s")  # fmt: skip
    cut = [b for b in cut if b not in late]
    ends += [min(b.start + b.duration, scene_end + SPAN_MARGIN) for b in cut
             if str(b.audio) in opened]  # fmt: skip
    hi = max([scene_end, *ends]) if end is None else end
    hi = min(max(float(hi), lo), lo + MAX_TRACK_SECONDS)
    g0 = round(lo * rate)  # the window's first sample, on the scene's sample grid
    size = max(0, int(np.ceil(hi * rate - 1e-6)) - g0)  # to the grid sample at or after hi
    if size > MAX_TRACK_SAMPLES:  # a very high rate: the track is cut, never a memory blow-up
        if problems is not None:
            problems.append(f"the track is cut after {MAX_TRACK_SAMPLES / rate:g} s: at most "
                            f"{MAX_TRACK_SAMPLES} samples are mixed")  # fmt: skip
        size = MAX_TRACK_SAMPLES
    stop = min(g0 + size, int(np.ceil(scene_end * rate)))  # clips stop where the scene ends
    work = [0]
    plan = _plan_spans(cut, opened, rate, g0, g0 + size, work, problems)
    plan += _plan_clips(root, files, clips, rate, g0, stop, problems, resolve, work)
    if not spoken and not plan:
        return None
    out = np.empty(size, np.int16)
    for a in range(0, size, _OUT_CHUNK):
        b = min(size, a + _OUT_CHUNK)
        buf = np.zeros(b - a, np.float32)
        for t, (samples, _) in spoken:
            _add(buf, samples, round(t * rate) - g0 - a)
        for clip in plan:
            clip.mix(buf, g0 + a, problems)
        np.round(buf, out=buf)
        np.clip(buf, -32768, 32767, out=buf)
        out[a:b] = buf
    return out, rate


def _open_spans(root: Path, files: Any, blocks: Sequence[NarrationBlock],
                resolve: Resolve | None, problems: list[str] | None
                ) -> dict[str, Source]:  # fmt: skip
    """The file each recorded block cuts, opened once per file (its header checked)."""
    out: dict[str, Source] = {}
    bad: set[str] = set()
    for b in blocks:
        name = str(b.audio)
        if name in out or name in bad:
            continue
        try:
            file = files.regular_file(root, name)
            wav = Path(resolve(file) if resolve is not None else file)
            info = wav_info(wav)
        except Exception as e:  # whatever a hostile file provokes: left out, said
            bad.add(name)
            if problems is not None:
                problems.append(f"narration audio {name!r} left out: {e}")
            continue
        out[name] = WavSource(wav, info, int(MAX_TRACK_SECONDS * info.rate) + 1)
    return out


def _plan_spans(blocks: Sequence[NarrationBlock], opened: Mapping[str, Source], rate: int,
                g0: int, g1: int, work: list[int], problems: list[str] | None
                ) -> list[_Planned]:  # fmt: skip
    """Each recorded block with samples in the window [g0, g1), within ``MAX_MIX_SAMPLES``."""
    plan: list[_Planned] = []
    for b in blocks:
        src = opened.get(str(b.audio))
        if src is None or b.audio_span is None:
            continue
        first = round(b.start * rate)
        skip = round(b.audio_span[0] * src.rate)
        count = max(0, min(round(b.duration * rate), src.frames - skip))
        p = _budget(_Planned(src, f"narration audio {b.audio!r}", first, count, 1.0, 1.0, True,
                             skip), g0, g1, rate, work, problems)  # fmt: skip
        if p is not None:
            plan.append(p)
    return plan


SPAN_MARGIN = 60.0  # seconds a recording's blocks may run past the scene's end
MAX_TRACK_SAMPLES = int(MAX_TRACK_SECONDS * SOUND_RATE)  # samples in one track (1 GiB int16)
MAX_SOUNDS = 256  # sound clips mixed into one track; the rest are left out (lint sound_limit)
MAX_MIX_SAMPLES = 1 << 30  # clip samples mixed in all (6 hours at 48 kHz: 2 full tracks)
_OUT_CHUNK = 1 << 20  # track samples mixed at a time


@dataclass
class _Planned:
    """A clip placed on the track's sample grid: output sample ``j`` (at the track's rate) is
    global sample ``first + j``, read from source position ``skip + j * ratio``; ``count`` of
    them (cut where the scene ends and at what the source holds)."""

    source: Source
    name: str
    first: int
    count: int
    ratio: float  # source samples per track sample
    gain: float
    same_rate: bool
    skip: int = 0  # source samples before the clip's first (a part of a recording)
    failed: bool = False

    @property
    def label(self) -> str:
        return self.name if self.name.startswith("narration") else f"sound {self.name!r}"

    def mix(self, buf: np.ndarray, g: int, problems: list[str] | None) -> None:
        """Add the part of the clip that falls in ``buf`` (global samples ``g`` on)."""
        j0, j1 = max(0, g - self.first), min(self.count, g + len(buf) - self.first)
        if self.failed or j1 <= j0:
            return
        try:
            if self.same_rate:
                y = self.source.read(self.skip + j0, j1 - j0)
            else:
                pos = np.arange(j0, j1) * self.ratio
                s0 = int(pos[0])
                s1 = min(self.source.frames - self.skip, int(pos[-1]) + 2)
                x = self.source.read(self.skip + s0, s1 - s0)
                y = np.interp(pos, np.arange(s0, s0 + len(x)), x).astype(np.float32)
        except Exception as e:  # changed or cut on disk since it was checked: said
            self.failed = True
            if problems is not None:
                problems.append(f"{self.label} left out from sample {g + j0}: {e}")
            return
        y = y * self.gain
        _add(buf, y * 32768.0, self.first + j0 - g)


def _budget(p: _Planned, g0: int, stop: int, rate: int, work: list[int],
            problems: list[str] | None) -> _Planned | None:  # fmt: skip
    """``p`` cut where ``stop`` is and to the ``MAX_MIX_SAMPLES`` left (``work[0]`` are used);
    None when none of it falls in [g0, stop). What is cut is said in ``problems``."""
    p.count = min(p.count, stop - p.first)
    mixed = min(p.first + p.count, stop) - max(p.first, g0)
    if mixed <= 0:  # it ended before the window
        return None
    if work[0] + mixed > MAX_MIX_SAMPLES:
        room = MAX_MIX_SAMPLES - work[0]
        if problems is not None:
            what = "left out" if room <= 0 else f"cut after {room / rate:g} s"
            problems.append(f"{p.label} {what}: a track mixes at most "
                            f"{MAX_MIX_SAMPLES} clip samples in all")  # fmt: skip
        if room <= 0:
            return None
        p.count = max(p.first, g0) + room - p.first
        mixed = room
    work[0] += mixed
    return p


def _plan_clips(root: Path, files: Any, clips: Sequence[Clip], rate: int, g0: int, stop: int,
                problems: list[str] | None, resolve: Resolve | None = None,
                work: list[int] | None = None) -> list[_Planned]:  # fmt: skip
    """Each clip that has samples in [g0, stop): checked (its header) before anything is
    mixed, within ``MAX_SOUNDS`` and ``MAX_MIX_SAMPLES``; the rest said in ``problems``."""
    plan: list[_Planned] = []
    work = [0] if work is None else work
    for k, c in enumerate(clips):
        if k >= MAX_SOUNDS:
            if problems is not None:
                problems.append(f"sound {c.path!r} left out: a track mixes at most "
                                f"{MAX_SOUNDS} sound clips")  # fmt: skip
            continue
        first = round(float(c.start) * rate)
        if not first < stop:
            continue
        try:
            src = c.source
            if src is None:
                file = files.regular_file(root, c.path)
                wav = Path(resolve(file) if resolve is not None else file)
                src = WavSource(wav, wav_info(wav))
        except Exception as e:  # whatever a hostile file provokes: left out, said
            if problems is not None:
                problems.append(f"sound {c.path!r} left out: {e}")
            continue
        same = src.rate == rate
        count = src.frames if same else int(src.frames * rate / src.rate)
        p = _budget(_Planned(src, c.path, first, count, src.rate / rate,
                             float(10 ** (c.gain / 20)), same), g0, stop, rate, work,
                    problems)  # fmt: skip
        if p is not None:
            plan.append(p)
    return plan


def _add(buf: np.ndarray, x: np.ndarray, i: int) -> None:
    """Add ``x`` into ``buf`` from index ``i`` (negative: its start is cut)."""
    if i < 0:
        x, i = x[-i:], 0
    x = x[: max(0, len(buf) - i)]
    buf[i : i + len(x)] += x
