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
from struct import error as struct_error

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
    """Mono int16 samples and the sample rate. Stereo is mixed down."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit WAV is supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return data, w.getframerate()


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
    frames: int

    @property
    def seconds(self) -> float:
        return self.frames / self.rate


def wav_info(path: str | Path) -> WavInfo:
    """A WAV file's header, checked against the bounds above. ``ValueError`` (with the
    reason) for anything that is not PCM WAV phasor can mix: not a WAV, compressed, a sample
    rate, channel count or sample width out of range."""
    try:
        with wave.open(str(path), "rb") as w:
            info = WavInfo(w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes())
    except (wave.Error, EOFError, OSError, struct_error, ValueError) as e:
        raise ValueError(f"not a PCM WAV file ({e or type(e).__name__})") from None
    if not SOUND_RATES[0] <= info.rate <= SOUND_RATES[1]:
        raise ValueError(f"a sample rate of {info.rate} Hz (from {SOUND_RATES[0]} to "
                         f"{SOUND_RATES[1]})")  # fmt: skip
    if not 1 <= info.channels <= MAX_CHANNELS:
        raise ValueError(f"{info.channels} channels (1 to {MAX_CHANNELS})")
    if info.width not in (1, 2, 3, 4):
        raise ValueError(f"{info.width * 8}-bit samples (8, 16, 24 or 32-bit PCM)")
    return info


def read_sound(path: str | Path, max_frames: int = MAX_CLIP_FRAMES) -> tuple[np.ndarray, int]:
    """A WAV clip as mono float32 in -1..1 and its rate: at most ``max_frames`` frames (and
    never more than ``MAX_CLIP_FRAMES``), read a chunk at a time, so a lying header or a
    huge file costs no more than that. ``ValueError`` for a file it cannot read."""
    info = wav_info(path)
    want = max(0, min(int(max_frames), info.frames, MAX_CLIP_FRAMES))
    out = np.zeros(want, dtype=np.float32)
    got = 0
    try:
        with wave.open(str(path), "rb") as w:
            while got < want:
                raw = w.readframes(min(_CHUNK, want - got))
                n = len(raw) // (info.width * info.channels)
                if n == 0:
                    break
                out[got : got + n] = _pcm(raw[: n * info.width * info.channels], info)
                got += n
    except (wave.Error, EOFError, OSError, struct_error) as e:
        raise ValueError(f"not a PCM WAV file ({e or type(e).__name__})") from None
    return out[:got], info.rate


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


def resample(x: np.ndarray, rate: int, to: int) -> np.ndarray:
    """``x`` at ``rate`` as samples at ``to`` (linear interpolation; enough for a sound
    effect under narration)."""
    if rate == to or not len(x):
        return x
    n = int(len(x) * to / rate)
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
) -> tuple[np.ndarray, int] | None:
    """The narration and any sound clips as one mono int16 track ``duration`` seconds long
    (at most ``MAX_TRACK_SECONDS``), each block's audio placed at its start, each clip at its
    ``start`` and cut where the scene ends. None when nothing has audio. A clip that is
    missing, outside the folder or unreadable is left out and said in ``problems`` (lint
    names it too: ``missing_file``, ``bad_file``); narration audio that cannot be read
    raises, as it always has."""
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
    cap = min(max(float(duration), 0.0), MAX_TRACK_SECONDS)
    end = max([cap] + [t + len(s) / rate for t, (s, _) in spoken])
    end = min(end, MAX_TRACK_SECONDS)
    sounds = []
    for c in clips:
        if not (0 <= c.start < cap):
            continue
        try:
            file = files.regular_file(root, c.path)
            info = wav_info(file)
            need = int(np.ceil((cap - c.start) * info.rate)) + 1
            x, r = read_sound(file, need)
        except ValueError as e:
            if problems is not None:
                problems.append(f"sound {c.path!r} left out: {e}")
            continue
        sounds.append((c.start, resample(x, r, rate) * float(10 ** (c.gain / 20))))
    if not spoken and not sounds:
        return None
    track = np.zeros(int(np.ceil(end * rate)), dtype=np.float32)
    for t, (samples, _) in spoken:
        i = round(t * rate)
        seg = samples[: max(0, len(track) - i)]
        track[i : i + len(seg)] += seg
    for t, x in sounds:
        i = round(t * rate)
        seg = x[: max(0, int(np.ceil(cap * rate)) - i)] * 32768.0
        track[i : i + len(seg)] += seg
    return np.clip(np.round(track), -32768, 32767).astype(np.int16), rate
