"""MIDI notes in and out, and a small synthesizer (Milestone 23). Pure NumPy and stdlib.

- ``read_smf(data)``: a Standard MIDI File (format 0 or 1; PPQ or SMPTE time) as a ``Song``:
  its notes (pitch, start and length in seconds through the tempo map, velocity, channel,
  GM program), track names, first tempo and time signature. Every size the file states is
  checked against what it holds; tracks, events, notes, tempo changes, ticks and note
  lengths are bounded, so a hostile file costs at most ``MAX_FILE_BYTES`` of reading and
  never raises: it is a ``Song`` with ``error`` (nothing read; lint ``bad_file``) or ``cut``
  (what was read up to a fault or a bound; lint ``truncated``). Format 2 (independent
  sequences) is refused with a reason.
- ``write_smf(tracks)``: notes as a format 1 file (a conductor track with the tempo and a
  4/4 time signature, one track per named part, running status).
- ``Synth``: notes as samples, a ``time.narration.Source``: the mixer plays it exactly as it
  plays a WAV clip (chunked, resampled to the track's rate, counted towards
  ``MAX_MIX_SAMPLES``). Every sample is a pure function of the notes, the seed and its own
  index: phases are integers, waveforms and envelope curves are fixed tables, the voices
  of one sample are added in one order in float64. So any window read is bit for bit the
  same slice of a read of the whole, in any process.
"""

from __future__ import annotations

import bisect
import functools
import heapq
import struct
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------- bounds

MAX_FILE_BYTES = 4 << 20  # a .mid larger than this is not read (bad_file)
MAX_TRACKS = 256  # tracks read; the rest are left out (truncated)
MAX_EVENTS = 400_000  # events read in all (measured: ~1 s of pure-Python parsing)
MAX_NOTES = 100_000  # notes kept (from a file, or after ``repeat``)
MAX_TEMPOS = 10_000  # tempo changes followed
MAX_CHUNKS = 1024  # chunks walked (unknown ones are skipped)
MAX_TICK = 1 << 40  # a track's running tick; past it the track stops
MAX_SECONDS = 3 * 3600.0  # notes starting later are left out (the track's own bound)
MAX_NOTE_SECONDS = 60.0  # longer notes (a note never let go) are cut to this
MIN_BPM, MAX_BPM = 1.0, 1000.0  # a tempo outside is ignored (said)

DRUMS = 9  # channel 10, counted from 0: General MIDI percussion

# The 128 General MIDI program names (program 0 = "Acoustic Grand Piano").
GM_NAMES = (
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet", "Celesta", "Glockenspiel",
    "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
    "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello",
    "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba",
    "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon",
    "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi",
    "Whistle", "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
    "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
    "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)",
    "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo",
    "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo",
    "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring",
    "Helicopter", "Applause", "Gunshot",
)  # fmt: skip
assert len(GM_NAMES) == 128


class Note(NamedTuple):
    """One note: MIDI ``pitch`` (60 = middle C; fractional pitches from a script play their
    exact frequency and are rounded when written to a file), ``start`` and ``length`` in
    seconds, ``velocity`` 1-127, ``channel`` 0-15 (9: drums), General MIDI ``program``."""

    pitch: float
    start: float
    length: float
    velocity: int = 100
    channel: int = 0
    program: int = 0


@dataclass(frozen=True)
class Song:
    """What ``read_smf`` found. ``error``: why nothing could be read (then no notes);
    ``cut``: why only part was read (the notes up to there are kept)."""

    notes: tuple[Note, ...] = ()
    format: int = 0
    tracks: int = 0
    names: tuple[str, ...] = ()
    division: str = ""
    bpm: float = 120.0  # the tempo at the start
    tempo_changes: int = 0
    time_signature: tuple[int, int] | None = None
    error: str | None = None
    cut: str | None = None
    problems: tuple[str, ...] = field(default=())


class SmfError(ValueError):
    """A file that is not a Standard MIDI File phasor reads (the reason)."""


# ---------------------------------------------------------------- reading


def read_smf(data: bytes) -> Song:
    """The notes of a Standard MIDI File (``data``: its bytes, at most ``MAX_FILE_BYTES``).
    Never raises: a file that cannot be read is a ``Song`` with ``error``."""
    if len(data) > MAX_FILE_BYTES:
        return Song(error=f"larger than {MAX_FILE_BYTES} bytes")
    try:
        return _read(bytes(data))
    except SmfError as e:
        return Song(error=str(e))
    except Exception as e:  # whatever else a hostile file provokes: said, never raised
        return Song(error=f"{type(e).__name__}: {e}")


_FPS = {24: 24.0, 25: 25.0, 29: 29.97, 30: 30.0}


class _Stop(Exception):
    """A bound reached: stop reading every track (``_read`` says which)."""


@dataclass
class _Found:
    raw: list[tuple[int, int, int, int, int]] = field(default_factory=list)  # on, off, p, v, ch
    tempos: list[tuple[int, int, int]] = field(default_factory=list)  # tick, order, us/quarter
    programs: list[tuple[int, int, int, int]] = field(default_factory=list)  # tick, order, ch, p
    names: list[str] = field(default_factory=list)
    signature: tuple[int, int] | None = None
    problems: list[str] = field(default_factory=list)
    events: int = 0
    notes: int = 0  # notes begun (the bound is checked before any is kept)
    order: int = 0
    stop: str | None = None


def _read(data: bytes) -> Song:
    size = len(data)
    if size < 14 or data[:4] != b"MThd":
        raise SmfError("not a Standard MIDI File (no MThd header)")
    (hlen,) = struct.unpack(">I", data[4:8])
    if hlen < 6:
        raise SmfError(f"its header chunk says {hlen} bytes (at least 6)")
    if 8 + hlen > size:
        raise SmfError(f"its header chunk says {hlen} bytes; the file has {size - 8}")
    fmt, ntrk, div = struct.unpack(">HHH", data[8:14])
    if fmt == 2:
        raise SmfError("format 2 (independent sequences) is not played: save it as format 0 or 1")
    if fmt not in (0, 1):
        raise SmfError(f"format {fmt} (0, 1 or 2)")
    if ntrk == 0:
        raise SmfError("it says it has no tracks")
    if div & 0x8000:
        code = 256 - (div >> 8)
        per_frame = div & 0xFF
        if code not in _FPS or per_frame == 0:
            raise SmfError(f"SMPTE time of {code} frames a second, {per_frame} ticks a frame "
                           "(24, 25, 29 or 30 frames; at least one tick)")  # fmt: skip
        tick_seconds: float | None = 1.0 / (_FPS[code] * per_frame)
        ppq = 0
        division = f"SMPTE {_FPS[code]:g} fps x {per_frame}"
    else:
        if div == 0:
            raise SmfError("0 ticks a quarter note")
        tick_seconds, ppq = None, div
        division = f"{div} ticks a quarter note"
    found = _Found()
    cut: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 8 + hlen
    for _ in range(MAX_CHUNKS):
        if pos + 8 > size or len(spans) >= min(ntrk, MAX_TRACKS):
            break
        kind, n = data[pos : pos + 4], struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        body = pos + 8
        if n > size - body:
            cut.append(f"track {len(spans) + 1} says {n} bytes; the file has {size - body}")
            n = size - body
        if kind == b"MTrk":
            spans.append((body, body + n))
        pos = body + n
    if ntrk > MAX_TRACKS:
        cut.append(f"{ntrk} tracks; the first {MAX_TRACKS} are read")
    elif len(spans) < ntrk:
        cut.append(f"its header says {ntrk} tracks; {len(spans)} are there")
    if not spans:
        raise SmfError("no track (MTrk) chunk")
    for k, (a, b) in enumerate(spans):
        try:
            _track(data, a, b, k, found)
        except _Stop:
            break
    if found.stop:
        cut.append(found.stop)
    cut += found.problems
    notes, bpm, changes = _timed(found, ppq, tick_seconds, cut)
    return Song(notes=notes, format=fmt, tracks=len(spans), names=tuple(found.names),
                division=division, bpm=bpm, tempo_changes=changes,
                time_signature=found.signature, cut="; ".join(cut[:4]) or None,
                problems=tuple(cut))  # fmt: skip


_LENGTHS = {0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2, 0xC0: 1, 0xD0: 1, 0xE0: 2}


def _vlq(data: bytes, i: int, b: int) -> tuple[int, int]:
    """A variable-length quantity at ``i`` (at most 4 bytes, before ``b``): (value, next i).
    ``ValueError`` when it runs past ``b`` or is longer than 4 bytes."""
    v = 0
    for _ in range(4):
        if i >= b:
            raise ValueError("cut short")
        c = data[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i
    raise ValueError("a number longer than 4 bytes")


def _track(data: bytes, a: int, b: int, k: int, f: _Found) -> None:
    """Read track ``k`` (bytes ``a`` to ``b``) into ``f``. A fault stops this track (said in
    ``f.problems``); a bound stops every track (``_Stop``, said in ``f.stop``)."""
    i, tick, status = a, 0, 0
    open_: dict[tuple[int, int], deque] = {}
    named = False
    where = f"track {k + 1}"
    try:
        while i < b:
            f.events += 1
            if f.events > MAX_EVENTS:
                f.stop = f"more than {MAX_EVENTS} events; the rest are left out"
                raise _Stop
            d, i = _vlq(data, i, b)
            tick += d
            if tick > MAX_TICK:
                raise ValueError(f"its time passes {MAX_TICK} ticks")
            if i >= b:
                raise ValueError("cut short after a delta time")
            c = data[i]
            if c == 0xFF:  # meta event
                if i + 2 > b:
                    raise ValueError("a meta event cut short")
                kind = data[i + 1]
                n, j = _vlq(data, i + 2, b)
                if n > b - j:
                    raise ValueError(f"a meta event says {n} bytes; {b - j} are left")
                body = data[j : j + n]
                i = j + n
                status = 0  # meta and system events cancel running status
                if kind == 0x2F:
                    break
                if kind == 0x51:
                    if n != 3:
                        f.problems.append(f"{where}: a tempo event of {n} bytes is ignored")
                        continue
                    us = int.from_bytes(body, "big")
                    if not 60e6 / MAX_BPM <= us <= 60e6 / MIN_BPM:
                        f.problems.append(f"{where}: a tempo of {us} us a quarter note is "
                                          "ignored")  # fmt: skip
                        continue
                    if len(f.tempos) >= MAX_TEMPOS:
                        f.stop = f"more than {MAX_TEMPOS} tempo changes; the rest are left out"
                        raise _Stop
                    f.tempos.append((tick, f.order, us))
                    f.order += 1
                elif kind == 0x03 and not named:
                    named = True
                    f.names.append(body[:64].decode("latin-1").strip())
                elif kind == 0x58 and n >= 2 and f.signature is None and body[1] <= 6:
                    f.signature = (body[0], 1 << body[1])
                continue
            if c in (0xF0, 0xF7):  # system exclusive: skipped
                n, j = _vlq(data, i + 1, b)
                if n > b - j:
                    raise ValueError(f"a system exclusive event says {n} bytes; {b - j} are left")
                i = j + n
                status = 0
                continue
            if c & 0x80:
                if c >= 0xF0:
                    raise ValueError(f"status byte 0x{c:02X} does not belong in a file")
                status = c
                i += 1
            elif status == 0:
                raise ValueError("a data byte with no status before it")
            n = _LENGTHS[status & 0xF0]
            if i + n > b:
                raise ValueError("an event cut short")
            x = data[i]
            y = data[i + 1] if n == 2 else 0
            if (x | y) & 0x80:
                raise ValueError(f"a data byte 0x{x | y:02X} past 127")
            i += n
            hi, ch = status & 0xF0, status & 0x0F
            if hi == 0x90 and y > 0:
                if f.notes >= MAX_NOTES:
                    f.stop = f"more than {MAX_NOTES} notes; the rest are left out"
                    raise _Stop
                f.notes += 1
                open_.setdefault((ch, x), deque()).append((tick, y))
            elif hi in (0x80, 0x90):
                q = open_.get((ch, x))
                if q:
                    t0, v = q.popleft()
                    f.raw.append((t0, tick, x, v, ch))
            elif hi == 0xC0:
                f.programs.append((tick, f.order, ch, x))
                f.order += 1
    except ValueError as e:
        f.problems.append(f"{where}: {e} at byte {i}; the rest of the track is skipped")
    finally:  # notes still held when the track ends end there
        for (ch, x), q in open_.items():
            for t0, v in q:
                f.raw.append((t0, tick, x, v, ch))


def _timed(f: _Found, ppq: int, tick_seconds: float | None,
           cut: list[str]) -> tuple[tuple[Note, ...], float, int]:  # fmt: skip
    """Notes in seconds through the tempo map, each with its channel's program then."""
    if not f.raw:
        bpm = 60e6 / min(f.tempos)[2] if f.tempos and min(f.tempos)[0] == 0 else 120.0
        return (), round(bpm, 6), len(f.tempos)
    raw = np.array(f.raw, dtype=np.int64)
    on, off = raw[:, 0], raw[:, 1]
    if tick_seconds is not None:  # SMPTE time: ticks are seconds already
        t_on, t_off = on * tick_seconds, off * tick_seconds
        bpm = 120.0
    else:
        tempos = sorted(f.tempos)
        ticks = [0]
        us = [500000]  # 120 beats a minute until the file says otherwise
        for t, _, u in tempos:
            if t == ticks[-1]:
                us[-1] = u
            else:
                ticks.append(t)
                us.append(u)
        tk = np.array(ticks, dtype=np.float64)
        per = np.array(us, dtype=np.float64) / 1e6 / ppq  # seconds a tick, from each change
        at = np.concatenate([[0.0], np.cumsum(np.diff(tk) * per[:-1])])

        def seconds(x: np.ndarray) -> np.ndarray:
            j = np.searchsorted(tk, x, side="right") - 1
            return at[j] + (x - tk[j]) * per[j]

        t_on, t_off = seconds(on.astype(np.float64)), seconds(off.astype(np.float64))
        bpm = 60e6 / us[0]
    length = np.minimum(t_off - t_on, MAX_NOTE_SECONDS)
    if np.any(t_off - t_on > MAX_NOTE_SECONDS):
        cut.append(f"notes longer than {MAX_NOTE_SECONDS:g} s are cut to it")
    late = t_on > MAX_SECONDS
    if late.any():
        cut.append(f"{int(late.sum())} notes after {MAX_SECONDS:g} s are left out")
    keep = (length > 0) & ~late
    progs = _programs(f.programs, raw[:, 4], on)
    order = np.lexsort((raw[:, 2], raw[:, 4], t_on))  # by start, then channel, then pitch
    notes = tuple(Note(float(raw[k, 2]), float(t_on[k]), float(length[k]), int(raw[k, 3]),
                       int(raw[k, 4]), int(progs[k])) for k in order if keep[k])  # fmt: skip
    return notes, round(bpm, 6), len(f.tempos)


def _programs(changes: list[tuple[int, int, int, int]], channel: np.ndarray,
              tick: np.ndarray) -> np.ndarray:  # fmt: skip
    """The program each note's channel has at its tick (0 before any change)."""
    out = np.zeros(len(channel), dtype=np.int64)
    by: dict[int, list[tuple[int, int, int]]] = {}
    for t, o, ch, p in changes:
        by.setdefault(ch, []).append((t, o, p))
    for ch, lst in by.items():
        lst.sort()
        ts = np.array([t for t, _, _ in lst], dtype=np.int64)
        ps = np.array([p for _, _, p in lst], dtype=np.int64)
        m = channel == ch
        j = np.searchsorted(ts, tick[m], side="right") - 1
        out[m] = np.where(j >= 0, ps[np.maximum(j, 0)], 0)
    return out


# ---------------------------------------------------------------- writing

PPQ = 960  # ticks a quarter note in files phasor writes
BPM = 120.0  # their tempo: a tick is 1/1920 s


def _vlq_bytes(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F))
        v >>= 7
    return bytes(reversed(out))


def _chunk(kind: bytes, body: bytes) -> bytes:
    return kind + struct.pack(">I", len(body)) + body


def _meta(kind: int, body: bytes) -> bytes:
    return bytes([0xFF, kind]) + _vlq_bytes(len(body)) + body


def write_smf(tracks: Sequence[tuple[str, Iterable[Note]]], *, bpm: float = BPM,
              ppq: int = PPQ, title: str = "") -> bytes:  # fmt: skip
    """``tracks`` (a name and its notes each) as a format 1 Standard MIDI File: a conductor
    track (``title``, the tempo, 4/4), then one track per entry. Times are rounded to ticks
    (1/1920 s at the defaults), pitches to whole semitones and into 0-127, velocities into
    1-127; a note lasts at least one tick. Each channel's program is set before its first
    note and again where a note asks for another one."""
    us = round(60e6 / bpm)
    per_second = ppq * bpm / 60.0
    conductor = _meta(0x03, title.encode("latin-1", "replace")[:127]) if title else b""
    conductor += b"\x00" + _meta(0x51, us.to_bytes(3, "big"))
    conductor += b"\x00" + _meta(0x58, bytes([4, 2, 24, 8]))
    conductor += b"\x00" + _meta(0x2F, b"")
    if title:
        conductor = b"\x00" + conductor
    chunks = [_chunk(b"MTrk", conductor)]
    for name, notes in tracks:
        events: list[tuple[int, int, int, bytes]] = []  # tick, kind (off, program, on), seq
        program: dict[int, int] = {}
        for seq, n in enumerate(notes):
            ch = min(max(int(n.channel), 0), 15)
            p = min(max(round(float(n.pitch)), 0), 127)
            v = min(max(int(n.velocity), 1), 127)
            t0 = max(0, round(float(n.start) * per_second))
            t1 = max(t0 + 1, round((float(n.start) + float(n.length)) * per_second))
            prog = min(max(int(n.program), 0), 127)
            if ch != DRUMS and program.get(ch) != prog:
                program[ch] = prog
                events.append((t0, 1, seq, bytes([0xC0 | ch, prog])))
            events.append((t0, 2, seq, bytes([0x90 | ch, p, v])))
            events.append((t1, 0, seq, bytes([0x80 | ch, p, 64])))
        events.sort(key=lambda e: (e[0], e[1], e[2]))
        body = bytearray()
        if name:
            body += b"\x00" + _meta(0x03, name.encode("latin-1", "replace")[:127])
        last, status = 0, 0
        for tick, _, _, msg in events:
            body += _vlq_bytes(tick - last)
            last = tick
            if msg[0] == status:  # running status
                body += msg[1:]
            else:
                body += msg
                status = msg[0]
        body += b"\x00" + _meta(0x2F, b"")
        chunks.append(_chunk(b"MTrk", bytes(body)))
    head = _chunk(b"MThd", struct.pack(">HHH", 1, len(chunks), ppq))
    return head + b"".join(chunks)


# ---------------------------------------------------------------- the synthesizer

RATE = 48000  # the synthesizer's sample rate; the mixer resamples to the track's
MAX_VOICES = 64  # notes sounding at once (with their release); later ones are dropped
_SUB = 1 << 16  # samples made at a time inside one read (bounded temporaries)
_TABLE_BITS = 12
_TABLE = 1 << _TABLE_BITS  # samples in a waveform table
_FRAC_BITS = 32 - _TABLE_BITS
_EXP_SPAN = 16.0  # the decay table covers exp(-u) for u in [0, 16]
_NOISE = 1 << 16  # samples in a drum's noise table
LEVEL = 0.25  # a voice at full velocity peaks here (several can play without clipping)

INSTRUMENTS = ("auto", "sine", "pluck", "organ", "bell", "pad", "lead")


@dataclass(frozen=True)
class Timbre:
    """A voice: harmonic amplitudes (``partials``: ratio -> amplitude; whole ratios make one
    table, others are added as sines), and an envelope: ``attack`` seconds up, then down to
    ``sustain`` with time constant ``decay`` (``sustain`` 0: a plucked string dying away,
    slower for low notes), and ``release`` seconds after the note ends."""

    partials: tuple[tuple[float, float], ...]
    attack: float
    decay: float
    sustain: float
    release: float
    detune: float = 0.0  # a second oscillator this many cents up (a chorus), 0: none


TIMBRES = {
    "sine": Timbre(((1, 1.0),), 0.01, 0.3, 0.8, 0.12),
    "pluck": Timbre(tuple((n, 1.0 / n**1.6) for n in range(1, 25)), 0.004, 1.1, 0.0, 0.12),
    "organ": Timbre(((1, 1.0), (2, 0.8), (3, 0.6), (4, 0.5), (6, 0.3), (8, 0.25)), 0.02,
                    0.05, 0.9, 0.08),
    "bell": Timbre(((1, 1.0), (2.76, 0.6), (5.4, 0.4), (8.93, 0.25)), 0.002, 1.6, 0.0, 0.3),
    "pad": Timbre(tuple((n, 1.0 / n) for n in range(1, 13)), 0.3, 0.5, 0.7, 0.6, detune=7.0),
    "lead": Timbre(tuple((n, 1.0 / n) for n in range(1, 21)), 0.02, 0.2, 0.7, 0.1),
}  # fmt: skip


def timbre_of(program: int, channel: int, instrument: str = "auto") -> str:
    """The built-in voice for a General MIDI ``program`` (or the one ``instrument`` names);
    channel 10 (9 from 0) is always the drum kit."""
    if channel == DRUMS:
        return "drums"
    if instrument != "auto":
        return instrument
    family = program // 8
    return {0: "pluck", 1: "bell", 2: "organ", 3: "pluck", 4: "pluck", 5: "pad", 6: "pad",
            7: "lead", 8: "lead", 9: "lead", 10: "lead", 11: "pad", 12: "bell", 13: "pluck",
            14: "bell", 15: "sine"}[family]  # fmt: skip


@functools.lru_cache(maxsize=64)
def _wave(name: str, top: int) -> np.ndarray:
    """One cycle of ``name``'s whole-ratio partials up to ratio ``top`` (band limit), peak 1,
    with one guard sample (``_TABLE + 1``)."""
    x = np.arange(_TABLE + 1, dtype=np.float64) * (2 * np.pi / _TABLE)
    w = np.zeros(_TABLE + 1)
    for r, amp in TIMBRES[name].partials if name in TIMBRES else ((1, 1.0),):
        if float(r).is_integer() and r <= top:
            w += amp * np.sin(int(r) * x)
    peak = np.abs(w).max()
    return w / peak if peak > 0 else w


@functools.lru_cache(maxsize=1)
def _exp_table() -> np.ndarray:
    return np.exp(-np.linspace(0.0, _EXP_SPAN, _TABLE + 1))


def _decay(u: np.ndarray) -> np.ndarray:
    """exp(-u) for u >= 0 from the fixed table (0 past ``_EXP_SPAN``), linear in between."""
    g = np.minimum(u, _EXP_SPAN) * (_TABLE / _EXP_SPAN)
    i = np.minimum(g.astype(np.int64), _TABLE - 1)
    t = _exp_table()
    out = t[i] + (t[i + 1] - t[i]) * (g - i)
    return np.where(u >= _EXP_SPAN, 0.0, out)


def _lookup(table: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """The table at integer phases (32-bit fractions of a cycle), linearly interpolated."""
    i = (phase >> _FRAC_BITS).astype(np.int64)
    frac = (phase & ((1 << _FRAC_BITS) - 1)).astype(np.float64) * (1.0 / (1 << _FRAC_BITS))
    return table[i] + (table[i + 1] - table[i]) * frac


def _phase(k: np.ndarray, inc: int) -> np.ndarray:
    """Integer phase at note-local samples ``k`` for a step of ``inc`` / 2^32 cycle."""
    return (k.astype(np.uint64) * np.uint64(inc)) & np.uint64(0xFFFFFFFF)


def _inc(freq: float) -> int:
    return round(freq / RATE * 4294967296.0) & 0xFFFFFFFF


# drum kinds by GM percussion key: (kind, seconds it rings, decay seconds, tone Hz)
_KICK, _SNARE, _HAT, _CYMBAL, _TOM, _TICK = range(6)


def _drum(pitch: int) -> tuple[int, float, float, float]:
    if pitch in (35, 36):
        return _KICK, 0.5, 0.15, 50.0
    if pitch in (37, 38, 39, 40):
        return _SNARE, 0.3, 0.06, 185.0
    if pitch in (42, 44):
        return _HAT, 0.12, 0.02, 0.0
    if pitch == 46:
        return _HAT, 0.6, 0.15, 0.0
    if pitch in (49, 51, 52, 53, 55, 57, 59):
        return _CYMBAL, 2.0, 0.45, 0.0
    if pitch in (41, 43, 45, 47, 48, 50):
        return _TOM, 0.5, 0.18, 80.0 * 2 ** ((pitch - 41) / 12)
    return _TICK, 0.15, 0.03, 0.0


@dataclass(frozen=True)
class _Voice:
    first: int  # the track sample (at ``RATE``) the note starts on
    held: int  # samples it is held
    total: int  # samples it sounds, with its release (or a drum's ring)
    timbre: str
    freq: float
    amp: float
    pitch: int
    noise_at: int  # where in the noise table a drum hit starts


class Synth:
    """Notes as a ``time.narration.Source`` at ``RATE``: ``frames`` covers the last note's
    release (at most ``MAX_SECONDS``); ``read(skip, count)`` makes those samples a window
    at a time. At most ``MAX_VOICES`` notes sound at once (a note starting when that many
    are sounding is dropped: ``dropped``). ``noise``: the drum kit's noise (``rng(id,
    seed)``: the only randomness)."""

    def __init__(self, notes: Sequence[Note], instrument: str = "auto",
                 noise: np.ndarray | None = None) -> None:  # fmt: skip
        voices: list[_Voice] = []
        for j, n in enumerate(notes):
            ch, pitch = int(n.channel), float(n.pitch)
            name = timbre_of(int(n.program), ch, instrument)
            first = round(float(n.start) * RATE)
            held = max(1, round(float(n.length) * RATE))
            v = min(max(int(n.velocity), 1), 127)
            amp = LEVEL * v * v / (127 * 127)
            if name == "drums":
                _, ring, _, _ = _drum(round(pitch))
                total, freq = round(ring * RATE), 0.0
            else:
                total = held + round(TIMBRES[name].release * RATE)
                freq = 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)
            voices.append(_Voice(first, held, total, name, freq, amp, round(pitch),
                                 (j * 7919) % _NOISE))  # fmt: skip
        voices.sort(key=lambda v: v.first)  # stable: notes keep their order at one start
        kept, ends = [], []  # min-heap of the kept voices' last samples
        self.dropped = 0
        self.dropped_at: float | None = None  # seconds: the first note dropped
        for v in voices:
            while ends and ends[0] <= v.first:
                heapq.heappop(ends)
            if len(ends) >= MAX_VOICES:
                self.dropped += 1
                if self.dropped_at is None:
                    self.dropped_at = v.first / RATE
                continue
            heapq.heappush(ends, v.first + v.total)
            kept.append(v)
        self.voices = tuple(kept)
        self._starts = [v.first for v in kept]
        self._longest = max((v.total for v in kept), default=0)
        self.frames = min(max((v.first + v.total for v in kept), default=0),
                          int(MAX_SECONDS * RATE))  # fmt: skip
        self.rate = RATE
        self.noise = np.zeros(_NOISE) if noise is None else np.asarray(noise, np.float64)

    def read(self, skip: int, count: int) -> np.ndarray:
        """Samples ``skip`` to ``skip + count`` (cut at ``frames``) as float32 in -1..1."""
        skip, count = max(0, int(skip)), max(0, min(int(count), self.frames - int(skip)))
        out = np.zeros(count, np.float64)
        for a in range(skip, skip + count, _SUB):
            b = min(skip + count, a + _SUB)
            lo = bisect.bisect_left(self._starts, a - self._longest)
            hi = bisect.bisect_left(self._starts, b)
            for v in self.voices[lo:hi]:  # one fixed order: the same sum in any window
                s, e = max(a, v.first), min(b, v.first + v.total)
                if e > s:
                    k = np.arange(s - v.first, e - v.first, dtype=np.int64)
                    out[s - skip : e - skip] += self._voice(v, k)
        np.clip(out, -1.0, 1.0, out=out)
        return out.astype(np.float32)

    def _voice(self, v: _Voice, k: np.ndarray) -> np.ndarray:
        if v.timbre == "drums":
            return v.amp * 2.0 * self._hit(v, k)
        t = TIMBRES[v.timbre]
        top = max(1, int(0.45 * RATE // max(v.freq, 1e-9)))
        sound = _lookup(_wave(v.timbre, top), _phase(k, _inc(v.freq)))
        for r, a in t.partials:  # inharmonic partials (a bell), each dying faster
            if not float(r).is_integer() and r * v.freq < 0.45 * RATE:
                part = _lookup(_wave("sine", 1), _phase(k, _inc(v.freq * r)))
                sound = sound + a * part * _decay(k * (r / (t.decay * RATE)))
        if t.detune:
            f2 = v.freq * 2.0 ** (t.detune / 1200.0)
            top2 = max(1, int(0.45 * RATE // f2))
            sound = 0.5 * (sound + _lookup(_wave(v.timbre, top2), _phase(k, _inc(f2))))
        return v.amp * sound * self._envelope(t, v, k)

    @staticmethod
    def _envelope(t: Timbre, v: _Voice, k: np.ndarray) -> np.ndarray:
        """Attack, decay to sustain while held; from the level at release down to 0 over
        ``release`` (an exponential fade times a straight line, so it ends at exactly 0)."""
        attack = max(1, round(t.attack * RATE))
        tau = t.decay * RATE
        if t.sustain == 0:  # a plucked string: low notes ring longer
            tau *= min(4.0, max(0.25, (261.6 / max(v.freq, 1.0)) ** 0.5))

        def held(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float64)
            rise = x / attack
            fall = t.sustain + (1 - t.sustain) * _decay(np.maximum(x - attack, 0) / tau)
            return np.where(x < attack, rise, fall)

        out = held(np.minimum(k, v.held))
        rel = v.total - v.held
        if rel > 0:
            past = k >= v.held
            if past.any():
                at_off = held(np.array([v.held]))[0]
                x = (k[past] - v.held).astype(np.float64) / rel
                out[past] = at_off * _decay(4.6 * x) * np.maximum(1.0 - x, 0.0)
        return out

    def _hit(self, v: _Voice, k: np.ndarray) -> np.ndarray:
        kind, _, decay, tone = _drum(v.pitch)
        env = _decay(k / (decay * RATE))
        idx = (k + v.noise_at) % _NOISE
        noise = self.noise[idx]
        if kind == _KICK:  # a sine falling from 150 Hz to its tone
            tau = 0.04 * RATE
            cycles = (tone * k + (150.0 - tone) * tau * (1.0 - _decay(k / tau))) / RATE
            frac = cycles - np.floor(cycles)
            sine = _wave("sine", 1)
            g = frac * _TABLE
            i = np.minimum(g.astype(np.int64), _TABLE - 1)
            body = sine[i] + (sine[i + 1] - sine[i]) * (g - i)
            return env * (body + 0.1 * noise * _decay(k / (0.004 * RATE)))
        if kind == _TOM:
            body = _lookup(_wave("sine", 1), _phase(k, _inc(tone)))
            return env * body
        if kind == _SNARE:
            body = _lookup(_wave("sine", 1), _phase(k, _inc(tone)))
            return 0.6 * noise * env + 0.4 * body * _decay(k / (0.04 * RATE))
        if kind in (_HAT, _CYMBAL):  # noise with its low end taken out (a first difference)
            before = self.noise[(idx - 1) % _NOISE]
            return 0.5 * (noise - before) * env
        return noise * env


def drum_noise(rng) -> np.ndarray:
    """The drum kit's noise table, from ``core.rng.rng(object_id, seed)``."""
    return rng.random(_NOISE) * 2.0 - 1.0
