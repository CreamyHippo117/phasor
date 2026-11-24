"""MIDI notes as an object (Milestone 23, STUDIO.md): one type with an import end, a play end
and an export end.

``Notes`` holds notes from a ``.mid`` file in the scene's folder (read through
``core/files.py``, re-read when it changes) or from the script (``notes=[(pitch, start,
length, velocity[, channel]), ...]``: any code can make them, say the partials of a Fourier
series). ``phasor render`` plays them with a small built-in synthesizer
(``midi_math.Synth``), mixed into the video's sound from ``start`` (seconds, or a marker's
name) exactly as a ``Sound`` clip is. ``Engine.export_midi`` writes every ``Notes`` of the
scene to one ``.mid``. Nothing is drawn unless ``show``: a piano roll in a box.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core import files
from phasor.core.ids import is_valid_name
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, register
from phasor.core.rng import rng
from phasor.core.state import SceneState
from phasor.domains import midi_math as mm
from phasor.domains.strokes import segment_batch, with_alpha

MIDI_EXT = (".mid", ".midi", ".smf")
MAX_SCRIPT_NOTES = 20_000  # notes a script may list in ``notes`` (the document holds them)
MAX_START = 1e6  # seconds
MIN_GAIN, MAX_GAIN = -60.0, 24.0  # dB
MAX_REPEAT = 64


def _note(row: Any, k: int) -> tuple[float, ...]:
    """One script note, checked: (pitch, start, length, velocity[, channel])."""
    if not isinstance(row, list | tuple) or len(row) not in (4, 5):
        raise ValueError(f"note {k}: (pitch, start, length, velocity[, channel]), not {row!r}")
    try:
        vals = [float(x) for x in row]
    except (TypeError, ValueError):
        raise ValueError(f"note {k}: numbers only, not {row!r}") from None
    pitch, start, length, vel = vals[:4]
    ch = vals[4] if len(vals) == 5 else 0.0
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"note {k}: not a finite number in {row!r}")
    if not 0 <= pitch <= 127:
        raise ValueError(f"note {k}: pitch {pitch:g} (0 to 127; 60 is middle C)")
    if not 0 <= start <= MAX_START:
        raise ValueError(f"note {k}: start {start:g} s (0 to {MAX_START:g})")
    if not 0 < length <= mm.MAX_NOTE_SECONDS:
        raise ValueError(f"note {k}: length {length:g} s (more than 0, at most "
                         f"{mm.MAX_NOTE_SECONDS:g})")  # fmt: skip
    if not 1 <= vel <= 127 or not vel.is_integer():
        raise ValueError(f"note {k}: velocity {vel:g} (a whole number, 1 to 127)")
    if not 0 <= ch <= 15 or not ch.is_integer():
        raise ValueError(f"note {k}: channel {ch:g} (a whole number, 0 to 15; 9 is drums)")
    return (pitch, start, length, vel, ch) if len(vals) == 5 else (pitch, start, length, vel)


@register
class Notes(ObjectBase):
    """MIDI notes, played by the built-in synthesizer and mixed into the video's sound like
    a ``Sound`` clip, from ``start`` (seconds into the scene, or a marker's name: the notes
    follow the marker when it moves), ``gain`` dB louder. The notes come from ``path`` (a
    ``.mid`` in the scene's folder: format 0 or 1) or from ``notes`` (the script's list of
    ``(pitch, start, length, velocity[, channel])``, seconds from the clip's start), never
    both. ``transpose`` moves them by semitones, ``tempo_scale`` plays them faster (2: twice
    as fast), ``channels`` keeps only those channels (0-15), ``repeat`` plays them that many
    times end to end. ``instrument`` ``auto`` gives each note the voice of its General MIDI
    program (``program`` for script notes); or one of ``sine``, ``pluck``, ``organ``,
    ``bell``, ``pad``, ``lead`` for all of them. Channel 10 (9 from 0) is the drum kit,
    whose noise comes from ``rng(id, seed)``. ``show`` draws a piano roll ``width`` x
    ``height`` centred on ``pos`` (otherwise nothing is drawn). A missing file is lint
    ``missing_file``, one that is not MIDI ``bad_file``, one read only in part
    ``truncated``; notes past the scene's end, notes dropped past ``MAX_VOICES`` at once
    and an unknown marker are ``notes_limit``."""

    type: ClassVar[str] = "notes"
    Files: ClassVar = ("path",)  # in the scene's folder (core/files.py)
    Life: ClassVar = frozenset()  # notes are a score: life never moves them
    Fixed: ClassVar = ("path", "notes", "start", "gain", "transpose", "tempo_scale",
                       "instrument", "program", "channels", "repeat", "seed")  # fmt: skip
    Style: ClassVar = {
        "color": ("emphasis", "primary"),  # the notes of the piano roll
        "played": ("emphasis", "highlight"),  # notes sounding now, and the playhead
        "frame": ("emphasis", "muted"),
        "stroke": ("stroke", "bold"),
        "glow": ("glow", "soft"),
    }

    class Params(ParamsBase):
        path: str = ""  # a .mid file in the scene's folder, instead of ``notes``
        notes: tuple[tuple[float, ...], ...] = ()  # (pitch, start, length, velocity[, ch])
        start: float | str = 0.0  # seconds into the scene, or a marker's name
        gain: float = Field(0.0, ge=MIN_GAIN, le=MAX_GAIN)  # dB
        transpose: int = Field(0, ge=-48, le=48)  # semitones
        tempo_scale: float = Field(1.0, ge=0.1, le=10.0)  # 2: twice as fast
        instrument: Literal["auto", "sine", "pluck", "organ", "bell", "pad", "lead"] = "auto"
        program: int = Field(0, ge=0, le=127)  # General MIDI program of script notes
        channels: tuple[int, ...] = ()  # keep only these channels (0-15); empty: all
        repeat: int = Field(1, ge=1, le=MAX_REPEAT)
        seed: int = Field(0, ge=0, le=2**31 - 1)  # the drum kit's noise
        show: bool = False  # draw a piano roll
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # the piano roll's centre
        width: float = Field(6.0, gt=0, le=1e4)
        height: float = Field(2.0, gt=0, le=1e4)

        @field_validator("path")
        @classmethod
        def _midi_file(cls, v: str) -> str:
            if not v:
                return v
            if not v.lower().endswith(MIDI_EXT):
                raise ValueError(f"{v!r} is not a MIDI file ({', '.join(MIDI_EXT)})")
            return files.relative_path(v)

        @field_validator("notes", mode="before")
        @classmethod
        def _script_notes(cls, v: Any) -> Any:
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if not isinstance(v, list | tuple):
                raise ValueError("notes is a list of (pitch, start, length, velocity[, channel])")
            if len(v) > MAX_SCRIPT_NOTES:
                raise ValueError(f"{len(v)} notes; a script lists at most {MAX_SCRIPT_NOTES} "
                                 "(put more in a .mid file)")  # fmt: skip
            return tuple(_note(row, k) for k, row in enumerate(v))

        @field_validator("start")
        @classmethod
        def _start(cls, v: float | str) -> float | str:
            if isinstance(v, str):
                if not is_valid_name(v):
                    raise ValueError(f"start {v!r}: seconds, or a marker's name")
                return v
            if not (math.isfinite(v) and 0 <= v <= MAX_START):
                raise ValueError(f"start {v} s (0 to {MAX_START:g})")
            return v

        @field_validator("channels")
        @classmethod
        def _channels(cls, v: tuple[int, ...]) -> tuple[int, ...]:
            if any(not 0 <= c <= 15 for c in v):
                raise ValueError(f"channels {v}: each 0 to 15 (9 is drums)")
            return tuple(sorted(set(v)))

        @model_validator(mode="after")
        def _one_source(self):
            if self.path and self.notes:
                raise ValueError("give notes or a path to a .mid file, not both")
            return self

    # ------------------------------------------------------------ the notes

    def song(self, state: SceneState) -> mm.Song | None:
        """The file's ``Song`` (None: no file named, or it is not there)."""
        p = self.me(state).params
        if not p.path:
            return None
        stamp = files.stamp(state.base_dir, p.path)
        if stamp[1] is None:
            return None
        return _read_song(stamp)

    def placed(self, state: SceneState) -> _Placed:
        """The notes as heard, in seconds from the clip's start (transposed, scaled,
        filtered, repeated), with the synthesizer that plays them; cached."""
        p = self.me(state).params
        stamp = files.stamp(state.base_dir, p.path) if p.path else None
        key = (self.id, stamp, _listed(p.notes), p.transpose, p.tempo_scale, p.instrument,
               p.program, p.channels, p.repeat, p.seed, p.gain)  # fmt: skip
        return _placed(key, p.notes)

    def midi_notes(self, state: SceneState) -> tuple[str, tuple[mm.Note, ...], float | str]:
        """What ``Engine.export_midi`` writes: (track name, notes from the clip's start,
        ``start``). With an ``instrument`` other than ``auto`` each melodic note carries that
        voice's program (``midi_math.PROGRAM_OF``), so it plays the same when read back."""
        p = self.me(state).params
        notes = self.placed(state).notes
        if p.instrument != "auto":
            prog = mm.PROGRAM_OF[p.instrument]
            notes = tuple(n if n.channel == mm.DRUMS else n._replace(program=prog) for n in notes)
        return self.id, notes, p.start

    # ------------------------------------------------------------ contract members

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        pl = self.placed(state)
        out: dict[str, Any] = {"kind": "notes", "source": "file" if p.path else "script",
                               "start": p.start, "gain": p.gain, "count": len(pl.notes),
                               "instrument": p.instrument,
                               "drawn": 1.0 if p.show else 0.0}  # fmt: skip
        out.update({k: list(v) if isinstance(v, list) else v for k, v in pl.summary.items()})
        if pl.dropped:
            out["voices_dropped"] = pl.dropped
        if pl.out_of_range:
            out["out_of_range"] = pl.out_of_range  # transposed past 0-127: not heard
        if p.path:
            out["file"] = p.path
            out["missing"] = not files.exists(state.base_dir, p.path)
            song = self.song(state)
            if song is not None and song.error:
                out["unreadable"] = song.error  # lint bad_file
            elif song is not None:
                out.update(format=song.format, tracks=song.tracks, bpm=song.bpm)
                if any(song.names):
                    out["names"] = [n for n in song.names if n]
                if song.time_signature:
                    out["time_signature"] = list(song.time_signature)
        if pl.cut:
            out["truncated"] = pl.cut  # lint truncated
        return out

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        if not p.show:
            return Bounds((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        x, y, z = p.pos
        return Bounds(
            (x - p.width / 2, y - p.height / 2, z), (x + p.width / 2, y + p.height / 2, z)
        )

    def audio_clip(self, state: SceneState):
        """What ``Engine.audio_track`` mixes: the synthesizer as a clip
        (``time.narration.Clip``; a marker ``start`` is resolved by the engine)."""
        from phasor.time.narration import Clip

        p = self.me(state).params
        pl = self.placed(state)
        if not pl.notes:
            return None
        marker = p.start if isinstance(p.start, str) else None
        start = 0.0 if marker else float(p.start)
        # gain is the synthesizer's (before its limiter), not the mix's
        return Clip(path=p.path or self.id, start=start, gain=0.0, source=pl.synth,
                    marker=marker)  # fmt: skip

    def note_report(self, state: SceneState) -> dict[str, Any]:
        """What lint ``notes_limit`` judges: the marker (or seconds) the notes start at, each
        note's start from there, and where the synthesizer dropped one (seconds)."""
        p = self.me(state).params
        pl = self.placed(state)
        return {"start": p.start, "starts": pl.table[:, 1],
                "dropped": pl.dropped, "dropped_at": pl.dropped_at, "over_work": pl.over_work,
                "work": pl.work, "voices": mm.MAX_VOICES,
                "max_work": mm.MAX_SYNTH_WORK}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        """A piano roll when ``show``: each note a bar (time across, pitch up) in a frame;
        with ``start`` in seconds, the notes sounding at ``t`` light up and a playhead
        moves across. Without ``show``, nothing: notes are heard, not seen."""
        me = self.me(state)
        p = me.params
        pl = self.placed(state)
        if not p.show or p.opacity <= 0:
            return
        cam = state.camera_for(me)
        x0, y0, z = p.pos[0] - p.width / 2, p.pos[1] - p.height / 2, p.pos[2]
        corners = np.array([[x0, y0, z], [x0 + p.width, y0, z], [x0 + p.width, y0 + p.height, z],
                            [x0, y0 + p.height, z]])  # fmt: skip
        frame = with_alpha(me.style["frame"], p.opacity)
        segment_batch(ctx, cam, corners, np.roll(corners, -1, axis=0), me.style["stroke"] * 0.4,
                      colors=frame, glow=1.0)  # fmt: skip
        if not pl.notes:
            return
        a = pl.table
        span = max(float(a[:, 2].max()), 1e-6)
        lo, hi = float(a[:, 0].min()) - 1, float(a[:, 0].max()) + 1
        pad = 0.04 * p.width
        xs = x0 + pad + (p.width - 2 * pad) * a[:, 1:] / span
        ys = y0 + p.height * (0.08 + 0.84 * (a[:, 0] - lo) / max(hi - lo, 1.0))
        on = np.zeros(len(a), bool)
        if not isinstance(p.start, str):
            local = state.t - float(p.start)
            on = (a[:, 1] <= local) & (local < a[:, 2])
            if 0 <= local <= span:
                px = x0 + pad + (p.width - 2 * pad) * local / span
                segment_batch(ctx, cam, np.array([[px, y0, z]]), np.array([[px, y0 + p.height, z]]),
                              me.style["stroke"] * 0.5, colors=with_alpha(me.style["played"],
                              p.opacity), glow=float(me.style["glow"]))  # fmt: skip
        base = np.array(with_alpha(me.style["color"], p.opacity))
        lit = np.array(with_alpha(me.style["played"], p.opacity))
        cols = np.where(on[:, None], lit, base)
        ends_a = np.stack([xs[:, 0], ys, np.full(len(ys), z)], axis=1)
        ends_b = np.stack([np.maximum(xs[:, 1], xs[:, 0] + 1e-3), ys, np.full(len(ys), z)],
                          axis=1)  # fmt: skip
        segment_batch(ctx, cam, ends_a, ends_b, me.style["stroke"], colors=cols,
                      glow=float(me.style["glow"]))  # fmt: skip


class _Placed:
    """The notes of one ``Notes`` as heard, and their synthesizer."""

    def __init__(self, notes: tuple[mm.Note, ...], synth: mm.Synth, cut: str | None,
                 out_of_range: int, instrument: str) -> None:  # fmt: skip
        self.notes, self.synth, self.cut, self.out_of_range = notes, synth, cut, out_of_range
        self.dropped = synth.dropped
        self.dropped_at = synth.dropped_at
        self.over_work = synth.over_work
        self.work = synth.work
        # (pitch, start, end) a row: what the semantics, the roll and lint read, made once
        self.table = np.array([(n.pitch, n.start, n.start + n.length) for n in notes],
                              dtype=np.float64).reshape(-1, 3)  # fmt: skip
        self.summary: dict[str, Any] = {}
        if notes:
            kinds = {(n.channel == mm.DRUMS, n.program, n.channel) for n in notes}
            self.summary = {
                "span": [round(float(self.table[:, 1].min()), 6),
                         round(float(self.table[:, 2].max()), 6)],
                "pitch_range": [float(self.table[:, 0].min()), float(self.table[:, 0].max())],
                "channels": sorted({c for _, _, c in kinds}),
                "programs": sorted({"Drum kit" if d else mm.GM_NAMES[p] for d, p, _ in kinds}),
                "voices": sorted({mm.timbre_of(p, c, instrument) for _, p, c in kinds}),
            }  # fmt: skip


CACHED_NOTES = 400_000  # notes kept placed (with their synthesizers) across objects
CACHED_SONGS = 256  # files kept read (also bounded by CACHED_NOTES notes in all)
_cache: OrderedDict[tuple, _Placed] = OrderedDict()
_songs: OrderedDict[tuple, mm.Song] = OrderedDict()
_lists: OrderedDict[int, tuple[tuple, str]] = OrderedDict()
_lock = threading.Lock()


def _read_song(stamp: tuple) -> mm.Song:
    """The ``.mid`` file ``stamp`` names, cached by its path, time and size (read again when
    it changes): the most recently used files are kept while they hold at most
    ``CACHED_NOTES`` notes and ``CACHED_SONGS`` files (at least one)."""
    with _lock:
        hit = _songs.get(stamp)
        if hit is not None:
            _songs.move_to_end(stamp)
            return hit
    try:
        song = mm.read_smf(files.read_bytes(stamp[0], mm.MAX_FILE_BYTES))
    except ValueError as e:
        song = mm.Song(error=str(e))
    with _lock:
        _songs[stamp] = song
        while len(_songs) > 1 and (
            len(_songs) > CACHED_SONGS or sum(len(v.notes) for v in _songs.values()) > CACHED_NOTES
        ):
            _songs.popitem(last=False)
    return song


def _listed(notes: tuple) -> str:
    """A digest of a script's notes, made once per notes tuple (a cache key need not hash
    20 000 rows at every call); the tuple is kept with it, so its id is never reused."""
    import hashlib

    with _lock:
        hit = _lists.get(id(notes))
        if hit is not None and hit[0] is notes:
            return hit[1]
    digest = hashlib.sha256(repr(notes).encode()).hexdigest()
    with _lock:
        _lists[id(notes)] = (notes, digest)
        while len(_lists) > 512:
            _lists.popitem(last=False)
    return digest


def _placed(key: tuple, script: tuple = ()) -> _Placed:
    """``_place(key)``, cached: the most recently used are kept while they hold at most
    ``CACHED_NOTES`` notes in all (at least one), so many small objects all stay cached and
    a few huge ones cost a bounded amount of memory."""
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    made = _place(key, script)
    with _lock:
        _cache[key] = made
        while len(_cache) > 1 and sum(len(v.notes) for v in _cache.values()) > CACHED_NOTES:
            _cache.popitem(last=False)
    return made


def _place(key: tuple, script: tuple) -> _Placed:
    (
        object_id,
        stamp,
        _listed_digest,
        transpose,
        scale,
        instrument,
        program,
        channels,
        repeat,
        seed,
        gain,
    ) = key
    song = _read_song(stamp) if stamp is not None and stamp[1] is not None else None
    cut = song.cut if song is not None else None
    if song is not None:
        base = list(song.notes)
    else:
        base = [mm.Note(r[0], r[1], r[2], int(r[3]), int(r[4]) if len(r) == 5 else 0,
                        program) for r in script]  # fmt: skip
    if channels:
        base = [n for n in base if n.channel in channels]
    out_of_range = 0
    moved = []
    for n in base:
        pitch = n.pitch + (transpose if n.channel != mm.DRUMS else 0)
        if not 0 <= pitch <= 127:
            out_of_range += 1
            continue
        moved.append(n._replace(pitch=pitch, start=n.start / scale, length=n.length / scale))
    period = max((n.start + n.length for n in moved), default=0.0)
    notes = []
    for r in range(repeat):
        if len(notes) + len(moved) > mm.MAX_NOTES:
            cut = "; ".join(x for x in (cut, f"repeated past {mm.MAX_NOTES} notes: "
                                        f"{r} of {repeat} repeats are played") if x)  # fmt: skip
            break
        notes += [n._replace(start=n.start + r * period) for n in moved]
    notes.sort(key=lambda n: (n.start, n.channel, n.pitch))
    synth = mm.Synth(
        notes, instrument, mm.drum_noise(rng(object_id, seed)), gain=10.0 ** (gain / 20.0)
    )
    return _Placed(tuple(notes), synth, cut, out_of_range, instrument)
