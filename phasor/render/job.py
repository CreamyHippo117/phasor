"""Render a scene to a video file or one still: the one place that loop lives.

The CLI (`phasor render`) and the MCP `render` tool both call this. ``engine`` is
anything with ``document``, ``duration``, ``frame(t, size, output=...)``,
``frame_blurred(...)`` and ``audio_track()`` (an ``Engine``); this layer must not import it.
Frames come back as finished sRGB bytes (``output="srgb8"``), encoded on the card.

The presets (``render.presets``) say what each quality tier draws. A ``draft`` renders the
scene without its decoration (``Engine.without_decoration``) and with a plain look, and
repeats a frame whose drawing is identical to the last one instead of drawing it again
(``FrameDrawer``): still moments cost almost nothing (Milestone 21).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from phasor.core.timing import stage
from phasor.render.doctor import run_doctor
from phasor.render.encode import BackgroundWriter, mux_audio, mux_temp, open_writer
from phasor.render.output import save_png
from phasor.render.presets import FULL, MIN_FPS, PRESETS, Effects, check_fps, preset_named

# Motion blur (shutter as a share of a frame interval, samples per frame) by preset: off while
# iterating, on for finals, where it is what makes fast motion look smooth rather than stuttery.
BLUR: dict[str, tuple[float, int]] = {n: (p.shutter, p.blur_samples) for n, p in PRESETS.items()}

# A source of finished frames, in order: (times, size, fps, shutter, samples) -> each frame's
# (h, w, 3) sRGB bytes. ``scene.parallel_frames`` is one (render workers, ADR 0003); without
# one, ``render_file`` renders on ``engine`` itself, one frame after another. ``render_file``
# also passes ``engine=`` (the engine to draw: a draft's is ``without_decoration``),
# ``effects=`` and ``dedup=``; a source must use them.
FrameSource = Callable[..., Iterator[Any]]

Cancel = Any  # a ``threading.Event`` (``is_set``) or a callable returning True to stop


class RenderCancelled(RuntimeError):
    """A render stopped by its ``cancel``; the partial output was removed."""


def render_frame(engine: Any, t: float, fps: float, size: tuple[int, int], shutter: float,
                 samples: int, effects: Effects = FULL) -> Any:  # fmt: skip
    """One finished frame of a render: motion-blurred when the shutter is open and there is
    more than one sample, else sharp. The one rule for the serial loop and the workers."""
    extra = {} if effects is FULL else {"effects": effects}  # FULL: the call as it always was
    if shutter > 0 and samples > 1:
        return engine.frame_blurred(t, fps, shutter=shutter, samples=samples, size=size,
                                    output="srgb8", **extra)  # fmt: skip
    return engine.frame(t, size, output="srgb8", **extra)


# Frames one render may have: 4.6 hours at 60 fps, 9.3 at 30, 16.7 minutes at 1000. Their
# times are a list, so an end of 1e12 s must be refused before it is made, not run out of memory.
MAX_FRAMES = 1_000_000

REPEAT = object()  # what ``FrameDrawer.draw`` returns for a frame identical to the last


class FrameDrawer:
    """Draws a render's frames one after another on ``engine``. With ``dedup`` (and no
    motion blur) each frame's drawing is recorded first (``GLRenderer.frame_plan``); when
    its key equals the last frame's, ``draw`` returns ``REPEAT`` instead of drawing, and
    the caller repeats the last frame's bytes. The recorded drawing is then the one drawn,
    so an object's ``draw`` runs once per frame either way. ``drawn``/``repeated`` count."""

    def __init__(self, engine: Any, fps: float, size: tuple[int, int], shutter: float,
                 samples: int, effects: Effects = FULL, dedup: bool = False) -> None:  # fmt: skip
        self.engine, self.fps, self.size = engine, fps, size
        self.shutter, self.samples, self.effects = shutter, samples, effects
        self.dedup = bool(dedup) and not (shutter > 0 and samples > 1)
        self._last: bytes | None = None
        self.drawn = self.repeated = 0

    def _plan(self, t: float) -> Any:
        runner = self.engine.pass_runner()
        if runner is None or not hasattr(runner, "frame_plan"):
            return None
        return runner.frame_plan(self.engine.state_at(t), self.size, fog=self.effects.fog)

    def prime(self, t: float) -> None:
        """Take the frame at ``t`` as the last one drawn (a render worker starting a stretch
        that follows another worker's: its first frame may repeat that one)."""
        if self.dedup:
            plan = self._plan(t)
            self._last = None if plan is None else plan.key

    def draw(self, t: float) -> Any:
        """The frame at ``t`` as (h, w, 3) sRGB bytes, or ``REPEAT``."""
        if not self.dedup:
            self.drawn += 1
            return render_frame(self.engine, t, self.fps, self.size, self.shutter, self.samples,
                                self.effects)  # fmt: skip
        with stage("dedup"):
            plan = self._plan(t)
        if plan is not None and plan.key == self._last:
            self.repeated += 1
            return REPEAT
        self._last = None if plan is None else plan.key
        self.drawn += 1
        return self.engine.frame(t, self.size, output="srgb8", effects=self.effects, plan=plan)


def serial_frames(engine: Any, times: list[float], size: tuple[int, int], fps: float,
                  shutter: float, samples: int, *, effects: Effects = FULL, dedup: bool = False,
                  counts: dict[str, int] | None = None) -> Iterator[Any]:  # fmt: skip
    """Frames rendered here, in order, with ``FrameDrawer`` (a repeated frame is the last
    one's bytes again). ``counts`` gets ``drawn`` and ``repeated``."""
    drawer = FrameDrawer(engine, fps, size, shutter, samples, effects, dedup)
    last = None
    for t in times:
        out = drawer.draw(t)
        if out is REPEAT:
            out = last
        last = out
        if counts is not None:
            counts.update(drawn=drawer.drawn, repeated=drawer.repeated)
        yield out


def _tier_b_warning(report: Any) -> str:
    cpu = report.get("cpu_kernels")
    if cpu is not None and cpu.ok:
        return (f"no CUDA device (tier B): kernels run on the CPU "
                f"({cpu.data['threads']} threads), slower than on a card")  # fmt: skip
    return "no CUDA device (tier B) and no C++ compiler: kernels run on NumPy, so this is slow"


def engine_for(engine: Any, preset: str) -> Any:
    """The engine a render with ``preset`` draws: ``engine`` itself, or, for a preset without
    life (``draft``), ``engine.without_decoration()``. Call it before building a frame
    source, so the workers get that engine's document and bake."""
    p = preset_named(preset)
    if p.life or not hasattr(engine, "without_decoration"):
        return engine
    return engine.without_decoration()


def resolve_time(engine: Any, value: Any, what: str = "time") -> float | None:
    """A render window's ``start`` or ``end``: seconds (a number or a numeric string) or a
    marker's name. None stays None. Not a finite number and not a marker: ``ValueError``.
    Times outside the scene are kept as they are (they always rendered)."""
    if value is None:
        return None
    if isinstance(value, str):
        # a marker first: a marker may be called ``inf`` or ``nan`` (ids are [a-z][a-z0-9_]*)
        markers = engine.timeline()["markers"] if hasattr(engine, "timeline") else {}
        if value in markers:
            return float(markers[value])
        try:
            value = float(value)
        except ValueError:
            known = ", ".join(sorted(markers)) or "none"
            raise ValueError(f"{what} {value!r} is neither seconds nor a marker "
                             f"(markers: {known})") from None  # fmt: skip
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be seconds or a marker name, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number of seconds, got {value!r}")
    return float(value)


def _stopped(cancel: Cancel) -> bool:
    if cancel is None:
        return False
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set() if is_set is not None else cancel())


def _new_folders(out: Path) -> list[Path]:
    """The folders writing ``out`` will create (none of them exist yet), deepest first."""
    folder = out if out.suffix == "" else out.parent
    new = []
    while not folder.exists() and folder != folder.parent:
        new.append(folder)
        folder = folder.parent
    return new


def _remove_partial(writer: Any, out: Path, folders: list[Path] = ()) -> None:
    """What a cancelled render wrote: its PNGs, or the video file and a half-muxed copy of it;
    then the folders it created, if they are empty. Never a file or folder it did not make."""
    inner = getattr(writer, "inner", writer)
    files = inner.written() if hasattr(inner, "written") else [out, mux_temp(out)]
    for f in files:
        with contextlib.suppress(OSError):
            f.unlink(missing_ok=True)
    for d in folders:
        with contextlib.suppress(OSError):
            d.rmdir()  # only if empty


def render_file(
    engine: Any,
    out: str | Path,
    *,
    preset: str = "preview",
    fps: float | None = None,
    start: float | str | None = None,
    end: float | str | None = None,
    frame: float | str | None = None,
    progress: Callable[[int, int], None] | None = None,
    shutter: float | None = None,
    blur_samples: int | None = None,
    frame_source: FrameSource | None = None,
    cancel: Cancel = None,
) -> dict[str, Any]:
    """Render ``engine`` to ``out`` (.mp4/.mov, a folder or a ``%05d.png`` pattern; a PNG
    when ``frame`` is given). Returns what was written. Raises ``RuntimeError`` on a machine
    with no OpenGL, and ``EncodeError`` when no encoder works. ``frame_source`` makes the
    frames instead of ``engine`` (``scene.parallel_frames``: several processes).
    ``start``/``end``/``frame``: seconds or a marker's name. ``progress(k, n)`` is called
    after each frame; ``cancel`` (a ``threading.Event`` or a callable) is checked between
    frames, and when it is set the render stops, removes what it wrote and raises
    ``RenderCancelled``. A bad preset, fps or time is a ``ValueError``."""
    p = preset_named(preset)
    if fps is not None and fps != 0:  # 0 and None: the preset's (as ever)
        fps = check_fps(fps)
    start, end = resolve_time(engine, start, "start"), resolve_time(engine, end, "end")
    frame = resolve_time(engine, frame, "frame")
    engine = engine_for(engine, preset)
    meta = engine.document.meta
    size = p.size(meta.width, meta.height)
    fps = fps or p.fps or meta.fps
    if fps < MIN_FPS:  # a scene's own rate the encoder cannot write (its rate would be 0/1)
        raise ValueError(f"fps must be at least 1/1001, the scene's is {fps!r}")
    if frame is None:  # the window, bounded before anything is set up
        start = start or 0.0
        end = engine.duration if end is None else end
        span = (end - start) * fps
        if not span <= MAX_FRAMES:  # also catches an overflow to inf
            raise ValueError(f"the window {start:g}-{end:g} s at {fps:g} fps is {span:.3g} "
                             f"frames; a render takes at most {MAX_FRAMES} "
                             "(render.job.MAX_FRAMES)")  # fmt: skip
    with stage("setup"):  # once per render, not per frame
        report = run_doctor()
        tier = report.tier
    if tier == "C":
        raise RuntimeError("no OpenGL here (tier C); nothing can be rendered")
    warnings = [_tier_b_warning(report)] if tier == "B" else []
    out = Path(out)
    shutter = p.shutter if shutter is None else shutter
    blur_samples = p.blur_samples if blur_samples is None else blur_samples
    effects = p.effects
    if frame is not None:
        save_png(out, render_frame(engine, frame, fps, size, 0.0, 1, effects))
        return {"path": str(out), "size": size, "t": frame, "frames": 1, "warnings": warnings,
                "preset": preset}  # fmt: skip
    count = max(1, round(span))
    times = [start + k / fps for k in range(count)]
    # The sound first: a clip that cannot be mixed is said before any frame is rendered,
    # never after the last one.
    problems: list[str] = []
    sound = out.suffix in (".mp4", ".mov")
    track = engine.audio_track(problems, start, start + count / fps) if sound else None
    warnings.extend(problems)
    counts: dict[str, int] = {}
    # The encoder runs on its own thread: frame k is encoded while frame k+1 renders.
    # "encode" then times only the hand-over (and any wait for a slow encoder).
    if frame_source is None:
        frames = serial_frames(engine, times, size, fps, shutter, blur_samples,
                               effects=effects, dedup=p.dedup, counts=counts)  # fmt: skip
    elif effects is FULL and not p.dedup and p.life:  # the call every source has taken
        frames = frame_source(times, size, fps, shutter, blur_samples)
    else:
        frames = frame_source(times, size, fps, shutter, blur_samples, engine=engine,
                              effects=effects, dedup=p.dedup, counts=counts)  # fmt: skip
    if _stopped(cancel):
        frames.close()
        raise RenderCancelled("render cancelled before its first frame")
    folders = _new_folders(out)
    writer = BackgroundWriter(open_writer(out, *size, fps, speed=p.encoder))
    try:
        for k, srgb8 in enumerate(frames):  # finished bytes, encoded on the card if any
            with stage("encode"):
                writer.write(srgb8)
            if progress:
                progress(k + 1, count)
            if k + 1 < count and _stopped(cancel):  # checked between frames
                raise RenderCancelled(f"render cancelled after {k + 1} of {count} frames")
    except BaseException as e:
        writer.abort()
        if isinstance(e, (RenderCancelled, KeyboardInterrupt)):  # stopped on purpose (Ctrl-C)
            _remove_partial(writer, out, folders)
        raise
    finally:
        frames.close()  # a frame source stops its workers
    sound = track is not None
    try:
        with stage("encode"):  # the frames still queued
            writer.close()
        if sound:
            samples, rate = track
            mux_audio(out, samples[: round(count / fps * rate)], rate)
    except KeyboardInterrupt:  # Ctrl-C while finishing: no half-written file stays behind
        _remove_partial(writer, out, folders)
        raise
    return {
        "path": str(out), "size": size, "fps": fps, "frames": count, "encoder": writer.encoder,
        "narration": bool(sound), "warnings": warnings,
        "motion_blur": shutter > 0 and blur_samples > 1, "preset": preset,
        "drawn": counts.get("drawn", count), "repeated": counts.get("repeated", 0),
    }  # fmt: skip
