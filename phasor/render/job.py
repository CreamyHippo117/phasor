"""Render a scene to a video file or one still: the one place that loop lives.

The CLI (`phasor render`) and the MCP `render` tool both call this. ``engine`` is
anything with ``document``, ``duration``, ``frame(t, size, output=...)``,
``frame_blurred(...)`` and ``audio_track()`` (an ``Engine``); this layer must not import it.
Frames come back as finished sRGB bytes (``output="srgb8"``), encoded on the card.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from phasor.core.timing import stage
from phasor.render.doctor import run_doctor
from phasor.render.encode import BackgroundWriter, mux_audio, open_writer
from phasor.render.output import save_png

# name -> (width, height, fps); None means "the scene's own"
PRESETS: dict[str, tuple[int | None, int | None, float | None]] = {
    "preview": (960, 540, 30.0),
    "final": (None, None, None),
    "4k": (3840, 2160, None),
}
# Motion blur (shutter as a share of a frame interval, samples per frame) by preset: off while
# iterating, on for finals, where it is what makes fast motion look smooth rather than stuttery.
BLUR: dict[str, tuple[float, int]] = {"preview": (0.0, 1), "final": (0.5, 4), "4k": (0.5, 4)}

# A source of finished frames, in order: (times, size, fps, shutter, samples) -> each frame's
# (h, w, 3) sRGB bytes. ``scene.parallel_frames`` is one (render workers, ADR 0003); without
# one, ``render_file`` renders on ``engine`` itself, one frame after another.
FrameSource = Callable[[list[float], tuple[int, int], float, float, int], Iterator[Any]]


def render_frame(engine: Any, t: float, fps: float, size: tuple[int, int], shutter: float,
                 samples: int) -> Any:  # fmt: skip
    """One finished frame of a render: motion-blurred when the shutter is open and there is
    more than one sample, else sharp. The one rule for the serial loop and the workers."""
    if shutter > 0 and samples > 1:
        return engine.frame_blurred(t, fps, shutter=shutter, samples=samples, size=size,
                                    output="srgb8")  # fmt: skip
    return engine.frame(t, size, output="srgb8")


def _tier_b_warning(report: Any) -> str:
    cpu = report.get("cpu_kernels")
    if cpu is not None and cpu.ok:
        return (f"no CUDA device (tier B): kernels run on the CPU "
                f"({cpu.data['threads']} threads), slower than on a card")  # fmt: skip
    return "no CUDA device (tier B) and no C++ compiler: kernels run on NumPy, so this is slow"


def render_file(
    engine: Any,
    out: str | Path,
    *,
    preset: str = "preview",
    fps: float | None = None,
    start: float | None = None,
    end: float | None = None,
    frame: float | None = None,
    progress: Callable[[int, int], None] | None = None,
    shutter: float | None = None,
    blur_samples: int | None = None,
    frame_source: FrameSource | None = None,
) -> dict[str, Any]:
    """Render ``engine`` to ``out`` (.mp4/.mov, a folder or a ``%05d.png`` pattern; a PNG
    when ``frame`` is given). Returns what was written. Raises ``RuntimeError`` on a machine
    with no OpenGL, and ``EncodeError`` when no encoder works. ``frame_source`` makes the
    frames instead of ``engine`` (``scene.parallel_frames``: several processes)."""
    meta = engine.document.meta
    w, h, preset_fps = PRESETS[preset]
    size = (w or meta.width, h or meta.height)
    fps = fps or preset_fps or meta.fps
    with stage("setup"):  # once per render, not per frame
        report = run_doctor()
        tier = report.tier
    if tier == "C":
        raise RuntimeError("no OpenGL here (tier C); nothing can be rendered")
    warnings = [_tier_b_warning(report)] if tier == "B" else []
    out = Path(out)
    p_shutter, p_samples = BLUR[preset]
    shutter = p_shutter if shutter is None else shutter
    blur_samples = p_samples if blur_samples is None else blur_samples
    if frame is not None:
        save_png(out, engine.frame(frame, size, output="srgb8"))
        return {"path": str(out), "size": size, "t": frame, "frames": 1, "warnings": warnings}
    start = start or 0.0
    end = engine.duration if end is None else end
    count = max(1, round((end - start) * fps))
    # The encoder runs on its own thread: frame k is encoded while frame k+1 renders.
    # "encode" then times only the hand-over (and any wait for a slow encoder).
    times = [start + k / fps for k in range(count)]
    if frame_source is None:
        frames = (render_frame(engine, t, fps, size, shutter, blur_samples) for t in times)
    else:
        frames = frame_source(times, size, fps, shutter, blur_samples)
    writer = BackgroundWriter(open_writer(out, *size, fps))
    try:
        for k, srgb8 in enumerate(frames):  # finished bytes, encoded on the card if any
            with stage("encode"):
                writer.write(srgb8)
            if progress:
                progress(k + 1, count)
    except BaseException:
        writer.abort()
        raise
    finally:
        frames.close()  # a frame source stops its workers
    with stage("encode"):  # the frames still queued
        writer.close()
    problems: list[str] = []
    sound = out.suffix in (".mp4", ".mov")
    track = engine.audio_track(problems, start, start + count / fps) if sound else None
    warnings.extend(problems)
    sound = track is not None
    if sound:
        samples, rate = track
        mux_audio(out, samples[: round(count / fps * rate)], rate)
    return {
        "path": str(out), "size": size, "fps": fps, "frames": count, "encoder": writer.encoder,
        "narration": bool(sound), "warnings": warnings,
        "motion_blur": shutter > 0 and blur_samples > 1,
    }  # fmt: skip
