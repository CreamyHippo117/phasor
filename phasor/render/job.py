"""Render a scene to a video file or one still: the one place that loop lives.

The CLI (`phasor render`) and the MCP `render` tool both call this. ``engine`` is
anything with ``document``, ``duration``, ``frame(t, size, output=...)``,
``frame_blurred(...)`` and ``audio_track()`` (an ``Engine``); this layer must not import it.
Frames come back as finished sRGB bytes (``output="srgb8"``), encoded on the card.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from phasor.core.timing import stage
from phasor.render.doctor import run_doctor
from phasor.render.encode import mux_audio, open_writer
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
) -> dict[str, Any]:
    """Render ``engine`` to ``out`` (.mp4/.mov, a folder or a ``%05d.png`` pattern; a PNG
    when ``frame`` is given). Returns what was written. Raises ``RuntimeError`` on a machine
    with no OpenGL, and ``EncodeError`` when no encoder works."""
    meta = engine.document.meta
    w, h, preset_fps = PRESETS[preset]
    size = (w or meta.width, h or meta.height)
    fps = fps or preset_fps or meta.fps
    with stage("setup"):  # once per render, not per frame
        tier = run_doctor().tier
    if tier == "C":
        raise RuntimeError("no OpenGL here (tier C); nothing can be rendered")
    warnings = (
        ["no CUDA device (tier B): kernels run on NumPy, so this is slow"] if tier == "B" else []
    )
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
    writer = open_writer(out, *size, fps)
    for k in range(count):
        t = start + k / fps
        srgb8 = (  # finished bytes, encoded on the card when there is one
            engine.frame_blurred(t, fps, shutter=shutter, samples=blur_samples, size=size,
                                 output="srgb8")
            if shutter > 0 and blur_samples > 1 else engine.frame(t, size, output="srgb8")
        )  # fmt: skip
        with stage("encode"):
            writer.write(srgb8)
        if progress:
            progress(k + 1, count)
    writer.close()
    track = engine.audio_track()
    sound = track is not None and out.suffix in (".mp4", ".mov")
    if sound:
        samples, rate = track
        first = round(start * rate)
        mux_audio(out, samples[first : first + round(count / fps * rate)], rate)
    return {
        "path": str(out), "size": size, "fps": fps, "frames": count, "encoder": writer.encoder,
        "narration": bool(sound), "warnings": warnings,
        "motion_blur": shutter > 0 and blur_samples > 1,
    }  # fmt: skip
