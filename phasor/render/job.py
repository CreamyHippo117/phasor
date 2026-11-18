"""Render a scene to a video file or one still: the one place that loop lives.

The CLI (`phasor render`) and the MCP `render` tool both call this. ``engine`` is
anything with ``document``, ``duration``, ``frame(t, size)`` and ``audio_track()``
(an ``Engine``); this layer must not import it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from phasor.render.doctor import run_doctor
from phasor.render.encode import mux_audio, open_writer
from phasor.render.output import encode_srgb8, save_png

# name -> (width, height, fps); None means "the scene's own"
PRESETS: dict[str, tuple[int | None, int | None, float | None]] = {
    "preview": (960, 540, 30.0),
    "final": (None, None, None),
    "4k": (3840, 2160, None),
}


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
) -> dict[str, Any]:
    """Render ``engine`` to ``out`` (.mp4/.mov, a folder or a ``%05d.png`` pattern; a PNG
    when ``frame`` is given). Returns what was written. Raises ``RuntimeError`` on a machine
    with no OpenGL, and ``EncodeError`` when no encoder works."""
    meta = engine.document.meta
    w, h, preset_fps = PRESETS[preset]
    size = (w or meta.width, h or meta.height)
    fps = fps or preset_fps or meta.fps
    tier = run_doctor().tier
    if tier == "C":
        raise RuntimeError("no OpenGL here (tier C); nothing can be rendered")
    warnings = (
        ["no CUDA device (tier B): kernels run on NumPy, so this is slow"] if tier == "B" else []
    )
    out = Path(out)
    if frame is not None:
        save_png(out, encode_srgb8(engine.frame(frame, size)))
        return {"path": str(out), "size": size, "t": frame, "frames": 1, "warnings": warnings}
    start = start or 0.0
    end = engine.duration if end is None else end
    count = max(1, round((end - start) * fps))
    writer = open_writer(out, *size, fps)
    for k in range(count):
        writer.write(encode_srgb8(engine.frame(start + k / fps, size)))
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
    }  # fmt: skip
