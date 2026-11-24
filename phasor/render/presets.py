"""Render presets: one table of what each quality tier draws, and how (Milestone 21).

| preset    | picture                     | fps   | motion blur | look            | encoder |
| --------- | --------------------------- | ----- | ----------- | --------------- | ------- |
| `draft`   | long side 640, scene aspect | 15    | off         | plain (DRAFT)   | fast    |
| `preview` | 960 x 540                   | 30    | off         | full            | quality |
| `final`   | the scene's own             | scene | 0.5, 4 sub  | full            | quality |
| `4k`      | 3840 x 2160                 | scene | 0.5, 4 sub  | full            | quality |

A plain look (``Effects``) leaves out the decoration around the picture: no bloom, no lens
(fringe, vignette), no depth fog in 3D. Paint, strokes, fills, text and kernel pictures still
draw; the tone map and the sRGB encode still run exactly once (contract 4). ``life=False``
renders the scene with life and the theme's camera drift off (``Engine.without_decoration``),
so still moments are really still and ``dedup`` can repeat them instead of drawing them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Effects:
    """Which parts of the final look run. ``FULL`` is every one (what ``preview``, ``final``
    and ``4k`` draw); ``DRAFT`` none. The tone map and sRGB encode are not optional."""

    bloom: bool = True  # the glow's halo (theme [bloom] strength, levels)
    lens: bool = True  # fringe and vignette (theme [bloom] fringe, vignette)
    fog: bool = True  # 3D depth fog toward the background (theme [light] fog)


FULL = Effects()
DRAFT = Effects(bloom=False, lens=False, fog=False)


@dataclass(frozen=True)
class Preset:
    """One quality tier. Size: ``width`` x ``height`` when both are given, else the scene's
    own picture scaled so its long side is at most ``long_side`` (``None``: the scene's own
    size). ``fps`` None is the scene's own. ``shutter``/``blur_samples``: motion blur (0 or 1
    sample is off). ``encoder``: ``"quality"`` or ``"fast"`` (``render.encode``)."""

    name: str
    width: int | None
    height: int | None
    long_side: int | None
    fps: float | None
    shutter: float
    blur_samples: int
    effects: Effects = FULL
    life: bool = True  # False: life and the theme's camera drift off (a plainer document)
    dedup: bool = False  # repeat a frame whose drawing is identical to the last one
    encoder: str = "quality"
    note: str = ""

    def size(self, scene_w: int, scene_h: int) -> tuple[int, int]:
        """The picture size for a scene of ``scene_w`` x ``scene_h``: even sides (video
        needs them), the scene's aspect kept when the preset says only a long side."""
        if self.width is not None and self.height is not None:
            return self.width, self.height
        if self.long_side is None:
            return scene_w, scene_h
        scale = min(1.0, self.long_side / max(scene_w, scene_h))
        return (max(2, 2 * round(scene_w * scale / 2)), max(2, 2 * round(scene_h * scale / 2)))

    def describe(self) -> dict[str, Any]:
        """Plain data for ``phasor doctor``, the docs and tools."""
        if self.width is not None:
            size = f"{self.width}x{self.height}"
        elif self.long_side is not None:
            size = f"long side {self.long_side}, scene aspect"
        else:
            size = "the scene's own"
        e = self.effects
        return {"name": self.name, "size": size,
                "fps": "the scene's own" if self.fps is None else self.fps,
                "motion_blur": self.shutter > 0 and self.blur_samples > 1,
                "bloom": e.bloom, "lens": e.lens, "fog": e.fog, "life": self.life,
                "dedup": self.dedup, "encoder": self.encoder, "note": self.note}  # fmt: skip


PRESETS: dict[str, Preset] = {
    "draft": Preset("draft", None, None, 640, 15.0, 0.0, 1, DRAFT, life=False, dedup=True,
                    encoder="fast", note="quickest look at the motion: small, 15 fps, plain"),
    "preview": Preset("preview", 960, 540, None, 30.0, 0.0, 1,
                      note="iterate: 960x540, 30 fps, the full look, no motion blur"),
    "final": Preset("final", None, None, None, None, 0.5, 4,
                    note="the scene's size and frame rate, motion blur"),
    "4k": Preset("4k", 3840, 2160, None, None, 0.5, 4, note="3840x2160, motion blur"),
}  # fmt: skip

MAX_FPS = 1000.0


def preset_named(name: str) -> Preset:
    """The preset called ``name``; a clear ``ValueError`` for anything else."""
    if not isinstance(name, str) or name not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {name!r}")
    return PRESETS[name]


def check_fps(fps: Any) -> float:
    """A frame rate a render may use: finite, above 0, at most ``MAX_FPS``."""
    try:
        f = float(fps)
    except (TypeError, ValueError):
        raise ValueError(f"fps must be a number, got {fps!r}") from None
    if not (math.isfinite(f) and 0 < f <= MAX_FPS):
        raise ValueError(f"fps must be above 0 and at most {MAX_FPS:g}, got {fps!r}")
    return f


def table() -> list[dict[str, Any]]:
    return [p.describe() for p in PRESETS.values()]
