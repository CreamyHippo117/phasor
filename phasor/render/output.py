"""The end of the color pipeline: tone map and sRGB-encode exactly once (SPEC contract 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phasor.core.color import linear_to_srgb
from phasor.core.timing import stage


def encode_srgb8(linear_rgba: np.ndarray) -> np.ndarray:
    """Linear RGBA in display range (after ``render.post.finish``) -> (h, w, 3) uint8 sRGB.
    The one place sRGB is encoded; values outside 0..1 are clipped."""
    with stage("srgb"):
        rgb = linear_to_srgb(linear_rgba[..., :3])
        return np.round(rgb * 255).astype(np.uint8)


def save_png(path: str | Path, srgb8: np.ndarray) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(srgb8, "RGB").save(path)
    return path
