"""Color math. Everything inside phasor is linear RGBA; sRGB exists only at the edges
(theme files in, final frames out). SPEC contract 4."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_linear(hex_color: str) -> tuple[float, float, float, float]:
    """``#rrggbb`` or ``#rrggbbaa`` (sRGB) -> linear RGBA."""
    h = hex_color.lstrip("#")
    if len(h) not in (6, 8):
        raise ValueError(f"bad hex color: {hex_color!r}")
    rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    a = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
    r, g, b = (float(x) for x in srgb_to_linear(rgb))
    return (r, g, b, a)
