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


def luminance(rgb) -> float:
    """Relative luminance of a linear RGB color (Rec. 709 weights)."""
    r, g, b = (float(x) for x in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def over(top, bottom) -> tuple[float, float, float, float]:
    """Composite linear RGBA ``top`` over an opaque ``bottom``."""
    a = float(top[3])
    mixed = (float(t) * a + float(b) * (1 - a) for t, b in zip(top[:3], bottom[:3], strict=True))
    return (*mixed, 1.0)


def contrast_ratio(a, b) -> float:
    """WCAG contrast ratio between two opaque linear colors, from 1 (same) to 21."""
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def mark_color(index: int) -> tuple[int, int, int]:
    """A distinct sRGB color for object index ``index`` (the value in the id pass), used to
    draw set-of-mark outlines. Golden-angle hue steps keep neighbors far apart."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.75, 1.0)
    return (round(r * 255), round(g * 255), round(b * 255))
