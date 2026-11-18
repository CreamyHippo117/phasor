"""Text metrics and rasterization on the CPU, shared by bounds (inspector, layout) and the
renderer so they can never disagree about how big a label is.

Milestone 1 uses the scalable font bundled inside Pillow (Aileron), which is identical on
every OS. Theme fonts and LaTeX arrive with the first full theme (Milestone 3).
"""

from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REF = 64


@functools.cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


@functools.lru_cache(maxsize=4096)
def aspect(text: str) -> float:
    """Width / height of one line of text, where height is the full line (ascent + descent)."""
    f = _font(_REF)
    ascent, descent = f.getmetrics()
    return max(f.getlength(text), 1.0) / (ascent + descent)


@functools.lru_cache(maxsize=512)
def rasterize(text: str, px_height: int) -> np.ndarray:
    """Coverage bitmap (uint8, row 0 = top), exactly ``px_height`` tall, ``aspect`` wide."""
    px_height = max(int(px_height), 2)
    ascent, descent = _font(_REF).getmetrics()
    f = _font(max(1, round(_REF * px_height / (ascent + descent))))
    width = max(1, math.ceil(aspect(text) * px_height))
    img = Image.new("L", (width, px_height), 0)
    ImageDraw.Draw(img).text((0, 0), text, fill=255, font=f)
    out = np.asarray(img, dtype=np.uint8)
    out.setflags(write=False)
    return out


_ANCHOR_SHIFT = {"center": -0.5, "left": 0.0, "right": -1.0}


def quad(text: str, origin, height: float, anchor: str, right, up):
    """The world-space rectangle a line of text occupies: (lower-left corner, width vector,
    height vector). Bounds and rendering both use this, so they always agree."""
    w = aspect(text) * height
    corner = np.asarray(origin, dtype=np.float64) + _ANCHOR_SHIFT[anchor] * w * right
    return corner - (height / 2) * up, w * right, height * up


@functools.cache
def _tofu() -> bytes:
    return bytes(_font(_REF).getmask("\U000f0000"))  # a private-use code point: never drawn


@functools.lru_cache(maxsize=4096)
def missing_glyphs(text: str) -> tuple[str, ...]:
    """Characters the font cannot draw (they would show as empty boxes), in order, once each.
    Whitespace is never missing."""
    font, tofu, out = _font(_REF), _tofu(), []
    for ch in dict.fromkeys(text):
        if not ch.isspace() and bytes(font.getmask(ch)) == tofu:
            out.append(ch)
    return tuple(out)
