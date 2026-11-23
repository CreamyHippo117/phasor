"""Glyph outlines of a line of text (pure: NumPy and matplotlib's font code, no drawing).

``outline(text)`` gives each glyph of ``text`` as subpaths of cubic Béziers
(``domains/path_math.py``), in the same box ``core.text.quad`` gives the text when it is drawn
as a picture, so a text drawn as shapes lies where its picture would:

- plain text in the font the picture uses (Aileron, bundled in Pillow), laid out with its
  advances and kerning and stretched to the picture's width (as the picture is);
- math (``$...$``) through matplotlib's mathtext in Computer Modern, like the picture with
  the built-in engine. With the ``latex`` engine the picture comes from TeX but the outline
  still comes from mathtext (they can differ a little).

Coordinates are in line heights of the text's role (the ``size`` style), from the lower-left
corner of the box, x to the right and y up; the caller places them with the camera's
``right`` and ``up``. TrueType quadratic curves become cubics exactly.

A text keeps at most ``MAX_GLYPHS`` glyphs as shapes (the rest are left out; lint
``glyph_limit`` says so). Results are cached per text (the fonts never change in a process).
"""

from __future__ import annotations

import functools
import io
import itertools
import threading
import unicodedata
from dataclasses import dataclass

import numpy as np

from phasor.core import text as text_metrics
from phasor.core.texconv import to_mathtext

MAX_GLYPHS = 400  # glyphs one text draws as shapes; the rest are left out (lint glyph_limit)
FONT_SCALE = 100.0  # pixels per em the outlines are read at (72 dpi: one point per pixel)
NAME_MAX = 24  # characters of a glyph's name (before a suffix that keeps it unique)
_LOCK = threading.Lock()  # matplotlib's fonts are shared objects with state

_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


@dataclass(frozen=True)
class Glyph:
    name: str  # what it shows: ``a``, ``cap_a``, ``two``, ``plus_sign``, ``bar``
    key: str  # the same glyph of the same font (``transform_matching`` pairs equal keys)
    subpaths: tuple[np.ndarray, ...]  # (1 + 3k, 3) each, in line heights from the corner
    order: int  # its place in the text, left to right (for ``write``)


@dataclass(frozen=True)
class Outline:
    glyphs: tuple[Glyph, ...]
    total: int  # glyphs the text has (more than ``len(glyphs)`` when capped)


def glyph_name(ch: str) -> str:
    """A name for a character, by meaning: ``a``, ``cap_a``, ``two``, ``plus_sign``."""
    if ch.isascii() and ch.isalpha():
        return ch if ch.islower() else f"cap_{ch.lower()}"
    if ch.isascii() and ch.isdigit():
        return _DIGITS[int(ch)]
    try:
        words = unicodedata.name(ch).lower()
    except ValueError:
        return "glyph"
    words = words.replace("greek small letter ", "").replace("greek capital letter ", "cap_")
    name = "".join(c if c.isalnum() else "_" for c in words).strip("_")[:NAME_MAX].rstrip("_")
    return name if name and name[0].isalpha() and name.isascii() else "glyph"


def _font_glyph_name(font, gi: int, ccode: int) -> str:
    """A math glyph's name, from the font (``pi``, ``plus``, ``Gamma`` as ``cap_gamma``),
    else from its character."""
    try:
        raw = font.get_glyph_name(gi)
    except (RuntimeError, ValueError):
        raw = ""
    if raw and raw.isascii() and raw.isalpha():
        name = f"cap_{raw.lower()}" if raw[0].isupper() else raw.lower()
        return name[:NAME_MAX]
    return glyph_name(chr(ccode)) if 0 < ccode < 0x110000 else "glyph"


# ---------------------------------------------------------------- paths


def subpaths(verts: np.ndarray, codes: np.ndarray, scale: float = 1.0,
             shift=(0.0, 0.0)) -> list[np.ndarray]:  # fmt: skip
    """A matplotlib path (vertices, codes) as subpaths of cubics in 3D (z 0): lines become
    straight cubics, quadratic curves (TrueType) the cubics equal to them; ``CLOSEPOLY``
    closes a subpath back to its start."""
    from matplotlib.path import Path as MPath

    v = np.asarray(verts, dtype=np.float64) * scale + np.asarray(shift, dtype=np.float64)
    out: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    start = None

    def finish() -> None:
        if len(cur) >= 4:
            out.append(np.array(cur))

    i, n = 0, len(codes)
    while i < n:
        code = codes[i]
        p = np.array([v[i, 0], v[i, 1], 0.0])
        if code == MPath.MOVETO:
            finish()
            cur, start = [p], p
            i += 1
        elif code == MPath.LINETO and cur:
            a = cur[-1]
            cur += [a + (p - a) / 3, a + 2 * (p - a) / 3, p]
            i += 1
        elif code == MPath.CURVE3 and cur and i + 1 < n:
            a, q = cur[-1], p
            b = np.array([v[i + 1, 0], v[i + 1, 1], 0.0])
            cur += [a + 2 * (q - a) / 3, b + 2 * (q - b) / 3, b]
            i += 2
        elif code == MPath.CURVE4 and cur and i + 2 < n:
            c2 = np.array([v[i + 1, 0], v[i + 1, 1], 0.0])
            b = np.array([v[i + 2, 0], v[i + 2, 1], 0.0])
            cur += [p, c2, b]
            i += 3
        elif code == MPath.CLOSEPOLY and cur:
            a = cur[-1]
            if start is not None and np.linalg.norm(a - start) > 1e-12:
                cur += [a + (start - a) / 3, a + 2 * (start - a) / 3, start.copy()]
            elif start is not None:
                cur[-1] = start.copy()
            finish()
            cur, start = [], None
            i += 1
        else:  # STOP, or a curve with nothing before it
            i += 1
    finish()
    return out


# ---------------------------------------------------------------- fonts


@functools.cache
def _plain_font():
    """Aileron, the font Pillow bundles (``core.text`` draws plain text with it)."""
    from matplotlib.ft2font import FT2Font

    data = text_metrics._font(64).font_bytes
    font = FT2Font(io.BytesIO(data))
    font.set_size(FONT_SCALE, 72)
    return font


def _plain(text: str) -> Outline:
    from matplotlib.ft2font import Kerning, LoadFlags

    font = _plain_font()
    ascent, descent = text_metrics._font(text_metrics._REF).getmetrics()
    line = text_metrics._REF / (ascent + descent)  # ems per line height, as Pillow sizes it
    base = descent / (ascent + descent)  # the baseline above the box's bottom
    pen, prev = 0.0, None
    placed: list[tuple[str, str, list[np.ndarray]]] = []
    total = 0
    cut = len(text)
    for at, ch in enumerate(text):
        if len(placed) >= MAX_GLYPHS:  # the rest is only counted (bounded work)
            cut = at
            total += sum(1 for c in text[at:] if not c.isspace())
            break
        gi = font.get_char_index(ord(ch))
        if prev is not None and gi:
            pen += font.get_kerning(prev, gi, Kerning.DEFAULT) / 64.0
        try:
            glyph = font.load_glyph(gi, flags=LoadFlags.NO_HINTING)  # 0: the missing glyph
        except RuntimeError:
            prev = None
            continue
        if gi and not ch.isspace():
            total += 1
            verts, codes = font.get_path()
            subs = subpaths(verts, codes, 1.0, (pen, 0.0))
            if subs:
                placed.append((glyph_name(ch), f"{font.postscript_name}:{gi}", subs))
        pen += glyph.linearHoriAdvance / 65536.0
        prev = gi or None
    # The picture is stretched to Pillow's width: so are the shapes (of the part drawn).
    if cut < len(text):
        width = text_metrics._font(text_metrics._REF).getlength(text[:cut]) / (ascent + descent)
    else:
        width = text_metrics.aspect(text)
    sx = width / pen if pen > 0 else line / FONT_SCALE  # pixels to line heights
    out = []
    for k, (name, key, subs) in enumerate(placed):
        moved = []
        for s in subs:
            s = s.copy()
            s[:, 0] *= sx
            s[:, 1] = s[:, 1] / FONT_SCALE * line + base
            s.setflags(write=False)
            moved.append(s)
        out.append(Glyph(name, key, tuple(moved), k))
    return Outline(tuple(out), total)


def _math(text: str) -> Outline:
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.ft2font import LoadFlags
    from matplotlib.mathtext import MathTextParser

    with rc_context({"mathtext.fontset": "cm"}):
        parsed = MathTextParser("path").parse(to_mathtext(text), dpi=72,
                                              prop=FontProperties(size=FONT_SCALE))  # fmt: skip
    em = text_metrics.MATH_EM  # line heights per em
    lift = float(parsed.depth) / FONT_SCALE  # depth below the baseline, in ems (mathtext)
    found: list[tuple[float, str, str, list[np.ndarray]]] = []
    total = len(parsed.rects) + sum(not (0 < g[2] < 0x110000 and chr(g[2]).isspace())
                                    for g in parsed.glyphs)  # fmt: skip
    for font, fontsize, ccode, gi, ox, oy in parsed.glyphs:
        if len(found) >= MAX_GLYPHS:
            break
        font.clear()
        font.set_size(FONT_SCALE, 72)
        font.load_glyph(gi, flags=LoadFlags.NO_HINTING)
        verts, codes = font.get_path()
        s = fontsize / FONT_SCALE
        subs = subpaths(
            verts,
            codes,
            s * em / FONT_SCALE,
            (float(ox) * em / FONT_SCALE, (float(oy) / FONT_SCALE + lift) * em),
        )
        if subs:
            name = _font_glyph_name(font, gi, ccode)
            found.append((float(ox), name, f"{font.postscript_name}:{gi}", subs))
    for ox, oy, w, h in parsed.rects:  # fraction bars, root tops
        if len(found) >= MAX_GLYPHS:
            break
        x0, y0 = float(ox) * em / FONT_SCALE, (float(oy) / FONT_SCALE + lift) * em
        x1, y1 = x0 + float(w) * em / FONT_SCALE, y0 + float(h) * em / FONT_SCALE
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        sub = []
        for a, b in itertools.pairwise(ring):
            pa, pb = np.array([*a, 0.0]), np.array([*b, 0.0])
            sub += ([pa] if not sub else []) + [pa + (pb - pa) / 3, pa + 2 * (pb - pa) / 3, pb]
        ratio = round(float(w) / max(float(h), 1e-12), 1)
        found.append((float(ox), "bar", f"rect:{ratio}", [np.array(sub)]))
    found.sort(key=lambda f: f[0])  # left to right: the order ``write`` draws them in
    out = []
    for k, (_, name, key, subs) in enumerate(found):
        for s in subs:
            s.setflags(write=False)
        out.append(Glyph(name, key, tuple(subs), k))
    return Outline(tuple(out), total)


@functools.lru_cache(maxsize=512)
def outline(text: str) -> Outline:
    """The glyphs of one line of text (math when it holds ``$...$``), as shapes."""
    with _LOCK:
        return _math(text) if text_metrics.is_math(text) else _plain(text)
