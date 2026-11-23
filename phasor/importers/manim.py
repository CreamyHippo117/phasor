"""Manim import (SPEC "Manim import"): a Manim Community script runs as it is, and phasor
records the scene it builds.

``import_manim("scene.py")`` runs the file with a small stand-in for the ``manim`` module on
the import path: ``from manim import *`` gives the usual names (``Scene``, ``Circle``,
``Create``, ``FadeIn``, ``Text``, ``MathTex``, ``Axes`` ...), and ``construct`` is called. Every
``add`` and ``play`` becomes objects and transitions on a phasor ``Scene`` at the right time:
what ``self.play(Create(circle), run_time=2)`` does for two seconds is a track on the
circle's ``progress`` here. The result is an ordinary phasor scene (save it, lint it, edit it
in the editor, render it), with the Manim frame (8 units tall, y up, origin in the middle).

What carries over is listed in ``docs/ai/API.md`` ("Manim import"). Shapes become phasor's
shapes (``shapes.Circle``, ``shapes.Rectangle`` ...) with their fills; a shape stretched out
of its kind, a line and a curve become a ``shapes.Path``. Diagrams become the diagram kit
(``dg``), ``ValueTracker`` a ``Value``, ``add_sound`` a ``media.Sound``, a
``MovingCameraScene``'s frame the camera. Updaters and ``always_redraw`` are run by the
stand-in at ``UPDATE_RATE`` frames a second, as Manim runs them, and what they do becomes
tracks (a baked record: the document never runs the script again). Anything phasor cannot
express is an entry in ``ManimImport.warnings``; a name phasor's stand-in lacks raises an
error that says so.

The stand-in was checked against real Manim CE 0.22.1 (``tests/manim_real``, Milestone 20):
colors, default sizes and rate functions are Manim's own.
"""

from __future__ import annotations

import colorsys
import copy
import inspect
import math
import re
import runpy
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from phasor.core import files as core_files
from phasor.core import text as text_metrics
from phasor.core import ticks
from phasor.core.ids import name_from
from phasor.core.issues import DocumentError
from phasor.core.schema import MAX_SIZE, Document, check_color
from phasor.scene import Scene as PhasorScene
from phasor.time.verbs import VerbError

FRAME_HEIGHT = 8.0
FRAME_WIDTH = FRAME_HEIGHT * 16 / 9
PI, TAU = math.pi, 2 * math.pi
DEGREES = PI / 180

ORIGIN = np.array([0.0, 0.0, 0.0])
UP, DOWN = np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
LEFT, RIGHT = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
OUT, IN = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
UL, UR, DL, DR = UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT
X_AXIS, Y_AXIS, Z_AXIS = RIGHT, UP, OUT

# Manim's named colors, copied from Manim CE 0.22.1 (``manim.utils.color.manim_colors``):
# ``color=RED`` reads the same, as an explicit color. (Before Milestone 20 the table was
# written from the documentation: YELLOW was #FFFF00, Manim's is #F7D96F.)
_COLORS = {
    "WHITE": "#FFFFFF", "BLACK": "#000000", "GRAY": "#888888", "GREY": "#888888",
    "LIGHT_GRAY": "#BBBBBB", "LIGHT_GREY": "#BBBBBB", "DARK_GRAY": "#444444",
    "DARK_GREY": "#444444", "LIGHTER_GRAY": "#DDDDDD", "LIGHTER_GREY": "#DDDDDD",
    "DARKER_GRAY": "#222222", "DARKER_GREY": "#222222",
    "GRAY_A": "#DDDDDD", "GRAY_B": "#BBBBBB", "GRAY_C": "#888888", "GRAY_D": "#444444",
    "GRAY_E": "#222222", "GREY_A": "#DDDDDD", "GREY_B": "#BBBBBB", "GREY_C": "#888888",
    "GREY_D": "#444444", "GREY_E": "#222222", "GRAY_BROWN": "#736357", "GREY_BROWN": "#736357",
    "RED": "#FC6255", "RED_A": "#F7A1A3", "RED_B": "#FF8080", "RED_C": "#FC6255",
    "RED_D": "#E65A4C", "RED_E": "#CF5044", "MAROON": "#C55F73", "MAROON_A": "#ECABC1",
    "MAROON_B": "#EC92AB", "MAROON_C": "#C55F73", "MAROON_D": "#A24D61", "MAROON_E": "#94424F",
    "PINK": "#D147BD", "LIGHT_PINK": "#DC75CD", "PURPLE": "#9A72AC", "PURPLE_A": "#CAA3E8",
    "PURPLE_B": "#B189C6", "PURPLE_C": "#9A72AC", "PURPLE_D": "#715582", "PURPLE_E": "#644172",
    "BLUE": "#58C4DD", "BLUE_A": "#C7E9F1", "BLUE_B": "#9CDCEB", "BLUE_C": "#58C4DD",
    "BLUE_D": "#29ABCA", "BLUE_E": "#236B8E", "DARK_BLUE": "#236B8E", "TEAL": "#5CD0B3",
    "TEAL_A": "#ACEAD7", "TEAL_B": "#76DDC0", "TEAL_C": "#5CD0B3", "TEAL_D": "#55C1A7",
    "TEAL_E": "#49A88F", "GREEN": "#83C167", "GREEN_A": "#C9E2AE", "GREEN_B": "#A6CF8C",
    "GREEN_C": "#83C167", "GREEN_D": "#77B05D", "GREEN_E": "#699C52", "YELLOW": "#F7D96F",
    "YELLOW_A": "#FFF1B6", "YELLOW_B": "#FFEA94", "YELLOW_C": "#F7D96F", "YELLOW_D": "#F4D345",
    "YELLOW_E": "#E8C11C", "GOLD": "#F0AC5F", "GOLD_A": "#F7C797", "GOLD_B": "#F9B775",
    "GOLD_C": "#F0AC5F", "GOLD_D": "#E1A158", "GOLD_E": "#C78D46", "ORANGE": "#FF862F",
    "DARK_BROWN": "#8B4513", "LIGHT_BROWN": "#CD853F", "PURE_RED": "#FF0000",
    "PURE_GREEN": "#00FF00", "PURE_BLUE": "#0000FF", "PURE_YELLOW": "#FFFF00",
    "PURE_CYAN": "#00FFFF", "PURE_MAGENTA": "#FF00FF", "LOGO_WHITE": "#ECE7E2",
    "LOGO_GREEN": "#87C2A5", "LOGO_BLUE": "#525893", "LOGO_RED": "#E07A5F",
    "LOGO_BLACK": "#343434",
}  # fmt: skip

TEXT_ROLES = (("title", 60), ("math", 40), ("label", 30), ("annotation", 0))  # font_size -> role
# Manim sizes in world units of the 8-unit frame: Manim's default font size (48) lands where
# the ``math`` text role does in the house themes (0.048 of the frame height), its default
# stroke width (4) where ``regular`` does (0.004).
FONT_UNIT = 0.048 * FRAME_HEIGHT / 48
STROKE_UNIT = 0.004 * FRAME_HEIGHT / 4
DEFAULT_DOT_RADIUS = 0.08
DEFAULT_ARROW_BUFF = 0.25  # Manim's ``Arrow`` stops this far short of both ends
UPDATE_RATE = 30  # frames a second at which updaters run (Manim: its frame rate)
MAX_FRAMES = 20_000  # updater frames in one import; past it the rate halves, with a warning
KEY_TOLERANCE = 1e-4  # sampled keys closer than this to a straight line between are dropped
RATE_PIECES = 16  # a rate function phasor has no name for is sampled into this many segments
SAMPLES = 72  # points round a circle (the box of a shape; phasor draws the true curve)


class ManimImportError(NotImplementedError):
    """A Manim feature phasor's import does not have, or a value it cannot carry over (the
    message says which, and for which object)."""


def _validation_message(e: ValidationError) -> str:
    """A pydantic error as one line per problem: ``style.size: bad size -1.0 ...``."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg'].removeprefix('Value error, ')}"
        for err in e.errors()
    )


@dataclass
class ManimImport:
    scene: PhasorScene
    warnings: list[str] = field(default_factory=list)
    duration: float = 0.0
    name: str = ""
    others: list[str] = field(default_factory=list)  # other Scene classes the file defines
    trace: list[dict] = field(default_factory=list)  # each play (tests/manim_real/compare.py)

    def document(self) -> Document:
        """The imported scene as a plain document: free to edit by hand or by an AI (a scene
        built by a script only takes tweaks), since the Manim script is not its source of
        truth any more."""
        doc = self.scene.engine.document
        raw = doc.model_dump(mode="json")
        raw["meta"]["source"] = "document"
        return Document.model_validate(raw)


def looks_like_manim(path: str | Path) -> bool:
    """Whether a .py file is a Manim script (it imports ``manim``)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    return bool(re.search(r"^\s*(from\s+manim(lib)?\s+import|import\s+manim)", text, re.M))


# ---------------------------------------------------------------- colors and sizes


def _hex_of(color: Any) -> str | None:
    """``#RRGGBB`` for a Manim color (a name, a hex string, a ManimColor-like object)."""
    if hasattr(color, "to_hex"):
        try:
            color = color.to_hex()
        except Exception:
            return None
    c = str(color)
    c = _COLORS.get(c.upper(), c)
    if not (c.startswith("#") and len(c) in (4, 7, 9)):
        return None
    if len(c) == 4:
        c = "#" + "".join(ch * 2 for ch in c[1:])
    return c[:7]


def color_role(color: Any) -> str:
    """The nearest phasor emphasis role for a Manim color (hex string or name)."""
    if color is None:
        return "primary"
    c = _hex_of(color)
    if c is None:
        return "primary"
    try:
        r, g, b = (int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))
    except ValueError:
        return "primary"
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.18 or v < 0.2:
        return "primary" if v > 0.62 else "muted"
    hue = h * 360
    if hue < 18 or hue >= 300:
        return "danger"
    if hue < 75:
        return "highlight"
    return "secondary"


class _Default(str):
    """A mobject class's own default color (the script did not set one): it becomes the
    nearest role, as phasor's own objects would be styled."""


# Class defaults (Manim's WHITE, RED, BLUE, YELLOW): roles, since the script did not choose them.
_WHITE, _RED, _BLUE = _Default("#FFFFFF"), _Default("#FC6255"), _Default("#58C4DD")
_YELLOW = _Default("#F7D96F")


def style_color(color: Any) -> str:
    """The phasor style value for a Manim color: the color the script set, as an explicit
    hex color; a class default, or a color phasor cannot read, as the nearest role."""
    if isinstance(color, _Default):
        return color_role(color)
    if hasattr(color, "to_hex"):
        color = _hex_of(color) or color
    c = _COLORS.get(str(color).upper(), str(color))
    try:
        if c.startswith("#"):
            return str(check_color(c))  # the document's own reading of a hex color
    except ValueError:
        pass
    if _CTX is not None:
        _CTX.warn(f"color {color!r} is not a hex color or Manim name: the nearest role is used")
    return color_role(color)


def text_role(font_size: float) -> str:
    return next(role for role, floor in TEXT_ROLES if font_size >= floor or floor == 0)


def carried_size(value: float, unit: float, what: str) -> float | None:
    """A Manim size (``what``: ``stroke_width=-1``, ...) times ``unit`` as a phasor explicit
    size in world units, or None (the role stays) with a warning when phasor's sizes cannot
    hold it: zero, negative, not a number or too large."""
    try:
        v = round(float(value) * unit, 6)
    except (TypeError, ValueError, OverflowError):
        v = float("nan")
    if 0.0 < v <= MAX_SIZE:
        return v
    if _CTX is not None:
        limit = f"a phasor size is a positive number of world units, at most {MAX_SIZE:g}"
        _CTX.warn(f"{what} is not carried over ({limit}): the role is used")
    return None


def _num(v: Any, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        raise ManimImportError(f"{what} must be a number, not {v!r}") from None
    if not math.isfinite(f):
        raise ManimImportError(f"{what} must be finite, not {v!r}")
    return f


def _vec3(v: Any) -> np.ndarray:
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        raise ManimImportError(f"{v!r} is not a point") from None
    if a.size == 0:
        return np.zeros(3)
    return np.array([a[0], a[1] if a.size > 1 else 0.0, a[2] if a.size > 2 else 0.0])


def _tuples(a: np.ndarray) -> list[tuple[float, float, float]]:
    return [tuple(round(float(x), 9) for x in row) for row in np.asarray(a, float)]


def _pt(v: np.ndarray) -> tuple[float, float, float]:
    return tuple(round(float(x), 9) for x in v)  # type: ignore[return-value]


def _rot(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_axis(angle: float, axis: Any) -> np.ndarray:
    """Rotation by ``angle`` about ``axis`` (Rodrigues)."""
    k = _vec3(axis)
    n = float(np.linalg.norm(k))
    if n == 0:
        return np.eye(3)
    k = k / n
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)


def _ring(r: float, start: float = 0.0, sweep: float = TAU, n: int = SAMPLES) -> np.ndarray:
    full = abs(sweep) >= TAU
    count = n if full else max(2, int(abs(sweep) / TAU * n) + 1)
    t = start + np.linspace(0.0, sweep, count, endpoint=not full)
    return np.stack([r * np.cos(t), r * np.sin(t), np.zeros(len(t))], axis=1)


# ---------------------------------------------------------------- rate functions (Manim's)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def smooth(t: float, inflection: float = 10.0) -> float:
    error = _sigmoid(-inflection / 2)
    return min(max((_sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0.0), 1.0)


def linear(t: float) -> float:
    return t


def smoothstep(t: float) -> float:
    return 0.0 if t <= 0 else 1.0 if t >= 1 else 3 * t**2 - 2 * t**3


def smootherstep(t: float) -> float:
    return 0.0 if t <= 0 else 1.0 if t >= 1 else 6 * t**5 - 15 * t**4 + 10 * t**3


def rush_into(t: float, inflection: float = 10.0) -> float:
    return 2 * smooth(t / 2.0, inflection)


def rush_from(t: float, inflection: float = 10.0) -> float:
    return 2 * smooth(t / 2.0 + 0.5, inflection) - 1


def slow_into(t: float) -> float:
    return math.sqrt(1 - (1 - t) * (1 - t))


def double_smooth(t: float) -> float:
    return 0.5 * smooth(2 * t) if t < 0.5 else 0.5 * (1 + smooth(2 * t - 1))


def there_and_back(t: float, inflection: float = 10.0) -> float:
    return smooth(2 * t if t < 0.5 else 2 * (1 - t), inflection)


def ease_in_sine(t: float) -> float:
    return 1 - math.cos((t * math.pi) / 2)


def ease_out_sine(t: float) -> float:
    return math.sin((t * math.pi) / 2)


def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2


def ease_in_cubic(t: float) -> float:
    return t * t * t


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def wiggle(t: float, wiggles: float = 2) -> float:
    return there_and_back(t) * math.sin(wiggles * math.pi * t)


rate_functions = types.SimpleNamespace(**{f.__name__: f for f in (
    smooth, linear, smoothstep, smootherstep, rush_into, rush_from, slow_into, double_smooth,
    there_and_back, ease_in_sine, ease_out_sine, ease_in_out_sine, ease_in_cubic,
    ease_out_cubic, ease_in_out_cubic, wiggle)})  # fmt: skip

# Rate functions phasor has an easing for, exactly (the same curve): one segment each.
_EASE = {linear: "linear", smooth: "sigmoid", smoothstep: "smooth", smootherstep: "smoother",
         ease_in_cubic: "in", ease_out_cubic: "out", ease_in_out_cubic: "in_out"}  # fmt: skip


def _rate_name(rate: Any) -> str:
    return getattr(rate, "__name__", "?")


def _rate(rate: Any, u: float) -> float:
    try:
        v = float(rate(min(max(u, 0.0), 1.0)))
    except Exception as e:
        raise ManimImportError(f"rate function {_rate_name(rate)!r} failed: {e}") from None
    if not math.isfinite(v):
        raise ManimImportError(f"rate function {_rate_name(rate)!r} gave {v}")
    return v


# ---------------------------------------------------------------- mobjects


class Mobject:
    """A thing on the Manim canvas. Leaves become one phasor object each; groups hold
    children and pass every change to them.

    A leaf's place: ``pts`` (points of its outline, world space: its box), ``origin`` (where
    its own centre went), ``s`` and ``theta`` (how much it was scaled and turned since it
    was made) and ``similar`` (False once it was stretched out of its kind: it then becomes
    a ``shapes.Path`` through ``pts``)."""

    kind = "group"
    outline = False  # a closed or open path phasor can fill, trace and morph
    id_base: str | None = None  # the id's stem (default: the class name)

    def __init__(self, *children: Mobject, color: Any = None, **kw: Any) -> None:
        self.children: list[Mobject] = [c for c in children if isinstance(c, Mobject)]
        self.color = style_color(color) if color is not None else None  # stroke: role or hex
        self.fill_color: str | None = None
        if kw.get("fill_color") is not None:
            self.fill_color = style_color(kw["fill_color"])
        self.fill_opacity = 0.0
        if kw.get("fill_opacity") is not None:
            self.fill_opacity = min(max(_num(kw["fill_opacity"], "fill_opacity"), 0.0), 1.0)
        self.stroke_opacity = 1.0
        if kw.get("stroke_opacity") is not None:
            self.stroke_opacity = min(max(_num(kw["stroke_opacity"], "stroke_opacity"), 0.0), 1.0)
        if kw.get("stroke_color") is not None:
            self.color = style_color(kw["stroke_color"])
        self.stroke: float | None = None
        self._set_width(kw.get("stroke_width"), f"stroke_width={kw.get('stroke_width')!r}")
        self.dash = "solid"
        self.opacity = 1.0
        self.pts = np.zeros((0, 3))
        self.origin = np.zeros(3)
        self.s, self.theta, self.similar = 1.0, 0.0, True
        self.z = 0
        self.updaters: list[Callable] = []
        self.suspended = False
        self.obj_id: str | None = None
        self.ptype: str | None = None  # the phasor type it became
        self.alias: Mobject | None = None  # after a Transform: this is now that mobject's
        self.target: Mobject | None = None
        self.saved: Mobject | None = None
        self.name = type(self).__name__  # the Manim class (the trace; real Manim's names)

    def _set_width(self, width: Any, what: str) -> None:
        if width is None:
            return
        try:
            w = float(width)
        except (TypeError, ValueError):
            w = float("nan")
        if w == 0:  # Manim's "no outline" (common on filled shapes): no stroke, no warning
            self.stroke_opacity = 0.0
            return
        self.stroke = carried_size(width, STROKE_UNIT, what)

    # -- bookkeeping
    def _warn(self, msg: str) -> None:
        if _CTX is not None:
            _CTX.warn(msg)

    def leaves(self) -> list[Mobject]:
        if self.alias is not None:
            return self.alias.leaves()
        if not self.children:
            return [self] if self.kind != "group" else []
        return [leaf for ch in self.children for leaf in ch.leaves()]

    def get_family(self) -> list[Mobject]:
        if self.alias is not None:
            return [self, *self.alias.get_family()[1:]]
        return [self, *(m for ch in self.children for m in ch.get_family())]

    family_members_with_points = leaves

    @property
    def submobjects(self) -> list[Mobject]:
        return self.children

    @property
    def animate(self) -> _AnimateBuilder:
        return _AnimateBuilder(self)

    def copy(self) -> Mobject:
        if self.alias is not None:  # a copy of what it became
            return self.alias.copy()
        c = copy.copy(self)
        c.children = [ch.copy() for ch in self.children]
        c.pts = self.pts.copy()
        c.origin = self.origin.copy()
        c.updaters = []
        c.obj_id, c.ptype, c.alias, c.target, c.saved = None, None, None, None, None
        return c

    def generate_target(self) -> Mobject:
        self.target = self.copy()
        return self.target

    def save_state(self) -> Mobject:
        self.saved = self.copy()
        return self

    def restore(self) -> Mobject:
        if self.saved is None:
            raise ManimImportError(f"{type(self).__name__}.restore needs save_state() first")
        self._take(self.saved)
        return self

    # -- geometry
    def _bbox(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = None, None
        for leaf in self.leaves():
            a, b = leaf._leaf_box()
            if a is None:
                continue
            lo = a if lo is None else np.minimum(lo, a)
            hi = b if hi is None else np.maximum(hi, b)
        if lo is None:
            z = np.zeros(3)
            return z, z
        return lo, hi

    def _leaf_box(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not len(self.pts):
            return None, None
        return self.pts.min(axis=0), self.pts.max(axis=0)

    def get_center(self) -> np.ndarray:
        lo, hi = self._bbox()
        return (lo + hi) / 2

    def get_critical_point(self, direction: Any) -> np.ndarray:
        d = _vec3(direction)
        lo, hi = self._bbox()
        c = (lo + hi) / 2
        return np.where(d > 0, hi, np.where(d < 0, lo, c))

    get_corner = get_critical_point
    get_edge_center = get_critical_point
    get_boundary_point = get_critical_point

    def get_left(self):
        return self.get_critical_point(LEFT)

    def get_right(self):
        return self.get_critical_point(RIGHT)

    def get_top(self):
        return self.get_critical_point(UP)

    def get_bottom(self):
        return self.get_critical_point(DOWN)

    def get_start(self) -> np.ndarray:
        leaves = [x for x in self.leaves() if len(x.pts)]
        return leaves[0].pts[0].copy() if leaves else self.get_center()

    def get_end(self) -> np.ndarray:
        leaves = [x for x in self.leaves() if len(x.pts)]
        return leaves[-1].pts[-1].copy() if leaves else self.get_center()

    def get_x(self) -> float:
        return float(self.get_center()[0])

    def get_y(self) -> float:
        return float(self.get_center()[1])

    def get_z(self) -> float:
        return float(self.get_center()[2])

    @property
    def width(self) -> float:
        lo, hi = self._bbox()
        return float(hi[0] - lo[0])

    @width.setter
    def width(self, value: float) -> None:
        self.set_width(value)

    @property
    def height(self) -> float:
        lo, hi = self._bbox()
        return float(hi[1] - lo[1])

    @height.setter
    def height(self, value: float) -> None:
        self.set_height(value)

    def get_width(self) -> float:
        return self.width

    def get_height(self) -> float:
        return self.height

    def length_over_dim(self, dim: int) -> float:
        lo, hi = self._bbox()
        return float(hi[dim] - lo[dim])

    # every change of place goes through _affine (one way)
    def _affine(self, m: np.ndarray, about: np.ndarray, kind: str, value: float = 0.0) -> Mobject:
        for leaf in self.leaves():
            leaf.pts = about + (leaf.pts - about) @ m.T
            leaf.origin = about + (leaf.origin - about) @ m.T
            leaf._after(kind, value)
        return self

    def _after(self, kind: str, value: float) -> None:
        """What a leaf keeps of a change besides its points (its scale, its turn)."""
        if kind == "scale":
            self.s *= abs(value)
            if value < 0:
                self.theta += PI
        elif kind == "rotate":
            self.theta += value
        elif kind == "other":
            self.similar = False

    def shift(self, *vecs: Any) -> Mobject:
        if not vecs:
            return self
        d = np.sum([_vec3(v) for v in vecs], axis=0)
        if not np.all(np.isfinite(d)):
            raise ManimImportError(f"shift by {d.tolist()} (not finite)")
        for leaf in self.leaves():
            leaf.pts = leaf.pts + d
            leaf.origin = leaf.origin + d
        return self

    def _about(self, about_point: Any, about_edge: Any = None) -> np.ndarray:
        if about_point is not None:
            return _vec3(about_point)
        if about_edge is not None:
            return self.get_critical_point(about_edge)
        return self.get_center()

    def move_to(self, point_or_mobject: Any, aligned_edge: Any = None,
                coor_mask: Any = None) -> Mobject:  # fmt: skip
        target = point_or_mobject
        if isinstance(target, Mobject):
            p = (target.get_critical_point(aligned_edge) if aligned_edge is not None
                 else target.get_center())  # fmt: skip
        else:
            p = _vec3(target)
        me = (
            self.get_critical_point(aligned_edge) if aligned_edge is not None else self.get_center()
        )
        d = p - me
        if coor_mask is not None:
            d = d * _vec3(coor_mask)
        return self.shift(d)

    def set_x(self, x: float) -> Mobject:
        return self.shift(RIGHT * (_num(x, "x") - self.get_x()))

    def set_y(self, y: float) -> Mobject:
        return self.shift(UP * (_num(y, "y") - self.get_y()))

    def scale(self, scale_factor: float, about_point: Any = None, about_edge: Any = None,
              **kw: Any) -> Mobject:  # fmt: skip
        f = _num(scale_factor, "scale factor")
        c = self._about(about_point, about_edge)
        self._affine(np.eye(3) * f, c, "scale", f)
        for leaf in self.leaves():
            leaf._rescaled(abs(f), scale_factor)
        return self

    def _rescaled(self, f: float, factor: Any) -> None:
        return None

    def stretch(self, factor: float, dim: int, about_point: Any = None, **kw: Any) -> Mobject:
        f = _num(factor, "stretch factor")
        m = np.eye(3)
        m[int(dim), int(dim)] = f
        return self._affine(m, self._about(about_point), "other" if f != 1 else "none")

    def rotate(self, angle: float, axis: Any = OUT, about_point: Any = None,
               about_edge: Any = None, **kw: Any) -> Mobject:  # fmt: skip
        a = _num(angle, "rotation angle")
        ax = _vec3(axis)
        c = self._about(about_point, about_edge)
        if np.allclose(ax[:2], 0) and ax[2] != 0:
            a = a if ax[2] > 0 else -a
            return self._affine(_rot(a), c, "rotate", a)
        return self._affine(_rot_axis(a, ax), c, "other")

    def flip(self, axis: Any = UP, **kw: Any) -> Mobject:
        return self.rotate(PI, axis, **kw)

    def apply_matrix(self, matrix: Any, about_point: Any = None, **kw: Any) -> Mobject:
        m = np.eye(3)
        a = np.asarray(matrix, float)
        m[: a.shape[0], : a.shape[1]] = a
        return self._affine(m, ORIGIN if about_point is None else _vec3(about_point), "other")

    def apply_function(self, function: Callable, **kw: Any) -> Mobject:
        for leaf in self.leaves():
            if len(leaf.pts):
                leaf.pts = np.array([_vec3(function(p)) for p in leaf.pts])
            leaf.origin = _vec3(function(leaf.origin))
            leaf.similar = False
        return self

    def rescale_to_fit(self, length: float, dim: int, stretch: bool = False, **kw: Any) -> Mobject:
        old = self.length_over_dim(dim)
        if old == 0:
            return self
        if stretch:
            return self.stretch(length / old, dim, **kw)
        return self.scale(length / old, **kw)

    def set_width(self, width: float, stretch: bool = False, **kw: Any) -> Mobject:
        return self.rescale_to_fit(_num(width, "width"), 0, stretch, **kw)

    def set_height(self, height: float, stretch: bool = False, **kw: Any) -> Mobject:
        return self.rescale_to_fit(_num(height, "height"), 1, stretch, **kw)

    scale_to_fit_width = set_width
    scale_to_fit_height = set_height

    def stretch_to_fit_width(self, width: float, **kw: Any) -> Mobject:
        return self.rescale_to_fit(_num(width, "width"), 0, True, **kw)

    def stretch_to_fit_height(self, height: float, **kw: Any) -> Mobject:
        return self.rescale_to_fit(_num(height, "height"), 1, True, **kw)

    def match_width(self, other: Mobject, **kw: Any) -> Mobject:
        return self.set_width(other.width, **kw)

    def match_height(self, other: Mobject, **kw: Any) -> Mobject:
        return self.set_height(other.height, **kw)

    def set(self, **kw: Any) -> Mobject:
        """``set(width=4)``, ``set(color=RED)``, ``set(fill_opacity=0.5)``."""
        for k, v in kw.items():
            setter = getattr(type(self), f"set_{k}", None)
            if setter is None:
                raise ManimImportError(f"{type(self).__name__}.set({k}=...) is not supported by "
                                       "phasor's Manim import yet")  # fmt: skip
            setter(self, v)
        return self

    def set_fill_opacity(self, opacity: float) -> Mobject:
        return self.set_fill(opacity=opacity)

    def set_stroke_width(self, width: float) -> Mobject:
        return self.set_stroke(width=width)

    def next_to(self, mobject_or_point: Any, direction: Any = RIGHT, buff: float = 0.25,
                aligned_edge: Any = ORIGIN, **kw: Any) -> Mobject:  # fmt: skip
        d = _vec3(direction)
        edge = _vec3(aligned_edge)
        other = mobject_or_point
        is_mob = isinstance(other, Mobject)
        anchor = other.get_critical_point(d + edge) if is_mob else _vec3(other)
        mine = self.get_critical_point(edge - d)
        return self.shift(anchor - mine + d * _num(buff, "buff"))

    def to_edge(self, edge: Any = LEFT, buff: float = 0.5) -> Mobject:
        e = _vec3(edge)
        lo, hi = self._bbox()
        half = np.array([FRAME_WIDTH / 2, FRAME_HEIGHT / 2, 0.0])
        move = np.zeros(3)
        for i in (0, 1):
            if e[i] > 0:
                move[i] = half[i] - buff - hi[i]
            elif e[i] < 0:
                move[i] = -half[i] + buff - lo[i]
        return self.shift(move)

    def to_corner(self, corner: Any = DL, buff: float = 0.5) -> Mobject:
        return self.to_edge(corner, buff)

    def center(self) -> Mobject:
        return self.move_to(ORIGIN)

    def align_to(self, mobject_or_point: Any, direction: Any = UP, **kw: Any) -> Mobject:
        d = _vec3(direction)
        other = mobject_or_point
        point = other.get_critical_point(d) if isinstance(other, Mobject) else _vec3(other)
        mine = self.get_critical_point(d)
        return self.shift(np.where(d != 0, point - mine, 0.0))

    def arrange(self, direction: Any = RIGHT, buff: float = 0.25, center: bool = True,
                aligned_edge: Any = ORIGIN, **kw: Any) -> Mobject:  # fmt: skip
        prev = None
        for ch in self.children:
            if prev is not None:
                ch.next_to(prev, direction, buff, aligned_edge=aligned_edge)
            prev = ch
        if center and self.children:
            self.move_to(ORIGIN)
        return self

    def arrange_in_grid(self, rows: int | None = None, cols: int | None = None,
                        buff: float = 0.25, **kw: Any) -> Mobject:  # fmt: skip
        n = len(self.children)
        if not n:
            return self
        cols = cols or (math.ceil(n / rows) if rows else math.ceil(math.sqrt(n)))
        w = max(c.width for c in self.children) + buff
        h = max(c.height for c in self.children) + buff
        for k, ch in enumerate(self.children):
            ch.move_to(np.array([(k % cols) * w, -(k // cols) * h, 0.0]))
        return self.move_to(ORIGIN)

    # -- look
    def set_color(self, color: Any = None, family: bool = True, **kw: Any) -> Mobject:
        if color is None:
            return self
        value = style_color(color)
        self.color, self.fill_color = value, value
        for ch in self.children:
            ch.set_color(color)
        return self

    def get_color(self) -> str | None:
        leaves = self.leaves()
        return leaves[0].color if leaves else self.color

    def set_opacity(self, opacity: float, **kw: Any) -> Mobject:
        o = min(max(_num(opacity, "opacity"), 0.0), 1.0)
        for leaf in self.leaves():
            leaf.stroke_opacity = o
            leaf.fill_opacity = o
            if not leaf.outline:
                leaf.opacity = o
        return self

    def fade(self, darkness: float = 0.5, **kw: Any) -> Mobject:
        return self.set_opacity(1.0 - _num(darkness, "darkness"))

    def set_fill(self, color: Any = None, opacity: float | None = None, **kw: Any) -> Mobject:
        for leaf in self.leaves():
            if color is not None:
                leaf.fill_color = style_color(color)
                if not leaf.outline:  # a dot or text: its fill is its color
                    leaf.color = leaf.fill_color
            if opacity is not None:
                leaf.fill_opacity = min(max(_num(opacity, "fill opacity"), 0.0), 1.0)
        return self

    def set_stroke(self, color: Any = None, width: float | None = None,
                   opacity: float | None = None, **kw: Any) -> Mobject:  # fmt: skip
        for leaf in self.leaves():
            if color is not None:
                leaf.color = style_color(color)
            if width is not None:
                leaf.stroke_opacity = 1.0
                leaf._set_width(width, f"set_stroke(width={width!r})")
            if opacity is not None:
                leaf.stroke_opacity = min(max(_num(opacity, "stroke opacity"), 0.0), 1.0)
        return self

    def set_z_index(self, z_index: int, **kw: Any) -> Mobject:
        for leaf in self.leaves():
            leaf.z = int(z_index)
        return self

    def add(self, *mobs: Mobject) -> Mobject:
        self.children.extend(m for m in mobs if isinstance(m, Mobject))
        return self

    def remove(self, *mobs: Mobject) -> Mobject:
        self.children = [c for c in self.children if c not in mobs]
        return self

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return VGroup(*self.children[i])
        return self.children[i]

    def __iter__(self):
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise ManimImportError(f"{type(self).__name__}.{name} is not supported by phasor's "
                               "Manim import yet")  # fmt: skip

    # -- updaters (run by the stand-in at UPDATE_RATE; their effect becomes tracks)
    def add_updater(self, update_function: Callable, index: int | None = None,
                    call_updater: bool = False) -> Mobject:  # fmt: skip
        if not callable(update_function):
            raise ManimImportError("add_updater needs a function")
        self.updaters.insert(len(self.updaters) if index is None else index, update_function)
        if call_updater:
            _call_updater(self, update_function, 0.0)
        return self

    def remove_updater(self, update_function: Callable) -> Mobject:
        self.updaters = [u for u in self.updaters if u is not update_function]
        return self

    def clear_updaters(self, recursive: bool = True) -> Mobject:
        for m in self.get_family() if recursive else [self]:
            m.updaters = []
        return self

    def suspend_updating(self, recursive: bool = True) -> Mobject:
        for m in self.get_family() if recursive else [self]:
            m.suspended = True
        return self

    def resume_updating(self, recursive: bool = True) -> Mobject:
        for m in self.get_family() if recursive else [self]:
            m.suspended = False
        return self

    def get_updaters(self) -> list[Callable]:
        return list(self.updaters)

    def become(self, mobject: Mobject, **kw: Any) -> Mobject:
        """Take ``mobject``'s place, shape and look (``always_redraw`` uses it each frame)."""
        mine, theirs = self.leaves(), mobject.leaves()
        if len(mine) != len(theirs) or any(type(a) is not type(b)
                                           for a, b in zip(mine, theirs, strict=True)):  # fmt: skip
            self._warn(f"{type(self).__name__}.become: a mobject with other parts keeps its "
                       "first parts (phasor objects cannot change kind)")  # fmt: skip
            return self
        self._take(mobject)
        return self

    def _take(self, other: Mobject) -> None:
        """Copy ``other``'s geometry and look into this mobject (leaf by leaf)."""
        mine, theirs = self.leaves(), other.leaves()
        if len(mine) != len(theirs):
            return
        keep = {"children", "obj_id", "ptype", "alias", "target", "saved", "updaters",
                "suspended", "name", "_spec", "_look"}  # fmt: skip
        for a, b in zip(mine, theirs, strict=True):
            for k, v in b.__dict__.items():
                if k in keep:
                    continue
                a.__dict__[k] = v.copy() if isinstance(v, np.ndarray) else copy.copy(v)

    # -- what phasor gets (leaves)
    def _style(self, **extra: Any) -> dict[str, Any]:
        """Style for the phasor object: the stroke color (a role or the script's hex color),
        a stroke width the script set (world units), the fill color, plus ``extra``."""
        out = dict(extra)
        if self.color:
            out["color"] = self.color
        if self.stroke is not None and (self.outline or self.kind in ("arrow", "axes")):
            out["stroke"] = self.stroke
        if self.outline:
            out["fill"] = "solid"
            out["fill_color"] = self.fill_color or self.color or "primary"
            if self.dash != "solid":
                out["dash"] = self.dash
        return out

    def _style_extra(self) -> dict[str, Any]:
        return {}

    def _spec(self) -> tuple[Any, dict[str, Any]]:  # pragma: no cover (leaves override)
        raise NotImplementedError

    def _look(self) -> dict[str, Any]:
        """Parameters of the look Manim animates besides geometry."""
        if self.outline:
            return {"fill_opacity": self.fill_opacity, "stroke_opacity": self.stroke_opacity}
        return {}

    def phasor_spec(self) -> tuple[Any, dict[str, Any]]:
        """The phasor class and parameters this leaf is now: its own kind while it keeps
        its shape, else a ``shapes.Path`` through its outline."""
        cls, params = self._spec()
        return cls, {**params, **self._look()}

    def phasor_object(self, ident: str):
        cls, params = self.phasor_spec()
        extra = {"z_index": self.z} if self.z else {}
        return cls(id=ident, opacity=0.0, style=self._style(**self._style_extra()), **extra,
                   **params)  # fmt: skip


def _call_updater(mob: Mobject, fn: Callable, dt: float) -> None:
    try:
        params = inspect.signature(fn).parameters
        wants_dt = "dt" in params or len(params) >= 2
    except (TypeError, ValueError):
        wants_dt = False
    try:
        fn(mob, dt) if wants_dt else fn(mob)
    except ManimImportError:
        raise
    except Exception as e:
        raise ManimImportError(f"an updater of {type(mob).__name__} failed: "
                               f"{type(e).__name__}: {e}") from None  # fmt: skip


class VMobject(Mobject):
    """A free outline (``set_points_as_corners``, ``VMobject().add(...)``)."""

    outline = True
    closed = False

    def __init__(self, color: Any = _WHITE, **kw: Any) -> None:
        super().__init__(color=color, **kw)

    def set_points_as_corners(self, points: Any) -> VMobject:
        pts = [_vec3(p) for p in points]
        self.pts = np.array(pts) if pts else np.zeros((0, 3))
        self.kind = "path" if len(self.pts) >= 2 else "group"
        self.origin = self.pts.mean(axis=0) if len(self.pts) else np.zeros(3)
        self.similar = False
        return self

    set_points_smoothly = set_points_as_corners

    def _spec(self):
        return _path_spec(self.pts, closed=False)


def _path_spec(pts: np.ndarray, closed: bool) -> tuple[Any, dict[str, Any]]:
    from phasor.domains import path_math as pm
    from phasor.domains import shapes

    sub = pm.polyline([tuple(p) for p in np.asarray(pts, float)], closed=closed)
    return shapes.Path, {"subpaths": [_tuples(sub)]}


class VGroup(Mobject):
    kind = "group"

    def __init__(self, *mobs: Mobject, **kw: Any) -> None:
        super().__init__(*mobs, **kw)


Group = VGroup


class Dot(Mobject):
    kind = "dot"

    def __init__(self, point: Any = ORIGIN, radius: float = DEFAULT_DOT_RADIUS,
                 color: Any = _WHITE, **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        self.origin = _vec3(point)
        self.r0 = abs(_num(radius, "radius"))
        self.fill_opacity = 1.0
        self.pts = self.origin[None, :].copy()

    def _leaf_box(self):
        r = np.array([self.r0 * self.s, self.r0 * self.s, 0.0])
        return self.origin - r, self.origin + r

    def _spec(self):
        from phasor import Dot as D

        return D, {"pos": _pt(self.origin), "r": max(self.r0 * self.s, 1e-6)}


SmallDot = Dot


class _Shape(Mobject):
    """A closed or open outline with a phasor shape of its own (``_named``) while it keeps
    its kind; ``pts`` is its outline in the world (its box)."""

    kind = "shape"
    outline = True
    closed = True

    def __init__(self, local: np.ndarray, color: Any = _WHITE, **kw: Any) -> None:
        super().__init__(color=color, **kw)
        self.pts = np.asarray(local, float).reshape(-1, 3).copy()
        if self.fill_color is None and self.color is not None:
            self.fill_color = self.color

    def _named(self) -> tuple[Any, dict[str, Any]] | None:
        return None

    def _spec(self):
        named = self._named() if self.similar else None
        if named is not None:
            return named
        return _path_spec(self.pts, closed=self.closed)

    def _deg(self, a: float) -> float:
        return round(math.degrees(a), 9)


class Circle(_Shape):
    def __init__(self, radius: float | None = None, color: Any = _RED,
                 arc_center: Any = ORIGIN, **kw: Any) -> None:  # fmt: skip
        self.r0 = abs(_num(1.0 if radius is None else radius, "radius"))
        super().__init__(_ring(self.r0) + _vec3(arc_center), color, **kw)
        self.origin = _vec3(arc_center)

    def _named(self):
        from phasor.domains import shapes

        r = self.r0 * self.s
        return (shapes.Circle, {"pos": _pt(self.origin), "r": r}) if r > 0 else None

    def point_at_angle(self, angle: float) -> np.ndarray:
        a = _num(angle, "angle") + self.theta
        return self.origin + self.r0 * self.s * np.array([math.cos(a), math.sin(a), 0.0])

    def surround(self, mobject: Mobject, buffer_factor: float = 1.2, **kw: Any) -> Circle:
        self.move_to(mobject)
        w = max(mobject.width, mobject.height) * _num(buffer_factor, "buffer_factor")
        return self.set_width(w) if self.width and w else self


class Ellipse(_Shape):
    def __init__(self, width: float = 2.0, height: float = 1.0, color: Any = _WHITE,
                 **kw: Any) -> None:  # fmt: skip
        self.w0, self.h0 = abs(_num(width, "width")), abs(_num(height, "height"))
        ring = _ring(1.0) * np.array([self.w0 / 2, self.h0 / 2, 0.0])
        super().__init__(ring, color, **kw)

    def _named(self):
        from phasor.domains import shapes

        w, h = self.w0 * self.s, self.h0 * self.s
        if w <= 0 or h <= 0:
            return None
        return shapes.Ellipse, {"pos": _pt(self.origin), "width": w, "height": h,
                                "rotation": self._deg(self.theta)}  # fmt: skip


class Arc(_Shape):
    closed = False

    def __init__(self, radius: float = 1.0, start_angle: float = 0.0, angle: float = TAU / 4,
                 arc_center: Any = ORIGIN, color: Any = _WHITE, **kw: Any) -> None:  # fmt: skip
        self.r0 = abs(_num(radius, "radius"))
        self.a0, self.sweep = _num(start_angle, "start_angle"), _num(angle, "angle")
        super().__init__(_ring(self.r0, self.a0, self.sweep) + _vec3(arc_center), color, **kw)
        self.origin = _vec3(arc_center)

    def _named(self):
        from phasor.domains import shapes

        r, sweep = self.r0 * self.s, math.degrees(self.sweep)
        if r <= 0 or not (0 < abs(sweep) <= 360):
            return None
        return shapes.Arc, {
            "pos": _pt(self.origin),
            "r": r,
            "start": self._deg(self.a0 + self.theta),
            "angle": round(sweep, 9),
        }


class ArcBetweenPoints(Arc):
    def __init__(self, start: Any, end: Any, angle: float = TAU / 4, **kw: Any) -> None:
        a, b = _vec3(start), _vec3(end)
        sweep = _num(angle, "angle")
        chord = float(np.linalg.norm(b - a))
        if chord == 0 or abs(math.sin(sweep / 2)) < 1e-9:
            super().__init__(radius=0.0, **kw)
            self.pts = np.array([a, b])
            self.similar = False
            return
        r = chord / (2 * math.sin(sweep / 2))
        mid = (a + b) / 2
        normal = np.array([-(b - a)[1], (b - a)[0], 0.0]) / chord
        center = mid + normal * r * math.cos(sweep / 2)
        start_angle = math.atan2(a[1] - center[1], a[0] - center[0])
        super().__init__(radius=abs(r), start_angle=start_angle, angle=sweep, arc_center=center,
                         **kw)  # fmt: skip


class Sector(_Shape):
    def __init__(self, radius: float = 1.0, angle: float = TAU / 4, start_angle: float = 0.0,
                 inner_radius: float = 0.0, arc_center: Any = ORIGIN, color: Any = _WHITE,
                 outer_radius: float | None = None, fill_opacity: float = 1.0,
                 **kw: Any) -> None:  # fmt: skip
        self.r0 = abs(_num(outer_radius if outer_radius is not None else radius, "radius"))
        self.inner0 = abs(_num(inner_radius, "inner_radius"))
        self.a0, self.sweep = _num(start_angle, "start_angle"), _num(angle, "angle")
        rim = np.vstack([_ring(self.r0, self.a0, self.sweep),
                         _ring(self.inner0, self.a0, self.sweep)[::-1]])  # fmt: skip
        super().__init__(rim + _vec3(arc_center), color, fill_opacity=fill_opacity, **kw)
        self.origin = _vec3(arc_center)

    def _named(self):
        from phasor.domains import shapes

        r, inner = self.r0 * self.s, self.inner0 * self.s
        sweep, start = math.degrees(self.sweep), self.a0 + self.theta
        if sweep < 0:
            start, sweep = start + self.sweep, -sweep
        if r <= 0 or not (0 < sweep <= 360) or inner >= r:
            return None
        return shapes.Sector, {"pos": _pt(self.origin), "r": r, "start": self._deg(start),
                               "angle": round(sweep, 9), "inner": inner}  # fmt: skip


AnnularSector = Sector


class Annulus(_Shape):
    def __init__(self, inner_radius: float = 1.0, outer_radius: float = 2.0,
                 color: Any = _WHITE, fill_opacity: float = 1.0,
                 **kw: Any) -> None:  # fmt: skip
        self.inner0 = abs(_num(inner_radius, "inner_radius"))
        self.outer0 = abs(_num(outer_radius, "outer_radius"))
        super().__init__(_ring(max(self.inner0, self.outer0)), color,
                         fill_opacity=fill_opacity, **kw)  # fmt: skip

    def _named(self):
        from phasor.domains import shapes

        inner, outer = self.inner0 * self.s, self.outer0 * self.s
        if not (0 < inner < outer):
            return None
        return shapes.Annulus, {"pos": _pt(self.origin), "outer": outer, "inner": inner}


def _distinct(pts: np.ndarray) -> int:
    return len({tuple(np.round(p, 9)) for p in pts})


class Polygon(_Shape):
    def __init__(self, *vertices: Any, color: Any = _BLUE, **kw: Any) -> None:
        super().__init__(np.array([_vec3(v) for v in vertices]).reshape(-1, 3), color, **kw)
        self.origin = self.pts.mean(axis=0) if len(self.pts) else np.zeros(3)

    def get_vertices(self) -> np.ndarray:
        return self.pts.copy()

    def _spec(self):
        from phasor.domains import shapes

        if len(self.pts) >= 3 and _distinct(self.pts) >= 3:
            return shapes.Polygon, {"vertices": _tuples(self.pts)}
        return _path_spec(self.pts, closed=True)


Polygram = Polygon


class RegularPolygon(Polygon):
    def __init__(self, n: int = 6, radius: float = 1.0, start_angle: float | None = None,
                 color: Any = _BLUE, **kw: Any) -> None:  # fmt: skip
        self.n = int(_num(n, "n"))
        if start_angle is None:  # Manim: a vertex at the right for even n, at the top for odd
            start_angle = 0.0 if self.n % 2 == 0 else TAU / 4
        self.r0, self.a0 = abs(_num(radius, "radius")), _num(start_angle, "start_angle")
        t = self.a0 + np.arange(min(max(self.n, 0), 10_000)) * TAU / max(self.n, 1)
        super().__init__(*[(self.r0 * math.cos(a), self.r0 * math.sin(a), 0) for a in t],
                         color=color, **kw)  # fmt: skip
        self.origin = np.zeros(3)

    def _spec(self):
        from phasor.domains import shapes

        r = self.r0 * self.s
        if self.similar and 3 <= self.n <= 64 and r > 0:
            # phasor's first vertex is at the top for rotation 0
            rot = math.degrees(self.a0 + self.theta) - 90.0
            return shapes.RegularPolygon, {"n": self.n, "pos": _pt(self.origin), "r": r,
                                           "rotation": round(rot, 9)}  # fmt: skip
        return super()._spec()


RegularPolygram = RegularPolygon


class Triangle(RegularPolygon):
    def __init__(self, **kw: Any) -> None:
        super().__init__(3, **kw)


class Rectangle(Polygon):
    def __init__(self, color: Any = _WHITE, height: float = 2.0, width: float = 4.0,
                 corner_radius: float = 0.0, **kw: Any) -> None:  # fmt: skip
        self.w0, self.h0 = _num(width, "width"), _num(height, "height")
        self.cr0 = abs(_num(corner_radius, "corner_radius"))
        w, h = self.w0 / 2, self.h0 / 2
        super().__init__((w, h, 0), (-w, h, 0), (-w, -h, 0), (w, -h, 0), color=color, **kw)
        self.origin = np.zeros(3)

    def _spec(self):
        from phasor.domains import shapes

        w, h = abs(self.w0) * self.s, abs(self.h0) * self.s
        if self.similar and w > 0 and h > 0:
            return shapes.Rectangle, {"pos": _pt(self.origin), "width": w, "height": h,
                                      "corner_radius": min(self.cr0 * self.s, w / 2, h / 2),
                                      "rotation": self._deg(self.theta)}  # fmt: skip
        if self.cr0:
            self._warn("a rounded rectangle stretched out of shape is drawn with sharp corners")
        return super()._spec()


class Square(Rectangle):
    def __init__(self, side_length: float = 2.0, **kw: Any) -> None:
        super().__init__(height=side_length, width=side_length, **kw)


class RoundedRectangle(Rectangle):
    def __init__(self, corner_radius: float = 0.5, **kw: Any) -> None:
        super().__init__(corner_radius=corner_radius, **kw)


class Line(_Shape):
    closed = False

    def __init__(self, start: Any = LEFT, end: Any = RIGHT, buff: float = 0.0,
                 color: Any = _WHITE, **kw: Any) -> None:  # fmt: skip
        a, b = self._ends(start, end)
        d = b - a
        n = float(np.linalg.norm(d))
        bf = _num(buff, "buff")
        if bf and n > 2 * bf:
            a, b = a + d / n * bf, b - d / n * bf
        super().__init__(np.array([a, b]), color, **kw)
        self.origin = (a + b) / 2

    @staticmethod
    def _ends(start: Any, end: Any) -> tuple[np.ndarray, np.ndarray]:
        a = start.get_center() if isinstance(start, Mobject) else _vec3(start)
        b = end.get_center() if isinstance(end, Mobject) else _vec3(end)
        return a, b

    def get_start(self) -> np.ndarray:
        return self.pts[0].copy()

    def get_end(self) -> np.ndarray:
        return self.pts[-1].copy()

    def get_length(self) -> float:
        return float(np.linalg.norm(self.pts[-1] - self.pts[0]))

    def get_angle(self) -> float:
        d = self.pts[-1] - self.pts[0]
        return math.atan2(d[1], d[0])

    def put_start_and_end_on(self, start: Any, end: Any) -> Line:
        self.pts = np.array([_vec3(start), _vec3(end)])
        self.origin = self.pts.mean(axis=0)
        return self

    def set_points_as_corners(self, points: Any) -> Line:
        self.pts = np.array([_vec3(p) for p in points])
        return self


class DashedLine(Line):
    def __init__(self, *a: Any, dash_length: float = 0.05, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.dash = "dashed"


class DashedVMobject(VGroup):
    """``DashedVMobject(mob)``: ``mob`` with a dashed stroke (phasor's ``dash`` style)."""

    def __init__(self, vmobject: Mobject, num_dashes: int = 15, **kw: Any) -> None:
        dashed = vmobject.copy()
        for leaf in dashed.leaves():
            if leaf.outline:
                leaf.dash = "dashed"
        super().__init__(dashed)


class TangentLine(Line):
    def __init__(self, vmob: Mobject, alpha: float, length: float = 1.0, **kw: Any) -> None:
        pts = vmob.leaves()[0].pts if vmob.leaves() else np.zeros((2, 3))
        k = min(int(_num(alpha, "alpha") * (len(pts) - 1)), len(pts) - 2)
        p, d = pts[k], pts[k + 1] - pts[k]
        d = d / (np.linalg.norm(d) or 1.0) * _num(length, "length") / 2
        super().__init__(p - d, p + d, **kw)


class Arrow(Mobject):
    kind = "arrow"

    def __init__(self, start: Any = LEFT, end: Any = RIGHT, buff: float = DEFAULT_ARROW_BUFF,
                 color: Any = _WHITE, **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        a, b = Line._ends(start, end)
        d = b - a
        n = float(np.linalg.norm(d))
        bf = _num(buff, "buff")
        if bf and n > 2 * bf:
            a, b = a + d / n * bf, b - d / n * bf
        self.pts = np.array([a, b])
        self.origin = (a + b) / 2

    get_start = Line.get_start
    get_end = Line.get_end
    get_length = Line.get_length

    def put_start_and_end_on(self, start: Any, end: Any) -> Arrow:
        self.pts = np.array([_vec3(start), _vec3(end)])
        return self

    def _spec(self):
        from phasor import linalg

        return linalg.Arrow, {"start": _pt(self.pts[0]), "end": _pt(self.pts[1])}


class Vector(Arrow):
    def __init__(self, direction: Any = RIGHT, buff: float = 0.0, **kw: Any) -> None:
        super().__init__(ORIGIN, direction, buff=buff, **kw)


DoubleArrow = Arrow


class ParametricFunction(Line):
    id_base = "graph"

    def __init__(self, function: Callable[[float], Any], t_range: Sequence[float] = (0, 1),
                 color: Any = _YELLOW, **kw: Any) -> None:  # fmt: skip
        lo, hi = _num(t_range[0], "t_range"), _num(t_range[1], "t_range")
        pts = []
        with np.errstate(all="ignore"):
            for t in np.linspace(lo, hi, 240):
                try:
                    p = _vec3(function(t))
                except (ZeroDivisionError, ValueError, OverflowError, ManimImportError):
                    continue
                if np.all(np.isfinite(p)):
                    pts.append(p)
        super().__init__(color=color, **kw)
        if len(pts) < 2:
            self._warn("a curve with fewer than two finite points draws nothing")
            self.kind = "group"
            return
        self.pts = np.array(pts)
        self.origin = self.pts.mean(axis=0)
        self.similar = False


FunctionGraph = ParametricFunction


class Brace(_Shape):
    """``Brace(mob, direction)``: a brace along the side of ``mob`` facing ``direction``."""

    closed = False

    def __init__(self, mobject: Mobject, direction: Any = DOWN, buff: float = 0.2,
                 color: Any = _WHITE, **kw: Any) -> None:  # fmt: skip
        d = _vec3(direction)
        n = float(np.linalg.norm(d)) or 1.0
        d = d / n
        side = np.array([-d[1], d[0], 0.0])  # start -> end along the side; the tip points d
        lo, hi = mobject._bbox()
        c = (lo + hi) / 2
        half = abs(float(np.dot((hi - lo) / 2, side)))
        reach = abs(float(np.dot((hi - lo) / 2, d)))
        base = c + d * (reach + _num(buff, "buff"))
        self.depth0 = 0.25
        start, end = base + side * half, base - side * half
        super().__init__(np.array([start, end, base + d * self.depth0]), color, **kw)
        self.origin = base
        self.dir = d

    def _named(self):
        from phasor.domains import shapes

        start, end = self.pts[0], self.pts[1]
        if float(np.linalg.norm(end - start)) <= 0:
            return None
        return shapes.Brace, {"start": _pt(start), "end": _pt(end),
                              "depth": self.depth0 * self.s}  # fmt: skip

    def _look(self) -> dict[str, Any]:
        return {}

    def _style(self, **extra: Any) -> dict[str, Any]:
        out = super()._style(**extra)
        out.pop("fill", None)
        out.pop("fill_color", None)
        return out

    def get_tip(self) -> np.ndarray:
        return self.pts[2].copy()

    def put_at_tip(self, mob: Mobject, buff: float = 0.2, **kw: Any) -> Mobject:
        return mob.next_to(self.pts[2], self.dir, buff=buff)

    def get_text(self, *text: str, **kw: Any) -> Text:
        t = Text(*text, **kw)
        self.put_at_tip(t)
        return t

    def get_tex(self, *tex: str, **kw: Any) -> MathTex:
        t = MathTex(*tex, **kw)
        self.put_at_tip(t)
        return t


class BraceBetweenPoints(Brace):
    def __init__(self, point_1: Any, point_2: Any, direction: Any = None, **kw: Any) -> None:
        a, b = _vec3(point_1), _vec3(point_2)
        d = b - a
        if direction is None:
            direction = np.array([d[1], -d[0], 0.0])
        super().__init__(Line(a, b), direction, buff=0.0, **kw)


def _scene_file(name: Any, what: str) -> str | None:
    """A file a script names, as a path relative to the scene's folder (where Manim looks
    first); None, with a warning, for one outside the folder."""
    if _CTX is None:
        return None
    raw = str(name)
    base = Path(_CTX.base_dir).resolve()
    try:
        p = Path(raw)
        full = (p if p.is_absolute() else base / p).resolve()
        rel = full.relative_to(base).as_posix()
        core_files.relative_path(rel)
    except (ValueError, OSError, RuntimeError):
        _CTX.warn(f"{what}: {raw!r} is not inside the scene's folder: left out (a scene reads "
                  "files only from its own folder)")  # fmt: skip
        return None
    if not core_files.exists(base, rel):
        _CTX.warn(f"{what}: {raw!r} is not there (lint missing_file until it is)")
    return rel


def _svg_aspect(rel: str | None) -> float:
    """Width over height of an SVG's view box (1 when it cannot be read)."""
    if rel is None or _CTX is None:
        return 1.0
    try:
        with core_files.regular_file(_CTX.base_dir, rel).open("rb") as f:
            head = f.read(4096).decode("utf-8", "replace")
    except (ValueError, OSError):
        return 1.0
    m = re.search(r'viewBox\s*=\s*"\s*[-\d.eE]+[\s,]+[-\d.eE]+[\s,]+([\d.eE+]+)[\s,]+([\d.eE+]+)',
                  head)  # fmt: skip
    try:
        w, h = (float(m.group(1)), float(m.group(2))) if m else (1.0, 1.0)
    except ValueError:
        return 1.0
    return w / h if w > 0 and h > 0 and math.isfinite(w / h) else 1.0


class SVGMobject(_Shape):
    """An SVG file next to the scene (``shapes.SVG``): its shapes are parts phasor reads
    itself, so the stand-in only knows its box (``height`` tall, centred)."""

    def __init__(self, file_name: Any = "", height: float | None = 2.0, color: Any = None,
                 **kw: Any) -> None:  # fmt: skip
        self.path = _scene_file(file_name, "SVGMobject")
        self.h0 = abs(_num(2.0 if height is None else height, "height"))
        w = self.h0 * _svg_aspect(self.path)
        box = np.array([(-w / 2, -self.h0 / 2, 0), (w / 2, self.h0 / 2, 0)], float)
        super().__init__(box, _WHITE if color is None else color, **kw)
        if self.path is None:
            self.kind = "group"  # nothing to draw (said in a warning)

    def _spec(self):
        from phasor.domains import shapes

        return shapes.SVG, {"path": self.path, "pos": _pt(self.origin), "height": self.h0 * self.s}

    def _look(self) -> dict[str, Any]:
        return {}

    def _style(self, **extra: Any) -> dict[str, Any]:
        out = {k: v for k, v in super()._style(**extra).items() if k not in ("fill", "dash")}
        out["fill_color"] = self.color or "primary"
        return out


class Text(Mobject):
    kind = "text"

    def __init__(self, *strings: Any, font_size: float | None = None, color: Any = _WHITE,
                 math: bool = False, tex: bool = False, **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **{k: v for k, v in kw.items() if k != "font"})
        raw = "".join(str(s) for s in strings)
        size = None if font_size is None else carried_size(font_size, FONT_UNIT,
                                                           f"font_size={font_size!r}")  # fmt: skip
        self.role = text_role(48 if size is None else float(font_size))
        self.raw = raw
        if math:
            self.text = raw if "$" in raw else f"${raw}$"
        elif tex:  # Tex: text mode, with $...$ maths inside
            self.text = raw
        else:
            self.text = raw.replace("$", r"\$")
        if raw.strip():
            try:
                text_metrics.check_math(self.text)
            except ValueError as e:
                self._warn(f"{raw!r} could not be typeset ({e}); shown as plain text")
                self.text = raw.replace("$", "").replace("\\", "")
        else:
            self.kind = "group"  # an empty text draws nothing (as in Manim)
        self.fill_opacity = 1.0
        self.pts = np.zeros((1, 3))
        self._ref = FRAME_HEIGHT * {"title": 0.075, "math": 0.048, "label": 0.042,
                                    "annotation": 0.032}[self.role]  # fmt: skip
        self.text_size: str | float = self.role  # none set: the role (Manim's default)
        if size is not None:
            self._ref = self.text_size = size
        self.as_shapes = False

    @property
    def glyphs(self) -> int:
        """How many glyphs Manim draws (it sets ``Write``'s run time): for TeX, a command
        is one glyph (``\\pi``), a fraction adds its bar, spacing and grouping draw none."""
        s = self.raw
        if "\\" in s or "$" in s or isinstance(self, MathTex):
            s = re.sub(r"\\[dt]?frac", "|", s)
            s = re.sub(r"\\(left|right|text\w*|math\w*|operatorname|displaystyle|quad|qquad)"
                       r"(?![a-zA-Z])|\\[,;:! ]", "", s)  # fmt: skip
            s = re.sub(r"\\[a-zA-Z]+", "x", s)
        return sum(1 for ch in s if not ch.isspace() and ch not in "{}$^_&")

    def _leaf_box(self):
        w, h = text_metrics.aspect(self.text) * self._ref, text_metrics.box(self.text) * self._ref
        pad = np.array([w / 2, h / 2, 0.0])
        return self.origin - pad, self.origin + pad

    def _after(self, kind: str, value: float) -> None:
        super()._after(kind, value)
        if kind in ("rotate", "other") and _CTX is not None:
            _CTX.warn("rotated or stretched text is drawn level, at its new place")

    def _rescaled(self, f: float, factor: Any) -> None:
        self._ref *= f
        size = carried_size(self._ref, 1.0, f"a text size after scale({factor!r})")
        self.text_size = self.role if size is None else size

    def get_text(self) -> str:
        return self.raw

    def _style_extra(self) -> dict[str, Any]:
        return {"size": self.text_size}

    def _spec(self):
        from phasor import Text as T

        return T, {"text": self.text, "pos": _pt(self.origin), "as_shapes": self.as_shapes}


class MathTex(Text):
    """``MathTex("a", "+", "b")``: one string is a math ``Text``; several are an
    ``Equation`` with a part each (``TransformMatchingTex`` pairs them)."""

    def __init__(self, *strings: Any, tex_template: Any = None, **kw: Any) -> None:
        super().__init__(*strings, math=True, **kw)
        self.parts = [str(s) for s in strings] if len(strings) > 1 else None
        self.colors: list[str] = []
        if self.parts:
            self.text = "$" + " ".join(self.parts) + "$"

    def __getitem__(self, i: Any) -> Any:
        if self.parts is None:
            raise ManimImportError("indexing a one-string MathTex: give the parts as separate "
                                   "strings (MathTex('a', '+', 'b'))")  # fmt: skip
        return _TexPart(self, i)

    def get_part_by_tex(self, tex: str, **kw: Any) -> Any:
        if self.parts and tex in self.parts:
            return _TexPart(self, self.parts.index(tex))
        raise ManimImportError(f"MathTex has no part {tex!r}")

    def set_color_by_tex(self, tex: str, color: Any, **kw: Any) -> MathTex:
        for k, p in enumerate(self.parts or []):
            if tex in p:
                _TexPart(self, k).set_color(color)
        return self

    def _spec(self):
        if not self.parts:
            return super()._spec()
        from phasor import Equation

        parts = [f"${p}$" if p.strip() else "$\\,$" for p in self.parts]
        params: dict[str, Any] = {"parts": parts, "pos": _pt(self.origin),
                                  "as_shapes": self.as_shapes}  # fmt: skip
        if self.colors:
            params["colors"] = list(self.colors)
        return Equation, params


class _TexPart:
    """``eq[1]``: one part of a several-part MathTex. Coloring it colors that part (the
    nearest role: an equation colors its parts by role); anything else is said."""

    def __init__(self, eq: MathTex, i: Any) -> None:
        self.eq, self.i = eq, int(i)
        if not 0 <= self.i < len(eq.parts or []):
            raise IndexError(i)

    def set_color(self, color: Any = None, **kw: Any) -> _TexPart:
        roles = list(self.eq.colors) or ["primary"] * len(self.eq.parts or [])
        roles[self.i] = color_role(color)
        self.eq.colors = roles
        if not isinstance(color, _Default) and _CTX is not None:
            _CTX.warn("a MathTex part's color becomes the nearest role (an equation's parts "
                      "take roles)")  # fmt: skip
        return self

    def get_center(self) -> np.ndarray:
        return self.eq.get_center()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise ManimImportError(f"a MathTex part's {name} is not supported by phasor's Manim "
                               "import yet (only set_color)")  # fmt: skip


class Tex(Text):
    def __init__(self, *strings: Any, tex_template: Any = None, **kw: Any) -> None:
        super().__init__(*strings, tex=True, **kw)


Title = Text
MarkupText = Text
Paragraph = Text


class DecimalNumber(Text):
    """A number on screen (``Counter``): ``set_value`` (and an updater that calls it)
    changes it."""

    def __init__(self, number: float = 0.0, num_decimal_places: int = 2, color: Any = _WHITE,
                 include_sign: bool = False, group_with_commas: bool = True,
                 **kw: Any) -> None:  # fmt: skip
        self.value = _num(number, "number")
        self.digits = min(max(int(_num(num_decimal_places, "num_decimal_places")), 0), 10)
        self.sign, self.group = bool(include_sign), bool(group_with_commas)
        super().__init__(f"{self.value:.{self.digits}f}", color=color, **kw)
        self.kind = "counter"

    def set_value(self, number: float) -> DecimalNumber:
        self.value = _num(number, "number")
        self.raw = self.text = f"{self.value:.{self.digits}f}"
        return self

    def get_value(self) -> float:
        return self.value

    def increment_value(self, delta_t: float = 1.0) -> DecimalNumber:
        return self.set_value(self.value + _num(delta_t, "increment"))

    def _spec(self):
        from phasor import Counter

        return Counter, {"value": self.value, "digits": self.digits, "pos": _pt(self.origin),
                         "sign": self.sign, "group": self.group}  # fmt: skip


class Integer(DecimalNumber):
    def __init__(self, number: float = 0, num_decimal_places: int = 0, **kw: Any) -> None:
        super().__init__(number, num_decimal_places=num_decimal_places, **kw)


class ValueTracker(Mobject):
    """A number to animate, with no drawing: a phasor ``Value``."""

    kind = "value"

    def __init__(self, value: float = 0.0, **kw: Any) -> None:
        super().__init__()
        self.value = _num(value, "ValueTracker value")
        if _CTX is not None:
            _CTX.realize(self)

    def _leaf_box(self):
        return None, None

    def get_value(self) -> float:
        return self.value

    def set_value(self, value: float) -> ValueTracker:
        self.value = _num(value, "ValueTracker value")
        if _CTX is not None and not _CTX.in_frames and self.obj_id:
            _CTX.instant(self, "value", self.value)
        return self

    def increment_value(self, d_value: float) -> ValueTracker:
        return self.set_value(self.value + _num(d_value, "increment"))

    def __iadd__(self, d_value: float) -> ValueTracker:
        return self.increment_value(d_value)

    def _spec(self):
        from phasor import Value

        return Value, {"value": self.value}

    def phasor_object(self, ident: str):
        cls, params = self._spec()
        return cls(id=ident, **params)


# ---------------------------------------------------------------- the diagram kit


def _range(r: Sequence[Any] | None, default: tuple, what: str, note: Callable[[str], None]):
    """A Manim range ``[min, max, step]`` as phasor takes it: finite, min < max, a step that
    makes at most ``ticks.MAX_TICKS`` ticks (widened to a nice one, said in ``note``)."""
    v = [*list(r if r is not None else default), 1][:3]
    lo, hi, step = (_num(x, what) for x in v)
    if not hi > lo:
        raise ManimImportError(f"{what} {list(v)}: the end must be past the start")
    wider = ticks.step_of(lo, hi, step) if step > 0 else step
    if wider != step:  # past phasor's tick limit: a wider nice step, said here
        note(ticks.clamp_note(lo, hi, step, what))
    return [lo, hi, wider]


class Axes(Mobject):
    kind = "axes"

    def __init__(
        self,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        x_length: float | None = None,
        y_length: float | None = None,
        color: Any = _WHITE,
        axis_config: dict | None = None,
        x_axis_config: dict | None = None,
        y_axis_config: dict | None = None,
        tips: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(color=color, **kw)
        note = lambda m: self._warn(f"axes {m}")  # noqa: E731
        self.xr = _range(x_range, (-7, 7, 1), "x_range", note)
        self.yr = _range(y_range, (-4, 4, 1), "y_range", note)
        self.x_length = abs(_num(12.0 if x_length is None else x_length, "x_length"))
        self.y_length = abs(_num(6.0 if y_length is None else y_length, "y_length"))
        configs = [c for c in (axis_config, x_axis_config, y_axis_config) if isinstance(c, dict)]
        self.numbers = any(c.get("include_numbers") for c in configs)
        self.labels: tuple[str, ...] = ()
        xr, yr = self.xr, self.yr
        self.origin = np.array(
            [-(xr[0] + xr[1]) / 2 * self.sx, -(yr[0] + yr[1]) / 2 * self.sy, 0.0]
        )
        self.pts = np.array([(-self.x_length / 2, -self.y_length / 2, 0),
                             (self.x_length / 2, self.y_length / 2, 0)], float)  # fmt: skip

    @property
    def sx(self) -> float:
        return self.x_length * self.s / (self.xr[1] - self.xr[0])

    @property
    def sy(self) -> float:
        return self.y_length * self.s / (self.yr[1] - self.yr[0])

    def c2p(self, x: Any, y: float = 0.0, z: float = 0.0) -> np.ndarray:
        if isinstance(x, list | tuple | np.ndarray):
            v = _vec3(x)
            x, y = v[0], v[1]
        return self.origin + np.array([_num(x, "x") * self.sx, _num(y, "y") * self.sy, 0.0])

    coords_to_point = c2p

    def p2c(self, point: Any) -> tuple[float, float]:
        p = _vec3(point) - self.origin
        return float(p[0] / self.sx), float(p[1] / self.sy)

    point_to_coords = p2c

    def i2gp(self, x: float, graph: Mobject) -> np.ndarray:
        leaves = graph.leaves()
        if not leaves or not len(leaves[0].pts):
            return self.c2p(x, 0)
        g = leaves[0].pts
        xs = (g[:, 0] - self.origin[0]) / self.sx
        return g[int(np.argmin(abs(xs - _num(x, "x"))))].copy()

    input_to_graph_point = i2gp

    def get_origin(self) -> np.ndarray:
        return self.c2p(0, 0)

    def get_x_axis(self) -> Mobject:
        return Line(self.c2p(self.xr[0], 0), self.c2p(self.xr[1], 0))

    def get_y_axis(self) -> Mobject:
        return Line(self.c2p(0, self.yr[0]), self.c2p(0, self.yr[1]))

    def plot(self, function: Callable[[float], float], x_range: Sequence[float] | None = None,
             color: Any = _YELLOW, **kw: Any) -> Mobject:  # fmt: skip
        lo, hi = (x_range or self.xr)[:2]
        pts = []
        with np.errstate(all="ignore"):
            for x in np.linspace(_num(lo, "x_range"), _num(hi, "x_range"), 240):
                try:
                    y = float(function(x))
                except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                    continue
                if math.isfinite(y):
                    pts.append(self.c2p(x, y))
        g = ParametricFunction(lambda t: (t, 0, 0), color=color,
                               **{k: v for k, v in kw.items() if k == "stroke_width"})  # fmt: skip
        if len(pts) < 2:
            self._warn("a plot with fewer than two finite points draws nothing")
            g.kind = "group"
            return g
        g.pts, g.origin = np.array(pts), np.array(pts).mean(axis=0)
        return g

    get_graph = plot

    def plot_line_graph(self, x_values: Sequence[float], y_values: Sequence[float],
                        line_color: Any = _YELLOW, **kw: Any) -> Mobject:  # fmt: skip
        g = VMobject(color=line_color)
        return g.set_points_as_corners([self.c2p(x, y) for x, y in zip(x_values, y_values,
                                                                       strict=False)])  # fmt: skip

    def get_axis_labels(self, x_label: Any = "x", y_label: Any = "y", **kw: Any) -> Mobject:
        """The axis names become the axes' own ``labels`` (phasor places them at the ends);
        the group returned is empty."""
        self.labels = tuple(_label_text(v) for v in (x_label, y_label))
        if self.obj_id and _CTX is not None:
            _CTX.instant(self, "labels", list(self.labels))
        return VGroup()

    def add_coordinates(self, *a: Any, **kw: Any) -> Axes:
        self.numbers = True
        if self.obj_id and _CTX is not None:
            _CTX.instant(self, "numbers", True)
        return self

    def get_area(self, *a: Any, **kw: Any) -> Mobject:
        self._warn("Axes.get_area is not carried over (draw a filled shapes.Path)")
        return VGroup()

    def _spec(self):
        from phasor import geo

        unequal = {"y_scale": self.sy} if abs(self.sx - self.sy) > 1e-9 * self.sx else {}
        params: dict[str, Any] = {"dims": 2, "x_range": tuple(self.xr), "y_range": tuple(self.yr),
                                  "pos": _pt(self.origin), "scale": self.sx, **unequal}  # fmt: skip
        if self.numbers:
            params["numbers"] = True
        if self.labels:
            params["labels"] = self.labels
        return geo.Axes, params


def _label_text(v: Any) -> str:
    if isinstance(v, Text):
        return v.text
    s = str(v)
    return s if "$" in s or not s.strip() else f"${s}$"


class NumberPlane(Axes):
    kind = "number_plane"

    def __init__(self, x_range: Sequence[float] | None = None,
                 y_range: Sequence[float] | None = None, x_length: float | None = None,
                 y_length: float | None = None, **kw: Any) -> None:  # fmt: skip
        xr = x_range if x_range is not None else (-FRAME_WIDTH / 2, FRAME_WIDTH / 2, 1)
        yr = y_range if y_range is not None else (-FRAME_HEIGHT / 2, FRAME_HEIGHT / 2, 1)
        xl = x_length if x_length is not None else _num(xr[1], "x_range") - _num(xr[0], "x_range")
        yl = y_length if y_length is not None else _num(yr[1], "y_range") - _num(yr[0], "y_range")
        super().__init__(xr, yr, abs(xl), abs(yl), **kw)

    def _spec(self):
        from phasor import dg

        unequal = {"y_scale": self.sy} if abs(self.sx - self.sy) > 1e-9 * self.sx else {}
        return dg.NumberPlane, {"x_range": tuple(self.xr), "y_range": tuple(self.yr),
                                "pos": _pt(self.c2p(0, 0)), "scale": self.sx, **unequal,
                                "numbers": self.numbers}  # fmt: skip


class ComplexPlane(NumberPlane):
    def n2p(self, number: complex) -> np.ndarray:
        z = complex(number)
        return self.c2p(z.real, z.imag)

    number_to_point = n2p


class NumberLine(Mobject):
    kind = "number_line"

    def __init__(self, x_range: Sequence[float] | None = None, length: float | None = None,
                 include_numbers: bool = False, include_tip: bool = False, color: Any = _WHITE,
                 unit_size: float = 1.0, **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        self.xr = _range(x_range, (-8, 8, 1), "x_range", lambda m: self._warn(f"number line {m}"))
        span = self.xr[1] - self.xr[0]
        self.length = abs(_num(span * _num(unit_size, "unit_size") if length is None else length,
                               "length"))  # fmt: skip
        self.numbers, self.tip = bool(include_numbers), bool(include_tip)
        mid = -(self.xr[0] + self.xr[1]) / 2 * self.unit  # Manim centres the line's length
        self.origin = np.array([0.0, 0.0, 0.0])
        self.pts = np.array([(-self.length / 2, -0.1, 0), (self.length / 2, 0.1, 0)], float)
        self._mid = mid

    @property
    def unit(self) -> float:
        return self.length * self.s / (self.xr[1] - self.xr[0])

    def _dir(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta), 0.0])

    def n2p(self, number: float) -> np.ndarray:
        mid = (self.xr[0] + self.xr[1]) / 2
        return self.origin + self._dir() * (_num(number, "number") - mid) * self.unit

    number_to_point = n2p

    def p2n(self, point: Any) -> float:
        mid = (self.xr[0] + self.xr[1]) / 2
        return float(np.dot(_vec3(point) - self.origin, self._dir()) / self.unit + mid)

    point_to_number = p2n

    def add_numbers(self, *a: Any, **kw: Any) -> NumberLine:
        self.numbers = True
        if self.obj_id and _CTX is not None:
            _CTX.instant(self, "numbers", True)
        return self

    def _spec(self):
        from phasor import dg

        return dg.NumberLine, {"x_range": tuple(self.xr), "pos": _pt(self.origin),
                               "scale": self.unit, "angle": round(math.degrees(self.theta), 9),
                               "numbers": self.numbers,
                               "tips": "end" if self.tip else "none"}  # fmt: skip


class BarChart(Mobject):
    kind = "bar_chart"

    def __init__(self, values: Sequence[float], bar_names: Sequence[str] | None = None,
                 y_range: Sequence[float] | None = None, x_length: float | None = None,
                 y_length: float | None = None, bar_colors: Any = None,
                 color: Any = _WHITE, **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        self.values = [_num(v, "bar value") for v in values]
        self.names = [str(n) for n in bar_names] if bar_names else []
        self.yr = [_num(v, "y_range") for v in y_range][:3] if y_range else []
        self.w0 = abs(_num(FRAME_WIDTH - 4 if x_length is None else x_length, "x_length"))
        self.h0 = abs(_num(FRAME_HEIGHT - 4 if y_length is None else y_length, "y_length"))
        self.bar_colors = [color_role(c) for c in bar_colors] if bar_colors is not None else []
        if bar_colors is not None:
            self._warn("bar colors become the nearest roles (a chart colors its bars by role)")
        self.pts = np.array([(-self.w0 / 2, -self.h0 / 2, 0), (self.w0 / 2, self.h0 / 2, 0)], float)

    def change_bar_values(self, values: Sequence[float], **kw: Any) -> BarChart:
        self.values = [_num(v, "bar value") for v in values]
        return self

    def _spec(self):
        from phasor import dg

        params: dict[str, Any] = {"values": tuple(self.values), "pos": _pt(self.origin),
                                  "width": self.w0 * self.s,
                                  "height": self.h0 * self.s}  # fmt: skip
        if self.names:
            params["names"] = tuple(self.names)
        if len(self.yr) >= 2:
            params["y_range"] = tuple([*self.yr, 0.0][:3])
        if self.bar_colors:
            params["colors"] = tuple(self.bar_colors)
        return dg.BarChart, params


def _cell_text(v: Any) -> str:
    if isinstance(v, Text):
        return v.text
    if isinstance(v, Mobject):
        raise ManimImportError(f"a cell that is a {type(v).__name__} is not supported (text and "
                               "numbers are)")  # fmt: skip
    return str(v)


def _cells_box(cells: list[list[str]], gap: float = 0.6) -> tuple[float, float]:
    ref = FRAME_HEIGHT * 0.042
    cols = list(zip(*cells, strict=False)) or [("",)]
    widths = [max(text_metrics.aspect(c or " ") * ref for c in col) for col in cols]
    return sum(widths) + gap * len(widths), len(cells) * ref * 1.8


class Table(Mobject):
    kind = "table"
    math_cells = False

    def __init__(self, table: Sequence[Sequence[Any]], row_labels: Any = None,
                 col_labels: Any = None, include_outer_lines: bool = False, color: Any = _WHITE,
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        rows = [[self._cell(v) for v in row] for row in table]
        self.header = col_labels is not None
        if col_labels is not None:
            rows.insert(0, [_cell_text(c) for c in col_labels])
        if row_labels is not None:
            labels = [_cell_text(c) for c in row_labels]
            if self.header:
                labels.insert(0, "")
            rows = [[lab, *row] for lab, row in zip(labels, rows, strict=False)]
        self.cells = rows
        self.lines = "grid" if include_outer_lines else "header" if self.header else "none"
        w, h = _cells_box(rows)
        self.pts = np.array([(-w / 2, -h / 2, 0), (w / 2, h / 2, 0)], float)

    def _cell(self, v: Any) -> str:
        if isinstance(v, Mobject):
            return _cell_text(v)
        s = str(v)
        return f"${s}$" if self.math_cells and "$" not in s else s

    def _spec(self):
        from phasor import dg

        return dg.Table, {"cells": tuple(tuple(r) for r in self.cells), "pos": _pt(self.origin),
                          "header": self.header, "lines": self.lines}  # fmt: skip


class MathTable(Table):
    math_cells = True


IntegerTable = DecimalTable = MathTable
MobjectTable = Table


class Matrix(Mobject):
    kind = "matrix"

    def __init__(self, matrix: Sequence[Sequence[Any]], color: Any = _WHITE,
                 left_bracket: str = "[", **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        self.entries = [[_cell_text(v).strip("$") for v in row] for row in matrix]
        self.bracket = {"[": "square", "(": "round", "|": "bar"}.get(str(left_bracket), "square")
        w, h = _cells_box(self.entries)
        self.pts = np.array([(-w / 2, -h / 2, 0), (w / 2, h / 2, 0)], float)

    def get_entries(self) -> VGroup:
        self._warn("a Matrix's entries are parts of one object: they move with it")
        return VGroup()

    def _spec(self):
        from phasor import dg

        return dg.MatrixDisplay, {"entries": tuple(tuple(r) for r in self.entries),
                                  "bracket": self.bracket, "pos": _pt(self.origin)}  # fmt: skip


IntegerMatrix = DecimalMatrix = MobjectMatrix = Matrix


class Code(Mobject):
    kind = "code"

    def __init__(self, code_file: Any = None, code_string: str | None = None,
                 language: str | None = None, add_line_numbers: bool = True,
                 color: Any = None, **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **{k: v for k, v in kw.items() if k == "stroke_width"})
        source = code_string if code_string is not None else kw.get("code")
        file = code_file if code_file is not None else kw.get("file_name")
        self.path = _scene_file(file, "Code") if source is None and file is not None else None
        self.code = str(source or "")
        self.language = str(language or (Path(str(file)).suffix.lstrip(".") if file else "python"))
        self.numbers = bool(add_line_numbers)
        lines = self.code.splitlines() or [""]
        if self.path is not None and _CTX is not None:
            try:
                with core_files.regular_file(_CTX.base_dir, self.path).open("rb") as f:
                    lines = f.read(64_000).decode("utf-8", "replace").splitlines()[:120] or [""]
            except (ValueError, OSError):
                pass
        ref = FRAME_HEIGHT * 0.032
        w = max(len(x) for x in lines) * 0.52 * ref + 0.5
        h = len(lines) * ref * 1.4
        self.pts = np.array([(-w / 2, -h / 2, 0), (w / 2, h / 2, 0)], float)
        if not self.code and self.path is None:
            self.kind = "group"

    def _spec(self):
        from phasor import dg

        src = {"path": self.path} if self.path else {"code": self.code}
        return dg.Code, {**src, "language": self.language, "pos": _pt(self.origin),
                         "line_numbers": self.numbers}  # fmt: skip


def always_redraw(func: Callable[[], Mobject]) -> Mobject:
    """Manim's own definition: the mobject ``func`` makes, made again every frame."""
    mob = func()
    if not isinstance(mob, Mobject):
        raise ManimImportError("always_redraw needs a function that returns a mobject")
    mob.add_updater(lambda m: m.become(func()))
    return mob


def always(method: Callable, *args: Any, **kwargs: Any) -> Mobject:
    mob = method.__self__
    mob.add_updater(lambda m: getattr(m, method.__name__)(*args, **kwargs))
    return mob


def f_always(method: Callable, *arg_generators: Callable, **kwargs: Any) -> Mobject:
    mob = method.__self__
    mob.add_updater(lambda m: getattr(m, method.__name__)(*(g() for g in arg_generators), **kwargs))
    return mob


# ---------------------------------------------------------------- animations


class Animation:
    default_time = 1.0
    default_rate: Callable = smooth
    remover = False

    def __init__(self, mobject: Mobject | None = None, run_time: float | None = None,
                 rate_func: Any = None, lag_ratio: float = 0.0, **kw: Any) -> None:  # fmt: skip
        self.mob = mobject
        self.run_time = None if run_time is None else _num(run_time, "run_time")
        if self.run_time is not None and self.run_time < 0:
            raise ManimImportError(f"run_time {run_time!r} is negative")
        self.rate_func = rate_func if rate_func is not None else type(self).default_rate

    @property
    def duration(self) -> float:
        return self.run_time if self.run_time is not None else self.default_time

    def mobjects(self) -> list[Mobject]:
        return [self.mob] if self.mob is not None else []

    def apply(self, ctx: _Context, t0: float, d: float) -> None:  # pragma: no cover
        raise NotImplementedError


def _fade(ctx: _Context, mob: Mobject, t0: float, d: float, a: float, b: float, rate: Any) -> None:
    for leaf in mob.leaves():
        if leaf.kind == "value":
            continue
        ctx.show(leaf, t0, opacity=a)
        ctx.track(leaf, "opacity", b, t0, d, rate, start=a)
        leaf.opacity = b


class FadeIn(Animation):
    def __init__(self, *mobjects: Mobject, shift: Any = None, target_position: Any = None,
                 scale: float = 1.0, **kw: Any) -> None:  # fmt: skip
        super().__init__(mobjects[0] if len(mobjects) == 1 else VGroup(*mobjects), **kw)
        self.shift = None if shift is None else _vec3(shift)
        if target_position is not None and len(mobjects) == 1:
            p = (target_position.get_center() if isinstance(target_position, Mobject)
                 else _vec3(target_position))  # fmt: skip
            self.shift = self.mob.get_center() - p
        self.scale = _num(scale, "scale")

    def apply(self, ctx, t0, d):
        if self.shift is not None or self.scale != 1:
            start = self.mob.copy()
            if self.scale != 1:
                start.scale(self.scale)
            if self.shift is not None:
                start.shift(-self.shift)
            for leaf, s in zip(self.mob.leaves(), start.leaves(), strict=True):
                ctx.move(leaf, s, leaf, t0, d, self.rate_func)
        _fade(ctx, self.mob, t0, d, 0.0, 1.0, self.rate_func)


class FadeOut(Animation):
    remover = True

    def __init__(self, *mobjects: Mobject, shift: Any = None, scale: float = 1.0,
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(mobjects[0] if len(mobjects) == 1 else VGroup(*mobjects), **kw)
        self.shift = None if shift is None else _vec3(shift)
        self.scale = _num(scale, "scale")

    def apply(self, ctx, t0, d):
        if self.shift is not None or self.scale != 1:
            goal = self.mob.copy()
            if self.scale != 1:
                goal.scale(self.scale)
            if self.shift is not None:
                goal.shift(self.shift)
            for leaf, g in zip(self.mob.leaves(), goal.leaves(), strict=True):
                ctx.move(leaf, leaf, g, t0, d, self.rate_func)
        _fade(ctx, self.mob, t0, d, 1.0, 0.0, self.rate_func)
        ctx.gone(self.mob)


def _draw_in(ctx: _Context, mob: Mobject, t0: float, d: float, rate: Any, lag: float) -> None:
    """Draw each leaf in (a path traced, text written, the rest faded), each ``lag`` of its
    run time after the last (Manim's ``lag_ratio``)."""
    leaves = [x for x in mob.leaves() if x.kind != "value"]
    n = len(leaves)
    span = d / (1 + lag * (n - 1)) if n > 1 and lag else d
    for k, leaf in enumerate(leaves):
        s = t0 + k * span * lag if n > 1 and lag else t0
        if leaf.kind == "text":
            ctx.realize(leaf)
            ctx.show(leaf, s, write=0.0)
            ctx.track(leaf, "write", 1.0, s, span, rate, start=0.0)
        elif leaf.outline:
            ctx.show(leaf, s, progress=0.0)
            ctx.track(leaf, "progress", 1.0, s, span, rate, start=0.0)
        else:
            _fade(ctx, leaf, s, span, 0.0, 1.0, rate)


class Create(Animation):
    def __init__(self, mobject: Mobject, lag_ratio: float = 1.0, **kw: Any) -> None:
        super().__init__(mobject, **kw)
        self.lag = _num(lag_ratio, "lag_ratio")

    def apply(self, ctx, t0, d):
        _draw_in(ctx, self.mob, t0, d, self.rate_func, self.lag)


ShowCreation = Create


class Write(Create):
    """Manim's defaults: linear, 1 s (2 s from 15 glyphs or parts), each part ``lag_ratio``
    after the last."""

    default_rate = linear

    def __init__(self, mobject: Mobject, run_time: float | None = None,
                 lag_ratio: float | None = None, **kw: Any) -> None:  # fmt: skip
        length = sum(leaf.glyphs if isinstance(leaf, Text) else 1 for leaf in mobject.leaves())
        if run_time is None:
            run_time = 1 if length < 15 else 2
        if lag_ratio is None:
            lag_ratio = min(4.0 / max(1.0, length), 0.2)
        super().__init__(mobject, run_time=run_time, lag_ratio=lag_ratio, **kw)

    def apply(self, ctx, t0, d):
        # a text writes itself glyph by glyph; several mobjects follow one another
        lag = self.lag if len(self.mob.leaves()) > 1 else 0.0
        _draw_in(ctx, self.mob, t0, d, self.rate_func, lag)


class DrawBorderThenFill(Animation):
    """Manim's defaults: 2 s, ``double_smooth``. The outline is traced in the first half,
    the fill comes in in the second."""

    default_time = 2.0
    default_rate = double_smooth

    def apply(self, ctx, t0, d):
        for leaf in self.mob.leaves():
            if leaf.outline:
                ctx.show(leaf, t0, progress=0.0, fill_opacity=0.0)
                ctx.track(leaf, "progress", 1.0, t0, d / 2, smooth, start=0.0)
                ctx.track(leaf, "fill_opacity", leaf.fill_opacity, t0 + d / 2, d / 2, smooth,
                          start=0.0)  # fmt: skip
            else:
                _fade(ctx, leaf, t0, d, 0.0, 1.0, self.rate_func)


class Uncreate(Animation):
    remover = True

    def apply(self, ctx, t0, d):
        for leaf in self.mob.leaves():
            if leaf.outline:
                ctx.track(leaf, "progress", 0.0, t0, d, self.rate_func, start=1.0)
            elif leaf.kind == "text":
                ctx.track(leaf, "write", 0.0, t0, d, self.rate_func, start=1.0)
            else:
                _fade(ctx, leaf, t0, d, 1.0, 0.0, self.rate_func)
        ctx.gone(self.mob)


class Unwrite(Uncreate):
    default_rate = linear


class GrowFromCenter(Animation):
    def __init__(self, mobject: Mobject, point: Any = None, **kw: Any) -> None:
        super().__init__(mobject, **kw)
        self.point = point

    def apply(self, ctx, t0, d):
        c = self.mob.get_center() if self.point is None else _vec3(self.point)
        start = self.mob.copy()
        start.scale(1e-3, about_point=c)
        for leaf, s in zip(self.mob.leaves(), start.leaves(), strict=True):
            ctx.show(leaf, t0)
            ctx.move(leaf, s, leaf, t0, d, self.rate_func)


class GrowFromPoint(GrowFromCenter):
    def __init__(self, mobject: Mobject, point: Any, **kw: Any) -> None:
        super().__init__(mobject, point=point, **kw)


class GrowFromEdge(GrowFromCenter):
    def __init__(self, mobject: Mobject, edge: Any, **kw: Any) -> None:
        super().__init__(mobject, point=mobject.get_critical_point(edge), **kw)


class GrowArrow(GrowFromCenter):
    def __init__(self, arrow: Mobject, **kw: Any) -> None:
        super().__init__(arrow, point=arrow.get_start(), **kw)


SpinInFromNothing = GrowFromCenter


class ShrinkToCenter(Animation):
    remover = True

    def apply(self, ctx, t0, d):
        goal = self.mob.copy()
        goal.scale(1e-3)
        for leaf, g in zip(self.mob.leaves(), goal.leaves(), strict=True):
            ctx.move(leaf, leaf, g, t0, d, self.rate_func)
        ctx.gone(self.mob)


class _MorphTo(Animation):
    """Move each leaf to its goal's place and look: parameter tracks while it keeps its
    phasor kind; an outline that changes kind morphs (the ``transform`` verb)."""

    def _to_goal(self, ctx: _Context, t0: float, d: float, goal: Mobject) -> None:
        mine, theirs = self.mob.leaves(), goal.leaves()
        if len(mine) != len(theirs):
            raise ManimImportError(f"{type(self).__name__}: the target has {len(theirs)} parts, "
                                   f"the mobject {len(mine)}")  # fmt: skip
        for leaf, g in zip(mine, theirs, strict=True):
            ctx.move(leaf, leaf, g, t0, d, self.rate_func)
            if leaf.alias is None:
                leaf._take(g)


class MoveToTarget(_MorphTo):
    def apply(self, ctx, t0, d):
        if self.mob.target is None:
            raise ManimImportError("MoveToTarget needs mob.generate_target() first")
        self._to_goal(ctx, t0, d, self.mob.target)


class Restore(_MorphTo):
    def apply(self, ctx, t0, d):
        if self.mob.saved is None:
            raise ManimImportError("Restore needs mob.save_state() first")
        self._to_goal(ctx, t0, d, self.mob.saved)


class ApplyMethod(_MorphTo):
    def __init__(self, method: Callable, *args: Any, **kw: Any) -> None:
        mob = getattr(method, "__self__", None)
        if not isinstance(mob, Mobject):
            raise ManimImportError("ApplyMethod needs a mobject's method")
        anim_kw = {k: kw.pop(k) for k in ("run_time", "rate_func") if k in kw}
        super().__init__(mob, **anim_kw)
        self.goal = mob.copy()
        getattr(self.goal, method.__name__)(*args, **kw)

    def apply(self, ctx, t0, d):
        self._to_goal(ctx, t0, d, self.goal)


class _MorphToGoal(_MorphTo):
    def __init__(self, mob: Mobject, goal: Mobject, **kw: Any) -> None:
        super().__init__(mob, **kw)
        self.goal = goal

    def apply(self, ctx, t0, d):
        self._to_goal(ctx, t0, d, self.goal)


class Transform(Animation):
    """``Transform(a, b)``: ``a`` becomes ``b``. A leaf that keeps its phasor kind moves its
    parameters; outlines (and text) of other kinds morph (``transform``;
    ``transform_matching`` for ``TransformMatchingTex`` / ``TransformMatchingShapes``);
    anything else crossfades. Afterwards ``a`` stands for what ``b`` was (as in Manim)."""

    matching = False
    replace = False

    def __init__(self, mobject: Mobject, target_mobject: Mobject, **kw: Any) -> None:
        super().__init__(mobject, **{k: v for k, v in kw.items()
                                     if k in ("run_time", "rate_func")})  # fmt: skip
        self.goal = target_mobject

    def apply(self, ctx, t0, d):
        mine, theirs = self.mob.leaves(), self.goal.leaves()
        same = (not self.matching and len(mine) == len(theirs) and all(
            ctx.same_kind(a, b) for a, b in zip(mine, theirs, strict=True)))  # fmt: skip
        if same:
            for a, b in zip(mine, theirs, strict=True):
                ctx.move(a, a, b, t0, d, self.rate_func)
                a._take(b)
            if self.replace:
                self.goal.alias = self.mob
            return
        ctx.morph(mine, theirs, t0, d, self.rate_func, self.matching)
        ctx.gone(self.mob)
        self.mob.alias = self.goal


class ReplacementTransform(Transform):
    replace = True


class TransformMatchingTex(ReplacementTransform):
    matching = True
    default_rate = linear  # Manim's TransformMatching* play linearly


class TransformMatchingShapes(TransformMatchingTex):
    pass


class FadeTransform(ReplacementTransform):
    def apply(self, ctx, t0, d):
        for leaf in self.mob.leaves():
            _fade(ctx, leaf, t0, d, 1.0, 0.0, self.rate_func)
        for leaf in self.goal.leaves():
            _fade(ctx, leaf, t0, d, 0.0, 1.0, self.rate_func)
        ctx.gone(self.mob)
        self.mob.alias = self.goal


ClockwiseTransform = CounterclockwiseTransform = Transform
TransformFromCopy = ReplacementTransform


class Rotate(Animation):
    def __init__(self, mobject: Mobject, angle: float = PI, axis: Any = OUT,
                 about_point: Any = None, **kw: Any) -> None:  # fmt: skip
        super().__init__(mobject, **kw)
        self.angle, self.axis, self.about = _num(angle, "angle"), axis, about_point

    def apply(self, ctx, t0, d):
        about = self.mob.get_center() if self.about is None else _vec3(self.about)
        leaves = self.mob.leaves()
        start = [leaf.copy() for leaf in leaves]

        def at(u: float) -> list[Mobject]:
            out = []
            for s in start:
                g = s.copy()
                g.rotate(self.angle * u, self.axis, about_point=about)
                out.append(g)
            return out

        ctx.sweep(leaves, at, t0, d, self.rate_func)


class Rotating(Rotate):
    default_rate = linear
    default_time = 5.0

    def __init__(self, mobject: Mobject, angle: float = TAU, **kw: Any) -> None:
        kw.pop("radians", None)
        super().__init__(mobject, angle=angle, **kw)


class Indicate(Animation):
    """Manim: scale by 1.2 and turn yellow, there and back."""

    default_rate = there_and_back

    def __init__(self, mobject: Mobject, scale_factor: float = 1.2, color: Any = _YELLOW,
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(mobject, **kw)
        self.factor, self.color = _num(scale_factor, "scale_factor"), color

    def apply(self, ctx, t0, d):
        goal = self.mob.copy()
        goal.scale(self.factor)
        goal.set_color(self.color)
        for leaf, g in zip(self.mob.leaves(), goal.leaves(), strict=True):
            ctx.move(leaf, leaf, g, t0, d, self.rate_func)


class _VerbAnimation(Animation):
    """An emphasis phasor has its own verb for: played with Manim's run time."""

    verb = ""

    def apply(self, ctx, t0, d):
        leaves = [x for x in self.mob.leaves() if x.kind != "value"]
        for leaf in leaves[:1] if self.verb == "circumscribe" else leaves:
            ctx.realize(leaf)
            ctx.verb(self.verb, t0, d, leaf.obj_id)


class Circumscribe(_VerbAnimation):
    verb = "circumscribe"


class Flash(_VerbAnimation):
    verb = "flash"

    def __init__(self, point: Any, **kw: Any) -> None:
        if not isinstance(point, Mobject):
            raise ManimImportError("Flash at a point: phasor flashes objects (Flash(mobject))")
        super().__init__(point, **{k: v for k, v in kw.items() if k in ("run_time", "rate_func")})


FocusOn = Flash


class Wiggle(_VerbAnimation):
    verb = "wiggle"
    default_time = 2.0


ApplyWave = Wiggle


class Wait(Animation):
    default_rate = linear

    def __init__(self, run_time: float = 1.0, **kw: Any) -> None:
        super().__init__(None, run_time=run_time, **kw)

    def apply(self, ctx, t0, d):
        return None


class AnimationGroup(Animation):
    """Manim: each animation starts ``lag_ratio`` of the one before's run time after it; a
    group ``run_time`` stretches the whole schedule to fit."""

    default_rate = linear

    def __init__(self, *animations: Any, lag_ratio: float = 0.0, run_time: float | None = None,
                 rate_func: Any = None, **kw: Any) -> None:  # fmt: skip
        super().__init__(None, run_time=run_time, rate_func=rate_func)
        self.anims = [_as_animation(a) for a in animations]
        self.lag = _num(lag_ratio, "lag_ratio")

    def mobjects(self) -> list[Mobject]:
        return [m for a in self.anims for m in a.mobjects()]

    def _schedule(self) -> list[tuple[float, float]]:
        out, t = [], 0.0
        for a in self.anims:
            out.append((t, a.duration))
            t += a.duration * self.lag
        return out

    @property
    def natural(self) -> float:
        return max((s + d for s, d in self._schedule()), default=0.0)

    @property
    def duration(self) -> float:
        return self.run_time if self.run_time is not None else self.natural

    def flat(self, t0: float, d: float) -> list[tuple[Animation, float, float]]:
        """Each animation inside with its start and run time (the schedule fit to ``d``)."""
        k = d / self.natural if self.natural > 0 else 0.0
        out: list[tuple[Animation, float, float]] = []
        for (s, dur), a in zip(self._schedule(), self.anims, strict=True):
            at, run = t0 + s * k, dur * k
            out += a.flat(at, run) if isinstance(a, AnimationGroup) else [(a, at, run)]
        return out

    def apply(self, ctx, t0, d):
        for a, at, run in self.flat(t0, d):
            a.apply(ctx, at, run)


class LaggedStart(AnimationGroup):
    def __init__(self, *animations: Any, lag_ratio: float = 0.05, **kw: Any) -> None:
        super().__init__(*animations, lag_ratio=lag_ratio, **kw)


class Succession(AnimationGroup):
    def __init__(self, *animations: Any, lag_ratio: float = 1.0, **kw: Any) -> None:
        super().__init__(*animations, lag_ratio=lag_ratio, **kw)


class LaggedStartMap(LaggedStart):
    def __init__(self, AnimationClass: Any, mobject: Mobject, arg_creator: Any = None,
                 run_time: float = 2.0, **kw: Any) -> None:  # fmt: skip
        super().__init__(*[AnimationClass(m) for m in mobject], run_time=run_time, **kw)


class _AnimateBuilder:
    """``mob.animate.shift(RIGHT).set_color(RED)``: the target is what those calls do to a
    copy; playing it moves the mobject there."""

    def __init__(self, mob: Mobject) -> None:
        self.mob = mob
        self.calls: list[tuple[str, tuple, dict]] = []
        self.anim_kw: dict[str, Any] = {}

    def __call__(self, **kw: Any) -> _AnimateBuilder:  # mob.animate(run_time=2).shift(...)
        self.anim_kw.update({k: v for k, v in kw.items() if k in ("run_time", "rate_func")})
        return self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*a: Any, **kw: Any) -> _AnimateBuilder:
            self.calls.append((name, a, kw))
            return self

        return call

    def build(self) -> Animation:
        goal = self.mob.copy()
        for name, a, kw in self.calls:
            getattr(goal, name)(*a, **kw)
        return _MorphToGoal(self.mob, goal, **self.anim_kw)


def _as_animation(a: Any) -> Animation:
    if isinstance(a, _AnimateBuilder):
        return a.build()
    if isinstance(a, Animation):
        return a
    if isinstance(a, Mobject):
        raise ManimImportError("play() got a mobject: wrap it in an animation (Create, "
                               "FadeIn ...)")  # fmt: skip
    raise ManimImportError(f"play() got {type(a).__name__}, which is not an animation")


# ---------------------------------------------------------------- the scene


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, str | bool) or isinstance(b, str | bool) or a is None or b is None:
        return a == b
    try:
        x, y = np.asarray(a, float), np.asarray(b, float)
    except (TypeError, ValueError):
        return a == b
    return x.shape == y.shape and bool(np.all(np.abs(x - y) <= tol))


def _same_shape(a: Any, b: Any) -> bool:
    if isinstance(a, str | bool) or isinstance(b, str | bool) or a is None or b is None:
        return False
    try:
        return np.asarray(a, float).shape == np.asarray(b, float).shape
    except (TypeError, ValueError):
        return False


def _nested(v: np.ndarray) -> Any:
    if v.ndim == 0:
        return float(v)
    if v.ndim == 1:
        return tuple(round(float(x), 9) for x in v)
    return [_nested(x) for x in v]


def _lerp(a: Any, b: Any, u: float) -> Any:
    if not _same_shape(a, b):
        return b if u >= 1 else a
    x, y = np.asarray(a, float), np.asarray(b, float)
    return _nested(x + (y - x) * u)


def _simplify(samples: list[tuple[float, Any]], tol: float) -> list[tuple[float, Any]]:
    """The keys a straight line between kept keys does not reproduce within ``tol``."""
    if len(samples) <= 2:
        return samples
    kept = [samples[0]]
    i, j = 0, 2
    while j < len(samples):
        t0, v0 = samples[i]
        t1, v1 = samples[j]
        ok = all(_close(_lerp(v0, v1, (samples[k][0] - t0) / (t1 - t0) if t1 > t0 else 1.0),
                        samples[k][1], tol) for k in range(i + 1, j))  # fmt: skip
        if ok:
            j += 1
            continue
        kept.append(samples[j - 1])
        i, j = j - 1, j + 1
    kept.append(samples[-1])
    return kept


def _seg(path: str, t0: float, t1: float, to: Any, ease: str) -> dict:
    return {"op": "add_segment", "path": path,
            "segment": {"t0": t0, "t1": t1, "to": to, "ease": ease}}  # fmt: skip


@dataclass
class _Context:
    """The phasor scene being built and the Manim clock."""

    scene: PhasorScene
    base_dir: str = ""
    warnings: list[str] = field(default_factory=list)
    t: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    shown: set[int] = field(default_factory=set)  # python ids of leaves currently visible
    trace: list[dict] = field(default_factory=list)
    values: dict[tuple[str, str], Any] = field(default_factory=dict)  # last value per path
    busy: dict[str, float] = field(default_factory=dict)  # path -> end of its last segment
    ops: list[dict] = field(default_factory=list)  # pending, applied together
    in_frames: bool = False
    frames: int = 0
    rate: float = UPDATE_RATE
    sounds: int = 0

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ident(self, leaf: Mobject) -> str:
        base = name_from(leaf.id_base or type(leaf).__name__, "mobject", 30)
        doc = self.scene.engine.document
        taken = {o.id for o in doc.objects} | set(doc.retired_ids)
        while True:
            self.counts[base] = self.counts.get(base, 0) + 1
            ident = f"{base}_{self.counts[base]}"
            if ident not in taken:
                return ident

    def realize(self, leaf: Mobject) -> str:
        """Make sure the leaf exists in the phasor scene; returns its object id. Objects start
        invisible (opacity 0): ``show`` switches them on at the time Manim first shows them."""
        if leaf.obj_id is not None:
            return leaf.obj_id
        if leaf.kind == "frame":
            leaf.obj_id, leaf.ptype = "camera", "frame"
            for k, v in leaf.phasor_spec()[1].items():
                self.values[("camera", k)] = v
            return "camera"
        self.flush()
        ident = self.ident(leaf)
        try:
            obj = leaf.phasor_object(ident)
            self.scene.add(obj)
        except (ValidationError, DocumentError) as e:
            obj = self._fallback(leaf, ident, e)
        leaf.obj_id, leaf.ptype = ident, obj.type
        for k, v in self.scene.engine.document.object(ident).params.items():
            self.values[(ident, k)] = v
        return ident

    def _fallback(self, leaf: Mobject, ident: str, error: ValidationError | DocumentError):
        """A shape phasor's own kind cannot hold (a zero radius, a coordinate past the
        bounds): a path through its outline, else a line, with a warning."""
        what = f"{ident} ({type(leaf).__name__})"
        why = _validation_message(error) if isinstance(error, ValidationError) else str(error)
        if not (leaf.outline and len(leaf.pts) >= 2):
            raise ManimImportError(f"{what}: {why}") from None
        from phasor import Line as PLine

        closed = leaf.closed

        def as_line() -> tuple[Any, dict[str, Any]]:
            pts = np.vstack([leaf.pts, leaf.pts[:1]]) if closed else leaf.pts
            return PLine, {"points": _tuples(pts)}

        for make in (lambda: _path_spec(leaf.pts, closed), as_line):
            cls, params = make()
            style = {k: v for k, v in leaf._style().items()
                     if cls is not PLine or k in ("color", "stroke")}  # fmt: skip
            look = leaf._look() if cls is not PLine else {}
            try:
                obj = cls(id=ident, opacity=0.0, style=style, **params, **look)
                self.scene.add(obj)
            except (ValidationError, DocumentError):
                continue
            self.warn(f"{what} cannot be drawn as its own kind ({why[:120]}): drawn as a "
                      f"{obj.type}")  # fmt: skip
            leaf.similar = False
            leaf._spec = make  # type: ignore[method-assign]
            if cls is PLine:
                leaf._look = dict  # type: ignore[method-assign]
                leaf._style = lambda _s=style, **kw: dict(_s)  # type: ignore
            return obj
        raise ManimImportError(f"{what}: {why}") from None

    def same_kind(self, a: Mobject, b: Mobject) -> bool:
        """Whether ``a`` can become ``b`` by moving its parameters (the same phasor type,
        and for text the same text)."""
        if a.alias is not None or b.alias is not None:
            return False
        ta = a.ptype or a.phasor_spec()[0].type
        cls_b, pb = b.phasor_spec()
        if ta != cls_b.type:
            return False
        if a.kind in ("text", "counter") and ta != "counter":
            return a.phasor_spec()[1].get("text", None) == pb.get("text") and getattr(
                a, "parts", None) == getattr(b, "parts", None)  # fmt: skip
        return True

    def show(self, leaf: Mobject, t: float, **start: Any) -> None:
        """Make the leaf visible from time ``t`` (with starting values for animated params)."""
        self.realize(leaf)
        if leaf.kind == "frame" or (id(leaf) in self.shown and not start):
            return
        first = id(leaf) not in self.shown
        for k, v in start.items():
            self.instant(leaf, k, v, t)
        if first and "opacity" not in start and leaf.kind != "value":
            self.instant(leaf, "opacity", leaf.opacity or 1.0, t)
        self.shown.add(id(leaf))

    def instant(self, leaf: Mobject, param: str, value: Any, t: float | None = None) -> None:
        self.realize(leaf)
        self._segments(f"{leaf.obj_id}.{param}", None, value, self.t if t is None else t, 0.0,
                       None)  # fmt: skip

    def track(self, leaf: Mobject, param: str, value: Any, t0: float, d: float, rate: Any,
              start: Any = None) -> None:  # fmt: skip
        self.realize(leaf)
        path = f"{leaf.obj_id}.{param}"
        if start is not None:
            self._segments(path, None, start, t0, 0.0, None)
        self._segments(path, start, value, t0, d, rate)

    def _key(self, path: str) -> tuple[str, str]:
        if ".style." in path:
            return (path, "")
        head, _, tail = path.rpartition(".")
        return (head, tail)

    def _segments(self, path: str, start: Any, value: Any, t0: float, d: float,
                  rate: Any) -> None:  # fmt: skip
        """Segments that take ``path`` to ``value`` over [t0, t0 + d] by the rate function:
        one segment when phasor has its easing, else ``RATE_PIECES`` straight pieces (a rate
        function that does not end at 1, ``there_and_back``, comes back)."""
        key = self._key(path)
        if start is None:
            start = self.values.get(key)
        t0, d = round(t0, 9), max(round(d, 9), 0.0)
        if t0 < self.busy.get(path, -1.0) - 1e-9:  # never overlap what the path already has
            d = max(0.0, round(t0 + d - self.busy[path], 9))
            t0 = self.busy[path]
        end = value
        if d == 0 or rate is None:
            self.ops.append(_seg(path, round(t0 + d, 9), round(t0 + d, 9), value, "linear"))
        elif rate in _EASE:
            self.ops.append(_seg(path, t0, round(t0 + d, 9), value, _EASE[rate]))
        elif _same_shape(start, value):
            for k in range(1, RATE_PIECES + 1):
                u = k / RATE_PIECES
                v = _lerp(start, value, _rate(rate, u))
                a, b = round(t0 + d * (k - 1) / RATE_PIECES, 9), round(t0 + d * u, 9)
                self.ops.append(_seg(path, a, b, v, "linear"))
                end = v
        elif start is not None and _rate(rate, 1.0) < 0.5:  # a color there and back
            mid = round(t0 + d / 2, 9)
            self.ops.append(_seg(path, t0, mid, value, "sigmoid"))
            self.ops.append(_seg(path, mid, round(t0 + d, 9), start, "sigmoid"))
            end = start
        else:
            self.ops.append(_seg(path, t0, round(t0 + d, 9), value, "sigmoid"))
        self.values[key] = end
        self.busy[path] = max(self.busy.get(path, 0.0), round(t0 + d, 9))

    def flush(self) -> None:
        if self.ops:
            ops, self.ops = self.ops, []
            self.scene._apply(ops)

    def restyle(self, leaf: Mobject, start: Mobject, goal: Mobject, t0: float, d: float,
                rate: Any) -> None:  # fmt: skip
        """Style keys that change from ``start`` to ``goal``: style tracks (colors mix in
        linear light; a size jumps at the end)."""
        old, new = start._style(**start._style_extra()), goal._style(**goal._style_extra())
        for key in sorted(new):
            if old.get(key) == new[key] or key in ("fill", "dash"):
                continue
            path = f"{leaf.obj_id}.style.{key}"
            if self.values.get((path, "")) is None:
                self.values[(path, "")] = old.get(key)
            colored = "color" in key and old.get(key) is not None
            self._segments(path, old.get(key), new[key], t0, d, rate if colored else None)

    def move(self, leaf: Mobject, start: Mobject, goal: Mobject, t0: float, d: float,
             rate: Any) -> None:  # fmt: skip
        """``leaf`` goes from ``start``'s parameters and look to ``goal``'s."""
        if leaf.kind == "group":
            return
        oid = self.realize(leaf)
        _, ps = start.phasor_spec()
        cls_g, pg = goal.phasor_spec()
        if cls_g.type != leaf.ptype:  # an outline stretched out of its kind
            twin = goal.copy()
            self.morph([leaf], [twin], t0, d, rate, False)
            leaf.alias = twin
            return
        self.show(leaf, t0)
        for k in sorted(pg):
            if k == "as_shapes" or (k == "text" and pg[k] != ps.get(k)):
                continue
            a = ps.get(k, self.values.get((oid, k)))
            now = self.values.get((oid, k))
            if _close(a, pg[k]) and _close(now, pg[k]):
                continue
            if not _close(a, now):
                self._segments(f"{oid}.{k}", None, a, t0, 0.0, None)
            self._segments(f"{oid}.{k}", a, pg[k], t0, d, rate)
        if leaf.kind != "frame":
            if goal.opacity != start.opacity:
                self._segments(f"{oid}.opacity", start.opacity, goal.opacity, t0, d, rate)
            self.restyle(leaf, start, goal, t0, d, rate)

    def sweep(self, leaves: list[Mobject], at: Callable[[float], list[Mobject]], t0: float,
              d: float, rate: Any) -> None:  # fmt: skip
        """Leaves along a path that is not a straight line in their parameters (a rotation):
        keyed ``RATE_PIECES`` times."""
        prev = [leaf.copy() for leaf in leaves]
        for k in range(1, RATE_PIECES + 1):
            u0, u1 = (k - 1) / RATE_PIECES, k / RATE_PIECES
            goals = at(_rate(rate, u1))
            for leaf, p, g in zip(leaves, prev, goals, strict=True):
                self.move(leaf, p, g, t0 + d * u0, d / RATE_PIECES, linear)
            prev = goals
        for leaf, g in zip(leaves, prev, strict=True):
            if leaf.alias is None:
                leaf._take(g)

    def morph(self, mine: list[Mobject], theirs: list[Mobject], t0: float, d: float, rate: Any,
              matching: bool) -> None:  # fmt: skip
        """``mine`` becomes ``theirs`` (new objects): the ``transform`` verb (or
        ``transform_matching``), played in Manim's time, for each pair with outlines; a
        crossfade for the rest."""
        pairs = list(zip(mine, theirs, strict=True)) if len(mine) == len(theirs) else []
        done_a, done_b = set(), set()
        for a, b in pairs:
            if self._morph_pair(a, b, t0, d, rate, matching):
                done_a.add(id(a))
                done_b.add(id(b))
        for leaf in mine:
            if id(leaf) not in done_a and leaf.kind != "value":
                _fade(self, leaf, t0, d, 1.0, 0.0, rate)
        for leaf in theirs:
            if id(leaf) not in done_b and leaf.kind != "value":
                _fade(self, leaf, t0, d, 0.0, 1.0, rate)

    def _morph_pair(self, a: Mobject, b: Mobject, t0: float, d: float, rate: Any,
                    matching: bool) -> bool:  # fmt: skip
        shaped = ("text",)
        if not ((a.outline or a.kind in shaped) and (b.outline or b.kind in shaped)):
            return False
        if a.kind == "counter" or b.kind == "counter":
            return False
        for leaf in (a, b):
            if leaf.kind == "text" and not leaf.as_shapes:
                leaf.as_shapes = True
                if leaf.obj_id:
                    self.flush()
                    self.scene._apply([{"op": "set_param", "path": f"{leaf.obj_id}.as_shapes",
                                        "value": True}])  # fmt: skip
        self.realize(a)
        self.realize(b)
        try:
            self.verb("transform_matching" if matching else "transform", t0, d, a.obj_id,
                      b.obj_id, rate=rate)  # fmt: skip
        except VerbError as e:
            self.warn(f"{type(a).__name__} into {type(b).__name__} crossfades ({e})")
            return False
        self.shown.discard(id(a))
        self.shown.add(id(b))
        return True

    def verb(self, name: str, t0: float, d: float, *ids: str, rate: Any = None) -> None:
        """Play phasor's verb ``name`` at ``t0``, stretched to Manim's ``d`` seconds."""
        self.flush()
        motion = self.scene.at(round(t0, 9)).motion(name, *ids)
        k = d / motion.end if motion.end > 0 else 1.0
        ease = _EASE.get(rate) if rate is not None else None
        ops = []
        for op in motion.ops:
            seg = op.get("segment")
            if seg is not None and isinstance(seg.get("t0"), int | float):
                seg = dict(seg)
                a, b = float(seg["t0"]), float(seg["t1"])
                seg["t0"], seg["t1"] = round(t0 + (a - t0) * k, 9), round(t0 + (b - t0) * k, 9)
                if (
                    ease is not None
                    and b > a
                    and seg.get("verb") in ("transform", "transform_matching")
                ):
                    seg["ease"] = ease
                op = {**op, "segment": seg}
                self.busy[op["path"]] = max(self.busy.get(op["path"], 0.0), seg["t1"])
                self.values[self._key(op["path"])] = seg["to"]
            ops.append(op)
        self.scene._apply(ops)

    def gone(self, mob: Mobject) -> None:
        for leaf in mob.leaves():
            self.shown.discard(id(leaf))


_CTX: _Context | None = None


class _Frame(Mobject):
    """A ``MovingCameraScene``'s ``self.camera.frame``: moving it moves the camera
    (``camera.center``, ``camera.height``; its width is the height x 16/9)."""

    kind = "frame"

    def __init__(self) -> None:
        super().__init__()
        w, h = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
        self.pts = np.array([(-w, -h, 0.0), (w, h, 0.0)])

    def phasor_spec(self):
        lo, hi = self.pts.min(axis=0), self.pts.max(axis=0)
        c = (lo + hi) / 2
        center = (round(float(c[0]), 9), round(float(c[1]), 9), 0.0)
        return types.SimpleNamespace(type="frame"), {
            "center": center,
            "height": round(float(hi[1] - lo[1]), 9),
        }

    def rescale_to_fit(self, length: float, dim: int, stretch: bool = False, **kw: Any) -> Mobject:
        # the frame keeps the picture's shape: a new width scales it as a whole
        old = self.length_over_dim(dim)
        return self.scale(length / old, **kw) if old and length > 0 else self

    def _after(self, kind: str, value: float) -> None:
        if kind == "rotate" and _CTX is not None:
            _CTX.warn("a turned camera frame is not carried over (the camera stays level)")


class _Camera:
    def __init__(self, moving: bool) -> None:
        self.background_color = "#000000"
        self.frame = _Frame() if moving else None

    def __getattribute__(self, name: str) -> Any:
        if name == "frame" and object.__getattribute__(self, "frame") is None:
            raise ManimImportError("self.camera.frame needs a MovingCameraScene")
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


class Scene:
    """Manim's ``Scene``: ``construct`` runs on ``render`` (here, on import)."""

    moving_camera = False

    def __init__(self, **kw: Any) -> None:
        self.camera = _Camera(self.moving_camera)
        self.mobjects: list[Mobject] = []
        self.renderer = None

    # -- the clock and the scene
    @property
    def _c(self) -> _Context:
        assert _CTX is not None
        return _CTX

    @property
    def time(self) -> float:
        return self._c.t

    def add(self, *mobjects: Mobject) -> Scene:
        for m in mobjects:
            if not isinstance(m, Mobject) or m.kind == "frame":
                continue
            for leaf in m.leaves():
                self._c.show(leaf, self._c.t)
            if m not in self.mobjects:
                self.mobjects.append(m)
        self._c.flush()
        return self

    add_foreground_mobject = add_foreground_mobjects = add

    def remove(self, *mobjects: Mobject) -> Scene:
        for m in mobjects:
            if not isinstance(m, Mobject):
                continue
            for leaf in m.leaves():
                if id(leaf) in self._c.shown and leaf.obj_id and leaf.kind != "value":
                    self._c.instant(leaf, "opacity", 0.0)
            self._c.gone(m)
            if m in self.mobjects:
                self.mobjects.remove(m)
        self._c.flush()
        return self

    def clear(self) -> Scene:
        return self.remove(*list(self.mobjects))

    def bring_to_front(self, *mobjects: Mobject) -> Scene:
        return self

    bring_to_back = bring_to_front

    def _family(self) -> list[Mobject]:
        seen: list[Mobject] = []
        ids: set[int] = set()
        for m in self.mobjects:
            for x in m.get_family():
                if id(x) not in ids:
                    ids.add(id(x))
                    seen.append(x)
        return seen

    def _updating(self) -> list[Mobject]:
        mobs = [m for m in self._family() if m.updaters and not m.suspended]
        frame = self.camera.frame if self.moving_camera else None
        if frame is not None and frame.updaters and not frame.suspended:
            mobs.append(frame)
        return mobs

    def play(self, *args: Any, run_time: float | None = None, rate_func: Any = None,
             lag_ratio: float | None = None, **kw: Any) -> None:  # fmt: skip
        c = self._c
        built = [_as_animation(a) for a in args]
        if not built:
            raise ManimImportError("play() needs at least one animation")
        for a in built:
            if run_time is not None:
                a.run_time = _num(run_time, "run_time")
            if rate_func is not None:
                a.rate_func = rate_func
        d = max((a.duration for a in built), default=0.0)
        t0 = c.t
        flat = [x for a in built for x in (a.flat(t0, a.duration) if isinstance(
            a, AnimationGroup) else [(a, t0, a.duration)])]  # fmt: skip
        before = {id(leaf): leaf.copy() for a, _, _ in flat for m in a.mobjects()
                  for leaf in m.leaves()}  # fmt: skip
        trackers = {id(m): m.value for a, _, _ in flat for m in a.mobjects()
                    for m in m.get_family() if m.kind == "value"}  # fmt: skip
        family = {id(x) for x in self._family()}
        for a, _, _ in flat:  # Manim adds an animation's mobject before it plays
            for m in a.mobjects():
                if m.kind != "frame" and id(m) not in family and m not in self.mobjects:
                    self.mobjects.append(m)
                    family |= {id(x) for x in m.get_family()}
        for a in built:
            a.apply(c, t0, a.duration)
        c.trace.append({"t0": t0, "duration": d, "anims": [
            {"cls": type(a).__name__, "run_time": a.duration,
             "rate_func": _rate_name(a.rate_func)} for a in built], "mobjects": self._listing(),
            "frame": self.moving_camera})  # fmt: skip
        for a, _, _ in flat:  # removers leave at the end
            if a.remover and a.mob in self.mobjects:
                self.mobjects.remove(a.mob)
            if isinstance(a, ReplacementTransform) and a.mob in self.mobjects:
                i = self.mobjects.index(a.mob)
                if a.goal in self.mobjects:
                    self.mobjects.pop(i)
                else:
                    self.mobjects[i] = a.goal
        c.trace[-1]["after"] = self._listing()
        self._frames(t0, d, flat, before, trackers)
        c.flush()
        c.t = t0 + d

    def _listing(self) -> list[dict]:
        out = []
        for m in self.mobjects:
            ids = [leaf.obj_id for leaf in m.leaves() if leaf.obj_id and leaf.kind != "value"]
            if ids:
                out.append({"cls": m.name, "ids": ids})
        return out

    def _frames(self, t0: float, d: float, flat: list[tuple[Animation, float, float]],
                before: dict[int, Mobject], tracker_start: dict[int, float]) -> None:  # fmt: skip
        """Run the updaters frame by frame through [t0, t0 + d], with every mobject an
        animation moves where Manim has it at that frame, and key what they change."""
        c = self._c
        if not self._updating():
            return
        c.flush()
        moving: list[tuple[Mobject, Mobject, Mobject, float, float, Any]] = []
        seen: set[int] = set()
        trackers: list[tuple[Mobject, float, float, float, float, Any]] = []
        for anim, at, dur in flat:
            for m in anim.mobjects():
                for x in m.get_family():
                    if x.kind == "value" and id(x) in tracker_start and id(x) not in seen:
                        seen.add(id(x))
                        trackers.append((x, tracker_start[id(x)], x.value, at, dur,
                                         anim.rate_func))  # fmt: skip
                for leaf in m.leaves():
                    if id(leaf) in before and id(leaf) not in seen and leaf.kind != "value":
                        seen.add(id(leaf))
                        moving.append((leaf, before[id(leaf)], leaf.copy(), at, dur,
                                       anim.rate_func))  # fmt: skip
        # Manim's frames: t0, t0 + 1/rate, ... before t0 + d (the first with dt 0), then the
        # end, with every animation finished and the updaters run once more with dt 0
        n = math.ceil(d * c.rate - 1e-9) if d > 0 else 0
        if c.frames + n + 1 > MAX_FRAMES and c.rate > 1:
            c.rate = max(1.0, c.rate / 2)
            c.warn(f"updaters run at {c.rate:g} frames a second past {MAX_FRAMES} frames")
            n = math.ceil(d * c.rate - 1e-9) if d > 0 else 0
        c.frames += n + 1
        samples: dict[tuple[int, str], list[tuple[float, Any]]] = {}
        keyed: dict[int, Mobject] = {}
        c.in_frames = True
        try:
            for k in range(n + 1):
                t = round(t0 + (k / c.rate if k < n else d), 9)
                for leaf, s, g, at, dur, rate in moving:
                    u = 1.0 if dur <= 0 else _rate(rate, (t - at) / dur)
                    _interpolate(leaf, s, g, u)
                for m, v0, v1, at, dur, rate in trackers:
                    u = 1.0 if dur <= 0 else _rate(rate, (t - at) / dur)
                    m.value = v0 + (v1 - v0) * u
                dt = 0.0 if k in (0, n) else 1.0 / c.rate
                for m in self._updating():
                    for fn in list(m.updaters):
                        _call_updater(m, fn, dt)
                    for leaf in m.leaves():
                        keyed[id(leaf)] = leaf
                for leaf in keyed.values():
                    if leaf.kind in ("group", "value"):
                        continue
                    for p, v in leaf.phasor_spec()[1].items():
                        samples.setdefault((id(leaf), p), []).append((t, v))
        finally:
            c.in_frames = False
        for leaf, _s, g, _a, _d, _r in moving:
            if id(leaf) not in keyed and leaf.alias is None:
                leaf._take(g)
        for m, _v0, v1, *_ in trackers:
            m.value = v1
        warned: set[int] = set()
        for (lid, p), series in samples.items():
            leaf = keyed[lid]
            if leaf.kind != "frame" and (leaf.obj_id is None or id(leaf) not in self._shown_ids()):
                continue
            oid = c.realize(leaf)
            if leaf.kind != "frame" and leaf.phasor_spec()[0].type != leaf.ptype:
                if lid not in warned:
                    warned.add(lid)
                    c.warn(f"{oid}: an updater changed it into another kind of shape; the "
                           "change is not carried over")  # fmt: skip
                continue
            if p in ("text", "as_shapes", "n", "parts"):
                continue
            path = f"{oid}.{p}"
            busy = c.busy.get(path, -1.0)
            if busy > t0 + 1e-9:  # an animation of this play owns it: it wins
                continue
            series = [(t, v) for t, v in series if t > busy + 1e-9 or busy < t0 - 1e-9]
            if not series or all(_close(v, c.values.get((oid, p)), KEY_TOLERANCE)
                                 for _, v in series):  # fmt: skip
                continue
            keys = _simplify(series, KEY_TOLERANCE)
            prev = None
            for t, v in keys:
                c.ops.append(_seg(path, round(prev if prev is not None else t, 9), round(t, 9),
                                  v, "linear"))  # fmt: skip
                prev = t
            c.values[(oid, p)] = keys[-1][1]
            c.busy[path] = max(c.busy.get(path, 0.0), round(keys[-1][0], 9))
        c.flush()

    def _shown_ids(self) -> set[int]:
        return {id(leaf) for m in self.mobjects for leaf in m.leaves()} | self._c.shown

    def wait(self, duration: float = 1.0, stop_condition: Any = None, frozen_frame: Any = None,
             **kw: Any) -> None:  # fmt: skip
        self.play(Wait(run_time=_num(duration, "wait duration")))

    def pause(self, duration: float = 1.0) -> None:
        self.wait(duration)

    def wait_until(self, stop_condition: Callable, max_time: float = 60.0) -> None:
        self.wait(max_time)

    def next_section(self, *a: Any, **k: Any) -> None:
        return None

    def add_sound(self, sound_file: Any, time_offset: float = 0.0, gain: float | None = None,
                  **kw: Any) -> None:  # fmt: skip
        """A ``media.Sound`` clip (a WAV file in the scene's folder), mixed in at render."""
        from phasor.domains import media

        c = self._c
        rel = _scene_file(sound_file, "add_sound")
        if rel is None:
            return
        start = c.t + _num(time_offset, "time_offset")
        params: dict[str, Any] = {"path": rel, "start": min(max(round(start, 6), 0.0),
                                                            media.MAX_START)}  # fmt: skip
        if gain is not None:
            params["gain"] = min(max(_num(gain, "gain"), media.MIN_GAIN), media.MAX_GAIN)
        c.sounds += 1
        doc = c.scene.engine.document
        taken = {o.id for o in doc.objects} | set(doc.retired_ids)
        ident = next(f"sound_{k}" for k in range(c.sounds, c.sounds + len(taken) + 2)
                     if f"sound_{k}" not in taken)  # fmt: skip
        c.flush()
        try:
            c.scene.add(media.Sound(id=ident, **params))
        except ValidationError as e:
            c.warn(f"add_sound({sound_file!r}) is left out: {_validation_message(e)}")

    def add_subcaption(self, *a: Any, **k: Any) -> None:
        return None

    def setup(self) -> None:
        return None

    def construct(self) -> None:  # pragma: no cover
        pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise ManimImportError(f"Scene.{name} is not supported by phasor's Manim import yet")


def _interpolate(leaf: Mobject, s: Mobject, g: Mobject, u: float) -> None:
    """Where Manim has ``leaf`` at ``u`` of its way from ``s`` to ``g`` (points mix)."""
    if s.pts.shape == g.pts.shape:
        leaf.pts = s.pts + (g.pts - s.pts) * u
    leaf.origin = s.origin + (g.origin - s.origin) * u
    leaf.s = s.s + (g.s - s.s) * u
    leaf.theta = s.theta + (g.theta - s.theta) * u
    leaf.opacity = s.opacity + (g.opacity - s.opacity) * u
    if leaf.kind == "counter":
        leaf.value = s.value + (g.value - s.value) * u


class MovingCameraScene(Scene):
    moving_camera = True


class ThreeDScene(Scene):
    """``set_camera_orientation(phi, theta)`` and ambient rotation become the orbit camera."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._rotation_rate = 0.0
        self._rotation_from: float | None = None

    def set_camera_orientation(self, phi: float | None = None, theta: float | None = None,
                               gamma: float | None = None, zoom: float | None = None,
                               **kw: Any) -> None:  # fmt: skip
        params: dict[str, Any] = {"distance": 18.0, "target": (0.0, 0.0, 0.0), "drift": 0.0}
        params["elevation"] = 90.0 - math.degrees(phi if phi is not None else 70 * DEGREES)
        params["azimuth"] = math.degrees(theta if theta is not None else -90 * DEGREES) + 90.0
        self._c.flush()
        self._c.scene.use_camera("orbit", **params)

    def begin_ambient_camera_rotation(self, rate: float = 0.02, **kw: Any) -> None:
        self._rotation_rate, self._rotation_from = _num(rate, "rate"), self._c.t

    def stop_ambient_camera_rotation(self, **kw: Any) -> None:
        self._finish_rotation()

    def _finish_rotation(self) -> None:
        if self._rotation_from is None:
            return
        span = self._c.t - self._rotation_from
        if span > 0:
            deg = math.degrees(self._rotation_rate) * span  # Manim: radians per second
            self._c.flush()
            doc = self._c.scene.engine.document
            cam = doc.camera(doc.timeline.camera)
            az = float(cam.params.get("azimuth", 0.0)) if cam else 0.0
            self._c.scene.at(self._rotation_from).to("camera.azimuth", az + deg, duration=span,
                                                     ease="linear")  # fmt: skip
        self._rotation_from = None


ThreeDAxes = Axes


# ---------------------------------------------------------------- the stand-in module

_NOT_MANIM = ("ManimImport", "ManimImportError", "Path", "PhasorScene", "Document", "VerbError",
              "ValidationError", "DocumentError", "Any", "Callable", "Sequence", "TEXT_ROLES",
              "FONT_UNIT", "STROKE_UNIT", "UPDATE_RATE", "MAX_FRAMES", "KEY_TOLERANCE",
              "RATE_PIECES", "SAMPLES", "MAX_SIZE")  # fmt: skip


def _module() -> types.ModuleType:
    mod = types.ModuleType("manim")
    names = {k: v for k, v in globals().items() if not k.startswith("_") and k not in _NOT_MANIM}
    names.update(dict(_COLORS))
    names["np"] = np  # Manim's star-import gives scripts numpy as ``np``
    names["config"] = types.SimpleNamespace(
        frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH, pixel_height=1080, pixel_width=1920,
        frame_rate=60, frame_x_radius=FRAME_WIDTH / 2, frame_y_radius=FRAME_HEIGHT / 2,
        background_color="#000000")  # fmt: skip
    mod.__dict__.update(names)
    keep = ("config", "np", "always_redraw", "always", "f_always", "rate_functions",
            *vars(rate_functions))  # fmt: skip
    mod.__all__ = [k for k in names if k[0].isupper() or k in keep]  # type: ignore[attr-defined]

    def missing(name: str) -> Any:
        raise AttributeError(f"{name!r} is not in phasor's Manim stand-in yet (see docs/ai/API.md,"
                             " 'Manim import')")  # fmt: skip

    mod.__getattr__ = missing  # type: ignore[attr-defined]
    return mod


def import_manim(path: str | Path, scene: str | None = None) -> ManimImport:
    """Run the Manim script at ``path`` (a ``Scene`` subclass named ``scene``, or the first one
    it defines) and return the phasor scene it builds. The scene reads files (an SVG, a
    sound) from the script's folder and never runs scripts itself."""
    global _CTX
    path = Path(path)
    fake = _module()
    saved = {k: sys.modules.get(k) for k in ("manim", "manimlib")}
    sys.modules["manim"] = fake
    try:
        ns = runpy.run_path(str(path), run_name="manim_import")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    classes = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Scene)
               and v.__module__ == "manim_import"]  # fmt: skip
    if not classes:
        raise ManimImportError(f"{path.name} defines no Manim Scene class")
    if scene is not None:
        match = [c for c in classes if c.__name__ == scene]
        if not match:
            raise ManimImportError(f"no Scene {scene!r} in {path.name}; it defines "
                                   f"{[c.__name__ for c in classes]}")  # fmt: skip
        cls = match[0]
    else:
        cls = classes[0]
    base_dir = str(path.resolve().parent)
    s = PhasorScene(cls.__name__, width=1920, height=1080, base_dir=base_dir,
                    allow_scripts=False)  # fmt: skip
    s.set("camera.height", FRAME_HEIGHT)
    s.set("camera.center", (0.0, 0.0, 0.0))
    ctx = _Context(scene=s, base_dir=base_dir)
    _CTX = ctx
    try:
        instance = cls()
        instance.setup()
        instance.construct()
        ctx.flush()
        if isinstance(instance, ThreeDScene):
            instance._finish_rotation()
        s.marker("end", ctx.t)
        background = str(instance.camera.background_color)
        if background.upper() not in ("#000000", "BLACK"):  # Manim's own: keep the theme's
            value = style_color(background)
            if value.startswith("#"):  # not a role: a color phasor could read
                if len(value) == 9:  # a background is opaque: nothing is behind it
                    if value[7:] != "FF":
                        ctx.warn(f"the background's alpha ({background!r}) is not carried "
                                 "over: a background is opaque")  # fmt: skip
                    value = value[:7]
                op = {"op": "set_meta", "field": "background", "value": value}
                s.engine.apply(op, origin="script").raise_for_errors()
    except ValidationError as e:  # anything the checks above missed: said, not a traceback
        raise ManimImportError(f"{path.name}: {_validation_message(e)}") from None
    except (DocumentError, VerbError) as e:
        raise ManimImportError(f"{path.name}: {e}") from None
    except RecursionError:
        raise ManimImportError(f"{path.name}: the script recursed too deeply") from None
    finally:
        _CTX = None
    return ManimImport(
        scene=s,
        warnings=list(dict.fromkeys(ctx.warnings)),
        duration=ctx.t,
        name=cls.__name__,
        others=[c.__name__ for c in classes if c is not cls],
        trace=ctx.trace,
    )
