"""The diagram kit: number lines, number planes, bar charts, tables, matrices and code.

    from phasor import dg

    s.add(dg.NumberLine(id="line", x_range=(-3, 3, 1), pos=(0, -3, 0)))
    s.add(dg.NumberPlane(id="plane", x_range=(-4, 4, 1), y_range=(-2, 2, 1), y_scale=1.5))
    s.add(dg.BarChart(id="sales", values=(3, 5, 2), names=("ants", "bees", "cats")))
    s.add(dg.Table(id="t", cells=(("x", "$x^2$"), ("1", "1"), ("2", "4"))))
    s.add(dg.MatrixDisplay(id="m", entries=(("a", "b"), ("c", "d"))))
    s.add(dg.Code(id="code", code="def f(x):\\n    return x * x"))

Each is a composite (ADR 0005) that also draws: the furniture (lines, ticks, grid, rules,
brackets) is its own drawing, and everything a reader reads is a part the inspector, layout
and lint see by itself: a number at a tick (``line.n_2``, ``plane.axes.x_m1``), a bar and
its name (``sales.bees``, ``sales.bees_name``), a cell (``t.b2``, spreadsheet names), a
matrix entry (``m.e1_2``), a line of code (``code.line_3``). Parts are placed by the
composite (they cannot take constraints); numbers at ticks are judged together by lint
``crowded_ticks``. Text sizes and colors come from roles (or explicit values, Milestone 18).

Axes with numbers and unequal scales are ``geo.Axes`` (``numbers=True``, ``x_scale``,
``y_scale``); a number that follows a value is ``Counter``; a live 2x2 or 3x3 matrix of
numbers is ``linalg.Matrix``. Text and counts are bounded by ``diagrams_math``.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core import extras, files, record, ticks
from phasor.core import text as text_metrics
from phasor.core.ids import children, name_from, unique_names
from phasor.core.objects import DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import diagrams_math as dm
from phasor.domains.basic import Text
from phasor.domains.geometry import NUMBER_GAP, Axes, _finite_scale, crowding, number_parts
from phasor.domains.shapes import Rectangle
from phasor.domains.strokes import arrow_head, polyline, segment_batch, with_alpha

PAD_X = 0.8  # text heights of room across a table cell (half each side)
PAD_Y = 0.45  # ... and up and down
CELL = 0.52  # a code character's cell, in text heights (monospace by placement)
WIDE = 1.15  # a letter wider than this many cells (m, W, @) is set smaller to fit
LINE = 1.3  # a line of code's height, in text heights
TIP = 0.22  # an arrow tip on a number line, world units
CODE_STYLE: dict[str, tuple[str, str]] = {
    "color": ("emphasis", "primary"),  # plain text
    "keyword": ("emphasis", "secondary"),
    "name": ("emphasis", "highlight"),  # functions, classes, built-ins
    "string": ("emphasis", "danger"),
    "number": ("emphasis", "highlight"),
    "comment": ("emphasis", "muted"),  # comments and line numbers
    "size": ("text", "annotation"),
}
_KEY = {
    "p": "color",
    "k": "keyword",
    "f": "name",
    "s": "string",
    "n": "number",
    "c": "comment",
    "g": "comment",
}


def _vec(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64)


def _bounded_pos(v):
    for c in v:
        if abs(ticks.number(c, "pos")) > ticks.MAX_VALUE:
            raise ValueError(f"pos must be within +-{ticks.MAX_VALUE:g}")
    return v


def _height(me, cam, at, key: str = "size") -> float:
    """A text role's height in world units at ``at``."""
    return float(me.style[key]) * float(cam.units_per_height(_vec(at)))


def _as_math(s: str) -> str:
    """A matrix entry as math: ``a_{11}`` -> ``$a_{11}$``; text already in ``$`` stays."""
    return s if text_metrics.is_math(s) else f"${s}$"


# ---------------------------------------------------------------- NumberLine


@register
class NumberLine(ObjectBase):
    """A number line: ``x_range`` (min, max, step; step 0 a nice one) drawn ``scale`` world
    units per unit, centered on ``pos``, turned ``angle`` degrees, with ticks and (when
    ``numbers``) the number at every tick as a part (``line.n_2``, ``line.n_m1``, ``line.n_0``).
    ``tips`` puts arrow heads on its ends; ``label`` is written past its end."""

    type: ClassVar[str] = "number_line"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "numbers": ("text", "label"),
        "size": ("text", "label"),  # the label
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        x_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # the middle of the line
        scale: float = Field(1.0, gt=0)  # world units per unit
        angle: float = Field(0.0, ge=-360, le=360)  # degrees, counterclockwise from +x
        tick: float = Field(0.16, gt=0, le=10)  # tick length, world units
        numbers: bool = True
        digits: int = Field(-1, ge=-1, le=ticks.MAX_DECIMALS)  # -1: as the step needs
        tips: Literal["none", "end", "both"] = "none"
        label: str = Field("", max_length=dm.MAX_CELL)

        @field_validator("pos")
        @classmethod
        def _pos(cls, v):
            return _bounded_pos(v)

        @field_validator("label")
        @classmethod
        def _label(cls, v):
            return dm.check_text(v, "label")

        @model_validator(mode="after")
        def _range(self):
            _finite_scale(self.scale, "scale", zero=False)
            lo, hi, step = self.x_range
            ticks.check_range(lo, hi, step, "x_range")
            if max(abs(lo), abs(hi)) * self.scale > ticks.MAX_VALUE:
                raise ValueError(f"x_range times scale must be within +-{ticks.MAX_VALUE:g}")
            return self

    def _frame(self, p):
        """(direction, normal, the world point of a value)."""
        a = math.radians(p.angle)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        n = np.array([-math.sin(a), math.cos(a), 0.0])
        lo, hi, _ = p.x_range
        mid, o = (lo + hi) / 2, _vec(p.pos)

        def at(v: float) -> np.ndarray:
            return o + d * (v - mid) * p.scale

        return d, n, at

    def _numbers(self, state: SceneState):
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        d, n, at = self._frame(p)
        lo, hi, step = p.x_range
        step = ticks.step_of(lo, hi, step)
        h = _height(me, cam, p.pos, "numbers")
        digits = ticks.decimals(step) if p.digits < 0 else p.digits
        # Numbers go under a level line and left of a steep one (as on x and y axes).
        off = p.tick / 2 + NUMBER_GAP * h
        if abs(float(n @ cam.right)) > abs(float(n @ cam.up)):  # steep
            away = n if float(n @ cam.right) < 0 else -n
            shift, anchor = away * off, "right"
        else:
            away = n if float(n @ cam.up) < 0 else -n
            shift, anchor = away * (off + h / 2), "center"
        nums = [
            (v, ticks.text(v, digits), at(v) + shift, anchor) for v in ticks.values(lo, hi, step)
        ]
        return nums, step, h, d, cam

    def build(self, state: SceneState) -> list:
        me = self.me(state)
        if not me.params.numbers:
            return []
        nums, *_ = self._numbers(state)
        roles = {"color": me.roles["color"], "numbers": me.roles["numbers"]}
        return number_parts("n", nums, roles=roles, reserved=children(state.reserved, self.id))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        out = record.hybrid_features(self, state)
        p = self.me(state).params
        _, _, at = self._frame(p)
        lo, hi, _ = p.x_range
        out.update(start=at(lo), end=at(hi))
        if lo <= 0 <= hi:
            out["zero"] = at(0.0)
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        lo, hi, step = p.x_range
        step = ticks.step_of(lo, hi, step)
        out: dict[str, Any] = {
            "kind": "number_line",
            "range": [lo, hi],
            "step": step,
            "scale": p.scale,
            "ticks": len(ticks.values(lo, hi, step)),
        }
        if p.label:
            out["label"] = p.label
        if p.numbers:
            nums, step, h, d, cam = self._numbers(state)
            out["numbers"] = {
                "n": {
                    "step": step,
                    **crowding(
                        [ticks.as_math(t) for _, t, _, _ in nums], step * p.scale, d, h, cam
                    ),
                }
            }
        out["parts"] = [o.id[len(self.id) + 1 :] for o in record.parts(state, self.id)]
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        d, n, at = self._frame(p)
        lo, hi, step = p.x_range
        col = with_alpha(me.style["color"], p.opacity)
        glow = float(me.style["glow"])
        a, b = at(lo), at(hi)
        w = me.style["stroke"] * float(cam.units_per_height((a + b) / 2))
        # A tip sits a little past the range's end, so the last tick stays clear of it.
        start = a - d * TIP if p.tips == "both" else a
        end = b + d * TIP if p.tips != "none" else b
        line_a = start + d * TIP if p.tips == "both" else start
        line_b = end - d * TIP if p.tips != "none" else end
        polyline(ctx, cam, [line_a, line_b], w, color=col, glow=glow)
        if p.tips != "none":
            arrow_head(ctx, cam, end, d, TIP, color=col, glow=glow)
        if p.tips == "both":
            arrow_head(ctx, cam, start, -d, TIP, color=col, glow=glow)
        vs = np.array(ticks.values(lo, hi, step))
        pts = np.array([at(v) for v in vs])
        segment_batch(
            ctx,
            cam,
            pts - n * p.tick / 2,
            pts + n * p.tick / 2,
            me.style["stroke"] * 0.8,
            colors=col,
            glow=glow,
        )
        if p.label:
            h = _height(me, cam, b)
            ctx.text(p.label, end + d * h * 0.4, h, color=col, anchor="left", glow=glow)


# ---------------------------------------------------------------- NumberPlane


@register
class NumberPlane(ObjectBase):
    """A coordinate plane: a grid at every step of ``x_range`` and ``y_range`` (and
    ``faint`` lighter lines between), with axes through the origin and numbers at their ticks
    (an ``Axes`` part, ``plane.axes``, whose numbers are ``plane.axes.x_2``). ``scale`` is
    world units per unit; ``x_scale``/``y_scale`` (0: ``scale``) make the scales unequal.
    Labels may sit on it (a backdrop)."""

    type: ClassVar[str] = "number_plane"
    Style: ClassVar = {
        "axes": ("emphasis", "primary"),
        "grid": ("emphasis", "secondary"),
        "stroke": ("stroke", "hairline"),  # the grid's lines
        "axis_stroke": ("stroke", "regular"),
        "numbers": ("text", "label"),
        "size": ("text", "label"),  # the axis labels
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        x_range: tuple[float, float, float] = (-7.0, 7.0, 1.0)
        y_range: tuple[float, float, float] = (-4.0, 4.0, 1.0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the axes cross (0, 0)
        scale: float = Field(1.0, gt=0)
        x_scale: float = Field(0.0, ge=0)  # 0: ``scale``
        y_scale: float = Field(0.0, ge=0)
        faint: int = Field(1, ge=0, le=4)  # lighter lines between two grid lines
        numbers: bool = True
        digits: int = Field(-1, ge=-1, le=ticks.MAX_DECIMALS)
        labels: tuple[str, ...] = ()  # ("x", "y"): written at the axes' ends

        @field_validator("pos")
        @classmethod
        def _pos(cls, v):
            return _bounded_pos(v)

        @field_validator("x_scale", "y_scale")
        @classmethod
        def _scale(cls, v, info):
            return _finite_scale(v, info.field_name)

        @field_validator("labels")
        @classmethod
        def _labels(cls, v):
            if len(v) > 2:
                raise ValueError("at most 2 labels")
            return tuple(dm.check_text(x, "label") for x in v)

        @model_validator(mode="after")
        def _ranges(self):
            _finite_scale(self.scale, "scale", zero=False)
            for name in ("x_range", "y_range"):
                lo, hi, step = getattr(self, name)
                ticks.check_range(lo, hi, step, name)
                sc = getattr(self, f"{name[0]}_scale") or self.scale
                if max(abs(lo), abs(hi)) * sc > ticks.MAX_VALUE:
                    raise ValueError(f"{name} times its scale must be within +-{ticks.MAX_VALUE:g}")
            return self

    def _lines(self, p, axis: str):
        """(value, faint) of every grid line across ``axis`` (its ticks and between them)."""
        lo, hi, step = getattr(p, f"{axis}_range")
        step = ticks.step_of(lo, hi, step)
        out = []
        for v in ticks.values(lo, hi, step):
            out.append((v, False))
            for k in range(1, p.faint + 1):
                u = v + step * k / (p.faint + 1)
                if u <= hi + 1e-12:
                    out.append((u, True))
        first = ticks.values(lo, hi, step)[0]
        for k in range(1, p.faint + 1):  # between the range's start and its first tick
            u = first - step * k / (p.faint + 1)
            if u >= lo - 1e-12:
                out.append((u, True))
        return out

    def build(self, state: SceneState) -> list:
        me = self.me(state)
        p = me.params
        r = me.roles
        if "axes" in children(state.reserved, self.id):
            return []  # the document holds "<plane>.axes" itself: no axes part (lint says)
        return [
            Axes(
                id="axes",
                dims=2,
                x_range=p.x_range,
                y_range=p.y_range,
                pos=p.pos,
                scale=p.scale,
                x_scale=p.x_scale,
                y_scale=p.y_scale,
                tick=0.1,
                numbers=p.numbers,
                digits=p.digits,
                labels=p.labels,
                style={
                    "color": r["axes"],
                    "stroke": r["axis_stroke"],
                    "numbers": r["numbers"],
                    "size": r["size"],
                },
            )
        ]

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sx, sy = p.x_scale or p.scale, p.y_scale or p.scale
        return {
            "kind": "number_plane",
            "backdrop": True,
            "x": {"range": list(p.x_range[:2]), "step": ticks.step_of(*p.x_range), "scale": sx},
            "y": {"range": list(p.y_range[:2]), "step": ticks.step_of(*p.y_range), "scale": sy},
            "grid_lines": len(self._lines(p, "x")) + len(self._lines(p, "y")),
            "parts": [o.id[len(self.id) + 1 :] for o in record.parts(state, self.id)],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        o = _vec(p.pos)
        sx, sy = p.x_scale or p.scale, p.y_scale or p.scale
        (x0, x1, _), (y0, y1, _) = p.x_range, p.y_range
        a, b, faint = [], [], []
        for v, f in self._lines(p, "x"):
            a.append(o + np.array([v * sx, y0 * sy, 0]))
            b.append(o + np.array([v * sx, y1 * sy, 0]))
            faint.append(f)
        for v, f in self._lines(p, "y"):
            a.append(o + np.array([x0 * sx, v * sy, 0]))
            b.append(o + np.array([x1 * sx, v * sy, 0]))
            faint.append(f)
        base = np.asarray(me.style["grid"], dtype=np.float64)
        cols = np.array([(*base[:3], base[3] * p.opacity * (0.35 if f else 0.8)) for f in faint])
        segment_batch(
            ctx,
            cam,
            np.array(a),
            np.array(b),
            me.style["stroke"],
            colors=cols,
            glow=float(me.style["glow"]),
        )


# ---------------------------------------------------------------- BarChart

_UP, _RIGHT = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class _Bars:
    """Where a bar chart's box is (world units) and its y range."""

    cam: Any
    lo: float
    hi: float
    step: float
    left: float
    bottom: float
    height: float
    base: float  # the value bars rise from: 0, or the range's end nearest it
    slot: float  # the width each bar has, gap included
    h: float  # the numbers' text height
    z: float

    def y(self, v: float) -> float:
        return self.bottom + (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo) * (
            self.height
        )


@register
class BarChart(ObjectBase):
    """Bars for ``values`` in a ``width`` by ``height`` box centered on ``pos``, with a y axis
    (``y_range``; (0, 0, 0) fits a nice range to the values) and its numbers. Each bar is a
    ``Rectangle`` part named after its entry of ``names`` (``chart.bees``; ``bar_1`` without
    names), with its name under it (``chart.bees_name``) and, when ``show_values``, its value
    over it (``chart.bees_value``). ``colors`` gives emphasis roles per bar (cycled);
    ``grow`` (0..1) raises the bars from the baseline (animate it)."""

    type: ClassVar[str] = "bar_chart"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),  # axes, names and numbers
        "stroke": ("stroke", "hairline"),
        "bar_stroke": ("stroke", "regular"),
        "fill": ("fill", "solid"),
        "numbers": ("text", "label"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        values: tuple[float, ...] = (3.0, 5.0, 2.0, 4.0)
        names: tuple[str, ...] = ()
        colors: tuple[str, ...] = ()  # emphasis roles, one per bar (cycled)
        pos: Vec3 = (0.0, 0.0, 0.0)  # the middle of the chart's box
        width: float = Field(6.0, gt=0, le=ticks.MAX_VALUE)
        height: float = Field(4.0, gt=0, le=ticks.MAX_VALUE)
        y_range: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (0, 0, 0): fit the values
        gap: float = Field(0.3, ge=0, le=0.9)  # share of each bar's slot left empty
        grow: float = Field(1.0, ge=0, le=1)
        numbers: bool = True
        show_values: bool = False
        digits: int = Field(-1, ge=-1, le=ticks.MAX_DECIMALS)

        @field_validator("values")
        @classmethod
        def _values(cls, v):
            if not 1 <= len(v) <= dm.MAX_BARS:
                raise ValueError(f"1 to {dm.MAX_BARS} values")
            for x in v:
                if abs(ticks.number(x, "a value")) > ticks.MAX_VALUE:
                    raise ValueError(f"values must be within +-{ticks.MAX_VALUE:g}")
            return v

        @field_validator("names")
        @classmethod
        def _names(cls, v):
            return tuple(dm.check_text(x, "a name") for x in v)

        @field_validator("colors")
        @classmethod
        def _colors(cls, v):
            from phasor.domains.basic import EMPHASIS

            if len(v) > dm.MAX_BARS or any(c not in EMPHASIS for c in v):
                raise ValueError(f"colors are emphasis roles: {', '.join(EMPHASIS)}")
            return v

        @field_validator("pos")
        @classmethod
        def _pos(cls, v):
            return _bounded_pos(v)

        @model_validator(mode="after")
        def _fit(self):
            if self.names and len(self.names) != len(self.values):
                raise ValueError("names: one per value (or none)")
            if tuple(self.y_range) != (0.0, 0.0, 0.0):
                ticks.check_range(*self.y_range, "y_range")
            return self

    def _range(self, p) -> tuple[float, float, float]:
        if tuple(p.y_range) != (0.0, 0.0, 0.0):
            lo, hi, step = p.y_range
            return lo, hi, ticks.step_of(lo, hi, step)
        return ticks.nice_range(min(0.0, *p.values), max(0.0, *p.values))

    def _layout(self, state: SceneState) -> _Bars:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        lo, hi, step = self._range(p)
        c = _vec(p.pos)
        return _Bars(
            cam=cam,
            lo=lo,
            hi=hi,
            step=step,
            left=c[0] - p.width / 2,
            bottom=c[1] - p.height / 2,
            height=p.height,
            base=min(max(0.0, lo), hi),
            slot=p.width / len(p.values),
            h=_height(me, cam, c, "numbers"),
            z=c[2],
        )

    def _names(self, state: SceneState) -> list[str]:
        p = self.me(state).params
        raw = [name_from(n, "bar", 30) for n in p.names] or [
            f"bar_{i + 1}" for i in range(len(p.values))
        ]
        return unique_names(raw, children(state.reserved, self.id))

    def _ynumbers(self, state: SceneState, b: _Bars):
        p = self.me(state).params
        digits = ticks.decimals(b.step) if p.digits < 0 else p.digits
        x = b.left - 0.06 - NUMBER_GAP * b.h
        return [(v, ticks.text(v, digits), np.array([x, b.y(v), b.z]), "right")
                for v in ticks.values(b.lo, b.hi, b.step)]  # fmt: skip

    def build(self, state: SceneState) -> list:
        me = self.me(state)
        p = me.params
        b = self._layout(state)
        r = me.roles
        out: list = []
        names = self._names(state)
        digits = ticks.decimals(b.step) if p.digits < 0 else p.digits
        text_style = {"color": r["color"], "size": r["numbers"]}
        for i, (v, name) in enumerate(zip(p.values, names, strict=True)):
            role = p.colors[i % len(p.colors)] if p.colors else "secondary"
            x = b.left + b.slot * (i + 0.5)
            top = b.base + (v - b.base) * p.grow
            y0, y1 = b.y(b.base), b.y(top)
            if abs(y1 - y0) > 1e-9:
                style = {"color": role, "fill_color": role, "fill": r["fill"],
                         "stroke": r["bar_stroke"], "glow": r["glow"]}  # fmt: skip
                out.append(Rectangle(id=name, pos=(x, (y0 + y1) / 2, b.z),
                                     width=b.slot * (1 - p.gap), height=abs(y1 - y0),
                                     style=style))  # fmt: skip
            if p.names:
                out.append(Text(id=f"{name}_name", text=p.names[i], style=text_style,
                                pos=(x, b.bottom - NUMBER_GAP * b.h - b.h / 2, b.z),
                                tags=["tick", "along_names"]))  # fmt: skip
            if p.show_values and p.grow > 0:
                up = 1 if top >= b.base else -1
                out.append(Text(id=f"{name}_value", text=ticks.as_math(ticks.text(v, digits)),
                                pos=(x, y1 + up * (NUMBER_GAP * b.h + b.h / 2), b.z),
                                style=text_style))  # fmt: skip
        if p.numbers:
            taken = children(state.reserved, self.id) | {o.id for o in out}
            taken |= {f"{n}_{k}" for n in names for k in ("name", "value")}
            out += number_parts("y", self._ynumbers(state, b), roles=r, reserved=taken)
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        b = self._layout(state)
        names = self._names(state)
        bars = [{"part": n, **({"name": p.names[i]} if p.names else {}), "value": v}
                for i, (n, v) in enumerate(zip(names, p.values, strict=True))]  # fmt: skip
        out: dict[str, Any] = {"kind": "bar_chart", "y_range": [b.lo, b.hi], "step": b.step,
                               "bars": bars, "largest": names[int(np.argmax(p.values))],
                               "grow": p.grow}  # fmt: skip
        nums: dict[str, Any] = {}
        if p.numbers:
            texts = [ticks.as_math(t) for _, t, _, _ in self._ynumbers(state, b)]
            spacing = b.step / (b.hi - b.lo) * p.height
            nums["y"] = {"step": b.step, **crowding(texts, spacing, _UP, b.h, b.cam)}
        if p.names:
            nums["names"] = {"step": 1, **crowding(list(p.names), b.slot, _RIGHT, b.h, b.cam)}
        if nums:
            out["numbers"] = nums
        out["parts"] = [o.id[len(self.id) + 1 :] for o in record.parts(state, self.id)]
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        b = self._layout(state)
        cam = b.cam
        col = with_alpha(me.style["color"], p.opacity)
        glow = float(me.style["glow"])
        w = me.style["stroke"] * float(cam.units_per_height(_vec(p.pos)))
        top, base = b.bottom + p.height, b.y(b.base)
        polyline(ctx, cam, [(b.left, b.bottom, b.z), (b.left, top, b.z)], w, color=col, glow=glow)
        polyline(ctx, cam, [(b.left, base, b.z), (b.left + p.width, base, b.z)], w, color=col,
                 glow=glow)  # fmt: skip
        a = np.array([(b.left - 0.06, b.y(v), b.z) for v in ticks.values(b.lo, b.hi, b.step)])
        segment_batch(ctx, cam, a, a + np.array([0.12, 0, 0]), me.style["stroke"], colors=col,
                      glow=glow)  # fmt: skip


# ---------------------------------------------------------------- Table


@functools.lru_cache(maxsize=32)
def _read_table(stamp: tuple) -> tuple[tuple[tuple[str, ...], ...], str | None, str | None]:
    """(rows, problem, cut) of the CSV file ``stamp`` names (cached by its path, time, size)."""
    try:
        data = files.read_bytes(stamp[0], dm.MAX_FILE_BYTES)
    except ValueError as e:
        return (), str(e), None
    rows, problem, cut = dm.parse_csv(data)
    return tuple(tuple(r) for r in rows), problem, cut


@functools.lru_cache(maxsize=32)
def _read_code(stamp: tuple) -> tuple[str, str | None, str | None]:
    try:
        data = files.read_bytes(stamp[0], dm.MAX_FILE_BYTES)
    except ValueError as e:
        return "", str(e), None
    return dm.read_code(data)


def _file_semantics(out: dict[str, Any], path: str, state: SceneState, problem, cut) -> None:
    out["file"] = path
    if not files.exists(state.base_dir, path):
        out["missing_file"] = path
    if problem:
        out["unreadable"] = problem  # lint bad_file
    if cut:
        out["truncated"] = cut  # lint truncated


def _grid(cells, h: float):
    """Column widths and row heights (world units) of text cells ``h`` tall."""
    rows, cols = len(cells), max((len(r) for r in cells), default=0)
    widths = [
        max([text_metrics.aspect(cells[i][j]) * h for i in range(rows) if cells[i][j]] + [h])
        + PAD_X * h
        for j in range(cols)
    ]
    heights = [max([text_metrics.box(c) * h for c in row if c] + [h]) + PAD_Y * h for row in cells]
    return widths, heights


@register
class Table(ObjectBase):
    """A table of text (maths in ``$...$``) from ``cells`` (rows of text) or a CSV file
    ``path`` in the scene's folder, centered on ``pos``. Cells are parts named like a
    spreadsheet's (``t.a1`` is the first row's first cell, ``t.b2`` the second's second);
    with ``header`` the first row is set in the ``header`` role. ``lines`` draws rules
    (``header``: one under the header; ``grid``: every row and column); ``reveal`` (0..1)
    shows the rows one by one (animate it)."""

    type: ClassVar[str] = "table"
    Files: ClassVar = ("path",)
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "header": ("emphasis", "secondary"),
        "rules": ("emphasis", "muted"),
        "stroke": ("stroke", "hairline"),
        "size": ("text", "label"),
    }

    class Params(ParamsBase):
        cells: tuple[tuple[str, ...], ...] = ()
        path: str = ""  # a .csv file in the scene's folder, instead of ``cells``
        pos: Vec3 = (0.0, 0.0, 0.0)  # the table's middle
        header: bool = True
        lines: Literal["none", "header", "grid"] = "header"
        align: Literal["center", "left", "right"] = "center"
        reveal: float = Field(1.0, ge=0, le=1)  # share of the rows shown

        @field_validator("cells")
        @classmethod
        def _cells(cls, v):
            return dm.check_grid(v, "cells", dm.MAX_ROWS, dm.MAX_COLS)

        @field_validator("path")
        @classmethod
        def _path(cls, v):
            return files.relative_path(v) if v else v

        @field_validator("pos")
        @classmethod
        def _pos(cls, v):
            return _bounded_pos(v)

        @model_validator(mode="after")
        def _one_source(self):
            if self.cells and self.path:
                raise ValueError("give cells or a path to a .csv file, not both")
            return self

    def data(self, state: SceneState) -> tuple[list[list[str]], str | None, str | None]:
        """(rows padded to one width, problem, cut): ``cells``, or the file's."""
        p = self.me(state).params
        if not p.path:
            return dm.padded(p.cells), None, None
        stamp = files.stamp(state.base_dir, p.path)
        if stamp[1] is None:
            return [], None, None
        rows, problem, cut = _read_table(stamp)
        return dm.padded(rows), problem, cut

    def _layout(self, state: SceneState):
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        rows, *_ = self.data(state)
        h = _height(me, cam, p.pos)
        widths, heights = _grid(rows, h)
        w, hh = sum(widths), sum(heights)
        corner = _vec(p.pos) - cam.right * (w / 2) + cam.up * (hh / 2)  # top left
        shown = min(len(rows), math.ceil(p.reveal * len(rows) - 1e-9))
        return cam, rows, widths, heights, corner, shown

    def build(self, state: SceneState) -> list:
        me = self.me(state)
        p = me.params
        cam, rows, widths, heights, corner, shown = self._layout(state)
        reserved = children(state.reserved, self.id)
        out, y = [], 0.0
        for i, row in enumerate(rows[:shown]):
            x = 0.0
            role = me.roles["header"] if p.header and i == 0 else me.roles["color"]
            for j, cell in enumerate(row):
                name = f"{dm.column_name(j)}{i + 1}"
                if cell and name not in reserved:
                    k = {"left": 0.0, "center": 0.5, "right": 1.0}[p.align]
                    inner = PAD_X * _height(me, cam, p.pos) / 2
                    dx = x + inner + k * (widths[j] - 2 * inner)
                    at = corner + cam.right * dx - cam.up * (y + heights[i] / 2)
                    out.append(
                        Text(
                            id=name,
                            text=cell,
                            pos=tuple(float(c) for c in at),
                            anchor=p.align,
                            style={"color": role, "size": me.roles["size"]},
                        )
                    )
                x += widths[j]
            y += heights[i]
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        rows, problem, cut = self.data(state)
        out: dict[str, Any] = {
            "kind": "table",
            "rows": len(rows),
            "columns": max((len(r) for r in rows), default=0),
        }
        if p.header and rows:
            out["header"] = rows[0]
        if p.path:
            _file_semantics(out, p.path, state, problem, cut)
        out["parts"] = [o.id[len(self.id) + 1 :] for o in record.parts(state, self.id)]
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if p.lines == "none":
            return
        cam, _, widths, heights, corner, shown = self._layout(state)
        if not shown:
            return
        w, hh = sum(widths), sum(heights[:shown])
        a, b = [], []
        if p.lines == "header" and p.header and shown > 1:
            y = heights[0]
            a.append(corner - cam.up * y)
            b.append(corner - cam.up * y + cam.right * w)
        elif p.lines == "grid":
            for y in np.cumsum([0.0, *heights[:shown]]):
                a.append(corner - cam.up * y)
                b.append(corner - cam.up * y + cam.right * w)
            for x in np.cumsum([0.0, *widths]):
                a.append(corner + cam.right * x)
                b.append(corner + cam.right * x - cam.up * hh)
        if a:
            segment_batch(
                ctx,
                cam,
                np.array(a),
                np.array(b),
                me.style["stroke"],
                colors=with_alpha(me.style["rules"], p.opacity),
            )


# ---------------------------------------------------------------- MatrixDisplay


@register
class MatrixDisplay(ObjectBase):
    """A matrix of maths: ``entries`` are rows of text, each typeset as maths
    (``"a_{11}"``, ``"\\\\frac{1}{2}"``, ``"0"``), in ``bracket`` (``square``, ``round``,
    ``bar`` for a determinant, ``none``), centered on ``pos``. Each entry is a part,
    ``m.e1_2`` (row 1, column 2). For a live matrix of numbers that follows another
    object's matrix, use ``linalg.Matrix``."""

    type: ClassVar[str] = "matrix_display"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "size": ("text", "math"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        entries: tuple[tuple[str, ...], ...] = (("1", "0"), ("0", "1"))
        bracket: Literal["square", "round", "bar", "none"] = "square"
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("entries")
        @classmethod
        def _entries(cls, v):
            if not v or not any(v):
                raise ValueError("a matrix needs at least one entry")
            dm.check_grid(v, "entries", dm.MAX_MATRIX, dm.MAX_MATRIX, dm.MAX_MATRIX**2)
            for row in v:
                for e in row:
                    if e:
                        text_metrics.check_math(_as_math(e))
            return v

        @field_validator("pos")
        @classmethod
        def _pos(cls, v):
            return _bounded_pos(v)

    def _layout(self, state: SceneState):
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        rows = [[_as_math(e) if e else "" for e in r] for r in dm.padded(p.entries)]
        h = _height(me, cam, p.pos)
        widths, heights = _grid(rows, h)
        row_h = max(heights)
        w, hh = sum(widths), row_h * len(rows)
        corner = _vec(p.pos) - cam.right * (w / 2) + cam.up * (hh / 2)
        return cam, rows, widths, row_h, corner, w, hh, h

    def build(self, state: SceneState) -> list:
        me = self.me(state)
        cam, rows, widths, row_h, corner, *_ = self._layout(state)
        reserved = children(state.reserved, self.id)
        style = {"color": me.roles["color"], "size": me.roles["size"]}
        out = []
        for i, row in enumerate(rows):
            x = 0.0
            for j, e in enumerate(row):
                name = f"e{i + 1}_{j + 1}"
                if e and name not in reserved:
                    at = corner + cam.right * (x + widths[j] / 2) - cam.up * (row_h * (i + 0.5))
                    out.append(Text(id=name, text=e, pos=tuple(float(c) for c in at), style=style))
                x += widths[j]
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        rows = dm.padded(p.entries)
        return {
            "kind": "matrix_display",
            "rows": len(rows),
            "columns": len(rows[0]),
            "entries": rows,
            "bracket": p.bracket,
            "parts": [o.id[len(self.id) + 1 :] for o in record.parts(state, self.id)],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if p.bracket == "none":
            return
        cam, *_, w, hh, h = self._layout(state)
        col = with_alpha(me.style["color"], p.opacity)
        glow = float(me.style["glow"])
        width = me.style["stroke"] * float(cam.units_per_height(_vec(p.pos)))
        mid = _vec(p.pos)
        reach = 0.25 * h  # how far a bracket's serif or bulge reaches
        for sgn in (-1, 1):
            x = mid + cam.right * (sgn * (w / 2 + 0.1 * h))
            top, bot = x + cam.up * (hh / 2), x - cam.up * (hh / 2)
            inward = cam.right * (-sgn * reach)
            if p.bracket == "square":
                pts = [top + inward, top, bot, bot + inward]
            elif p.bracket == "bar":
                pts = [top, bot]
            else:  # round: a bow out from the entries
                s = np.linspace(-1, 1, 25)[:, None]
                pts = x + cam.up * (hh / 2) * s + inward * (s**2 - 1) * 0.9 + inward * 0.9
                pts = [*pts]
            polyline(ctx, cam, pts, width, color=col, glow=glow)


# ---------------------------------------------------------------- Code


_LANGUAGE = re.compile(r"[a-z0-9][a-z0-9+#._-]*")


@register
class CodeLine(ObjectBase):
    """One line of code, in a monospace layout: every character centered in a cell
    ``CELL`` text heights wide, colored by its letter in ``classes`` (``p`` plain, ``k``
    keyword, ``f`` name, ``s`` string, ``n`` number, ``c`` comment, ``g`` line number).
    ``pos`` is the line's left end, at its middle height. ``Code`` builds these as its
    parts; one can also stand alone."""

    type: ClassVar[str] = "code_line"
    Style: ClassVar = CODE_STYLE

    class Params(ParamsBase):
        text: str = Field("", max_length=dm.MAX_LINE + 8)
        classes: str = Field("", max_length=dm.MAX_LINE + 8)  # one letter per character
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("pos")
        @classmethod
        def _pos(cls, v):
            return _bounded_pos(v)

        @model_validator(mode="after")
        def _classes(self):
            bad = [ch for ch in self.text if not ch.isprintable()]
            if bad:
                raise ValueError(f"one line of printable text (not {bad[0]!r})")
            if self.classes and len(self.classes) != len(self.text):
                raise ValueError("classes: one letter per character of text")
            if set(self.classes) - set(_KEY):
                raise ValueError(f"classes are letters of {''.join(_KEY)}")
            return self

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "label", "text": p.text.strip() or " ", "code": True}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        h = _height(me, cam, p.pos)
        cell = CELL * h
        start = _vec(p.pos)
        classes = p.classes or "p" * len(p.text)
        for k, (ch, cl) in enumerate(zip(p.text, classes, strict=True)):
            if ch.isspace():
                continue
            hh = h
            wide = text_metrics.aspect(ch) * h
            if wide > WIDE * cell:  # set smaller, on the same baseline
                hh = h * WIDE * cell / wide
            at = start + cam.right * (cell * (k + 0.5))
            at = at - cam.up * ((h - hh) * (0.5 - text_metrics.depth(ch)))
            color = with_alpha(me.style[_KEY[cl]], p.opacity)
            ctx.text(ch, at, hh, color=color, anchor="center")


@register
class Code(ObjectBase):
    """A block of code in a monospace layout, highlighted by pygments when it is installed
    (the ``code`` extra; without it the code is plain, and lint says so), from ``code`` or a
    file ``path`` in the scene's folder, in ``language`` (a pygments name: ``python``,
    ``c``, ``rust``, ``javascript``...), centered on ``pos``. Each line is a ``CodeLine``
    part, ``code.line_3`` (blank lines have none); ``line_numbers`` writes them in a
    gutter; ``reveal`` (0..1) shows the lines one by one (animate it)."""

    type: ClassVar[str] = "code"
    Files: ClassVar = ("path",)
    Style: ClassVar = CODE_STYLE

    class Params(ParamsBase):
        code: str = ""
        path: str = ""  # a source file in the scene's folder, instead of ``code``
        language: str = Field("python", min_length=1, max_length=dm.MAX_LANGUAGE)
        pos: Vec3 = (0.0, 0.0, 0.0)  # the block's middle
        line_numbers: bool = False
        reveal: float = Field(1.0, ge=0, le=1)  # share of the lines shown

        @field_validator("code")
        @classmethod
        def _code(cls, v):
            return dm.check_code(v)

        @field_validator("language")
        @classmethod
        def _language(cls, v):
            if not _LANGUAGE.fullmatch(v):
                raise ValueError("a language is a lower-case name such as 'python' or 'c++'")
            return v

        @field_validator("path")
        @classmethod
        def _path(cls, v):
            return files.relative_path(v) if v else v

        @field_validator("pos")
        @classmethod
        def _pos(cls, v):
            return _bounded_pos(v)

        @model_validator(mode="after")
        def _one_source(self):
            if self.code and self.path:
                raise ValueError("give code or a path to a file, not both")
            return self

    def source(self, state: SceneState) -> tuple[str, str | None, str | None]:
        """(code, problem, cut): ``code``, or the file's."""
        p = self.me(state).params
        if not p.path:
            return p.code, None, None
        stamp = files.stamp(state.base_dir, p.path)
        if stamp[1] is None:
            return "", None, None
        return _read_code(stamp)

    def _lines(self, state: SceneState):
        """(text, classes) per line, gutter included, and how it was highlighted."""
        p = self.me(state).params
        src, *_ = self.source(state)
        if not src:
            return [], "none"
        lines = dm.code_lines(src)
        classes, how = dm.highlight(src, p.language)
        if p.line_numbers:
            wide = len(str(len(lines)))
            lines = [f"{i + 1:>{wide}}  {x}" for i, x in enumerate(lines)]
            classes = [("g" * wide) + "pp" + c for c in classes]
        return list(zip(lines, classes, strict=True)), how

    def build(self, state: SceneState) -> list:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        lines, _ = self._lines(state)
        if not lines:
            return []
        h = _height(me, cam, p.pos)
        cols = max(len(x) for x, _ in lines)
        w, hh = cols * CELL * h, len(lines) * LINE * h
        corner = _vec(p.pos) - cam.right * (w / 2) + cam.up * (hh / 2)
        shown = min(len(lines), math.ceil(p.reveal * len(lines) - 1e-9))
        reserved = children(state.reserved, self.id)
        style = {k: me.roles[k] for k in CODE_STYLE}
        out = []
        for i, (text, classes) in enumerate(lines[:shown]):
            name = f"line_{i + 1}"
            if not text.strip() or name in reserved:
                continue
            at = corner - cam.up * (LINE * h * (i + 0.5))
            out.append(
                CodeLine(
                    id=name,
                    text=text.rstrip(),
                    classes=classes[: len(text.rstrip())],
                    pos=tuple(float(c) for c in at),
                    style=style,
                )
            )
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        src, problem, cut = self.source(state)
        lines, how = self._lines(state)
        out: dict[str, Any] = {
            "kind": "code",
            "language": p.language,
            "lines": len(lines),
            "highlight": how,
        }
        if how == "plain" and src and not extras.available("pygments"):
            out["missing_extra"] = "code"  # lint missing_extra: drawn plain
        if how == "unknown":
            out["unknown_language"] = p.language  # lint unknown_language
        if p.path:
            _file_semantics(out, p.path, state, problem, cut)
        out["parts"] = [o.id[len(self.id) + 1 :] for o in record.parts(state, self.id)]
        return out
