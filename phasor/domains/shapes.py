"""Shapes on ``Path`` (Milestone 16), boolean operations, and SVG drawings.

    from phasor import shapes

    s.add(shapes.Circle(id="disc", r=1.2, style={"fill": "soft"}))
    s.add(shapes.Rectangle(id="card", width=3, height=2, corner_radius=0.3))
    s.add(shapes.Arc(id="swing", r=2, start=0, angle=120, tips="end"))
    s.add(shapes.union(shapes.Circle(id="a", pos=(-0.5, 0)), shapes.Circle(id="b", pos=(0.5, 0)),
                       id="both", style={"fill": "solid"}))
    s.add(shapes.SVG(id="logo", path="logo.svg", height=2))

Every shape lies in the plane z = ``pos.z`` (the screen in 2D) and is a ``Path``: it has a
fill, a stroke, dashes, ``progress`` and the same style keys. Angles are in degrees,
counterclockwise from the x axis. The maths is in ``path_math``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, model_validator

from phasor.core.objects import ObjectBase, register
from phasor.core.values import Vec3
from phasor.domains import path_math as pm
from phasor.domains.path import MEASURE, Path, PathBase, PathParams, Tips
from phasor.domains.svg import SVG

__all__ = ["SVG", "Annulus", "Arc", "Brace", "Circle", "Ellipse", "Path", "Polygon",
           "Rectangle", "RegularPolygon", "Sector", "difference", "exclusion", "intersection",
           "union"]  # fmt: skip


def _xy(v) -> list[float]:
    return [round(float(x), 6) for x in v]


@register
class Circle(PathBase):
    """A circle of radius ``r`` around ``pos``."""

    type: ClassVar[str] = "circle"
    KIND: ClassVar[str] = "circle"

    class Params(PathParams):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(1.0, gt=0)

    def bezier(self, p):
        return [pm.circle(p.pos, p.r)]

    def facts(self, p):
        return {"center": _xy(p.pos), "radius": p.r, "area": round(math.pi * p.r**2, 4),
                "length": round(2 * math.pi * p.r, 4)}  # fmt: skip

    def shape_features(self, p):
        return {"center": np.asarray(p.pos, dtype=np.float64)}


@register
class Ellipse(PathBase):
    """An ellipse ``width`` by ``height`` around ``pos``, turned by ``rotation``."""

    type: ClassVar[str] = "ellipse"
    KIND: ClassVar[str] = "ellipse"

    class Params(PathParams):
        pos: Vec3 = (0.0, 0.0, 0.0)
        width: float = Field(2.0, gt=0)
        height: float = Field(1.0, gt=0)
        rotation: float = 0.0

    def bezier(self, p):
        return [pm.ellipse(p.pos, p.width / 2, p.height / 2, math.radians(p.rotation))]

    def facts(self, p):
        return {"center": _xy(p.pos), "width": p.width, "height": p.height,
                "area": round(math.pi * p.width * p.height / 4, 4)}  # fmt: skip

    def shape_features(self, p):
        return {"center": np.asarray(p.pos, dtype=np.float64)}


@register
class Arc(PathBase):
    """Part of a circle: radius ``r`` around ``pos``, from ``start`` through ``angle``
    degrees (negative: clockwise). Open, so it can carry arrow ``tips``."""

    type: ClassVar[str] = "arc"
    KIND: ClassVar[str] = "arc"

    class Params(PathParams):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(1.0, gt=0)
        start: float = 0.0
        angle: float = 90.0
        tips: Tips = "none"

        @model_validator(mode="after")
        def _angle(self):
            if self.angle == 0 or abs(self.angle) > 360:
                raise ValueError("an arc's angle must not be 0, and at most a full turn "
                                 "(-360 to 360 degrees)")  # fmt: skip
            return self

    def is_closed(self, p, sub) -> bool:
        return False  # a full turn is still an arc: it keeps its tips and has no area

    def bezier(self, p):
        return [pm.arc(p.pos, p.r, math.radians(p.start), math.radians(p.angle))]

    def facts(self, p):
        return {"center": _xy(p.pos), "radius": p.r, "start": p.start, "angle": p.angle,
                "length": round(abs(math.radians(p.angle)) * p.r, 4)}  # fmt: skip

    def shape_features(self, p):
        return {"center": np.asarray(p.pos, dtype=np.float64)}


@register
class Sector(PathBase):
    """A pie slice of radius ``r``, from ``start`` through ``angle`` degrees; with ``inner``
    > 0 a slice of a ring."""

    type: ClassVar[str] = "sector"
    KIND: ClassVar[str] = "sector"

    class Params(PathParams):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(1.0, gt=0)
        start: float = 0.0
        angle: float = Field(90.0, gt=0, le=360)
        inner: float = Field(0.0, ge=0)

        @model_validator(mode="after")
        def _inner(self):
            if self.inner >= self.r:
                raise ValueError("a sector's inner radius must be less than r")
            return self

    def bezier(self, p):
        return [pm.sector(p.pos, p.r, math.radians(p.start), math.radians(p.angle), p.inner)]

    def facts(self, p):
        area = math.radians(p.angle) / 2 * (p.r**2 - p.inner**2)
        return {"center": _xy(p.pos), "radius": p.r, "inner": p.inner, "start": p.start,
                "angle": p.angle, "area": round(area, 4)}  # fmt: skip

    def shape_features(self, p):
        return {"center": np.asarray(p.pos, dtype=np.float64)}


@register
class Annulus(PathBase):
    """A ring between radii ``inner`` and ``outer`` around ``pos`` (a fill with a hole)."""

    type: ClassVar[str] = "annulus"
    KIND: ClassVar[str] = "annulus"

    class Params(PathParams):
        pos: Vec3 = (0.0, 0.0, 0.0)
        outer: float = Field(1.0, gt=0)
        inner: float = Field(0.5, gt=0)

        @model_validator(mode="after")
        def _radii(self):
            if self.inner >= self.outer:
                raise ValueError("an annulus's inner radius must be less than its outer one")
            return self

    def bezier(self, p):
        return [pm.circle(p.pos, p.outer), pm.circle(p.pos, p.inner)[::-1].copy()]

    def facts(self, p):
        return {"center": _xy(p.pos), "outer": p.outer, "inner": p.inner,
                "area": round(math.pi * (p.outer**2 - p.inner**2), 4)}  # fmt: skip

    def shape_features(self, p):
        return {"center": np.asarray(p.pos, dtype=np.float64)}


@register
class Polygon(PathBase):
    """A closed polygon through ``vertices`` (any number, in order). The type is
    ``polygon_path``: ``polygon`` is ``algebra.Polygon``, a regular n-gon with numbered
    vertices for symmetry groups."""

    type: ClassVar[str] = "polygon_path"
    KIND: ClassVar[str] = "polygon"

    class Params(PathParams):
        vertices: tuple[Vec3, ...] = Field(((-1.0, -0.6, 0.0), (1.0, -0.6, 0.0), (0.0, 1.0, 0.0)),
                                           min_length=3)  # fmt: skip

    def bezier(self, p):
        return [pm.polyline(p.vertices, closed=True)]

    def facts(self, p):
        return {"vertices": len(p.vertices)}

    def shape_features(self, p):
        return {"vertices": np.asarray(p.vertices, dtype=np.float64)}


@register
class RegularPolygon(PathBase):
    """A regular ``n``-gon of circumradius ``r`` around ``pos``: its first vertex at the top,
    turned by ``rotation`` degrees."""

    type: ClassVar[str] = "regular_polygon"
    KIND: ClassVar[str] = "polygon"

    class Params(PathParams):
        n: int = Field(6, ge=3, le=64)
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(1.0, gt=0)
        rotation: float = 0.0

    def _vertices(self, p) -> np.ndarray:
        return pm.regular_vertices(p.n, p.r, p.rotation, p.pos)

    def bezier(self, p):
        return [pm.polyline(self._vertices(p), closed=True)]

    def facts(self, p):
        side = 2 * p.r * math.sin(math.pi / p.n)
        return {"center": _xy(p.pos), "n": p.n, "radius": p.r, "side": round(side, 6),
                "area": round(p.n * p.r**2 * math.sin(2 * math.pi / p.n) / 2, 4)}  # fmt: skip

    def shape_features(self, p):
        return {"center": np.asarray(p.pos, dtype=np.float64), "vertices": self._vertices(p)}


@register
class Rectangle(PathBase):
    """A ``width`` by ``height`` rectangle around ``pos``, its corners rounded by
    ``corner_radius`` (at most half the shorter side), turned by ``rotation`` degrees."""

    type: ClassVar[str] = "rectangle"
    KIND: ClassVar[str] = "rectangle"

    class Params(PathParams):
        pos: Vec3 = (0.0, 0.0, 0.0)
        width: float = Field(2.0, gt=0)
        height: float = Field(1.0, gt=0)
        corner_radius: float = Field(0.0, ge=0)
        rotation: float = 0.0

    def bezier(self, p):
        return [pm.rectangle(p.pos, p.width, p.height, p.corner_radius, p.rotation)]

    def facts(self, p):
        r = min(p.corner_radius, p.width / 2, p.height / 2)
        area = p.width * p.height - (4 - math.pi) * r * r
        return {"center": _xy(p.pos), "width": p.width, "height": p.height,
                "corner_radius": r, "area": round(area, 4)}  # fmt: skip

    def shape_features(self, p):
        t = math.radians(p.rotation)
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        w, h = p.width / 2, p.height / 2
        xy = np.array([(w, -h), (w, h), (-w, h), (-w, -h)]) @ rot.T + np.asarray(p.pos[:2])
        return {"center": np.asarray(p.pos, dtype=np.float64),
                "vertices": np.c_[xy, np.full(4, p.pos[2])]}  # fmt: skip


@register
class Brace(PathBase):
    """A curly brace from ``start`` to ``end``, its point ``depth`` out to the right of that
    direction: left to right, it points down (a brace under a term). Feature ``tip`` is its
    point, where a label goes."""

    type: ClassVar[str] = "brace"
    KIND: ClassVar[str] = "brace"
    Style: ClassVar = {**PathBase.Style, "stroke": ("stroke", "bold")}

    class Params(PathParams):
        start: Vec3 = (-1.0, 0.0, 0.0)
        end: Vec3 = (1.0, 0.0, 0.0)
        depth: float = Field(0.25, gt=0)

    def bezier(self, p):
        return [pm.brace(p.start, p.end, p.depth)]

    def is_closed(self, p, sub) -> bool:
        return False  # a brace of no width is a point, not a loop

    def _tip(self, p) -> np.ndarray:
        sub = pm.brace(p.start, p.end, p.depth)
        return sub[len(sub) // 2].copy()

    def facts(self, p):
        width = float(np.linalg.norm(np.subtract(p.end, p.start)))
        return {"width": round(width, 6), "tip": _xy(self._tip(p))}

    def shape_features(self, p):
        return {"tip": self._tip(p)}


# ---------------------------------------------------------------- boolean operations


def _region_of(shape: ObjectBase) -> tuple[list[np.ndarray], str]:
    if not isinstance(shape, PathBase):
        raise TypeError(f"a boolean operation takes shapes or paths, got {shape!r}")
    try:
        p = type(shape).Params.model_validate(shape._params)
    except Exception as e:
        raise ValueError(f"{shape.id}: a boolean operation needs plain values ({e})") from None
    subs = shape.subpaths(p)
    tol = max(pm.size_of(subs), 1e-9) * MEASURE * 20
    return list(shape.flat(p, tol)), shape.rule(p)


def _boolean(op: str, a: ObjectBase, b: ObjectBase, id: str, **kwargs: Any) -> Path:
    (ra, rule_a), (rb, rule_b) = _region_of(a), _region_of(b)
    rings = pm.boolean(op, ra, rb, rule_a, rule_b)
    kwargs.setdefault("style", dict(a._spec_fields["style"]))
    subs = [np.round(pm.polyline(r), 9).tolist() for r in rings]
    return Path(id, subpaths=subs, **kwargs)


def union(a: ObjectBase, b: ObjectBase, *, id: str, **kwargs: Any) -> Path:
    """A ``Path`` covering what ``a`` or ``b`` covers (each as its fill rule says). The
    shapes are measured once, when this is called: the result is a plain path. It takes
    ``a``'s style unless given one. Needs shapely (the ``boolean`` extra)."""
    return _boolean("union", a, b, id, **kwargs)


def intersection(a: ObjectBase, b: ObjectBase, *, id: str, **kwargs: Any) -> Path:
    """A ``Path`` covering what both ``a`` and ``b`` cover (see ``union``)."""
    return _boolean("intersection", a, b, id, **kwargs)


def difference(a: ObjectBase, b: ObjectBase, *, id: str, **kwargs: Any) -> Path:
    """A ``Path`` covering what ``a`` covers and ``b`` does not (see ``union``)."""
    return _boolean("difference", a, b, id, **kwargs)


def exclusion(a: ObjectBase, b: ObjectBase, *, id: str, **kwargs: Any) -> Path:
    """A ``Path`` covering what exactly one of ``a`` and ``b`` covers (see ``union``)."""
    return _boolean("exclusion", a, b, id, **kwargs)
