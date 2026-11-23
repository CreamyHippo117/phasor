"""Vector paths (Milestone 16): ``Path`` and the base every outline shape is drawn by.

A path is subpaths of cubic Béziers (``domains/path_math.py``). It is drawn as

- a **fill** (style ``fill``: none, soft, solid; its opacity comes from the theme) in
  ``fill_color``, under the even-odd or non-zero rule, holes included, triangulated by
  earcut and drawn through ``ctx.triangles``. A fill is plain paint: it never glows, so a
  color below the theme's bloom knee shows exactly as written;
- a **stroke** in ``color``, ``stroke`` wide, glowing by ``glow``, cut into dashes or dots by
  ``dash`` (by arc length from each subpath's start, so dashes do not crawl as a path
  grows), with optional arrow ``tips`` at its open ends.

Curves are flattened adaptively by their size on screen: within ``FLATNESS`` of the frame
height. ``progress`` traces the stroke (the ``trace`` verb) and fades the fill in with it.

A shape (``domains/shapes.py``) is a ``PathBase`` with its own parameters and a ``bezier``
method that turns them into subpaths; drawing, bounds, features and semantics are shared.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core import record
from phasor.core.ids import is_valid_id
from phasor.core.objects import DrawContext, ObjectBase, ParamsBase, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import morph_math as mm
from phasor.domains import path_math as pm
from phasor.domains.strokes import arrow_head, disc_batch, polyline, with_alpha

FLATNESS = 2.5e-4  # frame heights a drawn curve may stray from the true one (0.27 px at 1080p)
MEASURE = 1e-5  # share of a path's size: the tolerance its semantics are measured at
DASH_ON = 0.55  # share of a dash period that is drawn
DOT = 0.7  # a dot's radius, in stroke widths
TIP = 7.0  # an arrow tip's length, in stroke widths (as ``Arrow``'s head)
TIP_MAX = 0.3  # ... and at most this share of its subpath's length
# Parameters that change how a path is drawn but not its geometry (the cache key leaves
# them out, so a fade does not flatten and triangulate again every frame).
NOT_GEOMETRY = frozenset({"opacity", "dim", "z_index", "flash", "progress", "fill_opacity",
                          "stroke_opacity", "morph", "morph_from", "morph_kind"})  # fmt: skip
CACHE_BYTES = 256 * 2**20  # geometry kept between frames (flattened curves, fills, measures)
MAX_DASHES = 2000  # dashes (or dots) per path: past it they are drawn longer (lint dash_limit)
MAX_COORD = 1e9  # world units: no parameter of a path may be larger (it could not be drawn)

Tips = Literal["none", "start", "end", "both"]

STYLE: dict[str, tuple[str, str]] = {
    "color": ("emphasis", "primary"),
    "stroke": ("stroke", "regular"),
    "glow": ("glow", "none"),
    "fill": ("fill", "none"),
    "fill_color": ("emphasis", "primary"),
    "dash": ("dash", "solid"),
}
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "outline": {},
    "filled": {"style": {"fill": "soft"}},
    "solid": {"style": {"fill": "solid"}},
    "glowing": {"style": {"glow": "soft", "fill": "soft"}},
    "dashed": {"style": {"dash": "dashed"}},
}

_CACHE: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()  # key -> (value, bytes)
_CACHE_USED = [0]
_LOCK = threading.Lock()


def _nbytes(value: Any) -> int:
    """About how much memory a cached value holds (its arrays, plus a little per entry)."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, tuple | list):
        return 64 + sum(_nbytes(v) for v in value)
    if isinstance(value, dict):
        return 64 + sum(_nbytes(v) for v in value.values())
    return 64


def _cached(key: tuple, make: Callable[[], Any]) -> Any:
    """A path's geometry, kept while it fits in ``CACHE_BYTES`` (least recently used out):
    a drawing of thousands of parts (an SVG) keeps all of them, so a new frame does not
    flatten and measure them all again, and memory stays bounded."""
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit[0]
    value = make()
    size = 256 + _nbytes(value)
    with _LOCK:
        old = _CACHE.pop(key, None)
        if old is not None:
            _CACHE_USED[0] -= old[1]
        _CACHE[key] = (value, size)
        _CACHE_USED[0] += size
        while _CACHE_USED[0] > CACHE_BYTES and len(_CACHE) > 1:
            _CACHE_USED[0] -= _CACHE.popitem(last=False)[1][1]
    return value


def _frozen(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.setflags(write=False)


MAX_ID = 200  # characters of an object id a parameter names (``morph_from``)


class MorphParams(ParamsBase):
    """Becoming this object from another (the ``transform`` and ``transform_matching``
    verbs, ``domains/morph.py``): the source's id, how far along (1: done, drawn as itself)
    and how: ``whole`` (outline into outline) or ``matching`` (matching pieces move, the
    rest fade)."""

    morph_from: str = ""
    morph: float = Field(1.0, ge=0, le=1)
    morph_kind: Literal["whole", "matching"] = "whole"

    @field_validator("morph_from")
    @classmethod
    def _source(cls, v: str) -> str:
        if v and (len(v) > MAX_ID or not is_valid_id(v)):
            raise ValueError(f"morph_from must be an object id (at most {MAX_ID} characters)")
        return v


class PathParams(MorphParams):
    """What every path and shape has, besides its geometry."""

    progress: float = Field(1.0, ge=0, le=1)  # drawn share of the stroke (the trace verb)
    fill_opacity: float = Field(1.0, ge=0, le=1)  # times the fill role's opacity
    stroke_opacity: float = Field(1.0, ge=0, le=1)  # 0: a fill with no outline

    @model_validator(mode="after")
    def _bounded(self):
        """Every number of a path is within ``MAX_COORD``: a size, a position, a point.
        Beyond that nothing can be drawn (lengths overflow, dashes never end)."""

        def big(v) -> bool:
            if isinstance(v, int | float) and not isinstance(v, bool):
                return abs(v) > MAX_COORD
            if isinstance(v, tuple | list) and v:
                try:  # points: one array check
                    return bool(np.abs(np.asarray(v, dtype=np.float64)).max() > MAX_COORD)
                except (TypeError, ValueError):  # ragged (subpaths of different lengths)
                    return any(big(x) for x in v)
            return False

        for name in type(self).model_fields:
            if big(getattr(self, name)):
                raise ValueError(f"{name}: numbers must be within +-{MAX_COORD:g}")
        return self


class PathBase(ObjectBase):
    """Draws, measures and describes a path; a subclass says what the path is (``bezier``)."""

    Style: ClassVar = STYLE
    Presets: ClassVar = PRESETS
    KIND: ClassVar[str] = "path"  # the semantics' ``kind``

    # ---------------------------------------------------------------- what the path is

    def bezier(self, p: Any) -> list[np.ndarray]:
        """The subpaths (each (1 + 3k, 3)) these parameters describe."""
        raise NotImplementedError

    def facts(self, p: Any) -> dict[str, Any]:
        """What this shape knows about itself (a radius, an exact area), for semantics."""
        return {}

    def rule(self, p: Any) -> str:
        return getattr(p, "rule", "nonzero")

    def tips(self, p: Any) -> str:
        return getattr(p, "tips", "none")

    def is_closed(self, p: Any, sub: np.ndarray) -> bool:
        """Whether a subpath is a closed loop (no tips; it has an area). A shape that is
        open by nature (an arc of a full turn, a brace of no width) says no."""
        return pm.is_closed(sub)

    def shape_features(self, p: Any) -> dict[str, np.ndarray]:
        """Named points of the shape (its ``vertices``, a ``tip``) besides the box's."""
        return {}

    # ---------------------------------------------------------------- geometry (cached)

    def _key(self, p: Any) -> tuple:
        return (type(self), tuple((k, getattr(p, k)) for k in type(p).model_fields
                                  if k not in NOT_GEOMETRY))  # fmt: skip

    def subpaths(self, p: Any) -> tuple[np.ndarray, ...]:
        """The subpaths of these parameters, read-only."""

        def make():
            subs = tuple(np.array(pm.check_subpath(s)) for s in self.bezier(p))
            _frozen(*subs)
            return subs

        return _cached(("bezier", self._key(p)), make)

    def flat(self, p: Any, tol: float) -> tuple[np.ndarray, ...]:
        """Each subpath flattened within ``tol`` world units, read-only."""
        tol = float(f"{tol:.3g}")  # a camera that moves a hair reuses the polylines

        def make():
            polys = tuple(pm.flatten(s, tol) for s in self.subpaths(p))
            _frozen(*polys)
            return polys

        return _cached(("flat", self._key(p), tol), make)

    def _fill(self, p: Any, tol: float):
        tol = float(f"{tol:.3g}")

        def make():
            pos, tris = pm.fill(self.flat(p, tol), self.rule(p))
            _frozen(pos, tris)
            return pos, tris

        return _cached(("fill", self._key(p), tol, self.rule(p)), make)

    def measure(self, p: Any) -> dict[str, Any]:
        """Length, area, closed and crossings at a tolerance relative to the path's size, so
        they do not depend on the camera."""

        def make():
            subs = self.subpaths(p)
            tol = max(pm.size_of(subs), 1e-9) * MEASURE
            polys = self.flat(p, tol)
            return {
                "closed": all(self.is_closed(p, s) for s in subs) if subs else False,
                "length": sum(pm.length(q) for q in polys),
                "area": pm.region(polys, self.rule(p)).area if polys else 0.0,
                "crossing": pm.crossings(self.flat(p, tol * 30)),
            }

        return _cached(("measure", self._key(p)), make)

    def _tol(self, state: SceneState) -> float:
        me = self.me(state)
        subs = self.subpaths(me.params)
        center = np.vstack(subs).mean(axis=0) if subs else np.zeros(3)
        return FLATNESS * float(state.camera_for(me).units_per_height(center))

    def _fill_alpha(self, me) -> float:
        p = me.params
        return float(me.style["fill"]) * p.fill_opacity * p.opacity * p.progress

    # ---------------------------------------------------------------- drawing

    def morph_shape(self, state: SceneState) -> mm.Shape:
        """Its outline for a morph (``domains/morph.py``): one group, the whole path."""
        from phasor.domains.morph import path_look

        me = self.me(state)
        p = me.params
        subs = self.subpaths(p)
        group = mm.Group(self.KIND, mm.signature(subs), tuple(range(len(subs))), path_look(me))
        cam = state.camera_for(me)
        return mm.Shape(subs, tuple(self.is_closed(p, s) for s in subs), (group,), self.rule(p),
                        right=tuple(cam.right), up=tuple(cam.up))  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        me = self.me(state)
        p = me.params
        if p.morph_from and p.morph < 1.0:  # becoming this from another (``transform``)
            from phasor.domains import morph

            items = morph.frame(state, self.id)
            if items is not None:
                morph.paint(ctx, state, me, items)
                return
        subs = self.subpaths(p)
        if not subs:
            return
        tol = self._tol(state)
        polys = self.flat(p, tol)
        alpha = self._fill_alpha(me)
        if alpha > 0:
            pos, tris = self._fill(p, tol)
            if len(tris):
                fc = me.style["fill_color"]
                # Plain paint: glow 1 always (a glow role, which life breathes, never fills).
                ctx.triangles(pos, tris, color=(fc[0], fc[1], fc[2], fc[3] * alpha), glow=1.0)
        if p.opacity * p.stroke_opacity > 0 and p.progress > 0:
            self._stroke(ctx, state, polys, [self.is_closed(p, s) for s in subs], tol / FLATNESS)

    def _stroke(self, ctx, state, polys, closed, unit: float) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        color = with_alpha(me.style["color"], p.opacity * p.stroke_opacity)
        glow = me.style["glow"]
        width = me.style["stroke"] * unit
        period = me.style["dash"] * unit
        dotted = me.roles.get("dash") == "dotted"
        tips = self.tips(p)
        lengths = [pm.length(q) for q in polys]
        left = p.progress * sum(lengths)
        if period > 0:  # at most MAX_DASHES per path: past that they are drawn longer
            period = max(period, sum(lengths) / MAX_DASHES)
        for i, (poly, total) in enumerate(zip(polys, lengths, strict=True)):
            if left <= 1e-12:
                break
            upto, left = min(left, total), left - total
            start, end = 0.0, upto
            head = min(TIP_MAX * total, TIP * width)
            if not closed[i] and len(poly) >= 2:
                if tips in ("start", "both") and i == 0 and upto > head:
                    a = pm.piece(poly, 0.0, head * 0.2)
                    self._tip(ctx, cam, poly[0], a[0] - a[-1], head, color, glow)
                    start = head * 0.8
                if tips in ("end", "both") and i == len(polys) - 1 and upto > head:
                    b = pm.piece(poly, upto - head * 0.2, upto)
                    self._tip(ctx, cam, b[-1], b[-1] - b[0], head, color, glow)
                    end = upto - head * 0.8
            if end <= start:
                continue
            if period > 0 and dotted:
                centers = pm.dots(poly, period, upto=end, start=start, limit=MAX_DASHES)
                disc_batch(ctx, cam, centers, DOT * width, colors=color, glow=glow)
            elif period > 0:
                for dash in pm.dashes(poly, period, DASH_ON, upto=end, start=start,
                                      limit=MAX_DASHES):  # fmt: skip
                    polyline(ctx, cam, dash, me.style["stroke"] * cam.units_per_height(dash),
                             color=color, glow=glow)  # fmt: skip
            else:
                line = pm.piece(poly, start, end)
                if closed[i] and start == 0.0 and end >= total and len(line) > 2:
                    # A whole loop starts and ends mid-way along its first segment, so the
                    # seam is mitered like every other corner.
                    mid = (line[0] + line[1]) / 2
                    line = np.vstack([mid, line[1:], mid])
                polyline(ctx, cam, line, me.style["stroke"] * cam.units_per_height(line),
                         color=color, glow=glow)  # fmt: skip

    @staticmethod
    def _tip(ctx, cam, at, direction, head, color, glow) -> None:
        n = float(np.linalg.norm(direction))
        if n > 1e-12:
            arrow_head(ctx, cam, at, np.asarray(direction) / n, head, color=color, glow=glow)

    def path_points(self, state: SceneState) -> np.ndarray:
        """The whole outline as one polyline, subpath after subpath (what a ``Tracer``
        travels along)."""
        polys = self.flat(self.me(state).params, self._tol(state))
        return np.vstack(polys) if polys else np.zeros((0, 3))

    # ---------------------------------------------------------------- what it is

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        """The box's compass points, ``start`` and ``end`` of what is drawn, and its shape:
        ``region`` (the outline of a filled path with one ring: layout and lint judge it as
        an area) or ``curve`` (an unfilled path of one subpath: judged as a thin stroke)."""
        me = self.me(state)
        p = me.params
        rec = record.record(self, state)
        out = record.box_features(record.derived_world_bounds(self, state), rec.right, rec.up)
        polys = pm.trim(self.flat(p, self._tol(state)), p.progress)
        if p.morph_from and p.morph < 1.0:  # mid-morph: what is drawn is not its own shape
            polys = []
        if polys:
            out["start"], out["end"] = polys[0][0], polys[-1][-1]
            if self._fill_alpha(me) > 0 and len(polys) == 1 and len(polys[0]) >= 3:
                out["region"] = polys[0]
            elif len(polys) == 1 and len(polys[0]) >= 2:
                out["curve"] = polys[0]
        out.update(self.shape_features(p))
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        me = self.me(state)
        p = me.params
        m = self.measure(p)
        out: dict[str, Any] = {
            "kind": self.KIND,
            "subpaths": len(self.subpaths(p)),
            "closed": m["closed"],
            "length": round(m["length"], 4),
            "rule": self.rule(p),
            "fill": me.roles.get("fill", "none"),
            "dash": me.roles.get("dash", "solid"),
        }
        if m["closed"] or self._fill_alpha(me) > 0:  # an open path is closed to be filled
            out["area"] = round(m["area"], 4)
        if self.tips(p) != "none":
            out["tips"] = self.tips(p)
        if p.progress < 1:  # partly traced; 0 means nothing drawn yet
            out["drawn"] = round(p.progress, 4)
        period = me.style["dash"] * self._tol(state) / FLATNESS
        if period > 0 and m["length"] / period > MAX_DASHES:
            out["dash_limit"] = int(m["length"] / period)  # lint dash_limit
        if m["crossing"] and self._fill_alpha(me) > 0:
            out["fill_crossing"] = True  # lint fill_crossing
        if m["crossing"] is None and self._fill_alpha(me) > 0:
            out["fill_unchecked"] = True  # lint fill_unchecked: too many edges to check
        if p.morph_from and p.morph < 1.0:
            from phasor.domains import morph

            out.update(morph.facts(state, me))
        out.update(self.facts(p))
        return out


def _check_subpaths(v: tuple) -> tuple:
    for k, sub in enumerate(v):
        try:
            pm.check_subpath(sub)
        except ValueError as e:
            raise ValueError(f"subpath {k}: {e}") from None
    return v


@register
class Path(PathBase):
    """Any outline: subpaths of cubic Béziers, filled and stroked. Each subpath is a list of
    points: an anchor, then (handle, handle, anchor) for each cubic; it is closed when it
    ends where it began. ``Path.through(id, points)`` makes one of straight segments."""

    type: ClassVar[str] = "path"

    class Params(PathParams):
        subpaths: tuple[tuple[Vec3, ...], ...] = ()
        rule: Literal["nonzero", "evenodd"] = "nonzero"
        tips: Tips = "none"  # arrow tips at the open ends

        @field_validator("subpaths")
        @classmethod
        def _subs(cls, v):
            return _check_subpaths(v)

    def bezier(self, p: Any) -> list[np.ndarray]:
        return [np.asarray(s, dtype=np.float64) for s in p.subpaths]

    @classmethod
    def through(cls, id: str, points, *, closed: bool = False, **kwargs: Any) -> Path:
        """A path of straight segments through ``points`` (``closed``: back to the first)."""
        sub = pm.polyline(points, closed=closed)
        return cls(id, subpaths=[sub.tolist()], **kwargs)
