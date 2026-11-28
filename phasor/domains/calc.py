"""Calculus objects (SPEC "Domain modules: calc / analysis").

    from phasor import calc

    s.add(calc.RiemannSum(id="area", function="x**2", a=0, b=3, n=expr("2 + 2 * floor(t)")))
    s.add(calc.Taylor(id="approx", function="sin(x)", center=0, order=5))
    s.add(calc.VectorField(id="flow", p="-y", q="x"))
    s.add(calc.Periodize(id="poisson", f="exp(-(x-1.3)^2/9.68)", period=10, fold=expr("t / 4")))

The graph lives in the plane with x along the world x axis and y = f(x) up; ``pos`` moves the
origin. Numerics are in ``calc_math``.
"""

from __future__ import annotations

import functools
import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import calc_math as cm
from phasor.domains.fourier import _ydir
from phasor.domains.strokes import disc_batch, polyline, segments, with_alpha

_FN_CACHE: dict[str, cm.RealFunction] = {}


def _fn(source: str) -> cm.RealFunction:
    if source not in _FN_CACHE:
        _FN_CACHE[source] = cm.RealFunction.parse(source)
    return _FN_CACHE[source]


def _check_function(v: str) -> str:
    try:
        _fn(v.strip())
    except cm.FunctionError as e:
        raise ValueError(str(e)) from None
    return v


def _curve(f: cm.RealFunction, x0: float, x1: float, pos, samples=400, ymax=20.0):
    """The graph as separate runs: breaks at NaN, infinity and huge values (a pole)."""
    xs = np.linspace(x0, x1, samples)
    ys = f(xs)
    ok = np.isfinite(ys) & (np.abs(ys) < ymax)
    runs, start = [], None
    for i, good in enumerate([*ok, False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= 2:
                runs.append(
                    np.stack(
                        [xs[start:i] + pos[0], ys[start:i] + pos[1], np.full(i - start, pos[2])],
                        axis=1,
                    )
                )
            start = None
    return runs


# ---------------------------------------------------------------- Graph


@register
class Graph(ObjectBase):
    """The graph of y = f(x), with breaks where the function is undefined or has a pole."""

    type: ClassVar[str] = "graph"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        function: str = "sin(x)"
        xmin: float = -5.0
        xmax: float = 5.0
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

    def _runs(self, p):
        return _curve(_fn(p.function.strip()), p.xmin, p.xmax, p.pos)

    def path_points(self, state: SceneState) -> np.ndarray | None:
        """The longest unbroken run of the graph (what a ``Tracer`` travels along)."""
        runs = self._runs(self.me(state).params)
        return max(runs, key=len) if runs else None

    def world_bounds(self, state: SceneState) -> Bounds:
        runs = self._runs(self.me(state).params)
        return Bounds.of_points(np.vstack(runs) if runs else [self.me(state).params.pos], pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        runs = self._runs(self.me(state).params)
        return {"curve": np.vstack(runs)} if runs else {}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _fn(p.function.strip())
        return {
            "kind": "graph",
            "function": p.function,
            "latex": f.latex(),
            "breaks": max(len(self._runs(p)) - 1, 0),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        for run in self._runs(me.params):
            polyline(
                ctx,
                cam,
                run,
                w,
                color=with_alpha(me.style["color"], me.params.opacity),
                glow=me.style["glow"],
            )


# ---------------------------------------------------------------- RiemannSum


@register
class RiemannSum(ObjectBase):
    """The area under y = f(x) on [a, b] approximated by ``n`` pieces (``rule``: left, right,
    mid, trapezoid, upper, lower): the rectangles, the curve, and in the semantics the sum, the
    exact integral and the error. Animate ``n`` upward to watch it converge."""

    type: ClassVar[str] = "riemann_sum"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "curve": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        function: str = "x**2"
        a: float = 0.0
        b: float = 3.0
        n: float = Field(6.0, ge=1, le=400)  # a number (so it can be animated); rounded
        rule: Literal["left", "right", "mid", "trapezoid", "upper", "lower"] = "left"
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

        @field_validator("b")
        @classmethod
        def _ab(cls, v, info):
            if v <= info.data.get("a", 0.0):
                raise ValueError("b must be greater than a")
            return v

    def _sum(self, p):
        return cm.riemann_sum(_fn(p.function.strip()), p.a, p.b, max(1, round(p.n)), p.rule)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        r = self._sum(p)["rects"]
        y = np.nan_to_num(r[:, 4])
        top, bot = max(float(y.max()), 0.0), min(float(y.min()), 0.0)
        return Bounds(
            (p.a + p.pos[0], bot + p.pos[1], p.pos[2]), (p.b + p.pos[0], top + p.pos[1], p.pos[2])
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        r = self._sum(p)["rects"]
        return {
            "tops": np.stack(
                [
                    (r[:, 0] + r[:, 1]) / 2 + p.pos[0],
                    np.nan_to_num(r[:, 4]) + p.pos[1],
                    np.full(len(r), p.pos[2]),
                ],
                axis=1,
            )
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _fn(p.function.strip())
        s = self._sum(p)
        exact = cm.exact_integral(f, p.a, p.b)
        return {
            "kind": "riemann_sum",
            "rule": p.rule,
            "pieces": len(s["rects"]),
            "sum": round(s["value"], 6),
            "exact": round(exact, 6),
            "error": round(abs(s["value"] - exact), 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        o = np.asarray(p.pos, dtype=np.float64)
        fill = with_alpha(me.style["color"], p.opacity * 0.35)
        edge = with_alpha(me.style["color"], p.opacity)
        for x0, x1, fl, fr, h in self._sum(p)["rects"]:
            if not math.isfinite(h):
                continue
            if p.rule == "trapezoid":
                quad = np.array([[x0, 0, 0], [x1, 0, 0], [x1, fr, 0], [x0, fl, 0]]) + o
            else:
                quad = np.array([[x0, 0, 0], [x1, 0, 0], [x1, h, 0], [x0, h, 0]]) + o
            ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=fill)
            polyline(ctx, cam, np.vstack([quad, quad[:1]]), w * 0.6, color=edge)
        for run in _curve(_fn(p.function.strip()), p.a, p.b, o):
            polyline(
                ctx,
                cam,
                run,
                w * 1.4,
                color=with_alpha(me.style["curve"], p.opacity),
                glow=me.style["glow"],
            )


# ---------------------------------------------------------------- Taylor


@register
class Taylor(ObjectBase):
    """f(x) and its Taylor polynomial of degree ``order`` about ``center``: more terms hug the
    curve over a wider window. ``order`` is a number so it can be animated."""

    type: ClassVar[str] = "taylor"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "poly": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        function: str = "sin(x)"
        center: float = 0.0
        order: float = Field(3.0, ge=0, le=30)
        xmin: float = -6.0
        xmax: float = 6.0
        ymax: float = Field(3.0, gt=0)  # the polynomial is clipped above this height
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

    def _coeffs(self, p):
        return cm.taylor_coefficients(_fn(p.function.strip()), p.center, round(p.order))

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds(
            (p.xmin + p.pos[0], -p.ymax + p.pos[1], p.pos[2]),
            (p.xmax + p.pos[0], p.ymax + p.pos[1], p.pos[2]),
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {
            "center": np.array(
                [
                    p.center + p.pos[0],
                    float(_fn(p.function.strip())(np.array(p.center))) + p.pos[1],
                    p.pos[2],
                ]
            )
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _fn(p.function.strip())
        c = self._coeffs(p)
        xs = np.linspace(max(p.xmin, p.center - 1.0), min(p.xmax, p.center + 1.0), 81)
        err = float(np.nanmax(np.abs(cm.taylor_polynomial(c, p.center)(xs) - f(xs))))
        return {
            "kind": "taylor",
            "function": p.function,
            "order": len(c) - 1,
            "center": p.center,
            "coefficients": [round(v, 8) for v in c],
            "max_error_within_1": round(err, 8),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        f = _fn(p.function.strip())
        for run in _curve(f, p.xmin, p.xmax, p.pos, ymax=p.ymax):
            polyline(ctx, cam, run, w, color=with_alpha(me.style["color"], p.opacity))
        poly = cm.taylor_polynomial(self._coeffs(p), p.center)
        xs = np.linspace(p.xmin, p.xmax, 400)
        ys = poly(xs)
        keep = np.abs(ys) <= p.ymax
        pts = np.stack([xs + p.pos[0], ys + p.pos[1], np.full(len(xs), p.pos[2])], axis=1)
        start = None
        for i, good in enumerate([*keep, False]):
            if good and start is None:
                start = i
            elif not good and start is not None:
                if i - start >= 2:
                    polyline(
                        ctx,
                        cam,
                        pts[start:i],
                        w * 1.3,
                        color=with_alpha(me.style["poly"], p.opacity),
                        glow=me.style["glow"],
                    )
                start = None


# ---------------------------------------------------------------- VectorField


@register
class VectorField(ObjectBase):
    """A planar field F = (P(x, y), Q(x, y)) as a grid of arrows (length scaled to the grid
    spacing). Divergence and curl are in the semantics.

    Presets: ``dense`` (many small thin arrows, lengths eased so weak ones still show, color
    saying how strong: the ``magnitude`` colormap), ``bold`` (few thick glowing arrows)."""

    type: ClassVar[str] = "vector_field"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "sequential"),
    }
    Presets: ClassVar = {
        "dense": {"params": {"spacing": 0.32, "length_by": "soft", "color_by_size": True},
                  "style": {"colormap": "magnitude", "stroke": "hairline"}},
        "bold": {"params": {"spacing": 1.0}, "style": {"stroke": "regular", "glow": "soft"}},
    }  # fmt: skip

    class Params(ParamsBase):
        p: str = "-y"
        q: str = "x"
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        extent: tuple[float, float] = (8.0, 5.0)
        spacing: float = Field(0.8, gt=0.1)
        color_by_size: bool = False  # color each arrow by its size through the colormap
        length_by: Literal["size", "soft"] = "size"  # "soft": weak arrows still show

        @field_validator("p", "q")
        @classmethod
        def _pq(cls, v):
            try:
                cm.parse_expression(v, ("x", "y"))
            except cm.FunctionError as e:
                raise ValueError(str(e)) from None
            return v

    def _field(self, p):
        return cm.VectorField.parse(p.p, p.q)

    def _samples(self, p):
        xs = np.arange(-p.extent[0] / 2, p.extent[0] / 2 + 1e-9, p.spacing)
        ys = np.arange(-p.extent[1] / 2, p.extent[1] / 2 + 1e-9, p.spacing)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return X, Y

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        return Bounds(
            tuple(c - np.array([p.extent[0] / 2, p.extent[1] / 2, 0.0])),
            tuple(c + np.array([p.extent[0] / 2, p.extent[1] / 2, 0.0])),
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        import sympy as sp

        p = self.me(state).params
        f = self._field(p)
        return {
            "kind": "vector_field",
            "p": p.p,
            "q": p.q,
            "divergence": sp.sstr(f.divergence()),
            "curl": sp.sstr(f.curl()),
            "arrows": int(self._samples(p)[0].size),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        from phasor.core.color import ramp
        from phasor.domains.geometry_math import normalize01, sample_ramp
        from phasor.domains.strokes import field_arrows

        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        X, Y = self._samples(p)
        U, V = self._field(p)(X, Y)
        o = np.asarray(p.pos, dtype=np.float64)
        pts = o + np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
        vec = np.c_[np.ravel(U), np.ravel(V), np.zeros(X.size)]
        w = me.style["stroke"] * float(cam.units_per_height(o)) * 2.5
        if p.color_by_size:
            mag = np.linalg.norm(vec, axis=1)
            rgb = sample_ramp(ramp(me.style["colormap"]), normalize01(mag, 0.0, None))
            cols = np.c_[rgb, np.full(len(rgb), p.opacity)]
        else:
            cols = with_alpha(me.style["color"], p.opacity)
        field_arrows(ctx, cam, pts, vec, p.spacing, width=w, colors=cols,
                     glow=me.style["glow"], length_by=p.length_by)  # fmt: skip


# ---------------------------------------------------------------- Limit (epsilon-delta)


@register
class Limit(ObjectBase):
    """The epsilon-delta picture of lim f(x) as x -> a: the graph, a band of half-height
    ``epsilon`` around the limit L, and the widest band of half-width delta around a whose
    graph stays inside it. Shrink ``epsilon`` and delta shrinks with it. The point (a, L) is
    drawn hollow: f need not be defined there."""

    type: ClassVar[str] = "limit"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "band": ("emphasis", "highlight"),
        "delta": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        function: str = "sin(x)/x"
        a: float = 0.0
        epsilon: float = Field(0.3, gt=0)
        xmin: float = -5.0
        xmax: float = 5.0
        pos: Vec3 = (0.0, 0.0, 0.0)
        bands: float = Field(0.14, ge=0, le=1)  # opacity of the epsilon and delta bands

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

        @model_validator(mode="after")
        def _has_limit(self):
            if cm.limit_at(_fn(self.function.strip()), self.a) is None:
                raise ValueError(f"{self.function} has no finite limit at x = {self.a:g}")
            return self

    def _numbers(self, p):
        f = _fn(p.function.strip())
        L = cm.limit_at(f, p.a)
        assert L is not None
        delta = cm.delta_for_epsilon(f, p.a, L, p.epsilon, search=(p.xmax - p.xmin) / 2)
        return f, L, delta

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        f, L, _ = self._numbers(p)
        runs = _curve(f, p.xmin, p.xmax, p.pos)
        pts = [
            *runs,
            np.array(
                [
                    [p.xmin + p.pos[0], L + p.pos[1] + p.epsilon, p.pos[2]],
                    [p.xmax + p.pos[0], L + p.pos[1] - p.epsilon, p.pos[2]],
                ]
            ),
        ]
        return Bounds.of_points(np.vstack(pts), pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        _, L, _ = self._numbers(p)
        return {"point": np.array([p.a + p.pos[0], L + p.pos[1], p.pos[2]])}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f, L, delta = self._numbers(p)
        x = np.concatenate(
            [p.a - np.linspace(1e-6, delta, 400), p.a + np.linspace(1e-6, delta, 400)]
        )
        worst = float(np.nanmax(np.abs(f(x) - L))) if delta > 0 else 0.0
        return {
            "kind": "limit",
            "function": p.function,
            "a": p.a,
            "limit": round(L, 9),
            "epsilon": round(p.epsilon, 9),
            "delta": round(delta, 6),
            "holds": bool(delta > 0 and worst < p.epsilon),
            "worst_inside": round(worst, 9),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        from phasor.domains.strokes import circle

        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        f, L, delta = self._numbers(p)
        ox, oy, oz = p.pos
        w = me.style["stroke"] * float(cam.units_per_height())
        y0, y1 = oy + L - p.epsilon, oy + L + p.epsilon
        x0, x1 = ox + p.xmin, ox + p.xmax
        bb = Bounds.of_points(np.vstack(_curve(f, p.xmin, p.xmax, p.pos) or [np.array([p.pos])]))
        top, bottom = max(bb.hi[1], y1), min(bb.lo[1], y0)

        def rect(xa, xb, ya, yb, color):
            quad = np.array([[xa, ya, oz], [xb, ya, oz], [xb, yb, oz], [xa, yb, oz]])
            ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=color)

        if p.bands > 0:
            rect(x0, x1, y0, y1, with_alpha(me.style["band"], p.opacity * p.bands))
            if delta > 0:
                rect(ox + p.a - delta, ox + p.a + delta, bottom, top,
                     with_alpha(me.style["delta"], p.opacity * p.bands * 0.6))  # fmt: skip
        for run in _curve(f, p.xmin, p.xmax, p.pos):
            polyline(ctx, cam, run, w, color=with_alpha(me.style["color"], p.opacity),
                     glow=me.style["glow"])  # fmt: skip
        hole = np.array([ox + p.a, oy + L, oz])
        r = 0.1 * float(cam.units_per_height(hole)) / 4
        circle(ctx, cam, hole, r, w, color=with_alpha(me.style["band"], p.opacity))


# ---------------------------------------------------------------- LoopIntegral (flux, Green)


@register
class LoopIntegral(ObjectBase):
    """A circle laid over a vector field, and a line integral around it, traced as
    ``progress`` goes 0 -> 1. ``measure="flux"``: arrows show F . n, what crosses the curve
    (outward bright, inward dim); the total equals the integral of the divergence inside.
    ``measure="circulation"``: arrows show F . T, what flows along it; the total equals the
    integral of the curl inside (Green's theorem). Field coordinates are relative to ``pos``,
    as for ``VectorField``, so the two line up when they share ``pos``."""

    type: ClassVar[str] = "loop_integral"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "positive": ("emphasis", "highlight"),
        "negative": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(ParamsBase):
        p: str = "x"
        q: str = "y"
        measure: Literal["flux", "circulation"] = "flux"
        center: tuple[float, float] = (0.0, 0.0)  # in field coordinates
        radius: float = Field(1.5, gt=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the field's origin is
        progress: float = Field(1.0, ge=0, le=1)
        arrows: int = Field(16, ge=0, le=128)

        @field_validator("p", "q")
        @classmethod
        def _pq(cls, v):
            try:
                cm.parse_expression(v, ("x", "y"))
            except cm.FunctionError as e:
                raise ValueError(str(e)) from None
            return v

    def _ring(self, p, t):
        o = np.asarray(p.pos, dtype=np.float64)
        return np.stack(
            [
                o[0] + p.center[0] + p.radius * np.cos(t),
                o[1] + p.center[1] + p.radius * np.sin(t),
                np.full_like(t, o[2]),
            ],
            1,
        )

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._ring(p, np.linspace(0, 2 * np.pi, 64)), pad=0.6)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        o = np.asarray(p.pos, dtype=np.float64)
        return {
            "center": o + np.array([*p.center, 0.0]),
            "curve": self._ring(p, np.linspace(0, 2 * np.pi, 64)),
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        import sympy as sp

        p = self.me(state).params
        f = cm.VectorField.parse(p.p, p.q)
        inside = f.divergence() if p.measure == "flux" else f.curl()
        total = cm.loop_integral(f, p.center, p.radius, p.measure)
        area = cm.disc_integral(f, inside, p.center, p.radius)
        return {
            "kind": "loop_integral",
            "measure": p.measure,
            "so_far": round(cm.loop_integral(f, p.center, p.radius, p.measure, upto=p.progress), 6),
            "total": round(total, 6),
            "inside": sp.sstr(inside),
            "area_integral": round(area, 6),
            "theorem": (
                "divergence theorem: flux = integral of div F"
                if p.measure == "flux"
                else "Green: circulation = integral of curl F"
            ),
            "theorem_holds": bool(abs(total - area) < 1e-6 * max(1.0, abs(area))),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        from phasor.domains.strokes import arrow

        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        f = cm.VectorField.parse(p.p, p.q)
        o = np.asarray(p.pos, dtype=np.float64)
        w = me.style["stroke"] * float(cam.units_per_height(o))
        full = self._ring(p, np.linspace(0, 2 * np.pi, 129))
        polyline(ctx, cam, full, w * 0.5, color=with_alpha(me.style["negative"], p.opacity * 0.5))
        if p.progress > 0:
            done = self._ring(
                p, np.linspace(0, 2 * np.pi * p.progress, max(2, int(128 * p.progress) + 2))
            )
            polyline(ctx, cam, done, w, color=with_alpha(me.style["color"], p.opacity),
                     glow=me.style["glow"])  # fmt: skip
        if p.arrows == 0:
            return
        t = (np.arange(p.arrows) + 0.5) / p.arrows * 2 * np.pi
        t = t[t <= 2 * np.pi * p.progress + 1e-9]
        if not len(t):
            return
        c, s = np.cos(t), np.sin(t)
        P, Q = f(p.center[0] + p.radius * c, p.center[1] + p.radius * s)
        normal = p.measure == "flux"
        comp = P * c + Q * s if normal else -P * s + Q * c
        top = float(np.nanmax(np.abs(comp))) if np.isfinite(comp).any() else 0.0
        if top <= 0:
            return
        base = self._ring(p, t)
        dirs = np.stack([c, s], 1) if normal else np.stack([-s, c], 1)
        for b, d, v in zip(base, dirs, comp, strict=True):
            if not np.isfinite(v) or abs(v) < 1e-9 * top:
                continue
            tip = b + np.array([*d, 0.0]) * (v / top) * p.radius * 0.45
            role = "positive" if v > 0 else "negative"
            arrow(ctx, cam, b, tip, w * 0.8, with_alpha(me.style[role], p.opacity))


# ---------------------------------------------------------------- Periodize (Poisson summation)


@functools.lru_cache(maxsize=16)
def _poisson(source: str, period: float, copies: tuple[int, int], cell: float, k: int):
    """What the semantics and the spectrum panel need, once per (function, period, copies, cell,
    harmonics): k, the coefficients c_k of g, fhat(k/P)/P, and the |fhat| curve for the plot."""
    f = _fn(source)
    ks = np.arange(-k, k + 1)
    ck = cm.fourier_coefficients(f, period, copies, cell, ks)
    fk = cm.fourier_transform(f, period, copies, cell, ks / period) / period
    xi = np.linspace(-(k + 0.6) / period, (k + 0.6) / period, 25 * (2 * k + 2) + 1)
    curve = np.abs(cm.fourier_transform(f, period, copies, cell, xi))
    for a in (ks, ck, fk, xi, curve):
        a.flags.writeable = False
    return ks, ck, fk, xi, curve


def _pairs(z) -> list[list[float]]:
    return [[round(float(c.real), 9), round(float(c.imag), 9)] for c in z]


@register
class Periodize(ObjectBase):
    """Poisson summation in one picture. Top: one bump f, its copies every ``period`` (faint),
    their sum g (bright) and the pieces of f, cut at the edges of the cell, each slid into the
    cell where they add up to g: ``fold`` 0 -> 1 slides them back out until they draw f exactly.
    Below (``spectrum``): |fhat| with a stem at every k/P; a stem is |c_k| * P, the size of the
    k-th Fourier coefficient of g. The semantics give c_k (by a sum over the cell) and
    fhat(k/P)/P (by an integral of f over the line) side by side: they are equal.

    ``copies=(lo, hi)`` are the shifts j in range(lo, hi) (the end is left out), so (-2, 3) is
    five copies; the pieces cover those cells (the ``window``). The figure fits ``extent``
    (width, height) around ``pos`` in the plane z = pos[2], or on the overlay layer."""

    type: ClassVar[str] = "periodize"
    Style: ClassVar = {
        "copy": ("emphasis", "muted"),
        "piece": ("emphasis", "primary"),
        "piece2": ("emphasis", "secondary"),
        "color": ("emphasis", "highlight"),  # g, and the spectrum's curve
        "guide": ("emphasis", "muted"),
        "text": ("emphasis", "muted"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        f: str = "exp(-(x-1.3)^2/9.68)"
        period: float = Field(10.0, gt=0)
        copies: tuple[int, int] = (-2, 3)
        cell: float = 0.0  # the cell is [cell, cell + period]
        fold: float = Field(1.0, ge=0, le=1)  # 0: pieces inside the cell (sum g); 1: they are f
        spectrum: bool = True
        harmonics: int = Field(4, ge=0, le=12)  # stems at k/P for k = -harmonics..harmonics
        show_copies: bool = True
        show_pieces: bool = True
        show_g: bool = True
        x_label: str = "x"
        xi_label: str = "k / P"
        labels: bool = True
        pos: Vec3 = (0.0, 0.0, 0.0)  # the figure's centre
        extent: tuple[float, float] = (12.0, 7.0)

        @field_validator("f")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

        @field_validator("copies")
        @classmethod
        def _copies(cls, v):
            if not 1 <= v[1] - v[0] <= 15:
                raise ValueError("copies=(lo, hi) takes between 1 and 15 shifts")
            return v

    # -- layout. A panel is (u0, u1, v0, v1): shares of the extent, v up from the bottom.
    @staticmethod
    def _panels(p):
        if p.spectrum:
            return (0.07, 0.97, 0.5, 0.96), (0.07, 0.97, 0.1, 0.34)
        return (0.07, 0.97, 0.12, 0.96), None

    @staticmethod
    def _plane(me, state):
        """``at(u, v)``: shares of the extent (v up on screen) -> world points."""
        p = me.params
        sign = _ydir(me, state)
        x0, y0, z = p.pos[0] - p.extent[0] / 2, p.pos[1], p.pos[2]

        def at(u, v):
            u, v = np.broadcast_arrays(
                np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
            )
            return np.stack([x0 + u * p.extent[0], y0 + sign * (v - 0.5) * p.extent[1],
                             np.full(u.shape, z)], axis=-1)  # fmt: skip

        return at

    @staticmethod
    def _view(p):
        """The spatial panel's data window: ((x0, x1), (y0, y1))."""
        f = _fn(p.f.strip())
        a, b = cm.periodize_window(p.period, p.copies, p.cell)
        xs = np.linspace(a, b, 600)
        g, y = cm.periodic_sum(f, p.period, p.copies, xs), np.nan_to_num(f(xs))
        top = max(float(g.max()), float(y.max()), 1e-9)
        return (a, b), (min(float(g.min()), float(y.min()), 0.0) * 1.15, top * 1.15)

    def _data_to_uv(self, p):
        """(x, y) data -> (u, v) shares, for the spatial panel."""
        pan, _ = self._panels(p)
        (a, b), (ylo, yhi) = self._view(p)

        def uv(x, y):
            x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
            return (pan[0] + (x - a) / (b - a) * (pan[1] - pan[0]),
                    pan[2] + (y - ylo) / (yhi - ylo) * (pan[3] - pan[2]))  # fmt: skip

        return uv

    def _spectrum_to_uv(self, p):
        """(xi, |fhat|) data -> (u, v) shares, for the spectrum panel."""
        _, spec = self._panels(p)
        _, _, _, xi, curve = _poisson(p.f.strip(), p.period, p.copies, p.cell, p.harmonics)
        top = max(float(curve.max()), 1e-12)

        def uv(x, y):
            x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
            return (spec[0] + (x - xi[0]) / (xi[-1] - xi[0]) * (spec[1] - spec[0]),
                    spec[2] + y / top * (spec[3] - spec[2]))  # fmt: skip

        return uv

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x - p.extent[0] / 2, y - p.extent[1] / 2, z),
                      (x + p.extent[0] / 2, y + p.extent[1] / 2, z))  # fmt: skip

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        me = self.me(state)
        p = me.params
        at = self._plane(me, state)
        uv = self._data_to_uv(p)
        mid = p.cell + p.period / 2
        gmid = cm.periodic_sum(_fn(p.f.strip()), p.period, p.copies, np.array([mid]))
        out = {
            "cell": at(*uv([p.cell, p.cell + p.period], [0.0, 0.0])),
            "g_mid": at(*uv([mid], gmid)),
        }
        if p.spectrum:
            ks, _, fk, _, _ = _poisson(p.f.strip(), p.period, p.copies, p.cell, p.harmonics)
            out["stems"] = at(*self._spectrum_to_uv(p)(ks / p.period, np.abs(fk) * p.period))
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _fn(p.f.strip())
        ks, ck, fk, _, _ = _poisson(p.f.strip(), p.period, p.copies, p.cell, p.harmonics)
        a, b = cm.periodize_window(p.period, p.copies, p.cell)
        xs = p.cell + np.arange(400) * p.period / 400
        g = cm.periodic_sum(f, p.period, p.copies, xs)
        pieces = cm.piece_range(p.copies)
        slid = sum(np.nan_to_num(f(xs + m * p.period)) for m in pieces)  # the pieces at fold = 0
        series = np.real(np.exp(2j * np.pi * np.outer(xs, ks) / p.period) @ ck)
        return {
            "kind": "periodize",
            "function": p.f,
            "period": p.period,
            "copies": list(p.copies),
            "pieces": len(pieces),
            "window": [a, b],
            "fold": round(p.fold, 6),
            "k": [int(k) for k in ks],
            "c_k": _pairs(ck),
            "fhat_over_P": _pairs(fk),
            "max_mismatch": float(np.max(np.abs(ck - fk))),  # c_k against fhat(k/P)/P
            "folded_sum_error": float(np.max(np.abs(slid - g))),  # pieces slid in, against g
            "series_error": float(np.max(np.abs(series - g))),  # g against its harmonics
            "tail": round(cm.tail_mass(f, p.period, p.copies, p.cell), 9),
            "spectrum": p.spectrum,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height(np.asarray(p.pos, dtype=np.float64)))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        at = self._plane(me, state)
        sg = _ydir(me, state)
        f = _fn(p.f.strip())
        (a, b), (ylo, yhi) = self._view(p)
        text = with_alpha(me.style["text"], p.opacity)
        size = me.style["size"] * unit
        faint = with_alpha(me.style["guide"], p.opacity * 0.5)
        tick = size * 0.35

        def off(dx, dy):  # an offset on screen: dy up
            return np.array([dx, dy * sg, 0.0])

        def label(s, q, anchor="center"):
            if p.labels:
                ctx.text(s, q, size, color=text, anchor=anchor, glow=1.0)

        def axes(to_world, x_ticks, y_ticks, x_end, y_end):
            """The two axes through the data origin of a panel, with ticks and number labels."""
            polyline(ctx, cam, to_world(x_end[0], 0.0), w * 0.6, color=faint)
            polyline(ctx, cam, to_world(y_end[0], y_end[1]), w * 0.6, color=faint)
            for x, name in x_ticks:
                q = to_world(x, 0.0)
                polyline(ctx, cam, [q - off(0, tick), q + off(0, tick)], w * 0.5, color=faint)
                if name:
                    label(name, q + off(0, -1.6 * size))
            for y in y_ticks:
                q = to_world(y_end[0][0], y)
                polyline(ctx, cam, [q - off(tick, 0), q], w * 0.5, color=faint)
                label(f"{y:g}", q - off(1.2 * size, 0), "right")

        # --- the spatial panel
        uv = self._data_to_uv(p)

        def px(x, y):
            return at(*uv(x, y))

        ctx.triangles(
            px([p.cell, p.cell + p.period, p.cell + p.period, p.cell], [ylo, ylo, yhi, yhi]),
            np.array([[0, 1, 2], [0, 2, 3]]),
            color=with_alpha(me.style["guide"], p.opacity * 0.12),
        )  # the cell, shaded
        edges = np.arange(a, b + p.period / 2, p.period)
        axes(px, [(float(x), f"{x:g}") for x in edges], cm.nice_ticks(max(ylo, 0.0), yhi, 3),
             ([a, b], 0.0), ([a, a], [ylo, yhi]))  # fmt: skip
        label(p.x_label, px(b, 0.0) + off(0, 1.2 * size), "right")

        xs = np.linspace(a, b, 500)
        if p.show_copies:
            for j in cm.copy_range(p.copies):
                ys = np.nan_to_num(f(xs - j * p.period))
                polyline(ctx, cam, px(xs, ys), w * 0.7,
                         color=with_alpha(me.style["copy"], p.opacity * 0.7))  # fmt: skip
        if p.show_g:
            g = cm.periodic_sum(f, p.period, p.copies, xs)
            polyline(ctx, cam, px(xs, g), w * 1.5,
                     color=with_alpha(me.style["color"], p.opacity), glow=glow)  # fmt: skip
        if p.show_pieces:
            for i, m in enumerate(cm.piece_range(p.copies)):
                x, y = cm.fold_piece(f, p.period, m, p.cell, p.fold)
                col = me.style["piece" if i % 2 == 0 else "piece2"]
                polyline(ctx, cam, px(x, y), w * 1.2, color=with_alpha(col, p.opacity), glow=glow)
        if p.labels and p.show_g:
            ctx.text("g", px(p.cell + p.period / 2, yhi) + off(0, 0.9 * size), size,
                     color=with_alpha(me.style["color"], p.opacity), glow=1.0)  # fmt: skip

        if not p.spectrum:
            return
        # --- the spectrum panel: |fhat| and a stem at every k/P
        ks, _, fk, xi, curve = _poisson(p.f.strip(), p.period, p.copies, p.cell, p.harmonics)
        suv = self._spectrum_to_uv(p)

        def sx(x, y):
            return at(*suv(x, y))

        step = max(1, math.ceil((2 * p.harmonics + 1) / 9))
        top = float(curve.max())
        axes(sx, [(float(k / p.period), f"{k / p.period:g}" if k % step == 0 else "") for k in ks],
             [], ([xi[0], xi[-1]], 0.0), ([xi[0], xi[0]], [0.0, top]))  # fmt: skip
        label(p.xi_label, sx(xi[-1], 0.0) + off(0, 1.2 * size), "right")
        label("|f^|", sx(xi[0], top) - off(1.0 * size, 0), "right")
        col = with_alpha(me.style["color"], p.opacity)
        polyline(ctx, cam, sx(xi, curve), w * 1.3, color=col, glow=glow)
        stem = with_alpha(me.style["piece"], p.opacity)
        heads = sx(ks / p.period, np.abs(fk) * p.period)
        feet = sx(ks / p.period, np.zeros(len(ks)))
        segments(ctx, cam, np.stack([feet, heads], axis=1), me.style["stroke"] * 0.9,
                 color=stem, glow=glow)  # fmt: skip
        disc_batch(ctx, cam, heads, unit * 0.007, colors=stem, glow=glow)
