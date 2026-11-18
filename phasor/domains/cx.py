"""Complex analysis (SPEC "Domain modules"): pictures of functions of z, drawn by GPU kernels.

    from phasor import Scene, cx

    s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8, 4.5)))
    s.add(cx.Fractal(id="set", kind="mandelbrot", pos=(-0.6, 0), size=(3.2, 1.8)))

Both are rectangles in the plane z = 0 whose world coordinates are the complex plane
(x = Re, y = Im; SPEC contract 4), so the camera zooms into them like into anything else.
To the id and solo passes each is one quad with one id. Their inner structure (zeros and
poles) is reported through ``features()``, projected analytically.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3

F64_BELOW = 1e-4  # view heights under this render domain coloring in float64
SUPERSAMPLE = 2  # kernel pixels per output pixel, per side, for clean contour edges


@functools.cache
def _parse(source: str):
    from phasor.render.gpu.codegen import ComplexFunction

    return ComplexFunction.parse(source)


@functools.cache
def _zeros_and_poles(source: str) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    """Zeros and poles of a rational function, with multiplicity; () for anything else."""
    import sympy as sp

    from phasor.render.gpu.codegen import Z

    expr = _parse(source).expr
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    if not (num.is_polynomial(Z) and den.is_polynomial(Z)):
        return (), ()

    def roots(p):
        poly = sp.Poly(p, Z)
        if poly.degree() < 1:
            return ()
        return tuple(complex(r) for r in poly.nroots(n=15))

    return roots(num), roots(den)


def _rect(p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lower-left corner, right edge vector, up edge vector) of a centered rectangle."""
    w, h = p.size
    corner = np.asarray(p.pos) - np.array([w / 2, h / 2, 0.0])
    return corner, np.array([w, 0.0, 0.0]), np.array([0.0, h, 0.0])


class _Plane(ObjectBase):
    """What both kernel-drawn rectangles share."""

    def world_bounds(self, state: SceneState) -> Bounds:
        corner, r, u = _rect(self.me(state).params)
        return Bounds.of_points([corner, corner + r + u])

    def _inside(self, state: SceneState, points: tuple[complex, ...]) -> np.ndarray:
        p = self.me(state).params
        corner, r, u = _rect(p)
        z = p.pos[2]
        pts = [(c.real, c.imag, z) for c in points
               if corner[0] <= c.real <= corner[0] + r[0]
               and corner[1] <= c.imag <= corner[1] + u[1]]  # fmt: skip
        return np.array(pts, dtype=np.float64).reshape(-1, 3)


@register
class DomainColoring(_Plane):
    """A function of z, shown as color: hue is the phase of f(z), and the brightness bands
    step every time |f(z)| doubles. Poles are white."""

    type: ClassVar[str] = "domain_coloring"
    Style: ClassVar = {"colormap": ("colormap", "phase")}

    class Params(ParamsBase):
        function: str = Field(min_length=1)  # e.g. "1/(z**2 + 1)"; see docs/ai/API.md
        pos: Vec3 = (0.0, 0.0, 0.0)  # center of the rectangle
        size: tuple[float, float] = (8.0, 4.5)  # width and height in world units
        contour: float = Field(0.35, ge=0, le=1)  # how dark the modulus bands get

        @field_validator("function")
        @classmethod
        def _check(cls, v: str) -> str:
            from phasor.render.gpu.codegen import FunctionError

            try:
                _parse(v.strip())
            except FunctionError as e:
                raise ValueError(str(e)) from None
            return v

        @field_validator("size")
        @classmethod
        def _positive(cls, v):
            if min(v) <= 0:
                raise ValueError("size must be positive")
            return v

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        zeros, poles = _zeros_and_poles(p.function.strip())
        return {
            "center": np.asarray(p.pos),
            "zeros": self._inside(state, zeros),
            "poles": self._inside(state, poles),
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _parse(p.function.strip())
        feats = self.features(state)
        return {
            "kind": "domain_coloring",
            "backdrop": True,
            "function": f.source,
            "latex": f.latex(),
            "zeros": [[round(float(x), 6), round(float(y), 6)] for x, y, _ in feats["zeros"]],
            "poles": [[round(float(x), 6), round(float(y), 6)] for x, y, _ in feats["poles"]],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = _rect(p)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        f = _parse(p.function.strip())

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            from phasor.render import gpu

            return gpu.render(
                "domain", pw, ph, supersample=SUPERSAMPLE, function=f,
                center=(corner[0] + center[0], corner[1] + center[1]), height=height,
                contour=p.contour, lut=lut, precision="f64" if height < F64_BELOW else "f32",
            )  # fmt: skip

        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity)


@register
class Fractal(_Plane):
    """The Mandelbrot set, or the Julia set of ``c``, colored by escape time. Points in the
    set take the theme's background color; deeper zooms switch to float64 by themselves."""

    type: ClassVar[str] = "fractal"
    Style: ClassVar = {"colormap": ("colormap", "sequential")}

    class Params(ParamsBase):
        kind: Literal["mandelbrot", "julia"] = "mandelbrot"
        c: tuple[float, float] = (-0.8, 0.156)  # the Julia constant (unused for mandelbrot)
        pos: Vec3 = (-0.6, 0.0, 0.0)
        size: tuple[float, float] = (3.2, 1.8)
        max_iter: int = Field(256, ge=1, le=100_000)
        density: float = Field(0.35, gt=0)  # color bands per sqrt(iteration)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        out: dict[str, Any] = {"kind": "fractal", "backdrop": True, "set": p.kind,
                               "max_iter": p.max_iter}  # fmt: skip
        if p.kind == "julia":
            out["c"] = list(p.c)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = _rect(p)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        inside = tuple(state.background[:3])

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            from phasor.render import gpu
            from phasor.render.gpu.bridge import to_host

            field = gpu.render(
                "escape", pw, ph, center=(corner[0] + center[0], corner[1] + center[1]),
                height=height, max_iter=p.max_iter, julia=p.c if p.kind == "julia" else None,
            )  # fmt: skip
            out = gpu.run("palette", gpu.Tile.full(pw, ph), field=field, lut=lut,
                          density=p.density, inside=inside)  # fmt: skip
            return to_host(out)

        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity)
