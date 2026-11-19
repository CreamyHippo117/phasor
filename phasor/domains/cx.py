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
from fractions import Fraction
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import fractal_math as fm

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
    # Hue means phase here, so the full wheel stays the default; presets vary the rest.
    Presets: ClassVar = {
        "classic": {"params": {"softness": 0.0, "contour": 0.35}},  # hard contour edges
        "soft_phase": {"params": {"softness": 0.8, "contour": 0.25},
                       "style": {"colormap": "soft_phase"}},  # restrained three-hue wheel
        "vivid": {"params": {"softness": 0.5, "contour": 0.4}},
    }  # fmt: skip

    class Params(ParamsBase):
        function: str = Field(min_length=1)  # e.g. "1/(z**2 + 1)"; see docs/ai/API.md
        pos: Vec3 = (0.0, 0.0, 0.0)  # center of the rectangle
        size: tuple[float, float] = (8.0, 4.5)  # width and height in world units
        contour: float = Field(0.35, ge=0, le=1)  # how dark the modulus bands get
        softness: float = Field(0.5, ge=0, le=1)  # feathering of the band edges (0: hard)
        hue: float = 0.0  # turns the color wheel (a share of one turn); life adds a little

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
        lut = np.roll(lut, -round(p.hue * len(lut)), axis=0)
        f = _parse(p.function.strip())

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            from phasor.render import gpu

            return gpu.render(
                "domain", pw, ph, supersample=SUPERSAMPLE, function=f,
                center=(corner[0] + center[0], corner[1] + center[1]), height=height,
                contour=p.contour, soft=p.softness, lut=lut,
                precision="f64" if height < F64_BELOW else "f32",
            )  # fmt: skip

        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity)


@functools.lru_cache(maxsize=64)
def _point(text: str) -> fm.Point:
    return fm.parse_point(text)


@functools.lru_cache(maxsize=8)
def _orbit(c: fm.Point, bits: int, max_iter: int) -> np.ndarray:
    """The reference orbit, computed once per (point, precision, steps): a steady dive
    recomputes it only when the precision steps up (every 32 bits, about 10 digits)."""
    orbit = fm.reference_orbit(c, bits, max_iter)
    orbit.setflags(write=False)
    return orbit


STEADY_ROWS = 18  # the coarse sample of the whole rectangle that ``steady`` reads


def _escape_field(p, c: fm.Point, k: float, pw: int, ph: int, center, height) -> np.ndarray:
    """The escape field of a ``pw`` x ``ph`` view of a Fractal whose rectangle-relative
    ``center`` and ``height`` are world units; ``c`` and ``k`` from ``Fractal._view``."""
    from phasor.render import gpu

    w, h = p.size
    off, fh = ((center[0] - w / 2) * k, (center[1] - h / 2) * k), height * k  # from c
    if _precision(p.kind, fh, ph) == "perturbation":
        orbit = _orbit(c, fm.bits_for(fh / ph), p.max_iter)
        return gpu.render("deep", pw, ph, center=off, height=fh, orbit=orbit,
                          max_iter=p.max_iter)  # fmt: skip
    return gpu.render(
        "escape", pw, ph, center=(float(c[0]) + off[0], float(c[1]) + off[1]), height=fh,
        max_iter=p.max_iter, julia=p.c if p.kind == "julia" else None,
    )  # fmt: skip


def _precision(kind: str, height: float, rows: int) -> str:
    """f32, f64 or perturbation for a view ``height`` plane units tall in ``rows`` pixels.
    Julia sets have no perturbation path here; past that they stay float64 (blocky)."""
    from phasor.render.gpu.kernels.fractal import precision_for

    if kind == "mandelbrot" and height / rows < fm.DEEP_BELOW:
        return "perturbation"
    return precision_for(height)


@register
class Fractal(_Plane):
    """The Mandelbrot set, or the Julia set of ``c``, colored by escape time. Points in the
    set take the theme's background color.

    The rectangle shows the plane around ``center`` (exact decimal text, ``"re im"``; empty:
    the rectangle's own position), magnified ``10**zoom`` times. Precision follows the pixel
    size (SPEC "Precision policy"): float32, then float64 under 1e-5, then perturbation
    against an arbitrary-precision reference orbit under 1e-10 per pixel (Mandelbrot only),
    good to a zoom of about 1e280. For a dive, animate ``zoom`` and set ``steady=True``.
    """

    type: ClassVar[str] = "fractal"
    Style: ClassVar = {"colormap": ("colormap", "sequential")}
    Presets: ClassVar = {
        "classic": {"style": {"colormap": "sequential"}},
        "ember": {"style": {"colormap": "ember"}, "params": {"density": 0.3}},
        "ocean": {"style": {"colormap": "ocean"}, "params": {"density": 0.3}},
    }  # fmt: skip

    class Params(ParamsBase):
        kind: Literal["mandelbrot", "julia"] = "mandelbrot"
        c: tuple[float, float] = (-0.8, 0.156)  # the Julia constant (unused for mandelbrot)
        pos: Vec3 = (-0.6, 0.0, 0.0)
        size: tuple[float, float] = (3.2, 1.8)
        max_iter: int = Field(256, ge=1, le=100_000)
        density: float = Field(0.35, gt=0)  # color bands per sqrt(iteration)
        hue: float = 0.0  # shifts colors along the ramp (share of a cycle); life adds a little
        center: str = ""  # the plane point at the middle, exact: "-0.7756837680090537974 0.13"
        zoom: float = Field(0.0, ge=-3, le=280)  # magnification 10**zoom about ``center``
        # Keep the outermost color fixed as counts grow (a dive's colors then stay put while
        # the structure flows); off: colors cycle with the escape count itself.
        steady: bool = False

        @field_validator("center")
        @classmethod
        def _point(cls, v: str) -> str:
            if v.strip():
                fm.parse_point(v)  # ValueError names the format
            return v

    def _view(self, p) -> tuple[fm.Point, float]:
        """(the exact plane point at the rectangle's middle, plane units per world unit)."""
        c = _point(p.center) if p.center.strip() else (Fraction(p.pos[0]), Fraction(p.pos[1]))
        return c, 10.0**-p.zoom

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        c, k = self._view(p)
        out: dict[str, Any] = {"kind": "fractal", "backdrop": True, "set": p.kind,
                               "max_iter": p.max_iter}  # fmt: skip
        if p.kind == "julia":
            out["c"] = list(p.c)
        if p.center.strip() or p.zoom:
            out["center"] = fm.format_point(c)
            out["zoom"] = p.zoom
            out["view_height"] = p.size[1] * k
            # at a 1080-row render of the whole rectangle
            out["precision"] = _precision(p.kind, p.size[1] * k, 1080)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = _rect(p)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        inside = tuple(state.background[:3])

        c, k = self._view(p)

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            from phasor.render import gpu
            from phasor.render.gpu.bridge import to_host

            field = _escape_field(p, c, k, pw, ph, center, height)
            offset = p.hue
            if p.steady:  # hold the color of the rectangle's outermost (lowest) count
                w, h = p.size
                coarse = _escape_field(p, c, k, STEADY_ROWS * 16 // 9, STEADY_ROWS,
                                       (w / 2, h / 2), h)  # fmt: skip
                out_ = coarse[coarse >= 0]
                offset -= p.density * float(np.sqrt(out_.min())) if out_.size else 0.0
            out = gpu.run("palette", gpu.Tile.full(pw, ph), field=field, lut=lut,
                          density=p.density, offset=offset, inside=inside)  # fmt: skip
            return to_host(out)

        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity)


# ---------------------------------------------------------------- Riemann surfaces

_BRANCHES = {  # kind -> (turns of the angle the surface needs, f(r, theta) as a complex number)
    "sqrt": (2, lambda r, th: np.sqrt(r) * np.exp(0.5j * th)),
    "cbrt": (3, lambda r, th: np.cbrt(r) * np.exp(1j * th / 3)),
    "log": (2, lambda r, th: np.log(r) + 1j * th),
}
_CUTS = {"sqrt": "negative real axis", "cbrt": "negative real axis", "log": "every turn"}


@register
class RiemannSurface(ObjectBase):
    """The Riemann surface of a multivalued function over the complex plane: each sheet is
    one branch, and walking once around the branch point carries you onto the next sheet.

    Height is ``Re f(z)`` (or ``Im``) times ``scale``; color is the phase of ``f(z)``, so each
    sheet has its own colors. Centered on ``pos``, out to ``radius``. A lit 3D surface: give
    the scene an orbit camera (``s.use_camera("orbit")``).
    """

    type: ClassVar[str] = "riemann_surface"
    Style: ClassVar = {"colormap": ("colormap", "phase"), "material": ("surface", "solid")}
    Presets: ClassVar = {
        "glass": {"style": {"material": "glass"}},  # sheets that do not hide each other
        "soft_glass": {"style": {"material": "glass", "colormap": "soft_phase"}},
    }

    class Params(ParamsBase):
        kind: Literal["sqrt", "cbrt", "log"] = "sqrt"
        pos: Vec3 = (0.0, 0.0, 0.0)  # the branch point
        radius: float = Field(2.0, gt=0)
        height: Literal["re", "im"] = "re"
        scale: float = Field(1.0, gt=0)
        rings: int = Field(28, ge=2, le=400)  # mesh resolution along the radius
        segments: int = Field(90, ge=8, le=2000)  # per turn around the branch point

    def _grid(self, state: SceneState):
        """(points (R, T, 3), r (R,), theta (T,), f (R, T) complex)."""
        p = self.me(state).params
        turns, fn = _BRANCHES[p.kind]
        r = np.linspace(p.radius * 0.01, p.radius, p.rings + 1)
        th = np.linspace(0.0, 2 * np.pi * turns, p.segments * turns + 1)
        if p.kind == "log":
            th -= 2 * np.pi * turns / 2
        rr, tt = np.meshgrid(r, th, indexing="ij")
        f = fn(rr, tt)
        h = (f.real if p.height == "re" else f.imag) * p.scale
        x0, y0, z0 = p.pos
        pts = np.stack([x0 + rr * np.cos(tt), y0 + rr * np.sin(tt), z0 + h], axis=-1)
        return pts, r, th, f

    def world_bounds(self, state: SceneState) -> Bounds:
        pts = self._grid(state)[0].reshape(-1, 3)
        return Bounds.of_points(pts)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        """``branch_point``, and ``sheets``: one point on each sheet, above the point
        ``0.75 * radius * i`` of the plane (where the sheets are furthest apart)."""
        p = self.me(state).params
        turns, fn = _BRANCHES[p.kind]
        r = 0.75 * p.radius
        out = []
        for k in range(turns):
            th = np.pi / 2 + 2 * np.pi * k - (2 * np.pi * turns / 2 if p.kind == "log" else 0)
            f = fn(r, th)
            h = (f.real if p.height == "re" else f.imag) * p.scale
            out.append((p.pos[0] + r * np.cos(th), p.pos[1] + r * np.sin(th), p.pos[2] + h))
        return {"branch_point": np.asarray(p.pos, dtype=np.float64), "sheets": np.array(out)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        turns, _ = _BRANCHES[p.kind]
        return {
            "kind": "riemann_surface",
            "function": {"sqrt": "sqrt(z)", "cbrt": "z**(1/3)", "log": "log(z)"}[p.kind],
            "sheets": turns,
            "branch_points": [list(p.pos)],
            "branch_cut": _CUTS[p.kind],
            "height": f"{'Re' if p.height == 're' else 'Im'} f(z)",
            "material": self.me(state).roles["material"],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        pts, _r, th, f = self._grid(state)
        nr, nt = pts.shape[:2]
        # Normals from the grid's tangents (one-sided at the edges).
        du = np.gradient(pts, axis=0)
        dv = np.gradient(pts, axis=1)
        normals = np.cross(du, dv)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        phase = (np.angle(f) % (2 * np.pi)) / (2 * np.pi)
        rgb = lut[np.minimum((phase * len(lut)).astype(int), len(lut) - 1)]
        colors = np.concatenate([rgb, np.ones((*rgb.shape[:-1], 1))], axis=-1)
        uv = np.stack(np.meshgrid(np.linspace(0, 1, nr), th / (2 * np.pi), indexing="ij"),
                      axis=-1)[..., ::-1]  # fmt: skip
        k = np.arange(nr - 1)[:, None] * nt + np.arange(nt - 1)[None, :]
        k = k.ravel()
        tris = np.vstack([np.c_[k, k + nt, k + 1], np.c_[k + 1, k + nt, k + nt + 1]])
        ctx.surface(pts.reshape(-1, 3), tris, normals=normals.reshape(-1, 3),
                    colors=colors.reshape(-1, 4), uv=uv.reshape(-1, 2),
                    material=me.roles["material"],
                    opacity=me.params.opacity * me.style["material"])  # fmt: skip
