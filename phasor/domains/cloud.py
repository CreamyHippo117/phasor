"""Point clouds (BEAUTY.md item 3): hundreds of thousands to millions of points, each a tiny
glowing splat, added together so dense regions build up light and fine structure appears.

    from phasor import Scene, cloud

    s.add(cloud.PointCloud(id="roots", source="roots", degree=12, size=(8, 4.5)))

The points come from a source in ``domains/cloud_math.py`` (the roots of every polynomial
with coefficients +-1, a strange attractor, the Buddhabrot) or are given directly. A point
sits at world (x, y) = ``pos`` + ``scale`` * (the source's point), so with ``pos`` at the
origin and ``scale`` 1 the roots are at their own places in the complex plane. The cloud is
drawn through ``ctx.field`` inside the ``size`` rectangle round ``pos`` by the ``splat``
kernel, which computes only the part on screen: zooming in keeps every point sharp.

Three things keep a cloud from sitting still: ``twist`` (animatable) turns one coefficient
of a root cloud round the origin, so every root glides and the fractal edge morphs;
``twinkle`` flickers each point's light a little (life, a pure function of (id, seed, t));
``white`` lets the densest parts run white-hot.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.rng import rng
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import cloud_math as clm

# Brightness is set against the cloud's mean density (points per pixel over the rectangle,
# at the current zoom): ``exposure`` 1 lights a uniform cloud to 1 - 1/e. The mean is held at
# least this, so a sparse cloud seen close up keeps single points as soft dots, not flares.
MIN_DENSITY = 0.02
ZOOM_LIGHT = 0.7  # 1: brightness held as the camera zooms in; 0: each point's light fixed
CACHE = 8  # point sets kept in memory (the latest ones)
TWINKLE_PERIOD = 2.5  # seconds, typical; each point (each polynomial's roots) has its own


@functools.lru_cache(maxsize=CACHE)
def _points(oid: str, seed: int, source: str, key: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(points (n, 2) float64, value (n,) in [0, 1)) for one source; cached per params."""
    gen = rng(oid, seed)
    if source == "roots":
        degree, coefficients, count, twist = key
        total = clm.every_polynomial_count(degree, coefficients)
        polys = None if total * degree <= count else max(1, count // degree)
        z, v = clm.polynomial_roots(degree, coefficients, polys, gen, twist=twist)
    elif source == "attractor":
        kind, constants, count = key
        z, v = clm.attractor(kind, count, gen, constants)
    elif source == "buddhabrot":
        max_iter, count = key
        z, v = clm.buddhabrot(count, max_iter, gen)
    else:
        pts, values = key
        z = np.array([complex(x, y) for x, y in pts], dtype=np.complex128)
        v = np.asarray(values, dtype=np.float64)
        if not values:  # spread along the list, so a hand-made cloud is not one colour
            v = np.linspace(0.0, 1.0, len(z), endpoint=False)
    out = np.ascontiguousarray(np.c_[z.real, z.imag], dtype=np.float64)
    out.flags.writeable = False
    v = np.asarray(v, dtype=np.float64) % 1.0
    v.flags.writeable = False
    return out, v


@functools.lru_cache(maxsize=CACHE)
def _flicker(oid: str, seed: int, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """(frequency, phase) per group of points that flicker together, from the seed."""
    gen = rng(f"{oid}/twinkle", seed)
    return gen.uniform(0.5, 1.5, groups) / TWINKLE_PERIOD, gen.uniform(0.0, 1.0, groups)


@register
class PointCloud(ObjectBase):
    """Many points of light, splatted additively: where they crowd, light builds up and
    saturates softly (``1 - exp(-density)``), so dense regions glow and the fine structure
    at their edges shows.

    ``source``: ``roots`` (the roots of every polynomial of ``degree`` with coefficients
    from ``coefficients``, or a random sample of them when there are more than ``count``
    roots), ``attractor`` (``kind`` clifford, dejong or lorenz, fitted to the square
    [-1, 1]^2; ``constants`` override the classic ones), ``buddhabrot`` (escaping orbits,
    ``max_iter``) or ``points`` (``points`` given directly, with optional ``values`` 0-1
    for colour). Colour is the ``colormap`` role at each point's value (``color_by``
    ``source``: what the source says, e.g. the polynomial's place in the list; ``phase``:
    the point's angle round ``pos``), turned by ``hue``. ``point_size`` is the splat's
    radius in pixels, ``exposure`` how much light a point gives; ``glow`` makes the bright
    parts shine and haze.

    Motion: ``twist`` (roots) turns the last coefficient of the set by that share of a turn
    (b -> b e^(2 pi i twist)); tween it with ``.to`` and every root glides continuously
    while the fractal edge morphs. ``twinkle`` (0-1, life) flickers each point's light by
    up to that share (the roots of one polynomial together), slowly and differently for
    each; ``white`` (0-1) turns the densest parts white-hot."""

    type: ClassVar[str] = "point_cloud"
    Life: ClassVar = frozenset({"hue", "glow", "twinkle"})
    Style: ClassVar = {"colormap": ("colormap", "spectrum"), "glow": ("glow", "none")}
    Presets: ClassVar = {
        # dark ground, palette hue, glow and haze (the look of the prism strokes)
        "prism": {"params": {"exposure": 0.06, "point_size": 0.9, "white": 0.8,
                             "twinkle": 0.6},
                  "style": {"colormap": "spectrum", "glow": "strong"}},
        "plain": {"params": {"exposure": 0.3, "point_size": 1.0},
                  "style": {"colormap": "sequential", "glow": "none"}},
    }  # fmt: skip

    class Params(ParamsBase):
        source: Literal["roots", "attractor", "buddhabrot", "points"] = "roots"
        degree: int = Field(10, ge=1, le=24)  # roots: the polynomials' degree
        coefficients: tuple[float, ...] = (-1.0, 1.0)  # roots: the coefficient set
        kind: Literal["clifford", "dejong", "lorenz"] = "clifford"  # attractor
        constants: tuple[float, ...] = ()  # attractor: () for the classic constants
        max_iter: int = Field(200, ge=8, le=5000)  # buddhabrot: longest orbit
        points: tuple[tuple[float, float], ...] = ()  # source "points": (x, y) each
        values: tuple[float, ...] = ()  # source "points": colour value 0-1 each (optional)
        count: int = Field(100_000, ge=1, le=20_000_000)  # how many points, at most
        seed: int = 0
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the rectangle and origin of the points
        size: tuple[float, float] = (8.0, 4.5)  # the rectangle the cloud is seen in
        scale: float = Field(1.0, gt=0)  # world units per unit of the source
        point_size: float = Field(1.0, ge=0.4, le=2.5)  # splat radius (sigma), pixels
        exposure: float = Field(0.3, gt=0)  # light per point, against the mean density
        hue: float = 0.0  # turns the palette (a share of it); life adds a little
        color_by: Literal["source", "phase"] = "source"
        twist: float = 0.0  # roots: turns the last coefficient by this share of a turn
        twinkle: float = Field(0.0, ge=0, le=1)  # flicker of each point's light (life)
        white: float = Field(0.0, ge=0, le=1)  # how white the densest parts run

        @field_validator("size")
        @classmethod
        def _positive(cls, v):
            if min(v) <= 0:
                raise ValueError("size must be positive")
            return v

        @field_validator("coefficients")
        @classmethod
        def _coeffs(cls, v):
            if not v or not any(c != 0 for c in v):
                raise ValueError("coefficients must include a nonzero number")
            return v

        @model_validator(mode="after")
        def _points_and_values(self):
            if self.values and len(self.values) != len(self.points):
                raise ValueError(f"{len(self.points)} points but {len(self.values)} values")
            if self.constants and len(self.constants) != len(clm.ATTRACTORS[self.kind]):
                n = len(clm.ATTRACTORS[self.kind])
                raise ValueError(f"a {self.kind} attractor takes {n} constants")
            return self

    def cloud(self, state: SceneState) -> tuple[np.ndarray, np.ndarray]:
        """(points (n, 2) in world x, y; value (n,) 0-1) at this state, from the cache."""
        p = self.me(state).params
        pts, v = _points(self.id, p.seed, p.source, _key(p))
        return pts * p.scale + np.asarray(p.pos[:2]), v

    def _rect(self, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, h = p.size
        corner = np.asarray(p.pos, dtype=np.float64) - np.array([w / 2, h / 2, 0.0])
        return corner, np.array([w, 0.0, 0.0]), np.array([0.0, h, 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        corner, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([corner, corner + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        pts, _ = self.cloud(state)
        corner, r, u = self._rect(p)
        inside = _inside(pts, corner, r, u)
        out: dict[str, Any] = {"kind": "point_cloud", "backdrop": True, "source": p.source,
                               "points": len(pts), "in_rectangle": int(inside.sum())}  # fmt: skip
        if p.source == "roots":
            out.update(degree=p.degree, coefficients=list(p.coefficients), twist=p.twist)
        elif p.source == "attractor":
            out.update(attractor=p.kind)
        elif p.source == "buddhabrot":
            out.update(max_iter=p.max_iter)
        if len(pts):
            out["extent"] = [[round(float(x), 6) for x in pts.min(0)],
                             [round(float(x), 6) for x in pts.max(0)]]  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = self._rect(p)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        bg = np.asarray(state.background[:3], dtype=np.float32)
        glow = float(me.style.get("glow", 1.0))

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            from phasor.render import gpu

            pts, v = self.cloud(state)
            inside = _inside(pts, corner, r, u)
            weights = None
            if p.twinkle > 0 and len(pts):
                group = p.degree if p.source == "roots" else 1  # a polynomial's roots
                freq, phase = _flicker(self.id, p.seed, -(-len(pts) // group))
                w = 1.0 + p.twinkle * np.sin(2 * np.pi * (freq * state.t + phase))
                weights = np.repeat(w, group)[: len(pts)][inside]
            pts = pts[inside] - corner[:2]  # quad coordinates, like ``center``
            v = v[inside]
            if p.color_by == "phase":
                d = pts - (np.asarray(p.pos[:2]) - corner[:2])
                v = np.arctan2(d[:, 1], d[:, 0]) / (2 * np.pi)
            x = ((v + p.hue) % 1.0) * len(lut)
            k0 = np.minimum(x.astype(np.int64), len(lut) - 1)
            t = (x - k0)[:, None].astype(np.float32)
            colors = lut[k0] * (1 - t) + lut[(k0 + 1) % len(lut)] * t
            per_px = len(pts) * (height / ph) ** 2 / (p.size[0] * p.size[1])
            # zooming in spreads the points over more pixels; light per point grows only as
            # the square root of that, so a dense region seen close up does not wash out
            whole = len(pts) * (p.size[1] / ph) ** 2 / (p.size[0] * p.size[1])
            base = max(whole, MIN_DENSITY)
            gain = p.exposure / base * (base / max(per_px, MIN_DENSITY)) ** ZOOM_LIGHT
            out = gpu.render("splat", pw, ph, points=pts, colors=colors, center=center,
                             height=height, sigma=p.point_size, gain=gain, weights=weights,
                             white=p.white)  # fmt: skip
            out[..., :3] += bg * (1.0 - out[..., 3:4])  # light on the scene's ground
            return out

        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity, glow=glow)


def _key(p) -> tuple:
    if p.source == "roots":
        return (p.degree, tuple(p.coefficients), p.count, float(p.twist))
    if p.source == "attractor":
        return (p.kind, tuple(p.constants), p.count)
    if p.source == "buddhabrot":
        return (p.max_iter, p.count)
    return (tuple(p.points[: p.count]), tuple(p.values[: p.count]))


def _inside(pts: np.ndarray, corner, r, u) -> np.ndarray:
    x0, y0 = corner[0], corner[1]
    return ((pts[:, 0] >= x0) & (pts[:, 0] <= x0 + r[0])
            & (pts[:, 1] >= y0) & (pts[:, 1] <= y0 + u[1]))  # fmt: skip
