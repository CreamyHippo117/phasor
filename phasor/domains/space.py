"""Objects the 3D and point verbs grow: a function's landscape, a cutting plane and a dashed
segment (the ``lift``, ``slice`` and ``project`` verbs make them; each is also a plain type).

    s.add(space.Landscape(id="land", function="1/(z**2 + 1)", height=0.6, lines=20))
    s.add(space.CutPlane(id="cut", surface="land", axis="x", at=0.5))
    s.add(space.Dashed(id="drop", start=(2, 1, 0), end=(2, 0, 0), square=0.15))

``Landscape`` is the plane of a ``DomainColoring`` raised into a surface: the height is
``log(1 + |f|)`` (clipped, so a pole is a flat-topped tower, not a spike), the color the
phase of f. ``CutPlane`` is a plane at ``at`` along ``axis`` through any surface that gives
``grid_points(state)`` (``ParametricSurface``, ``pde.Surface``, ``Landscape``), drawn with the
curve where it meets the surface. ``Dashed`` is a dashed line (with an optional right-angle
mark), the perpendicular a projection drops.
"""

from __future__ import annotations

import functools
import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains.cx import _parse
from phasor.domains.heatmap import lut_colour
from phasor.domains.strokes import polyline, polylines, with_alpha

AXES = {"x": 0, "y": 1, "z": 2}


def _vec(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


# ---------------------------------------------------------------- Landscape


@functools.lru_cache(maxsize=32)
def _landscape(
    function: str, to: str, blend: float, a: tuple, pos: tuple, size: tuple, n: int, cap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xy grid (nx, ny, 2), height share (nx, ny) in [0, 1], phase turns (nx, ny)) of the
    function over the rectangle: ``log(1 + |f|)`` over ``cap``, clipped to 1."""
    w, h = size
    nx, ny = (n, max(2, round(n * h / w))) if w >= h else (max(2, round(n * w / h)), n)
    xs = pos[0] + (np.linspace(0, 1, nx) - 0.5) * w
    ys = pos[1] + (np.linspace(0, 1, ny) - 0.5) * h
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    z = X + 1j * Y
    f = _parse(function).numpy()(z, complex(*a))
    if to and blend > 0:
        f = (1 - blend) * f + blend * _parse(to).numpy()(z, complex(*a))
    with np.errstate(all="ignore"):
        mag = np.log1p(np.abs(f)) / cap
        turns = np.angle(f) / (2 * math.pi)
    share = np.clip(np.nan_to_num(mag, nan=1.0, posinf=1.0), 0.0, 1.0)
    turns = np.nan_to_num(turns, nan=0.0)
    for arr in (share, turns):
        arr.flags.writeable = False
    xy = np.stack([X, Y], axis=-1)
    xy.flags.writeable = False
    return xy, share, turns


@register
class Landscape(ObjectBase):
    """The landscape of a function of z: over the rectangle (``pos``, ``size``, in the complex
    plane like ``DomainColoring``) a surface whose height is ``height * min(log(1 + |f|), cap)``
    and whose color is the phase of f through the ``colormap`` (a wheel), darker where |f| is
    small. Poles are clipped at ``cap``, so they are flat-topped towers. ``height`` 0 is flat:
    animate it (the ``lift`` verb does) to raise a flat coloring into its landscape.
    ``lines`` draws that many glowing grid lines each way, in the hue under them. ``a`` is the
    constant the function may use; ``to`` and ``blend`` as in ``DomainColoring``. Give the
    scene an orbit camera."""

    type: ClassVar[str] = "landscape"
    Life: ClassVar = frozenset({"hue", "glow"})
    Style: ClassVar = {"colormap": ("colormap", "phase"), "material": ("surface", "solid"),
                       "glow": ("glow", "none"), "stroke": ("stroke", "hairline")}  # fmt: skip
    Presets: ClassVar = {
        "prism": {"params": {"lines": 24},
                  "style": {"glow": "strong", "material": "glass"}},
    }  # fmt: skip

    class Params(ParamsBase):
        function: str = "z**2 - 1"
        to: str = ""
        blend: float = Field(0.0, ge=0, le=1)
        a: tuple[float, float] = (0.0, 0.0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # center of the rectangle; the flat landscape's height
        size: tuple[float, float] = (8.0, 4.5)
        n: int = Field(64, ge=8, le=300)  # grid cells along the longer side
        height: float = Field(0.6, ge=0)  # world height per unit of log(1 + |f|); 0: flat
        cap: float = Field(3.0, gt=0)  # log(1 + |f|) is clipped here (a pole's tower)
        hue: float = 0.0  # turns the color wheel (a share of one turn); life adds a little
        lines: int = Field(0, ge=0, le=100)  # glowing grid lines each way (the prism look)

        @field_validator("function", "to")
        @classmethod
        def _check(cls, v: str, info) -> str:
            from phasor.render.gpu.codegen import FunctionError

            if info.field_name == "to" and not v.strip():
                return v
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

    def _grid(self, p):
        return _landscape(p.function.strip(), p.to.strip(), float(p.blend), tuple(p.a),
                          tuple(p.pos[:2]), tuple(p.size), p.n, float(p.cap))  # fmt: skip

    def _mesh(self, state: SceneState):
        p = self.me(state).params
        xy, share, turns = self._grid(p)
        z = p.pos[2] + p.height * p.cap * share
        return np.concatenate([xy, z[..., None]], axis=-1), share, turns

    def grid_points(self, state: SceneState) -> np.ndarray:
        """The surface as a (nx, ny, 3) grid of world points (what ``slice`` cuts)."""
        return self._mesh(state)[0]

    def world_bounds(self, state: SceneState) -> Bounds:
        return Bounds.of_points(self._mesh(state)[0].reshape(-1, 3), pad=0.01)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pts, share, _ = self._mesh(state)
        i = np.unravel_index(int(np.argmax(share)), share.shape)
        return {"peak": pts[i], "center": pts.reshape(-1, 3).mean(axis=0)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        pts, share, _ = self._mesh(state)
        f = _parse(p.function.strip())
        return {"kind": "landscape", "function": f.source, "latex": f.latex(),
                "height": round(float(p.height), 6),
                "peak": round(float(pts[..., 2].max() - p.pos[2]), 6),
                "clipped": round(float((share >= 1.0).mean()), 6)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        me = self.me(state)
        p = me.params
        pts, share, turns = self._mesh(state)
        nx, ny = share.shape
        du, dv = np.gradient(pts, axis=0), np.gradient(pts, axis=1)
        normals = np.cross(du, dv)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        lut = np.roll(np.asarray(me.style["colormap"], dtype=np.float64)[:, :3],
                      -round(p.hue * len(me.style["colormap"])), axis=0)  # fmt: skip
        hue = lut_colour(lut, "wrap")(turns)
        rgb = hue * (0.45 + 0.55 * share[..., None])  # dark where the function is small
        colors = np.concatenate([rgb, np.ones((nx, ny, 1))], axis=-1)
        k = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
        tris = np.vstack([np.c_[k, k + ny, k + 1], np.c_[k + 1, k + ny, k + ny + 1]])
        uv = np.stack(np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij"), -1)
        ctx.surface(pts.reshape(-1, 3), tris, normals=normals.reshape(-1, 3),
                    colors=colors.reshape(-1, 4), uv=uv.reshape(-1, 2),
                    material=me.roles["material"],
                    opacity=p.opacity * float(me.style["material"]))  # fmt: skip
        if p.lines > 0 and p.opacity > 0:
            self._net(ctx, state, pts, normals, hue * (0.4 + 0.6 * share[..., None]))

    def _net(self, ctx: DrawContext, state: SceneState, pts, normals, rgb) -> None:
        """``lines`` grid lines each way as glowing strokes in the hue under them, lifted a
        hair off the surface so they never sink into it."""
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        lifted = pts + normals * (0.004 * max(p.size))
        rgba = np.concatenate([rgb, np.full((*rgb.shape[:-1], 1), p.opacity)], axis=-1)
        glow = float(me.style["glow"])
        for axis in (0, 1):
            grid, cols = np.moveaxis(lifted, axis, 0), np.moveaxis(rgba, axis, 0)
            step = max(1, (len(grid) - 1) // p.lines)
            rows = grid[::step]
            w = me.style["stroke"] * cam.units_per_height(rows.reshape(-1, 3))
            polylines(ctx, cam, rows, w.reshape(rows.shape[:2]), color=(1, 1, 1, p.opacity),
                      glow=glow, colors=cols[::step])  # fmt: skip


# ---------------------------------------------------------------- CutPlane


def cut_curves(grid: np.ndarray, axis: int, at: float) -> list[np.ndarray]:
    """Where the plane ``coordinate[axis] == at`` meets the surface whose vertices are the
    (nu, nv, 3) ``grid`` (two triangles a cell): the crossing points of each edge the plane
    crosses, chained into curves (a closed one repeats its first point at the end).
    A vertex exactly on the plane counts as on the high side, so no curve ever breaks."""
    nu, nv = grid.shape[:2]
    flat = grid.reshape(-1, 3)
    side = flat[:, axis] >= at
    k = (np.arange(nu - 1)[:, None] * nv + np.arange(nv - 1)[None, :]).ravel()
    tris = np.vstack([np.c_[k, k + nv, k + 1], np.c_[k + 1, k + nv, k + nv + 1]])
    s = side[tris]
    mixed = s.any(axis=1) & ~s.all(axis=1)
    tris = tris[mixed]
    if not len(tris):
        return []
    pairs = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1)  # (m, 3, 2)
    crossing = side[pairs[..., 0]] != side[pairs[..., 1]]  # (m, 3): exactly two per triangle
    total = len(flat)
    edges = np.sort(pairs, axis=-1)
    keys = edges[..., 0] * total + edges[..., 1]
    chosen = keys[crossing].reshape(-1, 2)
    point: dict[int, np.ndarray] = {}
    for key, (a, b) in zip(keys[crossing], edges[crossing], strict=True):
        if int(key) not in point:
            pa, pb = flat[a], flat[b]
            u = (at - pa[axis]) / (pb[axis] - pa[axis])
            point[int(key)] = pa + u * (pb - pa)
    nbr: dict[int, list[int]] = {}
    for a, b in chosen.tolist():
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    seen: set[int] = set()
    curves: list[np.ndarray] = []
    # open curves start at an end (one neighbour); what is left is closed loops
    for start in sorted(nbr, key=lambda q: (len(nbr[q]) != 1, q)):
        if start in seen:
            continue
        chain, prev, cur = [start], -1, start
        seen.add(start)
        while True:
            nxt = [q for q in nbr[cur] if q != prev and q not in seen]
            if not nxt:
                if len(chain) > 2 and start in nbr[cur] and prev != start:
                    chain.append(start)  # a closed loop
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            chain.append(cur)
        if len(chain) >= 2:
            curves.append(np.array([point[q] for q in chain]))
    return curves


@register
class CutPlane(ObjectBase):
    """A plane at coordinate ``at`` along ``axis`` ("x", "y" or "z", world units) cutting the
    surface ``surface`` (any object with ``grid_points``: ``ParametricSurface``,
    ``pde.Surface``, ``Landscape``), drawn as a see-through sheet a little wider than the
    surface, with the curve where it meets it as a glowing stroke. Animate ``at`` to sweep it
    through (the ``slice`` verb does). The curve is recomputed from the surface at every
    time, so it follows a surface that moves."""

    type: ClassVar[str] = "cut_plane"
    Life: ClassVar = frozenset({"glow"})
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "curve": ("emphasis", "highlight"),
        "stroke": ("stroke", "bold"),
        "glow": ("glow", "none"),
        "material": ("surface", "glass"),
    }  # fmt: skip

    class Params(ParamsBase):
        surface: str = ""  # the id of the surface to cut
        axis: Literal["x", "y", "z"] = "z"  # horizontal by default: visible from above too
        at: float = 0.0
        pad: float = Field(0.08, ge=0)  # the sheet reaches this share past the surface

    def _grid(self, state: SceneState) -> np.ndarray | None:
        name = self.me(state).params.surface
        if not name or name not in state:
            return None
        impl = state[name].impl
        return impl.grid_points(state) if hasattr(impl, "grid_points") else None

    def _sheet(self, state: SceneState) -> np.ndarray:
        """The sheet's four corners, round the surface's box in the two other coordinates."""
        p = self.me(state).params
        ax = AXES[p.axis]
        grid = self._grid(state)
        if grid is None:
            lo, hi = np.full(3, -1.0), np.full(3, 1.0)
        else:
            flat = grid.reshape(-1, 3)
            lo, hi = flat.min(0), flat.max(0)
        room = (hi - lo) * p.pad + 0.05
        lo, hi = lo - room, hi + room
        i, j = [k for k in range(3) if k != ax]
        corners = []
        for a, b in ((lo[i], lo[j]), (hi[i], lo[j]), (hi[i], hi[j]), (lo[i], hi[j])):
            c = np.zeros(3)
            c[ax], c[i], c[j] = p.at, a, b
            corners.append(c)
        return np.array(corners)

    def curves(self, state: SceneState) -> list[np.ndarray]:
        p = self.me(state).params
        grid = self._grid(state)
        return [] if grid is None else cut_curves(grid, AXES[p.axis], p.at)

    def world_bounds(self, state: SceneState) -> Bounds:
        return Bounds.of_points(self._sheet(state))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        out = {"center": self._sheet(state).mean(axis=0)}
        found = self.curves(state)
        if len(found) == 1:
            c = found[0]
            out.update(start=c[0], end=c[-1], curve=c)
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        found = self.curves(state)
        out: dict[str, Any] = {"kind": "cut_plane", "surface": p.surface, "axis": p.axis,
                               "at": round(float(p.at), 6), "curves": len(found),
                               "points": int(sum(len(c) for c in found))}  # fmt: skip
        if self._grid(state) is None:
            out["missing_surface"] = True
        if found:
            allp = np.vstack(found)
            for k, name in enumerate("xyz"):
                if k != AXES[p.axis]:
                    out[f"{name}_range"] = [round(float(allp[:, k].min()), 6),
                                            round(float(allp[:, k].max()), 6)]  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        sheet = self._sheet(state)
        n = np.cross(sheet[1] - sheet[0], sheet[3] - sheet[0])
        n /= max(np.linalg.norm(n), 1e-12)
        rgba = np.tile(np.asarray(me.style["color"], dtype=np.float64), (4, 1))
        ctx.surface(sheet, np.array([[0, 1, 2], [0, 2, 3]]), normals=np.tile(n, (4, 1)),
                    colors=rgba, uv=np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]]),
                    material=me.roles["material"],
                    opacity=0.3 * p.opacity * float(me.style["material"]))  # fmt: skip
        glow = float(me.style["glow"])
        color = with_alpha(me.style["curve"], p.opacity)
        for curve in self.curves(state):
            w = me.style["stroke"] * cam.units_per_height(curve)
            polyline(ctx, cam, curve, w, color=color, glow=glow)


# ---------------------------------------------------------------- Dashed


@register
class Dashed(ObjectBase):
    """A dashed straight line from ``start`` to ``end`` (``dashes`` of them, each ``duty`` of
    its slot long), drawn from ``start`` as far as ``progress``. ``square`` > 0 adds a
    right-angle mark of that size at ``end`` between the line and the direction ``back``
    (what a projection's perpendicular wears at its foot)."""

    type: ClassVar[str] = "dashed"
    Life: ClassVar = frozenset({"glow"})
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }  # fmt: skip

    class Params(ParamsBase):
        start: Vec3 = (0.0, 0.0, 0.0)
        end: Vec3 = (1.0, 0.0, 0.0)
        progress: float = Field(1.0, ge=0, le=1)  # drawn share of the length
        dashes: int = Field(8, ge=1, le=100)
        duty: float = Field(0.6, gt=0, le=1)
        square: float = Field(0.0, ge=0)  # right-angle mark at ``end``: its side, world units
        back: Vec3 = (-1.0, 0.0, 0.0)  # from ``end`` along the line the mark sits on

    def _ends(self, state: SceneState):
        p = self.me(state).params
        return _vec(p.start), _vec(p.end)

    def world_bounds(self, state: SceneState) -> Bounds:
        a, b = self._ends(state)
        return Bounds.of_points([a, b], pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        a, b = self._ends(state)
        return {"start": a, "end": b, "center": (a + b) / 2}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        a, b = self._ends(state)
        p = self.me(state).params
        out = {"kind": "dashed_line", "length": round(float(np.linalg.norm(b - a)), 6),
               "dashes": p.dashes}  # fmt: skip
        if p.progress < 1:
            out["drawn"] = round(float(p.progress), 4)
        if p.progress <= 0 or out["length"] < 1e-9:
            out["drawn"] = 0.0
        return out

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        a, b = self._ends(state)
        length = float(np.linalg.norm(b - a))
        if p.progress <= 0 or length < 1e-9:
            return
        w = me.style["stroke"] * float(cam.units_per_height((a + b) / 2))
        color = with_alpha(me.style["color"], p.opacity)
        glow = float(me.style["glow"])
        slot = 1.0 / p.dashes
        for k in range(p.dashes):
            u0, u1 = k * slot, min(k * slot + p.duty * slot, p.progress)
            if u1 > u0:
                polyline(ctx, cam, [a + (b - a) * u0, a + (b - a) * u1], w, color=color, glow=glow)
        if p.square > 0 and p.progress >= 1.0:
            back = _vec(p.back)
            toward = (a - b) / length
            if np.linalg.norm(back) > 1e-9:
                s = p.square
                back = back / np.linalg.norm(back)
                polyline(ctx, cam, [b + back * s, b + back * s + toward * s, b + toward * s], w,
                         color=color, glow=glow)  # fmt: skip
