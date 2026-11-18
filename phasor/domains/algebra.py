"""Abstract algebra objects (SPEC "Domain modules: algebra").

    from phasor import algebra

    s.add(algebra.Polygon(id="square", n=4, rotation=expr("90 * floor(t)")))  # D4 on a square
    s.add(algebra.CayleyTable(id="table", group="D4", pos=(4, 0, 0)))
    s.add(algebra.ClockDiagram(id="clock", n=12, step=4))  # Z12 and a subgroup

The numerics (the groups themselves) are in ``algebra_math``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import algebra_math as am
from phasor.domains.strokes import circle, disc, polyline, with_alpha


def group_named(name: str) -> am.Group:
    """``"D4"``, ``"Z6"``, ``"S3"``: dihedral, cyclic, symmetric."""
    kind, digits = name[:1].upper(), name[1:]
    if not digits.isdigit() or kind not in "DZS":
        raise ValueError(f"unknown group {name!r}; write D<n>, Z<n> or S<n>, e.g. 'D4'")
    n = int(digits)
    g = {"D": am.dihedral, "Z": am.cyclic, "S": am.symmetric}[kind](n)
    if g.order > 24:
        raise ValueError(f"{name} has {g.order} elements; at most 24 are drawn")
    return g


# ---------------------------------------------------------------- Polygon


@register
class Polygon(ObjectBase):
    """A regular n-gon with its vertices numbered, to show a dihedral group acting on it:
    ``rotation`` (degrees) turns it and ``flip`` (0 to 1) mirrors it over the vertical axis (a
    smooth animation passes through edge-on). When it lands on a symmetry (a multiple of
    360/n, flip 0 or 1) the semantics name the group element."""

    type: ClassVar[str] = "polygon"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "text": ("emphasis", "primary"),
        "size": ("text", "label"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        n: int = Field(4, ge=3, le=12)
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(1.5, gt=0)  # circumradius
        rotation: float = 0.0  # degrees, counterclockwise
        flip: float = Field(0.0, ge=0, le=1)
        fill: float = Field(0.25, ge=0, le=1)

    def _verts(self, p) -> np.ndarray:
        """Screen-plane vertex positions (n, 3): vertex 1 starts at the top."""
        k = np.arange(p.n)
        ang = math.pi / 2 + 2 * math.pi * k / p.n  # vertex 1 at the top, counterclockwise
        base = np.stack([np.cos(ang), np.sin(ang)], axis=1) * p.r
        t = math.radians(p.rotation)
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        mirror = np.array([[math.cos(math.pi * p.flip), 0.0], [0.0, 1.0]])
        xy = base @ (mirror @ rot).T  # rotate first, then mirror (r then s, as in algebra_math)
        return np.c_[xy + np.asarray(p.pos[:2]), np.full(p.n, p.pos[2])]

    def _element(self, p):
        step = 360.0 / p.n
        k = p.rotation / step
        near = abs(k - round(k)) < 1e-6
        if near and p.flip in (0.0, 1.0):
            return (round(k) % p.n) + (p.n if p.flip == 1.0 else 0)
        return None

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._verts(p), pad=0.05 * p.r)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        v = self._verts(self.me(state).params)
        return {"vertices": v, "first_vertex": v[0], "center": v.mean(axis=0)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        el = self._element(p)
        group = am.dihedral(p.n)
        return {
            "kind": "polygon",
            "n": p.n,
            "group": group.name,
            "rotation": round(p.rotation, 6),
            "flipped": bool(p.flip > 0.5),
            "element": None if el is None else group.labels[el],
            "element_index": el,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        v = self._verts(p)
        c = np.asarray(p.pos, dtype=np.float64)
        base = with_alpha(me.style["color"], p.opacity)
        w = me.style["stroke"] * float(cam.units_per_height(c))
        if p.fill > 0:
            pos = np.vstack([c[None, :], v])
            k = np.arange(p.n)
            ctx.triangles(
                pos,
                np.c_[np.zeros_like(k), 1 + k, 1 + (k + 1) % p.n],
                color=with_alpha(me.style["color"], p.opacity * p.fill),
                glow=1.0,
            )
        polyline(ctx, cam, np.vstack([v, v[:1]]), w, color=base, glow=me.style["glow"])
        size = me.style["size"] * float(cam.units_per_height(c))
        for k in range(p.n):
            col = with_alpha(me.style["accent"] if k == 0 else me.style["color"], p.opacity)
            disc(ctx, cam, v[k], size * 0.32, color=col, glow=me.style["glow"])
            out = v[k] + (v[k] - c) / max(np.linalg.norm(v[k] - c), 1e-9) * size * 1.0
            ctx.text(str(k + 1), out, size, color=with_alpha(me.style["text"], p.opacity))


# ---------------------------------------------------------------- CayleyTable


@register
class CayleyTable(ObjectBase):
    """The multiplication table of a small group (``D4``, ``Z6``, ``S3``): every cell colored
    by the element it holds, so the Latin-square structure shows."""

    type: ClassVar[str] = "cayley_table"
    Style: ClassVar = {
        "colormap": ("colormap", "phase"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
    }
    Presets: ClassVar = {"soft": {"style": {"colormap": "soft_phase"}}}

    class Params(ParamsBase):
        group: str = "D4"
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        cell: float = Field(0.7, gt=0)
        reveal: float = Field(1.0, ge=0, le=1)  # share of the rows drawn (animate to fill it in)

        @field_validator("group")
        @classmethod
        def _g(cls, v):
            group_named(v)
            return v

    def _g(self, p) -> am.Group:
        return group_named(p.group)

    def _origin(self, p, n):
        c = np.asarray(p.pos, dtype=np.float64)
        return c + np.array([-n * p.cell / 2, n * p.cell / 2, 0.0])  # top-left corner

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        n = self._g(p).order
        o = self._origin(p, n)
        return Bounds.of_points(
            [o - np.array([p.cell, 0, 0]), o + np.array([n * p.cell, -(n + 1) * p.cell, 0])]
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        g = self._g(p)
        t = np.array(g.table)
        latin = all(
            sorted(t[k]) == list(range(g.order)) and sorted(t[:, k]) == list(range(g.order))
            for k in range(g.order)
        )
        return {
            "kind": "cayley_table",
            "group": g.name,
            "order": g.order,
            "abelian": g.is_abelian(),
            "latin_square": latin,
            "identity": g.labels[0],
            "elements": list(g.labels),
            "subgroup_orders": sorted({len(h) for h in g.subgroups()}),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        g = self._g(p)
        n = g.order
        cam = state.camera_for(me)
        o = self._origin(p, n)
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        rows = max(0, min(n, math.ceil(p.reveal * n - 1e-9)))
        size = me.style["size"] * float(cam.units_per_height(o))

        def center(i, j):  # row i (a), column j (b); row/column 0 of the grid are the headers
            return o + cam.right * ((j + 0.5) * p.cell) - cam.up * ((i + 0.5) * p.cell)

        def color(k):
            rgb = lut[int(k / n * len(lut)) % len(lut)]
            return (*rgb, p.opacity)

        for j in range(n):  # column headers (the second factor)
            ctx.text(
                g.labels[j], center(-1, j), size, color=with_alpha(me.style["text"], p.opacity)
            )
        for i in range(rows):
            ctx.text(
                g.labels[i], center(i, -1), size, color=with_alpha(me.style["text"], p.opacity)
            )
            for j in range(n):
                k = g.table[i][j]
                c = center(i, j)
                half = cam.right * (p.cell * 0.46) + cam.up * (p.cell * 0.46)
                quad = np.array(
                    [
                        c - half,
                        c + cam.right * p.cell * 0.46 - cam.up * p.cell * 0.46,
                        c + half,
                        c - cam.right * p.cell * 0.46 + cam.up * p.cell * 0.46,
                    ]
                )
                ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=color(k))
                ctx.text(g.labels[k], c, size, color=(0.02, 0.02, 0.04, p.opacity))


# ---------------------------------------------------------------- ClockDiagram


@register
class ClockDiagram(ObjectBase):
    """The cyclic group Z_n as a clock: n points on a ring; the subgroup generated by ``step`` is
    highlighted and its cosets get one color each (``step`` = 4 on a 12-clock shows the
    four-hour hand's orbit and its three cosets)."""

    type: ClassVar[str] = "clock_diagram"
    Style: ClassVar = {
        "color": ("emphasis", "muted"),
        "accent": ("emphasis", "highlight"),
        "colormap": ("colormap", "phase"),
        "stroke": ("stroke", "hairline"),
        "size": ("text", "annotation"),
        "text": ("emphasis", "primary"),
    }

    class Params(ParamsBase):
        n: int = Field(12, ge=2, le=36)
        step: int = Field(1, ge=1)
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(2.0, gt=0)
        cosets: float = Field(0.0, ge=0, le=1)  # fade in the coset coloring

    def _pts(self, p):
        a = math.pi / 2 - 2 * math.pi * np.arange(p.n) / p.n  # 0 at the top, clockwise
        return np.stack([np.cos(a), np.sin(a), np.zeros(p.n)], axis=1) * p.r + np.asarray(p.pos)

    def _sub(self, p) -> frozenset[int]:
        return am.cyclic(p.n).generated([p.step % p.n])

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._pts(p), pad=0.25)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        pts = self._pts(p)
        return {"points": pts, "orbit": pts[sorted(self._sub(p))]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        g = am.cyclic(p.n)
        sub = self._sub(p)
        return {
            "kind": "clock_diagram",
            "n": p.n,
            "generator": p.step % p.n,
            "subgroup": sorted(sub),
            "subgroup_order": len(sub),
            "index": p.n // len(sub),
            "generator_order": g.element_order(p.step % p.n),
            "cosets": [sorted(c) for c in g.left_cosets(sub)],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        g = am.cyclic(p.n)
        sub = self._sub(p)
        pts = self._pts(p)
        w = me.style["stroke"] * float(cam.units_per_height(np.asarray(p.pos)))
        circle(ctx, cam, p.pos, p.r, w, color=with_alpha(me.style["color"], p.opacity * 0.6))
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        cosets = g.left_cosets(sub)
        owner = {x: i for i, c in enumerate(cosets) for x in c}
        size = me.style["size"] * float(cam.units_per_height(np.asarray(p.pos)))
        for k in range(p.n):
            in_sub = k in sub
            base = with_alpha(me.style["accent"] if in_sub else me.style["color"], p.opacity)
            if p.cosets > 0 and not in_sub:
                rgb = lut[int(owner[k] / max(len(cosets), 1) * len(lut)) % len(lut)]
                base = (*rgb, p.opacity * p.cosets)
            disc(ctx, cam, pts[k], size * (0.4 if in_sub else 0.28), color=base)
            out = pts[k] + (pts[k] - np.asarray(p.pos)) / p.r * size * 1.1
            ctx.text(str(k), out, size, color=with_alpha(me.style["text"], p.opacity))
        orbit = [pts[x] for x in sorted(sub)]
        if len(orbit) >= 2:
            polyline(
                ctx,
                cam,
                np.vstack([orbit, orbit[:1]]),
                w * 2,
                color=with_alpha(me.style["accent"], p.opacity * 0.8),
            )
