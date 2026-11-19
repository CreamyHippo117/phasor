"""Built-in 2D primitives: dot, line, text.

They draw in the plane facing the camera (its ``right``/``up`` vectors), so the same code
serves the orthographic 2D case now and billboards in 3D later. Sizes that come from style
roles (stroke weight, text size) are fractions of the frame height, converted to world
units through the camera, so they look the same at any zoom.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from phasor.core import text as text_metrics
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains.strokes import polyline

DISC_SEGMENTS = 64


def _rgba(color: tuple[float, ...], opacity: float) -> tuple[float, float, float, float]:
    return (color[0], color[1], color[2], color[3] * opacity)


@register
class Dot(ObjectBase):
    """A filled disc: a point, a pole, a sample."""

    type: ClassVar[str] = "dot"
    Life: ClassVar = frozenset({"wander", "glow"})
    Style: ClassVar = {"color": ("emphasis", "primary"), "glow": ("glow", "none")}
    Presets: ClassVar = {
        "plain": {},
        "glowing": {"style": {"glow": "soft"}},
        "beacon": {"style": {"color": "highlight", "glow": "strong"}},
    }

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(0.1, gt=0)  # radius in world units

    def _basis(self, state: SceneState):
        cam = state.camera_for(self.me(state))
        return cam.right, cam.up

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        right, up = self._basis(state)
        half = p.r * np.sqrt(right**2 + up**2)
        return Bounds(tuple(np.asarray(p.pos) - half), tuple(np.asarray(p.pos) + half))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "point", "position": list(p.pos), "radius": p.r}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        right, up = self._basis(state)
        a = np.linspace(0, 2 * np.pi, DISC_SEGMENTS, endpoint=False)
        rim = np.asarray(p.pos) + p.r * (np.outer(np.cos(a), right) + np.outer(np.sin(a), up))
        positions = np.vstack([np.asarray(p.pos)[None, :], rim])
        k = np.arange(DISC_SEGMENTS)
        indices = np.c_[np.zeros_like(k), 1 + k, 1 + (k + 1) % DISC_SEGMENTS]
        ctx.triangles(positions, indices, color=_rgba(me.style["color"], p.opacity),
                      glow=me.style["glow"])  # fmt: skip


@register
class Line(ObjectBase):
    """A polyline through two or more points, with mitered joins."""

    type: ClassVar[str] = "line"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {
        "ink": {},  # plain, regular stroke
        "glowing": {"style": {"glow": "soft", "stroke": "bold"}},
        "beam": {"style": {"glow": "strong", "stroke": "bold", "color": "highlight"}},
        "hairline": {"style": {"stroke": "hairline", "color": "muted"}},
    }

    class Params(ParamsBase):
        points: tuple[Vec3, ...] = Field(min_length=2)
        progress: float = Field(1.0, ge=0, le=1)  # drawn share of the length (the trace verb)

    def _width(self, state: SceneState, points=None):
        """Stroke width in world units: a share of the frame height, so it looks the same at
        any zoom. With ``points`` (N, 3), one width per point (perspective: by depth)."""
        me = self.me(state)
        return me.style["stroke"] * state.camera_for(me).units_per_height(points)

    def _drawn(self, state: SceneState) -> np.ndarray:
        """The points of the part drawn so far: the first ``progress`` of the length."""
        p = self.me(state).params
        pts = np.asarray(p.points, dtype=np.float64)
        if p.progress >= 1.0:
            return pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        stop = p.progress * cum[-1]
        k = int(np.searchsorted(cum, stop, side="right"))  # points fully inside
        k = min(max(k, 1), len(pts) - 1)
        u = (stop - cum[k - 1]) / seg[k - 1] if seg[k - 1] > 0 else 0.0
        return np.vstack([pts[:k], pts[k - 1] + u * (pts[k] - pts[k - 1])])

    def world_bounds(self, state: SceneState) -> Bounds:
        pts = self._drawn(state)
        return Bounds.of_points(pts, pad=float(np.max(self._width(state, pts))) / 2)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pts = self._drawn(state)
        return {"start": pts[0], "end": pts[-1], "curve": pts}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        pts = np.asarray(self.me(state).params.points)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        out: dict[str, Any] = {"kind": "polyline", "points": len(pts), "length": length}
        if self.me(state).params.progress < 1:  # partly traced; 0 means nothing drawn yet
            out["drawn"] = round(self.me(state).params.progress, 4)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        pts = self._drawn(state)
        if me.params.progress <= 0:
            return
        polyline(ctx, cam, pts, self._width(state, pts), color=_rgba(me.style["color"],
                 me.params.opacity), glow=me.style["glow"])  # fmt: skip


@register
class Text(ObjectBase):
    """One line of text facing the camera, sized by a text role."""

    type: ClassVar[str] = "text"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "size": ("text", "label"),
        "leader": ("stroke", "hairline"),
    }

    class Params(ParamsBase):
        text: str = Field(min_length=1)
        pos: Vec3 = (0.0, 0.0, 0.0)
        anchor: Literal["center", "left", "right"] = "center"
        # Leader line from the text box to ``leader``, drawn with strength ``leader_on``.
        # The label placer sets both when it has to move a label away from what it names.
        leader: Vec3 = (0.0, 0.0, 0.0)
        leader_on: float = Field(0.0, ge=0, le=1)

    def _box(self, state: SceneState):
        """(lower-left corner, width vector, height vector, line height) in world space."""
        me = self.me(state)
        cam = state.camera_for(me)
        p = me.params
        h = me.style["size"] * cam.units_per_height(p.pos)
        return (*text_metrics.quad(p.text, p.pos, h, p.anchor, cam.right, cam.up), h)

    def world_bounds(self, state: SceneState) -> Bounds:
        corner, wv, hv, _ = self._box(state)
        return Bounds.of_points([corner, corner + wv, corner + hv, corner + wv + hv])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        corner, wv, hv, _ = self._box(state)
        return {"anchor": np.asarray(self.me(state).params.pos), "center": corner + (wv + hv) / 2}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "label", "text": self.me(state).params.text}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, wv, hv, h = self._box(state)
        color = _rgba(me.style["color"], p.opacity)
        ctx.text(p.text, np.asarray(p.pos), h, color=color, anchor=p.anchor)
        if p.leader_on > 0:
            cam = state.camera_for(me)
            target = np.asarray(p.leader)
            # Start at the point of the text box nearest the target, a little outside it.
            rel = target - corner
            u = np.clip(rel @ wv / (wv @ wv), 0, 1)
            v = np.clip(rel @ hv / (hv @ hv), 0, 1)
            start = corner + u * wv + v * hv
            along = target - start
            length = float(np.linalg.norm(along))
            if length > 1e-9:
                along /= length
                start = start + along * min(0.15 * h, length / 2)
                side = np.cross(cam.forward, along)
                half = side * (me.style["leader"] * cam.units_per_height(target) / 2)
                quad = np.array([start - half, start + half, target + half, target - half])
                ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]),
                              color=_rgba(color, p.leader_on))  # fmt: skip
