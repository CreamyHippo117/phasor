"""Light that points: a spreading ripple, a spotlight, and the probe the camera follows.

- ``Ripple``: rings of light that spread from a spot on another object (one of its
  features: a zero, a pole, a tracer's head) or from a point, and fade. The ``ripple`` verb
  adds one (``<id>.ripple``) and runs its ``phase`` from 0 to 1.
- ``Spotlight``: a soft pool of light over part of the frame, the rest under a dark veil.
  It sits on the overlay, so it dims fields too (which ``dim_others`` leaves alone). The
  ``spotlight`` verb reveals it and glides it from one target to the next.
- ``Follow``: an invisible probe on a feature of an object. While its weight ``on`` is
  above 0 the layout bake moves the camera's center (2D) or target (3D) onto the feature
  (``layout/camera.py`` ``follow_keys``); the ``follow`` verb adds one and animates ``on``.

All three are pure in (params, state): the spot is found from the target's features each
time the scene is evaluated, so a ripple or the camera rides a zero that moves.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains.strokes import polyline

RING_SEGMENTS = 96
RIPPLE_LAG = 0.3  # each ring starts this share of the ripple after the one before it
VEIL_SEGMENTS = 96
VEIL_STEPS = 8  # rings across the spotlight's soft edge


def feature_point(state: SceneState, target: str, feature: str, index: int) -> np.ndarray | None:
    """The world point ``index`` of ``target``'s feature ``feature``, or None when there is
    no such object, feature or point at this time."""
    if target not in state:
        return None
    feats = state[target].impl.features(state)
    if feature not in feats:
        return None
    pts = np.asarray(feats[feature], dtype=np.float64)
    if pts.size == 0:
        return None
    pts = pts.reshape(-1, 3) if pts.shape[-1] == 3 else np.atleast_2d(pts)
    if not 0 <= index < len(pts):
        return None
    return pts[index]


def _circle(center, radius: float, right, up) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS + 1)
    return center + np.outer(radius * np.cos(a), right) + np.outer(radius * np.sin(a), up)


@register
class Ripple(ObjectBase):
    """Rings of light spreading from a spot and fading: "it happens here". The spot is
    point ``index`` of ``target``'s feature ``feature``, or ``point`` when ``feature`` is
    empty."""

    type: ClassVar[str] = "ripple"
    Life: ClassVar = frozenset()  # its timing is the verb's
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(ParamsBase):
        target: str = ""
        feature: str = "center"
        index: int = Field(0, ge=0)
        point: Vec3 = (0.0, 0.0, 0.0)  # the spot when ``feature`` is ""
        phase: float = Field(1.0, ge=0, le=1)  # 0: not started, 1: every ring gone
        rings: int = Field(3, ge=1, le=8)
        reach: float = Field(0.2, gt=0)  # how far a ring spreads, in frame heights

    def spot(self, state: SceneState) -> np.ndarray | None:
        p = self.me(state).params
        if not p.feature:
            return np.asarray(p.point, dtype=np.float64)
        return feature_point(state, p.target, p.feature, p.index)

    def _radius(self, state: SceneState, spot) -> float:
        me = self.me(state)
        return float(me.params.reach * state.camera_for(me).units_per_height(spot))

    def rings_now(self, state: SceneState) -> list[tuple[float, float]]:
        """(share of the reach, alpha) of every ring visible at this phase."""
        p = self.me(state).params
        span = 1.0 + RIPPLE_LAG * (p.rings - 1)
        out = []
        for k in range(p.rings):
            u = p.phase * span - k * RIPPLE_LAG
            if 0.0 < u < 1.0:
                out.append((1.0 - (1.0 - u) ** 2, (1.0 - u) ** 1.5))
        return out

    def world_bounds(self, state: SceneState) -> Bounds:
        spot = self.spot(state)
        if spot is None:
            return Bounds.of_points([(0.0, 0.0, 0.0)])
        return Bounds.of_points([spot], pad=self._radius(state, spot))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        spot = self.spot(state)
        return {} if spot is None else {"center": spot}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        spot = self.spot(state)
        return {"kind": "ripple", "target": p.target, "feature": p.feature,
                "rings_visible": len(self.rings_now(state)),
                **({"spot_missing": True} if spot is None else
                   {"spot": [round(float(v), 6) for v in spot]})}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        spot = self.spot(state)
        if spot is None or me.params.opacity <= 0:
            return
        cam = state.camera_for(me)
        reach = self._radius(state, spot)
        color = me.style["color"]
        for share, alpha in self.rings_now(state):
            pts = _circle(spot, max(share * reach, 1e-6), cam.right, cam.up)
            a = color[3] * me.params.opacity * alpha
            polyline(ctx, cam, pts, me.style["stroke"] * cam.units_per_height(pts),
                     color=(color[0], color[1], color[2], a), glow=me.style["glow"])  # fmt: skip


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@register
class Spotlight(ObjectBase):
    """A pool of light: the circle around ``center`` (``radius`` world units, with a soft
    edge ``soft`` times as wide) at normal brightness, everything else under a dark veil
    (``strength``: how much light the veil takes away). Put it on the overlay; it then dims
    fields and surfaces as well."""

    type: ClassVar[str] = "spotlight"
    Life: ClassVar = frozenset()
    Style: ClassVar = {}

    class Params(ParamsBase):
        center: Vec3 = (0.0, 0.0, 0.0)  # in world units, wherever the spotlight is drawn
        radius: float = Field(1.0, gt=0)
        soft: float = Field(0.6, ge=0)
        strength: float = Field(0.7, ge=0, le=1)

    def _pool(self, state: SceneState) -> tuple[np.ndarray, float, float]:
        """The pool's center, inner radius and outer radius in the spotlight's own layer."""
        me = self.me(state)
        p = me.params
        world = state.camera
        c = np.asarray(p.center, dtype=np.float64)
        r = float(p.radius / world.units_per_height(c))  # frame heights
        if me.layer == "overlay":
            ndc = world.project(c[None])[0]
            center = np.array([(ndc[0] + 1) / 2 * state.aspect, (1 - ndc[1]) / 2, 0.0])
            return center, r, r * (1.0 + p.soft)
        scale = float(world.units_per_height(c))
        return c, r * scale, r * scale * (1.0 + p.soft)

    def world_bounds(self, state: SceneState) -> Bounds:
        me = self.me(state)
        if me.layer == "overlay":  # the veil covers the frame
            return Bounds((0.0, 0.0, 0.0), (float(state.aspect), 1.0, 0.0))
        center, _, outer = self._pool(state)
        return Bounds.of_points([center], pad=outer)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": self._pool(state)[0]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        # A backdrop: labels may sit on it, the camera and dim_others leave it alone, and it
        # hides nothing (it draws only in the color pass).
        return {"kind": "spotlight", "backdrop": True,
                "center": [round(float(v), 6) for v in p.center],
                "radius": round(float(p.radius), 6),
                "strength": round(float(p.strength), 6)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if pass_ != "color" or p.opacity <= 0 or p.strength <= 0:
            return
        center, inner, outer = self._pool(state)
        cam = state.camera_for(me)
        right, up = np.asarray(cam.right), np.asarray(cam.up)
        unit = 1.0 if me.layer == "overlay" else float(state.camera.units_per_height(center))
        far = (float(state.aspect) + 1.0) * unit
        radii = np.r_[np.linspace(inner, max(outer, inner * (1 + 1e-3)), VEIL_STEPS + 1),
                      max(far + outer, outer * 2)]  # fmt: skip
        alpha = np.r_[_smoothstep(np.linspace(0.0, 1.0, VEIL_STEPS + 1)), 1.0]
        ang = np.linspace(0.0, 2.0 * np.pi, VEIL_SEGMENTS, endpoint=False)
        ring = np.outer(np.cos(ang), right) + np.outer(np.sin(ang), up)  # (S, 3)
        pos = (center[None, None] + radii[:, None, None] * ring[None]).reshape(-1, 3)
        colors = np.zeros((len(pos), 4))
        colors[:, 3] = np.repeat(alpha, VEIL_SEGMENTS)
        s = VEIL_SEGMENTS
        k = np.arange(s)
        quads = []
        for i in range(len(radii) - 1):
            a, b = i * s + k, i * s + (k + 1) % s
            c, d = a + s, b + s
            quads += [np.stack([a, b, d], 1), np.stack([a, d, c], 1)]
        ctx.triangles(pos, np.concatenate(quads), color=(0.0, 0.0, 0.0, p.strength * p.opacity),
                      colors=colors)  # fmt: skip


@register
class Follow(ObjectBase):
    """What the camera follows: point ``index`` of ``target``'s feature ``feature``. While
    ``on`` is above 0 the layout bake moves the camera's center (2D) or target (3D) that
    share of the way onto the point. Draws nothing; the ``follow`` verb adds one."""

    type: ClassVar[str] = "follow"
    Life: ClassVar = frozenset()
    Style: ClassVar = {}

    class Params(ParamsBase):
        target: str = Field(min_length=1)
        feature: str = "center"
        index: int = Field(0, ge=0)
        on: float = Field(0.0, ge=0, le=1)

    def world_bounds(self, state: SceneState) -> Bounds:
        me = self.me(state)
        p = me.params
        spot = feature_point(state, p.target, p.feature, p.index)
        if spot is None:
            return Bounds.of_points([(0.0, 0.0, 0.0)])
        # a speck around the point (a box of no size is nowhere on screen)
        return Bounds.of_points([spot], pad=0.01 * state.camera_for(me).units_per_height(spot))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        spot = feature_point(state, p.target, p.feature, p.index)
        return {} if spot is None else {"center": spot}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "follow", "backdrop": True, "target": p.target, "feature": p.feature,
                "index": p.index, "on": round(float(p.on), 6)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        """Nothing to see: the camera's motion is what shows it."""
