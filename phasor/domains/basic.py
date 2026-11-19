"""Built-in 2D primitives: dot, line, text.

They draw in the plane facing the camera (its ``right``/``up`` vectors), so the same code
serves the orthographic 2D case now and billboards in 3D later. Sizes that come from style
roles (stroke weight, text size) are fractions of the frame height, converted to world
units through the camera, so they look the same at any zoom.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

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
    """A polyline through two or more points, with mitered joins. With ``hue_along`` > 0 its
    color runs along the curve through the ``colormap`` (that many times over its length),
    shifted by ``hue``, which life slowly turns: the colors flow along the line."""

    type: ClassVar[str] = "line"
    Life: ClassVar = frozenset({"glow", "hue"})
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "spectrum"),
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
        hue_along: float = Field(0.0, ge=0)  # colormap cycles along the length; 0: one color
        hue: float = 0.0  # shifts the colors along the curve (a share of a cycle)

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
        colors = None
        p = me.params
        if p.hue_along > 0:
            lut = np.asarray(me.style["colormap"], dtype=np.float64)
            full = np.asarray(p.points, dtype=np.float64)
            total = max(float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum()), 1e-12)
            along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            u = (along / total * p.hue_along + p.hue) % 1.0
            g = u * len(lut)
            k0 = np.floor(g).astype(int) % len(lut)
            f = (g - np.floor(g))[:, None]
            rgb = lut[k0] * (1 - f) + lut[(k0 + 1) % len(lut)] * f
            colors = np.c_[rgb, np.ones(len(pts))]
        polyline(ctx, cam, pts, self._width(state, pts), color=_rgba(me.style["color"],
                 p.opacity), glow=me.style["glow"], colors=colors)  # fmt: skip


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

        @field_validator("text")
        @classmethod
        def _typeset(cls, v: str) -> str:
            text_metrics.check_math(v)  # "$...$" must be valid mathtext
            return v

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
        text = self.me(state).params.text
        out: dict[str, Any] = {"kind": "label", "text": text}
        if text_metrics.is_math(text):
            out["math"] = True
        return out

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


# ---------------------------------------------------------------- Equation


EMPHASIS = ("primary", "secondary", "muted", "highlight", "danger")
# Parts are typeset one by one, so TeX cannot space a relation that sits at a part's edge:
# "$E(x) =$" then "$\\psi$" would touch. Where a part ends or the next begins with one of
# these, ``Equation`` puts TeX's thick space (5/18 em) between them.
_RELATIONS = (
    "=",
    "<",
    ">",
    "+",
    "-",
    "\\approx",
    "\\sim",
    "\\le",
    "\\ge",
    "\\equiv",
    "\\to",
    "\\cdot",
    "\\times",
    "\\ne",
)
THICK = 5 / 18


def _spaced(a: str, b: str) -> bool:
    """Whether a relation sits where part ``a`` meets part ``b``."""
    return _edge(a, True).endswith(_RELATIONS) or _edge(b, False).startswith(_RELATIONS)


def _edge(part: str, end: bool) -> str:
    core = part.strip().strip("$").strip()
    return core.rstrip("\\ ").rstrip() if end else core


@register
class Equation(ObjectBase):
    """Typeset math in parts, each part its own color role, set on one baseline:
    ``parts=("$\\\\psi$", "$(x) \\\\approx x - \\\\ln 2\\\\pi - S(x)$")`` with
    ``colors=("secondary", "primary")``. A part is mathtext (``$...$``) or plain text. Life
    lets an equation float slowly (``[life] float``); a glow role makes it emit light."""

    type: ClassVar[str] = "equation"
    Life: ClassVar = frozenset({"float", "glow"})
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "size": ("text", "math"),
        "glow": ("glow", "none"),
        # every emphasis role, resolved by the theme, for the parts' ``colors``
        **{role: ("emphasis", role) for role in EMPHASIS},
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        parts: tuple[str, ...] = Field(min_length=1)
        colors: tuple[str, ...] = ()  # an emphasis role per part; missing ones use ``color``
        pos: Vec3 = (0.0, 0.0, 0.0)  # center of the equation, on its baseline's middle

        @field_validator("parts")
        @classmethod
        def _typeset(cls, v):
            for part in v:
                text_metrics.check_math(part)
            return v

        @field_validator("colors")
        @classmethod
        def _roles(cls, v):
            bad = [c for c in v if c not in EMPHASIS]
            if bad:
                raise ValueError(f"colors are emphasis roles {EMPHASIS}, got {bad[0]!r}")
            return v

    def _layout(self, state: SceneState):
        """[(part, origin, line height)] left to right on a common baseline, and the box."""
        me = self.me(state)
        cam = state.camera_for(me)
        p = me.params
        h = me.style["size"] * cam.units_per_height(p.pos)
        widths = [text_metrics.aspect(t) * h for t in p.parts]
        em = h * text_metrics.MATH_EM
        gaps = [0.0] + [
            THICK * em if _spaced(a, b) else 0.0 for a, b in zip(p.parts, p.parts[1:], strict=False)
        ]
        total = sum(widths) + sum(gaps)
        x = -total / 2
        out = []
        o = np.asarray(p.pos, dtype=np.float64)
        top = bottom = 0.0
        for part, w, gap in zip(p.parts, widths, gaps, strict=True):
            x += gap
            hb, d = text_metrics.box(part) * h, text_metrics.depth(part) * h
            lift = hb / 2 - d  # from the baseline up to the middle of this part's box
            out.append((part, o + cam.right * x + cam.up * lift, h))
            top, bottom = max(top, hb - d), max(bottom, d)
            x += w
        return out, h, total, top, bottom

    def world_bounds(self, state: SceneState) -> Bounds:
        me = self.me(state)
        cam = state.camera_for(me)
        _, _, width, top, bottom = self._layout(state)
        o = np.asarray(me.params.pos, dtype=np.float64)
        corners = [o + cam.right * sx * width / 2 + cam.up * sy
                   for sx in (-1, 1) for sy in (-bottom, top)]  # fmt: skip
        return Bounds.of_points(corners)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {
            "kind": "equation",
            "text": "".join(p.parts),
            "parts": list(p.parts),
            "colors": [p.colors[i] if i < len(p.colors) else None for i in range(len(p.parts))],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        for i, (part, origin, h) in enumerate(self._layout(state)[0]):
            color = me.style[p.colors[i]] if i < len(p.colors) else me.style["color"]
            ctx.text(part, origin, h, color=_rgba(color, p.opacity), anchor="left",
                     glow=me.style["glow"])  # fmt: skip


# ---------------------------------------------------------------- Haze


@register
class Haze(ObjectBase):
    """Soft light behind everything: out-of-focus vertical shafts of the ``color`` role,
    brightest low in the frame and fading upward, one of them brighter than the rest. A
    backdrop (labels may sit on it). Positions come from ``rng(id, seed)``; life sways them
    (``drift``). Keep ``intensity`` low: it is atmosphere, not a subject."""

    type: ClassVar[str] = "haze"
    Life: ClassVar = frozenset({"drift"})
    Style: ClassVar = {"color": ("emphasis", "secondary")}

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, -0.5)  # center (a little behind z = 0)
        size: tuple[float, float] = (16.0, 9.0)
        shafts: int = Field(14, ge=1, le=64)
        intensity: float = Field(0.15, ge=0, le=2)
        softness: float = Field(0.5, gt=0, le=1)  # blur of a shaft, a share of its spacing
        seed: int = 0
        drift: float = 0.0  # shifts the shafts sideways, a share of the width (life sways it)

    def _shafts(self, p):
        from phasor.core.rng import rng

        g = rng(self.id, p.seed)
        n = p.shafts
        x = (np.arange(n) + g.uniform(0.1, 0.9, n)) / n  # spread out, a little uneven
        w = g.uniform(0.08, 0.3, n) / n  # narrow: dark gaps between the shafts
        b = g.uniform(0.15, 0.6, n)
        b[int(g.integers(0, max(n // 3, 1)))] = 1.0  # one shaft is the light source
        return x, w, b

    def _rect(self, p):
        c = np.asarray(p.pos, dtype=np.float64)
        corner = c - np.array([p.size[0] / 2, p.size[1] / 2, 0.0])
        return corner, np.array([p.size[0], 0.0, 0.0]), np.array([0.0, p.size[1], 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        corner, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([corner, corner + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "haze", "backdrop": True, "shafts": p.shafts,
                "intensity": p.intensity}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        xs, ws, bs = self._shafts(p)
        rgb = np.asarray(me.style["color"][:3], dtype=np.float32)
        bg = np.asarray(state.background[:3], dtype=np.float32)
        sx, sy = p.size

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            half_w = height * pw / ph / 2
            u = (center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w) / sx
            v = (center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height) / sy  # 0 bottom
            light = np.zeros(pw, dtype=np.float64)
            for x, w, b in zip(xs, ws, bs, strict=True):
                d = (u - (x + p.drift) % 1.0) / (w * (0.5 + p.softness))
                light += b * np.exp(-0.5 * d * d)
            fall = np.clip(1.0 - v, 0, 1) ** 2 * 0.9 + 0.1  # strongest low in the frame
            k = (p.intensity * light[None, :] * fall[:, None]).astype(np.float32)
            out = np.ones((ph, pw, 4), dtype=np.float32)
            out[..., :3] = bg + k[..., None] * rgb
            return out

        corner, r, u = self._rect(p)
        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity)
