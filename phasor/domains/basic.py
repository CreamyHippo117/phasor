"""Built-in 2D primitives: dot, line, text.

They draw in the plane facing the camera (its ``right``/``up`` vectors), so the same code
serves the orthographic 2D case now and billboards in 3D later. Sizes that come from style
roles (stroke weight, text size) are fractions of the frame height, converted to world
units through the camera, so they look the same at any zoom.
"""

from __future__ import annotations

import itertools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core import text as text_metrics
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.texconv import join_latex, to_latex
from phasor.core.values import Vec3
from phasor.domains.strokes import hue_colors, polyline

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
        disc(ctx, p.pos, p.r, right, up, _rgba(me.style["color"], p.opacity), me.style["glow"])


def disc(ctx: DrawContext, center, r: float, right, up, color, glow: float = 1.0) -> None:
    """A filled disc facing the camera (a dot, a tracer's head)."""
    a = np.linspace(0, 2 * np.pi, DISC_SEGMENTS, endpoint=False)
    c = np.asarray(center, dtype=np.float64)
    rim = c + r * (np.outer(np.cos(a), right) + np.outer(np.sin(a), up))
    positions = np.vstack([c[None, :], rim])
    k = np.arange(DISC_SEGMENTS)
    indices = np.c_[np.zeros_like(k), 1 + k, 1 + (k + 1) % DISC_SEGMENTS]
    ctx.triangles(positions, indices, color=color, glow=glow)


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

    def path_points(self, state: SceneState) -> np.ndarray:
        """The whole curve as points (what a ``Tracer`` travels along)."""
        return np.asarray(self.me(state).params.points, dtype=np.float64)

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
            full = np.asarray(p.points, dtype=np.float64)
            total = float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum())
            colors = hue_colors(pts, me.style["colormap"], p.hue_along, p.hue, total)
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
        write: float = Field(1.0, ge=0, le=1)  # share written in, left to right (``write``)

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
            out["latex"] = to_latex(text)  # the same maths as LaTeX source, to show or export
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, wv, hv, h = self._box(state)
        color = _rgba(me.style["color"], p.opacity)
        ctx.text(p.text, np.asarray(p.pos), h, color=color, anchor=p.anchor, reveal=p.write)
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
        write: float = Field(1.0, ge=0, le=1)  # share written in, left to right (``write``)
        # Becoming this equation from another (the ``transform`` verb): its parts, colors and
        # place, and how far along (1: done). Parts that match glide over; the rest fade.
        from_parts: tuple[str, ...] = ()
        from_colors: tuple[str, ...] = ()
        from_pos: Vec3 = (0.0, 0.0, 0.0)
        morph: float = Field(1.0, ge=0, le=1)

        @field_validator("parts", "from_parts")
        @classmethod
        def _typeset(cls, v):
            for part in v:
                text_metrics.check_math(part)
            return v

        @field_validator("colors", "from_colors")
        @classmethod
        def _roles(cls, v):
            bad = [c for c in v if c not in EMPHASIS]
            if bad:
                raise ValueError(f"colors are emphasis roles {EMPHASIS}, got {bad[0]!r}")
            return v

    def _layout(self, state: SceneState, parts=None, pos=None):
        """[(part, origin, line height)] left to right on a common baseline, and the box:
        of this equation, or of ``parts`` at ``pos`` set the same way (a transform's source)."""
        me = self.me(state)
        cam = state.camera_for(me)
        p = me.params
        parts = p.parts if parts is None else parts
        pos = p.pos if pos is None else pos
        h = me.style["size"] * cam.units_per_height(pos)
        widths = [text_metrics.aspect(t) * h for t in parts]
        em = h * text_metrics.MATH_EM
        gaps = [0.0] + [THICK * em if _spaced(a, b) else 0.0 for a, b in itertools.pairwise(parts)]
        total = sum(widths) + sum(gaps)
        x = -total / 2
        out = []
        o = np.asarray(pos, dtype=np.float64)
        top = bottom = 0.0
        for part, w, gap in zip(parts, widths, gaps, strict=True):
            x += gap
            hb, d = text_metrics.box(part) * h, text_metrics.depth(part) * h
            lift = hb / 2 - d  # from the baseline up to the middle of this part's box
            out.append((part, o + cam.right * x + cam.up * lift, h))
            top, bottom = max(top, hb - d), max(bottom, d)
            x += w
        return out, h, total, top, bottom

    def _corners(self, state: SceneState, parts=None, pos=None) -> list[np.ndarray]:
        me = self.me(state)
        cam = state.camera_for(me)
        _, _, width, top, bottom = self._layout(state, parts, pos)
        o = np.asarray(me.params.pos if pos is None else pos, dtype=np.float64)
        return [o + cam.right * sx * width / 2 + cam.up * sy
                for sx in (-1, 1) for sy in (-bottom, top)]  # fmt: skip

    def _transforming(self, p) -> bool:
        return bool(p.from_parts) and p.morph < 1.0

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        corners = self._corners(state)
        if self._transforming(p):  # mid-transform: both places
            corners += self._corners(state, p.from_parts, p.from_pos)
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
            "latex": join_latex(list(p.parts)),
            **(
                {
                    "transforming_from": list(p.from_parts),
                    "morph": round(p.morph, 6),
                    "matched": len(match_parts(p.from_parts, p.parts)),
                }
                if self._transforming(p)
                else {}
            ),
        }

    def _color(self, me, roles, i):
        return me.style[roles[i]] if i < len(roles) else me.style["color"]

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        glow = me.style["glow"]
        layout, _, total, _, _ = self._layout(state)
        if not self._transforming(p):
            x0 = layout[0][1] if layout else None  # the left edge, for writing in
            for i, (part, origin, h) in enumerate(layout):
                reveal = 1.0
                if p.write < 1.0 and total > 0:
                    w = text_metrics.aspect(part) * h
                    left = float((origin - x0) @ state.camera_for(me).right)
                    reveal = min(max((p.write * total - left) / max(w, 1e-12), 0.0), 1.0)
                ctx.text(part, origin, h, color=_rgba(self._color(me, p.colors, i), p.opacity),
                         anchor="left", glow=glow, reveal=reveal)  # fmt: skip
            return
        # Mid-transform: matched parts glide from their old place (and color) to the new;
        # the source's other parts fade out in the first half, this one's fade in after.
        m = p.morph
        before = self._layout(state, p.from_parts, p.from_pos)[0]
        pairs = dict((j, i) for i, j in match_parts(p.from_parts, p.parts))
        for j, (part, origin, h) in enumerate(layout):
            if j in pairs:
                i = pairs[j]
                at = before[i][1] * (1 - m) + origin * m
                a = np.asarray(self._color(me, p.from_colors, i), dtype=np.float64)
                b = np.asarray(self._color(me, p.colors, j), dtype=np.float64)
                ctx.text(part, at, h, color=_rgba(tuple(a * (1 - m) + b * m), p.opacity),
                         anchor="left", glow=glow)  # fmt: skip
            elif m > 0.5:
                ctx.text(part, origin, h, color=_rgba(self._color(me, p.colors, j),
                         p.opacity * (2 * m - 1)), anchor="left", glow=glow)  # fmt: skip
        gone = {i for i in pairs.values()}
        if m < 0.5:
            for i, (part, origin, h) in enumerate(before):
                if i not in gone:
                    ctx.text(part, origin, h, color=_rgba(self._color(me, p.from_colors, i),
                             p.opacity * (1 - 2 * m)), anchor="left", glow=glow)  # fmt: skip


def match_parts(a, b) -> list[tuple[int, int]]:
    """Pairs (i, j) of equal parts ``a[i] == b[j]`` in order: the longest common
    subsequence, so ``x = 2`` -> ``x + 1 = 3`` keeps ``x`` and ``=`` and fades the rest."""
    n, k = len(a), len(b)
    best = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            best[i][j] = best[i + 1][j + 1] + 1 if a[i] == b[j] else max(best[i + 1][j],
                                                                      best[i][j + 1])  # fmt: skip
    out, i, j = [], 0, 0
    while i < n and j < k:
        if a[i] == b[j]:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


# ---------------------------------------------------------------- motion helpers


def along(points: np.ndarray, share: float) -> tuple[np.ndarray, int]:
    """The point a ``share`` (0..1) of the way along a polyline by arc length, and the index
    of the last vertex before it."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    stop = min(max(share, 0.0), 1.0) * cum[-1]
    k = int(np.clip(np.searchsorted(cum, stop, side="right") - 1, 0, len(pts) - 2))
    u = (stop - cum[k]) / seg[k] if seg[k] > 0 else 0.0
    return pts[k] + u * (pts[k + 1] - pts[k]), k


def _path_of(state: SceneState, object_id: str) -> np.ndarray | None:
    """The polyline another object offers to travel along (``path_points``), if any."""
    if object_id not in state:
        return None
    impl = state[object_id].impl
    fn = getattr(impl, "path_points", None)
    pts = None if fn is None else np.asarray(fn(state), dtype=np.float64)
    return pts if pts is not None and len(pts) >= 2 else None


@register
class Tracer(ObjectBase):
    """A glowing point travelling along another curve (``path``: the id of a line, graph,
    parametric curve or partial sum), leaving a fading trail. ``progress`` is how far along
    (the ``travel`` verb animates it); ``trail`` is the trail's length as a share of the path."""

    type: ClassVar[str] = "tracer"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "stroke": ("stroke", "bold"),
        "glow": ("glow", "soft"),
    }
    Presets: ClassVar = {"comet": {"params": {"trail": 0.3}, "style": {"glow": "strong"}}}

    class Params(ParamsBase):
        path: str = Field(min_length=1)  # the id of the curve to travel along
        progress: float = Field(0.0, ge=0, le=1)
        trail: float = Field(0.15, ge=0, le=1)
        r: float = Field(0.08, gt=0)

    def _tip(self, state: SceneState):
        p = self.me(state).params
        pts = _path_of(state, p.path)
        if pts is None:
            return None, None, None
        tip, k = along(pts, p.progress)
        return pts, tip, k

    def world_bounds(self, state: SceneState) -> Bounds:
        pts, tip, _ = self._tip(state)
        if pts is None:
            return Bounds.of_points([(0.0, 0.0, 0.0)])
        return Bounds.of_points([tip], pad=self.me(state).params.r)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        _, tip, _ = self._tip(state)
        return {} if tip is None else {"center": tip, "tip": tip}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        _, tip, _ = self._tip(state)
        out: dict[str, Any] = {"kind": "tracer", "path": p.path, "progress": round(p.progress, 6)}
        if tip is None:
            out["path_missing"] = True  # no such curve, or it offers no path
        else:
            out["position"] = [round(float(v), 6) for v in tip]
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        pts, tip, k = self._tip(state)
        if pts is None or p.opacity <= 0:
            return
        cam = state.camera_for(me)
        color = _rgba(me.style["color"], p.opacity)
        if p.trail > 0 and p.progress > 0:
            tail, k0 = along(pts, max(p.progress - p.trail, 0.0))
            body = np.vstack([tail[None, :], pts[k0 + 1 : k + 1], tip[None, :]])
            if len(body) >= 2:
                fade = (
                    np.linspace(0.0, 1.0, len(body)) ** 1.5
                )  # faint at the end, bright at the head
                cols = np.c_[np.tile(np.asarray(color[:3]), (len(body), 1)), fade]
                w = me.style["stroke"] * cam.units_per_height(body)
                polyline(ctx, cam, body, w, color=color, glow=me.style["glow"], colors=cols)
        disc(ctx, tip, p.r, cam.right, cam.up, color, me.style["glow"])


@register
class Callout(ObjectBase):
    """A ring (or box) that draws itself around another object (``target``) to point at it:
    the ``circumscribe`` verb adds one, draws it and fades it."""

    type: ClassVar[str] = "callout"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(ParamsBase):
        target: str = Field(min_length=1)
        shape: Literal["circle", "box"] = "circle"
        pad: float = Field(0.2, ge=0)  # room around the target, as a share of its size
        progress: float = Field(1.0, ge=0, le=1)

    def _outline(self, state: SceneState) -> np.ndarray | None:
        me = self.me(state)
        p = me.params
        if p.target not in state:
            return None
        cam = state.camera_for(me)
        corners = np.asarray(state[p.target].impl.world_bounds(state).corners(), dtype=np.float64)
        c = corners.mean(axis=0)
        x, y = (corners - c) @ cam.right, (corners - c) @ cam.up
        hx = max(float(np.abs(x).max()), 1e-3) * (1 + p.pad)
        hy = max(float(np.abs(y).max()), 1e-3) * (1 + p.pad)
        if p.shape == "circle":
            a = np.linspace(0.5 * np.pi, 2.5 * np.pi, 97)  # from the top, counterclockwise
            r = np.hypot(hx, hy) * 0.85
            rx, ry = max(r, hx * 1.05), max(r * 0.75, hy * 1.05)
            return c + np.outer(rx * np.cos(a), cam.right) + np.outer(ry * np.sin(a), cam.up)
        box = [(-hx, hy), (-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        return np.array([c + bx * cam.right + by * cam.up for bx, by in box])

    def world_bounds(self, state: SceneState) -> Bounds:
        pts = self._outline(state)
        return Bounds.of_points([(0.0, 0.0, 0.0)] if pts is None else pts)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pts = self._outline(state)
        return {} if pts is None else {"center": pts.mean(axis=0)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "callout", "target": p.target, "shape": p.shape,
                **({} if p.target in state else {"target_missing": True})}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        pts = self._outline(state)
        if pts is None or p.progress <= 0 or p.opacity <= 0:
            return
        if p.progress < 1:
            end, k = along(pts, p.progress)
            pts = np.vstack([pts[: k + 1], end[None, :]])
        cam = state.camera_for(me)
        polyline(ctx, cam, pts, me.style["stroke"] * cam.units_per_height(pts),
                 color=_rgba(me.style["color"], p.opacity), glow=me.style["glow"])  # fmt: skip


@register
class Counter(ObjectBase):
    """A number on screen that follows a value: ``value=expr("wave.terms")`` shows the
    number of terms as it changes; ``prefix``/``suffix`` around it (maths when ``math``).
    The ``sweep`` verb adds one beside what it sweeps."""

    type: ClassVar[str] = "counter"
    Style: ClassVar = {"color": ("emphasis", "primary"), "size": ("text", "label"),
                       "glow": ("glow", "none")}  # fmt: skip

    class Params(ParamsBase):
        value: float = 0.0
        digits: int = Field(2, ge=0, le=10)
        prefix: str = ""
        suffix: str = ""
        math: bool = True  # typeset as maths: "$" + prefix + number + suffix + "$"
        pos: Vec3 = (0.0, 0.0, 0.0)
        anchor: Literal["center", "left", "right"] = "center"

        @model_validator(mode="after")
        def _typeset(self):
            if self.math:
                text_metrics.check_math(f"${self.prefix}0{self.suffix}$")
            return self

    @staticmethod
    def text_of(p) -> str:
        num = f"{p.value:.{p.digits}f}"
        if float(num) == 0:
            num = num.lstrip("-")
        return f"${p.prefix}{num}{p.suffix}$" if p.math else f"{p.prefix}{num}{p.suffix}"

    def _box(self, state: SceneState):
        me = self.me(state)
        cam = state.camera_for(me)
        p = me.params
        h = me.style["size"] * cam.units_per_height(p.pos)
        return (*text_metrics.quad(self.text_of(p), p.pos, h, p.anchor, cam.right, cam.up), h)

    def world_bounds(self, state: SceneState) -> Bounds:
        corner, wv, hv, _ = self._box(state)
        return Bounds.of_points([corner, corner + wv, corner + hv, corner + wv + hv])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        corner, wv, hv, _ = self._box(state)
        return {"anchor": np.asarray(self.me(state).params.pos), "center": corner + (wv + hv) / 2}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "label", "text": self.text_of(p), "value": p.value, "counter": True}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        _, _, _, h = self._box(state)
        ctx.text(self.text_of(p), np.asarray(p.pos), h, color=_rgba(me.style["color"], p.opacity),
                 anchor=p.anchor, glow=me.style["glow"])  # fmt: skip
