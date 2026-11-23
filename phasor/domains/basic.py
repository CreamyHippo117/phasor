"""Built-in 2D primitives: dot, line, text.

They draw in the plane facing the camera (its ``right``/``up`` vectors), so the same code
serves the orthographic 2D case now and billboards in 3D later. Sizes that come from style
roles (stroke weight, text size) are fractions of the frame height, converted to world
units through the camera, so they look the same at any zoom.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core import text as text_metrics
from phasor.core import ticks
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.texconv import join_latex, to_latex
from phasor.core.values import Vec3
from phasor.domains import glyph_math as gm
from phasor.domains import morph_math as mm
from phasor.domains.path import MAX_COORD, MorphParams, Path
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

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
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

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
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
    """One line of text facing the camera, sized by a text role. With ``as_shapes`` it is
    drawn as glyph outlines, one ``Path`` part per glyph (``<id>.<glyph>``), filled
    (``fill_opacity``) and stroked (``stroke_opacity``, ``OUTLINE`` of its height); ``write``
    draws each glyph's outline, then fills it. It can become another outline, or be
    becoming one (``morph_from``, the ``transform`` verbs)."""

    type: ClassVar[str] = "text"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "size": ("text", "label"),
        "leader": ("stroke", "hairline"),
    }

    class Params(MorphParams):
        text: str = Field(min_length=1)
        pos: Vec3 = (0.0, 0.0, 0.0)
        anchor: Literal["center", "left", "right"] = "center"
        # Leader line from the text box to ``leader``, drawn with strength ``leader_on``.
        # The label placer sets both when it has to move a label away from what it names.
        leader: Vec3 = (0.0, 0.0, 0.0)
        leader_on: float = Field(0.0, ge=0, le=1)
        write: float = Field(1.0, ge=0, le=1)  # share written in, left to right (``write``)
        as_shapes: bool = False  # glyph outlines, one Path part per glyph (not a picture)
        fill_opacity: float = Field(1.0, ge=0, le=1)  # as shapes: the glyphs' fill
        stroke_opacity: float = Field(0.0, ge=0, le=1)  # as shapes: their outline, once written

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
        box = Bounds.of_points([corner, corner + wv, corner + hv, corner + wv + hv])
        return _with_morph(self, state, box)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        corner, wv, hv, _ = self._box(state)
        return {"anchor": np.asarray(self.me(state).params.pos), "center": corner + (wv + hv) / 2}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        text = p.text
        out: dict[str, Any] = {"kind": "label", "text": text}
        if text_metrics.is_math(text):
            out["math"] = True
            out["latex"] = to_latex(text)  # the same maths as LaTeX source, to show or export
        out.update(_shape_facts(state, self.me(state), [text]))
        return out

    def morph_shape(self, state: SceneState) -> mm.Shape | None:
        """Its glyphs as outlines in world space, one group per glyph (``domains/morph``);
        None when it has no outline (``glyph_math.Outline.error``)."""
        me = self.me(state)
        outline = gm.outline(me.params.text)
        if outline.error:
            return None
        corner, _, _, h = self._box(state)
        look = _glyph_look(me, me.style["color"])
        return _glyph_shape([(outline, corner, look, None, "")], h, state.camera_for(me))

    def build(self, state: SceneState) -> list[ObjectBase]:
        """As shapes: a ``Path`` part per glyph (none while it is becoming something)."""
        me = self.me(state)
        if not me.params.as_shapes or _becoming(state, me):
            return []
        return _glyph_parts(state, me, _outlined(me, self.morph_shape(state), [me.params.text]))

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, wv, hv, h = self._box(state)
        color = _rgba(me.style["color"], p.opacity)
        becoming = _becoming(state, me)
        morphed = becoming and _draw_morph(ctx, state, self)
        if not morphed and (becoming or not p.as_shapes):  # as shapes, its parts draw glyphs
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
MAX_ARC = 10.0  # line heights a part may arc by in a transform (the theme's is about 1)
# Relations mathtext sets with a space before them even alone (measured: about 0.2 em of
# left margin); a lone "+", "-", "\cdot" or "\times" is set tight, like a sign.
_SET_APART = tuple(r for r in _RELATIONS if r not in ("+", "-", "\\cdot", "\\times"))


def _spaced(a: str, b: str) -> bool:
    """Whether a relation sits where part ``a`` meets part ``b``. A part that is only a
    relation (``"$=$"``, kept apart so ``transform`` can carry terms across it) is already
    set with TeX's space before it, so it takes the extra space only after it."""
    if _edge(b, False) in _SET_APART:
        return False
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

    class Params(MorphParams):
        parts: tuple[str, ...] = Field(min_length=1)
        colors: tuple[str, ...] = ()  # an emphasis role per part; missing ones use ``color``
        pos: Vec3 = (0.0, 0.0, 0.0)  # center of the equation, on its baseline's middle
        write: float = Field(1.0, ge=0, le=1)  # share written in, left to right (``write``)
        drift: float = Field(1.0, ge=0)  # share of the theme's ``[life] float`` it drifts
        # A name per part ("" for none): in a ``transform`` the part with the same key in the
        # source becomes this one wherever it was, even with other text ("+a" -> "-a" moving
        # across the "="), and the text changes as it travels.
        keys: tuple[str, ...] = ()
        # Becoming this equation from another (the ``transform`` verb): its parts, colors,
        # keys and place, and how far along (1: done). Parts that match glide over (a part
        # carried past others arcs ``arc`` line heights round them); the rest fade.
        from_parts: tuple[str, ...] = ()
        from_colors: tuple[str, ...] = ()
        from_keys: tuple[str, ...] = ()
        from_pos: Vec3 = (0.0, 0.0, 0.0)
        from_id: str = ""  # the source: life's drift passes from its to this one's, no jump
        morph: float = Field(1.0, ge=0, le=1)
        arc: float = Field(0.0, ge=0, le=MAX_ARC)
        as_shapes: bool = False  # glyph outlines, one Path part per glyph (not pictures)
        fill_opacity: float = Field(1.0, ge=0, le=1)  # as shapes: the glyphs' fill
        stroke_opacity: float = Field(0.0, ge=0, le=1)  # as shapes: their outline, once written

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
        return bool(p.from_parts) and p.morph < 1.0 and not p.morph_from

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        corners = self._corners(state)
        if self._transforming(p):  # mid-transform: both places
            corners += self._corners(state, p.from_parts, p.from_pos)
        return _with_morph(self, state, Bounds.of_points(corners))

    def morph_shape(self, state: SceneState) -> mm.Shape | None:
        """Its glyphs as outlines in world space, one group per glyph, each knowing its part
        (its TeX and key: ``transform_matching`` pairs equations' parts by TeX); None when
        a part has no outline (``glyph_math.Outline.error``)."""
        me = self.me(state)
        p = me.params
        layout, h, _, _, _ = self._layout(state)
        cam = state.camera_for(me)
        up = np.asarray(cam.up, dtype=np.float64)
        pieces = []
        glyphs = 0
        for i, (part, origin, hh) in enumerate(layout):
            if glyphs >= gm.MAX_GLYPHS:  # every part together keeps MAX_GLYPHS
                break
            outline = gm.outline(part)
            if outline.error:
                return None
            glyphs += len(outline.glyphs)
            corner = np.asarray(origin) - up * (text_metrics.box(part) * hh / 2)
            look = _glyph_look(me, self._color(me, p.colors, i))
            key = p.keys[i] if i < len(p.keys) else ""
            pieces.append((outline, corner, look, (i, part), key))
        return _glyph_shape(pieces, h, cam)

    def build(self, state: SceneState) -> list[ObjectBase]:
        """As shapes: a ``Path`` part per glyph (none while it is becoming something)."""
        me = self.me(state)
        if not me.params.as_shapes or _becoming(state, me) or self._transforming(me.params):
            return []
        return _glyph_parts(state, me, _outlined(me, self.morph_shape(state), me.params.parts))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {
            **_shape_facts(state, self.me(state), p.parts),
            "kind": "equation",
            "text": "".join(p.parts),
            "parts": list(p.parts),
            "colors": [p.colors[i] if i < len(p.colors) else None for i in range(len(p.parts))],
            "latex": join_latex(list(p.parts)),
            **(
                {
                    "transforming_from": list(p.from_parts),
                    "morph": round(p.morph, 6),
                    "matched": len(pairs := pair_parts(p.from_parts, p.parts, p.from_keys, p.keys)),
                    "changed": sum(p.from_parts[i] != p.parts[j] for i, j in pairs),
                    "carried_over": sum(crossing(pairs, p.from_parts, p.parts)),
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
        if _becoming(state, me):
            if _draw_morph(ctx, state, self, arc=p.arc):
                return
        elif p.as_shapes and not self._transforming(p):
            return  # its parts draw the glyphs
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
        matched = pair_parts(p.from_parts, p.parts, p.from_keys, p.keys)
        over = crossing(matched, p.from_parts, p.parts)
        pairs = dict((j, i) for i, j in matched)
        cam = state.camera_for(me)
        right, up = np.asarray(cam.right), np.asarray(cam.up)
        for k, (i, j) in enumerate(matched):
            part, origin, h = layout[j]
            old = p.from_parts[i]
            # centers glide (parts of different widths stay centered on each other)
            wa, wb = text_metrics.aspect(old) * h, text_metrics.aspect(part) * h
            c0, c1 = before[i][1] + right * wa / 2, origin + right * wb / 2
            center = c0 * (1 - m) + c1 * m
            if over[k]:  # carried past other parts: over them going right, under going left
                side = 1.0 if float((c1 - c0) @ right) >= 0 else -1.0
                center = center + up * (side * p.arc * h * math.sin(math.pi * m))
            a = np.asarray(self._color(me, p.from_colors, i), dtype=np.float64)
            b = np.asarray(self._color(me, p.colors, j), dtype=np.float64)
            color = tuple(a * (1 - m) + b * m)
            if old == part:
                ctx.text(part, center - right * wb / 2, h, color=_rgba(color, p.opacity),
                         anchor="left", glow=glow)  # fmt: skip
                continue
            u = min(max((m - 0.2) / 0.6, 0.0), 1.0)  # the text changes mid-way, as it travels
            u = u * u * (3 - 2 * u)
            if u < 1:
                ctx.text(old, center - right * wa / 2, h, color=_rgba(color, p.opacity * (1 - u)),
                         anchor="left", glow=glow)  # fmt: skip
            if u > 0:
                ctx.text(part, center - right * wb / 2, h, color=_rgba(color, p.opacity * u),
                         anchor="left", glow=glow)  # fmt: skip
        for j, (part, origin, h) in enumerate(layout):
            if j in pairs:
                continue
            if m > 0.5:
                ctx.text(part, origin, h, color=_rgba(self._color(me, p.colors, j),
                         p.opacity * (2 * m - 1)), anchor="left", glow=glow)  # fmt: skip
        gone = {i for i in pairs.values()}
        if m < 0.5:
            for i, (part, origin, h) in enumerate(before):
                if i not in gone:
                    ctx.text(part, origin, h, color=_rgba(self._color(me, p.from_colors, i),
                             p.opacity * (1 - 2 * m)), anchor="left", glow=glow)  # fmt: skip


# ---------------------------------------------------------------- text as shapes


WRITE_SPAN = 0.5  # share of a ``write`` each glyph takes; they start one after another
OUTLINE = 0.04  # a glyph outline's width, as a share of the text's line height
GLYPH = "glyph"  # the tag of a glyph part: its text is judged (tiny, focal point), not it


def written(write: float, k: int, n: int) -> tuple[float, float]:
    """How far glyph ``k`` of ``n`` is written at ``write`` (0..1): (its outline drawn, its
    fill in), each 0..1. Like Manim's ``Write``: the outline is drawn, then filled."""
    start = k / n * (1.0 - WRITE_SPAN) if n > 1 else 0.0
    u = min(max((write - start) / WRITE_SPAN, 0.0), 1.0)
    return min(2.0 * u, 1.0), max(2.0 * u - 1.0, 0.0)


def _becoming(state: SceneState, me) -> bool:
    """Morphing from an object that is in the scene (``domains/morph.py``)."""
    p = me.params
    return bool(p.morph_from) and p.morph < 1.0 and p.morph_from in state


def _glyph_look(me, color) -> mm.Look:
    p = me.params
    c = tuple(color)
    return mm.Look((c[0], c[1], c[2], c[3] * p.stroke_opacity),
                   (c[0], c[1], c[2], c[3] * p.fill_opacity), OUTLINE * float(me.style["size"]),
                   float(me.style.get("glow", 1.0)))  # fmt: skip


def _glyph_shape(pieces, h: float, cam) -> mm.Shape:
    """Outlines placed in world space: ``pieces`` are (outline, its box's lower-left corner,
    look, (part index, TeX) or None, key); ``h`` is the line height."""
    right, up = np.asarray(cam.right, dtype=np.float64), np.asarray(cam.up, dtype=np.float64)
    subs: list[np.ndarray] = []
    groups: list[mm.Group] = []
    for outline, corner, look, part, key in pieces:
        c = np.asarray(corner, dtype=np.float64)
        for g in outline.glyphs[: max(gm.MAX_GLYPHS - len(groups), 0)]:  # all parts together
            members = []
            for sub in g.subpaths:
                world = c + np.outer(sub[:, 0] * h, right) + np.outer(sub[:, 1] * h, up)
                world.setflags(write=False)
                members.append(len(subs))
                subs.append(world)
            groups.append(mm.Group(g.name, g.key, tuple(members), look, len(groups),
                                   part[1] if part else None, key,
                                   part[0] if part else -1))  # fmt: skip
    closed = tuple(True for _ in subs)  # glyph contours are closed
    return mm.Shape(tuple(subs), closed, tuple(groups), "nonzero", float(h),
                    tuple(float(v) for v in right), tuple(float(v) for v in up))  # fmt: skip


def _glyph_parts(state: SceneState, me, shape: mm.Shape) -> list[ObjectBase]:
    """A ``Path`` part per glyph, named by what it shows (``cap_a``, ``plus``), written in
    one after another by ``write`` (its outline drawn, then filled)."""
    from types import SimpleNamespace

    from phasor.core.ids import children
    from phasor.domains.svg import part_names

    p = me.params
    names = part_names([SimpleNamespace(id=g.name, tag="glyph") for g in shape.groups],
                       children(state.reserved, me.id))  # fmt: skip
    style = {
        "color": _typed(state, me, "color"),
        "fill": "solid",
        "stroke": max(OUTLINE * shape.line, 1e-9),
    }  # world units: it scales with the text
    if "glow" in me.roles:
        style["glow"] = me.roles["glow"]
    colors = getattr(p, "colors", ())
    n = len(shape.groups)
    out: list[ObjectBase] = []
    for k, (g, name) in enumerate(zip(shape.groups, names, strict=True)):
        if _far(shape, g):  # a Path cannot hold it: left out (lint glyph_limit)
            continue
        drawn, filled = written(p.write, k, n)
        own = dict(style)
        if 0 <= g.part < len(colors):  # an equation part's own color role
            own["color"] = colors[g.part]
        own["fill_color"] = own["color"]
        subs = [shape.subpaths[i].tolist() for i in g.members]
        out.append(Path(name, subpaths=subs, style=own, progress=drawn, tags=(GLYPH,),
                        fill_opacity=p.fill_opacity * filled,
                        stroke_opacity=1.0 - (1.0 - p.stroke_opacity) * filled))  # fmt: skip
    return out


def _far(shape: mm.Shape, g: mm.Group) -> bool:
    """A glyph with a point past ``path.MAX_COORD`` (or not finite): no ``Path`` part can
    hold it, so it is left out of the parts (lint ``glyph_limit`` says why)."""
    return any(not float(np.abs(shape.subpaths[i]).max()) <= MAX_COORD for i in g.members)


def _typed(state: SceneState, me, key: str):
    """The style value a part takes from its text: what the scene typed, or, while a style
    transition moves it, the value at t (a linear color, a world size)."""
    if key not in me.restyling:
        return me.roles[key]
    value = me.style[key]
    if isinstance(value, tuple):
        return tuple(float(min(max(v, 0.0), 1.0)) for v in value)
    world = float(value) * float(state.camera_for(me).units_per_height(me.params.pos))
    return max(world, 1e-9)


def _outlined(me, shape: mm.Shape | None, texts) -> mm.Shape:
    """The outline a text drawn as shapes is built from; a text with none is refused,
    saying why (mathtext cannot read what TeX typeset)."""
    if shape is None:
        why = next((o.error for t in texts if (o := gm.outline(t)).error), "")
        raise ValueError(f"{me.id!r} cannot be drawn as shapes: {why}; draw it as a picture "
                         "(as_shapes off) or write maths mathtext reads")  # fmt: skip
    return shape


def _draw_morph(ctx: DrawContext, state: SceneState, impl, arc: float = 0.0) -> bool:
    """A text or equation becoming itself from another object: the in-between, or its own
    glyphs when the source has no outline. False (nothing drawn) when it has no outline
    itself: the caller draws its picture."""
    from phasor.domains import morph

    me = impl.me(state)
    items = morph.frame(state, impl.id, arc=arc)
    if items is None:
        shape = impl.morph_shape(state)
        if shape is None:
            return False
        items = morph.own(state, me, shape)
    morph.paint(ctx, state, me, items)
    return True


def _with_morph(impl, state: SceneState, box: Bounds) -> Bounds:
    """A morphing text's box also covers what it draws on the way."""
    me = impl.me(state)
    if not _becoming(state, me):
        return box
    from phasor.core import record

    drawn = record.derived_world_bounds(impl, state)
    return Bounds.of_points([box.lo, box.hi, drawn.lo, drawn.hi])


def _shape_facts(state: SceneState, me, texts) -> dict[str, Any]:
    """Semantics of a text drawn as shapes, or becoming something."""
    from phasor.domains import morph

    p = me.params
    out: dict[str, Any] = {}
    if p.as_shapes:
        total = drawn = 0
        for k, t in enumerate(texts):
            o = gm.outline(t)
            total, drawn = total + o.total, drawn + len(o.glyphs)
            if drawn >= gm.MAX_GLYPHS:  # the rest is not read: at least one glyph a part
                total += len(texts) - k - 1
                break
        drawn = min(drawn, gm.MAX_GLYPHS)
        shape = me.impl.morph_shape(state)
        far = sum(_far(shape, g) for g in shape.groups) if shape is not None else 0
        out.update(as_shapes=True, glyphs=drawn - far)
        if drawn < total:
            out["glyph_limit"] = total  # lint glyph_limit (at least this many)
        if far:
            out["glyphs_far"] = far  # lint glyph_limit: past path.MAX_COORD, not drawn
    out.update(morph.facts(state, me))
    return out


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


def pair_parts(a, b, a_keys=(), b_keys=()) -> list[tuple[int, int]]:
    """Which part ``a[i]`` of the old equation becomes part ``b[j]`` of the new, as (i, j)
    sorted by j. Parts with the same key pair first, in any order and whatever their text
    (an equation being solved: ``+a`` on the left becomes ``-a`` on the right); the others
    pair by equal text in order (``match_parts``)."""
    keyed: dict[int, int] = {}
    for j, key in enumerate(b_keys[: len(b)]):
        if key:
            i = next((i for i, k in enumerate(a_keys[: len(a)]) if k == key
                      and i not in keyed.values()), None)  # fmt: skip
            if i is not None:
                keyed[j] = i
    ra = [i for i in range(len(a)) if i not in keyed.values()]
    rb = [j for j in range(len(b)) if j not in keyed]
    plain = {rb[q]: ra[p] for p, q in match_parts([a[i] for i in ra], [b[j] for j in rb])}
    return sorted(((i, j) for j, i in {**keyed, **plain}.items()), key=lambda ij: ij[1])


def crossing(pairs, a_parts=(), b_parts=()) -> list[bool]:
    """For each pair of ``pair_parts`` (sorted by j): whether it is carried past other parts
    (the order changes), so it should travel round them, not through them. The parts that
    keep their order are the largest set that can (unchanged text counts double, so ``=``
    stays put and ``+a`` goes round it); the rest arc."""
    n = len(pairs)
    weight = [2.0 if a_parts[i] == b_parts[j] else 1.0 for i, j in pairs] if a_parts else [1.0] * n
    best, prev = list(weight), [-1] * n  # heaviest increasing run of i ending at each pair
    for k in range(n):
        for q in range(k):
            if pairs[q][0] < pairs[k][0] and best[q] + weight[k] > best[k]:
                best[k], prev[k] = best[q] + weight[k], q
    keep, k = set(), max(range(n), key=lambda q: best[q]) if n else -1
    while k >= 0:
        keep.add(k)
        k = prev[k]
    return [k not in keep for k in range(n)]


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
        prefix: str = ""  # no length limit, as for any Text (documents saved before M19)
        suffix: str = ""
        math: bool = True  # typeset as maths: "$" + prefix + number + suffix + "$"
        pos: Vec3 = (0.0, 0.0, 0.0)
        anchor: Literal["center", "left", "right"] = "center"
        group: bool = False  # a comma between thousands: 1,234,567
        sign: bool = False  # a "+" before a positive number

        @model_validator(mode="after")
        def _typeset(self):
            if self.math:
                text_metrics.check_math(f"${self.prefix}0{self.suffix}$")
            return self

    @staticmethod
    def text_of(p) -> str:
        """The text shown: the number with ``digits`` decimals (no ``-0``), between
        ``prefix`` and ``suffix``. A number of 1e15 or more in size is written in scientific
        notation and one that is not finite as ``nan`` or ``inf``, so the text stays short
        whatever an expression feeds it."""
        num = ticks.text(p.value, p.digits, group=p.group, sign=p.sign)
        if p.math:
            return f"${p.prefix}{num.replace(',', '{,}')}{p.suffix}$"
        return f"{p.prefix}{num}{p.suffix}"

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


@register
class Value(ObjectBase):
    """A named number to animate, with no drawing (Manim's ``ValueTracker``, ADR 0005).
    Animate ``value`` like any parameter; read it in an expression (``expr("speed.value")``),
    in an updater (``look("speed").value``) or show it with a ``Counter``."""

    type: ClassVar[str] = "value"
    Life: ClassVar = frozenset()  # a number that means something: life never touches it

    class Params(ParamsBase):
        value: float = 0.0

    def semantics(self, state: SceneState) -> dict[str, Any]:
        # drawn 0: nothing shows, so no visual lint rule judges it
        return {"kind": "value", "value": self.me(state).params.value, "drawn": 0.0}

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        """Nothing: a value is only a number."""
