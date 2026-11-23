"""One object becoming another (the ``transform`` and ``transform_matching`` verbs).

A target that is morphing (``morph_from`` names the source, ``morph`` runs 0 to 1) draws the
in-between itself, every frame, from the source's outline as the state has it at ``t`` and
its own: a pure function of (document, t) (contract 3). Nothing is baked into the document,
so a source that moves is followed.

- **Outline.** Any object with ``morph_shape(state)`` (paths and shapes, text and equations,
  which give their glyphs) or a composite whose parts have one (an ``SVG``) has an outline:
  ``morph_math.Shape``, its subpaths in world space in ``Group``s (a glyph, a whole path),
  each with its ``Look`` (stroke and fill colors in linear RGBA, stroke width, glow).
- **Whole** (``morph_kind="whole"``, the ``transform`` verb): the subpaths of one become the
  subpaths of the other (``morph_math.plan``); colors, opacities and widths mix in linear
  light. Subpaths past ``morph_math.MAX_SUBPATHS`` fade out (the source's) or in (the
  target's).
- **Matching** (``morph_kind="matching"``, ``transform_matching``): between two equations,
  parts pair by TeX (``basic.pair_parts``: keys first, then equal parts in order) and an
  equal part's glyphs pair one to one; otherwise glyphs (or whole paths) pair by shape (a
  glyph's font and index; any other outline's ``morph_math.signature``), in reading order.
  A pair of the same shape glides to its new place (and size); a pair of different shapes
  (a keyed part whose text changes) morphs; a pair carried past others arcs round them
  (``arc`` line heights, as an equation's ``transform``). The source's other groups fade
  out in the first half, the target's fade in in the second.

``frame(state, target)`` gives what to draw: ``Item``s of world polylines with a look and an
opacity factor; ``paint`` draws them as a path draws itself (fill, then stroke). Dashes and
arrow tips are not drawn in between.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phasor.core.cache import ByteCache
from phasor.core.state import ObjectState, SceneState
from phasor.domains import morph_math as mm
from phasor.domains import path_math as pm
from phasor.domains.morph_math import Group, Look, Shape
from phasor.domains.path import FLATNESS

MAX_GROUPS = 2000  # groups an outline is matched by; the rest of a composite's are left out
MAX_FILL_RING = 4096  # points of one ring the in-between's fill is triangulated through
# Plans kept between frames, in bytes. The largest plan is MAX_TOTAL points a side, about
# 2 MB, so this keeps some 30 of the largest at once (a scene of many morphs running
# together still finds each one's plan the next frame).
PLAN_BYTES = 64 * 2**20
_PLANS = ByteCache(lambda: PLAN_BYTES)


@dataclass(frozen=True)
class Item:
    polys: tuple[np.ndarray, ...]  # world polylines
    closed: tuple[bool, ...]
    look: Look
    alpha: float  # times the target's opacity
    rule: str = "nonzero"
    fill: tuple[np.ndarray, ...] | None = None  # rings to fill, when not ``polys``
    # Mid-morph the fill crossfades from the rings laid out as the source's subpaths
    # (``fill_from``, even-odd: exact at the start) to the target's (``fill``, its rule:
    # exact at the end); ``blend`` is the target's share.
    fill_from: tuple[np.ndarray, ...] | None = None
    blend: float = 1.0


def morphing(p) -> bool:
    """Whether an object with these parameters is becoming another right now."""
    return bool(getattr(p, "morph_from", "")) and float(getattr(p, "morph", 1.0)) < 1.0


# ---------------------------------------------------------------- outlines


def shape_of(state: SceneState, object_id: str) -> Shape | None:
    """The outline of an object as the state has it (None: it has none, or is not there)."""
    if not object_id or object_id not in state:
        return None
    impl = state[object_id].impl
    fn = getattr(impl, "morph_shape", None)
    if callable(fn):
        return fn(state)
    parts = [o for o in state.objects if o.parent == object_id]
    shapes = [s for o in parts if (s := shape_of(state, o.id)) is not None]
    return merge(shapes) if shapes else None


def merge(shapes: Sequence[Shape]) -> Shape:
    """Several outlines as one (a composite's parts), groups in order."""
    subs: list[np.ndarray] = []
    closed: list[bool] = []
    groups: list[Group] = []
    for s in shapes:
        base = len(subs)
        subs += s.subpaths
        closed += s.closed
        for g in s.groups:
            groups.append(Group(g.name, g.key, tuple(base + i for i in g.members), g.look,
                                len(groups), g.tex, g.tag, g.part))  # fmt: skip
    groups = groups[:MAX_GROUPS]
    keep = sorted({i for g in groups for i in g.members})
    if len(keep) < len(subs):  # subpaths of groups left out go too
        where = {i: k for k, i in enumerate(keep)}
        groups = [Group(g.name, g.key, tuple(where[i] for i in g.members), g.look, g.order,
                        g.tex, g.tag, g.part) for g in groups]  # fmt: skip
        subs, closed = [subs[i] for i in keep], [closed[i] for i in keep]
    first = shapes[0]
    return Shape(tuple(subs), tuple(closed), tuple(groups), first.rule,
                 max(s.line for s in shapes), first.right, first.up)  # fmt: skip


def path_look(me: ObjectState) -> Look:
    """A path's look from its resolved style and parameters."""
    p = me.params
    c = me.style["color"]
    f = me.style.get("fill_color", c)
    stroke = float(getattr(p, "stroke_opacity", 1.0))
    fill = float(me.style.get("fill", 0.0)) * float(getattr(p, "fill_opacity", 1.0))
    return Look((c[0], c[1], c[2], c[3] * stroke), (f[0], f[1], f[2], f[3] * fill),
                float(me.style.get("stroke", 0.0)), float(me.style.get("glow", 1.0)))  # fmt: skip


# ---------------------------------------------------------------- what to draw


def _tol(state: SceneState, target: ObjectState, shapes: Sequence[Shape]) -> float:
    pts = [np.vstack(s.subpaths) for s in shapes if s.subpaths]
    center = np.vstack(pts).mean(axis=0) if pts else np.zeros(3)
    return float(f"{FLATNESS * float(state.camera_for(target).units_per_height(center)):.3g}")


def _flat(subs: Sequence[np.ndarray], tol: float) -> tuple[np.ndarray, ...]:
    return tuple(pm.flatten(s, tol) for s in subs)


def _plan(a: Sequence[np.ndarray], b: Sequence[np.ndarray], ac, bc) -> mm.Plan:
    """The plan for two outlines, cached by every byte of both (and which are closed), in
    their own pool of ``PLAN_BYTES``: a source that moves makes a plan a frame (pairs are
    by least travel in the world), and those must neither grow without bound nor push a
    drawing's flattened curves out of the geometry cache."""
    key = ("morph", mm.digest(a), mm.digest(b), tuple(ac), tuple(bc))
    return _PLANS.get(key, lambda: mm.plan(a, b, ac, bc))


def frame(state: SceneState, target_id: str, arc: float = 0.0) -> list[Item] | None:
    """What the morphing ``target_id`` draws at ``state.t``; None when it is not morphing or
    either side has no outline (it then draws itself)."""
    me = state[target_id]
    p = me.params
    if not morphing(p) or p.morph_from == target_id:
        return None
    src, dst = shape_of(state, p.morph_from), me.impl.morph_shape(state)
    if src is None or dst is None:
        return None
    m = max(float(p.morph), 0.0)
    tol = _tol(state, me, (src, dst))
    if getattr(p, "morph_kind", "whole") == "matching":
        return _matching(src, dst, m, tol, arc)
    return _whole(src, dst, m, tol)


def _whole(src: Shape, dst: Shape, m: float, tol: float) -> list[Item]:
    if not dst.groups:
        return [Item(_flat(src.subpaths, tol), src.closed, src.groups[0].look, 1.0 - m,
                     src.rule)] if src.groups else []  # fmt: skip
    if not src.groups:
        return [Item(_flat(dst.subpaths, tol), dst.closed, dst.groups[0].look, m, dst.rule)]
    plan = _plan(src.subpaths, dst.subpaths, src.closed, dst.closed)
    items: list[Item] = []
    if plan.identical:
        for g in dst.groups:
            look = src.look_of(g.members[0]).mix(g.look, m) if g.members else g.look
            items.append(Item(_flat([dst.subpaths[i] for i in g.members], tol),
                              tuple(dst.closed[i] for i in g.members), look, 1.0,
                              dst.rule))  # fmt: skip
        return items
    owner = {i: k for k, g in enumerate(dst.groups) for i in g.members}
    first_src: dict[int, int] = {}  # target subpath -> the source subpath of its first piece
    for s, d in zip(plan.src, plan.dst, strict=True):
        first_src.setdefault(d, s)
    by_group: dict[int, list[tuple[np.ndarray, bool, int]]] = {}
    for poly, closed, d in mm.at(plan, m):
        by_group.setdefault(owner.get(d, -1), []).append((poly, closed, d))
    fills: dict[int, list[np.ndarray]] = {}
    for ring, d in mm.rings(plan, m):  # a ring cut into pieces fills as one
        fills.setdefault(owner.get(d, -1), []).append(ring)
    before: dict[int, list[np.ndarray]] = {}  # the same points, as the source's subpaths
    for ring, s_ in mm.rings(plan, m, "source"):
        first = next((d for s2, d in zip(plan.src, plan.dst, strict=True) if s2 == s_), -1)
        before.setdefault(owner.get(first, -1), []).append(ring)
    for k in sorted(by_group):
        pieces = by_group[k]
        b = dst.groups[k].look if k >= 0 else dst.groups[0].look
        s = first_src.get(pieces[0][2], -1)
        a = src.look_of(s) if s >= 0 else b
        items.append(Item(tuple(q for q, _, _ in pieces), tuple(c for _, c, _ in pieces),
                          a.mix(b, m), 1.0, dst.rule, tuple(fills.get(k, ())),
                          tuple(before.get(k, ())), m))  # fmt: skip
    for i in plan.gone:
        items.append(Item(_flat([src.subpaths[i]], tol), (src.closed[i],), src.look_of(i),
                          1.0 - m, src.rule))  # fmt: skip
    for i in plan.new:
        items.append(Item(_flat([dst.subpaths[i]], tol), (dst.closed[i],), dst.look_of(i), m,
                          dst.rule))  # fmt: skip
    return items


Pair = tuple[tuple[int, ...], tuple[int, ...]]


def _equal(a: Sequence[str], b: Sequence[str], skip_a=(), skip_b=()) -> list[tuple[int, int]]:
    """(i, j) with ``a[i] == b[j]``: each ``b[j]`` in order takes the first ``a[i]`` left
    with its value (whatever the order: a term moved across the ``=`` still pairs)."""
    waiting: dict[str, list[int]] = {}
    for i, v in enumerate(a):
        if i not in skip_a:
            waiting.setdefault(v, []).append(i)
    out = []
    for j, v in enumerate(b):
        if j not in skip_b and waiting.get(v):
            out.append((waiting[v].pop(0), j))
    return out


def pair_groups(src: Shape, dst: Shape) -> list[Pair]:
    """Which source groups become which target groups, sorted by the target's first group.
    Two equations pair by TeX: parts with the same key first (whatever their text), then
    parts with equal TeX, in reading order (an equal part glyph by glyph; a keyed part
    whose text changes as a whole, its outline morphing). Anything else pairs glyph by
    glyph (or path by path) by shape, in reading order. The same answer whatever order a
    script wrote things in."""
    a, b = src.groups, dst.groups
    if a and b and all(g.tex is not None for g in (*a, *b)):
        parts_a, parts_b = _parts(a), _parts(b)
        keyed = _equal([k or f"\0{i}" for i, (_, k, _) in enumerate(parts_a)],
                       [k or f"\1{j}" for j, (_, k, _) in enumerate(parts_b)])  # fmt: skip
        found = keyed + _equal([t for t, _, _ in parts_a], [t for t, _, _ in parts_b],
                               {i for i, _ in keyed}, {j for _, j in keyed})  # fmt: skip
        pairs: list[Pair] = []
        for pi, pj in found:
            (ta, _, ga), (tb, _, gb) = parts_a[pi], parts_b[pj]
            if ta == tb and len(ga) == len(gb):
                pairs += [((x,), (y,)) for x, y in zip(ga, gb, strict=True)]
            elif ga and gb:
                pairs.append((ga, gb))
        return sorted(pairs, key=lambda ij: ij[1][0])
    order_a = sorted(range(len(a)), key=lambda i: (a[i].order, i))
    order_b = sorted(range(len(b)), key=lambda j: (b[j].order, j))
    found = _equal([a[i].key for i in order_a], [b[j].key for j in order_b])
    return sorted((((order_a[i],), (order_b[j],)) for i, j in found), key=lambda ij: ij[1][0])


def _parts(groups: Sequence[Group]) -> list[tuple[str, str, tuple[int, ...]]]:
    """An equation's parts from its glyph groups: (TeX, key, glyph groups in order)."""
    seen: dict[int, tuple[str, str, list[int]]] = {}
    for k, g in enumerate(groups):
        seen.setdefault(g.part, (g.tex or "", g.tag, []))[2].append(k)
    return [(t, key, tuple(ks)) for _, (t, key, ks) in sorted(seen.items())]


def _box(subs: Sequence[np.ndarray]) -> tuple[np.ndarray, float]:
    pts = np.vstack(subs) if subs else np.zeros((1, 3))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2, float((hi - lo).max())


def _members(shape: Shape, groups: Sequence[int]) -> list[int]:
    return [i for k in groups for i in shape.groups[k].members]


def _matching(src: Shape, dst: Shape, m: float, tol: float, arc: float) -> list[Item]:
    from phasor.domains.basic import crossing

    pairs = pair_groups(src, dst)
    simple = [(ga[0], gb[0]) for ga, gb in pairs]
    tex = bool(pairs) and src.groups[0].tex is not None and dst.groups[0].tex is not None
    over = crossing(simple, [g.tex for g in src.groups], [g.tex for g in dst.groups]) if tex \
        else crossing(simple)  # fmt: skip
    up, right = np.asarray(dst.up, dtype=np.float64), np.asarray(dst.right, dtype=np.float64)
    items: list[Item] = []
    for k, (ga, gb) in enumerate(pairs):
        ia, ib = _members(src, ga), _members(dst, gb)
        sa, sb = [src.subpaths[i] for i in ia], [dst.subpaths[i] for i in ib]
        if not sa or not sb:
            continue
        (ca, za), (cb, zb) = _box(sa), _box(sb)
        lift = np.zeros(3)
        if over[k] and arc > 0:  # carried past others: over them going right, under going left
            side = 1.0 if float((cb - ca) @ right) >= 0 else -1.0
            lift = up * (side * arc * max(dst.line, src.line) * math.sin(math.pi * m))
        look = src.groups[ga[0]].look.mix(dst.groups[gb[0]].look, m)
        if len(ga) == len(gb) == 1 and src.groups[ga[0]].key == dst.groups[gb[0]].key \
                and len(sa) == len(sb):  # fmt: skip
            scale = 1.0 + ((zb / za if za > 0 else 1.0) - 1.0) * m  # the same shape glides
            centre = ca + (cb - ca) * m + lift
            moved = [centre + (s - ca) * scale for s in sa]
            items.append(Item(_flat(moved, tol), tuple(src.closed[i] for i in ia), look, 1.0,
                              dst.rule))  # fmt: skip
            continue
        plan = _plan(sa, sb, [src.closed[i] for i in ia], [dst.closed[i] for i in ib])
        if plan.identical:
            polys, closed = _flat(sb, tol), tuple(dst.closed[i] for i in ib)
            fill = None
        else:
            got = mm.at(plan, m)
            gone = _flat([sa[q] for q in plan.gone], tol)
            polys = tuple(q for q, _, _ in got) + gone
            closed = tuple(c for _, c, _ in got) + tuple(src.closed[ia[q]] for q in plan.gone)
            fill = tuple(r + lift for r, _ in mm.rings(plan, m)) + tuple(q + lift for q in gone)
            fill_from = tuple(r + lift for r, _ in mm.rings(plan, m, "source"))
            fill_from += tuple(q + lift for q in gone)
            items.append(Item(tuple(q + lift for q in polys), closed, look, 1.0, dst.rule, fill,
                              fill_from, m))  # fmt: skip
            continue
        items.append(Item(tuple(q + lift for q in polys), closed, look, 1.0, dst.rule, fill))
    matched_a = {g for ga, _ in pairs for g in ga}
    matched_b = {g for _, gb in pairs for g in gb}
    out_alpha, in_alpha = max(0.0, 1.0 - 2.0 * m), max(0.0, 2.0 * m - 1.0)
    for i, g in enumerate(src.groups):
        if i not in matched_a and out_alpha > 0:
            items.append(Item(_flat([src.subpaths[q] for q in g.members], tol),
                              tuple(src.closed[q] for q in g.members), g.look, out_alpha,
                              src.rule))  # fmt: skip
    for j, g in enumerate(dst.groups):
        if j not in matched_b and in_alpha > 0:
            items.append(Item(_flat([dst.subpaths[q] for q in g.members], tol),
                              tuple(dst.closed[q] for q in g.members), g.look, in_alpha,
                              dst.rule))  # fmt: skip
    return items


def _thin(ring: np.ndarray) -> np.ndarray:
    """A ring to fill, with at most ``MAX_FILL_RING`` points (every k-th kept). Only the
    in-between makes such rings (the pieces of one subpath joined along it, 40 000 points
    at most); triangulating one that crosses itself grows faster than its size."""
    if len(ring) <= MAX_FILL_RING:
        return ring
    return ring[:: -(-len(ring) // MAX_FILL_RING)]


def paint(ctx, state: SceneState, me: ObjectState, items: Sequence[Item]) -> None:
    """Draw items as a path draws itself: the fill (plain paint, never glowing), then the
    stroke (mitered; a closed loop's seam too)."""
    from phasor.domains.strokes import polyline

    cam = state.camera_for(me)
    opacity = float(me.params.opacity)
    for it in items:
        drawn = [(q, c) for q, c in zip(it.polys, it.closed, strict=True) if len(q) >= 2]
        if not drawn or it.alpha <= 0:
            continue
        polys = [q for q, _ in drawn]
        a = opacity * it.alpha
        f = it.look.fill
        layers = [(it.fill if it.fill is not None else polys, it.rule, it.blend)]
        if it.fill_from is not None and it.blend < 1.0:
            layers.insert(0, (it.fill_from, "evenodd", 1.0 - it.blend))
        for rings_, rule, share in layers:
            rings = [_thin(q) for q in rings_ if len(q) >= 3]
            if f[3] * a * share > 0 and rings:
                pos, tris = pm.fill(rings, rule)
                if len(tris):
                    ctx.triangles(pos, tris, color=(f[0], f[1], f[2], f[3] * a * share), glow=1.0)
        c = it.look.color
        if c[3] * a > 0 and it.look.width > 0:
            for q, closed in drawn:
                line = q
                if closed and len(q) > 2 and np.array_equal(q[0], q[-1]):
                    mid = (q[0] + q[1]) / 2
                    line = np.vstack([mid, q[1:], mid])
                polyline(ctx, cam, line, it.look.width * cam.units_per_height(line),
                         color=(c[0], c[1], c[2], c[3] * a), glow=it.look.glow)  # fmt: skip


def own(state: SceneState, me: ObjectState, shape: Shape) -> list[Item]:
    """An outline drawn as itself (a morph whose source has no outline)."""
    tol = _tol(state, me, (shape,))
    return [Item(_flat([shape.subpaths[i] for i in g.members], tol),
                 tuple(shape.closed[i] for i in g.members), g.look, 1.0, shape.rule)
            for g in shape.groups]  # fmt: skip


def facts(state: SceneState, me: ObjectState) -> dict[str, object]:
    """Semantics of a morphing object: where from, how far, and what lint should know
    (``morph_missing``: no source, or one with no outline, so it draws itself;
    ``morph_limit``: subpaths past ``morph_math.MAX_SUBPATHS`` that fade instead)."""
    p = me.params
    if not morphing(p):
        return {}
    out: dict[str, object] = {"morphing_from": p.morph_from, "morph": round(float(p.morph), 6),
                              "morph_kind": p.morph_kind}  # fmt: skip
    src = shape_of(state, p.morph_from) if p.morph_from != me.id else None
    if src is None:
        out["morph_missing"] = True
        return out
    dst = me.impl.morph_shape(state)
    if dst is None:  # it has no outline itself (it draws itself)
        return out
    if p.morph_kind == "whole":
        plan = _plan(src.subpaths, dst.subpaths, src.closed, dst.closed)
        if plan.gone or plan.new:
            out["morph_limit"] = len(plan.gone) + len(plan.new)
    return out
