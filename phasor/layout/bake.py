"""The layout bake (SPEC contract 7).

The solver runs at every key time (markers, the ends of every transition, and a 10 Hz grid)
and writes positions into tracks. ``baked_state`` then gives the scene at any time with those
positions in place; renderers and the inspector both read it, so what is inspected is what
renders.

A baked position is stored on screen, relative to the object it is placed against. Between
key times only that offset is interpolated, so a label stays glued to a moving anchor and
keeps its gap while the camera zooms.

The bake is a pure function of (compiled document, occlusion oracle): the same inputs give
the same tracks, and it is cached per document version by the Engine.
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from phasor.core.constraints import Guard, Relation
from phasor.core.issues import Issue
from phasor.core.state import SceneState
from phasor.core.values import normalize
from phasor.layout import labels, solver
from phasor.layout.geometry import Box, Coverage, footprint, frame, screen_box, view_of
from phasor.layout.plan import MOVE_PARAM, Plan
from phasor.time.compile import Compiled, evaluate
from phasor.time.tracks import ResolvedTrack

SAMPLE_HZ = 10.0
ANALYTIC_WIDTH = 480
MAX_TIMES_IN_ISSUE = 12


class OcclusionOracle(Protocol):
    """What the label placer needs from a renderer (ADR 0001): which pixels are covered."""

    def coverage(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        """(h, w) bool, row 0 = top: pixels covered by an occluding object not in ``exclude``."""
        ...


class AnalyticOracle:
    """No renderer: every occluding object covers its ``geometry.footprint``."""

    def coverage(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        size = (ANALYTIC_WIDTH, max(1, round(ANALYTIC_WIDTH / state.aspect)))
        mask = np.zeros(size[::-1], dtype=bool)
        for obj in state.objects:
            if obj.id not in exclude and obj.params.opacity >= 0.5:
                mask |= footprint(state, obj, size)
        return mask


@dataclass(frozen=True)
class Track:
    """One baked parameter. ``keys`` are screen points (relative to ``ref``'s center when it
    is set) for a position, or plain numbers for a scalar."""

    path: str
    kind: str  # "point" | "scalar"
    ref: str | None
    keys: tuple
    # jumps[i] is True when the label changes slot between key i and key i + 1. It then
    # holds its place and jumps at key i + 1, instead of sliding across its own anchor.
    jumps: tuple[bool, ...] = ()


@dataclass(frozen=True)
class Bake:
    times: tuple[float, ...] = ()
    tracks: tuple[Track, ...] = ()
    issues: tuple[Issue, ...] = ()
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)  # label id -> slot per key
    occlusion: str = "analytic"  # "pixel" when the placer read the real id pass

    def values_at(self, t: float, pre: SceneState) -> dict[str, object]:
        """Solved parameter values at ``t``, given the scene as the document alone gives it."""
        if not self.tracks:
            return {}
        ts = self.times
        i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 1)
        j = min(i + 1, len(ts) - 1)
        u = 0.0 if j == i else min(max((t - ts[i]) / (ts[j] - ts[i]), 0.0), 1.0)
        out: dict[str, object] = {}
        for tr in self.tracks:
            a, b = np.asarray(tr.keys[i]), np.asarray(tr.keys[j])
            v = a + (b - a) * (0.0 if tr.jumps and j > i and tr.jumps[i] and u < 1.0 else u)
            if tr.kind == "scalar":
                out[tr.path] = float(v)
                continue
            obj = pre[tr.path.rpartition(".")[0]]
            if tr.ref is not None:
                v = v + np.asarray(screen_box(pre, pre[tr.ref]).center)
            out[tr.path] = _to_world(pre, obj, pre.values[tr.path], v)
        return out


def _to_world(state: SceneState, obj, base, screen_point) -> tuple:
    """The world point that appears at ``screen_point``, at the depth of ``base``."""
    view = view_of(state, obj)
    bx, by = view.point(base)
    return normalize(np.asarray(base) + view.shift(screen_point[0] - bx, screen_point[1] - by))


def key_times(c: Compiled) -> tuple[float, ...]:
    """Markers, the ends of every transition, and a uniform grid."""
    end = c.duration
    ts = {0.0, end, *c.time_map.markers.values()}
    for node in c.nodes.values():
        if isinstance(node, ResolvedTrack):
            for s in node.segments:
                ts.update((s.t0, s.t1))
    n = int(np.floor(end * SAMPLE_HZ + 1e-9))
    ts.update(k / SAMPLE_HZ for k in range(n + 1))
    return tuple(sorted(round(t, 9) for t in ts if 0.0 <= t <= end))


def baked_state(c: Compiled, baked: Bake, t: float) -> SceneState:
    """The scene at ``t`` with layout applied. Pure: depends only on (document, bake, t)."""
    pre = evaluate(c, t)
    values = baked.values_at(t, pre)
    return evaluate(c, t, values) if values else pre


def bake(c: Compiled, plan: Plan, oracle: OcclusionOracle | None = None) -> Bake:
    if plan.empty:
        return Bake()
    oracle = oracle or AnalyticOracle()
    lay = c.theme.layout
    times = key_times(c)
    placed_ids = frozenset(plan.placed)
    pos = {i: f"{i}.{MOVE_PARAM}" for i in plan.moving}

    # Which object each moving object follows between key times.
    ref: dict[str, str | None] = dict.fromkeys(plan.moving)
    for _, rel in plan.relations():
        if ref.get(rel.subject) is None:
            ref[rel.subject] = rel.target

    keys: dict[str, list] = {p: [] for p in pos.values()}
    for i in sorted(plan.leaders):
        keys[f"{i}.leader"], keys[f"{i}.leader_on"] = [], []
    failures: dict[tuple[int, str], tuple[tuple[str, ...], list[float]]] = {}
    slots: dict[str, list[str]] = {i: [] for i in plan.placed}
    previous: dict[str, labels.Previous] = {}

    def fail(rule, ids, message, t):
        failures.setdefault((rule.index, message), (tuple(ids), []))[1].append(t)

    for t in times:
        pre = evaluate(c, t)
        boxes = {o.id: screen_box(pre, o) for o in pre.objects}
        replace: dict[str, object] = {}

        def move(state, object_id, dx, dy, replace=replace):
            obj = state[object_id]
            path = pos[object_id]
            world = normalize(np.asarray(state.values[path]) + view_of(state, obj).shift(dx, dy))
            replace[path] = world

        # Stage 1: linear relations.
        items = []
        for rule, clause in ((r, cl) for r in plan.rules for cl in r.clauses):
            if isinstance(clause, Relation) and clause.kind != "near":
                unit = view_of(pre, pre[clause.subject]).units_per_height
                gap = lay["gap"] if clause.gap is None else clause.gap / unit
                items.append(solver.Item(rule.index, rule.spec.priority, clause, gap))
            elif (
                isinstance(clause, Guard)
                and clause.kind == "on_screen"
                and clause.ids[0] in plan.solved
            ):
                unit = view_of(pre, pre[clause.ids[0]]).units_per_height
                margin = lay["margin"] if clause.value is None else clause.value / unit
                items.append(solver.Item(rule.index, rule.spec.priority, clause, margin))
        if items:
            centers, unsolved = solver.solve(items, boxes, plan.solved, frame(pre))
            for object_id, (cx, cy) in centers.items():
                bx, by = boxes[object_id].center
                move(pre, object_id, cx - bx, cy - by)
            for u in unsolved:
                fail(plan.rules[u.index], u.clause.ids, u.message, t)
        state = evaluate(c, t, replace) if replace else pre

        # Stage 2: the label placer.
        if plan.placed:
            if replace:
                boxes = {o.id: screen_box(state, o) for o in state.objects}
            requests, rule_of = [], {}
            for rule, rel in plan.relations(("near",)):
                unit = view_of(state, state[rel.subject]).units_per_height
                b = boxes[rel.subject]
                requests.append(labels.Request(
                    rel.subject, b.w, b.h, boxes[rel.target],
                    lay["gap"] if rel.gap is None else rel.gap / unit,
                    lay["leader"] if rel.subject in plan.leaders else None,
                ))  # fmt: skip
                rule_of[rel.subject] = (rule, rel)
            # Other text is avoided as whole boxes (exact), so it is left out of the mask.
            fixed = [
                o.id for o in state.objects
                if o.id not in placed_ids and o.params.opacity > 0
                and o.impl.semantics(state).get("kind") == "label"
            ]  # fmt: skip
            fixed_labels = [boxes[i] for i in fixed]
            result = labels.place(
                requests,
                frame=frame(state),
                coverage=Coverage(oracle.coverage(state, placed_ids | set(fixed)), state.aspect),
                obstacles=fixed_labels,
                previous=previous,
            )
            for req in requests:
                p = result[req.id]
                bx, by = boxes[req.id].center
                move(state, req.id, p.center[0] - bx, p.center[1] - by)
                ax, ay = req.anchor.center
                previous[req.id] = labels.Previous(p.slot, (p.center[0] - ax, p.center[1] - ay))
                slots[req.id].append(p.slot)
                if req.id in plan.leaders:
                    tip = req.anchor.nearest_point(p.center)
                    obj = state[req.id]
                    replace[f"{req.id}.leader"] = _to_world(
                        state, obj, state.values[pos[req.id]], tip
                    )
                    replace[f"{req.id}.leader_on"] = 1.0 if p.leader else 0.0
                if p.blocked > c.theme.lint["crowded"]:
                    rule, rel = rule_of[req.id]
                    fail(rule, rel.ids, f"has no free place: even the best slot around "
                         f"{rel.target!r} is partly covered or cut by the frame", t)  # fmt: skip

        # Record this key: screen position of each solved parameter, relative to its ref.
        for path, world in replace.items():
            object_id, _, name = path.rpartition(".")
            if name == "leader_on":
                keys[path].append(world)
                continue
            point = np.asarray(view_of(pre, pre[object_id]).point(world))
            r = ref[object_id]
            if r is not None:
                point = point - np.asarray(boxes_pre_center(pre, r))
            keys[path].append((float(point[0]), float(point[1])))

    def jumps(object_id: str) -> tuple[bool, ...]:
        chosen = slots.get(object_id, ())
        return tuple(a != b for a, b in itertools.pairwise(chosen))

    tracks = tuple(
        Track(path, "scalar" if path.endswith(".leader_on") else "point",
              None if path.endswith(".leader_on") else ref[path.rpartition(".")[0]], tuple(ks),
              jumps(path.rpartition(".")[0]))
        for path, ks in sorted(keys.items())
    )  # fmt: skip
    issues = []
    for (index, message), (ids, ts) in sorted(failures.items()):
        rule = plan.rules[index]
        shown = ", ".join(f"{x:g}" for x in ts[:MAX_TIMES_IN_ISSUE])
        more = f" and {len(ts) - MAX_TIMES_IN_ISSUE} more" if len(ts) > MAX_TIMES_IN_ISSUE else ""
        issues.append(Issue(
            "unsolved", f"constraint {rule.spec.rule!r} {message} (at t = {shown}{more} s)",
            ids=ids, path=rule.path, times=tuple(ts),
            fix_hint="lower its priority, loosen it, or make room (move or shrink a neighbor)",
        ))  # fmt: skip
    kind = "analytic" if isinstance(oracle, AnalyticOracle) else "pixel"
    return Bake(times, tracks, tuple(issues), {k: tuple(v) for k, v in slots.items()}, kind)


def boxes_pre_center(pre: SceneState, object_id: str) -> tuple[float, float]:
    return screen_box(pre, pre[object_id]).center


__all__ = ["AnalyticOracle", "Bake", "Box", "OcclusionOracle", "Track", "bake", "baked_state",
           "key_times"]  # fmt: skip
