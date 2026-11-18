"""Choreography verbs (SPEC "Choreography verbs"): animate by intent, timed by the theme.

    s.at("poles_appear").reveal("pole_a", "pole_b")       # fade in, staggered
    s.at("loop_draw").trace("loop").then().reveal("loop.label")
    s.at("residue_glow").dim_others("loop", "loop.label")
    s.at("zoom").focus_pull("loop")

A verb expands into ordinary segments (and, where something must start hidden, a base
value), each tagged with the verb's name. The document stays plain data, and lint can tell
choreographed motion from raw ``to()`` transitions (``raw_motion`` in strict mode).

Pure: verbs read a compiled document and the scene at the anchor time, and return patch ops.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from phasor.core.schema import ExprSpec, MarkerRef, TimeRef
from phasor.core.state import SceneState
from phasor.time.compile import Compiled
from phasor.time.tracks import ResolvedTrack

MILESTONE_6 = {"orbit_reveal": "needs a 3D camera (Milestone 6)"}
FOCUS_MARGIN = 0.15  # focus_pull leaves this share of the frame around the object


class VerbError(ValueError):
    pass


@dataclass(frozen=True)
class Motion:
    ops: tuple[dict[str, Any], ...]
    end: float  # seconds after the anchor when the last transition ends


def shift(at: TimeRef, dt: float) -> TimeRef:
    if isinstance(at, MarkerRef):
        return MarkerRef(marker=at.marker, offset=round(at.offset + dt, 6))
    return round(float(at) + dt, 6)


def _ref(at: TimeRef) -> Any:
    return at.model_dump() if isinstance(at, MarkerRef) else at


class Verbs:
    """Verb expansion for one anchor time. ``state`` is the scene at that time."""

    def __init__(self, compiled: Compiled, state: SceneState, at: TimeRef) -> None:
        self.c, self.state, self.at = compiled, state, at
        self.t = compiled.time_map.resolve(at)
        self.motion = compiled.theme.motion

    # ---------------------------------------------------------- helpers

    def _object(self, object_id: str):
        if object_id not in self.state:
            known = ", ".join(sorted(o.id for o in self.state.objects))
            raise VerbError(f"no object {object_id!r}; objects are {known}")
        return self.state[object_id]

    def _has(self, object_id: str, param: str) -> bool:
        return param in type(self._object(object_id).params).model_fields

    def _base(self, path: str) -> Any:
        target, _, name = path.rpartition(".")
        spec = self.c.doc.object(target) or self.c.doc.camera(target)
        value = spec.params.get(name) if spec else None
        if isinstance(value, ExprSpec):
            raise VerbError(f"{path} is an expression; animate what it reads instead")
        return value

    def _starts_hidden(self, path: str) -> bool:
        """True when nothing animates ``path`` before the anchor, so a verb may set the
        base value (what the parameter is before its first transition)."""
        node = self.c.nodes.get(path)
        segs = node.segments if isinstance(node, ResolvedTrack) else ()
        return not any(s.t0 < self.t - 1e-9 for s in segs)

    def _segment(self, path: str, to: Any, verb: str, delay: float, timing: str | None = None):
        spec = self.motion[timing or verb]
        t0 = shift(self.at, delay)
        t1 = shift(self.at, delay + spec["duration"])
        seg = {"t0": _ref(t0), "t1": _ref(t1), "to": to, "ease": spec["ease"], "verb": verb}
        return {"op": "add_segment", "path": path, "segment": seg}, delay + spec["duration"]

    def _each(self, ids: Sequence[str], verb: str, make) -> Motion:
        if not ids:
            raise VerbError(f"{verb} needs at least one object id")
        ops: list[dict[str, Any]] = []
        end = 0.0
        for k, object_id in enumerate(ids):
            self._object(object_id)
            more, done = make(object_id, k * self.motion["stagger"])
            ops += more
            end = max(end, done)
        return Motion(tuple(ops), end)

    # ---------------------------------------------------------- verbs

    def _undo_dimming_before(self, object_id: str, t: float) -> list[dict[str, Any]]:
        """A reveal is a new focus: dimming applied to the object while it was still hidden
        (an earlier ``dim_others``) is dropped, so it arrives at full strength."""
        path = f"{object_id}.dim"
        segs = self.c.doc.timeline.tracks.get(path, ())
        keep = [s for s in segs if self.c.time_map.resolve(s.t0) >= t - 1e-9]
        if len(keep) == len(segs):
            return []
        ops: list[dict[str, Any]] = [{"op": "clear_track", "path": path}]
        ops += [{"op": "add_segment", "path": path, "segment": s.model_dump(mode="json")}
                for s in keep]  # fmt: skip
        return ops

    def reveal(self, *ids: str, verb: str = "reveal") -> Motion:
        """Fade in. The object is hidden until then (its base opacity becomes 0), and any
        dimming from before is dropped: what is revealed is the new focus."""

        def make(object_id: str, delay: float):
            path = f"{object_id}.opacity"
            obj = self.state[object_id]
            # params.opacity already includes dim; the opacity track is the undimmed value.
            current = obj.params.opacity / obj.params.dim if obj.params.dim > 0 else 0.0
            target = current if current > 0 else 1.0
            ops = []
            if self._starts_hidden(path):
                target = self._base(path) if self._base(path) not in (None, 0, 0.0) else 1.0
                ops.append({"op": "set_param", "path": path, "value": 0.0})
            # Opacity never overshoots, so the fade always uses the reveal timing.
            seg, done = self._segment(path, float(target), verb, delay, timing="reveal")
            undim = self._undo_dimming_before(object_id, self.t + delay)
            return [*undim, *ops, seg], done

        return self._each(ids, verb, make)

    def exit(self, *ids: str) -> Motion:
        """Fade out."""

        def make(object_id: str, delay: float):
            seg, done = self._segment(f"{object_id}.opacity", 0.0, "exit", delay)
            return [seg], done

        return self._each(ids, "exit", make)

    def trace(self, *ids: str) -> Motion:
        """Draw a curve from its start to its end. Hidden (nothing drawn) until then."""

        def make(object_id: str, delay: float):
            if not self._has(object_id, "progress"):
                kind = self.state[object_id].type
                raise VerbError(f"trace needs a curve with 'progress'; {object_id!r} is a {kind}")
            path = f"{object_id}.progress"
            ops = [{"op": "set_param", "path": path, "value": 0.0}] if self._starts_hidden(
                path) else []  # fmt: skip
            seg, done = self._segment(path, 1.0, "trace", delay)
            return [*ops, seg], done

        return self._each(ids, "trace", make)

    def bloom_in(self, *ids: str) -> Motion:
        """Fade in while growing from a point with a small overshoot (objects with a
        radius ``r``); others simply fade in."""
        faded = self.reveal(*ids, verb="bloom_in")
        ops, end = list(faded.ops), faded.end
        for k, object_id in enumerate(ids):
            if not self._has(object_id, "r"):
                continue
            path = f"{object_id}.r"
            r = self.state[object_id].params.r
            if self._starts_hidden(path):
                ops.append({"op": "set_param", "path": path, "value": r * 1e-3})
            seg, done = self._segment(path, r, "bloom_in", k * self.motion["stagger"])
            ops.append(seg)
            end = max(end, done)
        return Motion(tuple(ops), end)

    def dim_others(self, *keep: str) -> Motion:
        """Fade everything except ``keep`` (and their children) to the theme's dim level.
        Backdrops and hidden objects are left alone."""
        for object_id in keep:
            self._object(object_id)
        level = self.motion["dim_level"]
        others = [
            o.id for o in self.state.objects
            if o.id not in keep and not any(o.id.startswith(k + ".") for k in keep)
            and o.params.dim > level + 1e-9
            and not o.impl.semantics(self.state).get("backdrop", False)
        ]  # fmt: skip
        # Dimming animates ``dim``, never ``opacity``: an object revealed later is still
        # hidden until its reveal, and arrives already dimmed.
        ops = [self._segment(f"{i}.dim", level, "dim_others", 0.0, "dim")[0] for i in others]
        return Motion(tuple(ops), self.motion["dim"]["duration"] if ops else 0.0)

    def undim(self) -> Motion:
        """Bring every dimmed object back to full opacity."""
        dimmed = [o.id for o in self.state.objects if o.params.dim < 1 - 1e-9]
        ops = [self._segment(f"{i}.dim", 1.0, "undim", 0.0, "dim")[0] for i in dimmed]
        return Motion(tuple(ops), self.motion["dim"]["duration"] if ops else 0.0)

    def focus_pull(self, object_id: str, margin: float = FOCUS_MARGIN) -> Motion:
        """Move the camera so ``object_id`` fills the frame, with ``margin`` around it."""
        obj = self._object(object_id)
        if obj.layer == "overlay":
            raise VerbError(f"{object_id!r} is on the overlay; the camera cannot move to it")
        b = obj.impl.world_bounds(self.state)
        cx, cy, _ = (float(v) for v in b.center)
        w, h = b.hi[0] - b.lo[0], b.hi[1] - b.lo[1]
        height = max(h, w / self.state.aspect) * (1 + 2 * margin)
        cam = self.c.doc.timeline.camera
        z = float(self.state.values[f"{cam}.center"][2])  # keep the camera's depth
        s1, d1 = self._segment(f"{cam}.center", (cx, cy, z), "focus_pull", 0.0)
        s2, _ = self._segment(f"{cam}.height", float(height), "focus_pull", 0.0)
        return Motion((s1, s2), d1)

    def morph(self, path: str, value: Any) -> Motion:
        """Change any parameter (points of a curve, a constant, a color stop) smoothly."""
        target = path.rpartition(".")[0]
        if self.c.doc.camera(target) is None:
            self._object(target)
        seg, done = self._segment(path, value, "morph", 0.0)
        return Motion((seg,), done)

    def settle(self, path: str, value: Any) -> Motion:
        """Move to ``value`` like a released spring: overshoot, then come to rest."""
        if path.endswith(".opacity"):
            raise VerbError("settle overshoots, and opacity cannot go past 1; use reveal")
        seg, done = self._segment(path, value, "settle", 0.0)
        return Motion((seg,), done)

    def orbit_reveal(self, *_: Any, **__: Any) -> Motion:
        raise VerbError(f"orbit_reveal {MILESTONE_6['orbit_reveal']}")
