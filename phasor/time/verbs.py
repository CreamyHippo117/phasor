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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from phasor.core.objects import STRETCH_FLOOR
from phasor.core.record import record
from phasor.core.rng import rng
from phasor.core.schema import ExprSpec, MarkerRef, TimeRef
from phasor.core.state import SceneState
from phasor.time.compile import Compiled, evaluate
from phasor.time.tracks import ResolvedTrack

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

    def _written(self, path: str) -> float:
        """The document's own value of ``path`` at the anchor, without the bake: a turn is
        written relative to it, so a camera the solver aims turns by the same angle."""
        return float(evaluate(self.c, self.t).values[path])

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
        if self.c.doc.camera(cam).kind == "orbit":  # 3D: aim at it and come close enough
            radius = float(b.radius) * (1 + 2 * margin)
            fov = float(self.state.values[f"{cam}.fov"])
            dist = radius / max(1e-6, math.tan(math.radians(fov) / 2))
            s1, d1 = self._segment(f"{cam}.target", tuple(float(v) for v in b.center),
                                   "focus_pull", 0.0)  # fmt: skip
            s2, _ = self._segment(f"{cam}.distance", float(dist), "focus_pull", 0.0)
            return Motion((s1, s2), d1)
        z = float(self.state.values[f"{cam}.center"][2])  # keep the camera's depth
        s1, d1 = self._segment(f"{cam}.center", (cx, cy, z), "focus_pull", 0.0)
        s2, _ = self._segment(f"{cam}.height", float(height), "focus_pull", 0.0)
        return Motion((s1, s2), d1)

    def morph(self, path: str, value: Any, until: str | None = None) -> Motion:
        """Change any parameter (points of a curve, a constant, a color stop) smoothly, in
        the theme's ``morph`` time, or slowly until the marker ``until`` (a long, calm change:
        the constant of a function drifting while it is talked about)."""
        target = path.rpartition(".")[0]
        if self.c.doc.camera(target) is None:
            self._object(target)
        seg, done = self._segment(path, value, "morph", 0.0)
        if until is not None:
            done = self._until(until, "morph")
            seg["segment"]["t1"] = {"marker": until, "offset": 0.0}
        return Motion((seg,), done)

    def _until(self, until: str, verb: str) -> float:
        """Seconds from the anchor to the marker ``until``, which must come after it."""
        if until not in self.c.time_map.markers:
            raise VerbError(f"no marker {until!r}; markers are {sorted(self.c.time_map.markers)}")
        end = self.c.time_map.markers[until] - self.t
        if end <= 0:
            raise VerbError(f"marker {until!r} is not after the {verb} starts")
        return end

    def settle(self, path: str, value: Any) -> Motion:
        """Move to ``value`` like a released spring: overshoot, then come to rest."""
        if path.endswith(".opacity"):
            raise VerbError("settle overshoots, and opacity cannot go past 1; use reveal")
        seg, done = self._segment(path, value, "settle", 0.0)
        return Motion((seg,), done)

    def _orbit_camera(self, verb: str) -> str:
        cam = self.c.doc.timeline.camera
        if self.c.doc.camera(cam).kind != "orbit":
            raise VerbError(f"{verb} needs the 3D camera: s.use_camera('orbit')")
        return cam

    def orbit(self, degrees: float = 360.0, until: str | None = None) -> Motion:
        """Swing the camera ``degrees`` around its target (positive: counterclockwise seen
        from above). It takes the theme's ``orbit`` time per full turn, or runs until the
        marker ``until``. Camera goals keep working during it: the solver frames what you
        asked for at every angle."""
        cam = self._orbit_camera("orbit")
        path = f"{cam}.azimuth"
        start = self._written(path)
        spec = self.motion["orbit"]
        if until is None:
            seg, _ = self._segment(path, start + degrees, "orbit", 0.0)
            duration = spec["duration"] * abs(degrees) / 360.0
            seg["segment"]["t1"] = _ref(shift(self.at, duration))
            return Motion((seg,), duration)
        end = self._until(until, "orbit")
        seg = {
            "t0": _ref(self.at),
            "t1": {"marker": until, "offset": 0.0},
            "to": start + degrees,
            "ease": spec["ease"],
            "verb": "orbit",
        }
        return Motion(({"op": "add_segment", "path": path, "segment": seg},), end)

    def orbit_reveal(self, *ids: str) -> Motion:
        """Reveal objects one after another while the camera swings a little around them,
        so the eye gets depth from the motion (the theme's ``orbit_swing`` degrees)."""
        cam = self._orbit_camera("orbit_reveal")
        shown = self.reveal(*ids, verb="orbit_reveal")
        path = f"{cam}.azimuth"
        swing = self.motion["orbit_swing"]
        seg, done = self._segment(path, self._written(path) + swing, "orbit_reveal", 0.0)
        return Motion((*shown.ops, seg), max(shown.end, done))

    # ---------------------------------------------------------- the motion verbs

    def _instant(self, path: str, value: Any, verb: str, delay: float) -> dict[str, Any]:
        """A jump to ``value`` at ``delay`` seconds after the anchor."""
        t = _ref(shift(self.at, delay))
        seg = {"t0": t, "t1": t, "to": value, "ease": "linear", "verb": verb}
        return {"op": "add_segment", "path": path, "segment": seg}

    def _span(self, path: str, value: Any, verb: str, t0: float, t1: float, ease: str):
        seg = {"t0": _ref(shift(self.at, t0)), "t1": _ref(shift(self.at, t1)), "to": value,
               "ease": ease, "verb": verb}  # fmt: skip
        return {"op": "add_segment", "path": path, "segment": seg}

    def _show_at(self, object_id: str, verb: str, delay: float) -> list[dict[str, Any]]:
        """Appear (opacity 1) at ``delay``; hidden until then if nothing showed it before."""
        path = f"{object_id}.opacity"
        ops = []
        if self._starts_hidden(path):
            ops.append({"op": "set_param", "path": path, "value": 0.0})
        return [*ops, self._instant(path, 1.0, verb, delay)]

    def _of_type(self, object_id: str, verb: str, *names: str):
        obj = self._object(object_id)
        if obj.type not in names:
            raise VerbError(f"{verb} needs {' or '.join(names)}; {object_id!r} is a {obj.type}")
        return obj

    def transform(self, source: str, target: str) -> Motion:
        """An equation becoming the next: parts that appear in both glide from their old
        place (and color) to the new; parts with the same ``keys`` entry glide too, changing
        their text on the way, and arc round the parts they are carried past (the theme's
        ``transform_arc``); the old one's other parts fade out, the new one's fade in.
        ``target`` is hidden until now; ``source`` disappears as it starts."""
        a = self._of_type(source, "transform", "equation")
        self._of_type(target, "transform", "equation")
        ops: list[dict[str, Any]] = [
            {"op": "set_param", "path": f"{target}.from_parts", "value": list(a.params.parts)},
            {"op": "set_param", "path": f"{target}.from_colors", "value": list(a.params.colors)},
            # the place the script gave it; life's drift is added when drawn (``from_id``)
            {
                "op": "set_param",
                "path": f"{target}.from_pos",
                "value": [float(v) for v in self.state.values[f"{source}.pos"]],
            },
            {"op": "set_param", "path": f"{target}.from_keys", "value": list(a.params.keys)},
            {"op": "set_param", "path": f"{target}.from_id", "value": source},
            {"op": "set_param", "path": f"{target}.arc", "value": self.motion["transform_arc"]},
        ]
        if self._starts_hidden(f"{target}.morph"):
            ops.append({"op": "set_param", "path": f"{target}.morph", "value": 0.0})
        ops += self._show_at(target, "transform", 0.0)
        ops.append(self._instant(f"{source}.opacity", 0.0, "transform", 0.0))
        seg, done = self._segment(f"{target}.morph", 1.0, "transform", 0.0)
        return Motion((*ops, seg), done)

    def become(self, source: str, target: str) -> Motion:
        """One curve turning into another (a circle into a square): ``source``'s points move
        onto ``target``'s (both resampled to the same count, closed curves aligned so the
        motion is short), then ``target`` takes its place."""
        a = self._of_type(source, "become", "line")
        b = self._of_type(target, "become", "line")
        path = f"{source}.points"
        if not self._starts_hidden(path) or isinstance(self._base(path), ExprSpec):
            raise VerbError(f"become needs {source!r}'s points unanimated before it")
        pa, pb = _matched_shapes(np.asarray(a.params.points), np.asarray(b.params.points))
        spec = self.motion["become"]
        d = spec["duration"]
        # The same shape with more points (a track keeps one point count throughout).
        ops: list[dict[str, Any]] = [{"op": "set_param", "path": path, "value": pa.tolist()}]
        ops.append(self._span(path, pb.tolist(), "become", 0.0, d, spec["ease"]))
        ops += self._show_at(target, "become", d)
        ops.append(self._instant(f"{source}.opacity", 0.0, "become", d))
        return Motion(tuple(ops), d)

    def write(self, *ids: str) -> Motion:
        """Write text or an equation in, left to right (hidden until now)."""

        def make(object_id: str, delay: float):
            if not self._has(object_id, "write"):
                kind = self.state[object_id].type
                raise VerbError(f"write needs text or an equation; {object_id!r} is a {kind}")
            path = f"{object_id}.write"
            ops = [{"op": "set_param", "path": path, "value": 0.0}] if self._starts_hidden(
                path) else []  # fmt: skip
            seg, done = self._segment(path, 1.0, "write", delay)
            return [*ops, seg], done

        return self._each(ids, "write", make)

    def travel(self, *ids: str) -> Motion:
        """Send tracers along their paths, start to end, trailing light."""

        def make(object_id: str, delay: float):
            obj = self._of_type(object_id, "travel", "tracer")
            if obj.impl.semantics(self.state).get("path_missing"):
                raise VerbError(f"{object_id!r} travels along {obj.params.path!r}, which is not "
                                "a curve in the scene")  # fmt: skip
            path = f"{object_id}.progress"
            ops = [{"op": "set_param", "path": path, "value": 0.0}] if self._starts_hidden(
                path) else []  # fmt: skip
            seg, done = self._segment(path, 1.0, "travel", delay)
            return [*ops, seg], done

        return self._each(ids, "travel", make)

    def flash(self, *ids: str) -> Motion:
        """A quick brightening: light and glow swell, then fall back."""
        spec = self.motion["flash"]
        d = spec["duration"]

        def make(object_id: str, delay: float):
            path = f"{object_id}.flash"
            up = self._span(path, 1.0, "flash", delay, delay + 0.3 * d, spec["ease"])
            down = self._span(path, 0.0, "flash", delay + 0.3 * d, delay + d, "smooth")
            return [up, down], delay + d

        return self._each(ids, "flash", make)

    def circumscribe(self, object_id: str, shape: str = "circle") -> Motion:
        """Draw a ring (or a box) around an object, then fade it: "look here". The ring is
        the object's child ``<id>.ring``, made the first time and reused after."""
        self._object(object_id)
        if shape not in ("circle", "box"):
            raise VerbError(f"circumscribe draws a 'circle' or a 'box', not {shape!r}")
        ring = f"{object_id}.ring"
        spec = self.motion["circumscribe"]
        d = spec["duration"]
        ops: list[dict[str, Any]] = []
        if ring not in self.state:
            ops.append({"op": "add_object", "object": {"id": ring, "type": "callout", "params": {
                "target": object_id, "shape": shape, "progress": 0.0}}})  # fmt: skip
        else:
            ops += [self._instant(f"{ring}.progress", 0.0, "circumscribe", 0.0),
                    self._instant(f"{ring}.opacity", 1.0, "circumscribe", 0.0)]  # fmt: skip
        ops.append(self._span(f"{ring}.progress", 1.0, "circumscribe", 0.0, 0.6 * d, spec["ease"]))
        ops.append(self._span(f"{ring}.opacity", 0.0, "circumscribe", 0.6 * d, d, "smooth"))
        return Motion(tuple(ops), d)

    def wiggle(self, *ids: str) -> Motion:
        """A small shake from side to side: "this one"."""
        spec = self.motion["wiggle"]
        d = spec["duration"]

        def make(object_id: str, delay: float):
            if not self._has(object_id, "pos"):
                kind = self.state[object_id].type
                raise VerbError(f"wiggle moves an object's 'pos'; {object_id!r} is a {kind}")
            path = f"{object_id}.pos"
            self._base(path)  # refuses an expression
            obj = self.state[object_id]
            p0 = np.asarray(obj.params.pos, dtype=np.float64)
            size = float(obj.impl.world_bounds(self.state).radius)
            side = np.asarray(self.state.camera_for(obj).right) * max(0.06 * size, 1e-3)
            steps = (p0 + side, p0 - side, p0 + side / 2, p0)
            ops = [self._span(path, [float(v) for v in q], "wiggle", delay + k * d / 4,
                              delay + (k + 1) * d / 4, "smooth")
                   for k, q in enumerate(steps)]  # fmt: skip
            return ops, delay + d

        return self._each(ids, "wiggle", make)

    def sweep(self, path: str, value: float, show: bool = True) -> Motion:
        """Run a number from where it is to ``value`` at an even pace, showing it as it goes
        (``show``: a ``Counter`` beside the object, ``<id>.readout``, made the first time)."""
        target, _, name = path.rpartition(".")
        obj = self._object(target)
        current = getattr(obj.params, name, None)
        if not isinstance(current, int | float) or isinstance(current, bool):
            raise VerbError(f"sweep runs a number; {path} is {type(current).__name__}")
        self._base(path)  # refuses an expression
        seg, done = self._segment(path, float(value), "sweep", 0.0)
        ops: list[dict[str, Any]] = []
        readout = f"{target}.readout"
        if show and readout not in self.state:
            whole = type(obj.params).model_fields[name].annotation is int
            label = name.replace("_", "\\_")  # mathtext: an underscore is a subscript
            params = {
                "value": {"expr": path},
                "digits": 0 if whole else 2,
                "prefix": f"\\mathrm{{{label}}} = ",
                "opacity": 0.0,
            }
            ops.append(
                {"op": "add_object", "object": {"id": readout, "type": "counter", "params": params}}
            )
            ops.append({"op": "add_constraint", "rule": f"{readout} near {target}"})
            ops.append(self._instant(f"{readout}.opacity", 1.0, "sweep", 0.0))
        return Motion((*ops, seg), done)

    # ===== group S =====
    # Scale verbs. They animate ``<id>.stretch`` (ParamsBase: a per-axis scale of what the
    # object draws, about its ``pos`` or else its center), owned by these verbs like ``dim``
    # is by dim_others, so they work on any object and never fight a type's own size.

    def _stretch(self, ids: Sequence[str], verb: str, target) -> Motion:
        """Animate each object's stretch to ``target(current)`` (staggered), where
        ``current`` is its stretch at the anchor."""

        def make(object_id: str, delay: float):
            obj = self._object(object_id)
            if callable(getattr(obj.impl, "build", None)):
                parts = [o.id for o in self.state.objects if o.parent == object_id]
                raise VerbError(f"{verb} scales an object that draws itself; {object_id!r} is "
                                f"a composite of {parts}, which its build makes")  # fmt: skip
            path = f"{object_id}.stretch"
            self._base(path)  # refuses an expression
            now = tuple(float(v) for v in obj.params.stretch)
            to = [max(float(v), STRETCH_FLOOR) for v in target(now)]
            seg, done = self._segment(path, to, verb, delay)
            return [seg], done

        return self._each(ids, verb, make)

    @staticmethod
    def _factor(verb: str, by: float) -> float:
        if isinstance(by, bool) or not isinstance(by, int | float) or not by > 0:
            raise VerbError(f"{verb} scales by a positive number, got by={by!r}")
        return float(by)

    @staticmethod
    def _axis(verb: str, axis: str) -> int:
        if axis not in ("x", "y", "z"):
            raise VerbError(f"{verb} works along axis 'x', 'y' or 'z', not {axis!r}")
        return "xyz".index(axis)

    def grow(self, *ids: str, by: float = 1.5) -> Motion:
        """Scale objects up by ``by`` about their own center (their ``pos``, else the middle
        of what they draw), staggered. It multiplies what is there, so a squish is kept."""
        k = self._factor("grow", by)
        return self._stretch(ids, "grow", lambda now: [v * k for v in now])

    def shrink(self, *ids: str, by: float = 0.5) -> Motion:
        """Scale objects down by ``by`` (0.5: to half) about their own center, staggered."""
        k = self._factor("shrink", by)
        return self._stretch(ids, "shrink", lambda now: [v * k for v in now])

    def squish(self, *ids: str, axis: str = "y", to: float = 0.4) -> Motion:
        """Squash objects along one axis to ``to`` of their size (``to=1``: back to normal),
        keeping the other axes; staggered."""
        i = self._axis("squish", axis)
        if isinstance(to, bool) or not isinstance(to, int | float) or not to >= 0:
            raise VerbError(f"squish takes to= a size share of 0 or more, got to={to!r}")
        return self._stretch(ids, "squish", lambda now: (*now[:i], float(to), *now[i + 1 :]))

    def flatten(self, *ids: str, axis: str = "z") -> Motion:
        """Squash objects flat along ``axis`` (onto a plane through their center; in a 2D
        scene use 'x' or 'y'). ``squish(..., axis=..., to=1)`` brings them back."""
        i = self._axis("flatten", axis)
        return self._stretch(ids, "flatten", lambda now: (*now[:i], STRETCH_FLOOR, *now[i + 1 :]))

    # ===== end of group S =====

    # ===== group A =====

    # -- shared: tracks that must stay in time order, and steps that compose

    def _track(self, path: str) -> list[Any]:
        return list(self.c.doc.timeline.tracks.get(path, ()))

    def _initial(self, path: str) -> Any:
        """The value of ``path`` before any of its segments."""
        node = self.c.nodes.get(path)
        if isinstance(node, ResolvedTrack):
            return node.initial
        if type(node).__name__ == "Const":
            return node.value
        raise VerbError(f"{path} is an expression; animate what it reads instead")

    @staticmethod
    def _as_ref(v: Any) -> TimeRef:
        return MarkerRef(**v) if isinstance(v, dict) else float(v)

    @staticmethod
    def _plain(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (tuple, list)):
            return [Verbs._plain(x) for x in v]
        return float(v) if isinstance(v, (float, np.floating)) else v

    def _in_order(self, ops: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """``ops`` with every new segment placed in time order on its track: one that comes
        before a segment already written is merged in, so the document does not depend on
        the order the script wrote its verbs in."""
        out = [o for o in ops if o["op"] != "add_segment"]
        by_path: dict[str, list[dict[str, Any]]] = {}
        for o in ops:
            if o["op"] == "add_segment":
                by_path.setdefault(o["path"], []).append(o["segment"])
        res = self.c.time_map.resolve
        for path, new in by_path.items():
            old = self._track(path)
            if not old or res(self._as_ref(new[0]["t0"])) >= res(old[-1].t1) - 1e-9:
                out += [{"op": "add_segment", "path": path, "segment": s} for s in new]
                continue
            both = [s.model_dump(mode="json") for s in old] + new
            both.sort(key=lambda s: (res(self._as_ref(s["t0"])), res(self._as_ref(s["t1"]))))
            out.append({"op": "clear_track", "path": path})
            out += [{"op": "add_segment", "path": path, "segment": s} for s in both]
        return out

    def _compose(self, verb, paths, element, advance, decode, key, check=None):
        """Add one step (``element``) that acts on what is there at the anchor, so steps pile
        up: each new value is the step applied to the one before. Steps written for the same
        anchor (or out of time order) are put in a fixed order (by ``key``), those that
        would overlap are pushed back to follow each other, and every target is worked out
        again, so the document is the same whatever order the script wrote them in.
        ``advance(state, element)`` is the next state, ``decode(before, after)`` the element
        that led from one state to the next. Returns (ops, seconds after the anchor)."""
        spec = self.motion[verb]
        res = self.c.time_map.resolve
        tracks = {p: self._track(p) for p in paths}
        head, tail = {}, {}
        for p, segs in tracks.items():
            head[p] = [s for s in segs if res(s.t0) < self.t - 1e-9]
            tail[p] = [s for s in segs if res(s.t0) >= self.t - 1e-9]
            if any(s.verb != verb for s in tail[p]):
                raise VerbError(f"{verb} cannot be written before other motion of {p}; "
                                "write it after that motion")  # fmt: skip
        count = len(tail[paths[0]])
        if any(len(tail[p]) != count for p in paths):
            raise VerbError(f"the {verb} steps of {paths[0].partition('.')[0]!r} were edited "
                            "by hand; remove them and write the verbs again")  # fmt: skip
        start = tuple(head[p][-1].to if head[p] else self._initial(p) for p in paths)
        if check is not None:
            check(start)
        state, entries = start, []
        for i in range(count):
            segs = [tail[p][i] for p in paths]
            after = tuple(s.to for s in segs)
            t0 = res(segs[0].t0)
            entries.append({"elem": decode(state, after), "t0ref": segs[0].t0, "t0": t0,
                            "dur": res(segs[0].t1) - t0, "ease": segs[0].ease,
                            "new": False})  # fmt: skip
            state = after
        entries.append({"elem": element, "t0ref": self.at, "t0": self.t, "ease": spec["ease"],
                        "dur": spec["duration"], "new": True})  # fmt: skip
        entries.sort(key=lambda e: (e["t0"], key(e["elem"])))
        cursor, cursor_ref = -math.inf, None
        for p in paths:
            if head[p] and res(head[p][-1].t1) > cursor:
                cursor, cursor_ref = res(head[p][-1].t1), head[p][-1].t1
        ops: list[dict[str, Any]] = []
        for p in paths:
            if tracks[p]:
                ops.append({"op": "clear_track", "path": p})
            ops += [{"op": "add_segment", "path": p, "segment": s.model_dump(mode="json")}
                    for s in head[p]]  # fmt: skip
        state, end = start, 0.0
        for e in entries:
            if e["t0"] < cursor - 1e-9 and cursor_ref is not None:
                e["t0ref"] = cursor_ref  # pushed back to follow the step before it
            t1ref = shift(e["t0ref"], e["dur"])
            cursor, cursor_ref = res(t1ref), t1ref
            state = advance(state, e["elem"])
            for p, value in zip(paths, state, strict=True):
                seg = {"t0": _ref(e["t0ref"]), "t1": _ref(t1ref), "to": self._plain(value),
                       "ease": e["ease"], "verb": verb}  # fmt: skip
                ops.append({"op": "add_segment", "path": p, "segment": seg})
            if e["new"]:
                end = cursor - self.t
        return ops, end

    # -- act: a group element turns or flips a polygon

    def act(self, polygon_id: str, element: Any) -> Motion:
        """A symmetry acts on a polygon: it turns and flips on its ``rotation`` and ``flip``
        (the vertex numbers travel with it), on top of what it already did. ``element`` is a
        dihedral element, by label (``"r"``, ``"s"``, ``"sr2"``) or number."""
        obj = self._of_type(polygon_id, "act", "polygon")
        try:
            e = obj.impl.element_index(self.state, element)
        except ValueError as err:
            raise VerbError(str(err)) from None
        n = obj.params.n
        step = 360.0 / n
        paths = (f"{polygon_id}.rotation", f"{polygon_id}.flip")

        def advance(state, elem):
            theta, flip = float(state[0]), float(state[1])
            turn = (elem % n) * step
            turn -= 360.0 if turn > 180.0 + 1e-9 else 0.0  # the short way round
            sign = -1.0 if flip == 1.0 else 1.0  # on the mirrored side a turn reads clockwise
            return theta + sign * turn, (1.0 - flip if elem >= n else flip)

        def decode(before, after):
            sign = -1.0 if float(before[1]) == 1.0 else 1.0
            k = round(sign * (float(after[0]) - float(before[0])) / step) % n
            return k + (n if round(float(after[1])) != round(float(before[1])) else 0)

        def check(state):
            theta, flip = float(state[0]), float(state[1])
            if abs(theta / step - round(theta / step)) > 1e-6 or flip not in (0.0, 1.0):
                raise VerbError(f"act needs {polygon_id!r} on a symmetry (a rotation that is a "
                                f"multiple of {step:g} degrees, flip 0 or 1), not rotation "
                                f"{theta:g} and flip {flip:g}")  # fmt: skip

        ops, end = self._compose("act", paths, e, advance, decode, key=lambda x: x, check=check)
        return Motion(tuple(ops), end)

    # -- apply: a matrix composes onto a transform grid's current matrix

    def apply(self, grid_id: str, matrix: Any) -> Motion:
        """A matrix acts on what the grid already shows: the grid's matrix becomes
        ``matrix @ current``, moving linearly (2x2 or 3x3, as the grid is). Applies pile up
        in time order; applies at one marker go in a fixed order, so the document does not
        depend on how the script wrote them."""
        obj = self._of_type(grid_id, "apply", "transform_grid")
        if obj.params.flow:
            raise VerbError(f"{grid_id!r} shows a flow, which ignores its matrix; "
                            "clear 'flow' first")  # fmt: skip
        path = f"{grid_id}.matrix"
        try:
            m = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError):
            raise VerbError(f"apply needs a matrix of numbers, not {matrix!r}") from None
        d = len(obj.params.matrix)
        if m.shape != (d, d):
            raise VerbError(f"{grid_id!r} is {d}x{d}; apply needs a {d}x{d} matrix, "
                            f"not shape {m.shape}")  # fmt: skip
        if not np.isfinite(m).all():
            raise VerbError("apply needs a finite matrix")

        def advance(state, elem):
            return (np.asarray(elem) @ np.asarray(state[0], dtype=float),)

        def decode(before, after):
            prev = np.asarray(before[0], dtype=float)
            if abs(np.linalg.det(prev)) < 1e-9:
                raise VerbError("apply cannot put a step before one that follows a singular "
                                "matrix; write the applies in time order")  # fmt: skip
            return np.asarray(after[0], dtype=float) @ np.linalg.inv(prev)

        def key(x):
            return tuple(float(v) for v in np.round(np.asarray(x).ravel(), 9))

        ops, end = self._compose("apply", (path,), m, advance, decode, key=key)
        return Motion(tuple(ops), end)

    # -- generate: a Cayley graph grows outward from the identity

    def generate(self, graph_id: str, by: Any = None) -> Motion:
        """A Cayley graph grows from the identity, breadth first, one generator after
        another: ``by`` names the generators in the order they join (``"r s"``; default
        the graph's own). Its ``reach`` goes 0 to 1."""
        obj = self._of_type(graph_id, "generate", "cayley_graph")
        order = " ".join(by) if isinstance(by, (list, tuple)) else (by or "")
        gens = obj.params.generators.split()
        unknown = [x for x in order.split() if x not in gens]
        if unknown:
            raise VerbError(f"{unknown[0]!r} is not a generator of {graph_id!r}; "
                            f"its generators are {gens}")  # fmt: skip
        ops: list[dict[str, Any]] = []
        have = self._base(f"{graph_id}.order") or ""
        if order and have and have != order:
            raise VerbError(f"{graph_id!r} already grows in the order {have!r}")
        if order and have != order:
            ops.append({"op": "set_param", "path": f"{graph_id}.order", "value": order})
        path = f"{graph_id}.reach"
        if self._starts_hidden(path):
            ops.append({"op": "set_param", "path": path, "value": 0.0})
        seg, done = self._segment(path, 1.0, "generate", 0.0)
        return Motion(tuple(self._in_order([*ops, seg])), done)

    # -- coset and quotient

    def _cosets(self, verb: str, object_id: str, subgroup: Any, normal: bool = False):
        obj = self._of_type(object_id, verb, "clock_diagram", "cayley_graph")
        try:
            info = obj.impl.coset_info(self.state, subgroup)
        except ValueError as err:
            raise VerbError(str(err)) from None
        if normal and not info["normal"]:
            raise VerbError(f"{info['subgroup']} is not a normal subgroup of "
                            f"{info['group']}; its cosets do not make a quotient")  # fmt: skip
        ops: list[dict[str, Any]] = []
        if not info["same"]:
            if any(self._track(f"{object_id}.{x}") for x in ("gather", "collapse")):
                raise VerbError(f"{object_id!r} already gathers by a different subgroup; "
                                "one diagram shows one subgroup")  # fmt: skip
            for name, value in info["set"].items():
                ops.append({"op": "set_param", "path": f"{object_id}.{name}", "value": value})
        return info, ops

    def coset(self, object_id: str, subgroup: Any = None) -> Motion:
        """The elements of a clock or Cayley graph gather by coset of ``subgroup``: each
        coset glides into its own cluster (colored as its coset, the subgroup in the
        accent color). ``subgroup`` is what generates it (a step for a clock, labels such
        as ``"r2"`` for a graph); leave it out to use the diagram's own."""
        _, ops = self._cosets("coset", object_id, subgroup)
        path = f"{object_id}.gather"
        if self._starts_hidden(path):
            ops.append({"op": "set_param", "path": path, "value": 0.0})
        seg, done = self._segment(path, 1.0, "coset", 0.0)
        return Motion(tuple(self._in_order([*ops, seg])), done)

    def quotient(self, object_id: str, subgroup: Any = None) -> Motion:
        """Each coset collapses to a single node: the picture of G/H. The subgroup must be
        normal (else its cosets do not multiply). Works with or without ``coset`` before it:
        the elements glide to their coset's place either way."""
        _, ops = self._cosets("quotient", object_id, subgroup, normal=True)
        path = f"{object_id}.collapse"
        if self._starts_hidden(path):
            ops.append({"op": "set_param", "path": path, "value": 0.0})
        seg, done = self._segment(path, 1.0, "quotient", 0.0)
        return Motion(tuple(self._in_order([*ops, seg])), done)

    # -- permute: tokens swap places along arcs

    def permute(self, perm_id: str, sigma: Any) -> Motion:
        """The tokens of a ``Permutation`` swap places along arcs, from where they are now to
        the arrangement ``sigma``: token ``i`` ends in slot ``sigma[i]`` (from 0), or write
        cycles with the numbers on the tokens, ``"(1 2 3)(4 5)"``. Tokens keep the color of
        the cycle they move in."""
        obj = self._of_type(perm_id, "permute", "permutation")
        try:
            target = obj.impl.parse_arrangement(self.state, sigma)
        except ValueError as err:
            raise VerbError(str(err)) from None
        n = len(target)
        res = self.c.time_map.resolve
        paths = [f"{perm_id}.{x}" for x in ("arrangement", "perm", "progress")]
        tracks = {p: self._track(p) for p in paths}
        for p, segs in tracks.items():
            if any(s.verb != "permute" for s in segs):
                raise VerbError(f"permute cannot share {p} with other motion")
        base = self._initial(paths[0])
        ops: list[dict[str, Any]] = []
        if not base:
            base = tuple(range(n))
            ops.append({"op": "set_param", "path": paths[0], "value": list(base)})
        arr, mov, prog = (tracks[p] for p in paths)
        if not (len(arr) == len(mov) and len(prog) == 2 * len(arr)):
            raise VerbError(f"the permutes of {perm_id!r} were edited by hand; remove "
                            "them and write the verbs again")  # fmt: skip
        spec = self.motion["permute"]
        entries = []  # one per permute: where every token ends up, and when it moves
        for i in range(len(arr)):
            move = prog[2 * i + 1]
            ends = tuple(int(mov[i].to[int(a)]) for a in arr[i].to)  # token -> its slot
            entries.append({"t0ref": move.t0, "t1ref": move.t1, "ease": move.ease,
                            "target": ends, "new": False})  # fmt: skip
        entries.append({"t0ref": self.at, "t1ref": shift(self.at, spec["duration"]),
                        "ease": spec["ease"], "target": target, "new": True})  # fmt: skip
        for e in entries:
            e["t0"] = res(e["t0ref"])
        entries.sort(key=lambda e: e["t0"])
        for k in range(len(entries) - 1):
            a, b = entries[k], entries[k + 1]
            if abs(a["t0"] - b["t0"]) < 1e-9:
                raise VerbError(f"{perm_id!r} already permutes at this moment; put the "
                                "second one at a later marker (or use .then())")  # fmt: skip
        ops += [{"op": "clear_track", "path": p} for p in paths if tracks[p]]
        here, end = tuple(int(x) for x in base), 0.0
        for e in entries:
            token_at = [0] * n
            for token, slot in enumerate(here):
                token_at[slot] = token
            move = [e["target"][token_at[slot]] for slot in range(n)]  # slot -> slot it goes to
            t0, t1 = _ref(e["t0ref"]), _ref(e["t1ref"])
            for path, value in ((paths[0], list(here)), (paths[1], move), (paths[2], 0.0)):
                seg = {"t0": t0, "t1": t0, "to": value, "ease": "linear", "verb": "permute"}
                ops.append({"op": "add_segment", "path": path, "segment": seg})
            seg = {"t0": t0, "t1": t1, "to": 1.0, "ease": e["ease"], "verb": "permute"}
            ops.append({"op": "add_segment", "path": paths[2], "segment": seg})
            here = tuple(int(x) for x in e["target"])
            if e["new"]:
                end = res(e["t1ref"]) - self.t
        return Motion(tuple(ops), end)

    # ===== end of group A =====

    # ===== group E =====
    # Equations and emphasis: see docs/ai/verbs_equations.md

    def _part_index(self, eq, key: str, verb: str) -> int:
        keys = list(eq.params.keys[: len(eq.params.parts)])
        if not key or key not in keys:
            raise VerbError(f"{verb} addresses a part by its key; {eq.id!r} has keys "
                            f"{[k for k in keys if k]} (give the Equation `keys=`)")  # fmt: skip
        return keys.index(key)

    def _center_of(self, object_id: str, verb: str) -> list[float]:
        obj = self._of_type(object_id, verb, "equation", "counter", "text")
        feats = obj.impl.features(self.state)
        c = feats.get("center")
        c = obj.impl.world_bounds(self.state).center if c is None else c
        return [float(v) for v in c]

    def substitute(self, eq_id: str, key: str, value: Any, source: str | None = None) -> Motion:
        """Replace the part ``key`` of an equation by ``value`` (mathtext such as ``"$3$"``, or
        a number): the old text leaves where it stands and the new flies in from the object
        ``source`` (an equation, counter or text), or fades in where it is when there is none."""
        from phasor.core import text as text_metrics

        eq = self._of_type(eq_id, "substitute", "equation")
        j = self._part_index(eq, key, "substitute")
        if eq.params.swap_key:
            raise VerbError(
                f"{eq_id!r} already has a substitution ({eq.params.swap_key!r}); an equation "
                "takes one: `transform` to a new equation for the next"
            )
        new = f"${value:g}$" if isinstance(value, int | float) and not isinstance(value, bool) \
            else str(value)  # fmt: skip
        try:
            text_metrics.check_math(new)
        except ValueError as e:
            raise VerbError(f"substitute cannot typeset {new!r}: {e}") from None
        old = eq.params.parts[j]
        if new == old:
            raise VerbError(f"part {key!r} of {eq_id!r} already reads {old!r}")
        origin = None if source is None else self._center_of(source, "substitute")
        parts = list(eq.params.parts)
        parts[j] = new
        ops: list[dict[str, Any]] = [
            {"op": "set_param", "path": f"{eq_id}.parts", "value": parts},
            {"op": "set_param", "path": f"{eq_id}.swap_key", "value": key},
            {"op": "set_param", "path": f"{eq_id}.swap_old", "value": old},
            {"op": "set_param", "path": f"{eq_id}.swap_from", "value": origin or [0.0, 0.0, 0.0]},
            {"op": "set_param", "path": f"{eq_id}.swap_fly", "value": origin is not None},
        ]
        if self._starts_hidden(f"{eq_id}.swap"):
            ops.append({"op": "set_param", "path": f"{eq_id}.swap", "value": 0.0})
        seg, done = self._segment(f"{eq_id}.swap", 1.0, "substitute", 0.0)
        return Motion((*ops, seg), done)

    def cancel(self, eq_id: str, *keys: str) -> Motion:
        """Strike through two (or more) terms of an equation, then fade them and close the
        gap: the others glide together. Terms are addressed by the equation's ``keys``."""
        eq = self._of_type(eq_id, "cancel", "equation")
        if len(keys) < 2 or len(set(keys)) != len(keys):
            raise VerbError("cancel needs two or more different keys of terms to cross out")
        for key in keys:
            self._part_index(eq, key, "cancel")
        if eq.params.struck:
            raise VerbError(f"{eq_id!r} already has terms crossed out; `transform` to a new "
                            "equation for the next cancellation")  # fmt: skip
        order = [k for k in eq.params.keys if k in keys]  # the equation's order, whatever was typed
        spec = self.motion["cancel"]
        d = spec["duration"]
        ops: list[dict[str, Any]] = [{"op": "set_param", "path": f"{eq_id}.struck", "value": order}]
        for name in ("strike", "close"):
            if self._starts_hidden(f"{eq_id}.{name}"):
                ops.append({"op": "set_param", "path": f"{eq_id}.{name}", "value": 0.0})
        ops.append(self._span(f"{eq_id}.strike", 1.0, "cancel", 0.0, 0.5 * d, "smooth"))
        ops.append(self._span(f"{eq_id}.close", 1.0, "cancel", 0.5 * d, d, spec["ease"]))
        return Motion(tuple(ops), d)

    def _carry(self, source: str, target: str, verb: str) -> Motion:
        """``transform`` with the verb's own timing and tag, and its own way round: ``factor``
        draws terms together under a bracket, ``expand`` spreads them over the others."""
        base = self.transform(source, target)
        ops: list[dict[str, Any]] = []
        done = 0.0
        for op in base.ops:
            if op["op"] == "add_segment":
                seg = op["segment"]
                if seg["t0"] != seg["t1"]:  # the glide: the verb's own length and ease
                    op, done = self._segment(op["path"], seg["to"], verb, 0.0)
                else:
                    op = {**op, "segment": {**seg, "verb": verb}}
            elif op["path"].endswith(".arc"):
                op = {**op, "value": op["value"] * 1.4 + 0.3}
            ops.append(op)
        ops.append({"op": "set_param", "path": f"{target}.arc_mode", "value": verb})
        return Motion(tuple(ops), done)

    def factor(self, source: str, target: str) -> Motion:
        """An expression becoming its factored form: matching terms (same text or same
        ``keys``) travel under the others and are drawn together beneath a bracket."""
        return self._carry(source, target, "factor")

    def expand(self, source: str, target: str) -> Motion:
        """An expression becoming its expanded form: matching terms travel over the others
        and spread apart."""
        return self._carry(source, target, "expand")

    def ignite(self, *ids: str, settle: float = 0.3) -> Motion:
        """Light a stroke like a fuse: a bright head runs along it (a line or any curve) leaving
        it lit behind, then the whole stroke settles brighter (``settle`` of a flash). The
        head is the child ``<id>.fuse``, made the first time and reused after."""
        spec = self.motion["ignite"]
        d = spec["duration"]
        if not 0.0 <= settle <= 1.0:
            raise VerbError(f"ignite: settle is a share of a flash, 0 to 1, not {settle!r}")

        def make(object_id: str, delay: float):
            obj = self._object(object_id)
            path = getattr(obj.impl, "path_points", None)
            if path is None or len(np.asarray(path(self.state))) < 2:
                raise VerbError(f"ignite needs a stroke with a path (a line or a curve); "
                                f"{object_id!r} is a {obj.type}")  # fmt: skip
            fuse = f"{object_id}.fuse"
            ops: list[dict[str, Any]] = []
            if fuse not in self.state:
                params = {"path": object_id, "progress": 0.0, "trail": 1.0, "lit": 1.0,
                          "opacity": 0.0}  # fmt: skip
                made = {"id": fuse, "type": "tracer", "params": params}
                ops.append({"op": "add_object", "object": made})
            run = 0.7 * d
            flash = f"{object_id}.flash"
            ops += [
                self._instant(f"{fuse}.progress", 0.0, "ignite", delay),
                self._instant(f"{fuse}.opacity", 1.0, "ignite", delay),
                self._span(f"{fuse}.progress", 1.0, "ignite", delay, delay + run, spec["ease"]),
                self._span(flash, 1.0, "ignite", delay + run, delay + 0.85 * d, "out"),
                self._span(flash, settle, "ignite", delay + 0.85 * d, delay + d, "smooth"),
                self._span(f"{fuse}.opacity", 0.0, "ignite", delay + run, delay + d, "smooth"),
            ]
            return ops, delay + d

        return self._each(ids, "ignite", make)

    def pulse(self, *ids: str, n: int = 2) -> Motion:
        """``flash`` ``n`` times, quick and even: light and glow swell and fall back, beat
        after beat (a beat is half the theme's ``pulse`` duration)."""
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 12:
            raise VerbError(f"pulse: n is a count of beats from 1 to 12, not {n!r}")
        spec = self.motion["pulse"]
        beat = spec["duration"] / 2

        def make(object_id: str, delay: float):
            ops = []
            for k in range(n):
                t0 = delay + k * beat
                ops.append(self._span(f"{object_id}.flash", 1.0, "pulse", t0, t0 + 0.35 * beat,
                                      spec["ease"]))  # fmt: skip
                ops.append(self._span(f"{object_id}.flash", 0.0, "pulse", t0 + 0.35 * beat,
                                      t0 + beat, "smooth"))  # fmt: skip
            return ops, delay + n * beat

        return self._each(ids, "pulse", make)

    # ===== end of group E =====

    # ===== group F1 =====

    def _spot(self, object_id: str, at: Any, index: int, verb: str) -> dict[str, Any]:
        """Params naming a spot on ``object_id``: one of its features (point ``index``) or,
        for ``at`` = (x, y[, z]), a point."""
        obj = self._object(object_id)
        if isinstance(at, str):
            feats = dict(obj.impl.features(self.state))
            feats.setdefault("center", obj.impl.world_bounds(self.state).center)  # always
            if at not in feats:
                raise VerbError(f"{verb}: {object_id!r} ({obj.type}) has no feature {at!r}; its "
                                f"features are {sorted(feats)}")  # fmt: skip
            count = len(np.asarray(feats[at], dtype=np.float64).reshape(-1, 3))
            if not 0 <= index < count:
                raise VerbError(f"{verb}: {object_id!r} has {count} {at!r} point(s) now, so "
                                f"index {index} is not one of them")  # fmt: skip
            return {"target": object_id, "feature": at, "index": int(index)}
        try:
            pt = [float(v) for v in at]
        except (TypeError, ValueError):
            raise VerbError(f"{verb}: at= is a feature name or a point, not {at!r}") from None
        if len(pt) not in (2, 3):
            raise VerbError(f"{verb}: a point has 2 or 3 coordinates, got {pt}")
        return {"target": object_id, "feature": "", "point": [*pt, 0.0][:3]}

    def follow(self, object_id: str, feature: str = "center", index: int = 0,
               until: str | None = None) -> Motion:  # fmt: skip
        """Keep the camera on point ``index`` of ``object_id``'s ``feature`` (a zero of a
        domain coloring whose constant moves, a tracer's head) for the theme's ``follow``
        time, or until the marker ``until``: the camera's center (2D) or target (3D) glides
        onto it, rides along, and glides back. The probe ``<id>.follow`` is made the first
        time; the layout bake moves the camera (the document's center/target stay unset)."""
        if not isinstance(feature, str):
            raise VerbError("follow: feature= is the name of one of the object's features")
        spec = self.motion["follow"]
        d = self._until(until, "follow") if until is not None else spec["duration"]
        params = self._spot(object_id, feature, index, "follow")
        probe = f"{object_id}.follow"
        ops: list[dict[str, Any]] = []
        if probe not in self.state:
            ops.append({"op": "add_object", "object": {"id": probe, "type": "follow", "params": {
                **params, "on": 0.0}}})  # fmt: skip
        else:
            old = self.state[probe].params
            if (old.feature, old.index) != (params["feature"], params["index"]):
                raise VerbError(f"{object_id!r} is already followed at {old.feature!r}"
                                f"[{old.index}]; follow one feature per object")  # fmt: skip
        ramp = min(0.25 * spec["duration"], d / 2)
        path = f"{probe}.on"
        ops.append(self._span(path, 1.0, "follow", 0.0, ramp, spec["ease"]))
        ops.append(self._span(path, 0.0, "follow", d - ramp, d, spec["ease"]))
        return Motion(tuple(ops), d)

    def ripple(self, object_id: str, at: Any = "center", index: int = 0, rings: int = 3) -> Motion:
        """Rings of light spread from a spot on ``object_id`` and fade: ``at`` is one of its
        features (point ``index`` of a domain coloring's ``"zeros"``, a ``"center"``) or a
        point (x, y). The rings are the child ``<id>.ripple``, made the first time; the spot
        is found again at every frame, so the rings ride a zero that moves."""
        if not 1 <= int(rings) <= 8:
            raise VerbError(f"ripple draws 1 to 8 rings, not {rings}")
        params = {**self._spot(object_id, at, index, "ripple"), "rings": int(rings)}
        child = f"{object_id}.ripple"
        spec = self.motion["ripple"]
        d = spec["duration"]
        ops: list[dict[str, Any]] = []
        if child not in self.state:
            ops.append({"op": "add_object", "object": {"id": child, "type": "ripple", "params": {
                **params, "phase": 0.0}}})  # fmt: skip
        else:
            old = self.state[child].params
            now = {k: getattr(old, k) for k in params}
            if "point" in now:
                now["point"] = [float(v) for v in old.point]
            if now != params:
                raise VerbError(f"{object_id!r} already ripples with {now}; one spot per object "
                                "(ripple another object for another spot)")  # fmt: skip
        ops.append(self._instant(f"{child}.phase", 0.0, "ripple", 0.0))
        ops.append(self._span(f"{child}.phase", 1.0, "ripple", 0.0, d, spec["ease"]))
        return Motion(tuple(ops), d)

    def spotlight(self, *ids: str, until: str | None = None, pad: float = 0.25) -> Motion:
        """A pool of light on ``ids`` (around their bounds, ``pad`` more), the rest of the
        frame dimmed, fields included. The first call fades the light in, later ones glide
        it to the next target; ``until`` fades it out at that marker, and ``spotlight()``
        with no ids fades it out now. The light is the overlay object ``spotlight``."""
        spec = self.motion["spotlight"]
        d = spec["duration"]
        light = "spotlight"
        if not ids:
            if light not in self.state:
                raise VerbError("spotlight() turns the light off, but there is none yet: write "
                                "spotlight(target) first, or give it until=")  # fmt: skip
            return Motion((self._segment(f"{light}.opacity", 0.0, "spotlight", 0.0)[0],), d)
        corners = []
        for object_id in ids:
            obj = self._object(object_id)
            if obj.layer == "overlay":
                raise VerbError(f"{object_id!r} is on the overlay; the spotlight lights the world")
            corners.append(obj.impl.world_bounds(self.state).corners())
        pts = np.vstack(corners)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = [float(v) for v in (lo + hi) / 2]
        if self.state.camera.perspective:
            radius = float(np.linalg.norm(hi - lo)) / 2
        else:  # what the camera sees: the box in the plane
            radius = float(np.hypot(hi[0] - lo[0], hi[1] - lo[1])) / 2
        least = 0.12 * float(  # at least 0.12 of the frame height: a pool, even on a point
            self.state.camera.units_per_height(np.asarray(center))
        )
        radius = max(radius * (1.0 + pad), least)
        ops: list[dict[str, Any]] = []
        if light not in self.state:
            ops.append({"op": "add_object", "object": {
                "id": light, "type": "spotlight", "layer": "overlay",
                "params": {"center": center, "radius": radius, "opacity": 0.0}}})  # fmt: skip
        for name, value in (("center", center), ("radius", radius)):
            path = f"{light}.{name}"
            if self._starts_hidden(path):  # where it is when it first appears
                ops.append({"op": "set_param", "path": path, "value": value})
            ops.append(self._segment(path, value, "spotlight", 0.0)[0])
        ops.append(self._segment(f"{light}.opacity", 1.0, "spotlight", 0.0)[0])
        end = d
        if until is not None:
            off = self._until(until, "spotlight")
            if off < d:
                raise VerbError(f"spotlight: marker {until!r} comes before the light is on")
            ops.append(self._segment(f"{light}.opacity", 0.0, "spotlight", off)[0])
            end = off + d
        return Motion(tuple(ops), end)

    # ===== end of group F1 =====

    # ===== group F2a =====
    # (replace each stub below with the real verb; see docs/ai/verbs_*.md)

    ITERATE_STEPS_PER_DURATION = 10  # the theme's ``iterate`` time is for this many steps

    def iterate(self, orbit_id: str, n: int | None = None) -> Motion:
        """An orbit appears step by step: ``orbit_id`` (an ``orbit``) is shown and its
        ``progress`` runs 0 -> 1 at an even pace, a tenth of the theme's ``iterate`` time per
        step, so ``n`` steps (default: the orbit's own ``steps``) take ``n / 10`` of it. An
        orbit that ends early (it converged or escaped) is paced by the steps it has."""
        obj = self._of_type(orbit_id, "iterate", "orbit")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
            raise VerbError(f"iterate's n is a whole number of steps, at least 1, not {n!r}")
        path = f"{orbit_id}.progress"
        steps_path = f"{orbit_id}.steps"
        for each in (path, steps_path):
            self._base(each)  # refuses an expression
        ops: list[dict[str, Any]] = []
        if n is not None:
            ops.append(self._instant(steps_path, n, "iterate", 0.0))  # a track: the base stays
        count = obj.impl.orbit_length(self.state, n)
        spec = self.motion["iterate"]
        d = spec["duration"] * max(count, 1) / self.ITERATE_STEPS_PER_DURATION
        ops += self._show_at(orbit_id, "iterate", 0.0)
        if self._starts_hidden(path):
            ops.append({"op": "set_param", "path": path, "value": 0.0})
        else:
            ops.append(self._instant(path, 0.0, "iterate", 0.0))
        ops.append(self._span(path, 1.0, "iterate", 0.0, d, "linear"))
        return Motion(tuple(ops), d)

    def _spread(self, ids: Sequence[str], verb: str, to: float) -> Motion:
        def make(object_id: str, delay: float):
            if not self._has(object_id, "spread"):
                kind = self.state[object_id].type
                raise VerbError(f"{verb} needs a signal that splits into harmonics "
                                f"(partial_sum); {object_id!r} is a {kind}")  # fmt: skip
            self._base(f"{object_id}.spread")  # refuses an expression
            seg, done = self._segment(f"{object_id}.spread", to, verb, delay)
            return [seg], done

        return self._each(ids, verb, make)

    def decompose(self, *ids: str) -> Motion:
        """A Fourier sum peels apart: ``spread`` runs to 1 and each harmonic takes a row of
        its own under the sum, in its own colour."""
        return self._spread(ids, "decompose", 1.0)

    def recompose(self, *ids: str) -> Motion:
        """The inverse of ``decompose``: the harmonics return into the sum (``spread`` to 0)."""
        return self._spread(ids, "recompose", 0.0)

    # ===== end of group F2a =====

    # ===== group F2b =====

    def _child(self, child: str, kind: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """``add_object`` for a helper child the first time a verb needs it, else nothing."""
        if child in self.state:
            return []
        return [{"op": "add_object", "object": {"id": child, "type": kind, "params": params}}]

    def _written_expr(self, path: str) -> str | None:
        """The expression text the document has for ``path`` (None when it is not one)."""
        target, _, name = path.rpartition(".")
        spec = self.c.doc.object(target)
        value = spec.params.get(name) if spec else None
        return value.expr if isinstance(value, ExprSpec) else None

    def lift(self, plane_id: str, height: float = 0.6) -> Motion:
        """A flat domain coloring rises into the landscape of its function: ``|f|``
        (compressed to ``log(1 + |f|)``, poles clipped) becomes height, colored by phase,
        under a glowing net, while the flat colors fade. The surface is the child
        ``<id>.surface`` (a ``Landscape``), made the first time with the plane's function,
        position, size and constant ``a`` linked by expressions. ``height`` is world units
        per unit of ``log(1 + |f|)`` (the peak is three times it)."""
        plane = self._of_type(plane_id, "lift", "domain_coloring")
        if height <= 0:
            raise VerbError(f"lift needs a height above 0, not {height}")
        surf = f"{plane_id}.surface"
        spec = self.motion["lift"]
        d = spec["duration"]
        p = plane.params
        if surf in self.state and not self._has(surf, "height"):
            raise VerbError(f"{surf!r} exists but is not a landscape")
        ops = self._child(surf, "landscape", {
            "function": p.function.strip(), "to": p.to.strip(), "height": 0.0, "opacity": 0.0,
            "blend": {"expr": f"{plane_id}.blend"}, "a": {"expr": f"{plane_id}.a"},
            "pos": {"expr": f"{plane_id}.pos"}, "size": {"expr": f"{plane_id}.size"},
            "lines": 20})  # fmt: skip
        ops += [
            self._span(f"{surf}.opacity", 1.0, "lift", 0.0, 0.3 * d, "smooth"),
            self._span(f"{surf}.height", float(height), "lift", 0.0, d, spec["ease"]),
            self._span(f"{plane_id}.opacity", 0.0, "lift", 0.45 * d, d, "smooth"),
        ]
        return Motion(tuple(ops), d)

    def slice(self, surface_id: str, axis: str = "x", at: float | None = None,
              start: float | None = None) -> Motion:  # fmt: skip
        """A cutting plane sweeps through a surface (``ParametricSurface``, ``pde.Surface``
        or a ``Landscape``) along ``axis`` ("x", "y" or "z"), from ``start`` to ``at`` (world
        coordinates; by default the surface's whole extent), with the curve where it meets
        the surface drawn on it. The plane is the child ``<id>.cut`` (a ``CutPlane``), made
        the first time; a later slice on the same axis starts where the plane is."""
        obj = self._object(surface_id)
        if axis not in ("x", "y", "z"):
            raise VerbError(f"slice cuts along 'x', 'y' or 'z', not {axis!r}")
        if not hasattr(obj.impl, "grid_points"):
            raise VerbError("slice cuts a ParametricSurface, a pde Surface or a Landscape; "
                            f"{surface_id!r} is a {obj.type}")  # fmt: skip
        k = "xyz".index(axis)
        flat = np.asarray(obj.impl.grid_points(self.state)).reshape(-1, 3)
        lo, hi = float(flat[:, k].min()), float(flat[:, k].max())
        inset = 0.02 * (hi - lo)
        cut = f"{surface_id}.cut"
        old = self.state[cut] if cut in self.state else None  # noqa: SIM401
        if old is not None and old.params.axis != axis:
            raise VerbError(f"{cut!r} cuts along {old.params.axis!r}; slice the other way "
                            "with a second plane, not this one")  # fmt: skip
        end = hi - inset if at is None else float(at)
        begin = float(start) if start is not None else (
            lo + inset if old is None else float(old.params.at))  # fmt: skip
        spec = self.motion["slice"]
        d = spec["duration"]
        ops = self._child(cut, "cut_plane", {"surface": surface_id, "axis": axis, "at": begin,
                                             "opacity": 0.0})  # fmt: skip
        if old is not None:
            ops.append(self._instant(f"{cut}.at", begin, "slice", 0.0))
        ops += [
            self._span(f"{cut}.opacity", 1.0, "slice", 0.0, 0.2 * d, "smooth"),
            self._span(f"{cut}.at", end, "slice", 0.0, d, spec["ease"]),
        ]
        return Motion(tuple(ops), d)

    def project(self, v_id: str, onto: str, label: str | None = None) -> Motion:
        """Project an arrow (or a point) onto another arrow or a two-point line: a dashed
        perpendicular drops from the tip to the foot, then the projection arrow grows along
        the line from its start. Children: ``<id>.drop`` (a dashed line with a right-angle
        mark), ``<id>.proj`` (an arrow) and ``<id>.proj_grow`` (a number), all following the
        arrows through expressions; ``label`` writes a name beside the projection."""
        v = self._of_type(v_id, "project", "arrow", "dot")
        line = self._of_type(onto, "project", "arrow", "line")
        tip = f"{v_id}.end" if v.type == "arrow" else f"{v_id}.pos"
        if line.type == "arrow":
            o, e = f"{onto}.start", f"{onto}.end"
        else:
            n = len(line.params.points)
            if n != 2:
                raise VerbError(f"project onto a line needs a two-point line; {onto!r} has {n}")
            o, e = f"{onto}.points[0]", f"{onto}.points[1]"
        dirn = f"normalize({e} - {o})"
        foot = f"{o} + dot({tip} - {o}, {dirn}) * {dirn}"
        drop, proj, grow = f"{v_id}.drop", f"{v_id}.proj", f"{v_id}.proj_grow"
        if drop in self.state:
            for name, text in (("start", tip), ("end", foot)):
                if self._written_expr(f"{drop}.{name}") != text:
                    raise VerbError(
                        f"{v_id!r} already projects onto something else ({drop!r}); "
                        "project a copy of it to project twice"
                    )
        spec = self.motion["project"]
        d = spec["duration"]
        ops = self._child(drop, "dashed", {
            "start": {"expr": tip}, "end": {"expr": foot}, "progress": 0.0, "square": 0.12,
            "back": {"expr": f"normalize({o} - ({foot}))"}})  # fmt: skip
        ops += self._child(grow, "value", {"value": 0.0})
        ops += self._child(proj, "arrow", {
            "start": {"expr": o}, "end": {"expr": f"{o} + (({foot}) - {o}) * {grow}.value"},
            "opacity": 0.0})  # fmt: skip
        if drop in self.state:  # again: start over
            ops += [self._instant(f"{drop}.progress", 0.0, "project", 0.0),
                    self._instant(f"{grow}.value", 0.0, "project", 0.0),
                    self._instant(f"{proj}.opacity", 0.0, "project", 0.0)]  # fmt: skip
        ops += [
            self._span(f"{drop}.progress", 1.0, "project", 0.0, 0.5 * d, spec["ease"]),
            self._instant(f"{proj}.opacity", 1.0, "project", 0.5 * d),
            self._span(f"{grow}.value", 1.0, "project", 0.5 * d, d, spec["ease"]),
        ]
        if label:
            name = f"{proj}.label"
            ops += self._child(name, "text", {"text": label, "opacity": 0.0})
            if name not in self.state:
                ops.append({"op": "add_constraint", "rule": f"{name} near {proj}"})
            ops.append(self._span(f"{name}.opacity", 1.0, "project", 0.8 * d, d, "smooth"))
        return Motion(tuple(ops), d)

    # -- dissolve / assemble: an object and its cloud of points

    def _dust(self, object_id: str, count: int, spread: float | None, seed: int):
        """(child id, ``add_object`` ops or [], the spread) for the point cloud that stands
        for ``object_id``: points sampled from the geometry it records at the anchor."""
        obj = self._object(object_id)
        if not 12 <= count <= 20000:
            raise VerbError(f"count is between 12 and 20000, not {count}")
        dust = f"{object_id}.dust"
        b = obj.impl.world_bounds(self.state)
        lo, hi = np.asarray(b.lo, dtype=np.float64), np.asarray(b.hi, dtype=np.float64)
        extent = float(max(hi[0] - lo[0], hi[1] - lo[1], 0.05))
        far = max(0.35 * extent, 0.4) if spread is None else float(spread)
        if far <= 0:
            raise VerbError(f"spread must be above 0, not {spread}")
        pad = 0.2 * extent + 0.2
        w, h = (hi[0] - lo[0]) + 2 * (far + pad), (hi[1] - lo[1]) + 2 * (far + pad)
        if dust in self.state:
            old = self.state[dust].params
            if spread is None and old.scatter > 0:
                far = float(old.scatter)  # picks up where the points were left
            if old.size[0] < 0.9 * w or old.size[1] < 0.9 * h:  # life nudges bounds a hair
                raise VerbError(f"{dust!r} is too small for a spread of {far}; "
                                "dissolve with the spread it was made with")  # fmt: skip
            return dust, [], far
        center = (lo + hi) / 2
        pts = _sample_geometry(record(obj.impl, self.state), lo, hi, count,
                               rng(f"{object_id}/dust", seed))  # fmt: skip
        values = 0.5 + rng(f"{object_id}/dust-hue", seed).uniform(-0.08, 0.08, len(pts))
        params = {
            "source": "points", "seed": seed, "pos": [float(c) for c in center],
            "points": [[round(float(x - center[0]), 5), round(float(y - center[1]), 5)]
                       for x, y, _ in pts],
            "values": [round(float(v), 4) for v in values], "count": len(pts),
            "size": [float(w), float(h)], "point_size": 1.3, "exposure": 0.05,
            "opacity": 0.0, "scatter": 0.0,
        }  # fmt: skip
        return dust, self._child(dust, "point_cloud", params), far

    def dissolve(self, *ids: str, count: int = 900, spread: float | None = None,
                 seed: int = 0) -> Motion:  # fmt: skip
        """Shatter each object into ``count`` points of light that drift apart, each along
        its own random direction (``rng(id, seed)``) up to ``spread`` world units, while the
        object fades. The points are sampled from the geometry the object draws at the
        anchor and live in the child ``<id>.dust`` (a ``PointCloud``), which lies in the
        plane of the object (x and y of a 3D one)."""
        spec = self.motion["dissolve"]
        d = spec["duration"]

        def make(object_id: str, delay: float):
            dust, ops, far = self._dust(object_id, count, spread, seed)
            ops = list(ops)
            ops += [
                self._instant(f"{dust}.scatter", 0.0, "dissolve", delay),
                self._span(f"{dust}.opacity", 1.0, "dissolve", delay, delay + 0.25 * d, "smooth"),
                self._span(f"{object_id}.opacity", 0.0, "dissolve", delay, delay + 0.25 * d,
                           "smooth"),
                self._span(f"{dust}.scatter", far, "dissolve", delay, delay + d, spec["ease"]),
            ]  # fmt: skip
            return ops, delay + d

        return self._each(ids, "dissolve", make)

    def assemble(self, *ids: str, count: int = 900, spread: float | None = None,
                 seed: int = 0) -> Motion:  # fmt: skip
        """The reverse of ``dissolve``: scattered points of light (``<id>.dust``) fly back
        onto the object's geometry as it is at the anchor, and the object fades in as they
        arrive. Used on its own, the object is hidden until then."""
        spec = self.motion["assemble"]
        d = spec["duration"]

        def make(object_id: str, delay: float):
            dust, ops, far = self._dust(object_id, count, spread, seed)
            ops = list(ops)
            path = f"{object_id}.opacity"
            obj = self.state[object_id]
            target = obj.params.opacity / obj.params.dim if obj.params.dim > 0 else 0.0
            target = target if target > 0 else 1.0
            if self._starts_hidden(path):
                base = self._base(path)
                target = float(base) if base not in (None, 0, 0.0) else 1.0
                ops.append({"op": "set_param", "path": path, "value": 0.0})
            ops += [
                self._instant(f"{dust}.scatter", far, "assemble", delay),
                self._instant(f"{dust}.opacity", 1.0, "assemble", delay),
                self._span(f"{dust}.scatter", 0.0, "assemble", delay, delay + d, spec["ease"]),
                self._span(path, target, "assemble", delay + 0.75 * d, delay + d, "smooth"),
                self._span(f"{dust}.opacity", 0.0, "assemble", delay + 0.75 * d, delay + d,
                           "smooth"),
            ]  # fmt: skip
            return ops, delay + d

        return self._each(ids, "assemble", make)

    # ===== end of group F2b =====


def _sample_geometry(rec, lo: np.ndarray, hi: np.ndarray, count: int, gen) -> np.ndarray:
    """``count`` points (x, y, z) on what an object drew: along its recorded center lines
    in proportion to their length, else on its recorded vertices (with a little scatter when
    there are fewer than ``count``), else anywhere in its box (text, a kernel-drawn
    field)."""
    lines = [s for s in rec.strokes if len(s) >= 2]
    if lines:
        seg = [np.linalg.norm(np.diff(s, axis=0), axis=1) for s in lines]
        total = np.array([x.sum() for x in seg])
        if total.sum() > 1e-9:
            which = gen.choice(len(lines), size=count, p=total / total.sum())
            out = np.empty((count, 3))
            for k, s in enumerate(lines):
                mine = np.nonzero(which == k)[0]
                if not len(mine):
                    continue
                cum = np.concatenate([[0.0], np.cumsum(seg[k])])
                u = gen.uniform(0.0, cum[-1], len(mine))
                out[mine] = np.stack([np.interp(u, cum, s[:, c]) for c in range(3)], axis=1)
            return out
    verts = np.unique(np.round(rec.points, 6), axis=0) if len(rec.points) else np.zeros((0, 3))
    if len(verts) >= 12 and not set(rec.kinds) <= {"text", "field"}:
        pick = gen.choice(len(verts), size=count, replace=len(verts) < count)
        out = verts[pick]
        if len(verts) < count:
            out = out + gen.normal(0.0, 0.01 * max(float(np.ptp(verts[:, :2])), 0.1), out.shape)
        return out
    return lo + gen.uniform(0.0, 1.0, (count, 3)) * (hi - lo)


def _resample(points: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    u = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(u, cum, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def _matched_shapes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two polylines resampled to the same number of points; when both are closed, ``b`` is
    turned (and reversed if that is closer) so each point travels as little as possible."""
    n = max(len(a), len(b), 64)
    ra, rb = _resample(a, n), _resample(b, n)
    closed = np.allclose(a[0], a[-1]) and np.allclose(b[0], b[-1])
    if closed:
        best, best_cost = rb, np.inf
        ring = rb[:-1]
        for cand in (ring, ring[::-1]):
            for k in range(0, len(cand), max(1, len(cand) // 64)):
                c = np.roll(cand, -k, axis=0)
                c = np.vstack([c, c[:1]])
                cost = float(np.square(c - ra).sum())
                if cost < best_cost:
                    best, best_cost = c, cost
        rb = best
    return ra, rb
