"""A recording ``DrawContext`` and the contract members derived from it (ADR 0005).

A type that defines only ``Params`` and ``draw`` still has bounds, features and semantics:
``register`` gives it the ones here, which run its ``draw`` once against a ``Recorder`` and
read what it emitted. Pure NumPy, so ``core`` stays free of GPU libraries.

- ``world_bounds``: the box around every recorded vertex; for a ``field`` the quad's four
  corners (its ``pixels`` callback is never called); for text the box the text occupies.
- ``features``: ``center`` and the box's compass points (``top``, ``top_right``, ...: the
  box's sides as the camera sees them), plus ``start``, ``end`` and ``curve`` when the
  object drew exactly one stroke (through ``domains/strokes.py``).
- ``semantics``: ``{"type", "params"}``.

A recording is cached by (type, id, resolved params, resolved style, camera, scene folder,
and the path, modification time and size of each file the type names in ``Files``). What
else ``draw`` reads is remembered and the recording is made again when it changes: another
object's state (``state["other"]``), the time (``state.t``), entries of ``state.values``,
and any other part of the state. The arrays a recording keeps are shared, so they are
read-only.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from phasor.core import batches, files
from phasor.core import text as text_metrics
from phasor.core.cache import ByteCache
from phasor.core.objects import Bounds
from phasor.core.state import SceneState

CACHE_SIZE = 4096  # recordings kept, at most
CACHE_BYTES = 256 * 2**20  # ... and in bytes (a morph's in-between can be megabytes a frame)
_ABSENT = object()  # a ``state.values`` key the draw asked for that was not there
_ALL = object()  # the draw went through all of ``state.values``
# The compass points of the box, as (along the camera's right, along its up).
COMPASS = {
    "top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0),
    "top_left": (-1, 1), "top_right": (1, 1), "bottom_left": (-1, -1), "bottom_right": (1, -1),
}  # fmt: skip


class Recorder:
    """A ``DrawContext`` that keeps the geometry of what is drawn and draws nothing."""

    def __init__(self, right, up) -> None:
        self.right = np.asarray(right, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self.points: list[np.ndarray] = []  # (N, 3) blocks of world positions
        self.kinds: list[str] = []  # one per command: triangles | field | surface | text | batch
        self.strokes: list[np.ndarray] = []  # center lines noted by domains/strokes.py
        self.issues: list[str] = []  # repairs made to batch input (lint ``batch_input``)
        self.quality = "full"  # bounds, features and lint judge the full drawing

    def _add(self, kind: str, pts) -> None:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        self.kinds.append(kind)
        if len(pts):
            self.points.append(pts)

    def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None) -> None:
        self._add("triangles", positions)

    def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0) -> None:
        # Only the quad: ``pixels`` is the expensive part and says nothing about where it is.
        c, r, u = (np.asarray(v, dtype=np.float64) for v in (corner, right, up))
        self._add("field", [c, c + r, c + u, c + r + u])

    def surface(self, positions, indices, *, normals, colors, uv=None, material="solid",
                opacity=1.0, glow=1.0) -> None:  # fmt: skip
        self._add("surface", positions)

    def text(self, text, origin, height, *, color, anchor="center", glow=1.0,
             reveal=1.0) -> None:  # fmt: skip
        corner, wv, hv = text_metrics.quad(text, origin, height, anchor, self.right, self.up)
        self._add("text", [corner, corner + wv, corner + hv, corner + wv + hv])

    def note_path(self, points) -> None:
        """Called by ``domains/strokes.py`` with the center line of each stroke it draws."""
        kept = np.array(points, dtype=np.float64).reshape(-1, 3)  # a copy: it is made read-only
        kept.flags.writeable = False
        self.strokes.append(kept)

    # World-space batches: their points (sizes on screen are not counted in the box), each
    # line strip as a stroke, and what was repaired in the input (lint ``batch_input``).

    def _batch(self, made) -> None:
        b, issues = made
        self.issues.extend(issues)
        if b is None:
            return
        self._add("batch", b.positions if b.tips is None else np.r_[b.positions, b.tips])
        if b.kind == "lines":
            cuts = [] if b.strips is None else np.flatnonzero(np.diff(b.strips)) + 1
            for run in np.split(b.positions.astype(np.float64), cuts):
                if len(run) >= 2:
                    self.note_path(run)

    def points3d(self, centers, **kw) -> None:
        self._batch(batches.make_points(centers, **kw))

    def lines3d(self, points, **kw) -> None:
        self._batch(batches.make_lines(points, **kw))

    def arrows3d(self, tails, tips, **kw) -> None:
        self._batch(batches.make_arrows(tails, tips, **kw))


@dataclass(frozen=True)
class Recording:
    points: np.ndarray  # (N, 3): every recorded vertex and corner; may be empty
    kinds: tuple[str, ...]
    strokes: tuple[np.ndarray, ...]
    right: np.ndarray
    up: np.ndarray
    issues: tuple[str, ...] = ()  # repairs made to batch input (``core/batches.py``)


class _Values(Mapping):
    """``state.values`` as a recorded ``draw`` sees it: every entry it reads is noted."""

    def __init__(self, inner: Mapping[str, Any], log: dict[Any, Any]) -> None:
        self._inner, self._log = inner, log

    def __getitem__(self, key: str) -> Any:
        try:
            value = self._inner[key]
        except KeyError:
            self._log[key] = _ABSENT
            raise
        self._log[key] = value
        return value

    def __iter__(self) -> Iterator[str]:
        self._log[_ALL] = dict(self._inner)
        return iter(self._inner)

    def __len__(self) -> int:
        self._log[_ALL] = dict(self._inner)
        return len(self._inner)


# Parts of the state the cache key already covers (the object's camera, the folder).
_IN_KEY = frozenset({"camera_for", "base_dir"})


class _Watch:
    """The state a recorded ``draw`` sees: the real one, noting what it reads besides its own
    object: other objects, ``values`` entries and any other part of the state (``t``...)."""

    def __init__(self, state: SceneState, own: str) -> None:
        self._state, self._own = state, own
        self._objects: set[str] = set()
        self._values: dict[Any, Any] = {}
        self._parts: dict[str, Any] = {}

    def __getitem__(self, object_id: str):
        if object_id != self._own:
            self._objects.add(object_id)
        return self._state[object_id]

    def __contains__(self, object_id: object) -> bool:
        if object_id != self._own and isinstance(object_id, str):
            self._objects.add(object_id)
        return object_id in self._state

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._state, name)
        if name == "values":
            return _Values(value, self._values)
        if name not in _IN_KEY:
            self._parts[name] = _comparable(name, value)
        return value

    def reads(self, state: SceneState) -> tuple:
        objects = {i: _fingerprint(state, i) if i in state else None
                   for i in sorted(self._objects)}  # fmt: skip
        return objects, dict(self._values), dict(self._parts)


def _comparable(name: str, value: Any) -> Any:
    return _camera_key(value) if name in ("camera", "overlay") else value


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # arrays and the like: compare as unequal, so it is drawn again
        return False


def _still(state: SceneState, reads: tuple) -> bool:
    """Whether everything a recording read is the same in ``state``."""
    objects, values, parts = reads
    for i, fp in objects.items():
        if (i in state) != (fp is not None) or (fp is not None and _fingerprint(state, i) != fp):
            return False
    for key, value in values.items():
        now = dict(state.values) if key is _ALL else state.values.get(key, _ABSENT)
        if not (now is value or _same(now, value)):
            return False
    return all(_same(_comparable(n, getattr(state, n)), v) for n, v in parts.items())


# key -> (what draw read, the recording); the process's: the GUI and an MCP client share it
_CACHE = ByteCache(lambda: CACHE_BYTES, CACHE_SIZE)


def _camera_key(cam) -> tuple:
    return (cam.kind, cam.view_proj.tobytes(), cam.right.tobytes(), cam.up.tobytes(),
            float(cam.height), float(cam.tan_half_fov))  # fmt: skip


def _style_key(style) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in style.items()))


def _fingerprint(state: SceneState, object_id: str) -> tuple:
    o = state[object_id]
    return (o.params, _style_key(o.style), _camera_key(state.camera_for(o)))


def _files_key(impl, state: SceneState, me) -> tuple:
    """Each file the type names (``Files``) as it is now: a changed file is read again."""
    names = getattr(type(impl), "Files", ())
    paths = [getattr(me.params, n, None) for n in names]
    return tuple(files.stamp(state.base_dir, p) for p in paths if isinstance(p, str) and p)


def record(impl, state: SceneState) -> Recording:
    """What ``impl.draw`` emits for its object in ``state`` (cached)."""
    me = state[impl.id]
    cam = state.camera_for(me)
    key = (type(impl), impl.id, *_fingerprint(state, impl.id), state.base_dir,
           _files_key(impl, state, me))  # fmt: skip
    hit = _CACHE.peek(key)
    if hit is not None and _still(state, hit[0]):
        return hit[1]
    rec_ctx = Recorder(cam.right, cam.up)
    watch = _Watch(state, impl.id)
    impl.draw(rec_ctx, watch, "color")
    pts = np.vstack(rec_ctx.points) if rec_ctx.points else np.zeros((0, 3))
    pts.flags.writeable = False  # shared by everyone who asks: nobody may change it
    rec = Recording(pts, tuple(rec_ctx.kinds), tuple(rec_ctx.strokes), rec_ctx.right, rec_ctx.up,
                    tuple(dict.fromkeys(rec_ctx.issues)))  # fmt: skip
    _CACHE.put(key, (watch.reads(state), rec))
    return rec


def _anchor(state: SceneState, object_id: str) -> np.ndarray:
    """Where an object that drew nothing is: its ``pos`` when it has one, else the origin."""
    pos = getattr(state[object_id].params, "pos", None)
    return np.asarray(pos if pos is not None else (0.0, 0.0, 0.0), dtype=np.float64)


def bounds_of(rec: Recording, fallback: np.ndarray) -> Bounds:
    return Bounds.of_points(rec.points if len(rec.points) else fallback)


def box_features(bounds: Bounds, right, up) -> dict[str, np.ndarray]:
    """``center`` and the compass points of a box, its sides as seen along ``right``/``up``."""
    right, up = np.asarray(right, dtype=np.float64), np.asarray(up, dtype=np.float64)
    center = bounds.center
    half = (np.asarray(bounds.hi) - np.asarray(bounds.lo)) / 2
    along_r = float(np.abs(half) @ np.abs(right))  # the box's half extent across the view
    along_u = float(np.abs(half) @ np.abs(up))
    out = {"center": np.array(center, dtype=np.float64)}
    for name, (dx, dy) in COMPASS.items():
        out[name] = center + dx * along_r * right + dy * along_u * up
    for a in out.values():  # read-only like the recording's own points: nobody writes to one
        a.flags.writeable = False
    return out


def features_of(rec: Recording, bounds: Bounds) -> dict[str, np.ndarray]:
    out = box_features(bounds, rec.right, rec.up)
    if len(rec.strokes) == 1 and len(rec.strokes[0]) >= 2:
        curve = rec.strokes[0]
        out.update(start=curve[0], end=curve[-1], curve=curve)
    return out


def derived_world_bounds(self, state: SceneState) -> Bounds:
    """The box around everything ``draw`` emits (ADR 0005)."""
    return bounds_of(record(self, state), _anchor(state, self.id))


def derived_features(self, state: SceneState) -> dict[str, np.ndarray]:
    """``center``, the box's compass points, and ``start``/``end``/``curve`` for one stroke."""
    rec = record(self, state)
    return features_of(rec, bounds_of(rec, _anchor(state, self.id)))


def derived_semantics(self, state: SceneState) -> dict[str, Any]:
    """``{"type", "params"}``: what a type that says nothing more about itself is; plus
    ``batch_input`` (what was repaired in arrays given to a world-space batch) when any was."""
    me = state[self.id]
    out = {"type": me.type, "params": me.params.model_dump(mode="json")}
    issues = record(self, state).issues
    if issues:
        out["batch_input"] = list(issues)
    return out


def parts(state: SceneState, object_id: str):
    """The objects a composite built, in the order it built them."""
    return [o for o in state.objects if o.parent == object_id]


def composite_world_bounds(self, state: SceneState) -> Bounds:
    """The box around every part."""
    found = parts(state, self.id)
    if not found:
        return Bounds.of_points(_anchor(state, self.id))
    boxes = [o.impl.world_bounds(state) for o in found]
    return Bounds.of_points([p for b in boxes for p in (b.lo, b.hi)])


def composite_features(self, state: SceneState) -> dict[str, np.ndarray]:
    """``center`` and the compass points of the box around every part, and each part's own
    features as ``<part>.<feature>`` (``arrow.end``)."""
    cam = state.camera_for(state[self.id])
    out = box_features(composite_world_bounds(self, state), cam.right, cam.up)
    for o in parts(state, self.id):
        local = o.id[len(self.id) + 1 :]
        for name, pts in o.impl.features(state).items():
            out[f"{local}.{name}"] = pts
    return out


def composite_semantics(self, state: SceneState) -> dict[str, Any]:
    """``{"type", "params", "parts"}``: the parts by their own names."""
    return {**derived_semantics(self, state),
            "parts": [o.id[len(self.id) + 1 :] for o in parts(state, self.id)]}  # fmt: skip


def hybrid_world_bounds(self, state: SceneState) -> Bounds:
    """A composite that also draws (an axis draws its lines, its numbers are parts): the box
    around what it draws and every part."""
    boxes = [derived_world_bounds(self, state)]
    boxes += [o.impl.world_bounds(state) for o in parts(state, self.id)]
    return Bounds.of_points([p for b in boxes for p in (b.lo, b.hi)])


def hybrid_features(self, state: SceneState) -> dict[str, np.ndarray]:
    """As ``composite_features``, around what it draws and its parts."""
    cam = state.camera_for(state[self.id])
    out = box_features(hybrid_world_bounds(self, state), cam.right, cam.up)
    for o in parts(state, self.id):
        local = o.id[len(self.id) + 1 :]
        for name, pts in o.impl.features(state).items():
            out[f"{local}.{name}"] = pts
    return out


HYBRID = {
    "world_bounds": hybrid_world_bounds,
    "features": hybrid_features,
    "semantics": composite_semantics,
}

COMPOSITE = {
    "world_bounds": composite_world_bounds,
    "features": composite_features,
    "semantics": composite_semantics,
}

DERIVED = {
    "world_bounds": derived_world_bounds,
    "features": derived_features,
    "semantics": derived_semantics,
}
