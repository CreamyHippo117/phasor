"""The editor's logic (SPEC "GUI editor"), with no GUI toolkit in it.

``Editor`` holds what a person looking at a scene has in mind: the time under the playhead,
the selected object, the panels' contents. Every change goes through the shared
``Workspace`` and so through ``Engine.apply`` (contract 1): one undo step each, recorded as a
parameter or style change (the AI reading the scene sees intent, not pixels). The Qt window
(``window.py``) only draws this and forwards clicks, so all of it is tested without Qt.

The same ``Workspace`` is served to ``phasor mcp --attach`` over ``workspace.remote``; the
editor and the AI share one lock, and ``changed()`` tells the window when the AI has edited.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np

from phasor.engine import Engine
from phasor.workspace import ToolError, Workspace


def parse_value(text: str) -> Any:
    """What a person types into a field: JSON (``0.3``, ``[1, 2, 0]``, ``true``,
    ``{"expr": "t / 2"}``), else the text itself (``sin(x)``, ``D4``)."""
    try:
        return json.loads(text)
    except ValueError:
        return text


class Editor:
    def __init__(self, workspace: Workspace, lock: threading.RLock | None = None) -> None:
        self.ws = workspace
        self.lock = lock or threading.RLock()
        self.t = 0.0
        self.selected: str | None = None
        self.history: list[dict[str, Any]] = []  # what the person did here, newest last
        self._seen = self.version
        self._frame: tuple[tuple, np.ndarray] | None = None
        self._lint: tuple[int, list[dict[str, Any]]] | None = None

    # ------------------------------------------------------------ the scene

    @property
    def engine(self) -> Engine:
        return self.ws.engine

    @property
    def version(self) -> int:
        try:
            return self.engine.version
        except ToolError:
            return -1

    def changed(self) -> bool:
        """True once after anything (here or an attached AI) changed the scene."""
        v = self.version
        if v != self._seen:
            self._seen = v
            if self.selected and self.selected not in self._ids():
                self.selected = None
            return True
        return False

    def _ids(self) -> set[str]:
        return {o.id for o in self.engine.document.objects}

    def _edit(self, patch: Any, what: str) -> dict[str, Any]:
        with self.lock:
            r = self.ws.scene_edit(patch)
        if r["ok"]:
            self.history.append({"what": what, "version": self.version, "t": self.t})
        return r

    def undo(self) -> dict[str, Any]:
        with self.lock:
            return self.ws.undo()

    def redo(self) -> dict[str, Any]:
        with self.lock:
            return self.ws.redo()

    def save(self, path: str | None = None) -> dict[str, Any]:
        with self.lock:
            return self.ws.scene_save(path)

    # ------------------------------------------------------------ time

    def timeline(self) -> dict[str, Any]:
        with self.lock:
            return self.engine.timeline()

    def set_time(self, t: float) -> float:
        self.t = float(min(max(t, 0.0), max(self.engine.duration, 0.0)))
        return self.t

    def jump_to_marker(self, name: str) -> float:
        return self.set_time(self.timeline()["markers"][name])

    def play_step(self, dt: float) -> bool:
        """Advance the playhead; False when it reached the end."""
        end = self.engine.duration
        self.set_time(self.t + dt)
        return self.t < end

    # ------------------------------------------------------------ the viewport

    def frame(self, size: tuple[int, int]) -> np.ndarray:
        """The picture at the playhead, (h, w, 3) uint8, cached until time, size or the
        scene changes. Needs OpenGL."""
        key = (self.version, round(self.t, 6), tuple(size))
        if self._frame is None or self._frame[0] != key:
            with self.lock:
                img = np.ascontiguousarray(self.engine.still(self.t, size)[..., :3])
            self._frame = (key, img)
        return self._frame[1]

    def _to_output(self, u: float, v: float, size: tuple[int, int]) -> tuple[float, float]:
        meta = self.engine.document.meta
        return u * meta.width / size[0], v * meta.height / size[1]

    def select_at(self, u: float, v: float, size: tuple[int, int]) -> str | None:
        """Select the topmost object under a viewport point (pixels, origin top-left)."""
        x, y = self._to_output(u, v, size)
        with self.lock:
            ids = self.engine.hit_test(x, y, self.t)
        self.selected = ids[0] if ids else None
        return self.selected

    def select(self, object_id: str | None) -> None:
        if object_id is not None and object_id not in self._ids():
            raise ToolError(f"no object {object_id!r}")
        self.selected = object_id

    def screen_to_world(self, u: float, v: float, size: tuple[int, int], z: float = 0.0):
        """The world point under a viewport pixel, on the plane at height ``z`` (2D: z = 0)."""
        cam = self.engine.state_at(self.t).camera
        ndc = np.array([2 * u / size[0] - 1, 1 - 2 * v / size[1]])
        inv = np.linalg.inv(cam.view_proj)
        a, b = (inv @ np.array([*ndc, d, 1.0]) for d in (-1.0, 1.0))
        a, b = a[:3] / a[3], b[:3] / b[3]
        k = 0.0 if abs(b[2] - a[2]) < 1e-12 else (z - a[2]) / (b[2] - a[2])
        p = a + k * (b - a)
        return float(p[0]), float(p[1]), float(z)

    def drag_to(self, object_id: str, u: float, v: float, size: tuple[int, int]):
        """Move an object with a ``pos`` to the point under the mouse (one undo step, sent
        when the drag ends)."""
        spec = self.engine.document.object(object_id)
        if spec is None or "pos" not in spec.params:
            raise ToolError(f"{object_id!r} has no position to drag")
        z = float(spec.params["pos"][2]) if len(spec.params["pos"]) > 2 else 0.0
        x, y, _ = self.screen_to_world(u, v, size, z)
        pos = [round(x, 6), round(y, 6), z]
        return self._edit({"op": "set_param", "path": f"{object_id}.pos", "value": pos},
                          f"{object_id}.pos = {pos}")  # fmt: skip

    # ------------------------------------------------------------ the inspector

    def inspector(self, object_id: str | None = None) -> dict[str, Any] | None:
        oid = object_id or self.selected
        if oid is None:
            return None
        with self.lock:
            spec = self.engine.document.object(oid)
            if spec is None:
                return None
            st = self.engine.state_at(self.t)[oid]
            seen = self.engine.inspect(self.t, id=oid)["objects"][0]
        animated = [p for p in self.timeline()["lanes"] if p.startswith(oid + ".")]
        now = st.params.model_dump()
        params = {}
        for k, field in type(st.params).model_fields.items():
            if k in ("opacity", "dim", "z_index") and k not in spec.params:
                continue
            if f"{oid}.{k}" in animated:
                params[k] = now[k]  # a track: its value under the playhead
            elif k in spec.params:
                params[k] = spec.params[k]  # what the document says (an expression stays one)
            else:
                params[k] = field.get_default(call_default_factory=True)
        rules = [c.rule for c in self.engine.document.constraints if oid in c.rule]
        return {"id": oid, "type": spec.type, "params": params, "style": dict(st.roles),
                "semantic": seen.get("semantic", {}), "constraints": rules,
                "animated": animated}  # fmt: skip

    def set_param(self, object_id: str, name: str, text: str) -> dict[str, Any]:
        """Set a parameter from what was typed in its field."""
        value = parse_value(text)
        shown = text.strip()
        return self._edit({"op": "set_param", "path": f"{object_id}.{name}", "value": value},
                          f"{object_id}.{name} = {shown}")  # fmt: skip

    def set_style(self, object_id: str, key: str, role: str) -> dict[str, Any]:
        return self._edit({"op": "set_style", "id": object_id, "key": key, "role": role},
                          f"{object_id} {key}: {role}")  # fmt: skip

    # ------------------------------------------------------------ lint

    def lint_items(self, aesthetic: bool = False) -> list[dict[str, Any]]:
        """The lint panel: live issues (recomputed only after a change)."""
        v = self.version
        if self._lint is None or self._lint[0] != v or aesthetic:
            with self.lock:
                r = self.ws.lint(aesthetic=aesthetic)
            items = r["issues"] if isinstance(r, dict) else r
            if aesthetic:
                return items
            self._lint = (v, items)
        return self._lint[1]

    def jump(self, issue: dict[str, Any]) -> None:
        """Click on an issue: select its first object and move to its first time."""
        ids = [i for i in issue.get("ids", []) if i in self._ids()]
        self.selected = ids[0] if ids else self.selected
        times = issue.get("times") or []
        if times:
            self.set_time(float(times[0]))

    # ------------------------------------------------------------ looks and variants

    def looks(self) -> list[str]:
        from phasor.style.looks import available_looks

        return available_looks()

    def apply_look(self, name: str) -> dict[str, Any]:
        """Switch the scene to a look: its theme and life, and its defaults for the objects
        already there. One undo step."""
        from phasor.style.looks import load_look

        look = load_look(name)
        ops: list[dict[str, Any]] = [
            {"op": "set_meta", "field": "theme", "value": look.theme},
            {"op": "set_meta", "field": "life", "value": look.life},
            {"op": "set_meta", "field": "look", "value": name},
        ]
        for spec in self.engine.document.objects:
            d = look.defaults.get(spec.type, {})
            for k, v in d.get("params", {}).items():
                ops.append({"op": "set_param", "path": f"{spec.id}.{k}", "value": v})
            for k, role in d.get("style", {}).items():
                ops.append({"op": "set_style", "id": spec.id, "key": k, "role": role})
        return self._edit(ops, f"look: {name}")

    def variants(self, beat: str | float, n: int = 3) -> dict[str, Any]:
        with self.lock:
            return self.ws.variants(beat, n)

    def choose(self, winner: str, note: str = "") -> dict[str, Any]:
        """Pick a variant here: recorded as the person's choice (the AI's ``choose`` records
        its own as the AI's) and applied as one undo step."""
        from phasor.style import choices

        offered = getattr(self.ws, "_variants", {})
        if winner not in offered:
            raise ToolError(f"no variant {winner!r}; offered {sorted(offered)}")
        entry = choices.record(
            self.ws.root / choices.DEFAULT_PATH,
            scene=self.engine.document.meta.title,
            beat=getattr(self.ws, "_variant_beat", ""),
            options=list(offered),
            winner=winner,
            by="person",
            note=note,
        )
        ops = offered[winner]
        result = self._edit(ops, f"variant: {winner}") if ops else None
        return {"recorded": entry, "applied": bool(result and result["ok"]), "edit": result}
