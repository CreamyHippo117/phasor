"""Engine: owns the document, the evaluated-state cache and the inspector (SPEC contract 1).

The Python API, CLI, MCP server and GUI are thin clients of this class. Every mutation goes
through ``Engine.apply(patch) -> PatchResult``, which validates the whole result, records an
undo step and returns the lint delta. There is no other write path.

Importing this module loads no GPU library; a renderer is created on first use, and when
none can be created the inspector runs in analytic mode (tier C).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import phasor.domains  # noqa: F401  (registers the built-in object types)
from phasor.core import io
from phasor.core.issues import DocumentError, Issue
from phasor.core.patch import TWEAK_OPS, apply_ops, apply_overrides, parse_patch, record_tweaks
from phasor.core.schema import Document, Meta, Overrides
from phasor.core.state import SceneState
from phasor.inspect import inspector
from phasor.inspect.passes import PassRunner
from phasor.time.compile import Compiled, compile_document, evaluate

STATE_CACHE_SIZE = 256
UNDO_LIMIT = 200


@dataclass(frozen=True)
class PatchResult:
    ok: bool
    version: int
    errors: tuple[Issue, ...] = ()
    lint_added: tuple[Issue, ...] = ()
    lint_resolved: tuple[Issue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "errors": [i.to_dict() for i in self.errors],
            "lint_delta": {
                "added": [i.to_dict() for i in self.lint_added],
                "resolved": [i.to_dict() for i in self.lint_resolved],
            },
        }

    def raise_for_errors(self) -> PatchResult:
        if not self.ok:
            raise DocumentError(list(self.errors))
        return self


@dataclass(frozen=True)
class _Snapshot:
    """One point in history. Everything else the Engine holds is derived from this."""

    base: Document
    overrides: Overrides
    compiled: Compiled = field(compare=False)
    lint: tuple[Issue, ...] = field(compare=False)


def _build(base: Document, overrides: Overrides) -> _Snapshot:
    effective, orphans = apply_overrides(base, overrides)
    return _Snapshot(base, overrides, compile_document(effective), tuple(orphans))


class Engine:
    def __init__(
        self,
        document: Document | None = None,
        *,
        overrides: Overrides | None = None,
        title: str = "untitled",
        passes: PassRunner | str | None = "auto",
    ) -> None:
        base = document or Document(meta=Meta(title=title))
        self._snap = _build(base, overrides or Overrides())
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._version = 0
        self._states: OrderedDict[float, SceneState] = OrderedDict()
        self._snapshots: OrderedDict[float, dict[str, Any]] = OrderedDict()
        self._passes = passes

    # ------------------------------------------------------------ files

    @classmethod
    def open(cls, path: str | Path, **kwargs: Any) -> Engine:
        """Load ``name.phasor.json`` and, if present, ``name.overrides.json`` beside it."""
        return cls(
            io.load_document(path), overrides=io.load_overrides(io.overrides_path(path)), **kwargs
        )

    def save(self, path: str | Path) -> None:
        """Write the base document, and the overrides file when there are any tweaks.
        An emptied overrides file is removed so stale tweaks cannot come back."""
        io.save(path, self._snap.base)
        opath = io.overrides_path(path)
        if not self._snap.overrides.is_empty():
            io.save(opath, self._snap.overrides)
        elif opath.exists():
            opath.unlink()

    # ------------------------------------------------------------ read

    @property
    def document(self) -> Document:
        """The effective document: base with overrides applied."""
        return self._snap.compiled.doc

    @property
    def base(self) -> Document:
        return self._snap.base

    @property
    def overrides(self) -> Overrides:
        return self._snap.overrides

    @property
    def version(self) -> int:
        return self._version

    @property
    def scripted(self) -> bool:
        return self._snap.base.meta.source == "script"

    @property
    def duration(self) -> float:
        return self._snap.compiled.duration

    def lint(self) -> list[Issue]:
        """Current lint issues. Milestone 1 knows ``orphan_override``; visual rules arrive in M2."""
        return list(self._snap.lint)

    def state_at(self, t: float) -> SceneState:
        """Pure function of (document, t). Cached per document version."""
        t = float(t)
        if t in self._states:
            self._states.move_to_end(t)
            return self._states[t]
        state = evaluate(self._snap.compiled, t)
        self._states[t] = state
        if len(self._states) > STATE_CACHE_SIZE:
            self._states.popitem(last=False)
        return state

    # ------------------------------------------------------------ write

    def apply(self, patch: Any, *, origin: str = "tweak") -> PatchResult:
        """The only write path. Atomic: on any error nothing changes.

        ``origin="script"`` is used by the Python builder and edits the base document.
        ``origin="tweak"`` (GUI, AI, CLI) edits the base of a plain document, but on a scene
        built by a script it records into the overrides file instead, so re-running the
        script never destroys the tweak.
        """
        if origin not in ("tweak", "script"):
            raise ValueError(f"origin must be 'tweak' or 'script', got {origin!r}")
        old = self._snap
        try:
            parsed = parse_patch(patch)
            if origin == "tweak" and self.scripted:
                overrides = record_tweaks(old.overrides, parsed)
                # Validate the ops against the effective document for precise errors.
                apply_ops(old.compiled.doc, parsed)
                new = _build(old.base, overrides)
            else:
                base = apply_ops(old.base, parsed)
                overrides = old.overrides
                if origin == "tweak":
                    # A direct edit supersedes an older override on the same target.
                    overrides = _drop_superseded(overrides, parsed)
                new = _build(base, overrides)
        except DocumentError as e:
            return PatchResult(False, self._version, errors=tuple(e.issues))
        self._undo.append(old)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        return self._switch(new, old)

    def undo(self) -> PatchResult | None:
        if not self._undo:
            return None
        old = self._snap
        self._redo.append(old)
        return self._switch(self._undo.pop(), old)

    def redo(self) -> PatchResult | None:
        if not self._redo:
            return None
        old = self._snap
        self._undo.append(old)
        return self._switch(self._redo.pop(), old)

    def _switch(self, new: _Snapshot, old: _Snapshot) -> PatchResult:
        self._snap = new
        self._version += 1
        self._states.clear()
        self._snapshots.clear()
        return PatchResult(
            True,
            self._version,
            lint_added=tuple(i for i in new.lint if i not in old.lint),
            lint_resolved=tuple(i for i in old.lint if i not in new.lint),
        )

    # ------------------------------------------------------------ inspect

    def pass_runner(self) -> PassRunner | None:
        """The renderer behind the inspector, or None on a machine with no usable OpenGL."""
        if self._passes == "auto":
            try:
                from phasor.render.passes import default_renderer

                self._passes = default_renderer()
            except Exception:
                self._passes = None
        return self._passes  # type: ignore[return-value]

    def _snapshot(self, t: float) -> dict[str, Any]:
        t = float(t)
        if t not in self._snapshots:
            self._snapshots[t] = inspector.snapshot(self.state_at(t), self.pass_runner())
            if len(self._snapshots) > STATE_CACHE_SIZE:
                self._snapshots.popitem(last=False)
        return self._snapshots[t]

    def inspect(self, t: float, id: str | None = None, detail: str = "standard") -> Any:
        """Everything on screen at ``t`` (or one object), at the requested detail level."""
        self._require(id)
        return inspector.render_detail(self._snapshot(t), detail, only=id)

    def _entry(self, id: str, t: float) -> dict[str, Any]:
        self._require(id)
        return next(e for e in self._snapshot(t)["objects"] if e["id"] == id)

    def where(self, id: str, t: float) -> dict[str, Any]:
        """Region, bbox, depth and relations to objects tagged ``key``."""
        e = self._entry(id, t)
        keys = ("id", "region", "bbox", "center", "depth", "relations")
        return {"t": float(t), **{k: e[k] for k in keys}}

    def visible(self, id: str, t: float) -> dict[str, Any]:
        """Visible and on-screen fractions."""
        e = self._entry(id, t)
        return {
            "t": float(t),
            "id": id,
            "visible_fraction": e["visible_fraction"],
            "on_screen_fraction": e["on_screen_fraction"],
            "occlusion": self._snapshot(t)["occlusion"],
        }

    def occluders(self, id: str, t: float) -> dict[str, Any]:
        """Who hides it and by how much."""
        e = self._entry(id, t)
        return {"t": float(t), "id": id, "hidden_by": e["hidden_by"]}

    def hit_test(self, x: float, y: float, t: float) -> list[str]:
        """Object ids under an output-pixel point, topmost first."""
        return inspector.hit_test(self.state_at(t), self.pass_runner(), x, y)

    def still(self, t: float, size: tuple[int, int] | None = None):
        """Preview still: (h, w, 3) uint8 sRGB. Needs OpenGL (tier A or B)."""
        from phasor.render.output import encode_srgb8
        from phasor.render.passes import GLRenderer

        runner = self.pass_runner()
        if not isinstance(runner, GLRenderer):
            raise RuntimeError("no OpenGL renderer on this machine (run `phasor doctor`)")
        return encode_srgb8(runner.color(self.state_at(t), size))

    def _require(self, id: str | None) -> None:
        if id is not None and self.document.object(id) is None:
            known = sorted(o.id for o in self.document.objects)
            raise KeyError(f"no object {id!r}; objects are {known}")


def _drop_superseded(overrides: Overrides, patch: Any) -> Overrides:
    params, styles = dict(overrides.params), dict(overrides.styles)
    for op in patch.ops:
        if isinstance(op, TWEAK_OPS[0]):
            params.pop(op.path, None)
        elif isinstance(op, TWEAK_OPS[1]):
            styles.pop(f"{op.id}.{op.key}", None)
    return Overrides(params=params, styles=styles)
