"""What an AI client can do with phasor, as plain functions (no MCP library in here).

``Workspace`` holds the one open scene. Every method takes and returns JSON-friendly data,
so the same behavior is tested directly and served over MCP by ``server.py``. Nothing here
writes a document except through ``Engine.apply`` (SPEC contract 1).

Safety: paths must stay inside ``root``; ``.py`` scenes are executable code and are
refused unless ``allow_scripts`` is on; patches are validated before they apply and a
failed patch changes nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from phasor.core import io
from phasor.core.diff import diff_documents
from phasor.core.ids import file_stem
from phasor.core.issues import DocumentError, Issue
from phasor.core.objects import is_builtin, registered_types
from phasor.core.pycode import BlockError, read_tweaks, with_tweaks
from phasor.core.schema import Document, Meta
from phasor.engine import Engine
from phasor.scene import Scene, run_script
from phasor.style.looks import DEFAULT_THEME, available_looks, default_look, load_look
from phasor.style.theme import ROLE_GROUPS, available_themes
from phasor.time import verbs as _verbs
from phasor.workspace import docs
from phasor.workspace import explorer as _explorer
from phasor.workspace import jobs as _jobs

VERBS = (
    "reveal",
    "exit",
    "trace",
    "bloom_in",
    "dim_others",
    "undim",
    "focus_pull",
    "morph",
    "settle",
    "orbit",
    "orbit_reveal",
    "transform",
    "transform_matching",
    "become",
    "write",
    "travel",
    "flash",
    "circumscribe",
    "wiggle",
    "sweep",
)
MAX_CHECKPOINTS = 32


SHOWN_TIMES = 6


def _slim(issue: dict[str, Any]) -> dict[str, Any]:
    """An issue for a client: a rule that fires at 35 sample times lists the first few and
    the total (the full list is noise to read)."""
    times = issue.get("times", [])
    if len(times) > SHOWN_TIMES:
        issue = {**issue, "times": times[:SHOWN_TIMES], "times_total": len(times)}
    return issue


def _slim_result(r: dict[str, Any]) -> dict[str, Any]:
    d = r.get("lint_delta")
    if d:
        r = {**r, "lint_delta": {k: [_slim(i) for i in v] for k, v in d.items()}}
    return r


class ToolError(Exception):
    """A client mistake with a message the client can act on (not a crash)."""

    def __init__(self, message: str, issues: list[Issue] | None = None) -> None:
        super().__init__(message)
        self.issues = [i.to_dict() for i in issues or []]


class ScriptChanged(ToolError):
    """The open script changed on disk since it was last run or written here: nothing is
    written over it. ``disk`` and ``stamp``: the file as it is now (None when it is gone)."""

    def __init__(self, message: str, disk: str | None, stamp: tuple[int, int] | None) -> None:
        super().__init__(message)
        self.disk, self.stamp = disk, stamp


class Workspace:
    def __init__(
        self,
        root: str | Path | None = None,
        out_dir: str | Path | None = None,
        *,
        allow_scripts: bool = False,
    ) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.out_dir = Path(out_dir).resolve() if out_dir else self.root / "phasor_out"
        self.allow_scripts = allow_scripts
        self._engine: Engine | None = None
        self._variants: dict[str, list[dict[str, Any]]] = {}
        self._variant_beat = ""
        self.path: Path | None = None  # where the open scene was loaded from / last saved
        # Who changed the scene, in order: the editor's person ("you") and an attached AI ("ai").
        self.activity: list[dict[str, Any]] = []
        self.import_notes: list[str] = []  # what the last Manim import could not carry over
        # Explorer rows the person is typing in (row id -> the text so far): an edit of one
        # from anyone else is refused until they commit or leave it (``_explorer_typing``).
        self._drafts: dict[str, str] = {}
        # Set on the thread that runs an attached AI's call (``workspace.remote``): that call
        # never makes new code run, whatever the person allowed (``_no_new_code``).
        self._caller = threading.local()
        # A script scene (ADR 0004): the name its Scene has (the tweaks block calls
        # ``<var>.tweak``), the line that added each object, and the script as this
        # workspace last read or wrote it (text, stamp): a file that differs was changed
        # outside (the editor's file watching).
        self.script_var = "s"
        self.script_lines: dict[str, int] = {}
        self.script_disk: tuple[str, tuple[int, int]] | None = None
        self._jobs = _jobs.Jobs()  # render jobs (``render_start``)
        # Set while the editor runs the script on its worker thread: the scene it builds
        # replaces this one, so an edit now would be lost; edits are refused meanwhile.
        self.busy: str | None = None

    # ------------------------------------------------------------ plumbing

    @contextlib.contextmanager
    def _as_ai(self):
        """The calls made inside are an attached AI's (``RemoteServer`` wraps each one)."""
        before = self._by_ai
        self._caller.ai = True
        try:
            yield
        finally:
            self._caller.ai = before

    @property
    def _by_ai(self) -> bool:
        return getattr(self._caller, "ai", False)

    def _no_new_code(self, ops: list[Any]) -> None:
        """An attached AI may not make code run that the person did not ask for: no change to
        ``meta.modules`` and no new updater (``always``), even where the scene's own code may
        run (the person trusted the modules it named, not ones the AI names). Checked on the
        document the engine will edit, before anything is built, since building runs code."""
        if not self._by_ai:
            return
        from phasor.core.patch import SetMeta, apply_ops, parse_patch

        e = self.engine
        doc = e.document if e.scripted else e.base  # what ``Engine.apply`` applies the ops to
        try:
            parsed = parse_patch(ops)
        except (DocumentError, ValueError):
            return  # ``Engine.apply`` refuses the same patch the same way: nothing is built
        named = [op.value for op in parsed.ops if isinstance(op, SetMeta) and op.field == "modules"]
        new: list[str] = []
        if not named:
            try:
                after = apply_ops(doc, parsed, e.code_scope)
            except (DocumentError, ValueError):
                return  # refused the same way by ``Engine.apply``: nothing is built

            def refs(d: Document) -> set[str]:
                return {r for o in d.objects for r in o.always}

            new = sorted(refs(after) - refs(doc))
            if not new:
                return
        what = f"meta.modules {list(named[-1] or ())}" if named else f"updaters {new}"
        msg = (f"an attached AI may not make new code run ({what}): only the person at the "
               "editor may name a scene's modules or updaters")  # fmt: skip
        hint = "ask the person to make this change in the editor, or use phasor's own types"
        raise ToolError(msg, [Issue("scene_code", msg, fix_hint=hint)])

    def _open(self, path: str, *, allow_scripts: bool) -> dict[str, Any]:
        """``scene_open`` with the permission given here, not the workspace's: the editor
        passes ``True`` only for a script the person opened or ran themselves."""
        p = self._inside(path, must_exist=True)
        if p.suffix == ".py" and not allow_scripts:
            raise ToolError(
                "this server does not run scripts (a .py scene is code). Open the saved "
                ".phasor.json instead, or start the server with --allow-scripts"
            )
        from phasor.importers import import_manim, looks_like_manim
        from phasor.importers.manim import ManimImportError

        if p.suffix == ".py" and looks_like_manim(p):
            # A Manim script: run through the stand-in module; the result is a plain scene
            # (saved beside the script as name.phasor.json), free to edit.
            try:
                result = self._guard(lambda: import_manim(p))
            except ManimImportError as e:
                raise ToolError(f"{p.name} could not be imported: {e}") from e
            self._engine = Engine(result.document(), base_dir=p.parent)
            self.path = p.with_suffix(io.DOC_SUFFIX)
            self.import_notes = result.warnings
            return {**self.summary(), "imported_from": "manim", "notes": result.warnings}
        if p.suffix != ".py":
            self._engine = self._guard(lambda: Engine.open(p, allow_scripts=allow_scripts))
            self.path = p
            self.script_disk = None
            self.script_lines = {}
            return self.summary()
        run, text, stamp = self._load_script(p, allow_scripts)
        return self._adopt_script(p, run, text, stamp)

    def _load_script(self, p: Path, allow_scripts: bool) -> tuple[Any, str, tuple[int, int]]:
        """Run the script at ``p``: (its ``ScriptRun``, the text that ran, its stamp). The
        text is read *before* the script runs, so a change another editor makes while it
        runs is a change the editor has not seen (it reloads or asks). Touches nothing of
        this workspace: the editor runs it on a worker thread, then ``_adopt_script``."""
        stamp = file_stamp(p)  # the stamp first: a write between the two is seen as a change
        text = read_text(p)
        run = self._guard(lambda: run_script(p, allow_scripts=allow_scripts))
        return run, text, stamp

    def _adopt_script(self, p: Path, run: Any, text: str, stamp: tuple[int, int]) -> dict:
        """Make a finished script run the open scene (under the caller's lock)."""
        self._engine = run.engine
        self.path = p
        self.script_var, self.script_lines = run.var, run.lines
        self.script_disk = (text, stamp)
        out = self.summary()
        if run.overrides_file is not None:
            # ADR 0004: an old overrides file is folded into the script's tweaks block the
            # first time the script is opened here, then removed (its tweaks are in the code).
            # Not when the block's markers cannot be read for certain, or the file changed
            # meanwhile: then the overrides file stays, and it is read as before.
            try:
                self._write_script(with_tweaks(text, run.engine.overrides, self.script_var))
            except (BlockError, ToolError) as e:
                out["migration_skipped"] = str(e)
            else:
                run.overrides_file.unlink(missing_ok=True)
                out["migrated"] = str(run.overrides_file)
        return out

    def _check_disk(self) -> str:
        """The script on disk is the text this workspace last ran or wrote: else another
        editor changed it, and writing now would lose that change (``ScriptChanged``; the
        editor asks which version to keep). Returns the text."""
        assert self.path is not None and self.script_disk is not None
        try:
            disk, stamp = read_text(self.path), file_stamp(self.path)
        except FileNotFoundError:
            raise ScriptChanged(f"{self.path.name} is gone from the disk", None, None) from None
        if disk != self.script_disk[0]:
            raise ScriptChanged(
                f"{self.path.name} changed on disk since it was last run or saved here (another "
                "editor or an AI): nothing was written; open it again, or let the person choose "
                "which version to keep", disk, stamp)  # fmt: skip
        self.script_disk = (disk, stamp)
        return disk

    def _write_script(self, text: str) -> None:
        """Write the open script (its line endings as they are) and remember what was
        written, so the editor does not take this write for an outside edit. Refused when
        the file changed on disk since it was last run or written here (``_check_disk``)."""
        assert self.path is not None and self.path.suffix == ".py"
        seen = self._check_disk()
        write_text(self.path, text, expect=seen)
        self.script_disk = (text, file_stamp(self.path))

    def _save_tweaks(self) -> Path:
        """A script scene's save: its tweaks go into the block at the end of the script
        (ADR 0004); the code outside the block is not touched. Refused (nothing written)
        when the block's markers cannot be read for certain (``BlockError``: the person
        fixes them; the tweaks stay in the scene) or the file changed on disk since it was
        last run or written here. An old overrides file beside it is removed."""
        assert self.path is not None
        text = self._check_disk()
        try:
            new = with_tweaks(text, self.engine.overrides, self.script_var)
        except BlockError as e:
            raise ToolError(f"{self.path.name}: {e}; the tweaks were not written (they stay in "
                            "the scene): fix the markers, then save again") from e  # fmt: skip
        if new != text:
            self._write_script(new)
        self.path.with_name(self.path.stem + io.OVERRIDES_SUFFIX).unlink(missing_ok=True)
        return self.path

    def _trust_code(self) -> dict[str, Any]:
        """Let the open scene's own modules (``meta.modules``) run: the person at the editor
        asked (``Editor.trust_scene_code``). Not a tool: an attached AI cannot call it. The
        scene keeps its unsaved edits; its undo history starts again."""
        if self._by_ai:
            raise ToolError("only the person at the editor may trust a scene's code")
        e = self.engine
        self._engine = e.fork(allow_scripts=True)
        return {"modules": list(self._engine.document.meta.modules),
                "version": self._engine.version}  # fmt: skip

    def _writable(self) -> None:
        if self.busy:
            raise ToolError(f"{self.busy}: the scene is about to be replaced by what it builds; "
                            "try again in a moment")  # fmt: skip

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise ToolError("no scene is open: call scene_new or scene_open first")
        return self._engine

    def _inside(self, path: str | Path, *, must_exist: bool = False) -> Path:
        p = Path(path)
        p = (p if p.is_absolute() else self.root / p).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"{path} is outside the workspace folder {self.root}")
        if must_exist and not p.exists():
            raise ToolError(f"{path} does not exist (workspace folder: {self.root})")
        return p

    def _guard(self, fn):
        try:
            return fn()
        except DocumentError as e:
            raise ToolError(str(e), e.issues) from e
        except (KeyError, ValueError, OSError) as e:
            raise ToolError(str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)) from e
        except SystemExit as e:  # a scene script that calls exit(): the server goes on
            raise ToolError(f"the script called exit({e.code!r})") from None

    def summary(self) -> dict[str, Any]:
        e = self.engine
        d = e.document
        return {
            "title": d.meta.title,
            "theme": d.meta.theme,
            "size": [d.meta.width, d.meta.height],
            "fps": d.meta.fps,
            "duration": e.duration,
            "version": e.version,
            "path": str(self.path) if self.path else None,
            "source": "script" if e.scripted else "document",
            "objects": {o.id: o.type for o in d.objects},
            "markers": dict(sorted(d.timeline.markers.items(), key=lambda kv: kv[1])),
            "constraints": [f"{c.rule} ({c.priority})" for c in d.constraints],
            "narration": d.narration.script if d.narration else None,
            "checkpoints": e.checkpoints(),
        }

    # ------------------------------------------------------------ scenes

    def scene_new(
        self, title: str, theme: str | None = None, look: str | None = None,
        life: str | None = None, width: int = 1920, height: int = 1080, fps: float = 60.0,
        palette: str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Start an empty scene (replaces the open one; save first if it matters). It gets
        the default look, ``prism``, unless a ``look`` ("prism_everywhere", "calm", "glow",
        "chalk", or "none" for plain) or a ``theme`` is given; ``life`` ("off", "subtle",
        "lively") is the living motion; ``palette`` the colors run along strokes
        (``scene_palette``)."""
        self._writable()
        self._ai_not_while_typing("start a new scene")

        def go():
            name = default_look(look, theme)
            chosen = load_look(name) if name else None
            meta = Meta(
                title=title, theme=theme or (chosen.theme if chosen else DEFAULT_THEME),
                life=life or (chosen.life if chosen else "off"), look=name, width=width,
                height=height, fps=fps, palette=palette,
            )  # fmt: skip
            self._engine = Engine(Document(meta=meta), base_dir=self.root)
            self.path = None

        self._guard(go)
        return self.summary()

    def scene_palette(self, palette: str | None) -> dict[str, Any]:
        """Pick the colors run along strokes: ``spectrum`` (the prism rainbow), ``moonlight``
        (white with a cool and warm shimmer), ``gold``, ``glacier``, ``borealis``, ``rose``,
        or None for the theme's own. In the prism looks it recolors every glowing curve and
        the lines the look runs a hue along; fringes, vignette and glow stay. One undo step."""
        self._writable()
        from phasor.style.theme import PALETTES

        e = self.engine
        if palette is not None and palette not in PALETTES:
            raise ToolError(f"palette must be one of {PALETTES} (or none), got {palette!r}")
        ops: list[dict[str, Any]] = [{"op": "set_meta", "field": "palette", "value": palette}]
        look = load_look(e.document.meta.look) if e.document.meta.look else None
        targets = look.palette_targets() if look else {"line"}
        role = palette or (look.defaults.get("line", {}).get("style", {}).get("colormap")
                           if look else None) or "spectrum"  # fmt: skip
        for spec in e.document.objects:
            # Without a look, only lines that run a hue along themselves change.
            runs_hue = spec.type != "line" or look is not None or spec.params.get("hue_along", 0)
            keys = getattr(e.type_of(spec.type), "Style", {})
            if spec.type in targets and "colormap" in keys and runs_hue:
                ops.append({"op": "set_style", "id": spec.id, "key": "colormap", "role": role})
        r = e.apply(ops, origin="tweak")
        return {**_slim_result(r.to_dict()), "palette": palette}

    def scene_open(self, path: str) -> dict[str, Any]:
        """Open ``name.phasor.json`` (or a ``.py`` script when the server allows scripts; a
        script that imports ``manim`` is run through phasor's Manim stand-in and opens as a
        plain scene, with ``notes`` on what could not be carried over). The modules a
        document names run only where scripts may run; never for an attached AI."""
        self._writable()
        self._ai_not_while_typing("open another scene")
        return self._open(path, allow_scripts=self.allow_scripts and not self._by_ai)

    def scene_save(self, path: str | None = None) -> dict[str, Any]:
        """Save the document (``name.phasor.json``). A script scene keeps its tweaks in the
        script itself: saving writes them into the tweaks block at its end (``# --- phasor
        tweaks``) and leaves its other code as it is; ``path`` (a ``.phasor.json``) instead
        saves a copy of it as a document."""
        self._writable()
        e = self.engine
        if path is None and self.path is None:
            raise ToolError("this scene has no path yet: pass path, e.g. 'residues.phasor.json'")
        target = self._inside(path) if path else self.path
        assert target is not None
        if target.suffix == ".py":
            if target != self.path or not e.scripted:
                raise ToolError(
                    "a scene is saved as a script only to the script it runs "
                    "from; pass a .phasor.json path to save a copy as a document"
                )
            self._guard(self._save_tweaks)
            return {"saved": str(target), "version": e.version}
        if not target.name.endswith(io.DOC_SUFFIX):
            raise ToolError(f"scene files end in {io.DOC_SUFFIX}, got {target.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._guard(lambda: e.save(target))
        self.path = target
        return {"saved": str(target), "version": e.version}

    # ------------------------------------------------------------ editing

    def scene_edit(self, patch: Any) -> dict[str, Any]:
        """Apply a patch (one op, a list of ops, or ``{"ops": [...]}``). Atomic: if any op
        is invalid nothing changes. Returns the result and the lint issues it added or fixed.
        An ``add_object`` may name a ``preset`` (see ``object_types``); the scene's look fills
        in what is left unset, and explicit values always win. Attached to the editor, an
        edit that changes ``meta.modules`` or adds an updater is refused (``scene_code``),
        and so is one that would change an Explorer row the person is typing in
        (``row_locked``; also for ``undo``, ``redo`` and ``restore``)."""
        self._writable()
        ops = self._with_defaults(patch)
        self._no_new_code(ops)
        locked = self._ai_locked_ops(ops)
        if locked:
            return {"ok": False, "version": self.engine.version, "errors": locked,
                    "objects": len(self.engine.document.objects)}  # fmt: skip
        r = self.engine.apply(ops, origin="tweak")
        if r.ok:
            try:
                self._files_inside()
            except ToolError:
                self.engine.undo()  # the edit named a file outside the folder: take it back
                raise
        return {**_slim_result(r.to_dict()), "objects": len(self.engine.document.objects)}

    def _files_inside(self) -> None:
        """Every file the scene names (a picture's ``path``) is inside the workspace folder:
        the AI may show only what is in its folder."""
        from phasor.core import files

        e = self.engine
        for spec in e.document.objects:
            cls = e.type_of(spec.type)  # a scene's own type too: its files are checked
            for name in getattr(cls, "Files", ()):
                raw = spec.params.get(name)
                if isinstance(raw, str) and raw:
                    try:
                        self._inside(files.resolve(e.base_dir, raw))
                    except ValueError as err:  # outside the scene's folder
                        raise ToolError(str(err)) from None

    def _with_defaults(self, patch: Any) -> Any:
        """The patch as a list of ops, with ``preset`` and the scene's look resolved in every
        ``add_object`` (the document stores the resolved values). Other ops pass through."""
        ops = patch.get("ops") if isinstance(patch, dict) and "ops" in patch else patch
        listed = ops if isinstance(ops, list | tuple) else [ops]
        look_name = self.engine.document.meta.look
        look = load_look(look_name) if look_name else None
        out = []
        for op in listed:
            if not (isinstance(op, dict) and op.get("op") == "add_object"
                    and isinstance(op.get("object"), dict)):  # fmt: skip
                out.append(op)
                continue
            spec = dict(op["object"])
            cls = self.engine.type_of(str(spec.get("type")))
            preset = spec.pop("preset", None)
            params, style = dict(spec.get("params", {})), dict(spec.get("style", {}))
            if preset is not None:
                if cls is None or preset not in cls.Presets:
                    known = sorted(cls.Presets) if cls else []
                    raise ToolError(
                        f"no preset {preset!r} for {spec.get('type')!r}; presets: {known}"
                    )
                chosen = cls.Presets[preset]
                params = {**chosen.get("params", {}), **params}
                style = {**chosen.get("style", {}), **style}
            if look is not None:
                params, style = look.fill(str(spec.get("type")), params, style,
                                          self.engine.document.meta.palette)  # fmt: skip
            out.append({**op, "object": {**spec, "params": params, "style": style}})
        return out

    def undo(self) -> dict[str, Any]:
        """Revert the last edit (each patch, verb or narration is one step)."""
        self._writable()
        locked = self._ai_locked_doc(self.engine.undo_document())
        if locked:
            return {"undone": False, "ok": False, "version": self.engine.version,
                    "errors": locked}  # fmt: skip
        r = self.engine.undo()
        return {"undone": r is not None, **(_slim_result(r.to_dict()) if r else {})}

    def redo(self) -> dict[str, Any]:
        """Re-apply the edit that undo reverted."""
        self._writable()
        locked = self._ai_locked_doc(self.engine.redo_document())
        if locked:
            return {"redone": False, "ok": False, "version": self.engine.version,
                    "errors": locked}  # fmt: skip
        r = self.engine.redo()
        return {"redone": r is not None, **(_slim_result(r.to_dict()) if r else {})}

    def animate(
        self,
        verb: str,
        at: str | float,
        targets: list[str] | None = None,
        offset: float = 0.0,
        path: str | None = None,
        value: Any = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Animate with a choreography verb starting at a marker (or seconds). Timing comes
        from the theme. Returns when it starts and ends so the next step can follow it.
        ``orbit`` swings the 3D camera ``value`` degrees (default 360), until marker ``until``
        if given; ``morph`` too runs until ``until`` if given. ``transform``,
        ``transform_matching`` and ``become`` take ``targets=[from, to]``; ``sweep`` takes
        ``path`` and ``value``; ``circumscribe`` takes one target and ``value`` "circle" or
        "box"."""
        self._writable()
        if verb not in VERBS:
            raise ToolError(f"unknown verb {verb!r}; verbs are {list(VERBS)}")
        e = self.engine
        if e.scripted:
            raise ToolError(
                "this scene is built by a script, so structure and timing belong in the "
                "script; open its .phasor.json to animate here"
            )
        targets = targets or []
        if verb in ("morph", "settle"):
            if path is None or value is None:
                raise ToolError(f"{verb} needs path and value, e.g. path='dot.pos', value=[1, 2]")
            args: tuple = (path, value, until) if verb == "morph" else (path, value)
        elif verb == "orbit":
            args = (360.0 if value is None else float(value), until)
        elif verb == "focus_pull":
            if len(targets) != 1:
                raise ToolError("focus_pull needs exactly one target id")
            args = (targets[0],)
        elif verb in ("transform", "transform_matching", "become"):
            if len(targets) != 2:
                raise ToolError(f"{verb} needs two target ids: [from, to]")
            args = (targets[0], targets[1])
        elif verb == "circumscribe":
            if len(targets) != 1:
                raise ToolError("circumscribe needs exactly one target id")
            args = (targets[0], "circle" if value is None else str(value))
        elif verb == "sweep":
            if path is None or value is None:
                raise ToolError("sweep needs path and value, e.g. path='wave.terms', value=40")
            args = (path, float(value))
        else:
            args = tuple(targets)
        step = Scene.of(e).at(at, offset)

        def go():
            try:
                getattr(step, verb)(*args)
            except _verbs.VerbError as err:
                raise ToolError(str(err)) from err

        self._guard(go)
        base = e.document.timeline.markers[at] if isinstance(at, str) else float(at)
        start = base + offset
        return {"verb": verb, "start": start, "end": start + step._end, "version": e.version}

    def narrate(self, script: str = "", voice: str = "auto", pace: bool = True,
                audio: str | None = None) -> dict[str, Any]:  # fmt: skip
        """Set the narration. ``[[name]]`` puts a marker at the next word. Every animation
        anchored to a marker follows the voice. ``pace`` adds breathing room after busy
        sentences. ``audio``: a recording in the scene's folder (a WAV, or an mp4, m4a, mp3,
        ogg or flac) instead of a voice; ``script`` is then what it says (its sentences are
        found in the audio), or empty (its phrases become the blocks). A recording keeps its
        own timing, so ``pace`` is not applied to it; ``snap_to_narration`` moves your
        markers onto it."""
        self._writable()
        e = self.engine
        if e.scripted:
            raise ToolError("this scene is built by a script: change the narration there")
        scene = Scene.of(e)
        if audio is not None:
            self._inside(e.base_dir / audio)  # (and the scene's folder: core/files.py)
        from phasor.voice import VoiceError

        def go():
            if audio is not None:
                try:
                    scene.narrate(script, audio=audio)
                except VoiceError as err:
                    raise ToolError(str(err)) from err
                return
            scene.narrate(script, voice=voice)
            if pace:
                scene.pace()

        self._guard(go)
        return {
            "markers": dict(sorted(e.document.timeline.markers.items(), key=lambda kv: kv[1])),
            "duration": e.duration,
            "voice": e.document.narration.voice if e.document.narration else None,
        }

    def snap_to_narration(self, dry_run: bool = True, anchors: dict[str, int] | None = None,
                          allow_short: bool = False) -> dict[str, Any]:  # fmt: skip
        """Move the markers you placed by hand onto the narration's sentence starts (or where
        speech resumes after a pause), keeping their order and each scene long enough for its
        motions. ``dry_run`` (the default) only previews: ``moves`` lists marker, from, to,
        why and confidence. Call again with ``dry_run: false`` to apply (one undo step).
        ``anchors`` pins a marker to a narration block index (``{"poles": 2}``)."""
        if not dry_run:
            self._writable()
        e = self.engine
        if e.scripted and not dry_run:
            raise ToolError("this scene is built by a script: add s.snap_to_narration() there, "
                            "after s.narrate(...)")  # fmt: skip
        return self._guard(lambda: e.snap_to_narration(dry_run=dry_run, anchors=anchors,
                                                       allow_short=allow_short))  # fmt: skip

    # ------------------------------------------------------------ checkpoints

    def checkpoint(self, name: str) -> dict[str, Any]:
        """Remember the current document under ``name`` (to compare or return to later)."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", name) or name in ("current", "saved"):
            raise ToolError(
                "checkpoint names are 1-40 letters, digits, _ . -  (not 'current'/'saved')"
            )
        e = self.engine
        if name not in e.checkpoints() and len(e.checkpoints()) >= MAX_CHECKPOINTS:
            raise ToolError(f"at most {MAX_CHECKPOINTS} checkpoints; reuse a name")
        e.checkpoint(name)
        return {"checkpoint": name, "version": e.version, "all": e.checkpoints()}

    def restore(self, name: str) -> dict[str, Any]:
        """Go back to a checkpoint, as one step (undo brings you back again)."""
        self._writable()
        if self._by_ai and self._drafts and name in self.engine.checkpoints():
            locked = self._ai_locked_doc(self.engine.checkpoint_document(name))
            if locked:
                raise ToolError(locked[0]["message"])
        r = self._guard(lambda: self.engine.restore(name))
        return {**self.summary(), "restored": name, "version": r.version}

    def _version_of(self, ref: str) -> Document:
        if ref == "current":
            return self.engine.document
        if ref == "saved":
            path = self.path
            if path is not None and path.suffix == ".py" and path.exists():
                # A script's saved tweaks are its block (read, not run): laid over the
                # document its code makes now.
                from phasor.core.patch import apply_overrides

                saved = read_tweaks(read_text(path))
                return self._guard(lambda: apply_overrides(self.engine.base, saved)[0])
            if path is None or not path.exists():
                raise ToolError("nothing saved yet")
            # The saved document as it takes effect: base plus its overrides, like "current".
            allow = self.allow_scripts and not self._by_ai
            return self._guard(lambda: Engine.open(path, passes=None, allow_scripts=allow).document)
        if ref in self.engine.checkpoints():
            return self.engine.checkpoint_document(ref)
        raise ToolError(
            f"no version {ref!r}; use 'current', 'saved' or {self.engine.checkpoints()}"
        )

    def diff(self, a: str = "saved", b: str = "current") -> dict[str, Any]:
        """What changed from version ``a`` to version ``b`` ('current', 'saved' or a checkpoint)."""
        return self._guard(
            lambda: {"a": a, "b": b, **diff_documents(self._version_of(a), self._version_of(b))}
        )

    # ------------------------------------------------------------ seeing

    def inspect(self, t: float = 0.0, id: str | None = None, detail: str = "standard") -> Any:
        """What is on screen at ``t``: ``brief`` (one line each), ``standard`` or ``full``.
        ``full`` also gives each object's ``params`` as the document sets them (expressions
        stay expressions): how you see what a person changed in the editor."""
        if detail not in ("brief", "standard", "full"):
            raise ToolError("detail is brief, standard or full")
        out = self._guard(lambda: self.engine.inspect(t, id=id, detail=detail))
        if detail == "full" and isinstance(out, dict):
            doc = self.engine.document
            for o in out.get("objects", []):
                spec = doc.object(o["id"])
                if spec is not None:
                    o["params"] = _jsonable(dict(spec.params))
        return out

    def lint(
        self, fix: bool = False, strict: bool = False, step: float = 0.5, aesthetic: bool = False
    ) -> dict[str, Any]:
        """Find visual problems (overlap, off-screen, contrast, tiny text, clutter, ...).
        ``fix=True`` applies the fixes that are certain, then reports what is left.
        ``aesthetic=True`` also judges taste: focal point, color restraint, glow clipping,
        density, motion rhythm, margins, stroke weights."""
        e = self.engine
        applied: list[dict[str, Any]] = []
        if fix:
            self._writable()
            applied = e.autofix(aesthetic=aesthetic)["applied"]
        issues = e.lint(step, strict=strict, aesthetic=aesthetic)
        return {
            "clean": not issues,
            "errors": sum(i.severity == "error" for i in issues),
            "warnings": sum(i.severity == "warning" for i in issues),
            "issues": [_slim(i.to_dict()) for i in issues],
            "applied": applied,
        }

    def storyboard(
        self,
        times: list[float] | None = None,
        marks: bool = True,
        columns: int = 3,
        compare: bool = False,
    ) -> dict[str, Any]:  # fmt: skip
        """Render keyframes on one contact sheet (a PNG) with what the inspector sees in
        each. ``marks`` draws every object's box and id so you can tie picture to ids.
        ``compare`` puts each frame beside the three nearest frames in the project's
        ``gallery/`` (frames you love), with your notes on why they work."""
        from phasor.render.output import save_png

        e = self.engine
        gal = self.root / "gallery" if compare else None
        board = self._guard(lambda: e.storyboard(times, marks=marks, columns=columns, compare=gal))
        stem = f"{file_stem(e.document.meta.title)}.storyboard.v{e.version}"
        out = self.out_dir / f"{stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_png(out, board["image"])
        result = {"image": str(out), "frames": board["frames"], "version": e.version}
        if compare:
            cmp_path = self.out_dir / f"{stem}.compare.png"
            save_png(cmp_path, board["compare_image"])
            result.update(compare_image=str(cmp_path), compare=board["compare"],
                          gallery_size=board["gallery_size"])  # fmt: skip
        return result

    def events(self, t0: float = 0.0, t1: float | None = None, problems_only: bool = True,
               hz: float = 10.0) -> dict[str, Any]:  # fmt: skip
        """The event log: when things get hidden, leave or enter the frame, change region or
        gain an occluder, with the nearest marker and fix hints. ``problems_only`` keeps the
        ones a viewer would notice (a label or key object hidden or leaving the frame)."""
        from phasor.inspect.events import problems

        log = self._guard(lambda: self.engine.events(t0, t1, hz=hz))
        shown = problems(log) if problems_only else log
        return {"events": shown, "total": len(log), "problems": len(problems(log))}

    def when_hidden(self, id: str) -> dict[str, Any]:
        """Time spans where ``id`` is on screen but less than half of it is visible."""
        return {"id": id, "spans": self._guard(lambda: self.engine.when_hidden(id))}

    def free_space_near(self, id: str, t: float = 0.0) -> dict[str, Any]:
        """Empty label-sized places around ``id`` at ``t``, nearest first (boxes are 0-1,
        top-left origin): where a label or note could go."""
        return {"id": id, "t": t, "free": self._guard(lambda: self.engine.free_space_near(id, t))}

    # ------------------------------------------------------------ taste (Milestone 7)

    def _rubric(self) -> str:
        for folder in (self.root, docs.docs_root()):
            if folder is not None and (folder / "STYLE.md").exists():
                return (folder / "STYLE.md").read_text(encoding="utf-8")
        return ""

    def critique(self) -> dict[str, Any]:
        """The art-director's report: a taste score (0 = nothing to criticise), every
        aesthetic issue with the markers it happens at and its fixes (``fix`` is certain,
        ``suggested`` is judgement, ``snap`` puts an explicit color or size back on the
        nearest role and is only offered), and the rubric (STYLE.md) to judge the storyboard
        by. Look at the picture too: no rule sees everything."""
        from phasor.render import gallery

        report = self.engine.critique()
        report["rubric"] = self._rubric()
        report["gallery"] = len(gallery.entries(self.root / "gallery"))
        return report

    def polish(self, rounds: int = 3, snap: bool = False) -> dict[str, Any]:
        """Apply the critic's fixes (certain ones, then judgement suggestions), at most
        ``rounds`` (3) times; each round is undoable. Returns the score before and after and
        what is left for you to judge. Colors and sizes the scene typed (``off_palette``,
        ``off_scale``) snap to the nearest role only with ``snap=True``: ask the person."""
        self._writable()
        if not isinstance(snap, bool):  # a JSON "false" is truthy: never snap by accident
            raise ToolError(f"snap must be true or false, not {snap!r}")
        r = self.engine.polish(min(max(rounds, 1), 3), snap=snap)
        return {**r, "version": self.engine.version}

    def variants(self, beat: str | float, n: int = 3) -> dict[str, Any]:
        """Show alternative looks of one beat (a marker name or a time) side by side: as it
        is, calmer, bolder, each with its taste score. Nothing changes until you ``choose``."""
        from phasor.render.output import save_png

        e = self.engine
        r = self._guard(lambda: e.variants(beat, n))
        out = self.out_dir / f"{file_stem(e.document.meta.title)}.variants.v{e.version}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_png(out, r["image"])
        self._variants = {v["name"]: v["ops"] for v in r["variants"]}
        self._variant_beat = str(beat)
        return {"image": str(out), "beat": r["beat"], "t": r["t"],
                "variants": [{"name": v["name"], "score": v["score"], "changes": len(v["ops"])}
                             for v in r["variants"]]}  # fmt: skip

    def choose(self, winner: str, note: str = "", apply: bool = True) -> dict[str, Any]:
        """Pick the winner of the last ``variants`` call. The choice is appended to
        ``style/choices.jsonl`` (your taste as data) and, with ``apply``, its changes are
        made to the scene."""
        from phasor.style import choices

        if not getattr(self, "_variants", None):
            raise ToolError("call variants first: there is nothing to choose from")
        if winner not in self._variants:
            raise ToolError(
                f"no variant {winner!r}; the last call offered {sorted(self._variants)}"
            )
        e = self.engine
        entry = choices.record(self.root / choices.DEFAULT_PATH, scene=e.document.meta.title,
                               beat=self._variant_beat, options=list(self._variants),
                               winner=winner, by="ai", note=note)  # fmt: skip
        ops = self._variants[winner]
        result = self.scene_edit(ops) if apply and ops else None
        return {"recorded": entry, "applied": bool(result and result["ok"]),
                "edit": result}  # fmt: skip

    def hit_test(self, x: float, y: float, t: float = 0.0) -> dict[str, Any]:
        """Objects under an output-pixel point (origin top-left), topmost first."""
        return {"t": t, "x": x, "y": y, "ids": self.engine.hit_test(x, y, t)}

    def resolve(self, query: str, t: float = 0.0, limit: int = 5) -> dict[str, Any]:
        """Turn words like 'the pole on the left' or 'label of the loop' into object ids."""
        e = self.engine
        snap = e.inspect(t, detail="standard")["objects"]
        return {"query": query, "t": t, "matches": _resolve(e.document, snap, query)[:limit]}

    # ------------------------------------------------------------ outputs

    def review(
        self, step: float = 0.5, with_picture: bool = True, aesthetic: bool = False
    ) -> dict[str, Any]:
        """One call for the self-review loop: lint, what is on screen at every marker, and
        a storyboard. Fix what ``issues`` lists, then call again until ``clean``. Pass
        ``aesthetic=True`` for the beauty pass (taste rules on top of the correctness ones)."""
        e = self.engine
        lint = self.lint(step=step, strict=bool(e.document.narration), aesthetic=aesthetic)
        times = e.storyboard_times()
        marks = {t: n for n, t in e.document.timeline.markers.items()}
        frames = [
            {"t": t, "marker": marks.get(t), "objects": e.inspect(t, detail="brief")["objects"]}
            for t in times
        ]
        board: dict[str, Any] | None = None
        note = None
        if with_picture:
            try:
                board = self.storyboard(times)
            except (ToolError, RuntimeError) as err:
                note = f"no picture ({err}); judge from the inspector lines and lint"
        todo = [f"{i['code']} [{', '.join(i['ids'])}]: {i['message']}" for i in lint["issues"]]
        return {
            "clean": lint["clean"],
            "issues": lint["issues"],
            "todo": todo,
            "frames": frames,
            "storyboard": board["image"] if board else None,
            "note": note,
            "next": "done: report what you built and anything you could not resolve"
            if lint["clean"]
            else "fix the todo items (scene_edit, or lint with fix=true), then review again",
        }

    def render(
        self, preset: str = "preview", out: str | None = None, start: float | str | None = None,
        end: float | str | None = None, frame: float | str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Render a video or one PNG (``frame``). ``preset``: ``draft`` (quick look: long side
        640, 15 fps, no bloom, lens, fog or life, still frames repeated, not redrawn),
        ``preview`` (960x540@30), ``final`` or ``4k``. ``start``/``end``/``frame``: seconds
        or a marker name (a draft of one stretch: ``start="intro", end="proof"``). A draft
        is written to ``<scene>.draft.mp4`` by default."""
        from phasor.render.encode import EncodeError

        e = self.engine
        target = self._render_target(preset, out, frame)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            return e.render(target, preset=preset, start=start, end=end, frame=frame)
        except (RuntimeError, EncodeError, OSError, ValueError) as err:
            raise ToolError(str(err)) from err

    def _render_target(self, preset: str, out: str | None, frame: Any) -> Path:
        from phasor.render.job import PRESETS

        if not isinstance(preset, str) or preset not in PRESETS:
            raise ToolError(f"preset is one of {sorted(PRESETS)}")
        target = self._inside(out) if out else self.default_render_path(preset, frame)
        self._not_an_input(target)
        return target

    def default_render_path(self, preset: str, frame: Any = None) -> Path:
        """Where a render goes when no path is given: a draft beside the scene's file,
        ``<scene>.draft.mp4`` (the editor's Draft preview writes the same file; never the
        real render's name); other presets ``<scene>.mp4`` in the output folder. ``<scene>``
        is the file's name (a script's or a document's), else the title."""
        suffix = (".draft" if preset == "draft" else "") + (".png" if frame is not None
                                                           else ".mp4")  # fmt: skip
        if self.path is not None and preset == "draft":
            return self.path.parent / (self.path.name.split(".")[0] + suffix)
        return self.out_dir / (file_stem(self.engine.document.meta.title) + suffix)

    def _not_an_input(self, target: Path) -> None:
        """A render never writes over a file the scene reads (a clip, a picture, a sound,
        the narration's recording, the script itself)."""
        e = self.engine
        reads = set(e._files_read())
        n = e.document.narration
        if n is not None and n.voice.startswith("file:"):
            from phasor.core import files

            with contextlib.suppress(ValueError):
                reads.add(files.resolve(e.base_dir, n.voice[len("file:") :]))
        if self.path is not None:
            reads.add(self.path.resolve())
        if any(_same_target(target, r) for r in reads):
            raise ToolError(f"{target.name} is a file the scene reads: a render never writes "
                            "over it; choose another name")  # fmt: skip

    def render_start(
        self, preset: str = "draft", out: str | None = None, start: float | str | None = None,
        end: float | str | None = None, fps: float | None = None,
        frame: float | str | None = None, overwrite: bool | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Start a render that runs while you go on: returns at once with a ``job`` id; ask
        ``render_status`` how far it is, ``render_cancel`` to stop it (nothing is left
        behind). ``preset`` as for ``render`` (default ``draft``, written to
        ``<scene>.draft.mp4``, never over the real render). It renders the scene as it is
        now; later edits do not reach it. An existing file is replaced only with
        ``overwrite`` (a draft's own default file always is), and only when the render
        finished: until then the old file stays."""
        try:
            target = self._render_target(preset, out, frame)
        except OSError as err:
            raise ToolError(f"the render cannot write there ({err.strerror or err})") from err
        if overwrite is None and out is None and preset == "draft":
            overwrite = True  # a draft's own default file: the last draft, replaced
        return self._render_job(target, preset, start, end, fps, frame, overwrite)

    def _render_job(self, target: Path, preset: str, start: Any = None, end: Any = None,
                    fps: float | None = None, frame: Any = None,
                    overwrite: bool | None = None,
                    source: Engine | None = None) -> dict[str, Any]:  # fmt: skip
        """Start a render job to ``target``, a path the caller chose: the editor's person may
        render anywhere (not a tool, so an attached AI cannot reach it; ``render_start``
        keeps the AI in its folder). Same checks as ``render_start`` otherwise. A path the
        system refuses (a name too long, a folder that cannot be written) is a
        ``ToolError``, and no job is left behind. ``source``: a private fork to render
        instead of the open scene (the Explorer's export: a flow time range over a duration,
        never written into the document); the job owns it."""
        try:
            return self._start_job(Path(target), preset, start, end, fps, frame, overwrite,
                                   source)  # fmt: skip
        except OSError as err:
            raise ToolError(f"{Path(target).name[:80]}: the render cannot write there "
                            f"({err.strerror or err})") from err  # fmt: skip

    def _start_job(self, target: Path, preset: str, start: Any = None, end: Any = None,
                   fps: float | None = None, frame: Any = None,
                   overwrite: bool | None = None,
                   source: Engine | None = None) -> dict[str, Any]:  # fmt: skip
        from phasor.render.job import resolve_time
        from phasor.render.presets import check_fps

        e = self.engine if source is None else source
        target = Path(target).resolve()
        self._not_an_input(target)
        if not isinstance(overwrite, bool | type(None)):
            raise ToolError(f"overwrite must be true or false, not {overwrite!r}")
        replace = bool(overwrite)
        if target.exists() and not replace:
            raise ToolError(f"{target.name} exists: pass overwrite=true to replace it (it is "
                            "replaced only when the new render finished)")  # fmt: skip
        if target.suffix.lower() not in (".mp4", ".mov", ".png"):
            raise ToolError(f"{target.name}: a render job writes a .mp4, .mov or (with frame) "
                            "a .png")  # fmt: skip
        if (target.suffix.lower() == ".png") != (frame is not None):
            raise ToolError("a .png is one frame (pass frame); a video ends in .mp4 or .mov")
        try:  # a bad time or rate is said now, not by a job that fails later
            for what, v in (("start", start), ("end", end), ("frame", frame)):
                resolve_time(e, v, what)
            if fps is not None:
                check_fps(fps)
        except ValueError as err:
            raise ToolError(str(err)) from err
        target.parent.mkdir(parents=True, exist_ok=True)
        # the scene now; the job thread never reads ``e`` (a ``source`` fork is the job's own)
        snap = self._guard(e.snapshot) if source is None else source
        job = _jobs.RenderJob(snap, target, preset=preset, start=start, end=end, fps=fps,
                              frame=frame, overwrite=bool(overwrite))  # fmt: skip
        try:
            self._jobs.start(job)
        except ValueError as err:
            raise ToolError(str(err)) from err
        except OSError as err:  # a name too long, a folder that cannot be written
            raise ToolError(f"{target.name}: the render cannot write there ({err})") from err
        return job.status()

    def render_status(self, job: str | None = None) -> dict[str, Any]:
        """Where a render job is: ``state`` (running, cancelling, done, failed, cancelled),
        ``frame`` of ``frames``, ``progress`` 0-1, ``seconds``, the ``path`` it writes, and
        ``error`` or ``result`` at the end. Without ``job``: every job (``jobs``)."""
        if job is None:
            return {"jobs": [j.status() for j in self._jobs.all()]}
        try:
            return self._jobs.get(job).status()
        except KeyError as err:
            raise ToolError(str(err.args[0])) from err

    def render_cancel(self, job: str) -> dict[str, Any]:
        """Stop a render job between two frames; its partial file is removed and an earlier
        file of that name stays as it was. Returns its status (``cancelled``, or ``done``
        when it had already finished)."""
        try:
            j = self._jobs.get(job)
        except KeyError as err:
            raise ToolError(str(err.args[0])) from err
        if j.running:
            j.cancel(wait=jobs_wait())
            if j.running and (j._thread is None or not j._thread.is_alive()):
                j.state = "cancelled"  # a job whose thread is gone: cleared, never stuck
        return j.status()

    def export_midi(self, out: str | None = None, overwrite: bool = False) -> dict[str, Any]:
        """Write the scene's MIDI notes (every ``notes`` object, as heard, at its times in
        the scene) to a ``.mid`` file: format 1, one track per object. ``out``: a path in
        the workspace (default ``<scene>.mid`` next to the renders). Never over a file the
        scene reads; over another existing file only with ``overwrite``. Returns ``path``,
        ``tracks``, ``notes``, ``left_out`` and ``seconds``."""
        e = self.engine
        target = (
            self._inside(out) if out else self.out_dir / f"{file_stem(e.document.meta.title)}.mid"
        )
        if target.suffix.lower() not in (".mid", ".midi"):
            raise ToolError(f"{out}: a MIDI file ends in .mid")
        return self._guard(lambda: e.export_midi(target, overwrite=bool(overwrite)))

    # ------------------------------------------------------------ learning

    # ------------------------------------------------------------ the Explorer (Milestone 27)

    def _explorer(self, fn):
        try:
            return self._guard(fn)
        except _explorer.EditError as e:
            raise ToolError(str(e)) from e

    def explorer_state(self, t: float = 0.0) -> dict[str, Any]:
        """Call this first in the Explorer. The rows as typed (``id``, ``kind``, ``text``,
        ``error``, ``visible``, ``version``, ``locked`` while the person types in one), each
        system (equations, LaTeX, variables, divergence, window, ``dt``, flow time ``tau``,
        and its equilibria with eigenvalues, kinds and the window and grid searched), the
        variables and their slider ranges, the camera and lint. ``t`` is the scene time (the
        playhead is the editor's, not the document's)."""
        return self._explorer(lambda: _explorer.state(self.engine, t, self._drafts))

    def _row_refusals(self, rows: Any) -> list[dict[str, Any]]:
        """A row the person is typing in, or one changed since the ``version`` an edit names,
        is not edited (the whole call is refused)."""
        if not isinstance(rows, list | tuple):
            return []
        current = {e["id"]: e for e in _explorer.rows_of(self.engine.document)}
        objects: dict[str, list[str]] = {}
        for e in current.values():
            objects.setdefault(e["object"], []).append(e["id"])
        out = []
        for ed in rows:
            key = ed.get("id") if isinstance(ed, dict) else None
            if not isinstance(key, str):
                continue  # a new row, or an edit the planner refuses with its reason
            keys = [key] if key in current else objects.get(key, [])
            for k in keys:
                if k in self._drafts:
                    out.append({"code": "row_locked", "row": k, "version": current[k]["version"],
                                "message": f"the person is typing in row {k!r}; read "
                                "explorer_state again; retry once it is not locked"})  # fmt: skip
            want = ed.get("version")
            if want is not None and keys and current[keys[0]]["version"] != want:
                out.append({"code": "row_changed", "row": key,
                            "version": current[keys[0]]["version"],
                            "message": f"row {key!r} changed since version {want}; read it "
                            "again (explorer_state)"})  # fmt: skip
        return out

    def _locked_by(self, ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The rows the person is typing in that these ops would touch: the row's object, its
        parts (``system.field``), or any other row of the same object (an ODE line's
        system). Decided from the planned ops, so a new row that rewrites a locked one (a
        second ``rho = ...``, a third ``x' = ...``) is caught as well as an edit by id."""
        if not self._drafts:
            return []
        from phasor.core.ids import split_path

        current = {e["id"]: e for e in _explorer.rows_of(self.engine.document)}
        touched: set[str] = set()
        for op in ops:
            kind = op.get("op")
            path = None
            if kind in ("remove_object", "set_style"):
                touched.add(op["id"])
            elif kind == "add_object":
                touched.add(op["object"]["id"])
            elif kind == "set_param":
                path = op["path"]
            elif kind == "add_control":
                path = op["control"]["path"]
            elif kind == "remove_control":
                path = op["path"]
            if path is not None:
                touched.add(split_path(path)[0])
        out = []
        for key in sorted(self._drafts):
            row = current.get(key)
            if row is None:
                continue
            obj = row["object"]
            if obj in touched or any(t.startswith(f"{obj}.") for t in touched):
                out.append({"code": "row_locked", "row": key, "version": row["version"],
                            "message": f"the person is typing in row {key!r}, and this edit "
                            "would change it; read explorer_state again and retry once it "
                            "is not locked"})  # fmt: skip
        return out

    # An attached AI and a row the person is typing in (Milestone 29 audit B2): every tool
    # that changes the scene is refused when it would change or remove that row; the person's
    # own edits are not (they are the one typing).

    def _locked_message(self, key: str, version: str | None) -> dict[str, Any]:
        return {"code": "row_locked", "row": key, "version": version,
                "message": f"the person is typing in row {key!r}, and this would change it; "
                "read explorer_state again and retry once it is not locked"}  # fmt: skip

    def _ai_locked_ops(self, ops: Any) -> list[dict[str, Any]]:
        """``scene_edit`` by an AI: the ops' targets against the rows being typed in."""
        if not (self._by_ai and self._drafts):
            return []
        from phasor.core.patch import parse_patch

        try:
            dumped = [o.model_dump(mode="json") for o in parse_patch(ops).ops]
        except Exception:  # not a patch: ``apply`` refuses it with its reason
            return []
        return self._locked_by(dumped)

    def _ai_locked_doc(self, target: Document | None) -> list[dict[str, Any]]:
        """``undo``, ``redo``, ``restore`` by an AI: the document it would switch to, against
        the rows being typed in (a row that would vanish or change is refused)."""
        if not (self._by_ai and self._drafts) or target is None:
            return []
        now = self._row_states(self.engine.document)
        then = self._row_states(target)
        return [
            self._locked_message(k, now[k][0])
            for k in sorted(self._drafts)
            if k in now and then.get(k) != now[k]
        ]

    def _row_states(self, doc: Document) -> dict[str, tuple[str, str]]:
        """Each row being typed in: its version and the tracks that animate its object (a
        track on ``rho.value`` changes what the row means as much as its text does)."""
        rows = {e["id"]: e for e in _explorer.rows_of(doc) if e["id"] in self._drafts}
        out = {}
        for k, e in rows.items():
            tracks = {p: [seg.model_dump(mode="json") for seg in segs]
                      for p, segs in doc.timeline.tracks.items()
                      if p.startswith(f"{e['object']}.")}  # fmt: skip
            out[k] = (e["version"], json.dumps(tracks, sort_keys=True))
        return out

    def _ai_not_while_typing(self, what: str) -> None:
        if self._by_ai and self._drafts:
            raise ToolError(f"the person is typing in row {sorted(self._drafts)[0]!r}: {what} "
                            "once they are done (read explorer_state and retry)")  # fmt: skip

    def _locked_versions(self) -> tuple[Any, dict[str, Any]]:
        """The engine and the versions of the rows being typed in (``remote``'s check after
        a tool that has no check of its own)."""
        now = self._row_states(self.engine.document)
        return self._engine, {k: now.get(k) for k in self._drafts}

    def _undo_if_locked_changed(self, before: tuple[Any, dict[str, Any]],
                                version: int) -> list[dict[str, Any]]:  # fmt: skip
        """After an AI's tool that has no check of its own (``animate``, ``narrate``, a
        look, ``polish``, ...): if it changed a row the person is typing in, its steps are
        taken back (and not left to redo) and the call is refused."""
        engine, versions = before
        if engine is not self._engine or not versions:
            return []
        _, now = self._locked_versions()
        changed = [k for k, v in versions.items() if v is not None and now.get(k) != v]
        if not changed:
            return []
        e = self.engine
        for _ in range(max(0, e.version - version)):
            if e.undo() is None:
                break
        e.forget_redo()
        return [self._locked_message(k, versions[k][0]) for k in changed]

    def explorer_edit(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Add, retype, remove, show, hide, recolour and reorder Explorer rows and set
        variables, as one undoable edit. Each entry: ``{"text": "x' = sigma*(y - x)"}`` adds
        a row; ``{"id": "rho", "value": 24.74}`` (also ``min``, ``max``, ``step``), ``{"id":
        ..., "text": ...}``, ``{"id": ..., "remove": true}``, ``"visible"``, ``"color"`` (a
        role), ``"kind"`` (over the guess), ``"at"`` (its place), a system's ``window``,
        ``dt``, ``tau``, ``rate``; ``"version"`` refuses the edit if the row changed since;
        ``"base"`` on a new row is the id its object is named from (``"lorenz"``). A call
        that would change a row the person is typing in is refused whole (``row_locked``).
        Text is read like typing (``x^2``, ``2x``, Greek letters, ``x'``); a row that does not read
        is kept with its ``error``. Names no variable defines get a slider. Returns the rows
        changed (ids, kinds, errors) and the lint delta: read them before you answer."""
        e = self.engine
        if e.scripted:
            raise ToolError("Explorer rows edit a .phasor.json document; this scene is a script "
                            "(edit the script, or start one with scene_new)")  # fmt: skip
        refused = self._row_refusals(rows)
        if refused:
            return {"ok": False, "version": e.version, "errors": refused}
        cam = e.document.cameras[0].kind if e.document.cameras else "ortho"
        ops, report = self._explorer(lambda: _explorer.plan(e.document, rows, cam))
        locked = self._locked_by(ops)
        if locked:
            return {"ok": False, "version": e.version, "errors": locked}
        return self._explorer_commit_ops(ops, report)

    def _explorer_commit_ops(self, ops: list[dict[str, Any]], report: dict[str, Any]) -> dict:
        e = self.engine
        if not ops:
            return {"ok": True, "version": e.version, "errors": [], "changed": False, **report}
        r = self.scene_edit(ops)
        if r.get("ok"):
            now = {x["object"]: x for x in _explorer.rows_of(e.document)}
            for row in report["rows"]:
                if row["id"] in now:
                    row["version"] = now[row["id"]]["version"]
        return {**r, "changed": bool(r.get("ok")), **report}

    def explorer_preset(self, name: str, replace: bool = True) -> dict[str, Any]:
        """Load a classic system as rows (``lorenz``, ``rossler``, ``van_der_pol``,
        ``pendulum``, ``hopf``, ``saddle_node``, ``pitchfork``, ... : the list is in the
        error for an unknown name): its ODE rows, a slider per variable with the preset's
        range, the window and ``dt``. ``replace`` (default) removes the other rows in the same
        edit. Returns what ``explorer_edit`` returns."""
        e = self.engine
        if not isinstance(replace, bool):
            raise ToolError(f"replace is true or false, not {replace!r}")

        def go():
            edits = _explorer.preset_edits(name, _explorer.rows_of(e.document), replace)
            return self.explorer_edit(edits)

        return self._explorer(go)

    def find_equilibria(self, window: list[list[float]] | None = None, grid: int = 9,
                        variables: dict[str, float] | None = None, system: str | None = None,
                        t: float = 0.0) -> dict[str, Any]:  # fmt: skip
        """Every equilibrium Newton finds from a ``grid`` of starts per side of the ``window``
        (default: the system's), each with eigenvalues, kind, subspace dimensions, trace and
        determinant; ``variables`` ({"rho": 24}) try other values on a fork: nothing in the
        scene changes. Quote the window and grid: none found there is not none exist."""
        return self._explorer(lambda: _explorer.find_equilibria(
            self.engine, window, grid, variables, system, t))  # fmt: skip

    def flow_probe(self, x0: list[float] | list[list[float]], T: float = 10.0,
                   samples: int = 16, variables: dict[str, float] | None = None,
                   system: str | None = None, t: float = 0.0) -> dict[str, Any]:  # fmt: skip
        """Integrate one point ``x0`` (or up to 16) for flow time ``T`` (negative: backward)
        on a fork: ``samples`` positions (at most 64; only up to ``positions_until``, the
        smaller of ``chaotic_beyond`` and where ``dt`` and ``dt/2`` part, then statistics),
        escaped and when, bounding box, final speed, a period estimate, the largest Lyapunov
        exponent with its spread, and the dt check. At most 4000 steps (``clamped``)."""
        return self._explorer(lambda: _explorer.flow_probe(
            self.engine, x0, T, samples, variables, system, t))  # fmt: skip

    def sweep_parameter(self, name: str, lo: float, hi: float, n: int = 17,
                        measure: str = "equilibria", window: list[list[float]] | None = None,
                        grid: int = 9, variables: dict[str, float] | None = None,
                        system: str | None = None, t: float = 0.0) -> dict[str, Any]:  # fmt: skip
        """Move one variable from ``lo`` to ``hi`` (``n`` values, at most 64) on a fork: the
        equilibria at each (count, kinds, largest real part), and candidate bifurcations
        refined by bisection, each named by the local test that found it (saddle-node or
        pitchfork: the count changes; Hopf: a complex pair crosses; transcritical: a real
        eigenvalue crosses with the count unchanged), with its ``precision``. A count change
        at the window's edge is ``enters or leaves the window``; one beside non-isolated
        equilibria is ``not named``. Global bifurcations (homoclinic, crises) are not
        detected; say so."""
        return self._explorer(lambda: _explorer.sweep_parameter(
            self.engine, name, lo, hi, n, measure, window, grid, variables, system, t))  # fmt: skip

    # Not tools: the editor's own (an attached AI cannot call a name starting with _). The
    # Explorer panel (Milestone 29) must call these under the shared lock (``Editor.lock``),
    # like every other edit, so a lock and an AI call never interleave.

    def _explorer_typing(self, row: str, text: str) -> str:
        """The person is typing ``text`` in ``row``: it is locked to them until
        ``_explorer_commit`` or ``_explorer_leave``. Returns the row's version."""
        current = {e["id"]: e for e in _explorer.rows_of(self.engine.document)}
        if row not in current:
            raise ToolError(f"no row {row!r}")
        self._drafts[row] = text
        return current[row]["version"]

    def _explorer_leave(self, row: str) -> None:
        self._drafts.pop(row, None)

    def _explorer_commit(self, row: str, text: str) -> dict[str, Any]:
        """The person's row is done: unlock it and commit its text (one undoable edit)."""
        self._drafts.pop(row, None)
        return self.explorer_edit([{"id": row, "text": text}])

    def _explorer_preview(self, rows: list[dict[str, Any]]):
        """A fork of the scene with ``rows`` applied (what typing or a drag shows before the
        commit); the open scene and its history are untouched."""
        e = self.engine
        cam = e.document.cameras[0].kind if e.document.cameras else "ortho"
        ops, _ = self._explorer(lambda: _explorer.plan(e.document, rows, cam))
        fork = e.fork()
        if ops:
            r = fork.apply(self._with_defaults(ops), origin="tweak")
            if not r.ok:
                raise ToolError("; ".join(i.message for i in r.errors), list(r.errors))
        return fork

    def docs_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Search the API reference, patterns, pitfalls and examples."""
        return {"query": query, "results": docs.search(query, limit)}

    def object_types(self) -> dict[str, Any]:
        """What a patch may add: object types (use these exact names in ``add_object``) with their
        parameters and defaults and style keys (role group, default role), the role names each
        group accepts (style takes roles; color, text-size and stroke keys may instead hold an
        explicit value, checked by review), and the themes."""
        out: dict[str, Any] = {}
        scope = self._engine.code_scope if self._engine is not None else None
        for name, cls in sorted(registered_types(scope).items()):
            params = {}
            for k, f in cls.Params.model_fields.items():
                params[k] = (
                    "required" if f.is_required() else f.get_default(call_default_factory=True)
                )
            style = {
                k: {"group": g, "default": d} for k, (g, d) in getattr(cls, "Style", {}).items()
            }
            out[name] = {"params": _jsonable(params), "style": style,
                         "presets": sorted(getattr(cls, "Presets", {}))}  # fmt: skip
            if not is_builtin(cls):  # defined by a scene or a package, not shipped (ADR 0005)
                out[name]["scene_local"] = True
        return {
            "types": out,
            "roles": {g: list(r) for g, r in ROLE_GROUPS.items()},
            "themes": available_themes(),
            "looks": available_looks(),
        }


def read_text(path: Path) -> str:
    """A script as it is on disk: UTF-8, line endings kept (a tweak never changes them)."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _same_target(target: Path, read: Path) -> bool:
    """Whether writing ``target`` would write over ``read``: the same path (case folded
    where the system folds it), the same file (a hard link, another spelling), or, for a
    picture sequence ``shot_%04d.png``, any name the sequence could read in its folder."""
    from phasor.core import files

    t = target.resolve()
    if files.is_sequence(str(read)):
        name = read.name
        m = re.search(r"%(0(\d))?d", name)
        assert m is not None
        width = int(m.group(2)) if m.group(2) else 1
        pattern = (re.escape(name[: m.start()]) + rf"\d{{{width},}}"
                   + re.escape(name[m.end() :]))  # fmt: skip
        same_dir = os.path.normcase(str(t.parent)) == os.path.normcase(str(read.parent.resolve()))
        flags = re.IGNORECASE if os.path.normcase("A") == "a" else 0
        return same_dir and re.fullmatch(pattern, t.name, flags) is not None
    r = read.resolve()
    if os.path.normcase(str(t)) == os.path.normcase(str(r)):
        return True
    try:
        return t.exists() and r.exists() and os.path.samefile(t, r)
    except OSError:
        return False


def jobs_wait() -> float:
    """How long ``render_cancel`` waits for a job to stop (it is called under the editor's
    lock: short; the job finishes cleaning up on its own thread)."""
    return 0.5


def write_text(path: Path, text: str, *, expect: str | None = None) -> None:
    """Write ``text`` as it is (no line-ending translation), replacing the file in one step:
    a temporary file of its own (a unique name beside it, made exclusively), flushed to the
    disk, then renamed over the file. The file keeps its permissions; a read-only file is
    refused (``ToolError``), never replaced (on Windows replacing it fails; elsewhere it
    would silently lose its mode). ``expect``: the text the file must still hold, checked
    again just before the rename (``ScriptChanged`` otherwise; the window between the check
    and the rename is a few microseconds). A hard link to the file is not kept (the rename
    makes a new file)."""
    import stat
    import tempfile

    mode = None
    if path.exists():
        st = path.stat()
        mode = stat.S_IMODE(st.st_mode)
        if not os.access(path, os.W_OK) or not mode & stat.S_IWUSR:
            raise ToolError(f"{path.name} is read-only: nothing was written (make it writable, "
                            "or Save As another file)")  # fmt: skip
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".phasor-tmp",
                                    dir=path.parent)  # fmt: skip
    except OSError as e:
        raise ToolError(f"{path.name} could not be written: {e}") from e
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        if expect is not None:
            try:
                now = read_text(path)
            except FileNotFoundError:
                now = None
            if now != expect:
                raise ScriptChanged(f"{path.name} changed on disk just now (another editor): "
                                    "nothing was written", now, None)  # fmt: skip
        os.replace(tmp, path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise ToolError(f"{path.name} could not be written: {e}") from e
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def file_stamp(path: Path) -> tuple[int, int]:
    """(modification time in ns, size): what the editor compares to notice an outside edit
    (with the text itself, since Windows' clock is coarse)."""
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, tuple | list):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        return v.tolist()
    return v if isinstance(v, str | int | float | bool | type(None)) else str(v)


# ---------------------------------------------------------------- resolve

_STOP = {"the", "a", "an", "of", "at", "on", "in", "to", "for", "and", "is", "that", "this"}
_REGION_WORDS = {"left", "right", "top", "bottom", "upper", "lower", "center", "middle", "centre"}


def _tokens(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in _STOP]


def _resolve(doc: Document, snap: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Score objects against the words: id parts, tags, type, displayed text, screen region.
    "label of X" / "X's label" prefers the child ``X.label``."""
    words = _tokens(query)
    wants_label = "label" in words
    by_id = {e["id"]: e for e in snap}
    scored: list[tuple[float, str, list[str]]] = []
    for o in doc.objects:
        why: list[str] = []
        score = 0.0
        id_parts = set(re.split(r"[._]", o.id))
        text = ""
        tp = o.params.get("text")
        if isinstance(tp, str):
            text = tp
        text_tokens = set(_tokens(text))
        entry = by_id.get(o.id, {})
        region_tokens = set(re.split(r"[-\s]", str(entry.get("region", ""))))
        for w in words:
            if w in id_parts or w == o.id:
                score += 3
                why.append(f"id has '{w}'")
            elif w in {t.lower() for t in o.tags}:
                score += 2
                why.append(f"tagged '{w}'")
            elif w in text_tokens:
                score += 2.5
                why.append(f"text says '{w}'")
            elif w == o.type.split(".")[-1].lower():
                score += 1.5
                why.append(f"is a {w}")
            elif w in _REGION_WORDS and w in region_tokens:
                score += 1
                why.append(f"is on the {w}")
        if wants_label and o.id.endswith(".label"):
            score += 1.5 if score else 0
        elif wants_label and score and not o.id.endswith(".label"):
            score -= 0.5
        if score > 0:
            scored.append((score, o.id, why))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [{"id": i, "score": round(s, 2), "why": w, "region": by_id.get(i, {}).get("region")}
            for s, i, w in scored]  # fmt: skip
