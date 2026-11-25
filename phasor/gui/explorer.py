"""The Explorer panel's logic (Milestone 29, ``EXPLORER.md``): no Qt, tested directly.

The Explorer is a **view of the one engine**. Its rows are ``workspace.explorer.rows_of``
of the open document (no second store); every change the person makes is one patch through
the shared ``Workspace`` under the editor's lock (``Editor.lock``), with a readable name in
the history ("rho = 24.74", "added disk"; an attached AI's are "AI: ...").

Two kinds of state (``docs/decisions/explorer_advice_2026-11-24.md`` section 1):

- **document** (undoable, saved): rows, variables, sets, sections, the resting camera;
- **session** (never an op, never in the history): the playhead and transport (play,
  pause, speed, reverse, step), a row's text mid-typing, a slider mid-drag or playing, the
  camera while the mouse is down or the wheel turns, a handle mid-drag, the hover, the
  equilibrium card, the record marks.

The picture is drawn from a **view**: a fork of the engine (or of the typing preview) with
each flow system's ``rate`` set to 1, so the scene time is the playhead's flow time past the
document's resting ``tau``, plus the session's slider, camera and handle values. Building
the view is cheap (a fork and one ``apply`` on it, about a millisecond); the flow sets'
checkpoints live in a module-level cache keyed by their inputs, so scrubbing and orbiting
never integrate again.

Typing: each keystroke locks the row to the person (``Workspace._explorer_typing``: an AI's
``explorer_edit`` touching it is refused ``row_locked``), and the text is read
(``rows.normalize_row``) on a worker thread with a generation number, so a late answer for
older text is dropped. A clean row is previewed on a fork; it commits as one patch when the
debounce (``DEBOUNCE``, measured with the injected ``clock``) has passed since the last
keystroke, or when the person leaves the row (then even with an error: the row is kept with
its message, as ``explorer_edit`` keeps it).
"""

from __future__ import annotations

import contextlib
import functools
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core import io
from phasor.domains import flow_math as fm
from phasor.gui.controller import Editor
from phasor.workspace import ToolError
from phasor.workspace import explorer as X

DEBOUNCE = 0.2  # seconds after the last keystroke before a clean row commits
NAME_PAUSE = 1.0  # ... or this long when the commit would make a new variable (a slider)
SETTLE = 0.15  # full quality this long after the last input (Milestone 28's hint before)
WHEEL_IDLE = 0.4  # a turn of the wheel commits the zoom after this long without another
ZOOM_SECONDS = 0.5  # the camera's flight to an equilibrium
DRAG_PX = 4  # a press that moves less than this is a click
PICK_PX = 14  # how near (pixels) a click must be to an equilibrium or a handle
PLAY_SECONDS = 4.0  # a playing slider crosses its range in this many seconds
HOLD = 1.0  # scene seconds between two marks taken at the same flow time
TICK = 1 / 30  # one step of the transport, in seconds of play
SPEEDS = (1e-3, 1e3)  # the transport's speed range (times normal)
NEW = "+"  # the empty row at the bottom of the list
MAX_MARKS = 200  # marks in one recording
MAX_TEXT = 4000
EXPORT_SECONDS = (0.1, 3600.0)

# What a non-row object of the Explorer is called in the list.
OBJECT_LABELS = {
    "flow_set": "set", "flow_equilibrium_view": "equilibrium", "flow_manifold": "manifold",
    "flow_section": "section", "flow_twins": "twin tracers", "flow_divergence": "divergence map",
    "flow_bifurcation": "bifurcation diagram", "flow_limit_cycle": "limit cycle",
}  # fmt: skip
KINDS = tuple(k for k in X.flow.ROW_KINDS[:-1])  # the kinds a badge may choose (not "error")

CONVERT_WARNING = (
    "Convert to script writes this explorer document as a Python script (name.scene.py) "
    "and opens that instead. It is one-way for structure: from then on the Explorer can "
    "move sliders and parameters (written into the script's tweaks block) but cannot add, "
    "retype or remove rows; new rows are code. The .phasor.json stays as it is."
)
CONVERTED = ("this explorer is a script now: rows are code (edit the script); sliders and "
             "parameters still move")  # fmt: skip


def classify(text: str, kind: str | None = None) -> dict[str, Any]:
    """A typed row read as the planner will read it: ``kind``, ``error`` in plain words,
    the normalised ``text`` and ``latex``. Runs on the worker thread; never raises."""
    try:
        r = X.read(text, kind or None)
    except Exception as e:  # the reader never raises; a guard, so typing never crashes
        return {"kind": "error", "error": f"this row cannot be read ({type(e).__name__})",
                "text": text, "latex": "", "warnings": []}  # fmt: skip
    return {"kind": r.kind, "error": r.error, "text": r.text, "latex": r.latex,
            "warnings": list(r.warnings)}  # fmt: skip


def clean(answer: dict[str, Any] | None) -> bool:
    return answer is not None and answer["error"] is None and answer["kind"] != "error"


def _warm(fork, ops: list[dict[str, Any]], oid: str, t: float) -> None:
    """Apply ``ops`` to a private fork and compute what the new object shows (its semantics:
    the slow part, kept in the module-level memos), off the GUI thread."""
    if not fork.apply(ops).ok:
        return
    st = fork.state_at(t)
    st[oid].impl.semantics(st)


def equilibrium_names(points: list[dict[str, Any]], tol: float = 1e-6) -> list[str]:
    """Names for equilibria that do not change as a slider moves (positions in a list do):
    ``O`` for the origin; otherwise ``C`` and the sign of the first coordinate that is not
    zero (``C+``, ``C-``: Lorenz's two), with a number when several share one (``C+1``,
    ``C+2``, in order of their coordinates)."""
    base = []
    for p in points:
        at = [float(a) for a in p["at"]]
        first = next((a for a in at if abs(a) > tol), None)
        base.append("O" if first is None else ("C+" if first > 0 else "C-"))
    out = list(base)
    for name in set(base):
        same = sorted((k for k, b in enumerate(base) if b == name),
                      key=lambda k: tuple(points[k]["at"]))  # fmt: skip
        if len(same) > 1:
            for j, k in enumerate(same, 1):
                out[k] = f"{name}{j}"
    return out


class Immediate:
    """An executor that runs each job at once (tests; and the window when it wants an
    answer now)."""

    def submit(self, fn: Callable, *args: Any) -> Future:
        f: Future = Future()
        try:
            f.set_result(fn(*args))
        except BaseException as e:  # the future carries it, as a pool's would
            f.set_exception(e)
        return f

    def shutdown(self, wait: bool = True) -> None:
        pass


@dataclass
class Draft:
    """A row the person is typing in (session state)."""

    key: str  # the row's id in ``rows_of``, or ``NEW``
    text: str
    kind: str | None
    at: float  # the clock at the last keystroke
    gen: int
    answer: dict[str, Any] | None = None  # ``classify`` of ``text`` (None: not back yet)
    committed: bool = False  # ``text`` is in the document


def _num(v: float) -> str:
    return format(float(v), ".4g")


def _point_text(c) -> str:
    return "(" + ", ".join(_num(a) for a in c) + ")"


def _parse(v: Any, what: str) -> float:
    """A number typed into a box (text or a number): finite, or a plain-words refusal."""
    if isinstance(v, bool):
        raise ValueError(f"{what} must be a number")
    try:
        x = float(str(v).strip().replace(chr(0x2212), "-")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, not {str(v)[:40]!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{what} must be a finite number")
    return x


def _refused(message: str, **more: Any) -> dict[str, Any]:
    return {"ok": False, "errors": [{"code": "refused", "message": message}], **more}


@functools.lru_cache(maxsize=1)
def _example_text() -> str:
    from phasor.workspace.paths import examples_dir

    d = examples_dir()
    if d is None:
        return ""
    out = []
    for p in sorted(d.glob("*.py")):
        try:
            out.append(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(out)


# Types left out of the insert palette, and why: with their defaults alone they draw
# nothing (lint ``orphan``) or fall on each other; they are placed from code or by the AI.
PALETTE_HIDDEN = {
    **{t: "draws nothing without its content (a file, text, notes or a source)" for t in
       ("code", "notes", "path", "sound_rays", "svg", "table", "value", "wavefront3d")},
    "bar_chart": "its default bars overlap the Explorer's field; place it from code",
}  # fmt: skip


@functools.lru_cache(maxsize=1)
def palette() -> tuple[dict[str, Any], ...]:
    """The insert palette, generated from the object registry: every type shipped in phasor
    that has an example in ``examples/`` and can be placed with its defaults (no required
    parameter). Explorer rows themselves are typed, not inserted."""
    from phasor.core.objects import is_builtin, registered_types

    text = _example_text()
    out = []
    for name, cls in sorted(registered_types().items()):
        if not is_builtin(cls) or name in X.ROW_TYPES:
            continue
        if any(f.is_required() for f in cls.Params.model_fields.values()):
            continue
        cname = cls.__name__
        if not (re.search(rf"\b{re.escape(cname)}\(", text) or f'"{name}"' in text):
            continue
        try:
            cls.Params()
        except Exception:  # defaults that do not validate alone: not placeable by a click
            continue
        if name in PALETTE_HIDDEN:
            continue
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        out.append({"type": name, "label": cname, "about": doc,
                    "pos": "pos" in cls.Params.model_fields,
                    "flow": "system" in cls.Params.model_fields,
                    "near": "near" in cls.Params.model_fields})  # fmt: skip
    return tuple(out)


def marks_ops(doc, camera_id: str, marks: list[dict[str, Any]]) -> tuple[list[dict], list[str]]:
    """The ops that turn record marks into the scene (section 3.4): a marker per mark (at the
    playhead's scene time; one taken at the same time as the last waits ``HOLD`` seconds),
    and, for each camera parameter and variable that differs between marks, its value at the
    first mark and a ``smooth`` segment from each marker to the next (``MarkerRef`` times,
    so retiming a marker carries them). Systems that do not flow in the scene (``rate`` 0)
    get ``rate`` 1, so the video's flow time follows the playhead as it was recorded.
    Pure; returns (ops, marker names)."""
    ops: list[dict[str, Any]] = []
    taken = set(doc.timeline.markers)
    names: list[str] = []
    k = 1
    times: list[float] = []
    for m in marks:
        while f"mark_{k}" in taken:
            k += 1
        names.append(f"mark_{k}")
        taken.add(f"mark_{k}")
        t = max(0.0, float(m["t"]))
        if times and t < times[-1] + 0.05:
            t = times[-1] + HOLD
        times.append(t)
    for name, t in zip(names, times, strict=True):
        ops.append({"op": "set_marker", "name": name, "t": round(t, 6)})
    tracks = doc.timeline.tracks

    controls = {c.path for c in doc.controls}

    def animate(path: str, values: list[Any]) -> None:
        if all(_same(v, values[0]) for v in values):
            return
        if path in tracks:
            ops.append({"op": "clear_track", "path": path})
        if path in controls:  # a track drives it now: a slider cannot (lint orphan_control)
            ops.append({"op": "remove_control", "path": path})
        ops.append({"op": "set_param", "path": path, "value": values[0]})
        for j in range(1, len(values)):
            ops.append({"op": "add_segment", "path": path, "segment": {
                "t0": {"marker": names[j - 1]}, "t1": {"marker": names[j]},
                "to": values[j], "ease": "smooth"}})  # fmt: skip

    for key in sorted(set().union(*(m["camera"] for m in marks))):
        if all(key in m["camera"] for m in marks):
            animate(f"{camera_id}.{key}", [m["camera"][key] for m in marks])
    for vid in sorted(set().union(*(m["values"] for m in marks))):
        if all(vid in m["values"] for m in marks):
            animate(f"{vid}.value", [m["values"][vid] for m in marks])
    for o in doc.objects:
        rate = o.params.get("rate", 0.0)
        if o.type == "flow_system" and rate == 0 and f"{o.id}.rate" not in tracks:
            ops.append({"op": "set_param", "path": f"{o.id}.rate", "value": 1.0})
    return ops, names


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    return a == b


class Explorer:
    """The Explorer's state and actions over one ``Editor``. ``clock`` (seconds, monotonic)
    and the two executors are injected so tests never sleep: ``pool`` reads typed rows,
    ``analysis_pool`` runs the Analysis tab's estimates (a sweep never delays typing)."""

    def __init__(self, editor: Editor, *, clock: Callable[[], float] = time.monotonic,
                 pool: Any = None, analysis_pool: Any = None,
                 debounce: float = DEBOUNCE) -> None:  # fmt: skip
        self.ed = editor
        self.clock = clock
        self.pool = pool or ThreadPoolExecutor(1, thread_name_prefix="phasor-explorer")
        self.analysis_pool = analysis_pool or ThreadPoolExecutor(
            1, thread_name_prefix="phasor-analysis"
        )
        self.debounce = debounce
        self.message = ""  # the last thing said to the person (the status bar)
        self.dropped = 0  # answers dropped as stale (tests read it)
        self._reset_session()

    def _reset_session(self) -> None:
        self.playhead = 0.0  # flow time past the document's resting tau
        self.playing = False
        self.speed = 1.0
        self.reverse = False
        self.drafts: dict[str, Draft] = {}
        self.focus: str | None = None  # the row the person is typing in (its key now)
        self._gen = 0
        self._pending: list[tuple[int, Future]] = []
        self._typing_preview: tuple[int, int, Any] | None = None  # (gen, version, fork)
        self.slider_preview: dict[str, float] = {}  # variable id -> value mid-drag
        self.playing_sliders: dict[str, int] = {}  # variable id -> direction (+1 / -1)
        self.camera_session: dict[str, Any] = {}  # camera params while the mouse is down
        self._camera_pending = False  # the session camera has edits to commit
        self._wheel_at: float | None = None
        self._handle_ops: list[dict[str, Any]] = []  # a handle or a disk mid-drag
        self._press: dict[str, Any] | None = None
        self.mouse_down = False
        self.last_input = -1e9
        self.hover: tuple[float, float] | None = None
        self.card: dict[str, Any] | None = None
        self._anim: dict[str, Any] | None = None
        self.recording = False
        self.marks: list[dict[str, Any]] = []
        self._view: tuple[tuple, Any] | None = None
        self._frame: tuple[tuple, np.ndarray] | None = None
        self._shown_quality = "full"
        self._analysis: dict[str, dict[str, Any]] = {}
        self._placing: list[dict[str, Any]] = []  # flow objects computed before placing
        self.export_job: str | None = None
        self._times: list[tuple[float, float, str]] = []  # (done, seconds, quality) per frame
        self._latency: list[float] = []
        self._input_at: float | None = None  # wall clock of the input not drawn yet

    def close(self) -> None:
        for p in (self.pool, self.analysis_pool):
            p.shutdown(wait=False)

    # ------------------------------------------------------------ plumbing

    @property
    def ws(self):
        return self.ed.ws

    @property
    def read_only(self) -> bool:
        """After "Convert to script": rows are code; sliders still move (as tweaks)."""
        return self.ed.is_script

    def is_explorer_document(self) -> bool:
        """A ``.phasor.json`` with Explorer rows: the window opens it in the Explorer."""
        return not self.ed.is_script and any(X.is_row(o) for o in self._doc().objects)

    def _log(self, r: dict[str, Any], what: str) -> dict[str, Any]:
        """A change the person made through the Explorer: in the history and "who changed
        what" under a readable name (``Editor._edit`` does this for its own edits)."""
        if r.get("ok") and r.get("changed", True):
            entry = {"what": what, "version": self.ed.version, "t": self.ed.t}
            self.ed.history.append(entry)
            self.ws.activity.append({"who": "you", **entry})
            self.message = what
        elif not r.get("ok"):
            errs = r.get("errors") or [{"message": "refused"}]
            self.message = "Not applied: " + str(errs[0].get("message", ""))
        return r

    def _rows_edit(self, edits: list[dict[str, Any]], what: str | None = None) -> dict:
        """One ``explorer_edit`` (one undo step) under the shared lock."""
        if self.ed.running is not None:
            return self._log(_refused("the script is running: wait for it, or cancel it"), "")
        if self.read_only:
            return self._log(_refused(CONVERTED), "")
        with self.ed.lock:
            try:
                r = self.ws.explorer_edit(edits)
            except ToolError as e:
                r = _refused(str(e))
        return self._log(r, what or X.describe_edit(edits))

    def _edit(self, ops: list[dict[str, Any]], what: str) -> dict[str, Any]:
        """One patch (``Editor._edit``: the lock, the history and "who changed what")."""
        try:
            r = self.ed.edit(ops, what)
        except ToolError as e:
            r = _refused(str(e))
        if r.get("ok"):
            self.message = what
        else:
            self._log(r, what)
        return r

    def _doc(self):
        return self.ed.engine.document

    def _systems(self, doc=None) -> list[str]:
        return X.system_ids(doc or self._doc())

    # ------------------------------------------------------------ the list

    def rows(self) -> list[dict[str, Any]]:
        """The rows as the panel shows them: ``rows_of`` of the document (the one store),
        with what the person is typing over them (``typing``, its ``error`` and whether the
        answer is ``pending``), ``locked`` (to the person), the kind ``badge``, a variable's
        shown ``value`` (a slider mid-drag or playing) and ``animated`` (a kept track), then
        the Explorer's objects that are not typed rows (sets, sections, equilibrium views:
        ``is_object``)."""
        doc = self._doc()
        tracks = doc.timeline.tracks
        out = []
        for e in X.rows_of(doc):
            e = dict(e)
            with self.ed.lock:  # Editor.open_document clears it under the lock
                e["locked"] = e["id"] in self.ws._drafts
            e["badge"] = {"kind": e["kind"], "override": e.get("kind_override"),
                          "choices": KINDS}  # fmt: skip
            d = self.drafts.get(e["id"])
            if d is not None and not d.committed:
                e["typing"] = d.text
                e["pending"] = d.answer is None
                if d.answer is not None:
                    e["error"] = d.answer["error"]
                    e["badge"]["kind"] = d.answer["kind"]
            if e["kind"] == "variable":
                e["value"] = self.slider_preview.get(e["object"], e["value"])
                e["playing"] = e["object"] in self.playing_sliders
                e["animated"] = f"{e['object']}.value" in tracks
            out.append(e)
        n = len(out)
        for o in doc.objects:
            if o.type in OBJECT_LABELS and not X.is_row(o):
                p = o.params
                label = OBJECT_LABELS[o.type]
                if o.type == "flow_set":
                    label = str(p.get("shape", "disk"))
                    about = (f"{label}  c = {_point_text(p.get('center', (1.0, 0.0)))}  "
                             f"r = {_num(p.get('size', 0.5))}")  # fmt: skip
                elif o.type == "flow_equilibrium_view":
                    about = f"{label} {p.get('name') or ''} near {_point_text(p.get('near', ()))}"
                else:
                    about = f"{label} {o.id}"
                out.append({"id": o.id, "object": o.id, "kind": label, "source": about,
                            "text": about, "error": None, "row": n, "is_object": True,
                            "visible": float(p.get("opacity", 1.0) or 0) > 0, "locked": False,
                            "badge": {"kind": label, "override": None, "choices": ()}})  # fmt: skip
                n += 1
        if NEW in self.drafts and not self.drafts[NEW].committed:
            d = self.drafts[NEW]
            out.append({"id": NEW, "object": None, "kind": (d.answer or {}).get("kind", "new"),
                        "source": "", "typing": d.text, "pending": d.answer is None,
                        "error": (d.answer or {}).get("error"), "row": n, "locked": False,
                        "visible": True, "new": True,
                        "badge": {"kind": (d.answer or {}).get("kind", "new"), "override": None,
                                  "choices": KINDS}})  # fmt: skip
        return out

    def _row(self, key: str) -> dict[str, Any] | None:
        return next((e for e in X.rows_of(self._doc()) if e["id"] == key), None)

    # ------------------------------------------------------------ typing

    def type_row(self, key: str, text: str, kind: str | None = None) -> dict[str, Any]:
        """A keystroke in row ``key`` (``NEW`` for the empty row at the bottom): the row is
        locked to the person and its text read on the worker thread; nothing is committed
        here (``poll`` commits after the debounce, ``leave`` at once)."""
        if self.read_only:
            return _refused(CONVERTED)
        text = text if isinstance(text, str) else str(text)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1:
            return self.paste(key, lines)
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT]
            self.message = f"a row is at most {MAX_TEXT} characters; the rest was left out"
        if key != NEW:
            with self.ed.lock:
                try:
                    self.ws._explorer_typing(key, text)
                except ToolError as e:  # the row is gone (undone, or removed by the AI)
                    self.ws._explorer_leave(key)
                    gone = self.drafts.pop(key, None) or Draft(key, text, kind, 0.0, 0)
                    gone.text = text
                    self._orphan(gone)
                    return _refused(str(e), moved_to=NEW)
        self._gen += 1
        d = self.drafts.get(key)
        if d is None:
            d = self.drafts[key] = Draft(key, text, kind, self.clock(), self._gen)
        d.text, d.at, d.gen, d.answer, d.committed = text, self.clock(), self._gen, None, False
        if kind is not None:
            d.kind = kind or None
        self.focus = key
        self.last_input = self.clock()
        self._pending.append((d.gen, self.pool.submit(classify, text, d.kind)))
        return {"ok": True, "key": key, "gen": d.gen}

    def paste(self, key: str, lines: list[str]) -> dict[str, Any]:
        """Several lines pasted into a row: one row each, committed at once as one patch
        (the first replaces the row pasted into; the others follow it)."""
        lines = [ln[:MAX_TEXT] for ln in lines][: X.MAX_EDITS]
        row = self._row(key) if key != NEW else None
        if key != NEW and row is None:
            key = NEW
        if key == NEW:
            edits: list[dict[str, Any]] = [{"text": ln} for ln in lines]
        else:
            place = sum(1 for e in X.rows_of(self._doc()) if e["row"] <= row["row"])
            edits = [{"id": key, "text": lines[0]}]
            edits += [{"text": ln, "at": place + i} for i, ln in enumerate(lines[1:])]
        self.drafts.pop(key, None)
        with self.ed.lock:
            self.ws._explorer_leave(key)
        self.focus = None
        return self._rows_edit(edits, f"pasted {len(lines)} rows")

    def _orphan(self, d: Draft) -> None:
        """The row under a draft is gone (the person's undo, the AI's undo or edit): its
        lock is let go and the text typed is kept, never dropped: it moves to the new row
        at the bottom (or, when that holds text already, is added as a row of its own)."""
        with self.ed.lock:
            self.ws._explorer_leave(d.key)
        self.drafts.pop(d.key, None)
        if self.focus == d.key:
            self.focus = None
        if not d.text.strip():
            return
        self.message = ("the row you were typing in was removed (undo, or the AI); what you "
                        "typed is kept in a new row")  # fmt: skip
        if NEW in self.drafts and self.drafts[NEW].text.strip():
            self._rows_edit([{"text": d.text}], f"added {' '.join(d.text.split())[:60]}")
            return
        self.drafts.pop(NEW, None)
        self.type_row(NEW, d.text, d.kind)

    def _forget_stale(self) -> None:
        """Drafts whose row is gone become new rows; locks nobody types in are let go."""
        keys = {e["id"] for e in X.rows_of(self._doc())}
        for d in list(self.drafts.values()):
            if d.key != NEW and d.key not in keys and not d.committed:
                self._orphan(d)
        with self.ed.lock:
            for k in list(self.ws._drafts):
                if k not in self.drafts:
                    self.ws._explorer_leave(k)

    def _draft_of_gen(self, gen: int) -> Draft | None:
        return next((d for d in self.drafts.values() if d.gen == gen), None)

    def _preview(self, d: Draft) -> None:
        """A clean row shown before it commits: a fork with it applied (the document and
        its history are untouched)."""
        edit = {"text": d.text} if d.key == NEW else {"id": d.key, "text": d.text}
        if d.kind:
            edit["kind"] = d.kind
        with self.ed.lock:
            try:
                fork = self.ws._explorer_preview([edit])
            except ToolError:
                return  # the row cannot be planned (gone): the picture keeps the last good
            self._typing_preview = (d.gen, self.ed.version, fork)

    def _new_names(self, d: Draft) -> list[str]:
        """The variables committing this row would create (a dry run of the planner)."""
        edit = {"text": d.text} if d.key == NEW else {"id": d.key, "text": d.text}
        doc = self._doc()
        cam = doc.cameras[0].kind if doc.cameras else "ortho"
        try:
            return list(X.plan(doc, [edit], cam)[1]["new_variables"])
        except X.EditError:
            return []

    def _commit_draft(self, d: Draft) -> dict[str, Any]:
        """Commit a typed row: one patch named by its text. The row may get a new id (a new
        row, a kind change), so the draft follows it to its key in the list."""
        self._typing_preview = None
        what = " ".join(d.text.split())[:60] or "empty row"
        if d.key == NEW:
            edit: dict[str, Any] = {"text": d.text}
            if d.kind:
                edit["kind"] = d.kind
            r = self._rows_edit([edit], f"added {what}")
        else:
            if self.ed.running is not None:
                return _refused("the script is running")
            edit = {"id": d.key, "text": d.text}
            if d.kind:
                edit["kind"] = d.kind  # with the text: one patch, one undo step
            with self.ed.lock:
                self.ws._explorer_leave(d.key)  # the person's own commit: unlocked first
                try:
                    r = self.ws.explorer_edit([edit])
                except ToolError as e:
                    r = _refused(str(e))
            self._log(r, what)
        d.committed = True
        if not r.get("ok"):
            self.drafts.pop(d.key, None)
            if self.focus == d.key:
                self.focus = None
            if d.key != NEW and self._row(d.key) is None:
                self._orphan(d)  # the row went (undo, the AI): what was typed is kept
            return r
        new = self._rekey(d, r)
        if new and new != d.key:
            self.drafts.pop(d.key, None)
            d.key = new
            self.drafts[new] = d
            self.focus = new
        return r

    def _rekey(self, d: Draft, r: dict[str, Any]) -> str | None:
        """The key of the committed row now: the row with this text among the objects the
        commit reported (an ODE line is ``<system>:<coordinate>``)."""
        objects = {row["id"] for row in r.get("rows", [])}
        rows = X.rows_of(self._doc())
        for e in rows:
            if e["object"] in objects and e["source"] == d.text:
                return e["id"]
        for e in rows:
            if e["object"] in objects:
                return e["id"]
        return None if d.key == NEW else (d.key if any(e["id"] == d.key for e in rows) else None)

    def leave(self, key: str | None = None) -> dict[str, Any]:
        """The person left a row (another row, the viewport, Enter): commit what they typed
        if it differs from the document (with an error too: the row keeps its message), and
        unlock it."""
        key = self.focus if key is None else key
        if key is None:
            return {"ok": True}
        d = self.drafts.get(key)
        r: dict[str, Any] = {"ok": True}
        if d is not None and not d.committed:
            row = self._row(key) if key != NEW else None
            # an emptied row is not a deletion (delete is its own button): nothing to commit
            changed = bool(d.text.strip()) and (
                row is None or d.text != row["source"]
                or (d.kind is not None and d.kind != row.get("kind_override")))  # fmt: skip
            if changed:
                r = self._commit_draft(d)
        self.drafts.pop(key, None)
        if d is not None:
            self.drafts.pop(d.key, None)
        self.focus = None
        if key != NEW:
            with self.ed.lock:
                self.ws._explorer_leave(key)
                if d is not None:
                    self.ws._explorer_leave(d.key)
        self._typing_preview = None
        return r

    def settle(self) -> None:
        """Before anything that reads ``dirty`` or the document as a whole (Save, closing,
        R2/R3, convert): every row the person is typing in is committed (or let go when it
        is unchanged), and a wheel zoom waiting to rest is committed. So a row typed just
        before Ctrl+S or closing is saved, never lost."""
        for key in list(self.drafts):
            self.leave(key)
        self._forget_stale()
        if self._wheel_at is not None:
            self._wheel_at = None
            self._commit_camera("zoom")

    # ------------------------------------------------------------ the clock

    def poll(self) -> bool:
        """Called by the window's timer: take the reader's answers (dropping stale ones),
        preview a clean row, commit rows whose debounce has passed, commit a wheel zoom
        that came to rest, move the camera's flight, and say when the full-quality picture
        is due. True when the panel or the picture should be drawn again."""
        changed = False
        now = self.clock()
        self._forget_stale()
        waiting = []
        for gen, fut in self._pending:
            if not fut.done():
                waiting.append((gen, fut))
                continue
            d = self._draft_of_gen(gen)
            if d is None:  # newer text came since: this answer is for old text
                self.dropped += 1
                continue
            try:
                d.answer = fut.result()
            except Exception as e:  # the reader never raises; a guard all the same
                d.answer = {"kind": "error", "error": f"cannot read this row ({e})",
                            "text": d.text, "latex": "", "warnings": []}  # fmt: skip
            changed = True
            if clean(d.answer):
                self._preview(d)
        self._pending = waiting
        for d in list(self.drafts.values()):
            if not d.committed and clean(d.answer) and now - d.at >= self.debounce:
                if now - d.at < NAME_PAUSE and self._new_names(d):
                    continue  # a pause mid-name: no slider for a prefix ("sig") yet
                self._commit_draft(d)
                changed = True
        if self._anim is not None:
            self._fly(now)
            changed = True
        elif self.card is not None and self.card.get("_seen") != (
            self._view_key(),
            round(self.playhead, 6),
        ):
            self.refresh_card()
            changed = True
        if (self._wheel_at is not None and now - self._wheel_at >= WHEEL_IDLE
                and not self.mouse_down):  # fmt: skip
            self._wheel_at = None
            self._commit_camera("zoom")
            changed = True
        if self._shown_quality == "interactive" and not self._interacting():
            changed = True  # the full-quality picture is due
        for job in list(self._placing):
            if job["future"].done():
                self._finish_place(job)
                changed = True
        for name, job in self._analysis.items():
            if job.get("future") is not None and job["future"].done() and job["state"] == "running":
                self._finish_analysis(name, job)
                changed = True
        return changed

    def _interacting(self) -> bool:
        return (self.mouse_down or self._anim is not None
                or self.clock() - self.last_input < SETTLE)  # fmt: skip

    # ------------------------------------------------------------ row actions

    def _target(self, key: str) -> dict[str, Any] | None:
        return next((e for e in self.rows() if e["id"] == key), None)

    def set_kind(self, key: str, kind: str) -> dict[str, Any]:
        """The badge: a kind over the guess (``""``: the guess again)."""
        if kind and kind not in KINDS:
            return _refused(f"a row's kind is one of {', '.join(KINDS)}")
        if key in self.drafts and not self.drafts[key].committed:
            self.drafts[key].kind = kind or None
            return self.type_row(key, self.drafts[key].text)
        return self._rows_edit([{"id": key, "kind": kind}])

    def set_visible(self, key: str, visible: bool) -> dict[str, Any]:
        e = self._target(key)
        if e is not None and e.get("is_object"):
            return self._edit([{"op": "set_param", "path": f"{key}.opacity",
                                "value": 1.0 if visible else 0.0}],
                              f"{'showed' if visible else 'hid'} {key}")  # fmt: skip
        return self._rows_edit([{"id": key, "visible": bool(visible)}])

    def set_color(self, key: str, role: str) -> dict[str, Any]:
        """A row's colour, as a style role (``highlight``, ``danger``, ...)."""
        e = self._target(key)
        if e is not None and e.get("is_object"):
            from phasor.core.objects import get_type

            spec = self._doc().object(key)
            cls = get_type(spec.type) if spec else None
            keys = [
                k
                for k, (g, _) in getattr(cls, "Style", {}).items()
                if g in ("emphasis", "colormap")
            ]
            if not keys:
                return _refused(f"{key} has no colour")
            return self._edit([{"op": "set_style", "id": key, "key": keys[0], "role": role}],
                              f"{key} colour {role}")  # fmt: skip
        return self._rows_edit([{"id": key, "color": str(role)}])

    def delete(self, key: str) -> dict[str, Any]:
        if key == NEW:
            self.drafts.pop(NEW, None)
            return {"ok": True}
        e = self._target(key)
        if e is not None and e.get("is_object"):
            return self._edit([{"op": "remove_object", "id": key}], f"removed {key}")
        self.drafts.pop(key, None)
        with self.ed.lock:
            self.ws._explorer_leave(key)
        return self._rows_edit([{"id": key, "remove": True}])

    def move_row(self, key: str, at: int) -> dict[str, Any]:
        """Reorder: the row goes to place ``at`` (its ``row`` param; one patch)."""
        try:
            n = int(_parse(at, "the place"))
        except ValueError as e:
            return _refused(str(e))
        return self._rows_edit([{"id": key, "at": max(0, n)}])

    # ------------------------------------------------------------ sliders

    def sliders(self) -> list[dict[str, Any]]:
        """The variables' sliders (from the rows; values shown mid-drag or playing)."""
        return [{k: e.get(k) for k in ("id", "object", "name", "value", "min", "max", "step",
                                       "playing", "animated", "slider")}
                for e in self.rows() if e["kind"] == "variable"]  # fmt: skip

    def _variable(self, vid: str) -> dict[str, Any] | None:
        return next((e for e in self.rows() if e["kind"] == "variable"
                     and vid in (e["id"], e["object"])), None)  # fmt: skip

    def _snap(self, v: dict[str, Any], value: float) -> float:
        lo, hi = float(v["min"]), float(v["max"])
        x = min(max(float(value), lo), hi)
        step = v.get("step")
        if step:
            x = lo + round((x - lo) / step) * step
        return round(min(max(x, lo), hi), 10)

    def slider_drag(self, vid: str, value: Any) -> dict[str, Any]:
        """A slider moving: shown on the view (a fork), the document untouched."""
        v = self._variable(vid)
        if v is None:
            return _refused(f"no variable {vid!r}")
        if v["animated"]:
            return _refused(f"{v['name']} is animated by a track: clear it to move it by hand")
        try:
            x = self._snap(v, _parse(value, v["name"]))
        except ValueError as e:
            return _refused(str(e))
        self.slider_preview[v["object"]] = x
        self.last_input = self.clock()
        return {"ok": True, "value": x}

    def slider_release(self, vid: str) -> dict[str, Any]:
        """The slider let go: one patch named ``rho = 24.74`` (on a script, a tweak)."""
        v = self._variable(vid)
        oid = v["object"] if v else vid
        if oid not in self.slider_preview:
            return {"ok": True, "changed": False}
        x = self.slider_preview.pop(oid)
        self.last_input = self.clock()
        if v is None:
            return _refused(f"no variable {vid!r}")
        if self.read_only:
            key = f"{oid}.value"
            try:
                self.ed.set_control(key, x)
            except ToolError as e:
                return _refused(str(e))
            return self.ed.apply_controls()
        return self._rows_edit([{"id": oid, "value": x}], f"{v['name']} = {_num(x)}")

    def set_value(self, vid: str, text: Any) -> dict[str, Any]:
        """A number typed into a slider's box: committed at once."""
        r = self.slider_drag(vid, text)
        return r if not r.get("ok") else self.slider_release(vid)

    def slider_range(self, vid: str, *, min: Any = None, max: Any = None,
                     step: Any = None) -> dict[str, Any]:  # fmt: skip
        """The ``>`` box under a slider: its range and step, edited in place (one patch)."""
        v = self._variable(vid)
        if v is None:
            return _refused(f"no variable {vid!r}")
        edit: dict[str, Any] = {"id": v["object"]}
        try:
            for k, raw in (("min", min), ("max", max), ("step", step)):
                if raw is not None and str(raw).strip() != "":
                    edit[k] = _parse(raw, k)
        except ValueError as e:
            return _refused(str(e))
        lo, hi = edit.get("min", v["min"]), edit.get("max", v["max"])
        if not lo < hi:
            return _refused("min must be below max")
        if "step" in edit and not 0 < edit["step"] <= hi - lo:
            return _refused("step must be above 0 and at most max - min")
        if len(edit) == 1:
            return {"ok": True, "changed": False}
        return self._rows_edit([edit], f"{v['name']} range {_num(lo)} to {_num(hi)}")

    def slider_play(self, vid: str, on: bool = True) -> dict[str, Any]:
        """Play a slider (session): its value crosses the range and back, ``PLAY_SECONDS``
        a crossing; nothing is written ("keep" makes it a track)."""
        v = self._variable(vid)
        if v is None:
            return _refused(f"no variable {vid!r}")
        oid = v["object"]
        if not on:
            self.playing_sliders.pop(oid, None)
            self.slider_preview.pop(oid, None)
            return {"ok": True}
        if v["animated"]:
            return _refused(f"{v['name']} is animated by a track already")
        self.playing_sliders[oid] = 1
        self.slider_preview.setdefault(oid, float(v["value"]))
        return {"ok": True}

    def slider_keep(self, vid: str) -> dict[str, Any]:
        """Keep a playing slider: a track on ``<id>.value`` (min to max and back,
        ``PLAY_SECONDS`` each way, linear), a normal animation that renders. One patch."""
        v = self._variable(vid)
        if v is None:
            return _refused(f"no variable {vid!r}")
        if self.read_only:
            return _refused("a track is timing, which a script writes: add it in the code")
        oid, lo, hi = v["object"], float(v["min"]), float(v["max"])
        path = f"{oid}.value"
        ops: list[dict[str, Any]] = []
        if path in self._doc().timeline.tracks:
            ops.append({"op": "clear_track", "path": path})
        if any(c.path == path for c in self._doc().controls):
            # a track drives the value now: its slider goes in the same patch (a Control on
            # an animated parameter is lint orphan_control, and a script refuses it)
            ops.append({"op": "remove_control", "path": path})
        ops += [{"op": "set_param", "path": path, "value": lo},
                {"op": "add_segment", "path": path, "segment": {
                    "t0": 0.0, "t1": PLAY_SECONDS, "to": hi, "ease": "linear"}},
                {"op": "add_segment", "path": path, "segment": {
                    "t0": PLAY_SECONDS, "t1": 2 * PLAY_SECONDS, "to": lo,
                    "ease": "linear"}}]  # fmt: skip
        r = self._edit(ops, f"{v['name']} plays {_num(lo)} to {_num(hi)}")
        if r.get("ok"):
            self.playing_sliders.pop(oid, None)
            self.slider_preview.pop(oid, None)
        return r

    def clear_track(self, vid: str) -> dict[str, Any]:
        v = self._variable(vid)
        if v is None or not v["animated"]:
            return _refused(f"{vid} has no track")
        path = f"{v['object']}.value"
        ops: list[dict[str, Any]] = [{"op": "clear_track", "path": path}]
        if not any(c.path == path for c in self._doc().controls):  # its slider comes back
            ops.append({"op": "add_control", "control": {
                "path": path, "min": float(v["min"]), "max": float(v["max"]),
                "step": X.NEW_VARIABLE["step"], "label": v["name"]}})  # fmt: skip
        return self._edit(ops, f"{v['name']} track cleared")

    # ------------------------------------------------------------ presets, + row, palette

    @staticmethod
    def presets() -> list[dict[str, Any]]:
        return [{"name": k, "note": p.note, "dim": p.dim} for k, p in sorted(fm.PRESETS.items())]

    def load_preset(self, name: str, replace: bool = True) -> dict[str, Any]:
        if not isinstance(name, str) or name not in fm.PRESETS:
            return self._log(_refused(f"no preset {str(name)[:40]!r}; presets are "
                                      f"{', '.join(sorted(fm.PRESETS))}"), "")  # fmt: skip
        if self.read_only:
            return self._log(_refused(CONVERTED), "")
        with self.ed.lock:
            try:
                r = self.ws.explorer_preset(name, bool(replace))
            except ToolError as e:
                r = _refused(str(e))
        r = self._log(r, f"preset {name}")
        if r.get("ok"):
            self.playhead = 0.0
        return r

    def _system(self) -> tuple[str, Any] | tuple[None, None]:
        """The first system (an ODE row system first) and its resolved form at the view."""
        ids = self._systems()
        if not ids:
            return None, None
        st = self.view().state_at(self.playhead)
        sid = ids[0]
        return sid, st[sid].impl.resolve(st)

    def add_items(self) -> list[dict[str, Any]]:
        """The ``+ row`` menu: things that are not typed. Each is one patch."""
        sid, r = self._system()
        dim = r.dim if r is not None else 2
        none = None if sid else "type a system first (x' = ..., or a preset)"
        return [
            {"key": "tracer", "label": "tracer (a point that flows)", "why": none},
            {"key": "ball" if dim == 3 else "disk",
             "label": "ball of starting points" if dim == 3 else "disk of starting points",
             "why": none},
            {"key": "source", "label": "source (a field term)",
             "why": none or ("sources are terms of a 2D field" if dim == 3 else None)},
            {"key": "section", "label": "section plane (Poincare)", "why": none},
            {"key": "obstacle", "label": "obstacle", "why": "not in the Explorer's first "
             "version (collisions are out of scope, EXPLORER.md)"},
        ]  # fmt: skip

    def add(self, item: str, at: Any = None) -> dict[str, Any]:
        """Add a ``+ row`` item at ``at`` (coordinates; default the window's centre)."""
        if self.read_only:
            return self._log(_refused(CONVERTED), "")
        found = next((i for i in self.add_items() if i["key"] == item), None)
        if found is None:
            return _refused(f"no such item {str(item)[:40]!r}")
        if found["why"]:
            return self._log(_refused(found["why"]), "")
        sid, r = self._system()
        c = np.asarray(r.centre) if at is None else self._coords(at, r)
        c = [round(float(a), 6) for a in c]
        doc = self._doc()
        if item == "tracer":
            return self._rows_edit([{"text": _point_text(c)}], "added tracer")
        if item in ("disk", "ball"):
            fid = X.fresh_id(doc, item)
            return self._edit([{"op": "add_object", "object": {"id": fid, "type": "flow_set",
                                "params": {"system": sid, "shape": item, "center": c,
                                           "size": round(0.05 * r.diag, 6)}}}],
                              f"added {item}")  # fmt: skip
        if item == "source":
            terms = list(doc.object(sid).params.get("terms", ()))
            terms.append(f"source(pos=({_num(c[0])}, {_num(c[1])}), strength=1, core=0.1)")
            return self._edit([{"op": "set_param", "path": f"{sid}.terms", "value": terms}],
                              "added source")  # fmt: skip
        normal = [1.0, 0.0] if r.dim == 2 else [0.0, 0.0, 1.0]
        offset = c[0] if r.dim == 2 else c[2]
        params = {"system": sid, "normal": normal, "offset": offset,
                  "seeds": [[round(a + 0.01 * r.diag, 6) for a in c]]}  # fmt: skip
        probes = [e["object"] for e in X.rows_of(doc) if e["kind"] == "point"]
        if probes:
            params["of"] = probes[0]
        return self._edit([{"op": "add_object", "object": {
            "id": X.fresh_id(doc, "section"), "type": "flow_section", "params": params}}],
            "added section")  # fmt: skip

    def _coords(self, at: Any, r) -> np.ndarray:
        a = [float(x) for x in at][: r.dim]
        return np.asarray(a + [0.0] * (r.dim - len(a)))

    @staticmethod
    def palette() -> tuple[dict[str, Any], ...]:
        return palette()

    def place(self, type_name: str, at: tuple[float, float, float] | None = None) -> dict:
        """Place a palette type (one patch): at world point ``at`` when it has a position,
        on the first system when it is a flow object."""
        if self.read_only:
            return self._log(_refused(CONVERTED), "")
        item = next((p for p in palette() if p["type"] == type_name), None)
        if item is None:
            return _refused(f"{str(type_name)[:40]!r} is not in the insert palette")
        params: dict[str, Any] = {}
        if item["pos"] and at is not None:
            params["pos"] = [round(float(a), 6) for a in at][:3]
        if item["flow"]:
            ids = self._systems()
            if not ids:
                return self._log(_refused("type a system first (x' = ..., or a preset)"), "")
            params["system"] = ids[0]
            if item["near"]:  # an equilibrium's close-up: the one nearest the window's centre
                _, r = self._system()
                params["near"] = [round(float(a), 6) for a in r.centre]
        oid = X.fresh_id(self._doc(), re.sub(r"^flow_", "", type_name))
        ops = [{"op": "add_object", "object": {"id": oid, "type": type_name, "params": params}}]
        what = f"added {item['label']}"
        if not item["flow"]:
            return self._edit(ops, what)
        # A flow object may take seconds to compute (a limit cycle): it is computed on a fork
        # on the analysis thread (its memos are shared), then committed here, quickly.
        with self.ed.lock:
            fork = self.ed.engine.fork()
        self._gen += 1
        job = {"ops": ops, "what": what, "id": oid, "gen": self._gen, "t": self.playhead}
        job["future"] = self.analysis_pool.submit(_warm, fork, ops, oid, self.playhead)
        self._placing.append(job)
        self.message = f"placing {item['label']}: computing..."
        if job["future"].done():
            return self._finish_place(job)
        return {"ok": True, "pending": True, "id": oid}

    def _finish_place(self, job: dict[str, Any]) -> dict[str, Any]:
        """A placement computed on the analysis thread is committed (one patch)."""
        if job in self._placing:
            self._placing.remove(job)
        try:
            job["future"].result()
        except Exception as e:  # the fork refused it: the real edit says why, plainly
            self.message = f"placing failed: {type(e).__name__}"
        return self._edit(job["ops"], job["what"])

    def placing(self) -> list[str]:
        """Objects being computed before they are placed (the panel says "computing")."""
        return [j["id"] for j in self._placing]

    # ------------------------------------------------------------ the transport (session)

    def toggle_play(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle_reverse(self) -> bool:
        """Reverse replays the playhead backwards (the flow is never integrated backwards
        here; backward flow is a system's negative ``tau``)."""
        self.reverse = not self.reverse
        return self.reverse

    def restart(self) -> None:
        self.playhead = 0.0
        self.slider_preview = {k: v for k, v in self.slider_preview.items()
                               if k not in self.playing_sliders}  # fmt: skip

    def step(self, n: int = 1) -> float:
        self.playing = False
        self.playhead = max(0.0, min(X.flow.MAX_TAU, self.playhead + n * TICK * self.speed))
        return self.playhead

    def scrub(self, flow_time: Any) -> float:
        try:
            self.playhead = max(0.0, min(X.flow.MAX_TAU, _parse(flow_time, "the flow time")))
        except ValueError as e:
            self.message = str(e)
        return self.playhead

    def set_speed(self, x: Any) -> float:
        try:
            v = _parse(x, "the speed")
        except ValueError as e:
            self.message = str(e)
            return self.speed
        self.speed = min(max(abs(v), SPEEDS[0]), SPEEDS[1])
        return self.speed

    def speed_text(self) -> str:
        """The readout: how fast the flow plays against normal (one unit of flow time a
        second)."""
        return f"{self.speed:.3g}x normal" + (", reversed" if self.reverse else "")

    def flow_time(self) -> float:
        """The flow time shown: the first system's resting ``tau`` plus the playhead."""
        ids = self._systems()
        tau = self._doc().object(ids[0]).params.get("tau", 0.0) if ids else 0.0
        return (float(tau) if isinstance(tau, int | float) else 0.0) + self.playhead

    def advance(self, dt: float) -> bool:
        """The window's tick (``dt`` wall seconds): the playhead and playing sliders move.
        Never edits the document. True when the picture changes."""
        moved = False
        if self.playing:
            sign = -1.0 if self.reverse else 1.0
            self.playhead = max(0.0, min(X.flow.MAX_TAU, self.playhead + sign * dt * self.speed))
            if self.reverse and self.playhead <= 0.0:
                self.playing = False
            moved = True
        for oid, direction in list(self.playing_sliders.items()):
            v = self._variable(oid)
            if v is None:
                self.playing_sliders.pop(oid, None)
                continue
            lo, hi = float(v["min"]), float(v["max"])
            x = (
                self.slider_preview.get(oid, float(v["value"]))
                + direction * dt * (hi - lo) / PLAY_SECONDS
            )
            if x >= hi or x <= lo:
                self.playing_sliders[oid] = -direction
                x = min(max(x, lo), hi)
            self.slider_preview[oid] = x
            moved = True
        return moved

    # ------------------------------------------------------------ the view and the picture

    def _view_key(self) -> tuple:
        tp = self._typing_preview
        if tp is not None and tp[1] != self.ed.version:
            self._typing_preview = tp = None  # the document moved on (an AI edit): stale
        return (self.ed.generation, self.ed.version, tp[0] if tp else None,
                tuple(sorted(self.slider_preview.items())),
                repr(sorted(self.camera_session.items())), repr(self._handle_ops))  # fmt: skip

    def view(self):
        """The engine the picture is drawn from (a fork; see the module's docstring)."""
        key = self._view_key()
        if self._view is not None and self._view[0] == key:
            return self._view[1]
        with self.ed.lock:
            base = self._typing_preview[2] if self._typing_preview else self.ed.engine
            fork = base.fork()
        doc = fork.document
        rates = [{"op": "set_param", "path": f"{o.id}.rate", "value": 1.0}
                 for o in doc.objects if o.type == "flow_system"]  # fmt: skip
        session = [
            {"op": "set_param", "path": f"{k}.value", "value": v}
            for k, v in self.slider_preview.items()
            if doc.object(k) is not None
        ]
        if self.camera_session and doc.cameras:
            cam = doc.timeline.camera
            session += [{"op": "set_param", "path": f"{cam}.{k}", "value": v}
                        for k, v in self.camera_session.items()]  # fmt: skip
        for ops in (rates + session + self._handle_ops, rates + session, rates):
            if not ops or fork.apply(ops).ok:
                break
        self._view = (key, fork)
        return fork

    def render_plan(self, size: tuple[int, int]) -> tuple[tuple[int, int], str]:
        """Progressive quality: while the person interacts (and ``SETTLE`` seconds after),
        half size with ``quality="interactive"`` (thinned batches, Milestone 28); then the
        full picture. A render hint only: the document never changes."""
        if self._interacting():
            return (max(2, size[0] // 2), max(2, size[1] // 2)), "interactive"
        return (int(size[0]), int(size[1])), "full"

    def frame(self, size: tuple[int, int]) -> tuple[np.ndarray, str]:
        """The picture at the playhead and its quality. Needs OpenGL (``RuntimeError``
        without it, as ``Editor.frame``)."""
        rs, q = self.render_plan(size)
        view = self.view()
        key = (self._view[0] if self._view else None, round(self.playhead, 6), rs, q)
        if self._frame is None or self._frame[0] != key:
            start = time.monotonic()
            img = np.ascontiguousarray(view.still(self.playhead, rs, quality=q)[..., :3])
            self._frame = (key, img)
            done = time.monotonic()
            self._times.append((done, done - start, q))
            del self._times[:-120]
            if self._input_at is not None:
                self._latency.append(done - self._input_at)
                del self._latency[:-60]
                self._input_at = None
        self._shown_quality = q
        return self._frame[1], q

    def perf(self) -> dict[str, Any]:
        """Measured here, wall clock (the owner's feel check, ``pc_tasks/14``): frames a
        second over the last two seconds of drawing, the median ms to draw a frame (each
        quality), and the median ms from handling an input to the end of the first frame
        drawn after it. Not the whole latency: Qt's event queue, the paint and the display
        come on top; judge lag by eye too."""
        now = time.monotonic()
        recent = [t for t in self._times if now - t[0] <= 2.0]
        span = (recent[-1][0] - recent[0][0]) if len(recent) > 1 else 0.0
        out: dict[str, Any] = {"fps": (len(recent) - 1) / span if span > 0 else None}
        for q in ("interactive", "full"):
            ms = sorted(1000 * t[1] for t in self._times if t[2] == q)
            out[f"{q}_ms"] = ms[len(ms) // 2] if ms else None
        lat = sorted(1000 * x for x in self._latency)
        out["latency_ms"] = lat[len(lat) // 2] if lat else None
        return out

    def perf_text(self) -> str:
        p = self.perf()
        parts = [f"{p['fps']:.1f} fps" if p["fps"] else "",
                 f"draw {p['interactive_ms']:.0f} ms (moving)" if p["interactive_ms"] else "",
                 f"{p['full_ms']:.0f} ms (still)" if p["full_ms"] else "",
                 f"input to picture {p['latency_ms']:.0f} ms" if p["latency_ms"] else "",
                 ]  # fmt: skip
        return ", ".join(x for x in parts if x)

    # ------------------------------------------------------------ the viewport

    def _state(self):
        return self.view().state_at(self.playhead)

    def _cam_now(self) -> dict[str, Any]:
        """The camera's parameters as shown: the document's (with defaults) and the
        session's over them."""
        _, cur = self.ed.camera_params()
        return {**cur, **self.camera_session}

    @staticmethod
    def _ray(cam, u: float, v: float, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        ndc = np.array([2 * u / w - 1, 1 - 2 * v / h])
        inv = np.linalg.inv(cam.view_proj)
        near = np.array([*ndc, -1.0, 1.0]) @ inv.T
        far = np.array([*ndc, 1.0, 1.0]) @ inv.T
        a, b = near[:3] / near[3], far[:3] / far[3]
        return a, b - a

    def world_at(self, u: float, v: float, size: tuple[int, int],
                 through: Any = None) -> np.ndarray:  # fmt: skip
        """The world point under a viewport pixel: in 2D on the plane z = 0; in 3D on the
        plane facing the camera through ``through`` (default the orbit target)."""
        st = self._state()
        cam = st.camera
        if cam.kind != "perspective":
            z = 0.0 if through is None else float(np.asarray(through)[2])
            return np.asarray(cam.from_screen([(u, v)], size[0], size[1], z)[0])
        p0 = np.asarray(through if through is not None else self._cam_now().get("target",
                                                                                (0, 0, 0)),
                        dtype=np.float64)  # fmt: skip
        a, d = self._ray(cam, u, v, size[0], size[1])
        n = np.asarray(cam.forward)
        den = float(d @ n)
        k = 0.0 if abs(den) < 1e-12 else float((p0 - a) @ n) / den
        return a + k * d

    def coords_at(self, u: float, v: float, size: tuple[int, int], through: Any = None):
        """The first system's coordinates under a viewport pixel (None without a system)."""
        _, r = self._system()
        if r is None:
            return None
        w = self.world_at(u, v, size, through)
        return ((w - r.origin) / r.scale)[: r.dim]

    def _screen(self, world, size) -> np.ndarray:
        return self._state().camera.to_screen(np.atleast_2d(world), size[0], size[1])

    def equilibrium_at(self, u: float, v: float, size: tuple[int, int]) -> dict | None:
        """The equilibrium nearest a viewport pixel (within ``PICK_PX``), as a card: its
        system, position, kind, eigenvalues and subspace dimensions, from the equilibria
        object's semantics (the numbers ``find_equilibria`` gives)."""
        st = self._state()
        best = None
        for o in st.objects:
            if o.impl.type != "flow_equilibria" or o.params.opacity <= 0:
                continue
            sem = o.impl.semantics(st)
            sid = sem.get("system")
            if not sid or not sem.get("points"):
                continue
            r = st[sid].impl.resolve(st)
            at = np.array([p["at"] for p in sem["points"]], dtype=np.float64)
            px = self._screen(r.world(at), size)
            dist = np.hypot(px[:, 0] - u, px[:, 1] - v)
            k = int(np.argmin(dist))
            if dist[k] <= PICK_PX and (best is None or dist[k] < best[0]):
                best = (float(dist[k]), sid, k, sem["points"][k], r)
        if best is None:
            return None
        _, sid, k, p, r = best
        return self._card(sid, k, p, r, self._eq_names(st, sid)[k])

    def _eq_points(self, st, sid: str) -> list[dict[str, Any]]:
        obj = next((o for o in st.objects if o.impl.type == "flow_equilibria"
                    and o.impl.semantics(st).get("system") == sid), None)  # fmt: skip
        return list(obj.impl.semantics(st).get("points", ())) if obj is not None else []

    def _eq_names(self, st, sid: str) -> list[str]:
        return equilibrium_names(self._eq_points(st, sid))

    def refresh_card(self) -> dict[str, Any] | None:
        """The card follows its equilibrium (advice item 16): after any change it is found
        again by position (the nearest within 5% of the window's diagonal), with fresh
        numbers; one that is gone closes the card, one that changed kind says so."""
        c = self.card
        if c is None:
            return None
        key = self._view_key(), round(self.playhead, 6)
        if c.get("_seen") == key:
            return c
        st = self._state()
        sid = c["system"]
        if sid not in {o.id for o in st.objects}:
            self.card = None
            self.message = f"{c['name']} is gone (its system was removed)"
            return None
        r = st[sid].impl.resolve(st)
        pts = self._eq_points(st, sid)
        if pts:
            d = [float(np.linalg.norm(np.subtract(p["at"], c["at"]))) for p in pts]
            k = int(np.argmin(d))
        if not pts or d[k] > 0.05 * r.diag:
            self.card = None
            self.message = f"{c['name']} {_point_text(c['at'])} is gone at these values"
            return None
        new = self._card(sid, k, pts[k], r, equilibrium_names(pts)[k])
        if new["kind"] != c["kind"]:
            self.message = f"{c['name']} is now {new['name']}, a {new['kind']} (was a {c['kind']})"
        new["_seen"] = key
        self.card = new
        return new

    @staticmethod
    def _card(sid: str, k: int, p: dict[str, Any], r, name: str | None = None) -> dict:
        ev = []
        for a, b in p["eigenvalues"]:
            real = abs(b) <= p.get("tolerance", 1e-9)
            ev.append(_num(a) if real else f"{_num(a)} {'+' if b > 0 else '-'} {_num(abs(b))}i")
        d = p["dims"]
        name = name or f"E{k + 1}"
        lines = [f"{name} = {_point_text(p['at'])}  {p['kind']}",
                 "eigenvalues " + ", ".join(ev),
                 f"dim Es = {d['s']}, dim Eu = {d['u']}"
                 + (f", dim Ec = {d['c']}" if d.get("c") else "")]  # fmt: skip
        return {"system": sid, "index": k, "name": name, "at": list(p["at"]), "kind": p["kind"],
                "eigenvalues": p["eigenvalues"], "dims": d, "lines": lines,
                "diag": float(r.diag),
                "world": [float(a) for a in r.world(np.array(p["at"]))[0]]}  # fmt: skip

    def handles(self, size: tuple[int, int]) -> list[dict[str, Any]]:
        """Drag handles in viewport pixels: a set's centre (move) and rim (resize), a
        tracer's dot (move)."""
        st = self._state()
        cam = st.camera
        out = []
        for o in st.objects:
            p = o.params
            if o.impl.type == "flow_set" and p.opacity > 0:
                r = o.impl._sys(st)
                if r is None:
                    continue
                c = np.asarray(tuple(p.center[: r.dim]) + (0.0,) * (r.dim - len(p.center)))
                wc = r.world(c)[0]
                rim = wc + r.scale * float(p.size) * np.asarray(cam.right)
                sc, sr = self._screen(np.stack([wc, rim]), size)
                out.append({"id": o.id, "kind": "set", "at": sc.tolist(),
                            "radius": float(np.hypot(*(sr - sc))),
                            "world": wc.tolist()})  # fmt: skip
            elif o.impl.type == "flow_probe" and p.opacity > 0 and getattr(p, "visible", True):
                r = o.impl._sys(st)
                if r is None:
                    continue
                x0 = np.asarray(tuple(p.x0[: r.dim]) + (0.0,) * (r.dim - len(p.x0)))
                wc = r.world(x0)[0]
                sc = self._screen(wc, size)[0]
                out.append({"id": o.id, "kind": "tracer", "at": sc.tolist(), "radius": 0.0,
                            "world": wc.tolist()})  # fmt: skip
        return out

    def _handle_at(self, u: float, v: float, size) -> dict[str, Any] | None:
        for h in self.handles(size):
            d = math.hypot(h["at"][0] - u, h["at"][1] - v)
            if d <= PICK_PX:
                return {**h, "part": "centre"}
            if h["kind"] == "set" and abs(d - h["radius"]) <= PICK_PX / 2:
                return {**h, "part": "rim"}
        return None

    def press(self, u: float, v: float, size: tuple[int, int], shift: bool = False,
              pan: bool = False) -> None:  # fmt: skip
        """Mouse down in the viewport (frame pixels, top-left origin, at display size).
        ``shift``: draw a disk or ball; ``pan`` (ctrl or the middle button): slide the 3D
        camera instead of turning it."""
        self._input_at = time.monotonic()
        self.mouse_down = True
        self.last_input = self.clock()
        p: dict[str, Any] = {"u": u, "v": v, "size": tuple(size), "moved": False, "pan": pan}
        h = self._handle_at(u, v, size) if not shift else None
        if h is not None:
            p.update(mode="handle", handle=h)
        elif shift and self._systems() and not self.read_only:
            c = self.coords_at(u, v, size)
            p.update(mode="draw", centre=c, id=X.fresh_id(self._doc(), self._set_shape()))
        else:
            st = self._state()
            p.update(mode="camera", start=self._cam_now(), right=np.asarray(st.camera.right),
                     up=np.asarray(st.camera.up), perspective=st.camera.perspective)  # fmt: skip
        self._press = p

    def _set_shape(self) -> str:
        _, r = self._system()
        return "ball" if r is not None and r.dim == 3 else "disk"

    def move(self, u: float, v: float, size: tuple[int, int]) -> bool:
        """The mouse moved (with a button down: a drag; else hover). True: draw again."""
        p = self._press
        if p is None:
            self.hover = (u, v)
            return False
        if not p["moved"] and abs(u - p["u"]) + abs(v - p["v"]) < DRAG_PX:
            return False
        p["moved"] = True
        self.last_input = self.clock()
        self._input_at = time.monotonic()
        mode = p["mode"]
        if mode == "camera":
            self.camera_session = self._camera_drag(p, u - p["u"], v - p["v"], size)
            self._camera_pending = True
        elif mode == "handle":
            self._handle_ops = self._handle_drag(p["handle"], u, v, size)
        elif mode == "draw" and p["centre"] is not None:
            c = self.coords_at(u, v, size, through=self._world_of(p["centre"]))
            radius = float(np.linalg.norm(np.asarray(c) - np.asarray(p["centre"])))
            p["radius"] = max(radius, 1e-3)
            sid = self._systems()[0]
            self._handle_ops = [{"op": "add_object", "object": {"id": p["id"], "type": "flow_set",
                                 "params": {"system": sid, "shape": self._set_shape(),
                                            "center": [round(float(a), 6) for a in p["centre"]],
                                            "size": round(p["radius"], 6)}}}]  # fmt: skip
        return True

    def _world_of(self, coords) -> np.ndarray:
        _, r = self._system()
        return r.world(np.asarray(coords))[0]

    def _camera_drag(self, p: dict, dx: float, dy: float, size) -> dict[str, Any]:
        """Session camera params for a drag of ``dx``, ``dy`` pixels from the press: orbit
        (3D, 0.4 degrees a pixel, as ``Editor.drag_camera``) or pan (2D; shift in 3D)."""
        cur = p["start"]
        spec, _ = self.ed.camera_params()
        if spec.kind == "orbit" and not p["pan"]:
            el = min(max(float(cur["elevation"]) - dy * 0.4, -89.0), 89.0)
            return {
                "azimuth": round(float(cur["azimuth"]) - dx * 0.4, 6),
                "elevation": round(el, 6),
            }
        if spec.kind == "orbit":
            per_px = 2 * float(cur["distance"]) * math.tan(math.radians(float(cur["fov"])) / 2)
            key = "target"
        else:
            per_px, key = float(cur["height"]), "center"
        per_px /= size[1]
        move = p["right"] * (-dx * per_px) + p["up"] * (dy * per_px)
        return {key: [round(float(a + b), 9) for a, b in zip(cur[key], move, strict=True)]}

    def _handle_drag(self, h: dict, u: float, v: float, size) -> list[dict[str, Any]]:
        c = self.coords_at(u, v, size, through=h["world"])
        if c is None:
            return []
        spec = self._doc().object(h["id"])
        if h["kind"] == "tracer":
            return [{"op": "set_param", "path": f"{h['id']}.x0",
                     "value": [round(float(a), 6) for a in c]}]  # fmt: skip
        if h["part"] == "centre":
            return [{"op": "set_param", "path": f"{h['id']}.center",
                     "value": [round(float(a), 6) for a in c]}]  # fmt: skip
        centre = np.asarray(spec.params.get("center", (1.0, 0.0)), dtype=np.float64)
        centre = np.r_[centre, np.zeros(max(0, len(c) - len(centre)))][: len(c)]
        size_ = max(float(np.linalg.norm(np.asarray(c) - centre)), 1e-3)
        return [{"op": "set_param", "path": f"{h['id']}.size", "value": round(size_, 6)}]

    def release(self, u: float, v: float, size: tuple[int, int]) -> dict[str, Any] | None:
        """Mouse up: a click adds a tracer (empty space) or opens an equilibrium's card; a
        drag commits what it did as one patch (camera, handle, a new disk or ball)."""
        p, self._press = self._press, None
        self.mouse_down = False
        self.last_input = self.clock()
        if p is None:
            return None
        if not p["moved"]:
            return self.click(p["u"], p["v"], size)
        mode = p["mode"]
        if mode == "camera":
            return self._commit_camera("orbit" if "azimuth" in self.camera_session else "pan")
        ops, self._handle_ops = self._handle_ops, []
        if not ops:
            return None
        if mode == "draw":
            return self._edit(ops, f"added {self._set_shape()}")
        h = p["handle"]
        if h["kind"] == "tracer":
            return self._rows_edit([{"id": h["id"], "text": _point_text(ops[0]["value"])}],
                                   f"moved {h['id']}")  # fmt: skip
        return self._edit(ops, f"{'moved' if h['part'] == 'centre' else 'resized'} {h['id']}")

    def click(self, u: float, v: float, size: tuple[int, int]) -> dict[str, Any] | None:
        card = self.equilibrium_at(u, v, size)
        if card is not None:
            self.card = card
            return {"ok": True, "card": card}
        if self._handle_at(u, v, size) is not None:
            return None
        self.card = None
        if not self._systems():
            self.message = (
                "type a system first (x' = ..., or a preset), then click to drop a tracer"
            )
            return None
        c = self.coords_at(u, v, size)
        if c is None or self.read_only:
            return None
        return self._rows_edit([{"text": _point_text(c)}], "added tracer")

    def double_click(self, u: float, v: float, size: tuple[int, int]) -> dict[str, Any] | None:
        card = self.equilibrium_at(u, v, size)
        if card is None:
            return None
        self.card = card
        return self.zoom_in(card)

    def zoom_in(self, card: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fly the camera to an equilibrium (session, ``ZOOM_SECONDS``), then commit one
        patch: the camera where it arrived and the equilibrium's ``EquilibriumView``."""
        if card is not None:
            self.card = card
        card = self.refresh_card()  # fresh numbers: never a stale kind or position
        if card is None:
            return _refused(self.message or "click an equilibrium first")
        start = self._cam_now()
        spec, _ = self.ed.camera_params()
        if spec.kind == "orbit":
            to = {"target": [round(a, 6) for a in card["world"]],
                  "distance": round(max(float(start["distance"]) * 0.35, 1e-3), 6)}  # fmt: skip
        else:
            to = {"center": [round(a, 6) for a in card["world"]],
                  "height": round(max(float(start["height"]) * 0.35, 1e-3), 6)}  # fmt: skip
        self._anim = {"from": {k: start[k] for k in to}, "to": to, "start": self.clock(),
                      "card": card}  # fmt: skip
        self.last_input = self.clock()
        return {"ok": True, "flying": True}

    def _fly(self, now: float) -> None:
        a = self._anim
        u = min(1.0, (now - a["start"]) / ZOOM_SECONDS)
        s = u * u * (3 - 2 * u)
        cam = {}
        for k, end in a["to"].items():
            b = a["from"][k]
            if isinstance(end, list):
                cam[k] = [float(x) + s * (float(y) - float(x)) for x, y in zip(b, end, strict=True)]
            else:
                cam[k] = float(b) + s * (float(end) - float(b))
        self.camera_session = {**self.camera_session, **cam}
        self.last_input = now
        if u < 1.0:
            return
        self._anim = None
        self.card = a["card"]
        card = self.refresh_card() or a["card"]  # the scene may have changed during the flight
        doc = self._doc()
        spec, _ = self.ed.camera_params()
        tracks = doc.timeline.tracks
        ops = [{"op": "set_param", "path": f"{spec.id}.{k}", "value": v}
               for k, v in a["to"].items() if f"{spec.id}.{k}" not in tracks]  # fmt: skip
        ops.append({"op": "add_object", "object": {
            "id": X.fresh_id(doc, f"{card['system']}_{card['name'].lower()}"),
            "type": "flow_equilibrium_view",
            "params": {"system": card["system"], "near": card["at"],
                       "radius": round(0.08 * card["diag"], 6),
                       "name": card["name"]}}})  # fmt: skip
        r = self._edit(ops, f"zoom in to {card['name']} ({card['kind']})")
        if r.get("ok") and len(ops) == len(a["to"]) + 1:
            for k in a["to"]:
                self.camera_session.pop(k, None)

    def wheel(self, factor: Any) -> None:
        """The wheel (> 1 moves in): the session camera; the zoom commits once the wheel has
        been still ``WHEEL_IDLE`` seconds (``poll``), as one step."""
        try:
            f = min(max(_parse(factor, "the zoom"), 1e-3), 1e3)
        except ValueError:
            return
        cur = self._cam_now()
        spec, _ = self.ed.camera_params()
        key = "distance" if spec.kind == "orbit" else "height"
        self.camera_session[key] = float(cur[key]) / max(f, 1e-3)
        self._input_at = time.monotonic()
        self._camera_pending = True
        self._wheel_at = self.last_input = self.clock()

    def _commit_camera(self, what: str) -> dict[str, Any] | None:
        """The session camera becomes the resting camera: one patch. A camera a track
        animates (a recording) cannot be moved by hand: the view keeps the session camera
        and says why."""
        if not self._camera_pending or not self.camera_session:
            return None
        self._camera_pending = False
        _, cur = self.ed.camera_params()
        updates = {k: v for k, v in self.camera_session.items() if not _same(cur.get(k), v)}
        if not updates:
            self.camera_session = {}
            return None
        r = self.ed.set_camera(updates, what)
        if r.get("ok"):
            self.camera_session = {}
        else:
            self.message = ("the camera is animated by a track (a recording): this view is "
                            "not saved; clear the track to move the resting camera")  # fmt: skip
        return r

    def key(self, name: str) -> bool:
        """Keys over the viewport: space pauses, left/right step, r restarts."""
        if name == "space":
            self.toggle_play()
        elif name == "left":
            self.step(-1)
        elif name == "right":
            self.step(1)
        elif name == "r":
            self.restart()
        else:
            return False
        return True

    # ------------------------------------------------------------ R2 / R3

    def space(self) -> str:
        doc = self._doc()
        if doc.cameras and doc.cameras[0].kind == "orbit":
            return "R3"
        for o in doc.objects:
            if o.type == "flow_system" and len(o.params.get("equations", ())) == 3:
                return "R3"
        return "R2"

    def space_path(self, space: str) -> Path | None:
        p = self.ws.path
        if p is None or not p.name.endswith(io.DOC_SUFFIX):
            return None
        base = re.sub(r"_r[23]$", "", p.name[: -len(io.DOC_SUFFIX)])
        cand = p.with_name(f"{base}_{space.lower()}{io.DOC_SUFFIX}")
        plain = p.with_name(f"{base}{io.DOC_SUFFIX}")  # the file the person started from
        if not cand.exists() and plain.exists() and plain != p:
            return plain
        return cand

    def switch_space(self, space: str) -> dict[str, Any]:
        """The R2 / R3 tabs: two documents (two files, ``<name>_r2`` and ``<name>_r3``),
        each with its own field, camera and settings. The other one is opened (made from a
        preset the first time). ``needs_save`` when this one is unsaved."""
        if space not in ("R2", "R3"):
            return _refused("the spaces are R2 and R3")
        if space == self.space():
            return {"ok": True, "space": space}
        if self.read_only:
            return _refused(CONVERTED)
        target = self.space_path(space)
        if target is None:
            return {**_refused("save this explorer first: R2 and R3 are two files"),
                    "needs_save": True}  # fmt: skip
        self.settle()  # a row typed just now is part of what must be saved first
        if self.ed.dirty:
            return {**_refused("save this scene first: it has edits not saved"),
                    "needs_save": True}  # fmt: skip
        if not target.exists():
            from phasor.workspace.tools import Workspace

            w = Workspace(target.parent)
            meta = self._doc().meta
            w.scene_new(target.name[: -len(io.DOC_SUFFIX)], look=meta.look or "none",
                        width=meta.width, height=meta.height)  # fmt: skip
            w.explorer_preset("lorenz" if space == "R3" else "van_der_pol")
            w.scene_save(str(target))
        r = self.ed.open_document(target)
        if r.get("ok"):
            self._reset_session()
        return {**r, "space": space}

    # ------------------------------------------------------------ convert to script

    def prepare_convert(self) -> str:
        """Every row the person is typing in is committed or let go, and the editor's text
        becomes the code generated from the document now (what Run turns into the script).
        The window then runs it as Run does; ``convert`` runs it here."""
        self.settle()
        with self.ed.lock:
            text = self.ed.code()["text"]
        self.ed.set_buffer(text)
        return text

    def convert(self, *, wait: bool = True) -> dict[str, Any]:
        """Convert to script (ask with ``CONVERT_WARNING`` first): Run on the code generated
        from the document as it is now writes ``name.scene.py`` and opens it (``wait=False``:
        on the editor's worker thread, finished by ``Editor.poll_run``)."""
        if self.read_only:
            return _refused("this is a script already")
        self.prepare_convert()
        r = self.ed.run(wait=wait)
        if r.get("ok") or r.get("running"):
            self._reset_session()
        return r

    # ------------------------------------------------------------ record, send to scene

    def record(self, on: bool) -> dict[str, Any]:
        """The record toggle. On: marks start empty. Off: the marks become the scene (one
        patch, ``marks_ops``)."""
        if on:
            if self.read_only:
                return _refused("recording writes timing, which a script holds: add it in code")
            self.recording, self.marks = True, []
            return {"ok": True, "recording": True}
        self.recording = False
        marks, self.marks = self.marks, []
        if not marks:
            return {"ok": True, "marks": 0}
        spec, _ = self.ed.camera_params()
        ops, names = marks_ops(self._doc(), spec.id, marks)
        before = self.ed.engine.duration
        r = self._edit(ops, f"recorded {len(marks)} mark{'s' if len(marks) > 1 else ''}")
        after = self.ed.engine.duration
        if r.get("ok") and after > before + 1e-9:
            self.message = f"recorded {len(marks)} marks: the scene now lasts {_num(after)} s"
        return {**r, "marks": len(marks), "markers": names}

    def mark(self) -> dict[str, Any]:
        """Keep this moment: the playhead, the camera as shown and every variable's value.
        Only marks are kept (eased between), never every mouse wiggle."""
        if not self.recording:
            return _refused("press record first")
        if len(self.marks) >= MAX_MARKS:
            return _refused(f"at most {MAX_MARKS} marks in one recording: stop it, then "
                            "record again")  # fmt: skip
        if self.marks and self.playhead < self.marks[-1]["t"]:
            self.message = (f"marked at tau {_num(self.playhead)}, before the last mark: it "
                            f"plays {_num(HOLD)} s after it (marks keep the order you take "
                            "them)")  # fmt: skip
        _, cur = self.ed.camera_params()
        cam = {
            k: v
            for k, v in {**cur, **self.camera_session}.items()
            if k in ("azimuth", "elevation", "distance", "target", "center", "height", "fov")
        }
        values = {e["object"]: float(e["value"]) for e in self.rows() if e["kind"] == "variable"}
        self.marks.append({"t": self.playhead, "camera": cam, "values": values})
        return {"ok": True, "marks": len(self.marks)}

    # ------------------------------------------------------------ export

    def export_plan(self, tau0: Any, tau1: Any, seconds: Any) -> dict[str, Any]:
        """The export dialog's question, "from tau 0 to 30 in 20 s", checked: numbers
        finite, flow times within the systems' range, a positive duration, a rate the
        systems take. ``notes`` say when a system's steps will be clamped."""
        try:
            a, b, s = _parse(tau0, "from tau"), _parse(tau1, "to tau"), _parse(seconds, "seconds")
        except ValueError as e:
            return _refused(str(e))
        if not EXPORT_SECONDS[0] <= s <= EXPORT_SECONDS[1]:
            return _refused(f"the video lasts {EXPORT_SECONDS[0]:g} to {EXPORT_SECONDS[1]:g} s")
        if max(abs(a), abs(b)) > X.flow.MAX_TAU:
            return _refused(f"flow times are within +/-{X.flow.MAX_TAU:g}")
        rate = (b - a) / s
        if abs(rate) > 1e3:
            return _refused("that is more than 1000 units of flow time a second")
        notes = []
        for o in self._doc().objects:
            if o.type == "flow_system":
                dt = float(o.params.get("dt", 0.01))
                if max(abs(a), abs(b)) / dt > fm.MAX_STEPS:
                    notes.append(f"{o.id}: more than {fm.MAX_STEPS} steps of dt {dt:g}; the "
                                 "flow stops there (lint flow_budget)")  # fmt: skip
        return {"ok": True, "tau0": a, "tau1": b, "seconds": s, "rate": rate, "notes": notes,
                "text": f"from tau {_num(a)} to {_num(b)} in {_num(s)} s"}  # fmt: skip

    def export_path(self) -> Path:
        """The export's own default file (``<scene>.explore.mp4`` beside the renders): never
        the real render's name, so an export never stands in for it."""
        return self.ed.default_output(".mp4").with_suffix(".explore.mp4")

    def export_engine(self, plan: dict[str, Any]):
        """A private fork that plays ``plan`` (each system's ``tau`` and ``rate``, the
        duration): the document is not changed."""
        with self.ed.lock:
            fork = self.ed.engine.fork()
        tracks = fork.document.timeline.tracks
        ops: list[dict[str, Any]] = []
        for o in fork.document.objects:
            if o.type == "flow_system":
                for k, v in (("tau", plan["tau0"]), ("rate", plan["rate"])):
                    if f"{o.id}.{k}" not in tracks:
                        ops.append({"op": "set_param", "path": f"{o.id}.{k}", "value": v})
        ops.append({"op": "set_meta", "field": "duration", "value": plan["seconds"]})
        r = fork.apply(ops)
        if not r.ok:
            raise ToolError("; ".join(i.message for i in r.errors))
        return fork

    def export(self, path: str | Path, tau0: Any, tau1: Any, seconds: Any,
               preset: str = "draft", overwrite: bool | None = None) -> dict[str, Any]:  # fmt: skip
        """Export a video through the normal renderer (a render job, followed in the status
        bar like a draft)."""
        plan = self.export_plan(tau0, tau1, seconds)
        if not plan.get("ok"):
            return plan
        try:
            fork = self.export_engine(plan)
            st = self.ed.render_engine(
                fork, Path(path), preset, start=0.0, end=plan["seconds"], overwrite=overwrite
            )
        except ToolError as e:
            return _refused(str(e))
        self.export_job = st["job"]  # its own: the editor's draft render stays followed
        return {"ok": True, **st, "plan": plan}

    def export_status(self) -> dict[str, Any] | None:
        """The export job's status (None when there is none); a finished one is let go."""
        if self.export_job is None:
            return None
        try:
            st = self.ed.job_status(self.export_job)
        except ToolError:
            self.export_job = None
            return None
        if st["state"] not in ("running", "cancelling"):
            self.export_job = None
        return st

    def cancel_export(self) -> None:
        if self.export_job is not None:
            with contextlib.suppress(ToolError):  # it may have finished already
                self.ed.cancel_render(self.export_job)
            self.export_job = None

    def needs_replace(self, path: str | Path) -> bool:
        """The export's target exists: the person is asked before it is replaced."""
        return Path(path).exists()

    # ------------------------------------------------------------ the analysis tab

    def equilibria_table(self) -> list[dict[str, Any]]:
        """Every equilibrium the view's systems find, with its card lines."""
        st = self._state()
        out = []
        for o in st.objects:
            if o.impl.type != "flow_equilibria":
                continue
            sem = o.impl.semantics(st)
            sid = sem.get("system")
            if not sid:
                continue
            r = st[sid].impl.resolve(st)
            names = equilibrium_names(list(sem.get("points", ())))
            for k, p in enumerate(sem.get("points", ())):
                c = self._card(sid, k, p, r, names[k])
                out.append({k2: c[k2] for k2 in ("system", "name", "at", "kind", "lines")})
            if sem.get("window") is not None:
                out.append({"system": sid, "name": "", "at": [], "kind": "searched",
                            "lines": [f"searched the window {sem['window']} with a grid of "
                                      f"{sem.get('grid')} a side"]})  # fmt: skip
        return out

    def _start(self, name: str, fn: Callable, *args: Any) -> dict[str, Any]:
        with self.ed.lock:
            fork = self.ed.engine.fork()
            version = self.ed.version
        fut = self.analysis_pool.submit(fn, fork, *args)
        job = {"state": "running", "version": version, "future": fut, "result": None,
               "error": None}  # fmt: skip
        self._analysis[name] = job
        if fut.done():
            self._finish_analysis(name, job)
        return {"ok": True, "state": job["state"]}

    def _finish_analysis(self, name: str, job: dict[str, Any]) -> None:
        try:
            job["result"] = job["future"].result()
            job["state"] = "done"
        except X.EditError as e:
            job["state"], job["error"] = "error", str(e)
        except Exception as e:  # a numerical failure is said, never raised into the window
            job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"

    def lyapunov(self, x0: Any = None, T: Any = 50.0) -> dict[str, Any]:
        """Estimate the largest Lyapunov exponent from ``x0`` (default: the first tracer, or
        the window's centre) on a fork, off the GUI thread (``flow_probe``)."""
        sid, r = self._system()
        if sid is None:
            return _refused("type a system first")
        try:
            t = _parse(T, "T")
            if x0 is None:
                pts = [e for e in X.rows_of(self._doc()) if e["kind"] == "point"]
                spec = self._doc().object(pts[0]["object"]) if pts else None
                x0 = list(spec.params["x0"]) if spec else [float(a) + 0.1 for a in r.centre]
            x0 = [_parse(a, "x0") for a in x0]
        except (ValueError, TypeError) as e:
            return _refused(str(e))
        return self._start("lyapunov", X.flow_probe, x0, t, 16, None, sid, 0.0)

    def sweep(self, name: Any, lo: Any, hi: Any, n: Any = 17) -> dict[str, Any]:
        """A bifurcation sweep of one variable on a fork, off the GUI thread."""
        sid, _ = self._system()
        if sid is None:
            return _refused("type a system first")
        try:
            a, b, k = _parse(lo, "from"), _parse(hi, "to"), int(_parse(n, "n"))
        except ValueError as e:
            return _refused(str(e))
        return self._start("sweep", X.sweep_parameter, str(name)[:40], a, b, k, "equilibria",
                           None, 9, None, sid, 0.0)  # fmt: skip

    def analysis(self) -> dict[str, dict[str, Any]]:
        """The Analysis tab's estimates: state, result or error, and ``stale`` when the
        document changed since (the numbers are for the scene as it was)."""
        return {k: {"state": j["state"], "result": j["result"], "error": j["error"],
                    "stale": j["version"] != self.ed.version}
                for k, j in self._analysis.items()}  # fmt: skip

    # ------------------------------------------------------------ the AI notes tab, history

    def ai_notes(self) -> list[dict[str, Any]]:
        """What an attached AI changed, oldest first ("who changed what", the AI's lines)."""
        return [a for a in self.ws.activity if a.get("who") == "ai"]

    def history_names(self) -> list[str]:
        """The edits in readable words, oldest first: yours as named, the AI's as
        ``AI: ...``."""
        return [
            ("AI: " if a.get("who") == "ai" else "") + str(a.get("what", ""))
            for a in self.ws.activity
        ]
