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

import itertools
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core import io
from phasor.core.pycode import BlockError, find_tweaks, insert_code, read_tweaks, with_tweaks
from phasor.engine import Engine
from phasor.workspace import ToolError, Workspace

# What the narration flags of Milestone 22 mean, said plainly in the Narration panel.
_FLAG_TEXT = {
    "silence": "the recording is (nearly) silent",
    "constant_noise": "a steady noise covers the pauses",
    "low_snr": "speech is hard to tell from the noise",
    "clipping": "the recording clips (too loud)",
    "music": "music or a tone under the speech",
    "no_pauses": "no pauses found between sentences",
    "short": "the recording is very short",
    "cut": "only the first part was looked at",
    "unreliable": "the sentences could not be found reliably",
    "ambiguous_timing": "two readings of the timing disagree: nothing moves",
}
_NOTE_NAMES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
MAX_TYPED_NOTES = 2000


def parse_notes(text: str) -> list[tuple[float, float, float, float]]:
    """Notes typed in the Sound and MIDI panel, one per line or separated by ``,`` or
    ``;``: ``pitch start length [velocity]``. A pitch is a MIDI number (``60``) or a name
    (``C4``, ``F#3``, ``Bb2``; C4 is 60). Times in seconds from the notes' start."""
    import re

    out = []
    for k, item in enumerate(t for t in re.split(r"[,;\n]+", text) if t.strip()):
        parts = item.split()
        if not 3 <= len(parts) <= 4:
            raise ToolError(f"note {k + 1} ({item.strip()!r}): write pitch start length "
                            "[velocity], e.g. C4 0 0.5")  # fmt: skip
        m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", parts[0])
        try:
            if m:
                pitch = (12 * (int(m.group(3)) + 1) + _NOTE_NAMES[m.group(1).lower()]
                         + {"#": 1, "b": -1, "": 0}[m.group(2)])  # fmt: skip
            else:
                pitch = float(parts[0])
            start, length = float(parts[1]), float(parts[2])
            vel = float(parts[3]) if len(parts) == 4 else 100.0
        except ValueError:
            raise ToolError(f"note {k + 1} ({item.strip()!r}): a pitch (60 or C4), then "
                            "seconds") from None  # fmt: skip
        out.append((float(pitch), start, length, vel))
        if len(out) > MAX_TYPED_NOTES:
            raise ToolError(f"more than {MAX_TYPED_NOTES} typed notes: use a .mid file")
    if not out:
        raise ToolError("no notes: write pitch start length, e.g. C4 0 0.5, E4 0.5 0.5")
    return out


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def _with_line_ends(text: str, disk: str) -> str:
    """``text`` ("\\n" line ends) written with ``disk``'s: each line ``disk`` already has
    keeps its own ending (matched with ``difflib``, so a mixed file stays mixed and a CRLF
    file stays CRLF), a new line takes the ending most of ``disk`` uses."""
    import difflib

    if "\r\n" not in disk:
        return text
    old = disk.split("\n")[:-1]  # the lines that end in "\n" (a CR before it: CRLF)
    crlf = [piece.endswith("\r") for piece in old]
    plain = [piece[:-1] if c else piece for piece, c in zip(old, crlf, strict=True)]
    most = sum(crlf) * 2 >= len(old)
    *body, tail = text.split("\n")  # the last piece has no line end
    ends: list[bool] = []
    sm = difflib.SequenceMatcher(None, plain, body, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ends.extend(crlf[i1:i2] if tag == "equal" else [most] * (j2 - j1))
    return "".join(line + ("\r\n" if cr else "\n")
                   for line, cr in zip(body, ends, strict=True)) + tail  # fmt: skip


class RunCancelled(BaseException):
    """Raised inside a script's thread to stop it (Cancel, or past its time). A
    ``BaseException``, so a script's ``except Exception`` does not swallow it."""


RUN_TIMEOUT = 600.0  # seconds a Run may take before it is stopped (PHASOR_RUN_TIMEOUT)


def run_timeout() -> float:
    import os

    try:
        return max(1.0, float(os.environ.get("PHASOR_RUN_TIMEOUT", RUN_TIMEOUT)))
    except ValueError:
        return RUN_TIMEOUT


class RunTask:
    """A script run on a worker thread (``Workspace._load_script``: it builds a new engine
    and touches nothing shared), so the window and an attached AI stay alive while it runs.
    ``cancel`` stops it by raising ``RunCancelled`` in its thread at its next Python
    instruction (``PyThreadState_SetAsyncExc``): a pure-Python loop stops at once; a long
    call into C (NumPy, a GL draw) stops when it returns. A run stuck in C forever cannot be
    stopped in-process: the editor then lets it go and keeps the last good scene."""

    def __init__(self, ws: Workspace, path: Path, *, restore: str | None,
                 written: str | None = None, kind: str = "run") -> None:  # fmt: skip
        import time

        self.ws, self.path, self.kind = ws, Path(path), kind
        self.restore, self.written = restore, written
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()
        self.started = time.monotonic()
        self.timeout = run_timeout()
        self.cancelled = False
        self.timed_out = False
        self._thread = threading.Thread(target=self._go, name="phasor-run", daemon=True)

    def start(self) -> RunTask:
        self._thread.start()
        return self

    def _go(self) -> None:
        try:
            self.result = self.ws._load_script(self.path, True)
        except BaseException as e:  # the script's mistake, or RunCancelled
            self.error = e
        finally:
            self.done.set()

    def past_time(self) -> bool:
        import time

        late = not self.done.is_set() and time.monotonic() - self.started > self.timeout
        if late:
            self.timed_out = True
        return late

    def cancel(self) -> None:
        import ctypes

        if self.done.is_set() or self.cancelled:
            return
        self.cancelled = True
        tid = self._thread.ident
        if tid is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid),
                                                       ctypes.py_object(RunCancelled))  # fmt: skip


def _outside_block(text: str) -> str:
    """A script without its tweaks block: the code a Run builds the scene from (blank lines
    and trailing spaces left out, so the blank line written before a new block is not a
    change of code)."""
    found = find_tweaks(text, strict=False)
    rest = text if found is None else text[: found[0]] + text[found[1] :]
    return "\n".join(line.rstrip() for line in rest.splitlines() if line.strip())


def _error_line(e: BaseException, path: Path) -> int | None:
    """The line of ``path`` where running it failed: the deepest frame in that file, in the
    error or the error it was raised from; a syntax error's own line."""
    want = str(Path(path).resolve())
    seen: set[int] = set()
    err: BaseException | None = e
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if (
            isinstance(err, SyntaxError)
            and err.filename
            and Path(err.filename).resolve() == Path(want)
        ):
            return err.lineno
        line, tb = None, err.__traceback__
        while tb is not None:
            name = tb.tb_frame.f_code.co_filename
            if name == want or (name.endswith(".py") and str(Path(name).resolve()) == want):
                line = tb.tb_lineno
            tb = tb.tb_next
        if line is not None:
            return line
        err = err.__cause__ or err.__context__
    return None


def _names(line: str, oid: str) -> bool:
    """Whether a line of code names object ``oid`` in a string: ``'pole'``, or one of its
    paths (``'pole.r'``, as a tweak writes it)."""
    import re

    return re.search(rf"['\"]{re.escape(oid)}(?:\.[A-Za-z0-9_]+)*['\"]", line) is not None


def _class_name(type_name: str) -> str:
    """The class ``from phasor import ...`` gives for a shipped type (``video`` -> Video)."""
    from phasor.core.objects import registered_types

    cls = registered_types().get(type_name)
    if cls is None:
        raise ToolError(f"unknown object type {type_name!r}")
    return cls.__name__


def _same_file(a: Path, b: Path) -> bool:
    import filecmp

    try:
        return filecmp.cmp(a, b, shallow=False)  # sizes first, then chunk by chunk
    except OSError:
        return False


def parse_value(text: str) -> Any:
    """What a person types into a field: JSON (``0.3``, ``[1, 2, 0]``, ``true``,
    ``{"expr": "t / 2"}``), else the text itself (``sin(x)``, ``D4``)."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def open_workspace(path: str | None, root: str | None = None) -> Workspace:
    """A workspace with ``path`` open, or a new scene. Every way the desktop app opens a file
    (the Open dialog, recent scenes, examples, a path on the command line) comes here.

    Code runs only on the person's request: a ``.py`` the person opens is their own script
    and runs (with the modules it names); a ``.phasor.json`` opens without running the
    modules it names (``meta.modules``): its own types draw as placeholders and lint
    ``missing_type`` says why, until the person calls ``Editor.trust_scene_code``. The
    workspace itself never allows scripts, so an attached AI cannot open or run code."""
    ws = Workspace(root or (Path(path).parent if path else Path.cwd()))
    if path:
        p = Path(path).resolve()
        ws._open(str(p), allow_scripts=p.suffix == ".py")
    else:
        ws.scene_new("untitled")
    return ws


class Editor:
    def __init__(self, workspace: Workspace, lock: threading.RLock | None = None) -> None:
        self.ws = workspace
        self.lock = lock or threading.RLock()
        self.t = 0.0
        self.selected: str | None = None
        self.history: list[dict[str, Any]] = []  # what the person did here, newest last
        self._seen: tuple[int, int] = (0, self.version)
        self.preview: dict[str, float] = {}  # slider values shown but not yet in the document
        self._player: tuple[tuple, Any] | None = None  # ((generation, version), Player)
        self._frame: tuple[tuple, np.ndarray] | None = None
        self._lint: tuple[tuple[int, int], list[dict[str, Any]]] | None = None
        self.saved_version = self.version  # what is on disk (unsaved edits: ``dirty``)
        # The script view (Milestone 25, the code-first window). For a script scene
        # ``buffer`` is the text in the editor: the script, with its tweaks block kept in
        # step with the live scene (ADR 0004); for a document it is the code generated from
        # it. ``error`` is the last Run's mistake ({"line", "message"}); ``conflict`` the
        # file's new text when it changed outside while the buffer had unsaved edits.
        self.generation = 0  # counts the scenes a Run or a reload put in place
        self.buffer = ""  # always "\n" line ends (as the editor widget gives text)
        self._nl = "\n"  # the script file's own line end, kept when it is written
        self._block = ""  # the tweaks block as last written into the buffer
        self.block_edited = False  # the person typed in the block: Run to use it
        self.error: dict[str, Any] | None = None
        self.conflict: dict[str, Any] | None = None
        # The tweaks block's markers cannot be read for certain ({"line", "message"}):
        # nothing is written into it until the person fixes them (the tweaks stay in the
        # scene); ``unwritten`` says the scene holds tweaks the text does not show yet.
        self.block_problem: dict[str, Any] | None = None
        self.unwritten = False
        self.running: RunTask | None = None  # a Run or a reload on its worker thread
        # Code the app added to a script (footage, sound, notes, narration, the snap): undo
        # puts the text before it back and runs it, once the scene's own undo is used up.
        self._script_undo: list[str] = []
        self._script_redo: list[str] = []
        self.job: str | None = None  # the Draft preview's render job (the status bar shows it)
        self._ran = ""
        self._load_buffer()

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
        """True once after anything (here or an attached AI) changed the scene. A script's
        tweaks block follows (``_sync_block``)."""
        v = (self.generation, self.version)
        if v != self._seen:
            self._seen = v
            if self.selected and self.selected not in self._ids():
                self.selected = None
            self._sync_block()
            return True
        return False

    def _ids(self) -> set[str]:
        return {o.id for o in self.engine.document.objects}

    def _edit(self, patch: Any, what: str) -> dict[str, Any]:
        if self.running is not None:
            return self._refused("the script is running: wait for it, or cancel it")
        with self.lock:
            r = self.ws.scene_edit(patch)
        if r["ok"]:
            entry = {"what": what, "version": self.version, "t": self.t}
            self.history.append(entry)
            self.ws.activity.append({"who": "you", **entry})
            self._sync_block()
        elif self.is_script and any(e.get("code") == "scripted_scene" for e in r["errors"]):
            msg = ("this changes the scene's structure or timing, which the script writes: "
                   "change it in the code (markers: s.marker(...); a look: Scene(look=...)); "
                   "only parameters, styles and layout rules are tweaks")  # fmt: skip
            r = {**r, "errors": [{**r["errors"][0], "message": msg}, *r["errors"][1:]]}
        return r

    def activity(self) -> list[dict[str, Any]]:
        """Everything that changed the scene, oldest first: ``who`` is "you" or "ai"."""
        return list(self.ws.activity)

    @staticmethod
    def _refused(message: str) -> dict[str, Any]:
        return {"ok": False, "version": -1, "errors": [{"code": "refused", "message": message}]}

    def undo(self) -> dict[str, Any]:
        """Undo the last edit; on a script, when the scene's own undo is used up, the code
        the app added last (a panel's footage, sound, notes, narration or snap) is taken back
        out and the script runs again."""
        with self.lock:
            r = self.ws.undo()
        if not r.get("undone") and self.is_script and self._script_undo and not self.running:
            before = self._script_undo.pop()
            now = self.buffer
            got = self.run_code(before, keep_history=True)
            if got["ok"]:
                self._script_redo.append(now)
            else:
                self._script_undo.append(before)
            return {"undone": got["ok"], "script": True, **got}
        self._sync_block()
        return r

    def redo(self) -> dict[str, Any]:
        with self.lock:
            r = self.ws.redo()
        if not r.get("redone") and self.is_script and self._script_redo and not self.running:
            after = self._script_redo.pop()
            now = self.buffer
            got = self.run_code(after, keep_history=True)
            if got["ok"]:
                self._script_undo.append(now)
            else:
                self._script_redo.append(after)
            return {"redone": got["ok"], "script": True, **got}
        self._sync_block()
        return r

    def save(self, path: str | None = None) -> dict[str, Any]:
        """Ctrl+S. A script scene: when its code changed since it last ran, it runs first
        ("Run on save"; a mistake keeps the last good scene and saves nothing); else the
        buffer, tweaks block and all, is written as it is. Nothing is written when the file
        changed on disk since it was last run or saved here (``conflict``: the window asks
        which version to keep), or when the scene holds tweaks the text cannot show because
        the block's markers cannot be read (``block_problem``)."""
        from phasor.workspace.tools import ScriptChanged

        if self.is_script and path is None:
            if self.running is not None:
                return self._refused("the script is running: wait for it, or cancel it")
            with self.lock:
                try:
                    if self._needs_run():
                        r = self.run_code(self.buffer)
                        if not r["ok"]:
                            code = "file_changed" if r.get("conflict") else "script_error"
                            return {"ok": False, "errors": [{"code": code,
                                                             "message": r["error"]}]}  # fmt: skip
                    elif self.unwritten:
                        problem = self.block_problem or {"message": "", "line": None}
                        msg = (
                            f"not saved: {problem['message']}; the scene holds tweaks the "
                            "script cannot show until its tweaks block is fixed"
                        )
                        return {"ok": False, "errors": [{"code": "tweaks_block", "message": msg,
                                                         "line": problem["line"]}]}  # fmt: skip
                    else:
                        self.ws._write_script(self._native(self.buffer))
                except ScriptChanged as e:
                    return self._changed_on_disk(e)
                except ToolError as e:  # read-only, a full disk: nothing was written
                    return self._refused(str(e))
                self.ws.path.with_name(self.ws.path.stem + ".overrides.json").unlink(
                    missing_ok=True
                )  # the overrides file of an old scene: in the block now
                self.saved_version = self.version
                return {"saved": str(self.ws.path), "version": self.version}
        with self.lock:
            r = self.ws.scene_save(path)
            self.saved_version = self.version
            return r

    def _changed_on_disk(self, e: Any) -> dict[str, Any]:
        """A write was refused because the file changed outside: it becomes a ``conflict``
        (the window asks which version to keep); nothing was written."""
        if e.disk is not None:
            self.conflict = {"disk": e.disk, "stamp": e.stamp}
        return {
            "ok": False,
            "conflict": True,
            "errors": [{"code": "file_changed", "message": str(e)}],
        }

    @property
    def path(self) -> Path | None:
        """Where the scene is saved (None until the first save)."""
        return self.ws.path

    @property
    def dirty(self) -> bool:
        """Whether there are edits not saved yet (yours or an attached AI's). A script scene
        is dirty when the text in the editor is not the file's."""
        if self.is_script:
            disk = self.ws.script_disk
            return disk is None or self.buffer != _lf(disk[0]) or self.conflict is not None
        return self.version != self.saved_version

    def save_as(self, path: str | Path) -> dict[str, Any]:
        """Save to ``path`` anywhere the person chooses (a file dialog): the workspace
        folder moves there, so an attached AI follows the scene. ``.phasor.json`` is added
        when the name lacks it."""
        p = Path(path).resolve()
        if self.is_script:  # a script stays a script: the text is saved there and runs
            if p.suffix != ".py":
                p = p.with_name(p.name.removesuffix(".phasor.json").removesuffix(".json")
                                + ".py")  # fmt: skip
            return self._script_save_as(p)
        if not p.name.endswith(".phasor.json"):
            p = p.with_name(p.name.removesuffix(".json") + ".phasor.json")
        with self.lock:
            if p.parent != self.ws.root and self.ws.root not in p.parents:
                self.ws.root = p.parent
            return self.save(str(p))

    def _script_save_as(self, p: Path) -> dict[str, Any]:
        from phasor.workspace.tools import write_text

        with self.lock:
            text = self._native(self.buffer)
            if p.parent != Path(self.engine.base_dir).resolve():
                self.engine._bring_files(p.parent)  # pictures and sounds go with it
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                write_text(p, text)
            except ToolError as e:
                return self._refused(str(e))
            if p.parent != self.ws.root and self.ws.root not in p.parents:
                self.ws.root = p.parent
            r = self._finish_run(self._start_task(p, restore=None, kind="save_as"), wait=True)
            if not r["ok"]:
                return {"ok": False, "errors": [{"code": "script_error", "message": r["error"]}]}
            self.saved_version = self.version
            return {"saved": str(p), "version": self.version}

    # ------------------------------------------------------------ time

    def timeline(self) -> dict[str, Any]:
        with self.lock:
            return self.engine.timeline()

    def set_time(self, t: float) -> float:
        self.t = float(min(max(t, 0.0), max(self.engine.duration, 0.0)))
        return self.t

    def move_marker(self, name: str, t: float) -> dict[str, Any]:
        """Drag a marker to a new time (animations anchored to it follow)."""
        if name not in self.timeline()["markers"]:
            return self._refused(f"no marker {name!r}")
        return self._edit({"op": "set_marker", "name": name, "t": max(0.0, float(t))},
                          f"marker {name} -> {t:.3g}s")  # fmt: skip

    def retime_segment(self, path: str, index: int, t0: float, t1: float) -> dict[str, Any]:
        """Drag one animation span to new start and end times (the value it moves to stays)."""
        segs = self.engine.document.timeline.tracks.get(path)
        if segs is None or not 0 <= index < len(segs):
            return self._refused(f"{path} has no span number {index}")
        if t1 <= t0 or t0 < 0:
            return self._refused("a span needs 0 <= start < end")
        from phasor.core.schema import Segment

        old = segs[index]
        moved = Segment(t0=float(t0), t1=float(t1), to=old.to, ease=old.ease, verb=old.verb)
        new = [moved if i == index else seg for i, seg in enumerate(segs)]
        ops: list[dict[str, Any]] = [{"op": "clear_track", "path": path}]
        ops += [{"op": "add_segment", "path": path, "segment": seg.model_dump(mode="json")}
                for seg in new]  # fmt: skip
        return self._edit(ops, f"{path} span {index} -> {t0:.3g}..{t1:.3g}s")

    def waveform(self, audio: str, bins: int = 200,
                 span: tuple[float, float] | list[float] | None = None) -> list[float]:  # fmt: skip
        """Loudness of a narration audio file in ``bins`` steps (0 to 1, of full scale); an
        empty list when the file is missing or not 16-bit WAV. ``span``: only seconds (a, b)
        of it (a block of a recording, M22: any sound file ``domains.audio`` reads; only that
        part is read)."""
        from phasor.time.narration import parse_wav

        if span is not None:
            from phasor.domains import audio as sound

            try:
                x, _ = sound.samples(self.engine.base_dir, audio, (float(span[0]), float(span[1])))
            except Exception:  # a missing or hostile file: no waveform, never a crash
                return []
            if x.size == 0:
                return []
            edges = np.linspace(0, x.size, bins + 1).astype(int)
            return [min(1.0, float(np.abs(x[a:b]).max())) if b > a else 0.0
                    for a, b in itertools.pairwise(edges)]  # fmt: skip

        path = Path(self.engine.base_dir) / audio
        try:  # the one WAV header walk: a hostile file is an empty list, never a crash
            info = parse_wav(path)
            if info.width != 2:
                return []
            ch = info.channels
            with open(path, "rb") as f:
                f.seek(info.offset)
                data = np.frombuffer(f.read(info.frames * ch * 2), dtype="<i2")
        except (OSError, ValueError):
            return []
        if ch > 1:
            data = data[::ch]
        if data.size == 0:
            return []
        edges = np.linspace(0, data.size, bins + 1).astype(int)
        return [float(np.abs(data[a:b]).max()) / 32768.0 if b > a else 0.0
                for a, b in itertools.pairwise(edges)]  # fmt: skip

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
        key = (self.generation, self.version, round(self.t, 6), tuple(size),
               tuple(sorted(self.preview.items())))  # fmt: skip
        if self._frame is None or self._frame[0] != key:
            with self.lock:
                if self.preview:
                    img = self._preview_player().frame(self.t, size)
                else:
                    img = np.ascontiguousarray(self.engine.still(self.t, size)[..., :3])
            self._frame = (key, img)
        return self._frame[1]

    # ------------------------------------------------------------ sliders (the scene's controls)

    def _preview_player(self):
        from phasor.interactive import Player

        if self._player is None or self._player[0] != (self.generation, self.version):
            self._player = ((self.generation, self.version), Player(self.engine))
        player = self._player[1]
        for key, v in self.preview.items():
            player.set(key, v)
        return player

    def controls(self) -> list[dict[str, Any]]:
        """The scene's sliders with their current values (a preview, if one is set)."""
        from phasor.core.patch import param_value

        out = []
        e = self.engine
        for c in e.document.controls:
            try:
                cur = param_value(e.document, c.path, e.code_scope)
                value = float(cur if c.component is None else cur[c.component])
            except (KeyError, IndexError, TypeError, ValueError):
                continue  # its object's type is missing here: lint says why

            out.append({"key": c.key, "label": c.label or c.key, "min": c.min, "max": c.max,
                        "step": c.step, "value": self.preview.get(c.key, value)})  # fmt: skip
        return out

    def set_control(self, key: str, value: float) -> float:
        """Move a slider: the viewport shows it, the scene is not edited (until
        ``apply_controls``). Returns the value used (clamped, on the step)."""
        from phasor.interactive import snap

        ctl = next((c for c in self.engine.document.controls if c.key == key), None)
        if ctl is None or all(c["key"] != key for c in self.controls()):
            raise ToolError(f"no slider {key!r}")
        self.preview[key] = snap(ctl, value)
        return self.preview[key]

    def reset_controls(self) -> None:
        self.preview = {}

    def apply_controls(self) -> dict[str, Any]:
        """Make the slider values part of the scene: one undoable edit."""
        if not self.preview:
            return {"ok": True, "version": self.version, "errors": []}
        from phasor.core.patch import param_value

        doc, scope = self.engine.document, self.engine.code_scope
        by_path: dict[str, Any] = {}
        for key, v in self.preview.items():
            ctl = next(c for c in doc.controls if c.key == key)
            cur = by_path[ctl.path] if ctl.path in by_path else param_value(doc, ctl.path, scope)
            if ctl.component is None:
                by_path[ctl.path] = v
            else:
                new = list(cur)
                new[ctl.component] = v
                by_path[ctl.path] = new
        ops = [{"op": "set_param", "path": p, "value": v} for p, v in by_path.items()]
        what = "sliders: " + ", ".join(f"{k} = {v:g}" for k, v in self.preview.items())
        r = self._edit(ops, what)
        if r["ok"]:
            self.preview = {}
        return r

    # ------------------------------------------------------------ readouts (hover)

    def readout(self, u: float, v: float, size: tuple[int, int],
                hit: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
        """What is under the pointer: its plane coordinates and, for a function plot or a
        fractal, the value there (f(z), |f|, arg f; the escape count of a Julia/Mandelbrot
        point). ``hit`` (object ids under the point, topmost first) defaults to a hit test."""
        import cmath

        x, y, _ = self.screen_to_world(u, v, size)
        if hit is None:
            ox, oy = self._to_output(u, v, size)
            with self.lock:
                hit = self.engine.hit_test(ox, oy, self.t)
        out: dict[str, Any] = {"x": x, "y": y, "object": hit[0] if hit else None,
                               "text": f"x = {x:.4g}, y = {y:.4g}"}  # fmt: skip
        if not hit:
            return out
        spec = self.engine.document.object(hit[0])
        st = self.engine.state_at(self.t)[hit[0]]
        if spec is None:
            return out
        p = st.params
        if spec.type == "domain_coloring":
            from phasor.domains.cx import _parse

            z = complex(x - p.pos[0], y - p.pos[1])
            try:
                f = complex(_parse(p.function.strip()).numpy()(np.array([z]))[0])
            except (ValueError, ZeroDivisionError, FloatingPointError):
                f = complex("nan")
            out.update(z=z, f=f, abs=abs(f), arg=cmath.phase(f) if f == f else float("nan"))
            out["text"] = (f"z = {z.real:.4g}{z.imag:+.4g}i   f(z) = {f.real:.4g}{f.imag:+.4g}i   "
                           f"|f| = {abs(f):.4g}   arg f = {out['arg']:.4g}")  # fmt: skip
        elif spec.type == "fractal":
            zc = complex(x - p.pos[0], y - p.pos[1])
            c, z = (complex(*p.c), zc) if p.kind == "julia" else (zc, 0j)
            n = None
            for k in range(p.max_iter):
                z = z * z + c
                if abs(z) > 256:
                    n = k
                    break
            out.update(z=zc, escape=n)
            out["text"] = f"z = {zc.real:.4g}{zc.imag:+.4g}i   " + (
                f"escapes after {n} steps" if n is not None else "stays bounded"
            )
        return out

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

    def hit_ids(self, u: float, v: float, size: tuple[int, int]) -> list[str]:
        """Objects under a viewport point, topmost first (the selection is left alone)."""
        x, y = self._to_output(u, v, size)
        with self.lock:
            return list(self.engine.hit_test(x, y, self.t))

    def drag_camera(self, dx: float, dy: float, size: tuple[int, int],
                    pan: bool = False) -> dict[str, Any]:  # fmt: skip
        """A drag of the empty viewport by ``dx``, ``dy`` pixels: turns the 3D camera (0.4
        degrees a pixel); with ``pan``, or in 2D, slides the camera so the picture follows
        the mouse."""
        spec, cur = self._camera()
        if spec.kind == "orbit" and not pan:
            return self.orbit_camera(dx * 0.4, dy * 0.4)
        import math

        if spec.kind == "orbit":
            per_px = 2 * float(cur["distance"]) * math.tan(math.radians(float(cur["fov"])) / 2)
        else:
            per_px = float(cur["height"])
        per_px /= size[1]
        return self.pan_camera(-dx * per_px, dy * per_px)

    def select(self, object_id: str | None) -> None:
        if object_id is not None and object_id not in self._ids():
            raise ToolError(f"no object {object_id!r}")
        self.selected = object_id

    def screen_to_world(self, u: float, v: float, size: tuple[int, int], z: float = 0.0):
        """The world point under a viewport pixel, on the plane at height ``z`` (2D: z = 0)."""
        cam = self.engine.state_at(self.t).camera
        p = cam.from_screen([(u, v)], size[0], size[1], z)[0]
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

    # ------------------------------------------------------------ the camera

    def _camera(self) -> tuple[Any, dict[str, Any]]:
        from phasor.time.compile import CAMERA_PARAMS

        doc = self.engine.document
        spec = doc.camera(doc.timeline.camera)
        model = CAMERA_PARAMS[spec.kind]
        current = {
            k: f.get_default(call_default_factory=True) for k, f in model.model_fields.items()
        }
        return spec, {**current, **spec.params}

    def _set_camera(self, updates: dict[str, Any], what: str) -> dict[str, Any]:
        spec, _ = self._camera()
        tracks = self.engine.document.timeline.tracks
        busy = [k for k in updates if f"{spec.id}.{k}" in tracks]
        if busy:
            return self._refused(
                f"{spec.id}.{busy[0]} is animated by a track: scrub to a moment without the "
                "animation, or clear it, to move the camera by hand"
            )
        ops = [
            {"op": "set_param", "path": f"{spec.id}.{k}", "value": v} for k, v in updates.items()
        ]
        return self._edit(ops, f"camera {what}")

    def orbit_camera(self, d_azimuth: float, d_elevation: float) -> dict[str, Any]:
        """Turn the 3D camera around its target by degrees (dragging right turns the scene
        to the right). A 2D camera has nothing to turn."""
        spec, cur = self._camera()
        if spec.kind != "orbit":
            return self._refused("a 2D camera does not turn; use the orbit camera for 3D")
        az = float(cur["azimuth"]) - d_azimuth
        el = min(max(float(cur["elevation"]) - d_elevation, -89.0), 89.0)
        return self._set_camera(
            {"azimuth": az, "elevation": el}, f"orbit to {az:.1f}, {el:.1f} degrees"
        )

    def zoom_camera(self, factor: float) -> dict[str, Any]:
        """Move in (factor > 1) or out (< 1): the orbit camera's distance, the 2D camera's
        height."""
        spec, cur = self._camera()
        key = "distance" if spec.kind == "orbit" else "height"
        return self._set_camera({key: float(cur[key]) / max(factor, 1e-3)}, f"zoom x{factor:.2f}")

    def pan_camera(self, dx: float, dy: float) -> dict[str, Any]:
        """Slide the camera ``dx`` along the screen's right and ``dy`` along its up (world
        units)."""
        spec, cur = self._camera()
        cam = self.engine.state_at(self.t).camera
        key = "target" if spec.kind == "orbit" else "center"
        move = cam.right * dx + cam.up * dy
        new = [round(float(a + b), 9) for a, b in zip(cur[key], move, strict=True)]
        return self._set_camera({key: new}, f"pan to {new[0]:.2f}, {new[1]:.2f}")

    # ------------------------------------------------------------ the code view

    # ------------------------------------------------------------ the script (Milestone 25)

    @property
    def is_script(self) -> bool:
        """The open scene is a script the person edits (the code-first window); a
        ``.phasor.json`` document shows code generated from it instead."""
        p = self.ws.path
        return p is not None and p.suffix == ".py" and self.ws.script_disk is not None

    def _load_buffer(self) -> None:
        """Put the scene's code in the editor: the script as it is on disk, its tweaks block
        in step with the live scene; or the code generated from a document."""
        self.error = None
        self.conflict = None
        self.block_edited = False
        self.block_problem = None
        self.unwritten = False
        self._ran = ""
        if self.is_script:
            assert self.ws.script_disk is not None
            self.buffer = self._ran = _lf(self.ws.script_disk[0])
            self._block = self._block_of(self.buffer)
            self._sync_block()
        else:
            try:
                self.buffer = self.code()["text"]
            except (ToolError, ValueError):
                self.buffer = ""

    def set_buffer(self, text: str) -> None:
        """The person typed: the editor's text (not run, not saved, until Run or Save)."""
        self.buffer = _lf(text)

    def _native(self, text: str) -> str:
        """``text`` (with "\\n" line ends, as the editor holds it) with the file's own line
        ends: each line the file already has keeps its own (a mixed file stays mixed), a new
        line takes the ending most of the file uses."""
        disk = self.ws.script_disk[0] if self.ws.script_disk else ""
        return _with_line_ends(_lf(text), disk)

    @staticmethod
    def _block_of(text: str) -> str:
        """The tweaks block's text in ``text`` ("" when it has none, or its markers cannot be
        read)."""
        found = find_tweaks(text, strict=False)
        return text[found[0] : found[1]] if found else ""

    def _sync_block(self) -> None:
        """Write the live scene's tweaks (``engine.overrides``) into the buffer's tweaks
        block, so the script always says what the scene holds. The code outside the block
        is never touched. Not written: a block the person edited by hand and has not run
        (``block_edited``: Run takes it as typed), and a block whose markers cannot be read
        for certain (``block_problem``: a begin without an end, two blocks, the code not
        readable yet; the tweaks stay in the scene, ``unwritten``)."""
        if not self.is_script:
            return
        try:
            overrides = self.engine.overrides
        except ToolError:
            return
        if self._block_of(self.buffer) != self._block:
            self.block_edited = True
            return
        if io.dumps(read_tweaks(self.buffer)) == io.dumps(overrides):
            # the block already says what the scene holds: left as the person wrote it
            self.block_problem, self.unwritten, self.block_edited = None, False, False
            try:
                find_tweaks(self.buffer)
            except BlockError as e:
                self.block_problem = {"line": e.line, "message": str(e)}
            return
        try:
            new = with_tweaks(self.buffer, overrides, self.ws.script_var)
        except BlockError as e:
            self.block_problem = {"line": e.line, "message": str(e)}
            self.unwritten = overrides != read_tweaks(self.buffer)
            return
        self.block_problem = None
        self.unwritten = False
        self._block = self._block_of(new)
        self.buffer = new
        self.block_edited = False

    def _needs_run(self) -> bool:
        """The buffer's code is not the code the scene was built from, or its block is not
        the live scene's tweaks."""
        if _outside_block(self.buffer) != _outside_block(self._ran):
            return True
        if self.block_problem is not None:  # the block cannot be read: compare it as text
            return self.buffer != self._ran and not self.unwritten
        try:
            canon = with_tweaks(self.buffer, self.engine.overrides, self.ws.script_var)
        except BlockError:
            return self.buffer != self._ran
        return canon != self.buffer

    def code(self) -> dict[str, Any]:
        """The scene as Python: the script in the editor (``buffer``), or one generated from
        the document. ``path`` is where running edited code keeps it."""
        from phasor.core.ids import file_stem
        from phasor.core.pycode import document_to_python

        p = self.ws.path
        if self.is_script:
            assert p is not None
            return {"text": self.buffer, "is_script": True, "path": str(p)}
        stem = (p.name.removesuffix(".phasor.json") if p is not None
                else file_stem(self.engine.document.meta.title))  # fmt: skip
        return {"text": document_to_python(self.engine.document), "is_script": False,
                "path": str(self.ws.root / f"{stem}.scene.py")}  # fmt: skip

    def run(self, *, wait: bool = True) -> dict[str, Any]:
        """Run (Ctrl+Enter): the editor's text becomes the scene."""
        return self.run_code(self.buffer, wait=wait)

    def run_code(self, text: str, *, wait: bool = True, keep_history: bool = False) -> dict:
        """Run code as the scene (the person at the keyboard may: it is their code). The
        code is saved at ``path`` (the script; for a document, a new ``name.scene.py``
        beside it, which then is the scene) and runs on a worker thread, so the window stays
        alive; ``wait=False`` returns at once (``running``) and ``poll_run`` finishes it. A
        mistake reports its line (``error``), keeps the last good scene, and puts the file
        back as it was; the editor keeps the text so it can be fixed. Refused, writing
        nothing, when the script changed on disk since it last ran or was saved here (a
        ``conflict``). Running replaces the open scene (its undo history starts again)."""
        import ast

        from phasor.workspace.tools import ScriptChanged, write_text

        if self.running is not None:
            return {"ok": False, "error": "the script is already running: wait for it, or "
                    "cancel it", "line": None}  # fmt: skip
        text = _lf(text)
        self.buffer = text
        try:
            ast.parse(text)
        except SyntaxError as e:
            self.error = {"line": e.lineno, "message": f"{e.msg}"}
            return {"ok": False, "error": f"line {e.lineno}: {e.msg}", "line": e.lineno}
        path = Path(self.code()["path"])
        with self.lock:
            if self.is_script:
                try:
                    old = self.ws._check_disk()
                except ScriptChanged as e:
                    self._changed_on_disk(e)
                    return {"ok": False, "error": str(e), "line": None, "conflict": True}
                native = self._native(text)
            else:
                if path.exists():  # never over a script that is already there
                    msg = (f"{path.name} exists already: open it (it is a script), or move it "
                           "away, then Run again")  # fmt: skip
                    self.error = {"line": None, "message": msg}
                    return {"ok": False, "error": msg, "line": None}
                old, native = "", text
            try:
                write_text(path, native)
            except ToolError as e:
                self.error = {"line": None, "message": str(e)}
                return {"ok": False, "error": str(e), "line": None}
            task = self._start_task(path, restore=old, written=native,
                                    kind="undo" if keep_history else "run")  # fmt: skip
        if not wait:
            return {"ok": None, "running": True}
        return self._finish_run(task, wait=True)

    def _start_task(self, path: Path, *, restore: str | None, written: str | None = None,
                    kind: str = "run") -> RunTask:  # fmt: skip
        """Run the script at ``path`` on a worker thread (``RunTask``). ``restore``: what to
        write back if it fails ("" removes the file; None leaves it, as for an outside edit
        whose owner is the other editor), and only while the file still holds ``written``."""
        task = RunTask(self.ws, path, restore=restore, written=written, kind=kind)
        self.running = task
        self.ws.busy = "the script is running in the editor"
        return task.start()

    def poll_run(self) -> dict[str, Any] | None:
        """The window's timer: None while the run goes on (or when there is none); its
        result once it is done (``event``: run, reload, ...). A run past its time is
        cancelled (``RunTask.timeout``)."""
        task = self.running
        if task is None:
            return None
        if task.past_time():
            task.cancel()
        if not task.done.is_set():
            return None
        return self._finish_run(task, wait=False)

    def cancel_run(self) -> dict[str, Any] | None:
        """Stop a run that takes too long (Cancel): the last good scene stays."""
        task = self.running
        if task is None:
            return None
        task.cancel()
        task.done.wait(2.0)
        if not task.done.is_set():  # stuck in C code: let it go; nothing of it is used
            self.running = None
            self.ws.busy = None
            self.error = {"line": None, "message": "the run was cancelled (it is still "
                          "finishing in the background; nothing of it is used)"}  # fmt: skip
            return {"ok": False, "error": self.error["message"], "line": None}
        return self._finish_run(task, wait=False)

    def _finish_run(self, task: RunTask, *, wait: bool) -> dict[str, Any]:
        """Put a finished run in place (under the lock), or report its mistake."""
        from phasor.workspace.tools import ScriptChanged, read_text, write_text

        if wait:
            while not task.done.wait(0.05):
                if task.past_time():
                    task.cancel()
                    task.done.wait(2.0)
                    break
        if self.running is task:
            self.running = None
            self.ws.busy = None
        path = task.path
        if task.error is not None or task.result is None:
            e = task.error or RunCancelled("the run did not finish")
            line = _error_line(e, path)
            if task.restore is not None:
                try:  # put the file back, unless someone else changed it meanwhile
                    if task.written is None or read_text(path) == task.written:
                        if task.restore == "":
                            path.unlink(missing_ok=True)
                        else:
                            write_text(path, task.restore)
                except (OSError, ToolError):
                    pass
            where = f"line {line}: " if line else ""
            if isinstance(e, RunCancelled):
                msg = str(e) or (f"the run was stopped: it took longer than {task.timeout:g} s "
                                 "(PHASOR_RUN_TIMEOUT)" if task.timed_out
                                 else "the run was cancelled")  # fmt: skip
                line = None
            else:
                msg = f"{type(e).__name__}: {e}"
            self.error = {"line": line, "message": msg}
            return {"ok": False, "error": f"{where}{msg}", "line": line, "event": task.kind}
        run, ran_text, stamp = task.result
        with self.lock:
            out = self.ws._adopt_script(path, run, ran_text, stamp)
            self.generation += 1
            self._frame = self._lint = self._player = None
            self.preview = {}
            if self.selected not in self._ids():
                self.selected = None
            self.t = min(self.t, max(self.engine.duration, 0.0))
            self.saved_version = self.version
            text = _lf(ran_text)
            self._ran = text
            self.buffer = text
            self.error = None
            self.conflict = None
            self._block = self._block_of(text)
            self.block_edited = False
            if task.kind not in ("undo", "add"):  # typed code: the app's additions are history
                self._script_undo.clear()
                self._script_redo.clear()
            self._sync_block()  # the block in its canonical form (what the scene holds)
            if self.is_script and self.buffer != text and self.block_problem is None:
                try:
                    self.ws._write_script(self._native(self.buffer))
                    self._ran = self.buffer
                except ScriptChanged as e:  # changed while it ran: ask, write nothing
                    self._changed_on_disk(e)
        return {
            "ok": True,
            "version": self.version,
            "event": task.kind,
            "migrated": out.get("migrated"),
        }

    # ------------------------------------------------------------ the file on disk

    def poll_file(self, *, background: bool = False) -> dict[str, Any] | None:
        """Called by the window's timer: has the script changed on disk (another editor,
        Claude Code over MCP)? With no unsaved edits here it is reloaded and run (an error
        shows at its line and keeps the last good scene; ``background``: on the worker
        thread, finished by ``poll_run``). With unsaved edits it is a ``conflict``: both
        versions are kept and the person chooses (``resolve_conflict``); nothing is merged.
        None when nothing changed."""
        from phasor.workspace.tools import file_stamp, read_text

        if not self.is_script or self.conflict is not None or self.running is not None:
            return None
        path = self.ws.path
        assert path is not None and self.ws.script_disk is not None
        text0, stamp0 = self.ws.script_disk
        try:
            stamp = file_stamp(path)
        except OSError:
            return None  # gone or being replaced: looked at again on the next tick
        if stamp == stamp0:
            return None
        try:
            disk = read_text(path)
        except (OSError, UnicodeDecodeError):
            return None
        if disk == text0:  # touched, not changed
            self.ws.script_disk = (disk, stamp)
            return None
        if self.buffer == _lf(text0):  # nothing unsaved here: the file's version runs
            self.ws.activity.append({"who": "file", "what": "the script changed on disk",
                                     "version": self.version, "t": self.t})  # fmt: skip
            task = self._start_task(path, restore=None, kind="reload")
            if background:
                return {"event": "reloading"}
            r = self._finish_run(task, wait=True)
            if not r["ok"]:
                self.buffer = _lf(disk)  # the file's text, with its mistake shown
                self.ws.script_disk = (disk, stamp)
            return {**r, "event": "reloaded"}
        self.conflict = {"disk": disk, "stamp": stamp}
        return {"event": "conflict"}

    def resolve_conflict(self, keep: str) -> dict[str, Any]:
        """``keep="file"``: take the file's version (it runs); ``"mine"``: keep the editor's
        text (saving it later replaces the file's)."""
        if self.conflict is None:
            return {"ok": True}
        disk, stamp = self.conflict["disk"], self.conflict["stamp"]
        if keep == "file":
            self.conflict = None
            r = self._finish_run(self._start_task(self.ws.path, restore=None, kind="reload"),
                                 wait=True)  # fmt: skip
            if not r["ok"]:
                self.buffer = _lf(disk)
                self.ws.script_disk = (disk, stamp)
            return r
        if keep != "mine":
            raise ValueError(f"keep is 'file' or 'mine', not {keep!r}")
        self.conflict = None
        self.ws.script_disk = (disk, stamp)  # seen: the next save writes the editor's text
        return {"ok": True, "kept": "mine"}

    # ------------------------------------------------------------ objects and their lines

    def line_of(self, object_id: str | None = None) -> int | None:
        """The script line that made an object (selecting it in the preview shows it): the
        ``s.add`` that added it (or its parent, for a part), else the first line naming it."""
        oid = object_id or self.selected
        if oid is None or not self.is_script:
            return None
        lines = self.ws.script_lines
        probe = oid
        while probe:
            if probe in lines:
                return lines[probe]
            probe = probe.rpartition(".")[0]
        for k, line in enumerate(self.buffer.splitlines(), 1):
            if _names(line, oid):
                return k
        return None

    def object_at_line(self, line: int) -> str | None:
        """The object a script line makes or names (clicking a line selects it in the
        preview): the one ``s.add`` there added, else an id written on that line."""
        for oid, n in self.ws.script_lines.items():
            if n == line and oid in self._ids():
                return oid
        lines = self.buffer.splitlines()
        if not 1 <= line <= len(lines):
            return None
        text = lines[line - 1]
        named = [i for i in self._ids() if _names(text, i)]
        return max(named, key=len) if named else None

    def select_line(self, line: int) -> str | None:
        oid = self.object_at_line(line)
        if oid is not None:
            self.selected = oid
        return oid

    def problems(self) -> list[dict[str, Any]]:
        """The Problems tab: the script's error first (with its line), then lint, each with
        the script line to go to (``line``) when it names an object."""
        out: list[dict[str, Any]] = []
        if self.error is not None:
            out.append({"severity": "error", "code": "script_error", "ids": [], "times": [],
                        "message": self.error["message"], "line": self.error["line"]})  # fmt: skip
        if self.conflict is not None:
            out.append({"severity": "warning", "code": "file_changed", "ids": [], "times": [],
                        "message": "the script changed on disk while you had unsaved edits",
                        "line": None})  # fmt: skip
        if self.block_problem is not None:
            out.append({"severity": "error", "code": "tweaks_block", "ids": [], "times": [],
                        "message": f"{self.block_problem['message']}: phasor writes nothing "
                        "into the script until it is fixed" + (" (the scene holds tweaks the "
                        "script does not show yet)" if self.unwritten else ""),
                        "line": self.block_problem["line"]})  # fmt: skip
        if self.block_edited:
            out.append({"severity": "info", "code": "tweaks_edited", "ids": [], "times": [],
                        "message": "you edited the tweaks block: Run to use it (until then "
                        "new tweaks are not written into it)", "line": None})  # fmt: skip
        for issue in self.lint_items():
            ids = [i for i in issue.get("ids", []) if i in self._ids()]
            out.append({**issue, "line": self.line_of(ids[0]) if ids else None})
        return out

    # ------------------------------------------------------------ adding things

    def _add_code(self, lines: list[str], what: str) -> dict[str, Any]:
        """A structural change to a script scene is code (``scripted_scene``): ``lines`` go
        into the script before the tweaks block (and before a trailing ``__main__`` guard),
        and it runs. A mistake takes the lines back out and says why. Undo takes it back."""
        if self.running is not None:
            return self._refused("the script is running: wait for it, or cancel it")
        before = self.buffer
        try:
            added = insert_code(before, lines)
        except BlockError as e:
            return self._refused(f"{e}: fix the script's tweaks block markers first")
        r = self.run_code(added, keep_history=True)
        if not r["ok"]:
            self.buffer = before
            self.error = None
            code = "file_changed" if r.get("conflict") else "script_error"
            return {"ok": False, "errors": [{"code": code, "message": r["error"]}]}
        self._script_undo.append(before)
        self._script_redo.clear()
        entry = {"what": what, "version": self.version, "t": self.t}
        self.history.append(entry)
        self.ws.activity.append({"who": "you", **entry})
        return {"ok": True, "version": self.version, "errors": [], "code": lines}

    def _add_object(self, type_name: str, params: dict[str, Any], *, prefix: str, what: str,
                    code: Any = None, copied: str | None = None) -> dict[str, Any]:  # fmt: skip
        """Add an object: an ``add_object`` edit to a document (one undo step), a line of
        code in a script (``code(var, id)`` gives the lines, else ``s.add(Type(...))`` with
        its import). A file copied for it is removed again when it is not added."""
        oid = self._new_id(prefix)
        if self.is_script:
            var = self.ws.script_var
            if code is None:
                cls = _class_name(type_name)
                args = ", ".join([f"id={oid!r}", *(f"{k}={v!r}" for k, v in params.items())])
                lines = [f"from phasor import {cls}", f"{var}.add({cls}({args}))"]
            else:
                lines = code(var, oid)
            r = {**self._add_code(lines, what), "id": oid}
        else:
            op = {"op": "add_object", "object": {"id": oid, "type": type_name, "params": params}}
            r = {**self._edit(op, what), "id": oid}
        if not r.get("ok") and copied:
            Path(copied).unlink(missing_ok=True)
        return {**r, "copied": copied if r.get("ok") else None}

    def _new_id(self, prefix: str) -> str:
        from phasor.core.ids import name_from

        base = name_from(prefix, "media")
        doc = self.engine.document
        taken = self._ids() | set(getattr(doc, "retired_ids", ()) or ())
        oid, k = base, 2
        while oid in taken:
            oid, k = f"{base}_{k}", k + 1
        return oid

    def bring_file(self, path: str | Path) -> dict[str, Any]:
        """A file the person picked, as the scene names it: its path relative to the scene's
        folder. One from elsewhere is copied into ``media/`` there first (a scene reads
        files only from its own folder) and ``copied`` says so."""
        import shutil

        src = Path(path).resolve()
        folder = Path(self.engine.base_dir).resolve()
        if not src.is_file():
            raise ToolError(f"{path} is not a file")
        if folder in src.parents:
            return {"path": src.relative_to(folder).as_posix(), "copied": None}
        dest = folder / "media" / src.name
        k = 2
        while dest.exists() and not _same_file(dest, src):
            dest = dest.with_name(f"{src.stem}_{k}{src.suffix}")
            k += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():  # the same file is there already: used, not copied (nor removed)
            return {"path": dest.relative_to(folder).as_posix(), "copied": None}
        shutil.copy2(src, dest)
        return {"path": dest.relative_to(folder).as_posix(), "copied": str(dest)}

    def _at_seconds(self, at: str | float | None) -> float:
        markers = self.timeline()["markers"]
        if at in (None, ""):
            return 0.0
        if isinstance(at, str) and at in markers:
            return float(markers[at])
        try:
            return max(0.0, float(at))
        except (TypeError, ValueError):
            raise ToolError(f"{at!r} is neither seconds nor a marker "
                            f"({', '.join(markers) or 'none'})") from None  # fmt: skip

    # Footage (Milestone 24) ------------------------------------------------------

    def add_footage(self, path: str | Path, *, background: bool = True) -> dict[str, Any]:
        """A video clip (.mp4, .mov, ...), a picture or a picture sequence: as the full-frame
        background (filling the frame behind everything, labels may sit on it), or placed
        like any object."""
        from phasor.core.files import VIDEO_EXT
        from phasor.domains.frames import extension
        from phasor.scene import BACKGROUND_Z

        brought = self.bring_file(path)
        rel = brought["path"]
        video = extension(rel) in VIDEO_EXT
        kind = "video" if video else "image"
        prefix = "background" if background else Path(rel).stem
        if background:
            params = {"path": rel, "fill": True, "fit": "cover", "z_index": BACKGROUND_Z}

            def code(var: str, oid: str) -> list[str]:
                return [f"{var}.background({rel!r}, id={oid!r})"]
        else:
            params, code = {"path": rel}, None
        return self._add_object(kind, params, prefix=prefix, code=code, copied=brought["copied"],
                                what=f"{'background' if background else kind}: {rel}")  # fmt: skip

    def add_sequence(self, frame: str | Path, *, background: bool = True,
                     fps: float = 24.0) -> dict[str, Any]:  # fmt: skip
        """A picture sequence from one of its frames (``shot_0007.png`` -> ``shot_%04d.png``,
        every frame of it in the scene's folder), played at ``fps``."""
        import re

        p = Path(frame)
        m = re.fullmatch(r"(.*?)(\d+)", p.stem)
        if m is None:
            raise ToolError(f"{p.name}: a frame of a sequence ends in its number (shot_0001.png)")
        folder = Path(self.engine.base_dir).resolve()
        if folder not in p.resolve().parents:
            raise ToolError("a picture sequence is used where it is: put its folder inside the "
                            f"scene's folder ({folder}) first")  # fmt: skip
        digits = m.group(2)
        width = len(digits) if digits.startswith("0") else 0
        pattern = p.with_name(f"{m.group(1)}%0{width}d{p.suffix}" if width
                              else f"{m.group(1)}%d{p.suffix}")  # fmt: skip
        rel = pattern.resolve().relative_to(folder).as_posix()
        from phasor.scene import BACKGROUND_Z

        params: dict[str, Any] = {"path": rel, "fps": float(fps)}
        code = None
        if background:
            params.update(fill=True, fit="cover", z_index=BACKGROUND_Z)

            def code(var: str, oid: str) -> list[str]:
                return [f"{var}.background({rel!r}, id={oid!r}, fps={float(fps)!r})"]

        return self._add_object("image", params, prefix="background" if background else "frames",
                                code=code, what=f"picture sequence: {rel}")  # fmt: skip

    # Sound and MIDI (Milestones 22, 23) -------------------------------------------

    def add_sound(self, path: str | Path, at: str | float | None = None,
                  gain: float = 0.0) -> dict[str, Any]:  # fmt: skip
        """A sound clip that plays from a marker (it follows the marker when the marker
        moves: a snap, a drag) or from a number of seconds."""
        brought = self.bring_file(path)
        markers = self.timeline()["markers"]
        start: float | str = at if isinstance(at, str) and at in markers else self._at_seconds(at)
        params: dict[str, Any] = {"path": brought["path"], "start": start}
        if gain:
            params["gain"] = float(gain)
        when = start if isinstance(start, str) else f"{start:g}s"
        return self._add_object("sound", params, prefix=Path(brought["path"]).stem,
                                copied=brought["copied"],
                                what=f"sound: {brought['path']} at {when}")  # fmt: skip

    def add_notes(self, path: str | Path | None = None, typed: str = "",
                  at: str | float | None = None, instrument: str = "auto",
                  show: bool = False) -> dict[str, Any]:  # fmt: skip
        """MIDI notes played over the scene from ``at`` (a marker's name: they follow it;
        or seconds): from a ``.mid`` file, or typed (``parse_notes``)."""
        params: dict[str, Any] = {}
        copied = None
        if path:
            brought = self.bring_file(path)
            params["path"], copied = brought["path"], brought["copied"]
            prefix = Path(brought["path"]).stem
        else:
            params["notes"] = [list(n) for n in parse_notes(typed)]
            prefix = "notes"
        markers = self.timeline()["markers"]
        if isinstance(at, str) and at in markers:
            params["start"] = at
        elif at not in (None, ""):
            params["start"] = self._at_seconds(at)
        if instrument != "auto":
            params["instrument"] = instrument
        if show:
            params["show"] = True
        return self._add_object("notes", params, prefix=prefix, copied=copied,
                                what=f"notes: {params.get('path') or 'typed'}")  # fmt: skip

    def export_midi(self, out: str | Path | None = None, overwrite: bool = False) -> dict:
        """Write the scene's notes to a ``.mid`` (default ``<scene>.mid`` beside the scene;
        never over a file the scene reads; over another only with ``overwrite``)."""
        target = Path(out) if out else self.default_output(".mid")
        with self.lock:
            return self.ws.export_midi(str(target), overwrite=overwrite)

    # Narration (Milestone 22) -----------------------------------------------------

    def narration(self) -> dict[str, Any]:
        """The Narration panel: what speaks (a voice or a recording), its script, blocks
        and the markers it owns."""
        n = self.engine.document.narration
        if n is None:
            return {"voice": None, "audio": None, "script": "", "blocks": 0}
        audio = n.voice[len("file:") :] if n.voice.startswith("file:") else None
        return {"voice": n.voice, "audio": audio, "script": n.script, "blocks": len(n.blocks),
                "markers": sorted(n.markers())}  # fmt: skip

    def set_narration_audio(self, path: str | Path, script: str = "") -> dict[str, Any]:
        """Narrate from a recording (a WAV, or an mp4's sound...): ``script`` is what it
        says (its sentences are found in it; ``[[markers]]`` fall on their words), or empty
        (its phrases become the blocks)."""
        brought = self.bring_file(path)
        rel = brought["path"]
        if self.is_script:
            var = self.ws.script_var
            r = self._add_code([f"{var}.narrate({script!r}, audio={rel!r})"],
                               f"narration: {rel}")  # fmt: skip
        else:
            with self.lock:
                try:
                    out = self.ws.narrate(script, audio=rel)
                except ToolError as e:
                    r = {"ok": False, "errors": [{"code": "narration", "message": str(e)}]}
                else:
                    r = {"ok": True, "version": self.version, "errors": [], **out}
                    self.ws.activity.append({"who": "you", "what": f"narration: {rel}",
                                             "version": self.version, "t": self.t})  # fmt: skip
        if not r.get("ok") and brought["copied"]:
            Path(brought["copied"]).unlink(missing_ok=True)
        return {**r, "copied": brought["copied"] if r.get("ok") else None}

    def speech_preview(self, audio: str | None = None) -> dict[str, Any]:
        """Speech and pauses in a recording (the narration's, or ``audio`` in the scene's
        folder), as the snap sees them: segments, pauses, how far the pauses can be trusted
        (``quality`` 0-1) and its flags said plainly (``notes``)."""
        from phasor.domains import audio as sound

        name = audio or self.narration()["audio"]
        if not name:
            return {"ok": False, "message": "no recording: choose one first"}
        try:
            sp = sound.speech(self.engine.base_dir, name).to_dict()
        except ValueError as e:
            return {"ok": False, "message": f"{name}: {e}"}
        notes = [_FLAG_TEXT.get(f, f) for f in sp["flags"]]
        text = (f"{len(sp['segments'])} stretches of speech, {len(sp['pauses'])} pauses in "
                f"{sp['seconds']:.1f} s; pauses trusted {100 * sp['quality']:.0f}%")  # fmt: skip
        return {"ok": True, "audio": name, **sp, "notes": notes, "text": text}

    def snap_preview(self) -> dict[str, Any]:
        """What "Snap markers to narration" would do, without doing it: each move (marker,
        from, to, why, how sure) as a line to read, and the recording's flags plainly (an
        ``unreliable`` alignment moves nothing)."""
        with self.lock:
            r = self.ws.snap_to_narration(dry_run=True)
        rows = [
            f"{m['marker']}: {m['from']:.2f}s -> {m['to']:.2f}s  ({m.get('why', '')}, "
            f"{100 * float(m.get('confidence', 0)):.0f}% sure)"
            for m in r.get("moves", [])
        ]
        kept = [
            f"{m['marker']} stays at {m.get('from', m.get('t', 0)):.2f}s"
            + (f" ({m['why']})" if m.get("why") else "")
            for m in r.get("kept", [])
        ]
        flags = list(r.get("flags", []))
        notes = [_FLAG_TEXT.get(f, f) for f in flags]
        if r.get("unreliable"):
            notes.insert(0, "the alignment is unreliable: nothing would move")
        return {**r, "rows": rows, "kept_rows": kept, "notes": notes,
                "can_apply": bool(r.get("moves")) and not r.get("unreliable")}  # fmt: skip

    def snap_apply(self) -> dict[str, Any]:
        """Move the markers onto the narration: one undoable edit on a document; on a script
        the line ``s.snap_to_narration()`` is added (the same plan, every time it runs)."""
        if self.is_script:
            var = self.ws.script_var
            if f"{var}.snap_to_narration(" in _outside_block(self.buffer):
                return self._refused("the script already snaps its markers to the narration")
            return self._add_code([f"{var}.snap_to_narration()"], "snap markers to narration")
        with self.lock:
            try:
                r = self.ws.snap_to_narration(dry_run=False)
            except ToolError as e:
                return self._refused(str(e))
        if r.get("moves") and not r.get("errors"):
            entry = {"what": f"snap {len(r['moves'])} markers to narration",
                     "version": self.version, "t": self.t}  # fmt: skip
            self.history.append(entry)
            self.ws.activity.append({"who": "you", **entry})
        return {"ok": not r.get("errors"), **r}

    # ------------------------------------------------------------ renders (jobs)

    def default_output(self, suffix: str) -> Path:
        """``<scene><suffix>`` in the scene's folder: the script's or document's name (the
        title before the first save)."""
        from phasor.core.ids import file_stem

        p = self.ws.path
        stem = p.name.split(".")[0] if p else file_stem(self.engine.document.meta.title)
        folder = p.parent if p else self.ws.root
        return folder / f"{stem}{suffix}"

    def render_plan(self, preset: str) -> dict[str, Any]:
        """What the Render dialog shows for a quality: picture size, frames a second, the
        file it writes by default (a draft: ``<scene>.draft.mp4`` beside the scene, the same
        file ``render_start`` writes; never the real one)."""
        from phasor.render.presets import preset_named

        p = preset_named(preset)
        meta = self.engine.document.meta
        path = (self.ws.default_render_path("draft") if preset == "draft"
                else self.default_output(".mp4"))  # fmt: skip
        return {"preset": preset, "size": p.size(meta.width, meta.height),
                "fps": p.fps or meta.fps, "note": p.note, "path": str(path)}  # fmt: skip

    def start_render(self, preset: str = "draft", out: str | Path | None = None,
                     fps: float | None = None, overwrite: bool | None = None,
                     start: float | str | None = None, end: float | str | None = None,
                     frame: float | None = None) -> dict[str, Any]:  # fmt: skip
        """Start a render job of the scene as it is now, to ``out`` anywhere the person
        chose (a file dialog; the attached AI's ``render_start`` stays in its folder).
        Returns its status: the caller keeps its ``job`` id and follows it
        (``job_status``, ``cancel_render``). ``frame``: one picture at that time (a
        ``.png``). ``fps`` equal to the preset's is the preset's (None)."""
        plan = self.render_plan(preset)
        target = Path(out) if out else Path(plan["path"])
        if overwrite is None and out is None and preset == "draft":
            overwrite = True  # a draft replaces the last draft (never the real render)
        if frame is not None or (fps is not None and abs(fps - float(plan["fps"])) < 1e-9):
            fps = None
        with self.lock:
            return self.ws._render_job(target, preset, start=start, end=end, fps=fps,
                                       frame=frame, overwrite=overwrite)  # fmt: skip

    def draft_preview(self) -> dict[str, Any]:
        """The Draft preview button: the ``draft`` preset (15 fps, small, no blur, bloom,
        haze or life; repeated frames not drawn again), to ``<scene>.draft.mp4``, played
        when done (``poll_job``)."""
        r = self.start_render("draft")
        self.job = r["job"]
        return r

    def job_status(self, job: str) -> dict[str, Any]:
        return self.ws.render_status(job)

    def cancel_render(self, job: str) -> dict[str, Any]:
        """Stop that render job (the shared lock is not held while it stops)."""
        return self.ws.render_cancel(job)

    def poll_job(self) -> dict[str, Any] | None:
        """The Draft preview's status; ``open`` is the file to play when it is done."""
        if self.job is None:
            return None
        st = self.ws.render_status(self.job)
        if st["state"] in ("done", "failed", "cancelled"):
            self.job = None
            if st["state"] == "done":
                st = {**st, "open": st["path"]}
        return st

    def cancel_job(self) -> dict[str, Any] | None:
        """Cancel the Draft preview (only it: a render the dialog started is its own)."""
        if self.job is None:
            return None
        r = self.cancel_render(self.job)
        self.job = None
        return r

    def trust_scene_code(self) -> dict[str, Any]:
        """Let the modules the open document names (``meta.modules``: its own object types
        and updaters) run. A document opens without running them, whoever wrote it; lint
        ``missing_type`` names what is waiting for this. Only the person at the editor can:
        it is not a tool, so an attached AI cannot. Unsaved edits stay; undo starts again."""
        with self.lock:
            dirty = self.dirty
            r = self.ws._trust_code()
            self.saved_version = -2 if dirty else self.version
            self._seen = (-2, -2)  # the window redraws (``changed``)
            self._frame = self._lint = self._player = None
        return r

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
        v = (self.generation, self.version)
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
        already there, except where the scene typed a value (a color, a size: it stays, and
        a typed color gets no hue along it). One undo step."""
        from phasor.style.looks import load_look

        look = load_look(name)
        ops: list[dict[str, Any]] = [
            {"op": "set_meta", "field": "theme", "value": look.theme},
            {"op": "set_meta", "field": "life", "value": look.life},
            {"op": "set_meta", "field": "look", "value": name},
        ]
        for spec in self.engine.document.objects:
            params, style = look.defaults_for(spec.type, dict(spec.style))
            for k, v in params.items():
                ops.append({"op": "set_param", "path": f"{spec.id}.{k}", "value": v})
            for k, role in style.items():
                ops.append({"op": "set_style", "id": spec.id, "key": k, "role": role})
        return self._edit(ops, f"look: {name}")

    def palettes(self) -> list[str]:
        from phasor.style.theme import PALETTES

        return list(PALETTES)

    def set_palette(self, name: str | None) -> dict[str, Any]:
        """Recolor the strokes with a palette (``spectrum``, the prism rainbow, ...): one
        undo step, recorded as your edit."""
        with self.lock:
            r = self.ws.scene_palette(name)
        if r["ok"]:
            entry = {"what": f"palette: {name}", "version": self.version, "t": self.t}
            self.history.append(entry)
            self.ws.activity.append({"who": "you", **entry})
        return r

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
