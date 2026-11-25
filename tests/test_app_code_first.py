"""Milestone 25: the code-first window's logic (``Editor``), without Qt.

The script is the scene: the editor holds its text (``buffer``), Run builds the scene from
it, an error is shown at its line and keeps the last good scene, tweaks (the inspector, a
drag, an AI) are written into the tweaks block (ADR 0004), the file is watched (an outside
edit reloads; a conflict asks), an object shows its line and a line its object, and the
panels (footage, sound and MIDI, narration, renders) work on scripts and documents alike.

The invariant of ADR 0004: after any mix of typed edits, editor tweaks, AI edits, undo and
saves, running the script fresh gives the document the live engine holds."""

from __future__ import annotations

import os
import random
import re
import shutil
import threading
import time
from pathlib import Path

import pytest

from phasor.core import io
from phasor.gui.controller import Editor, open_workspace, parse_notes
from phasor.scene import run_script
from phasor.workspace import ToolError, Workspace
from phasor.workspace import jobs as jobmod

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SCRIPT = """from phasor import Dot, Scene, Text

s = Scene("code first", passes=None)
s.add(Dot(id="pole", pos=(1, 0, 0), r=0.2))
s.add(Text(id="note", text="hello", pos=(0, 1, 0)))
s.marker("go", 1.0)
s.marker("end", 2.0)
"""


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "scene.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return path


@pytest.fixture
def ed(scene):
    return Editor(open_workspace(str(scene)))


def _doc(engine) -> str:
    return io.dumps(engine.document)


def _fresh(text: str, folder: Path) -> str:
    """The document a fresh run of ``text`` makes (a copy beside the script)."""
    probe = folder / "fresh_probe.py"
    probe.write_text(text, encoding="utf-8")
    try:
        return _doc(run_script(probe, passes=None).engine)
    finally:
        probe.unlink()


def _touch_later(path: Path, text: str) -> None:
    """Write ``text`` as another program would, with a later modification time (a coarse
    clock must still see it)."""
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


# ---------------------------------------------------------------- the script and Run


def test_the_script_is_the_buffer_and_run_rebuilds_the_scene(ed, scene):
    assert ed.is_script and ed.buffer == SCRIPT and not ed.dirty
    assert ed.code() == {"text": SCRIPT, "is_script": True, "path": str(scene.resolve())}
    ed.set_buffer(SCRIPT.replace("r=0.2", "r=0.45"))
    assert ed.dirty  # typed, not run
    g = ed.generation
    r = ed.run()
    assert r["ok"] and ed.generation == g + 1 and not ed.dirty
    assert ed.engine.document.object("pole").params["r"] == 0.45
    assert "r=0.45" in scene.read_text(encoding="utf-8")  # Run keeps the code on disk
    assert ed.changed()  # the window redraws


def test_an_error_shows_its_line_and_keeps_the_last_good_scene(ed, scene):
    before = _doc(ed.engine)
    broken = SCRIPT + "s.add(Dot(id='bad', r=-1))\n"
    r = ed.run_code(broken)
    assert not r["ok"] and r["line"] == 8 and ed.error["line"] == 8
    assert _doc(ed.engine) == before  # the preview keeps the last good scene
    assert ed.buffer == broken  # the text stays to be fixed
    assert scene.read_text(encoding="utf-8") == SCRIPT  # the file keeps the code that runs
    assert ed.problems()[0]["code"] == "script_error" and ed.problems()[0]["line"] == 8
    r = ed.run_code("s = Scene(")
    assert not r["ok"] and r["line"] == 1
    assert ed.run_code(SCRIPT)["ok"] and ed.error is None


def test_tweaks_are_written_into_the_block_and_save_does_not_rerun(ed, scene):
    assert ed.set_param("pole", "r", "0.3")["ok"]
    assert "s.tweak('pole.r', 0.3)" in ed.buffer and ed.dirty
    assert ed.buffer.startswith(SCRIPT)  # the code above the block is untouched
    g = ed.generation
    r = ed.save()
    assert r["saved"] == str(scene.resolve()) and ed.generation == g  # no rerun for a tweak
    assert scene.read_text(encoding="utf-8") == ed.buffer and not ed.dirty
    ed.undo()
    assert "pole.r" not in ed.buffer and ed.dirty
    ed.redo()
    assert "pole.r" in ed.buffer
    # an AI's tweak over the shared workspace shows up the same way
    with ed.lock:
        ed.ws.scene_edit({"op": "set_style", "id": "note", "key": "color", "role": "muted"})
    assert ed.changed() and "s.tweak_style('note', 'color', 'muted')" in ed.buffer


def test_typing_in_the_block_is_kept_until_run(ed):
    ed.set_param("pole", "r", "0.3")
    ed.set_buffer(ed.buffer.replace("0.3)", "0.33)"))  # the person edits a tweak by hand
    assert ed.set_param("note", "pos", "[0, 2, 0]")["ok"]  # another tweak arrives meanwhile
    assert ed.block_edited and "0.33" in ed.buffer  # the hand edit is not overwritten
    assert any(p["code"] == "tweaks_edited" for p in ed.problems())
    ed.run()
    assert ed.engine.document.object("pole").params["r"] == 0.33 and not ed.block_edited


def test_save_runs_changed_code_first_and_refuses_a_broken_script(ed, scene):
    ed.set_buffer(SCRIPT.replace("hello", "bonjour"))
    assert ed.save()["saved"] and ed.engine.document.object("note").params["text"] == "bonjour"
    ed.set_buffer(SCRIPT + "s.add(Dot(id='bad', r=-1))\n")
    r = ed.save()
    assert r["ok"] is False and "line 8" in r["errors"][0]["message"]
    assert "bonjour" in scene.read_text(encoding="utf-8")  # the file keeps the good code


# ---------------------------------------------------------------- the file on disk


def test_an_outside_edit_reloads_and_a_conflict_asks(ed, scene):
    assert ed.poll_file() is None
    _touch_later(scene, SCRIPT.replace("hello", "from outside"))
    r = ed.poll_file()
    assert r["event"] == "reloaded" and r["ok"]
    assert ed.engine.document.object("note").params["text"] == "from outside"
    assert ed.buffer == scene.read_text(encoding="utf-8") and not ed.dirty
    # an outside edit that breaks the script: shown at its line, the last good scene stays
    _touch_later(scene, SCRIPT + "s.add(Dot(id='bad', r=-1))\n")
    r = ed.poll_file()
    assert r["event"] == "reloaded" and not r["ok"] and ed.error["line"] == 8
    assert ed.engine.document.object("note").params["text"] == "from outside"
    _touch_later(scene, SCRIPT)
    assert ed.poll_file()["ok"]
    # unsaved edits here and a change there: a conflict, nothing merged
    ed.set_buffer(SCRIPT.replace("hello", "mine"))
    _touch_later(scene, SCRIPT.replace("hello", "theirs"))
    assert ed.poll_file() == {"event": "conflict"}
    assert ed.buffer.count("mine") == 1 and ed.dirty
    assert ed.poll_file() is None  # asked once, until answered
    ed.resolve_conflict("mine")
    assert ed.buffer.count("mine") == 1 and ed.dirty
    ed.save()
    assert "mine" in scene.read_text(encoding="utf-8")
    ed.set_buffer(SCRIPT.replace("hello", "mine again"))
    _touch_later(scene, SCRIPT.replace("hello", "theirs"))
    assert ed.poll_file() == {"event": "conflict"}
    assert ed.resolve_conflict("file")["ok"]
    assert ed.engine.document.object("note").params["text"] == "theirs" and not ed.dirty


def test_the_editors_own_writes_are_not_outside_edits(ed, scene):
    ed.set_param("pole", "r", "0.3")
    ed.save()
    assert ed.poll_file() is None
    with ed.lock:  # an attached AI saves: the workspace wrote it, so it is not "outside"
        ed.ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.25})
        ed.ws.scene_save()
    ed.changed()
    assert ed.poll_file() is None and not ed.dirty
    assert ed.buffer == scene.read_text(encoding="utf-8")


# ---------------------------------------------------------------- objects and lines


def test_selecting_an_object_shows_its_line_and_back(ed):
    assert ed.line_of("pole") == 4 and ed.line_of("note") == 5
    assert ed.object_at_line(4) == "pole" and ed.object_at_line(5) == "note"
    assert ed.object_at_line(1) is None
    assert ed.select_line(5) == "note" and ed.selected == "note"
    ed.set_param("pole", "r", "0.3")  # a tweak line names its object too
    line = ed.buffer.splitlines().index("s.tweak('pole.r', 0.3)") + 1
    assert ed.object_at_line(line) == "pole"


# ---------------------------------------------------------------- panels


@pytest.fixture
def media(tmp_path_factory):
    out = tmp_path_factory.mktemp("elsewhere")  # outside the scene's folder
    for name in ("clip.mp4", "motif.mid", "narration_demo.wav"):
        shutil.copy2(EXAMPLES / "assets" / name, out / name)
    return out


def test_footage_sound_and_notes_on_a_script_become_code(ed, scene, media):
    r = ed.add_footage(media / "clip.mp4")
    assert r["ok"] and r["copied"].endswith(str(Path("media") / "clip.mp4"))
    assert "s.background('media/clip.mp4', id='background')" in ed.buffer
    bg = ed.engine.document.object("background")
    assert bg.type == "video" and bg.params["fill"] and bg.params["z_index"] == -1000
    r = ed.add_sound(media / "narration_demo.wav", at="go", gain=-6)
    assert r["ok"] and ed.engine.document.object(r["id"]).params["start"] == 1.0
    r = ed.add_notes(typed="C4 0 0.5, E4 0.5 0.5", at="go", instrument="bell")
    notes = ed.engine.document.object(r["id"]).params
    assert notes["start"] == "go" and notes["notes"][1][0] == 64 and notes["instrument"] == "bell"
    assert ed.add_notes(path=media / "motif.mid")["ok"]
    assert ed.buffer.index("from phasor import Sound") < ed.buffer.index("s.add(Sound(")
    assert _fresh(ed.buffer, scene.parent) == _doc(ed.engine)
    out = scene.parent / "scene.mid"
    assert ed.export_midi()["notes"] > 0 and out.exists()
    with pytest.raises(ToolError, match="exists"):
        ed.export_midi()  # not over an existing file unless asked
    assert ed.export_midi(overwrite=True)["path"] == str(out)
    with pytest.raises(ToolError, match="pitch start length"):
        ed.add_notes(typed="C4 0")


def test_a_bad_addition_to_a_script_is_taken_back(ed, scene, media):
    before, g = ed.buffer, ed.generation
    with pytest.raises(ToolError, match="neither seconds nor a marker"):
        ed.add_sound(media / "narration_demo.wav", at="nowhere")
    (media / "notes.txt").write_text("not a picture", encoding="utf-8")
    r = ed.add_footage(media / "notes.txt", background=False)  # Image refuses a .txt
    assert r["ok"] is False and "line" in r["errors"][0]["message"]
    assert ed.buffer == before and ed.generation == g and ed.error is None
    assert scene.read_text(encoding="utf-8") == SCRIPT


def test_footage_and_sound_on_a_document_are_undoable_edits(tmp_path, media):
    ws = Workspace(tmp_path)
    ws.scene_new("doc")
    ed = Editor(ws)
    assert not ed.is_script
    v = ed.version
    r = ed.add_footage(media / "clip.mp4", background=False)
    assert r["ok"] and ed.version == v + 1 and ed.engine.document.object(r["id"]).type == "video"
    ed.undo()
    assert ed.engine.document.object(r["id"]) is None
    assert ed.add_notes(typed="60 0 1")["ok"]


def test_parse_notes_reads_names_and_numbers():
    assert parse_notes("C4 0 0.5; F#3 0.5 0.25 90\nBb2 1 1") == [
        (60.0, 0.0, 0.5, 100.0), (54.0, 0.5, 0.25, 90.0), (46.0, 1.0, 1.0, 100.0)]  # fmt: skip
    with pytest.raises(ToolError):
        parse_notes("")


def test_narration_from_a_recording_and_the_snap_on_a_document(tmp_path, media):
    ws = Workspace(tmp_path)
    ws.scene_new("narrated")
    ed = Editor(ws)
    ed.ws.scene_edit({"op": "add_object", "object": {"id": "d", "type": "dot"}})
    r = ed.set_narration_audio(media / "narration_demo.wav")
    assert r["ok"] and ed.narration()["audio"] == "media/narration_demo.wav"
    for name, t in (("a", 0.4), ("b", 3.4), ("c", 6.1)):
        ed.ws.scene_edit({"op": "set_marker", "name": name, "t": t})
    v = ed.version
    pre = ed.snap_preview()
    assert ed.version == v  # a preview moves nothing
    assert isinstance(pre["rows"], list) and isinstance(pre["notes"], list)
    if pre["can_apply"]:
        r = ed.snap_apply()
        assert r["ok"] and ed.version == v + 1  # one undoable step
        ed.undo()
        assert ed.version == v + 2 and ed.timeline()["markers"]["b"] == 3.4


def test_narration_and_snap_on_a_script_are_code(ed, scene, media):
    r = ed.set_narration_audio(media / "narration_demo.wav")
    assert r["ok"] and "s.narrate('', audio='media/narration_demo.wav')" in ed.buffer
    r = ed.snap_apply()
    assert r["ok"] and "s.snap_to_narration()" in ed.buffer
    assert not ed.snap_apply()["ok"]  # once is enough
    assert _fresh(ed.buffer, scene.parent) == _doc(ed.engine)


# ---------------------------------------------------------------- renders


class _Fake:
    def __init__(self):
        self.gate = threading.Event()

    def __call__(self, engine, out, *, progress, cancel, preset, **kw):
        with open(out, "wb") as f:
            for k in range(10):
                if k == 5:
                    self.gate.wait(10)
                f.write(b"x")
                progress(k + 1, 10)
                if cancel.is_set():
                    f.close()
                    Path(out).unlink()
                    from phasor.render.job import RenderCancelled

                    raise RenderCancelled("cancelled")
        return {"frames": 10, "fps": 15.0, "size": (64, 36)}


def _until(fn, timeout=10.0):
    end = time.monotonic() + timeout
    while True:
        r = fn()
        if r is not None and r.get("state") not in ("running", "cancelling"):
            return r
        assert time.monotonic() < end
        time.sleep(0.01)


def test_draft_preview_plays_when_done_and_render_never_takes_the_finals_name(ed, scene,
                                                                              monkeypatch):  # fmt: skip
    fake = _Fake()
    monkeypatch.setattr(jobmod, "_render", fake)
    plan = ed.render_plan("draft")
    assert plan["path"] == str(scene.with_name("scene.draft.mp4"))
    assert plan["fps"] == 15.0 and max(plan["size"]) <= 640
    assert ed.render_plan("final")["path"] == str(scene.with_name("scene.mp4"))
    st = ed.draft_preview()
    assert st["state"] == "running" and ed.job == st["job"]
    fake.gate.set()
    done = _until(ed.poll_job)
    assert done["state"] == "done" and done["open"] == plan["path"]  # the window plays it
    assert ed.job is None
    again = ed.draft_preview()  # a new draft replaces the last draft
    _until(ed.poll_job)
    assert again["path"] == plan["path"]
    fake.gate.clear()
    final = scene.with_name("scene.mp4")
    final.write_bytes(b"kept")
    with pytest.raises(ToolError, match="exists"):
        ed.start_render("final")
    ed.start_render("final", overwrite=True)
    time.sleep(0.05)
    r = ed.cancel_job()
    assert r["state"] in ("cancelled", "cancelling") or r["state"] == "running"
    fake.gate.set()
    for j in ed.ws._jobs.all():
        j.wait(10)
    assert final.read_bytes() == b"kept"  # the cancelled render left the old file alone
    assert not jobmod.partial_path(final).exists()


# ---------------------------------------------------------------- the invariant (ADR 0004)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_fresh_run_equals_the_live_scene_after_any_mix_of_edits(scene, seed):
    """Random sequences of typed edits (run), editor tweaks, AI edits, undo, redo and saves;
    after each step a fresh run of the editor's script, and of the file once saved, gives the
    document the live engine holds."""
    rng = random.Random(seed)
    ed = Editor(open_workspace(str(scene)))
    folder = scene.parent
    radii = [0.1, 0.15, 0.25, 0.3, 0.35]
    added = 0
    orphans = 0
    # First a tweak whose object the next run removes: an orphan, kept through every step.
    from phasor.core.pycode import insert_code

    assert ed.run_code(insert_code(ed.buffer, ["s.add(Dot(id='gone', pos=(2, 2, 0)))"]))["ok"]
    assert ed.set_param("gone", "r", "0.3")["ok"]
    text = "".join(ln for ln in ed.buffer.splitlines(keepends=True) if "id='gone'" not in ln)
    assert ed.run_code(text)["ok"] and "s.tweak('gone.r', 0.3)" in ed.buffer
    assert _fresh(ed.buffer, folder) == _doc(ed.engine)
    for _ in range(18):
        step = rng.choice(["tweak", "style", "ai", "undo", "redo", "typed", "save", "rule",
                           "extra", "outside"])  # fmt: skip
        extras = sorted(i for i in ed._ids() if i.startswith("extra"))
        if step == "extra" and extras:  # a tweak to an object a later run may remove
            ed.set_param(rng.choice(extras), "r", str(rng.choice(radii)))
        elif step == "outside":  # another editor changes the code outside the block
            ed.save()
            text = re.sub(
                r'text="[^"]*"', f'text="v{rng.random():.3f}"', scene.read_text(encoding="utf-8")
            )
            _touch_later(scene, text)
            assert ed.poll_file()["ok"]
        elif step == "tweak":
            ed.set_param(rng.choice(["pole", "note"]), "pos", f"[{rng.random():.3f}, 0, 0]")
        elif step == "style":
            ed.set_style("pole", "color", rng.choice(["highlight", "muted", "danger"]))
        elif step == "ai":
            with ed.lock:
                ed.ws.scene_edit({"op": "set_param", "path": "pole.r", "value": rng.choice(radii)})
            ed.changed()
        elif step == "rule":
            with ed.lock:
                ed.ws.scene_edit({"op": "add_constraint", "rule": "note near pole"})
            ed.changed()
        elif step == "undo":
            ed.undo()
        elif step == "redo":
            ed.redo()
        elif step == "typed":
            added += 1
            text = ed.buffer
            if rng.random() < 0.4 and "s.add(Dot(id='extra" in text:
                # remove an added object: its tweaks become orphans, never lost
                lines = [ln for ln in text.splitlines(keepends=True) if "id='extra" not in ln]
                text = "".join(lines)
            else:
                from phasor.core.pycode import insert_code

                text = insert_code(text, [f"s.add(Dot(id='extra{added}', pos=({added}, 1, 0)))"])
            assert ed.run_code(text)["ok"]
        elif step == "save":
            ed.save()
            assert _fresh(scene.read_text(encoding="utf-8"), folder) == _doc(ed.engine)
        assert not ed.block_edited
        assert _fresh(ed.buffer, folder) == _doc(ed.engine), step
        orphans += any(i.code == "orphan_override" for i in ed.engine.lint())
    assert orphans == 18  # the orphaned tweak was never dropped


def test_a_crlf_script_keeps_its_line_ends(tmp_path):
    """A script with Windows line ends (git autocrlf, Notepad): not dirty on open, a tweak and
    a save keep CRLF outside the block, an outside CRLF edit reloads (no false conflict)."""
    path = tmp_path / "crlf.py"
    path.write_bytes(SCRIPT.replace("\n", "\r\n").encode("utf-8"))
    ed = Editor(open_workspace(str(path)))
    assert not ed.dirty and ed.buffer == SCRIPT  # the editor holds "\n" text
    ed.set_buffer(ed.buffer)  # what the widget hands back
    assert not ed.dirty
    ed.set_param("pole", "r", "0.3")
    ed.save()
    raw = path.read_bytes().decode("utf-8")
    assert raw.startswith(SCRIPT.replace("\n", "\r\n")) and "\n" not in raw.replace("\r\n", "")
    assert "s.tweak('pole.r', 0.3)\r\n" in raw and not ed.dirty
    edited = raw.replace("hello", "outside")
    path.write_bytes(edited.encode("utf-8"))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    r = ed.poll_file()
    assert r["event"] == "reloaded" and r["ok"] and not ed.dirty
    ed.set_buffer(ed.buffer.replace("outside", "typed"))
    assert ed.run()["ok"]
    assert b"\r\n" in path.read_bytes() and b"typed" in path.read_bytes()
