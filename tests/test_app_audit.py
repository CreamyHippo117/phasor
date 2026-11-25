"""Milestone 25, the independent audit: a regression test for each finding (B1-B3, S1-S12
and the nits fixed). The principle: the person's script is sacred. phasor writes only inside
the tweaks block, never loses text the person or another editor wrote, and never writes
when it cannot be sure."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from phasor.core import io
from phasor.core.pycode import BlockError, find_tweaks, with_tweaks
from phasor.core.schema import Overrides
from phasor.gui.controller import Editor, open_workspace
from phasor.scene import run_script
from phasor.workspace import ToolError, Workspace
from phasor.workspace import jobs as jobmod

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SCRIPT = """from phasor import Dot, Scene, Text

s = Scene("audit", passes=None)
s.add(Dot(id="pole", pos=(1, 0, 0), r=0.2))
s.add(Text(id="note", text="hello", pos=(0, 1, 0)))
s.marker("go", 1.0)
s.marker("end", 2.0)
"""
BEGIN = "# --- phasor tweaks: x ---\n"
END = "# --- end phasor tweaks ---\n"


def _script(tmp_path: Path, text: str, name: str = "scene.py") -> Path:
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return p


def _later(p: Path, text: str) -> None:
    """Write as another program would, with a later time (a coarse clock still sees it)."""
    p.write_bytes(text.encode("utf-8"))
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def _doc(engine) -> str:
    return io.dumps(engine.document)


# ---------------------------------------------------------------- B1, S5: the markers


def test_b1_a_begin_marker_without_its_end_never_costs_the_persons_code(tmp_path):
    """Audit B1 (p1 case 3): the end marker deleted by hand, code after the block: every
    later save used to replace everything from the orphan begin marker to a new end."""
    p = _script(tmp_path, SCRIPT + BEGIN + "s.tweak('pole.r', 0.3)\n"
                + "s.add(Dot(id='mine', pos=(0, 0, 0)))  # my code\n")  # fmt: skip
    before = p.read_bytes()
    ed = Editor(open_workspace(str(p)))
    assert ed.block_problem is not None and ed.block_problem["line"] == 8
    assert ed.set_param("pole", "r", "0.4")["ok"]  # the tweak is in the scene...
    assert "id='mine'" in ed.buffer and ed.buffer.count("phasor tweaks") == 1
    r = ed.save()  # ...but the script cannot show it: nothing is written
    assert r["ok"] is False and r["errors"][0]["code"] == "tweaks_block"
    assert "no end marker" in r["errors"][0]["message"]
    assert p.read_bytes() == before
    assert any(i["code"] == "tweaks_block" and i["line"] == 8 for i in ed.problems())
    # an AI's save over MCP is refused the same way
    ws = Workspace(tmp_path, allow_scripts=True)
    ws.scene_open("scene.py")
    ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.5})
    for _ in range(2):
        with pytest.raises(ToolError, match="no end marker"):
            ws.scene_save()
    assert p.read_bytes() == before


@pytest.mark.parametrize("text, problem", [
    (SCRIPT + BEGIN + END + BEGIN + END, "more than one tweaks block"),
    (SCRIPT + END + "s.tweak('pole.r', 0.3)\n" + BEGIN, "comes before"),
    (SCRIPT + END, "no begin marker"),
    (SCRIPT + "if True:\n    " + BEGIN + "    pass\n" + END, "no begin marker"),
    (SCRIPT + "    " + BEGIN + END, "indented"),
    (SCRIPT + "x = [\n" + BEGIN + "1,\n" + END + "]\n", None),  # inside a bracket: no block
    (SCRIPT.replace("\n", "\r") + BEGIN + END, "lone CR"),
])  # fmt: skip
def test_b1_markers_that_cannot_be_read_for_certain_are_refused(text, problem):
    if problem is None:  # markers inside a statement are no markers: a new block is added
        assert find_tweaks(text) is None
        assert with_tweaks(text, Overrides(params={"pole.r": 0.4})).startswith(text)
        return
    with pytest.raises(BlockError, match=problem):
        find_tweaks(text)
    with pytest.raises(BlockError):
        with_tweaks(text, Overrides(params={"pole.r": 0.4}))


def test_s5_two_blocks_keep_the_fresh_run_equal_to_the_live_scene(tmp_path):
    """Audit S5 (p1 case 4): only the first of two blocks was rewritten; the second kept a
    stale value (live r=0.1, fresh r=0.9). Now nothing is written and the live scene is the
    script's."""
    blk = BEGIN + "s.tweak('pole.r', 0.7)\n" + END
    p = _script(tmp_path, SCRIPT + blk + "\n" + blk.replace("0.7", "0.9"))
    ed = Editor(open_workspace(str(p)))
    ed.set_param("pole", "r", "0.1")
    assert ed.save()["ok"] is False
    fresh = run_script(p, passes=None).engine.document.object("pole").params["r"]
    assert fresh == 0.9 and "0.1" not in p.read_text(encoding="utf-8")


def test_s5_markers_inside_a_docstring_are_not_the_block(tmp_path):
    """Audit S5 (p1 case 5): the block was written inside a docstring, the tweak lost on a
    fresh run and the docstring rewritten."""
    doc = '"""Docs: the block looks like\n# --- phasor tweaks: here\n# --- end phasor tweaks ---\nend"""\n'
    p = _script(tmp_path, doc + SCRIPT)
    ed = Editor(open_workspace(str(p)))
    ed.set_param("pole", "r", "0.3")
    assert ed.save()["saved"]
    text = p.read_text(encoding="utf-8")
    assert text.startswith(doc + SCRIPT)  # the docstring and the code untouched
    assert text.rstrip().endswith("# --- end phasor tweaks ---")  # a real block at the end
    assert run_script(p, passes=None).engine.document.object("pole").params["r"] == 0.3


# ---------------------------------------------------------------- B2, B3: never lose an outside edit


RACE = """import os
from pathlib import Path
from phasor import Dot, Scene
s = Scene("race", passes=None)
s.add(Dot(id="pole", pos=(1, 0, 0), r=0.2))
if os.environ.get("PHASOR_TEST_RACE") == "1":
    os.environ["PHASOR_TEST_RACE"] = "0"
    p = Path(__file__)
    p.write_text(p.read_text().replace("pos=(1, 0, 0)", "pos=(" + "7, 0, 0)"))
"""


def test_b2_an_outside_edit_during_a_run_is_seen_not_dropped(tmp_path, monkeypatch):
    """Audit B2 (p15): the file was recorded as seen after the run, so a change written
    while the script ran was never noticed and the next save erased it."""
    p = _script(tmp_path, RACE)
    ed = Editor(open_workspace(str(p)))
    monkeypatch.setenv("PHASOR_TEST_RACE", "1")
    assert ed.run_code(ed.buffer + "# typed\n")["ok"]
    assert "(7, 0, 0)" in p.read_text(encoding="utf-8")  # the other editor's change
    r = ed.poll_file()
    assert r is not None and r["event"] == "reloaded"  # seen: nothing unsaved here, it runs
    assert ed.engine.document.object("pole").params["pos"] == (7.0, 0.0, 0.0)
    ed.set_param("pole", "r", "0.3")
    ed.save()
    assert "(7, 0, 0)" in p.read_text(encoding="utf-8")


def test_b3_a_save_never_overwrites_an_edit_made_since_the_last_look(tmp_path):
    """Audit B3a (p4 A): an outside edit between two polls was overwritten by Save."""
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    ed.set_param("pole", "r", "0.3")
    outside = SCRIPT + 's.add(Dot(id="outside", pos=(0, 0, 0)))\n'
    _later(p, outside)
    r = ed.save()
    assert r["ok"] is False and r["conflict"] and ed.conflict is not None
    assert p.read_text(encoding="utf-8") == outside
    assert ed.resolve_conflict("file")["ok"]  # the person takes the file's: it runs
    assert ed.engine.document.object("outside") is not None


def test_b3_an_ais_save_never_absorbs_an_outside_code_edit(tmp_path):
    """Audit B3b (p4 B): ``scene_save`` read whatever was on disk, wrote it back with the
    block and marked it seen; the editor never reloaded it and the next save erased it."""
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    outside = SCRIPT + 's.add(Dot(id="outside", pos=(0, 0, 0)))\n'
    _later(p, outside)
    with ed.lock:
        ed.ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.5})
        with pytest.raises(ToolError, match="changed on disk"):
            ed.ws.scene_save()
    assert p.read_text(encoding="utf-8") == outside
    ed.changed()
    assert ed.poll_file() == {"event": "conflict"}  # the tweak is unsaved here: asked
    ed.resolve_conflict("file")
    assert ed.engine.document.object("outside") is not None


# ---------------------------------------------------------------- S6: file modes


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_s6_a_read_only_script_is_refused_and_modes_are_kept(tmp_path):
    p = _script(tmp_path, SCRIPT)
    os.chmod(p, 0o444)
    ed = Editor(open_workspace(str(p)))
    ed.set_param("pole", "r", "0.3")
    r = ed.save()
    assert r["ok"] is False and "read-only" in str(r)
    assert stat.S_IMODE(p.stat().st_mode) == 0o444 and p.read_text(encoding="utf-8") == SCRIPT
    q = _script(tmp_path, SCRIPT, "private.py")
    os.chmod(q, 0o600)
    ed = Editor(open_workspace(str(q)))
    ed.set_param("pole", "r", "0.3")
    assert ed.save()["saved"]
    assert stat.S_IMODE(q.stat().st_mode) == 0o600


def test_s6_read_only_is_refused_everywhere(tmp_path):
    """The same through the read-only attribute Windows knows (``stat.S_IWRITE``)."""
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    ed.set_param("pole", "r", "0.3")
    os.chmod(p, stat.S_IREAD)
    try:
        r = ed.save()
        assert r["ok"] is False and "read-only" in str(r)
        assert p.read_text(encoding="utf-8") == SCRIPT
    finally:
        os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


# ---------------------------------------------------------------- S7: runs on a worker thread


def test_s7_a_runaway_script_can_be_cancelled_and_keeps_the_last_good_scene(tmp_path):
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    before = _doc(ed.engine)
    r = ed.run_code(SCRIPT + "while True:\n    pass\n", wait=False)
    assert r["ok"] is None and ed.running is not None  # the window is free meanwhile
    assert ed.set_param("pole", "r", "0.3")["ok"] is False  # edits wait for the run
    time.sleep(0.2)
    r = ed.cancel_run()
    assert r["ok"] is False and "cancelled" in r["error"] and ed.running is None
    assert _doc(ed.engine) == before
    assert p.read_text(encoding="utf-8") == SCRIPT  # the file keeps the code that ran
    assert ed.run_code(SCRIPT.replace("hello", "again"))["ok"]


def test_s7_a_run_past_its_time_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_RUN_TIMEOUT", "1")
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    t0 = time.monotonic()
    r = ed.run_code(SCRIPT + "while True:\n    pass\n")
    assert r["ok"] is False and "longer than 1 s" in r["error"]
    assert time.monotonic() - t0 < 30
    assert ed.engine.document.object("pole") is not None


# ---------------------------------------------------------------- S8: undo


def test_s8_a_panel_addition_to_a_script_is_one_undo_step(tmp_path, tmp_path_factory):
    media = tmp_path_factory.mktemp("media")
    shutil.copy2(EXAMPLES / "assets" / "chime.wav", media / "chime.wav")
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    r = ed.add_sound(media / "chime.wav", at="go")
    assert r["ok"] and "s.add(Sound(" in ed.buffer
    assert ed.undo()["undone"]  # the code the app added comes back out, and it runs
    assert "Sound(" not in ed.buffer and ed.engine.document.object(r["id"]) is None
    assert p.read_text(encoding="utf-8") == SCRIPT
    assert ed.redo()["redone"] and ed.engine.document.object(r["id"]) is not None


# ---------------------------------------------------------------- S9: sound at a marker


def test_s9_a_sound_at_a_marker_follows_the_marker(tmp_path, tmp_path_factory):
    media = tmp_path_factory.mktemp("media")
    shutil.copy2(EXAMPLES / "assets" / "chime.wav", media / "chime.wav")
    ws = Workspace(tmp_path)
    ws.scene_new("sound")
    ws.scene_edit({"op": "set_marker", "name": "go", "t": 1.0})
    ed = Editor(ws)
    r = ed.add_sound(media / "chime.wav", at="go")
    assert ed.engine.document.object(r["id"]).params["start"] == "go"

    def start() -> float:
        e = ed.engine
        st = e.state_at(0.0)
        clips = [c for o in st.objects
                 if (c := getattr(o.impl, "audio_clip", lambda s: None)(st))]  # fmt: skip
        return e._at_markers(clips, [])[0].start

    assert start() == 1.0
    ed.move_marker("go", 2.5)  # as a snap would move it
    assert start() == 2.5
    # an old document with seconds still loads, and "1.5" stays seconds
    ws.scene_edit({"op": "set_param", "path": f"{r['id']}.start", "value": "1.5"})
    assert ed.engine.state_at(0.0)[r["id"]].params.start == 1.5


# ---------------------------------------------------------------- S1, S3, S10, S11, S12: renders


class _Fake:
    def __init__(self, frames: int = 6, gate: bool = False) -> None:
        self.frames, self.gate = frames, threading.Event()
        if not gate:
            self.gate.set()

    def __call__(self, engine, out, *, progress, cancel, preset, **kw):
        with open(out, "wb") as f:
            for k in range(self.frames):
                self.gate.wait(10)
                f.write(b"x")
                progress(k + 1, self.frames)
                if cancel.is_set():
                    from phasor.render.job import RenderCancelled

                    raise RenderCancelled("cancelled")
        return {"frames": self.frames, "fps": 15.0, "size": (64, 36)}


def _settle(ws) -> None:
    for j in ws._jobs.all():
        j.wait(10)


def test_s1_the_person_renders_anywhere_the_ai_stays_in_its_folder(tmp_path, tmp_path_factory,
                                                                   monkeypatch):  # fmt: skip
    monkeypatch.setattr(jobmod, "_render", _Fake())
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    away = tmp_path_factory.mktemp("videos") / "mine.mp4"
    r = ed.start_render("final", out=away)  # the Render dialog, a path in ~/Videos
    _settle(ed.ws)
    assert ed.job_status(r["job"])["state"] == "done" and away.exists()
    with pytest.raises(ToolError, match="outside"):
        ed.ws.render_start("final", out=str(tmp_path_factory.mktemp("x") / "ai.mp4"))


def test_s3_a_file_named_like_a_partial_is_never_touched(tmp_path, monkeypatch):
    monkeypatch.setattr(jobmod, "_render", _Fake())
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    mine = tmp_path / "a.partial.mp4"
    mine.write_bytes(b"USER DATA")
    ed.start_render("final", out=tmp_path / "a.mp4")
    _settle(ed.ws)
    assert mine.read_bytes() == b"USER DATA" and (tmp_path / "a.mp4").exists()


def test_s10_one_draft_default_beside_the_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(jobmod, "_render", _Fake())
    _script(tmp_path, SCRIPT)
    ws = Workspace(tmp_path, allow_scripts=True)
    ws.scene_open("scene.py")
    ed = Editor(ws)
    assert ws.render_start()["path"] == str(tmp_path.resolve() / "scene.draft.mp4")
    assert ed.render_plan("draft")["path"] == str(tmp_path.resolve() / "scene.draft.mp4")
    _settle(ws)


def test_s11_a_render_never_writes_over_a_file_the_scene_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(jobmod, "_render", _Fake())
    shutil.copytree(EXAMPLES / "assets", tmp_path / "assets")
    shutil.copy2(EXAMPLES / "video_input.py", tmp_path / "v.py")
    ws = Workspace(tmp_path, allow_scripts=True)
    ws.scene_open("v.py")
    before = (tmp_path / "assets" / "clip.mp4").read_bytes()
    with ws._as_ai():
        with pytest.raises(ToolError, match="reads"):
            ws.render_start("final", out="assets/clip.mp4", overwrite=True)
        with pytest.raises(ToolError, match="reads"):
            ws.render("draft", out="assets/clip.mp4")
    assert (tmp_path / "assets" / "clip.mp4").read_bytes() == before


def test_s12_a_draft_open_in_a_player_goes_to_the_next_free_name(tmp_path, monkeypatch):
    monkeypatch.setattr(jobmod, "_render", _Fake())
    real = os.replace

    def locked(a, b):  # Windows: a player holds the old draft open
        if Path(b).name == "scene.draft.mp4":
            raise PermissionError(13, "the file is in use")
        return real(a, b)

    monkeypatch.setattr(jobmod.os, "replace", locked)
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    ed.draft_preview()
    _settle(ed.ws)
    st = ed.poll_job()
    assert st["state"] == "done" and st["open"].endswith("scene (2).draft.mp4")
    assert "could not be replaced" in st["note"] and (tmp_path / "scene (2).draft.mp4").exists()


# ---------------------------------------------------------------- S4: the web export


def test_s4_a_cancelled_export_keeps_the_file_that_was_there(tmp_path, monkeypatch):
    from phasor import Dot, Scene
    from phasor.gui.app import ExportJob

    s = Scene("web", passes=None)
    s.add(Dot(id="d"))
    out = tmp_path / "page.html"
    out.write_text("the page I had", encoding="utf-8")
    job = ExportJob(s.engine, "web", out, scene_dir=tmp_path)
    slow = "import sys, time; open(sys.argv[1], 'w').write('half'); time.sleep(60)"

    def command(scene, o=None):  # ``phasor web`` writing where it is told
        return [sys.executable, "-c", slow, str(o if o is not None else out)]

    monkeypatch.setattr(job, "command", command)
    job.start()
    end = time.monotonic() + 30

    def written() -> bool:  # the process has written something somewhere
        partial = [q for q in tmp_path.glob(".page.*.partial.html") if q.stat().st_size]
        return bool(partial) or out.read_text(encoding="utf-8") != "the page I had"

    while not written() and time.monotonic() < end:
        time.sleep(0.05)
    job.cancel()
    assert out.read_text(encoding="utf-8") == "the page I had"
    assert not list(tmp_path.glob(".*.partial.*"))


# ---------------------------------------------------------------- nits


def test_nit_mixed_line_endings_stay_as_they_were(tmp_path):
    mixed = SCRIPT.replace("\n", "\r\n", 2)
    p = _script(tmp_path, mixed)
    ed = Editor(open_workspace(str(p)))
    ed.set_param("pole", "r", "0.3")
    ed.save()
    assert p.read_bytes().decode("utf-8").startswith(mixed)


def test_nit_braces_in_a_picked_file_name_are_kept(tmp_path, tmp_path_factory):
    media = tmp_path_factory.mktemp("media")
    name = "sun{s}set{id}.png"
    shutil.copy2(EXAMPLES / "assets" / "sunset.png", media / name)
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    r = ed.add_footage(media / name)
    assert r["ok"] and f"s.background('media/{name}', id='background')" in ed.buffer


def test_nit_layout_rules_keep_their_order_and_opening_is_not_an_edit(tmp_path):
    """p8: an orphan rule moved to the end on open, and the scene showed unsaved at once."""
    p = _script(tmp_path, SCRIPT + BEGIN + "s.tweak_rule('gone above pole')\n"
                "s.tweak_rule('note above pole')\n" + END)  # fmt: skip
    ed = Editor(open_workspace(str(p)))
    assert [c.rule for c in ed.engine.overrides.constraints] == ["gone above pole",
                                                                  "note above pole"]  # fmt: skip
    assert not ed.dirty


def test_nit_a_copied_file_goes_away_when_the_addition_fails(tmp_path, tmp_path_factory):
    media = tmp_path_factory.mktemp("media")
    (media / "notes.txt").write_text("not a picture", encoding="utf-8")
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    r = ed.add_footage(media / "notes.txt", background=False)
    assert r["ok"] is False and not list((tmp_path / "media").glob("*"))


def test_nit_run_on_a_document_never_overwrites_an_existing_script(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("doc")
    ed = Editor(ws)
    (tmp_path / "doc.scene.py").write_text("# someone's script\n", encoding="utf-8")
    r = ed.run_code(ed.code()["text"])
    assert r["ok"] is False and "exists" in r["error"]
    assert (tmp_path / "doc.scene.py").read_text(encoding="utf-8") == "# someone's script\n"


def test_nit_structure_edits_on_a_script_say_where_they_belong(tmp_path):
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    r = ed.move_marker("go", 1.5)
    assert r["ok"] is False and "s.marker" in r["errors"][0]["message"]


# ---------------------------------------------------------------- re-audit (2026-11-25)


def test_r2_a_long_data_line_does_not_freeze_the_editor(tmp_path):
    """Re-audit S-2: on Python 3.12 ``tokenize`` is quadratic in line length; a 38 KB data
    line cost 3.3 s and about 1 GB per tweak. The markers are now found with the C parser."""
    data = "DATA = [" + ", ".join(str(i % 97) for i in range(10_000)) + "]\n"
    p = _script(tmp_path, SCRIPT + data)
    t0 = time.monotonic()
    ed = Editor(open_workspace(str(p)))
    for v in ("0.25", "0.3", "0.35"):
        assert ed.set_param("pole", "r", v)["ok"]
    assert ed.save()["saved"]
    assert time.monotonic() - t0 < 5.0  # was about 10 s on 3.12 (generous for slow CI)
    assert p.read_text(encoding="utf-8").startswith(SCRIPT + data)


def test_r2_undo_never_throws_away_typing_that_has_not_run(tmp_path):
    """Re-audit B-C: undo after a panel addition restored the text from before it,
    dropping code typed since; a later Redo overwrote newer typing."""
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    assert ed.add_notes(typed="C4 0 0.5", at="go")["ok"]
    ed.set_buffer(ed.buffer + "MINE = 'typed, not run yet'\n")
    r = ed.undo()
    assert r["ok"] is False and "has not run" in r["errors"][0]["message"]
    assert "MINE" in ed.buffer and "Notes(" in ed.buffer and ed.dirty
    assert ed.run()["ok"]  # once typed code ran, the addition is no longer undone by itself
    assert ed.undo().get("undone") is not True and "MINE" in ed.buffer
    # Redo is cleared by typing
    q = _script(tmp_path, SCRIPT, "two.py")
    ed = Editor(open_workspace(str(q)))
    assert ed.add_notes(typed="C4 0 0.5")["ok"] and ed.undo()["undone"]
    ed.set_buffer(ed.buffer + "LATER = 1\n")
    assert not ed.redo().get("redone") and "LATER" in ed.buffer


SWALLOW = SCRIPT + (
    "while True:\n    try:\n        while True:\n            pass\n"
    "    except BaseException:\n        pass\n"
)


def test_r2_a_script_that_swallows_cancel_is_let_go_and_the_file_restored(tmp_path, monkeypatch):
    """Re-audit S-1: a script catching ``RunCancelled`` kept ``busy`` set forever, its code
    stayed in the file, and the next poll raised a conflict about phasor's own write."""
    monkeypatch.setenv("PHASOR_RUN_TIMEOUT", "1")
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    t0 = time.monotonic()
    r = ed.run_code(SWALLOW)
    assert r["ok"] is False and "did not stop" in r["error"]
    assert time.monotonic() - t0 < 30
    assert ed.running is None and ed.ws.busy is None
    assert p.read_text(encoding="utf-8") == SCRIPT  # the last code that ran
    assert ed.poll_file() is None and ed.conflict is None  # our own write: no conflict
    assert ed.run_code(SCRIPT.replace("hello", "after"))["ok"]


def test_r2_opening_a_runaway_script_is_stopped(tmp_path, monkeypatch):
    """Re-audit S-1: opening a script ran it on the GUI thread with no timeout."""
    monkeypatch.setenv("PHASOR_RUN_TIMEOUT", "1")
    p = _script(tmp_path, SCRIPT + "while True:\n    pass\n")
    with pytest.raises(ToolError, match="not opened"):
        open_workspace(str(p))


def test_r2_an_outside_write_between_run_and_read_is_not_adopted(tmp_path):
    """Re-audit S-3: the worker ran whatever was on disk and the editor took it as its own."""
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    orig = ed.ws._load_script
    other = SCRIPT + "z = 'OTHER EDITOR'\n"

    def racy(path, allow):
        path.write_text(other, encoding="utf-8")
        return orig(path, allow)

    ed.ws._load_script = racy
    typed = SCRIPT + "MINE = 1\n"
    r = ed.run_code(typed)
    assert r["ok"] is False and r["conflict"] and ed.conflict is not None
    assert ed.buffer == typed


def test_r2_saving_a_deleted_script_asks_and_can_write_it_again(tmp_path):
    """Re-audit S-4: Save on a deleted script did nothing and said nothing."""
    p = _script(tmp_path, SCRIPT)
    ed = Editor(open_workspace(str(p)))
    p.unlink()
    ed.set_buffer(ed.buffer + "y = 2\n")
    r = ed.save()
    assert r["ok"] is False and r["conflict"] and ed.conflict["deleted"]
    assert ed.resolve_conflict("mine")["recreated"] == str(p)
    assert p.read_text(encoding="utf-8").endswith("y = 2\n")


def test_r2_a_render_never_writes_over_a_frame_of_a_sequence(tmp_path, monkeypatch):
    """Re-audit S-5: the pattern was compared, not its frames."""
    monkeypatch.setattr(jobmod, "_render", _Fake())
    shutil.copytree(EXAMPLES / "assets" / "frames", tmp_path / "shots")
    first = sorted((tmp_path / "shots").glob("*.png"))[1]
    stem, number = first.stem.rsplit("_", 1)
    digits = len(number)
    seq = f's.add(Image(id="seq", path="shots/{stem}_%0{digits}d.png"))\n'
    _script(tmp_path, SCRIPT.replace("Text\n", "Text, Image\n") + seq)
    ws = Workspace(tmp_path, allow_scripts=True)
    ws.scene_open("scene.py")
    before = first.read_bytes()
    with pytest.raises(ToolError, match="reads"):
        ws.render_start("draft", out=f"shots/{first.name}", frame=0.0, overwrite=True)
    with pytest.raises(ToolError, match="reads"):  # a frame not there yet: the pattern's too
        ws.render_start("draft", out=f"shots/{stem}_{'9' * digits}.png", frame=0.0)
    assert first.read_bytes() == before


def test_r2_a_render_that_cannot_write_leaves_no_zombie_job(tmp_path):
    """Re-audit S-6: an OSError making the partial file registered a job that never ended;
    after two, no render started; the raw OSError dropped an AI's connection."""
    _script(tmp_path, SCRIPT)
    ws = Workspace(tmp_path, allow_scripts=True)
    ws.scene_open("scene.py")
    for k in range(3):
        with pytest.raises(ToolError, match="cannot write there"):
            ws.render_start("draft", out=f"{k}{'x' * 300}.mp4")
    assert ws.render_status() == {"jobs": []}


def test_r2_the_ai_socket_answers_every_request(tmp_path):
    import json
    import socket

    from phasor.workspace.remote import RemoteServer

    ws = Workspace(tmp_path)
    ws.scene_new("x")
    server = RemoteServer(ws).start()
    try:
        with socket.create_connection(server.address, timeout=10) as c:
            f = c.makefile("rwb")
            for req in (
                {"token": server.token, "method": ["unhashable"]},
                {"token": server.token, "method": "render_start",
                 "params": {"out": "a" * 300 + ".mp4"}},
            ):  # fmt: skip
                f.write((json.dumps(req) + "\n").encode())
                f.flush()
                reply = json.loads(f.readline())
                assert reply["ok"] is False and reply["error"]
    finally:
        server.stop()


def test_r2_the_block_says_what_happens_and_run_keeps_what_the_person_wrote(tmp_path):
    """Re-audit S-8: the header said "edit freely" but Run rewrote the block."""
    from phasor.core.pycode import TWEAKS_HEADER

    assert "rewrites this block" in TWEAKS_HEADER and "edit freely" not in TWEAKS_HEADER
    block = BEGIN + "# chosen to match chapter 2\ns.tweak('pole.r', 0.1 * 3)\n" + END
    p = _script(tmp_path, SCRIPT + block)
    ed = Editor(open_workspace(str(p)))
    assert not ed.dirty and ed.run()["ok"]
    assert p.read_text(encoding="utf-8") == SCRIPT + block  # nothing rewritten by a Run
    assert ed.save()["saved"] and p.read_text(encoding="utf-8") == SCRIPT + block


def test_r2_nits(tmp_path, monkeypatch):
    from phasor.workspace.tools import write_text

    # a temporary name of the person's is never used
    p = _script(tmp_path, SCRIPT)
    mine = tmp_path / ".scene.py.phasor-tmp"
    mine.write_text("mine", encoding="utf-8")
    write_text(p, SCRIPT + "x = 1\n")
    assert mine.read_text(encoding="utf-8") == "mine"
    # a sound may start at a marker named inf
    ws = Workspace(tmp_path)
    ws.scene_new("s")
    ws.scene_edit({"op": "set_marker", "name": "inf", "t": 1.0})
    shutil.copy2(EXAMPLES / "assets" / "chime.wav", tmp_path / "chime.wav")
    sound = {"id": "c", "type": "sound", "params": {"path": "chime.wav", "start": "inf"}}
    ws.scene_edit({"op": "add_object", "object": sound})
    assert ws.engine.state_at(0.0)["c"].params.start == "inf"
    # a file made at the target while the render ran is the person's: kept
    fake = _Fake(gate=True)
    monkeypatch.setattr(jobmod, "_render", fake)
    r = ws.render_start("final", out="late.mp4")
    (tmp_path / "late.mp4").write_bytes(b"made meanwhile")
    fake.gate.set()
    _settle(ws)
    assert (tmp_path / "late.mp4").read_bytes() == b"made meanwhile"
    assert ws.render_status(r["job"])["path"].endswith("late (2).mp4")
