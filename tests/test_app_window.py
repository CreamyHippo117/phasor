"""Milestone 25: the code-first window on Qt's offscreen platform (the logic is tested in
``test_app_code_first.py``; here the widgets show it and forward input to it).

Phase 2's exit tests: an outside edit reloads; a conflict asks; an error keeps the last good
scene and marks its line. Also: the script is on the left and the preview on the right,
selecting an object shows its line and clicking a line selects it, the Draft preview button
renders in the background and plays the result, the Render dialog's Cancel leaves nothing,
the panels add footage, sound and notes, the app asks once before running a script it did
not write, and without OpenGL the preview says so while everything else works."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", reason="the gui extra is not installed")

from PySide6.QtWidgets import QSplitter

from phasor.gui.window import open_editor
from phasor.workspace import jobs as jobmod

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SCRIPT = """from phasor import Dot, Scene, Text

s = Scene("window", passes=None)
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
def win(scene):
    _app, w, _server = open_editor(str(scene), serve_ai=False, show=False)
    w.timer.stop()  # the tests tick it by hand
    w.resize(1200, 760)
    yield w
    w.ed.ws._jobs.cancel_all()
    w.close()


def _later(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def _poll(w, n: int = 1) -> None:
    for _ in range(n):
        w._poll_file()  # a reload runs on the worker thread
        w.wait_for_run()


def test_the_script_is_on_the_left_and_the_preview_on_the_right(win):
    split = win.centralWidget()
    assert isinstance(split, QSplitter) and split.widget(0) is win.left
    assert win.left.currentWidget() is win.code  # the script, or the Explorer's rows (M29)
    assert win.code.text.toPlainText() == SCRIPT
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles[:8] == ["Problems", "Inspector", "Looks", "Narration", "Sound and MIDI",
                          "Footage", "Sliders", "AI"]  # fmt: skip
    assert {"Run", "Draft preview", "Render...", "Save"} <= set(win.actions)
    assert not win.activity_dock.isVisible()  # under View, hidden, not deleted
    assert "phase 4" in win.ai.findChildren(type(win.code.title))[0].text()


def test_an_error_keeps_the_last_good_scene_and_marks_its_line(win, scene):
    win.code.text.setPlainText(SCRIPT + "s.add(Dot(id='bad', r=-1))\n")
    assert win.code.edited
    win.code.run()
    assert win.code.status.text().startswith("not run: line 8")
    assert win.ed.engine.document.object("bad") is None
    assert win.ed.engine.document.object("pole") is not None  # the last good scene
    marks = win.code.text.extraSelections()
    assert any(m.cursor.blockNumber() == 7 for m in marks)  # line 8, marked
    win.lint.refresh()
    assert win.lint.item(0).text().startswith("[error] script_error (line 8)")
    assert scene.read_text(encoding="utf-8") == SCRIPT
    win.code.text.setPlainText(SCRIPT.replace("hello", "fixed"))
    win.code.run()
    assert "ran:" in win.code.status.text()
    assert win.ed.engine.document.object("note").params["text"] == "fixed"


def test_an_outside_edit_reloads(win, scene):
    _later(scene, SCRIPT.replace("hello", "from claude code"))
    _poll(win)
    assert win.ed.engine.document.object("note").params["text"] == "from claude code"
    assert "from claude code" in win.code.text.toPlainText()
    assert "reloaded" in win.statusBar().currentMessage()


def test_a_conflict_asks_and_keeps_what_the_person_chooses(win, scene, monkeypatch):
    asked = []
    monkeypatch.setattr(win, "ask_conflict", lambda: asked.append(1) or "mine")
    win.code.text.setPlainText(SCRIPT.replace("hello", "mine"))
    _later(scene, SCRIPT.replace("hello", "theirs"))
    _poll(win, 3)
    assert asked == [1]  # asked once
    assert "mine" in win.code.text.toPlainText() and win.code.edited
    monkeypatch.setattr(win, "ask_conflict", lambda: asked.append(2) or "file")
    _later(scene, SCRIPT.replace("hello", "theirs again"))
    _poll(win)
    assert asked == [1, 2]
    assert win.ed.engine.document.object("note").params["text"] == "theirs again"
    assert "theirs again" in win.code.text.toPlainText() and not win.code.edited


def test_a_tweak_in_the_inspector_appears_in_the_script(win):
    win.ed.select("pole")
    win.inspector.refresh()
    win.inspector.fields["r"].setText("0.4")
    win.inspector.fields["r"].editingFinished.emit()
    assert "s.tweak('pole.r', 0.4)" in win.code.text.toPlainText()
    win.actions["Undo"].trigger()
    assert "pole.r" not in win.code.text.toPlainText()


def test_selecting_an_object_shows_its_line_and_a_line_selects_it(win):
    win.ed.select("note")
    win._after_selection()
    assert win.code.text.textCursor().blockNumber() == 4  # line 5
    win.code.line_clicked.emit(4)
    assert win.ed.selected == "pole"
    assert win.inspector.title.text().startswith("pole")


class _Fake:
    def __init__(self):
        self.gate = threading.Event()

    def __call__(self, engine, out, *, progress, cancel, preset, **kw):
        with open(out, "wb") as f:
            for k in range(8):
                if k == 4:
                    self.gate.wait(10)
                f.write(b"x")
                progress(k + 1, 8)
                if cancel.is_set():
                    from phasor.render.job import RenderCancelled

                    f.close()
                    Path(out).unlink()
                    raise RenderCancelled("cancelled")
        return {"frames": 8, "fps": 15.0, "size": (64, 36)}


def _tick_until(w, cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        w.tick()
        time.sleep(0.01)


def test_draft_preview_renders_in_the_background_and_plays_the_result(win, scene, monkeypatch):
    fake = _Fake()
    monkeypatch.setattr(jobmod, "_render", fake)
    played = []
    monkeypatch.setattr(win, "play_file", played.append)
    win.actions["Draft preview"].trigger()
    assert win.ed.job is not None
    _tick_until(win, lambda: win.job_bar.value() >= 4)
    assert not win.job_bar.isHidden() and not win.job_cancel.isHidden()
    fake.gate.set()
    _tick_until(win, lambda: played)
    assert played == [str(scene.with_name("scene.draft.mp4"))]
    assert win.job_bar.isHidden()
    # the Cancel button really stops one and leaves nothing behind
    fake.gate.clear()
    win.draft_preview()
    _tick_until(win, lambda: win.job_bar.value() >= 4)
    threading.Timer(0.3, fake.gate.set).start()
    win.job_cancel.click()
    assert win.ed.job is None and win.job_bar.isHidden()
    assert "cancelled" in win.statusBar().currentMessage()
    assert scene.with_name("scene.draft.mp4").read_bytes() == b"x" * 8  # the last good draft
    assert not list(scene.parent.glob(".*.partial.*"))


def test_the_render_dialog_chooses_the_quality_and_its_cancel_leaves_nothing(win, scene,
                                                                             monkeypatch):  # fmt: skip
    fake = _Fake()
    monkeypatch.setattr(jobmod, "_render", fake)
    win.open_render()
    dlg = win.render_dialog
    dlg.preset.setCurrentIndex(dlg.preset.findData("draft"))
    assert dlg.out.text().endswith("scene.draft.mp4") and "640" in dlg.size.text()
    assert dlg.fps.value() == 15.0
    dlg.preset.setCurrentIndex(dlg.preset.findData("final"))
    assert dlg.out.text().endswith("scene.mp4") and dlg.fps.value() == 60.0
    final = scene.with_name("scene.mp4")
    final.write_bytes(b"the real one")
    monkeypatch.setattr(dlg, "confirm_replace", lambda p: True)
    dlg.start()
    assert dlg.job is not None
    _tick_until(win, lambda: dlg.bar.value() >= 4)
    threading.Timer(0.3, fake.gate.set).start()
    dlg.cancel()
    assert "nothing was left behind" in dlg.status.text()
    for j in win.ed.ws._jobs.all():
        j.wait(10)
    assert final.read_bytes() == b"the real one"
    assert not list(scene.parent.glob(".*.partial.*"))


def test_the_panels_add_footage_sound_and_notes_to_the_script(win, scene, tmp_path_factory):
    media = tmp_path_factory.mktemp("media")
    for name in ("clip.mp4", "narration_demo.wav", "motif.mid"):
        shutil.copy2(EXAMPLES / "assets" / name, media / name)
    win.footage.add(str(media / "clip.mp4"))
    assert "s.background('media/clip.mp4', id='background')" in win.code.text.toPlainText()
    assert "copied" in win.footage.note.text()
    win.sound.refresh()
    win.sound.at.setCurrentText("go")
    win.sound.add_sound(str(media / "narration_demo.wav"))
    win.sound.typed.setPlainText("C4 0 0.5, G4 0.5 0.5")
    win.sound.add_typed()
    win.sound.add_midi(str(media / "motif.mid"))
    text = win.code.text.toPlainText()
    assert "s.add(Sound(" in text and "s.add(Notes(" in text and "motif.mid" in text
    r = win.sound.export(str(scene.with_name("out.mid")))
    assert r["notes"] > 0 and "wrote" in win.sound.note.text()
    win.narration.refresh()
    assert "No narration" in win.narration.now.text()
    assert not win.narration.record.isEnabled()  # said why: no audio input library
    win.narration.choose(str(media / "narration_demo.wav"))
    assert "narration_demo.wav" in win.narration.now.text()
    win.narration.show_speech()
    assert "stretches of speech" in win.narration.report.item(0).text()
    win.narration.preview()
    assert win.narration.report.count() >= 1


def test_without_opengl_the_preview_says_so_and_the_rest_works(win, monkeypatch):
    def no_gl(size):
        raise RuntimeError("no OpenGL here (tier C)")

    monkeypatch.setattr(win.ed, "frame", no_gl)
    win.refresh_all()
    assert "No preview here" in win.viewport.text() and "still work" in win.viewport.text()
    win.ed.select("pole")
    win.inspector.refresh()
    win.inspector.fields["r"].setText("0.3")
    win.inspector.fields["r"].editingFinished.emit()
    assert "s.tweak('pole.r', 0.3)" in win.code.text.toPlainText()


def test_a_script_the_app_did_not_write_asks_once_before_it_runs(tmp_path, monkeypatch):
    from phasor.gui.desktop import Desktop

    monkeypatch.setenv("PHASOR_CONFIG_DIR", str(tmp_path / "settings"))
    script = tmp_path / "someone.py"
    script.write_text(SCRIPT + "open('ran.txt', 'w').close()\n", encoding="utf-8")
    d = Desktop(serve_ai=False)
    asked = []
    answers = iter([False, True])
    monkeypatch.setattr(d, "confirm_run", lambda p: asked.append(p) or next(answers))
    monkeypatch.chdir(tmp_path)
    assert d.open(script) is None and not (tmp_path / "ran.txt").exists()  # said no
    w = d.open(script)
    assert w is not None and (tmp_path / "ran.txt").exists() and len(asked) == 2
    w.close()
    w = d.open(script)  # asked once: the next time it just opens
    assert w is not None and len(asked) == 2
    w.close()


def test_a_tweak_keeps_the_text_boxs_own_undo(win):
    """Audit S8: every tweak replaced the whole text, wiping the box's undo history."""
    box = win.code.text
    cur = box.textCursor()
    cur.movePosition(cur.MoveOperation.End)
    box.setTextCursor(cur)
    box.insertPlainText("# typing\n")
    assert box.document().isUndoAvailable()
    win.ed.set_param("pole", "r", "0.3")
    win.code.refresh()
    assert "s.tweak('pole.r', 0.3)" in box.toPlainText() and "# typing" in box.toPlainText()
    assert box.document().isUndoAvailable()


def test_the_run_button_keeps_the_window_alive_and_cancel_stops_a_runaway(win):
    """Audit S7: a ``while True`` froze every window and the attached AI."""
    win.code.QUICK = 0.1
    win.code.text.setPlainText(SCRIPT + "while True:\n    pass\n")
    win.code.run()
    assert not win.code.stop.isHidden() and "running" in win.code.status.text()
    win.tick()  # the window goes on
    win.code.stop.click()
    assert win.code.stop.isHidden() and "cancelled" in win.code.status.text()
    assert win.ed.engine.document.object("pole") is not None


def test_the_command_line_asks_before_running_a_script(tmp_path, monkeypatch):
    """Audit nit: ``phasor gui X.py --root DIR`` ran the script without the question."""
    from phasor.gui import cli, desktop

    monkeypatch.setenv("PHASOR_CONFIG_DIR", str(tmp_path / "settings"))
    script = tmp_path / "s.py"
    script.write_text(SCRIPT, encoding="utf-8")
    asked = []
    monkeypatch.setattr(desktop, "confirm_run", lambda p: asked.append(p) or False)
    assert cli._may_run(script) is False and len(asked) == 1
    monkeypatch.setattr(desktop, "confirm_run", lambda p: asked.append(p) or True)
    assert cli._may_run(script) and cli._may_run(script) and len(asked) == 2


@pytest.mark.parametrize("how", ["save", "run", "panel"])
@pytest.mark.parametrize("answer", ["mine", "file"])
def test_a_conflict_found_by_save_run_or_a_panel_is_asked(win, scene, monkeypatch,
                                                         tmp_path_factory, how, answer):  # fmt: skip
    """Audit follow-up: an outside edit between two polls, then Ctrl+S, Run or a panel: the
    write is refused and the window asks which version to keep (both answers work)."""
    asked = []
    monkeypatch.setattr(win, "ask_conflict", lambda: asked.append(how) or answer)
    win.code.text.setPlainText(SCRIPT.replace("hello", "mine"))  # typed, not saved
    _later(scene, SCRIPT.replace("hello", "theirs"))
    if how == "save":
        win.save_scene()
    elif how == "run":
        win.code.run()
    else:
        media = tmp_path_factory.mktemp("m")
        shutil.copy2(EXAMPLES / "assets" / "chime.wav", media / "chime.wav")
        win.sound.add_sound(str(media / "chime.wav"))
    win.tick()
    win.wait_for_run()
    assert asked == [how]
    win.tick()
    assert asked == [how]  # asked once
    if answer == "file":
        assert win.ed.engine.document.object("note").params["text"] == "theirs"
        assert not win.code.edited
    else:
        win.save_scene()
        win.wait_for_run()
        assert "mine" in scene.read_text(encoding="utf-8")


def test_ctrl_s_on_a_runaway_script_keeps_the_window_alive(win, scene):
    """Audit follow-up: Run on save went through the GUI thread."""
    win.code.QUICK = 0.1
    win.code.text.setPlainText(SCRIPT + "while True:\n    pass\n")
    win.save_scene()
    assert win.ed.running is not None and not win.code.stop.isHidden()
    win.tick()
    win.code.stop.click()
    assert win.ed.running is None and "cancelled" in win.code.status.text()
    assert scene.read_text(encoding="utf-8") == SCRIPT  # nothing that did not run is kept
