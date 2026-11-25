"""Milestone 29: the Explorer panel in the code-first window, on Qt's offscreen platform.

The logic is tested in ``test_explorer_panel.py``; here the widgets show it and forward
input: View > Explorer swaps the script for the rows, an explorer document opens in the
Explorer, typing in a row goes through the lock and commits after the debounce, an AI's edit
appears live, the transport plays without touching the document, the viewport's clicks
reach the Explorer, and Convert to script asks first. No OpenGL is needed: without it the
preview says so and everything else works."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", reason="the gui extra is not installed")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from phasor.gui.explorer import CONVERT_WARNING, DEBOUNCE, NEW, Immediate
from phasor.gui.window import open_editor
from phasor.workspace import explorer as X
from phasor.workspace.tools import Workspace
from tests.conftest import gl_available


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _doc(tmp_path, preset="van_der_pol"):
    w = Workspace(tmp_path)
    w.scene_new("lab", look="none")
    w.explorer_preset(preset)
    path = tmp_path / "lab.phasor.json"
    w.scene_save(str(path))
    return path


def _open(path, clock):
    _app, win, _server = open_editor(str(path), serve_ai=False, show=False)
    win.timer.stop()  # the tests tick it by hand
    win.resize(1200, 760)
    xp = win.xp
    xp.clock, xp.pool, xp.analysis_pool = clock, Immediate(), Immediate()
    xp.last_input = -1e9
    return win


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def win(tmp_path, clock):
    w = _open(_doc(tmp_path), clock)
    yield w
    w.close()


def test_an_explorer_document_opens_in_the_explorer_and_view_toggles_it(win):
    assert win.explorer_on and win.left.currentWidget() is win.explorer
    assert win.viewport.explorer is win.xp
    ids = [e["id"] for e in X.rows_of(win.ed.engine.document)]
    assert all(i in win.explorer.rows for i in ids) and NEW in win.explorer.rows
    win.actions["Explorer"].setChecked(False)
    assert win.left.currentWidget() is win.code and win.viewport.explorer is None
    win.actions["Explorer"].setChecked(True)
    assert win.explorer_on


def test_a_script_opens_in_the_code_view(tmp_path, clock):
    path = tmp_path / "s.py"
    path.write_text("from phasor import Dot, Scene\n\ns = Scene('s', passes=None)\n"
                    "s.add(Dot(id='d'))\n", encoding="utf-8")  # fmt: skip
    w = _open(path, clock)
    try:
        assert not w.explorer_on
    finally:
        w.close()


def _box(win, key=NEW):
    """The line edit the person types in (the list may have been rebuilt meanwhile)."""
    return win.explorer.rows[win.xp.focus or key].text


def test_typing_in_the_new_row_commits_after_the_debounce(win, clock):
    v0 = win.ed.version
    for k, ch in enumerate("a = 3"):
        box = _box(win)
        assert box is win.explorer.rows[NEW].text  # the row typed in keeps its widget
        box.insert(ch)  # textEdited, as typing does
        clock.t += 0.05
        win.tick()
        assert win.ed.version == v0, k
    clock.t += DEBOUNCE
    win.tick()
    assert win.ed.version == v0 + 1
    names = [e.get("name") for e in X.rows_of(win.ed.engine.document)]
    assert "a" in names
    assert win.xp.history_names()[-1] == "added a = 3"


def test_an_error_shows_under_the_row_in_plain_words(win, clock):
    box = win.explorer.rows[NEW].text
    box.insert("x' = sin(")
    clock.t += 1.0
    win.tick()
    err = win.explorer.rows[NEW].error
    assert not err.isHidden() and err.text()


def test_the_row_being_typed_is_locked_and_an_ai_edit_appears_live(win, clock):
    mu = next(e["id"] for e in X.rows_of(win.ed.engine.document) if e["kind"] == "variable")
    box = win.explorer.rows[mu].text
    box.setFocus()
    box.end(False)
    box.insert("5")
    win.tick()
    with win.ed.lock:
        r = win.ed.ws.explorer_edit([{"id": mu, "value": 2}])
    assert r["errors"][0]["code"] == "row_locked"
    with win.ed.lock:  # an attached AI adds a tracer meanwhile
        assert win.ed.ws.explorer_edit([{"text": "(1, 1)"}])["ok"]
    win.tick()
    assert any(e["kind"] == "point" for e in win.xp.rows())
    assert any(w.row["kind"] == "point" for w in win.explorer.rows.values())


def test_a_slider_drag_previews_and_its_release_is_one_step(win):
    row = next(w for w in win.explorer.rows.values() if w.slider is not None)
    v0 = win.ed.version
    row.slider.setSliderDown(True)
    row.slider.setSliderPosition(800)  # sliderMoved
    assert win.ed.version == v0 and win.xp.slider_preview
    row.slider.setSliderDown(False)  # sliderReleased
    assert win.ed.version == v0 + 1
    assert not win.xp.slider_preview


def test_the_transport_plays_without_touching_the_document(win):
    v0, d0 = win.ed.version, len(win.ed.engine._undo)
    win.actions["Play"].trigger()  # Space: the Explorer's transport in this view
    assert win.xp.playing
    for _ in range(60):
        win.tick()
    assert win.xp.playhead > 1.5
    win.explorer.reverse.click()
    win.explorer.restart.click()
    win.explorer.speed.setText("3")
    win.explorer.speed.returnPressed.emit()
    assert "3x normal" in win.explorer.readout.text()
    assert win.ed.version == v0 and len(win.ed.engine._undo) == d0


def _mouse(win, kind, x, y, button=Qt.MouseButton.LeftButton,
           mods=Qt.KeyboardModifier.NoModifier):  # fmt: skip
    vp = win.viewport
    pos = QPointF(x, y)
    buttons = button if kind != "release" else Qt.MouseButton.NoButton
    types = {"press": QMouseEvent.Type.MouseButtonPress,
             "move": QMouseEvent.Type.MouseMove,
             "release": QMouseEvent.Type.MouseButtonRelease,
             "double": QMouseEvent.Type.MouseButtonDblClick}  # fmt: skip
    ev = QMouseEvent(types[kind], pos, vp.mapToGlobal(QPoint(int(x), int(y))).toPointF(),
                     button, buttons, mods)  # fmt: skip
    QApplication.sendEvent(vp, ev)


def test_a_click_in_the_viewport_drops_a_tracer_and_a_drag_pans(win):
    vp = win.viewport
    vp.refresh()  # without OpenGL it says so, and clicks still map
    assert vp._size[0] > 0
    d0 = len(win.ed.engine._undo)
    cx, cy = vp.width() / 2, vp.height() / 2
    _mouse(win, "press", cx + 40, cy - 30)
    _mouse(win, "release", cx + 40, cy - 30)
    assert len(win.ed.engine._undo) == d0 + 1
    assert win.xp.history_names()[-1] == "added tracer"
    _mouse(win, "press", cx, cy)
    _mouse(win, "move", cx + 60, cy)
    assert len(win.ed.engine._undo) == d0 + 1  # the camera while the mouse is down: session
    _mouse(win, "release", cx + 60, cy)
    assert len(win.ed.engine._undo) == d0 + 2
    assert win.xp.history_names()[-1].startswith("camera pan")


def test_the_add_menu_and_the_palette_are_filled_from_the_explorer(win):
    win.explorer.fill_add_menu()
    labels = [a.text() for a in win.explorer.add_menu.actions()]
    assert labels[0].startswith("tracer") and "Insert" in labels
    disk = next(a for a in win.explorer.add_menu.actions() if a.text().startswith("disk"))
    d0 = len(win.ed.engine._undo)
    disk.trigger()
    assert len(win.ed.engine._undo) == d0 + 1
    assert any(w.row.get("is_object") for w in win.explorer.rows.values())


def test_presets_and_the_analysis_tab(win):
    idx = win.explorer.presets.findData("lorenz")
    win.explorer.presets.activated.emit(idx)
    assert win.xp.space() == "R3" and win.explorer.r3.isChecked()
    assert win.explorer.analysis.count() >= 3
    win.explorer.lyap.click()
    assert any("lyapunov" in win.explorer.analysis.item(i).text()
               for i in range(win.explorer.analysis.count()))  # fmt: skip
    assert win.explorer.ai_notes.count() >= 1


def test_export_dialog_checks_the_question(win):
    d = win.explorer.open_export()
    assert "from tau 0 to 30 in 20 s" in d.plan.text() and d.go.isEnabled()
    d.seconds.setText("zero")
    assert not d.go.isEnabled() and "Cannot export" in d.plan.text()
    d.close()


def test_convert_to_script_asks_first(win, monkeypatch):
    asked = []
    monkeypatch.setattr(win, "confirm_convert", lambda: asked.append(1) or False)
    win.explorer.convert.click()
    assert asked and not win.ed.is_script
    assert "one-way" in CONVERT_WARNING
    monkeypatch.setattr(win, "confirm_convert", lambda: True)
    win.explorer.convert.click()
    win.wait_for_run()
    assert win.ed.is_script and win.xp.read_only
    assert win.explorer.rows[next(iter(win.explorer.rows))].text.isReadOnly()
    assert not win.explorer.add_button.isEnabled()


def test_without_opengl_the_preview_says_so(win, monkeypatch):
    def no_gl(size):
        raise RuntimeError("no OpenGL here")

    monkeypatch.setattr(win.xp, "frame", no_gl)
    win.viewport.refresh()
    assert "No preview here" in win.viewport.text()
    win.explorer.refresh()  # the rows still draw


def test_closing_lets_go_of_a_row_being_typed(tmp_path, clock):
    w = _open(_doc(tmp_path), clock)
    mu = next(e["id"] for e in X.rows_of(w.ed.engine.document) if e["kind"] == "variable")
    w.xp.type_row(mu, "mu = 2")
    assert w.ed.ws._drafts
    w.close()
    assert not w.ed.ws._drafts
    assert w.ed.engine.document.object(mu).params["value"] == 2  # committed as it was left


def test_save_and_closing_the_desktop_window_keep_a_row_typed_just_now(tmp_path, monkeypatch):
    """The debounce has not passed: Ctrl+S, closing (the save question) and R2/R3 commit
    the row first, so it is saved, asked about, never lost."""
    from PySide6.QtWidgets import QMessageBox

    from phasor.gui.desktop import Desktop

    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    for k in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(k, str(home))
    monkeypatch.setenv("PHASOR_CONFIG_DIR", str(home / "settings"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "share"))
    asked = []
    monkeypatch.setattr(QMessageBox, "exec",
                        lambda box: asked.append(1) or QMessageBox.StandardButton.Save)  # fmt: skip
    path = _doc(tmp_path)
    d = Desktop(serve_ai=False)
    w = d.open(path)
    w.timer.stop()
    w.xp.pool = Immediate()
    mu = next(e["id"] for e in X.rows_of(w.ed.engine.document) if e["kind"] == "variable")
    w.xp.type_row(mu, "mu = 3")
    w.save_scene()  # Ctrl+S inside the debounce
    saved = Workspace(tmp_path)
    saved.scene_open(str(path))
    assert saved.engine.document.object(mu).params["value"] == 3
    w.xp.type_row(mu, "mu = 4")
    assert w.close()
    assert asked == [1]  # the typed row made it dirty: the question was asked
    saved.scene_open(str(path))
    assert saved.engine.document.object(mu).params["value"] == 4


def test_export_has_its_own_file_and_asks_before_replacing(win, monkeypatch):
    d = win.explorer.open_export()
    out = d.out.text()
    assert out.endswith(".explore.mp4")
    assert out != str(win.ed.default_output(".mp4"))  # never the real render's name
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(b"an earlier video")
    asked = []
    monkeypatch.setattr(d, "confirm_replace", lambda p: asked.append(p) or False)
    r = d.start()
    assert asked and r["ok"] is False
    assert Path(out).read_bytes() == b"an earlier video"
    d.close()


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_export_renders_the_fork_to_a_video(tmp_path, clock):
    pytest.importorskip("av", reason="the encode extra is not installed")
    import time

    w = _open(_doc(tmp_path), clock)
    try:
        w.xp.add("disk")
        v0 = w.ed.version
        out = tmp_path / "flow.mp4"
        r = w.xp.export(out, 0, 2, 0.3, "draft")
        assert r["ok"], r
        end = time.monotonic() + 600  # generous: llvmpipe on a shared box; no timing assert
        while w.ed.ws.render_status(r["job"])["state"] == "running":
            assert time.monotonic() < end, "the export did not finish"
            time.sleep(0.05)
        st = w.ed.ws.render_status(r["job"])
        assert st["state"] == "done", st
        assert out.exists() and out.stat().st_size > 0
        assert w.ed.version == v0  # the document was not changed
        assert w.ed.engine.document.object("van_der_pol").params.get("rate", 0) == 0
    finally:
        w.ed.ws._jobs.cancel_all()
        w.close()


def test_a_commit_keeps_the_cursor_and_the_line_edit(win, clock):
    """Audit S3: the debounce commit must not move the cursor or replace the line edit (its
    own undo goes with it)."""
    mu = next(e["id"] for e in X.rows_of(win.ed.engine.document) if e["kind"] == "variable")
    box = win.explorer.rows[mu].text
    box.setFocus()
    box.setCursorPosition(5)  # after "mu = "
    box.insert("2")  # "mu = 21"
    assert box.cursorPosition() == 6
    clock.t += 1.5
    win.tick()
    assert win.xp.history_names()[-1] == "mu = 21"  # committed
    now = win.explorer.rows[win.xp.focus].text
    assert now is box and box.cursorPosition() == 6
    assert box.isUndoAvailable()


def test_closing_the_window_commits_a_wheel_zoom_at_rest(tmp_path, clock):
    """Audit nit: closing settles (a typed row and a pending wheel zoom are committed)."""
    w = _open(_doc(tmp_path), clock)
    d0 = len(w.ed.engine._undo)
    w.xp.wheel(1.5)
    w.close()
    assert len(w.ed.engine._undo) == d0 + 1
