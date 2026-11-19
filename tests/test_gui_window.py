"""Milestone 10: the Qt window, driven headless (Qt's offscreen platform). The logic is
tested in ``test_gui_controller.py``; here: the widgets show it and forward input to it."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets", reason="the gui extra is not installed")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from phasor.gui.window import open_editor  # noqa: E402
from phasor.workspace.remote import RemoteWorkspace  # noqa: E402
from tests.conftest import gl_available  # noqa: E402

SCENE = {
    "schema_version": 5,
    "meta": {"title": "window test"},
    "objects": [
        {"id": "pole", "type": "dot", "params": {"pos": [1.0, 0.5, 0.0], "r": 0.25}},
        {"id": "far", "type": "dot", "params": {"pos": [40.0, 0.0, 0.0]}},
    ],
    "timeline": {"markers": {"go": 1.0, "end": 2.0}},
}


@pytest.fixture
def window(tmp_path):
    import json

    path = tmp_path / "w.phasor.json"
    path.write_text(json.dumps(SCENE))
    app, win, server = open_editor(str(path), show=False)
    win.resize(1000, 640)
    yield app, win, server, tmp_path
    win.timer.stop()
    server.stop()
    win.close()


def test_the_panels_show_the_scene(window):
    app, win, _server, _root = window
    app.processEvents()
    assert win.windowTitle() == "phasor: window test"
    assert set(win.timeline.data["markers"]) == {"go", "end"}
    texts = [win.lint.item(i).text() for i in range(win.lint.count())]
    assert any("far" in t or "off_screen" in t or "offscreen" in t for t in texts), texts
    assert "phasor mcp --attach" in win.statusBar().currentMessage()
    assert win.inspector.title.text().startswith("Nothing selected")
    assert win.looks.look.count() >= 4 and win.looks.beat.count() == 2


def test_typing_in_the_inspector_edits_the_scene(window):
    _app, win, _server, _root = window
    win.ed.select("pole")
    win.inspector.refresh()
    field = win.inspector.fields["r"]
    field.setText("0.4")
    field.editingFinished.emit()
    assert win.ed.engine.document.object("pole").params["r"] == 0.4
    win.inspector.fields["r"].setText("-5")  # refused: nothing changes, the reason is shown
    win.inspector.fields["r"].editingFinished.emit()
    assert win.ed.engine.document.object("pole").params["r"] == 0.4
    assert "Not applied" in win.statusBar().currentMessage()
    win.inspector.roles["color"].setCurrentText("highlight")
    assert win.ed.engine.state_at(0.0)["pole"].roles["color"] == "highlight"


def test_clicking_an_issue_jumps_to_it(window):
    _app, win, _server, _root = window
    row = next(i for i in range(win.lint.count())
               if "far" in str(win.lint.item(i).data(Qt.ItemDataRole.UserRole)))  # fmt: skip
    win.lint.itemClicked.emit(win.lint.item(row))
    assert win.ed.selected == "far"
    assert win.inspector.title.text().startswith("far")


def test_scrubbing_and_playing_move_the_playhead(window):
    _app, win, _server, _root = window
    win.timeline.scrubbed.emit(1.5)
    assert win.ed.t == pytest.approx(1.5)
    win.ed.set_time(0.0)
    win.toggle_play()
    for _ in range(6):
        win.tick()
    assert win.ed.t > 0.15 and win.playing


def test_an_attached_ai_edit_shows_up_in_the_open_window(window):
    """The exit demo's other half: the AI edits through the socket; the window redraws."""
    _app, win, _server, root = window
    win.ed.select("pole")
    win.refresh_all()
    remote = RemoteWorkspace.from_handoff(root)
    r = remote.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.33})
    assert r["ok"]
    win.tick()  # the timer notices the new version
    assert win.inspector.fields["r"].text() == "0.33"
    remote.close()


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_viewport_shows_the_picture_and_selects_by_click(window):
    _app, win, _server, _root = window
    win.viewport.resize(640, 360)
    win.viewport.refresh()
    assert win.viewport.error is None and not win.viewport.pixmap().isNull()
    fw, fh = win.viewport._size
    cam = win.ed.engine.state_at(0.0).camera
    ndc = cam.project([[1.0, 0.5, 0.0]])[0]
    u = (ndc[0] + 1) / 2 * fw + (win.viewport.width() - fw) / 2
    v = (1 - ndc[1]) / 2 * fh + (win.viewport.height() - fh) / 2
    for kind in (QMouseEvent.Type.MouseButtonPress, QMouseEvent.Type.MouseButtonRelease):
        ev = QMouseEvent(kind, QPointF(u, v), QPointF(u, v), Qt.MouseButton.LeftButton,
                         Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)  # fmt: skip
        (win.viewport.mousePressEvent if kind == QMouseEvent.Type.MouseButtonPress
         else win.viewport.mouseReleaseEvent)(ev)  # fmt: skip
    assert win.ed.selected == "pole"
    assert win.inspector.title.text().startswith("pole")
