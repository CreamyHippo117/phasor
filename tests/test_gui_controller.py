"""Milestone 10: the editor's logic, without Qt. Everything the window does goes through
``phasor.gui.controller.Editor``; the window only draws it (``test_gui_window.py``)."""

from __future__ import annotations

import pytest

from phasor.gui.controller import Editor
from phasor.workspace import Workspace
from tests.conftest import gl_available

POLE = {"op": "add_object", "object": {"id": "pole", "type": "dot",
                                       "params": {"pos": [1.0, 0.5, 0.0], "r": 0.2}}}  # fmt: skip


@pytest.fixture
def ed(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("edit me")
    e = Editor(ws)
    e.ws.scene_edit(POLE)
    e.ws.scene_edit({"op": "set_marker", "name": "go", "t": 1.0})
    e.ws.animate("morph", "go", path="pole.r", value=0.4)
    return e


def test_time_follows_the_timeline_and_markers(ed):
    tl = ed.timeline()
    assert tl["markers"] == {"go": 1.0} and "pole.r" in tl["lanes"]
    start, end = tl["lanes"]["pole.r"][0]
    assert start == pytest.approx(1.0) and end > start
    ed.set_time(99.0)
    assert ed.t == pytest.approx(tl["duration"])  # clamped to the scene
    ed.set_time(-3)
    assert ed.t == 0.0
    ed.jump_to_marker("go")
    assert ed.t == 1.0
    ed.set_time(0.0)
    ed.play_step(0.25)  # what the play timer calls each tick
    assert ed.t == pytest.approx(0.25)


def test_inspector_shows_params_semantics_and_style_and_edits_them(ed):
    ed.select("pole")
    panel = ed.inspector()
    assert panel["id"] == "pole" and panel["type"] == "dot"
    assert panel["params"]["r"] == 0.2 and panel["semantic"]["kind"]
    assert "color" in panel["style"]  # the role names, which are what can be edited
    v = ed.version
    r = ed.set_param("pole", "r", "0.3")  # typed text, parsed as JSON
    assert r["ok"] and ed.engine.document.object("pole").params["r"] == 0.3
    assert ed.version == v + 1
    r = ed.set_param("pole", "pos", "[0, 0, 0]")
    assert r["ok"] and ed.inspector()["params"]["pos"] == (0.0, 0.0, 0.0)
    bad = ed.set_param("pole", "r", "-1")
    assert not bad["ok"] and ed.engine.document.object("pole").params["r"] == 0.3  # unchanged
    assert ed.set_style("pole", "color", "highlight")["ok"]
    assert ed.inspector()["style"]["color"] == "highlight"
    ed.undo()
    assert ed.inspector()["style"]["color"] != "highlight"
    assert [h["what"] for h in ed.history][:2] == ["pole.r = 0.3", "pole.pos = [0, 0, 0]"]


def test_lint_items_jump_to_their_object_and_time(ed):
    ed.ws.scene_edit({"op": "add_object", "object": {"id": "far", "type": "dot",
                                                     "params": {"pos": [40.0, 0.0, 0.0]}}})  # fmt: skip
    items = ed.lint_items()
    off = next(i for i in items if "far" in i["ids"])
    ed.jump(off)
    assert ed.selected == "far" and ed.t == pytest.approx(off["times"][0] if off["times"] else 0)


def test_external_edits_are_noticed(ed):
    """The AI (attached over the socket) edits the same workspace: the window must redraw."""
    ed.changed()  # (the fixture's own setup)
    assert not ed.changed()
    ed.ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.25})
    assert ed.changed() and not ed.changed()  # reported once


def test_drag_moves_an_object_to_where_the_mouse_is(ed):
    """2D: the viewport point maps back to the plane the object lives in."""
    ed.select("pole")
    w, h = 800, 450
    st = ed.engine.state_at(0.0)
    cam = st.camera
    ndc = cam.project([[2.0, -1.0, 0.0]])[0]
    u, v = (ndc[0] + 1) / 2 * w, (1 - ndc[1]) / 2 * h
    x, y, _z = ed.screen_to_world(u, v, (w, h), z=0.0)
    assert (x, y) == pytest.approx((2.0, -1.0), abs=1e-6)
    r = ed.drag_to("pole", u, v, (w, h))
    assert r["ok"] and ed.engine.document.object("pole").params["pos"] == pytest.approx(
        (2.0, -1.0, 0.0), abs=1e-6
    )


def test_looks_can_be_switched_and_undone(ed):
    assert "prism" in ed.looks()
    v = ed.version
    r = ed.apply_look("prism")
    assert r["ok"] and ed.engine.document.meta.theme == "prism"
    assert ed.engine.document.meta.look == "prism" and ed.engine.document.meta.life == "lively"
    assert ed.version == v + 1  # one undo step
    ed.undo()
    assert ed.engine.document.meta.theme == "midnight_glow"


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_viewport_frames_and_click_to_select(ed):
    img = ed.frame((320, 180))
    assert img.shape == (180, 320, 3) and img.dtype.name == "uint8"
    st = ed.engine.state_at(0.0)
    ndc = st.camera.project([[1.0, 0.5, 0.0]])[0]
    u, v = (ndc[0] + 1) / 2 * 320, (1 - ndc[1]) / 2 * 180
    assert ed.select_at(u, v, (320, 180)) == "pole"
    assert ed.select_at(2, 2, (320, 180)) is None and ed.selected is None
    assert ed.frame((320, 180)) is ed.frame((320, 180))  # cached until something changes
    ed.set_time(0.5)
    assert ed.frame((320, 180)) is not img


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_variant_picker(ed):
    ed.set_style("pole", "glow", "soft")  # something a calmer or bolder look can change
    r = ed.variants("go")
    names = [v["name"] for v in r["variants"]]
    assert len(names) >= 2 and r["image"]
    v = ed.version
    out = ed.choose("calmer")
    assert out["applied"] and ed.version == v + 1  # one undo step
    assert out["recorded"]["by"] == "person" and out["recorded"]["winner"] == "calmer"
    with pytest.raises(Exception, match="no variant"):
        ed.choose("louder")
