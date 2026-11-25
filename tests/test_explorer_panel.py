"""Milestone 29: the Explorer panel's logic (``gui/explorer.py``), tested directly, no Qt.

The panel is a view of the one document: its rows are ``rows_of`` of the document, every
change is one patch through the shared ``Workspace`` under the editor's lock with a readable
name, and the session (playhead, a row mid-typing, a slider mid-drag, the camera while the
mouse is down) never reaches the document or its history. Time is an injected clock and
the reader runs on an injected executor: no test sleeps."""

from __future__ import annotations

import ast
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from phasor.gui.controller import Editor
from phasor.gui.explorer import (
    CONVERTED,
    DEBOUNCE,
    NEW,
    SETTLE,
    WHEEL_IDLE,
    ZOOM_SECONDS,
    Explorer,
    Immediate,
    classify,
    marks_ops,
    palette,
)
from phasor.workspace import explorer as X
from phasor.workspace.tools import Workspace
from tests.conftest import gl_available

SIZE = (640, 360)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, s: float) -> None:
        self.t += s


class Held:
    """An executor whose jobs run only when the test says (answers can come late)."""

    def __init__(self) -> None:
        self.jobs: list[tuple[Future, object, tuple]] = []

    def submit(self, fn, *args):
        f: Future = Future()
        self.jobs.append((f, fn, args))
        return f

    def run(self, k: int) -> None:
        f, fn, args = self.jobs[k]
        f.set_result(fn(*args))

    def shutdown(self, wait: bool = True) -> None:
        pass


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.scene_new("lab", look="none")
    return w


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ex(ws, clock):
    ed = Editor(ws, threading.RLock())
    e = Explorer(ed, clock=clock, pool=Immediate(), analysis_pool=Immediate())
    yield e
    e.close()


def _depth(ex) -> int:
    return len(ex.ed.engine._undo)


def _rows(ex) -> list[dict]:
    return X.rows_of(ex.ed.engine.document)


def _typed(ex, clock, key, text, step=0.05):
    """Type ``text`` a character at a time, ``step`` seconds apart, polling as the window
    does."""
    for k in range(1, len(text) + 1):
        ex.type_row(key, text[:k])
        clock.tick(step)
        ex.poll()


def _id_of(ex, kind, name=None):
    for e in _rows(ex):
        if e["kind"] == kind and (name is None or e.get("name") == name):
            return e["id"]
    raise AssertionError(f"no {kind} row")


# ---------------------------------------------------------------- the list is a view


def test_the_rows_are_the_documents_rows_and_follow_an_ai_edit(ex, ws):
    ex.load_preset("lorenz")
    assert [e["id"] for e in ex.rows()] == [e["id"] for e in _rows(ex)]
    ws.explorer_edit([{"text": "(1, 1, 1)"}])  # an attached AI, say
    assert [e["id"] for e in ex.rows()] == [e["id"] for e in _rows(ex)]
    assert ex.rows()[-1]["kind"] == "point"
    assert ex.space() == "R3"


def test_the_gui_logic_imports_no_qt():
    src = Path(__file__).resolve().parents[1] / "phasor" / "gui" / "explorer.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(n.startswith(("PySide6", "PyQt", "shiboken")) for n in names)


# ---------------------------------------------------------------- typing


def test_fast_typing_commits_once_after_the_debounce(ex, clock):
    v0, d0 = ex.ed.version, _depth(ex)
    _typed(ex, clock, NEW, "sigma = 10")
    assert ex.ed.version == v0 and _depth(ex) == d0  # every keystroke inside the debounce
    names = [e["name"] for e in _rows(ex) if e["kind"] == "variable"]
    assert names == []
    clock.tick(DEBOUNCE)
    assert ex.poll()
    assert _depth(ex) == d0 + 1  # one patch
    assert [e["name"] for e in _rows(ex) if e["kind"] == "variable"] == ["sigma"]
    assert ex.ed.history[-1]["what"] == "added sigma = 10"
    assert ex.focus == _id_of(ex, "variable", "sigma")  # the draft follows the new row


def test_typing_a_name_never_makes_its_prefixes_variables(ex, clock):
    _typed(ex, clock, NEW, "(sigma*y, -x)")
    clock.tick(DEBOUNCE)
    ex.poll()
    ex.leave()
    names = sorted(e["name"] for e in _rows(ex) if e["kind"] == "variable")
    assert names == ["sigma"]  # never s, si, sig, sigm


def test_a_pause_mid_name_does_not_make_a_slider_for_the_prefix(ex, clock):
    from phasor.gui.explorer import NAME_PAUSE

    v0 = ex.ed.version
    _typed(ex, clock, NEW, "(sig*y, -x)")
    clock.tick(DEBOUNCE + 0.1)  # longer than the debounce: but it would make a slider "sig"
    ex.poll()
    assert ex.ed.version == v0
    ex.type_row(NEW, "(sigma*y, -x)")
    clock.tick(NAME_PAUSE)
    ex.poll()  # a real pause: committed
    names = sorted(e["name"] for e in _rows(ex) if e["kind"] == "variable")
    assert names == ["sigma"]
    key = ex.focus
    ex.type_row(key, "(sigma*y, -2x)")  # no new name: the plain debounce
    clock.tick(DEBOUNCE)
    ex.poll()
    assert any(e["source"] == "(sigma*y, -2x)" for e in _rows(ex))


def test_a_late_answer_for_older_text_is_dropped(ws, clock):
    held = Held()
    ex = Explorer(Editor(ws), clock=clock, pool=held, analysis_pool=Immediate())
    ex.type_row(NEW, "a = (")
    ex.type_row(NEW, "a = 2")
    held.run(1)  # the newer answer first
    ex.poll()
    held.run(0)  # then the stale one
    ex.poll()
    assert ex.dropped == 1
    assert ex.rows()[-1]["error"] is None and ex.rows()[-1]["kind"] == "variable"


def test_retyping_a_row_previews_on_a_fork_then_commits_one_named_step(ex, clock, ws):
    ex.load_preset("lorenz")
    rho = _id_of(ex, "variable", "rho")
    v0, d0 = ex.ed.version, _depth(ex)
    ex.type_row(rho, "rho = 24.74")
    ex.poll()  # the answer is back: a preview, not a commit
    assert ex.ed.version == v0
    view = ex.view()
    assert view.document.object(rho).params["value"] == pytest.approx(24.74)
    assert ws.engine.document.object(rho).params["value"] == pytest.approx(28)
    clock.tick(DEBOUNCE)
    ex.poll()
    assert _depth(ex) == d0 + 1
    assert ws.engine.document.object(rho).params["value"] == pytest.approx(24.74)
    assert ex.history_names()[-1] == "rho = 24.74"


def test_a_row_that_does_not_read_shows_why_and_is_kept_only_when_left(ex, clock):
    v0 = ex.ed.version
    _typed(ex, clock, NEW, "x' = sin(")
    clock.tick(1.0)
    ex.poll()
    row = ex.rows()[-1]
    assert row["new"] and isinstance(row["error"], str) and row["error"]
    assert ex.ed.version == v0  # never committed by the debounce
    ex.leave()
    kept = [e for e in _rows(ex) if e["kind"] == "error"]
    assert len(kept) == 1 and kept[0]["source"] == "x' = sin("


HOSTILE = [
    "9^9^9", "x^9^9^9", "2^100000", "(" * 40 + "x" + ")" * 40, "x" + "+x" * 600,
    "x' = ", "\x00\x01", "x' = y\ny' = -x", "ẋ = ẋ", "rho = 1e400", "__import__('os')",
    "lambda: 0", "a" * 3000, "j = 2", "x' = t", "((((", "x' = 1/0", "", "   ", chr(0x2028),
    "x" * 5000, "😀 = 3",
]  # fmt: skip


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_text_never_raises(ex, clock, text):
    ex.type_row(NEW, text)
    clock.tick(DEBOUNCE + 0.01)
    ex.poll()
    ex.leave()
    for e in ex.rows():
        assert e["error"] is None or isinstance(e["error"], str)
    a = classify(text)
    assert a["error"] is None or isinstance(a["error"], str)


def test_undo_while_typing_takes_the_row_away_without_a_traceback(ex, clock):
    ex.type_row(NEW, "a = 3")
    clock.tick(DEBOUNCE)
    ex.poll()
    key = ex.focus
    ex.type_row(key, "a = 4")  # keeps typing in the row it made
    ex.ed.undo()  # Ctrl+Z: the row is gone
    clock.tick(DEBOUNCE)
    ex.poll()  # the commit is refused, plainly
    assert not any(e["kind"] == "variable" for e in _rows(ex))
    assert ex.message
    assert ex.type_row(key, "a = 5")["ok"] is False


# ---------------------------------------------------------------- typing against an AI


def test_the_row_being_typed_is_locked_against_the_ai_then_versions_guard_it(ex, clock, ws):
    ex.load_preset("lorenz")
    rho = _id_of(ex, "variable", "rho")
    ex.type_row(rho, "rho = 2")
    assert ex.rows()[[e["id"] for e in ex.rows()].index(rho)]["locked"]
    r = ws.explorer_edit([{"id": rho, "value": 99}])
    assert not r["ok"] and r["errors"][0]["code"] == "row_locked"
    r = ws.explorer_edit([{"text": "rho = 5"}])  # a new row rewriting it: refused too
    assert not r["ok"] and r["errors"][0]["code"] == "row_locked"
    r = ws.explorer_preset("rossler", True)
    assert not r["ok"] and r["errors"][0]["code"] == "row_locked"
    old = r["errors"][0]["version"]
    clock.tick(DEBOUNCE)
    ex.poll()  # the person's commit
    assert not any(e["locked"] for e in ex.rows())
    r = ws.explorer_edit([{"id": rho, "value": 7, "version": old}])
    assert not r["ok"] and r["errors"][0]["code"] == "row_changed"
    assert ws.engine.document.object(rho).params["value"] == pytest.approx(2)


def test_an_ai_edit_elsewhere_while_typing_drops_a_stale_preview(ex, clock, ws):
    ex.load_preset("lorenz")
    rho = _id_of(ex, "variable", "rho")
    ex.type_row(rho, "rho = 20")
    ex.poll()
    assert ex._typing_preview is not None
    beta = _id_of(ex, "variable", "beta")
    assert ws.explorer_edit([{"id": beta, "value": 3}])["ok"]
    ex.view()
    assert ex._typing_preview is None  # the document moved on: the picture is the document's
    clock.tick(DEBOUNCE)
    ex.poll()
    doc = ws.engine.document
    assert doc.object(rho).params["value"] == pytest.approx(20)
    assert doc.object(beta).params["value"] == pytest.approx(3)


def test_ai_edits_are_named_in_the_history(ex, ws, tmp_path):
    from phasor.workspace.remote import RemoteServer, RemoteWorkspace

    ex.load_preset("lorenz")
    server = RemoteServer(ws, ex.ed.lock).start()
    server.write_handoff()
    ai = RemoteWorkspace.from_handoff(tmp_path)
    try:
        ai.explorer_edit([{"id": "rho", "value": 24.74}])
        ai.explorer_edit([{"text": "(1, 2, 3)"}])
        ai.explorer_state()  # reading is not an edit
    finally:
        ai.close()
        server.stop()
    names = ex.history_names()
    assert names[-2:] == ["AI: rho = 24.74", "AI: added (1, 2, 3)"]
    assert names[0] == "preset lorenz"
    assert len(ex.ai_notes()) == 2


def test_describe_edit_never_raises():
    for rows in (None, [], [1, None], [{"id": 3, "text": 5}], [{"remove": "yes"}] * 9,
                 [{"id": "a", "min": 0, "max": 1}], [{"value": float("nan")}]):  # fmt: skip
        assert isinstance(X.describe_edit(rows), str)


# ---------------------------------------------------------------- row actions


def test_badge_visibility_colour_delete_and_order_are_one_step_each(ex):
    ex.load_preset("van_der_pol")
    ex.add("tracer")
    pt = _id_of(ex, "point")
    d0 = _depth(ex)
    ex.set_visible(pt, False)
    assert _rows(ex)[-1]["visible"] is False
    ex.set_color(pt, "danger")
    assert ex.ed.engine.document.object(pt).style["color"] == "danger"
    ex.move_row(pt, 0)
    assert _rows(ex)[0]["id"] == pt
    ex.type_row(NEW, "y = x^2")
    ex.leave()
    g = _id_of(ex, "graph")
    ex.set_kind(g, "implicit_curve")  # over the guess: one patch (a new object: a new id)
    row = next(e for e in ex.rows() if e["source"] == "y = x^2")
    assert row["kind"] == "implicit_curve" and row["badge"]["override"] == "implicit_curve"
    assert _depth(ex) == d0 + 5
    ex.delete(pt)
    assert _depth(ex) == d0 + 6
    assert ex.history_names()[-6:] == [f"hid {pt}", f"{pt} colour danger", f"moved {pt} to row 0",
                                       "added y = x^2", f"{g} as implicit_curve",
                                       f"removed {pt}"]  # fmt: skip
    assert ex.set_kind(pt, "nonsense")["ok"] is False
    assert ex.move_row(pt, "abc")["ok"] is False


# ---------------------------------------------------------------- sliders


def test_a_slider_drag_is_a_preview_and_its_release_is_one_named_step(ex, ws):
    ex.load_preset("lorenz")
    v0, d0 = ex.ed.version, _depth(ex)
    for x in np.linspace(28, 24.74, 30):
        assert ex.slider_drag("rho", x)["ok"]
    assert ex.ed.version == v0 and _depth(ex) == d0
    assert ex.view().document.object("rho").params["value"] == pytest.approx(24.74)
    assert ex.sliders()[1]["value"] == pytest.approx(24.74)
    ex.slider_release("rho")
    assert _depth(ex) == d0 + 1 and ex.history_names()[-1] == "rho = 24.74"
    assert ws.engine.document.object("rho").params["value"] == pytest.approx(24.74)


@pytest.mark.parametrize("bad", [{"min": "abc"}, {"max": "nan"}, {"min": "inf"},
                                 {"min": 50, "max": 1}, {"step": -1}, {"step": "1e9"},
                                 {"min": [1]}])  # fmt: skip
def test_bad_slider_ranges_are_refused_plainly(ex, bad):
    ex.load_preset("lorenz")
    v0 = ex.ed.version
    r = ex.slider_range("rho", **bad)
    assert r["ok"] is False and isinstance(r["errors"][0]["message"], str)
    assert ex.ed.version == v0
    assert ex.slider_drag("rho", "twenty")["ok"] is False


def test_slider_range_edit_and_typed_value(ex):
    ex.load_preset("lorenz")
    assert ex.slider_range("rho", min="0", max="50", step="0.5")["ok"]
    s = next(s for s in ex.sliders() if s["name"] == "rho")
    assert (s["min"], s["max"], s["step"]) == (0, 50, 0.5)
    assert ex.set_value("rho", "24.6")["ok"]
    assert ex.ed.engine.document.object("rho").params["value"] == pytest.approx(24.5)


def test_slider_play_is_session_and_keep_makes_a_track(ex, ws):
    ex.load_preset("lorenz")
    v0, d0 = ex.ed.version, _depth(ex)
    ex.slider_play("rho")
    seen = set()
    for _ in range(300):
        ex.advance(1 / 30)
        seen.add(round(ex.sliders()[1]["value"]))
    assert len(seen) > 20 and ex.ed.version == v0 and _depth(ex) == d0
    assert ex.slider_keep("rho")["ok"]
    assert "rho.value" in ws.engine.document.timeline.tracks
    assert ex.sliders()[1]["animated"] and not ex.sliders()[1]["playing"]
    assert ex.slider_drag("rho", 3)["ok"] is False  # a track moves it now
    s = ex.sliders()[1]
    st = ws.engine.state_at(2.0)  # half way up (linear, PLAY_SECONDS a crossing)
    assert st["rho"].params.value == pytest.approx((s["min"] + s["max"]) / 2)
    assert ex.clear_track("rho")["ok"]
    assert "rho.value" not in ws.engine.document.timeline.tracks


# ---------------------------------------------------------------- session state


def test_a_minute_of_play_and_scrubbing_leave_the_document_and_history_alone(ex):
    ex.load_preset("lorenz")
    ex.add("ball")
    v0, d0 = ex.ed.version, _depth(ex)
    ex.set_speed("2")
    assert ex.speed_text() == "2x normal"
    ex.play()
    for _ in range(1800):  # a minute at 30 Hz
        ex.advance(1 / 30)
    assert ex.playhead == pytest.approx(120.0)
    ex.toggle_reverse()
    assert ex.speed_text() == "2x normal, reversed"
    ex.scrub(3.0)
    ex.play()
    for _ in range(100):
        ex.advance(1 / 30)
    assert ex.playhead == 0.0 and not ex.playing  # reverse stops at the start
    ex.toggle_reverse()
    ex.step(1)
    ex.step(-3)
    ex.key("space")
    ex.key("r")
    assert ex.playhead == 0.0
    for bad in ("fast", "nan", None, -5):
        ex.set_speed(bad)
    assert ex.speed == 5.0
    assert ex.ed.version == v0 and _depth(ex) == d0 and ex.ed.history[-1]["what"] == "added ball"


def test_the_playhead_is_flow_time_past_the_resting_tau(ex):
    ex.load_preset("van_der_pol")
    sid = ex._systems()[0]
    ex._rows_edit([{"id": sid, "tau": 2.0}])
    ex.scrub(3.0)
    st = ex.view().state_at(ex.playhead)
    assert st[sid].impl.resolve(st).tau == pytest.approx(5.0)
    assert ex.flow_time() == pytest.approx(5.0)
    assert ex.ed.engine.document.object(sid).params.get("rate", 0) == 0  # the document's


def test_quality_is_interactive_while_dragging_and_full_after(ex, clock):
    ex.load_preset("lorenz")
    assert ex.render_plan(SIZE) == (SIZE, "full")
    ex.press(300, 200, SIZE)
    ex.move(340, 200, SIZE)
    assert ex.render_plan(SIZE) == ((320, 180), "interactive")
    ex.release(340, 200, SIZE)
    assert ex.render_plan(SIZE)[1] == "interactive"
    clock.tick(SETTLE + 0.01)
    assert ex.render_plan(SIZE) == (SIZE, "full")


# ---------------------------------------------------------------- the viewport


def test_orbit_while_the_mouse_is_down_is_session_and_one_step_on_release(ex):
    ex.load_preset("lorenz")
    v0, d0 = ex.ed.version, _depth(ex)
    az0 = ex._cam_now()["azimuth"]
    ex.press(300, 200, SIZE)
    for k in range(1, 40):
        ex.move(300 + k * 2, 200, SIZE)
    assert ex.ed.version == v0 and _depth(ex) == d0
    assert ex.view().document.cameras[0].params["azimuth"] == pytest.approx(az0 - 78 * 0.4)
    ex.release(378, 200, SIZE)
    assert _depth(ex) == d0 + 1 and ex.history_names()[-1].startswith("camera orbit")
    assert ex.ed.engine.document.cameras[0].params["azimuth"] == pytest.approx(az0 - 78 * 0.4)
    assert ex.camera_session == {}


def test_a_2d_drag_pans_and_the_wheel_commits_once_at_rest(ex, clock):
    ex.load_preset("van_der_pol")
    d0 = _depth(ex)
    ex.press(300, 200, SIZE)
    ex.move(250, 200, SIZE)
    ex.release(250, 200, SIZE)
    assert _depth(ex) == d0 + 1 and ex.history_names()[-1].startswith("camera pan")
    for _ in range(5):
        ex.wheel(1.1)
        clock.tick(0.05)
        ex.poll()
    assert _depth(ex) == d0 + 1
    clock.tick(WHEEL_IDLE)
    ex.poll()
    assert _depth(ex) == d0 + 2 and ex.history_names()[-1].startswith("camera zoom")
    ex.wheel("nan")  # never raises


def test_a_click_on_empty_space_drops_a_tracer_under_the_mouse_2d(ex):
    ex.load_preset("van_der_pol")
    d0 = _depth(ex)
    ex.press(400, 120, SIZE)
    ex.release(400, 120, SIZE)
    assert _depth(ex) == d0 + 1 and ex.history_names()[-1] == "added tracer"
    pt = _id_of(ex, "point")
    x0 = ex.ed.engine.document.object(pt).params["x0"]
    _, r = ex._system()
    px = ex._screen(r.world(np.asarray(x0)), SIZE)[0]
    assert np.allclose(px, (400, 120), atol=0.5)


def test_a_click_in_3d_drops_a_tracer_on_the_plane_through_the_target(ex):
    ex.load_preset("lorenz")
    ex.press(200, 140, SIZE)
    ex.release(200, 140, SIZE)
    pt = _id_of(ex, "point")
    x0 = np.asarray(ex.ed.engine.document.object(pt).params["x0"], dtype=float)
    _, r = ex._system()
    w = r.world(x0)[0]
    cam = ex._state().camera
    target = np.asarray(ex._cam_now().get("target", (0, 0, 0)), dtype=float)
    assert abs(float((w - target) @ np.asarray(cam.forward))) < 1e-3 * r.diag
    assert np.allclose(ex._screen(w, SIZE)[0], (200, 140), atol=0.5)


def test_shift_drag_draws_a_disk_and_its_handles_move_and_resize_it(ex):
    ex.load_preset("van_der_pol")
    ex._system()
    d0 = _depth(ex)
    ex.press(320, 180, SIZE, shift=True)
    c = ex._press["centre"]
    ex.move(360, 180, SIZE)
    ex.move(380, 180, SIZE)
    assert _depth(ex) == d0  # the disk mid-drag is the view's only
    assert any(o.type == "flow_set" for o in ex.view().document.objects)
    ex.release(380, 180, SIZE)
    assert _depth(ex) == d0 + 1 and ex.history_names()[-1] == "added disk"
    disk = next(o for o in ex.ed.engine.document.objects if o.type == "flow_set")
    assert np.allclose(disk.params["center"], c, atol=1e-5)
    edge = ex.coords_at(380, 180, SIZE)
    assert disk.params["size"] == pytest.approx(float(np.linalg.norm(edge - c)), rel=1e-3)
    assert any(e.get("is_object") and e["kind"] == "disk" for e in ex.rows())
    h = next(h for h in ex.handles(SIZE) if h["id"] == disk.id)
    ex.press(*h["at"], SIZE)
    ex.move(h["at"][0] + 30, h["at"][1] + 10, SIZE)
    ex.release(h["at"][0] + 30, h["at"][1] + 10, SIZE)
    assert ex.history_names()[-1] == f"moved {disk.id}"
    moved = ex.ed.engine.document.object(disk.id).params["center"]
    assert not np.allclose(moved, c)
    h = next(h for h in ex.handles(SIZE) if h["id"] == disk.id)
    ex.press(h["at"][0] + h["radius"], h["at"][1], SIZE)
    ex.move(h["at"][0] + h["radius"] + 25, h["at"][1], SIZE)
    ex.release(h["at"][0] + h["radius"] + 25, h["at"][1], SIZE)
    assert ex.history_names()[-1] == f"resized {disk.id}"
    assert ex.ed.engine.document.object(disk.id).params["size"] > disk.params["size"]
    assert _depth(ex) == d0 + 3


def test_dragging_a_tracer_retypes_its_row(ex):
    ex.load_preset("van_der_pol")
    ex.add("tracer", at=(1.0, 1.0))
    pt = _id_of(ex, "point")
    h = next(h for h in ex.handles(SIZE) if h["id"] == pt)
    ex.press(*h["at"], SIZE)
    ex.move(h["at"][0] + 40, h["at"][1], SIZE)
    ex.release(h["at"][0] + 40, h["at"][1], SIZE)
    row = next(e for e in _rows(ex) if e["id"] == pt)
    assert row["source"] != "(1, 1)" and row["source"].startswith("(")
    assert ex.history_names()[-1] == f"moved {pt}"


def _c_plus_pixel(ex):
    st = ex._state()
    sem = st["lorenz.eq"].impl.semantics(st)
    k = next(i for i, p in enumerate(sem["points"]) if p["at"][0] > 1)
    r = st["lorenz"].impl.resolve(st)
    return ex._screen(r.world(np.asarray(sem["points"][k]["at"])), SIZE)[0], sem["points"][k]


def test_clicking_an_equilibrium_opens_its_card_and_zoom_in_is_one_step(ex, clock):
    ex.load_preset("lorenz")
    (u, v), point = _c_plus_pixel(ex)
    d0 = _depth(ex)
    ex.press(u, v, SIZE)
    r = ex.release(u, v, SIZE)
    assert r["card"]["kind"] == point["kind"] and _depth(ex) == d0  # a card is not an edit
    assert r["card"]["lines"][1].startswith("eigenvalues")
    assert "i" in r["card"]["lines"][1]  # C+ at rho 28 has a complex pair
    ex.double_click(u, v, SIZE)
    clock.tick(ZOOM_SECONDS / 2)
    ex.poll()
    assert _depth(ex) == d0 and ex.camera_session  # in flight: session only
    clock.tick(ZOOM_SECONDS)
    ex.poll()
    assert _depth(ex) == d0 + 1
    assert ex.history_names()[-1].startswith("zoom in to E")
    doc = ex.ed.engine.document
    view = next(o for o in doc.objects if o.type == "flow_equilibrium_view")
    assert np.allclose(view.params["near"], point["at"])
    assert np.allclose(doc.cameras[0].params["target"], point["at"], atol=1e-5)
    assert ex.camera_session == {}


def test_the_analysis_tab(ex):
    ex.load_preset("lorenz")
    table = ex.equilibria_table()
    assert sum(1 for e in table if e["kind"] != "searched") == 3
    assert any(e["kind"] == "searched" for e in table)
    ex.add("tracer", at=(1, 1, 1))
    assert ex.lyapunov(T=20)["ok"]
    a = ex.analysis()["lyapunov"]
    assert a["state"] == "done" and not a["stale"]
    assert "largest_lyapunov" in a["result"]["seeds"][0]
    ex.slider_drag("rho", 20)
    ex.slider_release("rho")
    assert ex.analysis()["lyapunov"]["stale"]
    assert ex.sweep("rho", "0", "30", 5)["ok"]
    assert ex.analysis()["sweep"]["state"] == "done"
    ex.sweep("nobody", 0, 1, 5)
    assert ex.analysis()["sweep"]["state"] == "error"
    assert ex.sweep("rho", "x", 1)["ok"] is False
    assert ex.lyapunov(x0=["a", 1, 2])["ok"] is False


# ---------------------------------------------------------------- menus


def test_the_add_menu(ex):
    assert all(i["why"] for i in ex.add_items())  # nothing to add without a system
    assert ex.add("tracer")["ok"] is False
    ex.load_preset("van_der_pol")
    d0 = _depth(ex)
    for item in ("tracer", "disk", "source", "section"):
        assert ex.add(item)["ok"], item
    assert _depth(ex) == d0 + 4
    assert ex.history_names()[-4:] == ["added tracer", "added disk", "added source",
                                       "added section"]  # fmt: skip
    assert ex.add("obstacle")["ok"] is False and ex.add("bogus")["ok"] is False
    ex.load_preset("lorenz")
    keys = {i["key"]: i["why"] for i in ex.add_items()}
    assert "ball" in keys and keys["source"]
    assert ex.add("ball")["ok"]


def test_the_insert_palette_comes_from_the_registry_and_each_places_in_one_step(ws, clock):
    items = palette()
    types = {p["type"] for p in items}
    assert "flow_set" in types and not types & set(X.ROW_TYPES)
    assert len(items) > 20
    for p in items:
        ws.scene_new("lab", look="none")
        ex = Explorer(Editor(ws), clock=clock, pool=Immediate(), analysis_pool=Immediate())
        ex.load_preset("van_der_pol")
        d0 = _depth(ex)
        r = ex.place(p["type"], at=(0.5, 0.5, 0.0))
        assert r["ok"], (p["type"], r)
        assert _depth(ex) == d0 + 1
    assert ex.place("no_such_type")["ok"] is False


def test_presets(ex):
    names = [p["name"] for p in ex.presets()]
    assert "lorenz" in names and "van_der_pol" in names
    assert ex.load_preset("../etc")["ok"] is False
    assert ex.load_preset(None)["ok"] is False
    assert ex.load_preset("rossler")["ok"]


# ---------------------------------------------------------------- R2 / R3


def test_r2_and_r3_are_two_files_each_with_its_own_camera(ex, tmp_path):
    ex.load_preset("van_der_pol")
    r = ex.switch_space("R3")
    assert r["ok"] is False and r["needs_save"]
    ex.ed.save(str(tmp_path / "lab.phasor.json"))
    ex.playhead = 7.0
    r = ex.switch_space("R3")
    assert r["ok"], r
    assert ex.ws.path == tmp_path / "lab_r3.phasor.json"
    assert ex.space() == "R3" and ex.ed.engine.document.cameras[0].kind == "orbit"
    assert ex.playhead == 0.0 and _depth(ex) == 0
    ex.slider_drag("rho", 20)
    ex.slider_release("rho")
    ex.ed.save()
    r = ex.switch_space("R2")
    assert r["ok"] and ex.ws.path == tmp_path / "lab.phasor.json"
    assert ex.space() == "R2"
    assert ex.switch_space("R4")["ok"] is False


# ---------------------------------------------------------------- record, export


def test_marks_become_markers_and_eased_segments_in_one_patch(ex):
    ex.load_preset("lorenz")
    assert ex.mark()["ok"] is False
    ex.record(True)
    ex.mark()
    ex.camera_session = {"azimuth": 10.0}
    ex.playhead = 3.0
    ex.slider_drag("rho", 20)
    ex.mark()
    ex.mark()  # the same moment again: held a second later
    d0 = _depth(ex)
    r = ex.record(False)
    assert r["ok"] and r["markers"] == ["mark_1", "mark_2", "mark_3"]
    assert _depth(ex) == d0 + 1 and ex.history_names()[-1] == "recorded 3 marks"
    doc = ex.ed.engine.document
    assert doc.timeline.markers == {"mark_1": 0.0, "mark_2": 3.0, "mark_3": 4.0}
    seg = doc.timeline.tracks["camera.azimuth"][0]
    assert seg.t0.marker == "mark_1" and seg.t1.marker == "mark_2" and seg.ease == "smooth"
    assert "rho.value" in doc.timeline.tracks
    assert doc.object("lorenz").params["rate"] == 1.0
    st = ex.ed.engine.state_at(3.0)
    assert st["rho"].params.value == pytest.approx(20)
    assert ex.record(False)["marks"] == 0


def test_marks_ops_is_pure_and_skips_what_does_not_change(ex):
    ex.load_preset("lorenz")
    doc = ex.ed.engine.document
    marks = [{"t": 0, "camera": {"azimuth": 1.0}, "values": {"rho": 28.0}},
             {"t": 2, "camera": {"azimuth": 1.0}, "values": {"rho": 28.0}}]  # fmt: skip
    ops, names = marks_ops(doc, "camera", marks)
    assert names == ["mark_1", "mark_2"]
    assert not any(o["op"] == "add_segment" for o in ops)
    assert ex.ed.engine.document == doc


def test_export_plan_and_engine(ex):
    ex.load_preset("lorenz")
    plan = ex.export_plan("0", "30", "20")
    assert plan["ok"] and plan["text"] == "from tau 0 to 30 in 20 s"
    v0 = ex.ed.version
    fork = ex.export_engine(plan)
    st = fork.state_at(20.0)
    assert st["lorenz"].impl.resolve(st).tau == pytest.approx(30.0)
    assert fork.duration == pytest.approx(20.0)
    assert ex.ed.version == v0 and ex.ed.engine.document.object("lorenz").params.get("rate", 0) == 0
    for bad in (("a", 1, 1), (0, 1, 0), (0, 1, "inf"), (0, 1e9, 10), (0, 1e5, 1), (0, 1, -2)):
        assert ex.export_plan(*bad)["ok"] is False
    assert ex.export_plan(0, 1e4, 3000)["notes"]  # more steps than a run takes: said


# ---------------------------------------------------------------- convert to script


def test_convert_to_script_keeps_the_rows_and_refuses_new_ones(ex, tmp_path):
    ex.load_preset("lorenz")
    ex.add("tracer", at=(1, 1, 1))
    ex.ed.save(str(tmp_path / "lab.phasor.json"))
    before = [{k: e[k] for k in ("id", "kind", "source", "row")} for e in _rows(ex)]
    r = ex.convert()
    assert r["ok"], r
    assert ex.read_only and ex.ws.path.suffix == ".py"
    after = [{k: e[k] for k in ("id", "kind", "source", "row")} for e in _rows(ex)]
    assert after == before
    assert ex.type_row(NEW, "a = 1")["errors"][0]["message"] == CONVERTED
    assert ex.add("tracer")["ok"] is False and ex.load_preset("rossler")["ok"] is False
    assert ex.record(True)["ok"] is False
    assert ex.slider_drag("rho", 20)["ok"]
    assert ex.slider_release("rho")["ok"]  # a tweak, in the script's block
    assert "rho.value" in ex.ed.buffer.split("# --- phasor tweaks")[-1]


# ---------------------------------------------------------------- the picture (OpenGL)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_picture_is_the_view_half_size_while_interacting(ex, clock):
    ex.load_preset("van_der_pol")
    ex.add("disk")
    img, q = ex.frame((320, 180))
    assert q == "full" and img.shape == (180, 320, 3)
    ex.scrub(2.0)
    later, _ = ex.frame((320, 180))
    assert np.abs(later.astype(int) - img.astype(int)).sum() > 0  # the disk flowed
    ex.press(100, 100, (320, 180))
    ex.move(140, 100, (320, 180))
    small, q = ex.frame((320, 180))
    assert q == "interactive" and small.shape == (90, 160, 3)
    ex.release(140, 100, (320, 180))
    clock.tick(SETTLE + 0.01)
    assert ex.poll()  # the full picture is due
    assert ex.frame((320, 180))[1] == "full"
    p = ex.perf()  # what the owner's feel check reads (pc_tasks/14), wall clock
    assert p["interactive_ms"] > 0 and p["full_ms"] > 0 and p["latency_ms"] > 0
    assert "input to picture" in ex.perf_text()


def test_typing_on_the_real_reader_thread_while_an_ai_thread_edits(ws, clock):
    from concurrent.futures import wait

    ed = Editor(ws, threading.RLock())
    ex = Explorer(ed, clock=clock)  # the real worker threads; the clock is still the test's
    try:
        ex.load_preset("lorenz")
        refused: list[str] = []
        added: list[bool] = []

        def ai():
            for i in range(15):
                with ed.lock:
                    r = ws.explorer_edit([{"text": f"({i}, 1, 1)"}])
                    r2 = ws.explorer_edit([{"id": "rho", "value": 40 + i}])
                added.append(bool(r["ok"]))
                refused.extend(e["code"] for e in r2.get("errors", []))

        ex.type_row("rho", "rho = 2")
        t = threading.Thread(target=ai)
        t.start()
        for text in ("rho = 24", "rho = 24.7", "rho = 24.74"):
            ex.type_row("rho", text)
            wait([f for _, f in ex._pending])
            ex.poll()
        t.join()
        clock.tick(DEBOUNCE)
        ex.poll()
        doc = ws.engine.document
        assert doc.object("rho").params["value"] == pytest.approx(24.74)
        assert all(added) and sum(1 for e in _rows(ex) if e["kind"] == "point") == 15
        assert set(refused) == {"row_locked"} and len(refused) == 15
    finally:
        ex.close()
