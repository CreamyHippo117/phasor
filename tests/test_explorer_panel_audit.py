"""Milestone 29: regression tests for the independent audit's findings (each fails on the
code before the fixes and passes after; see MILESTONE_29.md "Independent audit")."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from phasor.gui.controller import Editor
from phasor.gui.explorer import NEW, Explorer, Immediate
from phasor.workspace import explorer as X
from phasor.workspace.tools import Workspace


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, s: float) -> None:
        self.t += s


class Held:
    """An executor whose jobs run only when the test says."""

    def __init__(self) -> None:
        self.jobs: list = []

    def submit(self, fn, *args):
        f: Future = Future()
        self.jobs.append((f, fn, args))
        return f

    def run_all(self) -> None:
        for f, fn, args in self.jobs:
            if not f.done():
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
    e = Explorer(Editor(ws, threading.RLock()), clock=clock, pool=Immediate(),
                 analysis_pool=Immediate())  # fmt: skip
    yield e
    e.close()


def _rows(ex):
    return X.rows_of(ex.ed.engine.document)


def _depth(ex) -> int:
    return len(ex.ed.engine._undo)


def _typed(ex, clock, key, text):
    for k in range(1, len(text) + 1):
        ex.type_row(key, text[:k])
        clock.tick(0.05)
        ex.poll()


def _key(ex, kind):
    return next(e["id"] for e in _rows(ex) if e["kind"] == kind)


def _lint_codes(ex):
    return {i.code for i in ex.ed.engine.lint() if i.severity == "error"}


def _same_scene(a, b) -> None:
    """Two documents say the same scene: objects, params, styles, tracks, markers, controls."""
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    for k in ("objects", "controls", "timeline", "cameras"):
        assert da[k] == db[k], k


# ---------------------------------------------------------------- B1: keep / record vs sliders


def test_b1_keep_takes_the_slider_away_in_the_same_patch_and_undo_brings_both_back(ex):
    ex.load_preset("lorenz")
    doc0 = ex.ed.engine.document
    d0 = _depth(ex)
    ex.slider_play("beta")
    assert ex.slider_keep("beta")["ok"]
    assert _depth(ex) == d0 + 1
    doc = ex.ed.engine.document
    assert "beta.value" in doc.timeline.tracks
    assert all(c.path != "beta.value" for c in doc.controls)
    assert "orphan_control" not in {i.code for i in ex.ed.engine.lint()}
    ex.ed.undo()
    _same_scene(ex.ed.engine.document, doc0)  # the track and the slider: both as before
    ex.ed.redo()
    assert all(c.path != "beta.value" for c in ex.ed.engine.document.controls)
    assert ex.clear_track("beta")["ok"]  # the slider comes back with the track gone
    assert any(c.path == "beta.value" for c in ex.ed.engine.document.controls)


def test_b1_record_then_convert_runs_and_the_script_is_the_document(ex, tmp_path):
    ex.load_preset("lorenz")
    ex.record(True)
    ex.mark()
    ex.scrub(5.0)
    ex.camera_session = {"azimuth": 10.0}
    ex.slider_drag("rho", 20)
    ex.mark()
    ex.camera_session = {}
    ex.slider_preview = {}
    assert ex.record(False)["ok"]
    assert "orphan_control" not in {i.code for i in ex.ed.engine.lint()}
    ex.ed.save(str(tmp_path / "lab.phasor.json"))
    doc = ex.ed.engine.document
    r = ex.convert()
    assert r["ok"], r
    assert ex.ed.is_script
    _same_scene(ex.ed.engine.document, doc)


def test_b1_keep_then_convert(ex, tmp_path):
    ex.load_preset("lorenz")
    ex.slider_play("beta")
    ex.slider_keep("beta")
    ex.ed.save(str(tmp_path / "lab.phasor.json"))
    assert ex.convert()["ok"]


# ---------------------------------------------------------------- B2: an AI against a locked row


def _ai(ws):
    return ws._as_ai()


def _typing_in_a_committed_row(ex, clock):
    _typed(ex, clock, NEW, "y = x^2")
    clock.tick(2)
    ex.poll()
    key = ex.focus
    ex.type_row(key, "y = x^2 + sin(x")
    return key


def test_b2_an_ai_undo_redo_or_restore_cannot_take_the_row_being_typed(ex, clock, ws):
    ex.load_preset("lorenz")
    ws.checkpoint("before")
    key = _typing_in_a_committed_row(ex, clock)
    with _ai(ws):
        r = ws.undo()
        assert r["undone"] is False and r["errors"][0]["code"] == "row_locked"
        with pytest.raises(Exception, match="typing"):
            ws.restore("before")
        with pytest.raises(Exception, match="typing"):
            ws.scene_new("other")
        with pytest.raises(Exception, match="typing"):
            ws.scene_open("nowhere.phasor.json")
    assert any(e["id"] == key for e in _rows(ex))
    ex.leave()
    ex.ed.undo()  # the person's own undo works
    ex.ed.undo()
    with _ai(ws):  # not typing now: the AI may redo
        assert ws.redo()["redone"]


def test_b2_an_ai_redo_cannot_change_the_row_being_typed(ex, clock, ws):
    ex.load_preset("lorenz")
    ws.explorer_edit([{"id": "rho", "value": 3}])
    ws.undo()  # redo would set rho to 3
    ex.type_row("rho", "rho = 2")
    with _ai(ws):
        r = ws.redo()
    assert r["redone"] is False and r["errors"][0]["code"] == "row_locked"


def test_b2_an_ai_raw_scene_edit_cannot_rewrite_or_remove_the_row_being_typed(ex, ws):
    ex.load_preset("lorenz")
    ex.type_row("rho", "rho = 2")
    with _ai(ws):
        r1 = ws.scene_edit([{"op": "set_param", "path": "rho.value", "value": 99}])
        r2 = ws.scene_edit([{"op": "remove_object", "id": "rho"}])
        r3 = ws.scene_edit({"op": "set_param", "path": "sigma.value", "value": 11})
    assert r1["ok"] is False and r1["errors"][0]["code"] == "row_locked"
    assert r2["ok"] is False and r2["errors"][0]["code"] == "row_locked"
    assert r3["ok"] is True  # another row: fine
    assert ex.ed.engine.document.object("rho").params["value"] == pytest.approx(28)
    r = ex.ed.edit([{"op": "set_param", "path": "rho.value", "value": 5}], "rho by hand")
    assert r["ok"]  # the person's own edit is not refused


def test_b2_other_ai_tools_are_taken_back_when_they_change_the_row(ex, ws, tmp_path):
    from phasor.workspace.remote import RemoteServer, RemoteWorkspace

    ex.load_preset("lorenz")
    ex.type_row("rho", "rho = 2")
    v0, d0 = ex.ed.version, _depth(ex)
    server = RemoteServer(ws, ex.ed.lock).start()
    server.write_handoff()
    ai = RemoteWorkspace.from_handoff(tmp_path)
    try:
        with pytest.raises(Exception, match="typing in row"):
            ai.animate("sweep", 0.0, path="rho.value", value=5)
        ai.animate("sweep", 0.0, path="sigma.value", value=5)  # another row: kept
    finally:
        ai.close()
        server.stop()
    doc = ex.ed.engine.document
    assert "rho.value" not in doc.timeline.tracks and "sigma.value" in doc.timeline.tracks
    assert ex.ed.engine.redo_document() is None  # the refused step is not left to redo
    assert ex.ed.version > v0 and _depth(ex) >= d0 + 1


def test_b2_a_row_gone_under_the_draft_keeps_the_text_and_lets_go_of_the_lock(ex, clock, ws):
    ex.load_preset("lorenz")
    key = _typing_in_a_committed_row(ex, clock)
    ex.ed.undo()  # the person's own Ctrl+Z (or an AI before the fix) takes the row away
    r = ex.type_row(key, "y = x^2 + sin(x)")
    assert r["ok"] is False and r["moved_to"] == NEW
    assert key not in ws._drafts  # no lock left behind
    assert ex.drafts[NEW].text == "y = x^2 + sin(x)"  # the typed text is kept
    ex.ed.redo()  # the row returns: nobody types in it, so it is not locked
    assert ws.explorer_edit([{"id": key, "text": "y = 2x"}])["ok"]
    ex.leave()
    assert any(e["source"] == "y = x^2 + sin(x)" for e in _rows(ex))


def test_b2_a_row_removed_while_typing_is_noticed_at_the_next_tick(ex, clock, ws):
    ex.load_preset("lorenz")
    key = _typing_in_a_committed_row(ex, clock)
    ex.ed.undo()
    ex.poll()
    assert key not in ws._drafts and ex.drafts[NEW].text == "y = x^2 + sin(x"
    ex.leave()  # an unfinished row commits when left (with its error), never dropped
    assert any(e["source"] == "y = x^2 + sin(x" for e in _rows(ex))


# ---------------------------------------------------------------- S1, S2: kind with the text


def test_s1_text_and_kind_commit_as_one_patch(ex, clock):
    ex.load_preset("lorenz")
    _typed(ex, clock, NEW, "y = x^2")
    clock.tick(2)
    ex.poll()
    ex.leave()
    key = _key(ex, "graph")
    d0, h0 = _depth(ex), len(ex.ed.history)
    ex.type_row(key, "y = x^3")
    ex.set_kind(key, "implicit_curve")
    clock.tick(2)
    ex.poll()
    assert _depth(ex) == d0 + 1 and len(ex.ed.history) == h0 + 1
    row = next(e for e in _rows(ex) if e["source"] == "y = x^3")
    assert row["kind_override"] == "implicit_curve"
    assert ex.focus == row["id"]  # the draft followed the row to its new id


def test_s2_a_kind_override_on_a_field_row_reads_back_and_leave_is_quiet(ex, clock):
    _typed(ex, clock, NEW, "(0, 0)")
    clock.tick(2)
    ex.poll()
    ex.leave()
    key = _key(ex, "point")
    assert ex.set_kind(key, "vector_field")["ok"]
    row = next(e for e in _rows(ex) if e["source"] == "(0, 0)")
    assert row["kind"] == "vector_field" and row["kind_override"] == "vector_field"
    assert ":" not in row["id"]
    d0 = _depth(ex)
    for _ in range(3):
        k = next(e["id"] for e in _rows(ex) if e["source"] == "(0, 0)")
        ex.type_row(k, "(0, 0)", kind="vector_field")
        ex.leave(k)
    assert _depth(ex) == d0  # nothing changed: nothing committed
    assert {e["id"] for e in _rows(ex)} == {row["id"]}


# ---------------------------------------------------------------- S4: the card follows


def test_s4_the_card_follows_its_equilibrium_and_closes_when_it_goes(ex, clock):
    ex.load_preset("lorenz")
    size = (640, 360)
    st = ex._state()
    sem = st["lorenz.eq"].impl.semantics(st)
    origin = next(p for p in sem["points"] if abs(p["at"][0]) < 1e-6)
    r = st["lorenz"].impl.resolve(st)
    u, v = ex._screen(r.world(np.asarray(origin["at"])), size)[0]
    card = ex.click(u, v, size)["card"]
    assert card["name"] == "O" and card["kind"] == "saddle"
    ex.set_value("rho", 0.5)  # now the origin is a stable node and C+/C- are gone
    ex.poll()
    assert ex.card["name"] == "O" and ex.card["kind"] != "saddle"
    assert "now" in ex.message
    ex.zoom_in()
    clock.tick(1)
    ex.poll()
    assert ex.history_names()[-1] == f"zoom in to O ({ex.card['kind']})"
    ex.set_value("rho", 28)
    cplus = next(p for p in ex._eq_points(ex._state(), "lorenz") if p["at"][0] > 1)
    u, v = ex._screen(r.world(np.asarray(cplus["at"])), size)[0]
    ex.card = None
    ex.camera_session = {}
    ex.click(u, v, size)
    if ex.card is not None:  # C+ is clickable from where the camera is now
        assert ex.card["name"] == "C+"
        ex.set_value("rho", 0.5)
        ex.poll()
        assert ex.card is None and "gone" in ex.message
        assert ex.zoom_in()["ok"] is False


def test_s4_names_do_not_move_with_a_slider():
    from phasor.gui.explorer import equilibrium_names

    pts = [{"at": [8.0, 8.0, 27.0]}, {"at": [0.0, 0.0, 0.0]}, {"at": [-8.0, -8.0, 27.0]}]
    assert equilibrium_names(pts) == ["C+", "O", "C-"]
    assert equilibrium_names(pts[::-1]) == ["C-", "O", "C+"]
    assert equilibrium_names([{"at": [1.0, 0]}, {"at": [2.0, 0]}]) == ["C+1", "C+2"]


# ---------------------------------------------------------------- S5: memos under threads


def test_s5_the_flow_memos_are_safe_under_threads():
    import sys

    from phasor.domains import flow

    class Runner:
        def state_at(self, t):
            return t

    class Sim:
        dt = 0.01

        def trail_from(self, s, back):
            x = 0
            for i in range(200):
                x += i
            return np.zeros((1, back, 3))

    runner, sim = Runner(), Sim()
    errors: list[str] = []
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def work(off):
        try:
            for i in range(4000):
                flow._trail(runner, sim, off + i % 9, 2)
        except Exception as e:  # a KeyError before the fix
            errors.append(repr(e))

    try:
        ts = [threading.Thread(target=work, args=(k * 100,)) for k in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert errors == []


# ---------------------------------------------------------------- S6: slow placement off thread


def test_s6_a_slow_flow_object_is_computed_off_the_gui_thread(ws, clock):
    held = Held()
    ex = Explorer(Editor(ws), clock=clock, pool=Immediate(), analysis_pool=held)
    ex.load_preset("van_der_pol")
    d0 = _depth(ex)
    r = ex.place("flow_limit_cycle")
    assert r["ok"] and r["pending"] and _depth(ex) == d0  # nothing computed or committed yet
    assert ex.placing() == [r["id"]]
    assert held.jobs  # the computation is the analysis thread's
    held.run_all()
    ex.poll()
    assert _depth(ex) == d0 + 1 and ex.placing() == []
    assert ex.history_names()[-1] == "added LimitCycle"


# ---------------------------------------------------------------- nits


def test_nit_marks_are_capped_and_going_backwards_is_said(ex):
    from phasor.gui.explorer import MAX_MARKS

    ex.load_preset("van_der_pol")
    ex.record(True)
    ex.scrub(30)
    ex.mark()
    ex.scrub(20)
    ex.mark()
    assert "before the last mark" in ex.message
    ex.marks = ex.marks * (MAX_MARKS // 2)
    assert ex.mark()["ok"] is False
    ex.marks = ex.marks[:3]
    ex.record(False)
    assert "lasts" in ex.message  # the duration it stretched to is said


def test_nit_export_has_its_own_job_and_the_draft_stays_followed(ex, tmp_path, monkeypatch):
    ex.load_preset("van_der_pol")
    ex.ed.job = "render_9"  # a draft being followed
    calls = {}

    def fake(source, target, preset, **kw):
        calls["source"] = source
        return {"job": "render_10", "state": "running", "path": str(target)}

    monkeypatch.setattr(ex.ed, "render_engine", fake)
    r = ex.export(tmp_path / "x.mp4", 0, 1, 1)
    assert r["ok"] and ex.export_job == "render_10" and ex.ed.job == "render_9"
    assert calls["source"] is not ex.ed.engine


def test_nit_pasting_several_rows_makes_a_row_each_in_one_patch(ex):
    d0 = _depth(ex)
    ex.type_row(NEW, "x' = y\ny' = -x\nmu = 2")
    assert _depth(ex) == d0 + 1
    kinds = sorted(e["kind"] for e in _rows(ex))
    assert kinds == ["ode", "ode", "variable"]


def test_nit_the_palette_leaves_out_what_draws_nothing_alone():
    from phasor.gui.explorer import PALETTE_HIDDEN, palette

    types = {p["type"] for p in palette()}
    assert not types & set(PALETTE_HIDDEN) and "flow_limit_cycle" in types


@pytest.mark.skipif(os.name == "nt", reason="the window tests cover Windows")
def test_nit_public_editor_ways_in(ex):
    src = Path(__file__).resolve().parents[1] / "phasor" / "gui" / "explorer.py"
    text = src.read_text(encoding="utf-8")
    for private in ("ed._edit(", "ed._camera(", "ed._set_camera(", "ws._render_job("):
        assert private not in text
