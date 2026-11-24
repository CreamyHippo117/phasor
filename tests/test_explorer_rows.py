"""Milestone 27 part B: Explorer rows wired to the document (``workspace/explorer.py``).

The list of rows is a view of the document: every row is an object (or one line of a
``flow.System``) with its typed text; a commit is one patch; ids survive delete-and-retype;
session state (playhead, a slider mid-drag, text mid-typing) never enters the document."""

from __future__ import annotations

import json

import pytest

from phasor import flow
from phasor.core.ids import ID_RE
from phasor.domains import flow_math as fm
from phasor.domains import rows
from phasor.render.gpu import codegen
from phasor.workspace import explorer
from phasor.workspace.tools import ToolError, Workspace
from tests.conftest import gl_available


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.scene_new("explorer", look="none")
    return w


def _types(ws) -> dict[str, str]:
    return {o.id: o.type for o in ws.engine.document.objects}


def _rows(ws) -> list[dict]:
    return explorer.rows_of(ws.engine.document)


def _undo_depth(ws) -> int:
    return len(ws.engine._undo)


def _ok(r):
    assert r["ok"], r.get("errors")
    return r


# ---------------------------------------------------------------- rows to objects


def test_rows_become_objects_with_their_text(ws):
    r = _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -sin(x) - g y"},
                              {"text": "(1, 0.5)"}, {"text": "y = x^2"},
                              {"text": "q = 9^9^9"}]))  # fmt: skip
    assert r["new_variables"] == ["g"]
    t = _types(ws)
    assert t["system"] == "flow_system" and t["system.field"] == "flow_arrows"
    assert t["system.eq"] == "flow_equilibria" and t["point"] == "flow_probe"
    assert t["g"] == "variable" and t["graph"] == "flow_note" and t["row"] == "flow_note"
    doc = ws.engine.document
    sysp = doc.object("system").params
    assert sysp["equations"] == ("y", "-sin(x) - g*y")
    assert sysp["source"] == "x' = y\ny' = -sin(x) - g y"  # as typed, a line per row
    assert sysp["constants"] == "g=1"  # every name the system uses has a value
    assert doc.object("point").params["system"] == "system"
    assert doc.object("point").params["x0"] == (1.0, 0.5)
    (ctl,) = doc.controls
    assert (ctl.path, ctl.min, ctl.max, ctl.step) == ("g.value", -10.0, 10.0, 0.01)
    view = {e["id"]: e for e in _rows(ws)}
    assert list(view) == ["system:x", "system:y", "point", "graph", "row", "g"]
    assert view["row"]["kind"] == "error" and "too large" in view["row"]["error"]
    assert view["graph"]["kind"] == "graph" and view["graph"]["drawn"] is False
    assert view["g"]["text"] == "g = 1" and view["g"]["slider"] is True
    assert [e["row"] for e in view.values()] == [0, 0, 1, 2, 3, 4]


def test_the_list_is_rebuilt_from_the_saved_document_alone(ws, tmp_path):
    _ok(ws.explorer_edit([{"text": "x' = a x - y"}, {"text": "y' = x + a y"},
                          {"text": "a = 0.25"}, {"text": "sin x"}]))  # fmt: skip
    before = _rows(ws)
    ws.scene_save("rows.phasor.json")
    other = Workspace(tmp_path)
    other.scene_open("rows.phasor.json")
    assert explorer.rows_of(other.engine.document) == before
    assert before[-1]["kind"] == "error" and "sin" in before[-1]["error"]


def test_a_kind_change_is_one_undoable_patch_that_retires_the_old_id(ws):
    _ok(ws.explorer_edit([{"text": "y = x^2"}]))
    before = ws.engine.document.model_dump()
    depth = _undo_depth(ws)
    r = _ok(ws.explorer_edit([{"id": "graph", "text": "(y, -x)"}]))
    assert _undo_depth(ws) == depth + 1
    assert r["removed"] == ["graph"]
    assert _types(ws) == {"field": "flow_system", "field.field": "flow_arrows"}
    assert "graph" in ws.engine.document.retired_ids
    ws.undo()
    assert ws.engine.document.model_dump() == before


def test_an_edit_that_keeps_the_kind_keeps_the_id(ws):
    _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -x"}]))
    r = _ok(ws.explorer_edit([{"id": "system:y", "text": "y' = -x - 0.1 y"}]))
    assert r["removed"] == [] and "system" in _types(ws)
    assert ws.engine.document.object("system").params["equations"] == ("y", "-x - 0.1*y")
    # a third row makes it 3D: the same system, a 3D window
    _ok(ws.explorer_edit([{"text": "z' = -z"}]))
    p = ws.engine.document.object("system").params
    assert len(p["equations"]) == 3 and len(p["window"]) == 3


@pytest.mark.parametrize("text", ["a = 1", "y = x^2", "(1, 2)", "(y, -x)"])
def test_delete_and_retype_five_times_never_reuses_an_id(ws, text):
    ids = []
    for _ in range(5):
        r = _ok(ws.explorer_edit([{"text": text}]))
        new = [row["id"] for row in r["rows"] if row["new"]]
        assert len(new) == 1
        ids.append(new[0])
        _ok(ws.explorer_edit([{"id": new[0], "remove": True}]))
    assert len(set(ids)) == 5
    assert all(ID_RE.fullmatch(i) for i in ids)
    assert set(ids) <= set(ws.engine.document.retired_ids)


def test_a_removed_system_and_its_children_retype_cleanly(ws):
    for _ in range(3):
        _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -x"}]))
        sid = next(o.id for o in ws.engine.document.objects if o.type == "flow_system")
        assert {f"{sid}.field", f"{sid}.eq"} <= set(_types(ws))
        _ok(ws.explorer_edit([{"id": f"{sid}:x", "remove": True},
                              {"id": f"{sid}:y", "remove": True}]))  # fmt: skip
        assert not any(i.startswith(sid) for i in _types(ws))


def test_ids_are_named_by_meaning_never_by_text_and_never_collide(ws):
    _ok(ws.explorer_preset("lorenz"))
    # a variable called like the system, and one called like a child
    _ok(ws.explorer_edit([{"text": "lorenz = 2"}, {"text": "field = 3"}]))
    t = _types(ws)
    assert t["lorenz"] == "flow_system" and t["lorenz.field"] == "flow_arrows"
    names = {ws.engine.document.object(i).params["name"]: i for i, k in t.items()
             if k == "variable"}  # fmt: skip
    assert names["lorenz"] != "lorenz" and names["field"] == "field"
    # no id holds a formula or an equilibrium's label
    for i in t:
        assert ID_RE.fullmatch(i), i
        assert "+" not in i and "'" not in i


def test_typing_a_variable_that_exists_sets_it_instead_of_a_second_one(ws):
    _ok(ws.explorer_edit([{"text": "x' = rho y"}, {"text": "y' = -x"}]))
    assert ws.engine.document.object("rho").params["value"] == 1.0
    _ok(ws.explorer_edit([{"text": "rho = 24"}]))
    vars_ = [o for o in ws.engine.document.objects if o.type == "variable"]
    assert len(vars_) == 1 and vars_[0].params["value"] == 24.0
    assert not [i for i in ws.engine.lint() if i.code == "duplicate_variable"]
    # the value widens the slider's range rather than sit outside it
    _ok(ws.explorer_edit([{"id": "rho", "value": 50}]))
    (ctl,) = ws.engine.document.controls
    assert ctl.max >= 50


def test_loading_a_preset_twice_keeps_its_variables(ws):
    _ok(ws.explorer_preset("van_der_pol"))
    ids = set(_types(ws))
    _ok(ws.explorer_edit([{"id": "mu", "value": 2.5}]))
    _ok(ws.explorer_preset("van_der_pol"))
    t = _types(ws)
    assert "mu" in t and ws.engine.document.object("mu").params["value"] == 1.0
    assert len([i for i, k in t.items() if k == "variable"]) == 1
    assert ids - {"van_der_pol", "van_der_pol.field", "van_der_pol.eq"} <= set(t)


@pytest.mark.parametrize("name", sorted(fm.PRESETS))
def test_every_preset_loads_as_rows_that_mean_the_preset(ws, name):
    p = fm.PRESETS[name]
    _ok(ws.explorer_preset(name))
    doc = ws.engine.document
    (sysid,) = [o.id for o in doc.objects if o.type == "flow_system"]
    sp = doc.object(sysid).params
    got = fm.build(sp["equations"])
    want = fm.build(p.equations)
    assert [str(e) for e in got.exprs] == [str(e) for e in want.exprs]
    assert tuple(tuple(w) for w in sp["window"]) == tuple(tuple(map(float, w)) for w in p.window)
    assert sp["dt"] == p.dt
    for v, (value, lo, hi) in p.variables.items():
        o = next(o for o in doc.objects if o.type == "variable" and o.params["name"] == v)
        assert (o.params["value"], o.params["min"], o.params["max"]) == (value, lo, hi)
    assert not [i for i in ws.engine.lint() if i.severity == "error"]


HOSTILE = [
    "9^9^9", "x^9^9^9", "2^100000", "(" * 40 + "x" + ")" * 40, "x" + "+x" * 600,
    "x' = ", "x'' = y", "\x00\x01", "x' = y\ny' = -x", "؀؁", "ẋ = ẋ", "rho = 1e400",
    "(1, 2, 3, 4)", "F(x) = 2", "x' = y; import os", "__import__('os')", "lambda: 0",
    "a" * 3000, "j = 2", "I = 3", "x' = t", "y' = x", "((((", "x' = 1/0",
]  # fmt: skip


def test_hostile_text_is_an_error_row_never_a_traceback(ws):
    for text in HOSTILE:
        r = ws.explorer_edit([{"text": text}])
        assert r["ok"], (text, r.get("errors"))
        json.dumps(r)
    kinds = {e["source"]: e for e in _rows(ws)}
    for text in ("9^9^9", "x' = y; import os", "__import__('os')", "j = 2", "I = 3", "x' = t"):
        assert kinds[text]["kind"] == "error" and kinds[text]["error"], text
    # a line break is read as a space (a row is one line)
    assert "x' = y y' = -x" in kinds
    assert not [i for i in ws.engine.lint() if i.severity == "error"]


def test_a_row_too_long_to_store_is_refused_cleanly(ws):
    with pytest.raises(ToolError, match="at most"):
        ws.explorer_edit([{"text": "x" * (flow.MAX_ROW_SOURCE + 1)}])
    assert ws.engine.version == 0


def test_names_the_flow_core_reserves_and_time_make_notes_not_refusals(ws):
    _ok(ws.explorer_edit([{"text": "x' = j y"}, {"text": "y' = -x"}]))
    view = _rows(ws)
    assert all(e["kind"] == "error" for e in view)
    assert "reserved" in view[0]["error"]
    _ok(ws.explorer_edit([{"id": view[0]["id"], "text": "x' = y + t"}]))
    assert "uses t" in _rows(ws)[0]["error"]
    # fixed: the notes become one system, in one patch
    r = _ok(ws.explorer_edit([{"id": _rows(ws)[0]["id"], "text": "x' = y"}]))
    assert "system" in _types(ws) and len(r["removed"]) == 2


def test_bad_edits_are_refused_with_a_message(ws):
    _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -x"}]))
    for bad, match in (([{"id": "nope", "text": "1"}], "no row"),
                       ([{"id": "system", "text": "y = 1"}], "has 2 rows"),
                       ([{"text": "a = 1", "kind": "teapot"}], "kind is one of"),
                       ([{"text": "a = 1", "bogus": 1}], "unknown fields"),
                       ({"text": "a"}, "list"),
                       ([{"id": "system", "color": "highlight"}, {"text": "b = 1",
                         "color": "highlight"}], "no colour")):  # fmt: skip
        v = ws.engine.version
        with pytest.raises(ToolError, match=match):
            ws.explorer_edit(bad)
        assert ws.engine.version == v


def test_visible_colour_order_and_settings(ws):
    _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -x"}, {"text": "(1, 0)"}]))
    _ok(ws.explorer_edit([{"id": "system", "visible": False, "color": "highlight",
                           "window": [[-2, 2], [-2, 2]], "dt": 0.02},
                          {"id": "point", "at": 0}]))  # fmt: skip
    o = ws.engine.document.object("system")
    assert o.params["visible"] is False and o.style["frame"] == "highlight"
    assert o.params["window"] == ((-2.0, 2.0), (-2.0, 2.0)) and o.params["dt"] == 0.02
    assert [e["id"] for e in _rows(ws)] == ["point", "system:x", "system:y"]
    st = ws.engine.state_at(0.0)
    assert st["system.field"].impl._hidden(st) and st["system.eq"].impl._hidden(st)
    assert not st["point"].impl._hidden(st)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_hidden_system_draws_nothing(ws):
    _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -x"}]))
    shown = ws.engine.still(0.0, (320, 180))
    _ok(ws.explorer_edit([{"id": "system", "visible": False}]))
    hidden = ws.engine.still(0.0, (320, 180))
    ws.undo()
    blank = ws.engine.fork()
    for i in ("system.field", "system.eq", "system"):
        assert blank.apply({"op": "remove_object", "id": i}).ok
    assert (hidden == blank.still(0.0, (320, 180))).all()
    assert (shown != hidden).any()


# ---------------------------------------------------------------- the person and the AI


def test_a_row_being_typed_is_locked_to_the_person(ws):
    from phasor.workspace.remote import tool_names

    _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -x"}, {"text": "a = 1"}]))
    version = ws._explorer_typing("system:y", "y' = -x - 0.")
    r = ws.explorer_edit([{"id": "system:y", "text": "y' = 0"}])
    assert not r["ok"]
    (err,) = r["errors"]
    assert (err["code"], err["row"], err["version"]) == ("row_locked", "system:y", version)
    # the whole system is that row's object: also locked
    assert not ws.explorer_edit([{"id": "system", "visible": False}])["ok"]
    # other rows are free
    _ok(ws.explorer_edit([{"id": "a", "value": 2}]))
    assert ws.explorer_state()["locked"] == ["system:y"]
    # the person commits: unlocked, one edit
    depth = _undo_depth(ws)
    _ok(ws._explorer_commit("system:y", "y' = -x - 0.5 y"))
    assert _undo_depth(ws) == depth + 1 and ws._drafts == {}
    # the lock is not a tool an attached AI could call
    assert not {"_explorer_typing", "_explorer_commit", "_explorer_preview"} & set(tool_names())


def test_an_edit_naming_an_old_version_is_refused(ws):
    _ok(ws.explorer_edit([{"text": "a = 1"}]))
    old = _rows(ws)[0]["version"]
    _ok(ws.explorer_edit([{"id": "a", "value": 3}]))
    r = ws.explorer_edit([{"id": "a", "value": 4, "version": old}])
    assert not r["ok"] and r["errors"][0]["code"] == "row_changed"
    new = r["errors"][0]["version"]
    _ok(ws.explorer_edit([{"id": "a", "value": 4, "version": new}]))


def test_a_preview_is_a_fork_and_the_commit_is_one_step(ws):
    _ok(ws.explorer_edit([{"text": "x' = y"}, {"text": "y' = -k x"}]))
    v, depth = ws.engine.version, _undo_depth(ws)
    for text in ("y' = -k x - 0", "y' = -k x - 0.", "y' = -k x - 0.3", "y' = -k x - 0.3y"):
        fork = ws._explorer_preview([{"id": "system:y", "text": text}])
        assert fork is not ws.engine
    assert (ws.engine.version, _undo_depth(ws)) == (v, depth)
    assert fork.document.object("system").params["equations"] == ("y", "-k*x - 0.3*y")
    _ok(ws.explorer_edit([{"id": "system:y", "text": "y' = -k x - 0.3y"}]))
    assert _undo_depth(ws) == depth + 1


def test_playing_scrubbing_and_dragging_leave_the_document_alone(ws):
    from phasor.gui.controller import Editor

    _ok(ws.explorer_preset("pendulum"))
    _ok(ws.explorer_edit([{"id": "pendulum", "rate": 1.0}]))
    ed = Editor(ws)
    v, depth = ed.version, _undo_depth(ws)
    for _ in range(600):  # a minute at 10 steps a second
        ed.play_step(0.1)
    for t in (3.0, 0.5, 9.0, 0.0, 4.2):
        ed.set_time(t)
    for value in (0.1, 0.5, 1.2):
        ed.set_control("gamma.value", value)  # a slider mid-drag
    assert (ed.version, _undo_depth(ws)) == (v, depth)
    assert ws.engine.document.object("gamma").params["value"] == 0.3
    # release: one edit, and the list shows the slider's value
    _ok(ed.apply_controls())
    assert _undo_depth(ws) == depth + 1
    row = next(e for e in _rows(ws) if e["id"] == "gamma")
    assert row["value"] == 1.2 and row["text"] == "gamma = 1.2"
    assert row["source"] == "gamma = 0.3"  # what was typed stays; the value wins


def test_a_script_scene_is_refused(tmp_path):
    (tmp_path / "s.py").write_text("from phasor import Scene\ns = Scene('s')\n", encoding="utf-8")
    w = Workspace(tmp_path, allow_scripts=True)
    w.scene_open("s.py")
    with pytest.raises(ToolError, match="script"):
        w.explorer_edit([{"text": "a = 1"}])


def test_the_rows_caps_are_the_stricter_of_the_two_layers():
    assert rows.MAX_CHARS <= codegen.MAX_SOURCE
    assert rows.MAX_TOKENS <= codegen.MAX_NODES
    assert rows.MAX_DEPTH <= codegen.MAX_DEPTH
    assert 10.0**rows.MAX_MAGNITUDE <= 2.718281828**codegen.MAX_POWER
    assert rows.normalize_row("9^9^9").kind == "error"


def test_a_commit_that_changes_nothing_writes_nothing(ws):
    def still():
        assert explorer.plan(ws.engine.document, [])[0] == []
        v, depth = ws.engine.version, _undo_depth(ws)
        _ok(ws.explorer_edit([]))
        assert (ws.engine.version, _undo_depth(ws)) == (v, depth)

    still()
    _ok(ws.explorer_edit([{"text": "x' = a y"}, {"text": "y' = -x"}, {"text": "(1, 0)"},
                          {"text": "y = x^2"}, {"text": "q = sin x"}]))  # fmt: skip
    still()
    _ok(ws.explorer_preset("lorenz", replace=False))
    still()
    v = ws.engine.version
    _ok(ws.explorer_edit([{"id": "rho", "value": 28}]))  # its value already
    assert ws.engine.version == v


@pytest.mark.parametrize("edit, match", [
    ({"id": "system", "dt": 5}, "dt must be between"),
    ({"id": "system", "window": 5}, "window"),
    ({"id": "system", "window": [[-1, 1], [-1, 1], [0, 1]]}, "2 coordinates"),
    ({"id": "system", "window": [[1, -1], [-1, 1]]}, "window"),
    ({"id": "a", "min": 5, "max": 1}, "below max"),
    ({"id": "a", "min": 20}, "below max"),
    ({"id": "a", "value": None}, "finite number"),
    ({"id": "a", "value": float("inf")}, "finite number"),
    ({"id": "a", "step": 0}, "step must be between"),
    ({"id": ["a"]}, "id is"),
    ({"id": "a", "at": None}, "whole number"),
    ({"id": "a", "remove": "false"}, "true or false"),
    ({"id": "a", "visible": 0}, "true or false"),
])  # fmt: skip
def test_a_bad_setting_is_refused_and_destroys_nothing(ws, edit, match):
    _ok(ws.explorer_edit([{"text": "x' = a y"}, {"text": "y' = -x"}]))
    before = ws.engine.document.model_dump()
    with pytest.raises(ToolError, match=match):
        ws.explorer_edit([edit])
    assert ws.engine.document.model_dump() == before
