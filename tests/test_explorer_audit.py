"""Milestone 27 part B, independent audit (2026-11-24): a regression test per finding.
B1, B2 the blockers; S1-S8 the should-fix items; N the nits (MILESTONE_27.md, "Audit")."""

from __future__ import annotations

import json
import math

import pytest

from phasor import Scene
from phasor.workspace import explorer
from phasor.workspace.tools import ToolError, Workspace

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.scene_new("explorer", look="none")
    return w


def _ok(r):
    assert r["ok"], r.get("errors")
    return r


def _dump(ws) -> str:
    return json.dumps(ws.engine.document.model_dump(mode="json"), sort_keys=True)


def _strict(r) -> None:
    json.dumps(r, allow_nan=False)


# ---------------------------------------------------------------- B1: the lock from the plan


@pytest.fixture
def typed(ws):
    _ok(ws.explorer_edit([{"text": "x' = a*y"}, {"text": "y' = -x"}, {"text": "rho = 3"}]))
    return ws


def test_b1_a_new_row_cannot_rewrite_a_locked_variable(typed):
    version = typed._explorer_typing("rho", "rho = 4")
    before = _dump(typed)
    r = typed.explorer_edit([{"text": "rho = 99"}])
    assert not r["ok"] and r["errors"][0]["code"] == "row_locked"
    assert (r["errors"][0]["row"], r["errors"][0]["version"]) == ("rho", version)
    assert _dump(typed) == before


def test_b1_a_new_ode_row_cannot_break_a_locked_system(typed):
    typed._explorer_typing("system:x", "x' = a*y + 1")
    before = _dump(typed)
    for edit in ([{"text": "x' = 5"}], [{"id": "system:y", "text": "y' = -2x"}],
                 [{"id": "system", "dt": 0.02}]):  # fmt: skip
        r = typed.explorer_edit(edit)
        assert not r["ok"] and r["errors"][0]["row"] == "system:x", edit
    assert not typed.explorer_preset("lorenz")["ok"]
    assert _dump(typed) == before
    # rows the person is not typing in are free; then the person's commit lands
    _ok(typed.explorer_edit([{"id": "rho", "value": 5}, {"text": "(1, 1)"}]))
    _ok(typed._explorer_commit("system:x", "x' = a*y + 1"))
    assert typed.engine.document.object("system").params["equations"][0] == "a*y + 1"


# ---------------------------------------------------------------- B2: what a count change is


def test_b2_equilibria_entering_through_the_window_edge_are_not_bifurcations(ws):
    _ok(ws.explorer_edit([{"text": "x' = sin(k*x)"}, {"text": "y' = -y"}]))
    r = ws.sweep_parameter("k", 1, 6, n=12)
    kinds = [c["kind"] for c in r["candidates"]]
    assert "saddle-node" not in kinds and "pitchfork" not in kinds, r["candidates"]
    first = min(r["candidates"], key=lambda c: c["value"])
    assert first["kind"] == "enters or leaves the window"
    # 2 pi / k reaches x = 4 (plus the search's pad of 1e-6 of the window)
    assert first["value"] == pytest.approx(math.pi / 2, abs=1e-5)
    # points the coarse grid finds or misses are not named either, and not converged
    for c in r["candidates"]:
        if c["kind"] == "not named":
            assert c["converged"] is False and c["why"]
    assert len({(c["kind"], c["value"]) for c in r["candidates"]}) == len(kinds)


def test_b2_non_isolated_equilibria_are_not_named_and_not_converged(ws):
    _ok(ws.explorer_edit([{"text": "x' = a*x*y"}, {"text": "y' = a*y"}]))
    r = ws.sweep_parameter("a", -1, 1, n=3)
    assert r["candidates"] and all(c["kind"] == "not named" for c in r["candidates"])
    assert all(not c["converged"] for c in r["candidates"]) and r["converged"] is False
    for c in r["candidates"]:
        a, b = c["test"].rsplit("from ", 1)[1].split(" to ")
        assert a != b


# ---------------------------------------------------------------- should-fix


def test_s1_numbers_keep_six_significant_digits(ws):
    assert explorer._r6(1.23456789e-9) == 1.23457e-9 and explorer._r6(123456.789) == 123457.0
    assert explorer._r6(float("inf")) is None
    _ok(ws.explorer_preset("van_der_pol"))
    assert ws.find_equilibria()["tolerance"]["newton_residual"] == 1e-9
    s = ws.flow_probe([2, 0], T=10, samples=4)["seeds"][0]
    assert s["dt_check"]["max_difference"] > 0  # tiny, not rounded to 0


def test_s2_positions_stop_where_dt_and_dt_half_part(ws):
    _ok(ws.explorer_preset("lorenz"))
    s = ws.flow_probe([1, 1, 1], T=40, samples=32)["seeds"][0]
    until = s["positions_until"]
    assert until <= s["chaotic_beyond"]
    if s["dt_check"]["agree_until"] is not None:
        assert until <= s["dt_check"]["agree_until"]
    assert all(x["t"] <= until for x in s["samples"])


def test_s3_a_stopped_seed_or_too_few_steps_is_not_converged(ws):
    _ok(ws.explorer_edit([{"text": "x' = -1000000 x"}, {"text": "y' = -y"}]))
    r = ws.flow_probe([1, 1], T=1, samples=4)
    assert r["converged"] is False and "stopped" in r["seeds"][0]["dt_check"]["why"]
    ws2 = Workspace(ws.root / "two")
    ws2.scene_new("e", look="none")
    _ok(ws2.explorer_preset("van_der_pol"))
    r = ws2.flow_probe([2, 0], T=0.05, samples=2)
    assert r["converged"] is False and "compared" in r["seeds"][0]["dt_check"]["why"]


def test_s4_a_transcritical_bifurcation_is_found_and_named(ws):
    _ok(ws.explorer_edit([{"text": "x' = mu*x - x^2"}, {"text": "y' = -y"}]))
    for n in (4, 6, 9):
        r = ws.sweep_parameter("mu", -1, 1, n=n)
        (c,) = r["candidates"]
        assert c["kind"] == "transcritical" and c["value"] == pytest.approx(0, abs=1e-6), n


def test_s5_candidates_say_what_limits_their_precision(ws):
    _ok(ws.explorer_preset("pitchfork"))
    (c,) = ws.sweep_parameter("mu", -1, 1, n=8)["candidates"]
    p = c["precision"]
    assert {"bracket", "merge_distance", "newton_residual", "note"} <= set(p)
    assert p["merge_distance"] > p["bracket"]
    _ok(ws.explorer_preset("hopf"))
    (c,) = ws.sweep_parameter("mu", -1, 1, n=8)["candidates"]
    assert c["kind"] == "Hopf" and "eigenvalue_tolerance" in c["precision"]


def test_s6_every_answer_is_strict_json_and_overflow_is_refused(ws):
    _ok(ws.explorer_preset("lorenz"))
    with pytest.raises(ToolError, match="finite"):
        ws.sweep_parameter("rho", -1e308, 1e308, n=3)
    with pytest.raises(ToolError, match="window"):
        ws.find_equilibria(window=[[-1e308, 1e308]] * 3)
    for r in (
        ws.explorer_state(),
        ws.find_equilibria(),
        ws.flow_probe([1e150, 1, 1], T=1),
        ws.sweep_parameter("rho", 20, 28, n=4),
        ws.explorer_edit([{"text": "y = x"}]),
    ):
        _strict(r)


def test_s7_ids_keep_clear_of_the_camera(ws):
    r = _ok(ws.explorer_edit([{"text": "camera = 3"}, {"text": "q = 1", "base": "camera"}]))
    ids = [row["id"] for row in r["rows"]]
    assert "camera" not in ids and len(set(ids)) == 2


@pytest.mark.parametrize("passes", MODES)
def test_s8_a_kept_row_says_it_is_not_drawn(ws, passes):
    r = _ok(ws.explorer_edit([{"text": "y = x^2"}, {"text": "q = sin x"}]))
    graph, bad = r["rows"]
    assert graph["drawn"] is False and "not drawn" in graph["why"] and graph["error"] is None
    assert bad["drawn"] is False and bad["error"]
    s = Scene("rows", passes=passes)
    ops, _ = explorer.plan(s.engine.document, [{"text": "y = x^2"}, {"text": "q = sin x"}])
    assert s.engine.apply(ops, origin="script").ok
    found = {i.ids[0]: i for i in s.engine.lint() if i.code == "explorer_row"}
    assert "not drawn" in found["graph"].message and "does not read" in found["row"].message
    assert all(i.severity == "warning" and i.times for i in found.values())


# ---------------------------------------------------------------- nits


def test_nits(ws):
    _ok(ws.explorer_preset("pendulum"))
    with pytest.raises(ToolError, match="true or false"):
        ws.explorer_preset("pendulum", replace="yes")
    for step in (1e-300, 1e300):
        with pytest.raises(ToolError, match="step"):
            ws.explorer_edit([{"id": "gamma", "step": step}])
    # an undamped pendulum's exponent is 0: never 'converged' near 0
    lam = ws.flow_probe([1, 0], T=50, variables={"gamma": 0})["seeds"][0]["largest_lyapunov"]
    assert lam["converged"] is False
    from phasor.domains import explore_math as em

    big = ws.flow_probe([1, 0], T=1e4, samples=2)
    assert big["clamped"] and big["steps"] == em.MAX_RUN
