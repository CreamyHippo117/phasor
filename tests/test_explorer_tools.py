"""Milestone 27 part B: the Explorer's AI tools (``Workspace`` methods, tested without the
protocol): ``explorer_state``, ``find_equilibria``, ``flow_probe``, ``sweep_parameter``.
Numbers with their provenance; analysis never changes the scene."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from phasor.domains import flow_math as fm
from phasor.workspace.tools import ToolError, Workspace

PROVENANCE = ("system", "window", "grid", "dt", "T", "variables", "converged", "tolerance")


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.scene_new("explorer", look="none")
    return w


@pytest.fixture
def lorenz(ws):
    assert ws.explorer_preset("lorenz")["ok"]
    assert ws.explorer_edit([{"id": "rho", "value": 27.7}, {"id": "beta", "value": 2.67}])["ok"]
    return ws


def _still(ws):
    return ws.engine.version, len(ws.engine._undo), ws.engine.document.model_dump()


def test_lorenz_equilibria_match_milestone_26(lorenz):
    before = _still(lorenz)
    r = lorenz.find_equilibria()
    assert _still(lorenz) == before
    json.dumps(r)
    for k in (*PROVENANCE, "starts"):
        assert k in r, k
    assert r["count"] == 3 and r["grid"] == 9 and r["non_isolated"] is False
    by_x = {round(p["at"][0], 1): p for p in r["points"]}
    c = by_x[8.4]
    assert c["kind"] == "saddle-focus" and c["dims"] == {"s": 1, "u": 2, "c": 0}
    ev = sorted(c["eigenvalues"])
    assert ev[0][0] == pytest.approx(-13.840, abs=1e-3)
    assert ev[1][0] == pytest.approx(0.0852, abs=1e-4) and abs(ev[1][1]) == pytest.approx(
        10.149, abs=1e-3
    )
    o = by_x[0.0]
    assert sorted(a for a, _ in o["eigenvalues"]) == pytest.approx([-22.741, -2.67, 11.741],
                                                                   abs=1e-3)  # fmt: skip


def test_other_values_are_tried_on_a_fork(lorenz):
    before = _still(lorenz)
    r = lorenz.find_equilibria(variables={"rho": 0.5})
    assert r["count"] == 1 and r["variables"]["rho"] == 0.5  # below 1 only the origin
    assert _still(lorenz) == before
    r = lorenz.find_equilibria(window=[[0, 25], [0, 30], [0, 55]], grid=5)
    assert r["window"] == [[0, 25], [0, 30], [0, 55]] and r["grid"] == 5
    with pytest.raises(ToolError, match="does not use"):
        lorenz.find_equilibria(variables={"nope": 1})
    with pytest.raises(ToolError, match="window"):
        lorenz.find_equilibria(window=[[0, 1]])


def test_sweep_finds_the_lorenz_hopf_at_its_formula(lorenz):
    sigma, beta = 10.0, 8.0 / 3.0
    want = sigma * (sigma + beta + 3) / (sigma - beta - 1)
    before = _still(lorenz)
    r = lorenz.sweep_parameter("rho", 20, 28, variables={"sigma": sigma, "beta": beta})
    assert _still(lorenz) == before
    json.dumps(r)
    for k in (*PROVENANCE, "table", "candidates", "not_detected"):
        assert k in r, k
    (hopf,) = [c for c in r["candidates"] if c["kind"] == "Hopf"]
    assert hopf["value"] == pytest.approx(want, abs=1e-8)
    assert len(hopf["at"]) == 2  # C+ and C- cross together: one candidate
    assert hopf["converged"] and "complex pair" in hopf["test"]
    assert "global" in " ".join(r["notes"])
    assert [row["count"] for row in r["table"]] == [3] * 17
    with pytest.raises(ToolError, match="measure"):
        lorenz.sweep_parameter("rho", 20, 28, measure="lyapunov")
    with pytest.raises(ToolError, match="n is"):
        lorenz.sweep_parameter("rho", 20, 28, n=65)


@pytest.mark.parametrize("name, lo, hi, kind, at", [
    ("pitchfork", -1, 1, "pitchfork", 0.0),
    ("saddle_node", -1, 1, "saddle-node", 0.0),
    ("hopf", -1, 1, "Hopf", 0.0),
])  # fmt: skip
def test_local_bifurcations_are_named_by_their_test(name, lo, hi, kind, at):
    p = fm.PRESETS[name]
    r = fm.sweep(fm.build(p.equations), {"mu": 0.0}, "mu", lo, hi, 9,
                 [w[0] for w in p.window], [w[1] for w in p.window])  # fmt: skip
    assert [c["kind"] for c in r["candidates"]] == [kind]
    assert r["candidates"][0]["value"] == pytest.approx(at, abs=1e-6)


def test_probe_reports_a_period_and_the_dt_check(ws):
    assert ws.explorer_preset("van_der_pol")["ok"]
    before = _still(ws)
    r = ws.flow_probe([2, 0], T=40, samples=8)
    assert _still(ws) == before
    json.dumps(r)
    for k in (*PROVENANCE, "steps", "clamped"):
        assert k in r, k
    (s,) = r["seeds"]
    assert s["period"] == pytest.approx(6.663, abs=0.05)
    assert s["dt_check"]["agree"] and r["converged"]
    assert s["chaotic_beyond"] is None and len(s["samples"]) == 8
    assert s["escaped"] is False and s["bounding_box"][1][0] == pytest.approx(2.01, abs=0.02)


def test_probe_of_a_chaotic_path_gives_statistics_past_chaotic_beyond(lorenz):
    r = lorenz.flow_probe([1, 1, 1], T=40, samples=16)
    (s,) = r["seeds"]
    lam = s["largest_lyapunov"]["value"]
    assert 0.5 < lam < 1.4
    beyond = s["chaotic_beyond"]
    assert beyond == pytest.approx(math.log(1e9) / lam, rel=1e-3)
    assert all(x["t"] <= beyond for x in s["samples"])
    assert "after_chaotic_beyond" in s and s["final_speed"] is None
    assert s["dt_check"]["compared_up_to"] <= beyond


def test_probe_escape_backward_and_bounds(ws):
    assert ws.explorer_edit([{"text": "x' = x"}, {"text": "y' = y"}])["ok"]
    r = ws.flow_probe([[1, 0], [0, 0]], T=30, samples=4)
    out, rest = r["seeds"]
    assert out["escaped"] is True and 0 < out["when"] < 30
    assert rest["escaped"] is False and rest["period"] is None
    back = ws.flow_probe([1, 1], T=-2, samples=3)
    assert back["T"] == pytest.approx(-2) and back["seeds"][0]["samples"][-1]["t"] < 0
    big = ws.flow_probe([0.1, 0], T=1e6, samples=2)
    assert big["clamped"] is True and big["steps"] == fm.MAX_PROBE_STEPS // 2
    for bad, match in ((dict(x0=[1, 2, 3]), "coordinates"), (dict(x0=[1, 0], samples=65),
                       "samples"), (dict(x0=[[0, 0]] * 17), "at most"),
                       (dict(x0=[1, 0], T=float("nan")), "finite")):  # fmt: skip
        with pytest.raises(ToolError, match=match):
            ws.flow_probe(**bad)


def test_explorer_state_is_the_one_call_an_ai_needs(lorenz):
    before = _still(lorenz)
    st = lorenz.explorer_state()
    assert _still(lorenz) == before
    json.dumps(st)
    assert st["space"] == "R3" and st["camera"] == "orbit"
    assert [r["kind"] for r in st["rows"]] == ["ode"] * 3 + ["variable"] * 3
    (sysm,) = st["systems"]
    assert sysm["dim"] == 3 and sysm["variables"]["rho"] == 27.7
    eq = sysm["equilibria"]
    for k in ("window", "grid", "dt", "T", "tolerance", "converged"):
        assert k in eq, k
    assert eq["count"] == 3 and eq["grid"] == 9 and eq["window"] == [[-25, 25], [-30, 30],
                                                                      [-2, 55]]  # fmt: skip
    assert st["variables"]["rho"] == {"id": "rho", "value": 27.7, "min": 0.0, "max": 60.0,
                                      "step": 0.01, "slider": True}  # fmt: skip
    assert st["locked"] == [] and isinstance(st["lint"], list)


def test_no_system_is_a_clear_message(ws):
    for call in (lambda: ws.find_equilibria(), lambda: ws.flow_probe([0, 0]),
                 lambda: ws.sweep_parameter("a", 0, 1)):  # fmt: skip
        with pytest.raises(ToolError, match="no system"):
            call()
    assert ws.explorer_state()["systems"] == []


def test_the_tools_are_named_for_mcp_and_documented():
    from pathlib import Path

    from phasor.mcp.server import TOOL_NAMES

    names = ("explorer_state", "explorer_edit", "explorer_preset", "find_equilibria",
             "flow_probe", "sweep_parameter")  # fmt: skip
    assert set(names) <= set(TOOL_NAMES)
    mcp = (Path(__file__).parents[1] / "docs" / "ai" / "MCP.md").read_text(encoding="utf-8")
    assert all(f"`{n}`" in mcp for n in names)


def test_answers_are_plain_json_without_numpy():
    p = fm.PRESETS["duffing"]
    r = fm.probe(fm.build(p.equations), {"delta": 0.3, "alpha": -1.0, "beta": 1.0},
                 np.array([[1.0, 0.0]]), 5.0, 0.01, samples=4, size=6.4)  # fmt: skip

    def walk(v):
        if isinstance(v, dict):
            return all(walk(x) for x in v.values())
        if isinstance(v, list):
            return all(walk(x) for x in v)
        return v is None or type(v) in (bool, int, float, str)

    assert walk(r)
