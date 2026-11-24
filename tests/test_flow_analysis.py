"""The Explorer's analysis pack (Milestone 30, EXPLORER.md; advice section 2.6 rows 5, 7-12):
sections, Lyapunov exponents, time series, twin probes, divergence maps, bifurcation sweeps
and limit cycles, against known numbers; cache keys, batch independence, scrubbing, budgets
and hostile formulas, as for the flow core (Milestone 26)."""

from __future__ import annotations

import json
import math
import runpy
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, flow
from phasor.domains import flow_analysis as fl
from phasor.domains import flow_analysis_math as fa
from phasor.domains import flow_math as fm
from phasor.engine import Engine
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))  # nothing in the repo


def _preset(name: str, **over):
    p = fm.PRESETS[name]
    vals = {k: v[0] for k, v in p.variables.items()}
    vals.update(over)
    return fm.System.parse(p.equations), vals, p


def _window(p):
    lo, hi = zip(*p.window, strict=True)
    return lo, hi


# ---------------------------------------------------------------- sections


def test_a_crossing_is_interpolated_inside_its_step():
    """A uniform field moves on straight lines: linear interpolation is exact there."""
    s = fm.System.parse(("1", "0.5"))
    run = fa.SectionRun(s.fn, (), np.array([[0.0, 0.0], [0.013, 1.0]]), 0.1,
                        fa.Sampler.plane((1, 0), 0.37, +1)).advance(20)  # fmt: skip
    cr = run.crossings()
    assert cr["seed"].tolist() == [0, 1]
    assert cr["points"] == pytest.approx(np.array([[0.37, 0.185], [0.37, 1.1785]]), abs=1e-12)
    assert cr["times"] == pytest.approx([0.37, 0.357], abs=1e-12)


def test_a_centre_crosses_its_section_once_a_turn():
    s = fm.System.parse(("y", "-x"))  # x = cos t, y = -sin t from (1, 0): y falls at 2 pi k
    run = fa.SectionRun(s.fn, (), np.array([[1.0, 0.0]]), 0.01,
                        fa.Sampler.plane((0, 1), 0.0, -1)).advance(3200)  # fmt: skip
    cr = run.crossings()
    assert cr["times"] == pytest.approx([2 * np.pi * k for k in (1, 2, 3, 4, 5)], abs=1e-4)
    assert np.abs(cr["points"][:, 0] - 1).max() < 1e-4  # linear: wrong by about dt^2
    up = fa.SectionRun(s.fn, (), np.array([[1.0, 0.0]]), 0.01,
                       fa.Sampler.plane((0, 1), 0.0, +1)).advance(3200).crossings()  # fmt: skip
    assert up["times"] == pytest.approx([np.pi * k for k in (1, 3, 5, 7, 9)], abs=1e-4)
    down = fa.SectionRun(s.fn, (), np.array([[1.0, 0.0]]), 0.01,
                         fa.Sampler.plane((0, 1), 0.0, 0)).advance(3200).crossings()  # fmt: skip
    assert len(down["times"]) == 10  # both ways: twice a turn


def test_crossings_do_not_depend_on_the_batch_or_how_far_was_asked_first():
    s, vals, _ = _preset("lorenz")
    v = s.values_of(vals)
    seeds = np.random.default_rng(4).normal(0, 5, (300, 3)) + np.array([0.0, 0.0, 25.0])
    plane = fa.Sampler.plane((0, 0, 1), 27.0, +1)
    big = fa.SectionRun(s.fn, v, seeds, 0.01, plane, max_step=4.0).advance(300).crossings()
    for k in (0, 7, 299):
        for batch in (seeds[k : k + 1], seeds[k : k + 7]):
            one = fa.SectionRun(s.fn, v, batch, 0.01, plane, max_step=4.0).advance(300).crossings()
            mine = one["seed"] == 0
            theirs = big["seed"] == k
            assert np.array_equal(one["points"][mine], big["points"][theirs])
            assert np.array_equal(one["times"][mine], big["times"][theirs])
    stepped = fa.SectionRun(s.fn, v, seeds, 0.01, plane, max_step=4.0)
    stepped.advance(120)
    stepped.advance(300)
    again = stepped.crossings()
    assert all(np.array_equal(again[k], big[k]) for k in big)
    assert np.array_equal(stepped.crossings(upto=120)["times"],
                          big["times"][big["times"] <= 1.2])  # fmt: skip


def test_maxima_are_crossings_of_the_velocity():
    s = fm.System.parse(("y", "-x"))
    cr = fa.SectionRun(s.fn, (), np.array([[0.0, -2.0]]), 0.01,
                       fa.Sampler.maxima(0)).advance(1000).crossings()  # fmt: skip
    assert cr["points"][:, 0] == pytest.approx(2.0, abs=1e-4)  # the amplitude
    assert np.diff(cr["times"]) == pytest.approx(2 * np.pi, abs=1e-6)


def test_a_section_keeps_at_most_its_cap(monkeypatch):
    monkeypatch.setattr(fa, "MAX_CROSSINGS", 5)
    s = fm.System.parse(("y", "-x"))
    run = fa.SectionRun(s.fn, (), np.array([[1.0, 0.0]] * 3), 0.01,
                        fa.Sampler.plane((0, 1), 0.0, +1)).advance(2000)  # fmt: skip
    assert run.crossings()["full"] and len(run.crossings()["times"]) == 5


# ---------------------------------------------------------------- Lyapunov


def test_lyapunov_of_lorenz_is_about_0_906():
    s, vals, _ = _preset("lorenz")
    res = fa.lyapunov(s, (1.0, 1.0, 20.0), 200.0, values=vals, dt=0.01)
    assert res["value"] == pytest.approx(0.906, abs=0.05) and res["converged"]
    assert res["chaotic_beyond"] == pytest.approx(math.log(1e16) / res["value"])
    assert 35 < res["chaotic_beyond"] < 45  # "Lorenz parts across machines near flow time 40"
    again = fa.lyapunov(s, (1.0, 1.0, 20.0), 200.0, values=vals, dt=0.01)
    assert again == res  # no random numbers: the same answer every time


def test_lyapunov_of_a_saddle_is_its_largest_eigenvalue():
    s = fm.System.parse(("x", "-2*y"))
    res = fa.lyapunov(s, (0.1, 1.0), 20.0, dt=0.01, transient=2.0)
    assert res["value"] == pytest.approx(1.0, abs=1e-3) and res["converged"]
    sink = fa.lyapunov(fm.System.parse(("-x", "-3*y")), (1.0, 1.0), 20.0, dt=0.01, transient=2)
    assert sink["value"] == pytest.approx(-1.0, abs=1e-3) and sink["chaotic_beyond"] is None


def test_lyapunov_of_a_limit_cycle_is_about_zero():
    s, vals, _ = _preset("van_der_pol")
    res = fa.lyapunov(s, (1.0, 0.0), 400.0, values=vals, dt=0.01, transient=50.0)
    assert abs(res["value"]) < 0.02 and res["chaotic_beyond"] is None


@pytest.mark.slow
def test_lyapunov_of_rossler_is_about_0_07():
    s, vals, _ = _preset("rossler")
    res = fa.lyapunov(s, (1.0, 1.0, 0.0), 1000.0, values=vals, dt=0.02, transient=50.0)
    assert res["value"] == pytest.approx(0.071, abs=0.015)


def test_lyapunov_of_a_path_that_escapes_says_so():
    s = fm.System.parse(("x*x", "-y"))
    res = fa.lyapunov(s, (1.0, 1.0), 10.0, dt=0.01, transient=0.0, escape=(np.zeros(2), 50.0))
    assert res["value"] is None and res["stopped"] == "escaped" and not res["converged"]


def test_lyapunov_budget_is_clamped():
    s, vals, _ = _preset("lorenz")
    res = fa.lyapunov(s, (1, 1, 20), 1e5, values=vals, dt=0.01, transient=1990.0)
    assert "steps" in res["clamped"] and res["steps"] <= fa.MAX_LYAPUNOV_STEPS


def test_a_statistic_is_rounded_to_its_spread():
    assert fa.statistic(0.906321, 0.024) == 0.906
    assert fa.statistic(0.906321, 0.0024) == 0.9063
    assert fa.statistic(0.906321, 2.4) == 0.9
    assert fa.statistic(float("nan"), 0.1) is None
    assert fa.statistic(1.23456789, 0.0) == 1.234568


# ---------------------------------------------------------------- limit cycles


def test_the_hopf_cycle_has_period_two_pi_and_radius_root_mu():
    s, vals, _ = _preset("hopf", mu=0.5)
    c = fa.limit_cycle(s, (1.0, 0.3), values=vals, dt=0.01, scale=5.6)
    assert c["kind"] == "cycle" and c["converged"] and c["loops"] == 1
    assert c["period"] == pytest.approx(2 * np.pi, abs=1e-7)
    assert np.linalg.norm(c["point"]) == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert np.linalg.norm(c["path"], axis=1) == pytest.approx(math.sqrt(0.5), abs=1e-5)


def test_van_der_pol_has_period_6_663():
    s, vals, _ = _preset("van_der_pol", mu=1.0)
    c = fa.limit_cycle(s, (1.0, 0.0), values=vals, dt=0.01, scale=12.8)
    assert c["kind"] == "cycle" and c["period"] == pytest.approx(6.6633, abs=1e-3)
    assert c["box"][1][0] == pytest.approx(2.0086, abs=1e-3)  # the amplitude


def test_a_period_doubled_cycle_takes_two_loops():
    s, vals, _ = _preset("rossler", c=3.5)
    c = fa.limit_cycle(s, (1, 1, 0), values=vals, dt=0.02, transient=100, max_time=200,
                       tol=1e-5, scale=41.0)  # fmt: skip
    assert c["kind"] == "cycle" and c["loops"] == 2
    chaos = fa.limit_cycle(s, (1, 1, 0), values={**vals, "c": 5.7}, dt=0.02, transient=100,
                           max_time=200, tol=1e-5, scale=41.0)  # fmt: skip
    assert chaos["kind"] == "none" and chaos["period"] is None and chaos["returns"] > 10


def test_rest_points_escapes_and_centres_are_reported_as_such():
    s, vals, _ = _preset("pendulum")
    rest = fa.limit_cycle(s, (1.0, 0.0), values=vals, dt=0.02, scale=12.8)
    assert rest["kind"] == "equilibrium" and np.abs(rest["point"]).max() < 1e-4
    away = fa.limit_cycle(fm.System.parse(("x", "1")), (1.0, 0.0), dt=0.01,
                          escape=(np.zeros(2), 100.0))  # fmt: skip
    assert away["kind"] == "stopped"
    centre = fa.limit_cycle(fm.System.parse(("y", "-x")), (1.0, 0.0), dt=0.01, scale=5.6)
    assert centre["kind"] == "none" and "centre" in centre["note"]  # (audit: not a cycle)


# ---------------------------------------------------------------- sweeps


@pytest.mark.parametrize(
    ("name", "var", "label", "value"),
    [
        ("saddle_node", "mu", "saddle-node", 0.0),
        ("pitchfork", "mu", "pitchfork", 0.0),
        ("hopf", "mu", "Hopf", 0.0),
    ],
)
def test_normal_forms_give_their_bifurcation(name, var, label, value):
    s, vals, p = _preset(name)
    lo, hi = _window(p)
    sw = fa.sweep(s, vals, var, -1.0, 1.0, 24, lo=lo, hi=hi)
    assert [c["label"] for c in sw["candidates"]] == [label]
    assert sw["candidates"][0]["value"] == pytest.approx(value, abs=1e-5)


def test_lorenz_pitchfork_at_one_and_hopf_at_the_formula():
    s, vals, p = _preset("lorenz")
    lo, hi = _window(p)
    sw = fa.sweep(s, vals, "rho", 0.5, 30.0, 24, lo=lo, hi=hi)
    sigma, beta = 10.0, 8.0 / 3.0
    hopf = sigma * (sigma + beta + 3) / (sigma - beta - 1)  # 24.7368...
    found = {c["label"]: c["value"] for c in sw["candidates"]}
    assert found == pytest.approx({"pitchfork": 1.0, "Hopf": hopf}, abs=1e-4)
    assert len(sw["candidates"]) == 2  # C+ and C- lose stability together: one candidate
    assert [r["count"] for r in sw["rows"]][:2] == [1, 3]


def test_transcritical_and_window_edges():
    s = fm.System.parse(("mu*x - x**2", "-y"))
    sw = fa.sweep(s, {"mu": 0.0}, "mu", -1.0, 1.0, 16, lo=(-2, -2), hi=(2, 2))
    assert [(c["label"], round(c["value"], 5)) for c in sw["candidates"]] == [
        ("transcritical", 0.0)]  # fmt: skip
    edge = fm.System.parse(("x - mu", "-y"))
    sw = fa.sweep(edge, {"mu": 0.0}, "mu", 1.0, 3.0, 12, lo=(-2, -2), hi=(2, 2))
    assert [c["label"] for c in sw["candidates"]] == ["enters or leaves the window"]


def test_a_sweep_of_a_name_the_system_does_not_use_says_so():
    from phasor.domains import explore_math as em

    s, vals, _ = _preset("hopf")
    with pytest.raises(fm.FunctionError, match="nope"):
        fa.sweep(s, vals, "nope", 0, 1, 8, lo=(-2, -2), hi=(2, 2))
    big = fa.sweep(s, vals, "mu", -1, 1, 10_000, lo=(-2, -2), hi=(2, 2))
    assert len(big["rows"]) == em.MAX_SWEEP and big["clamped"] == ["values"]


def _rossler_sweep(lo_c, hi_c, n):
    s, vals, _ = _preset("rossler")
    return fa.sweep(s, vals, "c", lo_c, hi_c, n, mode="attractor", x0=(1, 1, 0), dt=0.02,
                    transient=150.0, sample_time=100.0, escape=(np.zeros(3), 200.0),
                    max_step=2.0)  # fmt: skip


def test_rossler_doubles_its_period():
    sw = _rossler_sweep(2.0, 4.2, 23)
    per = [r["period"] for r in sw["rows"]]
    assert per[0] == 1 and 2 in per
    first = [per.index(1), per.index(2)] + ([per.index(4)] if 4 in per else [])
    assert first == sorted(first)  # 1, then 2, then 4: a cascade
    assert "period doubling" in {c["label"] for c in sw["candidates"]}
    # one value alone (a sweep has two ends: the other a hair away) moves as among the others
    k = per.index(2)
    v = float(sw["values"][k])
    alone = _rossler_sweep(v, v + 1e-9, 2)
    assert np.array_equal(alone["samples"][0], sw["samples"][k], equal_nan=True)


def test_an_attractor_sweep_is_bounded_by_its_work(monkeypatch):
    monkeypatch.setattr(fa, "MAX_SWEEP_WORK", 2000.0)
    sw = _rossler_sweep(2.0, 3.0, 8)
    assert "work" in sw["clamped"] and 8 * sw["steps"] <= 2000


# ---------------------------------------------------------------- areas


def test_polygon_area_and_liouville_for_a_constant_divergence():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    assert fa.polygon_area(np.c_[np.cos(t), np.sin(t)]) == pytest.approx(np.pi, rel=1e-3)
    sc = Scene("liouville", passes=None)
    sc.add(flow.System(id="d", equations=("y", "-sin(x) - 0.3*y"), tau=3.0,
                       window=((-4, 4), (-3, 3))))  # fmt: skip
    sc.add(flow.DivergenceMap(id="d.div", disk_center=(1.0, 0.5), disk_radius=0.4))
    sem = sc.engine.inspect(0.0, id="d.div")["objects"][0]["semantic"]
    assert sem["min"] == sem["max"] == -0.3 and float(sem["formula"]) == -0.3
    disk = sem["disk"]
    assert disk["predicted"] == pytest.approx(disk["polygon0"] * math.exp(-0.9), rel=1e-5)
    assert disk["area"] == pytest.approx(disk["predicted"], rel=2e-3)  # an outline's polygon


# ---------------------------------------------------------------- objects


def _scene(passes=None) -> Scene:
    s = Scene("analysis_all", passes=passes)
    s.add(flow.System(id="vdp", **flow.preset("van_der_pol"), tau=4.0, pos=(-3.5, 0.5, 0),
                      scale=0.45))  # fmt: skip
    s.add(flow.Variable(id="mu", name="mu", value=1.0, min=0.0, max=3.0))
    s.add(flow.Probe(id="vdp.probe", x0=(1.0, 0.0)))
    s.add(flow.Section(id="vdp.sec", normal=(0, 1), seeds=((1.0, 0.0), (0.5, 0.5))))
    s.add(flow.TwinProbes(id="vdp.twins", x0=(1.0, 0.0), eps=1e-5, lyapunov=20.0))
    s.add(flow.TimeSeries(id="vdp.ts", source="vdp.probe", pos=(0.5, 0.5, 0), extent=(3, 1.5)))
    s.add(flow.TimeSeries(id="vdp.dist", source="vdp.twins", pos=(0.5, -1.5, 0),
                          extent=(3, 1.5)))  # fmt: skip
    s.add(flow.DivergenceMap(id="vdp.div", counts=48, disk_center=(1, 0), disk_radius=0.3,
                             opacity=0.5))  # fmt: skip
    s.add(flow.LimitCycle(id="vdp.cycle", x0=(1.0, 0.0), transient=20.0, max_time=70.0,
                          tol=1e-5))  # fmt: skip
    s.add(flow.BifurcationDiagram(id="vdp.bif", variable="mu", range=(-1.0, 1.0), count=12,
                                  pos=(0.5, -3.8, 0), extent=(3, 1.8)))  # fmt: skip
    return s


@pytest.mark.parametrize("passes", MODES)
def test_every_analysis_type_inspects_and_lints_clean(passes):
    e = _scene(passes).engine
    data = e.inspect(0.0, detail="full")
    json.dumps(data)
    sem = {o["id"]: o["semantic"] for o in data["objects"]}
    assert sem["vdp.sec"]["crossings"] >= 1 and sem["vdp.sec"]["first"][0]["at"][1] == 0.0
    assert sem["vdp.ts"]["components"] == ["x", "y"] and sem["vdp.ts"]["range"]["x"]
    assert sem["vdp.dist"]["log"] and sem["vdp.twins"]["lyapunov"]["method"].startswith("Ben")
    assert sem["vdp.div"]["backdrop"] and sem["vdp.div"]["disk"]["area"] > 0
    assert sem["vdp.cycle"]["found"] == "cycle"
    assert sem["vdp.cycle"]["period"] == pytest.approx(6.6633, abs=1e-3)
    # mu = 0: the origin turns from a stable to an unstable spiral (a Hopf bifurcation; the
    # cycle it makes is not found: a sweep of equilibria sees only the equilibria)
    assert [c["label"] for c in sem["vdp.bif"]["candidates"]] == ["Hopf"]
    assert sem["vdp.bif"]["candidates"][0]["value"] == pytest.approx(0.0, abs=1e-5)
    assert "global" in sem["vdp.bif"]["not_detected"]
    assert not [i for i in e.lint() if i.severity == "error"]


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_every_analysis_type_draws():
    e = _scene("auto").engine
    full = e.still(0.0, (480, 270)).astype(int)
    for oid in ("vdp.sec", "vdp.twins", "vdp.ts", "vdp.dist", "vdp.div", "vdp.cycle",
                "vdp.bif"):  # fmt: skip
        f = e.fork()
        assert f.apply({"op": "remove_object", "id": oid}, origin="script").ok
        assert np.abs(full - f.still(0.0, (480, 270)).astype(int)).max() > 8, oid


def test_a_variable_moves_the_analyses_and_a_fresh_engine_agrees():
    e = _scene().engine
    assert e.apply({"op": "set_param", "path": "mu.value", "value": 2.0}).ok
    seen = e.inspect(0.0)["objects"]
    sem = {o["id"]: o["semantic"] for o in seen}
    assert sem["vdp.cycle"]["period"] == pytest.approx(7.6298, abs=2e-3)  # mu = 2
    fresh = {o["id"]: o["semantic"] for o in Engine(e.document).inspect(0.0)["objects"]}
    assert json.dumps(fresh, sort_keys=True) == json.dumps(sem, sort_keys=True)


def test_the_tools_lyapunov_reads_a_document():
    s = Scene("lyap_tool", passes=None)
    s.add(flow.System(id="lo", **flow.preset("lorenz")))
    st = s.engine.state_at(0.0)
    res = flow.lyapunov("lo", (1, 1, 20), 120.0, state=st)
    assert res["value"] == pytest.approx(0.906, abs=0.08) and res["statistic"]
    assert set(res) >= {"value", "spread", "T", "converged", "chaotic_beyond"}
    with pytest.raises(ValueError, match="no System"):
        flow.lyapunov("nope", (1, 1, 20), 10.0, state=st)
    direct = flow.lyapunov(fm.System.parse(("x", "-y")), (0.1, 1.0), 20.0, transient=2.0)
    assert direct["value"] == pytest.approx(1.0, abs=1e-3)


def test_convert_to_script_names_the_analysis_types_through_flow(tmp_path):
    from phasor.core.diff import diff_documents
    from phasor.core.pycode import document_to_python

    doc = _scene().engine.document
    code = document_to_python(doc)
    assert "flow.Section(" in code and "flow.BifurcationDiagram(" in code
    path = tmp_path / "again.py"
    path.write_text(code, encoding="utf-8")
    again = runpy.run_path(str(path), run_name="x")["s"].engine.document
    assert {o.id: o.type for o in again.objects} == {o.id: o.type for o in doc.objects}
    assert diff_documents(doc, again)["same"] or all(
        "track" in str(c) or "timeline" in str(c)
        for c in diff_documents(doc, again)["changes"])  # fmt: skip


def test_bad_parameters_are_refused_with_a_reason():
    s = Scene("bad_analysis", passes=None)
    s.add(flow.System(id="p"))
    for bad, match in [
        (lambda: flow.Section(id="p.a", normal=(0, 0)), "normal"),
        (lambda: flow.Section(id="p.b", seeds=((1.0,),)), "seed"),
        (lambda: flow.TimeSeries(id="p.c", components="xw"), "components"),
        (lambda: flow.BifurcationDiagram(id="p.d", range=(1.0, 1.0)), "range"),
        (lambda: flow.TwinProbes(id="p.e", eps=0.0), "greater"),
        (lambda: flow.LimitCycle(id="p.f", x0=(float("nan"), 0.0)), "x0"),
        (lambda: flow.TimeSeries(id="p.g", extent=(0.0, 1.0)), "extent"),
    ]:  # fmt: skip
        with pytest.raises(Exception, match=match):
            s.add(bad())


def test_a_sweep_of_an_unknown_variable_draws_and_says_why():
    s = Scene("unknown_var", passes=None)
    s.add(flow.System(id="p"))
    s.add(flow.BifurcationDiagram(id="p.bif", variable="gamma"))
    sem = s.engine.inspect(0.0, id="p.bif")["objects"][0]["semantic"]
    assert "gamma" in sem["why"] and sem["rows"] == []


# ---------------------------------------------------------------- hostile and budgets


@pytest.mark.parametrize("passes", MODES)
@pytest.mark.parametrize("formula", ["x**1e9", "exp(exp(exp(x)))", "1/(x-y)", "log(x)"])
def test_hostile_formulas_analyse_without_tracebacks(formula, passes):
    sc = Scene("hostile_analysis", passes=passes)
    sc.add(flow.System(id="h", equations=(formula, "-y + a"), constants="a=0.5", tau=2.0,
                       window=((-2, 2), (-2, 2))))  # fmt: skip
    for child in (flow.Section(id="h.sec", seeds=((0.5, 0.1), (-0.5, 1.0)), normal=(0, 1),
                               offset=0.3, direction="both"),
                  flow.TwinProbes(id="h.twins", x0=(0.5, 0.1), lyapunov=5.0),
                  flow.TimeSeries(id="h.ts", x0=(0.5, 0.1)),
                  flow.DivergenceMap(id="h.div", counts=32, disk_radius=0.3,
                                     disk_center=(0.5, 0.5)),
                  flow.LimitCycle(id="h.cyc", x0=(0.5, 0.1), transient=2.0, max_time=5.0),
                  flow.BifurcationDiagram(id="h.bif", variable="a", range=(-1, 1), count=6),
                  flow.BifurcationDiagram(id="h.bif2", variable="a", range=(-1, 1), count=4,
                                          mode="attractor", transient=1.0, sample_time=2.0,
                                          pos=(-3.6, 1.6, 0))):  # fmt: skip
        sc.add(child)
    json.dumps(sc.engine.inspect(0.0))
    assert not [i for i in sc.engine.lint() if i.severity == "error"]
    if passes is not None:
        assert sc.engine.still(0.0, (320, 180)).shape == (180, 320, 3)


@pytest.mark.parametrize("passes", MODES)
def test_budgets_are_clamped_and_linted(passes, monkeypatch):
    monkeypatch.setattr(fa, "MAX_SECTION_WORK", 2000.0)
    s = Scene("analysis_budget", passes=passes)
    s.add(flow.System(id="r", **flow.preset("rossler"), tau=20.0))  # 1000 steps
    s.add(flow.Section(id="r.sec", seeds=((1, 1, 0), (2, 1, 0), (3, 1, 0)), normal=(0, 1, 0)))
    sem = s.engine.inspect(0.0, id="r.sec")["objects"][0]["semantic"]
    assert "work" in sem["budget"]["clamped"] and sem["steps"] * 3 <= 2000
    assert sem["crossings"] is None and "note" in sem  # past SAFE_TAU: no chaotic counts
    found = {(i.code, i.ids[0]): i for i in s.engine.lint()}
    assert ("flow_budget", "r.sec") in found and found["flow_budget", "r.sec"].times


def _peak(fn):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        out = fn()
        return out, time.perf_counter() - start, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


# The at-maximum run is pc_tasks/13_analysis_maxima.md (minutes); the CI one is test_s_scaled_down_maxima_clamp_and_scale
# in test_flow_analysis_audit.py.


# ---------------------------------------------------------------- caches and scrubbing


def _clear_memos():
    from phasor.domains import flow as fl26

    for memo in (fl._SEC, fl._LYA, fl._AREA, fl._SWP, fl._CYC, fl26._PATHS, fl26._MAN,
                 fl26._EQ):  # fmt: skip
        memo.clear()


def _memo_scene() -> Scene:
    sc = Scene("analysis_memo", passes=None)
    sc.add(flow.System(id="p", equations=("y", "mu*(1 - x**2)*y - x"), constants="mu=1.0",
                       window=((-4, 4), (-5, 5)), tau=6.0))  # fmt: skip
    sc.add(flow.Variable(id="mu", name="mu", value=1.0, min=0, max=3))
    sc.add(flow.Section(id="p.sec", normal=(0, 1), seeds=((1.0, 0.0), (0.2, 0.1))))
    sc.add(flow.TwinProbes(id="p.twins", x0=(1.0, 0.0), lyapunov=10.0, transient=2.0))
    sc.add(flow.TimeSeries(id="p.ts", source="p.twins"))
    sc.add(flow.LimitCycle(id="p.cyc", x0=(1.0, 0.0), transient=10.0, max_time=30.0, tol=1e-5))
    sc.add(flow.DivergenceMap(id="p.div", counts=24, disk_center=(1.0, 0.0), disk_radius=0.3))
    sc.add(flow.BifurcationDiagram(id="p.bif", variable="mu", range=(-1.0, 1.0), count=6))
    sc.add(flow.BifurcationDiagram(id="p.att", variable="mu", range=(0.5, 2.0), count=4,
                                   mode="attractor", x0=(1.0, 0.0), transient=10.0,
                                   sample_time=20.0, pos=(-3.6, 1.6, 0)))  # fmt: skip
    return sc


CHANGES = [
    ("p.window", [[-3.0, 5.0], [-5.0, 5.0]]),
    ("p.escape", 3.0),
    ("p.dt", 0.02),
    ("p.tau", 4.0),
    ("p.constants", "mu=1.5"),
    ("mu.value", 1.3),
    ("p.sec.normal", [1.0, 1.0]),
    ("p.sec.offset", 0.2),
    ("p.sec.direction", "both"),
    ("p.sec.seeds", [[1.0, 0.0]]),
    ("p.sec.transient", 2.0),
    ("p.twins.x0", [1.1, 0.0]),
    ("p.twins.eps", 1e-4),
    ("p.twins.lyapunov", 12.0),
    ("p.twins.transient", 3.0),
    ("p.cyc.x0", [2.0, 0.0]),
    ("p.cyc.transient", 12.0),
    ("p.cyc.max_time", 25.0),
    ("p.cyc.tol", 5e-6),
    ("p.div.counts", 25),
    ("p.div.disk_center", [1.0, 0.5]),
    ("p.div.disk_radius", 0.2),
    ("p.bif.range", [-1.0, 0.5]),
    ("p.bif.count", 7),
    ("p.bif.grid", 7),
    ("p.bif.coord", "y"),
    ("p.att.x0", [0.5, 0.0]),
    ("p.att.transient", 8.0),
    ("p.att.sample_time", 15.0),
    ("p.att.sample", "section"),
]


# Provably the same answer: the Variable "mu" overrides the constant; an attracting cycle is
# reached from either start; a radius of 3 stops nothing reported here; the section's samples
# of the same cycles give the same periods (their counts are not reported for chaotic rows).
SAME = {"p.constants", "p.att.x0", "p.escape", "p.att.sample", "p.cyc.tol"}  # tol < dt^2: hidden


@pytest.mark.parametrize(("path", "value"), CHANGES)
def test_memos_hold_every_input(path, value):
    """Advice section 5 item 9 for the analysis memos: change one input alone; what the
    engine says changes, and equals a run with every memo cleared."""
    e = _memo_scene().engine

    def seen():
        return json.dumps(e.inspect(0.0)["objects"], sort_keys=True)

    before = seen()  # fills the memos
    assert e.apply({"op": "set_param", "path": path, "value": value}).ok
    warm = seen()
    _clear_memos()
    assert warm == seen()
    if path not in SAME:
        assert warm != before, path


def test_scrubbing_in_any_order_matches_a_fresh_engine():
    """Advice section 5 item 10: the flow time follows the scene time (``rate``); inspect at
    scrambled times, then compare each with a fresh engine."""
    sc = Scene("analysis_scrub", passes=None)
    sc.add(flow.System(id="r", **flow.preset("rossler"), rate=1.0))
    sc.add(flow.Section(id="r.sec", normal=(0, 1, 0), direction="down",
                        seeds=((1, 1, 0), (2, 0, 0), (0, 3, 1))))  # fmt: skip
    sc.add(flow.TwinProbes(id="r.twins", x0=(1, 1, 0), lyapunov=0.0))
    sc.add(flow.TimeSeries(id="r.ts", source="r.twins"))
    sc.add(flow.DivergenceMap(id="r.div", counts=16, z=1.0))
    e = sc.engine
    times = [7.5, 1.25, 9.0, 3.0, 9.0, 0.0, 6.0, 2.5]

    def said(engine, t):
        return json.dumps([o["semantic"] for o in engine.inspect(t)["objects"]], sort_keys=True)

    seen = {t: said(e, t) for t in times}
    for t in sorted(set(times)):
        _clear_memos()
        assert seen[t] == said(Engine(e.document), t), t


# ---------------------------------------------------------------- examples


def test_the_rossler_example_shows_its_section_and_its_exponent():
    e = runpy.run_path(str(ROOT / "examples" / "rossler_section.py"), run_name="x")["s"].engine
    sem = {o["id"]: o["semantic"] for o in e.inspect(e.duration)["objects"]}
    sec = sem["ro.section"]
    assert sec["seeds"] == 64 and sec["crossings"] is None  # long chaotic run: no counts
    lya = sem["ro.twins"]["lyapunov"]
    assert lya["value"] == pytest.approx(0.07, abs=0.03) and lya["statistic"]
    early = {o["id"]: o["semantic"] for o in e.inspect(1.0)["objects"]}
    assert early["ro"]["tau"] <= fm.SAFE_TAU  # the golden frames sit before the chaos
    assert e.lint(strict=True) == []


def test_the_period_doubling_example_shows_the_cascade():
    e = runpy.run_path(str(ROOT / "examples" / "period_doubling.py"), run_name="x")["s"].engine
    sem = {o["id"]: o["semantic"] for o in e.inspect(e.duration)["objects"]}
    rows = sem["cascade"]["rows"]
    periods = [r[1] for r in rows]
    assert periods[0] == 1 and 2 in periods and None in periods
    labels = {c["label"] for c in sem["cascade"]["candidates"]}
    assert "period doubling" in labels
    assert sem["ro2.cycle"]["found"] == "cycle" and sem["ro2.cycle"]["loops"] == 2
    assert e.lint(strict=True) == []


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_rossler_section_golden_storyboard():
    """Short flow time only (the chaos is machine-dependent past SAFE_TAU)."""
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "rossler_section.py"), run_name="x")["s"].engine
    st = e.state_at(5.0)
    assert st["ro"].impl.resolve(st).tau <= 5.0
    # the section from flow time 0 (the example waits 10): the golden shows its crossings
    assert e.apply({"op": "set_param", "path": "ro.section.transient", "value": 0.0}).ok
    sem = e.inspect(5.0, id="ro.section")["objects"][0]["semantic"]
    assert sem["crossings"] > 0
    sheet = e.storyboard([1.6, 3.6, 5.0], size=(480, 270))["image"]
    _golden("rossler_section", sheet, e.pass_runner().renderer_class())


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_period_doubling_golden_storyboard():
    """The sweep cut to its periodic part (c up to 4.15: attracting cycles forget rounding,
    so the long runs draw the same everywhere); the chaotic band is left out."""
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "period_doubling.py"), run_name="x")["s"].engine
    assert e.apply({"op": "set_param", "path": "cascade.range", "value": [2.0, 4.15]}).ok
    sheet = e.storyboard([2.5, e.duration], size=(480, 270))["image"]
    _golden("period_doubling", sheet, e.pass_runner().renderer_class())
