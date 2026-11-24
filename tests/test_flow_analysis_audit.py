"""Regression tests for the independent audit of Milestone 30 (MILESTONE_30.md, "Audit"):
each test failed on the commit before its fix."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from phasor import Scene, flow
from phasor.domains import explore_math as em
from phasor.domains import flow_analysis as fl
from phasor.domains import flow_analysis_math as fa
from phasor.domains import flow_math as fm
from phasor.engine import Engine

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))


def S(eqs):
    return fm.System.parse(eqs)


def _preset(name, **over):
    p = fm.PRESETS[name]
    vals = {k: v[0] for k, v in p.variables.items()}
    vals.update(over)
    return fm.System.parse(p.equations), vals, p


# ---------------------------------------------------------------- B1: ranges


def test_b1_a_range_whose_span_is_not_finite_is_refused_not_a_traceback():
    s = Scene("b1", passes=None)
    s.add(flow.System(id="h", **flow.preset("hopf")))
    with pytest.raises(Exception, match="span"):
        s.add(flow.BifurcationDiagram(id="h.b", variable="mu", range=(-1e308, 1e308)))
    sys_, vals, _ = _preset("rossler")
    for lo, hi in ((float("nan"), 1.0), (0.0, float("inf")), (-1e308, 1e308), (1.0, -1.0)):
        with pytest.raises(fm.FunctionError):
            fa.sweep(sys_, vals, "c", lo, hi, 4, mode="attractor", transient=1.0,
                     sample_time=1.0)  # fmt: skip
        with pytest.raises(fm.FunctionError):
            fa.sweep(sys_, vals, "c", lo, hi, 4, lo=(-12, -12, -1), hi=(12, 12, 25))


# ---------------------------------------------------------------- B2: false chaos


@pytest.mark.parametrize(
    ("eqs", "x0"),
    [(("y", "-sin(x)"), (2.0, 0.0)), (("y", "0"), (0.0, 1.0)),
     (("1", "sqrt(2)", "x"), (0.0, 0.0, 0.0))],
)  # fmt: skip
def test_b2_shear_and_neutral_flows_are_not_called_chaotic(eqs, x0):
    res = fa.lyapunov(S(eqs), x0, 100.0, dt=0.01)
    assert res["chaotic_beyond"] is None, res
    assert not (res["converged"] and res["value"] > 0.01), res


def test_b2_what_was_right_stays_right():
    s, vals, _ = _preset("lorenz")
    lo = fa.lyapunov(s, (1, 1, 20), 200.0, values=vals, dt=0.01)
    assert lo["value"] == pytest.approx(0.906, abs=0.05) and lo["converged"]
    assert lo["chaotic_beyond"] is not None
    short = fa.lyapunov(s, (1, 1, 20), 5.0, values=vals, dt=0.01)
    assert not short["converged"] and short["chaotic_beyond"] is None
    for eqs, want in ((("x", "-2*y"), 1.0), (("-x", "-3*y"), -1.0),
                      (("-0.2*x - y", "x - 0.2*y"), -0.2), (("y", "-x"), 0.0)):  # fmt: skip
        r = fa.lyapunov(S(eqs), (0.1, 1.0), 40.0, dt=0.01, transient=2.0)
        assert r["value"] == pytest.approx(want, abs=2e-3), eqs
        assert r["chaotic_beyond"] is None or want > 0


# ---------------------------------------------------------------- B3: periods


def test_b3_a_period_one_cycle_reads_period_one():
    vdp, _, _ = _preset("van_der_pol")
    sw = fa.sweep(vdp, {"mu": 0.0}, "mu", 1.0, 1.2, 12, mode="attractor", x0=(1, 0),
                  transient=100, sample_time=200)  # fmt: skip
    assert [r["period"] for r in sw["rows"]] == [1] * 12
    assert sw["candidates"] == []


def test_b3_the_cascade_reads_one_two_four_without_false_halvings():
    s, vals, _ = _preset("rossler")
    sw = fa.sweep(s, vals, "c", 2.0, 4.15, 24, mode="attractor", x0=(1, 1, 0), dt=0.02,
                  transient=150.0, sample_time=100.0, escape=(np.zeros(3), 200.0),
                  max_step=2.0)  # fmt: skip
    rows = sw["rows"]
    per = [r["period"] for r in rows]
    assert per[0] == 1 and 2 in per
    labels = [c["label"] for c in sw["candidates"]]
    assert "period halving" not in labels and labels[0] == "period doubling"
    chaotic = [r for r in rows if r["state"] == "chaos or a long period"]
    # (before: 19 of 24 None, five "onset of chaos"); below c = 3.8 every value is periodic or
    # honestly "not settled" (slow near a doubling); past it period 8, 16 and the onset of chaos
    # (about 4.2) are not told apart in 100 time units
    assert all(r["value"] > 3.8 for r in chaotic) and len(chaotic) <= 3


# ---------------------------------------------------------------- B4: window edges


def test_b4_roots_entering_at_both_edges_are_not_saddle_nodes():
    sw = fa.sweep(S(["sin(k*x)", "-y"]), {"k": 1.0}, "k", 0.5, 3.0, 20, lo=(-4, -1),
                  hi=(4, 1))  # fmt: skip
    kinds = {c["label"] for c in sw["candidates"]}
    assert kinds == {"enters or leaves the window"}, kinds


# ---------------------------------------------------------------- B5: scrub history


def test_b5_section_full_does_not_depend_on_what_was_asked_before(monkeypatch):
    monkeypatch.setattr(fa, "MAX_CROSSINGS", 5)

    def scene():
        sc = Scene("b5", passes=None)
        sc.add(flow.System(id="c", equations=("y", "-x"), window=((-2, 2), (-2, 2)), rate=1.0))
        sc.add(flow.Section(id="c.sec", normal=(0, 1), direction="both",
                            seeds=((1, 0), (0.5, 0), (1.5, 0))))  # fmt: skip
        return sc.engine

    e = scene()
    e.inspect(9.0)
    late = e.inspect(4.0, id="c.sec")["objects"][0]["semantic"]
    fl.clear_memos()
    fresh = scene().inspect(4.0, id="c.sec")["objects"][0]["semantic"]
    assert late == fresh and fresh["full"] is False


def test_b5_a_section_run_is_safe_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    s, vals, _ = _preset("lorenz")
    v = s.values_of(vals)
    seeds = np.random.default_rng(1).normal(0, 4, (50, 3)) + np.array([0, 0, 25.0])
    run = fa.SectionRun(s.fn, v, seeds, 0.01, fa.Sampler.plane((0, 0, 1), 27.0, +1))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda n: run.advance(n), [100, 300, 200, 300, 50, 300]))
    one = fa.SectionRun(s.fn, v, seeds, 0.01, fa.Sampler.plane((0, 0, 1), 27.0, +1)).advance(300)
    a, b = run.crossings(), one.crossings()
    assert all(np.array_equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------- B6: Lyapunov bounds


def test_b6_every_factor_of_the_lyapunov_run_is_bounded():
    s = S(("x", "-y"))
    t = time.perf_counter()
    r = fa.lyapunov(s, (0.1, 1.0), 10.0, dt=1e-9)
    r2 = fa.lyapunov(s, (0.1, 1.0), 10.0, dt=0.01, blocks=10**6)
    assert time.perf_counter() - t < 60  # a guard, not a timer
    for res in (r, r2):
        assert res["steps"] <= fa.MAX_LYAPUNOV_STEPS and res["clamped"]


@pytest.mark.parametrize(
    "kw",
    [{"T": float("nan")}, {"T": float("inf")}, {"T": 0.0}, {"dt": 0.0}, {"dt": float("nan")},
     {"x0": (1.0,)}, {"transient": float("inf")}],
)  # fmt: skip
def test_b6_nonsense_is_refused_with_a_reason(kw):
    args = {"T": 10.0, "dt": 0.01, "x0": (0.1, 1.0), "transient": 1.0, **kw}
    with pytest.raises(fm.FunctionError):
        fa.lyapunov(S(("x", "-y")), args.pop("x0"), args.pop("T"), **args)


@pytest.mark.parametrize("kw", [{"dt": 0.0}, {"max_time": float("inf")}, {"x0": (1, 2, 3)}])
def test_b6_limit_cycle_refuses_nonsense(kw):
    args = {"dt": 0.01, "max_time": 10.0, "x0": (1.0, 0.0), **kw}
    with pytest.raises(fm.FunctionError):
        fa.limit_cycle(S(("y", "-x")), args.pop("x0"), **args)


@pytest.mark.parametrize("passes", MODES)
def test_b6_twins_at_a_tiny_dt_are_clamped_and_linted(passes):
    sc = Scene("b6", passes=passes)
    sc.add(flow.System(id="p", dt=1e-5, tau=0.01))
    sc.add(flow.TwinProbes(id="p.t", x0=(1.0, 0.0), lyapunov=5.0, transient=1.0))
    sem = sc.engine.inspect(0.0, id="p.t")["objects"][0]["semantic"]
    assert "steps" in sem["budget"]["clamped"]
    # every factor bounded: before, renorm/dt = 50000 steps a renormalisation ran 425000 steps
    assert sem["budget"]["steps"] <= fa.MAX_LYAPUNOV_STEPS + fm.MAX_PROBE_STEPS
    assert ("flow_budget", "p.t") in {(i.code, i.ids[0]) for i in sc.engine.lint()}


# ---------------------------------------------------------------- B7: one implementation


def test_b7_one_lyapunov_one_sweep_one_period():
    assert not hasattr(em, "lyapunov")
    assert not hasattr(fa, "_row") and not hasattr(fa, "_partner")
    sc = Scene("b7", passes=None)
    sc.add(flow.System(id="h", **flow.preset("hopf")))
    sc.add(flow.BifurcationDiagram(id="h.b", variable="mu", range=(-1, 1), count=9))
    sem = sc.engine.inspect(0.0, id="h.b")["objects"][0]["semantic"]
    p = fm.PRESETS["hopf"]
    direct = em.sweep(fm.build(p.equations), {"mu": 0.0}, "mu", -1, 1, 9,
                      [w[0] for w in p.window], [w[1] for w in p.window])  # fmt: skip
    assert [c["label"] for c in sem["candidates"]] == [c["kind"] for c in direct["candidates"]]


# ---------------------------------------------------------------- should-fix


def test_s_limit_cycle_tolerance_is_capped_and_chaos_is_not_a_cycle():
    with pytest.raises(Exception, match="tol"):
        Scene("t", passes=None).add(flow.LimitCycle(id="x", tol=1e-2))
    s, vals, _ = _preset("lorenz")
    c = fa.limit_cycle(s, (1, 1, 20), values=vals, dt=0.01, transient=20, max_time=100,
                       tol=1e-5, scale=80.0)  # fmt: skip
    assert c["kind"] != "cycle"


def test_s_no_false_doubling_near_a_doubling():
    s, vals, _ = _preset("rossler", c=2.7)
    for tr in (300, 500):
        c = fa.limit_cycle(s, (1, 1, 0), values=vals, dt=0.02, transient=tr, max_time=150,
                           tol=1e-5, scale=41.0)  # fmt: skip
        if c["kind"] == "cycle":
            assert c["loops"] == 1 and c["period"] == pytest.approx(5.76, abs=0.05)


@pytest.mark.parametrize(
    ("eqs", "x0"),
    [(("-1e-7*x - y", "x - 1e-7*y"), (1.0, 0.0)), (("y", "-x", "1e-7"), (1.0, 0.0, 0.0)),
     (("y", "-x"), (1.0, 0.0))],
)  # fmt: skip
def test_s_weak_focus_drift_and_centre_are_not_limit_cycles(eqs, x0):
    c = fa.limit_cycle(S(eqs), x0, dt=0.01, transient=5, max_time=60, tol=1e-6, scale=5.0)
    assert c["kind"] != "cycle" and c.get("note")


def test_s_the_period_is_reported_to_its_real_accuracy():
    sc = Scene("vdp10", passes=None)
    sc.add(flow.System(id="v", **{**flow.preset("van_der_pol", mu=10.0),
                                  "window": ((-4, 4), (-20, 20))}))  # fmt: skip
    sc.add(flow.LimitCycle(id="v.c", x0=(2.0, 0.0), transient=40, max_time=150, tol=1e-5))
    sem = sc.engine.inspect(0.0, id="v.c")["objects"][0]["semantic"]
    assert sem["found"] == "cycle"
    assert abs(sem["period"] - 19.07837) <= 2e-4  # shown no finer than dt allows
    assert "dt" in sem["period_accuracy"]


@pytest.mark.parametrize(
    ("eq", "kind"),
    [("(x - mu)**2", None), ("mu + x**2", "saddle-node"), ("mu*x - x**2", "transcritical")],
)
def test_s_equilibria_labels(eq, kind):
    sc = Scene("labels", passes=None)
    sc.add(flow.System(id="s", equations=(eq, "-y"), constants="mu=0",
                       window=((-2, 2), (-2, 2))))  # fmt: skip
    sc.add(flow.BifurcationDiagram(id="s.b", variable="mu", range=(-1, 1), count=21))
    sem = sc.engine.inspect(0.0, id="s.b")["objects"][0]["semantic"]
    labels = [c["label"] for c in sem["candidates"]]
    assert labels == ([] if kind is None else [kind]), labels
    assert sem["columns"][-1] == "non_isolated"


def test_s_non_isolated_rows_and_clamped_count():
    sc = Scene("line", passes=None)
    sc.add(flow.System(id="s", equations=("a*x*y", "a*y"), constants="a=1",
                       window=((-2, 2), (-2, 2))))  # fmt: skip
    sc.add(flow.BifurcationDiagram(id="s.b", variable="a", range=(0.5, 1), count=4))
    rows = sc.engine.inspect(0.0, id="s.b")["objects"][0]["semantic"]["rows"]
    assert all(r[-1] is True for r in rows)


def test_s_reference_slope_is_rounded_to_its_spread():
    sc = Scene("ref", passes=None)
    sc.add(flow.System(id="r", **flow.preset("rossler"), tau=4.0))
    sc.add(flow.TwinProbes(id="r.t", x0=(1, 1, 0), lyapunov=100.0))
    sc.add(flow.TimeSeries(id="r.s", source="r.t"))
    sem = {o["id"]: o["semantic"] for o in sc.engine.inspect(0.0)["objects"]}
    spread = sem["r.t"]["lyapunov"]["spread"] / math.log(10)
    slope = sem["r.s"]["reference_slope"]
    assert slope == fa.statistic(slope, spread)  # no digit finer than the spread allows
    assert len(f"{slope}".split(".")[-1]) <= 3


def test_s_section_crossings_are_accurate_against_a_fine_run():
    s, vals, _ = _preset("lorenz")
    v = s.values_of(vals)
    plane = fa.Sampler.plane((0, 0, 1), 27.0, +1)
    coarse = fa.SectionRun(s.fn, v, np.array([[1.0, 1.0, 20.0]]), 0.01, plane).advance(300)
    fine = fa.SectionRun(s.fn, v, np.array([[1.0, 1.0, 20.0]]), 0.0005, plane).advance(6000)
    a, b = coarse.crossings(), fine.crossings()
    assert len(a["times"]) >= 2
    # the first crossing (before the paths part by RK4's own error): refined, not linear
    assert np.abs(a["points"][0] - b["points"][0]).max() < 1e-5  # linear: about 0.04 off
    assert abs(a["times"][0] - b["times"][0]) < 1e-6


@pytest.mark.parametrize("passes", MODES)
def test_s_dangling_and_cross_system_references_lint(passes):
    sc = Scene("refs", passes=passes)
    sc.add(flow.System(id="a", tau=1.0))
    sc.add(flow.System(id="b", **flow.preset("lorenz"), pos=(20, 0, 0), tau=1.0))
    sc.add(flow.FlowSet(id="a.set", count=20))
    sc.add(flow.Section(id="b.sec", of="a.set", normal=(0, 0, 1), offset=27))
    sc.add(flow.Section(id="b.sec2", of="nobody", normal=(0, 0, 1), offset=27))
    sc.add(flow.TimeSeries(id="a.ts", source="nobody"))
    sem = {o["id"]: o["semantic"] for o in sc.engine.inspect(0.0)["objects"]}
    for oid in ("b.sec", "b.sec2", "a.ts"):
        assert sem[oid]["bad_ref"], oid
    found = {(i.code, i.ids[0]) for i in sc.engine.lint()}
    assert {("flow_ref", "b.sec"), ("flow_ref", "b.sec2"), ("flow_ref", "a.ts")} <= found


def test_s_liouville_uses_the_initial_polygon():
    sc = Scene("liou", passes=None)
    sc.add(flow.System(id="d", equations=("y", "-sin(x) - 0.3*y"), tau=3.0,
                       window=((-4, 4), (-3, 3))))  # fmt: skip
    sc.add(flow.DivergenceMap(id="d.div", disk_center=(1.0, 0.5), disk_radius=0.4))
    disk = sc.engine.inspect(0.0, id="d.div")["objects"][0]["semantic"]["disk"]
    assert disk["ratio"] == pytest.approx(1.0, abs=2e-4)


def test_s_twins_overflow_says_not_finite():
    # the separation itself overflows (each twin finite): "not finite", not "the twins met"
    r = fa.lyapunov(S(("0", "0")), (0.0, 0.0), 10.0, dt=0.1, transient=0.0, d0=1e308)
    assert r["stopped"] == "not finite"


def test_s_scaled_down_maxima_clamp_and_scale(monkeypatch):
    """The at-maximum test, 100x smaller (the full one is pc_tasks/13_analysis_maxima.md)."""
    for name in ("MAX_SECTION_WORK", "MAX_LYAPUNOV_STEPS", "MAX_CYCLE_STEPS", "MAX_SWEEP_WORK"):
        monkeypatch.setattr(fa, name, getattr(fa, name) / 100 if "WORK" in name
                            else getattr(fa, name) // 100)  # fmt: skip
    s = Scene("small_max", passes=None)
    s.add(flow.System(id="m", **flow.preset("lorenz"), tau=2000.0))
    seeds = tuple(tuple(float(v) for v in q) for q in fm.fill("ball", (1, 1, 20), 5.0, 64, 3))
    s.add(flow.Section(id="m.sec", seeds=seeds, normal=(0, 0, 1), offset=27.0))
    s.add(flow.BifurcationDiagram(id="m.bif", variable="rho", range=(20, 30), count=64,
                                  mode="attractor", x0=(1, 1, 20), transient=1e4,
                                  sample_time=1e4))  # fmt: skip
    s.add(flow.BifurcationDiagram(id="m.eqs", variable="rho", range=(0, 30), count=64,
                                  grid=64, pos=(-3.6, 1.6, 0)))  # fmt: skip
    s.add(flow.TwinProbes(id="m.twins", x0=(1, 1, 20), lyapunov=1e4, transient=1e4))
    s.add(flow.LimitCycle(id="m.cyc", x0=(1, 1, 20), transient=1e4, max_time=1e4))
    took = {}
    for oid in ("m.sec", "m.bif", "m.eqs", "m.twins", "m.cyc"):
        t = time.perf_counter()
        took[oid] = (
            s.engine.inspect(0.0, id=oid)["objects"][0]["semantic"],
            time.perf_counter() - t,
        )
    sem = {k: v[0] for k, v in took.items()}
    assert "work" in sem["m.sec"]["budget"]["clamped"]
    assert "work" in sem["m.bif"]["budget"]["clamped"]
    assert "grid" in sem["m.eqs"]["budget"]["clamped"]
    assert "steps" in sem["m.twins"]["budget"]["clamped"]
    assert "steps" in sem["m.cyc"]["budget"]["clamped"]
    assert all(t < 120 for _, t in took.values()), took


def test_s_the_save_load_check_runs_cold():
    """test_final's save/load comparison clears the memos (they would hide a missed input)."""
    import inspect as ins

    import tests.test_final as tf

    assert "clear_memos" in ins.getsource(tf.test_every_example_survives_save_and_load)
    sc = Scene("cold", passes=None)
    sc.add(flow.System(id="h", **flow.preset("hopf")))
    sc.add(flow.LimitCycle(id="h.c", x0=(1, 0.3), transient=10, max_time=30))
    warm = sc.engine.inspect(0.0, id="h.c")
    fl.clear_memos()
    cold = Engine(sc.engine.document).inspect(0.0, id="h.c")
    assert json.dumps(warm["objects"][0]["semantic"]) == json.dumps(cold["objects"][0]["semantic"])
