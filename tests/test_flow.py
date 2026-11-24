"""The flow core and its objects (Milestone 26, EXPLORER.md; advice sections 1, 2 and 5):
known equilibria and eigenvalues, 3D classification, batch independence, cache keys,
scrubbing, set refinement, bounded backward runs, budgets and hostile formulas."""

from __future__ import annotations

import json
import runpy
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, flow
from phasor.core.simulation import SimRunner, sim_key
from phasor.domains import cloud_math
from phasor.domains import dynamics_math as dm
from phasor.domains import flow_math as fm
from phasor.engine import Engine
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]
LORENZ = {"sigma": 10.0, "rho": 27.7, "beta": 2.67}


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))  # nothing in the repo


def _lorenz() -> fm.System:
    return fm.System.parse(fm.PRESETS["lorenz"].equations)


# ---------------------------------------------------------------- equilibria, eigenvalues


def test_lorenz_equilibria_and_eigenvalues_match_the_reference():
    """Advice, measured table: C+ -13.840, 0.0852 +/- 10.149i; origin -22.741, -2.67,
    11.741 at (sigma, rho, beta) = (10, 27.7, 2.67)."""
    s = _lorenz()
    found = fm.equilibria(s, LORENZ, (-25, -30, -2), (25, 30, 55), 9)
    pts = found["points"]
    assert len(pts) == 3 and not found["non_isolated"]
    assert found["grid"] == 9 and found["window"][2] == [-2.0, 55.0]
    c = np.sqrt(2.67 * 26.7)
    assert np.allclose(pts, [[-c, -c, 26.7], [0, 0, 0], [c, c, 26.7]], atol=1e-9)
    plus = fm.classify(s.jacobian_exact(pts[2], LORENZ))
    assert plus["kind"] == "saddle-focus" and plus["dims"] == {"s": 1, "u": 2, "c": 0}
    ev = plus["eigenvalues"]
    assert ev[0] == pytest.approx((-13.840, 0.0), abs=5e-4)
    assert ev[1] == pytest.approx((0.0852, 10.149), abs=5e-4)
    assert ev[2] == pytest.approx((0.0852, -10.149), abs=5e-4)
    origin = fm.classify(s.jacobian_exact(pts[1], LORENZ))
    assert origin["kind"] == "saddle" and origin["dims"] == {"s": 2, "u": 1, "c": 0}
    assert [e[0] for e in origin["eigenvalues"]] == pytest.approx([-22.741, -2.67, 11.741],
                                                                  abs=5e-4)  # fmt: skip


def test_the_lorenz_objects_report_the_same_numbers():
    s = Scene("lorenz_numbers", passes=None)
    s.add(flow.System(id="lorenz", **flow.preset("lorenz", rho=27.7, beta=2.67)))
    s.add(flow.Equilibria(id="lorenz.eq"))
    s.add(flow.EquilibriumView(id="lorenz.cplus", near=(8, 8, 27), name="C+"))
    sem = {o["id"]: o["semantic"] for o in s.engine.inspect(0.0)["objects"]}
    eq = sem["lorenz.eq"]
    assert eq["count"] == 3 and [p["kind"] for p in eq["points"]] == [
        "saddle-focus", "saddle", "saddle-focus"]  # fmt: skip
    assert eq["points"][1]["at"] == [0.0, 0.0, 0.0]  # rounded, never -0.0
    card = sem["lorenz.cplus"]["card"]
    assert card[0].startswith("C+ = (8.443, 8.443, 26.7)") and "saddle-focus" in card[0]
    assert card[1] == "eigenvalues -13.84, 0.08516 +/- 10.15i"
    assert all(line.isascii() for line in card)
    div = sem["lorenz"]["divergence"]
    assert div["min"] == div["max"] == -13.67  # -sigma - 1 - beta: volumes contract


@pytest.mark.parametrize(
    ("matrix", "kind", "dims"),
    [
        (np.diag([-1.0, -2.0, -3.0]), "stable node", (3, 0, 0)),
        (np.diag([1.0, 2.0, 3.0]), "unstable node", (0, 3, 0)),
        (np.diag([1.0, -2.0, -3.0]), "saddle", (2, 1, 0)),
        ([[-1, -2, 0], [2, -1, 0], [0, 0, -3]], "stable spiral", (3, 0, 0)),
        ([[1, -2, 0], [2, 1, 0], [0, 0, 3]], "unstable spiral", (0, 3, 0)),
        ([[0.1, -2, 0], [2, 0.1, 0], [0, 0, -3]], "saddle-focus", (1, 2, 0)),
        (np.diag([0.0, -1.0, -2.0]), "non-hyperbolic", (2, 0, 1)),
        ([[0, -1, 0], [1, 0, 0], [0, 0, -1]], "non-hyperbolic", (1, 0, 2)),
        ([[-1, 1, 0], [0, -1, 1], [0, 0, -1]], "degenerate", (3, 0, 0)),
    ],
)
def test_classification_in_3d(matrix, kind, dims):
    for scale in (1.0, 1e-9, 1e6):  # the tolerance follows the Jacobian's size
        c = fm.classify(np.asarray(matrix, dtype=float) * scale)
        assert c["kind"] == kind and kind in fm.KINDS3
        assert (c["dims"]["s"], c["dims"]["u"], c["dims"]["c"]) == dims
        for key in ("Es", "Eu", "Ec"):
            if len(c[key]) > 1:
                B = np.array(c[key])
                assert pytest.approx(np.eye(len(B)), abs=1e-9) == B @ B.T


def test_2d_classification_keeps_its_kinds_and_adds_subspaces():
    c = fm.classify(np.array([[1.0, 0.0], [0.0, -1.0]]))
    assert c["kind"] == "saddle" and c["kind"] in fm.KINDS and dm.KINDS == fm.KINDS
    assert c["dims"] == {"s": 1, "u": 1, "c": 0}
    assert np.abs(c["Eu"][0]) == pytest.approx([1.0, 0.0])
    assert dm.classify is fm.classify


def test_a_line_of_equilibria_is_reported_as_non_isolated():
    s = fm.System.parse(("y", "0"))
    found = fm.equilibria(s, {}, (-2, -2), (2, 2), 9)
    assert found["non_isolated"] and np.abs(found["points"][:, 1]).max() < 1e-12
    none = fm.equilibria(fm.System.parse(("1", "x")), {}, (-1, -1), (1, 1), 5)
    assert len(none["points"]) == 0 and none["window"] == [[-1.0, 1.0], [-1.0, 1.0]]


def test_the_old_2d_front_is_the_flow_core():
    s = dm.System.parse("y", "-sin(x) - 0.3 * y")
    assert isinstance(s.flow, fm.System) and s.flow.dim == 2
    path = dm.trajectories(s, [[1.0, 0.0]], 4.0, 100)
    direct = fm.integrate(s.flow.fn, (), [[1.0, 0.0]], 100, 0.04,
                          box=((-1e3, -1e3), (1e3, 1e3))).points  # fmt: skip
    assert np.array_equal(path, direct)


def test_presets_parse_and_cloud_reads_lorenz_from_the_table():
    for name, p in fm.PRESETS.items():
        s = fm.System.parse(p.equations)
        assert s.dim == len(p.window), name
        vals = {k: v[0] for k, v in p.variables.items()}
        assert set(s.variables) <= set(vals), name
        lo, hi = zip(*p.window, strict=True)
        fm.equilibria(s, vals, lo, hi, 5)
        assert p.note and all(v[1] < v[2] for v in p.variables.values())
    assert cloud_math.ATTRACTORS["lorenz"] == (10.0, 28.0, 8.0 / 3.0)
    kw = flow.preset("lorenz", rho=27.7)
    assert "rho=27.7" in kw["constants"] and kw["equations"][0] == "sigma*(y - x)"
    with pytest.raises(ValueError, match="no variable"):
        flow.preset("lorenz", gamma=1)


# ---------------------------------------------------------------- stepping


def _run(fn, vals, seeds, n, dt=0.01, max_step=4.0):
    p = np.asarray(seeds, dtype=np.float64).copy()
    for _ in range(n):
        p, _ = fm.step(fn, vals, p, dt, max_step)
    return p


@pytest.mark.parametrize("formula", ["lorenz", "hostile"])
def test_a_seed_moves_the_same_alone_or_in_a_batch(formula):
    """Advice section 5 item 11: seed k's path with 1, 7 and 1000 seeds is bit-identical."""
    if formula == "lorenz":
        s, vals = _lorenz(), _lorenz().values_of(LORENZ)
        seeds = np.random.default_rng(1).normal(0, 6, (1000, 3)) + np.array([0.0, 0.0, 25.0])
    else:
        s, vals = fm.System.parse(("sin(3*y) + exp(-x**2)", "cos(x*y) - 0.1*y")), ()
        seeds = np.random.default_rng(2).normal(0, 2, (1000, 2))
    big = _run(s.fn, vals, seeds, 150)
    for k in (0, 1, 499, 993):
        assert np.array_equal(_run(s.fn, vals, seeds[k : k + 1], 150)[0], big[k])
        assert np.array_equal(_run(s.fn, vals, seeds[k : k + 7], 150)[0], big[k])


def test_backward_lorenz_is_bounded_by_the_escape_radius():
    s = _lorenz()
    vals = s.values_of(LORENZ)
    start = _run(s.fn, vals, [[1.0, 1.0, 20.0]], 500)  # on the attractor
    path = fm.integrate(s.fn, vals, start, 3000, -0.01, escape=(np.array([0, 0, 25.0]), 200.0),
                        max_step=4.0)  # fmt: skip
    assert path.code[0] in (fm.ESCAPED, fm.TOO_FAST) and 0 < path.at[0] < 3000
    assert np.isfinite(path.points[0, : path.at[0]]).all()
    assert np.isnan(path.points[0, path.at[0] :]).all()


@pytest.mark.parametrize("passes", MODES)
def test_a_stiff_field_is_stopped_not_flung(passes):
    s = fm.System.parse(("-1e6*x", "-y"))
    _q, code = fm.step(s.fn, (), np.array([[1.0, 1.0], [0.0, 1.0]]), 0.01, 0.4)
    assert code.tolist() == [fm.TOO_FAST, fm.ALIVE]
    sc = Scene("stiff", passes=passes)
    sc.add(flow.System(id="st", equations=("-1e6*x", "-y"), tau=0.5))
    sc.add(flow.FlowSet(id="st.disk", center=(1.0, 0.0), size=0.5, count=50))
    assert "flow_too_fast" in {i.code for i in sc.engine.lint()}


@pytest.mark.parametrize("passes", MODES)
@pytest.mark.parametrize("formula", ["x**1e9", "exp(exp(exp(x)))", "1/(x-y)", "log(x)"])
def test_hostile_formulas_draw_and_finish(formula, passes):
    """Advice section 5 item 4: 1000 seeds, 1000 steps, no exception, no hang."""
    s = fm.System.parse((formula, "-y + 0.5"))
    seeds = np.random.default_rng(3).uniform(-2, 2, (1000, 2))
    start = time.perf_counter()
    path = fm.integrate(s.fn, (), seeds, 1000, 0.01, escape=(np.zeros(2), 100.0), max_step=0.2)
    assert time.perf_counter() - start < 60.0  # a guard against a hang, not a timer
    assert path.points.shape == (1000, 1001, 2)
    fm.equilibria(s, {}, (-2, -2), (2, 2), 9)  # non-finite starts are skipped
    sc = Scene("hostile", passes=passes)
    sc.add(flow.System(id="h", equations=(formula, "-y + 0.5"), tau=1.0, window=((-2, 2), (-2, 2))))
    for child in (flow.FieldArrows(id="h.field"), flow.FlowSet(id="h.set", count=300),
                  flow.Tracer(id="h.probe", x0=(0.5, 0.1)), flow.Equilibria(id="h.eq")):  # fmt: skip
        sc.add(child)
    json.dumps(sc.engine.inspect(1.0))
    assert not [i for i in sc.engine.lint() if i.severity == "error"]
    if passes is not None:
        assert sc.engine.still(1.0, (320, 180)).shape == (180, 320, 3)  # it draws


# ---------------------------------------------------------------- the set as a Simulation


def _sim(**kw) -> fm.FlowSim:
    s = _lorenz()
    base = dict(system=s, values=s.values_of(LORENZ), dt=0.01,
                points=fm.fill("ball", (1, 1, 20), 2.0, 64, 3),
                loops=fm.curves("ball", (1, 1, 20), 2.0, 3, 2), samples=16, h_max=1.0,
                refine_every=10, max_step=4.0, escape=((0.0, 0.0, 25.0), 300.0), trail=4)  # fmt: skip
    base.update(kw)
    return fm.FlowSim(**base)


def test_every_parameter_of_a_set_is_in_its_cache_key():
    """Advice section 5 item 9: change each input alone; the key changes."""
    other = fm.System.parse(("sigma*(y - x)", "x*(rho - z) - y", "x*y - beta*z + 0.1"))
    changes = {
        "system": other, "values": (2.7, 27.7, 10.0), "dt": 0.02,
        "points": fm.fill("ball", (1, 1, 20), 2.0, 65, 3),
        "loops": fm.curves("ball", (1, 1, 21), 2.0, 3, 2), "samples": 17, "h_max": 1.5,
        "refine_every": 11, "cap": 1000, "max_step": 3.0,
        "escape": ((0.0, 0.0, 25.0), 301.0), "trail": 5, "direction": -1.0,
    }  # fmt: skip
    base = sim_key(_sim())
    assert sim_key(_sim()) == base
    for name, value in changes.items():
        assert sim_key(_sim(**{name: value})) != base, name


def test_a_slider_the_system_does_not_use_changes_nothing():
    sc = Scene("unused", passes=None)
    sc.add(flow.System(id="sy", equations=("y", "-a*x"), constants="a=1, b=2", tau=0.5))
    st = sc.engine.state_at(0.0)
    r = st["sy"].impl.resolve(st)
    assert r.sys.variables == ("a",) and r.values == {"a": 1.0}
    key = r.key
    assert sc.engine.apply({"op": "set_param", "path": "sy.constants",
                            "value": "a=1, b=3"}).ok  # fmt: skip
    st = sc.engine.state_at(0.0)
    assert st["sy"].impl.resolve(st).key == key


def test_scrubbing_in_any_order_matches_a_fresh_run():
    """Advice section 5 items 10 and 12: refinement included."""
    sim = _sim()
    a = SimRunner(sim, every=7, cache_dir=None)
    order = [0.5, 0.13, 0.9, 0.31, 0.9, 0.0, 0.77, 0.42]
    seen = {t: a.state_at(t) for t in order}
    for t in sorted(set(order)):
        fresh = SimRunner(_sim(), every=1000, cache_dir=None).state_at(t)
        for k in fresh:
            assert np.array_equal(seen[t][k], fresh[k], equal_nan=True), (t, k)


def test_refined_points_are_true_images_of_initial_points():
    sim = _sim(points=np.zeros((0, 3)), trail=0)
    st = SimRunner(sim, cache_dir=None).state_at(0.6)
    assert len(st["s"]) > 16 * len(sim.loops)  # it refined
    n = int(st["n"])
    for k in (0, len(st["s"]) // 2, len(st["s"]) - 1):
        x0 = fm._on_curve(sim.loops[int(st["loop"][k])], st["s"][k : k + 1])
        p, _code = sim._from_start(x0, n, 0.01)
        assert np.array_equal(p[0], st["q"][k], equal_nan=True)


def test_a_set_stops_growing_at_its_cap_and_says_when():
    sim = _sim(points=np.zeros((0, 3)), cap=200, h_max=0.2, trail=0)
    st = SimRunner(sim, cache_dir=None).state_at(1.0)
    assert len(st["s"]) <= 200 and int(st["under"]) > 0


@pytest.mark.parametrize("passes", MODES)
def test_set_under_resolved_is_a_lint(monkeypatch, passes):
    monkeypatch.setattr(fm, "MAX_SET_POINTS", 300)
    sc = Scene("under", passes=passes)
    sc.add(flow.System(id="lo", tau=1.0, **flow.preset("lorenz")))
    sc.add(flow.FlowSet(id="lo.ring", shape="circle", center=(1, 1, 20), size=2.0, samples=32))
    sem = sc.engine.inspect(0.0, id="lo.ring")["objects"][0]["semantic"]
    assert sem["outline"] <= 300 and sem["under_resolved_since"] > 0
    assert "set_under_resolved" in {i.code for i in sc.engine.lint()}


# ---------------------------------------------------------------- objects


def _scene(passes=None) -> Scene:
    s = Scene("flow_all", passes=passes)
    s.add(flow.System(id="pend", constants="gamma=0.3", equations=("y", "-sin(x) - gamma*y"),
                      window=((-4, 4), (-3, 3)), tau=1.5))  # fmt: skip
    s.add(flow.Variable(id="gamma", name="gamma", value=0.3, min=0, max=2))
    s.add(flow.FieldArrows(id="pend.field"))
    s.add(flow.Tracer(id="pend.probe", x0=(1.0, 0.5), label="phi^t(x0)"))
    s.add(flow.FlowSet(id="pend.disk", center=(1.0, 0.5), size=0.4, count=200, fill="both",
                       trail=3))  # fmt: skip
    s.add(flow.Equilibria(id="pend.eq"))
    s.add(flow.EquilibriumView(id="pend.rest", near=(0, 0), name="rest"))
    s.add(flow.Manifold(id="pend.sep", near=(3.1, 0), length=4.0))
    return s


@pytest.mark.parametrize("passes", MODES)
def test_every_flow_type_inspects_and_lints_clean(passes):
    e = _scene(passes).engine
    data = e.inspect(0.0, detail="full")
    json.dumps(data)
    sem = {o["id"]: o["semantic"] for o in data["objects"]}
    assert sem["gamma"]["used_by"] == ["pend"] and sem["pend"]["from_variables"] == ["gamma"]
    assert [p["kind"] for p in sem["pend.eq"]["points"]] == ["saddle", "stable spiral", "saddle"]
    assert sem["pend.rest"]["point"]["kind"] == "stable spiral"
    assert {b["which"] for b in sem["pend.sep"]["branches"]} == {"stable", "unstable"}
    assert sem["pend.disk"]["alive"] == sem["pend.disk"]["points"] + sem["pend.disk"]["outline"]
    assert sem["pend.probe"]["at"] is not None
    assert not [i for i in e.lint() if i.severity == "error"]


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_every_flow_type_draws():
    e = _scene("auto").engine
    full = e.still(0.0, (480, 270)).astype(int)
    for oid in ("pend.field", "pend.probe", "pend.disk", "pend.eq", "pend.rest", "pend.sep"):
        f = e.fork()
        assert f.apply({"op": "remove_object", "id": oid}, origin="script").ok
        assert np.abs(full - f.still(0.0, (480, 270)).astype(int)).max() > 8, oid


def test_a_child_reads_its_parent_and_a_fresh_engine_agrees():
    s = _scene()
    e = s.engine
    before = e.inspect(0.0, id="pend.eq")["objects"][0]["semantic"]
    assert e.apply({"op": "set_param", "path": "pend.equations",
                    "value": ["y", "-sin(x) - gamma*y + 0.5"]}).ok  # fmt: skip
    after = e.inspect(0.0, id="pend.eq")["objects"][0]["semantic"]
    assert after["points"] != before["points"]
    fresh = Engine(e.document)
    assert fresh.inspect(0.0, id="pend.eq")["objects"][0]["semantic"] == after
    # a Variable moves the system: its value overrides the constant
    assert e.apply({"op": "set_param", "path": "gamma.value", "value": 0.0}).ok
    pts = e.inspect(0.0, id="pend.eq")["objects"][0]["semantic"]["points"]
    assert all(p["kind"] in ("saddle", "centre", "non-hyperbolic", "unstable spiral",
                             "stable spiral") for p in pts)  # fmt: skip
    st = e.state_at(0.0)
    assert st["pend"].impl.resolve(st).values == {"gamma": 0.0}


def test_a_child_without_a_system_draws_nothing_and_says_why():
    s = Scene("orphan_child", passes=None)
    s.add(flow.FieldArrows(id="lonely"))
    sem = s.engine.inspect(0.0, id="lonely")["objects"][0]["semantic"]
    assert sem["system"] is None and "System" in sem["why"]


def test_bad_systems_are_refused_with_a_reason():
    s = Scene("bad_flow", passes=None)
    with pytest.raises(Exception, match="constants"):
        s.add(flow.System(id="a", equations=("rho*x", "y")))
    with pytest.raises(Exception, match="2 or 3"):
        s.add(flow.System(id="b", equations=("x",), window=((-1, 1),)))
    with pytest.raises(Exception, match="window"):
        s.add(flow.System(id="c", equations=("y", "x")))  # fine
        s.add(flow.System(id="d", equations=("y", "x", "z")))  # the 2D default window
    with pytest.raises(Exception, match="too large"):
        s.add(flow.System(id="e", equations=("9**9**9", "x")))
    with pytest.raises(Exception, match="unknown term kind"):
        s.add(flow.System(id="f", terms=("magic(pos=(0, 0))",)))
    with pytest.raises(Exception, match="cannot read the term|kind\\(name"):
        s.add(flow.System(id="h", terms=("__import__('os')",)))
    s.add(flow.System(id="g", terms=("vortex(pos=(0, 0), strength=2.0)", "uniform(v=(0.5, 0))")))
    st = s.engine.state_at(0.0)
    v = st["g"].impl.resolve(st).sys.fn.velocity(np.array([[1.0, 0.0]]), ())[0]
    assert v == pytest.approx([0.5, 2.0 / 1.01 - np.sin(1.0)])  # vortex + uniform + pendulum


@pytest.mark.parametrize("passes", MODES)
def test_budgets_are_clamped_and_linted(passes):
    s = Scene("budget", passes=passes)
    s.add(flow.System(id="b3", **flow.preset("rossler")))
    s.add(flow.FieldArrows(id="b3.field", counts=200))  # 8e6 asked
    s.add(flow.Tracer(id="b3.probe", x0=(1, 1, 1)))
    s.set("b3.tau", 500.0)  # 25000 steps at dt 0.02: past MAX_PROBE_STEPS
    sem = {o["id"]: o["semantic"] for o in s.engine.inspect(0.0)["objects"]}
    assert sem["b3.field"]["arrows"] <= flow.MAX_ARROWS
    assert sem["b3.probe"]["steps"] == fm.MAX_PROBE_STEPS and sem["b3.probe"]["at"] is None
    found = {(i.code, i.ids[0]) for i in s.engine.lint()}
    assert ("flow_budget", "b3.field") in found and ("flow_budget", "b3.probe") in found


# ---------------------------------------------------------------- bounds at the maximum


def _peak(fn):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        out = fn()
        return out, time.perf_counter() - start, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.slow
def test_each_type_at_its_documented_maximum_is_bounded():
    """Advice section 5 item 7: time and peak memory at the documented maxima (generous
    bounds: guards against a blow-up, not timers). A set of MAX_POINTS points (with the
    longest trail its points allow) asked for a long flow time is clamped by MAX_WORK, an
    outline likewise; a field of 200 per side is thinned; Newton's starts are capped."""
    s = Scene("max", passes=None)
    s.add(flow.System(id="m", **flow.preset("lorenz")))
    s.set("m.tau", 2000.0)  # 200000 steps: 4e10 point-steps unless clamped
    s.add(flow.FlowSet(id="m.set", count=fm.MAX_POINTS, center=(1, 1, 20), size=3,
                       shape="ball", trail=fm.MAX_TRAIL))  # fmt: skip
    s.add(flow.FlowSet(id="m.ring", shape="ball", fill="boundary", center=(1, 1, 20), size=3,
                       samples=4096, rings=8))  # fmt: skip
    s.add(flow.FieldArrows(id="m.field", counts=200))
    s.add(flow.Equilibria(id="m.eq", grid=64))
    s.add(flow.Tracer(id="m.probe", x0=(1, 1, 1)))
    s.add(flow.Manifold(id="m.wu", near=(0, 0, 0), length=1e4))
    e = s.engine
    sem, took, peak = _peak(lambda: {o["id"]: o["semantic"] for o in e.inspect(0.0)["objects"]})
    work = sem["m.set"]["budget"]
    assert sem["m.set"]["points"] == fm.MAX_POINTS and "work" in work["clamped"]
    assert work["points"] * work["steps"] <= fm.MAX_WORK
    assert "work" in sem["m.ring"]["budget"]["clamped"]
    assert sem["m.field"]["arrows"] <= flow.MAX_ARROWS
    assert sem["m.eq"]["starts"] <= fm.MAX_STARTS
    assert sem["m.probe"]["steps"] == fm.MAX_PROBE_STEPS
    assert took < 600 and peak < 3_000_000_000, (took, peak)
    assert {"flow_budget"} <= {i.code for i in e.lint()}


def _clear_memos():
    from phasor.domains import flow as fl

    fl._PATHS.clear()
    fl._MAN.clear()
    fl._EQ.clear()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("p.window", [[-3.0, 5.0], [-3.0, 3.0]]),  # shifted, same size
        ("p.window", [[-5.0, 5.0], [-4.0, 4.0]]),  # resized
        ("p.escape", 3.0),
        ("p.dt", 0.05),
        ("p.constants", "gamma=0.5"),
        ("gamma.value", 0.1),
    ],
)
def test_memos_hold_every_input(path, value):
    """Advice section 5 item 9 for the module memos: change one input; what the engine
    says equals a run with every memo cleared."""
    sc = Scene("memo", passes=None)
    sc.add(flow.System(id="p", equations=("y", "-sin(x) - gamma*y"), constants="gamma=0.3",
                       window=((-4, 4), (-3, 3)), tau=6.0))  # fmt: skip
    sc.add(flow.Variable(id="gamma", name="gamma", value=0.3, min=0, max=2))
    sc.add(flow.Tracer(id="p.probe", x0=(2.5, 1.5)))
    sc.add(flow.Manifold(id="p.sep", near=(3.1, 0), length=6.0))
    sc.add(flow.Equilibria(id="p.eq"))
    e = sc.engine

    def seen():
        return json.dumps(e.inspect(0.0)["objects"], sort_keys=True)

    seen()  # fills the memos
    assert e.apply({"op": "set_param", "path": path, "value": value}).ok
    warm = seen()
    _clear_memos()
    assert warm == seen()


# ---------------------------------------------------------------- the exit demo


def test_the_exit_demo_shows_what_it_promises():
    e = runpy.run_path(str(ROOT / "examples" / "lorenz_explorer.py"), run_name="x")["s"].engine
    sem = {o["id"]: o["semantic"] for o in e.inspect(e.duration)["objects"]}
    assert sem["lorenz"]["tau"] == 5.0 <= fm.SAFE_TAU  # goldens stay at short flow times
    assert sem["lorenz.eq"]["count"] == 3
    assert sem["lorenz.ball"]["points"] == 2000 and sem["lorenz.ball"]["alive"] == 2000
    assert sem["lorenz.cplus"]["card"][1] == "eigenvalues -13.84, 0.08516 +/- 10.15i"
    assert e.lint(strict=True) == []


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_lorenz_explorer_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "lorenz_explorer.py"), run_name="x")["s"].engine
    sheet = e.storyboard([2.0, 4.6, 9.0], size=(320, 180))["image"]
    _golden("lorenz_explorer", sheet, e.pass_runner().renderer_class())
