"""Milestone 9: pde, algebra and calc objects, and the roadmap's exit demo as tests: the heat
equation and the dihedral group, driven through the engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Scene, algebra, calc, expr, pde
from phasor.core.objects import REQUIRED_MEMBERS, registered_types
from phasor.domains import algebra_math as am
from tests.conftest import gl_available

NEW = {"pde_profile", "pde_field", "pde_surface", "polygon", "cayley_table", "clock_diagram",
       "graph", "riemann_sum", "taylor", "vector_field"}  # fmt: skip


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))


def sem(e, oid, t=0.0):
    return e.inspect(t, id=oid)["objects"][0]["semantic"]


def test_the_new_types_keep_the_object_contract():
    types = registered_types()
    assert set(types) >= NEW
    for name in NEW:
        assert all(hasattr(types[name], m) for m in REQUIRED_MEMBERS), name
        assert types[name].Style, name


# ---------------------------------------------------------------- exit demo: the heat equation


def test_exit_demo_the_heat_equation_through_the_engine():
    s = Scene("heat", passes=None)
    s.set("camera.height", 8)
    s.add(pde.Profile(id="rod", kind="heat", length=2.0, diffusivity=0.5, modes=(1.0, 0.4),
                      time=expr("t / 10"), pos=(-3, -1, 0)))  # fmt: skip
    s.add(pde.Field(id="plate", kind="heat", mode=(0, 0), diffusivity=0.5, time=expr("t / 10"),
                    size=(4, 4), pos=(4, 0, 0)))  # fmt: skip
    e = s.engine
    heat, error = [], []
    for t in (0.0, 1.0, 2.5, 5.0):
        r = sem(e, "rod", t)
        error.append(r["max_error_vs_exact"])
        heat.append(sem(e, "plate", t)["total_heat"])
    assert max(error) < 3e-3  # the numbers sit on the exact solution all the time
    assert heat == sorted(heat, reverse=True) and heat[-1] < heat[0]  # heat only leaves
    peaks = [sem(e, "plate", t)["max"] for t in (0.0, 2.5, 5.0)]
    assert peaks == sorted(peaks, reverse=True)  # a hot spot spreads out and cools
    assert sem(e, "rod", 5.0)["steps"] > sem(e, "rod", 1.0)["steps"] > 0  # real time stepping
    a, b = sem(e, "rod", 3.0), sem(e, "rod", 3.0)
    assert a == b  # scrubbing is pure
    state = e.state_at(2.0)
    f1 = state["rod"].impl.features(state)["curve"]
    again = e.state_at(2.0)
    assert np.array_equal(f1, again["rod"].impl.features(again)["curve"])


def test_the_decay_rate_the_inspector_reports_is_the_theory_rate():
    s = Scene("rate", passes=None)
    s.add(pde.Profile(id="rod", kind="heat", length=1.0, diffusivity=1.0, modes=(1.0,),
                      time=expr("t / 100")))  # fmt: skip
    e = s.engine
    u0, u1 = sem(e, "rod", 0.0)["max"], sem(e, "rod", 5.0)["max"]  # simulated time 0.05
    assert u1 / u0 == pytest.approx(math.exp(-(math.pi**2) * 0.05), rel=0.01)


def test_waves_advection_and_quantum_profiles_report_their_conserved_quantities():
    s = Scene("others", passes=None)
    s.add(pde.Profile(id="string", kind="wave", speed=1.0, modes=(2,), n=200, time=0.25))
    s.add(pde.Profile(id="pulse", kind="advection", speed=0.5, n=200, time=0.4))
    s.add(
        pde.Profile(id="psi", kind="schrodinger", length=20.0, n=256, dt=0.01, time=1.5, ymax=1.0)
    )
    e = s.engine
    assert sem(e, "string")["max_error_vs_exact"] < 0.03
    assert sem(e, "pulse")["max_error_vs_exact"] < 0.06
    assert sem(e, "psi")["norm"] == pytest.approx(1.0, abs=1e-6)  # probability is conserved
    bad = Scene("bad", passes=None)
    with pytest.raises(Exception, match="steps in"):
        bad.add(pde.Profile(id="far", kind="heat", time=1e6))
        bad.engine.inspect(0.0)


def test_2d_objects_surfaces_and_the_reaction_pattern():
    s = Scene("two", passes=None)
    s.use_camera("orbit", azimuth=-50, elevation=30, distance=12, drift=0)
    s.add(pde.Surface(id="drum", kind="wave", mode=(2, 1), time=0.1, size=(4, 4), n=24))
    s.add(
        pde.Field(id="pattern", kind="reaction", time=400, n=40, seed=2, size=(4, 4), pos=(6, 0, 0))
    )
    e = s.engine
    d = sem(e, "drum")
    assert d["equation"] == "wave" and abs(d["max_error_vs_exact"]) < 0.08
    p = sem(e, "pattern")
    assert p["max"] > p["min"] and p["backdrop"] is True
    assert sem(e, "pattern") == p  # same seed, same pattern
    st = e.state_at(0.0)
    assert st["drum"].impl.features(st)["peak"].shape == (3,)


# ---------------------------------------------------------------- exit demo: the dihedral group


def test_exit_demo_the_dihedral_group_acting_on_a_square_through_the_engine():
    """Rotate a square by r, then flip it: the polygon lands on the group element the table says
    r followed by s is, at every marker."""
    n = 4
    group = am.dihedral(n)
    s = Scene("d4", passes=None)
    s.add(algebra.Polygon(id="sq", n=n))
    s.add(algebra.CayleyTable(id="table", group="D4", pos=(4, 0, 0)))
    s.marker("start", 0.0)
    s.marker("turned", 2.0)
    s.marker("flipped", 4.0)
    s.marker("turned_again", 6.0)
    s.at("turned").morph("sq.rotation", 90.0)
    s.at("flipped").morph("sq.flip", 1.0)
    s.at("turned_again").morph("sq.rotation", 180.0)
    e = s.engine
    seen = [sem(e, "sq", t)["element"] for t in (0.0, 3.5, 5.5)]
    assert seen[0] == "e" and seen[1] == "r"
    assert seen[2] == group.labels[n + 1]  # rotate by one step, then flip: the reflection s r
    assert sem(e, "sq", 2.5)["element"] is None  # mid-turn: no symmetry, no name
    assert sem(e, "sq", 3.5)["flipped"] is False and sem(e, "sq", 5.5)["flipped"] is True
    # one more quarter turn after the flip lands on the element the table predicts
    final = sem(e, "sq", 8.0)
    assert final["element"] == group.labels[n + 2] and final["rotation"] == 180.0
    tab = sem(e, "table")
    assert tab["latin_square"] is True and tab["order"] == 8 and tab["abelian"] is False
    assert tab["subgroup_orders"] == [1, 2, 4, 8]


def test_the_polygons_vertices_match_the_matrices_of_the_group():
    n = 5
    for el in range(2 * n):
        rotation, flip = am.polygon_action(n, el)
        s = Scene("v", passes=None)
        s.add(
            algebra.Polygon(
                id="p", n=n, rotation=math.degrees(rotation), flip=1.0 if flip else 0.0, r=2.0
            )
        )
        v = s.engine.state_at(0.0)["p"].impl.features(s.engine.state_at(0.0))["vertices"][:, :2]
        base = (
            np.array(
                [
                    [
                        math.cos(math.pi / 2 + 2 * math.pi * k / n),
                        math.sin(math.pi / 2 + 2 * math.pi * k / n),
                    ]
                    for k in range(n)
                ]
            )
            * 2.0
        )
        want = base @ am.action_matrix(n, el).T
        assert np.allclose(v, want, atol=1e-9), el
        assert sem(s.engine, "p")["element_index"] == el


def test_the_clock_diagram_shows_a_subgroup_and_its_cosets():
    s = Scene("clock", passes=None)
    s.add(algebra.ClockDiagram(id="c", n=12, step=4))
    s.add(algebra.ClockDiagram(id="whole", n=12, step=5, pos=(6, 0, 0)))
    s.add(algebra.ClockDiagram(id="half", n=12, step=6, pos=(12, 0, 0)))
    e = s.engine
    c = sem(e, "c")
    assert c["subgroup"] == [0, 4, 8] and c["index"] == 4 and c["generator_order"] == 3
    assert all(len(x) == 3 for x in c["cosets"]) and sorted(sum(c["cosets"], [])) == list(range(12))
    assert sem(e, "whole")["subgroup_order"] == 12  # 5 generates everything: gcd(5, 12) = 1
    assert sem(e, "half")["subgroup"] == [0, 6]
    with pytest.raises(Exception, match="unknown group"):
        Scene("x", passes=None).add(algebra.CayleyTable(id="t", group="Q8"))
    with pytest.raises(Exception, match="at most 24"):
        Scene("x", passes=None).add(algebra.CayleyTable(id="t", group="D13"))


# ---------------------------------------------------------------- calc


def test_a_riemann_sum_converges_as_n_is_animated():
    s = Scene("riemann", passes=None)
    s.add(calc.RiemannSum(id="area", function="x**2", a=0.0, b=3.0, n=4.0, rule="left"))
    s.marker("more", 1.0)
    s.at("more").morph("area.n", 64.0)
    e = s.engine
    errs = [sem(e, "area", t)["error"] for t in (0.0, 2.5)]
    assert errs[1] < errs[0] / 8 and sem(e, "area", 2.5)["exact"] == pytest.approx(9.0)
    assert sem(e, "area", 2.5)["pieces"] == 64 and sem(e, "area", 0.0)["pieces"] == 4
    mid = Scene("mid", passes=None)
    mid.add(calc.RiemannSum(id="area", function="x**2", a=0.0, b=3.0, n=10.0, rule="mid"))
    assert sem(mid.engine, "area")["error"] == pytest.approx(
        3 * 0.3**2 / 24 * 2, rel=1e-3
    )  # (b - a) h^2 f'' / 24
    with pytest.raises(Exception, match="greater than a"):
        Scene("bad", passes=None).add(calc.RiemannSum(id="r", a=2.0, b=1.0))
    with pytest.raises(Exception, match="cannot use|unknown name"):
        Scene("bad2", passes=None).add(calc.RiemannSum(id="r", function="__import__('os')"))


def test_taylor_polynomials_improve_with_order_and_graphs_break_at_poles():
    s = Scene("taylor", passes=None)
    s.add(calc.Taylor(id="t", function="exp(x)", center=0.0, order=1.0))
    s.marker("more", 1.0)
    s.at("more").morph("t.order", 6.0)
    e = s.engine
    lo, hi = sem(e, "t", 0.0), sem(e, "t", 3.0)
    assert hi["max_error_within_1"] < lo["max_error_within_1"] / 50
    assert hi["coefficients"][:4] == pytest.approx([1.0, 1.0, 0.5, 1 / 6], abs=1e-6)
    g = Scene("g", passes=None)
    g.add(calc.Graph(id="tan", function="tan(x)", xmin=-4.0, xmax=4.0))
    assert sem(g.engine, "tan")["breaks"] >= 2  # tan jumps at +-pi/2 (and 3 pi / 2 is outside)


def test_vector_fields_report_divergence_and_curl():
    s = Scene("vf", passes=None)
    s.add(calc.VectorField(id="spin", p="-y", q="x"))
    s.add(calc.VectorField(id="out", p="x", q="y", pos=(0, 6, 0)))
    e = s.engine
    assert sem(e, "spin")["curl"] == "2" and sem(e, "spin")["divergence"] == "0"
    assert sem(e, "out")["divergence"] == "2" and sem(e, "out")["curl"] == "0"
    with pytest.raises(Exception, match="unknown name"):
        Scene("bad", passes=None).add(calc.VectorField(id="v", p="z + 1", q="x"))


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_everything_draws():
    s = Scene("all", passes="auto")
    s.set("camera.height", 12)
    s.add(pde.Profile(id="rod", kind="heat", length=2.0, modes=(1.0,), time=0.1, pos=(-6, 2, 0)))
    s.add(pde.Field(id="plate", kind="heat", mode=(0, 0), time=0.1, size=(3, 3), pos=(5, 4, 0)))
    s.add(algebra.Polygon(id="sq", n=5, rotation=72.0, pos=(-5, -2, 0)))
    s.add(algebra.CayleyTable(id="t", group="S3", pos=(0, -2, 0), cell=0.6))
    s.add(algebra.ClockDiagram(id="c", n=8, step=2, pos=(5, -3, 0), cosets=1.0))
    s.add(calc.RiemannSum(id="r", function="sin(x) + 1.5", a=0, b=6, n=12.0, pos=(-3, -5, 0)))
    s.add(calc.VectorField(id="v", p="-y", q="x", pos=(-1, 4, 0), extent=(4, 3)))
    img = s.engine.still(0.0, (480, 270))
    assert img.std() > 8
    assert not [i for i in s.engine.lint() if i.severity == "error"]
