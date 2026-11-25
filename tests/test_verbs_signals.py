"""The signal verbs (group F2a): ``iterate`` an orbit step by step, ``decompose`` a Fourier sum
into its harmonics and ``recompose`` it. Each is checked through ``Engine.state_at`` and the
inspector, and the new parameters are off by default (the pictures stay what they were)."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, Text, dyn, fourier
from phasor.domains import dynamics_math as dm
from phasor.domains import fourier_math as fm
from phasor.time.verbs import VerbError
from tests.test_beauty import _golden, needs_gl


def sem(e, i, t=0.0):
    return e.inspect(t, id=i)["objects"][0]["semantic"]


def verbs_on(e, path):
    segs = e.document.timeline.tracks.get(path, ())
    return {s.verb for s in segs}


# ---------------------------------------------------------------- the numerics


def test_newton_on_a_cube_root_of_unity_reaches_a_root():
    step = dm.orbit_map("z**3 - 1", "newton")
    pts, converged_at, escaped = dm.orbit(step, 0.45 + 0.9j, 0j, 40, 8.0, 1e-9)
    assert converged_at is not None and not escaped
    assert abs(pts[-1] ** 3 - 1) < 1e-8
    assert pts[1] == pytest.approx(
        0.45 + 0.9j - ((0.45 + 0.9j) ** 3 - 1) / (3 * (0.45 + 0.9j) ** 2)
    )


def test_the_mandelbrot_orbit_escapes_outside_and_stays_inside():
    step = dm.orbit_map("z**2 + a", "map")
    _, _, out = dm.orbit(step, 0j, 1 + 1j, 60, 8.0, 1e-9)
    _, _, inside = dm.orbit(step, 0j, -0.1 + 0.1j, 60, 8.0, 1e-9)
    assert out and not inside


def test_a_formula_is_read_by_the_whitelist_and_never_run():
    s = Scene("bad", passes=None)
    with pytest.raises(ValueError, match="cannot use|unknown name"):
        s.add(dyn.Orbit(id="o", function="__import__('os')"))
    with pytest.raises(ValueError, match="constant"):
        s.add(dyn.Orbit(id="p", kind="newton", function="3"))


def test_the_harmonics_add_up_to_the_partial_sum():
    x = np.linspace(-math.pi, math.pi, 50)
    for kind in ("square", "sawtooth", "triangle"):
        rows = fm.series_harmonics(kind, 6, x)
        assert rows.shape == (6, 50)
        assert rows.sum(axis=0) == pytest.approx(fm.series_partial_sum(kind, 6, x))


# ---------------------------------------------------------------- the object


def _orbit_scene():
    s = Scene("orbit", passes=None)
    s.add(dyn.Orbit(id="newton", kind="newton", function="z**3 - 1", start=(0.45, 0.9),
                    steps=10, pos=(-3, 0, 0)))  # fmt: skip
    s.add(dyn.Orbit(id="logistic", kind="cobweb", function="a*z*(1 - z)", a=(3.7, 0.0),
                    start=(0.2, 0.0), steps=12, scale=3.0, progress=0.5))  # fmt: skip
    return s


def test_an_orbit_reports_its_points_and_whether_it_converged():
    e = _orbit_scene().engine
    n = sem(e, "newton")
    assert n["steps_total"] == 10 and n["steps_shown"] == 10 and len(n["points"]) == 11
    assert n["converged"] and not n["escaped"]
    assert n["limit"] == pytest.approx([-0.5, -0.866025], abs=1e-5)
    c = sem(e, "logistic")
    assert c["steps_shown"] == 6 and not c["converged"] and c["limit"] is None  # chaos at 3.7


def test_an_orbit_that_leaves_is_escaped_and_ends_there():
    s = Scene("escape", passes=None)
    s.add(dyn.Orbit(id="m", function="z**2 + a", a=(1.0, 1.0), steps=50))
    m = sem(s.engine, "m")
    assert m["escaped"] and not m["converged"] and m["steps_total"] < 6


def test_orbit_bounds_features_and_cobweb_geometry():
    e = _orbit_scene().engine
    st = e.state_at(0.0)
    b = st["newton"].impl.world_bounds(st)
    # the start sits at pos + scale * z0 and the tip at the last point shown
    f = st["newton"].impl.features(st)
    assert f["start"] == pytest.approx([-3 + 0.45, 0.9, 0.0])
    assert b.lo[0] <= f["start"][0] <= b.hi[0] and b.lo[1] <= f["start"][1] <= b.hi[1]
    cb = st["logistic"].impl.world_bounds(st)  # the graph and the diagonal are inside
    assert cb.lo[0] <= 0.0 and cb.hi[0] >= 3.0 and cb.hi[1] >= 3.0 * 0.925


def test_the_cobweb_path_is_the_staircase():
    s = _orbit_scene()
    e = s.engine
    st = e.state_at(0.0)
    path = st["logistic"].impl._path(st["logistic"].params)
    x = st["logistic"].impl._orbit(st["logistic"].params)[0].real
    assert path[0] == complex(x[0], 0) and path[1] == complex(x[0], x[1])
    assert path[2] == complex(x[1], x[1]) and len(path) == 2 * len(x) - 1


def test_an_orbit_draws_nothing_at_progress_zero_but_the_start_point():
    s = Scene("zero", passes=None)
    s.add(dyn.Orbit(id="o", progress=0.0, function="z**2 + a", a=(-0.5, 0.2)))
    o = sem(s.engine, "o")
    assert o["steps_shown"] == 0 and len(o["points"]) == 1


# ---------------------------------------------------------------- iterate


def build_iterate(order=("a", "b")):
    s = Scene("iter", passes=None)
    s.add(dyn.Orbit(id="orbit", kind="newton", function="z**3 - 1", start=(0.45, 0.9), steps=10))
    s.add(Text(id="note", text="hi"))
    s.marker("go", 1.0)
    s.marker("end", 9.0)
    steps = {"a": lambda: s.at("go").iterate("orbit", n=8), "b": lambda: s.at("go").reveal("note")}
    for k in order:
        steps[k]()
    return s


def test_iterate_shows_the_steps_one_by_one_at_an_even_pace():
    s = build_iterate(("a",))
    e = s.engine
    assert verbs_on(e, "orbit.progress") == {"iterate"} and verbs_on(e, "orbit.opacity") == {
        "iterate"
    }
    assert e.state_at(0.9)["orbit"].params.opacity == 0  # hidden until its beat
    assert e.state_at(1.0)["orbit"].params.opacity == 1
    assert e.state_at(1.0)["orbit"].params.steps == 8
    # an even pace: halfway through, half the steps (the verb's ease is linear)
    d = 2.5 * sem(e, "orbit", 9.0)["steps_total"] / 10  # n / 10 of the theme's time
    assert e.state_at(1.0 + d / 2)["orbit"].params.progress == pytest.approx(0.5, abs=0.01)
    assert sem(e, "orbit", 1.0 + d / 2)["steps_shown"] == sem(e, "orbit", 9.0)["steps_total"] // 2
    assert sem(e, "orbit", 1.0 + d + 0.1)["steps_shown"] == sem(e, "orbit", 9.0)["steps_total"]
    assert sem(e, "orbit", 1.0)["steps_shown"] == 0


def test_iterate_end_follows_the_number_of_steps():
    s = Scene("pace", passes=None)
    s.add(dyn.Orbit(id="o", function="z**2 + a", a=(-0.3, 0.1), steps=5))
    s.marker("go", 0.0)
    s.marker("again", 20.0)
    a = s.at("go").iterate("o", n=10)
    b = s.at("again").iterate("o", n=20)
    assert a._end == pytest.approx(2.5) and b._end == pytest.approx(5.0)
    e = s.engine
    # each iterate keeps its own number of steps, and plays them all in its own time
    assert e.state_at(0.0)["o"].params.steps == 10 and e.state_at(19.0)["o"].params.steps == 10
    assert e.state_at(20.0)["o"].params.steps == 20
    assert isinstance(e.state_at(20.0)["o"].params.steps, int)
    assert sem(e, "o", 1.25)["steps_shown"] == 5  # halfway through the first: 5 of 10
    assert sem(e, "o", 22.5)["steps_shown"] == 10  # halfway through the second: 10 of 20


def test_an_orbit_that_ends_early_is_paced_by_its_real_steps():
    s = Scene("early", passes=None)
    s.add(dyn.Orbit(id="o", kind="newton", function="z**2 - 1", start=(2.0, 0.0), steps=100))
    s.marker("go", 0.0)
    real = sem(s.engine, "o")["steps_total"]
    assert real < 15
    assert s.at("go").iterate("o")._end == pytest.approx(2.5 * real / 10)


def test_iterate_errors_are_helpful():
    s = _orbit_scene()
    s.add(Text(id="note", text="hi"))
    s.marker("go", 1.0)
    with pytest.raises(VerbError, match="no object 'nope'"):
        s.at("go").iterate("nope")
    with pytest.raises(VerbError, match="iterate needs orbit; 'note' is a text"):
        s.at("go").iterate("note")
    with pytest.raises(VerbError, match="whole number of steps"):
        s.at("go").iterate("newton", n=0)
    with pytest.raises(VerbError, match="whole number of steps"):
        s.at("go").iterate("newton", n=2.5)
    with pytest.raises(VerbError, match="no marker"):
        s.at("missing").iterate("newton")


# ---------------------------------------------------------------- decompose / recompose


def build_wave(order=("d", "r", "n")):
    s = Scene("wave", passes=None)
    s.add(fourier.PartialSum(id="wave", kind="square", terms=4, extent=(4.0, 1.0)))
    s.add(Text(id="note", text="hi"))
    s.marker("apart", 1.0)
    s.marker("together", 6.0)
    s.marker("end", 10.0)
    steps = {"d": lambda: s.at("apart").decompose("wave"),
             "r": lambda: s.at("together").recompose("wave"),
             "n": lambda: s.at("apart").reveal("note")}  # fmt: skip
    for k in order:
        steps[k]()
    return s


def test_decompose_spreads_the_harmonics_and_recompose_brings_them_back():
    e = build_wave().engine
    assert verbs_on(e, "wave.spread") == {"decompose", "recompose"}
    spread = lambda t: e.state_at(t)["wave"].params.spread  # noqa: E731
    assert spread(0.5) == 0 and spread(1.0) == 0
    assert 0 < spread(2.0) < 1
    assert spread(4.0) == pytest.approx(1.0)
    assert 0 < spread(7.0) < 1 and spread(9.0) == 0
    mid = sem(e, "wave", 4.0)
    assert mid["spread"] == 1.0 and mid["harmonics_shown"] == 4
    assert mid["harmonic_amplitudes"][0] == pytest.approx(4 / math.pi, abs=1e-4)
    assert "spread" not in sem(e, "wave", 9.5)  # together again: the plain sum


def test_the_two_verbs_have_their_own_timing_and_ends():
    s = Scene("ends", passes=None)
    s.add(fourier.PartialSum(id="a", terms=3))
    s.add(fourier.PartialSum(id="b", terms=3))
    s.marker("go", 0.0)
    s.marker("back", 20.0)
    m = s.engine._snap.compiled.theme.motion
    assert s.at("go").decompose("a")._end == pytest.approx(m["decompose"]["duration"])
    assert s.at("back").recompose("a")._end == pytest.approx(m["recompose"]["duration"])
    s2 = Scene("two", passes=None)
    s2.add(fourier.PartialSum(id="a", terms=3))
    s2.add(fourier.PartialSum(id="b", terms=3))
    s2.marker("go", 0.0)
    two = s2.at("go").decompose("b", "a")._end  # staggered: the second starts a beat later
    assert two == pytest.approx(m["decompose"]["duration"] + m["stagger"])


def test_the_rows_hang_under_the_sum_and_the_bounds_hold_them():
    e = build_wave().engine
    for t, spread in ((0.0, 0.0), (4.0, 1.0)):
        st = e.state_at(t)
        me = st["wave"]
        b = me.impl.world_bounds(st)
        for curve in me.impl._row_curves(st, me.params) if spread else []:
            assert curve[:, 1].min() >= b.lo[1] - 1e-9 and curve[:, 1].max() <= b.hi[1] + 1e-9
            assert curve[:, 1].max() < 0.0  # under the sum's own row
        assert b.hi[1] == pytest.approx(0.5)  # the top is the sum's, never moved
    assert e.state_at(4.0)["wave"].impl.world_bounds(e.state_at(4.0)).lo[1] < -2.0
    # at spread 0 the bounds are exactly the old plot rectangle
    st = e.state_at(0.0)
    assert st["wave"].impl.world_bounds(st).lo[1] == pytest.approx(-0.5)


def test_more_terms_than_rows_show_only_the_first_rows():
    s = Scene("many", passes=None)
    s.add(fourier.PartialSum(id="w", terms=40, spread=1.0))
    assert sem(s.engine, "w")["harmonics_shown"] == fourier.PartialSum.ROWS


def test_decompose_errors_are_helpful():
    s = build_wave()
    with pytest.raises(VerbError, match="needs a signal that splits into harmonics"):
        s.at("apart").decompose("note")
    with pytest.raises(VerbError, match="no object 'nope'"):
        s.at("apart").recompose("nope")
    with pytest.raises(VerbError, match="at least one"):
        s.at("apart").decompose()


# ---------------------------------------------------------------- order


def test_verbs_give_the_same_scene_whatever_order_they_are_written_in():
    # (one object's own track is written in time order, as for every verb)
    a = build_wave(("d", "n", "r")).engine
    b = build_wave(("n", "d", "r")).engine
    for t in (0.5, 1.5, 4.0, 7.0, 9.0):
        assert a.inspect(t, detail="brief") == b.inspect(t, detail="brief")
    p, q = build_iterate(("a", "b")).engine, build_iterate(("b", "a")).engine
    for t in (0.5, 1.5, 2.5, 5.0):
        assert p.inspect(t, detail="brief") == q.inspect(t, detail="brief")


def test_the_new_params_are_off_by_default():
    s = Scene("off", passes=None)
    s.add(fourier.PartialSum(id="w", terms=5))
    st = s.engine.state_at(0.0)
    assert st["w"].params.spread == 0.0
    assert st["w"].impl.world_bounds(st).lo[1] == pytest.approx(-1.2)
    assert set(sem(s.engine, "w")) == {"kind", "series", "terms", "gibbs_overshoot",
                                       "max_error_away_from_jumps"}  # fmt: skip


# ---------------------------------------------------------------- the example


def test_the_example_is_clean_and_every_segment_is_tagged():
    e = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "verbs_signals.py"),
                       run_name="x")["s"].engine  # fmt: skip
    bad = [i for i in e.lint(strict=True) if i.code == "raw_motion" or i.severity == "error"]
    assert not bad, [i.message for i in bad]


@needs_gl
def test_the_signals_example_matches_its_golden():
    e = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "verbs_signals.py"),
                       run_name="x")["s"].engine  # fmt: skip
    sheet = e.storyboard([2.0, 6.0, 11.0, 14.5], size=(320, 180))["image"]
    _golden("verbs_signals", sheet, e.pass_runner().renderer_class())
