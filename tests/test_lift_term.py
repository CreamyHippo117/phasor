"""``lift_term``: a term of an overlay equation leaves a 3D object (projected through the
orbiting camera) and glides into the line; and a ``transform`` between two text sizes changes
size gradually (examples/lift_term.py). Checked through ``Engine.state_at`` and the inspector."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Equation, Scene, geo
from phasor.core import text as text_metrics
from phasor.time.verbs import VerbError
from tests.conftest import gl_available
from tests.test_beauty import _golden

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(ROOT / "examples" / "lift_term.py"), run_name="x")["s"]


def sem(e, i, t):
    return e.inspect(t, id=i)["objects"][0]["semantic"]


def _scene():
    s = Scene("lift", passes=None)
    s.use_camera("orbit")
    s.set("camera.distance", 16.0)
    s.set("camera.azimuth", -55.0)
    for oid, x in (("ball", -3.0), ("cell", 3.0)):
        s.add(geo.ParametricSurface(id=oid, fx="sin(u)*cos(v)", fy="sin(u)*sin(v)", fz="cos(u)",
                                    u_range=(0, 3.1416), v_range=(0, 6.2832), res=(8, 16),
                                    pos=(x, 0.0, 0.0)))  # fmt: skip
    s.add(Equation(id="eq", layer="overlay", pos=(0.9, 0.9), parts=("$N=$", "$a$", "$+$", "$b$"),
                   keys=("lhs", "a", "plus", "b")))  # fmt: skip
    s.marker("m1", 1.0)
    s.marker("m2", 4.0)
    s.marker("end", 8.0)
    s.at("m1").orbit(360.0, until="end")
    return s


def test_a_term_starts_on_the_objects_projected_centre_under_an_orbiting_camera(example):
    e = example.engine
    starts = []
    for t in (2.4, 2.8, 3.2):
        lift = sem(e, "count", t)["lifting"][0]
        ball = e.inspect(t, id="sphere")["objects"][0]
        # the inspector's centre of the sphere on screen, in frame heights from the top-left
        px, py = ball["center"]["px"]
        h = e.document.meta.height
        assert lift["start"][0] == pytest.approx(px / h, abs=2e-3)
        assert lift["start"][1] == pytest.approx(py / h, abs=2e-3)
        assert lift["start_norm"] == pytest.approx(ball["center"]["norm"], abs=2e-3)
        starts.append(lift["start"])
    assert np.ptp([s[0] for s in starts]) > 1e-3  # the camera moves, and the start with it


def test_the_term_is_hidden_before_flies_between_and_sits_in_the_line_after(example):
    e = example.engine
    assert "lifting" not in sem(e, "count", 1.5)  # not yet
    assert e.state_at(1.5)["count"].params.lift == (1.0, 0.0, 1.0, 0.0)  # waiting, not shown
    mid = sem(e, "count", 2.6)["lifting"]
    assert [m["part"] for m in mid] == [1] and 0 < mid[0]["progress"] < 1
    assert "lifting" not in sem(e, "count", 4.0)  # done: in the line
    assert e.state_at(4.0)["count"].params.lift[1] == 1.0
    both = e.state_at(7.0)["count"].params.lift
    assert both == (1.0, 1.0, 1.0, 1.0)


def test_it_lands_in_its_place_in_the_line(example):
    e = example.engine
    first, last = 2.0 + 1e-3, 2.0 + 1.3 - 1e-3  # the verb runs for the theme's transform time
    lo = sem(e, "count", first)["lifting"][0]
    assert lo["now"][:2] == pytest.approx(lo["start"][:2], abs=0.02)  # on the object
    hi = sem(e, "count", last)["lifting"][0]
    st = e.state_at(last)
    part, origin, h = st["count"].impl._layout(st)[0][1]
    slot = origin[:2] + np.array([text_metrics.aspect(part)
                                  * h / 2, 0.0])  # fmt: skip
    assert hi["now"][:2] == pytest.approx(slot, abs=0.02)  # in the band


def test_lifts_do_not_depend_on_the_order_they_are_written_in():
    a, b = _scene(), _scene()
    a.at("m1").lift_term("ball", "eq", "a")
    a.at("m2").lift_term("cell", "eq", "b")
    b.at("m2").lift_term("cell", "eq", "b")
    b.at("m1").lift_term("ball", "eq", "a")
    da, db = (s.engine.document.model_dump(mode="json") for s in (a, b))
    assert da["timeline"]["tracks"]["eq.lift"] == db["timeline"]["tracks"]["eq.lift"]
    for t in (0.5, 1.4, 2.0, 4.6, 6.0):
        assert a.engine.state_at(t)["eq"].params.lift == b.engine.state_at(t)["eq"].params.lift


def test_lifts_at_the_same_moment_fly_together_and_overlapping_ones_are_refused():
    s = _scene()
    s.at("m1").lift_term("ball", "eq", "a").lift_term("cell", "eq", "b")
    assert [m["part"] for m in sem(s.engine, "eq", 1.5)["lifting"]] == [1, 3]
    with pytest.raises(VerbError, match="already under way"):
        s.at("m1", 0.3).lift_term("ball", "eq", 0)
    s.at("m2").lift_term("ball", "eq", 0)
    assert s.engine.state_at(8.0)["eq"].params.lift == (1.0, 1.0, 1.0, 1.0)


def test_lift_term_checks_its_arguments():
    s = _scene()
    s.add(Equation(id="other", parts=("$x$",)))
    with pytest.raises(VerbError, match="needs equation"):
        s.at("m1").lift_term("ball", "ball", 0)
    with pytest.raises(VerbError, match="no part named"):
        s.at("m1").lift_term("ball", "eq", "zzz")
    with pytest.raises(VerbError, match="no part 9"):
        s.at("m1").lift_term("ball", "eq", 9)
    with pytest.raises(VerbError, match="no object"):
        s.at("m1").lift_term("nope", "eq", 1)
    s.at("m1").lift_term("ball", "eq", 1)
    with pytest.raises(VerbError, match="already comes from"):
        s.at("m2").lift_term("cell", "eq", 1)


def test_the_example_is_lint_clean_strict(example):
    assert example.engine.lint(strict=True) == []


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.slow
def test_lift_term_golden_storyboard(example):
    e = example.engine
    sheet = e.storyboard([1.0, 2.5, 4.0, 5.6, 7.0], size=(320, 180))["image"]
    _golden("lift_term", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- a size-changing transform


class _Texts:
    """A draw context that only keeps each text and its size."""

    def __init__(self):
        self.out = []

    def text(self, text, origin, height, **kw):
        self.out.append((text, float(height), np.asarray(origin, dtype=float)))


def _ledger():
    s = Scene("ledger", passes=None)
    s.add(Equation(id="title", parts=("$a$", "$=$", "$b$"), pos=(0.0, 0.0, 0.0),
                   style={"size": "title"}))  # fmt: skip
    s.add(Equation(id="note", parts=("$a$", "$=$", "$b$"), pos=(0.0, 0.0, 0.0),
                   style={"size": "annotation"}))  # fmt: skip
    s.marker("go", 1.0)
    s.at("go").transform("title", "note")
    return s


def test_a_transform_between_sizes_changes_size_gradually():
    e = _ledger().engine
    big = e.state_at(0.5)["title"].style["size"]
    small = e.state_at(0.5)["note"].style["size"]
    assert big > small
    assert e.state_at(2.5)["note"].params.from_size == pytest.approx(big)
    heights = []
    for t in (1.0 + 1e-3, 1.65, 2.3, 2.6):
        st = e.state_at(t)
        ctx = _Texts()
        st["note"].impl.draw(ctx, st, "color")
        heights.append(max(h for _, h, _ in ctx.out))
    assert heights[0] == pytest.approx(heights[-1] * big / small, rel=0.02)  # starts at big
    assert heights[0] > heights[1] > heights[2] >= heights[3]  # shrinking, not jumping


def test_a_transform_between_equal_sizes_sets_nothing_new():
    s = Scene("same", passes=None)
    s.add(Equation(id="one", parts=("$a$",)))
    s.add(Equation(id="two", parts=("$a$", "$b$")))
    s.marker("go", 1.0)
    s.at("go").transform("one", "two")
    assert s.engine.state_at(3.0)["two"].params.from_size == 0.0
