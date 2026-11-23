"""The diagram kit: number lines, number planes, axes with numbers and unequal scales, bar
charts, tables, matrices, code and an animated number (``MANIM_PARITY.md``, Milestone 19).

Sections: 1 ticks (core/ticks.py); 2 axes; 3 number line and plane; 4 bar chart; 5 table;
6 matrix; 7 code; 8 counter; 9 lint; 10 bounds and hostile input; 11 the examples."""

from __future__ import annotations

import math

import pytest

from phasor.core import ticks

# ---------------------------------------------------------------- 1. ticks


@pytest.mark.parametrize(
    "span, step",
    [(10, 2), (1, 0.2), (7, 1), (100, 20), (0.003, 0.0005), (2e9, 5e8), (3, 0.5), (1e-6, 2e-7)],
)
def test_a_nice_step_is_one_two_or_five_times_a_power_of_ten(span, step):
    assert ticks.nice_step(span) == pytest.approx(step)
    s = ticks.nice_step(span)
    mant = s / 10 ** math.floor(math.log10(s) + 1e-12)
    assert round(mant, 9) in (1.0, 2.0, 5.0)
    assert span / s <= ticks.TARGET + 1e-9


def test_ticks_are_the_multiples_of_the_step_in_the_range_and_deterministic():
    assert ticks.values(-3, 3, 1) == [-3, -2, -1, 0, 1, 2, 3]
    assert ticks.values(-0.5, 1.2, 0.25) == [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
    assert ticks.values(0.1, 0.35, 0.1) == [0.1, 0.2, 0.3]  # no 0.30000000000000004
    assert ticks.values(0, 10, 0) == [0, 2, 4, 6, 8, 10]  # a nice step
    assert all(str(v) != "-0.0" for v in ticks.values(-1, 1, 0.5))
    assert ticks.values(-7.3, 13.9, 0) == ticks.values(-7.3, 13.9, 0)


@pytest.mark.parametrize(
    "lo, hi, step, why",
    [
        (1, 1, 1, "min < max"),  # degenerate
        (2, 1, 1, "min < max"),  # reversed
        (0, 10, 0.001, "more than"),  # too many ticks
        (0, 1, -1, "positive"),
        (float("nan"), 1, 1, "finite"),
        (0, float("inf"), 1, "finite"),
        (0, 1, float("nan"), "finite"),
        (0, 10**400, 1, "too large"),  # a huge int: ValueError, never OverflowError
        (-2e9, 0, 0, "within"),
        (1e8, 1e8 + 1e-3, 0, "too small"),  # lost in rounding
        (0, 1, True, "number"),
    ],
)
def test_a_bad_range_is_a_value_error_naming_the_problem(lo, hi, step, why):
    with pytest.raises(ValueError, match=why):
        ticks.check_range(lo, hi, step, "x_range")


@pytest.mark.parametrize("lo, hi", [(-1e9, 1e9), (0, 1e-9 * 2), (1e9 - 1, 1e9), (-5, -4.9)])
def test_enormous_and_tiny_ranges_get_a_bounded_number_of_ticks(lo, hi):
    ticks.check_range(lo, hi, 0)
    vs = ticks.values(lo, hi, 0)
    assert 2 <= len(vs) <= ticks.TARGET + 2
    assert all(lo - 1e-9 * abs(lo) <= v <= hi + 1e-9 * abs(hi) for v in vs)
    d = ticks.decimals(ticks.step_of(lo, hi, 0))
    names = [ticks.text(v, d) for v in vs]
    assert len(set(names)) == len(names)  # every number written differently


def test_a_range_at_its_tick_limit_is_taken_and_one_past_is_refused():
    ticks.check_range(0, ticks.MAX_TICKS, 1)
    assert len(ticks.values(0, ticks.MAX_TICKS, 1)) == ticks.MAX_TICKS + 1
    with pytest.raises(ValueError):
        ticks.check_range(0, ticks.MAX_TICKS + 1, 1)


def test_nice_ranges_cover_the_data_even_when_it_is_flat():
    assert ticks.nice_range(0.3, 9.2) == (0.0, 10.0, 2.0)
    lo, hi, step = ticks.nice_range(5, 5)
    assert lo < 5 < hi and step > 0
    lo, hi, step = ticks.nice_range(-3, -12)
    assert lo <= -12 and hi >= -3


def test_numbers_are_written_without_minus_zero_and_stay_short():
    assert ticks.text(-0.0001, 2) == "0.00"
    assert ticks.text(1234567.5, 1, group=True) == "1,234,567.5"
    assert ticks.text(3, 0, sign=True) == "+3" and ticks.text(-3, 0, sign=True) == "-3"
    assert ticks.text(float("nan"), 2) == "nan" and ticks.text(float("-inf"), 2) == "-inf"
    assert len(ticks.text(1e308, 10)) < 20
    assert ticks.as_math("-1,000") == "$-1{,}000$"
    assert ticks.name("x", "-2.5") == "x_m2p5" and ticks.name("y", "1,000") == "y_1000"


def test_the_step_that_clears_crowded_numbers_is_nice_and_large_enough():
    s = ticks.clear_step(1.0, spacing=0.5, need=0.8, height=0.2, gap=0.25)
    assert s >= 1.0 * (0.8 + 0.05) / 0.5 and s in (2.0, 5.0)


# ---------------------------------------------------------------- 2. axes

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


def _sem(e, oid, t=0.0):
    return {o["id"]: o for o in e.inspect(t)["objects"]}[oid]["semantic"]


def _ids(e, t=0.0):
    return [o.id for o in e.state_at(t).objects]


def test_axes_take_unequal_scales_and_place_ticks_by_them():
    from phasor import Scene, geo

    s = Scene("a", passes=None)
    s.add(geo.Axes(id="ax", dims=2, x_range=(0, 10, 2), y_range=(0, 1, 0.25), x_scale=0.5,
                   y_scale=4, pos=(-2, -2, 0)))  # fmt: skip
    e = s.engine
    m = _sem(e, "ax")
    assert m["axes"]["x"]["scale"] == 0.5 and m["axes"]["y"]["scale"] == 4
    f = e.inspect(0, id="ax", detail="full")["objects"][0]["features"]
    assert f["x_end"]["world"][0] == pytest.approx([3.0, -2, 0])
    assert f["y_end"]["world"][0] == pytest.approx([-2, 2.0, 0])
    assert _ids(e) == ["ax"]  # no numbers asked for: no parts, as before


@pytest.mark.parametrize("passes", MODES)
def test_axes_numbers_are_parts_named_by_value_and_lint_clean(passes):
    from phasor import Scene, geo

    s = Scene("a", passes=passes)
    s.add(geo.Axes(id="ax", dims=2, x_range=(-4, 4, 1), y_range=(0, 1, 0.25), y_scale=3,
                   numbers=True, pos=(0, -1.5, 0)))  # fmt: skip
    e = s.engine
    ids = _ids(e)
    assert ids[0] == "ax"
    assert ids[1:] == ["ax.x_m4", "ax.x_m3", "ax.x_m2", "ax.x_m1", "ax.x_1", "ax.x_2", "ax.x_3",
                       "ax.x_4", "ax.y_0p25", "ax.y_0p50", "ax.y_0p75", "ax.y_1p00"]  # fmt: skip
    objs = {o["id"]: o for o in e.inspect(0)["objects"]}
    assert objs["ax.x_m2"]["semantic"]["text"] == "$-2$"
    assert objs["ax.x_m2"]["part_of"] == "ax" and "tick" in objs["ax.x_m2"]["tags"]
    st = e.state_at(0)
    box = st["ax"].impl.world_bounds(st)
    num = st["ax.x_m4"].impl.world_bounds(st)
    assert box.lo[1] <= num.lo[1] + 1e-9  # the axes' box holds its numbers
    x = st["ax.x_2"].impl.features(st)["anchor"]
    assert x[0] == pytest.approx(2.0) and x[1] < -1.5
    y = st["ax.y_0p50"].impl.features(st)["anchor"]
    assert y[1] == pytest.approx(-1.5 + 0.5 * 3) and y[0] < 0
    assert e.lint() == []


def test_axes_reject_ranges_and_scales_that_would_not_draw():
    from pydantic import ValidationError

    from phasor import geo

    P = geo.Axes.Params
    for bad in (
        {"x_range": (float("nan"), 1, 1)},
        {"x_range": (0, 1e6, 1)},  # 1e6 ticks
        {"x_range": (1, 1, 1)},
        {"x_scale": float("inf")},
        {"scale": 1e300},
        {"x_range": (0, 1e8, 0), "scale": 100},  # 1e10 world units
        {"labels": ("x" * 200,)},
        {"pos": (0, 10**400, 0)},
    ):
        with pytest.raises((ValidationError, ValueError)):
            P(**bad)
    P(x_range=(0, ticks.MAX_TICKS, 1), x_scale=0.01)  # at the bound: taken
    P(x_range=(-1e9, 1e9, 0), scale=0.5)  # enormous: a nice step
