"""The diagram kit: number lines, number planes, axes with numbers and unequal scales, bar
charts, tables, matrices, code and an animated number (``MANIM_PARITY.md``, Milestone 19).

Sections: 1 ticks (core/ticks.py); 2 axes; 3 number line and plane; 4 bar chart; 5 table;
6 matrix; 7 code; 8 counter; 9 lint; 10 bounds and hostile input; 11 the examples."""

from __future__ import annotations

import math

import pytest

import phasor.domains.diagrams  # noqa: F401  (registers the types for raw patches)
from phasor.core import ticks
from phasor.domains import diagrams_math as dm

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
        {"x_range": (1, 1, 1)},
        {"x_range": (0, 1, -1)},
        {"x_scale": float("inf")},
        {"x_scale": 1e300},  # a new parameter (Milestone 19): bounded
        {"x_range": (0, 1e300, 0), "scale": 1e300},  # world coordinates overflow
        {"pos": (0, 10**400, 0)},
    ):
        with pytest.raises((ValidationError, ValueError)):
            P(**bad)
    P(x_range=(0, ticks.MAX_TICKS, 1), x_scale=0.01)  # at the bound: taken
    P(x_range=(-1e9, 1e9, 0), scale=0.5)  # enormous: a nice step
    # What a document saved before Milestone 19 may hold still loads (audit B1): drawn
    # bounded, lint ``clamped`` (see section 12)
    P(x_range=(0, 1e6, 1))
    P(labels=("x" * 200, "y", "z", "w"))
    P(scale=1e300)


# ---------------------------------------------------------------- 3. number line and plane


def _scene(passes=None, **kw):
    from phasor import Scene

    return Scene("d", passes=passes, **kw)


def _objs(e, t=0.0):
    return {o["id"]: o for o in e.inspect(t)["objects"]}


@pytest.mark.parametrize("passes", MODES)
def test_a_number_line_writes_every_tick_as_a_part_and_lints_clean(passes):
    from phasor import dg

    s = _scene(passes)
    s.add(dg.NumberLine(id="line", x_range=(-3, 3, 1), pos=(0, -1, 0), tips="both", label="x"))
    e = s.engine
    assert _ids(e) == ["line"] + [f"line.n_{n}" for n in ("m3", "m2", "m1", "0", "1", "2", "3")]
    st = e.state_at(0)
    f = st["line"].impl.features(st)
    assert f["zero"] == pytest.approx([0, -1, 0]) and f["start"] == pytest.approx([-3, -1, 0])
    two = st["line.n_2"].impl.features(st)["anchor"]
    assert two[0] == pytest.approx(2.0) and two[1] < -1  # under the line
    m = _sem(e, "line")
    assert m["kind"] == "number_line" and m["ticks"] == 7 and m["numbers"]["n"]["step"] == 1
    assert e.lint() == []


def test_a_steep_number_line_writes_its_numbers_beside_it():
    from phasor import dg

    s = _scene()
    s.add(dg.NumberLine(id="up", x_range=(0, 4, 1), angle=90, pos=(0, 0, 0)))
    st = s.engine.state_at(0)
    one = st["up.n_1"]
    assert one.params.anchor == "right" and one.params.pos[0] < 0
    assert one.params.pos[1] == pytest.approx(-1.0)  # 1 is one unit above the middle (2)


@pytest.mark.parametrize("passes", MODES)
def test_a_number_plane_is_a_backdrop_grid_with_axes_and_numbers(passes):
    from phasor import Text, dg

    s = _scene(passes)
    s.add(dg.NumberPlane(id="plane", x_range=(-4, 4, 1), y_range=(-2, 2, 1), y_scale=1.5,
                         faint=1))  # fmt: skip
    s.add(Text(id="note", text="a point", pos=(1.5, 0.75, 0)))  # on the grid: fine
    e = s.engine
    ids = _ids(e)
    assert "plane.axes" in ids and "plane.axes.x_m4" in ids and "plane.axes.y_2" in ids
    m = _sem(e, "plane")
    assert m["backdrop"] and m["y"]["scale"] == 1.5 and m["x"]["scale"] == 1.0
    assert m["grid_lines"] == (9 + 8) + (5 + 4)  # lines at ticks and one between each
    st = e.state_at(0)
    assert st["plane.axes.y_2"].params.pos[1] == pytest.approx(3.0)  # unequal scales
    assert _objs(e)["plane.axes.x_1"]["part_of"] == "plane.axes"
    assert e.lint() == []


# ---------------------------------------------------------------- 4. bar chart


@pytest.mark.parametrize("passes", MODES)
def test_a_bar_chart_names_its_bars_by_their_names_and_scales_them(passes):
    from phasor import dg

    s = _scene(passes)
    s.add(dg.BarChart(id="c", values=(3, 5, 2), names=("Ants", "Bees", "Cats"), width=5,
                      height=3, show_values=True))  # fmt: skip
    e = s.engine
    ids = _ids(e)
    for part in ("c.ants", "c.ants_name", "c.ants_value", "c.bees", "c.cats", "c.y_0", "c.y_5"):
        assert part in ids, part
    st = e.state_at(0)
    hb, ha = st["c.bees"].params.height, st["c.ants"].params.height
    assert hb / ha == pytest.approx(5 / 3)
    m = _sem(e, "c")
    assert m["largest"] == "bees" and m["y_range"] == [0.0, 5.0]
    assert [b["part"] for b in m["bars"]] == ["ants", "bees", "cats"]
    assert _objs(e)["c.bees"]["type"] == "rectangle"
    assert e.lint() == []


def test_bars_grow_from_the_baseline_and_negative_bars_hang_below_it():
    from phasor import dg

    s = _scene()
    s.add(dg.BarChart(id="c", values=(-2, 0, 4), grow=0.5, height=6, pos=(0, 0, 0)))
    st = s.engine.state_at(0)
    m = _sem(s.engine, "c")
    lo, hi = m["y_range"]
    assert lo < 0 < hi
    zero_y = -3 + (0 - lo) / (hi - lo) * 6
    neg, pos = st["c.bar_1"].params, st["c.bar_3"].params
    assert neg.pos[1] < zero_y < pos.pos[1]
    assert pos.height == pytest.approx(2 / (hi - lo) * 6)  # half of 4, grown halfway
    assert "c.bar_2" not in _ids(s.engine)  # a zero bar has no rectangle
    s2 = _scene()
    s2.add(dg.BarChart(id="c", values=(1, 2), grow=0.0))
    assert [i for i in _ids(s2.engine) if i.startswith("c.bar")] == []


def test_a_flat_chart_and_names_that_clash_still_build():
    from phasor import dg

    s = _scene()
    s.add(dg.BarChart(id="c", values=(1, 2), names=("y 1", "Y-1")))  # both "y_1"
    ids = _ids(s.engine)
    assert "c.y_1" in ids and "c.y_1_2" in ids and "c.y_1_2_name" in ids
    assert len(ids) == len(set(ids))
    flat = _scene()
    flat.add(dg.BarChart(id="c", values=(0, 0)))  # nothing to draw a bar for, still a range
    lo, hi = _sem(flat.engine, "c")["y_range"]
    assert lo < 0 < hi and not [i for i in _ids(flat.engine) if i.startswith("c.bar")]


@pytest.mark.parametrize("passes", MODES)
def test_crowded_bar_names_are_one_issue_with_a_step_that_clears_them(passes):
    from phasor import dg

    s = _scene(passes)
    names = tuple(f"category {k}" for k in range(8))
    s.add(dg.BarChart(id="c", values=tuple(range(1, 9)), names=names, width=4))
    issues = s.engine.lint()
    codes = [i.code for i in issues]
    assert "crowded_ticks" in codes and "overlap" not in codes
    (issue,) = [i for i in issues if i.code == "crowded_ticks"]
    assert issue.ids == ("c",) and issue.path == "c.names" and issue.times == (0.0,)
    assert issue.fix == () and "wider" in issue.fix_hint and "step" not in issue.fix_hint


# ---------------------------------------------------------------- 5. table


@pytest.mark.parametrize("passes", MODES)
def test_a_table_names_its_cells_like_a_spreadsheet(passes):
    from phasor import dg

    s = _scene(passes)
    s.add(dg.Table(id="t", cells=(("x", "$x^2$"), ("1", "1"), ("2", "4"), ("3",)),
                   lines="grid"))  # fmt: skip
    e = s.engine
    ids = _ids(e)
    assert ids == ["t", "t.a1", "t.b1", "t.a2", "t.b2", "t.a3", "t.b3", "t.a4"]  # b4 empty
    m = _sem(e, "t")
    assert (
        m["rows"] == 4
        and m["columns"] == 2
        and m["header"]
        == [
            "x",
            "$x^2$",
        ][:2]
    )
    st = e.state_at(0)
    assert st["t.a1"].roles["color"] == "secondary" and st["t.a2"].roles["color"] == "primary"
    a2, b2, a3 = (st[i].params.pos for i in ("t.a2", "t.b2", "t.a3"))
    assert a2[1] == pytest.approx(b2[1]) and a2[0] < b2[0] and a3[1] < a2[1]
    assert e.lint() == []


def test_a_table_reveals_its_rows_one_by_one():
    from phasor import dg

    s = _scene()
    s.add(dg.Table(id="t", cells=(("a", "b"), ("1", "2"), ("3", "4"), ("5", "6")), reveal=0.5))
    assert _ids(s.engine) == ["t", "t.a1", "t.b1", "t.a2", "t.b2"]


def test_a_table_reads_a_csv_file_from_the_scenes_folder(tmp_path):
    from phasor import dg

    (tmp_path / "data.csv").write_text("n,square\n1,1\n2,4\n", encoding="utf-8")
    s = _scene(base_dir=tmp_path)
    s.add(dg.Table(id="t", path="data.csv"))
    assert _ids(s.engine) == ["t", "t.a1", "t.b1", "t.a2", "t.b2", "t.a3", "t.b3"]
    assert _sem(s.engine, "t")["file"] == "data.csv"
    assert s.engine.lint() == []
    # changed on disk: read again (contract 3)
    (tmp_path / "data.csv").write_text("n\n1\n", encoding="utf-8")
    import os

    os.utime(tmp_path / "data.csv", ns=(1, 1))
    assert _ids(s.engine) == ["t", "t.a1", "t.a2"]


@pytest.mark.parametrize("passes", MODES)
def test_a_bad_long_or_missing_csv_is_lint_and_never_raises(tmp_path, passes):
    from phasor import dg

    (tmp_path / "long.csv").write_text("\n".join(f"{k},x" for k in range(100)), encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00 not text")
    (tmp_path / "huge.csv").write_bytes(b"1," * (300 * 1024))
    s = _scene(passes, base_dir=tmp_path)
    s.add(dg.Table(id="long", path="long.csv", pos=(0, 2, 0)))
    s.add(dg.Table(id="bad", path="bad.csv"))
    s.add(dg.Table(id="huge", path="huge.csv"))
    s.add(dg.Table(id="gone", path="gone.csv"))
    e = s.engine
    ids = _ids(e)
    assert sum(1 for i in ids if i.startswith("long.a")) == dm.MAX_ROWS
    assert not [i for i in ids if i.startswith(("bad.", "huge.", "gone."))]
    by = {}
    for i in e.lint():
        by.setdefault(i.code, set()).update(i.ids)
    assert by["truncated"] == {"long"} and by["bad_file"] == {"bad", "huge"}
    assert "gone" in by["missing_file"]
    with pytest.raises(ValueError):
        dg.Table.Params(path="../outside.csv")
    with pytest.raises(ValueError, match="not both"):
        dg.Table.Params(path="a.csv", cells=(("x",),))


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="no FIFOs here")
def test_a_fifo_named_as_a_table_or_code_is_never_read(tmp_path):
    import os

    from phasor import dg

    os.mkfifo(tmp_path / "pipe.csv")
    os.mkfifo(tmp_path / "pipe.py")
    s = _scene(base_dir=tmp_path)
    s.add(dg.Table(id="t", path="pipe.csv"))
    s.add(dg.Code(id="c", path="pipe.py"))
    assert _ids(s.engine) == ["t", "c"]  # nothing read, nothing built, no hang
    assert {"t", "c"} <= {x for i in s.engine.lint() if i.code == "missing_file" for x in i.ids}


# ---------------------------------------------------------------- 6. matrix


@pytest.mark.parametrize("passes", MODES)
def test_a_matrix_display_typesets_each_entry_as_a_part(passes):
    from phasor import dg

    s = _scene(passes)
    s.add(dg.MatrixDisplay(id="m", entries=(("a_{11}", "a_{12}", "0"), ("\\frac{1}{2}", "", "1")),
                           bracket="round"))  # fmt: skip
    e = s.engine
    assert _ids(e) == ["m", "m.e1_1", "m.e1_2", "m.e1_3", "m.e2_1", "m.e2_3"]
    objs = _objs(e)
    assert objs["m.e2_1"]["semantic"]["text"] == "$\\frac{1}{2}$"
    m = objs["m"]["semantic"]
    assert m["rows"] == 2 and m["columns"] == 3 and m["bracket"] == "round"
    st = e.state_at(0)
    box = st["m"].impl.world_bounds(st)
    inner = st["m.e1_1"].impl.world_bounds(st)
    assert box.lo[0] < inner.lo[0]  # the bracket is outside the entries
    assert e.lint() == []


def test_a_matrix_refuses_entries_that_cannot_be_typeset_or_are_too_many():
    from pydantic import ValidationError

    from phasor import dg

    for bad in ((("\\frac{1}",),), (), ((),), tuple(("1",) * 11 for _ in range(2)),
                tuple(("1",) for _ in range(11)), (("x" * 100,),)):  # fmt: skip
        with pytest.raises(ValidationError):
            dg.MatrixDisplay.Params(entries=bad)


# ---------------------------------------------------------------- 7. code

SNIPPET = 'def area(r):\n    # a disc\n\n    return 3.14159 * r ** 2  # pi r^2\n\nprint("done")'


@pytest.mark.parametrize("passes", MODES)
def test_code_lines_are_parts_highlighted_by_pygments(passes):
    pytest.importorskip("pygments")
    from phasor import dg

    s = _scene(passes)
    s.add(dg.Code(id="code", code=SNIPPET, line_numbers=True))
    e = s.engine
    assert _ids(e) == ["code"] + [f"code.line_{k}" for k in range(1, 7)]  # numbered blanks
    plain = _scene(passes)
    plain.add(dg.Code(id="code", code=SNIPPET))  # without numbers a blank line has no part
    assert _ids(plain.engine) == ["code", "code.line_1", "code.line_2", "code.line_4",
                                  "code.line_6"]  # fmt: skip
    st = e.state_at(0)
    one = st["code.line_1"].params
    assert one.text == "1  def area(r):"
    assert one.classes[:3] == "gpp" and one.classes[3:6] == "kkk" and one.classes[7:11] == "ffff"
    four = st["code.line_4"].params
    assert "nnnnnnn" in four.classes and four.classes.endswith("c")
    m = _sem(e, "code")
    assert m["highlight"] == "pygments" and m["lines"] == 6
    assert "missing_extra" not in m
    assert e.lint() == []


def test_code_is_monospace_by_placement():
    from phasor import dg

    s = _scene()
    s.add(dg.CodeLine(id="a", text="iiii", pos=(0, 0, 0)))
    s.add(dg.CodeLine(id="b", text="MMMM", pos=(0, -1, 0)))
    st = s.engine.state_at(0)
    wa = st["a"].impl.world_bounds(st)
    wb = st["b"].impl.world_bounds(st)
    # four narrow and four wide letters end within one cell of each other
    h = st["a"].style["size"] * st.camera.units_per_height(None)
    assert abs((wa.hi[0] - wa.lo[0]) - (wb.hi[0] - wb.lo[0])) < 0.6 * h


@pytest.mark.parametrize("passes", MODES)
def test_without_pygments_code_is_drawn_plain_and_lint_names_the_extra(monkeypatch, passes):
    import sys

    from phasor import dg

    monkeypatch.setitem(sys.modules, "pygments", None)
    s = _scene(passes)
    s.add(dg.Code(id="code", code=SNIPPET))
    e = s.engine
    st = e.state_at(0)
    assert set(st["code.line_1"].params.classes) == {"p"}
    assert _sem(e, "code")["highlight"] == "plain"
    (issue,) = [i for i in e.lint() if i.code == "missing_extra"]
    assert issue.ids == ("code",) and "--extra code" in issue.fix_hint
    assert "plain" in issue.message and issue.severity == "warning"


@pytest.mark.parametrize("passes", MODES)
def test_an_unknown_language_is_drawn_plain_with_a_lint_warning(passes):
    pytest.importorskip("pygments")
    from phasor import dg

    s = _scene(passes)
    s.add(dg.Code(id="code", code="x := 1", language="no-such-language"))
    assert _sem(s.engine, "code")["highlight"] == "unknown"
    assert [i.code for i in s.engine.lint()] == ["unknown_language"]


@pytest.mark.parametrize("passes", MODES)
def test_code_from_a_file_is_read_cut_and_rechecked(tmp_path, passes):
    from phasor import dg

    (tmp_path / "f.py").write_text("x = 1\n\ty = 2\x07\n", encoding="utf-8")
    (tmp_path / "long.py").write_text("pass\n" * 500, encoding="utf-8")
    s = _scene(passes, base_dir=tmp_path)
    s.add(dg.Code(id="a", path="f.py"))
    s.add(dg.Code(id="b", path="long.py", pos=(0, 5, 0)))
    e = s.engine
    st = e.state_at(0)
    assert st["a.line_2"].params.text == "    y = 2?"  # tab as spaces, a bell as "?"
    assert sum(1 for i in _ids(e) if i.startswith("b.")) == dm.MAX_LINES
    assert {i.code for i in e.lint() if "b" in i.ids} >= {"truncated"}


def test_code_refuses_what_it_could_not_draw():
    from pydantic import ValidationError

    from phasor import dg

    for bad in ({"code": "x\x00"}, {"code": "x" * (dm.MAX_LINE + 1)},
                {"code": "\n" * dm.MAX_LINES}, {"code": "y" * (dm.MAX_CODE + 1)},
                {"language": "Python 3!"}, {"language": "x" * 40},
                {"code": "a", "path": "a.py"}):  # fmt: skip
        with pytest.raises(ValidationError):
            dg.Code.Params(**bad)
    dg.Code.Params(code="\n".join(["x" * dm.MAX_LINE] * dm.MAX_LINES)[: dm.MAX_CODE])


# ---------------------------------------------------------------- 8. counter


def test_a_counter_groups_signs_and_stays_short_whatever_it_is_fed():
    from phasor import Counter

    P = Counter.Params
    assert Counter.text_of(P(value=1234567.891, digits=1, group=True)) == "$1{,}234{,}567.9$"
    assert Counter.text_of(P(value=2, digits=0, sign=True, math=False)) == "+2"
    assert Counter.text_of(P(value=-0.001, digits=2)) == "$0.00$"  # as before: no -0
    assert Counter.text_of(P(value=3.14159, digits=3, prefix="\\pi = ")) == "$\\pi = 3.142$"
    assert len(Counter.text_of(P(value=1e300, digits=10))) < 30
    # No length limit on prefix and suffix, as for any Text: documents saved before
    # Milestone 19 hold longer ones (audit B1)
    assert Counter.text_of(P(value=1, digits=0, prefix="x" * 100, math=False)) == "x" * 100 + "1"


def test_a_counter_animates_with_the_value_it_follows():
    from phasor import Counter, Value, expr

    s = _scene()
    s.add(Value(id="n", value=0.0))
    s.add(Counter(id="shown", value=expr("n.value"), digits=0, group=True))
    s.marker("go", 0.5)
    s.marker("done", 1.5)
    s.at("go").morph("n.value", 12000.0, until="done")
    e = s.engine
    texts = [_objs(e, t)["shown"]["semantic"]["text"] for t in (0.0, 1.0, 1.5)]
    assert texts[0] == "$0$" and texts[-1] == "$12{,}000$" and texts[1] not in texts[::2]


# ---------------------------------------------------------------- 9. lint


@pytest.mark.parametrize("passes", MODES)
def test_crowded_axis_numbers_are_one_issue_per_axis_with_no_pairwise_noise(passes):
    from phasor import geo

    s = _scene(passes)
    s.add(geo.Axes(id="ax", dims=2, x_range=(0, 1000, 25), x_scale=0.006, numbers=True))
    issues = s.engine.lint()
    assert [(i.code, i.path) for i in issues] == [("crowded_ticks", "ax.x")]
    (issue,) = issues
    assert issue.ids == ("ax",) and issue.severity == "warning" and issue.times == (0.0,)
    assert issue.fix == ()  # the step is the author's to choose; the hint names one
    assert "step of at least 200" in issue.fix_hint or "step of at least 100" in issue.fix_hint
    s.engine.apply({"op": "set_param", "path": "ax.x_range", "value": [0, 1000, 200]})
    assert s.engine.lint() == []


def test_the_tick_gap_comes_from_the_theme():
    from phasor.style.theme import load_theme

    assert load_theme("midnight_glow").lint["tick_gap"] == 0.25


@pytest.mark.parametrize("passes", MODES)
def test_an_svg_without_its_extra_is_still_one_missing_extra_that_draws_nothing(
    monkeypatch, passes
):
    """The plain-fallback rule (Code without pygments) must not fire for an SVG, whose
    semantics also name a missing extra: one issue, the "draws nothing" one."""
    import sys
    from pathlib import Path

    from phasor import shapes

    monkeypatch.setitem(sys.modules, "svgelements", None)
    s = _scene(passes, base_dir=Path(__file__).parents[1] / "examples")
    s.add(shapes.SVG(id="emblem", path="assets/emblem.svg"))
    found = [i for i in s.engine.lint() if i.code == "missing_extra"]
    assert len(found) == 1 and "draws nothing" in found[0].message


def test_part_names_the_document_holds_get_a_suffix_never_a_collision():
    from phasor import Text, dg

    s = _scene()
    s.add(dg.Table(id="t", cells=(("x", "y"),)))
    s.add(dg.NumberPlane(id="plane", x_range=(-1, 1, 1), y_range=(-1, 1, 1)))
    s.add(dg.Code(id="code", code="x = 1"))
    s.add(Text(id="t.a1", text="mine", pos=(0, 3, 0)))
    s.add(Text(id="plane.axes", text="mine too", pos=(0, -3, 0)))
    s.add(Text(id="code.line_1", text="also mine", pos=(3, 3, 0)))
    ids = _ids(s.engine)
    assert "t.a1_2" in ids and "t.b1" in ids
    assert "plane.axes_2" in ids and "plane.axes_2.x_1" in ids
    assert "code.line_1_2" in ids
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- 10. bounds and hostile input

BIGGEST = {
    "axes": {"dims": 2, "x_range": (0, ticks.MAX_TICKS, 1), "y_range": (0, ticks.MAX_TICKS, 1),
             "scale": 0.05, "numbers": True},
    "number_line": {"x_range": (0, ticks.MAX_TICKS, 1), "scale": 0.1},
    "number_plane": {"x_range": (0, ticks.MAX_TICKS, 1), "y_range": (0, ticks.MAX_TICKS, 1),
                     "scale": 0.05, "faint": 4},
    "bar_chart": {"values": tuple(range(dm.MAX_BARS)),
                  "names": tuple(f"n{k}" for k in range(dm.MAX_BARS)), "show_values": True},
    "table": {"cells": tuple(tuple("x" * dm.MAX_CELL for _ in range(10)) for _ in range(30))},
    "matrix_display": {"entries": tuple(("1",) * dm.MAX_MATRIX for _ in range(dm.MAX_MATRIX))},
    "code": {"code": "\n".join(["x = 1 + 2"] * dm.MAX_LINES), "line_numbers": True},
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(BIGGEST))
def test_every_diagram_at_its_bounds_builds_inspects_and_lints_in_bounded_time(name):
    import time

    e = _scene().engine
    r = e.apply({"op": "add_object", "object": {"id": "big", "type": name,
                                                "params": BIGGEST[name]}}, origin="script")  # fmt: skip
    assert r.ok, [i.message for i in r.errors]
    t0 = time.perf_counter()
    st = e.state_at(0.0)
    e.inspect(0.0)
    e.lint()
    assert time.perf_counter() - t0 < 60, name
    assert len(st.objects) <= 1 + dm.MAX_CELLS + 2 * ticks.MAX_TICKS + 3 * dm.MAX_BARS + 2


HOSTILE = [
    ("axes", {"x_range": (0, 1e300, 0), "scale": 1e300}),  # world coordinates overflow
    ("axes", {"x_range": (0, 10**400, 1)}),
    ("number_line", {"x_range": (0, 1, 1e-9)}),
    ("number_line", {"scale": float("inf")}),
    ("number_line", {"pos": (1e300, 0, 0)}),
    ("number_plane", {"x_range": (-1e9, 1e9, 1)}),
    ("number_plane", {"y_scale": 1e300}),
    ("bar_chart", {"values": ()}),
    ("bar_chart", {"values": (1.0,) * (dm.MAX_BARS + 1)}),
    ("bar_chart", {"values": (1e300,)}),
    ("bar_chart", {"values": (10**400,)}),
    ("bar_chart", {"values": (1, 2), "names": ("a",)}),
    ("bar_chart", {"values": (1,), "colors": ("purple",)}),
    ("bar_chart", {"y_range": (5, 5, 1)}),
    ("table", {"cells": tuple(("x",) for _ in range(dm.MAX_ROWS + 1))}),
    ("table", {"cells": (("x",) * (dm.MAX_COLS + 1),)}),
    ("table", {"cells": (("$\\frac{$",),)}),
    ("table", {"cells": (("a\nb",),)}),
    ("matrix_display", {"entries": (("\\left(",),)}),
    ("code", {"code": "a\x1b[31m"}),
    ("code_line", {"text": "x", "classes": "zz"}),
    ("counter", {"value": 10**400}),
]


@pytest.mark.parametrize("name, params", HOSTILE)
def test_hostile_parameters_are_refused_as_errors_never_raised(name, params):
    e = _scene().engine
    r = e.apply({"op": "add_object", "object": {"id": "bad", "type": name, "params": params}},
                origin="script")  # fmt: skip
    assert not r.ok and r.errors
    assert _ids(e) == []
    assert e.lint() == []


def test_values_fed_by_an_expression_never_break_a_frame():
    """A counter or a chart fed a huge or non-finite number through an expression: the frame
    still evaluates (the document stays valid at every t) and lint never raises."""
    from phasor import Counter, Value, expr

    s = _scene()
    s.add(Value(id="v", value=1e300))
    s.add(Counter(id="c", value=expr("v.value * 10")))
    e = s.engine
    e.state_at(0.0)
    e.lint()  # whatever it reports, it reports


# ---------------------------------------------------------------- 11. the examples

NEW_TYPES = {"number_line", "number_plane", "bar_chart", "table", "matrix_display", "code",
             "code_line", "axes", "counter"}  # fmt: skip
_examples: dict[str, object] = {}


def _example(name: str):
    import runpy
    from pathlib import Path

    if name not in _examples:
        path = Path(__file__).parents[1] / "examples" / f"{name}.py"
        _examples[name] = runpy.run_path(str(path), run_name="x")["s"].engine
    return _examples[name]


def test_the_examples_show_every_diagram_type_and_lint_clean():
    pytest.importorskip("pygments")  # without it the gallery's code is (rightly) missing_extra
    seen = set()
    for name in ("diagram_kit", "data_viz"):
        e = _example(name)
        seen |= {o.type for o in e.state_at(e.duration).objects}
        assert e.lint(strict=True) == [], name
    assert seen >= NEW_TYPES
    e = _example("diagram_kit")
    sem = {o["id"]: o["semantic"] for o in e.inspect(e.duration)["objects"]}
    assert sem["axes"]["axes"]["x"]["scale"] != sem["axes"]["axes"]["y"]["scale"]
    assert sem["chart"]["largest"] == "bees" and sem["table.b2"]["text"] == "1"
    assert sem["matrix"]["rows"] == 2 and sem["plane"]["backdrop"]


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.parametrize("name, times", [("diagram_kit", [0.8, 2.2, 4.0, 6.0]),
                                         ("data_viz", [0.8, 2.0, 3.5, 6.0])])  # fmt: skip
def test_example_golden_storyboards(name, times):
    from tests.conftest import gl_available
    from tests.test_beauty import _golden

    if not gl_available():
        pytest.skip("needs OpenGL")
    pytest.importorskip("pygments")  # the gallery's code is highlighted in its golden
    e = _example(name)
    sheet = e.storyboard(times, size=(320, 180))["image"]
    _golden(name, sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- 12. independent audit (2026-11-23)

# B1: objects a document saved at bc99a79 (before Milestone 19) could hold; each must still
# load, draw a bounded picture, and lint must say when less is drawn than asked.
_OLD_DOC = {
    "cameras": [{"id": "camera", "kind": "ortho", "params": {}}], "constraints": [],
    "controls": [], "narration": None, "retired_ids": [], "schema_version": 9,
    "meta": {"background": None, "color_space": "srgb", "duration": None, "fps": 60.0,
             "height": 1080, "life": "off", "look": None, "modules": [], "palette": None,
             "seed": 0, "source": "script", "theme": "midnight_glow", "title": "old",
             "width": 1920},
    "timeline": {"camera": "camera", "markers": {}, "tracks": {}},
}  # fmt: skip
# name: (type, params, whether lint reports ``clamped``)
OLD_OBJECTS = {
    "counter_prefix": ("counter", {"value": 1.0, "prefix": "\\mathrm{total\\ visitors\\ so"
                                   "\\ far\\ today} = "}, False),
    "many_ticks": ("axes", {"scale": 0.4, "x_range": [-10, 10, 0.1], "y_range": [-1, 1, 0.1]},
                   True),
    "million_ticks": ("axes", {"dims": 2, "x_range": [0, 1e6, 1], "scale": 1e-5}, True),
    "tiny_step": ("axes", {"dims": 2, "x_range": [0, 1, 1e-12]}, True),
    "four_labels": ("axes", {"labels": ["x", "y", "z", "w"]}, True),
    "three_labels_2d": ("axes", {"dims": 2, "labels": ["x", "y", "z"]}, True),
    "long_label": ("axes", {"dims": 2, "labels": ["x" * 100]}, False),
    "bad_math_label": ("axes", {"dims": 2, "labels": ["$\\frac{$", "y"]}, True),
    "huge_range": ("axes", {"dims": 2, "x_range": [0, 2e10, 1e9], "scale": 5e-10}, False),
    "lost_span": ("axes", {"dims": 2, "x_range": [1e8, 1e8 + 1e-3, 1e-4], "scale": 1e3}, True),
    "huge_scale": ("axes", {"dims": 2, "scale": 1e300}, False),
    "far_pos": ("axes", {"dims": 2, "pos": [1e300, 0, 0]}, False),
    "long_tick": ("axes", {"dims": 2, "tick": 1e10}, False),
}  # fmt: skip


def _old_document(tmp_path, kind: str, params: dict):
    import json

    from phasor.scene import load

    obj = {"always": [], "id": "old", "layer": "world", "params": params, "style": {},
           "tags": [], "type": kind}  # fmt: skip
    path = tmp_path / "old.phasor.json"
    path.write_text(json.dumps({**_OLD_DOC, "objects": [obj]}, sort_keys=True), encoding="utf-8")
    return load(path, passes=None)


@pytest.mark.parametrize("name", sorted(OLD_OBJECTS))
def test_documents_saved_before_milestone_19_still_load_draw_bounded_and_lint_why(tmp_path, name):
    import numpy as np

    from phasor.core.record import Recorder

    kind, params, clamped = OLD_OBJECTS[name]
    e = _old_document(tmp_path, kind, params)
    st = e.state_at(0.0)
    sem = _sem(e, "old")
    for axis in (sem.get("axes") or {}).values():
        assert axis["ticks"] <= ticks.MAX_TICKS + 1
    b = st["old"].impl.world_bounds(st)
    assert np.isfinite(b.lo).all() and np.isfinite(b.hi).all()
    rec = Recorder(st.camera.right, st.camera.up)
    st["old"].impl.draw(rec, st, "color")
    assert 0 < len(rec.kinds) <= 4 * ticks.MAX_TICKS + 20
    issues = e.lint()
    found = [i for i in issues if i.code == "clamped"]
    assert bool(found) == clamped, [i.message for i in issues]
    if found:
        (issue,) = found
        assert issue.ids == ("old",) and issue.times == (0.0,) and issue.severity == "warning"


def test_a_range_past_the_tick_limit_is_drawn_at_a_wider_nice_step():
    assert ticks.step_of(-10, 10, 0.1) == 0.2  # 200 ticks asked: drawn at 0.2 (101)
    assert ticks.step_of(0, 200, 1) == 2.0
    assert ticks.step_of(0, ticks.MAX_TICKS, 1) == 1.0  # at the limit: as asked
    assert len(ticks.values(0, 1e6, 1)) <= ticks.MAX_TICKS + 1
    assert ticks.values(1e8, 1e8 + 1e-3, 1e-4) == []  # lost in rounding: no ticks
    assert "100" in ticks.clamp_note(-10, 10, 0.1, "x_range")
    assert ticks.clamp_note(-10, 10, 1, "x_range") is None


def test_a_write_past_the_tick_limit_is_taken_and_its_lint_delta_says_so():
    from phasor import geo

    s = _scene()
    s.add(geo.Axes(id="ax", dims=2))
    r = s.engine.apply({"op": "set_param", "path": "ax.x_range", "value": [0, 1000, 0.1]})
    assert r.ok
    assert [i.code for i in r.lint_added] == ["clamped"]
    assert "step of 10" in r.lint_added[0].message


def _grown_chart(names, show_values=True, reserved=()):
    """A chart whose bars grow in from 0 (no rectangles at t = 0) to full at t = 1."""
    from phasor import Text, dg

    s = _scene()
    s.add(dg.BarChart(id="c", values=tuple(range(1, len(names) + 1)), names=names, grow=0.0,
                      show_values=show_values))  # fmt: skip
    for k, rid in enumerate(reserved):
        s.add(Text(id=f"c.{rid}", text="mine", pos=(0, 4 + k, 0)))
    s.at(0.0).to("c.grow", 1.0, duration=1.0)
    s.marker("end", 2.0)
    return s.engine


def test_bar_names_never_collide_with_the_parts_derived_from_them():
    """B2: bars 'a' and 'a name' gave the label of 'a' and the bar 'a name' one id, and the
    chart raised once the bars grew (state_at, inspect, lint)."""
    e = _grown_chart(("a", "a name", "a value"))
    ids = {t: _ids(e, t) for t in (0.0, 1.0, 2.0)}
    for t, found in ids.items():
        assert len(found) == len(set(found)), t
    assert {"c.a", "c.a_name", "c.a_value", "c.a_name_2", "c.a_value_2"} <= set(ids[2.0])
    assert "c.a_name_2" in ids[0.0]  # the label's name is the same before the bars grow
    e.inspect(1.0, detail="full")
    assert "bad_part" not in {i.code for i in e.lint()}


def test_bar_part_names_fuzzed_over_shared_prefixes_and_suffixes():
    import random

    tokens = ["a", "a name", "a value", "a_name", "a_name_name", "A", "y 1", "y", "bar 1",
              "bar_1_name", "a_value_2", "a_name_2", "x"]  # fmt: skip
    rng = random.Random(19)
    for trial in range(24):
        names = tuple(rng.sample(tokens, rng.randint(1, 6)))
        reserved = tuple(rng.sample(["a_name", "y_1", "bar_1", "a_value"], rng.randint(0, 2)))
        e = _grown_chart(names, show_values=trial % 2 == 0, reserved=reserved)
        before: set[str] = set()
        for t in (0.0, 0.5, 1.0):
            found = _ids(e, t)
            assert len(found) == len(set(found)), (names, reserved, t)
            assert before <= set(found), (names, t)  # growing only adds parts: none renamed
            before = set(found)
        assert "bad_part" not in {i.code for i in e.lint()}, (names, reserved)


def test_an_empty_bar_name_has_no_label_and_never_fails():
    """S2: a name "" passed the validator but its label Text(text="") failed as bad_part."""
    from phasor import dg

    s = _scene()
    s.add(dg.BarChart(id="c", values=(1, 2), names=("", "b")))
    ids = _ids(s.engine)
    assert "c.b_name" in ids and "c.bar_name" not in ids and "c.bar" in ids
    assert "bad_part" not in {i.code for i in s.engine.lint()}
