"""Milestone 16: vector paths, fills, dashes, shapes, booleans and SVG.

Section 1 checks the geometry (``domains/path_math.py``) against known maths; later
sections the objects, lint, the example and its golden."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor.domains import path_math as pm

# ---------------------------------------------------------------- 1. geometry


def _area(pos, tris) -> float:
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)


def test_a_circle_flattens_close_to_the_circle_with_more_points_when_bigger_on_screen():
    sub = pm.circle((1, 2, 0), 3.0)
    assert pm.is_closed(sub) and len(sub) == 13
    for tol in (1e-2, 1e-3, 1e-4):
        poly = pm.flatten(sub, tol)
        r = np.linalg.norm(poly[:, :2] - [1, 2], axis=1)
        assert np.abs(r - 3).max() < tol + 3 * 2.8e-4  # the cubic circle is within 2.7e-4 r
    coarse, fine = pm.flatten(sub, 1e-2), pm.flatten(sub, 1e-4)
    assert len(fine) > 5 * len(coarse)
    # A straight cubic stays one segment, however fine the tolerance.
    assert len(pm.flatten(pm.polyline([(0, 0), (5, 0)]), 1e-9)) == 2


def test_lengths_and_areas_match_the_formulas():
    sub = pm.circle((0, 0, 0), 2.0)
    poly = pm.flatten(sub, 1e-5)
    assert pm.length(poly) == pytest.approx(4 * math.pi, rel=1e-3)
    pos, tris = pm.fill([poly])
    assert _area(pos, tris) == pytest.approx(4 * math.pi, rel=1e-3)
    rect = pm.flatten(pm.rectangle((0, 0, 0), 4, 2, radius=0.5), 1e-5)
    pos, tris = pm.fill([rect])
    assert _area(pos, tris) == pytest.approx(8 - (4 - math.pi) * 0.25, rel=1e-3)
    sq = pm.rectangle((0, 0, 0), 2, 2)
    assert pm.length(pm.flatten(sq, 1e-3)) == pytest.approx(8)
    pie = pm.flatten(pm.sector((0, 0, 0), 1.0, 0.0, math.pi / 2), 1e-5)
    assert pm.region([pie]).area == pytest.approx(math.pi / 4, rel=1e-3)


def test_holes_follow_the_rule():
    outer = pm.flatten(pm.circle((0, 0, 0), 2.0), 1e-4)
    inner_same = pm.flatten(pm.circle((0, 0, 0), 1.0), 1e-4)  # same direction
    inner_back = inner_same[::-1]
    full, ring = math.pi * 4, math.pi * 3
    # even-odd: a nested ring is a hole whichever way it runs
    assert pm.region([outer, inner_same], "evenodd").area == pytest.approx(ring, rel=1e-3)
    assert pm.region([outer, inner_back], "evenodd").area == pytest.approx(ring, rel=1e-3)
    # non-zero: a hole only when it runs the other way
    assert pm.region([outer, inner_same], "nonzero").area == pytest.approx(full, rel=1e-3)
    assert pm.region([outer, inner_back], "nonzero").area == pytest.approx(ring, rel=1e-3)
    # an island inside the hole (even-odd: filled again)
    island = pm.flatten(pm.circle((0, 0, 0), 0.5), 1e-4)
    a = pm.region([outer, inner_same, island], "evenodd").area
    assert a == pytest.approx(ring + math.pi * 0.25, rel=1e-3)
    pos, tris = pm.fill([outer, inner_same, island], "evenodd")
    assert _area(pos, tris) == pytest.approx(a, rel=1e-6)


def test_a_fill_in_a_tilted_plane_keeps_its_3d_points():
    sub = pm.circle((0, 0, 0), 1.0)
    t = math.radians(30)
    rot = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    poly = pm.flatten(sub, 1e-4) @ rot.T + [0, 0, 2]
    pos, tris = pm.fill([poly])
    assert _area(pos, tris) == pytest.approx(math.pi, rel=1e-3)
    normal = rot @ [0, 0, 1]
    assert np.abs((pos - [0, 0, 2]) @ normal).max() < 1e-9


def test_dashes_are_cut_by_arc_length_and_do_not_crawl():
    line = pm.flatten(pm.polyline([(0, 0), (10, 0)]), 1e-3)
    d = pm.dashes(line, 1.0, on=0.5)
    assert len(d) == 10 and d[3][0, 0] == pytest.approx(3.0) and d[3][-1, 0] == pytest.approx(3.5)
    # traced halfway: the same dashes over the first half, none past it
    half = pm.dashes(line, 1.0, on=0.5, upto=5.0)
    assert len(half) == 5
    for a, b in zip(half, d, strict=False):
        assert np.allclose(a, b)
    # a longer line: the dashes it shares with the shorter one stay where they were
    longer = pm.dashes(pm.flatten(pm.polyline([(0, 0), (14, 0)]), 1e-3), 1.0, on=0.5)
    for a, b in zip(d, longer, strict=False):
        assert np.allclose(a, b)
    # a curve: the same arc lengths
    arc = pm.flatten(pm.arc((0, 0, 0), 2.0, 0.0, math.pi), 1e-6)
    pieces = pm.dashes(arc, 0.5, on=0.5)
    assert all(pm.length(p) == pytest.approx(0.25, rel=1e-3) for p in pieces[:-1])
    dots = pm.dots(line, 2.0)
    assert np.allclose(dots[:, 0], [0, 2, 4, 6, 8, 10])
    loop = pm.dots(pm.flatten(pm.circle((0, 0, 0), 1.0), 1e-6), 2 * math.pi / 8)
    assert len(loop) == 8  # no dot on top of the first


def test_trim_takes_the_first_share_subpath_after_subpath():
    a = pm.flatten(pm.polyline([(0, 0), (4, 0)]), 1e-3)
    b = pm.flatten(pm.polyline([(0, 1), (4, 1)]), 1e-3)
    assert pm.trim([a, b], 1.0) == [a, b]
    (only,) = pm.trim([a, b], 0.25)
    assert only[-1, 0] == pytest.approx(2.0)
    first, second = pm.trim([a, b], 0.75)
    assert np.allclose(first, a) and second[-1, 0] == pytest.approx(2.0)
    assert pm.trim([a, b], 0.0) == []


def test_shapes_have_the_points_they_promise():
    v = pm.regular_vertices(6, 2.0)
    assert np.allclose(v[0], [0, 2, 0]) and np.allclose(np.linalg.norm(v[:, :2], axis=1), 2)
    br = pm.brace((-1, 0), (1, 0), 0.4)  # under a term, left to right: the tip points down
    poly = pm.flatten(br, 1e-4)
    assert np.allclose(poly[0], [-1, 0, 0]) and np.allclose(poly[-1], [1, 0, 0])
    assert poly[:, 1].min() == pytest.approx(-0.4, abs=1e-6) and poly[:, 1].max() <= 1e-9
    ring = pm.sector((0, 0, 0), 2.0, 0.0, math.pi, inner=1.0)
    assert pm.region([pm.flatten(ring, 1e-5)]).area == pytest.approx(1.5 * math.pi, rel=1e-3)
    with pytest.raises(ValueError, match="1 \\+ 3k"):
        pm.check_subpath([(0, 0, 0), (1, 0, 0)])


def test_crossings_find_a_ring_that_crosses_itself_or_another():
    square = pm.flatten(pm.polyline([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True), 1e-3)
    bow = pm.flatten(pm.polyline([(0, 0), (1, 1), (1, 0), (0, 1)], closed=True), 1e-3)
    assert not pm.crossings([square])
    assert pm.crossings([bow])
    other = square + np.array([0.5, 0.5, 0])
    assert pm.crossings([square, other])
    inner = (square - 0.5) * 0.5 + 0.5
    assert not pm.crossings([square, inner])
    assert not pm.crossings([pm.flatten(pm.circle((0, 0, 0), 1), 1e-4)])


def _two_discs():
    a = [pm.flatten(pm.circle((-0.5, 0, 0), 1.0), 1e-4)]
    b = [pm.flatten(pm.circle((0.5, 0, 0), 1.0), 1e-4)]
    return a, b


def test_boolean_operations_have_the_areas_set_algebra_says():
    pytest.importorskip("shapely")
    a, b = _two_discs()
    lens = 2 * math.acos(0.5) - 2 * 0.5 * math.sqrt(1 - 0.25)  # overlap of two unit discs, d=1
    areas = {op: pm.region(pm.boolean(op, a, b), "nonzero").area for op in pm.OPS}
    assert areas["intersection"] == pytest.approx(lens, rel=1e-3)
    assert areas["union"] == pytest.approx(2 * math.pi - lens, rel=1e-3)
    assert areas["difference"] == pytest.approx(math.pi - lens, rel=1e-3)
    assert areas["exclusion"] == pytest.approx(2 * math.pi - 2 * lens, rel=1e-3)
    # Holes come back running the other way, so they are holes under either rule.
    ring = pm.boolean("difference", [pm.flatten(pm.circle((0, 0, 0), 2.0), 1e-4)],
                      [pm.flatten(pm.circle((0, 0, 0), 1.0), 1e-4)])  # fmt: skip
    for rule in pm.RULES:
        assert pm.region(ring, rule).area == pytest.approx(3 * math.pi, rel=1e-3)
    # the same inputs, the same output (determinism)
    assert all(
        np.array_equal(x, y)
        for x, y in zip(pm.boolean("union", a, b), pm.boolean("union", a, b), strict=True)
    )


def test_a_missing_extra_is_a_clear_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "shapely", None)
    a, b = _two_discs()
    with pytest.raises(pm.MissingExtra, match="--extra boolean"):
        pm.boolean("union", a, b)


# ---------------------------------------------------------------- 2. objects

from phasor import Scene, shapes  # noqa: E402
from phasor.core.issues import DocumentError  # noqa: E402
from phasor.style.theme import load_theme  # noqa: E402
from tests.conftest import gl_available  # noqa: E402

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


class _Calls:
    """A DrawContext that keeps every triangles call."""

    def __init__(self):
        self.calls = []

    def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None):
        self.calls.append({"n": len(positions), "color": tuple(color), "glow": glow,
                           "positions": np.asarray(positions)})  # fmt: skip


def _draw(e, oid, t=0.0):
    st = e.state_at(t)
    ctx = _Calls()
    st[oid].impl.draw(ctx, st, "color")
    return ctx.calls


def _sem(e, oid, t=0.0):
    """An object's semantics; a composite's part too (``inspect`` takes document ids)."""
    (seen,) = [o for o in e.inspect(t)["objects"] if o["id"] == oid]
    return seen["semantic"]


def test_every_shape_says_what_it_is():
    s = Scene("shapes", passes=None)
    s.add(shapes.Circle(id="disc", pos=(1, 0), r=2))
    s.add(shapes.Ellipse(id="oval", width=4, height=2))
    s.add(shapes.Arc(id="swing", r=2, start=0, angle=90))
    s.add(shapes.Sector(id="pie", r=2, angle=90))
    s.add(shapes.Annulus(id="ring", outer=2, inner=1))
    s.add(shapes.Polygon(id="tri", vertices=[(0, 0), (2, 0), (0, 2)]))
    s.add(shapes.RegularPolygon(id="hexagon", n=6, r=1))
    s.add(shapes.Rectangle(id="card", width=4, height=2, corner_radius=0.5))
    s.add(shapes.Brace(id="under", start=(-1, -2), end=(1, -2), depth=0.4))
    e = s.engine
    want = {"disc": ("circle", math.pi * 4), "oval": ("ellipse", math.pi * 2),
            "pie": ("sector", math.pi), "ring": ("annulus", 3 * math.pi), "tri": ("polygon", 2.0),
            "hexagon": ("polygon", 3 * math.sqrt(3) / 2),
            "card": ("rectangle", 8 - (4 - math.pi) * 0.25)}  # fmt: skip
    for oid, (kind, area) in want.items():
        sem = _sem(e, oid)
        assert sem["kind"] == kind and sem["closed"] is True, oid
        assert sem["area"] == pytest.approx(area, rel=1e-3), oid
    arc = _sem(e, "swing")
    assert arc["kind"] == "arc" and arc["closed"] is False and "area" not in arc
    assert arc["length"] == pytest.approx(math.pi, rel=1e-4)
    assert _sem(e, "under")["tip"] == pytest.approx([0.0, -2.4, 0.0])
    st = e.state_at(0)
    assert np.allclose(st["hexagon"].impl.features(st)["vertices"][0], [0, 1, 0])
    assert np.allclose(st["disc"].impl.features(st)["center"], [1, 0, 0])
    assert np.allclose(st["under"].impl.features(st)["tip"], [0, -2.4, 0])
    b = st["disc"].impl.world_bounds(st)
    assert b.lo[0] == pytest.approx(-1, abs=0.05) and b.hi[0] == pytest.approx(3, abs=0.05)


def test_a_path_checks_its_subpaths_and_can_be_built_through_points():
    s = Scene("paths", passes=None)
    with pytest.raises(DocumentError, match="1 \\+ 3k"):
        s.add(shapes.Path(id="bad", subpaths=[[(0, 0), (1, 0)]]))
    s.add(shapes.Path.through("zig", [(0, 0), (1, 1), (2, 0)]))
    sem = _sem(s.engine, "zig")
    assert sem["kind"] == "path" and sem["closed"] is False
    assert sem["length"] == pytest.approx(2 * math.sqrt(2), abs=1e-4)
    s.add(shapes.Path.through("box", [(0, 0), (1, 0), (1, 1), (0, 1)], closed=True,
                              style={"fill": "solid"}))  # fmt: skip
    assert _sem(s.engine, "box")["area"] == pytest.approx(1.0)


def test_the_fill_is_plain_paint_at_the_roles_opacity_and_fades_with_progress():
    s = Scene("fill", passes=None)
    s.add(shapes.Circle(id="disc", r=1, opacity=0.5,
                        style={"fill": "soft", "glow": "strong", "fill_color": "secondary"}))  # fmt: skip
    e = s.engine
    me = e.state_at(0)["disc"]
    fill, stroke = _draw(e, "disc")[0], _draw(e, "disc")[1]
    theme = load_theme(e.document.meta.theme)
    soft = theme.values["fill"]["soft"]
    assert fill["glow"] == 1.0 and stroke["glow"] > 1.0  # the fill never glows
    assert fill["color"][3] == pytest.approx(me.style["fill_color"][3] * soft * 0.5)
    assert fill["color"][:3] == pytest.approx(tuple(theme.values["emphasis"]["secondary"][:3]))
    s.at(0).to("disc.progress", 0.0, duration=0.001)
    s.at(1).to("disc.progress", 0.5, duration=0.001)
    calls = _draw(s.engine, "disc", 2.0)
    assert calls[0]["color"][3] == pytest.approx(fill["color"][3] * 0.5)
    assert _sem(s.engine, "disc", 2.0)["drawn"] == 0.5
    assert _draw(s.engine, "disc", 0.5) == []  # nothing traced, nothing filled
    s.add(shapes.Circle(id="outline", pos=(3, 0)))
    assert len(_draw(s.engine, "outline")) == 1  # no fill: only the stroke
    s.add(shapes.Circle(id="blob", pos=(-3, 0), stroke_opacity=0, style={"fill": "solid"}))
    assert len(_draw(s.engine, "blob")) == 1  # no stroke: only the fill


def test_dashes_do_not_crawl_as_a_path_is_traced():
    s = Scene("dash", passes=None)
    s.add(shapes.Path.through("road", [(-4, 0), (4, 0)], style={"dash": "dashed"}))
    s.at(1).to("road.progress", 0.5, duration=0.001)
    full, half = _draw(s.engine, "road", 0.0), _draw(s.engine, "road", 2.0)
    assert len(full) > 4 and len(half) == pytest.approx(len(full) / 2, abs=1)
    for a, b in zip(half[:-1], full, strict=False):  # every whole dash is where it was
        assert np.allclose(a["positions"], b["positions"])
    s.add(shapes.Circle(id="dotted", r=1, pos=(0, 2), style={"dash": "dotted"}))
    (dots,) = _draw(s.engine, "dotted")
    st = s.engine.state_at(0)
    period = (
        load_theme(s.engine.document.meta.theme).values["dash"]["dotted"]
        * st.camera.units_per_height()
    )
    assert dots["n"] == pytest.approx(13 * (2 * math.pi) / period, rel=0.1)  # 13 vertices a dot


def test_arrow_tips_sit_on_the_open_ends():
    s = Scene("tips", passes=None)
    s.add(shapes.Arc(id="swing", r=2, angle=90, tips="both"))
    calls = _draw(s.engine, "swing")
    heads = [c for c in calls if c["n"] == 3]
    assert len(heads) == 2
    assert np.allclose(heads[0]["positions"][0], [2, 0, 0])
    assert np.allclose(heads[1]["positions"][0], [0, 2, 0], atol=1e-9)
    assert _sem(s.engine, "swing")["tips"] == "both"


def test_arrow_keeps_drawing_exactly_as_before():
    """``strokes.arrow`` now draws its head through ``arrow_head`` (shared with path tips):
    the same arithmetic as before, so the same bytes."""
    from phasor.domains import strokes

    cam = Scene("cam", passes=None).engine.state_at(0).camera
    ctx = _Calls()
    strokes.arrow(ctx, cam, (0, 0, 0), (2, 1, 0), 0.03, (1, 1, 1, 1))
    _, head = ctx.calls
    d = np.array([2.0, 1.0, 0.0])
    unit = d / np.linalg.norm(d)
    h = min(np.linalg.norm(d) * 0.3, 0.03 * 7)
    side = np.cross(cam.forward, unit)
    side /= max(np.linalg.norm(side), 1e-9)
    base = np.array([2.0, 1.0, 0.0]) - unit * h
    want = np.array([[2, 1, 0], base + side * h * 0.45, base - side * h * 0.45])
    assert np.array_equal(head["positions"], want)


def test_boolean_operations_make_plain_paths():
    pytest.importorskip("shapely")
    s = Scene("bool", passes=None)
    a = shapes.Circle(id="a", pos=(-0.5, 0), style={"fill": "soft"})
    b = shapes.Circle(id="b", pos=(0.5, 0))
    lens = 2 * math.acos(0.5) - math.sqrt(0.75)
    for op, area in (("union", 2 * math.pi - lens), ("intersection", lens),
                     ("difference", math.pi - lens), ("exclusion", 2 * math.pi - 2 * lens)):  # fmt: skip
        s.add(getattr(shapes, op)(a, b, id=op))
        sem = _sem(s.engine, op)
        assert sem["kind"] == "path" and sem["area"] == pytest.approx(area, rel=2e-3), op
        assert s.engine.document.object(op).style == {"fill": "soft"}  # a's style
    assert _sem(s.engine, "exclusion")["subpaths"] == 2
    with pytest.raises(TypeError, match="shapes or paths"):
        shapes.union(a, object(), id="nope")  # type: ignore[arg-type]


def test_a_boolean_without_shapely_says_how_to_install_it(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "shapely", None)
    with pytest.raises(pm.MissingExtra, match="--extra boolean"):
        shapes.union(shapes.Circle(id="a"), shapes.Circle(id="b", pos=(1, 0)), id="u")


# ---------------------------------------------------------------- 3. lint and layout


@pytest.mark.parametrize("passes", MODES)
def test_a_filled_path_whose_edges_cross_is_fill_crossing(passes):
    s = Scene("bow", passes=passes)
    bow = [(-2, -1), (2, 1), (2, -1), (-2, 1)]
    s.add(shapes.Path.through("bow", bow, closed=True, style={"fill": "soft"}))
    (issue,) = [i for i in s.engine.lint() if i.code == "fill_crossing"]
    assert issue.severity == "warning" and issue.ids == ("bow",) and issue.times
    assert issue.fix == ()  # no certain fix
    # the same outline unfilled, or a clean fill: nothing to say
    s2 = Scene("ok", passes=passes)
    s2.add(shapes.Path.through("bow", bow, closed=True))
    s2.add(shapes.Circle(id="disc", pos=(0, 2), style={"fill": "soft"}))
    assert s2.engine.lint() == []


@pytest.mark.parametrize("passes", MODES)
def test_an_svg_without_its_extra_or_its_file_draws_nothing_and_lint_says_why(
    passes, monkeypatch, tmp_path
):
    import sys

    pytest.importorskip("svgelements")
    (tmp_path / "logo.svg").write_text(SVG_TEXT, encoding="utf-8")
    s = Scene("svg", passes=passes, base_dir=tmp_path)
    s.add(shapes.SVG(id="logo", path="logo.svg"))
    s.add(shapes.SVG(id="gone", path="gone.svg", pos=(3, 0)))
    issues = {i.code: i for i in s.engine.lint()}
    # an empty drawing never appears, so it is an orphan too
    assert set(issues) == {"missing_file", "orphan"} and issues["missing_file"].ids == ("gone",)
    assert issues["orphan"].ids == ("gone",)
    assert _sem(s.engine, "gone")["missing_file"] == "gone.svg"
    monkeypatch.setitem(sys.modules, "svgelements", None)
    s2 = Scene("svg2", passes=passes, base_dir=tmp_path)
    s2.add(shapes.SVG(id="logo", path="logo.svg"))
    issues = {i.code: i for i in s2.engine.lint()}
    assert set(issues) == {"missing_extra", "orphan"}
    issue = issues["missing_extra"]
    assert issue.ids == ("logo",) and issue.severity == "warning"
    assert "--extra svg" in issue.fix_hint
    assert _sem(s2.engine, "logo") == {"kind": "svg", "file": "logo.svg", "parts": [],
                                       "missing_extra": "svg"}  # fmt: skip


@pytest.mark.parametrize("passes", MODES)
def test_a_label_beside_a_filled_disc_is_not_overlapping_it(passes):
    """Without OpenGL a filled path is judged by its outline (feature ``region``), not its
    box, so both modes agree that a label in the box's corner is clear of the disc."""
    s = Scene("near", passes=passes)
    s.add(shapes.Circle(id="disc", r=1.5, style={"fill": "solid"}, tags=["key"]))
    s.add(shapes.Circle(id="mark", pos=(1.35, 1.35), r=0.1, style={"fill": "solid"},
                        tags=["key"]))  # fmt: skip
    assert [i for i in s.engine.lint() if i.code == "overlap"] == []
    s.add(shapes.Circle(id="edge", pos=(1.06, 1.06), r=0.15, style={"fill": "solid"},
                        tags=["key"]))  # fmt: skip
    assert [i.ids for i in s.engine.lint() if i.code == "overlap"] == [("disc", "edge")]


# ---------------------------------------------------------------- 4. SVG

SVG_TEXT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50">
  <circle id="Sun-1" cx="20" cy="20" r="10" fill="red"/>
  <path id="frame" fill-rule="evenodd" d="M50 5 H90 V45 H50 Z M60 15 H80 V35 H60 Z"/>
  <g transform="translate(10 0)"><path d="M0 45 L20 45" fill="none" stroke="blue"/></g>
  <rect x="0" y="0" width="4" height="4"/>
  <rect id="1st" x="0" y="0" width="4" height="4"/>
  <image href="../outside.png" x="0" y="0" width="5" height="5"/>
</svg>"""


def test_an_svg_becomes_path_parts_named_from_its_ids(tmp_path):
    pytest.importorskip("svgelements")
    (tmp_path / "logo.svg").write_text(SVG_TEXT, encoding="utf-8")
    s = Scene("svg", passes=None, base_dir=tmp_path)
    s.add(shapes.SVG(id="logo", path="logo.svg", height=5.0, pos=(1, 1)))
    e = s.engine
    sem = _sem(e, "logo")
    assert sem["parts"] == ["sun_1", "frame", "path", "rect", "rect_1st"]
    st = e.state_at(0)
    assert [o.type for o in st.objects if o.parent == "logo"] == ["path"] * 5
    # 0.1 world units per SVG unit, the view box centered on pos, y up
    sun = st["logo.sun_1"].impl.features(st)
    assert np.allclose(sun["center"][:2], [1 + (20 - 50) * 0.1, 1 - (20 - 25) * 0.1], atol=0.02)
    frame = _sem(e, "logo.frame")
    assert frame["rule"] == "evenodd" and frame["area"] == pytest.approx(16 - 4, rel=1e-6)
    line = _sem(e, "logo.path")  # the moved group's line: stroked, not filled
    assert line["closed"] is False and line["length"] == pytest.approx(2.0)
    assert st["logo.path"].params.fill_opacity == 0 and st["logo.path"].params.stroke_opacity == 1
    assert st["logo.sun_1"].params.stroke_opacity == 0
    assert st["logo.frame"].roles["fill"] == "solid"  # the SVG's style roles
    assert e.lint() == []


def test_an_svg_is_read_again_when_its_file_changes(tmp_path):
    """Contract 3: the state follows the file. A composite builds its parts from it, so the
    engine forgets the states it made from the old file."""
    import os

    pytest.importorskip("svgelements")
    f = tmp_path / "logo.svg"
    f.write_text(SVG_TEXT, encoding="utf-8")
    s = Scene("svg", passes=None, base_dir=tmp_path)
    s.add(shapes.SVG(id="logo", path="logo.svg"))
    e = s.engine
    assert len(_sem(e, "logo")["parts"]) == 5
    before = e.state_at(0)["logo"].impl.world_bounds(e.state_at(0))
    f.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                 '<circle id="dot" cx="5" cy="5" r="1"/></svg>', encoding="utf-8")  # fmt: skip
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # a clock that did not tick
    assert _sem(e, "logo")["parts"] == ["dot"]
    st = e.state_at(0)
    assert st["logo"].impl.world_bounds(st) != before
    assert [o.id for o in st.objects if o.parent == "logo"] == ["logo.dot"]


def test_an_svg_reads_only_inside_the_scene_folder(tmp_path):
    s = Scene("svg", passes=None, base_dir=tmp_path)
    with pytest.raises(DocumentError, match="leaves the scene's folder"):
        s.add(shapes.SVG(id="logo", path="../logo.svg"))
    with pytest.raises(DocumentError, match="absolute"):
        s.add(shapes.SVG(id="logo", path=str(tmp_path / "logo.svg")))


def test_svg_part_names_are_valid_unique_and_stable():
    from phasor.core.ids import is_valid_name
    from phasor.domains.svg import part_names

    shapes_ = [pm.SvgShape(i, t, (), "nonzero", True, False)
               for i, t in [("A-b", "path"), ("a_b", "path"), ("", "circle"), ("", "circle"),
                            ("9lives", "rect"), ("é", "path")]]  # fmt: skip
    names = part_names(shapes_)
    assert names == ["a_b", "a_b_2", "circle", "circle_2", "rect_9lives", "path"]
    assert all(is_valid_name(n) for n in names) and names == part_names(shapes_)


# ---------------------------------------------------------------- 5. pixels


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_solid_fill_below_the_knee_comes_out_exactly_as_its_color():
    """Bloom only from the emission target: a fill is plain paint, and a role below the
    theme's knee tone-maps unchanged (CLAUDE.md beauty rules)."""
    s = Scene("paint", look="none")
    s.set("camera.height", 4)
    s.add(shapes.Rectangle(id="card", width=6, height=3, stroke_opacity=0,
                           style={"fill": "solid", "fill_color": "muted", "glow": "strong"}))  # fmt: skip
    img = s.engine.still(0.0, (320, 180))
    theme = load_theme(s.engine.document.meta.theme)
    lin = np.asarray(theme.values["emphasis"]["muted"][:3])
    assert lin.max() < theme.bloom["knee"]
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    centre = img[80:100, 150:170].reshape(-1, 3).astype(float)
    assert np.abs(centre - srgb * 255).max() <= 1.5


# ---------------------------------------------------------------- 6. the exit demo

ROOT = __import__("pathlib").Path(__file__).parents[1]
NEW_TYPES = {"path", "circle", "ellipse", "arc", "sector", "annulus", "polygon_path",
             "regular_polygon", "rectangle", "brace", "svg"}  # fmt: skip
_examples: dict[str, object] = {}


def _example(name: str):
    import runpy

    if name not in _examples:
        _examples[name] = runpy.run_path(str(ROOT / "examples" / f"{name}.py"),
                                         run_name="x")["s"].engine  # fmt: skip
    return _examples[name]


def test_the_two_examples_show_every_new_type_and_lint_clean():
    pytest.importorskip("shapely")
    pytest.importorskip("svgelements")
    seen = set()
    for name in ("shapes", "vector_toolkit"):
        e = _example(name)
        seen |= {o.type for o in e.state_at(e.duration).objects}
        assert e.lint(strict=True) == [], name
    assert seen >= NEW_TYPES
    e = _example("vector_toolkit")
    sem = {o["id"]: o["semantic"] for o in e.inspect(e.duration)["objects"]}
    assert sem["emblem"]["parts"] == ["ring", "star", "badge_1", "wave"]
    assert sem["emblem.badge_1"]["rule"] == "evenodd"
    assert sem["frame"]["area"] == pytest.approx(3.0 * 2.4 - (4 - math.pi) * 0.0625 - 1.8 * 1.2,
                                                 rel=1e-4)  # fmt: skip
    assert sem["intersection"]["area"] < sem["difference"]["area"] < sem["union"]["area"]


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize("name, times", [("shapes", [1.0, 3.4, 5.8, 7.8]),
                                         ("vector_toolkit", [1.0, 3.2, 5.6, 8.6])])  # fmt: skip
def test_example_golden_storyboards(name, times):
    pytest.importorskip("shapely")
    pytest.importorskip("svgelements")
    from tests.test_beauty import _golden

    e = _example(name)
    sheet = e.storyboard(times, size=(320, 180))["image"]
    _golden(name, sheet, e.pass_runner().renderer_class())


def test_doctor_names_a_missing_extra_without_failing(monkeypatch):
    import sys

    from phasor.render.doctor import check_extras

    assert check_extras().ok in (True, None)
    monkeypatch.setitem(sys.modules, "svgelements", None)
    c = check_extras()
    assert c.ok is None and "--extra svg" in c.detail


def test_crossings_check_a_detailed_outline_in_bounded_memory():
    """Edges are compared in blocks against those whose boxes meet, so a detailed outline (an
    SVG part, a boolean result) is checked without building N x N arrays."""
    import tracemalloc

    k = np.linspace(0, 2 * np.pi, 6000, endpoint=False)
    r = 2 + 0.3 * np.sin(40 * k)  # a wavy closed ring, 6000 edges, no crossings
    ring = np.c_[r * np.cos(k), r * np.sin(k), np.zeros_like(k)]
    tracemalloc.start()
    assert not pm.crossings([ring])
    bow = ring.copy()
    bow[[100, 3100]] = bow[[3100, 100]]  # two far vertices swapped: edges now cross
    assert pm.crossings([bow])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 200 * 1024 * 1024


@pytest.mark.parametrize("passes", MODES)
def test_tiny_waits_until_a_traced_curve_is_drawn(passes):
    from phasor import Line

    s = Scene("tiny", passes=passes)
    s.add(Line(id="long", points=[(-3, 0), (3, 0)], progress=0.001))
    assert [i for i in s.engine.lint() if i.code == "tiny"] == []
    s2 = Scene("tiny2", passes=passes)
    s2.add(Line(id="speck", points=[(0, 0), (0.005, 0)]))
    (issue,) = [i for i in s2.engine.lint() if i.code == "tiny"]
    assert issue.ids == ("speck",)


# ---------------------------------------------------------------- 6. hostile files (audit)
#
# The promise (svg.py, contract 3): a bad or hostile file in the scene's folder, or an edit
# to it, is a lint issue and draws nothing; it never makes state_at, inspect or lint raise,
# hang or run out of memory.

H = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
H += 'viewBox="0 0 100 100">'
DOT = '<circle id="dot" cx="50" cy="50" r="20"/>'


def _bump(f):
    import os

    st = os.stat(f, follow_symlinks=False)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9), follow_symlinks=False)


def _svg_scene(tmp_path, text, **kw):
    pytest.importorskip("svgelements")
    (tmp_path / "logo.svg").write_text(text, encoding="utf-8")
    s = Scene("svg", passes=None, base_dir=tmp_path, **kw)
    s.add(shapes.SVG(id="logo", path="logo.svg"))
    return s


def _survives(e, t=0.5):
    """state_at, inspect and lint all answer; the parts' points are finite."""
    st = e.state_at(t)
    e.inspect(t)
    issues = e.lint()
    for o in st.objects:
        if o.parent == "logo":
            assert all(np.isfinite(np.asarray(sub)).all() for sub in o.params.subpaths)
    return st, {i.code: i for i in issues}


def _parts(st):
    return [o.id for o in st.objects if o.parent == "logo"]


def test_svg_part_names_avoid_the_documents_ids_and_retired_ids(tmp_path):
    from phasor import Text

    f = tmp_path / "logo.svg"
    s = _svg_scene(tmp_path, H + DOT + "</svg>")
    s.add(Text(id="logo.label", text="logo"))
    s.add(Text(id="logo.star", text="x"))
    s.engine.apply([{"op": "remove_object", "id": "logo.star"}])
    e = s.engine
    assert _parts(e.state_at(0)) == ["logo.dot"]
    f.write_text(H + '<circle id="label" cx="5" cy="5" r="4"/>'
                 '<circle id="star" cx="5" cy="5" r="4"/></svg>', encoding="utf-8")  # fmt: skip
    _bump(f)
    st, _ = _survives(e)
    assert _parts(st) == ["logo.label_2", "logo.star_2"]  # a suffix, deterministically


def test_svg_part_names_are_capped_in_length():
    from phasor.domains.svg import MAX_NAME, part_names

    long = pm.SvgShape("A" * 5000, "circle", (), "nonzero", True, False)
    names = part_names([long, long])
    assert all(len(n) <= MAX_NAME + 4 for n in names) and names[0] != names[1]


def test_svg_shapes_with_coordinates_that_are_not_finite_are_dropped(tmp_path):
    bad = ('<path id="p" d="M0 0 L10 10" transform="scale(1e308) scale(1e308)"/>'
           '<path id="q" d="M0 0 L1e300 1e300 L-1e300 5 Z"/>'
           '<rect id="r" x="0" y="0" width="inf" height="5"/>')  # fmt: skip
    f = tmp_path / "logo.svg"
    s = _svg_scene(tmp_path, H + DOT + "</svg>")
    e = s.engine
    e.state_at(0)
    f.write_text(H + DOT + bad + "</svg>", encoding="utf-8")
    _bump(f)
    st, issues = _survives(e)
    assert _parts(st) == ["logo.dot"]
    assert "bad_file" in issues and issues["bad_file"].ids == ("logo",)
    assert "skipped" in issues["bad_file"].message


@pytest.mark.parametrize("box", ["0 0 0 0", "0 0 -10 -10", "0 0 nan 1e999", None])
def test_a_hostile_view_box_still_draws_finite_parts(tmp_path, box):
    head = ('<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="0">' if box is None
            else f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}">')  # fmt: skip
    s = _svg_scene(tmp_path, head + DOT + "</svg>")
    st, _ = _survives(s.engine)
    # svgelements drops every shape under an empty view box: nothing drawn, no error
    assert _parts(st) == ([] if box == "0 0 0 0" else ["logo.dot"])


@pytest.mark.parametrize("text", ["", "\x00\x01garbage" * 10, H + '<path id="p" d="M0 0 L10'])
def test_a_malformed_svg_is_bad_file_and_parsed_once(tmp_path, text, monkeypatch):
    se = pytest.importorskip("svgelements")
    calls = []
    real = se.SVG.parse

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(se.SVG, "parse", staticmethod(counted))
    pm._read_svg.cache_clear()
    s = _svg_scene(tmp_path, text)
    st, issues = _survives(s.engine)
    assert _parts(st) == []
    assert issues["bad_file"].ids == ("logo",) and issues["bad_file"].severity == "warning"
    assert len(calls) <= 1  # a bad file is remembered as bad, not parsed at every t


def _with_alarm(seconds, fn):
    import signal

    if not hasattr(signal, "SIGALRM"):
        return fn()

    def boom(*_):
        raise TimeoutError(f"took more than {seconds} s")

    old = signal.signal(signal.SIGALRM, boom)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_nested_use_that_fans_out_is_svg_too_complex_and_fast(tmp_path):
    fan = (H + '<defs><circle id="a0" cx="5" cy="5" r="1"/>'
           + "".join(f'<g id="a{i + 1}">' + f'<use href="#a{i}"/>' * 10 + "</g>"
                     for i in range(6))
           + '</defs><use href="#a6"/></svg>')  # fmt: skip
    s = _svg_scene(tmp_path, fan)
    st, issues = _with_alarm(60, lambda: _survives(s.engine))
    assert _parts(st) == []
    assert issues["svg_too_complex"].ids == ("logo",)
    d = pm.read_svg(str(tmp_path / "logo.svg"), ("x", 1, 1))
    assert d.shapes == () and "use" in d.limited and not d.problem


@pytest.mark.parametrize("kind", ["cycle", "deep"])
def test_use_cycles_and_deep_nesting_never_raise(tmp_path, kind):
    text = (H + '<g id="g1"><use xlink:href="#g2"/></g><g id="g2"><use xlink:href="#g1"/></g>'
            '<use id="self" xlink:href="#self"/>' + DOT + "</svg>" if kind == "cycle"
            else H + "<g>" * 5000 + DOT + "</g>" * 5000 + "</svg>")  # fmt: skip
    s = _svg_scene(tmp_path, text)
    st, issues = _with_alarm(60, lambda: _survives(s.engine))
    assert _parts(st) == [] and "svg_too_complex" in issues


def test_a_drawing_past_the_segment_limit_is_truncated_with_a_lint_note(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "SVG_MAX_SEGMENTS", 50)
    pm._read_svg.cache_clear()
    many = "".join(f'<rect id="r{i}" x="{i}" y="0" width="1" height="1"/>' for i in range(40))
    s = _svg_scene(tmp_path, H + many + "</svg>")
    st, issues = _survives(s.engine)
    assert 0 < len(_parts(st)) < 40
    assert issues["svg_too_complex"].ids == ("logo",)
    assert "segments" in issues["svg_too_complex"].message
    pm._read_svg.cache_clear()


def _loop(tmp_path, name):
    import os

    try:
        os.symlink(name, tmp_path / name)
    except (OSError, NotImplementedError):
        pytest.skip("no symbolic links here")


def test_a_link_that_loops_is_a_missing_file_for_every_type_that_reads_files(tmp_path):
    from phasor import Image

    pytest.importorskip("svgelements")
    _loop(tmp_path, "logo.svg")
    _loop(tmp_path, "pic.png")
    s = Scene("loop", passes=None, base_dir=tmp_path)
    s.add(shapes.SVG(id="logo", path="logo.svg"))
    s.add(Image(id="pic", path="pic.png"))
    st, _ = _survives(s.engine)
    assert _parts(st) == []
    found = {i.ids for i in s.engine.lint() if i.code == "missing_file"}
    assert found == {("logo",), ("pic",)}


def test_a_fifo_is_never_read(tmp_path):
    import os

    from phasor import Image
    from phasor.core import files

    if not hasattr(os, "mkfifo"):
        pytest.skip("no FIFOs here")
    pytest.importorskip("svgelements")
    os.mkfifo(tmp_path / "logo.svg")
    os.mkfifo(tmp_path / "pic.png")
    s = Scene("fifo", passes=None, base_dir=tmp_path)
    s.add(shapes.SVG(id="logo", path="logo.svg"))
    s.add(Image(id="pic", path="pic.png"))
    st, issues = _with_alarm(30, lambda: _survives(s.engine))
    assert _parts(st) == [] and issues["missing_file"]
    assert "not a regular file" in " ".join(i.message for i in s.engine.lint()
                                            if i.code == "missing_file")  # fmt: skip
    with pytest.raises(ValueError, match="regular file"):
        _with_alarm(10, lambda: files.read_bytes(tmp_path / "logo.svg", 100))
    assert not files.exists(tmp_path, "logo.svg") and files.stamp(tmp_path, "logo.svg")[1] is None


# ---------------------------------------------------------------- 7. bounded parameters (audit)

import phasor.domains.path as shapes_path  # noqa: E402


class _Any:
    """A DrawContext that counts every call, whatever it is."""

    def __init__(self):
        self.n = 0

    def __getattr__(self, name):
        def call(*a, **k):
            self.n += 1

        return call


EXTREMES = (1e9, 1e12, 1e200, -1e200, 1e-300, 1e8)


def _shape_cases():
    for cls in (shapes.Circle, shapes.Ellipse, shapes.Arc, shapes.Sector, shapes.Annulus,
                shapes.Polygon, shapes.RegularPolygon, shapes.Rectangle, shapes.Brace):  # fmt: skip
        for name, f in cls.Params.model_fields.items():
            if name in ("opacity", "dim", "z_index", "flash", "progress", "fill_opacity",
                        "stroke_opacity"):  # fmt: skip
                continue
            ann = str(f.annotation)
            for v in EXTREMES:
                if "float" in ann and "tuple" not in ann:
                    yield cls, {name: v}
                elif "Vec3" in ann or ("tuple" in ann and "float" in ann and name != "vertices"):
                    yield cls, {name: (v, -v, 0.0)}
                elif name == "vertices":
                    yield cls, {name: [(0, 0, 0), (v, 0, 0), (0, v, 0)]}
                elif "int" in ann:
                    yield cls, {name: int(min(abs(v), 1e9))}


@pytest.mark.parametrize("dash", ["dashed", "dotted"])
def test_shape_params_at_their_bounds_never_exhaust_memory(dash):
    """Every value a shape's parameters accept draws in bounded time and memory; a value
    past a bound is a clean validation error when it is set, never a crash later."""
    import time

    for cls, params in _shape_cases():
        s = Scene("fuzz", passes=None)
        try:
            s.add(cls(id="x", style={"dash": dash, "fill": "solid"}, **params))
        except DocumentError:
            continue  # refused when set: fine
        t0 = time.perf_counter()

        def go(s=s):
            st = s.engine.state_at(0)
            ctx = _Any()
            st["x"].impl.draw(ctx, st, "color")
            s.engine.inspect(0)
            return ctx.n

        n = _with_alarm(30, go)
        assert n <= 2 * shapes_path.MAX_DASHES + 16, (cls.__name__, params, n)
        assert time.perf_counter() - t0 < 20, (cls.__name__, params)


def test_arc_angle_is_bounded_and_a_full_turn_stays_open():
    s = Scene("arc", passes=None)
    for bad in (1e8, -720, 360.5):
        with pytest.raises(DocumentError, match="angle"):
            s.add(shapes.Arc(id="x", angle=bad))
    s.add(shapes.Arc(id="full", r=1, angle=360, tips="both"))
    sem = _sem(s.engine, "full")
    assert sem["closed"] is False and sem["tips"] == "both"
    assert "area" not in sem  # an open arc has no area unless filled
    assert sem["length"] == pytest.approx(2 * math.pi, rel=1e-4)
    ctx = _Any()
    st = s.engine.state_at(0)
    st["full"].impl.draw(ctx, st, "color")
    assert ctx.n >= 3  # the stroke and both tips


def test_a_brace_with_no_width_is_open():
    s = Scene("brace", passes=None)
    s.add(shapes.Brace(id="b", start=(1, 1, 0), end=(1, 1, 0)))
    assert _sem(s.engine, "b")["closed"] is False


def test_dashes_are_capped_per_path_and_counted_from_where_they_start():
    poly = np.array([[0.0, 0, 0], [1e6, 0, 0]])
    many = pm.dashes(poly, 1e-3, 0.5, limit=100)
    assert len(many) == 100 and many[-1][-1][0] < 1  # the first ones
    late = pm.dashes(poly, 1.0, 0.5, start=1e6 - 3)  # three dashes, found without a long loop
    assert len(late) == 3
    assert len(pm.dots(poly, 1e-3, limit=100)) <= 101
    assert pm.dashes(poly, 1.0, 0.5, upto=float("inf"), limit=10)  # never OverflowError


def test_a_huge_dashed_circle_draws_a_bounded_number_of_dashes_and_lint_says_so():
    import time

    s = Scene("dash", passes=None)
    s.add(shapes.Circle(id="big", r=1e4, style={"dash": "dashed"}))
    st = s.engine.state_at(0)
    ctx = _Any()
    t0 = time.perf_counter()
    st["big"].impl.draw(ctx, st, "color")
    assert ctx.n <= 2 * shapes_path.MAX_DASHES + 16 and time.perf_counter() - t0 < 10
    assert _sem(s.engine, "big")["dash_limit"] > shapes_path.MAX_DASHES
    (issue,) = [i for i in s.engine.lint() if i.code == "dash_limit"]
    assert issue.ids == ("big",) and issue.severity == "warning"
    small = Scene("dash2", passes=None)
    small.add(shapes.Circle(id="c", r=1, style={"dash": "dashed"}))
    assert "dash_limit" not in _sem(small.engine, "c")


def test_a_theme_dash_too_small_to_draw_is_a_theme_error(tmp_path, monkeypatch):
    from phasor.style import theme as th

    text = (th.THEME_DIR / "midnight_glow.toml").read_text(encoding="utf-8")
    (tmp_path / "tiny_dash.toml").write_text(text.replace("dashed = 0.03", "dashed = 1e-9"),
                                             encoding="utf-8")  # fmt: skip
    (tmp_path / "nan_dash.toml").write_text(text.replace("dotted = 0.014", "dotted = nan"),
                                            encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(th, "THEME_DIR", tmp_path)
    th.load_theme.cache_clear()
    try:
        for name in ("tiny_dash", "nan_dash"):
            with pytest.raises(th.ThemeError, match=r"\[dash\]"):
                th.load_theme(name)
    finally:
        th.load_theme.cache_clear()


# ---------------------------------------------------------------- 8. files and the engine (audit)


def test_a_worker_never_bakes_and_refuses_a_file_changed_mid_render(tmp_path, monkeypatch):
    """ADR 0003 / contract 7: a render worker applies its parent's bake. A file that changes
    under a render would mix old and new frames (and make the worker bake on its own), so
    the worker stops with a clear error instead; a payload made before the change is
    refused."""
    import phasor.engine as engine_module
    from phasor.engine import Engine

    f = tmp_path / "logo.svg"
    s = _svg_scene(tmp_path, H + DOT + "</svg>")
    payload = s.engine.worker_payload()

    def no_bake(*a, **k):
        raise AssertionError("a worker must not bake")

    monkeypatch.setattr(engine_module, "bake", no_bake)
    w = Engine.from_payload(payload, passes=None)
    assert _parts(w.state_at(0.0)) == ["logo.dot"]
    f.write_text(H + '<rect id="box" x="1" y="1" width="8" height="8"/></svg>', encoding="utf-8")
    _bump(f)
    with pytest.raises(RuntimeError, match="changed while"):
        w.state_at(0.1)
    with pytest.raises(ValueError, match="changed since"):
        Engine.from_payload(payload, passes=None)


def test_a_tracked_file_path_is_read_again_when_any_of_its_files_changes(tmp_path):
    """Contract 3 for a ``Files`` parameter on a track: every file it can name is watched,
    by the engine's caches and by lint ``missing_file``."""
    pytest.importorskip("svgelements")
    (tmp_path / "a.svg").write_text(H + DOT + "</svg>", encoding="utf-8")
    b = tmp_path / "b.svg"
    b.write_text(H + '<rect id="box" x="1" y="1" width="8" height="8"/></svg>', encoding="utf-8")
    s = Scene("track", passes=None, base_dir=tmp_path)
    s.add(shapes.SVG(id="logo", path="a.svg"))
    s.at(1.0).set("logo.path", "b.svg")
    e = s.engine
    assert _parts(e.state_at(0.5)) == ["logo.dot"]
    assert _parts(e.state_at(2.0)) == ["logo.box"]
    b.write_text(H + '<circle id="ring" cx="5" cy="5" r="4"/></svg>', encoding="utf-8")
    _bump(b)
    assert _parts(e.state_at(2.0)) == ["logo.ring"]
    b.unlink()
    assert [i.path for i in e.lint() if i.code == "missing_file"] == ["logo.path"]


# ---------------------------------------------------------------- 9. booleans on degenerate input


def test_a_boolean_on_degenerate_input_never_raises_a_raw_geos_error():
    """Found by fuzzing (seed 1): a tiny self-crossing ring beside a collinear one, against
    a single point. GEOS gave up ("Unable to determine overlay result"); the operation now
    snaps and retries, and otherwise says which operation failed."""
    pytest.importorskip("shapely")
    tiny = [(-2.0034340279292995e-05, 1.152610784114357e-05),
            (4.533513975001653e-06, -3.248761329274574e-06),
            (-1.158269124574922e-05, -2.7177632846330743e-05),
            (-1.5595981539131942e-07, 7.4402495924330395e-06),
            (2.991709870356918e-06, -1.2379767855284872e-05),
            (8.070447042553143e-06, 4.1491399696003755e-06),
            (-1.512200140513951e-05, -1.6331134656987116e-05),
            (-9.022840195666627e-06, 1.2991609933261012e-05)]  # fmt: skip
    xs = [1.3092309880051034, 1.6428978471782374, 1.8753429955333167, 2.012558812625586,
          2.339622530301911, 3.7008388585962013, 3.9569858057882845, 4.189676576177723,
          4.840356348224977, 8.90418981938739, 9.561977972281673]  # fmt: skip
    ys = [2.618461976010207, 3.285795694356475, 3.7506859910666335, 4.025117625251172,
          4.679245060603822, 7.401677717192403, 7.913971611576569, 8.379353152355446,
          9.680712696449953, 17.80837963877478, 19.123955944563345]  # fmt: skip
    ring = lambda pts: np.array([(x, y, 0.0) for x, y in [*pts, pts[0]]])  # noqa: E731
    a = [ring(tiny), ring(list(zip(xs, ys, strict=True)))]
    b = [np.full((1, 3), -0.08901580380121218)]
    for op in pm.OPS:
        try:
            out = pm.boolean(op, a, b, "evenodd", "evenodd")
        except ValueError as e:
            assert op in str(e)
        else:
            assert all(np.isfinite(r).all() for r in out)
