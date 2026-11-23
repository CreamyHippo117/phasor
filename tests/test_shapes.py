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
                                         ("vector_toolkit", [1.0, 3.0, 5.2, 8.2])])  # fmt: skip
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
