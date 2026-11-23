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
