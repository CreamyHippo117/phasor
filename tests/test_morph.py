"""Morphing paths: ``domains/morph_math.py`` (pure geometry) and the ``transform`` verb between
any two outlines. Sections: 1. the plan (counts, pairing, alignment, bounds); 2. objects."""

from __future__ import annotations

import time

import numpy as np
import pytest

from phasor.domains import morph_math as mm
from phasor.domains import path_math as pm


def _square(c=(0.0, 0.0), r=1.0, ccw=True):
    x, y = c
    pts = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    return pm.polyline(pts if ccw else pts[::-1], closed=True)


def _line(a, b):
    return pm.polyline([a, b])


def _ends(p: mm.Plan, m: float):
    return [poly for poly, _, _ in mm.at(p, m)]


# ---------------------------------------------------------------- 1. the plan


def test_identical_paths_make_an_identity_plan():
    a = [pm.circle((0, 0, 0), 1.0), _square((3, 0))]
    p = mm.plan(a, [s.copy() for s in a])
    assert p.identical and mm.at(p, 0.5) == []


def test_a_circle_becomes_a_square_with_points_moving_little():
    p = mm.plan([pm.circle((0, 0, 0), 1.0)], [_square(r=0.9)])
    assert not p.identical and p.closed == (True,)
    (start,) = _ends(p, 0.0)
    (end,) = _ends(p, 1.0)
    # every start point is on the circle, every end point on the square
    assert np.allclose(np.linalg.norm(start[:, :2], axis=1), 1.0, atol=2e-3)
    assert np.allclose(np.abs(end[:, :2]).max(axis=1), 0.9, atol=1e-9)
    # least travel: no point crosses the middle (a turned ring would)
    assert np.linalg.norm(end - start, axis=1).max() < 0.5


def test_counts_match_by_splitting_the_longest_pieces():
    one = [pm.circle((0, 0, 0), 1.0)]
    three = [_square((-3, 0), 0.3), _square((0, 0), 0.3), _square((3, 0), 0.3)]
    p = mm.plan(one, three)
    assert len(p.a) == 3 and sorted(p.dst) == [0, 1, 2] and p.src == (0, 0, 0)
    assert p.closed == (False, False, False)  # a split ring is open pieces
    back = mm.plan(three, one)
    assert len(back.a) == 3 and back.dst == (0, 0, 0)


def test_pieces_pair_with_the_nearest_partner():
    left, right = _square((-4, 0), 0.5), _square((4, 0), 0.5)
    p = mm.plan([left, right], [_square((4, 1), 0.5), _square((-4, 1), 0.5)])
    for a, b in zip(p.a, p.b, strict=True):
        assert np.sign(a[:, 0].mean()) == np.sign(b[:, 0].mean())


def test_a_hole_keeps_the_targets_winding():
    outer, hole = _square(r=2.0), _square(r=0.5, ccw=False)
    p = mm.plan([pm.circle((0, 0, 0), 2.0), pm.circle((0, 0, 0), 0.6)], [outer, hole])
    for a, b, closed in zip(p.a, p.b, p.closed, strict=True):
        assert closed
        assert np.sign(pm.signed_area(a)) == np.sign(pm.signed_area(b))
    # mid-way, the fill under non-zero still has its hole
    polys = _ends(p, 0.5)
    reg = pm.region(polys, "nonzero")
    assert reg.area < pm.region([polys[0]], "nonzero").area


def test_a_reversed_ring_is_turned_round_not_dragged_across():
    a = [_square(r=1.0, ccw=True)]
    b = [_square(r=1.0, ccw=False)]  # the same square, wound the other way
    p = mm.plan(a, b)
    (mid,) = _ends(p, 0.5)
    assert np.abs(mid[:, :2]).max() == pytest.approx(1.0, abs=1e-9)  # stays on the square


def test_closed_and_open_meet_without_a_jump():
    ring = [pm.circle((0, 0, 0), 1.0)]
    line = [_line((-1, -1), (1, -1))]
    for a, b in ((ring, line), (line, ring)):
        p = mm.plan(a, b)
        assert p.closed == (False,)
        (start,) = _ends(p, 0.0)
        (end,) = _ends(p, 1.0)
        assert len(start) >= mm.MIN_POINTS and len(end) >= mm.MIN_POINTS


def test_an_open_line_runs_the_shorter_way():
    p = mm.plan([_line((0, 0), (1, 0))], [_line((1, 0.1), (0, 0.1))])
    (mid,) = _ends(p, 0.5)
    assert np.allclose(mid[:, 1], 0.05) and np.ptp(mid[:, 0]) == pytest.approx(1.0)


def test_nothing_grows_from_and_shrinks_to_centres():
    p = mm.plan([], [_square((2, 0), 0.5)])
    assert _ends(p, 0.0) == []  # a point: nothing to draw yet
    (mid,) = _ends(p, 0.5)
    assert mid[:, :2].mean(axis=0) == pytest.approx((2, 0), abs=0.02)
    assert np.abs(mid[:, :2] - (2, 0)).max() == pytest.approx(0.25, abs=0.02)
    q = mm.plan([_square((2, 0), 0.5)], [])
    assert _ends(q, 1.0) == [] and len(_ends(q, 0.0)) == 1
    assert mm.plan([], []).a == ()


def test_a_zero_length_subpath_never_makes_a_degenerate_piece():
    dot = pm.polyline([(1, 1), (1, 1)])
    p = mm.plan([dot], [_square()])
    for m in (0.0, 1e-9, 0.3, 1.0):
        for poly, _, _ in mm.at(p, m):
            assert len(poly) >= 2 and np.ptp(poly, axis=0).max() > 0


def test_the_plan_does_not_depend_on_the_order_subpaths_are_written():
    a = [_square((-3, 0), 0.4), pm.circle((0, 0, 0), 1.0), _square((3, 1), 0.2)]
    b = [pm.circle((0, 2, 0), 0.5), _square((1, -1), 1.0)]
    one = _ends(mm.plan(a, b), 0.4)
    two = _ends(mm.plan(a[::-1], b[::-1]), 0.4)
    key = lambda ps: sorted(tuple(np.round(q.mean(axis=0), 9)) for q in ps)  # noqa: E731
    assert key(one) == key(two)
    assert all(np.array_equal(x, y) for x, y in zip(_ends(mm.plan(a, b), 0.4), one, strict=True))


def test_the_plan_is_bounded_whatever_the_input():
    """Every count is bounded: subpaths, points per piece, points in all; time too."""
    rng = np.random.default_rng(1)
    many = [_square(tuple(rng.uniform(-50, 50, 2)), 0.1) for _ in range(mm.MAX_SUBPATHS + 300)]
    huge = pm.polyline(rng.uniform(-1, 1, (200_001, 2)))  # 200 000 cubics
    t0 = time.perf_counter()
    p = mm.plan(many, [huge])
    assert len(p.a) <= mm.MAX_SUBPATHS and len(p.gone) == 300
    assert p.points <= mm.MAX_TOTAL + mm.MIN_POINTS * mm.MAX_SUBPATHS
    assert max(len(x) for x in p.a) <= mm.MAX_POINTS
    q = mm.plan([huge], many)
    assert len(q.new) == 300 and q.points <= mm.MAX_TOTAL + mm.MIN_POINTS * mm.MAX_SUBPATHS
    assert time.perf_counter() - t0 < 30
    assert len(mm.dense(huge)) <= 2 * mm.SAMPLE_CUBICS + 2


def test_a_shape_has_the_same_signature_anywhere_and_at_any_size():
    a = [_square((0, 0), 1.0)]
    assert mm.signature(a) == mm.signature([_square((5, -2), 3.0)])
    assert mm.signature(a) != mm.signature([pm.circle((0, 0, 0), 1.0)])
    assert mm.signature([]) == "empty"
    assert mm.digest(a) == mm.digest([s.copy() for s in a]) != mm.digest([_square((0, 0), 2.0)])


def test_growing_many_subpaths_from_nothing_is_bounded_too():
    many = [pm.circle((k * 0.1, 0, 0), 5.0) for k in range(mm.MAX_SUBPATHS + 10)]
    for p in (mm.plan([], many), mm.plan(many, [])):
        assert len(p.a) == mm.MAX_SUBPATHS
        assert p.points <= mm.MAX_TOTAL + mm.MIN_POINTS * mm.MAX_SUBPATHS
