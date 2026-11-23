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


# ---------------------------------------------------------------- fill cost (independent audit)


def _region_reference(polys, rule="nonzero"):
    """``path_math.region`` as Milestone 17 wrote it (a ring-by-ring, point-by-point loop):
    the vectorised one must give exactly the same regions."""
    flat = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 3]
    o, e1, e2 = pm.plane_of(np.vstack(flat))
    rings = []
    for p in flat:
        q = p - o
        ring = pm._open_ring(np.c_[q @ e1, q @ e2])
        if len(ring) >= 3 and abs(pm.signed_area(ring)) > 1e-14:
            rings.append(ring)
    n = len(rings)
    orient = [1 if pm.signed_area(r) > 0 else -1 for r in rings]
    probes = [pm._probe(r) for r in rings]
    contains = [[j != i and pm.inside(probes[i], rings[j]) for j in range(n)] for i in range(n)]
    filled = []
    for i in range(n):
        wind = sum(orient[j] for j in range(n) if contains[i][j])
        depth = sum(contains[i])
        if rule == "nonzero":
            fin, fout = (wind + orient[i]) != 0, wind != 0
        else:
            fin, fout = (depth + 1) % 2 == 1, depth % 2 == 1
        filled.append("outer" if fin and not fout else "hole" if fout and not fin else None)
    outers = [i for i in range(n) if filled[i] == "outer"]
    holes = {i: [] for i in outers}
    for i in range(n):
        if filled[i] == "hole":
            around = [j for j in outers if contains[i][j]]
            if around:
                holes[min(around, key=lambda j: abs(pm.signed_area(rings[j])))].append(rings[i])
    return tuple((rings[i], tuple(holes[i])) for i in outers)


def _same_shapes(a, b) -> bool:
    return len(a) == len(b) and all(
        np.array_equal(oa, ob) and len(ha) == len(hb)
        and all(np.array_equal(x, y) for x, y in zip(ha, hb, strict=True))
        for (oa, ha), (ob, hb) in zip(a, b, strict=True)
    )  # fmt: skip


def _awkward_rings(seed: int) -> list[np.ndarray]:
    """Nested, overlapping, touching and degenerate rings: holes both ways round, horizontal
    edges at shared heights, collinear points, a zero-area sliver, a repeated point."""
    rng = np.random.default_rng(seed)
    rings = []
    for _ in range(int(rng.integers(3, 25))):
        c, r = rng.uniform(-3, 3, 2), rng.uniform(0.05, 3)
        k = int(rng.integers(3, 40))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        if rng.random() < 0.5:
            ang = ang[::-1]  # clockwise
        pts = np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), np.zeros(k)]
        if rng.random() < 0.3:
            pts[:, 1] = np.round(pts[:, 1], 1)  # horizontal edges, shared heights
        rings.append(np.vstack([pts, pts[:1]]))
    rings.append(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [0, 0, 0]], float))
    rings.append(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0]], float))  # no area
    sq = np.array([[-1, -1, 0], [1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float)
    rings += [sq, sq[::-1] * 0.5, sq * 0.25]
    return rings


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("rule", pm.RULES)
def test_the_vectorised_fill_region_is_exactly_the_ring_by_ring_one(seed, rule):
    rings = _awkward_rings(seed)
    assert _same_shapes(pm.region(rings, rule).shapes, _region_reference(rings, rule))


def test_a_ring_of_very_many_points_is_tested_in_bounded_memory():
    """Probes are tested in chunks: one ring of 300 000 points around 600 small rings."""
    t = np.linspace(0, 2 * np.pi, 300_000, endpoint=False)
    big = np.c_[50 * np.cos(t), 50 * np.sin(t), np.zeros_like(t)]
    small = [pm.flatten(_square((x, y), 0.2), 1e-3) for x in range(-12, 12) for y in range(-12, 13)]
    reg = pm.region([np.vstack([big, big[:1]]), *small], "evenodd")
    assert len(reg.shapes) == 1 and len(reg.shapes[0][1]) == len(small)


def test_a_400_glyph_text_fills_into_a_circle_quickly():
    """Audit 17 #2: a 400-glyph sentence morphing into a filled circle took 8.5-11.6 s a
    frame (its even-odd layer of 512 rings: 261 632 ``inside`` calls one by one). The
    budget is generous (CI runners, Windows)."""
    from phasor import Scene, Text, shapes
    from phasor.domains import morph

    txt = ("The quick brown fox jumps over the lazy dog. " * 11)[:480]
    s = Scene("fill_cost", passes=None, look="none")
    s.set("camera.height", 8)
    s.marker("go", 1.0)
    s.add(Text(id="w", text=txt, style={"size": "annotation"}))
    s.add(shapes.Circle(id="c", r=1, style={"fill": "soft"}))
    s.at("go").transform("w", "c")
    from phasor.core import record

    st = s.engine.state_at(1.5)
    items = morph.frame(st, "c")
    rings = [r for it in items for r in (it.fill_from or ()) if len(r) >= 3]
    assert len(rings) >= 400
    t0 = time.perf_counter()
    pm.fill(rings, "evenodd")
    assert time.perf_counter() - t0 < 3.0
    # the whole frame: both fill layers (the target's one ring of ~40 000 points crossing
    # itself took 1.3-5.9 s in earcut alone) and the strokes; the plan is made once
    for t in (1.1, 1.9):
        st = s.engine.state_at(t)
        t0 = time.perf_counter()
        record.record(st["c"].impl, st)
        assert time.perf_counter() - t0 < 4.0


def test_the_in_between_fill_thins_only_rings_past_its_limit():
    from phasor.domains import morph

    small = pm.flatten(pm.circle((0, 0, 0), 1.0), 1e-3)
    assert len(small) <= morph.MAX_FILL_RING and morph._thin(small) is small
    t = np.linspace(0, 2 * np.pi, 3 * morph.MAX_FILL_RING + 7)
    big = np.c_[np.cos(t), np.sin(t), 0 * t]
    thin = morph._thin(big)
    assert len(thin) <= morph.MAX_FILL_RING and np.array_equal(thin[0], big[0])
