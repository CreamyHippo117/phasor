"""Morphing one path into another (pure NumPy, no drawing).

A path is a list of subpaths of cubic Béziers (``domains/path_math.py``). ``plan(a, b)``
decides how path ``a`` becomes path ``b``; ``at(plan, m)`` is the path ``m`` of the way
(0: ``a``, 1: ``b``). The plan is made once and each frame is a weighted mean:

1. **Sample.** Each subpath is sampled densely along its cubics (a bounded number of points
   per cubic; a subpath of very many cubics keeps every k-th anchor), so nothing below
   depends on how many control points a path has.
2. **Match counts.** Subpaths are put in a canonical order (longest first; ties by where
   they lie, then by index), so the result never depends on the order a script wrote them.
   When one side has fewer, its longest piece is split at half its length until the counts
   agree (a closed ring splits into two open halves). A side with no subpaths at all grows
   from (or shrinks to) the centre of each subpath of the other side. Past
   ``MAX_SUBPATHS`` the shortest subpaths of a side take no part: the caller fades them
   (``Plan.gone``, ``Plan.new``).
3. **Pair.** Pieces pair by least travel of their centres and sizes (an optimal assignment,
   up to ``MAX_ASSIGN`` pairs; past that, in canonical order).
4. **Resample** both pieces of a pair to the same number of points by arc length (between
   ``MIN_POINTS`` and ``MAX_POINTS``; all pairs together at most ``MAX_TOTAL``).
5. **Align.** Two closed rings keep the target's winding (the source ring is reversed when
   it turns the other way, so a hole stays a hole under the non-zero rule) and turn to the
   start that moves the points least (exactly: circular cross-correlation by FFT). An open
   piece runs forwards or backwards, whichever moves less. A closed ring paired with an
   open piece opens at the point that moves least.

Identical paths make an identical plan (``Plan.identical``): the caller draws the path
itself at every ``m``. ``at`` never returns a piece with fewer than two distinct points.

``signature(subpaths)`` names a shape whatever its place and size (``transform_matching``
pairs groups of equal signature).
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phasor.domains import path_math as pm

MAX_SUBPATHS = 512  # subpaths per side that morph; the shortest beyond fade (lint morph_limit)
MAX_POINTS = 512  # points per resampled piece
MIN_POINTS = 8  # points per resampled piece, at least
MAX_TOTAL = 40_000  # points in one morph, all pieces together
MAX_ASSIGN = 256  # pairs found by an optimal assignment; past it, paired in canonical order
SAMPLE_CUBICS = 4 * MAX_POINTS  # cubics a subpath is sampled through, at most
SAMPLES_PER_CUBIC = 64  # points sampled along each cubic, at most
SIGNATURE_DIGITS = 3  # decimals of the unit-sized outline a signature rounds to


@dataclass(frozen=True)
class Piece:
    """A stretch of one dense subpath, between arc lengths ``s0`` and ``s1``."""

    index: int  # which subpath of its side
    s0: float
    s1: float
    closed: bool  # a whole closed ring (not split)


@dataclass(frozen=True)
class Plan:
    a: tuple[np.ndarray, ...]  # resampled source pieces (n_i, 3)
    b: tuple[np.ndarray, ...]  # resampled target pieces, paired with ``a``
    closed: tuple[bool, ...]  # both ends of the pair are closed rings
    src: tuple[int, ...]  # the source subpath each pair comes from
    dst: tuple[int, ...]  # the target subpath each pair goes to
    gone: tuple[int, ...] = ()  # source subpaths that take no part (past MAX_SUBPATHS): fade
    new: tuple[int, ...] = ()  # target subpaths that take no part: fade in
    identical: bool = False

    @property
    def points(self) -> int:
        return int(sum(len(x) for x in self.a))


# ---------------------------------------------------------------- sampling


def dense(sub: np.ndarray) -> np.ndarray:
    """A subpath as a dense polyline: each cubic sampled at a bounded number of points; a
    subpath of more than ``SAMPLE_CUBICS`` cubics is first thinned to every k-th anchor
    (straight between them). Consecutive repeats are dropped (at least one point stays)."""
    sub = np.asarray(sub, dtype=np.float64)
    k = (len(sub) - 1) // 3
    if k <= 0:
        return sub[:1].copy()
    if k > SAMPLE_CUBICS:
        anchors = sub[::3]
        step = int(np.ceil(k / SAMPLE_CUBICS))
        keep = anchors[::step]
        if not np.array_equal(keep[-1], anchors[-1]):
            keep = np.vstack([keep, anchors[-1]])
        pts = keep
    else:
        c = pm.cubics(sub)
        per = max(2, min(SAMPLES_PER_CUBIC, (2 * SAMPLE_CUBICS) // k))
        u = (np.arange(1, per + 1) / per)[:, None]
        v = 1 - u
        p0, p1, p2, p3 = (c[:, i][:, None, :] for i in range(4))
        body = v**3 * p0 + 3 * v * v * u * p1 + 3 * v * u * u * p2 + u**3 * p3
        pts = np.vstack([sub[:1], body.reshape(-1, 3)])
    step = np.abs(np.diff(pts, axis=0)).max(axis=1)
    scale = max(1.0, float(np.abs(pts).max()))
    return pts[np.r_[True, step > 1e-12 * scale]]


def _resample(poly: np.ndarray, along: np.ndarray, s0: float, s1: float, n: int,
              endpoint: bool = True) -> np.ndarray:  # fmt: skip
    """``n`` points evenly spaced by arc length between ``s0`` and ``s1`` (with
    ``endpoint=False`` the last one is left out: a ring's start is not repeated)."""
    if len(poly) == 1 or along[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    u = np.linspace(s0, s1, n, endpoint=endpoint)
    return np.stack([np.interp(u, along, poly[:, k]) for k in range(3)], axis=1)


def _normal(ring: np.ndarray) -> np.ndarray:
    """Newell's normal of a ring (its length is twice the area; zero for no area)."""
    q, r = ring, np.roll(ring, -1, axis=0)
    return np.array([np.sum((q[:, 1] - r[:, 1]) * (q[:, 2] + r[:, 2])),
                     np.sum((q[:, 2] - r[:, 2]) * (q[:, 0] + r[:, 0])),
                     np.sum((q[:, 0] - r[:, 0]) * (q[:, 1] + r[:, 1]))])  # fmt: skip


# ---------------------------------------------------------------- the plan


@dataclass(frozen=True)
class _Side:
    polys: tuple[np.ndarray, ...]  # dense polylines
    along: tuple[np.ndarray, ...]  # arc lengths at their points
    closed: tuple[bool, ...]
    order: tuple[int, ...]  # canonical order of the subpaths that take part
    left_out: tuple[int, ...]  # past MAX_SUBPATHS


def _side(subs: Sequence[np.ndarray], closed: Sequence[bool]) -> _Side:
    polys = tuple(dense(s) for s in subs)
    along = tuple(pm.arc_lengths(p) if len(p) > 1 else np.zeros(1) for p in polys)
    centres = [p.mean(axis=0) for p in polys]
    order = sorted(range(len(polys)), key=lambda i: (-round(float(along[i][-1]), 9),
                                                     *np.round(centres[i], 9), i))  # fmt: skip
    return _Side(polys, along, tuple(bool(c) for c in closed), tuple(order[:MAX_SUBPATHS]),
                 tuple(sorted(order[MAX_SUBPATHS:])))  # fmt: skip


def _pieces(side: _Side, count: int) -> list[Piece]:
    """``count`` pieces of a side's subpaths: each whole subpath, then the longest piece split
    at half its length until there are ``count`` (``count`` >= the number of subpaths)."""
    pieces = [Piece(i, 0.0, float(side.along[i][-1]), side.closed[i]) for i in side.order]
    heap = [(-(p.s1 - p.s0), k, p) for k, p in enumerate(pieces)]
    heapq.heapify(heap)
    serial = len(pieces)
    while len(heap) < count:
        _, _, p = heapq.heappop(heap)
        mid = (p.s0 + p.s1) / 2
        for half in (Piece(p.index, p.s0, mid, False), Piece(p.index, mid, p.s1, False)):
            heapq.heappush(heap, (-(half.s1 - half.s0), serial, half))
            serial += 1
    # back in the order they lie: by subpath (canonical order), then along it
    rank = {i: k for k, i in enumerate(side.order)}
    return sorted((p for _, _, p in heap), key=lambda p: (rank[p.index], p.s0))


def _centre(side: _Side, p: Piece) -> tuple[np.ndarray, float]:
    """The centre and size of a piece (a few points along it)."""
    pts = _resample(side.polys[p.index], side.along[p.index], p.s0, p.s1, 9)
    return pts.mean(axis=0), float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _pair(sa: _Side, pa: list[Piece], sb: _Side, pb: list[Piece]) -> list[int]:
    """For each piece of ``pa``, the index of its partner in ``pb`` (least travel)."""
    n = len(pa)
    if n > MAX_ASSIGN:
        return list(range(n))
    from scipy.optimize import linear_sum_assignment

    ca = [_centre(sa, p) for p in pa]
    cb = [_centre(sb, p) for p in pb]
    a_c, a_s = np.array([c for c, _ in ca]), np.array([s for _, s in ca])
    b_c, b_s = np.array([c for c, _ in cb]), np.array([s for _, s in cb])
    cost = ((a_c[:, None, :] - b_c[None, :, :]) ** 2).sum(-1) + (a_s[:, None] - b_s[None]) ** 2
    rows, cols = linear_sum_assignment(cost)
    out = [0] * n
    for r, c in zip(rows, cols, strict=True):
        out[int(r)] = int(c)
    return out


def _best_roll(a: np.ndarray, b: np.ndarray) -> int:
    """The k for which ``np.roll(a, -k)`` is nearest ``b`` (both (n, 3), cyclic): circular
    cross-correlation by FFT, so every start is tried at O(n log n)."""
    fa, fb = np.fft.rfft(a, axis=0), np.fft.rfft(b, axis=0)
    corr = np.fft.irfft(np.conj(fb) * fa, n=len(a), axis=0).sum(axis=1)
    return int(np.argmax(np.round(corr, 9)))  # rounded: ties go to the smallest k


def _cost(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).sum())


def _align(a: np.ndarray, b: np.ndarray, a_ring: bool, b_ring: bool,
           both: bool) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``a`` and ``b`` (n points each) laid so that the points move least."""
    if both:  # two rings: keep b's winding, then the best start
        if float(np.dot(_normal(a), _normal(b))) < 0:
            a = a[::-1].copy()
        return np.roll(a, -_best_roll(a, b), axis=0), b
    if a_ring or b_ring:  # one ring opens (or closes) where the points move least
        ring, line = (a, b) if a_ring else (b, a)
        loop = ring[:-1]  # its last point repeats the first
        best = None
        for cand in (loop, loop[::-1].copy()):
            k = _best_roll(cand, line[:-1])
            r = np.roll(cand, -k, axis=0)
            r = np.vstack([r, r[:1]])
            c = _cost(r, line)
            if best is None or c < best[0] - 1e-12:
                best = (c, r)
        return (best[1], b) if a_ring else (a, best[1])
    rev = a[::-1]
    return (rev.copy(), b) if _cost(rev, b) < _cost(a, b) - 1e-12 else (a, b)


def plan(a: Sequence[np.ndarray], b: Sequence[np.ndarray], a_closed: Sequence[bool] | None = None,
         b_closed: Sequence[bool] | None = None) -> Plan:  # fmt: skip
    """How subpaths ``a`` become subpaths ``b`` (each (1 + 3k, 3)). ``a_closed`` and
    ``b_closed`` say which subpaths are closed rings (default: those that end where they
    begin)."""
    a = [np.asarray(s, dtype=np.float64) for s in a]
    b = [np.asarray(s, dtype=np.float64) for s in b]
    if len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y)
                                for x, y in zip(a, b, strict=True)):  # fmt: skip
        return Plan((), (), (), (), (), identical=True)
    ac = list(a_closed) if a_closed is not None else [pm.is_closed(s) for s in a]
    bc = list(b_closed) if b_closed is not None else [pm.is_closed(s) for s in b]
    sa, sb = _side(a, ac), _side(b, bc)
    n = max(len(sa.order), len(sb.order))
    if n == 0:
        return Plan((), (), (), (), (), sa.left_out, sb.left_out)
    if not sa.order or not sb.order:  # grow from (or shrink to) each subpath's centre
        full = sb if not sa.order else sa
        pieces = _pieces(full, n)
        counts = _within_total([_count(full, p) for p in pieces])
        pairs_full = [
            _resample(full.polys[p.index], full.along[p.index], p.s0, p.s1, c, not p.closed)
            for p, c in zip(pieces, counts, strict=True)
        ]
        points = [np.repeat(q.mean(axis=0, keepdims=True), len(q), axis=0) for q in pairs_full]
        idx = tuple(p.index for p in pieces)
        closed = tuple(p.closed for p in pieces)
        if not sa.order:
            return Plan(tuple(points), tuple(pairs_full), closed, (-1,) * n, idx,
                        sa.left_out, sb.left_out)  # fmt: skip
        return Plan(tuple(pairs_full), tuple(points), closed, idx, (-1,) * n, sa.left_out,
                    sb.left_out)  # fmt: skip
    pa, pb = _pieces(sa, n), _pieces(sb, n)
    partner = _pair(sa, pa, sb, pb)
    counts = _within_total([max(_count(sa, p), _count(sb, pb[partner[k]]))
                            for k, p in enumerate(pa)])  # fmt: skip
    out_a, out_b, closed, src, dst = [], [], [], [], []
    for k, p in enumerate(pa):
        q = pb[partner[k]]
        both = p.closed and q.closed
        m = counts[k]
        ra = _resample(sa.polys[p.index], sa.along[p.index], p.s0, p.s1, m, endpoint=not both)
        rb = _resample(sb.polys[q.index], sb.along[q.index], q.s0, q.s1, m, endpoint=not both)
        ra, rb = _align(ra, rb, p.closed and not both, q.closed and not both, both)
        out_a.append(ra)
        out_b.append(rb)
        closed.append(both)
        src.append(p.index)
        dst.append(q.index)
    return Plan(tuple(out_a), tuple(out_b), tuple(closed), tuple(src), tuple(dst),
                sa.left_out, sb.left_out)  # fmt: skip


def _within_total(counts: list[int]) -> list[int]:
    """Points per piece scaled down, when needed, so all of them stay near ``MAX_TOTAL``
    (each keeps at least ``MIN_POINTS``)."""
    total = sum(counts)
    if total <= MAX_TOTAL:
        return counts
    return [max(MIN_POINTS, c * MAX_TOTAL // total) for c in counts]


def _count(side: _Side, p: Piece) -> int:
    """Points for one piece: as many as its dense polyline has along it, within bounds."""
    along = side.along[p.index]
    inside = int(np.searchsorted(along, p.s1) - np.searchsorted(along, p.s0)) + 2
    return int(np.clip(inside, MIN_POINTS, MAX_POINTS))


def at(p: Plan, m: float) -> list[tuple[np.ndarray, bool, int]]:
    """The pieces ``m`` (0..1) of the way: (polyline, closed, target subpath) each, with
    consecutive repeats dropped; a piece with fewer than two distinct points is left out."""
    out = []
    for a, b, closed, d in zip(p.a, p.b, p.closed, p.dst, strict=True):
        pts = a + (b - a) * float(m)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        step = np.abs(np.diff(pts, axis=0)).max(axis=1) if len(pts) > 1 else np.zeros(0)
        scale = max(1.0, float(np.abs(pts).max()))
        keep = np.r_[True, step > 1e-12 * scale]
        pts = pts[keep]
        if closed and len(pts) > 1 and not np.array_equal(pts[-1], pts[0]):
            pts = np.vstack([pts, pts[:1]])  # the closing point was a repeat of the last
        if len(pts) >= 2 and np.ptp(pts, axis=0).max() > 1e-12 * scale:
            out.append((pts, closed, d))
    return out


def subpath(poly: np.ndarray, closed: bool) -> np.ndarray:
    """A polyline as a subpath of straight cubics (closed: it ends where it began)."""
    return pm.polyline(poly, closed=closed)


# ---------------------------------------------------------------- shapes by name


def signature(subs: Sequence[np.ndarray]) -> str:
    """A name for a shape whatever its place and size: its control points, moved to their
    centre and scaled to a unit box, rounded to ``SIGNATURE_DIGITS`` decimals, hashed. Two
    copies of a glyph (or of any outline) at different places and sizes share it."""
    if not subs:
        return "empty"
    pts = np.vstack([np.asarray(s, dtype=np.float64) for s in subs])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = float((hi - lo).max())
    unit = (pts - (lo + hi) / 2) / (size if size > 0 else 1.0)
    h = hashlib.blake2b(digest_size=16)
    for s in subs:
        h.update(len(s).to_bytes(8, "little"))
    h.update((np.round(unit, SIGNATURE_DIGITS) + 0.0).tobytes())  # + 0.0: no negative zero
    return h.hexdigest()


def digest(subs: Sequence[np.ndarray]) -> bytes:
    """A key for exactly these subpaths (every byte), for caches."""
    h = hashlib.blake2b(digest_size=20)
    for s in subs:
        a = np.ascontiguousarray(s, dtype=np.float64)
        h.update(len(a).to_bytes(8, "little"))
        h.update(a.tobytes())
    return h.digest()
