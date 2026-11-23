"""Vector path geometry (Milestone 16): pure NumPy, no drawing.

A *subpath* is an (N, 3) array with N = 1 + 3k: an anchor, then (handle, handle, anchor) for
each of its k cubic Béziers. It is closed when its last anchor is its first. A path is a list
of subpaths. Everything here works on that form or on its flattened polylines:

- building shapes (``circle``, ``arc``, ``rectangle``, ``brace``, ...) as subpaths;
- ``flatten``: cubic Béziers to polylines, adaptively, within a tolerance in world units
  (the caller turns a share of the frame height into one, so it follows size on screen);
- ``fill``: closed polylines to triangles under the even-odd or non-zero rule, holes
  included (mapbox-earcut);
- ``trim``, ``dashes`` and ``dots``: cut by arc length from the start of each polyline, so a
  dash stays where it is as the path grows;
- ``boolean``: union, intersection, difference and exclusion of filled regions (shapely, the
  ``boolean`` extra);
- ``read_svg``: the shapes of an SVG file as subpaths (svgelements, the ``svg`` extra).
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phasor.core import extras

KAPPA = 4.0 / 3.0 * (math.sqrt(2.0) - 1.0)  # cubic handle length of a quarter circle
MAX_PIECES = 256  # most segments one cubic flattens to
RULES = ("nonzero", "evenodd")
OPS = ("union", "intersection", "difference", "exclusion")
CROSSING_LIMIT = 20000  # segments: above this, a fill is not checked for crossings
CROSSING_BLOCK = 256  # segments compared with all others at once
SVG_MAX_BYTES = 8 * 1024 * 1024  # an SVG larger than this is not read


class MissingExtra(ImportError):
    """An optional dependency a feature needs is not installed."""

    def __init__(self, module: str, extra: str, feature: str) -> None:
        super().__init__(
            f"{feature} needs the {module!r} package: install the extra with "
            f"`uv sync --extra {extra}` (or `pip install phasor[{extra}]`)"
        )
        self.module, self.extra = module, extra


# ---------------------------------------------------------------- building subpaths


def _pt(p) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    return np.array([a[0], a[1], a[2] if len(a) > 2 else 0.0])


def line_cubic(a, b) -> np.ndarray:
    """The cubic of a straight segment: (3, 3) handle, handle, end (the start is ``a``)."""
    a, b = _pt(a), _pt(b)
    return np.array([a + (b - a) / 3, a + 2 * (b - a) / 3, b])


def polyline(points, closed: bool = False) -> np.ndarray:
    """A subpath of straight segments through ``points`` (closed: back to the first)."""
    pts = [_pt(p) for p in points]
    if closed and len(pts) > 1 and not np.allclose(pts[0], pts[-1]):
        pts.append(pts[0])
    out = [pts[0]]
    for a, b in itertools.pairwise(pts):
        out.extend(line_cubic(a, b))
    return np.array(out)


def arc(center, rx: float, start: float, sweep: float, ry: float | None = None,
        rotation: float = 0.0) -> np.ndarray:  # fmt: skip
    """An elliptic arc (radians) as a subpath, in pieces of at most a quarter turn."""
    c = _pt(center)
    ry = rx if ry is None else ry
    pieces = max(1, math.ceil(abs(sweep) / (math.pi / 2) - 1e-9))
    step = sweep / pieces
    k = 4.0 / 3.0 * math.tan(step / 4)
    cr, sr = math.cos(rotation), math.sin(rotation)

    def at(a: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        x, y = rx * (math.cos(a) + dx), ry * (math.sin(a) + dy)
        return c + np.array([x * cr - y * sr, x * sr + y * cr, 0.0])

    out = [at(start)]
    for i in range(pieces):
        a0, a1 = start + i * step, start + (i + 1) * step
        out.append(at(a0, -k * math.sin(a0), k * math.cos(a0)))
        out.append(at(a1, k * math.sin(a1), -k * math.cos(a1)))
        out.append(at(a1))
    return np.array(out)


def circle(center, r: float) -> np.ndarray:
    """A circle, counterclockwise from its rightmost point: four quarter cubics."""
    sub = arc(center, r, 0.0, 2 * math.pi)
    sub[-1] = sub[0]  # exactly closed
    return sub


def ellipse(center, rx: float, ry: float, rotation: float = 0.0) -> np.ndarray:
    sub = arc(center, rx, 0.0, 2 * math.pi, ry, rotation)
    sub[-1] = sub[0]
    return sub


def regular_vertices(n: int, r: float, rotation: float = 0.0, center=(0.0, 0.0, 0.0)):
    """(n, 3) vertices of a regular n-gon of circumradius ``r``: the first at the top, then
    counterclockwise; ``rotation`` in degrees, counterclockwise."""
    k = np.arange(n)
    ang = math.pi / 2 + 2 * math.pi * k / n + math.radians(rotation)
    c = _pt(center)
    return np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), np.full(n, c[2])]


def rectangle(center, width: float, height: float, radius: float = 0.0,
              rotation: float = 0.0) -> np.ndarray:  # fmt: skip
    """A rectangle, counterclockwise, its corners rounded by ``radius`` (at most half the
    shorter side); ``rotation`` in degrees."""
    c = _pt(center)
    w, h = width / 2, height / 2
    r = max(0.0, min(radius, w, h))
    t = math.radians(rotation)
    rot = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    if r <= 1e-12:
        sub = polyline([(w, -h), (w, h), (-w, h), (-w, -h)], closed=True)
    else:
        out = [np.array([w, -h + r, 0.0])]
        corners = [((w - r, h - r), 0.0), ((-w + r, h - r), 0.5), ((-w + r, -h + r), 1.0),
                   ((w - r, -h + r), 1.5)]  # fmt: skip
        for (cx, cy), quarter in corners:
            a0 = quarter * math.pi
            start = np.array([cx + r * math.cos(a0), cy + r * math.sin(a0), 0.0])
            if not np.allclose(start, out[-1]):
                out.extend(line_cubic(out[-1], start))
            out.extend(arc((cx, cy, 0.0), r, a0, math.pi / 2)[1:])
        if not np.allclose(out[-1], out[0]):
            out.extend(line_cubic(out[-1], out[0]))
        sub = np.array(out)
    sub = sub @ rot.T + c
    sub[-1] = sub[0]
    return sub


def sector(center, r: float, start: float, sweep: float, inner: float = 0.0) -> np.ndarray:
    """A pie slice (``inner`` 0) or a slice of a ring, closed; angles in radians."""
    c = _pt(center)
    outer = arc(c, r, start, sweep)
    if inner <= 0:
        out = [c, *line_cubic(c, outer[0]), *outer[1:], *line_cubic(outer[-1], c)]
    else:
        back = arc(c, inner, start + sweep, -sweep)
        out = [*outer, *line_cubic(outer[-1], back[0]), *back[1:], *line_cubic(back[-1], outer[0])]
    sub = np.array(out)
    sub[-1] = sub[0]
    return sub


def brace(a, b, depth: float) -> np.ndarray:
    """A curly brace from ``a`` to ``b`` (open), its tip ``depth`` out to the right of the
    direction a -> b (a brace under a term: a on the left, b on the right)."""
    a, b = _pt(a), _pt(b)
    d = b - a
    length = float(np.linalg.norm(d[:2]))
    if length < 1e-12:
        return polyline([a, b])
    u = d / length
    n = np.array([u[1], -u[0], 0.0])  # right of a -> b
    h = min(depth / 2, length / 4)
    k = KAPPA * h
    m = (a + b) / 2
    s1 = a + u * h + n * h
    s2 = m - u * h + n * h
    tip = m + n * (2 * h)
    s3 = m + u * h + n * h
    s4 = b - u * h + n * h
    out = [a, a + n * k, s1 - u * k, s1, *line_cubic(s1, s2), s2 + u * k, tip - n * k, tip,
           tip - n * k, s3 - u * k, s3, *line_cubic(s3, s4), s4 + u * k, b + n * k, b]  # fmt: skip
    return np.array(out)


# ---------------------------------------------------------------- checking and measuring


def check_subpath(sub) -> np.ndarray:
    """``sub`` as an (N, 3) float array with N = 1 + 3k, k >= 1, or ``ValueError``."""
    a = np.asarray(sub, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("a subpath is a list of points (x, y, z)")
    if len(a) < 4 or (len(a) - 1) % 3:
        raise ValueError(
            f"a subpath has 1 + 3k points (an anchor, then handle, handle, anchor per cubic); "
            f"got {len(a)}"
        )
    if not np.all(np.isfinite(a)):
        raise ValueError("a subpath's points must be finite")
    return a


def is_closed(sub: np.ndarray, eps: float = 1e-9) -> bool:
    scale = max(1.0, float(np.abs(sub).max()))
    return bool(np.linalg.norm(sub[-1] - sub[0]) <= eps * scale)


def cubics(sub: np.ndarray) -> np.ndarray:
    """(k, 4, 3): the control points of each cubic of a subpath."""
    k = (len(sub) - 1) // 3
    idx = 3 * np.arange(k)[:, None] + np.arange(4)[None, :]
    return sub[idx]


def flatten(sub: np.ndarray, tol: float) -> np.ndarray:
    """A polyline within ``tol`` (world units) of the subpath's cubics. Each cubic gets
    ceil(sqrt(3/4 * M / tol)) segments (Wang's formula; M is the largest second difference
    of its control points), so straight pieces stay one segment and tight curves get more."""
    c = cubics(sub)
    p0, p1, p2, p3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    m = np.maximum(np.linalg.norm(p0 - 2 * p1 + p2, axis=1),
                   np.linalg.norm(p1 - 2 * p2 + p3, axis=1))  # fmt: skip
    n = np.ceil(np.sqrt(0.75 * m / max(tol, 1e-12)))
    n = np.clip(n, 1, MAX_PIECES).astype(int)
    out = [sub[:1]]
    for i in range(len(c)):
        u = (np.arange(1, n[i] + 1) / n[i])[:, None]
        v = 1 - u
        out.append(v**3 * p0[i] + 3 * v * v * u * p1[i] + 3 * v * u * u * p2[i] + u**3 * p3[i])
    pts = np.vstack(out)
    if is_closed(sub):
        pts[-1] = pts[0]
    # A cubic of zero length (a corner arc ending where the next piece starts) adds a point
    # on top of the last one; a stroke cannot miter a joint with no direction.
    step = np.abs(np.diff(pts, axis=0)).max(axis=1)
    keep = np.r_[True, step > 1e-12 * max(1.0, float(np.abs(pts).max()))]
    if not keep.all():
        last = pts[-1].copy()
        pts = pts[keep]
        pts[-1] = last if len(pts) > 1 else pts[-1]
    return pts


def size_of(subs: Sequence[np.ndarray]) -> float:
    """The diagonal of the box around every control point (0 for nothing)."""
    if not subs:
        return 0.0
    pts = np.vstack(subs)
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def arc_lengths(poly: np.ndarray) -> np.ndarray:
    """Distance along a polyline at each of its points, from 0."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def length(poly: np.ndarray) -> float:
    return float(arc_lengths(poly)[-1]) if len(poly) > 1 else 0.0


def _cut(poly: np.ndarray, along: np.ndarray, a: float, b: float) -> np.ndarray:
    """The piece of a polyline between arc lengths ``a`` < ``b``."""
    i0 = int(np.searchsorted(along, a, side="right"))
    i1 = int(np.searchsorted(along, b, side="left"))
    pts = [np.array([np.interp(a, along, poly[:, k]) for k in range(3)])]
    pts.extend(poly[i0:i1])
    pts.append(np.array([np.interp(b, along, poly[:, k]) for k in range(3)]))
    return np.array(pts)


def trim(polys: Sequence[np.ndarray], share: float) -> list[np.ndarray]:
    """The first ``share`` of a path's length, subpath after subpath (the ``trace`` verb)."""
    if share >= 1.0:
        return list(polys)
    total = sum(length(p) for p in polys)
    left = max(share, 0.0) * total
    out = []
    for p in polys:
        if left <= 0:
            break
        along = arc_lengths(p)
        if along[-1] <= left:
            out.append(p)
        elif left > 0:
            out.append(_cut(p, along, 0.0, left))
        left -= along[-1]
    return out


def piece(poly: np.ndarray, a: float, b: float) -> np.ndarray:
    """The part of a polyline between arc lengths ``a`` and ``b`` (clamped to it)."""
    along = arc_lengths(poly)
    a, b = max(a, 0.0), min(b, along[-1])
    if a <= 0 and b >= along[-1]:
        return poly
    return _cut(poly, along, a, max(b, a))


def dashes(poly: np.ndarray, period: float, on: float = 0.55, upto: float | None = None,
           start: float = 0.0, limit: int | None = None) -> list[np.ndarray]:  # fmt: skip
    """The dashes of a polyline: pieces ``on * period`` long every ``period``, measured by arc
    length from its start, so a dash never moves when the polyline grows past it. Only what
    lies between arc lengths ``start`` and ``upto`` is kept (a path being traced, the room
    an arrow tip takes), and at most ``limit`` dashes (the first ones)."""
    along = arc_lengths(poly)
    end = along[-1] if upto is None else min(upto, along[-1])
    if not (math.isfinite(end) and math.isfinite(period)):
        return []
    if period <= 0:
        return [_cut(poly, along, start, end)] if end > start else []
    out = []
    first = max(0, math.floor(max(start, 0.0) / period))
    last = math.ceil(end / period)
    if limit is not None:
        last = min(last, first + limit)
    for k in range(first, last):
        a, b = max(k * period, start), min(k * period + on * period, end)
        if b > a + 1e-12:
            out.append(_cut(poly, along, a, b))
    return out


def dots(poly: np.ndarray, period: float, upto: float | None = None,
         start: float = 0.0, limit: int | None = None) -> np.ndarray:  # fmt: skip
    """(n, 3) points every ``period`` of arc length from the start (a dotted stroke), those
    between ``start`` and ``upto``, at most ``limit`` of them (the first ones)."""
    along = arc_lengths(poly)
    end = along[-1] if upto is None else min(upto, along[-1])
    if period <= 0 or len(poly) < 2 or not (math.isfinite(end) and math.isfinite(period)):
        return np.zeros((0, 3))
    first = max(0, math.ceil(max(start, 0.0) / period - 1e-9))
    count = math.floor(end / period + 1e-9) + 1 - first
    if limit is not None:
        count = min(count, limit + 1)
    s = (first + np.arange(max(count, 0))) * period
    if is_closed(poly) and len(s) > 1 and along[-1] - s[-1] < 0.5 * period:
        s = s[:-1]  # a closed loop: no dot on top of the first one
    s = s[(s >= start - 1e-12) & (s <= end + 1e-12)]
    return np.stack([np.interp(s, along, poly[:, k]) for k in range(3)], axis=1)


# ---------------------------------------------------------------- the plane a path lies in


def plane_of(points: np.ndarray):
    """(origin, e1, e2): the plane of a flat point set (Newell's normal); the world xy plane
    when the points are on it or do not span a plane."""
    p = np.asarray(points, dtype=np.float64)
    o = p.mean(axis=0)
    q, r = p - o, np.roll(p - o, -1, axis=0)
    n = np.array([np.sum((q[:, 1] - r[:, 1]) * (q[:, 2] + r[:, 2])),
                  np.sum((q[:, 2] - r[:, 2]) * (q[:, 0] + r[:, 0])),
                  np.sum((q[:, 0] - r[:, 0]) * (q[:, 1] + r[:, 1]))])  # fmt: skip
    if np.ptp(p[:, 2]) <= 1e-12 * max(1.0, float(np.abs(p).max())) or np.linalg.norm(n) < 1e-15:
        return o, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n /= np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    # e1 level (no z) when the plane is tilted, else the world x axis
    e1 = np.cross([0.0, 0.0, 1.0], n) if abs(n[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return o, e1, e2


def signed_area(ring: np.ndarray) -> float:
    """Shoelace area of a 2D ring (positive: counterclockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _open_ring(ring: np.ndarray) -> np.ndarray:
    """A closed polyline without its repeated last point."""
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        return ring[:-1]
    return ring


def inside(point: np.ndarray, ring: np.ndarray) -> bool:
    """Whether a 2D point is inside a 2D ring (even-odd ray casting)."""
    x, y = point
    xi, yi = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cross = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = (xj - xi) * (y - yi) / (yj - yi) + xi
    return bool(np.count_nonzero(cross & (x < xs)) % 2)


def _probe(ring: np.ndarray) -> np.ndarray:
    """A point just inside a ring, near its first edge (for nesting tests)."""
    a, b = ring[0], ring[1]
    mid = (a + b) / 2
    d = b - a
    nrm = np.array([-d[1], d[0]])
    nl = np.linalg.norm(nrm)
    if nl < 1e-15:
        return mid
    step = 1e-7 * max(1.0, float(np.ptp(ring, axis=0).max()))
    sign = 1.0 if signed_area(ring) > 0 else -1.0
    return mid + sign * nrm / nl * step


@dataclass(frozen=True)
class Region:
    """A filled region in its plane: outer rings, each with its holes (2D, open rings)."""

    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    shapes: tuple[tuple[np.ndarray, tuple[np.ndarray, ...]], ...]

    def to_world(self, pts2: np.ndarray) -> np.ndarray:
        return self.origin + pts2[:, :1] * self.e1 + pts2[:, 1:2] * self.e2

    @property
    def area(self) -> float:
        return float(sum(abs(signed_area(o)) - sum(abs(signed_area(h)) for h in hs)
                         for o, hs in self.shapes))  # fmt: skip


def region(polys: Sequence[np.ndarray], rule: str = "nonzero") -> Region:
    """The region closed polylines fill under ``rule``: every polyline is closed back to its
    start; each ring is an outer edge, a hole, or (where both sides are filled alike) no
    edge at all. Rings are assumed not to cross one another (lint ``fill_crossing``)."""
    if rule not in RULES:
        raise ValueError(f"fill rule must be one of {RULES}, got {rule!r}")
    flat = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 3]
    if not flat:
        z = np.zeros(3)
        return Region(z, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), ())
    o, e1, e2 = plane_of(np.vstack(flat))
    rings = []
    for p in flat:
        q = p - o
        ring = _open_ring(np.c_[q @ e1, q @ e2])
        if len(ring) >= 3 and abs(signed_area(ring)) > 1e-14:
            rings.append(ring)
    n = len(rings)
    orient = [1 if signed_area(r) > 0 else -1 for r in rings]
    probes = [_probe(r) for r in rings]
    contains = [[j != i and inside(probes[i], rings[j]) for j in range(n)] for i in range(n)]
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
    holes: dict[int, list[np.ndarray]] = {i: [] for i in outers}
    for i in range(n):
        if filled[i] != "hole":
            continue
        around = [j for j in outers if contains[i][j]]
        if around:
            parent = min(around, key=lambda j: abs(signed_area(rings[j])))
            holes[parent].append(rings[i])
    shapes = tuple((rings[i], tuple(holes[i])) for i in outers)
    return Region(o, e1, e2, shapes)


def fill(polys: Sequence[np.ndarray], rule: str = "nonzero"):
    """(positions (P, 3), triangles (T, 3)): the region the closed polylines fill under
    ``rule``, triangulated by earcut."""
    import mapbox_earcut as earcut

    reg = region(polys, rule)
    pos, tris, base = [], [], 0
    for outer, hs in reg.shapes:
        rings = [outer, *hs]
        verts = np.vstack(rings).astype(np.float64)
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        idx = earcut.triangulate_float64(verts, ends).reshape(-1, 3).astype(np.int64)
        pos.append(reg.to_world(verts))
        tris.append(idx + base)
        base += len(verts)
    if not pos:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(pos), np.vstack(tris)


def crossings(polys: Sequence[np.ndarray]) -> bool | None:
    """Whether any edges of the closed polylines cross (a ring crossing itself or another):
    then earcut's fill may be wrong. None (not checked) above ``CROSSING_LIMIT`` edges
    (lint ``fill_unchecked``)."""
    flat = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 3]
    if not flat:
        return False
    o, e1, e2 = plane_of(np.vstack(flat))
    segs, ring_id, idx = [], [], []
    for k, p in enumerate(flat):
        q = p - o
        r = _open_ring(np.c_[q @ e1, q @ e2])
        a, b = r, np.roll(r, -1, axis=0)
        segs.append(np.stack([a, b], axis=1))
        ring_id.append(np.full(len(r), k))
        idx.append(np.arange(len(r)))
    s = np.vstack(segs)
    if len(s) > CROSSING_LIMIT:
        return None
    if len(s) < 2:
        return False
    rid, ix = np.concatenate(ring_id), np.concatenate(idx)
    sizes = np.array([len(x) for x in idx])
    a, b = s[:, 0], s[:, 1]

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (
            q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])  # fmt: skip

    scale = max(1e-300, float(np.abs(s).max()) ** 2) * 1e-12
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    n_all = sizes[rid]
    for start in range(0, len(s), CROSSING_BLOCK):  # rows in blocks: memory stays small
        rows = slice(start, start + CROSSING_BLOCK)
        k = np.arange(len(s))[rows]
        # pairs (i, j), j > i, whose boxes meet
        near = (lo[rows, None, 0] <= hi[None, :, 0]) & (lo[None, :, 0] <= hi[rows, None, 0])
        near &= (lo[rows, None, 1] <= hi[None, :, 1]) & (lo[None, :, 1] <= hi[rows, None, 1])
        near &= np.arange(len(s))[None, :] > k[:, None]
        i, j = np.nonzero(near)
        if not len(i):
            continue
        i = i + start
        o1, o2 = orient(a[i], b[i], a[j]), orient(a[i], b[i], b[j])
        o3, o4 = orient(a[j], b[j], a[i]), orient(a[j], b[j], b[i])
        proper = (o1 * o2 < -scale) & (o3 * o4 < -scale)
        gap = np.abs(ix[i] - ix[j])
        neighbours = (rid[i] == rid[j]) & ((gap <= 1) | (gap == n_all[i] - 1))
        if np.any(proper & ~neighbours):
            return True
    return False


# ---------------------------------------------------------------- boolean operations


def boolean(op: str, a: Sequence[np.ndarray], b: Sequence[np.ndarray], rule_a: str = "nonzero",
            rule_b: str = "nonzero") -> list[np.ndarray]:  # fmt: skip
    """The region ``a op b`` (each a list of closed polylines filled under its rule) as
    closed polylines (outer rings counterclockwise, holes clockwise), in the plane of ``a``.
    Needs shapely (the ``boolean`` extra)."""
    if op not in OPS:
        raise ValueError(f"boolean op must be one of {OPS}, got {op!r}")
    if not extras.available("shapely"):
        raise MissingExtra("shapely", "boolean", f"shapes.{op}")
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    ra = region(a, rule_a)
    pts_b = [np.asarray(p, dtype=np.float64) for p in b]

    def geom(reg: Region, frame: Region):
        polys = []
        for outer, hs in reg.shapes:
            to2 = [_project(reg.to_world(r), frame) for r in (outer, *hs)]
            polys.append(Polygon(to2[0], to2[1:]))
        return areas(shapely.make_valid(MultiPolygon(polys))) if polys else Polygon()

    def areas(g):
        """Only the parts with an area: making a ring with none valid gives a line, and a
        collection of lines and polygons is one GEOS cannot always overlay."""
        keep = [x for x in shapely.get_parts(g) if isinstance(x, Polygon | MultiPolygon)]
        return shapely.union_all(keep) if keep else Polygon()

    rb = region(pts_b, rule_b)
    do = {
        "union": shapely.union,
        "intersection": shapely.intersection,
        "difference": shapely.difference,
        "exclusion": shapely.symmetric_difference,
    }[op]
    try:
        ga, gb = geom(ra, ra), geom(rb, ra)
        try:
            out = do(ga, gb)
        except shapely.errors.GEOSException:
            # Degenerate input of mixed scales (a ring with no area beside a tiny one):
            # snap both to a grid far below what shows, make them valid, try again.
            box = np.array(shapely.bounds(shapely.union(ga.envelope, gb.envelope)))
            grid = max(float(np.nanmax(np.abs(box))) if np.isfinite(box).any() else 1.0,
                       1e-300) * 1e-9  # fmt: skip
            snap = [areas(shapely.make_valid(shapely.set_precision(g, grid))) for g in (ga, gb)]
            out = do(*snap)
        out = shapely.normalize(out)
    except shapely.errors.GEOSException as e:
        raise ValueError(f"shapes.{op}: the shapes are too degenerate for a boolean "
                         f"operation (GEOS: {e})") from None  # fmt: skip
    rings = []
    for poly in getattr(out, "geoms", [out]):
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        poly = shapely.geometry.polygon.orient(poly, 1.0)
        for k, ring in enumerate([poly.exterior, *poly.interiors]):
            xy = np.asarray(ring.coords, dtype=np.float64)
            r = ra.to_world(xy)
            r[-1] = r[0]
            if (k == 0) == (signed_area(xy) < 0):
                r = r[::-1]
            rings.append(r)
    return rings


def _project(pts3: np.ndarray, frame: Region) -> np.ndarray:
    q = pts3 - frame.origin
    return np.c_[q @ frame.e1, q @ frame.e2]


# ---------------------------------------------------------------- SVG
#
# An SVG file sits in the scene's folder and may be hostile (Milestone 16 audit). It is read
# within these limits, checked while it is read, so a file of a few bytes can never expand
# into millions of shapes; past one, the drawing is empty or cut short (lint
# ``svg_too_complex``), never an error.

SVG_MAX_ELEMENTS = 50_000  # elements, once every <use> reference is expanded
SVG_MAX_DEPTH = 100  # elements (and <use> references) nested inside one another
SVG_MAX_SHAPES = 10_000  # shapes drawn; the rest are left out
SVG_MAX_SEGMENTS = 200_000  # cubic segments in all the shapes; the rest are left out
SVG_MAX_COORD = 1e12  # file units: a shape with a point farther out (or not finite) is skipped
_XLINK = "{http://www.w3.org/1999/xlink}href"
_UNIT_BOX = (0.0, 0.0, 1.0, 1.0)


class _OverBudget(Exception):
    """A drawing has more segments than ``SVG_MAX_SEGMENTS``."""


@dataclass(frozen=True)
class SvgShape:
    """One shape of an SVG file, in the file's own units (y down)."""

    id: str  # its ``id`` attribute, or "" when it has none
    tag: str  # circle, rect, path, ...
    subpaths: tuple[np.ndarray, ...]
    rule: str
    filled: bool
    stroked: bool


@dataclass(frozen=True)
class SvgDrawing:
    box: tuple[float, float, float, float]  # x, y, width, height of the view box (finite, > 0)
    shapes: tuple[SvgShape, ...]
    problem: str = ""  # why the file cannot be read (nothing drawn): lint ``bad_file``
    limited: str = ""  # which limit it reached (cut short or empty): lint ``svg_too_complex``
    skipped: int = 0  # shapes left out: points not finite or beyond SVG_MAX_COORD


def read_svg(path: str, stamp: tuple) -> SvgDrawing:
    """The shapes of the SVG file at ``path`` (cached by ``stamp``: path, mtime and size, so
    a changed file is read again; a bad file is remembered as bad). Never raises for what is
    in the file: a file that cannot be read has a ``problem``, one past a limit is ``limited``.
    Text, images and external references are ignored. Needs svgelements (the ``svg`` extra)."""
    return _read_svg(path, stamp)


@functools.lru_cache(maxsize=32)
def _read_svg(path: str, stamp: tuple) -> SvgDrawing:
    if not extras.available("svgelements"):
        raise MissingExtra("svgelements", "svg", "SVG")
    from phasor.core import files

    try:
        data = files.read_bytes(path, SVG_MAX_BYTES)
    except files.TooLarge:
        return SvgDrawing(_UNIT_BOX, (), limited=f"larger than {SVG_MAX_BYTES // 2**20} MB")
    except ValueError as e:
        return SvgDrawing(_UNIT_BOX, (), problem=str(e))
    try:
        why = _expansion_limit(data)
    except Exception as e:  # not XML at all
        return SvgDrawing(_UNIT_BOX, (), problem=f"not a valid SVG file ({e})")
    if why:
        return SvgDrawing(_UNIT_BOX, (), limited=why)
    try:
        return _svg_drawing(data)
    except RecursionError:
        return SvgDrawing(_UNIT_BOX, (), limited=f"nested more than {SVG_MAX_DEPTH} deep")
    except Exception as e:  # svgelements could not make sense of it
        return SvgDrawing(_UNIT_BOX, (), problem=f"not a valid SVG file ({type(e).__name__}: "
                                                 f"{e})")  # fmt: skip


def _expansion_limit(data: bytes) -> str:
    """Why the file is too large once its ``<use>`` references are expanded (as svgelements
    expands them: a ``<use>`` holds a copy of the element its ``href``, else its
    ``xlink:href``, names; the last element with an id wins), or "" when it is within
    ``SVG_MAX_ELEMENTS`` and ``SVG_MAX_DEPTH``. Counted on the file's tree without expanding
    anything, so a few bytes that would expand a million times are caught at once."""
    import io
    from xml.etree.ElementTree import iterparse

    kids: list[list[int]] = []
    ref: list[str | None] = []
    ids: dict[str, int] = {}
    stack: list[int] = []
    for event, el in iterparse(io.BytesIO(data), events=("start", "end")):
        if event == "end":
            stack.pop()
            continue
        i = len(kids)
        if i >= SVG_MAX_ELEMENTS:
            return f"more than {SVG_MAX_ELEMENTS} elements"
        if len(stack) >= SVG_MAX_DEPTH:
            return f"elements nested more than {SVG_MAX_DEPTH} deep"
        a = el.attrib
        url = a.get("href", a.get(_XLINK))
        ref.append(url[1:] if el.tag.rsplit("}", 1)[-1] == "use" and url else None)
        kids.append([])
        if "id" in a:
            ids[a["id"]] = i
        if stack:
            kids[stack[-1]].append(i)
        stack.append(i)
    if not kids:
        return ""
    deps = [k + ([ids[r]] if r in ids else []) for k, r in zip(kids, ref, strict=True)]
    size, depth, seen = [0] * len(kids), [0] * len(kids), [0] * len(kids)  # 1 open, 2 done
    todo = [0]
    while todo:  # children (and what a <use> names) before their parent, iteratively
        i = todo[-1]
        if seen[i] == 0:
            seen[i] = 1
            for j in deps[i]:
                if seen[j] == 1:
                    return "its <use> references loop"
                if seen[j] == 0:
                    todo.append(j)
            continue
        todo.pop()
        if seen[i] == 2:
            continue
        seen[i] = 2
        size[i] = min(1 + sum(size[j] for j in deps[i]), SVG_MAX_ELEMENTS + 1)
        depth[i] = 1 + max((depth[j] for j in deps[i]), default=0)
    if size[0] > SVG_MAX_ELEMENTS:
        return f"its <use> references expand it to more than {SVG_MAX_ELEMENTS} elements"
    if depth[0] > SVG_MAX_DEPTH:
        return f"its <use> references nest more than {SVG_MAX_DEPTH} deep"
    return ""


def _svg_drawing(data: bytes) -> SvgDrawing:
    import io

    import svgelements as se

    doc = se.SVG.parse(io.BytesIO(data))
    shapes: list[SvgShape] = []
    segments, skipped, limited = 0, 0, ""
    for el in doc.elements():
        if not isinstance(el, se.Shape) or isinstance(el, se.Group):
            continue
        if el.values.get("visibility") == "hidden" or el.values.get("display") == "none":
            continue
        if len(shapes) >= SVG_MAX_SHAPES:
            limited = f"more than {SVG_MAX_SHAPES} shapes: the rest are left out"
            break
        try:
            subs = _svg_subpaths(se, se.Path(el), SVG_MAX_SEGMENTS - segments)
        except _OverBudget:
            limited = f"more than {SVG_MAX_SEGMENTS} segments: the rest are left out"
            break
        except Exception:  # a shape svgelements cannot turn into a path
            skipped += 1
            continue
        good = [s for s in subs if np.isfinite(s).all() and np.abs(s).max() <= SVG_MAX_COORD]
        if len(good) < len(subs):
            skipped += 1
            continue
        if not good:
            continue
        segments += sum((len(s) - 1) // 3 for s in good)
        fill = el.values.get("fill", "black")
        stroke = el.values.get("stroke", "none")
        rule = "evenodd" if el.values.get("fill-rule") == "evenodd" else "nonzero"
        tag = type(el).__name__.lower()
        filled, stroked = str(fill).lower() != "none", str(stroke).lower() != "none"
        shapes.append(SvgShape(str(el.id or ""), tag, tuple(good), rule, filled, stroked))
    return SvgDrawing(_view_box(doc, shapes), tuple(shapes), limited=limited, skipped=skipped)


def _view_box(doc, shapes: Sequence[SvgShape]) -> tuple[float, float, float, float]:
    """The file's view box (or its width and height); the box around its shapes when that
    is missing, not finite, empty or absurd."""
    try:
        vb = doc.viewbox
        box = ((vb.x, vb.y, vb.width, vb.height) if vb is not None
               else (0.0, 0.0, doc.width or 1.0, doc.height or 1.0))  # fmt: skip
        box = tuple(float(v) for v in box)
    except (TypeError, ValueError, AttributeError):
        box = (math.nan,) * 4
    if (all(math.isfinite(v) and abs(v) <= SVG_MAX_COORD for v in box)
            and box[2] > 0 and box[3] > 0):  # fmt: skip
        return box  # type: ignore[return-value]
    if not shapes:
        return _UNIT_BOX
    pts = np.vstack([s for sh in shapes for s in sh.subpaths])
    lo, size = pts.min(axis=0), np.ptp(pts, axis=0)
    w, h = float(size[0]), float(size[1])
    side = max(w, h) if max(w, h) > 0 else 1.0
    return (float(lo[0]), float(lo[1]), w if w > 0 else side, h if h > 0 else side)


def _svg_subpaths(se, path, budget: int) -> list[np.ndarray]:
    """The subpaths of an svgelements path; ``_OverBudget`` past ``budget`` cubics."""
    subs: list[list[np.ndarray]] = []
    cur: list[np.ndarray] = []
    count = 0

    def p(c) -> np.ndarray:
        return np.array([float(c.x), float(c.y), 0.0])

    def done() -> None:
        if len(cur) >= 4:
            subs.append(cur.copy())

    for seg in path.segments():
        if count > budget:
            raise _OverBudget
        if isinstance(seg, se.Move):
            done()
            cur = [p(seg.end)]
        elif not cur:
            continue
        elif isinstance(seg, se.Close | se.Line):
            end = p(seg.end) if seg.end is not None else cur[0]
            if not np.allclose(end, cur[-1]):
                cur.extend(line_cubic(cur[-1], end))
                count += 1
        elif isinstance(seg, se.CubicBezier):
            cur.extend([p(seg.control1), p(seg.control2), p(seg.end)])
            count += 1
        elif isinstance(seg, se.QuadraticBezier):
            q0, q1, q2 = cur[-1], p(seg.control), p(seg.end)
            cur.extend([q0 + 2 / 3 * (q1 - q0), q2 + 2 / 3 * (q1 - q2), q2])
            count += 1
        elif isinstance(seg, se.Arc):
            for cb in seg.as_cubic_curves():
                cur.extend([p(cb.control1), p(cb.control2), p(cb.end)])
                count += 1
    if count > budget:
        raise _OverBudget
    done()
    return [np.array(s) for s in subs]
