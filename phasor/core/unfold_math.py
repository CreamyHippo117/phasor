"""Mirror unfolding: a ray bouncing in a room drawn as a straight line through mirror rooms.

A ray that reflects off a wall carries on straight into the mirror copy of the room across
that wall. Copy the room across every wall the ray reaches and the zig-zag becomes one
straight line through a row of rooms (the image-source picture of echoes). This module
works that out, pure NumPy and no drawing:

- ``box_unfold``: the walls a straight ray crosses, in order, through the lattice of mirror
  copies of a box (in 2D, a rectangle: a ray with no z never meets the floor or ceiling);
  for each, where and when the ray arrives and the hinge the copy turns about.
- ``hinge_matrix`` / ``chain_matrix``: a copy is the room carried through a chain of hinge
  turns (``Hinge``: a line and an angle). The last turn may be part done (``turn`` 0-1):
  that is the page turning.
- ``fold``, ``image_point``, ``zigzag``: from the unfolded frame back to the real room
  (the reflections the ray really makes) and out again (image sources).

A hinge is a turn about a line, never a reflection: a copy swings over like a page and its
handedness never flips, so a renderer can draw it as the room moved rigidly. For a flat
room (2D) half a turn about the wall's edge is exactly the reflection across it. For a box
(3D) the hinge runs through the middle of the wall: half a turn lands the box exactly on
its mirror copy (a box is symmetric about its middle), though what is inside comes out
turned rather than mirrored.

Other mirror unfoldings fit the same two pieces (a chain of ``Hinge`` per copy; a list of
crossings in time): a billiard in a polygon or a kaleidoscope (hinges on the polygon's
edges, half turns), the net of a cube and a geodesic on it (quarter turns about the edges
of its faces). Only the box is worked out here.

Layer: ``core``, so that both the ``unfold`` verb (``time``) and the ``HingedCopy`` object
(``domains``) can use it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EPS = 1e-9  # relative: crossings closer than this (times the room's size) happen together
MAX_FLIPS = 64
# Cell names: the direction of each axis, positive then negative (east/west, north/south,
# up/down: x right, y up the screen in 2D, z up).
AXIS_WORDS = (("e", "w"), ("n", "s"), ("u", "d"))
WALLS = (("west", "east"), ("south", "north"), ("floor", "ceiling"))  # (lo, hi) per axis


@dataclass(frozen=True)
class Hinge:
    """A turn of ``degrees`` about the line through ``point`` along ``axis`` (a unit vector;
    the turn is counterclockwise looking down the axis, the right-hand rule)."""

    point: tuple[float, float, float]
    axis: tuple[float, float, float]
    degrees: float = 180.0

    def as_params(self) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        return (self.point, self.axis, self.degrees)


@dataclass(frozen=True)
class Crossing:
    """The ray crossing from one copy of the room into the next."""

    distance: float  # along the ray from its start, in world units
    axis: int  # 0, 1, 2: the wall's normal axis
    side: int  # +1: the copy's high wall (east, north, ceiling); -1: its low wall
    cell_from: tuple[int, int, int]  # lattice index of the copy the ray leaves
    cell_to: tuple[int, int, int]
    point: tuple[float, float, float]  # where it crosses, unfolded (on the straight line)
    wall_point: tuple[float, float, float]  # the same point in the real room: the reflection
    wall: str  # the real wall it reflects off: west, east, south, north, floor, ceiling
    hinge: Hinge  # the turn that carries copy ``cell_from`` onto copy ``cell_to``


@dataclass(frozen=True)
class Unfolding:
    start: tuple[float, float, float]
    direction: tuple[float, float, float]  # unit
    crossings: tuple[Crossing, ...]
    end: float  # distance to where the ray leaves the last copy (its far wall)

    @property
    def end_point(self) -> np.ndarray:
        return np.asarray(self.start) + self.end * np.asarray(self.direction)

    def chains(self) -> list[tuple[Hinge, ...]]:
        """Each copy's chain of hinges, first crossing first (copy k: the first k + 1)."""
        out: list[tuple[Hinge, ...]] = []
        for c in self.crossings:
            out.append((*(out[-1] if out else ()), c.hinge))
        return out


def _box(lo, hi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(hi < lo):
        raise ValueError("a box needs lo <= hi, three numbers each")
    return lo, hi, hi - lo


def hinge_axis(axis: int, side: int) -> np.ndarray:
    """The direction of the hinge for a wall with normal ``axis`` on ``side``: along the wall
    and level (the y axis for an x wall, x for a y wall), signed so that half way through
    the turn the copy stands straight up out of the wall (towards +z, towards the viewer
    in 2D). The floor and ceiling turn about the x axis and swing towards -y."""
    u = np.zeros(3)
    if axis == 0:
        u[1] = side
    elif axis == 1:
        u[0] = -side
    else:
        u[0] = -side
    return u


def box_unfold(lo, hi, start, direction, flips: int) -> Unfolding:
    """The first ``flips`` walls a ray from ``start`` along ``direction`` crosses through
    the lattice of mirror copies of the box [lo, hi]. A box flat along an axis (a
    rectangle at z = c) is 2D: the ray must not move along that axis. When the ray meets
    a corner (two walls at once) both count, in axis order, at the same distance."""
    lo, hi, size = _box(lo, hi)
    p0 = np.asarray(start, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-12:
        raise ValueError("the ray needs a direction")
    d = d / np.linalg.norm(d)
    scale = float(size.max())
    if scale <= 0:
        raise ValueError("the room has no size")
    tol = EPS * scale
    moving = np.abs(d) > 1e-12
    if np.any(moving & (size <= tol)):
        raise ValueError("the ray leaves the plane of a flat room; give it no z")
    if np.any(p0 < lo - tol) or np.any(p0 > hi + tol):
        raise ValueError("the ray must start inside the room")
    if not 0 <= flips <= MAX_FLIPS:
        raise ValueError(f"flips must be 0 to {MAX_FLIPS}")
    cell = np.zeros(3, dtype=np.int64)

    def exits(c: np.ndarray) -> np.ndarray:
        """Distance to each axis's exit wall of copy ``c`` (inf where the ray is still)."""
        out = np.full(3, np.inf)
        wall = lo + (c + (d > 0)) * size
        out[moving] = (wall[moving] - p0[moving]) / d[moving]
        return out

    crossings: list[Crossing] = []
    while len(crossings) < flips:
        t = exits(cell)
        tmin = float(t.min())
        for ax in (int(a) for a in np.flatnonzero(t <= tmin + tol)):
            if len(crossings) == flips:
                break
            side = 1 if d[ax] > 0 else -1
            nxt = cell.copy()
            nxt[ax] += side
            q = p0 + tmin * d
            pivot = lo + (cell + 0.5) * size  # the middle of the copy being left
            pivot[ax] = lo[ax] + (cell[ax] + (side > 0)) * size[ax]  # on the wall
            hinge = Hinge(_tup(pivot), _tup(hinge_axis(ax, side)), 180.0)
            real = fold(lo, hi, q[None])[0]
            # Which real wall: the copy's wall folded back; a copy with an odd index on this
            # axis is mirrored, so its high wall is the room's low one.
            real_side = side if cell[ax] % 2 == 0 else -side
            crossings.append(Crossing(
                distance=tmin, axis=ax, side=side, cell_from=_itup(cell), cell_to=_itup(nxt),
                point=_tup(q), wall_point=_tup(real), wall=WALLS[ax][real_side > 0],
                hinge=hinge,
            ))  # fmt: skip
            cell = nxt
    end = float(exits(cell).min()) if moving.any() else 0.0
    return Unfolding(_tup(p0), _tup(d), tuple(crossings), end)


def _tup(v) -> tuple[float, float, float]:
    return tuple(float(x) for x in v)  # type: ignore[return-value]


def _itup(v) -> tuple[int, int, int]:
    return tuple(int(x) for x in v)  # type: ignore[return-value]


def hinge_matrix(point, axis, degrees: float) -> np.ndarray:
    """4x4: the turn of ``degrees`` about the line through ``point`` along ``axis``."""
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    a = np.radians(degrees)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    r = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)  # Rodrigues
    if abs(degrees % 180) < 1e-12:  # exact half and whole turns: no round-off
        r = np.round(r, 12)
    p = np.asarray(point, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = p - r @ p
    return m


def chain_matrix(hinges: Sequence, turn: float = 1.0) -> np.ndarray:
    """4x4: the hinges applied in order (the first one first), the last only ``turn`` of
    the way (0-1). ``hinges``: ``Hinge`` or (point, axis, degrees) triples."""
    m = np.eye(4)
    n = len(hinges)
    for k, h in enumerate(hinges):
        point, axis, degrees = h.as_params() if isinstance(h, Hinge) else h
        share = float(np.clip(turn, 0.0, 1.0)) if k == n - 1 else 1.0
        m = hinge_matrix(point, axis, degrees * share) @ m
    return m


def apply(m: np.ndarray, points) -> np.ndarray:
    """Points (..., 3) moved by the 4x4 ``m``."""
    p = np.asarray(points, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def fold(lo, hi, points) -> np.ndarray:
    """Points of the unfolded frame (N, 3) back in the real room: each axis folded like a
    mirror at every wall (a triangle wave). A flat axis stays as it is."""
    lo, hi, size = _box(lo, hi)
    p = np.array(points, dtype=np.float64, copy=True)
    for ax in range(3):
        if size[ax] <= 0:
            continue
        u = np.mod(p[..., ax] - lo[ax], 2 * size[ax])
        p[..., ax] = lo[ax] + np.where(u <= size[ax], u, 2 * size[ax] - u)
    return p


def image_point(lo, hi, point, cell) -> np.ndarray:
    """The image of ``point`` (in the room) in the mirror copy with lattice index ``cell``:
    an image source when ``point`` is the source."""
    lo, hi, size = _box(lo, hi)
    q = np.asarray(point, dtype=np.float64)
    c = np.asarray(cell, dtype=np.int64)
    odd = c % 2 == 1
    return np.where(odd, lo + (c + 1) * size - (q - lo), q + c * size)


def zigzag(u: Unfolding, lo, hi) -> np.ndarray:
    """The path the ray really takes in the room: (K + 2, 3), from the start through every
    reflection to where the straight line leaves the last copy, folded back."""
    pts = [u.start, *(c.point for c in u.crossings), tuple(u.end_point)]
    return fold(lo, hi, np.asarray(pts))


def cell_name(cell) -> str:
    """A copy's name from its place in the lattice: ``e2_n1`` is two rooms east and one
    north; the room itself is ``home``."""
    parts = [f"{AXIS_WORDS[ax][c < 0]}{abs(c)}" for ax, c in enumerate(cell) if c]
    return "_".join(parts) or "home"
