"""Shared drawing helpers for domain objects: curves, discs and circles that keep their width
on screen, so every domain draws strokes the same way the built-in ``Line`` does."""

from __future__ import annotations

import numpy as np

from phasor.core.objects import DrawContext
from phasor.core.spatial import Camera

DISC_SEGMENTS = 48


def polyline(ctx: DrawContext, cam: Camera, pts, width, *, color, glow: float = 1.0,
             colors=None) -> None:  # fmt: skip
    """A stroked polyline (mitered joins). ``width`` is in world units: a number, or one per
    point (perspective: wider where it is further away, so it looks constant on screen).
    ``colors`` (N, 4), one per point, runs a color along the curve."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return
    width = np.broadcast_to(np.asarray(width, dtype=np.float64), (len(pts),))
    flat = np.c_[pts @ cam.right, pts @ cam.up]  # coordinates in the camera plane
    seg = np.diff(flat, axis=0)
    norm = np.linalg.norm(seg, axis=1, keepdims=True)
    seg = seg / np.where(norm == 0, 1, norm)
    normals = np.c_[-seg[:, 1], seg[:, 0]]
    n_prev = np.vstack([normals[:1], normals])
    n_next = np.vstack([normals, normals[-1:]])
    miter = n_prev + n_next
    miter /= np.maximum(np.linalg.norm(miter, axis=1, keepdims=True), 1e-9)
    scale = 1 / np.maximum((miter * n_prev).sum(1), 0.25)  # miter limit 4x
    offset = (miter * scale[:, None]) * (width[:, None] / 2)
    off3 = np.outer(offset[:, 0], cam.right) + np.outer(offset[:, 1], cam.up)
    positions = np.empty((2 * len(pts), 3))
    positions[0::2], positions[1::2] = pts + off3, pts - off3
    k = 2 * np.arange(len(pts) - 1)
    indices = np.vstack([np.c_[k, k + 1, k + 2], np.c_[k + 1, k + 3, k + 2]])
    per_vertex = None
    if colors is not None:
        per_vertex = np.repeat(np.asarray(colors, dtype=np.float64), 2, axis=0)
    ctx.triangles(positions, indices, color=color, glow=glow, colors=per_vertex)


def segments(ctx: DrawContext, cam: Camera, pairs, width, *, color, glow: float = 1.0) -> None:
    """Many separate straight strokes: ``pairs`` is (N, 2, 3)."""
    for a, b in np.asarray(pairs, dtype=np.float64):
        w = float(np.mean(cam.units_per_height(np.array([a, b])))) * width
        polyline(ctx, cam, [a, b], w, color=color, glow=glow)


def disc(ctx: DrawContext, cam: Camera, center, radius: float, *, color, glow: float = 1.0) -> None:
    """A filled disc facing the camera."""
    a = np.linspace(0, 2 * np.pi, DISC_SEGMENTS, endpoint=False)
    c = np.asarray(center, dtype=np.float64)
    rim = c + radius * (np.outer(np.cos(a), cam.right) + np.outer(np.sin(a), cam.up))
    positions = np.vstack([c[None, :], rim])
    k = np.arange(DISC_SEGMENTS)
    ctx.triangles(positions, np.c_[np.zeros_like(k), 1 + k, 1 + (k + 1) % DISC_SEGMENTS],
                  color=color, glow=glow)  # fmt: skip


def circle(ctx: DrawContext, cam: Camera, center, radius: float, width, *, color, glow=1.0) -> None:
    """A ring facing the camera, ``width`` in world units."""
    a = np.linspace(0, 2 * np.pi, DISC_SEGMENTS + 1)
    c = np.asarray(center, dtype=np.float64)
    pts = c + radius * (np.outer(np.cos(a), cam.right) + np.outer(np.sin(a), cam.up))
    polyline(ctx, cam, pts, width, color=color, glow=glow)


def arrow(ctx: DrawContext, cam: Camera, a, b, width, color, glow: float = 1.0,
          head: float | None = None) -> None:  # fmt: skip
    """A straight arrow from ``a`` to ``b``, ``width`` in world units. The head is scaled to
    the width, or is ``head`` (a share of the arrow's length) when given."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return
    head = length * head if head is not None else min(length * 0.3, width * 7)
    unit = d / length
    side = np.cross(cam.forward, unit)
    side /= max(np.linalg.norm(side), 1e-9)
    base = b - unit * head
    polyline(ctx, cam, [a, base], width, color=color, glow=glow)
    tri = np.array([b, base + side * head * 0.45, base - side * head * 0.45])
    ctx.triangles(tri, np.array([[0, 1, 2]]), color=color, glow=glow)


LENGTH_BY = ("size", "soft")


def field_arrows(ctx: DrawContext, cam: Camera, centers, vectors, spacing: float, *, width,
                 colors, glow: float = 1.0, length_by: str = "size") -> None:  # fmt: skip
    """A field of arrows centred on ``centers`` (N, 3), pointing along ``vectors`` (N, 3),
    for grids ``spacing`` apart. ``colors``: one RGBA, or one per arrow (N, 4).

    ``length_by="size"``: length in proportion to size, the strongest 0.85 of the spacing.
    ``length_by="soft"``: lengths follow tanh of size over a typical size, so weak arrows
    still show and strong ones never reach their neighbours (the dense look: many small
    arrows, where color says how strong)."""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    vec = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    mag = np.linalg.norm(vec, axis=1)
    ok = np.isfinite(mag) & (mag > 1e-12)
    if not ok.any():
        return
    top = float(mag[ok].max())
    if length_by == "soft":
        typical = float(np.median(mag[ok]))
        length = spacing * 0.9 * np.tanh(mag / max(1.5 * typical, 1e-12))
    else:
        length = spacing * 0.85 * mag / top
    cols = np.asarray(colors, dtype=np.float64)
    per_arrow = cols.ndim == 2
    for k in np.flatnonzero(ok):
        if length[k] < 1e-6:
            continue
        d = vec[k] / mag[k] * length[k]
        c = tuple(cols[k]) if per_arrow else tuple(cols)
        # Soft fields are many small arrows: a head in proportion to each one, so the
        # direction still reads when the arrow is a few pixels long.
        arrow(ctx, cam, pts[k] - d / 2, pts[k] + d / 2, width, c, glow,
              head=0.38 if length_by == "soft" else None)  # fmt: skip


def with_alpha(color, opacity: float):
    return (color[0], color[1], color[2], color[3] * opacity)
