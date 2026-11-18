"""Shared drawing helpers for domain objects: curves, discs and circles that keep their width
on screen, so every domain draws strokes the same way the built-in ``Line`` does."""

from __future__ import annotations

import numpy as np

from phasor.core.objects import DrawContext
from phasor.core.spatial import Camera

DISC_SEGMENTS = 48


def polyline(ctx: DrawContext, cam: Camera, pts, width, *, color, glow: float = 1.0) -> None:
    """A stroked polyline (mitered joins). ``width`` is in world units: a number, or one per
    point (perspective: wider where it is further away, so it looks constant on screen)."""
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
    ctx.triangles(positions, indices, color=color, glow=glow)


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


def with_alpha(color, opacity: float):
    return (color[0], color[1], color[2], color[3] * opacity)
