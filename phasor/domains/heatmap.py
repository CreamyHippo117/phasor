"""Heat maps: a grid of values drawn as colour through ``ctx.field``, shared by every type
that shows one (``pde.Field``, ``pde.Slice``, ``acoustics.PressureSlice``,
``fourier.Spectrogram``).

The plain picture is what those types always drew. The prism look (BEAUTY.md, item 2) adds,
each optional:

- ``smooth``: Catmull-Rom (bicubic) sampling between grid values instead of straight-line
  blending, so a coarse grid reads as a smooth field;
- ``isolines``: that many contour levels drawn as thin light about a pixel wide at any zoom
  (the distance to a level is measured in pixels through the local gradient);
- a glow weight per pixel (alpha), so with the object's ``glow`` role the strong parts of the
  field shine and haze. See ``DrawContext.field``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Pixels = Callable[[int, int, tuple[float, float], float], np.ndarray]
AMBIENT = 0.12  # with a glow weight: the share of colour a quiet (weight 0) pixel keeps


def _catmull_rom(t: np.ndarray) -> list[np.ndarray]:
    t2, t3 = t * t, t * t * t
    return [
        0.5 * (-t3 + 2 * t2 - t),
        0.5 * (3 * t3 - 5 * t2 + 2),
        0.5 * (-3 * t3 + 4 * t2 + t),
        0.5 * (t3 - t2),
    ]


def sample(grid: np.ndarray, g: np.ndarray, h: np.ndarray, how: str) -> np.ndarray:
    """``grid[x, y]`` at fractional grid coordinates ``g`` (x, length pw) and ``h`` (y,
    length ph), clamped to the grid: ``(ph, pw)``. ``how``: ``nearest`` (the cell's value),
    ``linear`` (bilinear) or ``smooth`` (Catmull-Rom bicubic)."""
    nx, ny = grid.shape
    g, h = np.clip(g, 0, nx - 1), np.clip(h, 0, ny - 1)
    if how == "nearest":  # the cell the point is in (a cell spans its centre +- 1/2)
        ix = np.minimum(np.floor(g + 0.5).astype(int), nx - 1)
        iy = np.minimum(np.floor(h + 0.5).astype(int), ny - 1)
        return grid[ix][:, iy].T
    x0, y0 = np.floor(g).astype(int), np.floor(h).astype(int)
    fx, fy = g - x0, h - y0
    if how == "linear":
        x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
        fx, fy = fx[None, :], fy[:, None]
        return (grid[x0][:, y0].T * (1 - fx) * (1 - fy) + grid[x1][:, y0].T * fx * (1 - fy)
                + grid[x0][:, y1].T * (1 - fx) * fy + grid[x1][:, y1].T * fx * fy)  # fmt: skip
    if how != "smooth":
        raise ValueError(f"how must be nearest, linear or smooth, got {how!r}")
    wx, wy = _catmull_rom(fx), _catmull_rom(fy)
    out = np.zeros((len(h), len(g)))
    for k in range(4):
        ix = np.clip(x0 + k - 1, 0, nx - 1)
        col = grid[ix]  # (pw, ny)
        for m in range(4):
            iy = np.clip(y0 + m - 1, 0, ny - 1)
            out += wy[m][:, None] * wx[k][None, :] * col[:, iy].T
    return out


def isoline_weight(level: np.ndarray, count: int) -> np.ndarray:
    """1 on the ``count`` contour levels of ``level`` (values 0..1), falling to 0 about a
    pixel away from them, whatever the zoom: the distance to the nearest level, in level
    units, over how much ``level`` changes per pixel."""
    if count <= 0:
        return np.zeros_like(level)
    s = level * count
    frac = np.abs(s - np.round(s))
    gy, gx = np.gradient(s)
    per_px = np.hypot(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(per_px > 1e-9, frac / per_px, np.inf)
    inside = (s > 0.5) & (s < count - 0.5)  # not the floor or ceiling of the range
    return np.where(inside, np.exp(-d * d / 0.8), 0.0)


def heat_pixels(
    grid: np.ndarray,
    size: tuple[float, float],
    colour: Callable[[np.ndarray], np.ndarray],
    *,
    cells: bool = False,
    how: str = "linear",
    level: Callable[[np.ndarray], np.ndarray] | None = None,
    isolines: int = 0,
    weight: Callable[[np.ndarray], np.ndarray] | None = None,
    line_glow: float = 0.7,
    background: np.ndarray | None = None,
) -> Pixels:
    """``ctx.field`` pixels for ``grid[x, y]`` spanning a ``size`` rectangle.

    ``cells``: grid values sit at cell centres (``x / w * nx - 0.5``), else at nodes from edge
    to edge (``x / w * (nx - 1)``). ``colour(value) -> (ph, pw, 3)`` linear RGB. ``level``
    maps a value to 0..1 for the isolines (default: the value). ``weight(value)`` is the glow
    weight 0..1 (default: none, alpha 1 as before): with one, colour scales from ``AMBIENT`` at
    weight 0 to full at 1, from ``background`` (the scene's, so a quiet field melts into it);
    the isolines add to the weight and brighten."""
    nx, ny = grid.shape
    sx, sy = size

    def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
        half_w = height * pw / ph / 2
        xs = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w
        ys = center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height  # row 0 = top
        if cells:
            g, h = xs / sx * nx - 0.5, ys / sy * ny - 0.5
        else:
            g, h = xs / sx * (nx - 1), ys / sy * (ny - 1)
        val = sample(grid, g, h, how)
        rgb = np.asarray(colour(val), dtype=np.float64)
        out = np.ones((ph, pw, 4), dtype=np.float32)
        lines = isoline_weight(np.clip((level or (lambda v: v))(val), 0, 1), isolines)
        if weight is None and not isolines:
            out[..., :3] = rgb
            return out
        w = np.zeros_like(val) if weight is None else np.clip(weight(val), 0, 1)
        lit = w + line_glow * lines * (1 - w)
        # With a weight, quiet parts fall toward black and strong parts keep their colour
        # (the prism look: light out of darkness); the contour lines are light on top.
        bg = np.zeros(3) if background is None else np.asarray(background[:3], dtype=np.float64)
        shade = 1.0 if weight is None else AMBIENT + (1 - AMBIENT) * w
        out[..., :3] = (
            bg + (rgb - bg) * np.asarray(shade)[..., None] + rgb * (line_glow * lines)[..., None]
        )
        out[..., 3] = lit
        return out

    return pixels


def lut_colour(lut: np.ndarray, how: str = "nearest") -> Callable[[np.ndarray], np.ndarray]:
    """A colour function from a (n, 3) colormap table for values in 0..1: the table entry
    (``nearest``, as the plain pictures always did) or blended between entries (``linear``,
    no visible steps)."""

    def colour(v: np.ndarray) -> np.ndarray:
        x = np.clip(v, 0, 1) * (len(lut) - 1)
        if how == "nearest":
            return lut[np.clip(x.astype(int), 0, len(lut) - 1)]
        k = np.minimum(np.floor(x).astype(int), len(lut) - 2)
        t = (x - k)[..., None]
        return lut[k] * (1 - t) + lut[k + 1] * t

    return colour
