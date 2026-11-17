"""Escape-time fractals: Mandelbrot and Julia sets of z^2 + c.

Output is the smooth (fractional) escape count per pixel, float32, and -1 for points that
never escape. Coloring is a later step, so one field can be re-colored for free.
Precision policy (SPEC): float32 by default, float64 once the view is under about 1e-5.
"""

from __future__ import annotations

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.registry import VIEW_BODY, VIEW_PARAMS, kernel, plane, view_args

BAILOUT = 65536.0  # |z|^2 at escape; a large radius makes the smooth count accurate
F64_BELOW = 1e-5  # view heights under this need float64


def precision_for(height: float) -> str:
    return "f64" if height < F64_BELOW else "f32"


SOURCE = (
    """
extern "C" __global__ void phasor_escape("""
    + VIEW_PARAMS
    + """, int julia, double jx, double jy, int max_iter, float* out)
{"""
    + VIEW_BODY
    + """
    real zr, zi, cr, ci;
    if (julia) { zr = x; zi = y; cr = (real)jx; ci = (real)jy; }
    else       { zr = R(0); zi = R(0); cr = x; ci = y; }
    real r2 = zr * zr + zi * zi;
    int k = 0;
    while (k < max_iter && r2 <= R(65536.0)) {
        real t = zr * zr - zi * zi + cr;
        zi = R(2) * zr * zi + ci;
        zr = t;
        r2 = zr * zr + zi * zi;
        k++;
    }
    if (r2 <= R(65536.0)) { out[i] = -1.0f; return; }
    // smooth count: k + 1 - log2(log|z|), with log|z| = log(r2) / 2
    out[i] = (float)((double)k + 1.0 - log2(0.5 * log((double)r2)));
}
"""
)


def escape_reference(
    tile: Tile,
    center: tuple[float, float],
    height: float,
    max_iter: int = 256,
    julia: tuple[float, float] | None = None,
    precision: str | None = None,
) -> np.ndarray:
    real = np.float64 if (precision or precision_for(height)) == "f64" else np.float32
    x, y = (a.astype(real) for a in plane(tile, center, height))
    if julia is None:
        zr, zi, cr, ci = np.zeros_like(x), np.zeros_like(y), x, y
    else:
        zr, zi = x.copy(), y.copy()
        cr, ci = np.full_like(x, julia[0]), np.full_like(y, julia[1])
    two, bail = real(2), real(BAILOUT)
    count = np.zeros(x.shape, dtype=np.int32)
    r2 = zr * zr + zi * zi
    for _ in range(max_iter):
        live = r2 <= bail
        if not live.any():
            break
        t = zr * zr - zi * zi + cr
        zi = np.where(live, two * zr * zi + ci, zi)
        zr = np.where(live, t, zr)
        r2 = np.where(live, zr * zr + zi * zi, r2)
        count += live
    escaped = r2 > bail
    with np.errstate(all="ignore"):
        smooth = count + 1.0 - np.log2(0.5 * np.log(r2.astype(np.float64)))
    return np.where(escaped, smooth, -1.0).astype(np.float32)


@kernel(
    "escape",
    "(tile, center, height, max_iter=256, julia=None, precision=None) -> float32[h, w]",
    reference=escape_reference,
)
def escape(
    gpu,
    tile: Tile,
    center: tuple[float, float],
    height: float,
    max_iter: int = 256,
    julia: tuple[float, float] | None = None,
    precision: str | None = None,
):
    """Mandelbrot set (``julia=None``) or the Julia set of ``c = julia``."""
    compiled = gpu.compile(SOURCE, "phasor_escape", precision or precision_for(height))
    out = gpu.cp.empty((tile.h, tile.w), dtype=gpu.cp.float32)
    jx, jy = julia if julia is not None else (0.0, 0.0)
    args = (*view_args(tile, center, height), np.int32(julia is not None),
            np.float64(jx), np.float64(jy), np.int32(max_iter), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
