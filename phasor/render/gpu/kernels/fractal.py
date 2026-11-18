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
    # Iterate only the points that have not escaped yet; escaped ones keep their last z.
    zr, zi, cr, ci = (a.ravel().copy() for a in (zr, zi, cr, ci))
    count = np.zeros(zr.shape, dtype=np.int32)
    r2 = zr * zr + zi * zi
    live = np.flatnonzero(r2 <= bail)
    for _ in range(max_iter):
        if live.size == 0:
            break
        a, b = zr[live], zi[live]
        t = a * a - b * b + cr[live]
        b = two * a * b + ci[live]
        zr[live], zi[live] = t, b
        r2[live] = t * t + b * b
        count[live] += 1
        live = live[r2[live] <= bail]
    r2 = r2.reshape(x.shape)
    count = count.reshape(x.shape)
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


PALETTE = """
extern "C" __global__ void phasor_palette(int n, const float* field, const float* lut, int lut_n,
                                          float density, float offset, const float* inside,
                                          float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + 4 * (long long)i;
    float f = field[i];
    if (f < 0.0f) { o[0] = inside[0]; o[1] = inside[1]; o[2] = inside[2]; o[3] = 1.0f; return; }
    float t = sqrtf(f) * density + offset;
    t = t - floorf(t);
    float g = t * (float)lut_n;
    int k0 = (int)g;
    if (k0 >= lut_n) k0 = lut_n - 1;
    int k1 = (k0 + 1) % lut_n;
    float u = g - (float)k0;
    for (int c = 0; c < 3; c++) o[c] = lut[3 * k0 + c] * (1.0f - u) + lut[3 * k1 + c] * u;
    o[3] = 1.0f;
}
"""


def _palette_args(field, lut, inside):
    if field.ndim != 2 or str(field.dtype) != "float32":
        raise ValueError("palette needs a float32 (h, w) field from the escape kernel")
    table = np.ascontiguousarray(lut, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != 3 or len(table) < 2:
        raise ValueError("lut must be an (n, 3) table of linear RGB with n >= 2")
    return table, np.asarray(inside, dtype=np.float32)[:3]


def palette_reference(
    tile: Tile,
    field: np.ndarray,
    lut: np.ndarray,
    density: float = 0.35,
    offset: float = 0.0,
    inside: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    table, inside_rgb = _palette_args(field, lut, inside)
    t = np.sqrt(np.maximum(field, 0)).astype(np.float32) * np.float32(density) + np.float32(offset)
    t = (t - np.floor(t)).astype(np.float32)
    g = t * np.float32(len(table))
    k0 = np.minimum(g.astype(np.int64), len(table) - 1)
    u = (g - k0.astype(np.float32))[..., None]
    rgb = table[k0] * (np.float32(1) - u) + table[(k0 + 1) % len(table)] * u
    out = np.ones((*field.shape, 4), dtype=np.float32)
    out[..., :3] = np.where((field < 0)[..., None], inside_rgb, rgb)
    return out


@kernel(
    "palette",
    "(tile, field, lut, density=0.35, offset=0.0, inside=(0,0,0)) -> float32[h, w, 4]",
    reference=palette_reference,
)
def palette(gpu, tile: Tile, field, lut, density=0.35, offset=0.0, inside=(0.0, 0.0, 0.0)):
    """Color an escape field: a cyclic table indexed by sqrt(count), so bands stay even as
    counts grow; points in the set get ``inside``. ``tile`` is the field's own extent."""
    table, inside_rgb = _palette_args(field, lut, inside)
    src = gpu.cp.ascontiguousarray(field)
    out = gpu.cp.empty((*src.shape, 4), dtype=gpu.cp.float32)
    compiled = gpu.compile(PALETTE, "phasor_palette")
    args = (np.int32(src.size), src, gpu.cp.asarray(table), np.int32(len(table)),
            np.float32(density), np.float32(offset), gpu.cp.asarray(inside_rgb), out)  # fmt: skip
    gpu.launch(compiled, int(src.size), args)
    return out
