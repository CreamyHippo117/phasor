"""A scalar painted on a flat wall: a formula in x, y, z evaluated at every pixel.

The wall is a rectangle in space: ``origin`` is the world point of its lower-left corner
(measured from the room's corner, so a formula reads in meters from it), ``right`` and ``up``
are unit vectors along its edges. The view is the one ``ctx.field`` hands a kernel-drawn
object, in the rectangle's own coordinates (``center`` and ``height`` along ``right`` and
``up``), so a pixel at (u, v) is the point ``origin + u right + v up``, and the formula is
evaluated there as ``f(x, y, z, l, m, n, Lx, Ly, Lz)``: the point, three numbers the scene
animates (the room's mode numbers) and the room's size.

``to`` and ``blend`` (0-1) show (1 - blend) f + blend g, a straight line between two formulas,
so a note that changes into another passes through every value between and never flickers.
``scale`` is the value that comes out as 1 (full light); the output is ``f / scale``,
unclipped. Formulas and constants other than the sizes are run-time arguments, so animating
``l``, ``m``, ``n`` or ``blend`` never recompiles; only a new text does.

Output: float32 (h, w), the value per pixel, for ``heat_shade`` to colour (``image=``).
Plain arithmetic per pixel, no atomics: the same bits every run.
"""

from __future__ import annotations

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.codegen import POWI, RealFunction
from phasor.render.gpu.registry import VIEW_BODY, VIEW_PARAMS, KernelError, kernel, plane, view_args

NAMES = ("x", "y", "z", "l", "m", "n", "Lx", "Ly", "Lz")  # what a formula may use


def parse(source: str) -> RealFunction:
    """A wall formula, read by the whitelist parser (``FunctionError`` names what is wrong)."""
    return RealFunction.parse(source, NAMES)


def _prelude(function: RealFunction, to: RealFunction | None) -> str:
    sig = "(real x, real y, real z, real l, real m, real n, real Lx, real Ly, real Lz)"
    call = "(x, y, z, l, m, n, Lx, Ly, Lz)"
    if to is None:
        return POWI + f"\nstatic __device__ real phasor_g{sig} {{ return {function.c()}; }}"
    return (
        POWI
        + f"\nstatic __device__ real phasor_g0{sig} {{ return {function.c()}; }}"
        + f"\nstatic __device__ real phasor_g1{sig} {{ return {to.c()}; }}"
        + f"""
static __device__ real phasor_g{sig[:-1]}, real b) {{
    if (b <= R(0)) return phasor_g0{call};
    if (b >= R(1)) return phasor_g1{call};
    return (R(1) - b) * phasor_g0{call} + b * phasor_g1{call};
}}"""
    )


def source_for(function: RealFunction, to: RealFunction | None = None) -> str:
    blend = "" if to is None else ", (real)blend"
    return (
        _prelude(function, to)
        + """

extern "C" __global__ void phasor_wall("""
        + VIEW_PARAMS
        + """,
    double ox, double oy, double oz, double rx, double ry, double rz,
    double ux, double uy, double uz, double mode_l, double mode_m, double mode_n,
    double lx, double ly, double lz, double blend, double scale, float* out)
{"""
        + VIEW_BODY
        + """
    real wx = (real)ox + x * (real)rx + y * (real)ux;
    real wy = (real)oy + x * (real)ry + y * (real)uy;
    real wz = (real)oz + x * (real)rz + y * (real)uz;
    real v = phasor_g(wx, wy, wz, (real)mode_l, (real)mode_m, (real)mode_n,
                      (real)lx, (real)ly, (real)lz"""
        + blend
        + """);
    out[i] = (float)(v / (real)scale);
}
"""
    )


def _vec(name: str, v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (3,):
        raise KernelError(f"wall_field: {name} must be three numbers, got shape {a.shape}")
    return a


def wall_field_reference(
    tile: Tile,
    function: RealFunction,
    origin: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    center: tuple[float, float],
    height: float,
    mode: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scale: float = 1.0,
    to: RealFunction | None = None,
    blend: float = 0.0,
    precision: str = "f32",
) -> np.ndarray:
    o, r, u = _vec("origin", origin), _vec("right", right), _vec("up", up)
    u_, v_ = plane(tile, center, height)
    pts = o + u_[..., None] * r + v_[..., None] * u
    args = (pts[..., 0], pts[..., 1], pts[..., 2], *_vec("mode", mode), *_vec("size", size))
    val = function.numpy()(*args)
    if to is not None and blend > 0:
        other = to.numpy()(*args)
        val = other if blend >= 1 else (1 - blend) * val + blend * other
    return np.ascontiguousarray(val / scale, dtype=np.float32)


@kernel(
    "wall_field",
    "(tile, function, origin, right, up, center, height, mode=(0, 0, 0), size=(1, 1, 1), "
    "scale=1.0, to=None, blend=0.0, precision='f32') -> float32[h, w] (the formula / scale)",
    reference=wall_field_reference,
)
def wall_field(
    gpu,
    tile: Tile,
    function: RealFunction,
    origin: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    center: tuple[float, float],
    height: float,
    mode: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scale: float = 1.0,
    to: RealFunction | None = None,
    blend: float = 0.0,
    precision: str = "f32",
):
    """``function`` (``to`` blended in by ``blend``) divided by ``scale`` at each pixel of a
    wall: the rectangle at ``origin`` along unit vectors ``right`` and ``up`` (room
    coordinates), seen as ``center`` and ``height`` along them. ``mode`` is (l, m, n), ``size``
    the room's (Lx, Ly, Lz)."""
    o, r, u = _vec("origin", origin), _vec("right", right), _vec("up", up)
    lmn, lxyz = _vec("mode", mode), _vec("size", size)
    compiled = gpu.compile(source_for(function, to), "phasor_wall", precision)
    out = gpu.cp.empty((tile.h, tile.w), dtype=gpu.cp.float32)
    args = (*view_args(tile, center, height), *(np.float64(v) for v in (*o, *r, *u, *lmn, *lxyz)),
            np.float64(blend if to is not None else 0.0), np.float64(scale), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
