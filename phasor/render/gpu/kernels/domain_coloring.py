"""Domain coloring: a picture of a complex function.

Hue is the phase of f(z), looked up in a cyclic color table (so a theme can swap the
wheel). Brightness steps once every time |f| doubles, which draws the modulus contours.
Poles (and anything not finite) are white; zeros fall to the dark end of a contour band.
Output: linear RGBA float32, shape (h, w, 4).
"""

from __future__ import annotations

import numpy as np

from phasor.core.color import phase_lut
from phasor.render.gpu.budget import Tile
from phasor.render.gpu.codegen import ComplexFunction
from phasor.render.gpu.registry import VIEW_BODY, VIEW_PARAMS, kernel, plane, view_args

POLE = 1e30  # |f| above this is drawn as a pole


def source_for(function: ComplexFunction) -> str:
    return (
        "static __device__ cplx phasor_f(cplx z) { return "
        + function.c()
        + """; }

extern "C" __global__ void phasor_domain("""
        + VIEW_PARAMS
        + """, const float* lut, int lut_n, float contour, float* out)
{"""
        + VIEW_BODY
        + """
    cplx w = phasor_f(c_make(x, y));
    float* o = out + 4 * (long long)i;
    real m = c_abs(w);
    o[3] = 1.0f;
    if (!(m <= R(1e30))) { o[0] = 1.0f; o[1] = 1.0f; o[2] = 1.0f; return; }  // pole or NaN
    real band = R(0);
    if (m > R(0)) { real l = (real)log2(m); band = l - (real)floor(l); }
    real shade = R(1) - (real)contour * (R(1) - band);
    real h = c_arg(w) / R(2.0 * PHASOR_PI);
    h = h - (real)floor(h);
    real f = h * (real)lut_n;
    int k0 = (int)f;
    if (k0 >= lut_n) k0 = lut_n - 1;
    real u = f - (real)k0;
    int k1 = (k0 + 1) % lut_n;
    for (int c = 0; c < 3; c++)
        o[c] = (float)(shade * ((real)lut[3 * k0 + c] * (R(1) - u) + (real)lut[3 * k1 + c] * u));
}
"""
    )


def domain_reference(
    tile: Tile,
    function: ComplexFunction,
    center: tuple[float, float],
    height: float,
    contour: float = 0.35,
    lut: np.ndarray | None = None,
    precision: str = "f32",
) -> np.ndarray:
    lut = phase_lut() if lut is None else np.asarray(lut, dtype=np.float32)
    x, y = plane(tile, center, height)
    w = function.numpy()(x + 1j * y)
    with np.errstate(all="ignore"):
        m = np.abs(w)
        pole = ~(m <= POLE)
        log = np.log2(np.where(m > 0, m, 1.0))
        band = np.where(m > 0, log - np.floor(log), 0.0)
        shade = 1.0 - contour * (1.0 - band)
        h = np.angle(w) / (2 * np.pi)
        h = np.where(pole, 0.0, h - np.floor(h))
    f = h * len(lut)
    k0 = np.minimum(f.astype(np.int64), len(lut) - 1)
    u = (f - k0)[..., None]
    rgb = shade[..., None] * (lut[k0] * (1 - u) + lut[(k0 + 1) % len(lut)] * u)
    out = np.ones((tile.h, tile.w, 4), dtype=np.float32)
    out[..., :3] = np.where(pole[..., None], 1.0, rgb)
    return out


@kernel(
    "domain",
    "(tile, function, center, height, contour=0.35, lut=None, precision='f32') "
    "-> float32[h, w, 4] linear RGBA",
    reference=domain_reference,
)
def domain(
    gpu,
    tile: Tile,
    function: ComplexFunction,
    center: tuple[float, float],
    height: float,
    contour: float = 0.35,
    lut: np.ndarray | None = None,
    precision: str = "f32",
):
    """Color ``function`` over the view. ``contour`` (0-1) is how dark modulus bands get;
    ``lut`` is an (n, 3) cyclic table of linear RGB, default ``core.color.phase_lut()``."""
    table = np.ascontiguousarray(phase_lut() if lut is None else lut, dtype=np.float32)
    compiled = gpu.compile(source_for(function), "phasor_domain", precision)
    out = gpu.cp.empty((tile.h, tile.w, 4), dtype=gpu.cp.float32)
    args = (*view_args(tile, center, height), gpu.cp.asarray(table),
            np.int32(len(table)), np.float32(contour), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
