"""Post-processing kernels on a whole linear RGBA float32 image: tone map and blur.

They work on the full frame (the tile must be the whole image): a blur reads neighbors, so
it cannot be cut into independent tiles, and a 4K frame is only about 133 MB.
"""

from __future__ import annotations

import math

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.registry import KernelError, kernel

SOURCE = """
extern "C" __global__ void phasor_tonemap(int n, float exposure, const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int c = 0; c < 3; c++) {
        float x = in[4 * (long long)i + c] * exposure;
        if (!(x > 0.0f)) x = 0.0f;
        // ACES filmic curve (Narkowicz fit): linear HDR in, linear display 0..1 out
        float y = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
        out[4 * (long long)i + c] = y > 1.0f ? 1.0f : y;
    }
    out[4 * (long long)i + 3] = in[4 * (long long)i + 3];
}

extern "C" __global__ void phasor_gauss(int n, int w, int h, int axis, int radius,
                                        float sigma, const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int px = i % w, py = i / w;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float total = 0.0f;
    for (int k = -radius; k <= radius; k++) {
        int qx = px + (axis == 0 ? k : 0);
        int qy = py + (axis == 1 ? k : 0);
        qx = qx < 0 ? 0 : (qx >= w ? w - 1 : qx);   // clamp at the edges
        qy = qy < 0 ? 0 : (qy >= h ? h - 1 : qy);
        float g = expf(-0.5f * (float)(k * k) / (sigma * sigma));
        const float* p = in + 4 * ((long long)qy * w + qx);
        for (int c = 0; c < 4; c++) acc[c] += g * p[c];
        total += g;
    }
    for (int c = 0; c < 4; c++) out[4 * (long long)i + c] = acc[c] / total;
}
"""


def _check(tile: Tile, image) -> None:
    if (tile.x0, tile.y0, tile.w, tile.h) != (0, 0, tile.full_w, tile.full_h):
        raise KernelError("post-processing kernels need the whole image, not a tile")
    if tuple(image.shape) != (tile.h, tile.w, 4) or str(image.dtype) != "float32":
        raise KernelError(
            f"image must be float32 with shape {(tile.h, tile.w, 4)}, "
            f"got {image.dtype} {tuple(image.shape)}"
        )


def tonemap_reference(tile: Tile, image: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    _check(tile, image)
    x = np.maximum(np.nan_to_num(image[..., :3], nan=0.0) * np.float32(exposure), 0)
    x = x.astype(np.float32)
    y = (x * (np.float32(2.51) * x + np.float32(0.03))) / (
        x * (np.float32(2.43) * x + np.float32(0.59)) + np.float32(0.14)
    )
    return np.concatenate([np.minimum(y, 1.0), image[..., 3:]], axis=-1).astype(np.float32)


@kernel(
    "tonemap",
    "(tile, image, exposure=1.0) -> float32[h, w, 4]; ACES filmic, linear in, linear out",
    reference=tonemap_reference,
)
def tonemap(gpu, tile: Tile, image, exposure: float = 1.0):
    """Map linear HDR to linear display range. The sRGB encode still happens once, later."""
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(image)
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(SOURCE, "phasor_tonemap")
    gpu.launch(compiled, tile.n, (np.int32(tile.n), np.float32(exposure), src, out))
    return out


def gauss_radius(sigma: float) -> int:
    return max(1, math.ceil(3.0 * sigma))


def gauss_reference(tile: Tile, image: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    _check(tile, image)
    r = gauss_radius(sigma)
    pad = [(0, 0)] * 3
    pad[1 - axis] = (r, r)  # axis 0 blurs along x, which is array axis 1
    padded = np.pad(image, pad, mode="edge")
    acc = np.zeros_like(image, dtype=np.float32)
    total = np.float32(0)
    size = image.shape[1 - axis]
    for k in range(-r, r + 1):
        g = np.exp(np.float32(-0.5) * np.float32(k * k) / np.float32(sigma * sigma))
        window = [slice(None)] * 3
        window[1 - axis] = slice(r + k, r + k + size)
        acc += g.astype(np.float32) * padded[tuple(window)]
        total += g.astype(np.float32)
    return (acc / total).astype(np.float32)


@kernel(
    "gauss",
    "(tile, image, sigma, axis) -> float32[h, w, 4]; one pass of a separable Gaussian blur",
    reference=gauss_reference,
)
def gauss(gpu, tile: Tile, image, sigma: float, axis: int):
    """Blur along x (``axis=0``) or y (``axis=1``). Run both for a full blur (bloom)."""
    _check(tile, image)
    if axis not in (0, 1) or sigma <= 0:
        raise KernelError("gauss needs axis 0 or 1 and sigma > 0")
    src = gpu.cp.ascontiguousarray(image)
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(SOURCE, "phasor_gauss")
    args = (np.int32(tile.n), np.int32(tile.w), np.int32(tile.h), np.int32(axis),
            np.int32(gauss_radius(sigma)), np.float32(sigma), src, out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
