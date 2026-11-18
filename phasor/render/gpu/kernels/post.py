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
extern "C" __global__ void phasor_tonemap(int n, float exposure, float knee, const float* in,
                                          const float* bloom, float bloom_strength, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int c = 0; c < 3; c++) {
        float x = (in[4 * (long long)i + c] + bloom_strength * bloom[4 * (long long)i + c])
                  * exposure;
        if (!(x > 0.0f)) x = 0.0f;
        // Shoulder: identity up to the knee, then an exponential roll-off towards 1.
        // Paint colors stay exactly as the theme defines them; only light above the knee
        // (glow, bloom) is compressed. The curve is smooth (slope 1) at the knee.
        float s = 1.0f - knee;
        float y = x <= knee ? x : knee + s * (1.0f - expf(-(x - knee) / s));
        out[4 * (long long)i + c] = y;
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

extern "C" __global__ void phasor_bright(int n, float threshold, const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = in + 4 * (long long)i;
    float r = p[0] > 0.0f ? p[0] : 0.0f;
    float g = p[1] > 0.0f ? p[1] : 0.0f;
    float b = p[2] > 0.0f ? p[2] : 0.0f;
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float keep = lum > threshold ? (lum - threshold) / lum : 0.0f;
    float* o = out + 4 * (long long)i;
    o[0] = r * keep; o[1] = g * keep; o[2] = b * keep; o[3] = 1.0f;
}

extern "C" __global__ void phasor_resize(int n, int in_w, int in_h, int out_w, int out_h,
                                         const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ox = i % out_w, oy = i / out_w;
    float u = ((float)ox + 0.5f) * (float)in_w / (float)out_w - 0.5f;
    float v = ((float)oy + 0.5f) * (float)in_h / (float)out_h - 0.5f;
    u = u < 0.0f ? 0.0f : (u > (float)(in_w - 1) ? (float)(in_w - 1) : u);
    v = v < 0.0f ? 0.0f : (v > (float)(in_h - 1) ? (float)(in_h - 1) : v);
    int x0 = (int)u, y0 = (int)v;
    int x1 = x0 + 1 < in_w ? x0 + 1 : x0, y1 = y0 + 1 < in_h ? y0 + 1 : y0;
    float fx = u - (float)x0, fy = v - (float)y0;
    for (int c = 0; c < 4; c++) {
        long long r0 = (long long)y0 * in_w, r1 = (long long)y1 * in_w;
        float a = in[4 * (r0 + x0) + c], b = in[4 * (r0 + x1) + c];
        float d = in[4 * (r1 + x0) + c], e = in[4 * (r1 + x1) + c];
        float top = a + (b - a) * fx, bottom = d + (e - d) * fx;
        out[4 * (long long)i + c] = top + (bottom - top) * fy;
    }
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


def tonemap_reference(
    tile: Tile,
    image: np.ndarray,
    exposure: float = 1.0,
    bloom: np.ndarray | None = None,
    bloom_strength: float = 0.0,
    knee: float = 0.9,
) -> np.ndarray:
    _check(tile, image)
    rgb = image[..., :3]
    if bloom is not None:
        _check(tile, bloom)
        rgb = rgb + np.float32(bloom_strength) * bloom[..., :3]
    x = np.maximum(np.nan_to_num((rgb * np.float32(exposure)).astype(np.float32), nan=0.0), 0)
    x = x.astype(np.float32)
    k, s = np.float32(knee), np.float32(1.0) - np.float32(knee)
    with np.errstate(over="ignore"):
        y = np.where(x <= k, x, k + s * (np.float32(1) - np.exp(-(x - k) / s)))
    return np.concatenate([y, image[..., 3:]], axis=-1).astype(np.float32)


@kernel(
    "tonemap",
    "(tile, image, exposure=1.0, bloom=None, bloom_strength=0.0, knee=0.9) -> float32[h, w, 4]; "
    "shoulder curve of image + strength * bloom, linear in, linear 0..1 out",
    reference=tonemap_reference,
)
def tonemap(
    gpu, tile: Tile, image, exposure: float = 1.0, bloom=None, bloom_strength=0.0, knee=0.9
):
    """Map linear HDR to linear display range: unchanged up to ``knee``, then rolled off
    smoothly towards 1. The sRGB encode still happens once, later."""
    if not 0.0 < knee < 1.0:
        raise KernelError("knee must be between 0 and 1")
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(image)
    if bloom is None:
        glow, bloom_strength = src, 0.0
    else:
        _check(tile, bloom)
        glow = gpu.cp.ascontiguousarray(bloom)
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(SOURCE, "phasor_tonemap")
    args = (np.int32(tile.n), np.float32(exposure), np.float32(knee), src, glow,
            np.float32(bloom_strength), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
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


def bright_reference(tile: Tile, image: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    _check(tile, image)
    rgb = np.maximum(image[..., :3], 0).astype(np.float32)
    lum = (np.float32(0.2126) * rgb[..., 0] + np.float32(0.7152) * rgb[..., 1]
           + np.float32(0.0722) * rgb[..., 2])  # fmt: skip
    with np.errstate(all="ignore"):
        keep = np.where(lum > threshold, (lum - np.float32(threshold)) / lum, 0).astype(np.float32)
    out = np.ones_like(image, dtype=np.float32)
    out[..., :3] = rgb * keep[..., None]
    return out


@kernel(
    "bright",
    "(tile, image, threshold=1.0) -> float32[h, w, 4]; the light above threshold, for bloom",
    reference=bright_reference,
)
def bright(gpu, tile: Tile, image, threshold: float = 1.0):
    """Keep only what is brighter than ``threshold`` (luminance), scaled so the excess is
    kept and the hue is unchanged. Linear HDR: only emissive things exceed 1."""
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(image)
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(SOURCE, "phasor_bright")
    gpu.launch(compiled, tile.n, (np.int32(tile.n), np.float32(threshold), src, out))
    return out


def _resize_check(tile: Tile, image) -> None:
    if (tile.x0, tile.y0, tile.w, tile.h) != (0, 0, tile.full_w, tile.full_h):
        raise KernelError("resize needs the whole output image as its tile")
    if image.ndim != 3 or image.shape[2] != 4 or str(image.dtype) != "float32":
        raise KernelError("resize needs a float32 (h, w, 4) image")


def resize_reference(tile: Tile, image: np.ndarray) -> np.ndarray:
    _resize_check(tile, image)
    in_h, in_w = image.shape[:2]
    ox = np.arange(tile.w, dtype=np.float32)
    oy = np.arange(tile.h, dtype=np.float32)
    u = np.clip((ox + np.float32(0.5)) * np.float32(in_w) / np.float32(tile.w) - np.float32(0.5),
                0, in_w - 1).astype(np.float32)  # fmt: skip
    v = np.clip((oy + np.float32(0.5)) * np.float32(in_h) / np.float32(tile.h) - np.float32(0.5),
                0, in_h - 1).astype(np.float32)  # fmt: skip
    x0, y0 = u.astype(np.int64), v.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, in_w - 1), np.minimum(y0 + 1, in_h - 1)
    fx = (u - x0.astype(np.float32))[None, :, None]
    fy = (v - y0.astype(np.float32))[:, None, None]
    a, b = image[y0][:, x0], image[y0][:, x1]
    d, e = image[y1][:, x0], image[y1][:, x1]
    top, bottom = a + (b - a) * fx, d + (e - d) * fx
    return (top + (bottom - top) * fy).astype(np.float32)


@kernel(
    "resize",
    "(tile, image) -> float32[tile.h, tile.w, 4]; bilinear, sampled at pixel centers",
    reference=resize_reference,
)
def resize(gpu, tile: Tile, image):
    """Resample a whole image to the tile's size. Halving a dimension averages pixel pairs
    exactly, so repeated halving builds a box-filtered pyramid (bloom)."""
    _resize_check(tile, image)
    src = gpu.cp.ascontiguousarray(image)
    out = gpu.cp.empty((tile.h, tile.w, 4), dtype=gpu.cp.float32)
    compiled = gpu.compile(SOURCE, "phasor_resize")
    args = (np.int32(tile.n), np.int32(src.shape[1]), np.int32(src.shape[0]),
            np.int32(tile.w), np.int32(tile.h), src, out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
