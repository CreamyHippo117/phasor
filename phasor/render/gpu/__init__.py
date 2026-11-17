"""GPU math for the final renderer: the kernel registry, tiles and VRAM budget, the
expression code generator, and the bridge that hands frames to OpenGL.

    from phasor.render import gpu
    field = gpu.render("escape", 3840, 2160, center=(-0.5, 0), height=2.5)

Importing this package loads neither CuPy nor ModernGL.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from phasor.render.gpu.budget import Tile, budget_bytes, plan_tiles
from phasor.render.gpu.codegen import ComplexFunction, FunctionError
from phasor.render.gpu.registry import KernelError, default_gpu, get, kernel, registered, run

__all__ = [
    "ComplexFunction", "FunctionError", "KernelError", "Tile", "default_gpu", "get", "kernel",
    "plan_tiles", "registered", "render", "run",
]  # fmt: skip


def render(
    name: str,
    width: int,
    height_px: int,
    *,
    supersample: int = 1,
    backend: str = "auto",
    budget: int | None = None,
    **params: Any,
) -> np.ndarray:
    """Run a per-pixel kernel over a whole image, tile by tile, and stitch the result on
    the host. ``supersample=s`` renders each tile s times larger and averages it down, so
    only one enlarged tile is ever in memory.

    Averaging is right for color kernels (``domain``); a field such as ``escape`` should be
    supersampled after it has been colored.
    """
    from phasor.render.gpu.bridge import to_host

    if supersample < 1:
        raise ValueError("supersample must be at least 1")
    out: np.ndarray | None = None
    per_pixel = 16 * supersample * supersample  # RGBA float32 at the enlarged size
    budget = budget_bytes() if budget is None else budget
    for tile in plan_tiles(width, height_px, per_pixel, budget):
        part = to_host(run(name, tile.scaled(supersample), backend=backend, **params))
        if supersample > 1:
            s = supersample
            shape = (tile.h, s, tile.w, s, *part.shape[2:])
            part = part.reshape(shape).mean(axis=(1, 3), dtype=np.float64).astype(part.dtype)
        if out is None:
            out = np.empty((height_px, width, *part.shape[2:]), dtype=part.dtype)
        out[tile.y0 : tile.y0 + tile.h, tile.x0 : tile.x0 + tile.w] = part
    assert out is not None
    return out
