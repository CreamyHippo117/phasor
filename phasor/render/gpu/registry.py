"""The one kernel path (SPEC "GPU and CUDA plan").

GPU math is written as CUDA C strings, compiled by CuPy ``RawKernel`` through NVRTC, and
registered with ``@kernel(name, signature, reference=numpy_fn)``. A kernel without a NumPy
reference cannot be registered. The reference is the specification: tests compare the two,
and machines without CUDA (tier B) run the reference instead.

A kernel is two functions with the same parameters:

    launch(gpu, tile, **params) -> array on the GPU      (the decorated function)
    reference(tile, **params)   -> numpy array

``run(name, tile, **params)`` picks one. Nothing else in phasor may create a RawKernel.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phasor.render.gpu.budget import Tile

BLOCK = 256
HEADER = (Path(__file__).parent / "complex.cuh").read_text(encoding="utf-8")
PRECISIONS = {"f32": (), "f64": ("-DPHASOR_F64",)}
# No fused multiply-add: the card then rounds every + and * like the NumPy reference, so a
# fractal's chaotic edge comes out the same on both (found on an RTX 4070 Ti, 2026-11-21:
# with FMA, 3% of a Mandelbrot tile and 10% of a 1e-11 float64 view moved). Costs a little
# speed; the reference is the specification.
NVRTC_OPTIONS = ("--fmad=false",)


class KernelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Kernel:
    name: str
    signature: str
    launch: Callable[..., Any]
    reference: Callable[..., np.ndarray]


_KERNELS: dict[str, Kernel] = {}


def kernel(name: str, signature: str, *, reference: Callable[..., np.ndarray] | None = None):
    """Register a kernel. Fails without a NumPy reference, or when the reference does not
    take the same parameters as the launcher (after the launcher's leading ``gpu``)."""

    def register(launch: Callable[..., Any]) -> Callable[..., Any]:
        if reference is None or not callable(reference):
            raise KernelError(f"kernel {name!r} has no NumPy reference; every kernel needs one")
        want = list(inspect.signature(launch).parameters)[1:]
        have = list(inspect.signature(reference).parameters)
        if want != have:
            raise KernelError(
                f"kernel {name!r}: reference takes {have} but the launcher takes {want}"
            )
        if name in _KERNELS:
            raise KernelError(f"kernel {name!r} is already registered")
        _KERNELS[name] = Kernel(name, signature, launch, reference)
        return launch

    return register


def get(name: str) -> Kernel:
    import phasor.render.gpu.kernels  # noqa: F401  (registers the built-in kernels)

    if name not in _KERNELS:
        raise KernelError(f"no kernel {name!r}; registered: {sorted(_KERNELS)}")
    return _KERNELS[name]


def registered() -> dict[str, Kernel]:
    import phasor.render.gpu.kernels  # noqa: F401

    return dict(_KERNELS)


class Gpu:
    """What a launcher may use: the array module, ``compile`` and ``launch``.

    The real one wraps CuPy. Tests substitute an emulator that compiles the same source with
    a host C++ compiler, so launchers must stick to the calls NumPy and CuPy share.
    """

    def __init__(self) -> None:
        import cupy

        self.cp = cupy
        self._compiled: dict[tuple[str, str, str], Any] = {}

    def compile(self, source: str, entry: str, precision: str = "f32") -> Any:
        """A compiled kernel. NVRTC runs once per (source, entry, precision); CuPy also
        caches the binary on disk."""
        key = (source, entry, precision)
        if key not in self._compiled:
            self._compiled[key] = self.cp.RawKernel(
                HEADER + source, entry, options=NVRTC_OPTIONS + PRECISIONS[precision]
            )
        return self._compiled[key]

    def launch(self, compiled: Any, n: int, args: tuple) -> None:
        """Run ``compiled`` for thread indices 0..n-1. Kernels return early when their
        index is past ``n``; scalars in ``args`` must be NumPy scalars of the C type."""
        compiled((-(-n // BLOCK),), (BLOCK,), args)


@functools.cache
def default_gpu() -> Gpu | None:
    """The CUDA backend, or None when CuPy or a device is missing (tier B and C)."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        return Gpu()
    except Exception:
        return None


def run(name: str, tile: Tile, *, backend: str = "auto", **params: Any) -> Any:
    """Run a kernel on one tile. ``backend``: ``auto`` (CUDA when present, else the NumPy
    reference), ``cuda`` or ``reference``. Returns a CuPy or NumPy array accordingly."""
    k = get(name)
    if backend not in ("auto", "cuda", "reference"):
        raise ValueError(f"backend must be auto, cuda or reference, got {backend!r}")
    gpu = None if backend == "reference" else default_gpu()
    if gpu is None:
        if backend == "cuda":
            raise KernelError("no CUDA device (run `phasor doctor`)")
        return k.reference(tile, **params)
    return k.launch(gpu, tile, **params)


def view_args(tile: Tile, center: tuple[float, float], height: float) -> tuple:
    """The leading arguments shared by every per-pixel kernel: thread count, the tile, the
    full image, and the view (center and height of the frame in the plane)."""
    return (
        np.int32(tile.n), np.int32(tile.x0), np.int32(tile.y0), np.int32(tile.w),
        np.int32(tile.full_w), np.int32(tile.full_h),
        np.float64(center[0]), np.float64(center[1]), np.float64(height),
    )  # fmt: skip


# The matching C parameter list and pixel -> plane mapping, shared by every such kernel.
VIEW_PARAMS = (
    "int n, int tile_x0, int tile_y0, int tile_w, int full_w, int full_h, "
    "double cx, double cy, double height"
)
VIEW_BODY = """
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int px = tile_x0 + i % tile_w;
    int py = tile_y0 + i / tile_w;
    double aspect = (double)full_w / (double)full_h;
    real x = (real)(cx + (((double)px + 0.5) / full_w - 0.5) * height * aspect);
    real y = (real)(cy - (((double)py + 0.5) / full_h - 0.5) * height);
"""


def plane(tile: Tile, center: tuple[float, float], height: float) -> tuple[np.ndarray, np.ndarray]:
    """The NumPy counterpart of ``VIEW_BODY``: float64 x and y, shape (tile.h, tile.w)."""
    px = tile.x0 + np.arange(tile.w, dtype=np.float64)
    py = tile.y0 + np.arange(tile.h, dtype=np.float64)
    aspect = tile.full_w / tile.full_h
    x = center[0] + ((px + 0.5) / tile.full_w - 0.5) * height * aspect
    y = center[1] - ((py + 0.5) / tile.full_h - 0.5) * height
    return np.broadcast_to(x, (tile.h, tile.w)), np.broadcast_to(y[:, None], (tile.h, tile.w))
