"""``phasor bench``: milliseconds per frame per kernel, and peak VRAM (SPEC "Testing and
benchmarks"). Results go to ``benchmarks/`` so a slowdown shows up in review."""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core.color import phase_lut
from phasor.render import gpu
from phasor.render.gpu import ComplexFunction, Tile


def _cases(w: int, h: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(0)
    image = np.ascontiguousarray(rng.random((h, w, 4), dtype=np.float32) * 3)
    field = gpu.run("escape", Tile.full(w, h), center=(-0.6, 0.0), height=2.6, max_iter=256,
                    backend="reference")  # fmt: skip
    f = ComplexFunction.parse("1/(z**2 + 1)")
    # a 1e-20 view near a spiral of the Mandelbrot set: the perturbation path of a deep zoom
    # (for timing, a float64 orbit of the reference point is close enough)
    z, orbit = 0j, [0j]
    for _ in range(1000):
        z = z * z + complex(-0.7756837680090538, 0.1364673682946901)
        orbit.append(z)
    orbit = np.c_[np.real(orbit), np.imag(orbit)]
    return {
        "deep": {"center": (0.0, 0.0), "height": 1e-20, "orbit": orbit, "max_iter": 1000},
        "domain": {"function": f, "center": (0.0, 0.0), "height": 4.5},
        "escape": {"center": (-0.6, 0.0), "height": 2.6, "max_iter": 256},
        "palette": {"field": field, "lut": phase_lut()},
        "bright": {"image": image, "threshold": 1.0},
        "gauss": {"image": image, "sigma": 1.5, "axis": 0},
        "resize": {"image": image},
        "tonemap": {"image": image, "bloom": image},
    }


def run_bench(width: int = 1920, height: int = 1080, repeats: int = 3) -> dict[str, Any]:
    """Time every kernel on one ``width`` x ``height`` frame (``resize`` halves it)."""
    cuda = gpu.default_gpu()
    backend = "cuda" if cuda is not None else "reference"
    results: dict[str, Any] = {}
    peak = None
    for name, params in _cases(width, height).items():
        tile = Tile.full(width // 2, height // 2) if name == "resize" else Tile.full(width, height)
        times = []
        for k in range(repeats + 1):  # the first run compiles; it is not counted
            if cuda is not None:
                cuda.cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            gpu.run(name, tile, backend=backend, **params)
            if cuda is not None:
                cuda.cp.cuda.Device().synchronize()
            if k:
                times.append((time.perf_counter() - t0) * 1000)
        results[name] = {"ms": round(float(np.median(times)), 3)}
        if cuda is not None:
            used = cuda.cp.get_default_memory_pool().total_bytes()
            peak = max(peak or 0, int(used))
    return {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": platform.node(),
        "platform": platform.platform(),
        "backend": backend,
        "frame": [width, height],
        "repeats": repeats,
        "peak_vram_bytes": peak,
        "kernels": results,
    }


def save(result: dict[str, Any], directory: str | Path = "benchmarks") -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stamp = result["when"][:10]
    path = d / f"{stamp}_{result['backend']}_{result['frame'][0]}x{result['frame'][1]}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
