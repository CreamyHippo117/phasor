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
        "domain_glow": {"function": f, "center": (0.0, 0.0), "height": 4.5, "lines": 12},
        "escape": {"center": (-0.6, 0.0), "height": 2.6, "max_iter": 256},
        "palette": {"field": field, "lut": phase_lut()},
        "bright": {"image": image, "threshold": 1.0},
        "fringe": {"image": image, "amount": 0.004},
        "vignette": {"image": image, "strength": 0.3},
        "gauss": {"image": image, "sigma": 1.5, "axis": 0},
        "resize": {"image": image},
        "tonemap": {"image": image, "bloom": image},
        "srgb8": {"image": np.ascontiguousarray(image / np.float32(3.0))},  # display range
        "accumulate": {"image": image.astype(np.float16), "weight": 0.25, "acc": image},
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


# ---------------------------------------------------------------- whole frames (GPU_SPEED.md)

STAGES = (
    "state",
    "simulation",
    "draw",
    "gpu draw",
    "readback",
    "widen",
    "blur sum",
    "post",
    "srgb",
    "encode",
)


def run_frame_bench(engine: Any, name: str, count: int = 60, preset: str = "final",
                    out_dir: str | Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Render ``count`` frames from the middle of ``engine``'s scene the way ``phasor
    render`` does and report milliseconds per frame per stage (``core/timing.py``).
    ``other`` is what no stage claimed. The layout bake and kernel compiles happen before
    the clock starts, so they are not counted."""
    import tempfile

    from phasor.core.timing import Stopwatch
    from phasor.render.job import PRESETS, render_file

    fps = PRESETS[preset][2] or engine.document.meta.fps
    span = count / fps
    start = max(0.0, (engine.duration - span) / 2)
    with tempfile.TemporaryDirectory(prefix="phasor_bench_") as d:
        out = Path(out_dir or d) / f"{name}_bench.mp4"
        render_file(engine, out, preset=preset, start=start, end=start + 2 / fps)  # warm up
        with Stopwatch() as sw:
            r = render_file(engine, out, preset=preset, start=start, end=start + span)
    frames = r["frames"]
    setup = sw.seconds.get("setup", 0.0)  # once per render: kept out of the per-frame times
    per = {s: round(1000 * sw.seconds.get(s, 0.0) / frames, 2) for s in STAGES}
    per["other"] = round(1000 * sw.other() / frames, 2)
    cuda = gpu.default_gpu()
    return {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": platform.node(),
        "platform": platform.platform(),
        "scene": name,
        "preset": preset,
        "frame": list(r["size"]),
        "frames": frames,
        "motion_blur": r["motion_blur"],
        "encoder": r["encoder"],
        "backend": "cuda" if cuda is not None else "reference",
        "gl_renderer": engine.pass_runner().on_gl_info("GL_RENDERER"),
        "commit": _commit(),
        "wall_s": round(sw.wall, 3),
        "setup_s": round(setup, 3),
        "ms_per_frame": round(1000 * (sw.wall - setup) / frames, 2),
        "stages_ms_per_frame": per,
        "calls": dict(sorted(sw.calls.items())),
    }


def _commit() -> str | None:
    import subprocess

    from phasor.core.proc import NO_WINDOW

    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=10, creationflags=NO_WINDOW)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None


def save_frames(result: dict[str, Any], directory: str | Path = "benchmarks") -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    w, h = result["frame"]
    path = d / f"{result['when'][:10]}_frames_{result['scene']}_{result['backend']}_{w}x{h}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
