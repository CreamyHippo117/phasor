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
from phasor.render.cpu import threads as cpu_threads
from phasor.render.gpu import ComplexFunction, Tile
from phasor.render.gpu.registry import active_backend


def _cases(w: int, h: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(0)
    slow = active_backend() == "reference"
    image = np.ascontiguousarray(rng.random((h, w, 4), dtype=np.float32) * 3)
    field = gpu.run("escape", Tile.full(w, h), center=(-0.6, 0.0), height=2.6, max_iter=256,
                    backend="reference")  # fmt: skip
    f = ComplexFunction.parse("1/(z**2 + 1)")
    fa = ComplexFunction.parse("(z**5 - 1)/(z**2 + a)", constant=True)
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
        # two functions blended (prism_transitions): both are evaluated per pixel
        "domain_glow:blend": {
            "function": fa,
            "to": ComplexFunction.parse("(z**3 - 1)/(z**2 + a)", constant=True),
            "blend": 0.5,
            "a": 0.6j,
            "center": (0.0, 0.0),
            "height": 4.5,
            "lines": 12,
        },
        "escape": {"center": (-0.6, 0.0), "height": 2.6, "max_iter": 256},
        "palette": {"field": field, "lut": phase_lut()},
        "fractal_glow": {"field": field, "lut": phase_lut(), "radius": 3.0},
        "bright": {"image": image, "threshold": 1.0},
        "fringe": {"image": image, "amount": 0.004},
        "vignette": {"image": image, "strength": 0.3},
        "gauss": {"image": image, "sigma": 1.5, "axis": 0},
        "resize": {"image": image},
        "tonemap": {"image": image, "bloom": image},
        "srgb8": {"image": np.ascontiguousarray(image / np.float32(3.0))},  # display range
        "accumulate": {"image": image.astype(np.float16), "weight": 0.25, "acc": image},
        # a million points on CUDA or the CPU backend; a few thousand for the NumPy reference
        "splat": {
            "points": rng.normal(0.0, 0.6, (4000 if slow else 1_000_000, 2)),
            "colors": (1.0, 0.6, 0.3),
            "center": (0.0, 0.0),
            "height": 2.6,
            "gain": 4.0,
        },
    }


def run_bench(width: int = 1920, height: int = 1080, repeats: int = 3) -> dict[str, Any]:
    """Time every kernel on one ``width`` x ``height`` frame (``resize`` halves it)."""
    cuda = gpu.default_gpu()
    backend = active_backend()
    results: dict[str, Any] = {}
    peak = None
    for name, params in _cases(width, height).items():
        tile = Tile.full(width // 2, height // 2) if name == "resize" else Tile.full(width, height)
        times = []
        for k in range(repeats + 1):  # the first run compiles; it is not counted
            if cuda is not None:
                cuda.cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            gpu.run(name.partition(":")[0], tile, backend=backend, **params)  # "kernel:variant"
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
        "threads": cpu_threads() if backend == "cpu" else None,
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
    "dedup",
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

    fps = PRESETS[preset].fps or engine.document.meta.fps
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
    backend = active_backend()
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
        "backend": backend,
        "threads": cpu_threads() if backend == "cpu" else None,
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


# ---------------------------------------------------------------- a live view (Milestone 28)

VIEW_MOTIONS = ("orbit", "pan", "flow")


def _moved_camera(cam: Any, motion: str, k: int, pivot: np.ndarray) -> Any:
    """``cam`` after ``k`` steps of a mouse drag: ``orbit`` turns it 3 degrees a step about
    the world z axis through ``pivot``; ``pan`` slides it 1% of the frame height sideways."""
    import dataclasses

    if motion == "pan":
        d = cam.right * (0.01 * k * float(cam.units_per_height()))
        move = np.eye(4)
        move[:3, 3] = -d
        return dataclasses.replace(cam, view_proj=cam.view_proj @ move, eye=cam.eye + d)
    a = np.radians(3.0 * k)
    rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    back = np.eye(4)  # world -> where the unturned camera saw it
    back[:3, :3] = rot.T
    back[:3, 3] = pivot - rot.T @ pivot
    return dataclasses.replace(cam, view_proj=cam.view_proj @ back,
                               eye=rot @ (cam.eye - pivot) + pivot, right=rot @ cam.right,
                               up=rot @ cam.up, forward=rot @ cam.forward)  # fmt: skip


def run_view_bench(engine: Any, name: str, t: float, motions: tuple[str, ...], *,
                   width: int = 640, height: int = 360, count: int = 30,
                   quality: str = "full", cpu_expand: bool = False,
                   pivot=(0.0, 0.0, 0.0), step: float = 1 / 30) -> dict[str, Any]:  # fmt: skip
    """Frames per second of a live view of ``engine``'s scene at time ``t``, drawn the way
    the Explorer's viewport will draw (advice 1.4): the draft look (no bloom, lens or fog),
    sRGB bytes read back, ``quality`` as asked. Each motion draws ``count`` frames after one
    warm-up: ``orbit``/``pan`` move the camera only (a session camera, as while the mouse is
    down), ``flow`` advances the scene time by ``step`` a frame (the flow time with it when
    the system has a ``rate``). Each call starts with no batch buffers on the card, so runs
    at the same times are comparable. ``cpu_expand``: batches expanded on the CPU instead of in the
    vertex shader (the cost of building triangles every frame, as strokes do). Reports
    milliseconds per frame (mean, median), fps from the median, batch buffers uploaded per
    frame, and the load average (POSIX) so a shared machine's numbers can be judged."""
    import dataclasses
    import os
    import statistics

    from phasor.render.post import finish
    from phasor.render.presets import DRAFT

    runner = engine.pass_runner()
    theme = engine.theme
    size = (int(width), int(height))
    out: dict[str, Any] = {}
    old = runner.expand_on_cpu
    runner.expand_on_cpu = bool(cpu_expand)
    runner.forget_batches()  # every run uploads from nothing: runs are comparable
    try:
        for motion in motions:
            if motion not in VIEW_MOTIONS:
                raise ValueError(f"motion is one of {', '.join(VIEW_MOTIONS)}, not {motion!r}")
            base = engine.state_at(t)
            centre = np.asarray(pivot, dtype=np.float64)
            times, uploads = [], []
            for k in range(count + 1):
                start = time.perf_counter()
                if motion == "flow":
                    state = engine.state_at(t + k * step)
                else:
                    cam = _moved_camera(base.camera, motion, k, centre)
                    state = dataclasses.replace(base, camera=cam)
                before = runner.uploads["batch"]
                pic, _ = runner.start_color_raw(state, size, fog=False, emission=False,
                                                quality=quality).collect()  # fmt: skip
                finish(pic[::-1], theme, None, output="srgb8", effects=DRAFT)
                if k:  # the first frame warms up (programs, buffers, kernels)
                    times.append(time.perf_counter() - start)
                    uploads.append(runner.uploads["batch"] - before)
            ms = [1000 * x for x in times]
            med = statistics.median(ms)
            out[motion] = {"ms_mean": round(statistics.fmean(ms), 2),
                           "ms_median": round(med, 2),
                           "fps": round(1000 / med, 1) if med > 0 else None,
                           "batch_uploads_per_frame": round(statistics.fmean(uploads), 3),
                           }  # fmt: skip
    finally:
        runner.expand_on_cpu = old
    load = os.getloadavg() if hasattr(os, "getloadavg") else None  # (POSIX only)
    return {"case": name, "frame": list(size), "frames": count, "quality": quality,
            "expand": "cpu" if cpu_expand else "shader",
            "gl_renderer": runner.on_gl_info("GL_RENDERER"), "platform": platform.platform(),
            "cores": os.cpu_count(),
            "load_average": None if load is None else [round(x, 2) for x in load],
            "motions": out}  # fmt: skip


def save_view(results: list[dict[str, Any]], directory: str | Path = "benchmarks") -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    w, h = results[0]["frame"] if results else (0, 0)
    path = d / f"{stamp[:10]}_explorer_view_{active_backend()}_{w}x{h}.json"
    body = {"when": stamp, "commit": _commit(), "machine": platform.node(), "cases": results}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
