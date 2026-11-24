"""Every Milestone 30 analysis object at its documented maximum, each in a scene of its own:
seconds and peak memory (pc_tasks/13_analysis_maxima.md).

Run: uv run python benchmarks/analysis_maxima.py [name ...] [--memory]"""

from __future__ import annotations

import sys
import time
import tracemalloc

from phasor import Scene, flow
from phasor.domains import flow_analysis_math as fa
from phasor.domains import flow_math as fm


def _seeds():
    return tuple(tuple(float(v) for v in q)
                 for q in fm.fill("ball", (1, 1, 20), 5.0, fa.MAX_SECTION_SEEDS, 3))  # fmt: skip


CASES = {
    "section": lambda: flow.Section(id="m.x", seeds=_seeds(), normal=(0, 0, 1), offset=27.0),
    "attractor_sweep": lambda: flow.BifurcationDiagram(
        id="m.x", variable="rho", range=(20, 30), count=fa.MAX_SWEEP, mode="attractor",
        x0=(1, 1, 20), transient=1e4, sample_time=1e4),
    "equilibria_sweep": lambda: flow.BifurcationDiagram(id="m.x", variable="rho",
                                                        range=(0, 30), count=64, grid=64),
    "twins": lambda: flow.TwinProbes(id="m.x", x0=(1, 1, 20), lyapunov=1e4, transient=1e4),
    "limit_cycle": lambda: flow.LimitCycle(id="m.x", x0=(1, 1, 20), transient=1e4,
                                           max_time=1e4),
    "divergence_3d": lambda: flow.DivergenceMap(id="m.x", counts=256, z=27.0),
    "divergence_disk": None,  # a 2D system: below
}  # fmt: skip


def run(name: str) -> None:
    s = Scene("analysis_max", passes=None)
    if name == "divergence_disk":
        s.add(flow.System(id="m", tau=2000.0))
        s.add(flow.DivergenceMap(id="m.x", counts=256, disk_radius=0.5))
    else:
        s.add(flow.System(id="m", **flow.preset("lorenz"), tau=2000.0))
        s.add(CASES[name]())
    memory = "--memory" in sys.argv  # tracemalloc slows the run two or three times
    if memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    sem = s.engine.inspect(0.0, id="m.x")["objects"][0]["semantic"]
    took = time.perf_counter() - t0
    peak = f", peak {tracemalloc.get_traced_memory()[1] / 2**20:.0f} MB" if memory else ""
    tracemalloc.stop()
    print(f"{name}: {took:.1f} s{peak}, budget {sem.get('budget')}", flush=True)


if __name__ == "__main__":
    for name in [a for a in sys.argv[1:] if not a.startswith("--")] or CASES:
        run(name)
