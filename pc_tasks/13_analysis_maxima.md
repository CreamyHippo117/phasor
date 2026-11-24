# 13: the Explorer analysis pack at its maxima (Milestone 30)

Why: the audit of Milestone 30 found the full-size "every analysis type at its documented
maximum" test too heavy for CI (313 s on the cloud box, with `tracemalloc`). CI now runs a
version with every `MAX_*` constant 100x smaller
(`tests/test_flow_analysis_audit.py::test_s_scaled_down_maxima_clamp_and_scale`). This task
runs the full size once on the PC (Windows 11, RTX 4070 Ti; nothing here uses CUDA) and on
the cloud box, and records the times.

## Run

From the repository root, in PowerShell or a shell:

```
uv run python benchmarks/analysis_maxima.py
```

## Expect

Each line prints one object (each in a scene of its own), its seconds and its budget;
`--memory` adds the peak memory (tracemalloc: two or three times slower). Aim (audit): every
single inspect under about 20 s at its maximum; peak under 2 GB.

| Object | Cloud box, 2026-11-24 (s) | PC (s) |
| --- | --- | --- |
| `section` (4096 seeds, clamped by `MAX_SECTION_WORK`) | 8.3 | |
| `attractor_sweep` (256 values, clamped by `MAX_SWEEP_WORK`) | 19.8 | |
| `equilibria_sweep` (64 values, grid clamped to 9 in 3D) | 7.7 | |
| `twins` (`MAX_LYAPUNOV_STEPS` and a 20000-step path pair) | 12.4 | |
| `limit_cycle` (`MAX_CYCLE_STEPS`) | 3.5 | |
| `divergence_3d` (256 per side) | 0.0 | |
| `divergence_disk` (`DISK_STEPS`, `DISK_CAP`) | 0.6 | |

Report any object over 20 s: its `MAX_*` constant in `phasor/domains/flow_analysis_math.py`
(or `flow_analysis.py`) is to be lowered.
