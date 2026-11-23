# Milestone 14: use the whole CPU

Plan and owner's answers: `MANIM_PARITY.md` ("Milestone 14", answer 5: a C++ compiler found at
run time, the NumPy reference as the fallback, no prebuilt binaries). Exit demo: on a machine
with CUDA hidden, `phasor render` of `prism_primes` uses every core, is N times faster than the
reference path, and matches the software golden.

Status: **built on Linux (4 cores, no CUDA, llvmpipe).** CUDA was not available, so nothing
here says anything about tier A. Untested on Windows and macOS.

## Built

| Area | Where | What |
| --- | --- | --- |
| CPU backend (14.1) | `phasor/render/cpu/backend.py` | compiles each kernel's CUDA string, unchanged, with g++ or clang++ found at run time (`PHASOR_CXX` names one): `-O3 -ffp-contract=off -fvisibility=hidden`, OpenMP where the compiler builds it, else `std::thread`. A generated wrapper per `extern "C" __global__` entry runs the blocks in parallel and each block's thread indices in order. Libraries are cached in the user cache (`~/.cache/phasor/cpu_kernels`, `%LOCALAPPDATA%`, `~/Library/Caches`; `PHASOR_CPU_CACHE`), keyed by a SHA-256 of the whole translation unit, the compiler path and version, and the flags. They are written under a temporary name and moved into place, so render workers never load half a file |
| Determinism | same | integer `atomicAdd` is `__atomic_fetch_add`; the splat kernel's sums are already 64-bit fixed point, so the same bits at any thread count. A float `atomicAdd` is deliberately absent: a kernel that uses one does not compile on the CPU |
| Threads | same | `threads()`: `set_threads(n)`, else `PHASOR_THREADS`, else every core this process may use. `registry.run(..., threads=n)` per call. `OMP_WAIT_POLICY=PASSIVE` by default, so idle OpenMP threads do not spin against llvmpipe and the Python draw step |
| Registry | `render/gpu/registry.py` | `backend` is `auto` (cuda, then cpu, then reference), `cuda`, `cpu` or `reference`. `active_backend()` says which one `auto` is. With `auto`, a kernel the host compiler rejects warns once and runs on its reference |
| Post (14.2) | `render/post.py`, `engine.py` | no code change: bloom (`bright`, `resize`, `gauss`), `fringe`, `tonemap`, `vignette`, `srgb8` and the blur sum / widen (`accumulate`) already go through `registry.run`, so on tier B they now run on the CPU backend. The references stay the specification; the tone map and the sRGB encode still happen once each (contract 4) |
| Workers (14.3) | `scene.default_threads`, `parallel_frames(threads=)`, `phasor render --threads` | without CUDA: threads per process = cores // workers, so workers x threads <= cores; each worker calls `set_threads`. The CUDA default is unchanged; its measurement is `pc_tasks/cpu_workers.md` |
| Doctor | `render/doctor.py` | check `cpu_kernels` (when there is no CUDA): compiler name and version, OpenMP or threads, the thread count. It builds and runs a kernel through the backend; the build is cached, so the probe in every render is quick. With no compiler it says kernels run on NumPy, and it names an MSVC `cl` if one is on PATH (not supported). The tier B text says "kernels run on the CPU (N threads, g++)" when the check passes. `phasor render`'s tier B warning says the same |
| Bench | `render/bench.py` | reports `backend` (`cuda`, `cpu`, `reference`) and `threads`, so CPU results do not overwrite the reference baseline files |
| 14.4 | `MANIM_PARITY.md` "14.4 findings" | written investigation only |

## Verification

Measured on this box: 4 cores (Xeon, 2.1 GHz, one thread per core), no CUDA, OpenGL on
llvmpipe, g++ 13.3 with OpenMP. **CUDA was not available, so nothing here is a tier A number.**

**Frame bench**, `phasor bench --frames SCENE --count 6 --preset preview` (960x540):
- Setup: the reference and CPU runs were done back to back on a quiet box (load average
  2-3, nothing else running). The reference was forced with `PHASOR_CXX=none`.
- The reference reruns match the 14.0 baseline (530 / 987 ms).
- JSON: `benchmarks/2026-11-23_frames_*_cpu_960x540.json`.

| ms/frame | prism_primes ref | prism_primes cpu | prism_domain ref | prism_domain cpu |
| --- | --- | --- | --- | --- |
| draw | 25.9 | 27.4 | 482.1 | 157.1 |
| widen | 14.1 | 15.7 | 14.2 | 14.6 |
| post (bloom, tone map, lens) | 464.6 | 57.7 | 451.1 | 56.6 |
| srgb | 26.0 | 8.8 | 23.6 | 9.7 |
| **total** | **542.3** | **120.5** (4.5x) | **984.9** | **250.3** (3.9x) |

`prism_domain`'s draw is the domain-coloring kernels, which run on the CPU backend now.
The widen step (`accumulate`) is memory-bound and gains nothing.

**Kernel-by-kernel spot check** (960x540, 10 runs, while another agent's test suite was
also running on the box, so treat these as rough): `tonemap` 38 -> 20 ms on 1 thread and
15 ms on 4; `resize` 8 -> 1.1 ms; `gauss` 27 ms (NumPy) against 37 ms (1 thread) and 15 ms
(4). The CUDA source computes `expf` per tap per pixel, while NumPy computes it once per tap.
On this VM the 4th core added little over 3 (`escape`: 93 / 49 / 32 / 32 ms on 1-4 threads).

**OpenMP wait policy** (prism_primes, two runs each, box loaded): `ACTIVE` gave 473 and 477
ms/frame; `PASSIVE` gave 289 and 252. Spinning OpenMP threads fight llvmpipe and Python for
the cores. Hence `OMP_WAIT_POLICY=PASSIVE` by default.

**Exit demo**, `phasor render examples/prism_primes.py --preset preview --start 2 --end 5`
(90 frames, the whole command, quiet box):
- Wall clock: 60.5 s on the reference (`PHASOR_CXX=none`), 16.7 s on the CPU backend:
  **3.6x**.
- Repeated with `time`: 55.7 s against 20.3 s. CPU time was user+sys 61 s (about 1.1 cores
  busy) on the reference against 33 s (about 1.6 cores) on the CPU backend.
- All cores are busy only in the kernel stages. Loading, the bake and the Python draw step
  stay on one core.
- **N was never set in 14.0, so the "at least N times" clause has no number to meet.** The
  measured gain is 3.6x for the whole command and 4.5x per frame.
- It matches the software golden: `tests/test_prism.py::test_demo_..._matches_its_golden`
  passes on `auto`, which is the CPU backend here. `test_exit_demo_prism_primes_on_the_cpu_is_the_reference_picture`
  finds the finished bytes within one step of the NumPy references.

**Workers x threads** (same command, two runs each, quiet box): 1x4 gave 16.7 and 16.8 s;
2x2 gave 17.3 and 16.9 s; 4x1 gave 17.7 and 16.7 s. These are the same within noise, so the
default stays at one worker on 4 cores with every core for its kernels. Earlier, with the box
loaded by another agent's tests, 2x2 was 7% faster (25.9 against 28.0 s). Not adopted.

**Tests** (`uv run pytest`, full suite with the CPU backend live on `auto`):
- Full run: 1475 passed, 9 failed, 129 skipped, in 61 minutes. The box was shared with
  another agent's test runs, so this time is not a measurement.
  - 7 of the failures were `ModuleNotFoundError: mcp`: this worktree had been synced without
    the `mcp` extra. After `uv sync --extra mcp`, `tests/test_mcp.py` and `tests/test_kit.py`
    pass (50 passed).
  - 2 fail before this milestone too: `test_beauty.py::test_prism_domain_golden_storyboard`
    (SSIM 0.958) and `test_prism_heat_golden_storyboard` (SSIM 0.798). Forcing the NumPy
    reference gives the same SSIM (0.95825 and 0.79794), and the CPU sheet is within one
    byte step of the reference sheet. Their software goldens no longer match this box's
    llvmpipe, whatever the kernel backend. Left alone: regenerating a golden needs someone
    to look at it first.
  - The three slowest tests were `test_riemann_demo` and `test_final[riemann_sqrt]`: 1098,
    957 and 798 s.
    - A profile of loading that demo: 1071 s with the CPU backend and 640 s with the
      reference. In both, the time goes to the camera solver's depth reads,
      `passes._view_depth` (GL readback, then NumPy), not to kernels.
    - The two runs had different competing load from the other agent's suite. A
      `_view_depth`-sized NumPy matmul timed before and after loading the OpenMP library was
      equal within noise.
    - Rerun later on the quiet box, 4 alternating runs each: 1.6-2.5 ms both with and
      without a kernel library loaded. So no kernel cost was found. The Riemann tests
      themselves were not rerun quietly.
- `tests/test_cpu_backend.py`: 57 tests. Every `CASES` kernel case (43) is compared with its
  reference as strictly as the emulator, and must be bit-identical at 1, 2 and 4 threads.
  The rest cover:
  - splat with 60,000 points on few pixels: the same bits at every thread count;
  - backend order and fallback, the thread settings and the cache;
  - a refused float atomic and a refused reserved name;
  - the launch checks, compiler naming (MSVC `cl` refused) and the doctor;
  - workers x threads <= cores, and `--threads`;
  - the exit-demo frame.

  Cold kernel cache: 20 s, warm: 10 s (about 0.5 s per kernel library). The full suite built 54
  libraries (1.2 MB).
- The emulator tests (`tests/cuda_emu.py`) are unchanged and still pass.
- With `PHASOR_CXX=clang++` (clang 18.1.3, no libomp, so the `std::thread` path),
  `tests/test_cpu_backend.py` passes.
- ruff, ruff format and lint-imports are clean (2 contracts kept).
- `phasor doctor` reports `cpu_kernels` and tier B: "OpenGL, no CUDA: kernels run on the CPU
  (4 threads, g++)".

## Deviations from SPEC (SPEC updated)

1. SPEC's "one kernel path" is kept: the CPU backend is a second *compiler* for the same
   kernel strings, not a second implementation. SPEC's tier B row and an as-built note in
   "GPU and CUDA plan" are updated.
2. Float `atomicAdd` is refused on the CPU (a compile error), not implemented, because its sum
   would depend on thread order. No kernel uses one today.
3. With `backend="auto"`, a CPU compile failure warns once and falls back to the reference, so a
   render never stops over it. `backend="cpu"` raises, and the CASES tests use it, so failures
   stay loud in tests.
4. `Engine.still` (previews, storyboards) still encodes sRGB with the host
   `encode_srgb8`, as before; only `output="srgb8"` (video) uses the kernel. Both are one
   encode, and they agree to one byte step.
5. `-fvisibility=hidden` was added, beyond the plan's `-O3 -fopenmp`. Without it each kernel
   was a PLT call per thread index, and `accumulate` ran 4x slower than NumPy. With it, the
   kernel inlines into the wrapper's loop.

## Bugs caught while building

1. The wrapper's own loop variables (`b`, `t`) shadowed kernel parameters of the same name. The
   doctor's probe kernel `phasor_doctor_axpb(..., float b, ...)` read the block index as `b`.
   The wrapper now uses `phasor_*` names, and a kernel parameter with one of them is refused.

## Contract audit

This was done by the authoring model, not by an independent reviewer (as for 7-9).
- Contracts 1, 2, 5 and 7: untouched.
- 3: each backend is a pure function of its inputs and gives the same bits at any thread
  count. CPU and NumPy differ by at most one byte step, as CUDA and NumPy already do.
- 4: unchanged (one tone map, one sRGB encode).
- 6: `render/cpu` lives inside `render`, and lint-imports is clean.
- CLAUDE.md:
  - one source per kernel (the CPU backend compiles the strings in `render/gpu/kernels/`);
  - launchers use only calls NumPy and CuPy share;
  - every `subprocess` call passes `NO_WINDOW` (`tests/test_proc.py`);
  - text files use UTF-8.
- The serial emulator (`tests/cuda_emu.py`) stays separate from the backend, as the plan
  says. It overlaps in about 40 lines (the prelude and argument passing).

## Windows (written for, not run)

- On Windows a DLL's own dependencies are not searched for on PATH, so the compiler's folder
  is added with `os.add_dll_directory` before loading.
- A library that builds but will not load (a missing `libgomp-1.dll`) is a `CompileError`.
  `auto` then uses the reference, and the failure is remembered, so it is not rebuilt on
  every call (tested by mocking the load).
- The OpenMP probe's temporary folder is left behind if Windows will not delete a loaded
  DLL.

## Deferred

- The CUDA worker cap (14.3, second half): `pc_tasks/cpu_workers.md`, on the RTX 4070 Ti.
- 14.4: findings only (`MANIM_PARITY.md`). Lowering the GL requirement to 4.1 is the owner's
  call.
- Faster CPU forms of `gauss` and `srgb8`: on one thread they are slower than NumPy, since
  the CUDA source recomputes the weights and `pow` per pixel. Changing them would change the
  CUDA kernels too.

## Untested

- Windows (MinGW g++ or clang++ building a `.dll`: written for, never run) and macOS (clang++
  needs libomp for OpenMP; without it the `std::thread` path is used: never run).
- Anything with CUDA, including the CUDA worker count (`pc_tasks/cpu_workers.md`).
- clang++ was run only through `tests/test_cpu_backend.py` (`PHASOR_CXX=clang++`, clang 18.1.3
  without libomp, so the `std::thread` path: 56 passed). It was not benchmarked, and
  clang++ with OpenMP was never run.
