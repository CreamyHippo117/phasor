# Faster renders: keeping the card busy

Plan requested by the owner on 2026-11-20; **not yet approved**. Step 4's bake cache (ADR
0003) and step 5's new dependency need the owner's approval before they are built. This
file holds the detail and the status, like `TESTING.md`. Measure before and after every step. The goal is
**milliseconds per frame**; GPU utilization rises as a side effect once the card stops
waiting for Python.

## Where a frame goes today (RTX 4070 Ti, Windows 11, 2026-11-20)

Measured with `phasor bench --frames SCENE` (step 0), 60 final frames from the middle of
each scene (1920x1080, 60 fps, motion blur: 4 samples per moving frame):

| stage | motion_gallery | prism_primes | what it is |
| --- | --- | --- | --- |
| draw | 105 ms | 12 ms | Python: objects' `draw()`, strokes, GL calls |
| gpu draw | 2 ms | 1 ms | the card drawing all 4 samples |
| readback | 20 ms | 14 ms | copying the 2 float16 targets per sample to the host |
| widen | 137 ms | 100 ms | NumPy: float16 -> float32, flip, stack |
| blur sum | 121 ms | 77 ms | NumPy: float64 sum of the samples and the divide |
| post | 17 ms | 17 ms | bloom, tone map, lens on the card (with transfers) |
| srgb | 133 ms | 133 ms | NumPy: sRGB encode to 8 bits |
| encode | 8 ms | 8 ms | NVENC |
| state, other | 12 ms | 13 ms | |
| **per frame** | **555 ms** | **376 ms** | |

The card draws a whole frame in 1-2 ms. Three NumPy conversions (widen, blur sum, srgb)
are 70% of motion_gallery's frame and 82% of prism_primes'.

The first, rougher measurement (before the bench; draw included widen and readback):

| stage | motion_gallery | prism_primes | runs on |
| --- | --- | --- | --- |
| draw (x4 samples) + read back | 264 ms | 128 ms | mostly Python building strokes |
| summing the blur samples, the rest | 146 ms | 100 ms | NumPy, float64 |
| sRGB encode (`render/output.py`) | 134 ms | 135 ms | NumPy, float64 |
| bloom, tone map, lens (CUDA) | 16 ms | 20 ms | card |
| NVENC write | 8 ms | 8 ms | card |
| scene state | 4 ms | 6 ms | CPU |
| **per frame** | **571 ms** | **396 ms** | |

The whole showcase (10 scenes) took 71 minutes at 33% average GPU use and under 1% NVENC.

A profile of the draw step (`motion_gallery`) shows the time in `domains/strokes.py`
`polyline` and in `GLRenderer._draw_object`, which makes a new VBO, IBO and VAO for every
draw command, about 400 GL objects per sample. The read-back itself is about 2.5 ms.

## Step 0. Measure it the same way every time

Make the timing above a feature, so each step is judged by numbers, not estimates.

- `phasor bench --frames SCENE [--frames-count 60]`: renders N final frames from the
  middle of SCENE and reports ms/frame per stage (state, simulation, draw, blur sum, post,
  sRGB, encode), with the device and the commit. Saved to `benchmarks/` like the kernel bench.
- Timing hooks live in `render/job.py` (one `Stopwatch` passed down; no monkeypatching).
  CUDA stages synchronize before stopping the clock, or the card's work is misattributed.

Tests:
- `test_frame_bench_reports_every_stage` (`gl`: it draws): the report has every stage, the stage times add up to the wall time within 5%, and it is JSON.
- `test_frame_bench_runs_on_the_card` (`gpu`): the post stages say CUDA.

Done when: the table above can be regenerated with one command.

## Step 1. sRGB encode on the card (expected: -130 ms/frame)

Fuse the end of `post.finish` into one kernel: tone map, vignette, sRGB encode and the
8-bit conversion, producing the (h, w, 3) uint8 frame the encoder takes. The card then
sends back 6 MB per frame instead of 33 MB of float32.

- New kernel `display8` in `render/gpu/kernels/post.py`; its NumPy reference is exactly
  today's path (`tonemap_reference` -> `vignette_reference` -> `output.encode_srgb8`), so
  the reference *is* the current behavior. Compute the sRGB curve in double in the kernel
  (cheap at 2 M pixels), matching `core.color.linear_to_srgb`.
- `post.finish(..., output="linear" | "srgb8")`. `"linear"` stays for stills that need it
  and for the inspector; `job.render_file` asks for `"srgb8"`.
- Contract 4 still holds: the tone map and the sRGB encode each happen once. The kernel
  becomes the one place sRGB is encoded on the card; `encode_srgb8` stays as its reference
  and the CPU path. Update the docstrings and SPEC "Look" in the same commit.

Tests:
- Kernel tested the three required ways: add `display8` to `CASES` in
  `tests/test_kernels.py` (reference vs known maths, CUDA source on the CPU emulator,
  the real card with the `gpu` marker).
- `test_display8_matches_the_old_path_pixel_for_pixel`: random HDR frames plus edge cases
  (0, the knee, 1, NaN, negative, huge): card output equals `encode_srgb8(finish(...))` on
  at least 99.99% of pixels and is never off by more than 1 code value.
- `test_paint_below_the_knee_survives_display8`: theme paint colors come out as their
  exact sRGB bytes (the "paint is unchanged" rule).
- Goldens: every golden passes unchanged (SSIM >= 0.97) **and** a new strict check renders
  three showcase frames through old and new paths: max difference 1 code value.

Done when: bench shows the sRGB stage under 5 ms and all of the above pass.

## Step 2. Widen and sum the blur samples on the card (expected: -200 to -260 ms/frame)

Today each blur sample is read back as float16, widened to float32, stacked, added up in
float64 on the CPU, divided, narrowed again, and then sent to the card for post.

- `GLRenderer` gets an internal read that returns the raw float16 picture and emission
  without widening or stacking ("widen" is 100-137 ms per frame today: it goes away; the
  card widens with `astype`, a call NumPy and CuPy share) (`color_and_emission` keeps its public float32 contract for
  the inspector and stills).
- New kernel `accumulate` (out += weight * in, float32, any number of channels): each
  sample goes to the card once (2 x 16 MB, about 3 ms), is added there, and the sum goes
  straight into `finish` as a device array; `finish` stops forcing host arrays in.
- `Engine.frame_blurred` keeps its signature and its determinism (same samples, same
  order, same weights). Float32 accumulation on the card replaces float64 on the host; the
  tests below prove the difference never shows in 8-bit output.

Tests:
- `accumulate` in `CASES` (three ways).
- `test_blur_sum_on_the_card_matches_the_host_sum`: 1, 2, 4 and 8 samples, bright and dark
  frames: the final 8-bit frames equal the float64 host path within 1 code value.
- `test_a_still_moment_is_still_exactly_frame`: where nothing moves, `frame_blurred`
  is bit-identical to `frame` (it already is; keep it so).
- `test_same_blurred_frame_twice_is_identical` (`gpu`): determinism on the card.
- `test_memory_stays_flat_over_a_long_render` (exists) still passes: the device sum must
  not leak.

Done when: bench shows widen + blur sum under 20 ms on motion_gallery (from 258 ms).

## Step 3. Draw without building every stroke from scratch (expected: 264 -> 30-60 ms)

Three changes, in this order, each committed and measured on its own:

- **3a. Reuse GL buffers.** A small arena per program layout in `GLRenderer`: one growing
  VBO/IBO/CBO, written with `orphan` + `write`, and VAOs cached by layout instead of created
  and released per command. No picture change at all.
- **3b. One draw per object and kind.** Concatenate an object's mesh commands that share
  a program and uniforms (color, glow) into one buffer and one draw call; strokes keep
  their order inside the buffer, so blending order is unchanged. Text and fields stay as
  they are.
- **3c. Skip rebuilding what did not move.** The 4 blur samples of a frame, and most of
  consecutive frames, ask the same objects for the same strokes. Cache an object's draw
  commands keyed by (its resolved params, its whole camera, the render size and zoom, the
  pass, the theme hue); a hit skips `draw()` and the stroke maths. Bounded LRU, like
  `FIELD_CACHE`. This is **not** safe by construction: many `draw()` methods also read
  `state.t`, the frame size (strokes keep a constant width on screen) or other objects
  (tracer and callout read a guide). So the cache is opt-in per type: a class variable
  (`Cacheable = True`) a type may set only when its `draw()` reads nothing but its own
  params, its camera and the size; a test checks each opted-in type's `draw()` source.
- Then profile again; if `strokes.polyline` is still on top, vectorize its remaining
  per-call NumPy overhead (`np.r_`/`vstack` of tiny arrays).

Rules that still apply: all GL work stays in `render/passes.py` on the GL thread
(`on_gl_thread`), and objects keep the `draw(ctx, state, pass_)` contract (contract 5).

Tests:
- `test_batched_drawing_is_pixel_identical` (`gl`): for every example, frames at three
  times through the old and new draw paths are **bit-identical** in all three passes
  (color, id, solo). Keep the old path behind a flag until this has passed on Linux
  (Mesa) and Windows (NVIDIA), then delete it.
- `test_the_command_cache_never_serves_a_stale_object`: change one param, the camera, the
  size, the theme hue or the pass, and the cache misses; repeat a time, and it hits.
- `test_cached_render_equals_uncached_render` (`gl`): **every** frame of a short render of
  each example, with and without the cache, bit-identical; includes a partnered type
  (tracer), a Life-moving type and a 3D orbit.
- `test_only_self_contained_types_opt_in`: an opted-in type whose `draw()` reads
  `state.t`, `state.objects` or another object fails.
- `test_gl_objects_stay_flat`: rendering 300 frames does not grow the number of live GL
  buffers (count through `ctx`), the same idea as the VRAM test.
- The inspector tests and every lint `MODES` test pass unchanged: they read the id pass.
- Goldens unchanged.

Done when: bench shows the draw stage under 60 ms on motion_gallery with all of the above.

## Step 4. Render several stretches at once (expected: about x3 on wall time)

`Engine.state_at(t)` depends only on the document and `t` (contract 3), so frames can be
made in any order by any process. The CPU has 16 threads; one is busy today.

- Workers: `N` processes (default: half the cores, at most 4; `--workers` to set). Each
  loads the scene, takes every N-th chunk of 30 frames, and sends finished uint8 frames
  back through shared memory. The parent writes them **in order** to the one encoder, so
  there is one NVENC session, one audio mux, and the file is the same as a serial render.
- Spawned processes (Windows has no fork): each makes its own GL context and CUDA context
  (about 300 MB of VRAM each).
- **Shared VRAM.** Each worker's tile planner (`render/gpu/budget.py`) would size itself
  against the whole card's free memory; four of them would overcommit (deep_zoom at 4K).
  The parent splits the budget and passes each worker its share.
- **Shared simulation cache.** Workers rendering an FDTD or heat scene write the same
  `.phasor_cache` checkpoints at once. Writes become atomic (temp file, then rename), and a
  half-written file is never read.
- Placement under contract 6: the driver builds Engines, so it lives at or above the
  `engine` layer (in `phasor/scene.py`, next to `load`); `render/job.py` only gains a
  "frames from this iterator" path and never imports the engine.
- **Setup cost.** Each worker bakes the layout again (`riemann_sqrt`: 155 s). Persist the
  bake to disk keyed by the document hash and the solver sources, the way simulations
  cache checkpoints (`core/simulation.py`). Only `Engine` reads or writes it (contract 7);
  this needs an ADR (0003) and owner approval before building.
- Cancel and failure: a failed worker stops the render with its error; Ctrl+C and the
  desktop app's Cancel stop every worker and leave no files behind.

Tests:
- `test_parallel_render_equals_serial_render` (`gl`): a short scene to a PNG sequence with
  1 and 3 workers: every PNG is byte-identical. Also the mp4: same frame count, decoded
  frames identical.
- `test_frames_arrive_in_order_whatever_finishes_first`: a fake slow worker cannot reorder
  frames.
- `test_workers_share_the_vram_budget` (`gpu`): 4 workers on a 4K deep_zoom stay within
  the card's memory.
- `test_two_processes_filling_one_simulation_cache`: no torn reads, same states as one
  process.
- `test_a_failing_worker_fails_the_render_and_cleans_up`, `test_cancel_stops_every_worker`.
- `test_baked_layout_from_disk_equals_a_fresh_bake`: `state_at` at many times is equal;
  a changed document or solver source misses the cache; a damaged cache file is rebuilt.
- Narrated scenes keep audio in sync (C9 in `TESTING.md`): audio length equals video length.
- Packaged app: exports with workers work from `phasor.exe --cli` (frozen apps need
  `multiprocessing.freeze_support()`, which `packaging/entry.py` already calls).

Done when: the showcase renders at least 2.5x faster than after step 3, with identical
output.

## Step 5. Keep frames on the card end to end (expected: modest; measure first)

After steps 1-4 the remaining host traffic is the per-sample read-back (about 2.5 ms each)
and the 6 MB upload to the encoder. Worth doing only if the step 0 bench says so; SPEC
already requires an interop path to "ship only if it measures faster".

- **GL -> CUDA without the host.** In `render/gpu/bridge.py` (the only place pixels may
  cross): register the color and emission targets with CUDA (`cudaGraphicsGLRegisterImage`
  through the `cuda-python` bindings, to confirm against current docs; CuPy 14 has no GL
  interop calls) and copy them into CuPy
  arrays on the card. Runs on the GL thread with the context current. Falls back to the
  copy path when interop is unavailable (no CUDA, Mesa, a hybrid laptop).
- **CUDA -> NVENC without the host.** PyAV only takes host frames. NVIDIA's
  `PyNvVideoCodec` is said to take CUDA device memory (to confirm against current docs); the encoded H.264 packets are then muxed with
  PyAV as today. New optional dependency (`gpu` extra): owner approval needed.
- Order: interop first (removes the read-back on every sample), encoder second.

Tests:
- `test_interop_bridge_matches_copy_bridge` (`gpu`, `gl`): odd sizes, HDR values, float16
  edge values: bit-identical to the copy path (the existing
  `test_cuda_frames_reach_opengl_unchanged`, in the other direction).
- `test_interop_falls_back_cleanly`: no CUDA, or registration refused: the copy path runs
  and says so in `doctor`.
- `test_gpu_encoder_frames_equal_pyav_frames` (`gpu`): decoded frames from both encoders
  match within the codec's own tolerance (PSNR > 45 dB at cq 19), same frame count and
  timing.
- `doctor` reports which bridge and which encoder path are active.

Done when: bench shows it faster; otherwise record the numbers here and stop.

## Step 6 (candidate). Simulations on the card

The heat equation, waves and room acoustics step on the CPU (`domains/*_math.py`, pure
NumPy). The showcase hints at the cost: room_acoustics took about 830 ms per frame and
heat_dihedral about 590 ms, against 350 ms for fourier_linalg. Measure the "simulation"
stage with step 0 first; if it is large, move the `step` of those `Simulation`s into
registered kernels (NumPy reference = today's step, so the three-way kernel tests apply),
keeping fixed dt, the seed and the checkpoint cache (contract 3). Helps only those scenes.

## Rules for the whole plan

- One step per commit series; a step is done when its tests pass, the full suite passes
  (`uv run pytest`), the `gpu` tests pass on the RTX 4070 Ti, and the bench shows the gain.
- Every new kernel: NumPy reference first, then `CASES`, the emulator and the card.
- A speed-up may not change pictures. Steps 1-2 may differ by at most 1 code value (float
  rounding); steps 3, 4 and step 5's interop must be bit-identical; step 5's encoder
  switch is held to codec tolerance (PSNR > 45 dB at cq 19).
- Record each step's bench numbers in the table below, with the commit.

## Next session (written 2026-11-21, after step 2)

Start here. Steps 0, 1 and 2 are committed; the gate is green on the owner's PC.

1. **Re-measure first**: `uv run phasor bench --frames motion_gallery` and `prism_primes`.
   Expect about 164 and 46 ms/frame. If they differ a lot, find out why before changing code.
2. **Step 3a: reuse GL buffers** (no picture change; bit-identical). `draw` is now the
   biggest cost by far on motion_gallery (113 of 164 ms). Write the bit-identical test
   first (same frames before and after, `np.array_equal`), then the arena, then bench.
3. **Then 3b, 3c**, each with its bit-identical test first, measured on its own.

Open items:
- Done after step 2: sharp frames (`Engine.frame`: every frame where only life moves,
  which `is_moving` does not count) also widen on the card, through one `accumulate` with
  weight 1, bit-identical to the host path
  (`test_a_sharp_frame_through_the_card_is_the_host_path_exactly`). prism_primes
  66 -> 46.5 ms/frame. The inspector and stills keep `color_and_emission`.
- The emulator tests now run on the owner's PC (WinLibs g++, 2026-11-21): all 42 pass,
  `srgb8` and `accumulate` included.
- In a render the `srgb8` kernel takes 3-7 ms though it takes 2.2 ms alone (the card
  idles at low clocks between frames). Re-check after steps 3-4 before tuning it.
- Owner approval still needed before building step 4's on-disk bake cache (ADR 0003)
  and step 5's new dependency (`PyNvVideoCodec`).
- `benchmarks/2026-11-21_cuda_1920x1080.json` (a kernel bench from before this work) is
  untracked; the owner decides whether to commit it.
- Beauty upgrades: the owner has a list of about 10; deliberately not started.

## Status

| step | what | expected | measured | status |
| --- | --- | --- | --- | --- |
| 0 | frame bench | | 555 / 376 ms per frame (motion_gallery / prism_primes) | done |
| 1 | sRGB on the card | -130 ms | srgb 133 -> 5-12 ms, post 17 -> 10; frame 555 -> 425 / 376 -> 238 ms | done (the kernel alone is 2.2 ms; in a render the card idles at low clocks between frames, which steps 2-4 should fix) |
| 2 | widen + blur sum on the card | -200 to -260 ms | widen + blur sum 250 -> 12 ms, post 31 -> 3; frame 437 -> 164 / 238 -> 66 ms | done (sums on the card in float32; 8-bit output within 1 code value of the float64 host sum) |
| 3 | batched, cached drawing | draw 264 -> 30-60 ms | | to do |
| 4 | parallel workers | x3 wall time | | to do (ADR 0003 for the bake cache first) |
| 5 | GL-CUDA interop, GPU encoder | small | | measure first; new dependency needs approval |
| 6 | simulations on the card | those scenes only | | candidate: measure in step 0 |

Target after steps 0-4: motion_gallery from 571 ms to roughly 120 ms per frame, then about
3x more from workers; the showcase from 71 minutes to roughly 5-8.
