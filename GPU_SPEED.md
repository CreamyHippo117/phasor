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

## Step 8. Startup and the bake (2026-11-23, measured on Linux llvmpipe; untested on CUDA)

Plan written after finding that steps 0-4 left the *fixed* cost of a render as the largest part
of short renders, and that no GPU was at hand: only things that are CPU-side and exactly
repeatable are done here. The CPU kernels and C++ path belong to Milestone 14, not to this file.

Measured first (all examples; `load` = running the scene script, incl. its bakes):
- 19 of 22 examples load in under 0.25 s. Slow: `residue_theorem` 17.8 s (9 layout bakes of
  about 3.6 s each, one per verb that reads `state_at`), `motion_gallery` 6.7 s + 2.4 s first
  state, `prism_primes` 3.9 s (6.7 s of a profile in mpmath's `zetazero`).
- Frame reuse (skipping a frame identical to the last): dropped, 0 % of frames repeat (life).
- `run_doctor()` ran on every render (0.3 s here; on CUDA it also compiles a kernel).

Done, each bit-identical (finals pinned: `pc_tasks/pin_frames.py`, prism_primes, motion_gallery,
julia_explorer, 8-bit hashes equal before and after; a baked-layout hash of four scenes equal):
- `core/spatial.py`: `project`/`clip_w` build the homogeneous points without `np.c_` (the
  bake projects 258 000 tiny arrays); same matmul, same bits (`tests/test_speed_hotpaths.py`).
- `core/values.py`: `normalize` takes the plain-float case first (1.3 M calls a load).
- `layout/geometry.py`: the coverage summed-area table in int32 (same integers).
- `examples/prism_primes.py`: each zeta zero is found once, not once per approximation.
- `render/doctor.py` `cached_report()`: a render probes the machine once per process.
residue_theorem load 17.8 -> 16.2 s (profile: 31.7 -> 21.4 s), prism_primes 3.9 -> 3.2 s.

Not done, and why:
- The nine bakes while a script builds (`Scene._verb` calls `state_at`): letting verbs read an
  unbaked state would change what contract 7 says reads the bake; needs the owner. An
  incremental bake is the real fix (most verbs touch few objects).
- Worker start overlapping the parent's setup, and fewer workers for short renders: measure on
  the PC first (`pc_tasks/10_render_speed.md`, section 2).
- Step 5 (GL-CUDA interop, PyNvVideoCodec) and step 6 (simulations on the card): need the card
  and, for step 5, owner approval of the dependency.

## Step 7. Draft preset

A `draft` preset (`render/job.py`) for the quickest look at timing and layout: the owner allows
giving up quality on low-quality previews, never on `final` or `4k`. Each cut is a table entry
in `job.py`; a preset not listed in a table gets full quality, and `tests/test_draft_preset.py`
pins that `preview`, `final` and `4k` are unchanged and that their pictures are bit-identical
to before (hashes taken on the untouched tree, Mesa llvmpipe).

- Frame 640x360 at 15 fps, no motion blur (`PRESETS`, `BLUR`).
- Fast encode (`FAST_ENCODE`, `open_writer(..., fast=True)`): x264 `veryfast` crf 28, NVENC
  `p1` cq 30. Default writers are unchanged. NVENC settings untested on any card.
- Kernel-drawn fields at half resolution per side, stretched by the texture's linear filter
  (`FIELD_SCALE`, `field_scale` on `Engine.frame`/`frame_blurred` down to `GLRenderer._field_quad`;
  default 1.0 leaves every path as it was).
- Render workers get the field cut too: `parallel_frames`' source takes `field_scale` and passes it to
  each worker's `render_frame` (untested here beyond the unit tests; with the card, check on the PC).

Measured on this machine, which is a Linux box with Mesa llvmpipe and no GPU: NOT GPU data,
untested on CUDA and on a real GL card. `phasor render examples/prism_domain.py --workers 1`,
wall clock, run one after the other with nothing else running:

| preset | frames | size, fps | wall | per frame |
| --- | --- | --- | --- | --- |
| preview | 240 | 960x540, 30 | 291.2 s | 1213 ms |
| draft | 120 | 640x360, 15 | 43.6 s | 363 ms |

Draft is 6.7x faster on wall time; about 2x of that is only half as many frames (15 fps), the
rest is the smaller frame, half-resolution fields and the faster x264 setting. The file is
63 KB against 330 KB. (The in-test check, `prism_domain` 1 s: preview 30.2 s, draft 4.6 s.)
Each figure is one run, so treat the exact ratio loosely.

## Step 9. Kernel-drawn objects: stop doing their pixels on the CPU (2026-11-23, plan)

Plan written on the PC (RTX 4070 Ti) at `b579d7e`; approved by the owner the same day. No new
dependency. **9a and 9b are done** (results at the end of each); **9c is built** (untested
on CUDA); 9d is next.

### Where a frame goes now

Steps 1-4 brought the shared pipeline (readback, blur sum, post, sRGB, encode) to about 45-60
ms a frame. In every scene with a kernel-drawn object, the `draw` stage is now 90-98% of the
frame. That stage is CPU work inside the objects' pixel functions (`ctx.field`), and it runs
**once per motion-blur sample** (4 a moving frame in `final`). `phasor bench --frames NAME
--count 20 --preset final` (JSONs in `benchmarks/2026-11-23_frames_*_cuda_1920x1080.json`):

| example | draw | total | per blur sample | where the draw goes (cProfile) |
| --- | --- | --- | --- | --- |
| buddhabrot | 3035 | 3077 | ~760 | `cloud._colors` on 8 M points (LUT lerp, ~440 ms a sample); `cx.Fractal.orbits` culling and transforming 8 M points on the host (~245); uploading them again each sample |
| point_cloud | 3043 | 3138 | ~760 | `numpy.linalg.eigvals` re-solving ~1 M roots each sample while `twist` moves |
| interference | 1755 | 1895 | ~440 | `heatmap.shade_values`, `lut_colour` (wrap), `isoline_weight` (`np.gradient`), all float64; the `wave_field` kernel itself is ~9 ms |
| prism_fractal | 1280 | 1371 | ~320 | `cx.py` `fractal_glow`: the 4x supersampled field copied to the host and averaged in float64 (`.mean(axis=(1, 3))`) |
| prism_heat | 1125 | 1232 | ~280 | `heatmap.sample` (bicubic), `shade_values`, `lut_colour`, `isoline_weight` |
| bifurcation | 362 | 397 | ~90 | colours and cull on the host (cached orbits) |
| room_acoustics | 320 | 361 | ~80 | the pressure heat map (`heatmap`); simulation is 0.1 ms, so **step 6 is not needed** for it |
| prism_domain | 312 | 330 | ~78 | `gpu.render(supersample=)`: host float64 mean of the 4x tile |
| room_3d | 105 | 146 | | strokes (step 3's territory) |
| prism_primes | 17 | 66 | | reference: no kernel field |

So the lever is the same everywhere: the card makes the field in milliseconds, then the CPU
spends hundreds of milliseconds colouring, averaging, culling or re-solving it, four times a
frame. The sub-steps below move that work to the card. Each new kernel is a registered kernel
with a NumPy reference, so the Milestone 14 C++ CPU backend runs it too: machines without CUDA
get faster as well (compiled loops instead of float64 NumPy temporaries).

### 9a. Supersample on the card (expected: prism_fractal 1280 -> ~60 ms draw, prism_domain 312 -> ~30)

`gpu.render(supersample=s)` and `cx.Fractal`'s `fractal_glow` path copy the s x s enlarged
tile to the host and average it there in float64. Average before `to_host`, in float32, with
code that takes the array's own module (`cupy` or `numpy`: the CPU backend returns NumPy) so
both backends share it. A 2x2 box mean of four float32 values never needs float64 (the
consumer card's FP64 is slow anyway).

Tests: `test_device_supersample_matches_the_host_mean` (random tiles, s = 2, 3, 4; float32
result within 1 ulp-scale tolerance of the float64 host mean); the `prism_domain`,
`prism_fractal`, `deep_zoom` and fractal goldens unchanged; finals within 1 code value of
the old path (`pc_tasks/pin_frames.py` on three frames each, compared with a tolerance).

**Done 2026-11-23.** `gpu.box_mean` averages where the kernel left the picture, in the same
float64 arithmetic as before, so the bits do not change; `gpu.render(supersample=)` and
`cx.Fractal`'s glow path use it. prism_fractal 1371 -> 241 ms/frame (draw 1280 -> about 200),
prism_domain 330 -> 60 (draw 312 -> about 45). What is left of the fractal's draw is the escape
field going to the host and back (`_escape_field` returns a host array): a later trim.

### 9b. Point clouds stay on the card (expected: buddhabrot 3035 -> ~100 ms draw)

- Cache a source's points and colour values as device arrays, keyed as today
  (`source_points` key; bifurcation's orbit blocks likewise), so an 8 M point cloud is
  uploaded once, not once per sample.
- The `splat` kernel takes the colour value and reads the palette itself (`lut`, `hue`,
  wrap and lerp as `cloud._colors`): no (n, 3) colour array on the host.
- No host culling: the kernel already skips points outside its tile; the view transform
  (`center`, `zoom`, the rectangle's corner) becomes kernel parameters.
- Twinkle weights computed on the card from the cached (frequency, phase).
- Shared by `PointCloud` (all sources) and `cx.Fractal(kind="buddhabrot")` through
  `cloud.splat_pixels`.

Tests: the `splat` `CASES` gain the palette mode (three ways); the point_cloud, buddhabrot
and bifurcation goldens unchanged; finals bit-identical where the fixed-point splat allows
(the sum is already fixed point), else within 1 code value; `semantics` (`points`,
`in_rectangle`) unchanged, computed from the same cache.

**Done 2026-11-23, in a simpler form than planned.** `cloud.card_points` keeps a source's
points and values on the card (the latest two sets); placing, culling (`_inside`, the
Buddhabrot's view) and colouring (`_colors`, now `y - floor(y)` in place of `y % 1.0`, the
same bits) run there as CuPy operations, each one correctly rounded op as in NumPy, and the
`splat` launcher takes arrays already on the card. The palette was not moved into the kernel:
with the work on the card it costs a few milliseconds, and keeping NumPy's own operations
keeps the bits. Twinkle's `sin` and `color_by="phase"`'s `arctan2` stay on the host (libm and
CUDA differ in the last bit). buddhabrot 3077 -> 525 ms/frame (draw 3035 -> about 490); what is
left is the splat of 8 M points itself. Bifurcation keeps its host path (its points are
made per view). Tests: `tests/test_fields_on_card.py` (card and host pictures equal bit for
bit; `box_mean` and `_colors` equal the old code); finals of six examples hash-identical.

### 9c. One heat-shading kernel (expected: interference 1755 -> ~80, prism_heat 1125 -> ~60, room_acoustics 320 -> ~60)

`heatmap.py` is the shared path of `pde.Field`, `PressureSlice`, `Spectrogram`,
`ac.Interference` and `ac.PhasedArray`. Two registered kernels:

- `heat_sample`: the grid sampled at the pixels (`nearest`, `linear`, bicubic
  Catmull-Rom as `heatmap.sample`), for grid-based types.
- `heat_shade`: value -> colour (the `wrap` wheel with `hue_cycles` and `hue`, or the
  `linear`/`nearest` LUT), isoline weight from central differences (as `np.gradient` does
  inside, one-sided at the edges), weight, `contrast`, `ambient`, background and the edge
  `fade`; out (h, w, 4) with the glow weight in alpha, exactly `shade_values`' contract.
- `ac.Interference` chains `wave_field` -> `heat_shade` on the card; nothing returns to the
  host until the finished pixels.
- `colour_at` (a colour that depends on position, e.g. a spectrogram's hue up the frequency
  axis) becomes a kernel parameter (a hue ramp along an axis), or keeps the host path if a
  type needs something the kernel cannot express; list which types those are.

The NumPy references are today's functions, so the three-way `CASES` tests compare the
kernels against current behaviour. Tests: `test_heat_kernels_match_shade_values` over every
option combination the presets use; all prism heat goldens (`prism_heat`, `prism_drum_3d`,
`interference`, `room_acoustics`, spectrogram) unchanged; finals within 1 code value.

**Done 2026-11-25, as one kernel; untested on CUDA** (built on a Linux box with no GPU; the
card's checks are `pc_tasks/11_heat_kernel.md`). `heat_shade`
(`render/gpu/kernels/heat.py`) does the sampling, the colour, the isolines, the weight and
the fade in one pass, so nothing is stored between them: a pixel's neighbours (for the
isolines) are sampled again, or read from the image another kernel made. A type says how a
value becomes level, colour and weight with a `heatmap.Recipe` (the choices the five types
use; a function cannot run on the card). The maths moved from `domains/heatmap.py` into the
kernel's module as its reference, unchanged. `colour_at` is the recipe's `hue_of="height"`
(the spectrogram); no type keeps the host path, which stays for a scene's own types
(`heat_pixels` with plain functions).

The kernel works in double and in the reference's order of operations, with the pixel
coordinates computed as `heat_pixels` always did. On this box the compiled kernel (CPU
backend, g++ 13) drew all 48 heat-map fields of the examples (prism_heat, heat_dihedral,
interference, room_acoustics, prism_spectrogram; 5.8 M values) with **the same bits** as the
NumPy code; the reference path is the same bits by construction
(`tests/test_heat_kernel.py`: each type's old colouring written out and compared). On the
card `tanh`, `exp` and `pow` are not libm's, so expect the last bits to differ there.

`ac.Interference` and `ac.PhasedArray` run `wave_field` per tile and hand the result to
`heat_shade` where it is (`heatmap.field_pixels`): the field no longer comes to the host.

Measured here (4 cores, no GPU): one 1920x1080 prism heat map (smooth, 8 isolines) 778 ms
in NumPy -> 178 ms in the kernel on 4 threads (680 ms on one: five bicubic samples a pixel).
On the card: to measure. Double arithmetic is slow on consumer cards; if `phasor bench`
shows `heat_shade` above about 20 ms, a float32 variant is the next trim, and since it
would change pictures, for previews only (owner, 2026-11-26).

**Finals on the card must not change visibly.** On the card `tanh`, `exp` and `hypot` are
not libm's (double, last bits), so the float32 pixels may differ on rare values. By the
owner's rule (2026-11-26) `final` and `4k` frames must stay within 1 code value of before
(`pc_tasks/11_heat_kernel.md`, section 2); expected to hold easily.

### 9d. Polynomial roots on the card (expected: point_cloud 3043 -> ~150 ms draw while `twist` moves)

A batched Aberth-Ehrlich kernel, one thread per polynomial (degree <= 24), in place of
`eigvals` on companion matrices. **Every call starts from the same initial circle**, never
from the previous frame's roots: workers render frames in any order and contract 3 makes a
state a pure function of t. Fixed iteration cap plus convergence test, float64 inside (the
roots of +-1 polynomials cluster near the unit circle; check float32 is not enough before
choosing).

This changes the algorithm, so the reference is new and the tests say what correct means:
|p(z)| small for every root; the roots equal `eigvals`' as a multiset within 1e-9; a
cloud's picture within 1 code value of the eigvals picture; the point_cloud golden
unchanged; determinism (same t, same bits, any order). The `eigvals` path stays as the CPU
fallback where the kernel cannot run.

### 9e. Then measure again

The same ten-example bench, before and after, in the status table; plus the wall clock of
real `final` renders with `--workers 1` and `4`. Once the draw is on the card the four
workers share it, so per-frame bench gains will not all turn into wall-clock gains; that
measurement decides whether `scene.default_workers()` should change (`pc_tasks/cpu_workers.md`).

### Not in this step

- Step 5 (GL-CUDA interop, GPU encoder): the next lever after this one, still needs approval.
- Drawing a kernel field once per frame instead of once per blur sample would cut the draw
  another 4x, but it changes how a moving field blurs (a picture change): only with the
  owner's say-so, and as an opt-in per type.

## Step 10. Everything between a kernel and the finished frame (survey, 2026-11-25)

Asked by the owner: the card does its part of a frame in 1-2 ms, so what else makes a frame
slow, beyond heat maps? Sources: the saved CUDA frame benches in `benchmarks/` (RTX 4070
Ti), a cProfile of three `final` frames of eight examples on a Linux box with no GPU (so
its times are not the card's; it shows *which* code runs), and reading the path. **Nothing
below is built; items marked "card" cannot be measured or tested without it.**

The path of one kernel-drawn object, **once per motion-blur sample** (4 a moving frame):

| stage | what happens | cost | status |
| --- | --- | --- | --- |
| 1. kernel | the field on the card | 1-10 ms | fine |
| 2. host maths | colouring, averaging, culling, root finding in NumPy | 80-760 ms a sample | 9a, 9b, 9c done; 9d (roots) open |
| 3. card -> host -> card | `gpu.render` stitches tiles on the host, so a kernel's result is copied down and, for a second kernel, back up | fractal: 33 MB each way a sample (the escape field at 2x2 supersampling) | **10a** |
| 4. card -> host | the finished float32 pixels (33 MB at 1080p) | a few ms + | **10b** |
| 5. float32 -> float16 | `CopyBridge.upload`, one CPU core, also when the field did not change | 23 ms a field here (measured alone) | **10b** |
| 6. host -> GL texture | 16 MB | ~3 ms | 10b; step 5's interop removes it |
| 7. GL draw | | < 1 ms | fine |
| 8. readback, blur sum, post, sRGB, encode | shared by every scene | ~45 ms a frame | steps 1-4 done; step 5 next |

What is left after 9c, by scene (draw, ms/frame on the card, from the benches above):
prism_fractal ~200 (likely stages 3-5: to profile on the card), buddhabrot ~490 (the splat of 8 M points, 4 times),
point_cloud ~3000 while `twist` moves (9d), bifurcation ~360 (host colours and cull: its
points are made per view), prism_domain ~40 (stages 4-5), room_3d ~105 and room_acoustics'
lines (stroke building: `strokes.polylines` 1600 calls a frame; step 3b/3c).

- **10a. Chained kernels stay on the card.** `gpu.render(..., device=True)`: with one tile
  (every 1080p picture) return the kernel's array where it is; `cx._escape_field` passes it
  straight to `palette`/`fractal_glow`. Bit-identical (the same kernels, no arithmetic
  moved). Card.
- **10b. The field goes to GL as float16 made on the card.** Pixel functions may return a
  device array; `bridge` narrows and flips it there (`astype`, a call NumPy and CuPy share)
  and brings 16 MB to the host instead of 33; the renderer's field cache keeps the float16
  rows, so an unchanged field is not converted again. Changes what `ctx.field`'s `pixels`
  may return (`core/objects.py` says NumPy): the owner's OK, and a test that CuPy's
  float32 -> float16 rounds as NumPy's does (to nearest even). Card.
- **10c. One field per frame, not per blur sample.** It changes pictures (a moving field
  would blur differently, which can be seen), so by the owner's rule (2026-11-26) it is
  for previews only.
  Neither preview preset has motion blur (`BLUR`: one sample a frame), so today it would
  gain nothing there: **not built**. It becomes worth building only if a preview gets
  motion blur.
- **10d. Strokes.** Step 3b/3c (one draw per object, the command cache) for scenes made of
  lines; nothing new.
- **10e. Without a card.** `gpu.box_mean`'s float64 mean of the supersampled tile is the top
  line of prism_domain and prism_fractal on the CPU backend (250 and 950 ms a frame here):
  a box-mean kernel would run it on every core. Not the owner's machine.

Order suggested: 10a and 10b together on the PC (small, bit-identical, every field type
gains), then 9d. 9d replaces `eigvals` by another algorithm, so its roots differ in the
last bits: its finals must stay within 1 code value of the `eigvals` pictures (owner's
rule, 2026-11-26), else it is for previews only.

## Rules for the whole plan

- One step per commit series; a step is done when its tests pass, the full suite passes
  (`uv run pytest`), the `gpu` tests pass on the RTX 4070 Ti, and the bench shows the gain.
- Every new kernel: NumPy reference first, then `CASES`, the emulator and the card.
- **Owner, 2026-11-26: in `final` and `4k` a speed-up may change a picture only where the
  eye cannot see it**: every 8-bit value within 1 code value of before
  (`pc_tasks/pin_frames.py --compare`, on the card too). A change the eye could see (a
  field blurring differently, half-resolution fields) is for previews only (`draft`,
  `preview`: a table in `render/job.py`, like `FIELD_SCALE`), or is not done. Steps 1-2
  (within 1 code value) meet it and stay. Step 5's encoder switch is held to codec
  tolerance (PSNR > 45 dB at cq 19).
- Record each step's bench numbers in the table below, with the commit.

## Next session (written 2026-11-22, after step 4)

Start here. Step 4 (render workers, ADR 0003 accepted) is committed; the full suite is green.

What was learned building it, worth keeping:
- **Stretches must be short.** Frames are taken in order, stretch by stretch; with 30-frame
  stretches and a 3-frame queue each worker but one sat waiting (4 workers ran at one
  worker's speed). Now 4-frame stretches and a queue of about two stretches
  (`scene.WORKER_CHUNK`, `WORKER_BUFFER_BYTES`).
- **Spawn re-runs the parent's script** unless told not to: `scene._workers_skip_main`.
- **Workers rebuild the scene from the document**, so state a scene script sets up for its
  own types (a registry of drawings) was missing there: those objects drew nothing
  (owner's reverb_la, 2026-11-25). A worker now runs the scene script first when the
  scene came from one and names scene code (`scene._worker_engine`), and checks what
  scene-local objects draw at its first frame against the parent
  (`core.record.census`); a difference stops the render, naming the objects. Scenes
  without code of their own skip both (no cost).
- Two independent render processes on the RTX 4070 Ti each run at ~60 ms/frame (one alone:
  ~54): the card and driver share well; per-process CPU work is what scales.
- The fixed cost before the first frame was large: loading motion_gallery took 11.5 s,
  mostly SymPy's `lambdify` run on every draw (`geometry_math.compile_fn`, now cached) and a
  ComplexMap rebuilding its grid three times a moment (`cx._map_runs`, now cached). Load
  5.5 s, bake 1.5 s now. Other examples may have the same kind of cost: profile `load`.

Candidates next:
- With 4 workers, motion_gallery's frames take ~17 s of a 34 s render; the rest is the
  parent's load, bake and doctor, and workers starting. Profile `load` of the other
  examples, and consider starting the workers while the parent bakes (they need the bake
  only for their first frame).
- In-process items from before (PyAV's RGB -> YUV holding the GIL; readback 15 ms) now
  matter less: they are spread over 4 workers.

Open items:
- Done after step 2: sharp frames (`Engine.frame`: every frame where only life moves,
  which `is_moving` does not count) also widen on the card, through one `accumulate` with
  weight 1, bit-identical to the host path
  (`test_a_sharp_frame_through_the_card_is_the_host_path_exactly`). prism_primes
  66 -> 46.5 ms/frame. The inspector and stills keep `color_and_emission`.
- The emulator tests now run on the owner's PC (WinLibs g++, 2026-11-21): all 42 pass,
  `srgb8` and `accumulate` included.
- Lit surfaces (`ParametricSurface`, `RiemannSurface`) are not bit-repeatable on the RTX
  4070 Ti: rendering the same state twice in one process differs by one float16 step
  (<= 0.0005) on a few hundred pixels of the color and emission targets; id and solo are
  exact. Found while checking 3a; it predates it (old vs old differs as much as old vs new).
  Cause not found (the lit shader has nothing obviously order-dependent); likely the driver.
  Contract 3 (time is pure) holds for the document and state, not for these last bits.
  Golden tests tolerate it; bit-identical speed tests must leave lit surfaces out.
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
| 3 | batched, cached drawing | draw 264 -> 30-60 ms | 3a (reused GL buffers): draw 113 -> 92 ms, frame 164 -> 143 / 47 ms, pictures unchanged | 3a done; stroke builder rewritten (bit-identical): draw 92 -> 57 ms, frame 143 -> 105 ms; FlowParticles batched (pixels identical): draw 57 -> 40, frame 105 -> 99; 3b/3c not needed now |
| 3+ | async readback, encoder thread | | readback 21 -> 15 ms, frame 99 -> 93 (bench); encoder thread: wall 82 -> 76 / 52 -> 44 ms | done (wall clock: the bench's waits hide overlap) |
| 4 | parallel workers | x3 wall time | full final render of motion_gallery 61.5 -> 33.8 s (4 workers, CLI), packaged app 50.6 -> 24.0 s; prism_primes 34.8 -> 23.7 s; steady 70.6 -> 23.2 ms/frame | done 2026-11-22 (ADR 0003 option A; default 4 workers: 6 and 8 were no faster) |
| 5 | GL-CUDA interop, GPU encoder | small | | measure first; new dependency needs approval |
| 6 | simulations on the card | those scenes only | room_acoustics: simulation 0.1 ms, draw 320 ms (2026-11-23) | not needed for room_acoustics; its cost is the heat map (step 9c) |
| 9 | kernel-drawn objects on the card | draw 300-3000 -> 30-150 ms | 9a: prism_fractal 1371 -> 241, prism_domain 330 -> 60 ms/frame; 9b: buddhabrot 3077 -> 525 ms/frame; finals bit-identical (pin_frames on six examples) | approved 2026-11-23; 9a, 9b done; 9c built 2026-11-25 (same bits on the CPU backend; untested on CUDA: `pc_tasks/11_heat_kernel.md`); 9d next; then step 10 |

Target after steps 0-4: motion_gallery from 571 ms to roughly 120 ms per frame, then about
3x more from workers; the showcase from 71 minutes to roughly 5-8.
