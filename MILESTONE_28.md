# Milestone 28: world-space batches for the Explorer's live path

Branch: the worktree branch of this session, from `claude/eloquent-mayer-jiqxs9` at `9fa24f7`
(Milestones 21, 22, 24, 26, 27 and fixes merged). Plan: `EXPLORER.md`, advice
`docs/decisions/explorer_advice_2026-11-24.md` sections 1.4 and 1.5.

Exit condition: flow objects draw through `ctx.points3d/lines3d/arrows3d`; the camera is a
uniform, so an orbit uploads nothing and a flow-time step uploads one positions array per
moving batch (counted, not estimated); GL and the CPU expansion agree; hostile arrays lint,
never raise; frame rates measured on this box, with the PC's left to `pc_tasks/12`.
**Built and tested on the cloud box; not independently audited yet.**

## Built

| What | Where | Notes |
| --- | --- | --- |
| The batch model and its NumPy expansion | `phasor/core/batches.py` | `make_points`, `make_lines`, `make_arrows` sanitise input into a `Batch` (float32, C-contiguous, finite; one colour or size kept once as a single row; a single strip needs no strip array) and list every repair. `instances`/`expand` are the vertex shader in NumPy: 12-gon dots (`disc_batch`), mitered strips with miter limit 4 (`polylines`), flat arrows (`arrow_batch`), 8-sided cones (`cone_batch`). Units `world`, `height` (share of the frame height), `px`; measured per item or at an `anchor`. `Batch.digest` (BLAKE2b-128 per array) and `frame_key` |
| The protocol | `phasor/core/objects.py` `DrawContext` | `points3d(centers, radius=, units=, colors=, opacity=, glow=, anchor=)`, `lines3d(points, width=, ..., lengths=)`, `arrows3d(tails, tips, width=, ..., shape="flat"|"cone", head=)` |
| Recorder (`passes=None`, lint, inspector) | `phasor/core/record.py` | records the batch points (bounds), each line strip as a stroke (`start`/`end`/`curve` for one strip), the repairs (`Recording.issues`); derived semantics gain `batch_input` only when something was repaired |
| GL path | `phasor/render/passes.py` | `_VS_BATCH` with `POINTS`, `LINES`, `FLAT`, `CONE`: vertex pulling from shader storage buffers by `gl_InstanceID`/`gl_VertexID`, instanced draws of an empty vertex array over the template triangles; same fragment shaders (colour, id, solo) with `BATCH` (flash tint for one-colour batches). Storage buffers are kept by content (dtype, bytes, digest), let go of after 32 renders unused or past 512 buffers / 768 MB (`batch_memory()` reports them), 8 spares refilled instead of re-created, never evicting one drawn in the current render. `GLRenderer.uploads` counts `batch` writes and bytes and `arena` refills. `expand_on_cpu` draws the CPU expansion through the plain mesh path (the reference picture). `batch_vertices` reads the shader's world vertices back by transform feedback (tests) |
| Quality hint | `passes.py`, `engine.py` | `Engine.frame/still(..., quality="full"|"interactive")` -> `start_color_raw`/`frame_plan` -> `_Collector.quality`; colour pass only (id and solo, hence the inspector, always full); in the frame-plan digest; checked (`ValueError`). Half size while dragging is the caller's job (documented in `Engine.frame` and API.md) |
| Frame key | `render/framekey.py` | an object with `frame_key()` (a batch) is digested by it: arrays by their hash, computed once and shared with the buffer cache |
| Theme hue on batches | `passes.py`, `core/color.py` | a glowing one-colour `lines3d` runs the theme's `[strokes]` hue like `strokes.polylines`; `hue_colors`/`lut_colors` moved to `core/color.py` (render may not import domains), `strokes.py` names them as before |
| Flow objects moved | `phasor/domains/flow.py` | `System` frame, `FieldArrows` (flat and cone), `Probe` path and dot, `FlowSet` points, trails (one strip per point, fading in) and outline (all loops one batch), `Equilibria`, `Manifold` (one batch per branch, the old order). Opacity is a uniform, sizes are stroke roles at the window's middle (`anchor`), so nothing they hand over depends on the camera. `EquilibriumView` stays on `strokes.py` (its ring faces the camera: camera-dependent by nature); `dyn.*` and every other type keep `strokes.py` |
| Thinning | `flow.py` | `INTERACTIVE_ARROWS = 4000` (the grid is coarsened), `INTERACTIVE_POINTS = 5000` (the existing `MAX_DRAWN_POINTS` stride with a smaller cap) |
| Lint | `inspect/lint.py`, `style/theme.py`, four themes | `batch_input` (warning; threshold `[lint] batch_repairs = 0`, default added for themes without it): lists the repairs, with times |
| Speed found while measuring | `flow.py`, `flow_math.py` | a set's trail is kept per runner (an orbit no longer steps 10^4 points 8 steps every frame); `System.key` caches the expressions' `srepr` (7 resolves a frame were 8 ms of SymPy printing) |
| Bench | `render/bench.py`, `cli.py` | `phasor bench --explorer [--width --height --count]`: the advice 1.5 cases (`cli._explorer_cases`), each motion (`orbit`/`pan` move only the camera, `flow` advances time) in three ways (shader, CPU-built, interactive), draft look, sRGB read back; ms median, fps, batch uploads per frame, load average; saved to `benchmarks/` |
| Docs | `docs/ai/API.md`, `SPEC.md`, `EXPLORER.md`, `pc_tasks/12_world_batches.md`, `pc_tasks/README.md` | API section "World-space batches" (a running example), the flow section, the lint table; SPEC "Final renderer" bullet |

## Proofs (measured)

**(a) Uploads, counted** (`GLRenderer.uploads`, `tests/test_batches.py`):

- `test_orbiting_uploads_nothing_and_flow_time_uploads_one_positions_array`: the M26
  `lorenz_explorer` scene at t = 4: after the first frame, eleven more azimuths (a full turn)
  upload **0** batch buffers; the next flow-time frame uploads **exactly 1** (the ball's
  2000 positions; cones, equilibria and the frame unchanged).
- `test_a_growing_path_uploads_only_its_positions`: a probe whose path grows each frame
  uploads exactly 2 arrays a frame (the path's points and the dot's), not its colour, width
  or strips.
- `test_a_batch_keeps_nothing_that_depends_on_the_camera`: the flow objects' batch keys are
  equal from two azimuths.
- The bench's `uploads/frame`: 0.0 for every orbit and pan; flow frames 1 (10^5 set),
  2 (Lorenz set and its trails), 4 (2D: set points, outline, probe path, probe dot).

**(b) Frame rates on the cloud box.** **Cloud-box numbers**: 4 shared cores, no GPU, OpenGL
by llvmpipe (Mesa 25.2.8, LLVM 20), load average 3.1-3.9 during the run (other sessions
share the box), 640x360, draft look, 30 frames after one warm-up, median.
`benchmarks/2026-11-24_explorer_view_cpu_640x360.json`; command
`uv run phasor bench --explorer --width 640 --height 360 --count 30`.

| Case | Motion | shader | CPU-built triangles | shader, interactive | advice 1.5 estimate (cloud) |
| --- | --- | --- | --- | --- | --- |
| 2D portrait, `dyn.PhasePortrait` (strokes: today's path) | pan | 13.9 fps (71.9 ms) | - | - | 4-9 fps (measured in the window, with Qt) |
| 2D field, 2500 arrows + 2000 points + outline + probe | pan | 23.4 fps (42.8 ms) | 19.7 fps | 20.2 fps | 15-30 fps |
| same | flow | 19.3 fps (51.9 ms) | 19.2 fps | 19.2 fps | 15-30 fps |
| 3D Lorenz, 4913 cones + 10^4 tracers with 8-step trails | orbit | 4.2 fps (237 ms) | 4.2 fps | 7.7 fps (130 ms) | 8-15 fps |
| same | flow | 3.1 fps (326 ms) | 2.8 fps | 4.4 fps | 8-15 fps |
| 3D Lorenz, 343 cones + 10^5 tracers | orbit | 3.5 fps (284 ms) | 3.0 fps | 10.9 fps (92 ms) | - |
| same | flow | 2.9 fps (344 ms) | 2.4 fps | 6.2 fps (162 ms) | 10-20 fps at one step a frame |
| 10^6 tracers | - | not run (the CPU cap is 2*10^5 points) | | | no |

What these say. On llvmpipe the card is the CPU: a profile of the Lorenz orbit (load 5-6)
put 220 ms of 620 in `vao.render` and 205 ms in the MSAA resolve (`copy_framebuffer`), the
rest in Python `draw`. Building triangles in NumPy each frame costs about as much as
llvmpipe's vertex stage, so shader and CPU-built paths are close here (1.0-1.3x); the gain
the milestone is for is on a real card, where `GPU_SPEED.md` measured the card at 1-2 ms a
frame and Python stroke building at 46-66 ms. The advice's 8-15 fps estimate for the Lorenz
case is **not reached on this box** (4.2 fps; 7.7 interactive): its triangle count (about
360,000 with 4x MSAA) is llvmpipe's limit, not the batches'. The interactive hint is the
CPU's lever (1.8-3.1x in 3D). PC numbers: none claimed; `pc_tasks/12_world_batches.md` has
the commands and an expected-against-measured template.

**(c) Bounds at the documented maxima** (cloud box, load average 3.0-3.4, 640x360, wall
clock; `/tmp` script in the session, numbers here; `test_batches_at_the_documented_maxima_are_bounded`
checks the host side with generous bounds):

| Batch | Items | First frame | Again (nothing uploaded) | Uploaded to the card | Host peak (tracemalloc) |
| --- | --- | --- | --- | --- | --- |
| lines (points x trail at its cap: 62,500 strips of 64) | 4,000,000 | 3.3 s | 2.5 s ortho, 2.9 s orbit | 61 MB (positions 48 MB + strips 16 MB; one colour, one width) | 191 MB |
| points (a set at `flow_math.MAX_POINTS`) | 200,000 | 0.7 s | 0.6 s | 5.3 MB | 32 MB |
| arrows (`flow.MAX_ARROWS`) | 20,000 | 0.10 s | 0.09 s | 0.5 MB | 32 MB |

The storage-buffer cache is bounded (512 buffers, 768 MB). A 4e6-point batch is under the
128 MB storage-block size llvmpipe reports; the GL spec only promises 16 MB
(`GL_MAX_SHADER_STORAGE_BLOCK_SIZE`): **untested on NVIDIA** at that size (`pc_tasks/12`
exercises the 10^5 case, not the 4e6 cap; a driver with a smaller limit would need the batch
split, which is not built).

## Tests

`tests/test_batches.py` (52, all pass here, OpenGL on llvmpipe): the CPU expansion against a
scalar per-vertex reference written the way the shader is (exact to 1e-12; 4 kinds x ortho,
perspective x world, height, px, anchored); the expansion against the triangles
`strokes.polylines`, `disc_batch`, `arrow_batch`, `cone_batch` drew (same triangles); the
vertex shader against the CPU expansion by transform feedback (within 2e-6 relative,
float32); GL against CPU-expanded pictures for each kind and camera (the test allows 24/255
on under 0.2% of pixels and 99.8% of the id buffer; measured on llvmpipe: identical, colour
and id, all 8 cases); 14 hostile inputs (NaN, inf,
strings, complex, object, 4-column points, wrong-length colours and widths, negative and
infinite widths, non-contiguous, bad units, anchor, opacity, glow, lengths, shape, head) never
raise, and lint `batch_input` names each object in both MODES (`passes=None` and GL); a huge
batch is cut and noted; recorder bounds and features; the upload counters above; the quality
hint (thinned field, strided set, semantics unchanged, plan key differs, bad value refused,
id pass unchanged); bounds at the maxima (slow); the live-view bench; a 120-frame flow keeps
a bounded number of buffers.

Suites run here, one at a time: `test_batches` 52, `test_flow` 69 (with the slow golden),
`test_flow_parser` 36, `test_dynamics` 18, `test_render` 22 (+1 skipped), `test_draft` 66,
`test_cpu_backend` 59, `test_beauty` 31, `test_strokes` 31, `test_lint` 32, `test_layers` 9,
`test_contracts` 9, `test_docs` 5, `test_final -k "not slow"` 470 (+8 skipped),
`test_explorer_rows` 60, `test_explorer_tools` 13, `test_explorer_audit` 14,
`test_open_contract` 70, `test_room3d` 21, `test_frame_bench` 2, `test_proc` 2,
`test_mcp -k llms` 1. `ruff check`, `ruff format --check` and `lint-imports` clean. The
whole suite was not run in one go (shared box).

Goldens: compared frame by frame against renders saved before any change (480x270,
`pendulum_flow` at 4 times, `lorenz_explorer` at 3, the `test_flow` scene): **byte-identical**
except the Lorenz system's frame hairlines (below). `tests/golden/software/lorenz_explorer.png`
regenerated and looked at before and after (indistinguishable by eye: 409 pixels of 461,760
differ, all on the 12 hairline edges of the box). `phase_portrait` (old `dyn` objects) is
untouched and passes. There is no `pendulum_flow` golden and no GPU flow golden
(`pc_tasks/12` makes the GPU one).

## Deviations

1. **"The software backend (`passes=None`)" is the NumPy expansion, not a second
   rasteriser.** phasor has no CPU rasteriser: `passes=None` drives the recorder (bounds,
   features, semantics, lint) and `render/cpu/backend.py` compiles kernels; neither draws
   triangles. A NumPy rasteriser would be a second renderer ("one way to do each thing").
   So: `core/batches.py` is the expansion every platform runs (the recorder and lint use the
   same sanitiser on Windows CI without OpenGL), and "GL vs software images" compares the
   shader's pictures with the CPU-expanded triangles drawn through the plain mesh path
   (`GLRenderer.expand_on_cpu`); both need OpenGL, so those tests are `gl`. The exact
   CPU-vs-reference test runs everywhere. Redirect if a real CPU rasteriser was meant.
2. **Buffers are keyed by content, not by naming the parameters.** The renderer hashes each
   array (BLAKE2b, 83 ms for 48 MB here) and keeps buffers by hash. Since the arrays are a
   function of the object's parameters (the camera never enters), this is "keyed by the
   object's params" without trusting a type to list them, and it can never serve a stale
   buffer. The cost is the hash per frame of what is drawn.
3. **All lines are mitered strips**, two-point strips included (one way: `polylines`'
   maths). The `System` frame was `segment_batch` (sides from the view ray to the segment's
   middle); in perspective its 1-pixel hairlines move by antialiasing (the regenerated
   golden). `FlowSet` trails were per-segment fades with `segment_batch`; they are now one
   strip per point with the fade per sample (no golden has trails).
4. **"One positions array per batch"** holds when the batch's item count is fixed or it is
   one strip; a set's outline with several loops that is being refined also re-uploads its
   strip array (the loops' boundaries move). Counted in the bench (`uploads/frame`).
5. **The flash verb now brightens one-colour dots and frames.** `disc_batch` and
   `segment_batch` passed per-vertex colours, which the mesh path does not tint; a batch given
   one colour is tinted like a plain mesh. No scene flashes a flow object; pictures without
   a flash are unchanged.
6. **Not every flow object moved.** `EquilibriumView`'s ring and card face the camera (the
   geometry is the camera's), so it stays on `strokes.py`; in the counter test it is
   hidden. Its translucent subspace disc was already a camera-free mesh.
7. **The advice's cloud estimate for 3D Lorenz (8-15 fps) is not reached** (4.2; 7.7
   interactive), see (b).
8. **`batch_input` reaches types whose semantics are derived** (scene-local types, ADR
   0005), through the recorder. A type that writes its own `semantics` (every flow type) gets
   its input repaired but no lint, so it must hand the batches clean arrays: the flow types
   filter non-finite points first. Gated so that lint never draws objects it did not draw
   before (no extra cost for every scene).
9. **Buffer lifetime.** Storage buffers are kept by content and let go of when not drawn for
   `BATCH_KEEP_RENDERS` (32) renders of any pass, or past 512 buffers / 768 MB; 8 spares are
   refilled instead of re-created. A long render of a moving 10^5-point set therefore keeps
   about 32 frames of its positions (32 x 1.2 MB) per renderer, i.e. per render worker, not
   the whole render (`test_a_long_flow_keeps_few_buffers_on_the_card`).

## Deferred

- **Skipping `draw()` when only the camera moved.** The batches make it possible (their
  arrays do not depend on the camera), but other draw calls (strokes, text) do, and `draw`
  may read anything; it needs an opt-in like `GPU_SPEED.md` step 3c. Python `draw` is now
  the largest cost a real card will see; for Milestone 29.
- **MSAA off in drafts/the viewport** (M21 deferred it): the MSAA resolve is 205 ms of a
  Lorenz frame on llvmpipe.
- **Text and `EquilibriumView` on batches**: camera-facing by nature; unchanged.
- **A GL renderer without shader storage buffers** (GL < 4.3, macOS's 4.1): the batch path
  needs them; the shaders already need `#version 430`, so nothing new is lost, but there is
  no automatic fallback to `expand_on_cpu`.

## Untested

Windows (CI has no OpenGL: the GL tests skip there; the pure-NumPy and lint tests are written
for it), macOS, CUDA, and **any timing on a real GPU** (`pc_tasks/12_world_batches.md`). The
transform-feedback and picture-agreement tests have run only on llvmpipe. The 4e6-point batch
against an NVIDIA storage-block limit.
