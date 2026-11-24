# Milestone 21: the draft render

Plan: `STUDIO.md`, Milestone 21. Asked for by the owner on 2026-11-24: "a button that lets
you do a low quality preview render. no blur, motion blur, haze, etc. 15 fps, ridiculously
efficient in other ways however you choose". Built 2026-11-24 on branch
`claude/m21-draft-render`, branched from `claude/eloquent-mayer-jiqxs9` at `fa4232d`.

Exit condition: **a draft render of an example scene at 15 fps with no post effects, visibly
the same picture, several times faster than preview, cancellable with progress, from the CLI,
the Workspace tool and the Engine API.**

Status: **built; audited independently (2026-11-24): 1 blocker + 4 should-fix + 6 nits
fixed, each with a regression test that failed before the fix (see "Audit"); focused gates
green on Linux (tier B: llvmpipe, CPU kernels).** The full suite is left to CI. Windows,
macOS, CUDA, NVENC and the desktop dialog by hand untested.

## Built

| Area | Where | What |
| --- | --- | --- |
| The preset table | `phasor/render/presets.py` | `Preset` (size: `width` x `height`, or a `long_side` keeping the scene's aspect with even sides and never enlarging; `fps`; `shutter`, `blur_samples`; `effects`; `life`; `dedup`; `encoder`) and `PRESETS`: `draft`, `preview`, `final`, `4k`. `Effects(bloom, lens, fog)`: `FULL`, `DRAFT`. `preset_named` and `check_fps` (1/1001 to 1000, finite) give clean `ValueError`s. `render.job.BLUR` is read from it |
| The plain look | `render/post.py`, `render/passes.py`, `engine.py` | `finish(..., effects=)` skips the bloom pyramid and the lens (fringe, vignette); the tone map and sRGB encode run once as always. `GLRenderer.start_color_raw(fog=, emission=)`: no depth fog on lit surfaces, and the emission target is not read back or widened when nothing blooms. `Engine.frame(..., effects=)`, `frame_blurred(..., effects=)` |
| No decoration | `engine.py` | `Engine.without_decoration()`: a fork (`fork` + `apply`) with `meta.life="off"` and, on an orbit camera that takes the theme's drift, `drift=0`; made once per version of the document (keyed by `fingerprint`) and kept; the engine itself when there is nothing to strip. Its layout bake is made in it, once, like any engine's (never per frame; workers get its bake, ADR 0003) |
| Frame de-duplication | `render/framekey.py`, `render/passes.py`, `render/job.py`, `scene.py` | `GLRenderer.frame_plan(state, size, fog=)` records each object's color-pass draw calls (no GL) and a 256-bit BLAKE2 digest of everything the drawing depends on: the calls (arrays by bytes), draw order, each object's id, type, layer, opacity, flash and camera, the light, the background, the stroke hue, the size and the fog flag; kernel fields by type, parameters, roles, style and quad (as the field cache keys them). `render.job.FrameDrawer` repeats the last frame's bytes when the key is the last one's, and otherwise draws with the recorded calls (`Engine.frame(plan=)`), so `draw` runs once per frame either way. Workers: a worker starting a stretch takes the frame before it as "last" (`prime`) and sends `("repeat", k)`; the parent repeats its last bytes. Only with `dedup` (the draft) and never with motion blur. Unknown values in a digest get a token that never repeats (a frame is drawn rather than wrongly repeated) |
| Encoder | `render/encode.py` | `open_writer(..., speed="fast")`: libx264 `ultrafast`, CRF 26, no B-frames, a keyframe a second; NVENC `p1`, CQ 28, same; PNG sequence at zlib level 1. `speed="quality"` (every other preset) builds exactly the old command line and options |
| Progress, cancel, window | `render/job.py` | `render_file(cancel=)`: a `threading.Event` or a callable, checked before the first frame and between frames; cancelling aborts the encoder, removes what the render wrote (the video file, or the PNGs it wrote, never the folder or other files) and raises `RenderCancelled` (a `RuntimeError`). `progress(k, n)` as before. `start`/`end`/`frame` take seconds or a marker name (`resolve_time`); non-finite or unknown is a `ValueError` naming the markers; out-of-range times keep rendering as before. Results add `preset`, `drawn`, `repeated` |
| Surfaces | `cli.py`, `scene.py`, `engine.py`, `workspace/tools.py`, `gui/desktop.py`, `render/doctor.py`, `render/bench.py` | `phasor render --preset draft` and `phasor preview SCENE` (the same options, default `draft`; registered in `cli.py`, which owns `render`); a draft's default file is `<scene>.draft.mp4`; `--start`/`--end`/`--frame` take markers; the summary line says the preset, the repeated frames and the wall time. `Engine.render(out, preset=, ...)`; `Scene.render(out, preset=, workers=)` and `scene.render(engine, ...)` (resolve the draft engine before building `parallel_frames`, so workers draw it; the CLI uses it). The Workspace/MCP `render` tool accepts `draft` and markers, writes `<scene>.draft.mp4`. The desktop export dialog lists Draft (default still Preview). `phasor doctor` prints the preset table (and `--json` has `presets`). `phasor bench --frames --preset draft` and a `dedup` stage |
| Docs | `docs/ai/API.md`, `docs/ai/MCP.md`, `llms.txt` | "Render presets and the draft render": the preset table, the effect table below, how de-duplication decides, a running example; the command table; the MCP tool row |
| Tests | `tests/test_draft.py` (66), `tests/test_desktop.py` (+1), `tests/ssim.py` | below |

### The effect table: what the draft skips and keeps

| Effect | Where | Draft |
| --- | --- | --- |
| Bloom (bright pass, blur pyramid, sum) | `render/post.py`, theme `[bloom]` strength, levels | skipped; the emission target is not read back |
| Lens: fringe, vignette | `render/post.py`, theme `[bloom]` fringe, vignette | skipped |
| Tone map (exposure, knee) | `render/post.py` | kept, once (contract 4) |
| sRGB encode | `render/post.py` (`srgb8` kernel), `render/output.py` | kept, once |
| Depth fog ("haze") on lit surfaces | `render/passes.py` `_draw_lit`, theme `[light]` fog | skipped |
| Lit surfaces: three-point light, specular, glass fresnel, wire grid | `render/passes.py`, theme `[light]` | kept |
| Glow (HDR paint above 1) | `render/passes.py`, theme `[glow]` | kept in the picture; its halo is the bloom (skipped) |
| 4x MSAA | `render/passes.py` | kept (thin strokes break up without it) |
| Hue along strokes | theme `[strokes]` hue_along, `palette` | kept (the prism look's color) |
| Hue flowing along strokes over time | theme `[strokes]` flow (only `prism_vivid`) | kept (see Deferred) |
| Kernel pictures and their glow-weight "haze" | `render/gpu/kernels/` | kept, at the draft's pixel size only |
| Motion blur | `Engine.frame_blurred` | skipped |
| Life: glow breathing, shimmer, drifting dots and equations, field hue turning, twinkle | `core/life.py`, theme `[life]`, scene `life` | skipped (`life="off"`) |
| The theme's camera drift | `time/compile.py` `sway`, theme `[camera]` drift | skipped when `drift` < 0 (the theme's); an explicit drift is kept |
| Verbs, tracks, expressions, updaters, simulations, flash | theme `[motion]`, the document | kept |
| Narration and sound clips | `Engine.audio_track` | kept |

## Numbers (this container: 4 cores, no GPU, llvmpipe, CPU kernels, ffmpeg libx264)

Whole renders, wall clock, one process, after `load` and the scene's own bake; a draft's one
fork bake is inside the clock. `final` is a 2 s window (4-6 s); a whole final takes minutes.
`benchmarks/2026-11-24_draft_render_cpu.json`.

| Scene | Draft (640x360, 15 fps) | Preview (960x540, 30 fps) | Final (1920x1080, 60 fps, blur) | Draft vs preview, same scene |
| --- | --- | --- | --- | --- |
| `prism_primes` (13 s) | 195 frames in 4.8 s: **40.5 frames/s**, 2.7 s of video per second | 390 in 73.0 s: 5.3 frames/s, 0.18 s/s | 120 in 83.9 s: 1.43 frames/s, 0.024 s/s | **15x** |
| `motion_gallery` (12 s) | 180 in 9.4 s: **19.3 frames/s**, 1.28 s/s | 360 in 74.2 s: 4.9 frames/s, 0.16 s/s | 120 in 158 s: 0.76 frames/s, 0.013 s/s | **7.9x** |
| `basics_3d` (12 s, 3D) | 180 in 15.3 s: **11.8 frames/s**, 0.79 s/s | 360 in 91.2 s: 3.9 frames/s, 0.13 s/s | 120 in 239 s: 0.50 frames/s, 0.008 s/s | **6.0x** |
| `point_cloud`, window 3.0-3.4 s (kernel/data-bound) | 6 frames in 48.5 s: **8.1 s a frame**, 0.008 s/s | 12 in 76.2 s: 6.4 s a frame, 0.005 s/s | not measured | **1.6x** (only the halved frame rate) |

**The draft is not fast everywhere.** Its savings are in pixels and post effects. In
`point_cloud` the time goes into solving the roots of 65,536 polynomials (a million points)
whenever `twist` moves, which does not depend on the picture's size: a draft frame costs as
much as a preview frame (8.1 s against 6.4 s here; the draft's includes its one fork bake),
and only the 15 fps gives 1.6x. The splat itself does follow the draft's pixel size. A cheap
safe way to make that solve smaller in drafts (fewer points) would change what the scene
draws, so it is left as a limit (API.md says so).

The `prism_primes` preview row ran once beside the docs tests (91.4 s) and once alone
(73.0 s, the row above).

Per frame (`phasor bench --frames`, the middle of the scene, setup excluded), before
(preview) and after (draft); `benchmarks/2026-11-24_frames_*`:

| Scene | preview ms/frame (draw, widen, post, srgb) | draft ms/frame (dedup = recording the draws, draw, widen, post, srgb) |
| --- | --- | --- |
| `motion_gallery` | 102.0 (23.9, 12.7, 50.1, 5.6) | 45.5 (8.8, 20.4, 4.0, 3.0, 5.9) |
| `basics_3d` | 125.1 (47.1, 15.4, 46.3, 7.9) | 71.8 (30.3, 25.3, 2.7, 2.0, 7.8) |

The `dedup` stage is mostly the objects' own `draw` recording (moved out of `draw`): the
digest itself measured 2.3 ms a frame on `basics_3d` and 0.7 ms on `prism_primes`.

**De-duplication** (draft with and without it, whole scenes, PNG sequences compared byte for
byte; the fork's bake outside the clock):

| Scene | Frames | Repeated | With | Without | Byte-identical |
| --- | --- | --- | --- | --- | --- |
| `prism_primes` | 195 | 92 (47%) | 5.18 s | 8.29 s | yes |
| `prism_domain` | 120 | 99 (83%) | 3.47 s | 5.00 s | yes |
| `residues` | 38 | 15 (39%) | 1.27 s | 1.40 s | yes |
| `motion_gallery`, `basics_3d` | 180 | 0 | - | - | - |

`motion_gallery` and `basics_3d` repeat nothing, and rightly: flow particles and flow lines
draw from the clock in every frame, and `basics_3d` orbits its camera through the whole
scene (`/tmp` diagnosis: the camera and the `flow_lines` calls differ between every two
frames).

**Workers** (draft, this 4-core box): `basics_3d` 13.0 s with 1 worker, 11.9 s with 2, 14.1 s
with 4; `prism_primes` 5.6 s, 6.7 s, 9.1 s. Starting processes costs more than it saves on 4
cores; `default_workers()` is 1 here, which is what the CLI uses.

**Byte-for-byte check of preview and final**: frames 4.0-4.1 s of `prism_primes`,
`basics_3d` and `motion_gallery`, preview and final, as PNG sequences and `.mp4`, rendered
before any change and after: all 24 files have the same SHA-256 (two runs before agreed
with each other first).

## Audit (independent, 2026-11-24) and fixes

The audit confirmed the core (preview and final byte-identical, dedup 0 mismatches over 32
examples, cancel and Ctrl-C cleanup, the fork never leaks). Each fix has a regression test
that fails on the audited commit and passes now (`tests/test_draft.py`, the block "the
independent audit's findings", `tests/test_contracts.py`, `tests/test_desktop.py`).

| # | Finding | Fix |
| --- | --- | --- |
| 1 (blocker) | `phasor/cli.py` read the clock (`perf_counter`), failing the contract-3 grep rule in `test_contracts.py` | the wall time comes from `core.timing.Stopwatch`, the one place allowed to read it |
| 2 | the desktop dialog named a Draft export `<stem>.mp4`, overwriting the real render | the suggested name follows the quality: `a.draft.mp4` / `a.draft.png`, as the CLI and the Workspace tool |
| 3 | `--end 1e12` (or `--fps 1000 --end 1e6`) ran out of memory building the frame times | `render.job.MAX_FRAMES` = 1,000,000 frames (4.6 h at 60 fps, 9.3 h at 30, 16.7 min at 1000), checked before anything is set up; a clean `ValueError` for every preset |
| 4 | `--fps 1e-9` passed and gave the encoder `-r 0/1` | rates below 1/1001 are refused (`presets.MIN_FPS`), and so is a scene's own rate below it |
| 5 | the speed claim did not hold for kernel-bound scenes | measured and stated (the table above, API.md, PITFALLS.md); a limit |
| 6 | markers named `inf`/`nan` could not be a window | `resolve_time` tries the marker name first |
| 7 | cancel left empty folders; Ctrl-C inside `writer.close()`/`mux_audio` left a partial or `.mux.mp4` file | the folders a render made are removed if empty; Ctrl-C while finishing removes the file and the mux temp (`encode.mux_temp`) |
| 8 | a pointless fork and bake when nothing would change | no fork for the theme's drift when the theme's drift (or its period) is 0. The fork still bakes its own layout once even when the parent is baked: its document differs (life off), and contract 7 / ADR 0003 allow applying a bake only to the document it was made for |
| 9 | NVENC fast settings over the ffmpeg pipe had `-rc vbr -cq 28` without `-b:v 0` | added (the PyAV path already sets `bit_rate` 0); untested without a card |
| 10 | `tests/ssim.py` `busy=` with no busy window returned NaN | the plain mean over every window then |
| 11 | `skills/phasor/SKILL.md`, `kit/CLAUDE.md`, `docs/ai/PITFALLS.md` did not mention the draft | they do (`phasor preview`), `.claude/skills/phasor/SKILL.md` kept equal, `llms.txt` regenerated |

## Tests

`tests/test_draft.py`: the table and the old presets' values; draft sizes at the bounds
(portrait, square, smaller than 640, odd, 8000x2, 2x2); bad preset names, frame rates
(NaN, inf, negative, over 1000, text), windows (unknown marker, NaN, inf, booleans, lists)
fail cleanly before anything is written; `without_decoration` (life off, the open scene
untouched, cached per version, a new version after an edit, the theme's drift off and an
explicit one kept, nothing to strip, a saved document's tweaks kept); the digest's
exactness; frame keys without OpenGL (a still moment has one key, motion and size and fog
change it, `FlowParticles` drawing from `state.t` is never repeated, living glow is never
repeated); encoder settings (the quality command line unchanged, the fast one, NO_WINDOW,
a fast video has no B-frames and a keyframe every 15 frames); CLI parsing (`phasor preview`,
markers, bench's draft); default file names; the Workspace tool; `Scene.render` bounds;
with OpenGL: a draft is 640x360 at 15 fps with 25+ of 45 frames repeated, paint passes the
tone map unchanged; repeated frames byte-identical to drawing every frame (and the test
asserts frames were repeated); a clock-driven type drawn every frame; a 2-worker draft
byte-identical to a serial one; a window by markers; cancel by Event and by callable, with
progress, for a video and a PNG folder (other files kept); preview frames through the job
equal to `engine.frame`; **draft against final** for `prism_primes`, `basics_3d` and
`prism_domain` (SSIM of the draft and the downscaled final at least 0.75, and over the
windows with content at least 0.35 and 0.2 above a blank frame and another scene's final;
measured 0.44-0.69 against at most 0.30); a golden draft still
(`tests/golden/software/draft_prism_primes.png`, looked at); a storyboard snapshot: draft
frames on a contact sheet against a golden (`draft_storyboard.png`, looked at) with the
inspector agreeing that the dot rests; `--workers 0` still renders; Ctrl-C
(`KeyboardInterrupt`) removes the partial file. `tests/test_desktop.py`: the export dialog lists Draft and starts on Preview.

Gates run (Linux): `ruff check .`, `ruff format --check .`, `lint-imports` (2 kept);
`test_contracts.py` (added to the list after the audit), `test_draft.py`, `test_render.py`, `test_workers.py`, `test_prism.py`, `test_frame_bench.py`,
`test_cpu_backend.py`, `test_mcp.py`, `test_kit.py`, `test_layers.py`, `test_beauty.py`,
`test_engine.py`, `test_desktop.py` (with the gui extra), `test_doctor.py`,
`test_encode_thread.py`, `test_proc.py`, `test_life.py`, `test_docs.py`, `test_final.py -k
"not slow"`. There is no `test_post*.py` in the tree. Not run: the full suite (CI does it).

## Deviations

1. **A draft draws a plainer document, not the same one.** STUDIO.md says a draft "is a render
   of the same document". The owner's "no ... haze, etc." and the efficiency ask both point at
   life (it moves every glowing object every frame, so no two prism frames are ever alike
   and nothing could be repeated). Life is part of `Engine.state_at` (contract 3), so it
   cannot be switched off at draw time without breaking that contract; instead the draft
   draws `Engine.without_decoration()`, a private fork with `life="off"` (and the theme's
   camera drift off), edited through `apply` (contract 1) and baked once in the parent
   (contract 7, ADR 0003). Everything the document animates is unchanged. Contract text is
   untouched.
2. **Dedup keys on what `draw` recorded, not on the state.** A hash of the state without its
   time would freeze types that draw from `state.t` (flow particles, flow lines, a point
   cloud's twinkle); the recorded calls do not. Kernel pictures are keyed as the field
   cache keys them (not by pixels), the same assumption the renderer already makes.
3. **The SSIM test uses a "busy windows" SSIM besides the plain one.** Plain luma SSIM over a
   near-black prism frame is high for almost any two frames (a blank frame scored 0.73-0.90
   against the finals); `tests/ssim.py` gained an optional `busy` threshold (windows where
   either picture varies), and the controls use it.
4. `--start`, `--end` and `--frame` are parsed as text (seconds or a marker), no longer
   `float`: every number that parsed before parses the same; `nan`/`inf`, which parsed
   before, are now refused. `--fps 0` still means the preset's rate, as before.
5. **Bounds, and what still renders.** `workers` 1 or fewer renders in this process (as
   `--workers 0` always did) and more than 64 is clamped to 64 (`scene.MAX_WORKERS`); only a
   value that is not a whole number is refused. `fps` must be above 0 and at most 1000:
   NaN, inf and negative rates never rendered anything sensible; a rate over 1000 frames a
   second is refused rather than clamped, because clamping would silently make a video of
   another length and frame count than asked for (no shipped scene or doc uses one).
6. **Ctrl-C** (`KeyboardInterrupt`) during a render now removes the partial file like a
   cancel; `phasor render`/`preview` print "cancelled" and exit 130 instead of a traceback.

## Deferred

- **The app's preview button** (Milestone 25): the export dialog only lists Draft; the
  window is not redesigned. Left for 25: the dialog still cancels by killing the render
  process, which leaves the partial file (its suggested name for a Draft is `a.draft.mp4`
  since the audit).
- **The Workspace / MCP `render` tool has no progress and no cancel**: it renders a draft
  (and takes markers), but a tool call is one blocking request. Progress and `cancel` exist
  on `render_file`, `Engine.render` and `Scene.render`; the Workspace hook for the app is
  Milestone 25.
- **Hue flowing along strokes** (`[strokes] flow`, only the `prism_vivid` theme) is kept in
  the draft; it makes every frame differ there. Stripping it would need a theme override.
- **MSAA off** in the draft: not done (thin strokes alias badly at 640 pixels); not measured.
- **Reusing the host buffers of the post chain** (widen, tone map, sRGB allocate a frame each):
  not done. GL targets, pack buffers and vertex buffers were already reused per renderer;
  the three host arrays at 640x360 are 4.4 MB a frame and NumPy allocation is microseconds,
  while those stages take 9-13 ms a frame of computation. The frame handed to the encoder
  must not be reused anyway (`BackgroundWriter`).
- A pre-existing issue found, not fixed (it would change finals): `cloud.PointCloud`'s
  field closure reads `state.t` for `twinkle`, but the field cache key has no time, so a
  twinkling cloud's flicker can freeze while its parameters stay the same. The draft is not
  affected (life off sets twinkle to 0).

## Untested

Windows (CI has no OpenGL: the GL tests skip there; the 52 non-GL tests are
written for it), macOS, CUDA and NVENC (the fast NVENC options were not run on a card), the
desktop app's export of a draft by hand (its file naming is tested offscreen), the GUI cancel (the dialog still kills the process;
the in-process `cancel` is for Milestone 25).
