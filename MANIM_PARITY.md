# Manim parity, the CPU, and a freer object contract

Plan requested by the owner on 2026-11-23; **approved the same day** (answers below). Like `GPU_SPEED.md`, this file
holds the detail and the status. The contract changes are in
`docs/decisions/0005-open-object-contract.md` (accepted 2026-11-23, SPEC edited in the same commit).

What this plan closes, from the comparison of 2026-11-23:

| Gap | Milestone |
| --- | --- |
| phasor uses one CPU core for kernels without CUDA; workers are capped at 4 | 14 |
| no scene-local object types; five required members; no per-frame Python | 15 (ADR 0005) |
| no fills; no Arc, Sector, Annulus, Ellipse, Brace, dashes; no SVG; no boolean shape ops | 16 |
| `Transform` between different shapes crossfades; no `TransformMatchingTex`; colors don't animate | 17 |
| colors and font sizes only as roles; background only through the theme | 18 |
| no NumberLine, BarChart, Table, Code, axis labels, NumberPlane grid, unequal axis scales | 19 |
| importer warnings; `MovingCameraScene`, `add_sound`; untested on real Manim, Windows, macOS | 20 |

Order: 14 and 15 do not depend on each other. 16-19 need 15, so that a new shape is a short
class and not 80 lines. 20 comes last, and it removes one importer warning per feature landed.
One milestone per branch, as always. Where this sits relative to `BEAUTY.md` is the owner's
call (question 4 below).

## Milestone 14: use the whole CPU

Today (`render/gpu/registry.py`), a machine without CUDA (tier B) runs each kernel's NumPy
reference on one core. `phasor render` warns "this is slow". On the owner's RTX 4070 Ti, the
Python draw step (105 ms of 555 ms in `motion_gallery`, `GPU_SPEED.md`) runs in at most 4
workers (`scene.default_workers`: a quarter of the threads, 1 to 4), so most cores sit idle.

**14.0 Measure.** Run `phasor bench --frames` on `motion_gallery` and `prism_primes` with CUDA
hidden (`backend="reference"`) and with CUDA. Record the wall clock and per-core use in
`benchmarks/`. Each later step is judged against these numbers.

**14.1 A CPU kernel backend from the same CUDA source.** `tests/cuda_emu.py` already compiles
each kernel's CUDA string with g++ and runs it one thread index at a time. Promote it to
`render/cpu/backend.py`:
- Compile with `-O3 -fopenmp`, run the thread loop as `#pragma omp parallel for` over every
  core, and cache the built library in the user cache, keyed by the source hash.
- Use real atomics (`__atomic_fetch_add`) in place of the emulator's plain adds. The splat
  kernel's sums must not depend on thread order: each thread accumulates into its own buffer,
  then the buffers are reduced in a fixed order. The same frame renders the same every time.
- Registry `backend="auto"` order: `cuda`, then `cpu` (if a compiler is present), then
  `reference`. `phasor doctor` reports the C++ compiler and the thread count.
- The kernel rule in CLAUDE.md stands: there is one source string per kernel, in
  `render/gpu/kernels/`. The emulator tests keep running serially, so their output stays
  deterministic.
- Tests: every kernel in `CASES` (`tests/test_kernels.py`) is run on the CPU backend and
  compared with its reference. Results match whatever the thread count (1, 2, all).

**14.2 Post on the CPU through the same kernels.** On tier B, the post stages (`display8`,
blur sum, bloom) run through the CPU backend instead of NumPy, so they use every core too.

**14.3 Worker count from the hardware.** `default_workers()` decides per tier:
- Without CUDA: workers × kernel threads ≤ cores. An `--threads` option sets the threads.
- With CUDA: measure 4, 6, 8 and 12 workers on the 4070 Ti, then raise the cap to the fastest
  count that `render/gpu/budget.py` says fits in VRAM. The draw step is Python, so more
  processes is the direct way to use the CPU while the card waits.
- Done when the bench shows the gain on both machines (wall clock, before and after).

**14.4 Rendering with no OpenGL 4.3 (tier C, which includes every Mac).** macOS stops at
OpenGL 4.1, so phasor cannot render there today. First investigate Mesa's llvmpipe, a
multithreaded CPU OpenGL: it is already what headless Linux uses. On Windows, the
`mesa-dist-win` `opengl32.dll` could be picked up by `glcontext.py` on request. On macOS,
llvmpipe needs building. Write down the findings before choosing between that and a second
`DrawContext` (a NumPy rasterizer), which is far more work. Until something runs there, it
stays "untested on macOS".

Exit demo: `phasor render showcase/prism_primes.py` on a machine with CUDA hidden uses every
core, is at least N× faster than the reference path (N set from 14.0), and matches the
software golden.

## Milestone 15: a freer object contract (needs ADR 0005)

What ADR 0005 changes, in short:
- **Required members:** only `Params` and `draw(ctx, state)`. `pass_` becomes optional,
  because the renderer already decides what a command means in each pass.
- **Derived defaults:** `world_bounds`, `features` and `semantics` are worked out from what
  `draw` records (`core/record.py`, pure NumPy, cached per state). A type can still write its
  own when it knows more, for example the poles of a function.
- **Composites:** `build(state) -> [objects]` makes a type out of other types, like a Manim
  `VGroup` subclass. Children get dotted ids.
- **Types local to a scene:** a type can be defined in the scene script or in a `*.types.py`
  file next to it. The document names the modules it needs. A missing module is lint
  `missing_type`, and the object draws as a placeholder box (overrides are kept). The module
  is loaded only where scripts are allowed, which are the same rules as importers.
- **Updaters:** `obj.always(fn)`, with `fn(t, look) -> {param: value}`. `look("other")`
  reads another object's state, each read becomes an edge in the parameter DAG, and a cycle
  is a validation error. Updaters stay pure; one that needs the previous frame is a
  `Simulation`.
- **`Value`:** a named number to animate (Manim's `ValueTracker`), with no drawing.

Steps: the recorder and derived defaults (tests first), then the optional `pass_`, then
`build`, then scene-local modules with `missing_type`, then `always` and `Value`. Last, port
`Dot`, `Line` and two domain types to the short form, to prove that nothing in the inspector,
layout or lint gets worse. Their inspector output must be unchanged.

Exit demo: a scene script defines a new type in 10 lines (`draw` only). The inspector,
layout, lint and storyboard all work on it, and an `always` updater keeps a label on a
moving dot.

## Milestone 16: vector paths, fills and shapes

- `Path`: subpaths of cubic Béziers, flattened adaptively by size on screen. It has a fill
  and a stroke, and becomes the base of every outline shape.
- **Fills:** even-odd or non-zero rule, with holes, triangulated by earcut. The fill draws
  through the existing `ctx.triangles`, so no renderer change is needed. A new style group
  `fill` (`none soft solid`) takes its opacity from the theme.
- **Shapes on `Path`:** `Circle`, `Ellipse`, `Arc`, `Sector`, `Annulus`, `Polygon`,
  `RegularPolygon`, `Rectangle` (with corner radius), `Brace`, and arrow tips as path ends.
- **Dashes:** a `dash` style group (`solid dashed dotted`), cut by arc length so dashes
  don't crawl as a path grows.
- **Boolean operations:** `union`, `intersection`, `difference`, `exclusion` on flattened
  paths.
- **`SVG`:** an object with a `Files` param (contract 3: read from the scene's folder). Its
  shapes become `Path` children with ids from the SVG `id`s.
- Each type gets an API entry, an example, a storyboard snapshot and a golden.

## Milestone 17: text as shapes, morphing anything, animated style

- **Glyph outlines:** text outlines come from matplotlib's `TextPath`, which phasor already
  uses for math. Text can then be filled, stroked and morphed, with one child per glyph.
  `write` draws the stroke, then the fill, like Manim.
- **`transform` between any two paths:** match subpath counts by splitting or duplicating
  subpaths, resample by arc length, then align start points by least travel. This replaces
  today's crossfade.
- **`transform_matching`:** glyph groups matched by TeX substring (for formulas) or by shape
  (for anything else) move to their new place; the rest fade.
- **Style transitions:** `.to("id.style.color", "highlight")` and the same for fill and
  stroke interpolate the resolved linear RGBA. The document still stores a role at each key.

## Milestone 18: style freedom (needs the owner's OK)

- Accept explicit values beside roles: `color="#58C4DD"` or linear RGBA, a font size in
  world units, a stroke width, and a background color per scene.
- Taste stays checked: aesthetic lint rules `off_palette` and `off_scale` warn, and
  `polish()` can snap an explicit value to the nearest role. That snap is not certain to be
  right, so it is offered, not applied.
- This loosens the beauty rule "taste knobs are ... never numbers typed into a scene". It
  is the owner's decision (question 2).

## Milestone 19: a diagram kit

`NumberLine`, `NumberPlane` with a grid, `Axes` with unequal x and y scales and axis labels,
`BarChart`, `Table`, `MatrixDisplay`, `Code` (monospace, highlighted), and a number display
that animates (check the existing `Counter` first: extend, never duplicate). These are built
as composites (`build`) of `Path` and `Text`.

## Milestone 20: Manim import parity and platforms

- Map every feature above in `importers/manim.py` and delete its warning. Updaters and
  `always_redraw` become `always`; dt updaters become a generated `Simulation`.
  `ValueTracker` becomes `Value`.
- `MovingCameraScene`: `self.camera.frame` animations become camera tracks.
- `add_sound`: a `Sound` clip (a `Files` param, a start time) mixed in at encode, next to
  narration.
- Test against real Manim CE: install it in a test environment and compare positions and
  timings scene by scene with phasor's import. Today the stand-in is written from Manim's
  documentation only.
- Platforms: a full `TESTING.md` pass on Windows; macOS once 14.4 gives it a renderer.

## Owner's answers (2026-11-23)

1. ADR 0005 (contracts 2, 3 and 5): **accepted**. SPEC and CLAUDE.md edited in the same commit.
2. Explicit colors and sizes in scenes (Milestone 18): **allowed**. CLAUDE.md's beauty rule is
   amended: explicit values are checked by `off_palette` / `off_scale` and snapped by `polish()`.
3. Dependencies: **left to the builder**. Picks, all with Windows wheels: `mapbox-earcut`
   (ISC) for fill triangulation; `shapely` (BSD, vectorized, robust on messy input) for boolean
   operations, preferred over `pyclipper` (MIT; integer coordinates need scaling);
   `svgelements` (MIT) for SVG. `pygments` (BSD) only if Milestone 19's `Code` needs it. Each
   is a base dependency only if its object type is in the core; otherwise an extra.
4. Order against `BEAUTY.md`: **interleaved**. Between milestones, take the next runnable
   `BEAUTY.md` step; its steps 3, 4 and 7 are PC-only (`pc_tasks/`).
5. CPU kernels: **a C++ compiler at run time**, with the NumPy reference as the fallback when
   none is found. No prebuilt binaries.

Branches: the session may push only `claude/eloquent-mayer-jiqxs9`, so Milestones 14 and 15
are built in local worktrees and merged here, one milestone per commit series. Nothing starts
on 16-19 until 15 passes `phasor-milestone-gate`.

## Milestone 14 baseline (14.0, 2026-11-23)

4 cores, no CUDA, OpenGL on llvmpipe (tier B), `phasor bench --frames SCENE --count 6 --preset
preview` (960x540), kernels on the NumPy reference, nothing else running. JSON in `benchmarks/`.

| stage (ms/frame) | prism_primes | prism_domain |
| --- | --- | --- |
| draw | 27 | 518 |
| post (bloom, tone map, lens) | 454 | 421 |
| widen, srgb, encode, other | 48 | 47 |
| **total** | **530** | **987** |

Post is the biggest or second stage in both. Not measured, and
not claimable from this machine: anything with CUDA, and 14.3's worker counts on the RTX 4070
Ti (`pc_tasks/`).

## Milestone 14 result (2026-11-23)

The same box and command as the baseline, back to back on a quiet box; the reference rerun
matches the baseline above. CUDA not available.

| stage (ms/frame) | prism_primes ref -> cpu | prism_domain ref -> cpu |
| --- | --- | --- |
| draw | 26 -> 27 | 482 -> 157 |
| post | 465 -> 58 | 451 -> 57 |
| srgb | 26 -> 9 | 24 -> 10 |
| **total** | **542 -> 121 (4.5x)** | **985 -> 250 (3.9x)** |

## 14.4 findings (2026-11-23; written investigation, nothing built)

The question: how phasor can render where there is no OpenGL 4.3. That covers every Mac
(Apple stops at 4.1 core, deprecated since macOS 10.14 but still shipped, on Apple silicon
too) and Windows machines whose driver lacks 4.3.

**1. phasor does not use any 4.3 feature. Measured here.**
- `glcontext.REQUIRED_GL = 430` says "compute shaders, SSBOs, integer textures". The
  renderer uses none of the first two. Every shader is a vertex/fragment pair (`#version
  430`) with `layout(location = n)` outputs and a `uint` id target, which are GL 3.3
  features.
- The host calls in `render/passes.py` are `buffer`, `texture`, `depth_texture`,
  `depth_renderbuffer`, `framebuffer`, `vertex_array`, `program`, `copy_framebuffer` (a
  blit), `blend_func`, `enable` and `disable`. All of them are 3.x.
- Experiment, not committed: on this box's Mesa I asked for a 4.1 core context
  (`MESA_GL_VERSION_OVERRIDE=4.1`, `MESA_GLSL_VERSION_OVERRIDE=410`), with
  `create_standalone_context(require=410)` and every `#version 430` in `passes.py` changed
  to `410`. Mesa then reported "4.1 (Core Profile)" and compiled every shader.
- A 480x270 frame at mid-scene of `prism_primes`, `basics_3d` (lit surfaces, labels) and
  `motion_gallery` came out **byte-identical** (max difference 0) to the 4.5 context.
- Caveat: Mesa's override caps the version it reports and the GLSL it accepts. It does not
  remove the newer entry points, so a stray 4.2+ call would still have worked here. Only a
  real 4.1 driver proves it.
- **Recommendation (owner decision: it changes the tier rule in SPEC and CLAUDE.md):**
  - Lower the requirement to 4.1 core and write `#version 410` in the renderer's shaders.
  - Keep the doctor's triangle and id probes, at 410.
  - Then run the suite on a real Mac, on Apple's own OpenGL, before claiming anything. Until
    then: untested on macOS.

**2. Mesa llvmpipe, where the driver is missing or too old.**
- Linux: already the headless path (EGL, `glcontext.py`). It is multithreaded
  (`LP_NUM_THREADS`, default every core). In the 14.0 baseline it drew `prism_primes` at
  960x540 in 27 ms/frame, a small share of the frame.
- Windows: the `mesa-dist-win` builds ship an `opengl32.dll` with llvmpipe (GL 4.5 core).
  Windows loads `opengl32.dll` from the application's folder before the system's, so
  placing it next to `python.exe` (or the packaged app's exe) is enough.
  - The ModernGL `glcontext` WGL backend seems to take a `libgl` path, which would let
    `glcontext.py` load a named DLL on request without copying anything. Not verified:
    check this before building on it.
  - It costs about 30-60 MB in the installer and its own license notices. Untested on
    Windows.
- macOS: Homebrew's `mesa` builds llvmpipe, but for GLX/X11, not CGL, which is what
  ModernGL uses on macOS. Getting it to run under a CGL or EGL context is real porting work.
  If point 1 holds on a real Mac, Apple's own 4.1 makes this unnecessary.

**3. A NumPy rasterizer (a second `DrawContext`).** Rough cost; nothing measured:
- It would rasterize triangles, text quads and kernel quads with depth, premultiplied
  blending, the two color targets (paint and emission), the id and solo passes, and the lit
  surface shading.
- That is about 1500-3000 lines plus their own tests. Every golden would need a third
  variant, or it would have to match `software` within SSIM.
- Vectorized NumPy fills a few large triangles quickly. It does not fill thousands of small
  ones (curves, text): those need per-triangle Python, or a C++ inner loop, which would
  bring the compiler back. Expect it to run well behind llvmpipe, which is already
  multithreaded LLVM-compiled code.
- It also breaks "one way to do each thing": two renderers that must keep agreeing.

**Order proposed:**
1. Point 1, on a real Mac. Cheap, and it may settle macOS.
2. Point 2 on Windows, for drivers without 4.1, as an opt-in through `glcontext.py`.
3. Point 3 only if 1 and 2 both fail somewhere that matters.

## Waiting for the owner (Milestone 15 audits; Milestone 20)

1. Imported code (a package, a test module, a notebook cell) is shared by the process and
   usable by a document opened without `allow_scripts` (`MILESTONE_15.md` Deviation 13).
2. A scene split across a sibling module (`from gemlib import Gem2` beside `scene.py`) counts
   as imported code, so a later untrusted document naming `gemlib.py` uses the process's
   class (contract 3); repro `pending/m15_s1_sibling_module/` (`MILESTONE_15.md` Deferred).
3. How the desktop app offers `Editor.trust_scene_code()` for a document that names modules
   (today: no button; lint `missing_type` names it) (`MILESTONE_15.md` Deviation 17).
4. Milestone 20: should an imported text match Manim's size (an explicit size on every text,
   `off_scale` warns) or keep the theme's role (narrower than Manim's: Text 0.42x, MathTex
   0.6x)? (`MILESTONE_20.md` Deferred 1)

## Status

| Milestone | Status |
| --- | --- |
| 14 CPU | **done** 2026-11-23 (`MILESTONE_14.md`); Windows/CUDA/macOS on the PC |
| 15 contract | **done** 2026-11-23 (`MILESTONE_15.md`), two audit rounds; owner decisions open |
| 16 shapes | **done** 2026-11-23 (`MILESTONE_16.md`), audited |
| 18 style | **done** 2026-11-23 (`MILESTONE_18.md`), audited |
| 17 text shapes | **done** 2026-11-23 (`MILESTONE_17.md`), audited |
| 19 diagram kit | **done** 2026-11-23 (`MILESTONE_19.md`), audited |
| 20 importer parity | **built** 2026-11-23 (`MILESTONE_20.md`): real Manim CE 0.22.1 compared, 10 scenes pass; awaiting an independent audit; Windows/macOS on the PC (`pc_tasks/10`, `11`) |
