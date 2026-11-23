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

## Status

| Milestone | Status |
| --- | --- |
| 14 CPU | approved; baseline first, then built |
| 15 contract | ADR 0005 accepted; building |
| 16-20 | planned (16-19 wait for 15) |
