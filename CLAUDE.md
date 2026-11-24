# CLAUDE.md: building phasor

`SPEC.md` is the design. This file is the short form of the rules that keep errors from
compounding. If this file and SPEC.md disagree, stop and ask.

## Where we are

- Owner approved ADR 0001, ADR 0002 and id tombstones on 2026-11-18.
- Current milestone: **all of 0-13 built (`MILESTONE_*.md`; 11 Sketch in `MILESTONE_11.md`, reinstated 2026-11-19); the roadmap is complete; boundary reviews of 2-6 (2026-11-18) and 7-9 (2026-11-19, by the authoring model, not independent) done; 10-13 await the owner's final reviews. Windows and CUDA untested**. See `MILESTONE_*.md` for what is built and deferred.
- Roadmap (owner, 2026-11-19): Milestones 11 (Sketch) and 13 (Manim import) were dropped,
  then both reinstated. The owner approved the three layers between the engine and the
  clients (`interactive`, `importers`, `workspace`) on 2026-11-19.
- Test plan and its status: `TESTING.md` (A desktop app done; B Windows/NVIDIA mostly done on
  the owner's RTX 4070 Ti, 2026-11-21; C gaps to do).
- Now (owner, 2026-11-22): the beauty plan in `BEAUTY.md`, in its order (prism domain
  coloring, heat maps, point clouds, fractals, image import, graph import, 3D acoustics).
  Keep `prism_primes`' look exactly. Steps 3, 4 and 7 run on the PC from `pc_tasks/`.
  Next animations (signal, waves, dynamics): `ANIMATIONS.md`. Changes waiting for the
  owner's OK go in `pending/` (a patch and a before/after picture). Render speed and the
  finished Windows app come after, unless the owner says otherwise.
- Desktop app (owner, 2026-11-22): code-first, one shared script that the person, the AI and
  the app all edit. Plan: `APP_PLAN.md` (answered); ADR 0004 (tweaks inside the script,
  touches contract 2) accepted by the owner; phase 1 not started.
- Manim parity, the whole CPU, and an open object contract: `MANIM_PARITY.md` (Milestones 14-20,
  owner 2026-11-23, answers recorded there); ADR 0005 accepted, plan approved, interleaved with
  `BEAUTY.md`. **14 (CPU kernels) and 15 (open contract) are DONE** (gate 2026-11-23, after independent audits; `MILESTONE_14.md`, `MILESTONE_15.md`) on `claude/eloquent-mayer-jiqxs9`. **16 (paths, fills, shapes), 17 (text shapes, morphing), 18 (explicit styles) and 19 (diagram kit) are DONE** (audited; `MILESTONE_16.md` to `MILESTONE_19.md`); **20 (importer parity, real Manim compared) is DONE** (audited; `MILESTONE_20.md`): the whole plan 14-20 is built. What is left is the owner's PC checks (`pc_tasks/09`, `10_manim_real.md`, `11_platforms_after_20.md`) and decisions (`MANIM_PARITY.md`). Owner's PC (Windows 11, RTX 4070 Ti, tier A; 2026-11-23, results in `pc_tasks/10_manim_real.md` and `pc_tasks/11_platforms_after_20.md`): full suite 2362 passed, 0 failed after the fixes it found; real Manim 11 of 11 scenes pass; sound renders measured in sync (chime not yet listened to). Still untested: macOS, `TESTING.md` B1 (no OpenGL on the PC), a heard check of the chime, and the rest of **`pc_tasks/09_pc_checks_after_14_15.md`** (CUDA worker count, Riemann timing, GPU goldens); owner decisions are in `MANIM_PARITY.md`.
- Studio (owner, 2026-11-24): preview render, audio and media inputs, MIDI, and a desktop app that works: `STUDIO.md` (Milestones 21-25). 24 video and picture inputs is DONE (audited); 21 draft render and 22 audio inputs and narration snap are built and under audit; 23 MIDI follows 22; 25 the desktop app (`APP_PLAN.md` phases 1-2 plus the buttons) comes last.
- Explorer (owner, 2026-11-24): a Desmos-like dynamical-systems workbench on the one engine (rows, sliders, vector fields, flows, equilibria, AI tools): `EXPLORER.md` (Milestones 26-31; design review in `docs/decisions/explorer_advice_2026-11-24.md`). **26 flow core is DONE** (audited; `MILESTONE_26.md`); 27 part A (row normaliser) merged, 27 tools next; the panel (29) waits for 25.
- Render speed: `GPU_SPEED.md` (steps 1-4 done, workers included); resume at its "Next
  session"; measure with wall clock before and after each step.
- One milestone per branch. No work on milestone N+1 until N's exit demo and tests pass.
- Before declaring a milestone done, run the `phasor-milestone-gate` skill.

## The seven contracts (binding; changes need owner approval + SPEC edit in the same commit)

1. **One engine.** `phasor.engine.Engine` owns document, state cache, layout bake, inspector.
   CLI/MCP/GUI are thin clients. Every mutation is `Engine.apply(patch) -> PatchResult`
   (validate, record undo, return lint delta). No other write path.
2. **Document is truth.** pydantic v2 in `core/schema.py`, `schema_version` + `core/migrate.py`.
   `name.phasor.json` (sorted keys) + `name.overrides.json` (id + param path, plus added
   constraints; ADR 0002). Re-running a
   script never loses overrides; a dangling override is lint `orphan_override`.
   IDs: `[a-z][a-z0-9_]*`, dots for children, never reused (removed ids go to
   `retired_ids`), named by meaning not counters.
3. **Time is pure.** `Engine.state_at(t)` depends only on (document, t) and the occlusion oracle (ADR 0002), returns frozen data.
   Updaters (`obj.always`) are pure DAG nodes; `meta.modules` names code modules inside the scene's
   folder (ADR 0005, owner 2026-11-23).
   It may also depend on files the document names (an object's `Files` params, e.g. a
   `Room3D` `.obj`), read only from inside the scene's folder, re-read when they change;
   a missing one is lint `missing_file` (owner, 2026-11-22).
   Params are a DAG (constants, tracks, expressions); a cycle is a validation error.
   Narration is a separate `TimeMap`. Simulations use fixed dt, seed, cached checkpoints.
   Randomness only via `rng(object_id, seed)`.
4. **One spatial pipeline.** Everything is 3D; 2D = orthographic camera looking down −z at z=0.
   World: right-handed, z up, x=Re, y=Im. Screen: pixels top-left origin + normalized 0–1.
   Linear RGBA16F until the end; tone map and sRGB encode exactly once.
5. **Object contract.** Every type implements `Params` and `draw(ctx, state)`; missing either
   fails registration. `world_bounds`, `features` and `semantics` are derived from what `draw`
   records unless the type writes them; `pass_` is optional; `build(state)` makes a composite
   (ADR 0005, owner 2026-11-23).
6. **Layers.** `core ← style ← time ← layout ← inspect ← render ← domains ← engine ← interactive ← voice ← scene ← importers ← workspace ← (cli|gui|mcp|sketch)`
   (`interactive`, `importers`, `workspace` approved by the owner 2026-11-19).
   `core`, `style`, `time`, `layout` import no GPU/GL/video library. Lower layers that need
   render output define a protocol; `render` implements it; `Engine` injects it
   (`docs/decisions/0001-layers.md`). Enforced by `lint-imports` and `tests/test_layers.py`.
7. **Layout is baked.** Solver runs at markers + samples and writes tracks; renderers and
   inspector both read the bake. kiwisolver for linear relations, then the label placer
   (8 compass + leader slots, hysteresis 20%). Unsatisfiable constraints become lint issues.
   A render worker applies its parent Engine's bake for the same document (ADR 0003).

## Conventions

- Tests first for timeline, patches, inspector and solvers.
- Every object type shipped in phasor (scene-local types are exempt): semantics, bounds, inspector output, a lint rule if relevant, a
  `docs/ai/API.md` entry and an example in `examples/`.
- Every feature ships with a storyboard snapshot test (from Milestone 2).
- One way to do each thing. Extend, never duplicate. Grep before adding a helper.
- GPU math only as CuPy `RawKernel` CUDA C strings in `render/gpu/kernels/`, registered with
  a NumPy reference; registration fails without one. Nothing outside `render/gpu/bridge.py`
  moves pixels between CUDA and GL. GL contexts only via `render/glcontext.py`.
- Update `docs/ai/` in the same commit as the API change. Every ```python block there runs in CI.
- Where SPEC and working code disagree, the code wins and SPEC is updated in the same commit,
  except for the seven contracts above, which need owner approval.
- Record any deviation from SPEC in the current `MILESTONE_N.md` under "Deviations", with why.
- Don't claim something works on Windows/CUDA unless it ran there. Say "untested on X".
- All GL work goes through `render/passes.py` (`default_renderer()`), which shares one context
  and makes it current per render. Creating GL objects outside `with ctx:` corrupts state.
  The shared context lives on one thread: GL work from anywhere else goes through
  `glcontext.on_gl_thread(fn)` (a context current in one thread fails in every other).
- Text files are read and written with `encoding="utf-8"` (Windows defaults to cp1252).
- Every `subprocess` call passes `creationflags=NO_WINDOW` (`core/proc.py`), or the Windows
  desktop app flashes a console window; `tests/test_proc.py` checks it.
- Golden stills live in `tests/golden/<software|gpu>/`; regenerate with
  `PHASOR_UPDATE_GOLDEN=1 uv run pytest -k golden` and look at them before committing.
- Layout runs only inside `layout/bake.py`, and only `Engine` runs or applies the bake.
  Anything that draws or inspects takes `Engine.state_at(t)`, never `time.compile.evaluate`.
- A lint rule lives in `inspect/lint.py`, takes its threshold from the theme's `[lint]` table,
  reports `times`, and offers a `fix` only when the fix is certain to be right.
- New lint rules and layout features are tested with `passes=None` and with OpenGL (see
  `MODES` in `tests/test_lint.py`); CI on Windows has no OpenGL.
- GL queries (even `ctx.info`) must run inside `with ctx:`.
- Every kernel is tested three ways: its NumPy reference against known maths, its CUDA
  source compiled for the CPU (`tests/cuda_emu.py`), and on a real GPU (`gpu` marker).
  Add each new kernel to `CASES` in `tests/test_kernels.py`. Launchers may only use array
  calls NumPy and CuPy share, or the emulator cannot run them.
- Bloom comes only from the emission target (glow roles); never threshold the picture.
  Paint below the theme's knee must come out of the tone map unchanged.
- Kernel-drawn objects call `ctx.field(...)` and render only their visible part; mark them
  `"backdrop": True` in semantics if labels may sit on them.
- Narrated scenes anchor every motion to a marker and use verbs; `lint(strict=True)` is the
  check. Verbs live in `time/verbs.py`, take timing from the theme `[motion]`, and must give
  the same result whatever order the script writes them in.
- When a test and the code disagree, work out which one is wrong before changing either.
  In Milestone 1 it was the test about as often as the code.

- MCP tools are methods of `workspace/tools.py` `Workspace` (plain data in and out, tested
  without the protocol; shared by the MCP server and the GUI, which may not import each
  other); `mcp/server.py` only names them, `workspace/remote.py` carries them to an
  attached `phasor mcp --attach`. Add a tool there, to `TOOL_NAMES`, to
  `docs/ai/MCP.md` and regenerate `llms.txt` (`uv run python -m phasor.mcp.llms`).
- GUI: all logic in `gui/controller.py` (`Editor`, no Qt, tested directly); `gui/window.py`
  only draws it and forwards input. Window tests run on Qt's `offscreen` platform. Every
  edit goes through the `Workspace` under the shared lock.
- Importers (`phasor/importers/`) run foreign scripts only on the person's request; a stand-in
  module never leaks past the import (it is removed from `sys.modules`). Interactive features
  (`phasor/interactive.py`) work on `Engine.fork()`, never on the open scene.
- Client commands register through the `phasor.commands` entry point; `cli` never imports a
  sibling client. Tools that return a picture use `structured_output=False`.
- A ```json patch / ```json calls block in `docs/ai/MCP.md` runs in CI like a python block.

- 3D: the camera solver lives in `layout/camera.py` and runs only inside the bake; it may set
  only camera parameters the document leaves unset. Lower layers get depth through the
  oracle's `depth` (ADR 0001 pattern); 3D labels are judged by what is in front of them.

- Beauty: new scenes get the `prism` look (owner, 2026-11-22; `look="none"` for plain, which
  is the dark `midnight_glow` theme); dark always, light themes are opt-in. Tests run with
  `PHASOR_DEFAULT_LOOK=none` (set in `tests/conftest.py`). Fewer
  rainbows is not no rainbows: hue that carries meaning (domain-coloring phase) keeps the
  wheel. Taste knobs are theme `[lint]`/`[life]` values, roles, looks and presets. Explicit colors and
  sizes typed into a scene are allowed (owner, 2026-11-23, Milestone 18); the aesthetic rules
  `off_palette` and `off_scale` warn about them and `polish()` offers the nearest role. Life is a pure function of (id, seed, t); `exact` objects never
  move; a type moves only in the ways its `Life` class variable names. Aesthetic rules live in `inspect/aesthetic.py` and run only with `aesthetic=True`.

- Domains: numerics live in `domains/<name>_math.py` (pure NumPy, no drawing) and objects in
  `domains/<name>.py`; objects draw strokes through `domains/strokes.py`. A time-stepping model
  is a `Simulation` (`core/simulation.py`): fixed dt, seed in `params()`, pure `step`, and
  only `SimRunner.state_at(t)` reads it.

## Commands

```bash
uv sync --extra gl --extra encode --extra mcp --extra boolean --extra svg --extra code  # add --extra gpu on an NVIDIA machine
uv run ruff check . && uv run ruff format --check .
uv run lint-imports
uv run pytest                             # gl/gpu tests skip themselves without hardware
uv run phasor doctor
```

Headless Linux needs `libegl1 libgl1-mesa-dri` for the GL tests. The CPU-emulator kernel
tests need `g++` on PATH (Windows: `winget install BrechtSanders.WinLibs.POSIX.UCRT`).

## Model use

Use the strongest available model for: core contracts and schema (M1), layout solver and
label placer (M2), kernels and the CUDA/GL bridge (M3), camera solver (M6), perturbation
deep zoom (M9), and every milestone-boundary review. A smaller model is fine where the
spec is explicit and the gates are green.
