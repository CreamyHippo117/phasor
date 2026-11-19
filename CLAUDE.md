# CLAUDE.md: building phasor

`SPEC.md` is the design. This file is the short form of the rules that keep errors from
compounding. If this file and SPEC.md disagree, stop and ask.

## Where we are

- Owner approved ADR 0001, ADR 0002 and id tombstones on 2026-11-18.
- Current milestone: **2-6 built; boundary review of 2-6 done 2026-11-18 (see `MILESTONE_6.md`). 7, 8, 9 built (9b deep zoom included, `MILESTONE_9.md`); boundary review of 7-9 done 2026-11-19 by the authoring model, not independent (`MILESTONE_9.md`). Windows and CUDA untested**. See `MILESTONE_*.md` for what is built and deferred.
- Roadmap (owner, 2026-11-19): Milestones 11 (Sketch) and 13 (Manim import) are dropped;
  10 (GUI editor) and 12 (interactive export) remain. The `sketch` client slot in the layer
  rule stays empty.
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
   Params are a DAG (constants, tracks, expressions); a cycle is a validation error.
   Narration is a separate `TimeMap`. Simulations use fixed dt, seed, cached checkpoints.
   Randomness only via `rng(object_id, seed)`.
4. **One spatial pipeline.** Everything is 3D; 2D = orthographic camera looking down −z at z=0.
   World: right-handed, z up, x=Re, y=Im. Screen: pixels top-left origin + normalized 0–1.
   Linear RGBA16F until the end; tone map and sRGB encode exactly once.
5. **Object contract.** Every type implements `Params`, `world_bounds`, `features`,
   `semantics`, `draw(ctx, state, pass_)` for passes `color`/`id`/`solo`. Missing any member
   fails registration.
6. **Layers.** `core ← style ← time ← layout ← inspect ← render ← domains ← engine ← (cli|gui|mcp|sketch)`.
   `core`, `style`, `time`, `layout` import no GPU/GL/video library. Lower layers that need
   render output define a protocol; `render` implements it; `Engine` injects it
   (`docs/decisions/0001-layers.md`). Enforced by `lint-imports` and `tests/test_layers.py`.
7. **Layout is baked.** Solver runs at markers + samples and writes tracks; renderers and
   inspector both read the bake. kiwisolver for linear relations, then the label placer
   (8 compass + leader slots, hysteresis 20%). Unsatisfiable constraints become lint issues.

## Conventions

- Tests first for timeline, patches, inspector and solvers.
- Every object type: semantics, bounds, inspector output, a lint rule if relevant, a
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

- MCP tools are methods of `mcp/tools.py` `Workspace` (plain data in and out, tested without
  the protocol); `mcp/server.py` only names them. Add a tool there, to `TOOL_NAMES`, to
  `docs/ai/MCP.md` and regenerate `llms.txt` (`uv run python -m phasor.mcp.llms`).
- Client commands register through the `phasor.commands` entry point; `cli` never imports a
  sibling client. Tools that return a picture use `structured_output=False`.
- A ```json patch / ```json calls block in `docs/ai/MCP.md` runs in CI like a python block.

- 3D: the camera solver lives in `layout/camera.py` and runs only inside the bake; it may set
  only camera parameters the document leaves unset. Lower layers get depth through the
  oracle's `depth` (ADR 0001 pattern); 3D labels are judged by what is in front of them.

- Beauty: dark (`midnight_glow`) is the default theme, always; light themes are opt-in. Fewer
  rainbows is not no rainbows: hue that carries meaning (domain-coloring phase) keeps the
  wheel. Taste knobs are theme `[lint]`/`[life]` values, roles, looks and presets, never
  numbers typed into a scene. Life is a pure function of (id, seed, t); `exact` objects never
  move; a type moves only in the ways its `Life` class variable names. Aesthetic rules live in `inspect/aesthetic.py` and run only with `aesthetic=True`.

- Domains: numerics live in `domains/<name>_math.py` (pure NumPy, no drawing) and objects in
  `domains/<name>.py`; objects draw strokes through `domains/strokes.py`. A time-stepping model
  is a `Simulation` (`core/simulation.py`): fixed dt, seed in `params()`, pure `step`, and
  only `SimRunner.state_at(t)` reads it.

## Commands

```bash
uv sync --extra gl --extra encode         # add --extra gpu on an NVIDIA machine
uv run ruff check . && uv run ruff format --check .
uv run lint-imports
uv run pytest                             # gl/gpu tests skip themselves without hardware
uv run phasor doctor
```

Headless Linux needs `libegl1 libgl1-mesa-dri` for the GL tests.

## Model use

Use the strongest available model for: core contracts and schema (M1), layout solver and
label placer (M2), kernels and the CUDA/GL bridge (M3), camera solver (M6), perturbation
deep zoom (M9), and every milestone-boundary review. A smaller model is fine where the
spec is explicit and the gates are green.
