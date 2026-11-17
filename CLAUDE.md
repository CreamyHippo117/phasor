# CLAUDE.md: building phasor

`SPEC.md` is the design. This file is the short form of the rules that keep errors from
compounding. If this file and SPEC.md disagree, stop and ask.

## Where we are

- Current milestone: **1 built (awaiting boundary review and Windows check), 2 next**. See `MILESTONE_*.md` for what is built and deferred.
- One milestone per branch. No work on milestone N+1 until N's exit demo and tests pass.
- Before declaring a milestone done, run the `phasor-milestone-gate` skill.

## The seven contracts (binding; changes need owner approval + SPEC edit in the same commit)

1. **One engine.** `phasor.engine.Engine` owns document, state cache, layout bake, inspector.
   CLI/MCP/GUI are thin clients. Every mutation is `Engine.apply(patch) -> PatchResult`
   (validate, record undo, return lint delta). No other write path.
2. **Document is truth.** pydantic v2 in `core/schema.py`, `schema_version` + `core/migrate.py`.
   `name.phasor.json` (sorted keys) + `name.overrides.json` (id + param path). Re-running a
   script never loses overrides; a dangling override is lint `orphan_override`.
   IDs: `[a-z][a-z0-9_]*`, dots for children, never reused, named by meaning not counters.
3. **Time is pure.** `Engine.state_at(t)` depends only on (document, t), returns frozen data.
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
- When a test and the code disagree, work out which one is wrong before changing either.
  In Milestone 1 it was the test about as often as the code.

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
