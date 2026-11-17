# Milestone 1: Core + inspector

Status: **built and green on Linux (software OpenGL). Not yet verified on Windows or on a
real GPU. Awaiting the boundary review and two owner decisions (see the end).**

Exit demo: `examples/moving_dot.py`, a labeled dot that crosses the frame, passes behind a
disc and leaves. `where` and `visible` are checked at 21 times against closed-form geometry
(`tests/test_inspect.py::test_exit_demo_where_and_visible_are_correct_at_any_t`).

## Built

| Area | Files | What |
| --- | --- | --- |
| Schema | `core/schema.py`, `migrate.py`, `io.py` | pydantic v2, frozen, unknown keys rejected, `schema_version`, JSON Schema export, canonical byte-stable files |
| Values, ids | `core/values.py`, `ids.py`, `rng.py` | bool / float / str / nested tuples only; id rules; `rng(object_id, seed)` |
| Object contract | `core/objects.py`, `state.py` | `SceneObject` protocol, registry that refuses incomplete types, immutable `SceneState` |
| Spatial | `core/spatial.py`, `text.py`, `color.py` | one 3D pipeline, ortho camera, screen-space overlay camera, shared text metrics, linear color |
| Patches | `core/patch.py` | 10 ops, pure `apply_ops`, overrides and orphan detection |
| Style | `style/theme.py`, `themes/midnight_glow.toml` | role groups, theme loader (palette, stroke, text sizes) |
| Time | `time/expr.py`, `tracks.py`, `easing.py`, `compile.py` | whitelisted expressions, `TimeMap`, tracks, parameter DAG with cycle detection, pure `evaluate` |
| Inspector | `inspect/passes.py`, `inspector.py` | `PassRunner` protocol, snapshot, detail levels, `hit_test`, analytic fallback |
| Render | `render/passes.py`, `output.py` | `color` (RGBA16F, 4x MSAA), `id` (R32UI + depth), `solo` (R8); sRGB encode once; PNG stills |
| Objects | `domains/basic.py` | `Dot`, `Line`, `Text` |
| Engine | `engine.py` | `apply` (atomic, undo/redo, lint delta), `state_at` cache, `inspect`/`where`/`visible`/`occluders`/`hit_test`/`still`, `open`/`save` |
| Python API | `scene.py` | `Scene`, `expr`, `at().to()/.set()`, `load` (script or document + overrides) |
| CLI | `cli.py` | `phasor inspect SCENE [--t] [--id] [--detail]` |
| Docs | `docs/ai/*` | API, patterns, pitfalls; every example runs as a test |

## Verification

| Check | Result |
| --- | --- |
| `ruff check`, `ruff format --check`, `lint-imports` | pass |
| `pytest` here (llvmpipe GL 4.5) | 181 passed, 1 skipped (CUDA kernel) |
| `pytest` with OpenGL simulated away (tier C, like the Windows CI runner) | all non-GL tests pass, GL tests skip |
| Exit demo script, stills, saved document equals script | pass |
| Golden stills, SSIM ≥ 0.99, `software` class | pass (3 stills) |
| Golden stills, `gpu` class | **none yet**: create on your machine, look at them, commit |
| Anything on Windows or NVIDIA | **not verified here** |

To verify on Windows:

```bash
uv sync --extra gl --extra gpu --extra encode
uv run pytest
uv run python examples/moving_dot.py
PHASOR_UPDATE_GOLDEN=1 uv run pytest -k golden
```

In PowerShell the last line is `$env:PHASOR_UPDATE_GOLDEN=1; uv run pytest -k golden`.

## Bugs the tests caught while building

Worth recording, because each would have compounded silently:

1. Expressions reported **no dependencies** (the set was frozen before the tree was walked),
   so the parameter graph evaluated in the wrong order. Caught by the purity test.
2. With more than one GL context alive, renders went to whichever was current: tests passed
   alone and produced garbage together. Fixed with one shared context made current per
   render, plus a static test that contexts are created in one place.
3. Three of my own test expectations were wrong (a region boundary, a y-axis sign, a sliver
   still on screen). The inspector was right each time.

## Deviations from SPEC (each fixed in SPEC in the same commit unless marked)

1. **`on_screen_fraction`** is measured on a zoomed-out solo pass, not against an "analytic
   projected area", which does not exist for text or surfaces. `visible_fraction` is counted
   inside the frame.
2. **Coplanar ordering**: `z_index`, then document order, through a small depth bias. The
   spec did not say how 2D objects at z = 0 decide who is in front.
3. **Sub-pixel objects** report `visible_fraction: null` with a note, never a guess.
4. **No `parent` field**: the parent is the id's dotted prefix, so the two cannot disagree.
5. **`constraints` and `assets` are not in schema v1.** They arrive through migrations in
   M2 and M4, which also exercises the migration path early. (SPEC not edited: still the plan.)
6. **Tracks live in `timeline.tracks`** keyed by parameter path; a parameter itself is a
   constant or an expression. One place for animation.
7. **Tier C inspector is analytic** (decision 2 from Milestone 0, implemented provisionally).
8. **Golden stills per renderer class** (decision 3 from Milestone 0, implemented provisionally).
9. **Occlusion rule** is `opacity >= 0.5` for now; per-material `occludes` arrives with materials (M6).
10. **Preview antialiasing** is 4x MSAA; analytic antialiasing belongs to the final renderer (M3).
11. **Text** uses the font bundled in Pillow (new dependency; CPU only, allowed in `core`).
    Theme fonts and LaTeX arrive in M3.
12. **IDs are never reused** is not enforced: remove-then-add of the same id is allowed.
    Enforcing it needs a tombstone list in the document; proposed for M2.
13. `inspect(t)` returns every object, including hidden and off-screen ones, with their
    fractions, rather than only visible ones. An AI needs to know what went missing.
14. Milestone 1 sits on the same branch as Milestone 0 (this session can only push to one branch).

## Deferred (by the roadmap, not forgotten)

- Layout constraints, label placer, layout bake, visual lint rules, storyboard: M2.
- Kernels, CUDA bridge, bloom, tone mapping, video: M3. Narration and verbs: M4.
- `diff`, natural-language resolver, MCP: M5. Events, `when_hidden`, `free_space_near`,
  camera solver, perspective: M6. Simulations: M8.
- Live preview window with scrubbing and hot reload: not in the M1 row; proposed for M2.
- Parent transforms (children moving with a group): with composites in M2.

## Owner decisions still open

1. **ADR 0001** (`docs/decisions/0001-layers.md`): layer positions for `style`, `engine`
   and now `scene`, plus dependency inversion for render output. Milestone 1 is built on it
   and it worked as designed (`inspect` never imports `render`). Needs your approval because
   it touches contract 6.
2. **ID reuse** (deviation 12): enforce with tombstones, or relax the contract's wording?
