# Milestone 15: the open object contract (ADR 0005)

Accepted by the owner on 2026-11-23 (`docs/decisions/0005-open-object-contract.md`, the
"Milestone 15" section of `MANIM_PARITY.md`). Exit demo: **a scene script defines a new type
in about 10 lines (`draw` only); the inspector, layout, lint and storyboard all work on it,
and an `always` updater keeps a label on a moving dot** (`examples/open_contract.py`).

Status: **built; gates green on Linux (see Verification).** Untested on Windows and CUDA.

## Built

| Area | Where | What |
| --- | --- | --- |
| Recorder | `phasor/core/record.py` | a `DrawContext` that keeps geometry and draws nothing (pure NumPy): triangles, surfaces, text boxes (`core/text.quad`), and a `field`'s quad corners without ever calling its `pixels`. Derived `world_bounds` (box of everything drawn), `features` (`center`, the box's compass points, and `start`/`end`/`curve` for one stroke, noted by `domains/strokes.py`), `semantics` (`{"type", "params"}`). Cached per (type, id, params, style, camera, folder), and made again when an object it read changes |
| Two-member contract | `phasor/core/objects.py` | `register` needs `Params` and `draw` (or `build`); the other three come from the type or from the recorder; a `draw` without `pass_` gets a wrapper once (signature checked at registration), so every renderer still calls `draw(ctx, state, pass_)` and nothing in `render/` changed. A scene file may register its type again (a script run twice); built-ins never change |
| Ported types | `domains/basic.py`, `domains/cx.py` | `Dot`, `Line`, `DomainColoring` and `Fractal` use `draw(ctx, state)`; `DomainColoring` and `Fractal` no longer write `world_bounds` (derived from their one field quad). Inspector output frozen before the port in `tests/golden/contract/ported_inspector.json` (2D and orbit, three times) and byte-identical after |
| Composites | `time/compile.py`, `core/record.py`, `inspect/inspector.py`, `inspect/lint.py` | `build(state) -> [objects]`; parts get `<composite>.<name>`, are checked against document ids and `retired_ids` at every evaluation (`bad_part`), take the composite's layer, opacity and z, and are objects of their own in `SceneState` (`ObjectState.parent`). Bounds are the union, features the union box's plus `<part>.<feature>`. The inspector sees a composite where its parts are (union of their solo masks) and marks parts `"part_of"`; overlap lint ignores a composite against its own parts |
| Scene-local types | `core/schema.py`, `core/migrate.py`, `engine.py`, `scene.py` | schema 8: `meta.modules` (relative `.py` files in the scene's folder) and `ObjectSpec.always`; migration 7 to 8. The Engine loads a module only where scripts may run (`allow_scripts`; `load("x.py")` turns it on; the MCP workspace passes `--allow-scripts`), once per file change, with `files.resolve`; a failed load rolls back the type and updater registries and `sys.modules`. `Scene.add` names the module of a type or updater defined in the folder. Render workers get the permission when the parent had all the code |
| `missing_type` | `inspect/lint.py`, `core/objects.py` (`Placeholder`) | a missing, refused or broken module (or an updater nobody defined): error with `times`, no `fix`; the object draws as a placeholder box at its `pos`, keeps every parameter (unknown ones stay readable by expressions), its tracks and overrides; saving it writes the same bytes |
| Updaters | `time/updaters.py`, `time/compile.py` | `obj.always(fn)`, `fn(t, look) -> {param: value}`; each parameter it sets is an `UpdaterOut` node (a `CompiledExpr`, so `not_movable`, `track_on_expr` and motion blur treat it as an expression) whose edges are the parameters `look` read at t = 0; a cycle is the usual `cycle` error; a read it did not declare is an error at evaluation, never a stale value; it runs once per evaluation and leaves out what it does not return. `state_at` stays pure and frozen |
| `impure_updater` | `inspect/lint.py`, theme `[lint] impure` | in `lint(strict=True)`: every updater is called again at every sample time, twice, the second pass in reverse order of times and of updaters, on the same evaluated parameters; answers that differ by more than the threshold are an error with `times`, no `fix` |
| `Value` | `domains/basic.py` | a named animatable number with no drawing (`kind: value`, `drawn: 0`, never `orphan`) |
| Docs | `docs/ai/API.md`, `docs/ai/MCP.md`, `llms.txt` | the contract section with running examples; `object_types` marks `scene_local` types and `llms.txt` lists only shipped ones |
| Demo | `examples/open_contract.py` | `Diamond` (Params + draw), `Flag` (a composite of a `Line` and a `Diamond`), `Value` `turn` driving the planet's updater, a label kept on the planet by another updater, a `near` constraint on the composite; storyboard golden `tests/golden/software/open_contract.png` |

## Verification

- `uv run pytest` (full run, 75 min on a shared 4-core box): **1512 passed, 125 skipped, 5
  failed**. Three were this milestone's and are fixed (a `model_copy` that the contract-1
  grep in `tests/test_contracts.py` rightly flagged, the example's docstring naming a
  milestone, the example missing from `kit/examples.toml`); those files re-run green. Two
  fail the same way on the branch's starting commit and are not from this milestone:
  `test_prism_domain_golden_storyboard` (SSIM 0.958) and `test_prism_heat_golden_storyboard`
  (0.798) against the software goldens on this llvmpipe box. Skips: CUDA (no GPU here),
  opt-in packaging builds, empty parametrizations. ruff, ruff format and lint-imports clean;
  `phasor doctor` tier B (llvmpipe OpenGL 4.5, no CUDA).
- `tests/test_open_contract.py` (37 tests): the recorder (every command, pixels never
  computed, cache and re-recording when a read object changes); registration (two members,
  `pass_` kept for a draw that takes it, re-registration by the same file only); the four
  ported types' inspector output byte-identical; composites in both lint MODES (ids, indices,
  union bounds, features, `part_of`, visibility through the parts, opacity, `bad_part` for a
  retired, taken or bad part id); scene-local modules (refused, loaded, missing, broken with
  nothing left in the registry or `sys.modules`, outside the folder, a scene script run twice,
  the migration, a render worker loading what its parent ran, and a render through two
  spawned worker processes byte-identical to a serial one); updaters (a label on a moving
  dot in both MODES, chained updaters, cycles, undeclared reads, unknown parameters, an
  updater from outside the folder, a missing one); `impure_updater` in both MODES (a counter
  and state shared between two updaters fire; the writer and chained pure updaters do not);
  `Value`; the exit demo through inspector, layout, strict lint and a storyboard golden.
- Looked at the golden storyboard: the planet goes round the sun with its name above it, the
  composite's label sits beside the flag where the placer put it.

- Gate (`phasor-milestone-gate`): mechanical gates as above; exit demo run
  (`tests/test_open_contract.py::test_exit_demo_*`, both lint MODES, golden looked at); the
  contract audit was done by the authoring model, **not independently** (no subagent was
  available in the building session); it found deviation 4 below touching contract 3.

## Deviations

1. **The recording cache key is wider than ADR 0005's "(id, resolved params)".** Bounds depend
   on the camera (stroke widths, billboard basis) and on resolved style, so the key is (type,
   id, params, style, camera, scene folder), and a `draw` that reads another object records the
   read and is re-recorded when that object changes. Keyed on (id, params) alone, a box
   would go stale when the camera moves or a read object changes (both are tested).
2. **Derived features add `curve`** (the stroke's points) next to `start` and `end` for a
   one-stroke object: layout judges thin strokes by it, as it does for `Line`.
3. **Porting kept most members.** Derived defaults would change `Dot` (its `kind: point`
   semantics, which layout and lint read; its exact ellipse box in perspective) and `Line`
   (`kind: polyline`; its box padded by half the width, not the mitred corners), so those two
   only drop `pass_`. `DomainColoring` and `Fractal` also drop `world_bounds`. SPEC's "types
   shipped in phasor's packages still carry the full set" is read as: every type has all five
   members after registration, and shipped types keep their docs, example and goldens.
4. **Updaters defined outside the scene's folder** (a notebook, a doc page, a test file
   elsewhere) are allowed and written `"<file name>:<function>"`; they work only in the
   session that defined them, and elsewhere the document opens with `missing_type`. ADR 0005
   only says where scene code is loaded from; refusing these would have made `always`
   unusable in notebooks and in `docs/ai/API.md`. **This touches contract 3** (the state then
   depends on code the document does not name a file for, as a type from an installed
   package always has) **and needs the owner's OK**; the alternative is to refuse them in
   `register_updater` (one line) and mark the doc examples `no-run`.
5. **`look` reads parameters before living motion.** Life is applied after the parameter
   graph, so a label kept on a wandering dot is off by the wander; the demo tags its planet
   `exact`. Reading through life would make `look` depend on style and tags.
6. **`impure_updater` runs only in `lint(strict=True)`**, as ADR 0005 says, not in plain lint.
7. **Parts of a composite get no living motion** and constraints cannot name them (they are
   not document objects); a part is placed through its composite.
8. **Updater reads are discovered at t = 0.** An updater must read the same parameters at
   every t (branches that read something new later are an error, not a silent edge).
9. **No `--allow-scripts` on the CLI.** `phasor render scene.py` runs the script (and so its
   types); a saved `.phasor.json` that names modules opens through the CLI with
   `missing_type`. `cli.py` is shared with Milestone 14's work, so it was left alone.
10. `Value` has no style roles (it draws nothing); `tests/test_final.py` exempts it from
    "has style roles" and "draws something" (`DRAWLESS`), and lists only shipped types.

## Bugs caught while building

1. Test files register types at import, so `llms.txt` listed them: `object_types` now marks
   types not shipped in phasor `scene_local`, and `llms.txt` leaves them out.
2. A `Params` model written with text annotations in code that is not an importable module (a
   doc block run by `exec`, a notebook cell) could not resolve `Vec3`: `register` now resolves
   such annotations where it was called.
3. `source_file` turned "no source file" into the working folder.
4. A first cycle test added an updater that read an object that did not exist yet; the error
   was right (`bad_updater`) and the test was wrong.

## Untested

- Windows and CUDA (nothing here touches the GPU path, but the renderer calls the wrapped
  `draw` on every frame there too).
- Very large composites (hundreds of parts: each evaluation rebuilds them).
