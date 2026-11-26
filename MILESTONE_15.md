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
| Ported types | `domains/basic.py`, `domains/cx.py` | `Dot`, `Line`, `DomainColoring` and `Fractal` use `draw(ctx, state)`; `DomainColoring` and `Fractal` no longer write `world_bounds` (derived from their one field quad). Inspector output (analytic mode) frozen in `tests/golden/contract/ported_inspector.json` (2D and orbit, three times); separately, the full inspector output and lint of that scene in analytic **and pixel** mode were compared byte for byte between this branch and its starting commit 417a7af on this machine: identical |
| Composites | `time/compile.py`, `core/record.py`, `inspect/inspector.py`, `inspect/lint.py` | `build(state) -> [objects]`; parts get `<composite>.<name>`, are checked against document ids and `retired_ids` at every evaluation (`bad_part`), take the composite's layer, opacity and z, and are objects of their own in `SceneState` (`ObjectState.parent`). Bounds are the union, features the union box's plus `<part>.<feature>`. The inspector sees a composite where its parts are (union of their solo masks) and marks parts `"part_of"`; overlap lint ignores a composite against its own parts |
| Scene-local types | `core/schema.py`, `core/migrate.py`, `engine.py`, `scene.py` | schema 8: `meta.modules` (relative `.py` files in the scene's folder) and `ObjectSpec.always`; migration 7 to 8. The Engine loads a module only where scripts may run (`allow_scripts`; `load("x.py")` turns it on; the MCP workspace passes `--allow-scripts`), once per file change (a changed module is run again, contract 3), with `files.resolve`; a failed load rolls back the type and updater registries and `sys.modules`. `Scene.add` names the module of a type or updater defined in the folder. Render workers get the permission when the parent had all the code |
| `missing_type` | `inspect/lint.py`, `core/objects.py` (`Placeholder`) | a missing, refused or broken module (or an updater nobody defined): error with `times`, no `fix`; the object draws as a placeholder box at its `pos`, keeps every parameter (unknown ones stay readable by expressions), its tracks and overrides; saving it writes the same bytes |
| Updaters | `time/updaters.py`, `time/compile.py` | `obj.always(fn)`, `fn(t, look) -> {param: value}`; each parameter it sets is an `UpdaterOut` node (a `CompiledExpr`, so `not_movable`, `track_on_expr` and motion blur treat it as an expression) whose edges are the parameters `look` read at t = 0; a cycle is the usual `cycle` error; a read it did not declare is an error at evaluation, never a stale value; it runs once per evaluation and leaves out what it does not return. `state_at` stays pure and frozen |
| `impure_updater` | `inspect/lint.py`, theme `[lint] impure` | in `lint(strict=True)`: every updater is called again at every sample time, twice, the second pass in reverse order of times and of updaters, on the same evaluated parameters; answers that differ by more than the threshold are an error with `times`, no `fix` |
| `Value` | `domains/basic.py` | a named animatable number with no drawing (`kind: value`, `drawn: 0`, never `orphan`) |
| Docs | `docs/ai/API.md`, `docs/ai/MCP.md`, `llms.txt` | the contract section with running examples; `object_types` marks `scene_local` types and `llms.txt` lists only shipped ones |
| Demo | `examples/open_contract.py` | `Diamond` (Params + draw), `Flag` (a composite of a `Line` and a `Diamond`), `Value` `turn` driving the planet's updater, a label kept on the planet by another updater, a `near` constraint on the composite; storyboard golden `tests/golden/software/open_contract.png` |

## Verification

- `uv run pytest`, final full run at the last commit (75 min on a shared 4-core box):
  **1518 passed, 125 skipped, 2 failed**. The two failures are not from this milestone: they
  fail with the same scores on the branch's starting commit 417a7af on this machine
  (`test_prism_domain_golden_storyboard`, SSIM 0.958; `test_prism_heat_golden_storyboard`,
  0.798; software goldens against llvmpipe). Skips: CUDA (no GPU here), opt-in packaging
  builds, empty parametrizations. An earlier full run also caught three failures of this
  milestone, now fixed: a `model_copy` that the contract-1 grep in `tests/test_contracts.py`
  rightly flagged, the example's docstring naming a milestone, the example missing from
  `kit/examples.toml`. ruff, ruff format and lint-imports clean; `phasor doctor` tier B
  (llvmpipe OpenGL 4.5, no CUDA).
- `tests/test_open_contract.py` (38 tests): the recorder (every command, pixels never
  computed, cache and re-recording when a read object changes); registration (two members,
  `pass_` kept for a draw that takes it, re-registration by the same file only); the four
  ported types' inspector output byte-identical; composites in both lint MODES (ids, indices,
  union bounds, features, `part_of`, visibility through the parts, opacity, `bad_part` for a
  retired, taken or bad part id); scene-local modules (refused, loaded, missing, broken with
  nothing left in the registry or `sys.modules`, outside the folder, a scene script run twice,
  the migration, a render worker loading what its parent ran, and a render through two
  spawned worker processes byte-identical to a serial one); updaters (a label on a moving
  dot in both MODES, chained updaters, cycles, undeclared reads, unknown parameters, an
  updater from outside the folder refused, a missing one, a changed module read again); `impure_updater` in both MODES (a counter
  and state shared between two updaters fire; the writer and chained pure updaters do not);
  `Value`; the exit demo through inspector, layout, strict lint and a storyboard golden.
- Looked at the golden storyboard: the planet goes round the sun with its name above it, the
  composite's label sits beside the flag where the placer put it.

- Gate (`phasor-milestone-gate`): mechanical gates as above; exit demo run
  (`tests/test_open_contract.py::test_exit_demo_*`, both lint MODES, golden looked at); the
  contract audit was done by the authoring model, **not independently** (no subagent was
  available in the building session). It found one contract problem, fixed before handing
  back: updaters defined outside the scene's folder had been allowed (contract 3); they are
  refused now (see the proposal below).

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
4. **Updaters must live in a file in the scene's folder** (contract 3). The proposal for
   session-only references (`"<file>:<name>"` that open elsewhere as `missing_type`) was
   **refused by the owner** (2026-11-24): the state would depend on code the document names
   no file for. Instead (owner's idea, ADR 0006, branch `claude/notebook-updaters`, built, for
   review): an updater defined in a notebook cell or at a prompt is held by the `Scene`
   builder, never by the document, and `Scene.save_copy(path)` writes its source into
   `name.updaters.py` beside the copy, which names it (`meta.modules`, `always`, via the new
   patch op `set_always`). Only a small subset is captured (top-level `def`; builtins,
   importable modules and functions, plain-data constants); the rest raises
   `UpdaterCaptureError` at `add`. A file is never overwritten. The open scene does not
   change. `core/capture.py`, `tests/test_updater_capture.py`.
5. **`look` reads parameters before living motion.** Life is applied after the parameter
   graph, so a label kept on a wandering dot is off by the wander; the demo tags its planet
   `exact`. Reading through life would make `look` depend on style and tags.
6. **`impure_updater` runs only in `lint(strict=True)`**, as ADR 0005 says, not in plain lint.
7. **Parts of a composite get no living motion** and constraints cannot name them (they are
   not document objects); a part is placed through its composite.
8. **Updater reads are discovered at t = 0.** An updater must read the same parameters at
   every t (branches that read something new later are an error, not a silent edge).
9. **No `--allow-scripts` on the CLI. Resolved (2026-11-23, commit below).** `phasor inspect`,
   `lint`, `storyboard`, `render`, `web`, `bench --frames` and `sketch` take `--allow-scripts`
   (default off; one helper, `cli._open`); the flag reaches render workers through the
   engine's `worker_payload`. Tests: `tests/test_open_contract.py` (`test_the_cli_*`).
10. `Value` has no style roles (it draws nothing); `tests/test_final.py` exempts it from
    "has style roles" and "draws something" (`DRAWLESS`), and lists only shipped types.
11. **The `unfold` verb (added 2026-11-25, after 15) keeps its maths in `core/unfold_math.py`,
    not `domains/<name>_math.py`.** The verb lives in `time`, which may not import `domains`
    (contract 6), and the `HingedCopy` object in `domains/unfold.py` needs the same maths;
    `core` is the lowest layer both can import. Pure NumPy, no drawing, as the domains rule
    asks. `HingedCopy` has no style roles (it draws its source's look);
    `tests/test_final.py` exempts it as `BORROWED`.

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
