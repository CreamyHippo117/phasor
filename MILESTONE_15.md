# Milestone 15: the open object contract (ADR 0005)

Accepted by the owner on 2026-11-23 (`docs/decisions/0005-open-object-contract.md`, the
"Milestone 15" section of `MANIM_PARITY.md`). Exit demo: **a scene script defines a new type
in about 10 lines (`draw` only); the inspector, layout, lint and storyboard all work on it,
and an `always` updater keeps a label on a moving dot** (`examples/open_contract.py`).

Status: **built; gates green on Linux (see Verification); independent audit done 2026-11-23, blockers fixed** (Deviations 11-15). Untested on Windows and CUDA.

## Built

| Area | Where | What |
| --- | --- | --- |
| Recorder | `phasor/core/record.py` | a `DrawContext` that keeps geometry and draws nothing (pure NumPy): triangles, surfaces, text boxes (`core/text.quad`), and a `field`'s quad corners without ever calling its `pixels`. Derived `world_bounds` (box of everything drawn), `features` (`center`, the box's compass points, and `start`/`end`/`curve` for one stroke, noted by `domains/strokes.py`), `semantics` (`{"type", "params"}`). Cached per (type, id, params, style, camera, folder), and made again when an object it read changes |
| Two-member contract | `phasor/core/objects.py` | `register` needs `Params` and `draw` (or `build`); the other three come from the type or from the recorder; a `draw` without `pass_` gets a wrapper once (signature checked at registration), so every renderer still calls `draw(ctx, state, pass_)` and nothing in `render/` changed. A scene file may register its type again (a script run twice); built-ins never change |
| Ported types | `domains/basic.py`, `domains/cx.py` | `Dot`, `Line`, `DomainColoring` and `Fractal` use `draw(ctx, state)`; `DomainColoring` and `Fractal` no longer write `world_bounds` (derived from their one field quad). Inspector output (analytic mode) frozen in `tests/golden/contract/ported_inspector.json` (2D and orbit, three times); separately, the full inspector output and lint of that scene in analytic **and pixel** mode were compared byte for byte between this branch and its starting commit 417a7af on this machine: identical |
| Composites | `time/compile.py`, `core/record.py`, `inspect/inspector.py`, `inspect/lint.py` | `build(state) -> [objects]`; parts get `<composite>.<name>`, are checked against document ids and `retired_ids` at every evaluation (`bad_part`), take the composite's layer, opacity and z, and are objects of their own in `SceneState` (`ObjectState.parent`). Bounds are the union, features the union box's plus `<part>.<feature>`. The inspector sees a composite where its parts are (union of their solo masks) and marks parts `"part_of"`; overlap lint ignores a composite against its own parts |
| Scene-local types | `core/schema.py`, `core/migrate.py`, `engine.py`, `scene.py` | schema 8: `meta.modules` (relative `.py` files in the scene's folder) and `ObjectSpec.always`; migration 7 to 8. The Engine loads a module only where scripts may run (`allow_scripts`; `load("x.py")` turns it on; the MCP workspace passes `--allow-scripts`), once per file change (a changed module is run again, contract 3), with `files.resolve`; a failed load rolls back the type and updater registries and `sys.modules`. `Scene.add` names the module of a type or updater defined in the folder. Scene code is registered under its own file and a document reaches it only through its own `meta.modules` where scripts may run (`CodeScope`, Deviation 11). Render workers get exactly the engine's own permission |
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
- **Independent audit done 2026-11-23, blockers fixed.** It found two blockers and three
  should-fix problems (Deviations 11-15, each with the test that guards it in
  `tests/test_open_contract.py`, section 7) and five notes (Deviation 16, Deferred).

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
4. **Updaters must live in a file in the scene's folder** (contract 3). One defined in a
   notebook cell or at a prompt is refused by `always`/`Scene.add`, so the updater block in
   `docs/ai/API.md` is `no-run` (the example and the tests run the same code). **Proposal for
   the owner** (not built): allow such updaters as session-only references
   (`"<file>:<name>"`) that open elsewhere as `missing_type`; it would make `always` usable in
   notebooks but makes the state depend on code the document names no file for, so it needs
   a contract 3 change.
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

Found by the independent audit (2026-11-23; repro scripts x1-x11), and how they were fixed:

11. **Blocker: a document that may not run scripts ran code registered earlier** (x1, x9).
    `needs_code` asked the process-wide registries, so a type or updater any script had
    registered (another scene's, or this folder's script the person ran) was used with no
    `missing_type` and no check that it came from the document's own files. Fixed by scoping
    scene code to its file: code a script defines (`python x.py`, `load`, `runpy`, a module
    the engine loads: `objects.script_of`, decided from the class's own functions before
    `draw` is wrapped) is registered under (resolved file, name), and a document reaches it
    only through a `CodeScope`: the files of its own `meta.modules`, and only where scripts
    may run. Without `allow_scripts` a document gets shipped and imported code only, even in
    the script's own folder. Tests: `test_audit_x1_a_document_that_may_not_run_scripts_*`,
    `test_audit_x1_an_engine_that_loaded_a_module_lends_nothing_to_another`,
    `test_audit_where_code_comes_from`, `test_audit_a_short_form_draw_is_still_scene_code`.
12. **Blocker: a render worker could get permission its parent did not have** (x11).
    `worker_payload` sent `allow_scripts or not missing`, and because of 11 nothing was
    missing whenever the names were registered in the parent, so a fresh worker ran the
    document folder's `scene.py`. The payload now carries `self.allow_scripts` only. Test:
    `test_audit_x11_a_worker_never_gets_more_permission_than_its_parent` (a real subprocess
    worker; a marker file proves the folder's code never ran).
13. **Two scenes in one process collided** (x1, x9): `scene.py:orbit` of one replaced the
    other's, and a second scene defining the same type name failed to register. Fixed by 11
    (keys are per file). A scene's own type still may not take a shipped type's name. Tests:
    `test_audit_x1_two_scenes_with_the_same_names_never_see_each_others_code` (alternating
    `apply`/`state_at`/`fork`), `test_audit_scene_code_cannot_take_a_shipped_name`.
    Consequences visible to authors (in `docs/ai/API.md`): a `Scene(...)` built by a script
    may run that script's code (the immediate caller decides, so an importer inside a server
    lends nothing); one built by imported code (a test) may not unless it says so; an
    `Engine` made from a scene's document needs `allow_scripts=True` to keep the scene's
    types; render workers of an engine that may not run scripts draw imported types (a test
    module's) as placeholders, since a fresh process has not imported them. One existing test
    had passed only through finding 11: `tests/test_final.py`'s save-and-load of
    `examples/open_contract.py` saved into an empty folder and reopened without scripts; it
    now copies the named module beside the save and opens it with `allow_scripts=True`, as a
    person moving the scene would. Imported code (a
    package, a test module, a notebook cell) stays shared by the process and usable without
    `allow_scripts`, as phasor's own types are: **a trust decision for the owner** (it is code
    the person imported, not code a document names).
14. **An updater's parameter returned only after t = 0 was dropped silently** (x2); and
    strict lint returned nothing for a scene whose `state_at(2.0)` raised when the scene had
    no markers. A parameter the updater did not return at compile time is now an error naming
    it and the time; `lint(strict=True)` calls every updater at every sample time and over at
    least the first `PROBE_SPAN` (2 s), and reports one that fails as `bad_updater` with its
    times (it swallowed such failures before, or crashed). Tests:
    `test_audit_x2_an_updater_must_set_the_same_parameters_at_every_t`,
    `test_audit_x2_strict_lint_probes_updaters_beyond_t0`.
15. **The recording cache ignored files and time** (x4). The key now holds each `Files`
    file's path, modification time and size (`files.stamp`, the (mtime, size) rule the mesh
    and network loaders use), so a changed file is drawn again. A `draw` that reads `state.t`,
    `state.values` entries or any other part of the state has those reads remembered and
    compared on a hit, like reads of other objects (chosen over a lint: it is always right and
    keeps the cache for draws that read nothing more). Test:
    `test_audit_x4_a_recording_follows_the_files_and_the_time_a_draw_reads`.
16. Notes fixed because they were small: the recording's points and stroke curves are
    read-only copies (x10; `test_audit_x10_cached_geometry_is_read_only`); a composite part
    with an updater is `bad_part` instead of a part that silently never moves
    (`test_audit_a_part_with_an_updater_is_an_error_not_silently_still`); a module's updaters
    are adopted only by plain top-level names, never a dotted path
    (`test_audit_only_plain_names_are_adopted_as_updaters`); looking up, loading and compiling
    a document's code holds a process lock (`engine._CODE_LOCK`) and the recording cache has
    its own.

## Deferred

- `build()` purity is not checked (a composite whose parts differ for the same t is not
  caught the way `impure_updater` catches updaters).
- `patch.param_value` (a control on a parameter the document leaves at its default) reads
  the type's default from shipped and imported types only; for a scene's own type it does not
  find the default (it has no engine, so no `CodeScope`).
- The registries are still process-wide dictionaries; reads of them are not locked (only
  loading and compiling is).

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
