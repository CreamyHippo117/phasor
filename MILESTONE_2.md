# Milestone 2: Layout + linter

Roadmap row: kiwisolver stage, label placer with hysteresis, layout bake, lint rules,
storyboard with marks. Exit demo: **a deliberately cluttered scene is flagged, then
auto-fixed.**

Status: **built; all gates green on Linux with software OpenGL and with OpenGL blocked.**
Untested on Windows and on a real GPU. Fresh-eyes boundary review not run (see the end).

## Built

| Area | Where | What |
| --- | --- | --- |
| Constraint language | `core/constraints.py` | `a above b gap 0.2`, `label(x) near x`, guards, `and`; errors say why |
| Schema v2 | `core/schema.py`, `core/migrate.py` | `constraints` in the document; v1 files migrate on load |
| Patch ops | `core/patch.py` | `add_constraint`, `remove_constraint`; on scripted scenes they go to the overrides file |
| Plan | `layout/plan.py` | which objects move and which stage moves them; impossible rules are rejected with a reason |
| Stage 1 | `layout/solver.py` | kiwisolver: above/below/left_of/right_of/align_x/align_y/inside/on_screen, four priorities |
| Stage 2 | `layout/labels.py` | 8 compass + 8 leader slots; cost = overlap + occlusion + distance + movement; 20% hysteresis |
| Bake | `layout/bake.py` | solves at markers, transition ends and 10 Hz; `baked_state(t)` is what everyone reads |
| Occlusion oracle | `layout/bake.py` (protocol), `inspect/oracle.py` (id pass) | dependency inversion per ADR 0001 |
| Leader lines | `domains/basic.py` | `Text.leader`, `Text.leader_on`, drawn as a hairline |
| Linter | `inspect/lint.py` | overlap, offscreen, contrast, tiny, orphan, clutter, unsolved (+ orphan_override) |
| Auto-fix | `Engine.autofix`, `Issue.fix` | issues carry patch ops; applied through `Engine.apply`, one undo step per round |
| Storyboard | `Engine.storyboard`, `render/storyboard.py` | contact sheet with time and marker captions, set-of-mark overlay, inspector lines |
| CLI | `cli.py` | `phasor lint [--fix] [--strict] [--json]`, `phasor storyboard [--marks]` |
| Theme | `style/themes/midnight_glow.toml` | `[layout]` gaps and `[lint]` thresholds |
| Docs | `docs/ai/` | constraints, lint, auto-fix, storyboards; every example runs as a test |

## Verification

- `uv run pytest`: **256 passed, 1 skipped** (CUDA). With `moderngl` blocked (tier C, as on
  the Windows CI runner): 206 passed, 51 skipped.
- `ruff check`, `ruff format --check`, `lint-imports` (2 contracts kept): clean.
- **Exit demo** (`examples/cluttered.py`, tested in `tests/test_lint.py`, with real pixels
  and with geometry only):
  - before: 4 warnings (title cut by the frame; three label collisions);
  - `autofix()` adds `on_screen(title)` and one `near` rule per label;
  - after: 0 issues. Checked independently of the linter: at 30 Hz every label and pole is
    fully on screen and no label box touches a pole; no label changes slot more than once.
  - the script is never edited; the fixes are in `cluttered.overrides.json` and survive reload.
- Glue: with `label(dot) above dot gap 0.2` the gap is exact to 1e-9 at 53 times that are
  mostly not key times, while the dot moves with easing.
- Purity: two engines queried in opposite time order give identical states.
- Hysteresis: a held slot survives a wobbling obstacle for 40 steps; it is left only when
  another slot costs under 80% of it.
- Storyboard goldens: `tests/golden/software/cluttered_{before,after}.png` (SSIM >= 0.99).

## Bugs the tests caught while building

1. **Slot ties were decided by rounding noise** (`0.07999…` vs `0.08`), so "above" and
   "below" swapped unpredictably. Costs closer than 1e-9 are now equal and slot order decides.
2. **A label sliding between opposite slots crossed its own anchor.** The 30 Hz ground-truth
   check found it between lint samples. Slot changes are now jumps.
3. **Hysteresis kept a label in a covered slot while a clean one existed.** The weights were
   retuned so a slot covered by more than a few percent always loses.
4. **Milestone 1's golden test filed software renders under `gpu/`.** It read the renderer
   name without making the GL context current. Fixed (`GLRenderer.renderer_class()`); the
   three misfiled images are deleted. No real-GPU golden has ever existed.

## Deviations from SPEC (SPEC updated in the same commit unless marked)

1. **`align` is two relations, `align_x` and `align_y`.** One word could not say which axis.
2. **`label(x)` means the object `x.label`.** The spec used `label(...)` without defining it.
3. **Not built yet: `outside`, `tangent_to`, `attach_to`, `visible_in_3d`** (need 3D or
   non-linear solving; Milestone 6). Using one is a clear error naming the milestone.
4. **Lint rules `occluded_label` and `z_fight` wait for 3D** (Milestone 6). In 2D a covered
   label is reported as `overlap`.
5. **`orphan` means "never appears".** The spec's other half, "never leaves", would flag
   every object in every scene until there are enter/exit verbs (Milestone 5).
6. **`raw_style` cannot occur:** the schema only accepts role names, so a raw color is
   already an `apply` error (`bad_role`). No lint rule was added.
7. **Guards `no_overlap`, `min_size_px`, `readable` are checked, not solved** (only
   `on_screen` moves things). A broken guard is an `unsolved` error.
8. **Slot changes jump** (ADR 0002). Eased moves belong to choreography verbs (Milestone 5).
9. **Lint samples every 0.5 s plus markers and transition ends**, so a collision shorter
   than that between samples can be missed. The 10 Hz event log is Milestone 6.
10. **Needs owner approval, SPEC contract text not changed:** constraints in the overrides
    file (contract 2), and the bake depending on the occlusion oracle (contracts 3 and 7).
    Both are in `docs/decisions/0002-layout-bake.md`.

## Deferred (by the roadmap, not forgotten)

- Groups in rules (`no_overlap(ir.labels)`), `free_space_near`, `diff` (with MCP, Milestone 5).
- Camera solver and 3D label placement (Milestone 6); aesthetic lint rules (Milestone 7).
- ID tombstones and parent transforms, still open from Milestone 1.
- Undo keeps evaluated caches only for the current and previous step; older steps recompute.

## Untested on hardware

- Windows, and any real GPU. CI runs the non-GL tests on Windows.
- No `gpu`-class goldens exist. Create them on an NVIDIA machine with
  `PHASOR_UPDATE_GOLDEN=1 uv run pytest -k golden` and look at them before committing.

## Owner decisions still open

1. ADR 0001 (layers), from Milestone 0.
2. ADR 0002, both points.
3. ID reuse: enforce with tombstones, or relax the wording (from Milestone 1).

## Boundary review

The contract audit for this milestone was done by the same session that wrote the code,
plus the static guards in `tests/test_contracts.py` (two new ones for contract 7). The
build rules ask for a reviewer that did not write the code; that has not been run.
