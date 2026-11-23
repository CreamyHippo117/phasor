# Milestone 18: style freedom

Plan: `MANIM_PARITY.md`, "Milestone 18". Owner's answer 2 (2026-11-23): explicit colors and
sizes in scenes are **allowed**; they are checked by `off_palette` / `off_scale` and
snapped by `polish()`. Exit demo: **a scene with an explicit color, size and background
renders correctly; `off_palette` and `off_scale` warn; `polish` offers the snap; nothing is
applied silently; old documents load unchanged** (`examples/own_colors.py`).

Status: **built; focused gates green on Linux (see Verification)**. The full suite was not
run (over an hour here). Contract audit by the authoring model, **not independent**.
Untested on Windows, macOS and CUDA. **Independent audit done 2026-11-23; its findings are
fixed** (see "Audit" below).

## Built

| Area | Where | What |
| --- | --- | --- |
| Document | `core/schema.py`, `core/migrate.py`, `core/patch.py` | schema 9. A style value is a role name, or an explicit value: a color (`"#rrggbb"`, `"#rgb"`, `"#rrggbbaa"`, stored upper case; or linear RGBA, 3 or 4 numbers in 0..1, stored as 4) or a size (a positive number of world units, stored as float). Bools, NaN, infinities, out-of-range colors and bad hex are refused. `meta.background` (a color or None: the theme's). `ObjectSpec.style`, `Overrides.styles` and `SetStyle.role` take the wider value. Migration 8 to 9 adds `meta.background: None`; an old document reads the same |
| Resolution | `style/explicit.py`, `time/compile.py` | which groups take which explicit value (`emphasis`: color; `text`, `stroke`: size; glow, surface, colormap: roles only, `bad_role` otherwise). A hex color is sRGB-decoded to linear once, when the document compiles (contract 4); a tuple is linear already. A world size is turned into the share of the frame height that every `draw` already multiplies back by `units_per_height`, in `evaluate` (pure: document and t), through the camera of the object's layer at its `pos`; composite parts the same way. `roles[key]` holds the value the scene typed, for lint and the UI. No draw code changed |
| Background | `time/compile.py`, `render/passes.py` | `SceneState.background` is the scene's or the theme's. The field cache key now includes the background (Fractal, cloud and image fields fill empty parts with it; the renderer is shared between scenes) |
| Looks | `style/looks.py` | a typed color wins over the look: `Look.fill` adds no `hue_along` or colormap to an object whose color is explicit (in the prism look a typed red line was drawn as a white-hot spectrum before this). Roles get the look as before (prism_primes golden unchanged) |
| Scene API | `scene.py`, `core/pycode.py` | `Scene(..., background=...)`; `style={"color": "#58C4DD"}`, `{"size": 0.4}`, `{"stroke": 0.05}`; generated scripts carry `background=` |
| `off_palette` | `inspect/aesthetic.py`, theme `[lint] off_palette = 0.12` | aesthetic warning: an explicit color farther (OKLab) than the threshold from every emphasis role of the theme, or a scene background from the theme's; `times`, `path` (`<id>.<key>` or `meta.background`), no `fix` |
| `off_scale` | `inspect/aesthetic.py`, theme `[lint] off_scale = 1.25` | aesthetic warning: an explicit text size or stroke width more than the ratio off the nearest role of its group, measured at each beat (so a zoom that pushes it off is reported with its times); no `fix` |
| Snap, offered | `inspect/critic.py`, `engine.py`, `workspace/tools.py` | `critique()` lists a third kind of op per issue, `snap` (the nearest role; a document's background back to None), next to `fix` and `suggested`. `polish(snap=False)` by default; only `polish(snap=True)` (Engine and the MCP tool) applies them. `autofix` and `variants` never touch an explicit value |
| Guards | `inspect/aesthetic.py`, `inspect/critic.py`, `gui/window.py` | `stroke_mix` counts roles only (an explicit width is `off_scale`'s; mixing str and float would also have crashed its sort); the `focal_point` suggestion never proposes `highlight` for an object whose color was typed; the inspector shows a typed value as the first entry of the role menu and writes nothing back unless a role is picked |
| Manim import | `importers/manim.py` | colors the script sets (`RED`, `"#58C4DD"`), `stroke_width` / `set_stroke(width=)`, `font_size` and a changed `camera.background_color` become explicit values (sizes calibrated so Manim's defaults land on the `math` and `regular` roles); a class's own default color stays the nearest role; `scale` scales an explicit text size (warning removed). The reverb golden is regenerated (looked at: Manim's own colors and font sizes) |
| Docs | `docs/ai/API.md`, `docs/ai/MCP.md`, `docs/ai/PATTERNS.md`, `llms.txt`, both `phasor` skill copies, `kit/CLAUDE.md`, `BEAUTY.md`, `SPEC.md` | the new rule, the table of what each group takes, a running example; the Manim table; SPEC's Beauty-system line, aesthetic table and art-director pass (non-contract text) |
| Demo | `examples/own_colors.py`, `kit/examples.toml` | prism look, a navy background near the theme's (quiet), a typed cyan near `secondary` (quiet), a typed green (`off_palette`), a 0.12-unit orbit (`off_scale`, snap to `bold`); storyboard golden `tests/golden/software/own_colors.png` |

## Verification (2026-11-23, Linux, llvmpipe, no CUDA)

- `tests/test_style_freedom.py` (25 tests, written before or with the code): canonical
  storage and refusals; migration 8 to 9 (an old document reads the same, roles and
  background); save/load byte-identical with explicit values; hex decoded to linear once,
  tuple kept; world sizes at two zooms and in 3D at `pos`; roles-only groups refused;
  `set_style` and overrides with explicit values; the background in the state and through
  `set_meta`; `off_palette` (colors and background) and `off_scale` (text and stroke) in
  both lint MODES; `off_scale` reported only after a zoom; `stroke_mix` and `autofix` leave
  an explicit width alone; critique offers the snap, `polish()` and `autofix` change no typed
  value, `polish(snap=True)` snaps; a scripted scene's background gets no snap, a document's
  does; exact pixels (hex, linear tuple, background) through the GL renderer; the exit demo's
  warnings and snaps; the storyboard golden.
- Other tests changed: four encoded the old roles-only rule (`test_engine` invalid patch,
  `test_time` all problems, `test_mcp` style test, the importer's color test); each now
  checks a roles-only key or the new behavior. Registry tests in `test_aesthetic` and
  `test_critic` list the two new rules. A GUI test (`test_gui_window`) checks that the
  inspector neither hides nor replaces a typed color. `test_open_contract`'s migration test
  no longer pins the version to 8.
- Gate results: see "Gate" below.

## Deviations

1. **Schema bumped to 9** (the task allowed it only if the schema changed: it did, the value
   type of `style` widened and `meta.background` was added). Additive: old documents migrate
   by one `setdefault`.
2. **`Overrides` stays version 1.** Its `styles` values widened; no field was added, so an
   old overrides file reads the same. An older phasor reading a new overrides file with a
   hex color fails with `bad_role` (its schema took any string as a role name), not with a
   validation error or a version message.
3. **`set_style` keeps its field name `role`** for an explicit value too (one op, one field;
   the document's `style` mixes them the same way). A separate `value` field would have been
   a second way to do one thing.
4. **Explicit sizes are world units, converted at evaluation**, not at compile time (the
   camera can move). In 2D this is exact at any zoom; in 3D it is exact at the object's
   `pos` (the camera's target when it has none), and a stroke spanning depth varies along
   its length as role strokes do.
5. **Linear colors above 1 are refused**: an explicit color is paint; light comes from the
   glow roles and the emission target (CLAUDE.md, bloom rule).
6. **The snap of a scripted scene's background is not offered** (`set_meta` is not a tweak
   op on a scripted scene); the issue says to change the script. Snaps of composite parts are
   not offered either (a part's style is set by its composite's `build`).
7. **The field cache key gained `state.background`.** A latent bug before (two themes with
   different backgrounds could share a cached Fractal), likely with per-scene backgrounds.
8. **Manim import: font and stroke calibration is approximate** (48 pt = the `math` role's
   0.384 world units, width 4 = `regular`'s 0.032); Milestone 20 compares with real Manim.
   Class default colors stay roles, so an unstyled Manim scene keeps the house look and does
   not trip `off_palette` on every object.
9. **The reverb golden was regenerated** (it still passed at SSIM 0.97, but no longer showed
   what the import makes).
10. **Docs landed in later commits than the API change** (against CLAUDE.md's "same commit"
    rule): at commits 328b90c and bea305e, `test_mcp::test_llms_txt_is_current` and the
    three tests that encoded the old roles-only rule fail; they pass from 5ee0bd7 / ff1942a
    on. Squash the series on merge.

11. **The size bound lives in `core/schema.py`** (`MAX_SIZE`, 1e6 world units), not in
    `style/explicit.py` as the audit suggested: the schema is the one check every write path
    and `load_document` share, and `core` may not import `style`. `explicit.py` re-exports it.
12. **A background with alpha below 1 is now refused** (it was accepted and drawn opaque). A
    schema 9 document written on this branch before the audit with such a background no
    longer loads; none was released.
13. **Explicit sizes on an overlay-layer object are in frame heights** (the overlay camera's
    `units_per_height` is 1), not world units; documented in API.md, MCP.md and
    `style/explicit.py`.

## Audit (independent, 2026-11-23): what was fixed

| Finding | Fix | Guarding tests |
| --- | --- | --- |
| 1 (blocker) Manim scripts with `stroke_width=0`, `set_stroke(width=0)`, `font_size<=0`, `.scale(0)`, `.scale(-1)` died with a raw `ValidationError` | `importers/manim.py` `carried_size`: a size phasor cannot hold (<= 0, > `MAX_SIZE`) keeps the role, with a warning; a negative scale is a flip (`abs` for sizes, also a dot's radius); `text_role` no longer raises `StopIteration` below 0; any `ValidationError` / `DocumentError` while importing is a `ManimImportError` naming the object and key; `Workspace.scene_open` reports it as a `ToolError` | `test_manim_import.py`: `test_a_size_phasor_cannot_hold_is_dropped_with_a_warning` (6 cases), `test_a_negative_scale_is_a_flip_and_keeps_the_size_positive`, `test_any_invalid_value_surfaces_as_a_manim_import_error_naming_object_and_key`, `test_the_command_and_the_workspace_report_an_import_error_cleanly` |
| 2 huge ints crashed with `OverflowError` | `core/schema.py` `_float` turns it into a `ValueError` (style, color tuples, background, files on disk) | `test_style_freedom.py`: `test_huge_numbers_are_refused_cleanly_on_every_path` (ObjectSpec, `Engine.apply`, `set_style`, MCP `scene_edit`), `test_huge_numbers_in_a_background_or_a_file_on_disk_are_validation_errors` |
| 3 no upper bound on sizes (1e308 crashed `still()`) | `MAX_SIZE` = 1e6 world units (Deviation 11); API.md and MCP.md | the two tests above, `test_the_size_bound_is_finite_and_sizes_up_to_it_are_taken`, the importer's `scale_huge` case |
| 4 GUI "apply look" overwrote typed values and switched the hue back on | `Look.defaults_for` (shared by `Look.fill` and `Editor.apply_look`) leaves out typed keys and the hue along a typed color; still one undo step | `test_gui_controller.py::test_a_look_never_replaces_a_typed_value` |
| 5 background alpha accepted, then dropped | `Meta.background` refuses alpha < 1 (Deviation 12); the importer strips it with a warning; object colors keep their alpha | `test_a_background_is_opaque_and_alpha_below_one_is_refused` (Meta, Scene, `Engine.apply`, MCP), `test_a_background_with_alpha_imports_opaque_with_a_warning` |
| 6 stale AI-facing text | `mcp/server.py` INSTRUCTIONS and `docs/ai/PITFALLS.md` (`bad_role`); the skill copy and `kit/CLAUDE.md` were already current; `llms.txt` regenerated (unchanged: it does not carry the instructions) | `test_mcp.py`, `test_docs.py` |
| notes | Deviation 2 reworded; Deviation 13 (overlay units); `style_color` reuses the schema's `check_color`; the inspector uses `explicit.is_role`; `Workspace.polish` requires a real bool for `snap`; a direct palette test for the importer (the reverb golden does not guard colors) | `test_the_polish_tool_takes_snap_only_as_a_real_bool`, `test_manim_colors_import_as_these_values` |

The audit's repro scripts were the failing evidence for findings 1-5; the importer and GUI
tests were run red before their fixes, the schema tests for 2, 3 and 5 were written in the
same session as the fix, just after it.

## Deferred

- `Equation.colors` (a parameter of emphasis role names per term) stays roles only.
- Explicit values for glow, surface and colormap keys (not in the plan).
- Animating a style value (a color change is still instant; the importer still says so).
- Milestone 20's remaining importer warnings (fills, updaters, ValueTracker, ...).
- The field cache key in `render/passes.py` (~572) uses role names, not the resolved theme
  values (predates Milestone 18; noted by the audit). Two themes giving one role different
  values could share a cached field. Not fixed here: a render-layer change with no failing
  case yet.
- `importers/manim.py` `color_role` still reads hex itself (it needs a hue to pick the
  nearest role); `style_color` (the explicit value) uses the schema's parser.

## Untested

Windows, macOS and CUDA: not run. GPU goldens from the owner's PC: none yet for
`own_colors`. Qt on a real display: only Qt's offscreen platform.

## Shared files (for merging with Milestone 16)

`phasor/core/schema.py` (StyleValue/ColorValue validators, `ObjectSpec.style`,
`Overrides.styles`, `Meta.background`, version 9), `phasor/core/migrate.py` (`_v8_to_v9`),
`phasor/core/patch.py` (`SetStyle.role` type), `phasor/style/theme.py` (two required
`[lint]` keys), the four base theme tomls (`[lint]` lines only), `phasor/time/compile.py`
(the two style-resolution loops, `CompiledObject.world`, `world_sizes`, the background in
`evaluate`), `phasor/core/objects.py` (a comment and a type hint), `phasor/style/looks.py`
(a type hint). If Milestone 16 adds style groups (`fill`, `dash`), `style/explicit.py`
`EXPLICIT` is where a fill color would be allowed an explicit value.

## Gate (`phasor-milestone-gate`, 2026-11-23, Linux, llvmpipe, no CUDA)

| Step | Result | Evidence |
| --- | --- | --- |
| ruff, ruff format | pass | "All checks passed", 302 files formatted |
| lint-imports | pass | 2 kept, 0 broken |
| pytest, focused | pass | 23 suites (style_freedom, aesthetic, beauty, lint, final `-k "not slow"`, docs, layers, open_contract, manim_import, prism, critic, engine, time, mcp, core, gui_window, gui_controller, render, contracts, kit, pycode, remote, life): 796 passed, 3 skipped (no CUDA; two `value` cases with nothing to draw). Then style_freedom, workers, sketch, inspect, choices, prism, beauty, life, kit after the look fix: 181 passed, 2 skipped (no software golden yet for two beauty scenes, as before). prism_primes golden passes and its critique stays empty |
| pytest, full | **not run** | over an hour on this box |
| phasor doctor | tier B | llvmpipe OpenGL 4.5, g++ CPU kernels, no CUDA |
| exit demo | pass (Linux) | `examples/own_colors.py`: renders (golden looked at; exact pixels for hex, linear and background); `off_palette` x2 and `off_scale` x1 warn; critique offers the snaps; `polish()` and `autofix` change no typed value; `polish(snap=True)` snaps; a version 8 document migrates and reads the same |
| contract audit | done, **not independent** | by the authoring model (no subagent available). Every write goes through `Engine.apply`; `state_at` stays a function of (document, t) (world sizes go through the camera evaluated at t); hex decoded to linear once, linear to the end; `style/explicit.py` imports only core and style; no new object members; one way to set a style value (`set_style`). No contract text changed |
| docs and artifacts | pass | API.md section with a running block, MCP.md, llms.txt regenerated, example + catalog entry, golden, this file |

**MILESTONE 18: NOT DONE: the full suite was not run, and the contract audit was not
independent.** On Linux, the exit conditions are met.
