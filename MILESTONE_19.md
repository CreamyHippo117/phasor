# Milestone 19: a diagram kit

Plan: `MANIM_PARITY.md`, "Milestone 19". Owner's answer 3 (2026-11-23): `pygments` (BSD)
only if `Code` needs it, as an optional extra. Exit demo: **a gallery with a NumberLine, a
NumberPlane with grid, Axes with unequal scales and labels, a BarChart, a Table, a
MatrixDisplay and a Code block renders correctly; every type has inspector output, lint,
docs, an example and a golden; params are bounded and fuzzed** (`examples/diagram_kit.py`,
`examples/data_viz.py`).

Status: **built on Linux (llvmpipe OpenGL, no CUDA); focused gates green (see Gate).** The
full suite was not run (over an hour here). Contract audit by the authoring model, **not
independent**. Untested on Windows, macOS and CUDA. Built in parallel with Milestone 17
(another worktree); see "Shared files".

## Built

| Area | Where | What |
| --- | --- | --- |
| Ticks | `phasor/core/ticks.py` (new) | pure, deterministic: nice steps (1, 2, 5 x 10^k; step 0 in a range asks for one, from the range alone, never the camera or text size), `check_range` (finite, within `MAX_VALUE` 1e9, `min < max`, span not lost in rounding (`RESOLUTION` 1e-9 relative), at most `MAX_TICKS` 100 ticks), `values` (15 significant digits, no `-0`), `nice_range` (data, flat data widened), number text (no `-0`; 1e15 and more in scientific notation; `nan`/`inf`; thousands commas; a `+`), part names by value (`x_m2p5`), `clear_step`. In `core` because lint (below the domains) uses it |
| Composites that draw | `core/record.py` (`HYBRID`), `core/objects.py` (`register`) | a type with both `build` and `draw` keeps its `draw` (this was already so) and now derives bounds and features from its drawing **and** its parts (before: its parts only). Every diagram type draws its furniture (lines, ticks, grid, rules, brackets) and builds what a reader reads as parts |
| Names | `core/ids.py` | `name_from`, `unique_names`, `children` (`svg.part_names` and `SVG._parts` now use them; same names as before, its tests pass) |
| `geo.Axes` (extended) | `domains/geometry.py` | `x_scale`, `y_scale`, `z_scale` (0: `scale`; unequal scales), step 0 = nice, `numbers=True` (tick numbers as `Text` parts `axes.x_2`, `axes.y_m1`, maths for a true minus, style key `numbers` = text `label`), `digits`. Ranges, scales, `pos` and `labels` bounded. With `numbers` off it builds no parts and draws exactly as before |
| `dg.NumberLine` | `domains/diagrams.py` | range, `pos` (middle), `scale`, `angle`, `tips`, `label`; numbers `line.n_m1`, `line.n_0`; under a level line, left of a steep one; features `start`, `end`, `zero` |
| `dg.NumberPlane` | same | grid (with `faint` lines between) drawn by the plane; an `Axes` part `plane.axes` with numbers and labels; unequal scales; a backdrop (labels may sit on it) |
| `dg.BarChart` | same | `Rectangle` parts named after the bars' names (`chart.bees`, or `bar_1`), `bees_name`, `bees_value`, y numbers; auto nice range (negative values hang below the baseline), `grow` (animate), `colors` (emphasis roles) |
| `dg.Table` | same | cells from `cells` or a CSV `path` (a `Files` param: only inside the scene's folder, regular files, 256 KB cap, through `files.stamp`/`read_bytes`, cached by the stamp, read again when it changes); cells named like a spreadsheet (`t.a1`, `t.b2`); header role; rules `none`/`header`/`grid`; `align`; `reveal` (rows) |
| `dg.MatrixDisplay` | same | any matrix (up to 10 x 10) of maths entries (`m.e1_2`), brackets `square`/`round`/`bar`/`none` |
| `dg.Code`, `dg.CodeLine` | same, `domains/diagrams_math.py` | code from `code` or a file `path` (same file rules); one `CodeLine` part per line (`code.line_3`), monospace by placement (each character centered in a 0.52-text-height cell; wide letters set smaller on the same baseline); highlighted by pygments (classes keyword, name, string, number, comment, plain; line numbers); without pygments: plain, lint `missing_extra`; unknown language: plain, lint `unknown_language`; `line_numbers`, `reveal` |
| Counter (extended) | `domains/basic.py` | `group` (1,234,567), `sign` (+); text short for huge and non-finite values (was a 300-character string for 1e300); `prefix`/`suffix` at most 40 characters. Output for ordinary values is unchanged. It animates by following a `Value` (example, test) |
| Lint | `inspect/lint.py`, theme `[lint] tick_gap = 0.25` | `crowded_ticks` (numbers along one axis, or a chart's bar names, closer than `tick_gap` text heights; one issue per axis with `path` `<id>.<axis>`, `times`, the smallest nice step that clears it in the hint, no `fix`); generic `overlap` and `contrast` no longer judge numbers along one axis pair by pair (tags `along_<axis>`), and `contrast` no longer treats a composite as being under its own parts; `truncated` (a table or code file cut to the limits); `unknown_language`; `missing_extra` also for a type that draws plainer without its extra; `bad_file`'s hint no longer talks about SVG for other files |
| Bounds | `diagrams_math.py` | tables 40 rows, 12 columns, 300 cells, 60 characters a cell; charts 40 bars; matrices 10 x 10; code 120 lines x 120 characters, 12000 in all, no control characters; language names `[a-z0-9+#._-]`, 32 characters. A file is cut to them (lint `truncated`) instead of refused |
| Hostile numbers | `core/values.py` | `normalize` turns an int too large for a float into a `ValueError` (it raised `OverflowError` from `Engine.apply` for any type) |
| Extra | `pyproject.toml`, `uv.lock`, CI, packaging, README, TESTING, CLAUDE.md, `core/extras.py` | `code = ["pygments>=2.17"]`, in `all`; installed by CI and packaging |
| Docs | `docs/ai/API.md` (new section "Diagrams", Axes bullet, lint table), `llms.txt` | two running blocks (every type; a crowded axis) |
| Examples | `examples/diagram_kit.py` (reference), `examples/data_viz.py` (technique), `kit/examples.toml` | prism look; both pass `lint(strict=True)` with nothing |

## Verification (Linux, 4 cores, llvmpipe OpenGL 4.5, no CUDA)

- `tests/test_diagrams.py` (42 test functions, 104 cases, written with or before the code;
  both lint MODES for every type and every new lint code): ticks (nice steps, degenerate, reversed, too many, NaN/inf/huge int,
  enormous and tiny ranges with distinct numbers, determinism); every type's parts, names,
  semantics, features and placement, lint clean; CSV and code files (read, changed on disk,
  too long, not UTF-8, too big, missing, outside the folder, a FIFO never read); pygments
  highlighting and the fallback without it; `crowded_ticks` for axes and bar names (one
  issue, no pairwise `overlap`); every type at its bounds builds, inspects and lints in
  bounded time; part names the document holds get a suffix (`t.a1_2`, `plane.axes_2`); an
  SVG without its extra still gets exactly one `missing_extra`; 22 hostile parameter sets refused as errors through `Engine.apply` (never
  raised); the examples show every new type and lint strict-clean; storyboard goldens.
- Goldens (looked at): `tests/golden/software/diagram_kit.png`,
  `tests/golden/software/data_viz.png`. `test_final` covers the 7 new types (2D and orbit:
  inspect, draw, lint; `REQUIRED` entries for `table`, `code`, `code_line`).
- Cost: the gallery (77 objects with parts) with OpenGL: `state_at` 7 ms, a new inspector
  snapshot 0.38 s, strict lint 3.6 s. At the bounds, measured **without OpenGL**
  (`passes=None`): each type under 1.1 s for state, inspect and lint together; with OpenGL a
  300-cell table costs more per inspector snapshot (one solo mask per part), not measured.

## Deviations

1. **Hybrid composites** (`record.HYBRID`): ADR 0005 says a composite's bounds are the union
   of its parts'. A composite that also draws now unions its own drawing too. No such type
   existed before (the code path allowed it but ignored the drawing). Not a contract text
   change; the owner may want ADR 0005 to say so.
2. **`geo.Axes` was extended, not replaced**: lines, ticks and end labels are still drawn by
   the Axes itself (pixels unchanged when `numbers` is off); only the numbers are parts.
   Its end `labels` stay drawn, not parts (making them parts would add new lint subjects to
   old scenes). `step` 0 in a range used to be a validation error; it now asks for a nice
   step (`tests/test_geometry.py` changed to say so). Ranges now must make at most 100 ticks
   and stay within 1e9: no example, gallery or showcase script, test or saved
   `*.phasor.json` in the repository uses `axes` past it (the two saved documents have no
   axes).
3. **New name `MatrixDisplay`** (type `matrix_display`): `matrix` is `linalg.Matrix`, a live
   2x2/3x3 of numbers with a determinant; kept as it is. `MatrixDisplay` is any matrix of
   maths. The owner may want `linalg.Matrix` rebuilt on it later (its pictures would change).
4. **Module name `dg`** (`from phasor import dg`), next to `geo`, `nw`, `dyn`.
5. **Code is monospace by placement, in phasor's text font (Aileron)**: `ctx.text` has no
   font choice and text internals are Milestone 17's; mathtext's `\mathtt` was tried and
   rejected (boxes tight to the ink, so baselines jitter; `\#`, `\&` fail).
6. **The label placer does not place parts** (Milestone 15, Deviation 7: parts take no
   constraints). Tick numbers, bar names, cells and code lines are placed by their
   composite's rule; lint judges them (`crowded_ticks`, `overlap` between different groups,
   `offscreen`, `tiny`, `contrast`). A label of the whole diagram is a document object with
   `label(x) near x`.
7. **No `fix` for `crowded_ticks`**: a range may be an expression or a track; the hint gives
   the step.
8. **Single x-height letters in maths are short**: `tiny` judges a text's ink box, so a
   `MatrixDisplay` of `a`, `c` (no ascenders) is 21 px tall at the `math` role and is
   flagged at 1080p; digits and taller letters pass. Not changed (lint rule shared by all
   text).
9. Number lines write their numbers under a level line and left of a steep one (as axes do).
10. A `Code` without pygments reports `missing_extra` from its semantics key `plain_without`
    (it draws, plainly), not from `Requires` (whose message is "draws nothing"); the message
    comes from `core/extras.EXTRAS`.
11. **A part name the document already holds or retired under a diagram gets the next free
    suffix** (`t.a1_2`, `plane.axes_2`, `code.line_1_2`), as `SVG` does; nothing is dropped.

## Deferred

- Log scales, dates and categorical axes; plotting functions on `Axes` (`calc.Graph` does
  curves on its own); legends; stacked and grouped bars; table cell colors and spans.
- A background panel behind code (a filled shape under labels is `overlap`; needs a
  backdrop part).
- The Manim importer still maps `NumberPlane` to plain axes (Milestone 20 maps it to
  `dg.NumberPlane`, and `BarChart`, `Table`, `Code`, `DecimalNumber`).
- Animating a counter's digits (rolling); `Counter` follows any value smoothly already.
- A small mark above some Aileron letters with descenders (`y`, `g`) at certain sizes: seen
  in the gallery's stills; in `core/text` rasterization (Milestone 17's area), not touched.

## Untested

Windows, macOS, CUDA (no GPU goldens). pygments in the PyInstaller build (its lexers load
dynamically; pyinstaller-hooks-contrib has a hook, not checked). FIFO test skipped where
`os.mkfifo` is missing (Windows).

## Shared files (for merging with Milestone 17)

Small, additive changes: `docs/ai/API.md` (one new section before "Space", the Axes bullet,
three lint rows, the `missing_extra` row), `inspect/lint.py` (`_same_axis`, the overlap
skip, the contrast skip, four rule bodies in the per-frame loop, their messages, the
docstring table), `style/theme.py` (`tick_gap` in the required `[lint]` keys) and the four
base theme TOMLs (one `[lint]` line), `tests/test_final.py` (three `REQUIRED` entries),
`tests/test_geometry.py` (one assertion), `pyproject.toml`, `uv.lock`, `kit/examples.toml`,
`llms.txt`, `phasor/__init__.py`, `phasor/domains/__init__.py`, `core/pycode.py`,
`core/extras.py`, `core/objects.py` (register's derived members), `core/record.py`
(`HYBRID`), `core/ids.py`, `core/values.py`, `domains/basic.py` (Counter only),
`domains/geometry.py` (Axes), `domains/geometry_math.py` (`tick_values`), `domains/svg.py`
(uses the shared name helpers), CI, packaging, README, TESTING, CLAUDE.md (Commands line).
Nothing in `core/text.py`, `time/verbs.py`, glyphs or morphing.

## Gate (`phasor-milestone-gate`, 2026-11-23, Linux, llvmpipe, no CUDA)

| Step | Result | Evidence |
| --- | --- | --- |
| ruff check, ruff format --check | pass | "All checks passed", 318 files formatted |
| lint-imports | pass | 2 kept, 0 broken |
| pytest, focused | pass | test_diagrams, test_shapes, test_style_freedom, test_open_contract, test_lint, test_docs, test_layers, test_beauty, test_engine, test_aesthetic, test_kit, test_geometry, test_motion, test_mcp, test_pycode, test_contracts, test_core: 561 passed, 2 skipped; test_final `-k "not slow"`: 408 passed, 2 skipped. After the last fixes (commit 529e131): test_diagrams, test_shapes, test_lint, test_docs, test_kit, test_mcp, test_layers, test_doctor, test_final `-k "not slow"`: 697 passed, 4 skipped |
| pytest, full | **not run** | over an hour on this box (left to CI) |
| phasor doctor | tier B | llvmpipe OpenGL 4.5, g++ CPU kernels, no CUDA, all extras installed |
| exit demo | pass (Linux) | `examples/diagram_kit.py`: every type renders (golden looked at), strict lint empty; `examples/data_viz.py` likewise |
| contract audit | done, **not independent** | by the authoring model (no subagent available). Writes only through `Engine.apply` (builds return parts; nothing writes the document); `state_at` depends on the document, t and files named by `Files` params (read only inside the folder, cached by stamp, the engine's caches dropped when they change); caches keyed by everything they read (file stamp; source, language and whether pygments is there); `core/ticks.py` is pure and in `core` (no GPU/GL); every new type has the five members after registration; one way to do each thing (Axes extended; Counter extended; `MatrixDisplay` beside `linalg.Matrix` is Deviation 3). No contract text changed |
| docs and artifacts | pass | API.md section with two running blocks, lint table, llms.txt; examples and catalog entries; goldens; this file |

**MILESTONE 19: NOT DONE: the full suite was not run, and the contract audit was not
independent.** On Linux, the exit conditions are met.
