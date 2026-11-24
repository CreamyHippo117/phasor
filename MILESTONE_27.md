# Milestone 27: Explorer rows, variables, AI tools

Plan: `EXPLORER.md`, row 27; design: `docs/decisions/explorer_advice_2026-11-24.md` sections 7
and 8. Part A built 2026-11-24 in a worktree branched from the Explorer plan commit, in
parallel with Milestone 26 and the `parse_expression` hardening.

Status: **Part A (the pure-text row layer) and Part B (rows on the document, variables and
sliders, the six AI tools) built 2026-11-24.** Not gated, not independently audited. Linux
only: **untested on Windows and on CUDA** (pure Python and NumPy; nothing new uses GL,
CUDA or a subprocess; the one GL test skips itself without OpenGL).

## Built (Part A)

| Area | Where | What |
| --- | --- | --- |
| Notation | `phasor/domains/rows.py` `normalize_row` | a tokenizer in front of `parse_expression`: `^` to `**`, superscripts, implicit products (`2x`, `2(x+1)`, `x y`, `(a)(b)`, `3sin(x)`), a name followed by `(` is a call only when the parser itself reads it as a function (asked of the parser, cached; so new parser functions need no change here), `sin x` is an error with a hint, unicode (`·` `×` `−` `÷`, Greek letters to names, full-width and subscripts folded, invisible format characters as spaces), derivatives `x'` `x′` `ẋ` `dx/dt` only as a left-hand side; names are whole letter runs (`rho`, `xy` one name, with a warning for `xy`); `lambda`/`λ` become `lamda` (SymPy's spelling; `lambda` is a Python word). Every part goes through `parse_expression`, so the whitelist stays the one safety argument |
| Result | same, `Row` | `kind`, normalised `text` (idempotent), `parts` in the parser's grammar, `latex`, plain-words `error`, `name` (variable, ODE coordinate, named field), `value`, free `variables`, `coordinates`, `guess` (kind without override), `warnings` |
| Classification | same, `classify_row`, `normalize_row(text, kind=)` | section 7.3's table, in order: `variable`, `ode`, `point`, `vector_field` (also `F(x, y) = (P, Q)`), `parametric_curve`, `parametric_surface`, `graph`, `sideways_graph`, `polar`, `scalar_field`, `surface`, `implicit_curve`, `error`; an override kind the shape does not fit is an error row that says why. Reserved names `x y z t u v r theta pi e i` |
| ODE systems | same, `group_ode_rows` | `OdeSystem`: dimension = number of primed coordinates, components in x, y, z order, variables used, `autonomous`, row indices, errors (duplicates, `x'` alone, `z'` without `y'`, a coordinate used but not primed) |
| Hostile input | same | always an error value, never an exception: caps on characters (2000), symbols (400), bracket depth (30), digits in a number (50); constant powers above `1e300` refused before SymPy builds them (`9^9^9`, `x^9^9^9`, `2^100000`); `RecursionError`/`MemoryError` caught. Slowest measured here: a 150-deep `x**x**...` tower, 0.07 s |
| Tests | `tests/test_rows.py` | 181 tests: a table of 106 typed lines to (kind, text, error; two errors worded by the parser are matched only as errors), parts re-parse with the parser, overrides, Lorenz (three rows plus `σ ρ β` give one 3D system with three variables), ODE grouping errors, idempotency (text, kind, error, parts) on the table and on 600 seeded random rows, meaning kept for 15 inputs already in the parser's grammar, 48 hostile inputs (each an error value or a clean row, idempotent, under a generous 2 s bound; measured all under 0.1 s), a loose per-row time budget |

## Built (Part B)

| Area | Where | What |
| --- | --- | --- |
| Row fields | `domains/flow.py` `RowFields` | `source` (as typed; a `System` of ODE rows keeps one line per row), `row` (place; -1: not a row), `kind` (an override, one of `rows.KINDS`), `visible`, on `Variable`, `System`, `Probe` and the new `Note`. A hidden `System` hides its dotted parts; a `Probe` that is a row keeps its own `visible` |
| Inert rows | `flow.Note` (`flow_note`) | text, kind, plain-words `error`; draws nothing (drawless in lint and `test_final`); API.md entry; in `examples/explorer_rows.py` |
| Rows to the document | `workspace/explorer.py` `plan`, `rows_of` | the list is a view (`rows_of`, rebuilt from a saved file alone); `plan(doc, edits)` gives the ops of **one** commit: `name = number` -> `Variable` + `Control` (`<id>.value`, min/max/step); ODE rows -> one `System` with `<id>.field` and `<id>.eq`; `(P, Q)` -> `System` + `<id>.field`; `(a, b)` -> `Probe` on the system of its dimension; others -> `Note`. Free names get a `Variable` (1, -10..10, 0.01) at commit only; a System's `constants` name every variable it uses. A kind change removes and adds in one patch (children and controls removed explicitly); an unchanged kind keeps the id. Ids by meaning (`rho`, `system`, `point`, a preset's name), first free suffix against every id and retired id and their parents. Every planned object is validated first: a row the validator refuses (a name the flow core reserves such as `j`, `I`; `t` in an ODE) becomes a `Note` with the message, never a refused patch. Typing `rho = 24` when `rho` exists sets it; loading a preset again keeps its variables. Rows are one line (line breaks become spaces); a row over 4000 characters is refused. A new 3D system turns an ortho camera into orbit |
| Session vs document | `Workspace._explorer_typing/_commit/_leave/_preview` | a row being typed is locked: `explorer_edit` touching it (or its system) is refused whole with `row_locked` and the row's `version` (a content hash of its object); `"version"` on an edit refuses with `row_changed` if the row moved on. A preview is `Engine.fork` + the same ops; the commit is one undo step. These helpers start with `_`, so `remote.tool_names()` never exposes them to an attached AI |
| Tools | `workspace/tools.py`, `mcp/server.py` `TOOL_NAMES` | `explorer_state(t)`, `explorer_edit(rows)`, `explorer_preset(name, replace)`, `find_equilibria(window, grid, variables, system, t)`, `flow_probe(x0, T, samples<=64, variables, system, t)`, `sweep_parameter(name, lo, hi, n<=64, measure, window, grid, variables, system, t)`. Analysis runs on a fork (`variables` set the Variables there) and leaves version, undo depth and the document unchanged (tested). Every answer carries `system`, `window`, `dt`, `variables`, `tolerance`, `converged` (and `grid`/`starts`/`T` where they apply) and is plain JSON |
| Numerics | `domains/flow_math.py` | `classified_equilibria`; `lyapunov` (Benettin twin, renormalised, transient dropped, `spread` of two halves, `converged`); `probe` (samples only up to `chaotic_beyond` = ln(1e9)/lambda for lambda > `CHAOS_TOL` 0.05, statistics after; escaped and when; bounding box; final speed; period from returns to the mid-run point (three or more at steady intervals); dt against dt/2 over the run, or over five Lyapunov times on a chaotic path, plus `agree_until`; steps clamped to `MAX_PROBE_STEPS/2`); `sweep` (equilibria at n values; count changes bisected and named `saddle-node` or `pitchfork`, a tracked complex pair crossing bisected by Newton continuation and named `Hopf` with its frequency, a real crossing with the count unchanged named as such; candidates from one event merged; global bifurcations said not to be detected) |
| Caps | `domains/rows.py` | each cap is the stricter of the rows layer's and the hardened parser's (`min(..., codegen.MAX_*)`); the rows layer's are the stricter today; `9^9^9` tests kept |
| Docs | `docs/ai/MCP.md`, `docs/ai/API.md`, `llms.txt`, `skills/phasor/SKILL.md` (both copies) | the six tools in the table and a section with a runnable `json calls` block; an "Explorer rows" API section with a running python block; "Helping in the Explorer" in the skill |
| Example | `examples/explorer_rows.py`, `kit/examples.toml` | rows planned into a scene: an ODE system, a slider, a point, a kept graph, an error row |
| Tests | `tests/test_explorer_rows.py` (60), `tests/test_explorer_tools.py` (13) | rows to objects; rebuilt from the saved file; kind change one patch and undo restores the document exactly; delete-and-retype five times (variable, graph, point, field; systems with children) never `id_retired`; ids by meaning, no collision with `lorenz.field`, no label as id; setting an existing variable; every one of the 22 presets loads and means the preset; 24 hostile rows are error rows (and lint has no errors); too long refused cleanly; reserved names and `t` become notes; bad edits refused unchanged; visible/colour/order/window/dt; a hidden system draws nothing (GL); the lock, version refusal, preview fork; playing 600 steps, scrubbing and a slider mid-drag leave `Editor.version` and undo depth unchanged and `apply_controls` is one step; script scenes refused; a commit that changes nothing writes nothing (no undo entry); 13 bad settings (dt out of range, a wrong window, min above max, None, a non-bool remove, ...) refused with the document unchanged, never a row turned into a note. Lorenz (10, 27.7, 2.67) equilibria and eigenvalues as in Milestone 26; values on a fork; the Lorenz Hopf at sigma(sigma+beta+3)/(sigma-beta-1) = 24.736842... within 1e-8, C+ and C- one candidate; pitchfork, saddle-node and Hopf normal forms named right; Van der Pol period 6.66 and dt check; Lorenz Lyapunov 0.5-1.4 and no positions past `chaotic_beyond`; escape, backward, clamp and bad input; `explorer_state`; plain JSON |

## Still to do

- Parse every formula in `examples/`, `showcase/`, `gallery/`, `docs/ai/` through the rows
  layer (EXPLORER.md rule); the parser side is `tests/formula_corpus.py` (Milestone 26).
- Draw the row kinds that are only kept now (Deviation 5).
- `behaviour` in `explorer_state` (a Lyapunov estimate on demand, cached by the system's
  key) and a Lyapunov/extent `measure` for `sweep_parameter` (Milestone 30).
- The panel (`gui/explorer.py`, Milestone 29) uses `_explorer_typing`, `_explorer_preview`
  and `_explorer_commit`; "AI:" names in the undo history are the panel's.

## Deviations

1. **Size caps in the rows layer.** The design puts the caps in `parse_expression`; the
   hardening landed in Milestone 26 (2000 characters, 2000 nodes, depth 100, powers past
   e^700). The rows layer keeps its own caps as the stricter of each pair (2000 characters,
   400 tokens, depth 30, 50 digits, 1e300), written as `min(own, codegen.MAX_*)` so a looser
   parser never loosens rows. The whitelist is still only `parse_expression`'s.
2. **`lambda` is spelt `lamda`.** `lambda` cannot be a name in the parser's grammar (a
   Python word); `lamda` is SymPy's own spelling and prints as `\lambda`.
3. **A Greek letter is a name of its own** (`kα` is `k*alpha`), while ASCII letter runs are
   one name (`kalpha`). Typed Unicode has a clear boundary; ASCII does not.
4. **Row kinds named here** (`sideways_graph`, `polar`, `scalar_field`, ...) are the
   classifier's words; which object each becomes is Part B's choice (section 7.3's table).

## Open questions

- Should `xy` (a name made only of reserved letters) stay one name with a warning, as
  decided, or be refused? Desmos users will type it meaning `x*y`.
- `name = expression of other variables` (`b = 2a`) and user functions (`f(x) = ...`) are
  errors "not supported yet"; section 7.6 puts them in the second pass.
- Bare constants (`3`, `a`) are read as `graph` (a horizontal line), as Desmos draws them.
5. **Rows of kinds the Explorer does not draw yet are kept as `flow.Note`** (graph, sideways
   graph, polar, parametric curve and surface, surface, scalar field, implicit curve): read,
   classified and saved with their text, drawn nothing. `calc.Graph` and the `geo` curves
   take no free names, so a slider could not drive them; drawing them needs those types to
   read Variables (or new row types), which is more than this milestone's scope.
6. **One ODE system per document.** All `x'`/`y'`/`z'` rows form one `System`; two rows for
   one coordinate make every ODE row an error note saying so. A vector-field row is a
   system of its own. A row's visibility and colour on an ODE line apply to its system.
7. **Colour is a style role on the row object's first emphasis key** (a System's `frame`, a
   Probe's `color`); a Variable or Note has none and the edit is refused.
8. **Order is a `row` number per object** (no reorder op); a system's lines follow one
   another from its number. Numbers are dense (0, 1, 2, ...) after every commit.
9. **A Variable's `source` is what was typed; its value wins.** A slider commit
   (`Editor.apply_controls`) sets only `value`, so the list shows `name = value` and keeps
   `source` as typed.
10. **The analysis tools take the scene time `t`** (default 0): the playhead is the
   editor's session state, not the document's. `explorer_state` has no `behaviour`
   block yet (see Still to do); `flow_probe` gives the Lyapunov estimate.
11. **`find_equilibria` with `variables` uses a fork only for names a Variable holds**; a
   name only in the system's `constants` is overridden in the computation directly (the
   same answer, nothing changes either way).
12. **Deleting a variable a system still uses brings it back** (value 1) at the same commit:
   every name a row uses has a slider. The old id stays retired (`a_2`).

## Test runs here (Linux, llvmpipe, shared box; Part B, 2026-11-24)

Focused suites, one at a time: test_explorer_rows 60 and test_explorer_tools 13 (73 passed,
the GL test ran); test_rows + test_flow_parser 217; test_flow 69; test_dynamics 19; test_lint
32; test_docs 5 (`test_mcp_doc_examples_run_in_order` now takes about 150 s here, the
Explorer block included; Windows CI timing untested); test_mcp + test_remote +
test_gui_controller + test_layers + test_contracts 84; test_kit (catalogue, docs, skill
copies) passed; test_final not slow 467 passed, 8 skipped (then the new example and
`flow_note` again after the last fixes: 5 passed). ruff check, ruff format and lint-imports
clean. The whole suite was not run.
