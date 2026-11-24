# Milestone 27: Explorer rows, variables, AI tools

Plan: `EXPLORER.md`, row 27; design: `docs/decisions/explorer_advice_2026-11-24.md` sections 7
and 8. Part A built 2026-11-24 in a worktree branched from the Explorer plan commit, in
parallel with Milestone 26 and the `parse_expression` hardening.

Status: **Part A built (the pure-text row layer); tools and Variable/Control wiring still to
do.** Not gated. Linux only; Windows untested (pure Python, no GL, no subprocess).

## Built (Part A)

| Area | Where | What |
| --- | --- | --- |
| Notation | `phasor/domains/rows.py` `normalize_row` | a tokenizer in front of `parse_expression`: `^` to `**`, superscripts, implicit products (`2x`, `2(x+1)`, `x y`, `(a)(b)`, `3sin(x)`), a name followed by `(` is a call only when the parser itself reads it as a function (asked of the parser, cached; so new parser functions need no change here), `sin x` is an error with a hint, unicode (`·` `×` `−` `÷`, Greek letters to names, full-width and subscripts folded, invisible format characters as spaces), derivatives `x'` `x′` `ẋ` `dx/dt` only as a left-hand side; names are whole letter runs (`rho`, `xy` one name, with a warning for `xy`); `lambda`/`λ` become `lamda` (SymPy's spelling; `lambda` is a Python word). Every part goes through `parse_expression`, so the whitelist stays the one safety argument |
| Result | same, `Row` | `kind`, normalised `text` (idempotent), `parts` in the parser's grammar, `latex`, plain-words `error`, `name` (variable, ODE coordinate, named field), `value`, free `variables`, `coordinates`, `guess` (kind without override), `warnings` |
| Classification | same, `classify_row`, `normalize_row(text, kind=)` | section 7.3's table, in order: `variable`, `ode`, `point`, `vector_field` (also `F(x, y) = (P, Q)`), `parametric_curve`, `parametric_surface`, `graph`, `sideways_graph`, `polar`, `scalar_field`, `surface`, `implicit_curve`, `error`; an override kind the shape does not fit is an error row that says why. Reserved names `x y z t u v r theta pi e i` |
| ODE systems | same, `group_ode_rows` | `OdeSystem`: dimension = number of primed coordinates, components in x, y, z order, variables used, `autonomous`, row indices, errors (duplicates, `x'` alone, `z'` without `y'`, a coordinate used but not primed) |
| Hostile input | same | always an error value, never an exception: caps on characters (2000), symbols (400), bracket depth (30), digits in a number (50); constant powers above `1e300` refused before SymPy builds them (`9^9^9`, `x^9^9^9`, `2^100000`); `RecursionError`/`MemoryError` caught. Slowest measured here: a 200-deep `x**x**...` tower, 0.07 s |
| Tests | `tests/test_rows.py` | 181 tests: a table of 106 typed lines to (kind, text, error), parts re-parse with the parser, overrides, Lorenz (three rows plus `σ ρ β` give one 3D system with three variables), ODE grouping errors, idempotency (text, kind, error, parts) on the table and on 600 seeded random rows, meaning kept for 15 inputs already in the parser's grammar, 48 hostile inputs (each under a generous 2 s bound; measured all under 0.1 s), a loose per-row time budget |

## Still to do (Part B and on)

- `source` and `kind` params on the row objects; the inert error-row object (section 7.1).
- `Variable` + auto `Control` at commit time only (section 7.4); `Explorer.rows()` as a view.
- Tools: `explorer_state`, `explorer_edit`, `explorer_preset`, `find_equilibria`,
  `flow_probe`, `sweep_parameter` (`workspace/tools.py`, `TOOL_NAMES`, `docs/ai/MCP.md`,
  `llms.txt`, the skill); `docs/ai/API.md` for the row API once it settles.
- Parse every formula in `examples/`, `showcase/`, `gallery/`, `docs/ai/` through the rows
  layer and the hardened parser (EXPLORER.md rule).

## Deviations

1. **Size caps in the rows layer.** The design puts the caps (length, node count, constant
   towers) in `parse_expression`; that hardening was in progress on another branch, so
   `rows.py` carries its own caps to meet the 0.1 s rule now. The whitelist (nothing is
   evaluated as Python) is still only `parse_expression`'s. The caps may become redundant
   when the hardening lands; keep whichever is stricter, and keep the `9^9^9` tests.
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
