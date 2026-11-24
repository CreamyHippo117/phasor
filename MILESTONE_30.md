# Milestone 30: Explorer analysis pack (maths and objects)

Plan: `EXPLORER.md` row 30; design review `docs/decisions/explorer_advice_2026-11-24.md`
section 2.6 rows 5, 7-12 (and sections 1, 5 for the rules). Built 2026-11-24 in a worktree,
on top of Milestone 26 (flow core); Milestone 27 part B merged in and reconciled (one
implementation of each analysis, below). Nothing here touches `workspace/`, `gui/`,
`render/`, `rows.py` or `EXPLORER.md`.

Status: **built; audited independently, 7 blockers + 10 should-fix (and 3 nits) fixed, each
with a regression test that failed on the code before the fix (see "Audit"); not yet
gated.** Linux only: **untested on Windows and on CUDA** (nothing here uses CUDA; every new
type and the new lint are tested with `passes=None` too, since Windows CI has no OpenGL).

## Built

| Area | Where | What |
| --- | --- | --- |
| Numerics | `domains/flow_analysis_math.py` (pure NumPy) | `Sampler` (a plane `n . p = offset`, up/down/both, or the maxima of a coordinate = its velocity crossing 0 downward); `refine_crossings` (a crossing bracketed by a step, refined to rounding by regula falsi (Illinois) on a partial RK4 step, per point, so batch independent); `SectionRun` (seeds advanced together under a lock, continued step by step; `cut_at` for the crossing cap); `benettin` (batched Benettin, twin size relative to the point, aligned during the transient) and `judge` (spread over blocks, a trend test across them, `chaotic` only when converged, above `CHAOS_FLOOR` 0.02 and three spreads); `lyapunov` (one seed, every factor clamped, nonsense refused); `returns_repeat` (the one period test: windings m after which returns repeat, persisting, not drifting, not a simpler orbit still closing in); `limit_cycle` (plus an isolation check: a neighbour must come back); `path_returns` (returns of a stored path, for the probe tool); `divergence_grid`, `polygon_area`; `sweep` (equilibria: `explore_math.sweep`; attractor: every value advanced at once, period and `state` per value via `returns_repeat`) |
| Flow core extension | `domains/flow_math.py` | a variable's value may be an array, one value per point (`_values`, `_take`; `step` sub-selects them when it sub-steps); the scalar path is unchanged (M26 goldens pass) |
| Shared with Milestone 27 | `domains/explore_math.py` | `probe` uses `flow_analysis_math.benettin`/`judge` (its own `lyapunov` deleted) and `path_returns`/`returns_repeat` for its period (its own `_period` rewritten onto them); `sweep` is the one equilibria sweep (its transcritical test gained a tolerance) |
| Objects | `domains/flow_analysis.py`, reached as `flow.X` | `Section` (`flow_section`), `TwinProbes` (`flow_twins`), `TimeSeries` (`flow_time_series`, a panel), `DivergenceMap` (`flow_divergence`), `BifurcationDiagram` (`flow_bifurcation`, a panel), `LimitCycle` (`flow_limit_cycle`); `flow.lyapunov(system, x0, T, state=)` for tools; `clear_memos()` |
| Budgets | `flow_analysis_math`, `flow_analysis` | `MAX_SECTION_SEEDS` 4096, `MAX_SECTION_WORK` 2e7, `MAX_CROSSINGS` 2e5, `MAX_LYAPUNOV_STEPS` 25000 (transient at most half; renorm and blocks clamped), `MAX_CYCLE_STEPS` 20000, `MAX_SWEEP` 256 attractor values (equilibria: `explore_math.MAX_SWEEP` 64), `MAX_SWEEP_WORK` 4e6, sweep Newton grid at most 27 (2D) / 9 (3D) per side, disk `DISK_CAP` 4096 / `DISK_STEPS` 2000; a step of one or two points in a Python loop counts as `LOOP_POINTS` (1000) points for `flow_budget`. At its maximum each object inspects in at most about 20 s here (`pc_tasks/13_analysis_maxima.md`, `benchmarks/analysis_maxima.py`) |
| Memos | `flow_analysis.py` | `_SEC`, `_LYA`, `_AREA`, `_SWP`, `_CYC` (32 entries each) keyed by every input, read and written under a lock (worker threads) |
| Lint | `inspect/lint.py`, `docs/ai/API.md` | `flow_ref` (warning): a `Section` `of=` or `TimeSeries` `source=` that names nothing, another kind, or another System's object |
| Docs | `docs/ai/API.md` | "The analysis pack": what is not detected, the false positives guarded against and those still possible |
| Examples | `examples/rossler_section.py`, `examples/period_doubling.py` | laid out for 16:9 and 9:16 (looked at both) |
| Tests | `tests/test_flow_analysis.py`, `tests/test_flow_analysis_audit.py`, `tests/test_final.py` | see below |

## Reconciliation with Milestone 27 part B (one way to do each thing)

| What | Kept | Why | Deleted / rewired |
| --- | --- | --- | --- |
| Largest Lyapunov exponent | `flow_analysis_math.benettin` + `judge` | batched over seeds like 27B's, but with a twin size relative to the point (27B's absolute 1e-8 loses the twin below rounding on a growing path: a saddle read -2 instead of 1), a hashed direction per seed, alignment during the transient, a trend test (27B's half-difference test let a shear's ln T / T bias through when the halves happened to agree) | `explore_math.lyapunov` deleted; `probe` calls `benettin`/`judge` (renorm 10 steps, first fifth a transient, as before) |
| Equilibria sweep | `explore_math.sweep` | audited already, continuation-based matching, window edges at either side, degenerate roots counted once and joined, `converged`, `precision`; mine mislabelled symmetric window edges (blocker 4) | `flow_analysis_math` equilibria sweep (`_row`, `_partner`, `_track`, `_bisect`, `_merged`) deleted; `BifurcationDiagram` and `fa.sweep(mode="equilibria")` call it; its transcritical test gained a tolerance (a double root's eigenvalue 0 gave 8 spurious transcriticals) |
| Period detection | `flow_analysis_math.returns_repeat` (returns to a section, refined) | precise (returns refined to rounding), names the windings (period doubling), refuses drift and not-yet-settled orbits; 27B's `_period` matched returns within 1% of the path's size and could not tell period 2 | `explore_math._period` now calls `path_returns` + `returns_repeat` (loose tolerance 1e-3, two persisting returns, as a probe is short); `limit_cycle` and the attractor sweep call it too; the attractor sweep's clustering (`periods`) deleted |

The 27B tool tests (`test_explorer_tools`, `test_explorer_audit`, `test_explorer_rows`) pass
unchanged.

## Audit (independent, 2026-11-24)

| # | Finding | Before | After (fix) | Test (`tests/test_flow_analysis_audit.py`) |
| --- | --- | --- | --- | --- |
| B1 | a finite range with an infinite span crashed inspect; NaN/inf/lo>hi ranges raised or gave wrong values | `range=(-1e308, 1e308)` ValueError in `np.linspace`; `(1, -1)` 0.026 instead of 0 | the span is validated; every sweep refuses a range that is not finite with lo < hi with a reason | `test_b1_*` |
| B2 | Lyapunov called shear and neutral flows chaotic | undamped pendulum 0.0266 +/- 0.008 converged, chaotic_beyond 1385; shear 0.029; `1, sqrt(2), x` 0.0146 converged | `judge`: trend test across blocks, value above 3 spreads and above 0.02 for chaos; Lorenz 0.906, Rossler 0.07, saddle 1, node -1, focus -0.2, oscillator 0 still right, short T not converged | `test_b2_*` (3 + 1) |
| B3 | attractor periods wrong and locked into a golden | Van der Pol period-1 read [2,2,1,1,3,...]; golden config 19/24 None, five "onset of chaos", a false "halving" | maxima refined to rounding; `returns_repeat` per value with a tolerance tied to the window; honest `state` ("not settled" near a doubling); candidates only between settled values; golden regenerated and looked at | `test_b3_*` |
| B4 | roots entering at both window edges read as saddle-nodes | `sin(k*x)` sweep: saddle-node | the one sweep (27B's) labels them `enters or leaves the window` | `test_b4_*` |
| B5 | `Section.full` depended on scrub history; `SectionRun` not thread-safe | full=True at t=4 after t=9, False fresh | `cut_at` step, `full` = cut at or before the asked step; a lock in `SectionRun` and around the memos | `test_b5_*` (the thread test passes before too: a race is not reproducible on demand) |
| B6 | `MAX_LYAPUNOV_STEPS` not a bound; nonsense not refused | dt 1e-5 twins: 425000 steps; `dt=1e-9`: did not finish in 300 s; T=nan etc. bare exceptions | every factor clamped (blocks, renorm, transient, per-block); twins budget counts the real steps weighted by `LOOP_POINTS`; `FunctionError` with a reason for T, dt, transient, x0, max_time | `test_b6_*` |
| B7 | duplicate lyapunov, sweep, period with 27B | three pairs | one each (above) | `test_b7_*` |
| S1 | LimitCycle mistook chaos for a cycle at loose tol; false doubling near a doubling; weak focus, drift and centre read as cycles | Lorenz tol 1e-2 "cycle"; c=2.7 loops 2 or 1 by transient; centre "cycle" | tol at most 1e-5; `returns_repeat` (persistence, drift, simpler orbit still closing); a neighbour must come back (isolation) | `test_s_limit_cycle_*`, `test_s_no_false_doubling_*`, `test_s_weak_focus_*` |
| S2 | period printed past its accuracy | VdP mu=10 19.078443 vs 19.07837 | rounded to max(tol, dt^2); `period_accuracy` says so | `test_s_the_period_*` |
| S3 | equilibria-sweep labels | `(x-mu)**2`: 8 transcritical; odd counts "count change" | the shared sweep plus a tolerance on the sign test | `test_s_equilibria_labels` (3) |
| S4 | `non_isolated` missing from rows | a line read as 9 equilibria | a column | `test_s_non_isolated_rows_*` |
| S5 | statistics printed finer than their spread | reference slope to 1e-3; converged/chaotic_beyond flipping with 1 ulp of x0 | value and trend rounded one digit coarser than the spread, spread and chaotic_beyond to one significant digit; `converged` can still flip near its threshold (documented) | `test_s_reference_slope_*` |
| S6 | budget model did not match cost | LimitCycle 96 s, twins 92 s, equilibria sweep 74 s, disk 15 s at maxima, no lint | real-cost work model (`LOOP_POINTS`), caps lowered, sweep grid clamped, disk capped: at most about 20 s each | `test_s_scaled_down_maxima_*`; full size `pc_tasks/13_analysis_maxima.md` |
| S7 | 313 s at-maximum test in CI | | replaced by a 100x scaled-down one; the full one is a PC task | `test_s_scaled_down_maxima_*` |
| S8 | Section crossings linear (0.04 off on Lorenz) with 6 digits | | refined to rounding (first crossing within 1e-5 of a fine run) | `test_s_section_crossings_*` |
| S9 | dangling and cross-system references silently fell back | | `bad_ref` in semantics, lint `flow_ref` (both modes) | `test_s_dangling_*` |
| S10 | Liouville ratio used the circle's area | 0.9984 for a constant divergence | the outline's polygon at flow time 0 (`polygon0`) | `test_s_liouville_*` |
| S11 | test_final's save/load compared warm memos | | `clear_memos()` before the loaded engine inspects | `test_s_the_save_load_check_runs_cold` |
| N1 | "the twins met" for an overflow | | "not finite" | `test_s_twins_overflow_*` |
| N2 | Rossler example off the frame at 9:16, tiny labels | | strips 0.48 wide stacked lower left, camera pulled back, label text role; looked at 16:9 and 9:16 | looked at |
| N3 | the section golden showed no crossings | | the golden sets the section's transient to 0 (flow time <= 5): crossings show | `test_rossler_section_golden_storyboard` |

"Fails before" was checked by running the new tests against the commit before the fixes
(an export of it, run with the same environment; the GL variant of the tiny-dt test was not
run there): every test failed except two, one of them (`dt=1e-9`) by not finishing in 300 s.
The two that passed before are `test_b2_what_was_right_stays_right` (a guard that nothing
right broke) and the thread test (a race does not fail on demand).

## Deviations

1. **Numbers from long runs are statistics, not SAFE_TAU-cut values.** Positions and counts
   from a path are given only up to `SAFE_TAU`; a Lyapunov value is a statistic rounded no
   finer than its spread; a converged cycle's period is rounded to its real accuracy; a
   chaotic row's period, sample count and a cycle search's return count are None.
2. **Lint `flow_ref` is the one new rule**; "not converged" stays a semantic flag.
3. **TwinProbes does not draw its own log strip**: a `TimeSeries(source=twins)` is the strip
   (one object, one camera).
4. **Attractor-mode candidates are not refined** (each step would be a whole transient).
5. **All crossings are refined**, not linearly interpolated (the task said linear; the audit
   showed 0.04 errors printed to 6 digits).
6. **The period-doubling golden cuts the sweep to c <= 4.15** and uses long runs of
   attracting orbits; the Rossler section golden uses flow time <= 5 and the section's
   transient set to 0.
7. **Objects live in `flow_analysis.py`**, re-exported by `flow`.
8. **A centre, a weak focus or a drift is not a cycle** (returns that drift, or a neighbour
   that does not come back); a very weakly attracting cycle can be refused the same way.
9. **The equilibria sweep allows 64 values** (`explore_math.MAX_SWEEP`), not 256.

## Not done / open

- Gate (`phasor-milestone-gate`); Windows and CUDA runs (`pc_tasks/13_analysis_maxima.md`).
- `converged` of a Lyapunov estimate can flip for a value near its threshold under a 1-ulp
  change of x0 (a statistic); documented, not hidden.
- Not detected (by design, documented): global bifurcations, cycle bifurcations other than
  period changes, anything between swept values or outside the window. Basins, FTLE (PC,
  Milestone 31).

## Test runs here (Linux, llvmpipe, shared box)

After the merge of Milestone 27 part B and the audit fixes, focused suites one at a time
(slow tests included where a file has them): `test_flow_analysis` 78, `test_flow_analysis_audit`
42, `test_explorer_rows` 60, `test_explorer_tools` 13, `test_explorer_audit` 14, `test_rows`
181, `test_flow` 69, `test_dynamics` 19, `test_flow_parser` 36, `test_docs` 5,
`test_layers` 9, `test_contracts` 9, `test_beauty` 33 (2 skipped), `test_lint` 32,
`test_kit` 11, `test_mcp` 39 (after regenerating `llms.txt`), `test_final -k "not slow"` 500
(8 skipped). All passed; ruff check, ruff format and lint-imports clean. The whole suite was
not run. Goldens `rossler_section.png` and `period_doubling.png` regenerated and looked at;
the examples looked at in 16:9 and 9:16. Maxima (each object alone, no tracemalloc): see
`pc_tasks/13_analysis_maxima.md` (at most 19.8 s, the attractor sweep).
