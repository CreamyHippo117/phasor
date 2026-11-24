# Milestone 30: Explorer analysis pack (maths and objects)

Plan: `EXPLORER.md` row 30; design review `docs/decisions/explorer_advice_2026-11-24.md`
section 2.6 rows 5, 7-12 (and sections 1, 5 for the rules). Built 2026-11-24 in a worktree,
on top of Milestone 26 (flow core). Milestone 27's tools (`workspace/`) are another
agent's: nothing here touches `workspace/`, `gui/`, `render/`, `rows.py` or `EXPLORER.md`.

Status: **built; not gated, not independently audited.** Linux only: **untested on Windows
and on CUDA** (nothing here uses CUDA; every new type is tested with `passes=None` too,
since Windows CI has no OpenGL).

## Built

| Area | Where | What |
| --- | --- | --- |
| Numerics | `domains/flow_analysis_math.py` (pure NumPy) | `Sampler` (a plane `n . p = offset`, up/down/both, or the maxima of a coordinate = its velocity crossing 0 downward); `crossed` (linear interpolation inside a step); `SectionRun` (seeds advanced together, continued step by step, crossings with point, time, seed; first `MAX_CROSSINGS` kept); `lyapunov` (Benettin: twin at `1e-8 max(1, |x|)` along a hashed direction, renormalised every 0.5 to a size relative to the current point, aligned during the transient (not counted), 8 blocks, `spread` = standard error, `converged`, `chaotic_beyond` = ln(1e16)/lambda); `statistic` (a value rounded to its spread); `limit_cycle` (transient, plane across the flow, returns refined to rounding by secant steps of a partial RK4 step, `_loops`: repeats after m windings, period, point, path, box; `equilibrium`, `stopped`, `none`); `divergence_grid`, `polygon_area`; `sweep` (equilibria: counts, eigenvalues, candidates refined by bisection and labelled saddle-node / pitchfork / transcritical / Hopf / window edge, mutual-nearest matching, one candidate per label and value; attractor: every value advanced at once, samples after a transient, `periods` by clustering, candidates between values) |
| Flow core extension | `domains/flow_math.py` | a variable's value may be an array, one value per point (`_values`, `_take`; `step` sub-selects them when it sub-steps); the scalar path is unchanged |
| Objects | `domains/flow_analysis.py`, reached as `flow.X` (`flow.py` imports and re-exports them) | `Section` (`flow_section`), `TwinProbes` (`flow_twins`), `TimeSeries` (`flow_time_series`, a panel like `ImpulseResponse`), `DivergenceMap` (`flow_divergence`, heat map through `heatmap.heat_pixels`, `HeatParams`, backdrop; advected disk area vs A0 exp(integral of div)), `BifurcationDiagram` (`flow_bifurcation`, a panel), `LimitCycle` (`flow_limit_cycle`); `flow.lyapunov(system, x0, T, state=)` for tools |
| Budgets | `flow_analysis_math` | `MAX_SECTION_SEEDS` 4096, `MAX_SECTION_WORK` 2e7, `MAX_CROSSINGS` 2e5, `MAX_LYAPUNOV_STEPS` 5e4 (the transient at most half), `MAX_CYCLE_STEPS` 1e5, `MAX_SWEEP` 256 values, `MAX_SWEEP_WORK` 1e7, `MAX_CANDIDATES` 8, `BISECT_ITERS` 24; every object reports a `budget` dict, so the existing `flow_budget` lint (theme `[lint] flow_work`, reports times) covers them |
| Memos | `flow_analysis.py` | `_SEC`, `_LYA`, `_AREA`, `_SWP`, `_CYC` (32 entries each), keyed by every input (system key with the variable values it uses, `run_key()`, each param) |
| Docs | `docs/ai/API.md` | "The analysis pack" in the Flows section, with a running python block; what a sweep does not detect |
| Examples | `examples/rossler_section.py`, `examples/period_doubling.py`, `kit/examples.toml` | a Rossler section filling in, twin probes, their log distance against lambda (about 0.063 +/- 0.01 here, literature 0.07), time strips; the Rossler period-doubling cascade over c (a logistic-like picture, period 1, 2, 4, chaos, a period-3 window) and the two-loop cycle at c = 3.5 |
| Tests | `tests/test_flow_analysis.py`, `tests/test_final.py` | see below |
| Script conversion | `core/pycode.py` | `flow_analysis` classes convert to `flow.X` |

Known numbers checked: uniform-field crossing exact to 1e-12; centre crossings at 2 pi k
(and pi(2k+1) upward); Lorenz lambda 0.906 +/- 0.05 (T 200); a saddle's lambda = its
largest eigenvalue to 1e-3; Van der Pol |lambda| < 0.02; Rossler about 0.07 (slow test);
Hopf period 2 pi to 1e-7 and radius sqrt(0.5); Van der Pol period 6.6633 and amplitude
2.0086; Rossler c = 3.5 closes after 2 loops, c = 5.7 has none; saddle-node, pitchfork and
Hopf normal forms at mu = 0 (1e-5); Lorenz pitchfork at rho = 1 and Hopf at
sigma(sigma+beta+3)/(sigma-beta-1) = 24.7368 (1e-4); transcritical at 0; a window edge;
Liouville exact for constant divergence (the pendulum, -gamma).

Rules checked: batch independence (crossings of seed k alone, in 7, in 300; a sweep value
alone or in 23), continuing a section equals one run, every input in each memo key (31
single-param changes, each compared with all memos cleared, and each changes the output
except two that provably do not), scrub order equals a fresh engine, hostile formulas (four,
every type, both modes, still drawn with GL), bad params refused with a reason, budgets
clamped and linted in both modes, every type at its maximum (slow: time and tracemalloc peak
asserted generously), every type in `test_final` (2D and orbit, `passes=None` and GL).

## Deviations

1. **Numbers from long runs are statistics, not SAFE_TAU-cut values.** A Lyapunov exponent,
   section points after a transient and sweep samples are past `SAFE_TAU` by construction.
   Positions and counts from a path are still given only up to `SAFE_TAU` (Section, TwinProbes,
   TimeSeries, DivergenceMap's disk). The Lyapunov value, its spread and `chaotic_beyond` are
   reported as a `statistic` rounded to the decade of the spread (agreeing across machines
   within the spread, not bit for bit). A converged limit cycle's period and point, and a
   sweep's periods, come from an attracting orbit, which forgets rounding: they are rounded
   to the cycle's tolerance, and a chaotic row's period is None.
2. **No new lint rule.** `flow_budget` (with the budget dicts) covers the analysis objects;
   "not converged" is a semantic flag (`converged`), not a lint: a short Lyapunov estimate
   in a scene is not a mistake.
3. **TwinProbes does not draw its own log strip**: one object draws in one camera, so the
   twins draw in the world and a `TimeSeries(source=twins)` is the strip (on the overlay, as
   in the example). One panel type for both strips.
4. **Attractor-mode candidates are not refined** (each bisection step would be a whole
   transient); they are reported between swept values. Equilibria-mode candidates are.
5. **The limit cycle's returns are refined beyond linear interpolation** (secant on a partial
   RK4 step): linear interpolation is wrong by about dt^2, too much for the tolerance; the
   `Section` object keeps linear interpolation as asked.
6. **The period-doubling golden cuts the sweep to c <= 4.15** (periodic orbits only, which
   settle and draw the same everywhere) and uses long runs; the Rossler section golden uses
   flow time <= 5 only.
7. **Objects live in a new module** `flow_analysis.py`, re-exported by `flow` (a module
   `__getattr__`; importing `flow` registers them).
8. **A centre's closed orbits count as a cycle** (isolation is not checked); documented.

## Not done / open

- Gate (`phasor-milestone-gate`) and an independent audit.
- Windows and CUDA runs (`pc_tasks/`); timing guards use `perf_counter` with generous bounds.
- Milestone 27's tools calling `flow.lyapunov` and the sweep (`sweep_parameter`) are for the
  other agent; the functions take plain data.
- Not detected (by design, documented): global bifurcations, cycle bifurcations other than
  period changes, anything between swept values or outside the window. Basins, FTLE (PC,
  Milestone 31).

## Test runs here (Linux, llvmpipe, shared box)

Focused suites, one at a time: `test_flow_analysis` 80 (76 not slow + 4 slow: Rossler
lambda, every type at its maximum (313 s here, bound 900 s; peak under 2 GB) and the two
goldens), `test_flow` 67, `test_flow_parser` 36, `test_dynamics` 18, `test_docs` 5,
`test_layers` 9, `test_contracts` 9, `test_dynamics -m slow` 1 and `test_flow -m slow` 2 (the M26 goldens and maximum after the flow core change), `test_beauty` 31, `test_lint` 32, `test_kit` 10,
`test_final -k "not slow"` 492 (6 skipped), `test_mcp -k llms` 1 (after regenerating
`llms.txt`). All passed. ruff check, ruff format and lint-imports clean. The whole suite
was not run. Goldens `tests/golden/software/rossler_section.png` (flow time <= 5) and
`period_doubling.png` (c <= 4.15) were looked at.

Measured here (loaded box, not a benchmark): Lorenz lambda over T 200: about 4 s; a Rossler
sweep of 64 values x 25000 steps: about 7 s; `period_doubling` example inspect: about 8 s;
`rossler_section` example: about 17 s (the Lyapunov estimate over 300 dominates).
