# Milestone 26: Explorer flow core (headless)

Plan: `EXPLORER.md` row 26; design review `docs/decisions/explorer_advice_2026-11-24.md`
(sections 1, 2 and 5). Built 2026-11-24 in a worktree. Exit demo:
**`examples/lorenz_explorer.py` renders a cone field, the three equilibria, a ball of 2000
points folding (tau 0 -> 5) and the C+ card; Lorenz (10, 27.7, 2.67) eigenvalues match:
C+ -13.840, 0.0852 +/- 10.149i; origin -22.741, -2.67, 11.741.**

Status: **built; audited independently, 4 blockers + 5 should-fix (and 3 nits) fixed, each
with a regression test (see "Audit"); not yet gated** (the `phasor-milestone-gate` skill is
still to run). Linux only: **untested on Windows and on CUDA** (nothing here uses CUDA; the
Windows suite has no OpenGL, so every new type is tested with `passes=None` too).

## Built

| Area | Where | What |
| --- | --- | --- |
| Parser hardening (advice 5.1-5.6) | `render/gpu/codegen.py` | one entry `_read` for `parse_expression` and `ComplexFunction.parse`: source at most `MAX_SOURCE` 2000 characters, `MAX_NODES` 2000 parts, `MAX_DEPTH` 100 (a run of `+`/`-` or `*`/`/` counts once and is folded left to right without recursion, so a 999-term sum reads and keeps SymPy's order); a power of two numbers is refused when `|exponent| * |log |base||` passes 700 (`9**9**9`, `2**-1000000`, `(1/2)**1000000`: under 1 ms); `RecursionError`, `MemoryError`, `OverflowError`, `ValueError`, `TypeError` inside become `FunctionError`; real-only `sign`, `floor`, `atan2`, `min`, `max` (2-4 arguments) for `parse_expression` only (`ComplexFunction` still refuses `sign(z)`) |
| Bounded simplify | `domains/geometry_math.py` | `tidy`: `sp.simplify` only under 80 operations (divergence and curl of `VectorField3D`); the flow core never simplifies |
| Formula corpus | `tests/formula_corpus.py`, `tests/golden/contract/formulas.json` | every formula-shaped string in `examples/`, `showcase/`, `gallery/` and `docs/ai/` (Python constants, `.phasor.json`, the doc's python and json blocks), read by the parser **before** the change (637 strings, 630 formulas); the test asserts each still reads to the same `srepr` |
| Flow core | `domains/flow_math.py` | `System` (2 or 3 components; variables are *arguments*, compiled once per expression, `compiled` LRU); `Compiled.velocity`, batched `jacobians` (exact; central differences where the derivative is not finite or does not compile: `sign`, `floor`, `abs` at 0); `jacobian_exact` (SymPy at the point); sources that add (`term_exprs`/`parse_term`: `uniform`, `source`, `sink`, `vortex`, `linear`, written as text `vortex(pos=(0, 0), strength=2)`, softened by a core radius); `step` (fixed RK4; `max_step` sub-steps a fast seed by its own count up to 16, then stops it `TOO_FAST`; non-finite stops only that seed); `integrate` (stored paths, `box` and `escape` stops, continuation from an earlier `Path`); `FlowSim` (a set as a `Simulation`: filled points, curves refined every `refine_every` steps by integrating new parameters from flow time 0, cap with `under` step, trail in float32, direction); `fill` (Halton, no RNG), `curves` (circle, ball rings, box edges, segment); batched `newton` (`pinv` for lines of equilibria, `drop` as before) and `equilibria` (grid, cluster, sort, cap 64, `non_isolated`, window and grid returned); `classify` (2D: the old `KINDS` and code unchanged; 3D: `KINDS3` with `saddle-focus`, tolerance `1e-7 * max|J|`, eigenvalues sorted by real part, orthonormal `Es`/`Eu`/`Ec` with fixed signs, `dims`); `manifolds` (1D stable/unstable branches, ends `escaped`/`steps`/`equilibrium`/...); `PRESETS` (22: Lorenz, Rossler, Thomas, Chua, Halvorsen, Aizawa; Van der Pol, pendulum, Duffing, Lotka-Volterra, Brusselator, FitzHugh-Nagumo, SIR, double well, Hopf, saddle-node, pitchfork, linear saddle/node/focus/centre/degenerate) |
| 2D front | `domains/dynamics_math.py` | `System` keeps its fields and gains `.flow`; `_fn`, `jacobian`, `jacobian_at`, `trajectories`, `fixed_points`, `classify`, `KINDS`, `parse_constants` are the flow core's. Bit-identical to the old code on the `test_dynamics` cases and the example's seeds (trajectories, fields, Jacobians, fixed points: max difference 0.0); `test_dynamics.py` and the `phase_portrait` golden unchanged |
| One table | `domains/cloud_math.py` | `ATTRACTORS["lorenz"]` reads `PRESETS["lorenz"]` (same numbers) |
| Objects | `domains/flow.py` (`from phasor import flow`) | `Variable` (drawless), `System` (`flow_system`: equations, terms, constants, window, tau, rate, dt, escape, pos, scale, frame, label), `FieldArrows` (`flow_arrows`: cones in 3D, flat arrows in 2D, one draw call each), `Probe` (`flow_probe`; was `Tracer`, renamed: the motion `Tracer` owns that name), `FlowSet` (`flow_set`), `Equilibria` (`flow_equilibria`), `EquilibriumView` (`flow_equilibrium_view`: by `near`, never by index or label), `Manifold` (`flow_manifold`); `flow.preset(name, **overrides)` |
| Batched strokes | `domains/strokes.py` | `arrow_batch` (camera-facing arrows) and `cone_batch` (camera-free cones), one `ctx.triangles` each |
| Work bounded | `flow_math.MAX_WORK`, `flow.FlowSet` | a set runs at most `MAX_WORK` (1e8) point-steps: its steps are clamped before it starts to `MAX_WORK / (points + outline cap)` (an outline alone: 762 steps), refinement inserts no more than the work left allows, and `FlowSim` stops (`halted`) past it as a backstop; semantics `budget.clamped` names `work`, lint `flow_budget`. Module memos (`_PATHS`, `_MAN`, `_EQ`) are keyed by every argument passed to the integration (`Resolved.run_key`: dt, escape centre and radius, max_step; and eps) |
| Lint | `inspect/lint.py`, `style/theme.py`, themes | `flow_budget` (points x steps past `[lint] flow_work` 5e7 (a theme without the key gets the default, `_TABLE_DEFAULTS`), arrows past `[lint] flow_arrows` 4000, or clamped at a hard limit), `flow_too_fast`, `set_under_resolved`; `variable` is drawless (no `orphan`) |
| Second example | `examples/pendulum_flow.py` | a 2D flow with a `Variable` on a `Control`, separatrices (`Manifold`), a `Probe` with its label, a disk shrinking |
| Docs, example, kit | `docs/ai/API.md`, `examples/lorenz_explorer.py`, `kit/examples.toml`, `core/pycode.py` | the "Flows" section (a running block), three lint rows; the exit demo; the `flow` alias for "convert to script" |
| Tests | `tests/test_flow_parser.py`, `tests/test_flow.py`, `tests/test_final.py` | towers under 0.1 s; long, deep, broken input; extras and the complex whitelist; corpus before/after; Lorenz numbers (maths and objects); 3D classification table at three scales; non-isolated; presets; batch independence (1, 7, 1000 seeds, Lorenz and a sin/exp field, with sub-stepping on); bounded backward Lorenz; stiff field lint; four hostile formulas (1000 seeds x 1000 steps, objects inspect and lint); every `FlowSim` input in the cache key; an unused constant leaves the key alone; scrub order vs a fresh runner (refinement included); refined points are true images; the cap and its lint; children read the parent and a fresh engine agrees; a `Variable` overrides a constant; budgets clamped and linted; every type at its maximum in one scene, asked for flow time 2000 (200000 points with the longest trail they allow, a 4096-sample 8-ring outline, a 200-per-side field, a 64-grid Newton, a probe and a manifold past their step caps: clamped, time and `tracemalloc` peak asserted); each memo checked against a cold run after changing one input (window shift, resize, escape, dt, constants, a Variable); the stiff, budget, cap and hostile tests in both `MODES` (the hostile ones also render a still with GL); the exit demo; golden storyboard (`tests/golden/software/lorenz_explorer.png`, tau <= 5); every flow type in `test_final` (2D and orbit, `passes=None`, drawn with GL; partner `System` "sys") |

## Audit (independent, 2026-11-24)

| # | Finding | Fix | Test |
| --- | --- | --- | --- |
| B1 | the tower guard read the base at 15 digits, so `(1+10**-17)**(10**7)` (and `(1+10**-20)**(10**30)`, `((10**16+1)/10**16)**(10**30)`, the complex and half-integer variants) looked like 1 and SymPy computed billions of exact digits | `codegen._power`: an exact base (rationals, `I`) to a whole power is refused past `MAX_EXACT_BITS` (2e5) bits of answer (exponent x bits of numerator and denominator); other bases are measured at 60 digits | `test_powers_a_hair_from_one_are_refused_at_once` (7 cases, both parsers), `test_legitimate_powers_still_read` |
| B2 | the real-only extras reached every `parse_expression` caller, so calc, geometry and dyn accepted `floor`/`sign` and crashed at inspect; a flow System with them crashed (the divergence lambdified DiracDelta) | `parse_expression(..., extras=False)` by default: only `flow_math` passes `extras=True`, every other domain refuses them with the base commit's error text; the System's divergence is measured as the trace of the (numeric where needed) Jacobian, its formula None where there is none | `test_real_only_extras_are_for_the_flow_core_only` (exact base text), `test_other_domains_still_refuse_the_extras_at_validation`, `test_a_system_with_sign_and_floor_inspects_and_lints` (both modes) |
| B3 | `linear` used `sp.Poly`, which expands: `(x+y)**1000000` ran out of memory | linear = the Jacobian holds no coordinate (no expansion); the parser refuses a sum raised past `MAX_SUM_POWER` (100) in every domain | `test_linearity_is_decided_without_expanding`, `test_a_sum_to_a_huge_power_is_refused` |
| B4 | Newton stopped on the residual alone and clustered at a fixed 1e-6: a double or triple root came back as 5-44 points (saddle-node and pitchfork at mu 0, Lorenz at rho 1, x' = x^2), Lorenz rho 1 even flagged non-isolated | the flow core's Newton (`singular="pinv"`) runs up to `POLISH_ITERS` (150) until its steps stop moving (`XTOL` 1e-12) as well as the residual; points with a singular Jacobian closer than `MERGE` (1e-4 of the window) are one root and only separate singular points count toward `non_isolated`; `dynamics_math` ("drop") is unchanged | `test_a_degenerate_equilibrium_is_one_point` (4 presets + x^2), `test_lines_of_equilibria_stay_non_isolated` (y,0; y,0,0; xy,xy; SIR), `test_sweeping_a_slider_through_a_bifurcation_counts_right` |
| S5 | `tidy` counted 2 operations in `(x+y)**1000000` and simplified it for ever | no simplify past a power of 16 either | `test_simplify_skips_high_powers` |
| S6 | the exact Jacobian of a long formula took up to 50 s to build | past `JAC_OPS` (120) operations or `JAC_DEPTH` (24) nesting per component the Jacobian is central differences (`jac_expr` None); `classify` of a non-finite Jacobian is kind `undefined` instead of an exception | `test_a_huge_formula_compiles_quickly_with_a_numeric_jacobian` |
| S7 | a set's trail sat in every checkpoint and each slider value wrote a new folder: 1.5 GB of `.phasor_cache` for one setting | checkpoints are positions only and stay in memory (`cache_dir=None`); the trail is recomputed when drawn (`FlowSim.trail_from`: the last `trail` steps from the state `trail` steps back). Sizes: one checkpoint is points x dim x 8 bytes (2e5 points in 3D: 4.8 MB); a runner keeps 6, `runner_for` keeps 16 runners: at most about 460 MB at the maximum count, nothing on disk | `test_a_set_writes_nothing_to_disk_and_its_trail_is_not_checkpointed` |
| S8 | the exit demo's card was tiny and on the point cloud; the C+ subspaces not visible | the card uses the `label` text role; at "card" the field leaves, the ball steps back to 0.35 and the camera pulls in closer; the golden storyboard is 480x270 per frame (looked at) | `test_lorenz_explorer_golden_storyboard` |
| S9 | no example of `Tracer`, `Manifold`, `Variable`; `flow.Tracer` clashed with the motion `Tracer` | renamed `flow.Probe` (`flow_probe`); `examples/pendulum_flow.py` | `test_the_pendulum_example_shows_its_parts`, `test_final` |
| N | tower timing asserts at 0.1 s; two Variables of one name silently first-wins; the SymPy version not in the set's key | 2 s guards; lint `duplicate_variable`; `FlowSim.params()["sympy"]` | `test_two_variables_of_one_name_lint` (both modes) |

The audit's own probe scripts showed each blocker failing on the commit before these fixes;
the regression tests above reproduce them.

## Measured here (4 cores, llvmpipe, shared with other sessions; not a benchmark)

- Lorenz equilibria from a 9^3 grid: 34 ms. 10 RK4 steps of 1000 Lorenz seeds: 2.5 ms.
- `lorenz_explorer`: `inspect` 0.9 s cold; a 960x540 still about 0.7 s on llvmpipe; storyboard of three frames at 480x270 about 0.7 s.
- A 64-point Lorenz ball outline refined to 8251 points by tau 2.
- Parser: `9**9**9` and the other towers 0.1-0.7 ms; a 999-term sum 64 ms.

## Deviations

1. **`render/gpu/codegen.py` was edited** although this milestone was told not to touch
   `render/`: the whitelist parser lives there and `ComplexFunction.parse` shares `_build`,
   so the tower guard and size limits had to go in that file (a wrapper elsewhere would
   leave `cx.py` open to `9**9**9`). Only `_build`, the two entry points, the limits and a
   real-only function table changed; no renderer code.
2. **`System.terms` are text, not tables** (advice 2.2 sketches dicts): a document
   parameter may only be numbers, strings and tuples (`core/values.py`), so a source is
   written `vortex(pos=(0, 0), strength=2)` and read with the AST (`parse_term`, keyword
   numbers only). Implemented: `uniform`, `source`, `sink`, `vortex`, `linear`. Deferred:
   `saddle`, `centre`, `dipole`, `gradient`, `preset` as terms (a preset is
   `flow.preset(name)`, data for the System's own parameters).
3. **No `dim` parameter**: the dimension is the number of `equations` (one way to say it).
4. **Every named number needs a value in `constants`**; a `Variable` of the same name
   overrides it. The validator cannot see other objects, so a free name without a constant
   is refused at once instead of becoming an unbound-variable lint. Milestone 27's row
   classifier writes the constant when it creates the `Variable`.
5. **`Obstacle` is deferred** (allowed by the task).
6. **Positions and chaotic counts in semantics only up to `SAFE_TAU` (10)**, a fixed rule
   rather than an estimate of the Lyapunov time: past it `Probe.at`, `FlowSet.box`, its
   `outline`/`alive`/`escaped`/`not_finite` counts and `Manifold` lengths are `None` with a
   note. `too_fast` and `under_resolved_since` stay (the lint rules read them); the latter
   is a step on the refinement grid and may differ across machines near its threshold.
7. **Outlines are short-lived by budget**: counting an outline at its cap (2^17) bounds a
   set with an outline to 762 steps at `MAX_WORK` 1e8 (flow time 7.6 at dt 0.01) even when
   the outline stays small; a tighter bound needs work estimates the parameters alone do not
   give. Clamped and linted, never a hang.
8. **3D kinds**: a 3D focus is named `stable spiral` / `unstable spiral` (the 2D strings);
   the one new string is `saddle-focus`. A pure-imaginary pair in 3D is `non-hyperbolic`
   (there is no 3D `centre`).
9. **`FlowSet` trails are drawn as faded segments of the last `trail` steps** (one sample per
   step), for filled points only, recomputed at draw time (`trail` x points steps a frame);
   outlines have no trail.
10. **`flow.Tracer` is `flow.Probe`** (audit S9): the top-level motion `Tracer` already owns
   the name; the type string is `flow_probe`.
11. **The flow core's Newton differs from `dynamics_math`'s** (polishing, merging degenerate
   roots): `dynamics_math` keeps the residual-only Newton so `dyn.FixedPoints` and the
   `phase_portrait` golden stay bit-identical; a `dyn` portrait at a bifurcation still shows
   a small cloud, as before Milestone 26.
12. **`flow_budget` severity is warning** (there is no "info" severity); so are
   `flow_too_fast` and `set_under_resolved`.
13. **Checkpoints of sets stay in memory** (audit S7), unlike other `Simulation`s: a set is
   recomputed when a scene is reopened or in a render worker (deterministic, so the same).
14. **Manifolds start `1e-3` window diagonals off the point** (the core's default is `1e-5`
   times the point's size): a slow saddle (the pendulum's, rate 0.86) otherwise shows
   nothing in a few time units.

## Not done / open

- Gate and independent audit; Windows and CUDA runs (`pc_tasks/`), and the Windows coarse
  clock for the "under 0.1 s" tower test (it uses `perf_counter`, which is fine-grained on
  Windows; untested there).
- World-space batches (Milestone 28): flow objects still build camera-facing geometry on the
  CPU for disks and arrows (cones are camera-free).
- Lyapunov, sections, sweeps (Milestone 30); rows, normaliser and AI tools (Milestone 27).

## Test runs here (Linux, llvmpipe, 2026-11-24)

On a box shared with other sessions (load about 20 on 4 cores), the suite ran in parts:
`-m "not slow"`: 2288 passed, 133 skipped, 1 failed (`test_llms_txt_is_current`: fixed by
regenerating `llms.txt`, then passed); `-m slow`: 27 passed, 17 skipped, with
`test_every_example_runs_and_has_no_lint_errors[riemann_sqrt]` left out (it ran over 40
minutes under that load and was stopped; not run on this tree). `tests/test_flow.py`: 52
passed. ruff check, ruff format and lint-imports clean. Skips are the usual ones (no CUDA,
no gui extra, no TeX, no espeak-ng, opt-in packaging).
