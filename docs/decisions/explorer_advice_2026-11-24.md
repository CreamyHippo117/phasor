# Explorer: design advice (2026-11-24)

Read-only review. Nothing in the repo was changed. Numbers marked **measured** were run on
this box (4 cores, llvmpipe) today; everything else about speed is an **estimate** and says so.

Measured here:

| What | Result |
| --- | --- |
| `Engine.still` of `PhasePortrait(flow=True, n_seeds=40)` + `FixedPoints`, look none | 113 ms at 480x270, 250 ms at 960x540 (4-9 fps) |
| `Engine.state_at` for that scene | 1.2 ms |
| NumPy RK4, Lorenz, one step | 0.6 ms for 10^4 points, 11.8 ms for 10^5 points |
| `gm.compile_fn` field evaluation, 10^4 points, 3 components | 0.09 ms |
| Lorenz forward 20 time units then backward 20 | NaN (backward run blows up) |
| `parse_expression("9**9**9", ...)` | no answer in 20 s (killed) |
| `parse_expression("x+x+...+x")`, 20000 terms | `RecursionError`, not `FunctionError` |
| `parse_expression` of `2x`, `x^2`, `x'`, `sign(x)` | refused (`FunctionError`) |
| Lorenz (10, 27.7, 2.67) eigenvalues at C+ and at 0 | -13.840, 0.0852 +/- 10.149i; -22.741, -2.67, 11.741 (match the screenshots) |

---

## 1. Architecture

**Verdict: the hypothesis is right in its main claim (the explorer edits an ordinary document
through `Engine.apply`) and wrong in three details.** Do not build a standalone live viewport
that exports to a document: it would be a second engine (contract 1), and the AI tools, undo,
save, lint and the video renderer would all have to be written again for it.

### 1.1 What to change in the hypothesis

1. **Not every action is an op.** Split state in two:
   - *Document state* (goes through `Engine.apply`, undoable, saved): rows, expressions,
     variables' values, seeds, regions, overlays, the camera's resting pose, the flow rate.
   - *Session state* (never an op): playhead, paused, playback speed, reverse, the camera
     while the mouse is down, a slider while it is being dragged, text while it is being typed.
   A drag or a keystroke works on a fork (`Engine.fork`, exactly the `interactive.Player`
   pattern that `Editor._preview_player` already uses) and commits **one** patch on release
   or on a parse-clean debounce. Otherwise one orbit is 200 undo steps and each step pays
   validation and a lint delta.
2. **"Precompute all trajectories as arrays" does not survive 10^5 seeds.**
   10^5 seeds x 4000 steps x 3 x 8 bytes = 9.6 GB. Store full paths only for single probes.
   Sets of points keep *checkpoints of the current positions* plus a short trail (see 1.3).
3. **Reverse is the playhead moving backward over the forward run, not backward integration.**
   Measured: Lorenz integrated backward for 20 units is NaN. Backward flow (tau < 0, needed
   for stable manifolds) is a separate, bounded run with an escape radius; when it escapes the
   UI says "the backward flow leaves the window at tau = -1.3", which is true maths.

### 1.2 Time

- Each `System` object has an animatable param `tau` (flow time). In a rendered scene it is a
  normal track (`tau` 0 -> 30 over 20 s). In the explorer, the session clock drives the
  playhead `t`, and the System's `rate` param maps `t` to `tau` (`tau = tau0 + rate * t`).
  The speed slider is session playback rate (the "0.053x normal" readout); "Restart" sets
  `t = 0`. So `state_at(t)` stays pure in (document, t) and the same document renders to video.
- The document needs a duration for the renderer. The explorer plays past the end (the flow
  has no end); `meta.duration` is set only by "Send to scene" (section 3).

### 1.3 Cache and bounds design

One mechanism, the one that exists: the flow of a set of seeds **is a `Simulation`**
(`core/simulation.py`): fixed `dt`, pure `step` (one RK4 step of all seeds), `params()` holds
everything that decides the answer. `SimRunner` already gives nearest-checkpoint-plus-steps,
a cursor for playing forward, a memory LRU, disk checkpoints, and a key that includes the
solver's source hash. Backward scrubbing costs at most `every` steps.

- **Fixed-step RK4 only.** No adaptive RK45: with a shared error norm the step depends on
  which other seeds are in the batch, so adding a seed would change every other trajectory.
  Offer `dt` (default from the field's speed scale in the window, clamped) and one honest
  check: run a probe at `dt` and `dt/2`, report where they part by more than a pixel.
- **`params()` must contain:** dimension; every component's source text; every variable
  value the expressions use (only those: a slider the system does not use must not
  invalidate it); the summed sources' parameters; seeds as bytes (or the region's definition
  and count); `dt`; method name; direction; escape radius / bounds; dtype; trail length.
  `sim_key` already adds the class, cache version and source hash.
- **Budgets (clamp and lint `flow_budget`, never a validation error):**
  - single probes: at most 200 per system, at most 20000 stored steps each (48 MB worst case in float32);
  - a point set: at most 2*10^5 points on CPU (10^6 with the GPU kernel, PC only);
  - trail: at most 64 samples; points x trail <= 4*10^6 (48 MB in float32);
  - `SimRunner.MEMORY = 6` checkpoints of (N, 3) float64 at N = 2*10^5 is 29 MB: fine;
  - disk: checkpoints of a slider being dragged pile up under `.phasor_cache/`. Pass
    `cache_dir=None` on forks (the drag) and write to disk only for committed documents;
    add a size cap with oldest-first eviction (there is none today).
- **The existing `_CACHE` in `domains/dynamics.py` is the wrong shape for sliders:** it is
  cleared whole at 64 entries, and `_system` substitutes constants into SymPy and re-runs
  `lambdify` per value. For the explorer compile once per expression text with the variables
  as **arguments** (`compile_fn(source, ("x","y","z", *names))`); only the trajectory key
  contains the values.
- **Determinism, said honestly.** Same machine, same document: same bits (render workers
  agree, ADR 0003 holds). Across machines (Linux/Windows, NumPy SIMD paths, CPU vs CUDA) a
  chaotic trajectory differs visibly after roughly `ln(1e16)/lambda` ~ 37/lambda time units
  (Lorenz: about tau 40). So: golden stills of chaotic systems use tau <= 5; CPU-vs-GPU
  kernel tests compare at short tau with a tolerance, and compare *statistics* (bounding box,
  mean) at long tau; the docs say a long chaotic path is not reproducible across machines.
- **Per-seed independence.** A seed's path must not depend on its neighbours in the batch.
  Elementwise NumPy is close but not guaranteed bit-identical between a SIMD body and a scalar
  tail for `sin`/`exp`. Test it (section 5); if it fails, pad batches to a fixed multiple.

### 1.4 Is a separate real-time viewport path needed?

**Yes, but as a new primitive of the one renderer, not a second renderer.** Evidence:

- `gui/window.py` `Viewport` is a `QLabel` showing `Editor.frame` -> `Engine.still`
  (render, read back, QPixmap). Measured here: 4-9 fps for a small 2D portrait.
- `GPU_SPEED.md`: on the RTX 4070 Ti the card draws a frame in 1-2 ms; the time is Python
  building strokes (46-66 ms a frame after steps 1-4). Its command cache is keyed by "its
  whole camera", so **orbiting invalidates every stroke**: strokes are expanded to
  camera-facing ribbons on the CPU (`strokes.polyline(ctx, cam, ...)` takes the camera).
- M21's draft preset removes post and blur, not the stroke building. It cannot carry a 3D
  orbit of 5000 cones.

Recommendation: add **world-space batches** to `DrawContext`: `ctx.points3d`, `ctx.lines3d`,
`ctx.arrows3d` (instanced cones or flat arrows), taking plain world-space arrays plus
per-vertex colour and a width in pixels or world units; the width and facing are expanded in
the vertex shader. Buffers are keyed by the object's params only, the camera is a uniform.
Then orbit, zoom and pan re-upload nothing, and advancing `tau` re-uploads one positions
array. The final renderer, the draft preset and the explorer viewport all use the same call
("one way to do each thing"); the software backend (`render/cpu/backend.py`, `passes=None`)
expands the same arrays on the CPU so Windows CI still passes. The new flow objects draw
only through these; old objects keep `strokes.py`.

The explorer viewport then is: draft preset, 8-bit target, no post, `output="srgb8"` read
back (about 2.5 ms on the PC per `GPU_SPEED.md`), shown in the existing `Viewport`. It stays
inside `render/passes.py` and the shared context, so no convention is broken. A
`QOpenGLWidget` with its own context would break "GL contexts only via `render/glcontext.py`"
and the one-thread rule; do not do it unless the read-back proves to be the bottleneck.

Progressive quality: while the mouse is down or a slider moves, draw at half size, thin the
arrow field to a budget (e.g. 4000 arrows) and cut point sets to a stride; 150 ms after the
last input, redraw full. The thinning is a render hint, not a document change.

### 1.5 Realistic frame rates (estimates unless marked)

| Case | Cloud box (4 cores, llvmpipe) | Owner's PC (RTX 4070 Ti) |
| --- | --- | --- |
| Today's path, 2D portrait | 4-9 fps (**measured**) | about 15-20 fps (from `GPU_SPEED.md` 46-66 ms; untested for this scene) |
| World-space batches, 2D, a few thousand segments, 640x360 | 15-30 fps | 60+ fps |
| World-space batches, 3D Lorenz, 5000 arrows + 10^4 tracers with trails | 8-15 fps | 60+ fps |
| 10^5 tracers | integration alone is 11.8 ms a step (**measured**): 10-20 fps at one step a frame | needs the CUDA flow kernel for more than a few steps a frame; then 60 fps is plausible |
| 10^6 tracers | no | PC only, kernel + splat (`render/gpu/kernels/splat.py`) |

The reference app's 200 fps is C++ with ImGui. Do not chase it: the targets are 60 fps on
the PC and "usable" (15 fps, half-size while dragging) on a CPU. All PC numbers are untested
until `pc_tasks/` runs them.

---

## 2. The object model

### 2.1 Today's overlap to fix first

There are already two unrelated "vector field from formulas" families: `dyn.PhasePortrait` /
`dyn.FixedPoints` (2D, `f`, `g`, `constants="mu=1.5"`, each object repeats the formulas) and
`geo.VectorField3D` / `FlowLines` / `FlowParticles` (`fp`, `fq`, `fr`, no constants). A third
would break "extend, never duplicate". Build one core, `domains/flow_math.py`, an n-D
(n = 2 or 3) `System`, and make `dynamics_math.System` a thin 2D front of it. Old objects and
their params keep working unchanged (rule: nothing that worked before fails).

### 2.2 Types (all in `domains/flow.py`, maths in `domains/flow_math.py`)

| Type | What it is | Pure or Simulation |
| --- | --- | --- |
| `Variable` | one named number (`name`, `value`, `min`, `max`); a document `Control` points at `<id>.value`; a track on it animates it | pure |
| `System` | the field: `dim`, `terms` (a list of sources that **sum**), `tau`, `rate`, `window`, `dt`. Draws nothing or the LaTeX of the system | pure |
| `FieldArrows` (child) | the arrow field of its system (2D arrows, 3D cones), coloured by magnitude | pure |
| `Tracer` (child) | one seed `x0`: the path so far, the moving dot, the label `phi^t(x0)` | pure (stored path) |
| `FlowSet` (child) | a set of initial conditions advected as a set: `shape` (disk, ball, box, segment, circle, points), `count`, `fill` (boundary / filled / both), `trail` | Simulation (checkpoints) |
| `Equilibria` (child) | all equilibria in the window, classified; a click target | pure |
| `EquilibriumView` (child) | one equilibrium close up: Jacobian, eigenvalues, `E^s` / `E^u` / `E^c` as a line or a translucent plane with arrows, inside a sphere | pure |
| `Manifold` (child) | 1D stable/unstable manifold of a saddle, integrated from the eigen-direction | pure (stored path) |
| `Obstacle` | a region where tracers stop (v1: absorb only) | pure |

Children use dotted ids (`lorenz.field`, `lorenz.eq`, `lorenz.set_disk`) and read the parent's
params from the state, so the formulas are written once. Check that a child's `draw` can read
its parent from `SceneState` before committing to this; if not, the fallback is that the
explorer writes the same `terms` into each child in one patch (uglier, still correct).

**Sources (the `terms` list), all compiled to one SymPy expression and summed symbolically**,
so the Jacobian, the LaTeX, the divergence and the kernel code come from one place:

`uniform(v)`, `source/sink(pos, strength)`, `vortex(pos, strength)`, `saddle(pos, rates)`,
`centre(pos, omega)`, `dipole(pos, moment)`, `linear(A, pos)`, `expr(components)`,
`gradient(V)` (the field is -grad V, differentiated symbolically), `preset(name)`.
Singular sources (1/r) take a core radius (`r^2 + eps^2`), or the field is infinite at the
source and every nearby tracer dies. "Interact" in v1 means exactly this: sources add, and
everything else is advected by the sum.

### 2.3 What is pure and what needs a Simulation

- **Pure (v1):** passive advection of points and sets by an autonomous or explicitly
  time-dependent field; equilibria; Jacobians; manifolds; nullclines; sections.
- **Simulation, deterministic, cheap (v1):** `FlowSet` (it is the same passive advection,
  only stored as checkpoints).
- **Simulation, real state (defer):** inertia (x'' = F - gamma x'), reflecting walls,
  collisions between bodies, a body whose position feeds the field (two-way coupling makes
  the field depend on history).
- **Coupling in v1:** only what the param DAG already does: a param may be an `ExprSpec`
  (`{"expr": "..."}`, "computed from t and other parameters"), so a source's `pos` or
  `strength` can follow a variable or the time. Updaters (`always`) are code modules in the
  scene folder; panels cannot write them, so they are not the explorer's mechanism.
- A second-order system is typed as a first-order one in (x, v); that already works and
  covers the pendulum, Duffing and Van der Pol.

### 2.4 Advecting a region as a set

- **Filled points** (the Lorenz ball): `count` points from a deterministic low-discrepancy
  fill (no RNG needed, like `hash_seeds`), advected. Cheap, robust, beautiful with the splat.
- **Boundary kept resolved** (the deforming disk): the boundary is a closed curve
  parametrised by `s` in [0, 1). State = (`s` values, positions). At every checkpoint, where
  two neighbours are further apart than `h_max` (or the turning angle is large), insert
  `s_mid` and compute its position **by integrating that initial point from tau = 0**
  (never by interpolating positions: every drawn point is a true image of an initial point).
  The refinement happens on the fixed checkpoint grid, so it is a function of the params only.
  Cap at 2^17 points; in chaos the length grows like e^(lambda tau), so the cap is reached
  (Lorenz, 64 points to 1.3*10^5: tau about 8.5). After the cap: lint info
  `set_under_resolved` with the time, and the boundary is drawn as points.
- **3D ball boundary:** not a refined mesh in v1. Draw it as K latitude and longitude rings,
  each a refined curve as above (this is what the screenshot's wire rings are).

### 2.5 Equilibria and linear analysis

- **Find:** batched Newton from a grid (vectorised `np.linalg.solve` over all starts at once;
  today's `fixed_points` is a Python double loop over scalars), keep converged points inside
  the window, cluster with a tolerance relative to the window, sort. Cap at 64; more than
  that, or clusters forming a line, is reported as `non_isolated` (e.g. x' = y, y' = 0).
  Always report the window and grid used: "none found" is not "none exist".
- **Jacobian:** exact, by SymPy differentiation of the parsed expressions (already done in
  2D). Fall back to central differences when the derivative is not defined at the point
  (`abs`). No complex-step: `abs` and non-analytic sources break it.
- **Classify (n = 2, 3):** counts (dim E^s, dim E^u, dim E^c) with a tolerance relative to
  the Jacobian's scale; names: node, focus (spiral), saddle, saddle-focus, centre,
  degenerate, non-hyperbolic ("the linear part does not decide"). **Keep the existing 2D
  `KINDS` strings**; add 3D kinds beside them. Semantics stores `C+ = (8.44, 8.44, 26.7)`,
  eigenvalues, eigenvectors, the three dimensions.
- **Subspaces:** from real eigenvectors, and the real and imaginary parts of a complex pair
  (a plane). Fix signs deterministically (the existing rule in `classify`).

### 2.6 What else, ranked by value / cost

| # | Feature | Value | Cost | When |
| --- | --- | --- | --- | --- |
| 1 | Click to drop a tracer; drag it and the path follows | very high | low | v1 |
| 2 | Equilibria + click card + zoom into the linear picture | very high | medium | v1 |
| 3 | Deforming disk / ball of initial conditions | very high | medium | v1 |
| 4 | Separatrices: 1D stable/unstable manifolds of saddles (2D and 3D) | high | low | v1 |
| 5 | Sensitive dependence: two tracers eps apart, with their distance on a log strip | high | low | v1 |
| 6 | Auto sliders + "play" on a slider (a parameter sweeping, live) | high | low | v1 |
| 7 | Time-series strip (x(t), y(t), z(t) of the selected tracer) | high | low | v2 |
| 8 | Nullclines in 2D (exists), divergence as a 2D heat map, area of the disk vs exp(integral of div) | medium | low | v2 |
| 9 | Poincare section (crossings of a plane, splatted) | high | low-medium | v2 |
| 10 | Largest Lyapunov exponent (Benettin), with a convergence spread | medium | low | v2 |
| 11 | Bifurcation diagram over a variable | high | medium | v2 |
| 12 | Limit cycle detection and period (return to a section) | medium | medium | v2 |
| 13 | Basin colouring, 2D | high | high (N^2 trajectories: kernel) | PC |
| 14 | FTLE field, 2D | medium | high (kernel) | PC |
| 15 | Nullsurfaces and implicit surfaces in 3D (marching cubes) | medium | high | later |
| 16 | 2D manifolds in 3D, globalised (the Lorenz manifold) | high looks | very high, fragile | defer; v1 shows the local disc only |
| 17 | Inertial bodies, reflection, collisions, two-way coupling | medium | high | defer |

---

## 3. The panel UI

### 3.1 Layout

```
+----------------------------------------------------------------------------------+
| File  Edit  View (Code | Explorer)  Scene  AI                                    |
+----------------------------+-----------------------------------------------------+
| [R2] [R3]      [+ row v]   |                                 x' = 10(y - x)      |
| 1 (o) x' = sigma (y - x)   |                                 y' = x(rho - z) - y |
| 2 (o) y' = x (rho - z) - y |              viewport           z' = x y - beta z   |
| 3 (o) z' = x y - beta z    |    click: tracer   shift+drag: disk                 |
|   sigma [----o-----] 10  > |    drag: orbit/pan   wheel: zoom                    |
|   rho   [-------o--] 27.7 >|    click a red ball: card, "zoom in"                |
|   beta  [--o-------] 2.67 >|                                          0.053x     |
| 4 (o) disk  c=(1,1,20) r=2 +-----------------------------------------------------+
| 5 (o) tracer (1, 1, 1)     | [<<] [play] [>>]  reverse  restart  clear   speed   |
| 6 ( ) y = sin(x)           +-----------------------------------------------------+
|   ! cannot read "sin(x"    | Analysis | Inspector | Problems | AI                |
| Presets: [Lorenz        v] |  C+ = (8.44, 8.44, 26.7) saddle-focus  dim E^s = 1  |
+----------------------------+-----------------------------------------------------+
```

- **Left: the expression list** (section 7), with variable sliders appearing under the row
  that first uses them, a preset dropdown, and a `+ row` menu for things that are not typed
  (disk, ball, tracer, source, obstacle, section plane).
- **Viewport, direct manipulation:** click empty space adds a tracer at that point (in 3D:
  on the plane through the orbit target facing the camera, shown as a faint grid while
  hovering); shift+drag draws a disk/ball (centre, then radius); drag a handle to move or
  resize; click an equilibrium for its card; double-click or "zoom in" animates the camera to
  it and adds its `EquilibriumView`; drag empty space orbits (3D) or pans (2D); wheel zooms;
  space pauses; left/right step; `r` restarts. `Editor.drag_camera`, `orbit_camera`,
  `zoom_camera`, `pan_camera`, `hit_ids`, `screen_to_world`, `drag_to` exist and are reused.
- **Overlays:** the system in LaTeX top right (the app typesets maths already), `phi^t(x0)`
  labels on tracers, the speed readout bottom right. Each is a toggle that is an object's
  param, so a rendered video shows the same.
- **R2 / R3 tabs:** two documents (two files), each with its own field, camera and settings.
  That is the reference app's behaviour and it avoids a 2D system and a 3D system fighting
  over one camera.

### 3.2 Undo, save, load

- One history, the engine's. A typed edit commits when the row parses and the debounce
  fires; a drag commits on release; a slider drag commits on release. Each entry has a
  readable name ("rho = 24.74", "added disk", "AI: added tracer").
- Play, pause, speed, reverse and scrubbing are not in the history.
- Save writes a plain `name.phasor.json`. **An explorer document is not a script scene**:
  script scenes accept only the three tweak ops (`TWEAK_OPS`, refusal `scripted_scene`), so
  adding a row would be refused.

### 3.3 Shared with the code-first window (M25) and what is new

| Shared, unchanged | New |
| --- | --- |
| `Workspace` + lock, `Editor` (camera, selection, hit test, undo, save, frame), `EditorWindow`, `Viewport`, `Inspector`, `LintPanel`, `ActivityDock`, the AI tab | `gui/explorer.py` `Explorer` (no Qt): rows, classification results, session clock, drag-on-a-fork and commit |
| export, looks, storyboard | `ExplorerPanel` in `window.py`: draws the rows and forwards input, nothing else |
| the document, patches, `core/pycode.py` | the transport bar (session clock) |

Explorer is a **view of the same window** (View > Explorer), not a second app. The left pane
shows the script or the rows.

**One decision the owner must make** (it touches `APP_PLAN.md`): APP_PLAN says a
`.phasor.json` is converted to a script when opened. Explorer documents must be the
exception and stay JSON while they are explored. **"Convert to script"** runs
`document_to_python`, and from then on the Explorer panel can move sliders and params
(tweaks, written into the tweaks block per ADR 0004) but cannot add or remove rows; new
rows are code. The UI must say this at the moment of conversion.

### 3.4 Send to scene, export video

- **Export video:** the normal renderer on the document, with a duration and a `tau` track
  the dialog asks for ("from tau 0 to 30 in 20 s"). Nothing else is needed, which is the
  payoff of the architecture.
- **Send to scene:** a "record" toggle captures camera poses and slider values at marks the
  person sets ("mark" button); on stop, they become camera segments and tracks between
  markers through `Engine.apply` (the same ops `animate` uses). Then optionally "Convert to
  script". Do not record every mouse wiggle: marks only, eased between.

### 3.5 Presets

Lorenz, Rossler, Thomas, Chua, Halvorsen, Aizawa (3D); Van der Pol, damped pendulum,
Duffing (unforced), Lotka-Volterra, Brusselator, FitzHugh-Nagumo, SIR (2 of 3 variables),
double well, Hopf normal form, saddle-node normal form, pitchfork normal form, and the linear
zoo (saddle, node, focus, centre, degenerate). A preset is data: rows text, variable values
and ranges, window, `dt`, a starting camera, one sentence. Stored in one table
(`flow_math.PRESETS`); `cloud_math.attractor` should read its constants from the same table.

---

## 4. Milestones

| # | What | Depends on | Where | Exit demo |
| --- | --- | --- | --- | --- |
| 26 | **Flow core**, headless: `flow_math` (n-D system with variables as arguments, batched RK4, flow as a `Simulation`, equilibria, 3D classification, 1D manifolds, set refinement), parser hardening (section 5), the objects of 2.2 drawing through `strokes.py` for now, presets, API.md, examples, goldens | none | cloud | `examples/lorenz_explorer.py` renders: cone field, three red equilibria, a ball of 2000 points folding, the C+ card; eigenvalue tests match the table above |
| 27 | **Rows, variables and AI tools**: the row normaliser and classifier, row <-> object mapping, `Variable` + auto `Control`, `explorer_*` Workspace tools, `explorer_state`, MCP.md, llms.txt, skill | 26 (interfaces only: can start with stubs) | cloud | a test script of typed lines produces the same document as M26's example; an MCP session finds Lorenz's equilibria and the Hopf value of rho |
| 28 | **World-space batches and the live draft path**: `ctx.points3d/lines3d/arrows3d` in GL and in the software backend, flow objects moved onto them, half-size-while-dragging, thinning | M21 (draft preset) | cloud for correctness; **PC for timing** | bench: orbiting the M26 scene re-uploads no buffers; fps table of 1.5 filled in with measured numbers on both machines |
| 29 | **The Explorer panel**: `gui/explorer.py`, `ExplorerPanel`, transport, click/drag tools, equilibrium card and zoom, R2/R3 | 27; M25 phase 2 if it has landed, else it docks in today's window; 28 optional (works slower without) | cloud (offscreen Qt tests); **PC for feel** | the owner types Lorenz, drags rho, drops a ball, clicks C+, zooms: no code |
| 30 | **Analysis pack**: Poincare section, Lyapunov estimate, time-series strip, twin tracers, divergence map, bifurcation sweep, limit cycles; `sweep_parameter` tool | 26, 27 | cloud | Rossler section and period-doubling sweep; logistic-like bifurcation picture |
| 31 | **Scale and handoff**: CUDA flow kernel (expression -> real-valued CUDA C, NumPy reference, emulator test), 10^5-10^6 points with the splat, basins and FTLE kernels, record/marks -> scene, export dialog, implicit surfaces (marching cubes) | 28, 29 | **PC** (kernels, timing); cloud for references and emulator | the "Coming Up" picture: 10^5 trajectories of Lorenz at 60 fps on the PC, exported as a video |

Parallel worktrees:

- Now, at once: **26** (one agent on `flow_math`, one on objects/docs/goldens once the maths
  API is fixed), **27's parser and classifier** (pure text, no dependency), and **M21**.
- After 26: **27 tools**, **30**, and **28** in parallel (28 touches `render/`, 30 touches
  `domains/`, 27 touches `workspace/`: no shared files except `docs/ai/API.md`; give each
  its own section and merge docs last).
- **29** is the only one that waits for people-shaped feedback; keep it small and last
  before 31.
- Strongest model: 26's maths (refinement, classification tolerances), 28 (renderer), 31's
  kernels. A smaller model is fine for presets, docs, the panel widgets.

---

## 5. Risks and traps for this feature, with the test that catches each

**Expressions (found today, reproducible):**

1. `9**9**9` hangs `parse_expression` (SymPy evaluates the integer tower). In a box that
   re-parses while typing this freezes the app. Fix: refuse constant powers whose result
   would exceed a bound before building them (check exponent size on `ast.Pow` of two
   constants), cap literal magnitude and exponent. Test: a table of towers must return
   `FunctionError` in under 0.1 s.
2. A 20000-term sum raises `RecursionError`, not `FunctionError`. Fix: cap source length
   (e.g. 2000 characters) and node count; catch `RecursionError` and `MemoryError` at the
   one parse entry. Test: long, deep and wide inputs all give `FunctionError`.
3. `sp.simplify` (used in `geometry_math.divergence` and `curl`) has no time bound on
   nasty input. Use `sp.diff` without `simplify` in anything that runs while typing; simplify
   only for the LaTeX label with a node-count guard. Test: a 1500-character legal formula
   completes classification and semantics in under a second.
4. Legal but hostile: `x**1e9`, `exp(exp(exp(x)))`, `1/(x-y)`, `log(x)` for x < 0. They
   parse. Evaluation already runs under `np.errstate(all="ignore")`; the rule must be: a
   non-finite velocity kills that tracer (NaN from there, as `trajectories` does), never an
   exception, never a hang; equilibria skip non-finite starts. Test: each of these as a
   system, 1000 seeds, 1000 steps, finishes within a time budget and the object draws.
5. Stiff or huge fields (`1e6*x`): fixed `dt` RK4 explodes. Clamp speed x dt to a fraction of
   the window per step (sub-step up to a cap, then kill and lint `flow_too_fast`). Test: the
   system `x' = -1e6 x` gives a lint, not garbage flying across the screen.
6. `parse_expression` does not know `sign`, `min`, `max`, `atan2`, `floor`, `^`, implicit
   products or primes. The row normaliser (section 7) adds notation; adding functions to
   `_FUNCS` changes the complex-function kernels' whitelist too (`_C_FUNCS` has no code for
   them), so give `parse_expression` its own real-only extras. Test: `ComplexFunction.parse`
   still refuses `sign(z)`.

**Bounds:**

7. Every count is validated (`n_seeds le=200`, `steps le=4000`, `count le=5000` today), but
   the *products* are not: seeds x steps x trail, grid^3 arrows (`VectorField3D.spacing`
   has `ge=0` and no cap: `spacing=0.001` over a 4x4x4 box is 6.4*10^10 arrows), Newton
   grid^3 x iterations, refinement growth. Budget the product, clamp, lint. Test: each type
   at its documented maximum, time and peak memory measured and asserted.
8. Disk cache growth from slider drags (1.3). Test: 500 distinct variable values on a fork
   write nothing to disk.

**Purity and caches:**

9. A cache key missing an input: variable values, `dt`, direction, escape radius, the
   summed sources, dimension. Test (property style): for each param of `System` and
   `FlowSet`, change it alone and assert the output arrays change or are provably equal.
10. Scrub order: `state_at(t)` after any sequence of scrubs equals a fresh engine's. Test:
    random scrub sequences against a cold `SimRunner(cache_dir=None)`.
11. Batch independence: seed k's path with 1 seed, 7 seeds and 1000 seeds is bit-identical.
12. Refinement determinism: the refined boundary at tau, reached by playing forward, by
    jumping, and in a render worker, is the same array.
13. Session state leaking into the document: playing for a minute leaves `Editor.version`
    and the undo depth unchanged. Test it directly.
14. Chaos across machines: goldens only at short tau; never a golden of Lorenz at tau 40.

**Ids:**

15. `retired_ids`: delete the variable `a`, type `a` again: `AddObject` is refused
    (`id_retired`). The row/variable id must come from a counter-free scheme that survives
    this (`Variable.name` is the maths name; the id is `a`, then `a_b`-style next free
    name); the same for rows deleted and re-added, and for children of a deleted system.
    Test: add, delete, add the same text 5 times.
16. Derived child ids (`lorenz.eq`, `lorenz.set_disk`) colliding with a user row named the
    same; an equilibrium's derived label (`C+`) is not an id and must not become one
    (equilibria move and merge as a slider moves). Test: sweep rho across 1 (pitchfork: 1 ->
    3 equilibria) with the card open; nothing raises, the card follows or closes.

**Windows and CI:**

17. No OpenGL on Windows CI: every new object and lint is tested with `passes=None` and with
    GL (`MODES`). The world-space batches need a software implementation from day one or
    M28 turns the Windows suite red.
18. Qt tests on `offscreen`; no timing asserts on Windows' coarse clock (a 0 ms debounce can
    read as not elapsed); drive the debounce with an injected clock.
19. Any `subprocess` (none expected) passes `NO_WINDOW`; preset and row files are utf-8.

**Compatibility:**

20. Folding `dynamics_math.System` into `flow_math`: `test_dynamics.py` and the
    `phase_portrait` golden must pass unchanged, including the kind strings ("stable
    spiral") and `constants="mu=1.5"`.
21. Changing `_FUNCS` or the parser's limits must not reject a formula in `examples/`,
    `showcase/`, `gallery/` or `docs/ai/`. Test: parse every formula string found in those
    trees before and after.
22. New numeric analysis in semantics changes `inspect` output and storyboard snapshots:
    round to a fixed precision (6 digits, as `FixedPoints` does) and never put
    machine-dependent chaotic values in semantics.

**GUI:**

23. Typing on the GL thread: a parse or Newton run that takes 300 ms freezes the window.
    Run analysis in a worker with a generation counter (drop stale answers); GL only via
    `glcontext.on_gl_thread`.
24. The AI and the person editing at once: every edit takes the `Workspace` lock; a row
    being typed (uncommitted text) must not be overwritten by an AI edit to the same row:
    refuse with the row's version, like `script_edit` in APP_PLAN.

---

## 6. What to cut or defer, and where the hypothesis is wrong

Wrong or incomplete in the hypothesis:

- "Every action is an op": no (1.1). Document state yes, session state no.
- "Trajectories are precomputed arrays": only for single probes. Sets are checkpointed.
- "RK45": no. Fixed-step RK4, for purity under changing seed sets.
- "Reverse flips integration": no. Reverse replays; backward flow is separate and bounded.
- "The draft preset can carry it": for 2D on the PC, probably; for 3D orbiting, no, because
  strokes are built per camera. It needs world-space batches.
- "Converted to a script via pycode" is true but one-way for structure (3.3).

Cut from v1: reflecting walls, inertia, collisions, body-to-field coupling, global 2D
manifolds, FTLE, basins, implicit surfaces in 3D, adaptive solvers, a dedicated GL widget,
more than one system per document (allow it in the schema, show one in the panel),
maps/discrete systems, regression, tables, units, recording every mouse move.

Keep small: 9 object types, 6 MCP tools, 3 world-space primitives.

A different architecture is **not** better. The standalone-viewport-then-export design would
reach 200 fps sooner and would lose undo, the AI tools, lint, save/load and "what you explore
is what you render". The one-engine design costs one renderer feature (M28).

---

## 7. Desmos-like expression list

### 7.1 The model

- **Each row is one document object** (or one component of a `System`), with `visible`
  (the object's `opacity`/a `shown` param), colour (a style role, `set_style`; explicit
  colours are allowed since M18), delete (`RemoveObject`, id retired), and order (draw
  order; if the document has no reorder op, add a `row` integer param rather than a new op).
- The row's **source text is stored on the object** (a `source: str` param every row type
  has), so the list is rebuilt from the document alone, the AI reads exactly what the person
  typed, and a row that does not parse can still be saved: it is stored as a `Note`-like
  inert row object with the text and the error, drawing nothing.
- The list is a *view*: `Explorer.rows()` derives it from the document. No second store.

### 7.2 Notation: a normaliser in front of the whitelist parser

`parse_expression` refuses `2x`, `x^2`, `x'` (measured). Do not loosen the parser. Add one
function, `normalize_row(text) -> (kind, parts)`, a small tokenizer that rewrites into the
existing grammar: `^` -> `**`; implicit products (`2x`, `2(x+1)`, `x y`, `(a)(b)`,
`3sin(x)`; a name directly followed by `(` is a call only if it is a known function,
otherwise a product); unicode (`·`, `×`, `−`, `π`, `θ`, Greek letters to names); primes and
dots (`x'`, `ẋ`, `dx/dt`) as left-hand sides only. The hint line under the box lists
variables, constants and functions, like the reference app. The normalised text goes through
`parse_expression`, so the safety argument stays one argument. It also returns LaTeX (SymPy,
as `ComplexFunction.latex` does) for the overlay.

Names: a run of letters is **one name** (`rho`, `sigma`, `k1`), unlike Desmos, where `ab`
means `a*b`. Products need a space, `*`, a digit or a bracket (`x y`, `2x`, `a(b)`). Say so
in the hint line.

### 7.3 Classification (decided by the shape of the statement, in this order)

| Shape | Row kind | Object |
| --- | --- | --- |
| `name = number` | variable | `Variable` + `Control` |
| `x' = ...`, `y' = ...`, `z' = ...` (also `dx/dt`) | ODE component; rows with primes of x, y (, z) form one system, its dimension is the number of primed coordinates | `System` term `expr` |
| `(P, Q)` or `(P, Q, R)` in x, y, z; or `F(x,y) = (P, Q)` | vector field (drawn as arrows; a "flow it" button turns it into ODE rows) | `System` + `FieldArrows` |
| `(f, g)` or `(f, g, h)` in `t` only | parametric curve | `geo.ParametricCurve` |
| `(f, g, h)` in `u, v` | parametric surface | `geo.ParametricSurface` |
| `y = f(x)`; a bare expression in `x` only | graph | `calc.Graph` |
| `x = g(y)` | sideways graph | parametric curve |
| `r = f(theta)` | polar curve | parametric curve |
| `z = f(x, y)` | surface (switches the document to R3) | `geo.ParametricSurface` |
| a bare expression in `x, y` | scalar field: heat map in R2 | `heatmap` |
| `lhs = rhs` in `x, y`, not solved for one of them | implicit curve | new `ImplicitCurve` (vectorise the marching squares already in `dynamics_math.nullcline_segments`) |
| `lhs = rhs` with `z`, not solved | implicit surface | deferred (marching cubes); the row says "not yet" |
| `(a, b)` or `(a, b, c)` of numbers | point; a button offers "start a tracer here" | `Tracer` or a dot |
| anything else | error under the row |  |

Every row has a small kind badge with a dropdown to override the guess (a bare `x*y` as a
surface instead of a heat map). The override is a param (`kind`), so it is saved and undone.
Reserved names: `x y z t u v r theta`, `pi e i`. Any other free name is a variable.

### 7.4 Sliders

- A free name with no `Variable` creates one (value 1, range -10..10, step 0.01) **at
  commit time only** (parse-clean and debounce), or typing `sigma` would create `s`, `si`,
  `sig`. Variables are never deleted automatically; an unused one is shown dimmed with a
  remove button.
- The slider is a document `Control` (`path="<id>.value"`): this is exactly what
  `interactive.Player`, the editor's Controls tab, the notebook and `export_web` already
  read, so an explorer document gets a notebook and a web export for free.
- Dragging uses the fork (`Editor.set_control` / `apply_controls` already implement
  preview-then-commit). Typing a number into the slider's box commits.
- **Slider animation** ("play" on a slider): in the session it is a clock moving the preview
  value; "keep" turns it into a track on `<id>.value` (a normal animation, renderable).
- Ranges: the `Control`'s `min`, `max`, `step`, edited in place (the `>` in the sketch).

### 7.5 Updating while typing

- 150-250 ms debounce. The row is normalised, parsed and classified off the GL thread with a
  generation counter. Clean: one patch (`SetParam source` + whatever the kind implies) on a
  fork for the preview, committed to the real engine when the person stops or leaves the row.
- Not clean: the picture keeps the last good version of that row, and the message appears
  under the row in plain words (`FunctionError` text is already written for people: "unknown
  name 'q' ... use ..."). No dialog, no traceback, no lost rows.
- Changing a row's kind by editing it (`y = x^2` -> `x' = y`) replaces the object: remove
  and add in **one** patch so undo is one step. The old id is retired; ids must not be
  derived from the text.

### 7.6 Cheap and expensive

| Cheap (v1) | Medium | Expensive / out |
| --- | --- | --- |
| graphs, parametric curves and surfaces, vector fields, ODEs, points, variables and sliders, colour, hide, delete, reorder, slider play, free 2D pan/zoom and 3D orbit | implicit curves in 2D, heat maps of f(x, y), inequalities in 2D as shaded regions, piecewise (`{x<0: ..}` needs a parser extension), user-defined functions `f(x) = ...` used in other rows (a dependency graph with cycle detection) | implicit surfaces (marching cubes), lists and tables, sums/integrals/derivative notation, regression (out of scope), 3D inequalities |

User-defined functions are worth doing in the second pass: they make `V(x,y) = ...` then
`x' = -dV/dx` natural, but they need a row dependency DAG (a cycle is a row error, as params'
cycles are validation errors in contract 3).

---

## 8. An AI that understands the explorer

### 8.1 Principle

The AI reads the **same document** through the same `Workspace`. Nothing explorer-specific is
hidden in the GUI: rows keep their source text, analysis is in the objects' semantics, and
the tools are `Workspace` methods (so they are tested without the protocol, and serve both
`phasor mcp --attach` and the app's AI tab). Tools return **numbers with their provenance**
(window, grid, dt, T, tolerance, converged or not), never adjectives.

### 8.2 Semantics each object records (shows up in `inspect` today, no new plumbing)

- `Variable`: name, value, min, max, which rows use it.
- `System`: dimension; each component as plain text and LaTeX; variables used; divergence
  as a formula and its sampled min/max/sign over the window ("negative everywhere sampled":
  volumes contract); symmetries tested from a short fixed list of sign flips and swaps
  (Lorenz: invariant under (x, y, z) -> (-x, -y, z)); linear or not; autonomous or not;
  `tau`, `rate`, `dt`.
- `Equilibria`: window, grid, count, `non_isolated` flag; per point: position, eigenvalues
  (re, im), eigenvectors, kind, dim E^s / E^u / E^c, trace, det, hyperbolic or not with the
  tolerance used.
- `Tracer`: x0, tau, current position, speed, escaped (and when), path bounding box,
  closest equilibrium and distance, a period estimate if it returns.
- `FlowSet`: shape, count, alive / escaped counts, bounding box now, boundary length or
  area now versus at 0, `under_resolved` since when.
- `Manifold`: which equilibrium, stable or unstable, arc length, where it ends (another
  equilibrium within tolerance: a connection; the window edge; the step cap).
- Inert rows: the text and the parse error.

### 8.3 `explorer_state` (one call, the AI's first)

A composition of the above, not new analysis, shaped like this:

```
{"space": "R3", "t": 4.2, "playing": false,
 "system": {"plain": ["x' = sigma*(y - x)", ...], "latex": "...",
            "variables": {"sigma": {"value": 10, "min": 0, "max": 30}, ...},
            "divergence": {"formula": "-sigma - 1 - beta", "min": -13.67, "max": -13.67},
            "symmetry": ["(x,y,z)->(-x,-y,z)"]},
 "rows": [{"id": "...", "kind": "ode", "text": "...", "visible": true, "error": null}, ...],
 "equilibria": {"window": [...], "grid": 9, "points": [{"at": [8.443, 8.443, 26.7],
                "kind": "saddle-focus", "dims": {"s": 1, "u": 2, "c": 0},
                "eigenvalues": [[-13.840, 0], [0.0852, 10.149], [0.0852, -10.149]]}, ...]},
 "behaviour": {"probes": 16, "T": 50, "escaped": 0, "bounded_box": [...],
               "largest_lyapunov": {"value": 0.90, "spread": 0.04, "T": 200,
                                    "converged": true}},
 "camera": {...}, "lint": [...], "notes": ["equilibria searched only inside the window"]}
```

`behaviour` is computed on demand on a fork with a fixed, small budget and cached by the
system's key; the cheap parts are always present.

### 8.4 Tools (six; all `Workspace` methods, added to `TOOL_NAMES`, `MCP.md`, `llms.txt`)

| Tool | Does | Returns |
| --- | --- | --- |
| `explorer_state()` | the summary above | data |
| `explorer_edit(rows)` | add / set text / remove / show / hide / recolour rows and set variables; text goes through the same normaliser and classifier as typing | the changed rows with ids, kinds, errors, lint delta |
| `find_equilibria(window?, grid?, variables?)` | equilibria and classification, optionally for other variable values, on a fork (nothing changes) | points with eigen-data, the window and grid used |
| `flow_probe(x0, T, samples<=64, variables?)` | integrate one or a few seeds on a fork | sampled positions, escaped/when, bounding box, final speed, period estimate, `dt` check (did `dt/2` agree) |
| `sweep_parameter(name, lo, hi, n<=64, measure)` | for each value on a fork: equilibria count and kinds, max real part of eigenvalues at each, optionally largest Lyapunov exponent or attractor extent | the table, plus **candidate bifurcations**: values where the count changes or a real part changes sign, refined by bisection, each labelled by the local test it passed (saddle-node / pitchfork: count change; Hopf: complex pair crossing) |
| `explorer_preset(name)` | load a preset | as `explorer_edit` |

Not separate tools: `classify` (it is in `find_equilibria`), explaining an error (the error
text is in `explorer_state.rows`), pictures (`storyboard` and `review` exist), reading the
document as code (`core/pycode.py` through the existing scene tools). The existing
`inspect`, `lint`, `hit_test`, `resolve`, `undo`, `redo`, `render` work unchanged.

Checks to wire in: for Lorenz (10, 27.7, 2.67), `find_equilibria` returns the three points
and the eigenvalues in the table at the top; `sweep_parameter("rho", 20, 28)` reports a Hopf
candidate at 24.74 (= sigma(sigma+beta+3)/(sigma-beta-1) with beta = 8/3; recompute for 2.67).

### 8.5 How AI edits appear

- The MCP server attached to the app (`workspace/remote.py`) and the AI tab call the same
  `Workspace` under the same lock; `Editor.version` changes; the window's `tick` refreshes
  the rows and the picture. The `ActivityDock` already lists what was done.
- One undo history. An AI edit is one entry, named with its origin ("AI: rho = 24.74").
  The person's Ctrl+Z undoes it like any other.
- Analysis tools run on forks and never appear in the history or the picture. A tool that
  wants to *show* something (mark the Hopf value) must do it with `explorer_edit`.
- A row the person is typing in is locked to them: `explorer_edit` on it is refused with the
  row's version, and the AI is told to retry.

### 8.6 What an AI cannot do reliably, and the guard for each

| Limit | Guard |
| --- | --- |
| Judging whether a picture is pretty or "looks chaotic" | tools give numbers (exponent with spread, escaped counts, box); the docs tell the model to quote them and to say "the estimate did not converge" when `converged` is false |
| Stating where a chaotic trajectory *is* at a late time | `flow_probe` flags `chaotic_beyond: tau*` (from the exponent) and returns statistics, not positions, after it |
| "There are no other equilibria" | every answer carries the window and grid; the docs say to state them |
| Global bifurcations (homoclinic, e.g. Lorenz near rho 13.93; crises) | `sweep_parameter` labels candidates by the local test that found them and says global events are not detected; a Lyapunov or extent sweep can hint, labelled as a hint |
| Classifying near-degenerate points | `non-hyperbolic` within a stated tolerance; never a guess between centre and weak focus (a nonlinear centre cannot be decided from the linear part) |
| Seeing motion | `storyboard` over tau gives stills; `Tracer` semantics give speed and period |
| Trusting its own fix of an expression | `explorer_edit` returns the parse result and the lint delta; the skill says to read them before answering |
| Stiff or badly scaled systems | the `dt` check in `flow_probe` and lint `flow_too_fast` |

Docs to write in the same commits: an "Explorer" section in `docs/ai/API.md` (types of
2.2), the six tools in `MCP.md` with runnable `json calls` blocks, `llms.txt` regenerated,
and a short "helping in the explorer" part of the phasor skill: call `explorer_state` first;
answer with numbers and where they came from; change things only with `explorer_edit`; say
what was not checked.
