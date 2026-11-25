# Explorer: a Desmos-like math workbench (Milestones 26-31)

Asked for by the owner on 2026-11-24, after seeing another person's "Math Explorer" (real-time
dynamical systems: vector fields in R2/R3, disks and balls of initial conditions advected,
the Lorenz attractor, equilibria with eigenvalue classification, stable/unstable manifolds,
LaTeX overlays, speed/pause/reverse): "i need to have this, so i can play around and not
worry about the code so much". Follow-up: it should work "like Desmos" (rotate around, add
equations on the fly) and "some way for AI to understand your work".

The full design review (Fable, with measurements on this box) is
`docs/decisions/explorer_advice_2026-11-24.md`. This file is the plan and the status. Where
the two differ, this file wins; both yield to CLAUDE.md.

## The decision

**The explorer is a view of the one engine, not a second program.** Rows are ordinary
document objects edited through `Engine.apply`; a dragged slider or an orbit works on a fork
and commits one patch on release. So undo, save, lint, the AI tools, the notebook/web export
and the video renderer all work on an explorer document, and "what you explore is what you
render". The cost is one renderer feature (world-space batches, Milestone 28).

Two kinds of state:

- **Document** (ops, undoable, saved): rows and their source text, variables, seeds, regions,
  overlays, the resting camera, the flow rate.
- **Session** (never an op): playhead, paused, speed, reverse, camera while the mouse is down,
  a slider mid-drag, text mid-typing.

## Decisions taken from the review (the owner can redirect)

- One flow core, `domains/flow_math.py`: an n-D (2 or 3) `System`; `dynamics_math.System`
  becomes its thin 2D front. Old `dyn.*` and `geo.*` objects and params keep working.
- Fixed-step RK4 only (a seed's path must not depend on its neighbours). A set of seeds is a
  `Simulation` (checkpoints). Reverse = playhead replay; backward flow is a separate bounded
  run with an escape radius (backward Lorenz is NaN).
- Budgets are clamped and linted (`flow_budget`), never validation errors.
- Notation goes through a normaliser (`^`, `2x`, primes, unicode) in front of the existing
  whitelist parser; the parser itself is hardened first (`9**9**9` hangs, long sums raise
  `RecursionError`).
- Names are whole runs of letters (`rho` is one name); products need a space, `*`, digit or bracket.
- Sliders are document `Variable` + `Control`; the Notebook and `export_web` read them already.
- Chaotic paths are reproducible on one machine only; goldens of chaotic systems use tau <= 5.
- **Owner decision (raised, defaulted):** explorer documents stay `.phasor.json` while explored
  (an exception to APP_PLAN's convert-on-open). "Convert to script" is one-way for structure.
- Out of v1: collisions, inertia/two-way coupling, global 2D manifolds, FTLE, basins, 3D
  implicit surfaces, adaptive solvers, a separate GL widget, discrete maps.

## Milestones

| # | What | Depends on | Where | Status |
| --- | --- | --- | --- | --- |
| 26 | Flow core, headless: parser hardening, `flow_math`, batched RK4 + `Simulation`, equilibria, 3D classification, 1D manifolds, set refinement, objects (`Variable`, `System`, `FieldArrows`, `Tracer`, `FlowSet`, `Equilibria`, `EquilibriumView`, `Manifold`), presets, docs, examples, goldens | none | cloud | DONE (independently audited, 4 blockers fixed; `MILESTONE_26.md`) |
| 27 | Rows, variables, AI tools: `normalize_row`, row classifier, `source` params, auto `Control`, `explorer_state/edit/preset`, `find_equilibria`, `flow_probe`, `sweep_parameter`, MCP.md, llms.txt, skill | 26 interfaces | cloud | DONE (independently audited, 2 blockers fixed; `MILESTONE_27.md`); the panel's typed-row lock is in place, rows for graphs/curves/surfaces are saved but not drawn yet |
| 28 | World-space batches `ctx.points3d/lines3d/arrows3d` (GL + software), flow objects moved onto them, thinning while dragging | M21 | cloud; PC timing | built, independently audited (1 blocker + 5 should-fix fixed; `MILESTONE_28.md`); PC timing in `pc_tasks/12_world_batches.md` |
| 29 | The Explorer panel in the M25 window: transport, click/drag tools, equilibrium card, R2/R3; also record marks -> scene and the export dialog (from 31) and the insert palette's first form (from 32) | 27, M25 | cloud; PC feel | built, not audited (`MILESTONE_29.md`); feel on the PC: `pc_tasks/14_explorer_feel.md` |
| 30 | Analysis pack: Poincare section, Lyapunov, time-series strip, twin tracers, divergence map, bifurcation sweep, limit cycles | 26, 27 | cloud | planned |
| 31 | Scale and handoff: CUDA flow kernel, 1e5-1e6 points, basins/FTLE kernels (record marks -> scene and the export dialog were built in 29) | 28, 29 | PC | planned |

Parallel plan: 26 (maths first, then objects) and 27's parser/normaliser run at once; after
26, 27's tools, 28 and 30 in parallel (they touch `workspace/`, `render/`, `domains/`).
Window work (29) waits for M25.

## Beyond vector fields: a scene workbench (owner, 2026-11-24)

The owner's follow-up: not only fields, also **objects**. Example: build the 3D room scene
with sound rays by dropping points, shapes (circle, rectangle, octagon, custom) and sources
into space, then ask the AI for "a graph of when rays from source A reach source B" or "a
circle of eight octagons". Code only, AI only, or a hybrid are all fine.

Decision: **hybrid, and it costs no extra architecture**, because the explorer is a view of
the one document. A row, a clicked-in shape, a line of code and an AI call are four ways to
make the same objects through `Engine.apply`. So:

- **Insert palette** (the `+ row` menu): every shipped type that has an example (points,
  `shapes`, `Room3D`, sound sources, flow objects, graphs, text) with drag handles; placing
  one is one patch. The palette is generated from the registry, not written by hand.
- **Arrangements** are an op the AI and the person share: `arrange(ids_or_template, pattern)`
  with `ring(n, r)`, `grid(r, c)`, `line`, `along(path)`, writing ordinary objects with
  derived ids (`oct.1` ... `oct.8`), so "a circle of eights" is one undoable patch. Needs a
  ring/grid pattern helper in `core` (grep first; `algebra.ring` and the Manim importer's
  `arrange` are the only precedents).
- **Measurements as tools**, numbers with provenance (like `find_equilibria`): `ray_arrivals(
  source, receiver, ...)` returns arrival times, counts and the path for the room's rays, and
  `graph_of(measure, over)` puts the answer on screen as a `Graph` row. Each is a
  `Workspace` method on a fork, tested without the protocol. The acoustics maths already
  exists in `domains/ac_math.py`; the tool only exposes it.
- **Code view** shows the same document as a script (`core/pycode.py`); edits there and in
  the panels meet in one history. A scene built by clicking can be converted to code once
  (one-way for structure, per APP_PLAN).

Milestones (after 29, so the panel and tools exist): **32 Insert palette and direct
manipulation of any object type**; **33 Arrangements and measurement tools (rays, arrival
graph first)**. Both cloud-testable; feel on the PC.

## Rules

CLAUDE.md's contracts and the STUDIO.md rules (bounded inputs, lint not crash, Windows CI has
no OpenGL, `passes=None` and GL modes, no timing asserts on the coarse Windows clock, utf-8).
Plus: every cache key lists every input (property test: change each param alone); batch
independence (seed k's path is the same alone or in 1000); scrub order equals a fresh engine;
playing for a minute leaves `Editor.version` and undo depth unchanged; row ids survive
delete-and-retype (`retired_ids`); parse every formula in `examples/`, `showcase/`, `gallery/`,
`docs/ai/` before and after parser changes.

## Status

Planned 2026-11-24. 26 flow core DONE (audited, merged); 27 rows and AI tools DONE (audited, merged); 30 analysis pack built and under audit; 28 (world-space batches) built and audited, awaiting the PC's numbers; 29 (the panel) built 2026-11-25 on Milestone 25's window, not audited yet; its feel is the owner's check (`pc_tasks/14_explorer_feel.md`).
