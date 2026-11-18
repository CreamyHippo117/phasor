# Milestone 9: Domains wave 2

Roadmap row: `pde`, `algebra`, `calc` (listed with `calc` in the module table), perturbation deep
zoom. Exit demo: heat equation, dihedral group and a 1e-30 zoom. Model: Sonnet, strong for
perturbation.

Status: **partly built, by decision: everything except the perturbation deep zoom.** The owner
asked for what could be done without the stronger model; the 1e-30 zoom (a reference orbit in
arbitrary precision, per-pixel deltas, glitch detection and rebasing) is **Milestone 9b** and is
not started. The heat-equation and dihedral-group parts of the exit demo are built and tested.
Gates green on Linux (software OpenGL). Untested on Windows and a GPU.

## Built

| Area | Where |
| --- | --- |
| PDE solvers, each a `Simulation` with a validated step: heat 1D/2D, wave 1D/2D, advection, Gray-Scott, Schrodinger (split-step), Poisson/Laplace; exact solutions for comparison | `domains/pde_math.py` |
| PDE objects: `Profile`, `Field` (backdrop), `Surface` (lit 3D) | `domains/pde.py` (`from phasor import pde`) |
| Finite groups: dihedral, cyclic, symmetric; subgroups, cosets, normality, group actions | `domains/algebra_math.py` |
| Algebra objects: `Polygon` (a group acting on an n-gon), `CayleyTable`, `ClockDiagram` | `domains/algebra.py` |
| Calculus numerics: Riemann sums (6 rules), Taylor, epsilon-delta, vector fields, divergence, curl, circulation | `domains/calc_math.py` |
| Calculus objects: `Graph`, `RiemannSum`, `Taylor`, `VectorField` | `domains/calc.py` |
| A safe expression parser in named variables (`x`, `y`) beside the complex one | `render/gpu/codegen.parse_expression` |
| Docs, examples | `docs/ai/` (API, MCP, PITFALLS), `examples/heat_dihedral.py` |

## Verification

- `uv run pytest`: **625 passed, 17 skipped** (all need CUDA); ruff, ruff format, lint-imports clean.
- **Heat equation (exit demo)**: the 1D solver agrees with the exact modes to 3e-3 and decays at
  exp(-k (m pi / L)^2 t); 2D single mode likewise; a hot spot only cools (maximum principle);
  an unstable step is refused; through the engine the plotted numbers stay on the exact
  solution at every time, heat only leaves, scrubbing is pure.
- **Dihedral group (exit demo)**: D3..D8 satisfy the group axioms and have the right orders;
  the Cayley table is a Latin square; D4 has 10 subgroups and Lagrange holds; the polygon's
  vertices equal the 2x2 matrices of the group elements for every element (and the matrices
  multiply like the table); through the engine a square turned a quarter then flipped lands on
  `s r`, and a further half turn on `s r2`, as the table says.
- Other solvers against theory: standing wave frequency, drum modes, plucked string period,
  advection speed and mass, Schrodinger norm (unitary) and free-packet spreading, a harmonic
  well's period, Poisson to second order, Gray-Scott determinism by seed.
- Calculus: Riemann error orders (1 and 2), lower/upper sums squeeze the integral, Taylor
  coefficients and Lagrange's remainder bound, epsilon-delta, Green's theorem.
- Looked at every object rendered and at the heat/dihedral storyboard.

## Deviations from SPEC (SPEC updated in the same commit)

1. The perturbation deep zoom is not built (decision; Milestone 9b).
2. NumPy solvers only (no GPU solvers, no spectral PDE solvers); 2D fields and 3D surfaces,
   no volumes or slices of 3D PDEs.
3. A reasonable subset of each module table (see SPEC "As built (Milestone 9)").

## Bugs caught while building

1. The first leapfrog wave solver started one step late (its state at n was time n+1), a phase
   error of a few percent that the standing-wave test caught; the zero-velocity start is now
   symmetric in time (u(-dt) = u(dt)).
2. `Group.is_normal` and friends were fine, but my first Cayley label for `s r` was `sr1`;
   labels now read `e r r2 r3 s sr sr2 sr3`.
3. The Cayley table ran off the frame in the first draft of the example (found by looking).

## Untested

- Windows, NVIDIA GPU. Large 2D grids in NumPy are slow (default 48x48 is fast).
- `Surface` and `VectorField` only through the inspector and one still, not goldens.

## Open

- **Milestone 9b: perturbation deep zoom** (strongest model).
- **Boundary reviews of Milestones 7, 8 and 9** (strongest model).
