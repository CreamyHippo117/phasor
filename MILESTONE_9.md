# Milestone 9: Domains wave 2

Roadmap row: `pde`, `algebra`, `calc` (listed with `calc` in the module table), perturbation deep
zoom. Exit demo: heat equation, dihedral group and a 1e-30 zoom. Model: Sonnet, strong for
perturbation.

Status: **built, including the perturbation deep zoom (Milestone 9b, on the strongest model).**
All three parts of the exit demo (heat equation, dihedral group, a 1e-30 zoom) are built and
tested. Gates green on Linux (software OpenGL). Untested on Windows and a GPU.

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
| Deep zoom (9b): reference orbit in exact fixed point, perturbation kernel with rebasing (CUDA C + NumPy reference), `cx.Fractal` `center` / `zoom` / `steady` | `domains/fractal_math.py`, `render/gpu/kernels/fractal.py` (`deep`), `domains/cx.py` |
| Docs, examples | `docs/ai/` (API, MCP, PITFALLS), `examples/heat_dihedral.py`, `examples/deep_zoom.py` |

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
- **1e-30 zoom (exit demo, 9b)**: every pixel of a 1e-30 view (reference at and off the view
  center) agrees to 1e-3 of an escape count with exact iteration at 64 more bits; float64
  alone renders that view as one color; a reference that escapes after 19 steps still gives
  exact answers (pixels rebase onto its start); just above the hand-over both float64 and
  perturbation agree with exact iteration; the CUDA source matches the NumPy reference bit for
  bit when compiled for the CPU; through the engine a 1e-30 frame is the kernels' picture; the
  example dives f32 -> f64 -> perturbation once each, only ever deeper, and lints clean; a
  software golden storyboard.
- Other solvers against theory: standing wave frequency, drum modes, plucked string period,
  advection speed and mass, Schrodinger norm (unitary) and free-packet spreading, a harmonic
  well's period, Poisson to second order, Gray-Scott determinism by seed.
- Calculus: Riemann error orders (1 and 2), lower/upper sums squeeze the integral, Taylor
  coefficients and Lagrange's remainder bound, epsilon-delta, Green's theorem.
- Looked at every object rendered and at the heat/dihedral storyboard.

## Deviations from SPEC (SPEC updated in the same commit)

1. Deep zoom (9b): perturbation starts at 1e-10 per pixel, not "about 1e-13": measured against
   exact iteration on the demo spiral, plain float64 already gets ~1% of pixels visibly wrong
   (up to 30 escape counts) at 1e-10, perturbation none.
2. Deep zoom: arbitrary precision is fixed point on Python's integers, not `gmpy2` (exact,
   identical everywhere, no new dependency; a 20 000-step orbit at 200 bits takes ~30 ms).
3. Deep zoom: glitches are handled by rebasing (Zhuoran: when |z| < |dz|, or the reference
   runs out, continue from the reference's start), not by Pauldelbrot detection plus a second
   reference. Mandelbrot only; no series approximation; float64 deltas limit zoom to 1e280.
4. Deep zoom: `steady` (new) keeps the outer color fixed, because the escape count grows with
   depth and a dive's background otherwise cycles light/dark about once a second.
5. NumPy solvers only (no GPU solvers, no spectral PDE solvers); 2D fields and 3D surfaces,
   no volumes or slices of 3D PDEs.
6. A reasonable subset of each module table (see SPEC "As built (Milestone 9)").

## Bugs caught while building

1. The first leapfrog wave solver started one step late (its state at n was time n+1), a phase
   error of a few percent that the standing-wave test caught; the zero-velocity start is now
   symmetric in time (u(-dt) = u(dt)).
2. `Group.is_normal` and friends were fine, but my first Cayley label for `s r` was `sr1`;
   labels now read `e r r2 r3 s sr sr2 sr3`.
3. The Cayley table ran off the frame in the first draft of the example (found by looking).
4. (9b) Near a minibrot at 1e-14 and 5000+ steps, ~1% of edge pixels differ from exact
   iteration even with rebasing. Traced: with an exact reference orbit both methods are exact;
   the error is the reference rounded to float64, which every perturbation renderer shares.
   Rebasing halves it (2.4% -> 1%); the test pins both numbers.
5. (9b) First draft of the dive flashed light/dark (colors indexed by a growing count): `steady`.

## Untested

- Windows, NVIDIA GPU. Large 2D grids in NumPy are slow (default 48x48 is fast).
- `Surface` and `VectorField` only through the inspector and one still, not goldens.

## Open

- **Boundary reviews of Milestones 7, 8 and 9** (strongest model).
