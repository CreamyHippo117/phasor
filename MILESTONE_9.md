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
| The rest of the module tables: PDE boundary conditions (held / insulated, drawn), spectral 1D heat, `Modes`, `Heat3D` + `Slice`; `CayleyGraph`, cosets and quotients on `CayleyTable`, `Homomorphism`, `SubgroupLattice`, `GaloisDiagram`, `Permutation`; `Limit`, `LoopIntegral` (flux, Green) | `domains/pde*.py`, `domains/algebra*.py`, `domains/calc*.py` |
| One stroke helper for arrows (moved from `linalg` to `strokes.arrow`) | `domains/strokes.py` |
| Docs, examples | `docs/ai/` (API, MCP, PITFALLS), `examples/heat_dihedral.py`, `examples/deep_zoom.py`, `examples/more_wave2.py` |

## Verification

- `uv run pytest`: **673 passed, 20 skipped** (all need CUDA; after 9b, the rest of the tables and the 7-9 boundary review); ruff, ruff format, lint-imports clean.
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
- **The rest of the tables**: spectral heat equals the exact modes to 1e-12 at a step 30x
  the explicit limit; insulated ends keep the heat (explicit and spectral) and match cos(pi x);
  mode decomposition recovers the modes and its partial sums converge; Heat3D decays at the
  exact rate; Cayley graphs have one edge per element and generator and refuse generators that
  do not generate; D4/<r> and D4/Z(D4) are groups (order 2, and the Klein group) and a
  non-normal subgroup is refused; clock homomorphisms obey |G| = |ker| |im|; the D4 lattice has
  10 subgroups with covering edges only; the Galois correspondence holds ([L : F] = |H|, H fixes
  F) and SymPy agrees on the degree 8 of Q(sqrt 2, sqrt 3, sqrt 5); permutations' cycles, sign,
  order, and tokens land on their target slots; limits are found symbolically and refused where
  the sides disagree; Green's and the divergence theorem hold around circles for four fields;
  `examples/more_wave2.py` lints and critiques clean, with a software golden.
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
6. Each module table is now covered except: GPU PDE solvers, spectral solvers beyond 1D heat,
   volume rendering (3D PDEs show as slices), Stokes on 3D surfaces, rings beyond clocks, field
   towers other than square roots of primes, cube/icosahedron symmetry groups, group orbits as
   pictures (SPEC "As built (Milestone 9)").
7. `steady` on `cx.Fractal`, `Slice.scale = 0` (auto) and `core.color.ramp` are additions the SPEC did
   not name; each fixes something seen in a render (see Bugs).

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
6. **PDE heat maps drew their hottest spot dark** (found by looking at the 3D slice): `ember`,
   `ocean` and `sequential` are cyclic fractal palettes that end dark, and `Field`, `Surface`
   (since the first Milestone 9 commit) indexed all of them. They now use the rising half
   (`core.color.ramp`); a test pins it. The Milestone 8 `Spectrogram` had the same bug (its loudest bins dark) and is fixed with it.
7. The first Galois and lattice drawings overlapped their labels and lacked glyphs for the
   square root and sigma (the font has neither): labels are ASCII (`sqrt2`, `sig(2)`) on screen,
   Unicode in the semantics, and crowded levels put their outer labels sideways.

## Untested

- Windows, NVIDIA GPU. Large 2D grids in NumPy are slow (default 48x48 is fast).
- `Surface` and `VectorField` only through the inspector and one still, not goldens.
- `Homomorphism` with many elements is a tangle of arrows (straight lines cross the source
  ring); fine for teaching sizes (n <= 12), not beautiful beyond.

## Open

- **Boundary reviews of Milestones 7, 8 and 9** (strongest model).

## Boundary review (Milestones 7-9, 2026-11-19)

Done in this session by the model that wrote most of the code, not by an independent reviewer
(CLAUDE.md asks for subagents only on request): treat it as weaker evidence than the M2-M6
review. Audited the seven contracts over everything since `bd7dc40` (51 files, ~7,600 lines),
plus hostile input to the new expression parser. Contracts 1 (one write path: only
`Engine._commit` replaces the document; `polish` and `variants` go through `apply`), 2 (schema
v5 migration tested; new params have defaults), 5 (contract test covers every new type),
6 (lint-imports clean; `ramp` moved down to `core.color` so two domains share it) and 7
(kiwisolver, the label placer and the camera solver are imported only inside `layout`) held.
Found and fixed, each with a test that fails without the fix:

1. **Contract 3, across sessions (M8).** The simulation disk cache was keyed by parameters
   only, so after a solver bug fix `.phasor_cache` kept serving the old code's answers. The
   key now hashes the solver module's source too.
2. **Living motion moved meaningful geometry (M7, exposed by M8/M9).** Life guessed "a dot"
   from parameters named `pos` and `r`, so with `life` on, polygons, clock diagrams, Cayley
   graphs, homomorphisms and acoustic sources/listeners drifted, and a source's computed
   arrivals jittered. Types now declare what life may do (`ObjectBase.Life`, glow only by
   default; `Dot` wanders, `DomainColoring` and `Fractal` turn hue).
3. **Motion blur missed expressions (M7).** `is_moving` looked only at tracks, so anything
   driven by `expr("t ...")` (PDE time, permutations, most domain scenes) was never blurred.
4. **Motion blur averaged after the tone map (M7).** Still one tone map per frame, but bright
   moving glow was averaged after being squashed; the samples' light is now averaged first and
   finished once.
5. **Cyclic colormaps drew the hottest values dark (M8 spectrogram, M9 PDE heat maps)**, found
   while building `Slice` (see Bugs 6).

Checked and fine: no randomness or clock outside `rng`; deep zoom caches are pure functions;
the parser refuses attribute access, calls outside the whitelist, lambdas, subscripts and
overflowing constants.

Open from the review (not fixed):
- SymPy calls (`calc.Limit`, Taylor, divergence) have no time limit: a pathological formula
  sent over MCP could stall the server.
- Motion blur counts a finished expression (`min(t / 3, 1)` after t = 3) as moving: correct
  pictures, wasted samples.
- Milestone 10's exit demo names the reverb scene, which was skipped in Milestone 8.

