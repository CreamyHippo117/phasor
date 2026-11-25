# Milestone 8: Domains wave 1

Roadmap row: `fourier`, `linalg`, `acoustics`, `Simulation` checkpoints. Exit demo: the 3D
reverb scene. **Decision of the owner (2026-11-18): skip the scene, not the test.** The
acceptance test is `tests/test_reverb.py` (plus `tests/test_acoustics.py` for the physics).

Status: **built; gates green on Linux (software OpenGL).** Untested on Windows and a GPU.
**Boundary review not run** (CLAUDE.md asks for the strongest model; see Open).

## Built

| Area | Where |
| --- | --- |
| `Simulation`: fixed dt, seed, checkpoints (memory + `.phasor_cache/`), pure `state_at` | `core/simulation.py` |
| Room acoustics numerics: image sources, arrivals, impulse response, Schroeder RT60, Sabine and Eyring, FFT convolution, 3D FDTD with the stability limit | `domains/ac_math.py` |
| Acoustics objects: `Room`, `Source`, `Listener`, `Wavefront`, `ImageSources`, `ImpulseResponse`, `PressureSlice` | `domains/acoustics.py` (`from phasor import ac`) |
| Fourier numerics and objects: `Epicycles`, `PartialSum`, `Spectrum`, `Spectrogram` | `domains/fourier_math.py`, `domains/fourier.py` |
| Linear algebra numerics and objects: `TransformGrid` (2D and 3D), `EigenLines`, `Arrow`, `Matrix` | `domains/linalg_math.py`, `domains/linalg.py` |
| Shared stroke drawing for domains (curves, discs, rings); `Line` now uses it | `domains/strokes.py` |
| Layout relations `outside`, `tangent_to`, `attach_to ... at <spot>` (moved here from M6) | `core/constraints.py`, `layout/solver.py`, `layout/bake.py` |
| Docs: `docs/ai/API.md` (domains, simulations, relations), MCP.md (a tested patch), PITFALLS | `docs/ai/` |
| Examples | `examples/fourier_linalg.py`, `examples/room_acoustics.py` |

## Verification

- `uv run pytest`: **584 passed, 17 skipped** (all need CUDA); ruff, ruff format, lint-imports clean.
- **Exit test (`tests/test_reverb.py`)**: the source and the listener are not hidden by the
  translucent walls (real pixels) and a solid wall *would* hide them; the wavefront reaches
  the listener exactly when the direct sound does; the room reports volume, area and RT60;
  absorbent walls give a shorter fitted tail and plaster's is flagged unreliable; the
  pressure field is a cached pure simulation (same time, same field, scrubbing back gives the
  same answer, checkpoints on disk); the camera event log has no problems for the source and
  the listener.
- **Physics (`tests/test_acoustics.py`)**: image-source geometry (six mirrored ghosts, counts,
  gains), the direct arrival at d/c, echo density growing, RT60 inside the theory bracket and
  the Schroeder curve checked against a direct energy sum, convolution theorem, FDTD wave speed
  (a pressure peak 3 m further arrives 8.7 ms later), the CFL limit (an exactly-at-the-limit dt
  works, one above raises), boundedness, determinism and the checkpoint cache.
- **Maths (`tests/test_fourier.py`, `tests/test_linalg.py`)**: Gibbs overshoot tends to 0.0895
  of the jump and its position follows pi/(2n); epicycle chain ends on the curve; leakage and
  windows; spectrogram follows a chirp; determinant as area; eigenvectors satisfy Av = lambda v;
  SVD stages; rank, null space, Gram-Schmidt.
- Looked at the rendered room (walls, ghosts, wavefront, pressure slice, impulse-response
  panel with the RT60 fit) and the Fourier/linalg tour.

## Deviations from SPEC (SPEC updated in the same commit)

1. The reverb scene itself is not built (owner's decision); the exit demo is the test.
2. FDTD runs in NumPy, not on the GPU; its absorbing walls are a damping shell (an
   approximation). The SPEC's "rendered as a glowing isosurface or slice": slice only.
3. `tangent_to` is box tangency, not true curve tangency.
4. Rays beyond the image-source method (stochastic ray tracing) are not built.
5. A shoebox is not a diffuse field: the fitted RT60 rings longer than Eyring's; the tests
   check a theory bracket, and the object reports `rt60_reliable` (false when the decay runs
   past what `order` reflections describe).
6. Not built: Laplace/z-transforms, spherical harmonics, Bessel modes, null/column-space and
   change-of-basis pictures, an interactive slider, convolution-reverb audio export.

## Bugs caught while building

1. My first RT60 test asserted the fit equals Eyring; it does not (non-diffuse field), and
   the first IR was truncated so early the fit looked shorter. The test now brackets by
   theory and checks the Schroeder integral against a direct sum in the range the arrivals
   fully cover.
2. A `Wavefront` radius of `343 * (t - emit)` is negative before the emission, and the
   parameter's `ge=0` rejected the whole scene at t = 0; a negative radius now means "not
   emitted yet".
3. The impulse-response plot grew downward on the overlay (y runs down there); plots now grow
   upward on both layers.
4. A layout rule that could not run (an unknown `attach_to` spot) left a solved object with
   no key at that time, so reading the bake crashed; every solved object now records a key
   even when its rule fails.
5. `tangent_to` inherited the theme's label gap, so objects stood off instead of touching.
6. A FDTD checkpoint interval of 100 steps meant short runs cached nothing; 50.

## Untested

- Windows, NVIDIA GPU.
- Large FDTD grids (the 40x25x15 default is fast in NumPy; a fine grid is slow).
- `Spectrogram` and `ImageSources` of high order only through the inspector, not goldens.

## Open

- **Boundary review of Milestones 7 and 8** on the strongest model.
- A storyboard golden for the domain objects (the existing goldens are unchanged).
