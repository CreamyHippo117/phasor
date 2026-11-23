# Brief for ChatGPT / Codex: four animations for phasor

You are joining a Python project called **phasor** (a maths and signal-processing animation
engine). You have no memory of it, so everything you need is named here. Do the four items below,
in order, each on its own branch. Do not touch anything else.

## Where to go

- Repo: `CreamyHippo117/phasor` on GitHub.
- Start every branch from `claude/charming-einstein-hehoar` (not `main`).
- Read, in this order: `CLAUDE.md` (the rules, binding), `ANIMATIONS.md` (the rows for your four
  items), `BEAUTY.md` (the prism look), then the finished example below.
- Setup: `uv sync --extra gl --extra encode`. Headless Linux needs `libegl1 libgl1-mesa-dri`.
  No GPU is needed for these four items (plain NumPy).

## Copy this finished item

`dyn.PhasePortrait` / `dyn.FixedPoints` is done and is your template for structure:

| What | File |
|---|---|
| Maths (pure NumPy, no drawing) | `phasor/domains/dynamics_math.py` |
| Object (params, bounds, features, semantics, draw) | `phasor/domains/dynamics.py` |
| Strokes helper used for drawing | `phasor/domains/strokes.py` |
| Tests | `tests/test_dynamics.py` |
| Example scene | `examples/phase_portrait.py` |
| Golden still | `tests/golden/software/phase_portrait.png` |

## The four items (do in this order)

Branch names: `chatgpt/s2-winding`, `chatgpt/w3-directivity`, `chatgpt/w4-doppler`,
`chatgpt/d4-attractors`.

### 1. S2 Fourier transform, the winding picture
- **What:** a signal wound round a circle at a test frequency; its centre of mass is marked;
  sweeping the test frequency traces the spectrum, which spikes where the winding balances.
- **Maths:** `phasor/domains/fourier_math.py` (extend; grep first, reuse `Spectrum` maths).
  Object in `phasor/domains/fourier.py`. Params include the signal (formula or sum of tones),
  `freq` (the animated test frequency), `progress`, `pos`, `size`.
- **Tests must show:** a pure tone at f0 gives centre of mass about 0 away from f0 and a spike
  at f0; the centre of mass equals the DFT at that frequency; determinism.
- **Prism version:** the winding curve is a glowing stroke with hue running along it; the centre
  of mass is a bright beacon; the spectrum is drawn beside it as light, built up as `freq` sweeps.

### 2. W3 Directivity
- **What:** polar patterns: an array's beam pattern, a piston, a cardioid.
- **Maths:** `phasor/domains/ac_math.py`. Object in `phasor/domains/acoustics.py`. Params:
  `kind` (`array`, `piston`, `cardioid`), element count, spacing, frequency, steer angle, `pos`, `size`.
- **Tests must show:** the array's main lobe peaks at the steer angle within a beamwidth; a
  cardioid is 1 at 0 degrees and 0 at 180; a piston's first null is where theory puts it (J1 zero).
- **Prism version:** a glowing polar curve; the main lobe brightest, side lobes fainter; hue by
  angle only if it carries meaning, otherwise one palette hue.

### 3. W4 Doppler
- **What:** a moving source leaves bunched and stretched wavefronts; at sound speed, the shock cone.
- **Maths:** `phasor/domains/ac_math.py` (pure NumPy: source position at emission time, ring
  centre and radius at time t, observed frequency). Object in `phasor/domains/acoustics.py`.
  Params: `speed` (source Mach number), `frequency`, `time`, `pos`, `size`.
- **Tests must show:** observed frequency ahead is f/(1-M), behind f/(1+M); rings are tangent to
  a cone of half-angle asin(1/M) when M > 1; determinism.
- **Prism version:** wavefront circles as fading glowing rings; hue by frequency heard, so the
  bunched front reads warm and the stretched rear cool.

### 4. D4 Strange attractors
- **What:** Lorenz (and Rossler, Thomas) traced as a moving 3D curve.
- **Maths:** `phasor/domains/dynamics_math.py`. A time-stepping model must be a `Simulation`
  (`phasor/core/simulation.py`): fixed dt, seed in `params()`, pure `step`, read only through
  `SimRunner.state_at(t)`. Object in `phasor/domains/dynamics.py`. Params: `system`, parameters
  (sigma, rho, beta), `seed`, `progress`, camera orbit.
- **Tests must show:** Lorenz stays bounded; two nearby starts diverge exponentially (positive
  Lyapunov estimate); same seed gives identical output; the Thomas system conserves nothing but
  stays bounded.
- **Prism version:** a glowing 3D curve with hue running along time, the camera slowly orbiting.
  (The dense point-cloud version is a separate PC task: do not build it.)

## Rules that matter most (the full list is in CLAUDE.md; CLAUDE.md wins)

- Maths in `domains/<name>_math.py` (pure NumPy), objects in `domains/<name>.py`, strokes via
  `domains/strokes.py`. Extend existing files; grep before adding any helper.
- Every object implements `Params`, `world_bounds`, `features`, `semantics`, `draw(ctx, state,
  pass_)`. Missing one fails registration.
- Tests first. Randomness only via `rng(object_id, seed)`. `state_at(t)` is pure.
- New scenes use `look="prism"`. Taste values (glow, dim, hue) come from the theme and looks in
  `style/`, never numbers typed into a scene. Keep `prism_primes` looking exactly as it does.
- For each item: a `docs/ai/API.md` entry (its python blocks run in CI), an example in
  `examples/`, a golden storyboard (`PHASOR_UPDATE_GOLDEN=1 uv run pytest -k golden`, then look at
  the picture before committing), and a showcase entry. Add the tool-facing names to
  `llms.txt` with `uv run python -m phasor.mcp.llms`.
- Text files are read and written with `encoding="utf-8"`.
- Do not claim anything works on Windows or CUDA: say "untested on X".

## Done means (per item)

```bash
uv run ruff check . && uv run ruff format --check .
uv run lint-imports
uv run pytest -q
```

all pass. Commit with a clear message, push the branch, and report: branch name, files changed,
test output, and the golden picture. Open a **draft** pull request into
`claude/charming-einstein-hehoar`. Do not merge. If `CLAUDE.md` and this brief disagree, stop and
ask.

## Do not

- Do not edit `SPEC.md` or the seven contracts in `CLAUDE.md`.
- Do not work on D5, S1 or S3 (another agent has them) or any `pc_tasks/` item.
- Do not add dependencies.
