# 09. Render speed: measurements only the RTX card can give

Branch with the changes: see the PR for `GPU_SPEED.md` step 7 and 8. This session had no
GPU, so every number below is **untested on CUDA**. Run these *before* (commit `4643d37`)
and *after* (the PR head), serially, with nothing else running, and write the numbers into
`GPU_SPEED.md`'s Status table. No code changes are needed for this task.

```bash
git worktree add ..\phasor-speed -b perf/measure 4643d37     # before
cd ..\phasor-speed ; uv sync --extra gl --extra encode --extra gpu
```

## 1. Frame bench (steady state, per stage)

```bash
uv run phasor bench --frames prism_primes --count 60 --preset final
uv run phasor bench --frames motion_gallery --count 60 --preset final
uv run phasor bench --frames residue_theorem --count 60 --preset final
```

## 2. Wall clock of whole renders (this is where startup shows)

Use PowerShell's `Measure-Command { ... }`. Short scene = `moving_dot` (12 s), long scene =
`motion_gallery`. Each of preview, final; each with `--workers 1`, `2`, `4`:

```bash
uv run phasor render examples/moving_dot.py --preset preview --workers 1 --out $env:TEMP\a.mp4
uv run phasor render examples/motion_gallery.py --preset final --workers 4 --out $env:TEMP\b.mp4
```

Also time `uv run phasor doctor` (each render used to run it again: NVRTC compile checks
may take seconds on CUDA).

## 3. Scene load time

```bash
uv run python -c "import time; from phasor import scene; t=time.perf_counter(); scene.load('examples/residue_theorem.py'); print(time.perf_counter()-t)"
```
(same for `prism_primes`, `motion_gallery`). Linux llvmpipe numbers: residue_theorem 17.8 s
before, 16.2 s after; prism_primes 3.9 -> 3.2 s.

## 4. Finals must not change

`uv run python pc_tasks/pin_frames.py before.json prism_primes:6 motion_gallery:6 julia_explorer:4` at both commits, then compare the files (hashes must be identical); also the golden tests
(`uv run pytest -k golden`, `-m gpu`). Lit surfaces are not bit-repeatable on this card
(see `GPU_SPEED.md` "Next session"); leave them out of any bit comparison.

## 5. Draft preset (if merged)

`uv run phasor render examples/prism_domain.py --preset draft` vs `--preset preview`, with
`--workers 1`: wall clock, and look at the picture. Report whether the draft look is good
enough to judge timing and layout.
