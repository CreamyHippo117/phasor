# Checks only the Windows RTX PC can do (after Milestones 14 and 15)

Written 2026-11-23 by the cloud session. Milestones 14 (CPU kernels) and 15 (open object
contract) are built and merged on `claude/eloquent-mayer-jiqxs9`, but the cloud box has no
CUDA, no Windows and no Mac, so everything below is **untested** until you run it. Do these in
order; each is small. Write the result (a number or pass/fail) under "Results" at the bottom
and commit this file.

Set up once (PowerShell):

```powershell
cd C:\Users\Atlas\phasor-dev
git fetch origin
git checkout claude/eloquent-mayer-jiqxs9
git pull
uv sync --extra gl --extra encode --extra gui --extra mcp --extra gpu
uv run phasor doctor
```

Record the tier, the card, and the new `cpu_kernels` line (compiler and thread count).

## 1. The CPU backend on Windows (written for, never run)

`MILESTONE_14.md` "Windows (written for, not run)". It needs a MinGW compiler:
`winget install BrechtSanders.WinLibs.POSIX.UCRT`, then open a new window.

```powershell
uv run phasor doctor                       # cpu_kernels should name g++ and the threads
uv run pytest tests/test_cpu_backend.py -q # 58 tests; all 43 kernel cases against NumPy
```

If a built library will not load (DLL search path), the backend warns once and falls back to
NumPy: note the warning text. Do **not** fix it blind; tell the session what you saw.

## 2. The kernels on the real GPU

```powershell
uv run pytest tests/test_kernels.py tests/test_gpu.py -q   # the gpu-marked tests run here
```

Every `CASES` entry should pass on CUDA, and match the CPU backend.

## 3. The Riemann tests, timed (the cloud box takes 800-1100 s each)

The cloud box has no GPU; its OpenGL is software, and a profile put the time in GL depth
readback and NumPy, not in kernels. This is not confirmed. On the card:

```powershell
uv run pytest tests -k riemann --durations=10 -q
uv run python tests/diagnose_riemann.py      # TESTING.md B1
```

Record the seconds per test. If they are still minutes, tell the session: the depth readback
is then the next thing to move onto the card (the CUDA/GL bridge).

## 4. Goldens

- `test_prism_domain_golden_storyboard` and `test_prism_heat_golden_storyboard` fail against the
  software goldens (SSIM 0.958 and 0.798, threshold 0.98). **Cause found (cloud agent, checked by
  looking at the pictures): the two software goldens are stale**, not a renderer bug. Look changes
  (`ccffe76`, `76ff8bb`) regenerated only the GPU goldens. The current software render matches
  the GPU goldens (SSIM 0.996 and 1.000). Read `pending/golden_prism_domain_heat/README.md`,
  look at the before/after pictures there, and if you like the "after", apply the patch or
  regenerate with `PHASOR_UPDATE_GOLDEN=1 uv run pytest tests/test_beauty.py -k "prism_domain_golden or prism_heat_golden"`
  on a no-CUDA machine (the cloud session can do it once you say OK). Optional on the PC: run
  the two tests there to confirm the GPU goldens still pass.
- Milestone 15 made only a software golden for `examples/open_contract.py`
  (`tests/golden/software/open_contract.png`). Make the GPU one and **look at it**:

  ```powershell
  $env:PHASOR_UPDATE_GOLDEN = "1"
  uv run pytest -k "golden and open_contract"
  Remove-Item Env:PHASOR_UPDATE_GOLDEN
  ```

## 5. Render workers (Milestone 14.3, CUDA half)

Follow `pc_tasks/cpu_workers.md` (branch `perf/cuda-workers`). Nothing about the CUDA worker
default changes until this is measured.

## 6. Windows-only test failures seen on CI (not caused by 14 or 15)

Run `uv run pytest -q` once on the PC and compare with this list. Known on the GitHub Windows
runner, which has no OpenGL: `test_remote.py::test_strangers_and_private_methods_are_refused`
(`icacls` lists 3 entries, the test expects 1), `test_texconv.py::test_text_and_equation_report_their_latex`
and `test_beauty.py::test_picture_golden_storyboard` (no OpenGL), and
`test_final.py::...[room_acoustics]` (`unsolved` label in analytic mode, also on the base
commit). If any of these fail on the PC with OpenGL present, that is new information.

## 7. Decisions for you (no PC needed, but write them down here)

- **GL 4.1 on macOS.** The renderer uses no OpenGL 4.3 feature (shaders retagged `#version 410`
  gave byte-identical frames on Mesa capped at 4.1). Lowering the requirement would likely let
  Macs render. It changes the tier rule in SPEC and CLAUDE.md: yes or no?
- **Updaters from outside the scene folder** (proposal in `MILESTONE_15.md`): allow or keep
  refused?
- **Pushing separate branches** `milestone/14-cpu` and `milestone/15-contract`: the cloud session
  can push only `claude/eloquent-mayer-jiqxs9`, so both milestones are on that one branch.

## Results

(fill in)
