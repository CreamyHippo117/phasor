# Testing plan and status

The plan agreed with the owner on 2026-11-19, kept here so it outlives any one session.
Update the status column as items land. Commands are in `CLAUDE.md`.

Status today: **1,136 tests** (`uv run pytest`): all pass on Linux with software OpenGL;
60 need CUDA, 1 needs TeX, 1 a GPU for the doctor, 4 build the packaged app (opt-in).
**Untested: Windows, NVIDIA/CUDA, macOS, a real TeX install, a real pen, real handwriting.**

How to run the opt-in parts:

```bash
PHASOR_TEST_PACKAGE=1 uv run pytest tests/test_packaging.py    # builds the app (~1 min, 480 MB)
PHASOR_UPDATE_GOLDEN=1 uv run pytest -k golden                 # regenerate reference pictures, then look at them
```

## A. The desktop app: done (2026-11-19)

`tests/test_desktop.py` and `tests/test_packaging.py`, 41 tests.

| # | What | Status |
|---|---|---|
| A1 | Settings: round trip, damaged or missing file, platform folders, recent scenes (newest first, no duplicates, at most 12, deleted ones hidden) | done |
| A2 | Unsaved changes (yours or an AI's), Save As inside and outside the folder, tweaks to a script scene kept | done |
| A3 | Exports: each command; picture, web page, video with progress; cancel; failure says why; narration audio in the video; nothing left behind | done (found: a hidden scene copy left behind; fixed) |
| A4 | Start screen, templates, reopening an open scene, broken files, examples opened as copies, title marks unsaved edits | done |
| A5 | Closing: Save / Discard / Cancel; clean windows close without asking; Quit stops at the first Cancel; closing stops exports and the AI | done |
| A6 | No two actions share a keyboard shortcut | done |
| A7 | LaTeX without TeX is refused and reverted; the choice is remembered | done |
| A8 | Recent menus in every window; the export dialog shows only what each kind needs | done |
| A9 | Launch commands inside and outside the packaged app; menu shortcut on Linux; honest answer on macOS; the Windows shortcut command (not run on Windows) | done |
| A10 | Packaged app: entry point without a console; build; every command; a render with typeset maths; examples bundled; the window starts | done (Linux only) |

## B. Windows and NVIDIA (on the owner's PC)

GPU tests are written (`tests/test_gpu.py`, 38 tests, plus 22 in `test_kernels.py` and
`test_doctor.py`); they skip themselves without a card. On the PC, in the phasor folder:

```bash
uv sync --extra gl --extra encode --extra gui --extra mcp --extra gpu
uv run phasor doctor
uv run pytest
uv run pytest tests/test_gpu.py tests/test_kernels.py -v -m gpu
uv run phasor bench
```

Then the long ones: `$env:PHASOR_LONG_RENDER=2000; uv run pytest tests/test_gpu.py -k memory`
(PowerShell), and the GPU reference pictures: `$env:PHASOR_UPDATE_GOLDEN=1; uv run pytest -k golden`
(look at `tests/golden/gpu/` before committing them).

| # | What | Status |
|---|---|---|
| B1 | Full suite on Windows **without** OpenGL (what CI sees) and **with** it | to do |
| B2 | Windows paths: spaces, backslashes, a non-English user name, CRLF, deleting a temp file still open, `%APPDATA%` settings | to do |
| B3 | Packaged `.exe`: build, start with no console window, Start-menu shortcut, exports that do not flash a console | to do |
| B4 | CUDA: every kernel against NumPy on the card; CUDA picked automatically; CUDA frames reach OpenGL unchanged (odd sizes, HDR) | first run 2026-11-21 (RTX 4070 Ti, Windows 11): found two bugs, fixed, rerun to do |
| B5 | GPU reference pictures (`tests/golden/gpu/`): regenerate and look; the same kernel and the same whole frame twice are identical; a whole frame on the card matches the NumPy frame | tests written, to run |
| B6 | Deep zoom at 1e-30 on the card against exact iteration; real float64 on the card; tiling and supersampling; a 4K frame within the VRAM budget; the card faster than NumPy; `phasor bench`; memory flat over 300 (or 2,000) frames; clean fallback with the card hidden or CuPy broken | tests written, to run (fallback passes on Linux) |
| B7 | A packaged app with CuPy in it: `PHASOR_PACKAGE_GPU=1` and `PHASOR_TEST_PACKAGE=1`, then `pytest tests/test_packaging.py` | tests written, to run |
| B8 | Real TeX (MiKTeX): the LaTeX engine end to end, conversion on real LaTeX, a clear message for a missing LaTeX package | to do |

## C. Gaps in the suite

| # | What | Priority | Status |
|---|---|---|---|
| C1 | Random patch sequences keep the invariants (undo all restores, a bad patch changes nothing, ids never reused) | P1 | to do |
| C2 | Hostile formulas never run as code; the maths converter is idempotent on random input; random strokes never crash the classifier | P1 | to do |
| C3 | A file saved in each old format (schema 1 to 5) still loads and looks the same | P1 | to do |
| C4 | Security: the AI stays in its folder (`..`, symlinks); scripts refused without permission; wrong AI token refused; the connection file is private; a scene title cannot inject code into the web page | P1 | to do |
| C5 | Build the package, install it in a fresh environment, run the examples, docs search and command line from there | P1 | to do |
| C6 | A reference picture for every example; the vanishing thin lines in small pictures as a known failure | P2 | to do |
| C7 | Speed limits per example and per frame (`riemann_sqrt` takes 155 s to set up) | P2 | to do |
| C8 | Memory and GPU objects flat over long renders and many window opens and closes | P2 | to do |
| C9 | Video formats: mp4, ProRes `.mov`, PNG sequences, odd sizes, 4K; narration in sync | P2 | to do |
| C10 | A person and an AI editing at once under load: no lost edits, no deadlock | P2 | to do |
| C11 | Docs match the code: every public name in `API.md`, the command table, the AI tool list | P3 | to do |
| C12 | Light themes and every look pass lint on every example | P3 | to do |
| C13 | Sketching with a real pen; Manim import against real Manim | P3 | to do (needs the owner) |

## Found on the owner's PC (2026-11-21, RTX 4070 Ti, Windows 11, CuPy 14.2)

- **Every final render with CUDA crashed**: the picture kernels (glow, tone map, resize,
  fringe, vignette, fractal palette) passed host arrays to `cp.ascontiguousarray`, which CuPy
  refuses. Fixed (`cp.asarray` first). The CPU stand-in for the card (`tests/cuda_emu.py`)
  now refuses host arrays the same way, so this kind of bug fails on Linux too.
- **The card rounded differently from the reference**: NVRTC fuses `a*b + c` by default, and
  fractal edges amplify the difference (3% of a tile, 10% of a deep float64 view). Kernels now
  compile with `--fmad=false`.
- OpenGL ran on the processor's built-in AMD graphics, not the RTX card. Windows setting:
  Settings > System > Display > Graphics, set Python to "High performance".

## Found and not fixed

- Hairline strokes under one pixel wide vanish in small pictures (thumbnails). Full-size
  frames are fine. Fixing it changes every reference picture: waiting for the owner.
- A dense 3D *box* of arrows (preset `dense` with depth) is cluttered; a plane is fine.
- On macOS the renderer cannot start: it needs OpenGL 4.3 and macOS stops at 4.1 (see the
  Mac discussion of 2026-11-20 in the session notes; no decision yet).
