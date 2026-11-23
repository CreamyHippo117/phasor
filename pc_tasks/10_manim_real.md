# 10. The Manim import against real Manim (Milestone 20), on the PC

Already run on the cloud box (Linux, Manim CE 0.22.1, TeX Live 2023): all 10 scenes pass
(`MILESTONE_20.md`, "Real Manim"). On the PC this checks the same on **Windows**, where it has
never run. Branch: `claude/eloquent-mayer-jiqxs9` (no new branch). Folder:
`C:\Users\Atlas\phasor-dev`.

Manim goes in its **own** virtual environment, never in phasor's.

## 1. Install Manim (PowerShell)

```powershell
cd C:\Users\Atlas
uv venv manim_env --python 3.12
uv pip install --python manim_env\Scripts\python.exe manim
manim_env\Scripts\python.exe -c "import manim; print(manim.__version__)"
```

Manim's wheels bundle Cairo and Pango on Windows. `math_tex.py` and `manim_reverb.py` need
LaTeX (MiKTeX: `winget install MiKTeX.MiKTeX`, then let it install packages on demand); the
other scenes do not. Without LaTeX those two fail with a LaTeX error: write that down, it is
not a phasor failure.

## 2. Run the comparison

```powershell
cd C:\Users\Atlas\phasor-dev
$env:PHASOR_MANIM_PYTHON = "C:\Users\Atlas\manim_env\Scripts\python.exe"
uv run pytest tests/manim_real -v
uv run python -m tests.manim_real.compare -v
```

Expected: `11 passed` (`wait_until` was added by the M20 audit). In the table the second
command prints, the clock, positions and sizes of shapes, dots, lines and the camera frame
should match the one in `MILESTONE_20.md` to 3 decimals (both sides are deterministic). The
`text` numbers, and positions of anything placed next to a text, will differ: Manim's
`Text` is set by Pango in the system's fonts, which are not Linux's on Windows; they must
stay within the tolerance (0.35), not match the Linux numbers.

## Results

Write here: the Manim version, pass/fail per scene, and any number that differs from the
Linux table.

**2026-11-23, Windows 11 (RTX 4070 Ti PC), Manim CE 0.22.1** (`uv venv manim_env --python
3.12`, wheels only, no build), **MiKTeX 25.12** (user install via winget, auto-install of
packages on). `uv run pytest tests/manim_real -v`: **11 passed** (33.9 s), every scene
including `math_tex` and `manim_reverb` (real LaTeX).

`python -m tests.manim_real.compare -v` against the Linux table in `MILESTONE_20.md`:

| Scene | Windows | Same as Linux? |
| --- | --- | --- |
| shapes_layout | positions 0.008, sizes 0.250 (arrow tip, by design) | yes |
| animate_timing | positions 0.000, sizes 0.101 (by design) | yes |
| fills_colors | positions 0.011, sizes 0.045 | yes |
| text_write | positions 0.000, sizes 0.000; text 0.463 (by design, skipped) | text 0.463 here, 0.54 on Linux (fonts) |
| axes_plot | positions 0.003, sizes 0.026; text 0.050 | yes |
| math_tex | positions 0.000, sizes 0.000; text 0.108 | yes |
| moving_camera | camera frame 0.000, sizes 0.032 | yes |
| updaters | positions 0.000, sizes 0.045 | yes |
| manim_camera_tour | positions 0.001, sizes 0.037, camera 0.000; text 0.132 | text 0.132 here, 0.104 on Linux (fonts) |
| wait_until | clock 3.367 s, positions 0.000, sizes 0.032 | yes |
| manim_reverb | clock 14.5 s, positions 0.010, sizes 0.035; text 0.303 | yes |

Clock, play counts, run times and rate functions match in every scene. Every shape, dot,
line and camera number is the Linux one to 3 decimals. Only text differs (Pango on Windows
fonts), and it stays within 0.35 except `text_write`, which is by design.
