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

Expected: `10 passed`. The table the second command prints should match the one in
`MILESTONE_20.md` (the same numbers to 3 decimals: both sides are deterministic).

## Results

Write here: the Manim version, pass/fail per scene, and any number that differs from the
Linux table.
