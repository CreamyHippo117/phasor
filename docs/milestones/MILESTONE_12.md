# Milestone 12: interactive export

Roadmap row: desktop sliders, notebook widget, then web export. Exit demo: **a Julia explorer
with a slider for c** (`examples/julia_explorer.py`).

Status: **built; gates green on Linux (software OpenGL, Qt offscreen).** Untested on Windows,
a real GPU, a real display and a real Jupyter kernel (the widget is exercised through
ipywidgets' objects, not a running notebook).

## Built

| Area | Where | What |
| --- | --- | --- |
| Sliders in the document | `core/schema.py` (`Control`, schema v6), `core/patch.py` (`add_control`, `remove_control`), `scene.py` (`Scene.control`) | a numeric parameter (or one entry of a vector) with min, max, step, label; refused for expressions, text, and parameters a track animates; v5 documents migrate with none; `diff` names them; lint `orphan_control` |
| The player | `phasor/interactive.py` (`Player`), `Engine.fork()` | moves sliders on a private copy (snapped to the step, clamped to the range); the open scene never changes |
| Desktop | `gui/controller.py`, `gui/window.py` | Sliders tab (preview in the viewport; Apply is one undo step), hover readouts (f(z), \|f\|, arg f over a domain coloring; the escape count over a Julia/Mandelbrot set) |
| Notebook | `interactive.notebook` | picture, slider per control, time slider (extra `notebook`) |
| Web | `interactive.export_web`, `phasor web` | one self-contained offline HTML file: a picture per combination of slider positions (and optional time steps), sliders, dragging on the picture moves the first two; refuses more than 600 pictures |
| Exit demo | `examples/julia_explorer.py` | a Julia set with Re c, Im c, iterations |

## Verification

- `uv run pytest`: **774 passed, 25 skipped** (22 need CUDA, 1 needs a TeX install, 2 are empty
  parametrizations); ruff, ruff format and lint-imports clean; `phasor doctor` tier B.

- Controls are validated against what they drive (missing parameter, vector without a
  component, text, wrong component, animated by a track); they survive save/load; a slider
  whose object goes away is `orphan_control`; `diff` names added and removed controls.
- Player: snapping and clamping, the scene is untouched (version and document equal), reset
  restores the exact initial values, pictures are pure (back to the start, the same bytes).
- Julia explorer (real pixels): c in the Mandelbrot set leaves a connected set (more than 3% of
  the frame inside), c outside leaves dust (under 1%); different sliders give different pictures.
- Web export: six pictures for a 3 x 2 grid; no `http` anywhere in the page; the picture at
  (Re c = 0, Im c = 1.2) equals the player's pixel for pixel; too big a grid is refused and
  writes nothing.
- Notebook: moving a widget slider moves the player and changes the PNG shown.
- Editor: sliders preview without editing; Apply is one history step; the Sliders tab follows
  the scene after it rebuilds.

## Deviations from SPEC (SPEC updated)

1. Sliders are declared on the scene (`scene.control`) as document controls, not an
   `interactive=True` flag on each parameter: one place to look, one schema field.
2. Web export is baked pictures, not WebGPU/WGSL kernels: it works everywhere, offline, with no
   server; it is coarse (positions per slider) and large for many sliders.
3. No drag handles that write through sliders; hover readouts only for domain coloring and
   fractals (no winding number or pressure readouts yet).
4. The notebook widget needs `ipywidgets` (extra `notebook`).

## Bugs caught while building

1. A parameter still at its default is not in the document, so it could not be a slider:
   controls now fall back to the object type's default.
2. `Player.reset` snapped the starting value (0.156) to the slider grid (0.15): reset now writes
   exact initial values.
3. Slider tests on scripted scenes were refused ("cannot be applied to a scene built by a
   script"): the right behaviour (controls are structure), so the tests edit as the script.

## Untested

- Windows, GPU, a real notebook kernel, and the page in real browsers (the JavaScript is
  checked by reading it and by its data, not by running a browser).
