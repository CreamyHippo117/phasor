# Milestone 13: Manim import

Reinstated by the owner on 2026-11-19 (it had been dropped for a day). Roadmap row: a converter
for common Manim CE scenes. Exit demo: **an old Manim reverb scene runs in phasor**
(`examples/manim_reverb.py`).

Status: **built; gates green on Linux.** Untested on Windows and against a real Manim install
(the stand-in is written from Manim CE's documented API, not by importing Manim: no Manim was
available to compare against).

## Built

| Area | Where | What |
| --- | --- | --- |
| The import | `phasor/importers/manim.py` | the script runs unchanged with a stand-in `manim` module on the import path; `self.add` and `self.play` become objects and transitions at Manim's own times; the module is removed afterwards |
| Mobjects | same | `Dot`, `Circle`, `Square`, `Rectangle`, `Triangle`, `Polygon`, `RegularPolygon`, `Line`, `Arrow`, `Vector`, `ParametricFunction`, `Text`, `Tex`, `MathTex`, `Axes`, `NumberPlane`, `VGroup`; positioning (`shift`, `move_to`, `next_to`, `to_edge`, `to_corner`, `scale`, `rotate`, `arrange`, `align_to`) computed as Manim does, in its 8-unit frame |
| Animations | same | `Create`, `Uncreate`, `Write`, `FadeIn`, `FadeOut`, `GrowFromCenter`, `Transform`, `ReplacementTransform`, `Rotate`, `Indicate`, `MoveToTarget`, `AnimationGroup`, `LaggedStart`, `Succession`, `.animate`, rate functions, `wait` |
| 3D | same | `ThreeDScene`: `set_camera_orientation` and ambient rotation become the orbit camera |
| Honesty | same | colors become the nearest style role; fills are outlines; updaters, `always_redraw`, `ValueTracker` animations and font scaling are warnings; an unknown Manim name raises an error that says so |
| Use | `phasor manim`, `scene_open`, `phasor gui` | the result is a plain scene (not script-locked), saved as `name.phasor.json`; `--code` also writes it as a phasor script |
| Demo | `examples/manim_reverb.py` | a room, a source, expanding wavefronts, a decay plot and RT60 = 0.161 V/A |

## Verification

- `uv run pytest`: **774 passed, 25 skipped** (22 need CUDA, 1 needs a TeX install, 2 are empty
  parametrizations); ruff, ruff format and lint-imports clean; `phasor doctor` tier B.

- The demo runs unchanged; its duration is the sum of every `run_time` and `wait` (14.5 s);
  nothing is visible before Manim shows it; the room draws over 1.5 s, rings expand in turn,
  the formula writes in at the end; no lint errors; a software golden.
- Layout: positions, `next_to`, `to_edge` land where Manim's would; `Transform` morphs shapes
  with equal point counts (the target is never created); `.animate` moves smoothly; groups fade
  together; `axes.plot` and `c2p` match; math text is typeset; 3D orientation and rotation.
- Honesty: warnings for what cannot be carried over, errors for what is missing, scene choice
  by name, no Scene class, the stand-in removed from `sys.modules`.
- `phasor manim` saves a scene that loads and has its objects; `scene_open` on a Manim script
  opens it as a plain scene you can edit; a server without `--allow-scripts` refuses it; the
  editor opens it directly.

## Deviations from SPEC (SPEC updated)

1. A compatibility layer that runs the script, not a source-to-source converter: it gets control
   flow, helper functions and loops right for free, at the cost of running the script's code.
2. Only a documented subset of Manim; the rest is an explicit error or warning.

## Bugs caught while building

1. Objects existed from t = 0 at full opacity until their first "show": every imported object
   now starts invisible and is switched on at the time Manim first shows it (found by looking
   at the first storyboard).
2. An id starting with an underscore (from a private base class name) was invalid.

## Untested

- Windows; real Manim (see above); scripts using MovingCameraScene frame animations, updaters
  and graphs of functions with discontinuities.
