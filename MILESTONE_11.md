# Milestone 11: Sketch

Dropped by the owner on 2026-11-19, reinstated the same day. Roadmap row: stroke capture,
classification, fitting, handwriting to LaTeX, paint-over. Exit demo: **draw a contour; it
integrates and animates** (`examples/sketch_contour.py`, and the editor's Sketch tab).

Status: **built; gates green on Linux.** Untested on Windows, with a real pen or tablet
(strokes in the tests are drawn by code and by Qt mouse events), and against people's
actual handwriting.

## Built

| Area | Where | What |
| --- | --- | --- |
| Classification | `phasor/sketch/strokes.py` | by fitting, not a learned model: dot, line, arrow (a hook at the end, or a V drawn at the tip), circle (algebraic circle fit), closed, scribble (direction reversals), curve and wavy curve; small strokes close together are handwriting. Thresholds are shares of the stroke's own size |
| Proposals | `phasor/sketch/propose.py` | each shape becomes plain data (summary, patch ops, verbs, findings) that depends on what is under it; nothing changes until `accept` (one `Engine.apply`, then the verbs; undone if a verb fails) |
| Contours | same | a loop over a domain coloring: smoothed into a glowing closed Line, traced from the playhead, its label revealed with it; the integral of f dz numerically along the drawn path, and exactly as 2πi Σ winding × residue for rational f; the direction drawn counts; a warning when the path runs close to a pole |
| Paint-over | same | a loop around objects emphasizes them; an arrow from an object moves it; a scribble removes what it crosses (`hit_test`); a dot snaps onto a pole or zero |
| Handwriting | same | a PNG of the strokes for an AI (or the person) to read; `transcribe` makes the Text |
| Use | `phasor sketch`, the editor's Sketch tab | strokes as JSON, `--accept`, `--text K=...`; the tab draws on the viewport and lists proposals with checkboxes. It reaches the editor through the new `phasor.panels` entry point, so the two clients never import each other |
| Demo | `examples/sketch_contour.py` | a wobbly hand-drawn loop around i on 1/(z²+1): π exactly, 3.1420 along the path; traced from the marker; a software golden |

## Also in this round (the owner's other asks)

- **3D and camera tracking.** Every 3D object type is framed by camera goals
  (`tests/test_geo_camera.py`). Fixed: a `focus_pull` or `orbit` at a marker froze those
  camera parameters at their defaults from t = 0, so the goals failed before the move. Now
  the solver owns them until the move starts and the move starts from the solved camera
  (SPEC "Camera solver"). Fixed: `orbit` wrote its turn from the already-solved angle, so
  an orbit at a marker could turn twice as far.
- **Typeset and LaTeX both ways.** `phasor/core/texconv.py`: `to_mathtext` / `to_latex`;
  every math string passes through the one its engine needs, so either form works on
  either engine; Text and Equation report `latex` in their semantics. Mathtext stays the
  default.
- **Layers.** The owner approved `interactive`, `importers` and `workspace` (contract 6,
  ADR 0001 amendment, CLAUDE.md).
- **Final tests** (`tests/test_final.py`): every type in 2D and 3D (inspects as plain data,
  no lint error, draws something); every example runs with no lint error, survives save and
  load, and round-trips through its generated Python; purity (any order, fresh engine);
  identical bytes for the same frame; undo all and redo all restore each document exactly;
  a patch with one bad op changes nothing; a fork never touches the scene; a tweak to a
  script scene survives re-running it, and a dangling one is linted.

## Bugs caught while building

1. `examples/fourier_linalg.py` did not build at all (a `morph` of `terms`, an integer,
   produced 24.5). No test ran that example; the final tests now run every one. Integer
   parameters now animate in whole steps.
2. The overlap lint flagged labels sitting on a backdrop (a domain-colored plane), which the
   placer and CLAUDE.md both allow.
3. Sketch first proposed a contour label visible before the contour was drawn (seen in the
   first golden); it is now revealed with the trace.
4. Pixel to world was written twice (editor and sketch): it is now `Camera.from_screen`.

## Found, not fixed

- Hairline strokes (`stroke` role `hairline`, e.g. 3D axes in 2D) are under one pixel wide in
  small frames (a 192 × 108 thumbnail) and vanish. Full-size frames are fine. A minimum
  on-screen width with reduced alpha would fix it but changes every golden: left for the
  owner's review.

## Deviations from SPEC (SPEC updated)

1. Handwriting to LaTeX has no offline recognizer: an AI reads a picture of the strokes.
2. Not built from the SPEC's sketch section: photo import of paper, keyframe by drawing,
   3D camera sketching, storyboard import, a vision model proposing objects.
3. Sketch edits plain scenes; on a script scene only parameter and style proposals apply
   (the script's own code can apply the rest with `origin="script"`, as the example does).

## Verification

- See the commit for the full run; ruff, ruff format and lint-imports clean.
- The milestone gate's contract audit was done by the authoring model, not an independent
  reviewer (no subagent was asked for).

## Untested

- Windows; CUDA; a real pen or tablet; real handwriting; a real TeX install (the LaTeX path
  is tested with stand-in `latex` and `dvipng` programs).
