# Milestone 6: 3D + spatial awareness

Roadmap row: perspective camera, surfaces, OIT, event log, camera solver, 3D label placement,
auto-cinematography. Exit demo: **Riemann surface of sqrt(z): labels readable through a full
orbit, event log clean.**

Status: **built; gates green on Linux (software OpenGL).** Untested on Windows and on a
real GPU.

## Built

| Area | Where | What |
| --- | --- | --- |
| Orbit camera | `core/spatial.py` | perspective camera on a sphere around `target`, z up; `units_per_height(point)` by depth; `clip_w` |
| Choosing it | `set_camera` patch op, `Scene.use_camera` | switch a camera between `ortho` (2D) and `orbit` (3D) |
| Drift | `time/compile.sway` | slow azimuth sway, a pure function of t (theme `[camera]`; `drift=0` holds still) |
| Lit surfaces | `DrawContext.surface`, `render/passes.py` | three-point rig placed relative to the camera, two-sided Blinn-Phong, depth fog (theme `[light]`); materials solid / glass / wire / emissive |
| Glass | `render/passes.py` | see-through, does not hide things (`occludes`), drawn after opaque objects with its own triangles sorted back to front |
| Riemann surfaces | `cx.RiemannSurface` | `sqrt`, `cbrt`, `log`; height Re or Im; colored by phase; features `branch_point`, `sheets` |
| Depth in layout | `PassRunner.depth_buffer`, `OcclusionOracle.depth` | real view-space depth from the id pass (ADR 0001 pattern) |
| 3D labels | `layout/bake.py` | live at their anchor's depth (nudged 4% toward the camera), slide sideways, avoid only what is in front of them |
| Camera solver | `layout/camera.py` | goals `keep_visible`, `center`, `thirds`, `avoid_occlusion`, `min_size`; solves the parameters left unset; L-BFGS-B multi-start; stage 2 on the depth buffer; PCHIP; refine where it fails |
| Goals | `core/constraints.py` | `camera keep_visible(a, b)` etc. as rules with priorities; `visible_in_3d(a)` guard |
| Event log | `inspect/events.py`, `Engine.events` | 10 Hz plus markers; visibility 0.9/0.5/0.1, frame, region, new occluder; 5% hysteresis; marker, severity, fix hints |
| Queries | `Engine.when_hidden`, `Engine.free_space_near` | hidden spans; empty label-sized places around an object |
| Lint | `inspect/lint.py` | `occluded_label`; in 3D `overlap` only compares label-label and point-point |
| Verbs | `time/verbs.py` | `orbit(degrees, until=marker)`, `orbit_reveal`, `focus_pull` in 3D |
| MCP | `mcp/tools.py` | `events`, `when_hidden`, `free_space_near`; `animate` takes `orbit`; `set_camera` op |
| Exit demo | `examples/riemann_sqrt.py` | narrated, full orbit, solver-chosen camera, glass sheets, a walk that climbs to the other sheet |

## Verification

- `uv run pytest`: **460 passed, 16 skipped** (all need CUDA); ruff, ruff format and
  lint-imports clean; `phasor doctor` tier B. New: `test_camera3d.py`, `test_spatial.py`, `test_riemann_demo.py`
  (slow: the full 10 Hz event log with real pixels, about 3 minutes on llvmpipe).
- **Exit demo**: the camera turns exactly 360 degrees between the `walk` and `home` markers;
  elevation, distance and target come only from the solver; the event log has **no
  problems**; every world label is at least 85% visible and fully on screen at every
  0.25 s through the orbit; `lint(strict=True)` returns nothing; a
  software golden storyboard.
- Looked at every storyboard on the way. The tools found real problems in the first drafts
  and each fix was driven by them: points hidden behind the other sheet (fixed with glass),
  the walk crossing the sheet markers, a blown-out glow, the surface running into the title,
  a camera that swung overhead.

## Bugs caught while building

1. The solver's search box came from the first key only, so the camera could not follow a
   moving object. Now from the whole timeline.
2. A flat "behind the camera" penalty gave the optimizer no gradient; it stopped where it
   started. Depth is now clamped with a graded penalty.
3. Bounding spheres of flat surfaces are huge and look the same from every side: framing
   now uses projected box corners.
4. Smoothness penalized motion, so following a moving object cost too much; it now
   penalizes jerks, plus a little angular speed.
5. My own NDC aspect correction was wrong by a factor of about 3 (caught by a test that
   compares the solver's projection with the camera's).
6. `drift: None` broke the value universe (parameters cannot be None): `-1` means "theme".
7. The 2D "covered pixels" rule made every slot near a surface look blocked; 3D labels now
   use depth.
8. `Engine.events` (a method) shadowed the `events` module inside the class body.
9. **Labels cut dark boxes out of glass behind them** (found by watching the rendered
   video): a text quad wrote depth over its whole rectangle, so glass drawn later was
   rejected there. Empty text texels are now discarded; a test checks that nothing behind
   a label gets darker.

## Deviations from SPEC (SPEC updated in the same commit)

1. Glass uses per-triangle back-to-front sorting, not order-independent transparency.
   Exact except where translucent triangles cut through each other.
2. Camera goals are constraint rules (`camera keep_visible(...)`); `frame(ids, during)` is
   not a separate query, and goals hold for the whole scene (no `during`).
3. The field of view is never solved (with distance free it only adds ambiguity).
4. `thirds` is a goal you ask for, not on by default; `smooth` is always on.
5. Framing uses projected bounding boxes; spheres only for occlusion.
6. Elevation is kept within the theme's `min_elevation`/`max_elevation` (-60/65).
7. `outside`, `tangent_to`, `attach_to` moved to Milestone 8. `z_fight` lint not built.
8. No `facing` field in the inspector: every label is a camera-facing billboard.
9. No soft shadows or ambient occlusion yet (SPEC "Look"); lighting is the three-point
   rig plus fog.

## Untested

- Windows, NVIDIA GPU (all of this ran on Mesa llvmpipe).
- `cbrt` and `log` surfaces only through unit tests, not looked at in a demo.
- The camera solver on scenes with many goals and objects (speed: the demo's bake with
  stage 2 takes tens of seconds on llvmpipe).

## Open

- Fresh-eyes boundary review not run (Milestones 2-6).
- The exit demo takes about 3 minutes to check fully on a software renderer.
