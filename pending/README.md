# Pending: changes waiting for the owner's OK

Each change is a patch against the branch it sits on, plus pictures. Apply with
`git apply pending/<name>.patch`; delete both when it lands or is declined.

## `lift_verb.patch`: the `lift` verb (TRANSITIONS.md T1)

- `s.use_camera("orbit", flat=7.5)`: a 3D camera that shows exactly what the 2D camera of
  height 7.5 shows (looking down from far off through a 1.5 degree field of view; `distance`
  is an expression of `fov`). Sugar in `scene.py` only: no schema change.
- `s.at("m").lift(elevation=32, fov=35, until=None)`: widens the field of view (a dolly zoom,
  nothing on the plane moves: within 1.8 px at 1080p) over the first 55%, and tips the camera
  over from 45% on. Timing is theme `[motion.lift]` (4.5 s, smoother) in every theme, or
  until a marker. Also the MCP `animate` verb `lift`.
- Tests: `tests/test_lift.py` (the flat camera matches the 2D one to 0.5 px; the dolly does
  not move the plane; landing values; `until`; errors; script order). Docs: API.md 3D
  section (a runnable block), MCP.md, llms.txt. `examples/prism_lift.py` uses the verb.
- Limit: after a lift the distance stays tied to `fov`, so `focus_pull` on that camera
  refuses with a `VerbError` that says to move `target` and morph `fov`. The cue sheet shows
  the lift as one cue (`tests/test_cues.py` updated in the patch).
- Tested together with `text_depth.patch` on top: 155 passed (cues, lift, camera3d, docs,
  kit, mcp, motion, narration, layers) and `test_final -k "prism_lift or prism_dive"`.
- Pictures: `lift_verb_before.png` (the example as on the branch: done by hand with
  `morph`) and `lift_verb_after.png` (with the verb). Times 2, 4.5, 6, 7.5, 10, 14 s.

## `text_depth.patch`: text no longer cut by the plane it stands on when the camera tilts

- Cause: world text is a billboard facing the camera, centred on its origin. Tilted, the
  part below the baseline (a fraction's denominator) goes below z = 0, behind a plane at
  z = 0 (domain colouring, a heat map), and the depth test drops it. Top-down the text lies
  in the plane, so draw order decides and it shows.
- Fix (`render/passes.py`): a text corner below the horizontal plane through its origin is
  depth-tested where its view ray crosses that plane: same pixel, nearer depth. Only text,
  only `gl_Position.z`; the id/solo passes share it, so the inspector agrees. Ortho is
  unchanged by construction; all eight 3D examples render byte-identical.
- Trade-off: an opaque object in a thin slab just in front of the text's lower part no
  longer hides it (biggest at grazing angles: a dot 5% of the way to the eye at 10 degrees
  elevation goes from 21% to 93% visible; at 30% of the way nothing changes). The
  alternative, hinging text at its bottom edge, moves text on screen in every 3D view and
  changes goldens.
- SPEC.md (Milestone 6 "as built") gets one sentence. Tests: `tests/test_camera3d.py`
  (cut at 20/35/60 degrees before, whole after; still hidden by what is in front).
- Picture: `text_depth_before_after.png` (top left before at 35 degrees; then after at 35,
  10, 60).
