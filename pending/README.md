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
  cannot animate the distance; zoom by morphing `fov`.
- Pictures: `lift_verb_before.png` (the example done by hand with `morph`), and
  `lift_verb_after.png` (with the verb; the drum's phase shifted so it is at full swing as
  the camera lands). Times 2, 4.5, 6.5/6, 8/7.5, 10, 14 s.
