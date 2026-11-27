# Handoff 2026-11-26: echo follow-ups and the reverb_la wishlist

Branch `claude/great-lamport-kvky14`, on top of merged PR #21. Eight pieces of work are merged
here, each built by a separate agent. Merging them together needed two small conflict
fixes: a lint table row and an API.md section. Session paused for usage credits.

## Status: merged here, NOT yet verified as a whole

- **Checks run on the merged result:** ruff, ruff format --check and lint-imports are
  clean, and `llms.txt` is current.
- **Not yet run:** the full `uv run pytest` on the merged result. That is the first job
  next session. Each agent ran only its own subsets. The last three agents (unfold, net,
  shared glow points) were stopped while their own full-suite runs were still going, so
  their reports were never received.

## What is in it

1. **Unfold beyond the box** (`core/unfold_math.py`, `domains/unfold.py`, verb `unfold`):
   - convex polygons and polyhedra, so polygon billiards and the kaleidoscope work
     (`examples/kaleidoscope.py`);
   - convex `.obj` rooms;
   - `SoundRays` rays;
   - corner hits taken one after the other.
   - Not checked yet: the agent's handling of mirror image vs turn for asymmetric rooms.
     Read its commit `492c807` and tests.
2. **Nets and geodesics** (`domains/net.py`, `net_math.py`, `net.Polyhedron`, `Face`):
   - a cube or any convex polyhedron opens into a net;
   - the geodesic straightens on the net (the spider and fly gives 40; a unit cube gives √5);
   - example `spider_and_fly.py`.
3. **One way to draw glowing points that flash** (commit `fdc006e`): `StarLattice`,
   `SweepLattice` and `ac.ImageSources` should share it. Check it against its brief:
   - the shared flash and glowing-point code;
   - `ImageSources` no longer limited to 6 bounces;
   - an example of the step from animation 8 to 10.
4. **Wishlist: workers.** Workers run the scene script. `load` names the script last in
   `meta.modules` (owner approved 2026-11-26, contracts 2 and 3, recorded in SPEC). Each
   worker's first frame is checked against the parent's.
5. **Wishlist: layout.** The bake runs only while a label is shown, reuses itself across
   motion edits, and verbs bake only up to their start. New lint rule `slow_layout`.
   - The owner accepted this as a non-contract change (2026-11-26).
   - Still open: the camera solver checks depth at every key, and non-motion edits re-bake
     every shown span.
6. **Wishlist: `filt.PhasorField`** with exact shutter blur (example `reverb_modes`).
   - Open: the renderer could pass its fps and shutter to `draw`, which would touch
     contract 3 and needs the owner.
   - Open: the prism phase colormap shows only about 6 hues.
7. **Wishlist: conformal `cx.ComplexMap`** (`path="family"`, preset `roll`) and
   `ease="log:<curve>"`, the default for camera distance and height zooms (example
   `s_plane_roll`).
8. **Wishlist: `s.note` / `s.sound`** in the document (schema 9), mixed under the narration
   with a limiter that never changes the narration (example `sound_harmonics`).
   - Open: the editor plays no audio and shows no sound events.
   - Open: only 16-bit WAV files.
   - Open: reverb on notes through `ImpulseResponse`.

## Next session

1. `uv sync --extra gl --extra encode --extra mcp`, then the full `uv run pytest`, about
   30 min with nothing else running. Fix what fails; regenerating goldens needs a look
   before and after.
2. Read the three unreported commits (`492c807`, `a2d648d`, `fdc006e`) against their
   briefs, above.
3. Open the PR (draft) and watch CI.
4. Untested on Windows (own PC), CUDA and macOS. There are no GPU goldens for the new
   examples: add them to `pc_tasks/`.
