# Step 7: a 3D room with sound bouncing in it (on the PC, branch `beauty/acoustics-3d`)

## Goal

The owner's picture: a 3D room drawn as faint glowing outlines of its walls, with sound
leaving a source as rays and an expanding wavefront that reflect off the walls, so you can
watch reverberation build up. Camera orbits (the 3D camera and its solver exist). Prism look.

## What exists

`phasor/domains/acoustics.py` (2D-drawn `Room`, `Source`, `Listener`, `Wavefront`,
`ImageSources`, `ImpulseResponse`, `PressureSlice`) and `phasor/domains/ac_math.py`, whose
`image_sources(size, source, order, alpha)` is already 3D (a box room). The heat-map
helper (`phasor/domains/heatmap.py`) draws pressure slices in the prism look.

## What to build

1. **`Room3D`**: a box (`size`, `origin`) or a mesh from a Wavefront `.obj` file
   (`path`, next to the scene: declare `Files = ("path",)` like `Image` in
   `phasor/domains/media.py`, so a missing file is `missing_file` and the AI stays in its
   folder). Draw the edges (box) or the mesh's feature edges as thin glowing lines with
   `strokes.polyline`; optionally faint glass walls (`ctx.surface(..., material="glass")`).
2. **Ray tracing** in `ac_math.py` (pure NumPy; CUDA kernel only if profiling shows it
   pays): `rays(room, source, n, bounces, alpha, seed)` returning, per ray, its polyline of
   reflection points and the energy after each bounce (specular reflection; for a mesh,
   intersect triangles). Randomness through `rng(object_id, seed)`.
3. **`SoundRays`** object: rays drawn up to the distance sound has travelled at time `t`
   (speed of sound `c`), as glowing strokes whose brightness is the energy left, hue from
   the palette by bounce number. `count`, `bounces`, `speed` (slow-motion factor).
4. **`Wavefront3D`**: the expanding sphere, drawn as the circles where it meets the walls
   (and the image-source spheres, for the reflected fronts), fading with absorption.
5. **Example** `examples/room_3d.py`: a box room (or a simple L-shaped mesh room in
   `examples/assets/`), a clap at the source, rays and fronts spreading and bouncing, the
   camera orbiting; the impulse response building up in a corner panel.
6. **Tests**: rays reflect with angle in = angle out; ray energy follows (1 - alpha)^bounces;
   first arrivals match `image_sources`; the drawn rays end at distance c*t; the `.obj`
   loader; `missing_file`; determinism; a golden storyboard (GPU golden made here).
7. **Docs and showcase** as in the other tasks.

## Shared files (merge conflicts)

`phasor/domains/acoustics.py` and `ac_math.py` (only this task edits them), `docs/ai/API.md`,
`BEAUTY.md`, `showcase/*`. Merge last (third).

## Done when

The example renders smoothly on the RTX card with the camera orbiting, goldens made and
looked at, all tests pass, branch pushed, owner told it is ready to merge.
