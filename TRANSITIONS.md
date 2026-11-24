# Plan: spatial transitions and ambience

The owner, 2026-11-24: what makes the best maths videos immersive is ambient music (the owner
does the audio), beautiful geometry, and **spatial transitions**: the camera never cuts, it
moves; it zooms out, the geometry changes while you are far away, and it zooms back in. This
file plans the visual half. Same rules as `ANIMATIONS.md`: verbs in `time/verbs.py` with
timing from the theme `[motion]`, docs, an example, tests, a storyboard golden.

## What makes them work

1. **The camera never cuts.** Every change of view is a motion, so the viewer always knows
   where they are.
2. **Change the geometry while the viewer can't look closely.** Zoomed out, detail is lost,
   so a structure can be reshaped or swapped there without it reading as a trick; zooming
   back in lands on the new thing.
3. **A zoom-out shows that what you were looking at is one point in something bigger.** The
   move then carries meaning, not only motion.

## Transitions

| # | What | Status | Prism version |
|---|---|---|---|
| T1 | **Lift (2D -> 3D)**: a flat picture tilts up and becomes a landscape (a heat map into its surface, domain colouring into the \|f(z)\| surface, a spectrogram into terrain, a pole-zero plot into \|H(z)\| with the unit circle cut through it) | **prototype**: `examples/prism_lift.py`, built from existing pieces, no engine change | the flat picture glowing, then the landscape under its net of glowing lines |
| T2 | **Pull back / dive**: zoom out until the scene is one point in a bigger space, glide, zoom into another point (one filter -> the map of all filters over cutoff and Q; a Julia set -> its point on the Mandelbrot set -> another Julia set) | **example**: `examples/prism_dive.py` (Douady rabbit -> its c on the Mandelbrot set -> c glides to seahorse valley -> dive into its double spiral). The zoom is the fractals' own `zoom`, driven by one `Value`; one `c` drives everything | the big space dark with faint structure, the current point a beacon |
| T3 | **Reshape while out**: zoomed out, the space itself bends (a line rolls up into the unit circle: sampling -> aliasing; a time axis winds into the winding picture), then zoom back in | plan; needs per-object "embedding" morphs (a line object's points morphed onto a circle already works with `morph`/`transform`) | the bending axis as a glowing stroke |
| T4 | **Fly through**: the camera passes through an object into the next idea (through the wall of the room into the spectrum of what was recorded there) | plan; needs the near plane and fog checked so the wall dissolves rather than clips | the wall as faint light the camera passes through |

## How the lift works (T1, prototype)

- The scene uses the **orbit camera from t = 0**, looking almost straight down
  (`azimuth=-90`, `elevation=89`: x right, y up, as in 2D) with a very narrow field of view
  (`fov=1.5`) from far away. That renders the same as the flat 2D camera: tested on
  `prism_domain`, the mean difference is 2/255 per pixel.
- `camera.distance` is an **expression of `camera.fov`**:
  `FRAME / (2 * tan(camera.fov * pi / 360))`, which keeps the plane through the target
  `FRAME` units tall on screen. Morphing `fov` alone is then a dolly zoom: perspective fades
  in and nothing jumps. Two separate tracks on `fov` and `distance` would not do this: in the
  middle of the move the framing would be off by a factor of six.
- Then `elevation` is morphed down (the plane tips over) and the object's height grows
  (`pde.Surface.height` from 0.01).
- Measured: the drum's corners stay within 1.8 px at 1080p through the dolly. The remaining
  drift is the 1 degree of tilt that `elevation <= 89` (`OrbitCameraParams`) leaves.

### Open points before it becomes a verb

- **A `lift` verb** (`s.at("m").lift(elevation=32, until="landed")`): it would write the
  distance expression and the two morphs. That touches `time/verbs.py`, needs theme
  `[motion]` timing and order-independence tests, so it waits for the owner's OK.
- **Text in a lifted scene**: in a test still of `prism_domain` tilted to 35 degrees, the
  equation (a world-layer billboard at z = 0) lost its lower half. Probably it sits
  in the field quad's plane and is hidden by depth; not investigated yet. Until then, put text
  in lifted scenes on the overlay layer.
- **Drift**: keep `drift=0` while flat (near top-down, azimuth sway rotates the whole
  picture); orbit after landing instead.
- **Plane larger than the frame** for domain colouring: tilted, a frame-sized plane shows its
  edge and black beyond it. The drum, as a bounded object, looks right with its edge.
- **A |f(z)| surface type** does not exist yet; the domain-colouring lift needs one
  (`cx.ModulusSurface`, the domain colouring as its texture). Not built.
- **Camera expressions may read only the camera** (found building T2):
  `core/pycode.py` `document_to_python` writes the camera's `s.set(...)` before any
  `s.add(...)`, so a camera expression that reads an object can't be rebuilt from Python and
  fails `test_every_example_round_trips_through_its_python_code`. `prism_dive` zooms the
  fractals instead. Fixing the write order would let a camera follow an object.
- **The lift verb** is ready as `pending/lift_verb.patch` (with pictures), waiting for the
  owner's OK.
- **A true ortho <-> perspective blend** (a 2D scene converted without opting in to the
  orbit camera) would change contract 4 and the schema: needs owner approval. The
  prototype doesn't need it.

## Ambience, visual side

The owner does the music. On screen:

- slow camera drift after landing (`drift`, `orbit(..., until=)`); Life on everything not
  `exact`; depth fog and haze so far geometry fades into the dark; hold shots longer than
  feels natural.
- **Cue sheet export** (done): `phasor cues scene.py -o cues.csv` writes the markers and
  each verb's span as CSV (`time,timecode,kind,name,start,end,duration`), so swells and hits
  can be placed on transitions in the audio editor (`phasor/workspace/cues.py`).

## Order

Done: T1 prototype, cue sheet, T2 example. Next: T1 verb (after the owner's OK), T3, T4.
