# Beauty plan

The owner's plan of 2026-11-22, kept here so it outlives any one session. Update the status
column as items land. The reference look is `prism_primes`: dark background, glowing strokes
with hue along them, haze, vignette, fringe. **Keep that look exactly as it is**; extend it to
the things below. Palettes (`meta.palette`) apply to all of it.

Rules that hold for every item: the dark theme stays the default; the look is a `look` or a
`preset` (a taste knob), never numbers typed into a scene; glow comes only from emission
(never thresholding the picture); a new GPU kernel has a NumPy reference and the three-way
kernel tests; every new type gets docs, an example, a golden storyboard, and a GPU golden
from the owner's PC.

Style references from the owner (screenshots, 2026-11-22): luminous domain coloring on a
dark ground (petals of phase glowing out of a bright core), glowing hue-cycling curves with
haze, root point clouds where millions of points glow additively and fractal edges appear,
and a large 3D graph of glowing red and yellow edges. Phasor makes its own scenes in these
looks; it does not recreate anyone's videos.

## Order

| # | What | Why this order | Status |
|---|---|---|---|
| 1 | **Prism domain coloring.** A `prism` preset for `cx.DomainColoring`: phase as hue from the palette, brightness from closeness to zeros and poles so structure glows out of darkness, contour bands as faint light, emission so it blooms and hazes like the strokes | the most visible change; touches only the domain kernel and its look | **done** 2026-11-22: kernel `domain_glow`, preset `prism` (the prism looks use it), glowing rays and rings, `examples/prism_domain.py`; GPU golden to make on the PC |
| 2 | **Heat maps.** `pde.Field`, `PressureSlice`, `Spectrogram`: prism colormaps, smooth (bicubic) sampling, glowing isolines, haze; optional height (3D) mode | same machinery as 1 (a field drawn by colour), so it reuses it | to do |
| 3 | **Point clouds.** A `PointCloud` object: millions of points splatted additively on the GPU with hue and glow (polynomial root clouds, attractors, Buddhabrot-style fractals); plus a cloud mode for `cx.Fractal` | the new GPU feature; needs a splatting kernel (CUDA atomics) | to do |
| 4 | **Fractals in the prism look.** Escape-time colour from the palette, glowing edges, haze; Julia and Mandelbrot | builds on 1 and 3 | to do |
| 5 | **Imports: pictures.** An `Image` object: any PNG/JPG (the Mona Lisa) placed, sized and animated like any object, with optional glow or a frame | small, and needed by 6 and 7 | to do |
| 6 | **Imports: graphs.** A `Network` object loaded from a file (edge list, CSV, JSON, GraphML) or built in a script: 2D or 3D force layout computed once, glowing additive edges coloured by an attribute, thousands of nodes (a game tree of Connect 4 positions, say). Graph-theory types (Cayley graph and friends) gain the same look | big; the layout solver and edge batching are the work | to do |
| 7 | **3D acoustics.** A 3D room (box, or a mesh imported from OBJ) drawn as faint glowing wire outlines, with sound as rays and expanding wavefronts that bounce off the walls (image-source and ray tracing, on the GPU where it pays) | biggest; uses 5 (meshes) and the 3D camera | to do |

After each item: a showcase scene in `showcase/` so the owner can watch it.
