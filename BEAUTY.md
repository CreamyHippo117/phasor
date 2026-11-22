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
| 2 | **Heat maps.** `pde.Field`, `PressureSlice`, `Spectrogram`: prism colormaps, smooth (bicubic) sampling, glowing isolines, haze; optional height (3D) mode | same machinery as 1 (a field drawn by colour), so it reuses it | **done** 2026-11-22: shared helper `domains/heatmap.py` (bicubic, isolines a pixel wide, glow weight, fade into the background), preset `prism` on all four types, prism looks use it, two-colour waves, `examples/prism_heat.py`. The 3D height mode exists already as `pde.Surface`; giving it the prism look is left for the PC session |
| 3 | **Point clouds.** A `PointCloud` object: millions of points splatted additively on the GPU with hue and glow (polynomial root clouds, attractors, Buddhabrot-style fractals); plus a cloud mode for `cx.Fractal` | the new GPU feature; needs a splatting kernel (CUDA atomics) | **done** 2026-11-22 on the PC: kernel `splat` (atomicAdd, tone curve 1 - exp(-k density), alpha = glow weight; a million points at 1080p in about 15-20 ms on the RTX 4070 Ti), `cloud.PointCloud` (sources `roots`, `attractor`, `buddhabrot`, `points`; presets `prism` and `plain`; the prism looks use it), `examples/point_cloud.py` (2^16 polynomials of degree 16, about a million roots, falling toward z = i), GPU golden made. The cloud mode for `cx.Fractal` is left for step 4, which uses `splat` and `cloud_math.buddhabrot`. **Less static** 2026-11-22 (owner: "some points move around, some white portions"): `twist` (animatable) turns one coefficient so every root glides and the ring tears open into fern-like tails; `twinkle` (life) flickers points; `white` runs the densest parts white-hot (`splat` tone curve, by white x a^4); `splat` now sums in fixed point (same bits every run); roots solve on all cores (a million in about 0.5 s); the example tears and heals the ring during the fall |
| 4 | **Fractals in the prism look.** Escape-time colour from the palette, glowing edges, haze; Julia and Mandelbrot | builds on 1 and 3 | **done** 2026-11-22 on the PC (except cloud mode): kernel `fractal_glow` (the edge's distance from the escape count's gradient, so the `deep` path glows too), preset `prism` (the prism looks use it), `examples/prism_fractal.py`, GPU golden. The cloud mode (`pc_tasks/04` item 3) waits for step 3's splat kernel |
| 5 | **Imports: pictures.** An `Image` object: any PNG/JPG (the Mona Lisa) placed, sized and animated like any object, with optional glow or a frame | small, and needed by 6 and 7 | **done** 2026-11-22: `Image` (`domains/media.py`), files next to the scene (`missing_file` if absent), presets `plain`/`framed`/`luminous`, the AI kept to its folder, `examples/picture.py` |
| 6 | **Imports: graphs.** A `Network` object loaded from a file (edge list, CSV, JSON, GraphML) or built in a script: 2D or 3D force layout computed once, glowing additive edges coloured by an attribute, thousands of nodes (a game tree of Connect 4 positions, say). Graph-theory types (Cayley graph and friends) gain the same look | big; the layout solver and edge batching are the work | **done** 2026-11-22: `nw.Network` (`domains/network.py`, `network_math.py`): files (edge list, JSON, GraphML), script edges, generators incl. every tic-tac-toe position (765 up to symmetry), radial/spring/layers/circle layouts in 2D/3D, faint cross links, one-call edge batches, exposure for dense graphs, `grow`, `examples/game_tree.py`. The spring layout is plain NumPy (slow past a few thousand nodes): a GPU version belongs with the PC tasks |
| 7 | **3D acoustics.** A 3D room (box, or a mesh imported from OBJ) drawn as faint glowing wire outlines, with sound as rays and expanding wavefronts that bounce off the walls (image-source and ray tracing, on the GPU where it pays) | biggest; uses 5 (meshes) and the 3D camera | **done** 2026-11-22 (PC): `ac.Room3D` (box, or a mesh from `.obj` via the new `Files` mechanism and lint `missing_file`), `ac.SoundRays` (specular ray tracing in NumPy: fast enough, no kernel), `ac.Wavefront3D` (fronts where they meet the walls, image sources), `examples/room_3d.py`, GPU golden `room_3d.png` |

After each item: a showcase scene in `showcase/` so the owner can watch it.

Steps 3, 4 and 7 run on the owner's PC, three Claude Code sessions at once: see
`pc_tasks/README.md` (one instruction file per step, a branch and folder each).

## Also decided (owner, 2026-11-22)

- **Prism is the default look** for new scenes (`look="none"` for plain).
- **Smoother palettes**: the six palettes run round a smooth curve in OKLab (no corners at
  their colours, no grey mixtures between them).
- **A little more transparency**: in the prism looks, glowing curves (`dim` 0.9) and colour
  fields (`dim` 0.88) let what is behind them glow through. Tune in `style/looks/prism*.toml`.
- Everything is prismed except `axes` and `image` (plain on purpose, owner 2026-11-22):
  Cayley tables, clock diagrams, matrices and rooms got a `glow` role; the 3D surfaces turn
  to glass in the prism looks.
