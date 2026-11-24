# Showcase

Scenes to watch, rendered on your own PC into `showcase/videos/` (not kept in the
repository; they are big). On an NVIDIA card the maths runs on the card and the video goes
through its encoder:

```bash
uv run python showcase/render_all.py
```

`--preset 4k` for 3840 x 2160, `--preset preview` for the quick size, or name scenes to
render only those: `uv run python showcase/render_all.py prism_primes deep_zoom`.

| Scene | What it shows |
|---|---|
| `prism_primes` | the prism look: primes heard through the zeta zeros, glowing rainbow strokes |
| `prism_domain` | domain coloring in the prism look: glowing zeros, poles, rays and rings |
| `prism_transitions` | transitions: a constant drifting, one function blending into another, equations solved |
| `prism_heat` | heat maps in the prism look: a hot spot, a vibrating drum, a chirp |
| `prism_spectrogram` | a spectrogram of a formula, the playhead a glowing line sweeping it |
| `sampling` | sampling and aliasing: samples of a fast wave as beacons, the slow wave they also fit |
| `prism_drum_3d` | a vibrating drum in 3D, a net of glowing lines, orbiting |
| `point_cloud` | a million polynomial roots as glowing points, falling toward z = i |
| `prism_fractal` | fractals in the prism look: a morphing Julia set, then a dive, edges glowing |
| `picture` | a picture placed in a scene: it fades in framed in light, a curve traces it |
| `game_tree` | every game of tic-tac-toe: its 765 positions fanning out move by move |
| `strange_attractor` | the Lorenz trajectory builds its two lobes under an orbiting camera |
| `residue_theorem` | a narrated short: the residue theorem, every motion on a spoken cue |
| `riemann_sqrt` | 3D: the Riemann surface of sqrt(z), the camera orbits on its own |
| `deep_zoom` | a dive into the Mandelbrot set down to 1e-30 |
| `motion_gallery` | the motion verbs: morphing equations, a tracer, a counter, a warped grid |
| `basics_3d` | 3D basics: axes, a helix, a surface, a vector field |
| `residues` | domain coloring of 1/(z^2 + 1) with a glowing contour |
| `fourier_linalg` | Fourier series and linear algebra |
| `heat_dihedral` | the heat equation and the dihedral group |
| `directivity` | beam steering, a narrowing piston pattern, and a cardioid |
| `room_acoustics` | a room, its echoes and the pressure wave |
| `doppler` | wavefront compression, the sonic pileup, and the Mach cone |
| `room_3d` | a 3D room with sound bouncing in it: rays and fronts reflecting, reverberation building |
| `buddhabrot` | the Mandelbrot set's cloud mode: eight million escaping orbits glowing like a nebula |
| `bifurcation` | the logistic map's bifurcation diagram as glowing points: the period doublings, chaos, the window of period three |
| `interference` | two sources interfering, then four in a line steering a beam |

Narrated scenes have sound only where a speech engine is installed (espeak-ng); without one
the timing is estimated and the video is silent.
