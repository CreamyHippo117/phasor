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
| `prism_heat` | heat maps in the prism look: a hot spot, a vibrating drum, a chirp |
| `point_cloud` | a million polynomial roots as glowing points, falling toward z = i |
| `prism_fractal` | fractals in the prism look: a morphing Julia set, then a dive, edges glowing |
| `residue_theorem` | a narrated short: the residue theorem, every motion on a spoken cue |
| `riemann_sqrt` | 3D: the Riemann surface of sqrt(z), the camera orbits on its own |
| `deep_zoom` | a dive into the Mandelbrot set down to 1e-30 |
| `motion_gallery` | the motion verbs: morphing equations, a tracer, a counter, a warped grid |
| `basics_3d` | 3D basics: axes, a helix, a surface, a vector field |
| `residues` | domain coloring of 1/(z^2 + 1) with a glowing contour |
| `fourier_linalg` | Fourier series and linear algebra |
| `heat_dihedral` | the heat equation and the dihedral group |
| `room_acoustics` | a room, its echoes and the pressure wave |

Narrated scenes have sound only where a speech engine is installed (espeak-ng); without one
the timing is estimated and the video is silent.
