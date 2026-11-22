# Plan: signal processing, waves and dynamical systems

The owner's list of 2026-11-22: what has no animation yet in these three fields, what each
would be, where it gets built (**here**: plain NumPy, built in a cloud session; **PC**: needs
the GPU, built by Claude Code on the owner's PC, see `pc_tasks/08_wave_fields.md`), and its
prism version. Every item follows the usual rules: maths in `domains/<name>_math.py`,
the object in `domains/<name>.py`, docs, an example, tests, a golden storyboard, a showcase
scene. Prism versions are presets that the `prism` looks set (prism is the default look).

## Already there (for reference)

Fourier series (`Epicycles`, `PartialSum`), spectra with windows and leakage (`Spectrum`), a
chirp's spectrogram, waves on strings and drums with their modes, the heat equation, a 2D
room (pressure field, echoes, wavefronts), vector fields with flow lines and particles,
eigenvectors of a matrix, a grid being transformed.

## Signal processing

| # | What | Where | Prism version |
|---|---|---|---|
| S1 | **Live convolution**: the kernel flipped and sliding along the signal, their product shaded under the curve, its area tracing the output point by point | here | the two signals as glowing strokes in two palette hues, the overlap glowing where they multiply, the output drawn as light as it is made |
| S2 | **Fourier transform, the winding picture**: the signal wound round a circle at a test frequency, its centre of mass marked; sweeping the frequency traces the spectrum, which spikes where the winding balances | here | the winding as a glowing curve with hue along it, the centre of mass a beacon, the spectrum drawn as light beside it |
| S3 | **Filtering**: a filter's poles and zeros on the plane, its frequency response (gain and phase), and a signal before and after, with its spectrum losing the cut band | here | pole-zero plot over prism domain coloring of the transfer function; response curves glowing; the removed part of the spectrum fading out |
| S4 | **Spectrogram of any signal**: from a formula or a `.wav` file next to the scene, the playhead moving along | here | prism heat map (already built), the playhead a glowing line |
| S5 | **Sampling and aliasing**: samples taken from a fast wave, the slow wave they also fit | here | the true wave and its alias in two hues, the samples as beacons |

## Wave propagation

| # | What | Where | Prism version |
|---|---|---|---|
| W1 | **Interference**: two or more point sources, the field they make (crests, troughs, the dark lines where they cancel), animated in time | PC (a per-pixel field kernel) | the field as a prism heat map: crests warm, troughs cool, the quiet lines dark, haze |
| W2 | **Beamforming**: a line or ring of sources with phase delays steering the beam; change the delays and the beam swings | PC (the same kernel, with phases) | as W1, plus the array as glowing beacons and the beam direction as a ray |
| W3 | **Directivity**: polar patterns (an array's beam pattern, a piston, a cardioid) | here | a glowing polar curve, side lobes fainter, the main lobe brightest |
| W4 | **Doppler**: a moving source leaving bunched and stretched wavefronts; at sound speed, the shock cone | here | wavefront circles as fading glowing rings, hue by frequency heard |
| W5 | **3D room acoustics** | PC (beauty step 7, `pc_tasks/07_acoustics_3d.md`) | faint glowing walls, rays of light bouncing |

## Dynamical systems

| # | What | Where | Prism version |
|---|---|---|---|
| D1 | **Phase portrait** (done: `dyn.PhasePortrait`): a system x' = f(x, y), y' = g(x, y) from formulas, trajectories traced from seeds over time, the nullclines drawn | here | trajectories as glowing strokes with hue along time, nullclines faint, the flow field in dense small arrows |
| D2 | **Fixed points and stability** (done: `dyn.FixedPoints`): fixed points found, the Jacobian's eigenvalues computed, each classified (saddle, node, spiral, centre; stable or not) with its eigen-directions drawn | here | fixed points as beacons coloured by stability, eigen-directions as rays of light, a label with the eigenvalues |
| D3 | **Bifurcation diagram**: the logistic map (or any map) over a parameter, the period doubling into chaos | PC (millions of points: the point-cloud splat of step 3) | the classic picture as a glowing point cloud |
| D4 | **Strange attractors**: the Lorenz system traced as a moving trajectory in 3D (and Rossler, Thomas) | here (the curve); PC for the dense cloud version | a glowing 3D curve with hue along time, the camera orbiting |
| D5 | **Linear systems and their modes**: x' = A x, the eigenvectors as invariant lines, trajectories, the matrix exponential acting on a grid | here (reuses `EigenLines`, `TransformGrid`) | glowing, as above |

## Order

Here, in this order: **D1, D2** (phase portraits and stability, the core of the field), **S1,
S2** (convolution and the Fourier transform), **S3, S4, S5**, **W3, W4**, **D4, D5**.
On the PC, after the beauty steps 3, 4 and 7: **W1, W2, D3** (`pc_tasks/08_wave_fields.md`).
