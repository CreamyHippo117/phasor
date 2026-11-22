# Wave fields and dense dynamics (on the PC, branch `anim/wave-fields`)

From `ANIMATIONS.md`: the items that need the GPU. Start after beauty steps 3 (point clouds)
and 4 are merged, since D3 uses step 3's `splat` kernel and W1/W2 follow step 4's pattern for
a glowing field kernel.

## W1 Interference and W2 Beamforming

1. **Kernel `wave_field`** (`phasor/render/gpu/kernels/wave.py`): for each pixel, the sum
   over N point sources of `amp_k * cos(k |x - s_k| - w t + phase_k) / sqrt(|x - s_k|)`
   (2D spreading), output the instantaneous field and, optionally, its time-averaged
   intensity. NumPy reference; add to `CASES` in `tests/test_kernels.py`; three-way tests.
2. **Object `ac.Interference`** (`phasor/domains/acoustics.py` or a new `waves.py`):
   `sources` (positions), `frequency`, `phases`, `amplitudes`, `speed`, `time`, `mode`
   (`field` or `intensity`), `pos`, `size`. Drawn through `ctx.field` with the glow weight in
   alpha (see `cx.DomainColoring`'s prism preset and `domains/heatmap.py`): crests warm,
   troughs cool, the cancelling lines dark.
3. **Object `ac.PhasedArray`**: N sources on a line or ring, `steer` (degrees) setting the
   phase delays; draws the field (as W1), the elements as beacons, and the beam direction.
   Animating `steer` swings the beam. Its polar directivity (W3) is built in a cloud session.
4. **Tests**: two sources in phase give maxima on the perpendicular bisector; the nodal
   lines are where the path difference is a half wavelength; steering by theta puts the beam
   peak at theta (within a beamwidth); determinism; speed (1080p field well under 50 ms).
5. **Example** `examples/interference.py`: two sources, then four in a line steering a beam.

## D3 Bifurcation diagram

1. In `phasor/domains/cloud_math.py` (from step 3): `bifurcation(map, r0, r1, n_r,
   n_iter, n_keep)` returning points (r, x) for the last `n_keep` iterates at each r
   (logistic map by default, or a formula through the existing expression parser).
2. A `PointCloud(source="bifurcation", ...)` option drawn with the `splat` kernel.
3. Tests: the first period doubling at r = 3, the second near 1 + sqrt(6); chaos past
   ~3.5699; points match direct iteration.
4. Example `examples/bifurcation.py`: zoom into the first doublings and the window of
   period three.

## Shared files, done when

As in the other tasks (`pc_tasks/README.md`). Goldens made on the RTX card and looked at,
all tests pass, branch pushed, the owner told it is ready to merge.
