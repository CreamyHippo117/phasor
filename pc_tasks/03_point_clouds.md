# Step 3: point clouds (on the PC, branch `beauty/point-clouds`)

## Goal

A `PointCloud` object: hundreds of thousands to millions of points, each a tiny glowing
splat, added together so dense regions build up light and fine structure appears (the
owner's reference: root clouds of polynomials where fractal edges glow in many colours). In
the prism look: dark ground, hue from the palette, haze.

## What to build

1. **Kernel `splat`** (`phasor/render/gpu/kernels/splat.py`): input points in view
   coordinates (x, y), a per-point colour (linear RGB) and weight; output an (h, w, 4) float32
   image where each point adds a small Gaussian (radius ~0.7-1.5 px) into the pixels around
   it with `atomicAdd`. Then a tone curve per pixel, `1 - exp(-k * density)`, so dense areas
   saturate softly instead of clipping. Alpha = the glow weight (density-based).
   - NumPy reference: `np.add.at` over the same footprint.
   - The CPU emulator (`tests/cuda_emu.py`) runs threads one after another, so give its
     `PRELUDE` an `atomicAdd(float*, float)` that just adds and returns the old value.
   - Tile the output like the other kernels; points outside the tile are skipped.
2. **Point sources** in `phasor/domains/cloud_math.py` (pure NumPy):
   - `polynomial_roots(degree, coefficients=(-1, 1), count)`: roots of all (or a random
     sample of) polynomials with coefficients from a small set. Colour by the position of
     the root's polynomial in the sample, or by the root's phase.
   - `attractor(kind, n)`: Clifford / de Jong / Lorenz (projected) iterates.
   - `buddhabrot(n, max_iter)`: escaping orbits of random c (needed later by step 4).
   - Randomness only through `rng(object_id, seed)` (contract 3).
3. **Object `PointCloud`** (`phasor/domains/cloud.py`): params `source` (`roots`,
   `attractor`, `buddhabrot`, or `points` given directly), the source's options, `count`,
   `pos`, `size`, `point_size`, `exposure`, `hue` (turns the palette); style `colormap` and
   `glow`. Draw through `ctx.field(...)` with the `splat` kernel computing only the visible
   part (zooming in keeps fine detail). Cache generated points per params.
   Presets: `prism` (dark, palette hue, glow `strong`) and `plain`. The `prism` looks use it.
4. **Example** `examples/point_cloud.py`: the roots of every polynomial of degree 12 with
   coefficients +-1 (4096 polynomials, ~49k roots) or more, zooming slowly toward the unit
   circle where the fractal edges are.
5. **Tests** (`tests/test_cloud.py`): the reference against `np.add.at`; total light
   conserved before the tone curve; a single point lands where it should; the three-way
   kernel tests through `CASES`; roots are real roots (|p(z)| small); determinism (same
   seed, same picture); speed on the GPU (a million points in well under a second at 1080p).
6. **Docs and showcase**: `docs/ai/API.md` (object row, kernel row, a short runnable
   python block), `BEAUTY.md` status, `showcase/render_all.py` and its README.

## Shared files (merge conflicts)

You add lines to `tests/test_kernels.py` (`CASES`, the sorted kernel list), `docs/ai/API.md`,
`BEAUTY.md`, `showcase/*`, `phasor/domains/__init__.py`, `phasor/__init__.py`. Add, never
reorder, so merges stay simple.

## Done when

The example renders on the RTX card, GPU golden made and looked at, all tests pass, branch
pushed, and you tell the owner it is ready to merge (first of the three).
