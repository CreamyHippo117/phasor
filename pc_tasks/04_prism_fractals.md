# Step 4: fractals in the prism look (on the PC, branch `beauty/prism-fractals`)

## Goal

`cx.Fractal` (Mandelbrot and Julia, including the deep zoom to 1e-280) in the prism look:
the inside dark, the escape-time colour from the palette (smooth, cycling), and the edge of
the set glowing with haze, so the boundary is light coming out of darkness.

## What to build

1. **Kernel `fractal_glow`** (or new options on `palette`, if that stays clean) in
   `phasor/render/gpu/kernels/fractal.py`: from the escape field (`escape` / `deep` output),
   - colour: the palette ramp indexed smoothly by `sqrt(count) * density + offset`,
     blended between entries (no banding);
   - glow weight (alpha): high near the boundary. Use the escape count's gradient per pixel
     (large where the field changes fast) or a distance estimate (`|z| log|z| / |dz|`, which
     needs the derivative carried in the iteration: add it to `escape` as an option and to
     `deep` if feasible); fall to `ambient` far outside;
   - inside the set: the background, not black paint, so it melts into the scene.
   NumPy reference for each; add to `CASES` in `tests/test_kernels.py`.
2. **`cx.Fractal`**: params `radius` (> 0 turns the prism look on, like `DomainColoring`),
   `ambient`; style `glow`; preset `prism`; the `prism` looks set it. Keep `steady` (the
   dive's held colour) working.
3. **Cloud mode** (after step 3 is merged): `cx.Fractal(kind="buddhabrot")` or a `render`
   option drawing escaping orbits with step 3's `splat` kernel.
4. **Example** `examples/prism_fractal.py`: a Julia set whose `c` moves slowly along a
   path (the set morphing), and a short Mandelbrot dive in the prism look.
5. **Tests**: the glow weight is high on the boundary and low inside and far out; the plain
   fractal is pixel-identical to before (check the existing `deep_zoom` and fractal goldens
   do not change); the prism preset glows (emission > 0); the deep zoom still matches exact
   iteration (the tests in `tests/test_deep_zoom.py` keep passing).
6. **Docs and showcase** as in the other tasks.

## Shared files (merge conflicts)

`tests/test_kernels.py`, `docs/ai/API.md`, `BEAUTY.md`, `showcase/*`, the prism looks
(`phasor/style/looks/prism*.toml`, the `[defaults.fractal.*]` tables only). Merge after step 3.

## Done when

Example renders on the RTX card, GPU goldens made and looked at, the deep zoom tests pass,
branch pushed, owner told it is ready to merge (second).
