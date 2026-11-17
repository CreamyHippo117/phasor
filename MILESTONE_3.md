# Milestone 3: GPU beauty (in progress)

Roadmap row: kernel registry, `CopyBridge`, tiling, domain coloring, Mandelbrot/Julia, HDR,
bloom, tone map, encode chain, first theme, style roles. Exit demo: **domain-colored
1/(z²+1) with a glowing contour at 4K.**

Status: **the part marked for the strongest model (kernels and bridge) is built. The
milestone is NOT done:** the exit demo needs the remaining parts below, and nothing here
has run on CUDA.

## Built

| Area | Where | What |
| --- | --- | --- |
| Kernel registry | `render/gpu/registry.py` | `@kernel(name, signature, reference=...)`; fails without a NumPy reference with the same parameters; `run()` picks CUDA or the reference |
| Complex math header | `render/gpu/complex.cuh` | `cplx` with add/mul/div/exp/log/sqrt/trig/hyperbolic/powers; float or double |
| Expression codegen | `render/gpu/codegen.py` | `ComplexFunction.parse("1/(z**2+1)")` -> kernel C, NumPy function, LaTeX; whitelist parser, no `eval` |
| Tiles and budget | `render/gpu/budget.py`, `gpu.render` | every kernel takes a tile; plans keep one tile under 60% of free VRAM; supersampling per tile |
| `escape` kernel | `kernels/fractal.py` | Mandelbrot and Julia, smooth escape count, float32/float64 by the precision policy |
| `domain` kernel | `kernels/domain_coloring.py` | phase from a cyclic color table, modulus contours, poles white |
| `tonemap`, `gauss` | `kernels/post.py` | ACES curve; separable Gaussian (the building block of bloom) |
| Phase color wheel | `core/color.py` `phase_lut` | OKLCH circle at constant lightness: even brightness, no seam |
| Bridge | `render/gpu/bridge.py` | `FrameBridge` protocol, `CopyBridge` (RGBA16F), the only GPU-to-host transfer |
| CPU emulator | `tests/cuda_emu.py` | compiles the same CUDA source with g++ and runs the real launcher code |

## Verification

- `uv run pytest`: **326 passed, 10 skipped** (all 10 need CUDA). With OpenGL blocked:
  274 passed, 62 skipped. `ruff`, `lint-imports`: clean.
- **CUDA source, compiled for the CPU, against the NumPy references:** 9 kernel cases agree
  within 1e-4 on every pixel (fractals bit-for-bit), and 14 formulas through the generated C
  agree with SymPy's own NumPy function.
- References against known maths: membership of the Mandelbrot set, the Julia set of 0 is
  the unit disc, phase colors at 1, i, -1, contour brightness at known moduli, blur keeps
  energy and matches the Gaussian, tone map is monotone 0 to 1.
- Tiles stitch to exactly the single-tile image; 2x supersampling equals the 2x render
  averaged down.
- Bridge: a frame uploaded through `CopyBridge` reads back equal (float16) on software OpenGL.

## Not verified here

- **Nothing has run on CUDA.** NVRTC accepting the source, CuPy's argument passing, and GPU
  floating point are covered only by the 9 `gpu`-marked tests, which skipped. Run
  `uv run pytest -m gpu` on the NVIDIA machine before trusting the GPU path.
- The bridge has only uploaded NumPy arrays; the `cupy.asnumpy` step is one untested line.

## Deviations from SPEC

1. **Codegen uses its own emitter, not `sympy.ccode`.** `ccode` has no complex type; the
   emitter walks the SymPy tree and calls `complex.cuh`. One expression still feeds the
   kernel, the NumPy reference and the LaTeX label. (SPEC updated.)
2. **Formulas are parsed by a whitelist,** then built into SymPy, never `sympify`'d, because
   `sympify` evaluates text as Python. (SPEC updated.)
3. **Post-processing kernels take the whole image, not a tile.** A blur reads neighbors.
4. **Same branch as Milestones 0-2.** The session may only push to one branch.

## Remaining for Milestone 3 (a smaller model is fine)

- Scene object types for a domain-colored plane and fractals (object contract, one id).
- HDR color pass through the kernels: emission, bloom (threshold + `gauss` pyramid),
  `tonemap`, then the one sRGB encode.
- `render/encode.py` (probe order from SPEC), `phasor render`, `phasor bench`, `benchmarks/`.
- First full theme and remaining style roles (glow, surface); fractal coloring from the theme.
- The exit demo at 4K, its golden frames and storyboard snapshot.
- `InteropBridge` only if it measures faster (stretch).
