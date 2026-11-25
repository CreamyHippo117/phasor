# Milestone 3: GPU beauty

Roadmap row: kernel registry, `CopyBridge`, tiling, domain coloring, Mandelbrot/Julia, HDR,
bloom, tone map, encode chain, first theme, style roles. Exit demo: **domain-colored
1/(z²+1) with a glowing contour at 4K.**

Status: **built; all gates green on Linux with software OpenGL and NumPy kernels.**
**Nothing has run on CUDA**: the 15 `gpu`-marked tests skipped. The GPU path is checked by
compiling the same CUDA source for the CPU (`tests/cuda_emu.py`), which proves the kernel
logic but not that NVRTC and CuPy accept it.

## Built

| Area | Where | What |
| --- | --- | --- |
| Kernel registry | `render/gpu/registry.py` | `@kernel(name, signature, reference=...)`; fails without a NumPy reference with the same parameters; `run()` picks CUDA or the reference |
| Complex math header | `render/gpu/complex.cuh` | `cplx` with add/mul/div/exp/log/sqrt/trig/hyperbolic/powers; float or double |
| Expression codegen | `render/gpu/codegen.py` | `ComplexFunction.parse("1/(z**2+1)")` -> kernel C, NumPy function, LaTeX; whitelist parser |
| Tiles and budget | `render/gpu/budget.py`, `gpu.render` | per-tile kernels within 60% of free VRAM; supersampling per tile |
| Kernels (7) | `render/gpu/kernels/` | `escape`, `palette`, `domain`, `bright`, `resize`, `gauss`, `tonemap` |
| Bridge | `render/gpu/bridge.py` | `CopyBridge` (RGBA16F), the only GPU-to-host transfer |
| Kernel-drawn objects | `domains/cx.py`, `DrawContext.field` | `cx.DomainColoring`, `cx.Fractal`: one id in the id pass, only the visible part computed, zeros/poles as features |
| Final look | `render/post.py`, emission target in `render/passes.py` | bloom pyramid from emitted light, shoulder tone map, then the one sRGB encode |
| Theme | `midnight_glow.toml` | glow, surface, colormap (OKLCH phase wheel, sequential) and bloom settings |
| Encoding | `render/encode.py` | SPEC probe order; MP4, ProRes `.mov`, PNG sequences |
| Commands | `cli.py`, `render/bench.py` | `phasor render` (presets preview/final/4k, `--frame`), `phasor bench` |
| Lint | `inspect/lint.py` | `contrast` measured on the real color pass; new `missing_glyph` |
| Approvals | ADR 0001, ADR 0002, id tombstones | accepted by the owner; schema v3 adds `retired_ids` |

## Verification

- `uv run pytest`: **350 passed, 17 skipped** (15 need CUDA, 1 CUDA doctor probe, 1 had no
  golden before this commit). Without OpenGL every non-GL test passes.
- `ruff`, `lint-imports`: clean.
- **Exit demo** (`examples/residues.py`): renders at 3840x2160 (`test_exit_demo_renders_at_4k`,
  marked slow: about 15 s here). The loop's rim reaches white, the plane beside it is lit by
  the bloom, and lint is clean with real pixels and without. The 4K still was inspected by eye.
- **Kernels, three ways:** every kernel's CUDA source compiled for the CPU matches its NumPy
  reference (fractals bit for bit, others within 1e-4); 14 formulas through generated C
  match SymPy; references are checked against known maths.
- **Picture correctness:** the color pass at a pixel equals the kernel at that point; a
  zoomed camera computes exactly the visible window; theme colors survive the tone map.
- Storyboard snapshot: `tests/golden/software/residues_storyboard.png`.
- `benchmarks/2026-11-18_reference_1920x1080.json`: NumPy timings on this machine.

## Bugs caught while building

1. **ACES tone mapping changed every theme color** (white text turned gray). Replaced by a
   shoulder curve that leaves paint alone (deviation 5).
2. **Thin glowing lines did not bloom at low resolution**: antialiasing spread their light
   under the threshold. Bloom now comes from a separate emission target.
3. **Nested `with ctx:`** in the bridge left no context current for the renderer.
4. **Field texture one pixel too tall** from rounding before a ceiling.
5. **The label placer fled the domain plane** (it covers the whole frame). Planes are now
   backdrops.
6. **`²` rendered as an empty box** with nothing to flag it: new `missing_glyph` lint rule.
7. The NumPy fractal took 9.5 s per 1080p frame; now 1.5 s (iterates only live points).

## Deviations from SPEC (SPEC updated in the same commit)

1. Codegen uses its own emitter, not `sympy.ccode` (no complex type there).
2. Formula text is parsed by a whitelist, never `sympify`'d.
3. Post-processing kernels take the whole image, not a tile.
4. Same branch as Milestones 0-2 (the session may push to one branch only).
5. **Tone map is a knee-and-shoulder curve, not ACES/AgX.** Paint below the knee is exact.
6. **Bloom comes from an emission render target**, not a threshold on the picture.
7. **The preview still has the final look** (bloom and tone map), so storyboards show what
   renders. Milestone 1-2 goldens were regenerated; they differ only near white.
8. **First theme only.** `aurora` and `paper` are Milestone 7 in the roadmap.
9. Not built: motion blur, adaptive supersampling, depth of field, grain, LaTeX, `InteropBridge`.
   Fields are supersampled 2x per side.

## Untested on hardware

- **CUDA:** `uv run pytest -m gpu` on the NVIDIA machine, then
  `PHASOR_UPDATE_GOLDEN=1 uv run pytest -k golden` to create `gpu`-class goldens (look at them).
- NVENC encoding; the CuPy path of `phasor bench`; Windows in general.

## Open

- Fresh-eyes boundary review not run.
