# 11. Heat maps on the card (`GPU_SPEED.md` step 9c): what only the RTX card can check

Step 9c was built on a Linux box with no GPU. There the compiled kernel (the CPU backend and
the emulator) draws every heat map of the examples with the same bits as before. **Nothing
here has run on CUDA.** No code changes are expected; if a check fails, fix it on this branch.

```bash
uv sync --extra gl --extra encode --extra gpu
```

## 1. The kernel on the card

```bash
uv run pytest tests/test_kernels.py tests/test_heat_kernel.py tests/test_waves.py -m gpu -q
uv run pytest tests/test_beauty.py tests/test_pde.py tests/test_acoustics.py tests/test_fourier.py -q
```

`test_the_card_draws_the_reference_within_a_colour_step` is the one to read: the kernel is
written in double, so it should differ from NumPy only in the last bits of `tanh`, `exp` and
`pow`. If NVRTC refuses the source (a `struct` passed by reference to `__device__`
functions is the newest thing in it), say what it printed.

## 2. Pictures: finals before and after

`pc_tasks/pin_frames.py` on `prism_heat`, `interference`, `room_acoustics`,
`prism_spectrogram` and `heat_dihedral`, `final` preset, before (the PR's base) and after.
**The hashes must be equal** (owner, 2026-11-26: a change that is not bit-identical is for
previews only). Then look at one frame of each.

If any hash differs: count the differing 8-bit values and find the first stage they appear
in (the kernel's float32 pixels against its reference: `tests/test_heat_kernel.py`'s `gpu`
test prints the largest error). Do not merge for `final` until it is equal: either the
difference is removed, or `heat_shade` on CUDA is used only for `draft` and `preview` (a
table in `render/job.py`, like `FIELD_SCALE`) and finals keep the NumPy path on CUDA.
Report what you found.

## 3. Speed: the numbers for `GPU_SPEED.md`'s Status table

Before and after, serially, nothing else running:

```bash
uv run phasor bench
uv run phasor bench --frames prism_heat --count 20 --preset final
uv run phasor bench --frames interference --count 20 --preset final
uv run phasor bench --frames room_acoustics --count 20 --preset final
uv run phasor bench --frames prism_spectrogram --count 20 --preset final
```

Read `heat_shade` and `heat_shade:wave` in the kernel bench first. The kernel does all its
arithmetic in double; consumer cards are much slower in double than in float. Expected a few
milliseconds per 1080p frame. **If it is above about 20 ms**, the next step is a float32
variant for everything but the contour-line distance (say so in `GPU_SPEED.md`; do not
change the kernel without comparing pictures).

Before 9c (`b579d7e`): prism_heat 1232 ms/frame, interference 1895, room_acoustics 361.

## 4. What the survey found next (`GPU_SPEED.md` "Step 10"): measure, do not build yet

Each needs the card to measure and the owner's go-ahead. For `prism_fractal`, `prism_domain`
and `buddhabrot`, profile one `final` frame (`cProfile` round
`render_file`) and report the time in: `cupy.asnumpy` / `to_host`,
`numpy.ascontiguousarray` (the float16 conversion in `bridge.CopyBridge.upload`),
`Texture.write`, and the kernels themselves.
