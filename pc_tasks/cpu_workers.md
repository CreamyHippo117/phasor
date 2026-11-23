# Render workers on the RTX card (on the PC, branch `perf/cuda-workers`)

From `MANIM_PARITY.md`, Milestone 14.3. The tier B half is built (`scene.default_threads`:
without CUDA, workers x kernel threads <= cores, `phasor render --threads`). The CUDA half
needs the RTX 4070 Ti: today `scene.default_workers()` is a quarter of the CPU threads,
1 to 4, and the draw step is Python, so more processes may use the CPU better while the card
waits. Nothing about the CUDA default changes until this is measured.

## Measure

1. On `claude/eloquent-mayer-jiqxs9` after Milestone 14 is merged, nothing else running,
   the card idle. For `examples/motion_gallery.py` and `showcase/prism_primes.py`, preset
   `final` (PowerShell):

   ```powershell
   uv run phasor doctor                       # record the tier, card and VRAM free
   foreach ($w in 1, 4, 6, 8, 12) {
     (Measure-Command { uv run phasor render showcase/prism_primes.py --preset final `
        --workers $w -o "out_$w.mp4" }).TotalSeconds
   }
   ```

   Run each twice and keep the second: the first fills the kernel caches.
   Record the wall clock, the peak VRAM (`nvidia-smi --query-gpu=memory.used
   --format=csv -l 1` in a second window) and the CPU use (Task Manager, all cores) per count.
2. Check each video against the one-worker video: the same frames (`ffmpeg -i a -i b
   -lavfi psnr`: `inf`, or above 60 dB where lit surfaces differ by their last float16 bit,
   `GPU_SPEED.md` "Open items").
3. Ask `render/gpu/budget.py` what fits: each worker takes `1 / workers` of the VRAM budget
   (`set_share`). A count whose tiles no longer fit, or that swaps, is out.

## Then

- Raise the cap in `scene.default_workers()` (CUDA path only) to the fastest count that
  fits, as a rule from the hardware (for example threads // 2, at most the count that fits
  in VRAM), not a number for this one card. Keep the no-CUDA rule untouched.
- Save the numbers as `benchmarks/<date>_workers_cuda_<scene>.json` (the fields of
  `phasor bench --frames`, plus `workers` and `wall_s` per count) and add a table to
  `GPU_SPEED.md`.
- Test: `tests/test_workers.py` keeps passing; add one for the new rule (mock
  `os.cpu_count` and the budget).
- If no count beats 4 by more than noise (5%), change nothing and write that down.

## Shared files, done when

As in the other tasks (`pc_tasks/README.md`). Numbers measured on the RTX card only; all
tests pass; branch pushed; the owner told it is ready to merge.
