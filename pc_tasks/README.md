# Tasks for Claude Code on the owner's PC

**Back at the PC? Start with `09_pc_checks_after_14_15.md`**: short checks for what the cloud box could not run.

Steps 3, 4 and 7 of `BEAUTY.md` need the RTX card (GPU kernels, GPU reference pictures,
fast renders), so they are done on the PC. Each has its own instruction file here, written so
**three sessions can work at the same time** without touching each other's work.

| File | Step | Branch | Folder (git worktree) |
|---|---|---|---|
| `03_point_clouds.md` | 3. Point clouds: millions of glowing points, splatted on the GPU | `beauty/point-clouds` | `C:\Users\Atlas\phasor-clouds` |
| `04_prism_fractals.md` | 4. Fractals in the prism look | `beauty/prism-fractals` | `C:\Users\Atlas\phasor-fractals` |
| `07_acoustics_3d.md` | 7. A 3D room with sound bouncing in it | `beauty/acoustics-3d` | `C:\Users\Atlas\phasor-room` |
| `cpu_workers.md` | Milestone 14.3: how many render workers the RTX card wants | `perf/cuda-workers` | `C:\Users\Atlas\phasor-workers` |
| `08_wave_fields.md` | Interference, beamforming, bifurcation (`ANIMATIONS.md`); after 3 and 4 | `anim/wave-fields` | `C:\Users\Atlas\phasor-waves` |
| `09_pc_checks_after_14_15.md` | Milestones 14 and 15: Windows CPU backend, GPU kernels, Riemann timing, GPU golden for `open_contract`, decisions | `claude/eloquent-mayer-jiqxs9` (no new branch) | `C:\Users\Atlas\phasor-dev` |
| `11_heat_kernel.md` | Render speed step 9c: heat maps drawn by the `heat_shade` kernel; checks and measurements only the card can give | the PR's branch (no new branch) | `C:\Users\Atlas\phasor-dev` |
| `10_render_speed.md` | Render speed: measure before/after on the card (`GPU_SPEED.md` steps 7-8); measurements only, no code | `perf/measure` | `C:\Users\Atlas\phasor-speed` |

## Starting the three at once (PowerShell)

Each task works in its own copy of the project (a git *worktree*: same history, separate
folder and branch), so the sessions never edit the same files on disk. Once:

```bash
cd C:\Users\Atlas\phasor-dev
git pull
git worktree add ..\phasor-clouds -b beauty/point-clouds
git worktree add ..\phasor-fractals -b beauty/prism-fractals
git worktree add ..\phasor-room -b beauty/acoustics-3d
```

Then open three PowerShell windows, one per folder, and in each:

```bash
uv sync --extra gl --extra encode --extra gui --extra mcp --extra gpu
claude
```

and give it: **"Read CLAUDE.md, then pc_tasks/03_point_clouds.md, and do that task."**
(04 or 07 in the other windows.)

## Order of merging

The three branches touch a few shared files (`docs/ai/API.md`, `BEAUTY.md`, the showcase
list, `tests/test_kernels.py`). Merge them back into `claude/funny-ride-v6qgsf` one at a time,
**3, then 4, then 7**, resolving those small conflicts as you go (each task file says how).
Step 4's "cloud mode" uses step 3's splat kernel: do that part after step 3 is merged.

## Rules every task follows (from CLAUDE.md, in short)

- Keep `prism_primes`' look exactly; the new looks are presets, used by the `prism` looks.
- Glow comes only from emission. A kernel picture glows through its **alpha channel** (a
  per-pixel glow weight) and the object's `glow` role: see `DrawContext.field` and
  `cx.DomainColoring`'s `prism` preset (`domain_glow` kernel) for the pattern.
- A new GPU kernel is a CuPy `RawKernel` in `phasor/render/gpu/kernels/` with a NumPy
  reference, added to `CASES` in `tests/test_kernels.py`, tested three ways (reference
  maths, CPU emulator, real GPU).
- Every new type: docs in `docs/ai/API.md`, an example in `examples/`, a golden storyboard
  (software golden from Linux CI later; **GPU golden made here**, looked at before commit),
  a showcase entry.
- Before pushing: `uv run ruff check .`, `uv run ruff format --check .`, `uv run lint-imports`,
  `uv run pytest -q`. Push the task's branch, then tell the owner it is ready to merge.
