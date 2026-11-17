# Milestone 0: Skeleton

Status: **built and green on Linux; exit demo (doctor on clean Windows + CUDA) NOT verified.**

## Built

- Repo layout from SPEC "Tech stack, Windows, repo layout"; `uv` project, Python ≥ 3.11.
- Extras: `gl` (moderngl), `gpu` (`cupy-cuda12x[ctk]`, which brings NVRTC from pip), `encode` (PyAV).
- `phasor` CLI: `doctor` implemented; `inspect`, `lint`, `storyboard`, `render`, `bench`, `mcp`,
  `gui` registered and exit 2 naming their milestone.
- `phasor doctor [--json] [--require A|B|C]`. Every probe exercises the capability:
  - CUDA device, VRAM, compute capability (CuPy); `nvidia-smi` fallback;
  - NVRTC version; compiles and runs a `RawKernel`, checked against NumPy;
  - standalone GL ≥ 4.3 context; renders a triangle into an RGBA16F target and an
    R32UI + depth id target, read back and counted with `np.bincount`;
  - encoders: PyAV `hevc_nvenc` / `h264_nvenc` / `libx264` and system ffmpeg, each by
    encoding real frames; picks one in the SPEC probe order;
  - hardware tier A/B/C.
- `render/glcontext.py`: the one way to create a GL context (EGL fallback on headless Linux).
- Gates: ruff (lint + format), import-linter (layer order + GPU-free lower layers), runtime
  test that lower layers don't load GPU libraries, `docs/ai` code blocks run as tests.
- CI: GitHub Actions, Ubuntu + Windows × Python 3.11/3.12, Mesa EGL on Ubuntu so GL tests run.
- `CLAUDE.md`, `docs/ai/{API,PATTERNS,PITFALLS}.md`, `STYLE.md` (template for you to fill),
  `docs/decisions/0001-layers.md`.
- Anti-drift tooling: `.claude/skills/phasor-milestone-gate` (boundary review ritual) and a
  Stop hook (`.claude/hooks/stop_gate.py`) that blocks finishing while ruff, import-linter or
  the fast tests are red. Verified: blocks on a planted lint error, steps aside on re-entry.

## Verification

| Check | Result |
| --- | --- |
| `ruff check`, `ruff format --check` | pass |
| `lint-imports` (2 contracts) | pass |
| `pytest` | 37 passed, 1 skipped (CUDA kernel test: no GPU here) |
| `phasor doctor` (this Linux container) | tier B: llvmpipe GL 4.5, libx264; NVENC correctly rejected |
| doctor with CuPy but no driver | reports the CUDA error, skips kernel checks, does not crash |
| **Exit demo: doctor passes on clean Windows + NVIDIA** | **not verified here: needs your machine** |

To verify on Windows:

```bash
uv sync --extra gl --extra gpu --extra encode
uv run pytest
uv run phasor doctor --require A
```

## Library checks (step 4 of the kickoff)

Checked against installed current versions: ModernGL 5.13, PyAV 19.0, CuPy 14.2,
pydantic 2.14, import-linter 2.16, kiwisolver.

- ModernGL: `create_context(standalone=True, require=430, backend="egl")` works headless;
  `dtype="u4"` (R32UI) textures render and read back. Matches the spec.
- CuPy: `RawKernel(code, name)`, `cuda.nvrtc.getVersion()`, `MemoryPool.free_all_blocks()`
  match the spec. Kernel execution not run (no GPU).
- PyAV: matches the spec, with the caveat below.
- **Not checked yet**, on purpose (checked at the milestone that uses them): WhisperX (M4),
  MCP Python SDK (M5), `cuda-python` GL interop for `InteropBridge` (M3 stretch).

## Contradictions and gaps found in SPEC

1. **NVRTC is not in the plain CuPy wheel.** "No C++ compiler" is true, but NVRTC must come
   from the CUDA Toolkit or the `cupy-cuda12x[ctk]` extra. *Fixed in SPEC* (non-contract text).
2. **Codec lists lie.** PyAV lists NVENC with no NVIDIA GPU present, so probes must encode.
   *Fixed in SPEC* (non-contract text).
3. **Tier C vs inspector.** Tier C promises a working inspector, but `visible_fraction`,
   `hidden_by` and occlusion need the GL `id`/`solo` passes. Proposal: tier C runs an
   analytic inspector (bounds, regions, relations; no occlusion), labelled as such.
   **Needs your decision.**
4. **Contract 6 gaps.** `style` and `engine` have no layer, and `layout` (label-placer
   occlusion, camera-solver verify) and `inspect` (visibility) need `render` output.
   Proposal in `docs/decisions/0001-layers.md`: style below time, engine above domains,
   and dependency inversion (protocols in lower layers, implemented in render, injected by
   Engine). Import direction is unchanged. Implemented in `pyproject.toml` provisionally.
   **Needs your approval** (contract change rule).
5. **Golden frames across machines.** SSIM ≥ 0.99 will not hold between llvmpipe (CI) and
   NVIDIA (your machine). Proposal: goldens are keyed by renderer class (`software`,
   `nvidia`), and CI compares only against `software`. Decide at Milestone 1.
6. **Stdlib name clash.** `phasor.time` and `phasor.inspect` shadow stdlib names if someone
   runs a script from inside `phasor/`. Harmless with normal imports; documented in
   PITFALLS. Rename only if you want to.

## Deviations

- Layer order includes `style` and `engine` positions not in contract 6 (see 4 above).
- `phasor doctor` lives in `phasor/render/doctor.py` (it is GPU code), called from `cli.py`.

## Deferred

- The Milestone 1 plan (file-by-file with test names): this is where the schema, Engine,
  parameter DAG and object contract get designed, which is the highest-leverage design
  work in the project. Deferred to a session on the strongest model, per owner instruction.
- Windows verification of the exit demo.
