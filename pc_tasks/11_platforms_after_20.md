# 11. Platforms after Milestone 20: a full Windows pass, and macOS

What the cloud box cannot do. Short and exact; write results at the bottom.

## What Windows CI already covers (GitHub runner, no GPU, no OpenGL)

The whole suite runs there on every push. Everything passes except 4 tests known to fail on
that runner (not caused by Milestones 14-20; listed in `09_pc_checks_after_14_15.md` step 6):

- `tests/test_remote.py::test_strangers_and_private_methods_are_refused` (`icacls` lists 3
  entries, the test expects 1)
- `tests/test_texconv.py::test_text_and_equation_report_their_latex`
- `tests/test_beauty.py::test_picture_golden_storyboard` (no OpenGL)
- `tests/test_final.py::...[room_acoustics]` (an `unsolved` label in analytic mode; also on
  the base commit)

Milestone 20's new tests (`test_sound.py`, `test_manim_parity.py`, `tests/manim_real`) are
written to run there: no POSIX-only calls, WAV files made with the stdlib, OpenGL tests marked
`gl`, the Manim comparison skipped without `PHASOR_MANIM_PYTHON`. Untested on Windows until
CI or this task runs them.

## A. Windows full pass (the RTX PC), `C:\Users\Atlas\phasor-dev`

```powershell
git pull
uv sync --extra gl --extra encode --extra gui --extra mcp --extra gpu --extra boolean --extra svg --extra code
uv run phasor doctor
uv run pytest -q                                   # with OpenGL and CUDA: compare with the 4 above
uv run pytest -q tests/test_sound.py tests/test_manim_parity.py tests/test_manim_import.py
uv run phasor render examples/sound_cue.py -o sound_cue.mp4
uv run phasor manim examples/manim_camera_tour.py -o tour\tour.phasor.json
uv run phasor render tour\tour.phasor.json -o tour.mp4
```

Only a real machine can check, so write these down:
1. `sound_cue.mp4` and `tour.mp4` play with the chime **heard** (at 1.3 s, and when the dot sets
   off in the tour, about 4.5 s) and in sync with the picture.
2. The sound came with the saved scene: `tour\assets\chime.wav` exists.
3. TESTING.md B1 (without OpenGL) and B2 (paths with spaces, a non-English user name, CRLF):
   still "to do" there; run them now if time allows.
4. Then `pc_tasks/10_manim_real.md` (Manim on Windows).

## B. macOS (any Mac; untested so far)

phasor cannot render on a Mac today: Apple stops at OpenGL 4.1 and phasor asks for 4.3. The
Milestone 14.4 investigation (`MANIM_PARITY.md`, "14.4 findings") found that the renderer uses
no 4.3 feature: with every shader retagged `#version 410` and a 4.1 core context, Mesa gave
byte-identical frames. Lowering the requirement is an owner decision (it changes the tier rule
in SPEC and CLAUDE.md); nothing was changed.

What a Mac can run now (no rendering):

```bash
git clone <repo> phasor && cd phasor
uv sync --extra encode --extra mcp --extra boolean --extra svg --extra code
uv run phasor doctor            # expect tier C (no OpenGL 4.3)
uv run pytest -q                # gl tests skip themselves; write down any other failure
```

If the owner says yes to 4.1, then on the Mac: `uv sync --extra gl`, `uv run phasor doctor`
(expect tier B) and `uv run pytest -q -m gl`, then compare `tests/golden/software` stills
(`uv run pytest -k golden`). Until that has run: untested on macOS.

## Results

### A. Windows, 2026-11-23 (RTX 4070 Ti, Windows 11, commit `be6d2c0`)

`phasor doctor`: **tier A** (CUDA 4070 Ti, NVRTC 12.9, OpenGL 4.3, PyAV with `h264_nvenc`; no
system ffmpeg, not needed). Worktree `C:\Users\Atlas\phasor-m20` (so `phasor-dev` stayed on
its branch).

**Full suite** (OpenGL + CUDA): **2 failed, 2352 passed, 42 skipped** (22 min). None of the 4
CI-runner failures above fail here. The 2 failures are new, both fail again when run alone,
neither is a library bug:

1. `tests/test_cpu_backend.py::test_a_library_that_builds_but_will_not_load_falls_back_and_is_not_retried`:
   `DID NOT WARN`. The test patches `cpu.ctypes.CDLL`, which is the shared `ctypes` module;
   with a CUDA card `auto` picks CUDA, and CuPy's NVRTC loading (`compiler._get_nvrtc_version`)
   hits the patched `CDLL` and raises the fake `libgomp` error. Passes only where CUDA is
   absent (CI, the cloud box). Fix in the test: force the CPU path (hide CUDA / patch only
   the backend's own loader).
2. `tests/test_desktop.py::test_the_start_screen_lists_recent_scenes_and_the_examples`:
   the start screen shows `examples/manim_camera_tour.py`'s docstring, which says
   "uses what Milestone 20 carries over"; the test forbids project notes there. Fix: reword
   that docstring.

**M20 tests** (`test_sound.py`, `test_manim_parity.py`, `test_manim_import.py`): 127 passed,
3 skipped (2 GPU goldens not made yet, 1 symlink test: no symlink privilege).

**Renders**: `sound_cue.mp4` (90 frames) and `tour.mp4` (360 frames), both `h264_nvenc`
"with sound".
1. Sound: measured, not listened to (onset of the AAC track, 10 % of peak): chime at
   **1.32 s** in `sound_cue.mp4` (cue 1.3 s) and **4.52 s** in `tour.mp4` (about 4.5 s);
   audio and video lengths 3.02/3.00 s and 12.02/12.00 s, both streams start at 0, so in
   sync. Someone should still play them once to hear the chime.
2. `tour\assets\chime.wav` exists after `phasor manim`: yes.
3. TESTING.md B1 without OpenGL: not run (no switch to hide OpenGL on this PC). B2: run, below.
4. `10_manim_real.md`: done, 11 passed (results there).

**B2 (Windows paths)**, run by hand:
- Spaces + non-ASCII folders (`phasor b2\Ünïcødé dir ñ\`, output `видео ñ\`), backslash
  paths in PowerShell: lint, inspect, storyboard, render with sound, `phasor manim --code`,
  GUI export (and cancel): **pass**, no leftovers.
- CRLF scenes (and a `.py` with BOM + CRLF): **pass**, the render is byte-identical to LF.
- `%APPDATA%` settings (`gui/app.py:44`): **pass**, including a non-ASCII APPDATA.
- Non-English user name, **simulated** (USERPROFILE/APPDATA/LOCALAPPDATA/TEMP/TMP pointed at
  `Users\Jürgen Ñoño Ψ\`; no real account):
  - **FAIL, CUDA**: any non-ASCII `%TEMP%` (even "Jürgen") breaks CuPy's first compile of a
    kernel: `CompileException: Catastrophic error: cannot open source file
    "...JÃ¼rgen..."` (NVRTC reads CuPy's temp `.cu` path in the wrong encoding). The render
    exits 1; no fallback from `registry.py:159` (`post.py:479`, `accumulate`). Only uncached
    kernels, i.e. a fresh install. A non-ASCII `CUPY_CACHE_DIR` alone is fine.
  - **FAIL, CPU kernels (MinGW g++)**: a `%TEMP%` or kernel cache path with characters outside
    the ANSI code page (Ψ; Greek, Cyrillic, CJK names) fails to assemble/link
    (`cpu/backend.py:394`); Latin-1 names work. `auto` falls back to NumPy with a warning
    (slow but right). The OpenMP probe (`backend.py:195-211`) then caches `openmp: false`
    for good, so kernels stay single-threaded after the path is fixed.
  - LaTeX (MiKTeX) with such a `%TEMP%`: pass.
- Temp files still open: every `mkstemp` is closed before use; GUI export log, TeX temp
  folder, settings and simulation-cache writes: pass. **FAIL**: rendering with sound while a
  player holds the output open: `[WinError 5] Access is denied: '... .mux.mp4' -> '... .mp4'`
  (`render/encode.py:255`, `mux_audio`), exit 1, leaves the `.mux.mp4`, the silent video is
  already written, and the message does not say to close the player.
- Smaller: a `.phasor.json` saved with a UTF-8 BOM (Notepad) fails to load
  (`core/io.py:46`, `:54`, `encoding="utf-8"`; `utf-8-sig` would read it); a BOM settings
  file is treated as damaged and reset (`gui/app.py:74`). Test-only: `tests/cuda_emu.py:44`
  makes a `phasor_emu_*` temp folder on import and never removes it (133 in `%TEMP%`).

### B. macOS

Not run: no Mac here.

### Fixed the same day (uncommitted until the owner says so)

- The 2 suite failures: `test_cpu_backend` now hides CUDA itself (`registry.default_gpu`);
  the start screen also drops a trailing "(Milestone N)" / "(BEAUTY.md ...)" aside
  (`gui/app.py`, `_NOTE_ASIDE`; `sound_cue` and `vector_toolkit` had one), and
  `manim_camera_tour.py`'s docstring no longer says "Milestone 20".
- Non-English user name: `phasor/core/proc.py` `ascii_path` / `ascii_temp_env` /
  `ascii_tempdir` give NVRTC (via CuPy's temp folder) and g++ (its file arguments and
  `TEMP`/`TMP`) the 8.3 short name of a non-ASCII path. Checked with the simulated user
  (`Jürgen Ñoño Ψ`): a fresh CuPy cache compiles and renders (without the fix: the same
  `cannot open source file`), CPU kernels and the OpenMP probe build in a `Ψ` cache. Also,
  `auto` now falls back (warning once per kernel) when a kernel does not compile for CUDA,
  instead of stopping the render; `backend="cuda"` still raises.
- OpenMP probe: a compiler that fails for another reason (the probe also fails without
  `-fopenmp`, or does not run) is no longer saved as `openmp: false`.
- Sound render into a file a player holds: a clear message ("another program (a video
  player?) has it open. Close it and render again") and no `.mux.mp4` left.
- BOM: `.phasor.json`, overrides and the GUI settings read `utf-8-sig`.
- `tests/cuda_emu.py` removes its temp folder at exit.
- New tests: `tests/test_windows_robust.py` (8).
- Full suite after the fixes (OpenGL + CUDA): **2362 passed, 0 failed, 42 skipped** (20.5 min).
