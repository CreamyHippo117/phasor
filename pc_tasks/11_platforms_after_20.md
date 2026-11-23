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
