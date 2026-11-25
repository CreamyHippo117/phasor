# 14. The Explorer panel (Milestone 29): how it feels on the RTX 4070 Ti

Milestone 29 put the Explorer into the desktop window (View > Explorer). Everything that can
be checked without a person at a screen was checked on the cloud box (Linux, 4 shared cores,
llvmpipe: OpenGL drawn by the CPU, Qt offscreen; `MILESTONE_29.md`). **Nobody has used it at
a real screen yet.** This task is that: typing, dragging, dropping, zooming and orbiting by
hand, with numbers the window measures itself. Write results at the bottom; do not change
code. If something fails, write down what you did and what happened.

Branch: the Milestone 29 branch (or `claude/eloquent-mayer-jiqxs9` once merged). Folder:
`C:\Users\Atlas\phasor-dev`. Close other heavy programs: the numbers are wall clock.

## A. Setup (5 minutes)

```powershell
git pull
uv sync --extra gl --extra encode --extra gui --extra mcp --extra gpu --extra boolean --extra svg --extra code
uv run phasor doctor
uv run pytest -q tests/test_explorer_panel.py tests/test_explorer_window.py
```

Record: pass/fail counts and the GL renderer `phasor doctor` names (it must be the NVIDIA
card). The two GL tests there ran on llvmpipe only.

Start the app: `uv run phasor-gui`, File > New > 2D scene, then **View > Explorer**
(Ctrl+Shift+E). The left pane shows the rows instead of the script. Save it at once
(File > Save As, `feel.phasor.json`): R2/R3 switching needs a saved file.

**Where the numbers are.** The transport line under the rows reads, after a few frames,
for example `tau = 3.200   1x normal   24.0 fps, draw 31 ms (moving), 88 ms (still), input
to picture 45 ms`: frames a second over the last two seconds, the median time to draw a
frame while moving (half size, thinned) and when still (full), and the median time from
handling a mouse input to the end of the first picture drawn after it. That last one is not
the whole latency (Qt's event queue, the paint and the screen come on top): judge the lag by
eye as well. Read them while doing each step.

## B. The steps (about 30 minutes)

1. **Type Lorenz.** Click the empty row at the bottom and type, at your normal speed,
   `x' = sigma (y - x)`; then in the next empty row `y' = x (rho - z) - y`, then
   `z' = x y - beta z`. Expected: nothing is committed while you type (no new sliders named
   `s`, `si`, `sig`); about 0.2 s after you stop, the row commits (1 s when it brings a new
   name, which gets a slider: try pausing half a second inside `sigma`; until the third row is
   in, the first ones say in red that the system needs its other rows: that is right);
   after the third row the
   view turns 3D (R3 lights up) and `sigma`, `rho`, `beta` sliders appear (value 1). An
   unfinished row (`x' = sin(`) shows a red message under it in plain words, never a dialog.
   Spaces typed in a row are text: they must never start or stop play (Space is also the
   Play shortcut).
   Record: did any keystroke feel delayed? did the cursor jump? did a row commit before you
   stopped?
2. **Load the preset instead** (Presets... > lorenz) so the numbers match the reference
   (sigma 10, rho 28, beta 8/3).
3. **Drag rho** from 28 down to about 24.74 and back. Expected: the picture follows the
   slider while you drag (half size, then sharp 150 ms after you stop); one undo step
   named `rho = ...` when you let go (Edit > Undo shows it; "Who changed what" under View).
   Record: fps and "draw (moving)" while dragging; does the picture lag the slider?
4. **Play.** Press play (or Space with the mouse over the picture), try speed `0.25`, `4`,
   `20` (Enter in the speed box), reverse, restart, `<<` and `>>`, left/right arrows, `r`.
   Expected: the readout says `x normal`; reverse runs the playhead back and stops at 0;
   none of this is an undo step. Record: fps while playing at speed 1 and at 20.
5. **Drop a ball.** Pause at tau 0. Shift+drag in empty space: the press is the centre,
   the drag the radius. Expected: the ball shows while you drag, one undo step "added ball"
   when you let go, and it appears in the list as `ball c = (...) r = ...`. Play: the ball
   folds onto the wings. Drag its centre (the ball's middle) to move it, its rim to resize
   it. Record: fps while playing with the ball; does dragging the handle feel attached to
   the mouse?
6. **Click C+.** Pause. Click one of the two red balls off the origin. Expected: the card
   under the rows reads `E.. = (8.485, 8.485, 27)  saddle-focus` (or the kind the analysis
   tab gives), its eigenvalues (one real negative, a complex pair with a small positive
   real part) and `dim Es = 1, dim Eu = 2`. Click empty space: a tracer appears where you
   clicked (on the plane through the orbit centre that faces you).
7. **Zoom in.** Double-click C+ (or press zoom in on the card). Expected: the camera flies
   there in half a second, then one undo step "zoom in to E.. (saddle-focus)" adds the
   close-up (the stable line and unstable disc). Record: was the flight smooth?
8. **Orbit with many tracers.** Undo the zoom. Use + row > ball; its points are its `count`
   parameter (400 by default), which the panel has no box for yet: turn View > Explorer off,
   click the ball in the picture to select it, set `count` in the Inspector tab, then turn
   the Explorer back on (or ask the attached AI to set it). Measure twice: with **1e4**
   points (count 10000) and with **1e5** (count 100000; a set takes at most 200000). For
   each: play to tau 5, pause, then drag
   to orbit for about ten seconds with the mouse down the whole time; also turn the wheel
   and let it rest (one undo step "camera zoom" after 0.4 s). Record: fps while orbiting,
   "draw (moving)", "input to picture"; does the picture keep up with the mouse; how long
   until the sharp picture after you let go.
9. **R2 / R3.** File > Save, then press R2. Expected: a second file `feel_r2.phasor.json`
   opens (Van der Pol the first time), with its own 2D camera; R3 brings back the Lorenz
   file as saved. Record: anything lost?
10. **Record and export.** On the Lorenz file: press record, mark, orbit a quarter turn,
    play to tau 5, mark, pause, drag rho to 24, mark, stop record. Expected: one undo step
    "recorded 3 marks"; the timeline shows `mark_1`..`mark_3`; Play in the code view plays
    the camera and rho eased between the marks. Then Export video... "from tau 0 to 30 in
    20 s", quality preview. Expected: the status bar follows it; the mp4 plays the flow.
11. **An AI at the same time.** In a terminal: `uv run phasor mcp --attach --root <the
    folder>` from Claude Code, and ask it to "set rho to 24.74 with explorer_edit" while
    you are typing in the rho row. Expected: it is refused (`row_locked`) until you leave
    the row; its other edits appear in the window within a tick and in the AI notes tab as
    `AI: ...`.
12. **Convert to script.** Press Convert to script. Expected: a warning that it is one-way
    for structure; then `feel.scene.py` opens, the rows stay but cannot be typed in, sliders
    still move (written into the script's tweaks block).

## C. What to record (expected vs measured)

| Step | Expected | Measured on the 4070 Ti | Feels right? |
| --- | --- | --- | --- |
| 1 typing | commits 0.2 s after the last key, no prefix sliders, no dialog | | |
| 3 drag rho | picture follows the slider, sharp 150 ms after; one undo step | fps: ... draw (moving): ... ms | |
| 4 play, speed 1 | smooth, `1x normal`; no undo step | fps: ... | |
| 4 play, speed 20 | still smooth (more flow per frame, same draw cost) | fps: ... | |
| 5 ball, playing | the ball folds onto the wings | fps: ... | |
| 5 handle drag | the ball's centre stays under the mouse | | |
| 6 card | C+ saddle-focus, a complex pair, dim Eu 2 | card text: ... | |
| 7 zoom in | half-second flight, one undo step | | |
| 8 orbit, 1e4 points | 30 fps or better while orbiting (estimate) | fps: ... draw (moving): ... input to picture: ... | |
| 8 orbit, 1e5 points | 15-30 fps while orbiting (estimate) | fps: ... draw (moving): ... input to picture: ... | |
| 8 sharp after release | about 150 ms plus one full draw | ... ms | |
| 9 R2/R3 | two files, each its own camera | | |
| 10 record, export | 3 markers, eased; the video plays the flow | | |
| 11 AI while typing | `row_locked`, then its edits appear live | | |
| 12 convert | warning first; rows read-only, sliders tweak | | |

The 30 fps and 15-30 fps lines are estimates (the design review's section 1.5 and
`pc_tasks/12_world_batches.md`), never measured on an NVIDIA card with the window's
overheads (a fork and a still per frame, the picture copied into Qt).

**What feels wrong** (the most useful part): anything that lags, jumps, surprises you, takes
more clicks than it should, or reads badly. One line each, in your own words.

## Results

(write here)
