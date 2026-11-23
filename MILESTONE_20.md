# Milestone 20: Manim import parity and platforms

Plan: `MANIM_PARITY.md`, "Milestone 20". Built 2026-11-23 in a worktree branched from
`claude/eloquent-mayer-jiqxs9` (after the gate of 17 and 19). Exit demo:
**`examples/manim_camera_tour.py`, a Manim script using the camera frame, a filled shape, a
ValueTracker and an updater, `always_redraw`, `add_sound`, a morph, a color animation and a
number line, imports with no warning, and real Manim CE agrees on its clock, places, colors
and camera** (`tests/manim_real`, row `manim_camera_tour` below).

Status: **built; independent audit done 2026-11-23, its findings fixed (see "Audit");
focused gates green on Linux; real Manim compared here.** Windows, macOS and CUDA untested
(`pc_tasks/10_manim_real.md`, `pc_tasks/11_platforms_after_20.md`).

## Built

| Area | Where | What |
| --- | --- | --- |
| The stand-in, rewritten | `phasor/importers/manim.py` | a leaf keeps its outline (`pts`), its centre, scale, turn and whether it is still its own kind; every change of place goes through one `_affine`; a leaf becomes phasor's own kind (`shapes.Circle`, `Rectangle` ...) while it keeps its shape, else a `shapes.Path` (one way: `phasor_spec`) |
| Shapes and fills | same | `Circle`, `Ellipse`, `Arc`, `ArcBetweenPoints`, `Sector`/`AnnularSector`, `Annulus`, `Polygon`, `RegularPolygon`, `Triangle`, `Rectangle`, `Square`, `RoundedRectangle`, `Brace` (`get_text`, `get_tex`, `get_tip`), `DashedLine`, `DashedVMobject`, `VMobject.set_points_as_corners`; `fill_opacity`, `fill_color`, `set_fill`, `set_stroke`, `stroke_width=0` (`stroke_opacity` 0), `set_opacity` |
| Files | same | `SVGMobject` -> `shapes.SVG`, `Code(code_file=)` -> `dg.Code`, `add_sound` -> `media.Sound`: only files inside the scene's folder (`core/files.py`); the scene's `base_dir` is the script's folder; `phasor manim -o elsewhere/` copies them along (`Engine._bring_files`) |
| Text | same | `Tex` is text mode (was maths), `MathTex` with several strings an `Equation` (`eq[i].set_color` -> the part's role); `Write`'s run time from the glyph count as Manim (1 s under 15 glyphs, else 2 s; `\frac` counts its bar) |
| Morphs | same | `Transform`/`ReplacementTransform` between kinds -> the `transform` verb, `TransformMatchingTex`/`Shapes` -> `transform_matching` (linear, as Manim), played in Manim's run time by retiming the verb's segments (`scene.At.motion`: a verb's ops without applying them); text becomes `as_shapes` when it morphs; a shape stretched out of its kind by `.animate` morphs into a path |
| Color | same | `.animate.set_color`, `set_fill`, `Indicate` -> style tracks (`<id>.style.color`, `fill_color`), mixed in linear light; a rate function that comes back (`there_and_back`) comes back |
| Rate functions | same, `time/easing.py` | Manim's own (`smooth` is a logistic S, not smoothstep): new easing `sigmoid`; `linear`, `smoothstep`, `smootherstep`, `ease_*_cubic` map exactly; any other (`there_and_back`, `double_smooth`, `rush_into`, a lambda) becomes straight pieces within 0.2% of the motion (`RATE_TOLERANCE`; at most 64, Douglas-Peucker; was 16 even pieces, up to 8.9% off) |
| Diagram kit | same | `Axes` (`get_axis_labels` -> `labels`, `add_coordinates`/`include_numbers` -> `numbers`, `p2c`, `i2gp`), `NumberPlane`/`ComplexPlane` -> `dg.NumberPlane` (grid), `NumberLine` -> `dg.NumberLine`, `BarChart`, `Table`/`MathTable`, `Matrix`, `Code`, `DecimalNumber`/`Integer` -> `Counter`; style keys cut to the kit type's own |
| Trackers and updaters | same | `ValueTracker` -> `Value` with tracks; `add_updater` (with or without `dt`), `always_redraw`, `always`, `f_always`, `become`, `suspend_updating`: run by the stand-in frame by frame (`config.frame_rate`, 30 a second unless the script sets 1-120; Manim's frame times: the first frame of a play has `dt` 0 and the play's last frame is before its end, then a final `dt` 0 update), with every animated mobject and tracker where Manim has it; what changes becomes keys, simplified (Douglas-Peucker, 1e-4); bounded per import (`MAX_FRAMES` 20000, `MAX_UPDATER_CALLS` 2 million, `MAX_KEY_NUMBERS` 1 million sampled numbers: past half, fewer frames a second at Manim's frame times with `dt` adding up to Manim's; when spent, updaters stop, said with the time); `wait_until` / `wait(stop_condition=)` end at the first frame the condition holds, checked with Manim's clock a frame on |
| Camera | same | `MovingCameraScene`: `self.camera.frame` (`.animate.move_to`, `shift`, `scale`, `set(width=)`, `set_width`, `save_state`, `Restore`) -> tracks on `camera.center`/`camera.height` (width = height x 16/9); in a plain `Scene` (Manim has no `frame` there) it moves the camera too, with a warning (Deviation 6) |
| Other animations | same | `Unwrite`, `DrawBorderThenFill` (2 s, `double_smooth`), `GrowFromPoint`/`Edge`, `GrowArrow`, `ShrinkToCenter`, `Restore`, `ApplyMethod`, `Rotating`, `Circumscribe`, `Flash`, `Wiggle` (phasor's verbs, retimed), `FadeIn`/`FadeOut` `shift`/`scale`/`target_position`, `LaggedStartMap`; `AnimationGroup` stretches its schedule to a group `run_time` (it did not) |
| Sound | `domains/media.py`, `time/narration.py`, `engine.py`, `render/job.py` | `media.Sound(path, start, gain)`: a WAV in the scene's folder, drawless (`DRAWLESS` in lint, like `Value`), semantics `seconds` (what the file really holds)/`rate`/`unreadable`; `narration.parse_wav` (the one header walk: every size checked against the file), `wav_info`, `read_sound` (8-32 bit PCM, 1-16 channels, 8-192 kHz, chunked, at most `MAX_CLIP_FRAMES`), `Clip`; `mix_track` mixes narration and clips (narration's rate, else 48 kHz; clips cut where the scene ends; the track at most 3 hours; a chunk of 2^20 samples at a time on the scene's sample grid; at most `MAX_SOUNDS` 256 clips and `MAX_MIX_SAMPLES` 2^30 clip samples); `Engine.audio_track(problems)` (any type with `audio_clip`); `phasor render` mixes before the frames, lists left-out clips in its warnings and prints "with sound" |
| Lint | `inspect/lint.py` | `missing_file` and `bad_file` for a sound (the hint says PCM WAV); `sound_limit` (a clip past `MAX_SOUNDS`, or an animated `start`/`gain`); no `orphan` for drawless types |
| Real Manim | `tests/manim_real/` | `record_manim.py` (runs in Manim's own Python: plays, run times, rate functions, every top-level mobject at 30 frames a second, the camera frame), `compare.py` (phasor's import measured through the engine at the same times), `test_manim_real.py` (marker `manim_real`, skipped without `PHASOR_MANIM_PYTHON`), 8 scenes plus the two Manim examples |
| Docs, examples | `docs/ai/API.md`, `SPEC.md`, `examples/` | the Manim table rewritten; `Sound` in the types table and in "Narrate" (a running example); `examples/manim_camera_tour.py` (exit demo), `examples/sound_cue.py` (the type's example) |

### Every warning the importer could emit before (Milestone 13-19), and what became of it

| Warning before | Now |
| --- | --- |
| "filled shapes are drawn as outlines" (`fill_opacity=`, `set_fill`) | **deleted**: fills carry over (M16 `fill`, `fill_opacity`, `fill_color`) |
| "a NumberPlane is drawn as plain axes with ticks (no grid)" | **deleted**: `dg.NumberPlane` (M19) |
| "axis labels are not carried over: add Text next to the axes" | **deleted**: `geo.Axes(labels=)` (M19) |
| "ValueTracker animations are not carried over" | **deleted**: `Value` with tracks (M15) |
| "always_redraw / updaters are baked: the object keeps its first look" | **deleted**: run at Manim's frames, keyed (M15's rule kept: no foreign code in the document; see Deviation 1) |
| "a color change is instant here (style does not animate)" | **deleted**: style tracks (M17) |
| "add_sound is not carried over (use narration)" | **deleted**: `media.Sound` (M20) |
| `Transform` between different shapes crossfaded (documented, no warning) | **gone**: the morph verbs (M17) |
| `MovingCameraScene` frame animations silently did nothing | **gone**: camera tracks |
| "{size} is not carried over" (stroke or font size 0, negative, too large) | **kept** for negative / non-finite / too large; `stroke_width=0` is now `stroke_opacity` 0 with no warning (M16) |
| "color ... is not a hex color or Manim name: the nearest role is used" | **kept** (a color phasor cannot read) |
| "'...' could not be typeset" | **kept** (mathtext cannot read it) |
| "axes ... past the limit" (tick clamp) | **kept** (M19's limit), also for `NumberLine` |
| "the background's alpha ... is not carried over" | **kept** (M18: a background is opaque) |

New warnings, each for something phasor cannot express: rotated or stretched text (drawn
level); a turned camera frame; a `MathTex` part's color (nearest role); `BarChart` colors
(roles); a style key a kit type lacks; an updater that turns a shape into another kind;
`become` with other parts; `Axes.get_area`; `Matrix.get_entries`; a plot or curve with fewer
than two finite points; a file outside the folder or not there; a shape that cannot be its
own kind (drawn as a path or a line); a stretched rounded rectangle; the updater frame budget.

## Real Manim (2026-11-23, Linux, Manim CE 0.22.1, TeX Live 2023)

Installed in a throwaway venv outside the repo (`uv venv /tmp/claude-0/manim_env`,
`uv pip install manim`); its native build needed `apt-get install libpango1.0-dev
libcairo2-dev pkg-config`, and `MathTex` needed `texlive-latex-base texlive-latex-recommended
texlive-fonts-recommended texlive-latex-extra dvisvgm`. `PHASOR_MANIM_PYTHON=... uv run pytest
tests/manim_real`: **11 passed** (after the audit fixes, with the new `wait_until` scene). Tolerances (`compare.py`): shapes, dots, lines and the camera
frame 0.05 world units (of the 8-unit frame); text and diagram composites 0.35 (phasor's own
fonts); the plain look (the prism look's living motion moves things by design); a sample where
either side has drawn nothing yet is skipped; Manim's internal groups (the camera's
`ScreenRectangle`, a `ValueTracker`, an `AnimationGroup`'s `Group`, a `TransformMatching*`'s
`VGroup`) are not the script's.

| Scene | Matched | Differed (why) |
| --- | --- | --- |
| shapes_layout | clock; every mobject at every sample; positions within 0.008; colors (2 explicit the same, 7 class defaults as roles) | sizes 0.25: Manim's arrow tip is wider (0.35 against phasor's 0.1); phasor's boxes include the stroke |
| animate_timing | clock (8 plays: run times and rate functions, `LaggedStart` 0.5, `Succession`, `Indicate`, `Rotate`); positions **0.000** outside drawing | sizes 0.10 mid-`.animate.scale(2).rotate(PI/4)`: Manim mixes points in a straight line (the square shrinks on the way), phasor turns it; during `Succession` Manim lists the square inside its group |
| fills_colors | clock; every mobject; positions 0.011, sizes 0.045; fills; the animated colors at the end (2 the same, 6 roles) | none (a dashed line's box, which ends at its last dash, is measured apart) |
| text_write | clock (`Write` 1 s, linear); every mobject; positions 0.000 | text 0.54: a small text put in a corner by its edge; phasor's text is narrower (Deferred 1) |
| axes_plot | clock; every mobject; positions 0.003, sizes 0.026 once drawn | none |
| math_tex | clock (`Write` linear 1 s, `TransformMatchingTex` linear); every mobject; positions 0.000; text 0.108 | none beyond fonts |
| moving_camera | clock; camera frame within 0.000 at every sample (`move_to` + `set(width=)`, `shift` + `scale`, `Restore`) | none (sizes 0.032: the stroke) |
| updaters | clock; positions **0.000** at every frame (a tracker-driven updater, `next_to` a moving dot, a `dt` rotation, `always_redraw`) | none (sizes 0.045: the stroke of a turned square) |
| manim_camera_tour (exit demo) | clock; positions 0.001; sizes 0.037; camera 0.000; colors (4 the same, 2 roles); text 0.104 | none |
| wait_until (audit) | clock (3.367 s: `wait_until` ends at 1 s, `wait(3, stop_condition=)` at 2.867 s, as Manim's own clock says); every mobject; positions 0.000; sizes 0.032 | none (the recorder now measures a stopped wait by Manim's clock, not its `run_time`) |
| manim_reverb (M13 demo) | clock (14.5 s); every mobject; positions 0.010; sizes 0.035; colors (5 the same, 4 roles); text 0.303 | none |

While a `Create`/`Write`/`DrawBorderThenFill` plays, a shape is partly drawn: phasor traces
from its own first point and by arc length, Manim from its first Bezier point by pieces (a
square: Manim's starts at the top right going left). Those samples are measured apart
(`drawing`, up to 1.0 mid-trace) and reported, not held to the tolerance; every other sample
of the same scene is.

Before this milestone (the same harness against the Milestone 13 stand-in): 3 scenes could not
import at all (`NumberLine`, `Ellipse`, `save_state` missing), one refused `add_updater`; the
clock differed in 4 (`Write` was `Create` with `smooth`; `LaggedStart`/`Succession`/
`TransformMatchingTex` rates; `Write` of 15+ glyphs is 2 s); `Tex` was maths; text positions
were off by up to 2.5.

### Stand-in bugs found by real Manim, and fixed

1. `YELLOW` (and other named colors) were wrong: `#FFFF00`, Manim's is `#F7D96F`; the table is
   now Manim's own (100 names).
2. `smooth` was smoothstep; Manim's is a logistic S (0.07 vs 0.16 at a quarter): new `sigmoid`.
3. `Write` was `Create` with `smooth`: Manim's is linear, 1 s or 2 s by glyph count.
4. `Tex` was `MathTex` (maths italic, spaces lost).
5. `Arrow` ignored Manim's `buff` (0.25 short of both ends).
6. `RegularPolygon` with an even `n` started at the top; Manim's starts at the right.
7. `AnimationGroup(run_time=)` did not stretch its schedule; `LaggedStart`/`Succession`/
   `AnimationGroup` and `TransformMatching*` are linear in Manim.
8. `next_to` ignored `aligned_edge`; `move_to` ignored `aligned_edge`/`coor_mask`.
9. `set_fill(color)` changed the stroke color.
10. Updater `dt`: Manim's first frame of every play has `dt` 0 and its last frame is before
    the play's end (a 3 s play at 30 fps turns a `dt` rotation by 89/30, not 3): mirrored.
11. A `NumberLine` with numbers is taller in Manim (its box includes the numbers).
12. `ValueTracker`, `NumberLine`, `Ellipse`, `save_state`, `add_updater` and others were missing.

## Verification

- `ruff check .`, `ruff format --check .`, `lint-imports` (2 contracts kept): clean.
- Focused suites, `-m "not slow"` (`test_manim_import`, `test_manim_parity`, `test_sound`,
  `test_open_contract`, `test_final`, `test_lint`, `test_docs`, `test_layers`, `test_beauty`,
  `test_engine`, `test_diagrams`, `test_shapes`, `test_glyphs`, `test_transform`,
  `test_style_freedom`, `test_mcp`, `test_kit`, `test_narration`, `test_render`, `test_proc`,
  `test_gui_controller`, `tests/manim_real`): 1152 passed, 18 skipped, 2 failed in 7.5 min;
  the two (`llms.txt` stale, the kit catalogue missing the new examples) were fixed and pass
  on rerun, and the files changed after that run (sound window, budgets, lesson-1 warnings)
  were rerun: `test_manim_import` + `test_manim_parity` + `test_sound` 103 passed, 1
  skipped (no PySide6); `test_narration` passes; `tests/manim_real` 10 passed with Manim, 10
  skipped without. The full suite was not run here (over an hour; CI runs it).
- Gate: the independent audit was done on 2026-11-23 (see "Audit"), and its findings are fixed. After
  the fixes, the same focused suites plus `test_gui_controller`, `test_layers` and `test_proc`
  ran with `-m "not slow"`: 1190 passed, 8 skipped in 7.5 min. The slow golden storyboards of
  `test_manim_parity` pass. `tests/manim_real` with Manim gave 11 passed. The importer's
  documents for the 77-script audit corpus, the examples and the real-Manim scenes were compared
  with those from before the fixes, and only the rate-function pieces differ (Deviation 11). The
  end-to-end `phasor manim` + `phasor render` command below was rerun: 360 frames, with sound.
- Bounds measured (after the audit fixes, this machine, wall clock and peak RSS): a 600 s
  `always_redraw` of a 240-point plot imports in 5.4 s as a 6.9 MB document; 50 half-second
  plays driving an `always_redraw` plot 6.1 s, 220 MB, 8.4 MB; 200 such plays 9.3 s, 240 MB,
  10.6 MB (updaters stop at 69 s, said); 2000 dots with `dt` updaters over `wait(30)` 8.0 s,
  220 MB; 30000 `wait(0.001)` with an updater 3.2 s; 3 x `wait(1e7)` with an updater in
  under a minute; a WAV header claiming 2^31 frames costs what the file holds (under 5 MB,
  `tracemalloc`); 300 clips of a 10-minute 8 kHz WAV in a 60 s scene mix in 9 s at 71 MB
  (256 mixed, the rest said); one 2.3 h 8 kHz clip in a 3 h scene 6.9 s at 1.04 GB, the
  3-hour int16 track itself (was 8.5 GB).
- A render with sound, here: `phasor manim examples/manim_camera_tour.py -o
  tour/tour.phasor.json` then `phasor render tour/tour.phasor.json --preset preview`: 360
  frames, an AAC track at 48 kHz, silent before 4.4 s and the chime from 4.5 s;
  `tour/assets/chime.wav` came along.
- Goldens (looked at): `tests/golden/software/manim_camera_tour.png`, `sound_cue.png`;
  `manim_reverb.png` regenerated (the shapes are phasor's shapes now; the same picture).

## Deviations

1. **Updaters become tracks, not `always`** (`MANIM_PARITY.md` said "updaters and
   `always_redraw` become `always`; dt updaters become a generated `Simulation`"). A Manim
   updater is Python acting on stand-in mobjects; as an `always` updater the document would
   need the Manim script and the stand-in at every load and `allow_scripts` to render, which
   an import must never widen (lesson 4). The stand-in runs them at Manim's own frames (the
   same fixed `dt` a `Simulation` would have) and writes the result as keys. Lost: the result
   is baked, so changing the `Value` later does not move what the updater drove. Recorded in
   SPEC ("As built, Milestone 20").
2. **Class-default text keeps its role size**, which is narrower than Manim's text at the same
   font size (measured: Text 0.42x Manim's width, MathTex 0.6x; Deferred 1). Layout computed
   from those boxes (`next_to` a text, a text `to_corner`) lands up to 0.54 away.
3. `play()` with two animations on one mobject (`Transform(a, b), a.animate...`) is not
   merged as Manim does; the second wins. Not seen in real scripts; not handled.
4. A scene that moves a shape past 1e9 world units fails the import (shapes are bounded by
   `MAX_COORD`, M16); before, a `Line` took any number. A shape that *starts* past it falls back
   to a path or a line, as before.
5. One WAV header walk, two decoders, in `time/narration.py`: `parse_wav` (no `wave`
   module: every chunk size checked against the file, a RIFF size smaller than the data
   refused, the frames the file really holds) under `read_wav` (narration, exact int16, any
   rate: the same samples as before, tested against the stdlib) and `wav_info`/`read_sound`
   (sound clips: any PCM width, bounded rates and frames, chunked); the editor's waveform
   uses it too. Folding narration into `read_sound` would refuse the low-rate clips
   narration tests and tools write today (lesson 1), so the decoders stay two.
6. Kept for lesson 1 (they imported before, as nothing): `self.camera.frame` in a plain
   `Scene` (Manim has none) now moves the camera, with a warning; `play()` with no animation
   and a negative `run_time` are warnings (Manim refuses both); `MoveToTarget`/`Restore`
   with a different number of parts moves the parts both have, with a warning.
7. Updater work has three budgets per import: `MAX_FRAMES` (20000 frames),
   `MAX_UPDATER_CALLS` (2 million calls) and `MAX_KEY_NUMBERS` (1 million numbers sampled,
   counted as sampled: a document of at most about 11 MB of keys; it was 2 million, about
   20 MB, and was not enforced). Past half of any, fewer frames a second (down to 2), each
   at a Manim frame time, `dt` adding up to Manim's, the first frame of a play skipped (it
   repeats the last); when one is spent, updaters stop for the rest of the import and the
   warning says when. A long run of updater-driven plays is therefore cut, not coarsened
   forever: 200 half-second plays of a redrawn 240-point plot stop at 69 s of 100.
8. A render reads and mixes only its window of the sound (`audio_track(start, end)`).
9. Each `play` adds its mobjects to `self.mobjects` as Manim does (only those not already in
   the scene's family); removers leave at the end. The order inside `self.mobjects` is not
   checked against Manim's.
10. Ids and types an import gives changed from the Milestone 13 importer: a plot is
    `graph_N` (was `path_N`), a `DashedLine` `dashedline_N` (was `line_N`), `Tex` `tex_N`
    (was `mathtex_N`, as it is text now), shapes are phasor's own kinds. A scene imported
    before and re-imported gets the new ids.
11. The import queues its ops and applies them in one patch at the end (and before each
    morph verb, which reads the document); each new object is checked alone on a scratch
    engine first, so the fallbacks (a path, a line) are chosen as before. An error in a
    queued op surfaces at the next apply rather than at its play (its message has no script
    line; neither did it before). The documents of the 77-script corpus, the examples and
    the real-Manim scenes are identical to before except where a rate function phasor has
    no name for is used (more pieces, note 1 of the audit).
12. `config.frame_rate` is 30 (it said 60, Manim CE's default, while updaters ran at 30);
    a script that sets it gets updaters at that rate (1 to 120).
13. `wait_until` and `wait(stop_condition=)` (were ignored: the full `max_time`, no warning)
    stop where Manim's video does: the condition is checked after each frame with Manim's
    clock one frame on (Manim CE 0.22.1 measured: `Scene.time` 1.0333 at the stop for a
    1 unit/s dot reaching x >= 0.99). A condition that fails is an error (Manim stops too).
14. A track window (`phasor render --start`) is on the scene's own sample grid: the same
    samples as that slice of the whole mix (a window's narration moved by up to a sample).
15. An imported scene is not strict-lint clean, by design: its motion is tracks at Manim's
    times, never verbs anchored to markers, so `lint(strict=True)` says `raw_motion` for
    each moving object. The exit demo has nothing else (tested); `manim_reverb` also has two
    `overlap` warnings where a ring sweeps over a label, as in Manim.

## Deferred

1. **Text size parity (owner decision).** To match Manim's widths the import would set every
   text's size explicitly (Text about 0.019 world units per font point, MathTex 0.013, against
   0.008 today), and `off_scale` would warn on every imported text. Today a class-default text
   gets the theme's role (Milestone 18's rule). Which does the owner want?
2. `Create` traces a shape from phasor's own first point; Manim's square starts at its
   top-right corner going left. Cosmetic, mid-trace only. `.animate.rotate` turns the shape
   where Manim mixes points (it shrinks on the way). Arrow tips are narrower than Manim's.
3. `Axes.get_area`, `Matrix.get_entries`, MathTex parts beyond color, rotated text, a turned
   camera frame: warnings.
4. Sound formats other than PCM WAV (decoding MP3 would need ffmpeg in the time layer).
5. Re-importing a Manim script silently replaces the saved scene, with any tweaks made to it
   (`name.overrides.json` is not carried over to the new import's ids). Pre-existing (M13).
6. The time budget (`MAX_IMPORT_SECONDS`) is checked only when the script calls back into
   the stand-in (a play, an updater, a wait condition): a script that loops without calling
   back is not stopped. A thread or a process would stop it; not done.
7. A script's `NameError` printed with its traceback can reach `Scene.__getattr__`, which
   raises a `ManimImportError` for the unknown name (a confusing message, not a crash).
   Pre-existing; recorded only.

## Audit (independent, 2026-11-23): findings and fixes

Each fix has a failing test first; the auditor's repros are adapted into the tests named.

| Finding | Fix | Test |
| --- | --- | --- |
| A. A crafted WAV raised `RuntimeError` (Python's `wave` chunk seek) out of `Engine.lint`, `inspect`, `audio_track`; a RIFF size smaller than the data was dropped silently, the semantics said the header's 10416 s for a 500-frame file | one header walk without `wave` (Deviation 5); every failure is `bad_file`; frames are what the file holds; the render mixes before its frames | `test_sound.py`: `test_a_hostile_header_is_bad_file_in_lint_inspect_and_every_mix` (4 files, whole and windowed), `test_semantics_report_what_the_file_really_holds`, `test_narration_reads_wavs_exactly_as_the_stdlib_did`, `test_a_render_mixes_the_sound_before_any_frame`; `test_gui_controller.py` waveform |
| B. Many short plays driving `always_redraw`: each play's flush recompiled the whole document (50 plays 228 s, 3.9 GB, 16 MB; 100 killed at 600 s) | ops and objects queued, one patch (Deviation 11); budgets enforced (S1) | `test_many_short_plays_driving_a_redrawn_curve_are_one_patch[50, 200]` (scene writes <= 3, keys within the budget, < 13 MB): 50 plays 6 s, 200 plays 9 s |
| S1. Past the frame budget the rate became 0 for good (dt updaters froze) while frames and numbers kept growing; the docs' bounds were false | the budget as in Deviation 7 | `test_the_updater_budget_is_enforced_and_dt_stays_manims`, `test_the_numbers_budget_bounds_the_keys`, `test_updater_frames_are_bounded` |
| S2. 2000 dots with updaters over `wait(30)`: 524 s, 2.36 GB | profiled: one whole-document compile per object added (quadratic) and `inspect.signature` per updater call; objects checked alone and queued, the signature read once per function, `_shown_ids` once per play, a calls budget | `test_many_mobjects_with_updaters_cost_linear_work` (counted: scene writes, signatures, updater calls): 8 s, 220 MB |
| S3. The mix held every resampled clip (300 clips: 3.4 GB; one 2.3 h clip: 8.5 GB) | streamed a chunk at a time; `MAX_SOUNDS`, `MAX_MIX_SAMPLES`, lint `sound_limit` | `test_mixing_memory_does_not_grow_with_the_number_of_clips` (`tracemalloc`), `test_the_number_of_clips_and_the_samples_mixed_are_bounded`, `test_sound_limit_lints_...` (both MODES): 300 clips 71 MB (was 3.4 GB), a 2.3 h clip 1.04 GB, the track itself (was 8.5 GB) |
| S5. `wait_until` / `stop_condition` ignored | Deviation 13 | `test_wait_until_stops_at_the_first_frame_the_condition_holds`, `test_a_failing_wait_until_condition_is_a_clean_error`, real Manim `wait_until` |
| Note 1: 16 even pieces missed `slow_into` 8.9%, `wiggle` 3.6%, `there_and_back` 1.7% | adaptive pieces: 0.19%, 0.20%, 0.18% | `test_a_rate_function_without_a_name_stays_within_its_tolerance` |
| Note 2: a window of a resampled clip differed from the slice of the whole mix (30810 on a saw) | the scene's sample grid (Deviation 14) | `test_a_window_is_the_same_samples_as_that_slice_of_the_whole_mix` (exactly equal) |
| Note 3: `add_sound` clamped a negative offset or gain 99 silently | warnings | `test_add_sound_says_when_it_clamps` |
| Note 4: an animated `start`/`gain` was ignored | lint `sound_limit` (the mix reads them at 0 s) | `test_sound_limit_lints_...` |
| Note 5: `config.frame_rate` said 60, updaters ran at 30 | Deviation 12 | `test_config_frame_rate_is_the_rate_updaters_run_at` |
| Note 6: this file said a plain Scene's frame is an error; the Verification command wrote beside the folder | text fixed (Built, Verification) | - |
| Note 7: ids and types changed between importers; re-import drops overrides | Deviation 10; Deferred 5 | - |
| Note 8: the exit demo is not strict-clean | Deviation 15; the docs say so | `test_the_camera_tour_imports_with_every_feature_and_no_warning` (lint clean, strict only `raw_motion`) |
| Note 9: `pc_tasks/10` said Windows matches to 3 decimals | text rows will differ (Pango fonts): fixed | - |
| Note 10: no time budget; `SystemExit` escaped; `_bring_files` wrote through a symlink | `MAX_IMPORT_SECONDS` (Deferred 6); `SystemExit` -> `ManimImportError` / `ToolError`; links refused | `test_a_script_that_fails_is_a_clean_import_error` (exit cases), `test_a_script_past_the_time_budget_is_a_clean_error`, `test_a_scene_script_that_exits_is_a_tool_error`, `test_saving_elsewhere_never_writes_through_a_link` (skips without symlinks) |

## Untested

- Windows (`pc_tasks/11_platforms_after_20.md`; the new tests are written for Windows CI:
  stdlib WAVs, no POSIX-only calls, `gl` marked, Manim skipped without its Python).
- macOS (no renderer until the GL 4.1 decision; `pc_tasks/11_platforms_after_20.md` B).
- Real Manim on Windows (`pc_tasks/10_manim_real.md`).
- CUDA (nothing here touches kernels).
- Hearing the sound in a player (the track's samples were checked, not listened to).
