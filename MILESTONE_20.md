# Milestone 20: Manim import parity and platforms

Plan: `MANIM_PARITY.md`, "Milestone 20". Built 2026-11-23 in a worktree branched from
`claude/eloquent-mayer-jiqxs9` (after the gate of 17 and 19). Exit demo:
**`examples/manim_camera_tour.py`, a Manim script using the camera frame, a filled shape, a
ValueTracker and an updater, `always_redraw`, `add_sound`, a morph, a color animation and a
number line, imports with no warning, and real Manim CE agrees on its clock, places, colors
and camera** (`tests/manim_real`, row `manim_camera_tour` below).

Status: **built; focused gates green on Linux; real Manim compared here.** Windows, macOS
and CUDA untested (`pc_tasks/10_manim_real.md`, `pc_tasks/11_platforms_after_20.md`).

## Built

| Area | Where | What |
| --- | --- | --- |
| The stand-in, rewritten | `phasor/importers/manim.py` | a leaf keeps its outline (`pts`), its centre, scale, turn and whether it is still its own kind; every change of place goes through one `_affine`; a leaf becomes phasor's own kind (`shapes.Circle`, `Rectangle` ...) while it keeps its shape, else a `shapes.Path` (one way: `phasor_spec`) |
| Shapes and fills | same | `Circle`, `Ellipse`, `Arc`, `ArcBetweenPoints`, `Sector`/`AnnularSector`, `Annulus`, `Polygon`, `RegularPolygon`, `Triangle`, `Rectangle`, `Square`, `RoundedRectangle`, `Brace` (`get_text`, `get_tex`, `get_tip`), `DashedLine`, `DashedVMobject`, `VMobject.set_points_as_corners`; `fill_opacity`, `fill_color`, `set_fill`, `set_stroke`, `stroke_width=0` (`stroke_opacity` 0), `set_opacity` |
| Files | same | `SVGMobject` -> `shapes.SVG`, `Code(code_file=)` -> `dg.Code`, `add_sound` -> `media.Sound`: only files inside the scene's folder (`core/files.py`); the scene's `base_dir` is the script's folder; `phasor manim -o elsewhere/` copies them along (`Engine._bring_files`) |
| Text | same | `Tex` is text mode (was maths), `MathTex` with several strings an `Equation` (`eq[i].set_color` -> the part's role); `Write`'s run time from the glyph count as Manim (1 s under 15 glyphs, else 2 s; `\frac` counts its bar) |
| Morphs | same | `Transform`/`ReplacementTransform` between kinds -> the `transform` verb, `TransformMatchingTex`/`Shapes` -> `transform_matching` (linear, as Manim), played in Manim's run time by retiming the verb's segments (`scene.At.motion`: a verb's ops without applying them); text becomes `as_shapes` when it morphs; a shape stretched out of its kind by `.animate` morphs into a path |
| Color | same | `.animate.set_color`, `set_fill`, `Indicate` -> style tracks (`<id>.style.color`, `fill_color`), mixed in linear light; a rate function that comes back (`there_and_back`) comes back |
| Rate functions | same, `time/easing.py` | Manim's own (`smooth` is a logistic S, not smoothstep): new easing `sigmoid`; `linear`, `smoothstep`, `smootherstep`, `ease_*_cubic` map exactly; any other (`there_and_back`, `double_smooth`, `rush_into`, a lambda) is sampled into 16 straight segments |
| Diagram kit | same | `Axes` (`get_axis_labels` -> `labels`, `add_coordinates`/`include_numbers` -> `numbers`, `p2c`, `i2gp`), `NumberPlane`/`ComplexPlane` -> `dg.NumberPlane` (grid), `NumberLine` -> `dg.NumberLine`, `BarChart`, `Table`/`MathTable`, `Matrix`, `Code`, `DecimalNumber`/`Integer` -> `Counter`; style keys cut to the kit type's own |
| Trackers and updaters | same | `ValueTracker` -> `Value` with tracks; `add_updater` (with or without `dt`), `always_redraw`, `always`, `f_always`, `become`, `suspend_updating`: run by the stand-in frame by frame (30 a second, Manim's frame times: the first frame of a play has `dt` 0 and the play's last frame is before its end, then a final `dt` 0 update), with every animated mobject and tracker where Manim has it; what changes becomes keys, simplified (Douglas-Peucker, 1e-4); bounded (`MAX_FRAMES` 20000 per import, then fewer frames a second) |
| Camera | same | `MovingCameraScene`: `self.camera.frame` (`.animate.move_to`, `shift`, `scale`, `set(width=)`, `set_width`, `save_state`, `Restore`) -> tracks on `camera.center`/`camera.height` (width = height x 16/9); a plain `Scene` has no `frame` (an error that says so) |
| Other animations | same | `Unwrite`, `DrawBorderThenFill` (2 s, `double_smooth`), `GrowFromPoint`/`Edge`, `GrowArrow`, `ShrinkToCenter`, `Restore`, `ApplyMethod`, `Rotating`, `Circumscribe`, `Flash`, `Wiggle` (phasor's verbs, retimed), `FadeIn`/`FadeOut` `shift`/`scale`/`target_position`, `LaggedStartMap`; `AnimationGroup` stretches its schedule to a group `run_time` (it did not) |
| Sound | `domains/media.py`, `time/narration.py`, `engine.py`, `render/job.py` | `media.Sound(path, start, gain)`: a WAV in the scene's folder, drawless (`DRAWLESS` in lint, like `Value`), semantics `seconds`/`rate`/`unreadable`; `narration.wav_info`, `read_sound` (8-32 bit PCM, 1-16 channels, 8-192 kHz, chunked, at most `MAX_CLIP_FRAMES`), `resample`, `Clip`; `mix_track` mixes narration and clips (narration's rate, else 48 kHz; clips cut where the scene ends; the track at most 3 hours); `Engine.audio_track(problems)` (any type with `audio_clip`); `phasor render` lists left-out clips in its warnings and prints "with sound" |
| Lint | `inspect/lint.py` | `missing_file` and `bad_file` for a sound (the hint says PCM WAV); no `orphan` for drawless types |
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
tests/manim_real`: **10 passed**. Tolerances (`compare.py`): shapes, dots, lines and the camera
frame 0.05 world units (of the 8-unit frame); text and diagram composites 0.35 (phasor's own
fonts); the plain look (the prism look's living motion moves things by design); a sample where
either side has drawn nothing yet is skipped; Manim's internal groups (the camera's
`ScreenRectangle`, a `ValueTracker`, an `AnimationGroup`'s `Group`, a `TransformMatching*`'s
`VGroup`) are not the script's.

| Scene | Matched | Differed (why) |
| --- | --- | --- |
| shapes_layout | clock; every mobject at every sample; positions within 0.008; colors (2 explicit the same, 7 class defaults as roles) | sizes 0.25: Manim's arrow tip is wider (0.35 vs phasor's), and phasor's boxes include the stroke |
| animate_timing | clock (8 plays: run times and rate functions, `LaggedStart` 0.5, `Succession`, `Indicate`, `Rotate`) | while `Create` draws a square, phasor traces it from another corner (0.998 mid-trace; equal when done); during `Succession` Manim lists the square inside its group |
| fills_colors | clock; every mobject; fills; the animated colors at the end (2 the same, 6 roles) | a dashed line's box ends at its last dash (0.054) |
| text_write | clock (`Write` 1 s, linear); every mobject; positions 0.000 | text 0.54: a small text put in a corner by its edge; phasor's text is narrower (see Deferred 1) |
| axes_plot | clock; every mobject; the curve and dot where Manim has them once drawn | mid-`Create` a curve's drawn share is by arc length here, by Bezier pieces in Manim (0.089) |
| math_tex | clock (`Write` linear 1 s, `TransformMatchingTex` linear); every mobject; positions 0.000; text 0.108 | none beyond fonts |
| moving_camera | clock; camera frame within 0.000 at every sample (`move_to` + `set(width=)`, `shift` + `scale`, `Restore`) | none (sizes 0.032: the stroke) |
| updaters | clock; positions **0.000** at every frame (a tracker-driven updater, `next_to` a moving dot, a `dt` rotation, `always_redraw`) | sizes 0.045: the stroke of a turned square |
| manim_camera_tour (exit demo) | clock; positions 0.011; camera 0.000; colors (4 the same, 2 roles); text 0.104 | none |
| manim_reverb (M13 demo) | clock (14.5 s); every mobject; colors (5 the same, 4 roles); text 0.303 | the room's `Create` traced from another corner (as animate_timing) |

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
- Focused suites (`uv run pytest -q ...`): `test_manim_import` 28 passed, 1 skipped (no
  PySide6); `test_manim_parity` 40 passed; `test_sound` 26 passed; `tests/manim_real` 10 passed
  with Manim, 10 skipped without; the gate's list below.
- A render with sound, here: `phasor manim examples/manim_camera_tour.py -o tour/` then
  `phasor render tour/tour.phasor.json --preset preview`: 360 frames, an AAC track at 48 kHz,
  silent before 4.4 s and the chime from 4.5 s; `tour/assets/chime.wav` came along.
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
5. Two WAV readers in `time/narration.py`: `read_wav` (narration, exact int16, any rate, as
   before) and `read_sound` (sound clips: any PCM width, bounded rates and frames, chunked).
   Folding narration into `read_sound` would refuse the low-rate clips narration tests and
   tools write today (lesson 1); left as two, side by side.
6. Each `play` adds its mobjects to `self.mobjects` as Manim does (only those not already in
   the scene's family); removers leave at the end. The order inside `self.mobjects` is not
   checked against Manim's.

## Deferred

1. **Text size parity (owner decision).** To match Manim's widths the import would set every
   text's size explicitly (Text about 0.019 world units per font point, MathTex 0.013, against
   0.008 today), and `off_scale` would warn on every imported text. Today a class-default text
   gets the theme's role (Milestone 18's rule). Which does the owner want?
2. Morphing `Create`: phasor traces a shape from its own first point; Manim's square starts at
   its top-right corner going left. Cosmetic, mid-trace only.
3. `Axes.get_area`, `Matrix.get_entries`, MathTex parts beyond color, rotated text, a turned
   camera frame: warnings.
4. Sound formats other than PCM WAV (decoding MP3 would need ffmpeg in the time layer).

## Untested

- Windows (`pc_tasks/11_platforms_after_20.md`; the new tests are written for Windows CI:
  stdlib WAVs, no POSIX-only calls, `gl` marked, Manim skipped without its Python).
- macOS (no renderer until the GL 4.1 decision; `pc_tasks/11_platforms_after_20.md` B).
- Real Manim on Windows (`pc_tasks/10_manim_real.md`).
- CUDA (nothing here touches kernels).
- Hearing the sound in a player (the track's samples were checked, not listened to).
