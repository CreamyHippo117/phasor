# phasor over MCP (for AI agents)

You are connected to `phasor mcp`. You build a scene by sending **patches**, and you check it
with **review**. You never write coordinates for labels: name objects by meaning, use theme
roles, add constraints. Type an explicit color or size only when the person asks for one;
review says when it is off the theme.

Every ```json patch block (a patch) and ```json calls block (one tool call per line) in
this file runs in order, against one scene, in `tests/test_docs.py`.

## The loop

1. `object_types` (what exists, parameter names and defaults, style roles) and
   `docs_search "<what you want>"`.
2. `scene_new`, then `scene_edit` with patches.
3. `narrate` a script with `[[markers]]`, then `animate` verbs anchored to those markers.
4. `review`: lint + what is on screen at every marker + a storyboard picture.
   Fix each item in `todo` (`scene_edit`, or `lint` with `fix: true` for the certain fixes)
   and `review` again. Stop when `clean` is true or after 5 rounds; say what is left.
5. `scene_save`, then `render` (`draft` is quickest: small, 15 fps, plain look; `preview`
   shows the full look).

A failed patch changes nothing and tells you why (`errors[].message`, often with
`fix_hint`). `undo` reverts the last edit; `checkpoint` / `restore` / `diff` let you try
a variant and compare.

## Tools

| Tool | Use it to |
| --- | --- |
| `scene_new`, `scene_open`, `scene_save` | start, load or save `name.phasor.json` (inside the workspace folder only); a script scene's `scene_save` writes its tweaks into the tweaks block at the end of the script (ADR 0004) and leaves its other code alone; it writes nothing when the block's markers cannot be read for certain or the file changed on disk since it was last run or saved (open it again) |
| `scene_palette` | the colors run along strokes: `spectrum` (the prism rainbow), `moonlight`, `gold`, `glacier`, `borealis`, `rose`, or none; in the prism looks every glowing curve takes them |
| `scene_edit` | apply a patch (below); returns `ok`, `errors`, and the lint issues it added or fixed |
| `animate` | a choreography verb at a marker or time: `reveal`, `exit`, `trace`, `bloom_in`, `dim_others`, `undim`, `focus_pull`, `morph`, `settle`, `write`, `travel`, `flash`, `wiggle`, `circumscribe` (`value` "circle" or "box"), `transform`, `transform_matching` and `become` (`targets` [from, to]), `sweep` (`path`, `value`), and in 3D `orbit` (`value` degrees, `until` a marker) and `orbit_reveal` |
| `narrate` | set the script and voice; markers appear in `markers` |
| `snap_to_narration` | move the markers you placed by hand onto the narration's sentence starts (a recording: `narrate` with `audio`); `dry_run` (default true) previews each move (marker, from, to, why, confidence) with `alignment` and `unreliable`; `dry_run: false` applies it as one undo step |
| `inspect` | what is on screen at `t`: `brief` (a line each), `standard`, `full` |
| `lint` | visual problems; `fix: true` applies the certain fixes |
| `review` | the whole self-review round in one call |
| `storyboard` | keyframe picture with each object's box and id drawn on it |
| `hit_test`, `resolve` | what is at a pixel; which object ids "the label of the pole" means |
| `events`, `when_hidden`, `free_space_near` | 3D: when things get hidden or leave the frame (with fixes), spans an object is hidden, empty places next to an object |
| `critique`, `polish` | the art director: a taste score, aesthetic issues with fixes and the rubric; apply the fixes (3 rounds at most; `snap: true` also snaps explicit values to roles) |
| `variants`, `choose` | alternative looks of one beat side by side (as is, calmer, bolder); pick one, it is recorded in `style/choices.jsonl` and applied |
| `diff`, `checkpoint`, `restore`, `undo`, `redo` | compare and branch |
| `render` | a video (`draft`, `preview`, `final`, `4k`) or one PNG with `frame`; `start`/`end`/`frame` take seconds or a marker name (a draft goes to `<scene>.draft.mp4`) |
| `render_start`, `render_status`, `render_cancel` | a render that runs while you go on: `render_start` (as `render`; default `draft`, to `<scene>.draft.mp4` beside the scene's file) returns a `job` at once; `render_status` gives `state` (running, done, failed, cancelled), `frame` of `frames`, `progress`; `render_cancel` stops it and removes its partial file. It renders the scene as it was when started, writes beside the target and replaces it only when done (an existing file only with `overwrite: true`; a draft's own `<scene>.draft.mp4` always; never a file the scene reads) |
| `export_midi` | write the scene's MIDI notes (every `notes` object, as heard, at its times in the scene) to a `.mid` (format 1, a track per object); `out` defaults to `<scene>.mid`; never over a file the scene reads, nor over an existing file unless `overwrite: true` |
| `docs_search` | find the API, a pattern or an example |
| `explorer_state`, `explorer_edit`, `explorer_preset` | the Explorer: its rows, systems, equilibria and sliders in one call; add, retype, hide, recolour or remove rows and set variables (one undoable edit); load a classic system |
| `find_equilibria`, `flow_probe`, `sweep_parameter` | the Explorer's numbers, on a fork (nothing changes): equilibria with eigenvalues, a path's behaviour (period, escape, Lyapunov exponent, the dt check), candidate bifurcations along a variable |

## Patch operations

`scene_edit` takes one op, a list of ops, or `{"ops": [...]}`. Ops apply together or not at
all. Type names are lower case: `dot`, `line`, `text`, `domain_coloring`, `fractal`.

```json patch
[
  {"op": "set_param", "path": "camera.height", "value": 6},
  {"op": "add_object", "object": {"id": "pole_i", "type": "dot",
    "params": {"pos": [0, 1], "r": 0.15}, "style": {"color": "highlight"}, "tags": ["key"]}},
  {"op": "add_object", "object": {"id": "pole_i.label", "type": "text",
    "params": {"text": "pole at i"}}},
  {"op": "add_object", "object": {"id": "loop", "type": "line",
    "params": {"points": [[-2, -1], [2, -1], [2, 2], [-2, 2], [-2, -1]]},
    "style": {"color": "secondary"}}}
]
```

- A child id is `<owner>.<name>` and the owner must exist (`pole_i.label`).
- `pos` is world space: x = Re, y = Im, z up; `[x, y]` means z = 0.
- `style` takes role names from `object_types().roles`. A color key (group `emphasis`), a
  text size (`text`) or a stroke width (`stroke`) may instead hold an explicit value:
  `"#58C4DD"` (sRGB), linear `[r, g, b, a]` (0..1), or a size in world units (positive, at
  most 1e6; frame heights on an overlay). Glow, surface and colormap keys take roles only.
  Prefer roles; `set_meta` `background` sets the scene's own background color (opaque).
- Never reuse the id of a removed object (`id_retired`); pick a new meaningful name.

Constraints say where things go; the layout solver does the arithmetic:

```json patch
[
  {"op": "add_constraint", "rule": "label(pole_i) near pole_i"},
  {"op": "add_constraint", "rule": "on_screen(loop, 0.05)", "priority": "medium"}
]
```

Rules: `a above b gap N`, `label(x) near x`, `a outside b`, `a tangent_to b`,
`a attach_to b at top` (a named spot: top, bottom, left, right, center, ... or a feature such as
an arrow's `tip`), `align_x(a, b)`, `align_y(a, b)`, `inside(a, b)`, `on_screen(a, margin)`, `no_overlap(a, b)`, `min_size_px(a, N)`,
`readable(a)`; join with `and`; priorities `required`, `strong`, `medium`, `weak`.

Sliders (`add_control` / `remove_control`) make a number something a viewer can move in the
editor, a notebook or a web page (`phasor web`); a vector parameter needs its `component`:

```json patch
[
  {"op": "add_object", "object": {"id": "set", "type": "fractal",
    "params": {"kind": "julia", "c": [-0.8, 0.156]}}},
  {"op": "add_control", "control": {"path": "set.c", "component": 0, "min": -1.2, "max": 1.2,
    "step": 0.05, "label": "Re c"}}
]
```

Change things later with `set_param`, `set_style`, `set_tags`, `remove_object`,
`remove_constraint`:

```json patch
[
  {"op": "set_param", "path": "pole_i.r", "value": 0.2},
  {"op": "set_style", "id": "loop", "key": "stroke", "role": "bold"},
  {"op": "set_tags", "id": "loop", "tags": ["contour"]}
]
```

Domain objects are added the same way (`object_types` lists their parameters and presets):
`equation` (typeset math in colored parts); `axes`, `parametric_curve`, `parametric_surface`, `vector_field_3d`, `flow_lines` (space); `epicycles`, `partial_sum`, `spectrum`, `spectrogram` (fourier); `transform_grid`, `arrow`,
`eigen_lines`, `matrix` (linear algebra); `room`, `source`, `listener`, `wavefront`,
`image_sources`, `impulse_response`, `pressure_slice`, `room3d`, `sound_rays`, `wavefront3d` (acoustics); `pde_profile`, `pde_field`,
`pde_surface`, `pde_modes`, `pde_slice`; `polygon`, `cayley_table`, `clock_diagram`,
`cayley_graph`, `homomorphism`, `subgroup_lattice`, `galois_diagram`, `permutation` (algebra);
`graph`, `riemann_sum`, `taylor`, `vector_field`, `limit`, `loop_integral` (calculus). A parameter may be an
expression, which is how a `matrix` object reads a grid's matrix: `"matrix": {"expr": "grid.matrix"}`.

```json patch
[
  {"op": "add_object", "object": {"id": "grid", "type": "transform_grid",
    "params": {"matrix": [[2, 1], [1, 2]], "pos": [5, -1, 0], "extent": 2}}},
  {"op": "add_object", "object": {"id": "entries", "type": "matrix",
    "params": {"matrix": {"expr": "grid.matrix"}, "pos": [5, 3, 0]}}},
  {"op": "add_object", "object": {"id": "tone", "type": "spectrum",
    "params": {"frequencies": [100.5], "amplitudes": [1.0], "window": "hann", "pos": [-7, -4, 0]}}}
]
```

```json patch
[
  {"op": "add_object", "object": {"id": "rod", "type": "pde_profile",
    "params": {"kind": "heat", "length": 2.0, "diffusivity": 0.5, "modes": [1.0, 0.4],
               "time": {"expr": "t / 10"}, "pos": [-8, 1, 0]}}},
  {"op": "add_object", "object": {"id": "square", "type": "polygon",
    "params": {"n": 4, "pos": [2, 2, 0]}}},
  {"op": "add_object", "object": {"id": "area", "type": "riemann_sum",
    "params": {"function": "x**2", "a": 0, "b": 3, "n": 8, "rule": "mid", "pos": [-2, -4, 0]}}}
]
```

Backgrounds that compute a picture per pixel (`domain_coloring`, `fractal`) are objects too;
labels may sit on them.

```json patch
{"op": "add_object", "object": {"id": "plane", "type": "domain_coloring",
  "params": {"function": "1/(z**2 + 1)", "contour": 0.4}, "style": {"colormap": "phase"}}}
```

## 3D

Switch the camera to `orbit` (perspective, z up) and add surfaces. Leave the camera's
`elevation`, `distance` and `target` unset and give **goals** instead: the camera solver
chooses them at every moment, also while the camera orbits.

```json patch
[
  {"op": "set_camera", "kind": "orbit", "params": {"azimuth": -60}},
  {"op": "add_object", "object": {"id": "sheet", "type": "riemann_surface",
    "params": {"kind": "sqrt", "radius": 2.0}, "style": {"material": "glass"}}},
  {"op": "add_constraint", "rule": "camera keep_visible(sheet)"},
  {"op": "add_constraint", "rule": "camera avoid_occlusion(pole_i)", "priority": "medium"}
]
```

Goals: `keep_visible(a, b)`, `center(a)`, `thirds(a)`, `avoid_occlusion(a)`,
`min_size(a, 0.3)` (share of the frame height), each written after `camera`. Materials
(`surface` roles): `solid`, `glass` (see-through, does not hide things), `wire`,
`emissive`. In 3D a label is hidden only by what is in front of it: lint reports
`occluded_label`, and `events` lists every moment something important is hidden or
leaves the frame. A clean 3D scene has no `problems` in `events`.

## Narration and animation

Write the script first, mark the words things should happen on, then anchor every motion
to a marker with a verb. Do not animate by raw seconds in a narrated scene (lint
`raw_motion` says so).

```json calls
{"tool": "narrate", "script": "Here are the poles. [[poles]] Draw a loop around the first. [[loop]] The result is two pi.", "voice": "estimate"}
{"tool": "animate", "verb": "reveal", "at": "poles", "targets": ["pole_i"]}
{"tool": "animate", "verb": "trace", "at": "loop", "targets": ["loop"]}
```

A recording instead of a voice: put the file (WAV, or an mp4, m4a, mp3, ogg, flac) in the
scene's folder and call `narrate` with `audio` and the `script` it reads (its sentences are
found in the audio; `[[markers]]` fall on their words). Markers you placed by hand before
the recording existed snap onto it: `snap_to_narration` previews (`dry_run`, the default),
then `dry_run: false` moves them. Lint `narration_drift` lists markers the snap would move.
Read `unreliable` and `alignment` in the preview before applying: `ambiguous_timing` (the
scene's timing fits the voice two ways; nothing moves: pin markers with `anchors`),
`alignment_doubtful` (a sentence break is probably misplaced: the reader ran sentences
together or paused as long inside them), `unreadable`, `music`, `low_snr`. Anchors that
would reorder markers or cross a `[[marker]]` of the script are refused.

```json calls
{"tool": "snap_to_narration", "dry_run": true}
```

MIDI notes are a `notes` object: from a `.mid` in the scene's folder (`path`) or listed
(`notes`: `[pitch, start, length, velocity, channel]`, seconds), played by the built-in
synthesizer from `start` (seconds or a marker's name) and mixed into the render's sound like
a `sound` clip. `export_midi` writes them all to one `.mid`.

```json patch
{"op": "add_object", "object": {"id": "tune", "type": "notes",
  "params": {"notes": [[60, 0, 0.4, 100], [64, 0.4, 0.4, 90], [67, 0.8, 0.8, 80]],
             "instrument": "bell", "gain": -6}}}
```

```json calls
{"tool": "export_midi", "out": "out/tune.mid"}
```

A long render need not block you: start it, ask how far it is, stop it if the scene needs
another change (nothing half-written is left behind).

```json calls
{"tool": "render_start", "preset": "draft", "out": "out/look.draft.mp4"}
{"tool": "render_status"}
{"tool": "render_cancel", "job": "render_1"}
```

`animate` returns `start` and `end`: pass `offset: end - start` to start the next step when
the first ends. A verb moves nothing for an object that has no track yet, so `reveal` an
object you want to appear (it is hidden until then).

## What comes back from `review`

`clean`, `todo` (one line per issue: code, ids, what is wrong), `issues` (the same with
`fix` ops you can send to `scene_edit`), `frames` (a line per object at each marker) and
`storyboard` (an image path; the image also arrives inline). Read the picture, not only the
lint: lint cannot tell you a layout is ugly.

## The beauty pass

`review` and `lint` take `aesthetic: true` for the taste rules on top of the correctness
ones: `focal_point` (exactly one `highlight` per beat), `color_restraint` (at most three
saturated hues; a phase-coloring plane is exempt, hue is its meaning), `glow_clip`,
`density`, `motion_rhythm`, `margin`, `stroke_mix`, and for explicit values `off_palette`
(a color far from every palette role, or a background far from the theme's) and `off_scale`
(a size off the theme's type or stroke scale). Then:

1. `critique` returns `score` (0 = nothing to criticise), issues tied to ids and markers,
   `fix` ops (certain), `suggested` ops (judgement), `snap` ops (an explicit value back
   to its nearest role: offered, never applied by itself), and the rubric (`STYLE.md`).
2. Look at the storyboard (`storyboard` with `compare: true` puts each frame beside the three
   nearest frames in the project's `gallery/`) and judge by the rubric. Rules cannot see
   everything.
3. `polish` applies the fixes and suggestions, three rounds at most; whatever is left is
   yours to judge (an empty frame, two motions that collide). It snaps explicit colors and
   sizes only with `snap: true`: ask the person first, the value may be meant.
4. Unsure which look? `variants` on a marker, then `choose` one.

Start scenes from a look: `scene_new` with `look: "calm"` (soft edges, gentle life), `"glow"`
(lively bloom) or `"chalk"` (light paper page, still, crisp). `object_types` lists each
type's `presets`; in a patch, `{"preset": "soft_phase"}` inside an `add_object`. Explicit
values win, then the preset, then the look. The default theme is dark.

## Working beside a person (`--attach`)

If the person has `phasor gui` open, run `phasor mcp --attach --root <their folder>`: every
tool then works on the scene in their window, and they watch your edits land. Their edits
are parameter and style changes: `inspect(detail="full")` shows every object's `params` as
the document sets them, and `diff` compares with the saved file. Undo is one shared history, so
`undo` takes back the most recent edit, whoever made it: check `diff` before undoing. You
cannot open a `.py` scene through the editor (only the person can); a Manim script
(`from manim import ...`) opens as a plain scene, with `notes` on what could not be carried
over, on a server started with `--allow-scripts`. You never make code run in the editor: a
document you open there runs none of the modules it names (lint `missing_type`), and an
edit that changes `meta.modules` or adds an updater (`always`) is refused (`scene_code`),
even on a scene whose code the person trusted. Draw with phasor's own types, or ask the
person: only they can trust a scene's code (in the editor).

## Traps this loop has already hit

- **Overlay positions are in frame heights**, not pixels or world units: `[0, 0]` is the
  top-left corner and the frame is about 1.78 wide, so a centered title is `[0.8889, 0.08]`.
  An overlay added with no `pos` sits at the corner and lint reports `offscreen`.
- **Lint clean does not mean it looks good.** The first draft of the residue scene was lint
  clean with an off-center title, a hand-placed answer and a 24-sided "circle". Look at the
  storyboard image every round.
- **Do not place text by hand.** Hand-picked coordinates are how a result ended up floating in
  the middle of a colored background. Use an overlay position or a constraint.
- **A removed id is gone for good.** Re-adding `result` fails with `id_retired`; use `answer`.
- **A second `reveal` of the same object is refused** (`bad_segment`): its fade already
  exists. Change the first one's marker instead, or `undo`.
- **`.py` scenes are code.** The server refuses them unless started with `--allow-scripts`.
- **A scene may bring its own types** (ADR 0005: `meta.modules`, a scene script or
  `*.types.py` next to it). `object_types` marks those `"scene_local": true`. On a server
  without `--allow-scripts` their code does not run: lint `missing_type`, the object draws as
  a placeholder box, and its parameters and overrides stay as they are. Edit its parameters
  as usual; do not remove it to make the issue go away. A `value` object is a number with
  nothing to draw: animate its `value` and read it with `{"expr": "speed.value"}`.

## The Explorer (rows, sliders and numbers)

The Explorer is a list of typed rows on one document: `x' = sigma*(y - x)` (an ODE row; the
`x' y' z'` rows form one system), `rho = 28` (a variable with a slider), `(y, -x)` (a vector
field), `(1, 0.5)` (a point that follows the flow), and graphs or curves (read, not drawn
yet). Work like this:

1. `explorer_state` first: the rows as typed (`id`, `kind`, `text`, `error`, `locked`), each
   system with its equilibria (eigenvalues, kinds, the window and grid searched), the
   variables and their slider ranges, lint.
2. Answer with numbers and where they came from: `find_equilibria` (window, grid),
   `flow_probe` (`T`, `dt`, the dt check, `chaotic_beyond`: no positions after it),
   `sweep_parameter` (candidate bifurcations, each named by the local test that found it).
   They run on a fork: the scene does not change. Say what was not checked (outside the
   window; global bifurcations; a Lyapunov estimate that did not converge).
3. Change things only with `explorer_edit` (or `explorer_preset`): each call is one undo
   step. Read its `rows` (ids, kinds, errors) and `lint_delta` before you answer. A row the
   person is typing in is `locked`: the edit is refused with `row_locked` and the row's
   `version`; read `explorer_state` again and retry later. The same holds for every tool
   that would change or remove that row: `scene_edit`, `undo`, `redo` and `restore` are
   refused with `row_locked`, `scene_new` and `scene_open` wait, and any other tool that
   changed it (a verb on its value) is taken back and refused.

```json calls
{"tool": "scene_new", "title": "explorer", "look": "none"}
{"tool": "explorer_preset", "name": "pendulum"}
{"tool": "explorer_edit", "rows": [{"id": "gamma", "value": 0.5}, {"text": "(1.5, 1.5)"}]}
{"tool": "explorer_state"}
{"tool": "find_equilibria", "grid": 7}
{"tool": "flow_probe", "x0": [1.5, 1.5], "T": 5, "samples": 6}
{"tool": "sweep_parameter", "name": "gamma", "lo": 0, "hi": 1, "n": 5}
```

Rows are read the way people write: `x^2`, `2x`, `x y`, `sin(x)` (not `sin x`), Greek
letters (`σ` is `sigma`), `x'` or `dx/dt`. A run of letters is one name (`rho`), so a
product needs a space or `*`. A name no row defines gets a slider (value 1, range -10..10)
when the edit is committed. A row that does not read is kept, with its `error` in words;
a graph, curve, surface or heat map row is kept but not drawn yet (`drawn: false` and `why`
in the answer, lint `explorer_row`). A new row may name the id its object gets with `base`
(`{"text": "x' = y", "base": "pendulum"}`). Sweep candidates carry a `precision` (the
bracket, and what limits the value besides it: Newton's residual, the merge distance of
roots, the eigenvalue tolerance); a count change at the window's edge is `enters or leaves
the window`, not a bifurcation. `flow_probe` gives positions only up to `positions_until`.
