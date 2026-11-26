# phasor: math animation engine spec

Nov 15, 2026 · @Atlas

phasor is a Python library for beautiful, GPU-rendered math animations. It is built so an AI agent can see, check and fix what it makes, not just write code blind. It runs natively on Windows with any CUDA GPU. "phasor" is a working name; rename freely.

## How to use this doc with Claude Code

- Save this doc as `SPEC.md` at the root of a new repo (export it as Markdown). Then ask Claude Code to read it, propose `CLAUDE.md`, and start Milestone 0.
- Build in milestone order. Each milestone must end with a runnable demo scene and passing tests before moving on.
- Keep `docs/ai/` (section "AI integration") updated as the API grows. That folder is what makes any AI fluent in the library later.
- Where this spec and working code disagree, the code wins and the spec gets updated in the same commit.
- Credit: the audio-driven timing and state/transition model are inspired by 2swap's swaptube (C++, Linux, CUDA). phasor reimplements the ideas in Python; it copies no code.

## Vision and design principles

The goal is one tool that is equally good for human-only, AI-only and AI+human editing, and that makes complex analysis, Fourier, PDEs, algebra and 3D look stunning.

**Why existing tools fail AI (the problems phasor must solve)**

1. No spatial feedback: the AI cannot ask where an object is, how big it is on screen, or what it overlaps.
2. 3D projection is invisible: occlusion, depth, label facing and off-frame objects must be imagined.
3. Intent is not stored: `shift(2*LEFT)` records a move, not "label above the pole, never overlapping".
4. Time is implicit: scene state is built by mutation, so one early edit breaks later frames silently.
5. Errors are silent: overlaps, off-screen text, low contrast and leftover objects render without warnings.
6. Iteration is slow: render a full video, watch it, describe it in words, edit, repeat.
7. API drift: multiple dialects and changing APIs make AI write mixed or outdated code.
8. No math semantics: a contour is just a path; nothing knows about its poles, winding or residues.
9. Weak GPU story: fractals, domain coloring and bloom need hand-written shaders or slow CPU paths.

**Principles**

- **Queryable, always.** Every object has a stable ID, a semantic type, and world + screen bounds at any time t, returned as JSON.
- **Declarative over imperative.** Store intent (constraints, state targets, narration markers), let the engine solve positions and timing.
- **Pure time.** Scene state is a pure function of t. Seeking to any time is cheap and deterministic.
- **Loud errors.** A linter flags visual problems before render; strict mode turns warnings into failures.
- **Fast loop.** A storyboard preview (keyframe stills + JSON) returns in seconds; full CUDA render is the last step.
- **One small, stable API.** One way to do each thing, versioned, documented for AI in `docs/ai/`.
- **Math-native objects.** Objects know their mathematics (a contour knows enclosed poles; a matrix knows its eigenvectors).
- **Beauty by default.** HDR color, bloom and good typography are on by default, controlled by named themes.
- **Domain-agnostic core.** Complex analysis is one plugin among many; acoustics, PDEs, algebra and linear algebra are first-class too.

## Architecture overview

&#91;embedded content: phasor architecture · 3 editors, 1 scene file, 2 renderers\]

Python, the AI and the GUI all edit one scene document. The engine solves layout and time, the inspector reports exactly what is on screen back to every editor, and only then do the renderers run.

## Core architecture contracts

These seven decisions are hard to change later. Build them exactly as written in Milestones 0–2, and ask before deviating.

**1. One engine, three front ends**

- `phasor.engine.Engine` owns the document, the evaluated-state cache, the layout bake and the inspector. The Python API, CLI, MCP server and GUI are thin clients; none holds scene state of its own.
- Every mutation goes through `Engine.apply(patch) -> PatchResult`. It validates against the schema, records an undo step and returns the lint delta. There is no other write path.
- When the GUI is running it hosts the Engine, and `phasor mcp` attaches to it over a localhost JSON-RPC socket, so the AI and the human edit the same live scene. With no GUI open, `phasor mcp` hosts a headless Engine.

**2. The document is the source of truth**

- Schema: pydantic v2 models in `core/schema.py`. JSON Schema is exported from them for MCP validation and docs. The document carries `schema_version`; migrations live in `core/migrate.py`.
- Canonical file: `name.phasor.json`, written with sorted keys and stable ordering so git diffs stay clean.
- A Python script is a builder: running `scene.py` emits the base document. GUI and AI tweaks to a scripted scene are saved in `name.overrides.json`, keyed by object id + parameter path (plus layout constraints added on top of the script's own, ADR 0002), and applied on top at load. Re-running the script never destroys tweaks. An override whose target id vanished is a lint error (`orphan_override`). An object whose type comes from a module that is missing or not allowed to load is lint `missing_type`; it draws as a placeholder box and keeps its parameters and overrides (ADR 0005).
- IDs match `[a-z][a-z0-9_]*`, use dots for children (`res.pole_i`), and are never reused or renumbered: removing an object retires its id in the document (`retired_ids`), and adding it again is an error. Auto IDs come from type + meaning (`pole_i`), not counters.

**3. Time is a pure function**

- `Engine.state_at(t) -> SceneState` depends only on (document, t), given the machine's occlusion oracle: the layout bake reads the real id pass where OpenGL exists and geometry where it does not, so a near-tie can place a label differently on two machines (ADR 0002). It may also depend on files the document names: an object type lists its file-path params in `Files` (e.g. `Room3D.path`, a Wavefront `.obj`); paths are relative and stay inside the scene's folder (`core/files.py`), a changed file is re-read (cached on its mtime and size), and a missing one draws nothing and is lint `missing_file` (owner approved, 2026-11-22). A document may also name code modules in its folder (`meta.modules`, ADR 0005): read only from inside the scene's folder and loaded only where scripts are allowed, like the importers. When `phasor.load` runs a scene script and the document names scene code, it names the script too, after those modules, because what that code draws can depend on what the script sets up (a registry it fills); opening that document with scripts allowed then runs the whole script (owner approved, 2026-11-26). `SceneState` is immutable (frozen dataclasses) and holds every object's resolved parameters.
- Parameters form a DAG of constants, tracks (piecewise segments with easing), expressions over other parameters, and updaters (`obj.always(fn)`; ADR 0005: `fn(t, look)` is pure, and each `look(id)` read is an edge). Evaluate in topological order; a cycle is a validation error. An updater that gives different results on repeat evaluation is lint `impure_updater`; one that needs the previous frame is a `Simulation`.
- Narration lives in a separate `TimeMap` (marker name to seconds). Tracks store marker-relative anchors. Re-recording audio rebuilds only the `TimeMap`.
- Simulations (PDE, FDTD, particles) are `Simulation` objects with `step(state, dt)`, a fixed dt, a seed, and checkpoints every N steps cached in `.phasor_cache/` under a hash of their parameters. `state_at(t)` = nearest earlier checkpoint + remaining steps.
- Randomness only through `rng(object_id, seed)`. Same inputs must give the same frames on the same machine.

**4. One spatial pipeline**

- Everything is 3D. A 2D scene is a 3D scene with an orthographic camera looking down −z at the plane z = 0. There is no separate 2D code path.
- World space: right-handed, z up, scene units. The complex plane maps x = Re, y = Im.
- Screen space: pixels with origin top-left, plus normalized 0–1 coordinates in every inspector output. Titles and HUD panels live in a screen-space overlay layer, with normal objects and IDs.
- Color: linear RGBA16F from kernels through post-processing. Tone mapping and sRGB encoding happen exactly once, at the end.

**5. The object contract**

Every object type implements `Params` and `draw`; a type missing either fails registration (ADR 0005, owner 2026-11-23). `world_bounds`, `features` and `semantics` are optional: when a type does not write them, they are derived from what `draw` records (`core/record.py`), and a type that knows more writes its own. `draw`'s `pass_` argument is optional. A type may instead define `build(state)`, returning other objects (a composite; children get dotted ids). Types shipped in phasor's packages still carry the full set, an `docs/ai/API.md` entry and an example; scene-local types do not need them.

| Member | Returns | Used by |
| --- | --- | --- |
| `Params` | pydantic model of its parameters | validation, GUI panels, MCP patches |
| `world_bounds(state)` (optional) | AABB + bounding sphere | camera solver, culling |
| `features(state)` (optional) | named points/curves in world space (`pole_i`, `tip`, `front`) | inspector, labels, resolver |
| `semantics(state)` (optional) | JSON-serializable math facts | inspector, resolver |
| `draw(ctx, state, pass_)` | draw calls for pass `color`, `id` or `solo` | both renderers |

**6. Layer rules**

- Allowed import direction: `core` ← `style` ← `time` ← `layout` ← `inspect` ← `render` ← `domains` ← `engine` ← `interactive` ← `voice` ← `scene` ← `importers` ← `workspace` ← (`gui`, `mcp`, `cli`, `sketch`), as placed by ADR 0001 (owner-approved 2026-11-18). `interactive`, `importers` and `workspace` are layers added in Milestones 10 to 13 between the engine and the clients (they change no existing import); the owner approved all three on 2026-11-19. The clients are siblings and never import each other.
- `core`, `style`, `time` and `layout` import no GPU library, so they test on any machine.
- Enforce with `import-linter` in CI from Milestone 0.

**7. Layout is baked, then rendered**

- The solver runs at every marker plus sampled times and writes positions into tracks (the "layout bake"). Renderers and the inspector both read the bake, so what the AI inspects is exactly what renders.
- Stage 1: `kiwisolver` for linear relations (align, above, gap).
- Stage 2: label placer. Candidates = 8 compass slots around the anchor plus leader-line slots. Cost = overlap area + occlusion (from the ID buffer) + distance from anchor + movement since the previous key time.
- Hysteresis: a label changes slot only when cost improves by more than 20%, so labels never jitter.
- Constraints the solver cannot satisfy are returned as lint issues, never dropped silently.
- A render worker process may apply the bake its parent `Engine` computed for the same document (`Engine.worker_payload` / `Engine.from_payload`, checked by a fingerprint of the document); it never bakes itself (ADR 0003, accepted 2026-11-22).
- Layer order and dependency inversion for render output: `docs/decisions/0001-layers.md` (accepted).

## Scene document and semantic scene graph

A scene is one document that Python, the AI and the GUI all edit. Python scripts build it; the canonical saved form is `name.phasor.json` plus an optional `name.overrides.json` for tweaks, as fixed in "Core architecture contracts".

**Scene document contents**

- `meta`: title, resolution, fps, theme, color space.
- `objects`: each with `id`, `type`, parameters, style, tags, layer. The parent is the id's dotted prefix (`res.pole_i` belongs to `res`), so it cannot disagree with the id.
- `constraints`: layout rules (see Layout section).
- `timeline`: state tracks, transitions and narration markers.
- `sounds`: notes and sound files on the soundtrack (see "Sound events"; schema version 9).
- `cameras`: 2D/3D cameras and their tracks.
- `assets`: audio, images, sketches, fonts.

**Object model**

- Every object has: stable `id` (human-readable, e.g. `pole_i`), `type`, `semantic` dict (math meaning), `style`, `z`/layer, visibility.
- Types are grouped by layer: primitives (point, curve, surface, volume, text, LaTeX), math objects (from domain modules), composites (groups, axes, legends).
- Parameters can be constants, expressions of t, or references to other objects (`midpoint(pole_i, pole_neg_i)`).
- Objects expose `world_bounds(t)`, `screen_bounds(t, camera)`, `visible_fraction(t)` (occlusion-aware in 3D), `anchor_points` (named spots like `top`, `tip`, `center`).

**Natural references**

A resolver maps phrases to IDs using semantics: "the pole at i", "the left contour", "the third eigenvector", "the wall". Ambiguous matches return candidates instead of guessing.

**Example (Python authoring)**

```python
from phasor import Scene, cx

s = Scene("residues", theme="aurora")
f = cx.Function("1/(z**2+1)", id="f")
plane = s.add(cx.DomainColoring(f, id="plane"))
c = s.add(cx.Contour.circle(center=1j, r=0.5, id="loop"))
s.add(cx.ResidueMarkers(f, inside=c, id="res"))
s.constrain("label(res.pole_i) above res.pole_i gap 0.2")
```

## Timeline, state and narration sync

Time is driven by narration, swaptube-style: the script is split into audio blocks, and animations attach to words, not seconds.

**Two-level time**

- **Macroblock** = one narration chunk (a sentence or clause) with its audio clip. Its duration comes from the audio.
- **Microblock** = a subdivision of a macroblock, aligned to marked words inside it.
- Silent blocks (`pause(1.5)`) are allowed for breathing room.

**Narration script format**

```text
We start with a function that has two poles [[poles_appear]].
Draw a small loop around [[loop_draw]] the one at i.
Only that pole contributes [[residue_glow]].
```

- Audio: record your own voice or use TTS (pluggable: local Piper/Coqui, or a cloud provider).
- Alignment: synthesized voices are spoken piece by piece between markers, so each `[[marker]]` time is exact without a model. Recordings use WhisperX (or a word-timings file) to place markers on words.
- `[[pause 1.5]]` writes a silent block. A marker at the end of a sentence means its end.
- Re-recording re-aligns everything; nothing is hard-timed.

**State tracks and transitions**

- Every animatable parameter is a named state variable: `loop.r`, `camera.azimuth`, `julia.c`.
- A transition sets a target value over a span: `at("loop_draw").to("loop.progress", 1, ease="smooth")`.
- Evaluation is pure: `scene.state_at(t)` computes everything from the tracks. No hidden mutation.
- Spans can be anchored to markers, to the end of what was just started (`s.at("loop_draw").trace("loop").then()`), or to absolute times.
- Built-in easing library plus physical springs (`back`, `spring`). No custom easing functions: a document must mean the same on every machine. `log:<easing>` runs any of them by factors (`a (b/a)^ease(u)`, both ends positive): `focus_pull` and `morph` use it for a camera's `height` and `distance`, so a 600x dive moves evenly.

**Sound events**

As built (2026-11-25). Besides the narration, the soundtrack holds the document's `sounds`:
`Note` (`kind: "note"`: `midi`, `amp` = peak as a share of full scale, `dur` = whole length,
`voice` one of `sine`, `soft`, `pluck`, `bell`) and `SoundFile` (`kind: "file"`: `path` to a
16-bit PCM WAV inside the scene's folder, `gain`). Each has an `id` (same id space as objects;
retired on removal, never reused) and `at`, seconds or a marker plus offset, resolved through
the TimeMap like a track, so re-narrating moves it. Scripts write `s.note(midi, at, ...)` and
`s.sound(path, at, ...)`; patches `add_sound`, `set_sound`, `remove_sound`; the MCP tool is
`sound`. `phasor/time/sound.py` synthesizes and mixes (pure NumPy/SciPy):

- An event starts at sample `round(t * rate)`; the rate is the narration's when it has audio,
  else 48 kHz. Files at another rate are resampled (polyphase). Files are read at mix time,
  so an edited file is heard at once.
- Notes are dry and deterministic: a fixed sum of sines per voice under a linear 5 ms attack
  and 30 ms release inside `dur`, scaled so the peak is exactly `amp`; no randomness.
- Headroom: the narration is never altered. Where narration plus sounds would pass -1 dBFS
  (or the narration's own level, if higher) only the sounds are turned down, by a gain built
  from a running minimum then a running mean over 2 x 10 ms, which provably never exceeds
  the safe gain at any sample: no clipping and no clicks. Below the ceiling the mix is the
  plain sum.
- A note lengthens the scene to its end; a file only to its start (its length is in the file):
  lint `sound_cut` reports a file that runs past the end, `missing_file` a file that is not
  there, `bad_sound_file` one that is not 16-bit PCM. Each stays silent or is cut.
- Rendered videos (every preset, so previews too) carry the mix; the editor window plays no
  audio and lists the events in `Engine.timeline()["sounds"]`.

**Pacing helpers**

- Auto-pause after an idea, sized by how much changed on screen (`Scene.pace()`, run after the verbs).
- Lint warning when too many things change in one block.

## Layout constraints, inspector and linter

This is the part that lets an AI work with its eyes open: it states intent as constraints, then reads back exactly what is on screen.

**Constraints**

- Relations: `above`, `below`, `left_of`, `right_of`, `near`, `align_x`, `align_y`, `inside`, `outside`, `tangent_to`, `attach_to`. `outside` leaves by whichever side needs the least movement; `tangent_to` touches that side's edge of the target's screen box (boxes, not true curve tangency); `a attach_to b at top` puts `a` beside a named spot of `b` (a screen-box spot, or one of its feature points such as an arrow's `tip`). `outside`, `tangent_to` and `attach_to` arrived in Milestone 8.
- A rule reads `subject relation target [gap n]`; the subject is what moves. `label(x)` is the child object `x.label`. Clauses join with `and`.
- Guards: `no_overlap(a, b, ...)`, `on_screen(a, margin)`, `visible_in_3d(a)` (not occluded, Milestone 6), `min_size_px(a, n)`, `readable(a)` (contrast vs background). `on_screen` moves the object when it can; the others are checked by the linter and reported as `unsolved` when broken.
- Solved at every key time (markers, transition ends, a 10 Hz grid) where something it places is shown, and the key either side (while everything it places is hidden it does not run; after an edit that changes only motion from some time on, the keys before it are kept; a verb bakes only as far as where it starts), with priorities `required`, `strong` (default), `medium`, `weak`, and a report of any constraint it could not satisfy.
- A baked position is stored on screen relative to its anchor, so labels stay glued to moving anchors between key times. A label that changes slot jumps; eased moves come from choreography verbs.
- 3D labels: billboard toward camera, auto-nudge out of occlusion, optional leader lines.

**Inspector (query API)**

- `inspect(t)` returns JSON: every visible object with id, type, semantic summary, screen bbox, depth, occlusion %, color, opacity.
- `inspect(t, id)` returns full detail for one object, including math (enclosed poles, eigenvalues, current wavefront radius...).
- `diff(scene_a, scene_b)` returns a structured change list.
- `hit_test(x, y, t)` returns the object(s) under a screen point (used by sketch and paint-over features).

**Linter**

| Rule | Catches |
| --- | --- |
| overlap | Labels or key objects overlapping |
| offscreen | Text or important objects cut by the frame |
| occluded\_label | 3D labels hidden behind surfaces |
| contrast | Text unreadable against glow or domain coloring |
| orphan | Objects that never appear ("never leave" needs enter/exit verbs, Milestone 5) |
| z\_fight | Coplanar surfaces flickering |
| clutter | Too many simultaneous changes in one block |
| tiny | Objects smaller than N px |
| unsolved | Constraints the solver could not satisfy |
| slow\_layout | A layout bake that draws more occlusion buffers than the theme allows |

- `occluded_label` arrived with 3D (Milestone 6); `z_fight` is still to come. In 2D a covered label is an `overlap`; under a perspective camera `overlap` compares only labels with labels and points with points, and a label behind something is `occluded_label`.
- Issues carry the times they were seen and, where the linter is sure, a `fix` (patch ops). `Engine.autofix()` and `phasor lint --fix` apply them.
- `phasor lint scene.py` runs it at sampled times (every 0.5 s by default), every transition end and every marker; output is JSON and human-readable.
- Exit code 1 on errors; `--strict` also fails on warnings.

## 3D spatial awareness

The AI never guesses where things are from pixels: the renderer reports ground truth every frame, summarized in viewer terms. This is the project's most original feature, so build it in the core, not as an add-on.

**Render passes (`render/passes.py`)**

| Pass | Target | Content |
| --- | --- | --- |
| `color` | RGBA16F | The picture |
| `id` | R32UI + depth | Object index per pixel, nearest opaque surface |
| `solo` | R8, one object at a time, depth test off | The object's full silhouette as if nothing hid it |

- The inspector runs `id` and `solo` at 480×270, independent of output resolution. Read back with ModernGL and count with `np.bincount`.
- Each material has `occludes: bool`. Translucent surfaces (alpha below 0.5) do not write to the `id` pass by default.
- Kernel-drawn objects (a domain-colored plane) are quads with one ID. Their inner structure (poles, zeros, wavefronts) is reported through `features()`, projected analytically.

**What is computed per object, per sampled time**

- Screen bbox and center (pixels and 0–1), depth, distance to camera.
- `visible_fraction` = pixels in `id` pass ÷ pixels in `solo` pass, both counted inside the frame. Objects that do not occlude are measured with an `id` pass in which they are forced to write.
- `hidden_by` = IDs found in the `id` pass where the solo mask is set, with the share each one covers.
- `on_screen_fraction` = share of the full silhouette that lies inside the frame. When the bounding box crosses the frame edge it is measured on a zoomed-out `solo` pass (inside pixels ÷ all pixels); this works for any shape, where an analytic area does not exist for text or surfaces.
- An object too small or thin to cover a pixel at 480×270 reports `visible_fraction: null` with a `note`; it is never guessed.
- Coplanar objects (every 2D scene) are ordered by `z_index`, then document order, through a tiny per-object depth bias, so "nearest surface" is always well defined.
- Region name from a 3×3 grid (`upper-left` … `lower-right`), plus `facing` for labels.
- Pairwise relations for tagged objects only: `left_of`, `above`, `in_front_of`, `overlaps`. All are camera-relative.

**Event log (`inspect/events.py`)**

- Sample the whole timeline at 10 Hz plus every marker. Emit an event only when a value crosses a threshold: visibility 0.9 / 0.5 / 0.1, enters or leaves frame, region change, new occluder.
- Thresholds use hysteresis (5%) so one object grazing an edge does not spam events.
- Events carry time, nearest marker, ids and a suggested fix.

```json
{"t": 3.4, "marker": "rays", "event": "occluded", "id": "ir_label",
 "visible": 0.4, "hidden_by": [{"id": "wall_north", "share": 0.6}],
 "fix_hints": ["raise 0.3", "camera.azimuth +15deg", "set wall_north.occludes false"]}
```

**Query API (Python and MCP share it)**

| Query | Answers |
| --- | --- |
| `where(id, t)` | Region, bbox, depth, relations to tagged neighbors |
| `visible(id, t)` | Visible and on-screen fractions |
| `occluders(id, t)` | Who hides it and by how much |
| `when_hidden(id)` | Time spans where visibility is under 0.5 |
| `free_space_near(id, t)` | Empty screen rectangles close to it, for labels |
| `events(t0, t1)` | The event log for a span |
| `frame(ids, during)` | Asks the camera solver to keep these in view |

**Detail levels.** Every query takes `detail`: `brief` (one line per object, at most 40 lines), `standard`, or `full`. The MCP default is `brief`; too much JSON drowns the model.

**Set-of-mark storyboards.** `storyboard(marks=True)` draws each object's outline and ID onto the stills, in colors taken from the ID pass. Vision models then tie what they see to named objects.

**Camera solver (`layout/camera.py`)**

- Input is goals, not angles: `keep_visible(ids)`, `center(id)`, `thirds(id)`, `avoid_occlusion(id)`, `min_size(id, frac)`, `smooth`.
- Unknowns: azimuth, elevation, distance, target point, field of view.
- Stage 1: analytic cost from projected bounding spheres, minimized with `scipy.optimize` from several starts.
- Stage 2: verify the winner with the real `id` and `solo` passes and re-solve if it fails.
- Solve at markers, spline between them, then verify 10 Hz samples along the path. Failures become lint issues with times.
- The result is baked into camera tracks like any other layout.

**As built (Milestone 6).** Camera goals are constraint rules on the active camera: `camera keep_visible(a, b)`, `camera center(a)`, `camera thirds(a)`, `camera avoid_occlusion(a)`, `camera min_size(a, share)`, with the usual priorities. The solver chooses only the orbit-camera parameters the document leaves unset (azimuth, elevation, distance, target; the field of view is never solved, since with distance free it only adds ambiguity), so an animated azimuth (the `orbit` verb) is kept and framed around. A parameter animated but never set, whose first motion starts after 0 (a `focus_pull` or `orbit` at a marker), is solved until that motion starts, and the motion then starts from the solved value: an `orbit` turns by its angle from there (the verb writes its turn relative to the document's own angle, not the baked one), any other motion eases the difference away over its own duration (with its own easing; a `log:` zoom goes by factors from the solved value). Stage 1 frames with projected bounding-box corners (a flat surface seen edge-on is thin; its sphere would never say so) and uses spheres only for occlusion; depth is clamped with a graded penalty so the cost stays smooth. `smooth` penalizes jerks (distance from the straight-line continuation of the two previous solves) plus angular speed, not following a moving object. Overlay text claims its strip of the frame, so framing keeps the world out of a title. Solved at markers, transition ends and 1 Hz, joined with PCHIP, checked at every 10 Hz key and re-solved where a goal fails, up to three rounds; what still fails is `unsolved` with times. `frame(ids, during)` is not a query: add a `camera keep_visible` rule (goals hold for the whole scene). `thirds` is a goal you ask for, not on by default (a centered subject orbits better). 3D labels live at their anchor's depth, nudged 4% toward the camera, slide only sideways, and avoid only what is in front of them (the oracle's `depth`); there is no `facing` field because every label is a billboard. `z_fight` is not a lint rule yet. Glass uses per-triangle back-to-front sorting instead of OIT: exact except where translucent triangles cut through each other.

## Rendering pipeline and visual quality

Two renderers share one scene: a fast preview for iteration and a CUDA-accelerated final renderer for beauty.

**Preview renderer**

- Low-res keyframe stills at markers and sampled times, plus the inspector JSON, in seconds.
- `phasor storyboard scene.py` writes a contact sheet PNG (frames labeled with time + marker) for humans and vision-capable AI.
- Live window with scrubbing and hot reload on save.

**Final renderer**

- Compute: CuPy RawKernels for per-pixel math (domain coloring, fractals, PDE fields, wave simulation). CPU fallback via NumPy for machines without CUDA, slower but correct.
- Raster: ModernGL (OpenGL 4.3) for geometry, surfaces, volumes, text; GPU interop with CUDA buffers where possible to avoid round-trips.
- Vector: curves and text tessellated on GPU with analytic anti-aliasing; LaTeX via a local TeX install or MathJax-to-SVG.
- Encode: PyAV/FFmpeg with NVENC when available; ProRes/PNG sequence for editing.
- Motion blur via sub-frame accumulation; adaptive supersampling on high-detail regions.

**Look**

- Linear HDR pipeline, wide-gamut option. Tone mapping is a shoulder curve: identity below the theme's knee (so paint colors show exactly as the theme defines them), then a smooth roll-off to white. ACES and AgX shift every color, including flat paint, so they are not the default; a theme may add them later as an option.
- The sRGB encode follows the tone map once per pixel: `render/output.py` `encode_srgb8` on the host, and its twin, the `srgb8` kernel, on the card (`finish(..., output="srgb8")`, used by every render), which must agree with it to one 8-bit step.
- Physically based bloom and glow with per-object emission strength: the color pass writes a second target holding only the light glow roles add, and the bloom (a blurred pyramid) is built from that, so plain paint never blooms and thin curves bloom by their energy at any resolution.
- Kernel-drawn planes (domain coloring, fractals) are backdrops: layout lets labels sit on them, and lint judges the label by contrast against the real color pass.
- Depth of field, film grain, vignette, chromatic aberration (all subtle and optional).
- 3D: PBR-lite materials, soft shadows, ambient occlusion, translucent surfaces with order-independent transparency, glowing curves on surfaces.
- Named themes: palette, background, glow levels, typography, line weights. AI picks themes; it does not invent hex codes.
- Perceptual colormaps (and cyclic ones for phase); colorblind-safe checks in the linter.

## GPU and CUDA plan

CUDA does the math, OpenGL draws the picture, and one narrow bridge connects them. Verify every library claim below against current docs in Milestone 0; versions move.

**One kernel path**

- CuPy `RawKernel` is the only way to write GPU math in v1. No Numba, no second path.
- Kernels are CUDA C strings in `render/gpu/kernels/`, registered with `@kernel(name, signature, reference=numpy_fn)`. Registration fails without a NumPy reference.
- CuPy compiles through NVRTC at first use and caches the result, so users need no C++ compiler. NVRTC itself comes from pip via the `cupy-cuda12x[ctk]` extra (phasor's `gpu` extra), so no CUDA Toolkit install is needed either. `phasor doctor` must prove this on a clean Windows machine.
- SymPy expressions become kernel code through a small emitter (`render/gpu/codegen.py`; `sympy.ccode` has no complex type) plus a complex-math header (`render/gpu/complex.cuh`). One expression feeds the kernel, the NumPy reference and the LaTeX label. Formula text is read by a whitelist parser and never evaluated as Python.
- Kernel sources stay plain C++ plus the CUDA keywords, so the same string can be compiled for the CPU in tests (`tests/cuda_emu.py`) on machines without a GPU.
- As built (Milestone 14): that same string is also the tier B backend. `render/cpu/backend.py` compiles each kernel's CUDA source, unchanged, with a g++ or clang++ found at run time (`-O3`, OpenMP where available, no fused multiply-add), runs its blocks on every core and caches the library in the user cache, keyed by a hash of the code, compiler and flags. Integer atomics are real atomics and float atomics are refused, so results are the same bits at any thread count. `registry.run(backend="auto")` tries `cuda`, then `cpu`, then `reference`; no compiler means the NumPy reference. No prebuilt binaries. MSVC `cl` is not supported. Windows is untested.

**CUDA to OpenGL bridge (`render/gpu/bridge.py`)**

- One interface: `FrameBridge.upload(cupy_array) -> moderngl.Texture`.
- `CopyBridge` (build first): `cupy.asnumpy` then `texture.write`. A 4K RGBA16F frame is about 66 MB, which is fine for offline rendering.
- `InteropBridge` (optimization, Milestone 3 stretch): register a GL pixel buffer with CUDA through the `cuda-python` bindings and copy on the GPU. Ship only if it measures faster and passes the same golden-frame tests.
- Nothing outside the bridge may move pixels between CUDA and GL.

**Memory planning**

- `render/gpu/budget.py` reads free VRAM and plans tiles. Kernels always accept a tile rectangle; the caller stitches.
- Default budget: use at most 60% of free VRAM.
- Reference sizes: a 4K RGBA float32 buffer is about 133 MB; at 2×2 supersampling about 530 MB. A 256³ FDTD grid with three float32 fields is about 200 MB; 512³ is about 1.6 GB.
- Call `free_all_blocks()` on the CuPy pool between scenes and every N frames in long renders.

**Precision policy**

| Case | Method |
| --- | --- |
| Default per-pixel math | float32 |
| Fractal zoom past about 1e-5 scale | float64 kernels |
| Fractal zoom past 1e-10 per pixel (as built; first planned at about 1e-13) | Perturbation: one reference orbit in arbitrary precision on CPU (fixed point on Python's exact integers; first planned with `gmpy2`), per-pixel deltas in float64 on GPU, glitches caught by `|z| < |dz|` and repaired by rebasing onto the reference's start |
| PDE and FDTD fields | float32, with a float64 test mode |

- FDTD stability: enforce dt ≤ dx / (c·√3) in 3D and raise a validation error otherwise.

**Encoding**

- `render/encode.py` probes at startup, in order: PyAV with `h264_nvenc`/`hevc_nvenc`, a system `ffmpeg.exe` fed raw frames over a pipe, then PyAV `libx264`. H.264 comes first because it plays everywhere (owner, 2026-11-20); `h264_nvenc` runs at constant quality (cq 19), `libx264` at crf 16. A probe must encode real frames: PyAV wheels list NVENC codecs even on machines without an NVIDIA GPU.
- Audio is muxed in the same step from the narration blocks and the sound events ("Sound events").

**Hardware tiers**

| Tier | Hardware | Behavior |
| --- | --- | --- |
| A | NVIDIA with CUDA 12 | Everything |
| B | Any GPU with OpenGL 4.3, no CUDA | Kernels run on every CPU core through the CPU backend when g++ or clang++ is found (as built, Milestone 14), else on the NumPy references; final renders allowed, warned as slower than tier A |
| C | No usable GPU | Timeline and lint work; the inspector runs in analytic mode (exact geometry, no occlusion, marked `"occlusion": "analytic"`); no rendering |

**Testing and benchmarks**

- Every kernel: GPU output vs NumPy reference on a 64×64 tile, within tolerance.
- Golden frames: SSIM ≥ 0.99 against stored stills, per demo scene.
- `phasor bench` reports ms per frame per kernel and peak VRAM; results are stored in `benchmarks/` so regressions show in review.

## Beauty system

The AI handles structure, math and timing; the engine and your gallery hold the taste. An AI that does nothing special must still produce a good-looking frame.

**Themes (`style/themes/*.toml`)**

- A theme defines everything visual: palette, background, type scale, stroke weights, glow levels, spacing unit, easing defaults, lighting rig, post-processing.
- Scenes and AI patches may only use style roles. Raw hex codes, pixel widths and font sizes in a scene are a lint error (`raw_style`), unless marked `style_override=True`.

| Role group | Roles |
| --- | --- |
| Emphasis | `primary`, `secondary`, `muted`, `highlight`, `danger` |
| Stroke | `hairline`, `regular`, `bold` |
| Glow | `none`, `soft`, `strong` |
| Text | `title`, `label`, `annotation`, `math` |
| Surface | `solid`, `glass`, `wire`, `emissive` |

- Ship three themes tuned by hand: `midnight_glow`, `aurora`, `paper`. Add more only after these look great.
- **The dark theme is the default, always** (owner, 2026-11-18; replaces an earlier note asking for light defaults). `midnight_glow` stays the house theme. **The `prism` look is the default style** (owner, 2026-11-22): a scene that names neither a look nor a theme gets it; `look="none"` gives plain `midnight_glow`. Light and mid-tone themes are opt-in choices, and glow and bloom must still read on them (soft colored halos, not white blow-outs).

**Living motion: no frozen frames (owner, 2026-11-18)**

Things on screen should feel alive, not rigid. Every theme has a `[life]` section and every scene a `life` setting (`off`, `subtle`, `lively`), so the effect can be turned off entirely.

- Glow breathes: bloom strength and glow roles swell and fade slowly instead of sitting at one value.
- Points and markers drift: a dot or pole wanders a tiny distance around its true position (a fraction of its own size), never far enough to mislead. Labels follow their anchors.
- Fields drift: a fractal or domain-colored backdrop slowly pans, rotates its colors or zooms by a hair.
- Curves shimmer: glowing strokes vary slightly in brightness along their length over time.
- Rules that keep it honest:
  - **Purity holds (contract 3).** Life is a deterministic function of (object id, seed, t), built from smooth noise via `rng(object_id, seed)`. The same frame always renders the same, and `state_at(t)` still depends only on the document and t.
  - **The inspector sees true positions plus the wiggle**, and reports `life` amplitudes, so lint and layout account for it (a label's slot must hold through the whole wiggle).
  - **Meaningful geometry stays exact.** Life never moves a pole off a computed location by more than the theme's tolerance, never changes a value that a label states, and is off on objects tagged `exact`.
  - **Amplitudes and speeds live in the theme**; the AI picks a preset, it does not tune numbers.

**Softer, less clashing pictures (owner, 2026-11-18)**

- Boundaries should not clash: hard edges between regions (domain-coloring contour bands, the rim of a backdrop, a curve crossing a field) are softened or feathered by default. Sharp edges are a choice, not the default.
- Fewer rainbows, not no rainbows (owner, 2026-11-18): use judgement. Where hue carries meaning, a full wheel is right: domain coloring encodes phase as hue, so it keeps its rainbow. Elsewhere, vary the palettes and mix things (two or three theme hues, sequential ramps, a muted wheel, monochrome with one accent) instead of reaching for a rainbow every time. The `color_restraint` lint rule judges a frame's saturated hues but does not count a phase-coding backdrop against it.

**Presets**

Making something should start from a preset, not from a blank set of parameters.

- Look presets bundle theme choices: background (dark by default; mid or light on request), palette restraint, edge softness, glow level and life level. Examples: `calm` (soft edges, subtle life, gentle glow), `chalk` (opt-in paper background, no glow, crisp), `glow` (lively bloom).
- Object presets bundle parameters per object type: a domain-coloring preset picks colormap, contour softness and drift; a fractal preset picks palette, iteration depth and drift; a curve preset picks stroke, glow and shimmer.
- Usage: `Scene("x", look="calm")`, `cx.DomainColoring(..., preset="soft_phase")`. A preset is just named defaults; anything set explicitly wins, and the document stores the resolved values so it stays reproducible.
- The AI chooses presets by name (documented in `docs/ai/`), the same way it chooses style roles.

**As built (Milestone 7).** Looks (`style/looks/*.toml`: `calm`, `glow`, `chalk`) pick the theme and `life` and supply per-type defaults; presets live on the object classes (`Presets`); explicit values beat a preset, which beats the look, which beats the type's default, and the document stores resolved values (`meta.theme`, `meta.life`, `meta.seed`, `meta.look`; schema v5, older documents migrate with life off). Dark (`midnight_glow`) is the default; `aurora` is a second dark theme and `paper` a light, opt-in one. Palettes are colormap roles (`phase`, `soft_phase`, `sequential`, `ember`, `ocean`); domain coloring keeps the full `phase` wheel. Contour bands are feathered by the `domain` kernel's `soft` argument (object param `softness`, 0.5 default). Life (`core/life.py`) is applied where parameters are resolved: dots drift a fraction of their radius, glow breathes (strokes shimmer faster, uniformly along their length, not per point), color fields turn through a `hue` parameter (a field's geometry never moves); objects tagged `exact` do not move; `life` defaults to `off` for plain `Scene(...)` and for migrated documents, and looks turn it on. The inspector reports the true, wiggling positions. Aesthetic rules are `lint(aesthetic=True)` (`inspect/aesthetic.py`); `stroke_mix` fires above two weights (the theme has only three roles, so "more than three" could never fire); `glow_clip` counts light more than three times plain paint, before bloom; the full-wheel phase coloring is exempt from `color_restraint`. `Engine.critique()` scores them and `polish()` applies certain fixes and one judgement (which highlight to keep), three rounds at most; the vision critic itself is the AI reading `STYLE.md` and the storyboard through the MCP tools `critique`, `variants`, `choose`. The gallery compare (`storyboard(compare=...)`) matches frames by a coarse color layout and brightness histogram. `variants` offers as-is, calmer and bolder, rendered from a throwaway engine; `phasor style tune` reports what `style/choices.jsonl` favors but does not yet edit a theme. Motion blur (`Engine.frame_blurred`, on for `final` and `4k`) averages sub-frames only where a transition runs. Not built: a gallery of your own (the folder is yours to fill), a lint rule for labels in front of key objects in 3D, per-vertex stroke shimmer, shadows and ambient occlusion. Boundary review (2026-11-19): what living motion may do is declared per type (`ObjectBase.Life`: glow by default; dots wander, color fields turn hue), not guessed from parameter names; motion blur averages light before the single tone map and also blurs motion driven by expressions of `t`. The prism look (owner, 2026-11-19, after a reference video on the explicit formula for primes) added: typeset math, `$...$` in any text through matplotlib's mathtext in Computer Modern (as built instead of "a local TeX install or MathJax": no install, identical on every OS; mathtext's TeX subset), `Equation` (colored parts on one baseline), `Line(hue_along=...)` (a color per vertex, the `spectrum` colormap), the lens (`[bloom] fringe` on the glow only, `[bloom] vignette` after the tone map, both registered kernels, off by default), the life channel `float`, and the `prism` theme and look.

**Choreography verbs (`time/verbs.py`)**

- The AI animates with verbs, not raw tweens: `reveal`, `trace`, `focus_pull`, `bloom_in`, `settle`, `morph`, `orbit_reveal`, `dim_others`, `exit`.
- Each verb expands into state transitions with tuned duration, easing, stagger and overshoot, read from the theme. Segments remember the verb that made them.
- Dimming uses a separate `dim` factor per object (drawn opacity = opacity x dim), so `dim_others` and `reveal` never fight; a reveal drops earlier dimming.
- Raw `to()` transitions stay available but lint as `raw_motion` in strict mode.

**As built (palettes).** `meta.palette` (schema 7) picks the colors run along strokes: the prism looks keep their lens and glow and take any palette (`spectrum`, the rainbow, by default; `moonlight`, `gold`, `glacier`, `borealis`, `rose`, colormap roles in every theme). It colors the theme's `[strokes]` hue and the lines a look runs a hue along; `scene_palette` switches it on an open scene.

**As built (motion verbs, after the roadmap).** Eight more verbs, timed by the theme like the rest: `transform` (an equation becomes the next; parts matched by longest common subsequence glide over, the rest fade), `become` (a curve takes another's shape; closed curves aligned so points travel little; the source's points are resampled in place, so it needs them unanimated before), `write` (text and equations written in left to right), `travel` (a `Tracer` along any curve offering `path_points`), `flash` (every object has a `flash` parameter the renderer turns into light and glow), `circumscribe` (adds or reuses the child `<id>.ring`, a `Callout`), `wiggle` and `sweep` (adds the child `<id>.readout`, a `Counter` bound by an expression). New objects: `Tracer`, `Callout`, `Counter`, `cx.ComplexMap` (a grid through f, `amount` along the straight path from the identity, or with `path="family"` along a family f(z, a) holomorphic at every step, preset `roll` = (e^{az} - 1)/a; lines refined where they bend) and `geo.FlowParticles` (random starts from `rng(id, seed)`, a pure function of t); `geo.ParametricCurve`/`ParametricSurface` blend into other formulas (`to_f*`, `blend`). `write` crops text with a hard edge (the bitmap has no stroke order), not true handwriting.

**As built (unfold).** `unfold(room, ray, flips)` (echo video, animation 2): a ray (a `Line`) bouncing in a box room shown as a straight line through mirror rooms. The ray moves at a steady speed; at each wall it reaches, a `HingedCopy` of the room (`<room>.mirror_<where>`, named by its place in the lattice of mirror rooms: `mirror_e2_n1`) swings over in the theme's `[motion.unfold]` time and lands as the ray arrives. A copy is its source drawn through a chain of hinge turns (strokes are drawn for the camera moved the opposite way, facing the viewer mid-turn); half a turn about the wall's edge is the mirror image in 2D, about the wall's level middle line it lands a box exactly on its mirror box in 3D. The maths is `core/unfold_math.py` (in `core`, not `domains/`, because the verb in `time` uses it; see the deviation notes 11 and 12 in `MILESTONE_15.md`). Beyond the box (`room_unfold`): any convex room, a convex polygon (a closed `Line`, an `algebra.Polygon`: billiards, the kaleidoscope) or a convex mesh `Room3D`, its copies named by the walls crossed (`mirror_b_a_c`, walls `a`, `b`, ... in the room's order); a room that is not convex is refused. Where a turn cannot land on the true mirror room (a polyhedron not symmetric about the plane through the hinge, square to the wall), the hinge mirrors the copy in that plane first and the half turn lands it on the reflection; the landed copy is always the true mirror image. The ray may be one ray of a `SoundRays` (`index=`, drawn as `<rays>.unfolded_<index>`). At a corner the flips follow one another, all landed as the ray arrives.

**As built (nets).** `net.Polyhedron` (a composite): a box or any convex polyhedron opening into a flat net, its faces the parts `<id>.face_<name>`, each carried through its chain of hinge turns up a spanning tree of faces (the `Hinge` chains of `HingedCopy`; the turn about a shared edge is the angle between the outward normals), every turn `unfold` (0-1) of the way. With `geodesic` (two points on the surface) the shortest surface path (`domains/net_math.py`: every run of faces laid flat, a window of directions narrowed edge by edge) is the part `<id>.geodesic`, bent on the solid and straight on the net; the default tree keeps its faces joined. No new verb: the fold is one number, animated with `morph` (theme `[motion.morph]`, or `until=` a marker), like `HingedCopy.turn` and `Polygon.flip`.

**Aesthetic lint rules (added to the linter)**

| Rule | Fails when |
| --- | --- |
| `focal_point` | A beat has zero or more than one `highlight` object |
| `color_restraint` | More than 3 saturated hues on screen |
| `glow_clip` | More than 1% of pixels clip to white after tone mapping |
| `density` | Occupied screen area is under 8% or over 60% |
| `motion_rhythm` | Unrelated objects move in the same 0.3 s window |
| `margin` | Content enters the outer 5% safe margin |
| `stroke_mix` | More than 3 stroke weights in one frame |

Thresholds live in the theme, so your taste can move them.

**Golden gallery and style rubric**

- `gallery/` holds frames and short clips you love, each with a one-line note on why it works.
- `STYLE.md` is the rubric in your words: hierarchy, negative space, color restraint, motion rhythm.
- `storyboard(compare="gallery")` puts new frames beside the three nearest gallery frames for the critic.

**Art-director pass**

- A second agent that did not build the scene critiques the storyboard against `STYLE.md` and the gallery. It returns issues tied to object ids and markers.
- The builder fixes them, at most 3 rounds, then reports what is still open.

**Variants and your taste as data**

- `variants(beat, n=3)` renders cheap alternative storyboards of one beat (different verbs, camera, glow).
- You pick in the GUI, or the critic picks. Every choice is appended to `style/choices.jsonl` with the two options and the winner.
- `phasor style tune` nudges theme values toward the winners. This is how defaults drift toward your eye over time.

**3D auto-cinematography**

- Every 3D theme includes a three-point light rig, depth fog and a slow default camera drift of a few degrees.
- The camera solver's `thirds` and `smooth` goals are on by default.
- Flat lighting and a frozen camera are what make 3D look amateur; neither is the default here.

## Math domain modules

Each domain is a plugin package (`phasor.cx`, `phasor.fourier`, ...) that registers semantic object types, GPU kernels and lint rules. All share the core; none is special.

| Module | Objects and features |
| --- | --- |
| `cx` complex analysis | Domain coloring (phase, modulus contours, hybrid styles); conformal grid warps; contours that know enclosed poles, winding numbers and residues; animated contour integrals (running sum of f dz); Riemann surfaces and branch cuts in 3D; analytic continuation; Laurent/Taylor series convergence discs; Möbius transforms; argument principle |
| `fractal` | Mandelbrot, Julia, Newton, Burning Ship, multibrot; perturbation + series approximation for deep zooms; smooth/orbit-trap/distance-estimate coloring; parameter-space ↔ dynamical-space links; 3D Mandelbulb via ray marching |
| `fourier` harmonic analysis | Epicycles; series partial sums with Gibbs; DFT/FFT bins; spectrograms; windows and leakage; convolution theorem; Laplace and z-transforms; spherical harmonics on spheres; Bessel modes on drums |
| `linalg` linear algebra | Transform grids in 2D/3D; eigenvectors/eigenvalues; determinants as signed area/volume; SVD as rotate-stretch-rotate; projections; change of basis; null/column spaces; Gram–Schmidt; matrices with live entries linked to the visual |
| `algebra` abstract algebra | Cayley graphs and tables; symmetry groups acting on shapes (dihedral, cube, icosahedron); cosets and quotients; homomorphisms as maps; group orbits; rings mod n as clock diagrams; field extensions as towers; Galois correspondence diagrams; permutation animations |
| `pde` | Heat, wave, Schrödinger, Laplace/Poisson, advection, reaction–diffusion; GPU finite-difference and spectral solvers; 2D fields as surfaces/heatmaps, 3D as volumes or slices; boundary conditions as visible objects; separation of variables with mode decomposition |
| `acoustics` | Wavefronts and rays in 2D/3D rooms; image-source method; reflection/absorption per material; impulse response; RT60 decay; early reflections vs late tail; FDTD pressure field; convolution reverb; listener/source objects |
| `calc` / `analysis` | Riemann sums, limits, epsilon-delta, Taylor approximations, vector fields, flux, curl, divergence, Stokes/Green |
| `prob` | Distributions, CLT, random walks, Markov chains |
| `geometry` / `topology` | Parametric surfaces, curvature, geodesics, Möbius strip, torus maps, covering spaces |

**Symbolic link**

SymPy is the bridge: an expression drives numerics (lambdified to GPU), the LaTeX label and semantic metadata from one source. Editing the formula updates plot and label together. Equation morphs match terms structurally, not glyph by glyph.

**As built (Milestone 8).** `core/simulation.py` is the `Simulation` of contract 3: a fixed `dt`, `params()` (seed included), `initial()`, pure `step`; `SimRunner.state_at(t)` is the nearest earlier checkpoint plus the remaining steps, memoized and cached on disk under `.phasor_cache/<hash of the parameters>/` (`PHASOR_CACHE` overrides), so it depends only on the parameters and t however you scrub; a damaged checkpoint is recomputed. The three modules are `phasor.fourier` (epicycles, series partial sums with the Gibbs overshoot, spectrum with windows and leakage, spectrogram, FFT convolution), `phasor.linalg` (transform grids in 2D and 3D, eigen lines, vectors, matrices whose entries are linked by expression; determinant, eigenvalues, SVD stages, rank, null space, Gram-Schmidt) and `phasor.ac` (shoebox room, source, listener, wavefront, image sources, impulse response with Schroeder RT60, a 3D FDTD pressure slice that refuses an unstable `dt`). Numerics live in `domains/*_math.py` with no drawing. Not built: Laplace and z-transforms, spherical harmonics, Bessel drum modes, null/column-space pictures, change-of-basis and projection visuals, an interactive absorption slider (Milestone 12), ray tracing beyond the image-source method, convolution-reverb audio export, and the narrated reverb film (its acceptance test, `tests/test_reverb.py`, checks the physics and the inspector instead). The FDTD runs in NumPy (a GPU solver is a later optimization); its walls absorb through a damping shell, an approximation, not an exact boundary condition.

**As built (Milestone 9).** `phasor.pde` (heat 1D/2D, waves 1D/2D, advection, Gray-Scott reaction-diffusion, Schrodinger by split-step Fourier, Poisson/Laplace by a sparse direct solve; each time-stepping model is a `Simulation`, a step that would be unstable is a validation error, and every solver is tested against its exact solution), `phasor.algebra` (dihedral, cyclic and symmetric groups as tables: subgroups, cosets, normality; a polygon the dihedral group acts on, Cayley tables, clock diagrams) and `phasor.calc` (Riemann sums by six rules, Taylor polynomials, epsilon-delta, vector fields with divergence, curl and Green's theorem, graphs that break at poles; functions are text read by the codegen whitelist, `parse_expression`). Objects: `pde_profile`, `pde_field`, `pde_surface`, `polygon`, `cayley_table`, `clock_diagram`, `graph`, `riemann_sum`, `taylor`, `vector_field`. Milestone 9b added the perturbation deep zoom: `cx.Fractal` takes `center` (exact decimal text) and `zoom` (powers of ten, to 280), switches to perturbation under 1e-10 per pixel (measured: plain float64 already gets about 1% of pixels visibly wrong there), and `steady` holds a dive's outer color; every pixel of a 1e-30 view is tested against exact iteration. Mandelbrot only (no Julia perturbation), no series approximation. The rest of the module tables followed (still Milestone 9): `pde.Profile` boundary conditions drawn and chosen (held or insulated ends) and a spectral solver exact per mode, `pde.Modes` (separation of variables, each mode decaying at its own rate), `pde.Slice` (a plane through a 3D heating cube, `Heat3D`); `algebra.CayleyGraph`, cosets and quotients on `CayleyTable` (`subgroup`, `cosets`), `Homomorphism` (clocks Z_n -> Z_m with kernel and image), `SubgroupLattice`, `GaloisDiagram` (square-root towers over Q, fields beside the subgroups fixing them), `Permutation` (travelling tokens, cycles, sign); `calc.Limit` (epsilon-delta bands), `calc.LoopIntegral` (flux and circulation around a circle, with the divergence theorem and Green's theorem checked in the semantics). Still not built: GPU PDE solvers, spectral solvers beyond 1D heat, volume rendering (3D PDEs are shown as slices), Stokes' theorem on surfaces in 3D, rings beyond the clocks, field towers other than square roots of primes, symmetry groups of the cube and icosahedron, group orbits as pictures.

## Interactivity and outputs

One scene exports to several outputs without rewriting it.

- **Video**: MP4/H.264/H.265 (NVENC), ProRes, PNG/EXR sequences, GIF/WebM loops; vertical presets for shorts.
- **Stills**: high-res PNG/EXR posters, SVG for 2D vector content.
- **Interactive desktop player** (as built, Milestone 12): the live window with sliders bound to state variables (e.g. `julia.c`, `room.absorption`), drag handles on objects (move a contour, a source, a wall), hover readouts (f(z), |f|, arg f, winding number, pressure).
- **Interactive web export** (later milestone): WebGPU/WebGL build of the scene with the same sliders; GPU kernels transpiled to WGSL where possible, prebaked textures otherwise.
- **Notebook widget**: inline player in Jupyter for exploration.
- Parameters marked `interactive=True` become sliders automatically; constraints and linter keep running live.

**As built (Milestone 12).** Sliders are document **controls** (schema v6: `Control(path, component, label, min, max, step)`; ops `add_control`/`remove_control`; `Scene.control`; lint `orphan_control`; diff entries), declared by the scene instead of an `interactive=True` flag on each parameter. `phasor.interactive.Player` moves them on a private `Engine.fork()` (the scene never changes); the editor's Sliders tab previews them (Apply is one undo step), `notebook(engine)` returns ipywidgets (extra `notebook`), and `export_web` / `phasor web` bake one picture per combination of slider positions (and optional time steps) into a single offline HTML file with the same sliders, refusing more than 600 pictures. Not built, by decision: WebGPU/WGSL kernels (baked pictures work everywhere with no server, at the cost of a coarse grid), drag handles that edit through sliders, hover readouts beyond f(z)/|f|/arg f and the escape count (no winding number or pressure yet).

**As built (Milestone 13, Manim import).** `phasor/importers/manim.py` runs a Manim Community script unchanged with a stand-in `manim` module on the import path and records the phasor scene it builds at Manim's own times (see `docs/ai/API.md` "Manim import" for the table). It runs the script: only import scripts you trust. Colors become style roles, fills outlines, updaters and `always_redraw` are baked, `Transform` between different shapes crossfades; each is a warning. `phasor manim`, and `scene_open` / `phasor gui` on a script that imports `manim`; the result is a plain (not script-locked) scene.

**As built (space and math, 2026-11-19).** `phasor.geo`: `Axes`, `ParametricCurve`, `ParametricSurface` (colored by an output; `with_map` carries a flat input square onto the output shape), `VectorField3D`, `FlowLines`; `calc.VectorField(color_by_size)`. The vertical light shafts (`Haze`) of the first prism look were removed at the owner's request (the Fourier picture's own artifact, not a style element). The editor gained a code tab (the scene as Python: the script, or one generated from the document; Run rebuilds the scene), 3D orbit/pan/zoom by dragging, a narration lane with waveform, draggable markers and spans, a who-changed-what dock and hover readouts; `phasor-gui` starts it as a program. LICENSE: MIT.

## Sketch-driven authoring

Sketches become editable scene objects, not pixels. This works because the scene graph knows what sits under every stroke.

- **Sketch to scene**: draw on a tablet canvas in the GUI, or import a photo of paper. Strokes are classified (circle, line, arrow, dot, curve, handwriting). A vision model plus geometric fitting proposes objects: circle → contour, dot → pole/point, arrow → vector, wavy line → wave, handwriting → LaTeX (via handwriting-to-LaTeX OCR). You confirm or fix each proposal.
- **Freehand paths as math**: draw any closed curve over a plot; it is smoothed to a spline and becomes a real contour (integral computed and animated along it), integration region, boundary condition, ray path, or camera path.
- **Keyframe by drawing**: sketch frame A and frame B; objects are matched by ID/semantics and the engine creates the transitions.
- **Paint-over corrections**: scribble on a preview frame (circle a region, arrow a direction, write "bigger"). `hit_test` turns marks into object IDs, so the AI gets "move `label_res` up 0.3, it overlaps `loop`" instead of a pixel guess.
- **3D camera sketching**: draw a path on a top-down or side view; it becomes a smooth eased camera track that keeps targets framed.
- **Style from an image**: give a reference image; extract a palette, glow strength and background into a reusable theme.
- **Storyboard import**: a sequence of rough sketches + narration lines becomes a first-draft scene with markers.

**As built (Milestone 11).** `phasor/sketch/` is a client (the `sketch` slot of contract 6): `strokes.py` classifies a stroke by geometric fitting (dot, line, arrow from a hook or a V at the tip, circle by an algebraic circle fit, closed, scribble by direction reversals, curve and wavy curve; small strokes close together are handwriting), with thresholds as shares of the stroke's own size. `propose.py` turns shapes into proposals (patch ops plus verbs, applied with `accept`) that depend on what is under them: a loop over a domain coloring is a contour traced from the playhead with its integral (numeric along the path, exact from residues and winding numbers for rational functions) and a label; a loop around objects emphasizes them; a dot snaps onto a pole or zero; an arrow from an object moves it; a scribble removes what it crosses (paint-over through `hit_test`). `phasor sketch` takes strokes as JSON; the editor's **Sketch** tab draws on the viewport (it reaches the GUI through the `phasor.panels` entry point, so neither client imports the other). Handwriting to LaTeX has no offline recognizer: the proposal carries a picture of the strokes for an AI (or the person) to read, and `transcribe` makes the Text. Not built: photo import of paper, keyframe by drawing, 3D camera sketching and storyboard import (the other bullets above); a vision model proposing objects.

## AI integration

Any AI should be able to author, inspect, critique and repair scenes through tools, and learn the library from docs written for it.

**MCP server (`phasor mcp`)**

| Tool | Purpose |
| --- | --- |
| `scene_open` / `scene_save` | Load or save a scene document |
| `scene_edit` | Apply a structured patch (add/remove/update objects, constraints, transitions) |
| `inspect` | JSON snapshot at time t (all objects or one id) |
| `lint` | Run the linter; returns issues with object ids and suggested fixes |
| `storyboard` | Render keyframe stills + contact sheet; returns image paths/images |
| `hit_test` | Objects under a screen point |
| `resolve` | Natural-language reference to object ids |
| `diff` | Structured diff between two scene versions |
| `render` | Final render with preset; returns progress and output path |
| `docs_search` | Search the API reference and examples |

**As built (Milestone 5).** The table above is the minimum; the server also has `scene_new`, `animate` (the choreography verbs), `narrate`, `review` (the self-review loop below in one call: lint, inspector lines at every marker, storyboard image), `undo`/`redo`, `checkpoint`/`restore`, and `object_types` (type names, parameters, style roles). Tool logic is plain Python in `phasor/mcp/tools.py` (`Workspace`), served by `phasor/mcp/server.py`. Safety: paths stay inside `--root`; `.py` scenes are code and are refused unless `--allow-scripts`; a failed patch changes nothing. `phasor mcp` registers itself with the CLI through the `phasor.commands` entry-point group, because sibling clients may not import each other (contract 6). Milestone 10 moved the tool logic to `phasor/workspace/` (both clients share it) and added `phasor mcp --attach`, which edits the scene open in `phasor gui`.

**Self-review loop (built in)**

1. Edit the scene.
2. `lint` and `inspect` at every marker.
3. `storyboard` and look at the frames (vision models).
4. Fix issues; repeat until clean or N rounds.
5. Report what changed and any unresolved warnings.

**Docs written for AI (`docs/ai/`)**

- `API.md`: every public object/function, one canonical example each, no deprecated aliases.
- `PATTERNS.md`: idioms (label a pole, animate an integral, orbit a 3D camera, sync to narration).
- `PITFALLS.md`: known traps and how the linter reports them.
- `examples/`: 50+ small, tested scenes across all domains, each with narration, as a pattern library.
- A generated `llms.txt` summary, plus a Claude skill (`SKILL.md`) packaging the above so Claude uses phasor fluently.
- Docs are tested: every code block in `docs/ai/` runs in CI.

**Safety rails for AI edits**

- Patches are validated against the schema before applying.
- Undo history and named checkpoints; the AI can branch a scene and compare.
- Human GUI edits are recorded as constraint/parameter changes so the AI sees intent, not just new coordinates.

## GUI editor

The GUI is a third editor on the same scene document, so human tweaks and AI edits never fight.

- **Viewport**: live 2D/3D preview with orbit/pan, object selection, drag handles, gizmos for 3D.
- **Timeline panel**: narration text with markers, audio waveform, state tracks as lanes, drag spans to retime.
- **Inspector panel**: selected object's parameters, semantics, constraints; edit values directly.
- **Lint panel**: live warnings; click to jump to object and time.
- **Sketch layer**: pen/tablet input for sketch-to-scene and paint-over corrections.
- **AI chat dock**: talk to an agent connected via MCP; it sees the current selection, time and lint state.
- **Code view**: the generated Python/scene file, two-way synced (edit code or GUI).
- Tech: Dear PyGui or Qt (PySide6) hosting the ModernGL viewport; decide in Milestone 4.

**As built (Milestone 10).** Qt (PySide6-Essentials, the `gui` extra): it runs headless on Qt's `offscreen` platform, so the window is tested in CI, and it is mature on Windows. The viewport shows the engine's own frames (`Engine.still`) as an image rather than embedding a GL widget, so the editor sees exactly the final look. All logic is `phasor.gui.controller.Editor` (no Qt; every edit through the shared `Workspace` and so `Engine.apply`); `gui/window.py` draws it. Built: viewport (click to select, drag the selected object in its plane; a dashed box from the inspector's bounds), timeline (markers, one lane per animated parameter from `Engine.timeline()`, click or drag to scrub, play), inspector (each parameter as a JSON/text field, style roles as menus, constraints, semantics; animated parameters show their value at the playhead), lint panel (click to jump to object and time), looks tab (apply a look to the whole scene as one undo step; variants of a beat, a pick recorded as the person's), MCP attach (`Workspace` moved to `phasor.workspace`, shared by both clients; `workspace/remote.py` serves it on 127.0.0.1 with a token in `<root>/.phasor/gui.json`, mode 600; `phasor mcp --attach` builds the same MCP server on a proxy with identical tool signatures; calls share the editor's lock). Not built: 3D gizmos and orbit/pan of the view (the camera is a document object: edit it in the inspector), audio waveform and narration text in the timeline, dragging spans to retime, the AI chat dock (an MCP client such as Claude attaches instead), the two-way code view, the sketch layer (since built: Milestone 11, the Sketch tab). The exit demo uses the prism scene, since the reverb scene was skipped in Milestone 8.

**As built (desktop app, after Milestone 13).** `phasor-gui` is a desktop program: a start screen (templates, open, recent, examples copied to `Documents/phasor` before opening), one window per scene with a menu bar, unsaved-change prompts, Save As anywhere (the workspace folder moves with it), settings (math typesetting) in the platform's settings folder, and File > Export (picture, video, web page) run as a separate `phasor` process with progress and cancel, so the GL context is never shared between threads. Logic in `gui/app.py` (no Qt), widgets in `gui/desktop.py`. Packaged with PyInstaller (`packaging/`): built and run on Linux and Windows 11 (macOS untested); CuPy left out of the package unless `PHASOR_PACKAGE_GPU=1`, which bundles it with NVRTC and the CUDA runtime only.

## Tech stack, Windows, repo layout

Everything runs natively on Windows 10/11 with an NVIDIA GPU; no WSL, no C++ toolchain required for users.

| Layer | Choice | Notes |
| --- | --- | --- |
| Language | Python 3.11+ | Type hints throughout; `uv` for env and deps |
| GPU compute | CuPy (RawKernel) only | CUDA 12.x wheels; NumPy fallback |
| Graphics | ModernGL (+ moderngl-window) | OpenGL 4.3; consider wgpu-py later for WebGPU export |
| Encode | PyAV (FFmpeg) | NVENC when present |
| Symbolic | SymPy | Lambdify to CuPy/NumPy |
| Text/LaTeX | MiKTeX or TeX Live, or MathJax to SVG | Glyphs cached as GPU textures/curves |
| Audio | soundfile, WhisperX, pluggable TTS | Word-level alignment |
| Constraints | Custom solver (kiwisolver for linear layout) | Plus a nudger for overlaps/occlusion |
| AI | MCP Python SDK | `phasor mcp` command |
| Tests | pytest, image snapshot tests | Golden frames with tolerance |

**Repo layout**

```text
phasor/
  engine.py    Engine: document, state cache, apply(patch), undo
  core/        schema, migrate, IDs, resolver, SceneObject protocol
  time/        tracks, easing, TimeMap, narration alignment, verbs, simulations
  layout/      constraints, label placer, camera solver, layout bake
  inspect/     inspector, queries, events, hit test, diff, lint rules
  render/      passes, preview, final, post-fx, encode
  render/gpu/  kernels/, bridge.py, budget.py, complex.cuh
  style/       themes/, roles, choices.jsonl, tune
  domains/     cx, fractal, fourier, linalg, algebra, pde, acoustics, calc, prob, geometry
  sketch/      stroke capture, classification, fitting, OCR
  gui/         editor app
  mcp/         MCP server
  cli.py       phasor doctor | render | storyboard | lint | inspect | bench | mcp | gui
docs/ai/       API.md, PATTERNS.md, PITFALLS.md, SKILL.md
gallery/       frames and clips you love, with notes
examples/      tested example scenes per domain
benchmarks/
tests/
SPEC.md        this document
STYLE.md       your style rubric
CLAUDE.md      conventions for Claude Code
```

**Conventions for Claude Code (seed for `CLAUDE.md`)**

- Every new object type needs: semantics, bounds, inspector output, at least one lint rule if relevant, a doc entry and an example.
- Every feature ships with a storyboard snapshot test.
- Never add a second way to do something already in the API; extend the existing one.
- Keep kernels in `.cu` strings in `render/gpu/` with a NumPy reference implementation for tests.

## Worked example: 3D reverb scene

This is the acceptance test for the acoustics module and for 3D in general: a sound wave bouncing in a room, building up to how reverb software works.

**Story beats (each a narration macroblock)**

1. A glowing point source in a translucent 3D room emits a spherical wavefront.
2. The wavefront hits a wall; reflected fronts appear (image sources shown as ghost speakers behind the walls).
3. Rays from source to listener: direct path, first reflections, then a dense cloud; each arrival drops a spike on an impulse-response plot.
4. The impulse response splits visually into early reflections and late tail; RT60 decay curve is fitted and labeled.
5. Wall material slider (absorption) changes the tail live in interactive mode.
6. A dry audio clip is convolved with the impulse response; spectrograms of dry vs wet sit side by side.
7. Pull back to a reverb plugin's block diagram (pre-delay, early reflections, diffusion, decay, damping) mapped to what we just saw.

**Sketch of the scene code**

```python
from phasor import Scene, ac, fourier

s = Scene("reverb", theme="midnight_glow", dim=3)
s.narrate("reverb_script.txt")          # macroblocks + [[markers]]
room = s.add(ac.Room(size=(8, 5, 3), material="plaster", id="room"))
src  = s.add(ac.Source(pos=(2, 2.5, 1.5), id="src"))
mic  = s.add(ac.Listener(pos=(6, 2, 1.5), id="mic"))
wave = s.add(ac.Wavefront(src, room, max_order=3, id="wave"))
ir   = s.add(ac.ImpulseResponse(room, src, mic, id="ir"), panel="bottom")

s.at("emit").play(wave.expand, until="first_hit")
s.at("first_hit").show(ac.ImageSources(room, src, order=1, id="ghosts"))
s.at("rays").play(ac.Rays(src, mic, room, order=3).draw, stagger=0.02)
s.at("tail").show(ir.rt60_fit, label="RT60")
s.camera.orbit(around=room, during=("emit", "tail"))
s.interactive("room.absorption", range=(0.02, 0.9))
s.constrain("visible_in_3d(mic) and no_overlap(ir.labels)")
```

**Implementation notes**

- Wavefront: GPU FDTD on a 3D grid for the true pressure field (rendered as a glowing isosurface or slice), plus analytic image-source spheres for clean visuals.
- Rays: image-source method up to order 3–4, stochastic ray tracing beyond.
- Impulse response and convolution done in NumPy/CuPy; audio exported alongside the video.
- Lint checks: mic and source never occluded by walls (walls render translucent where they would block), IR plot labels readable.

## Roadmap and milestones

Build the AI-visible core first, then beauty, then breadth. Each milestone ends with a demo scene and passing tests.

| # | Milestone | Scope | Exit demo | Model |
| --- | --- | --- | --- | --- |
| 0 | Skeleton | Repo, `uv` env, CLI stub, `import-linter`, CI, `CLAUDE.md`, `phasor doctor` (CUDA, NVRTC, OpenGL 4.3, encoders) | `phasor doctor` passes on clean Windows | Strong |
| 1 | Core + inspector | Schema, Engine, patches, overrides file, pure timeline, object contract, 2D primitives, `color` + `id` + `solo` passes, `inspect`, preview stills | Moving labeled dot; `where` and `visible` correct at any t | Strong |
| 2 | Layout + linter | kiwisolver stage, label placer with hysteresis, layout bake, lint rules, storyboard with marks | A deliberately cluttered scene is flagged, then auto-fixed | Strong |
| 3 | GPU beauty | Kernel registry, `CopyBridge`, tiling, domain coloring, Mandelbrot/Julia, HDR, bloom, tone map, encode chain, first theme, style roles | Domain-colored 1/(z²+1) with a glowing contour at 4K | Strong for kernels and bridge, then Sonnet |
| 4 | Narration sync | Script markers, recording/TTS, WhisperX alignment, `TimeMap`, choreography verbs | Residue theorem short driven entirely by narration | Sonnet |
| 5 | MCP + AI docs | MCP server, detail levels, `docs/ai/`, self-review loop, skill package | An AI builds and fixes a scene from a one-line prompt | Sonnet |
| 6 | 3D + spatial awareness | Perspective camera, surfaces, OIT, event log, camera solver, 3D label placement, auto-cinematography | Riemann surface of sqrt(z): labels readable through a full orbit, event log clean | Strong for camera solver, then Sonnet |
| 7 | Beauty system | Aesthetic lint rules, gallery compare, art-director pass, variants, `choices.jsonl`, two more themes (dark stays the default), living motion (`life`), softened edges, varied palettes (rainbow only where hue carries meaning), look and object presets | Critic pass measurably improves a rough scene in 3 rounds | Sonnet |
| 8 | Domains wave 1 | `fourier`, `linalg`, `acoustics`, `Simulation` checkpoints | 3D reverb scene (section above) | Sonnet |
| 9 | Domains wave 2 | `pde`, `algebra`, `calc`, perturbation deep zoom | Heat equation, dihedral group and a 1e-30 zoom | Sonnet; strong for perturbation |
| 10 | GUI editor | Viewport, timeline, inspector, lint panel, MCP attach, variant picker | Edit a scene by hand and by AI in one session (built on the prism scene: the reverb scene was skipped) | Sonnet |
| 11 | Sketch | **Dropped, then reinstated by the owner (2026-11-19).** Stroke capture, classification, fitting, handwriting to LaTeX (read by an AI from a picture), paint-over | Draw a contour; it integrates and animates (built: `examples/sketch_contour.py`) | Sonnet |
| 12 | Interactive export | Desktop sliders, notebook widget, then web export | Julia explorer with a c slider (built; web export is baked pictures, not WebGPU) | Sonnet |
| 13 | Manim import | **Reinstated by the owner (2026-11-19).** Runs Manim CE scripts unchanged against a stand-in module and records the phasor scene they build | An old Manim reverb scene runs in phasor (built: `examples/manim_reverb.py`) | Sonnet |

## Build plan and kickoff prompt

Use the strongest model where mistakes are expensive, Sonnet where the spec is clear, and a strong-model review at every milestone boundary.

**Rules of the build**

- One milestone per branch. No work on milestone N+1 until N's exit demo and tests pass.
- Each milestone ends with: demo scene in `examples/`, golden frames, updated `docs/ai/`, and a short `MILESTONE_N.md` listing what was built and what was deferred.
- Review prompt at each boundary, run by the strong model: "Compare this branch against SPEC.md sections X and Y. List every deviation from the Core architecture contracts, then run the exit demo and report."
- Any change to the seven core contracts needs your approval and a spec edit in the same commit.

**Kickoff prompt (paste into Claude Code with `SPEC.md` at the repo root)**

```text
Read SPEC.md fully before writing anything.

You are building phasor, a Python math animation engine for Windows + CUDA.
The section "Core architecture contracts" is binding. If something there seems
wrong or impossible, stop and ask me; do not work around it.

Do Milestone 0 only:
1. Create the repo layout from "Tech stack, Windows, repo layout" with uv,
   pytest, ruff, import-linter (layer rules from contract 6) and CI.
2. Write CLAUDE.md from the conventions in SPEC.md plus the seven contracts
   in short form.
3. Implement `phasor doctor`: report Python, GPU model and VRAM, CUDA and
   NVRTC availability, compile and run one trivial CuPy RawKernel, create a
   standalone ModernGL 4.3 context, render one offscreen triangle, and probe
   encoders (PyAV nvenc, system ffmpeg, libx264). Print a hardware tier A/B/C.
4. Before using any library API you are not certain of (CuPy RawKernel,
   ModernGL integer textures, PyAV codecs, WhisperX, MCP SDK), check its
   current docs. List anything in SPEC.md that the docs contradict.
5. Stop. Show me the doctor output, the list of contradictions, and your
   plan for Milestone 1 as a file-by-file list with test names.

Rules for every milestone:
- Tests first for the timeline, patches, inspector and solvers.
- Every object type meets the full SceneObject contract.
- Every GPU kernel has a NumPy reference and a comparison test.
- One way to do each thing. Extend, never duplicate.
- Update docs/ai/ in the same commit as the API change.
- Never mark a milestone done with failing tests or a missing exit demo.
```

## Open questions

- [ ] Final name ("phasor" is a placeholder; check PyPI availability).
- [x] GUI toolkit: PySide6 (Milestone 10; testable headless, solid on Windows).
- [ ] Graphics backend long-term: stay on ModernGL or move to wgpu-py for WebGPU export.
- [ ] TTS provider default (local vs cloud), or your own voice only.
- [ ] Which GPU you have (VRAM sets FDTD grid sizes and tile sizes).
- [x] TeX backend: both, by choice: matplotlib mathtext built in (default, identical everywhere) and a real TeX (`latex` + `dvipng`, MiKTeX or TeX Live) via `PHASOR_MATH=latex|auto` (Milestone 13 session, 2026-11-19).
- [x] License: MIT (owner's lean, chosen 2026-11-19; `LICENSE`). Whether to publish it is still open.
