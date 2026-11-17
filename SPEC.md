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
- A Python script is a builder: running `scene.py` emits the base document. GUI and AI tweaks to a scripted scene are saved in `name.overrides.json`, keyed by object id + parameter path, and applied on top at load. Re-running the script never destroys tweaks. An override whose target id vanished is a lint error (`orphan_override`).
- IDs match `[a-z][a-z0-9_]*`, use dots for children (`res.pole_i`), and are never reused or renumbered. Auto IDs come from type + meaning (`pole_i`), not counters.

**3. Time is a pure function**

- `Engine.state_at(t) -> SceneState` depends only on (document, t). `SceneState` is immutable (frozen dataclasses) and holds every object's resolved parameters.
- Parameters form a DAG of constants, tracks (piecewise segments with easing) and expressions over other parameters. Evaluate in topological order; a cycle is a validation error.
- Narration lives in a separate `TimeMap` (marker name to seconds). Tracks store marker-relative anchors. Re-recording audio rebuilds only the `TimeMap`.
- Simulations (PDE, FDTD, particles) are `Simulation` objects with `step(state, dt)`, a fixed dt, a seed, and checkpoints every N steps cached in `.phasor_cache/` under a hash of their parameters. `state_at(t)` = nearest earlier checkpoint + remaining steps.
- Randomness only through `rng(object_id, seed)`. Same inputs must give the same frames on the same machine.

**4. One spatial pipeline**

- Everything is 3D. A 2D scene is a 3D scene with an orthographic camera looking down −z at the plane z = 0. There is no separate 2D code path.
- World space: right-handed, z up, scene units. The complex plane maps x = Re, y = Im.
- Screen space: pixels with origin top-left, plus normalized 0–1 coordinates in every inspector output. Titles and HUD panels live in a screen-space overlay layer, with normal objects and IDs.
- Color: linear RGBA16F from kernels through post-processing. Tone mapping and sRGB encoding happen exactly once, at the end.

**5. The object contract**

Every object type implements the `SceneObject` protocol. A type missing any member fails registration.

| Member | Returns | Used by |
| --- | --- | --- |
| `Params` | pydantic model of its parameters | validation, GUI panels, MCP patches |
| `world_bounds(state)` | AABB + bounding sphere | camera solver, culling |
| `features(state)` | named points/curves in world space (`pole_i`, `tip`, `front`) | inspector, labels, resolver |
| `semantics(state)` | JSON-serializable math facts | inspector, resolver |
| `draw(ctx, state, pass_)` | draw calls for pass `color`, `id` or `solo` | both renderers |

**6. Layer rules**

- Allowed import direction: `core` ← `time` ← `layout` ← `inspect` ← `render` ← `domains` ← (`gui`, `mcp`, `cli`).
- `core`, `time` and `layout` import no GPU library, so they test on any machine.
- Enforce with `import-linter` in CI from Milestone 0.

**7. Layout is baked, then rendered**

- The solver runs at every marker plus sampled times and writes positions into tracks (the "layout bake"). Renderers and the inspector both read the bake, so what the AI inspects is exactly what renders.
- Stage 1: `kiwisolver` for linear relations (align, above, gap).
- Stage 2: label placer. Candidates = 8 compass slots around the anchor plus leader-line slots. Cost = overlap area + occlusion (from the ID buffer) + distance from anchor + movement since the previous key time.
- Hysteresis: a label changes slot only when cost improves by more than 20%, so labels never jitter.
- Constraints the solver cannot satisfy are returned as lint issues, never dropped silently.

## Scene document and semantic scene graph

A scene is one document that Python, the AI and the GUI all edit. Python scripts build it; the canonical saved form is `name.phasor.json` plus an optional `name.overrides.json` for tweaks, as fixed in "Core architecture contracts".

**Scene document contents**

- `meta`: title, resolution, fps, theme, color space.
- `objects`: each with `id`, `type`, parameters, style, parent (for grouping), tags.
- `constraints`: layout rules (see Layout section).
- `timeline`: state tracks, transitions and narration markers.
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
- Alignment: WhisperX (or similar forced alignment) gives word timestamps; `[[markers]]` resolve to times.
- Re-recording re-aligns everything; nothing is hard-timed.

**State tracks and transitions**

- Every animatable parameter is a named state variable: `loop.r`, `camera.azimuth`, `julia.c`.
- A transition sets a target value over a span: `at("loop_draw").to("loop.progress", 1, ease="smooth")`.
- Evaluation is pure: `scene.state_at(t)` computes everything from the tracks. No hidden mutation.
- Spans can be anchored to markers, to other spans (`after("loop_draw")`), or to absolute times.
- Built-in easing library plus physical springs; custom easing as a Python function.

**Pacing helpers**

- Auto-pause after an idea, sized by how much changed on screen.
- Lint warning when too many things change in one block.

## Layout constraints, inspector and linter

This is the part that lets an AI work with its eyes open: it states intent as constraints, then reads back exactly what is on screen.

**Constraints**

- Relations: `above`, `below`, `left_of`, `right_of`, `near`, `align`, `inside`, `outside`, `tangent_to`, `attach_to(anchor)`.
- Guards: `no_overlap(a, b, ...)`, `on_screen(a, margin)`, `visible_in_3d(a)` (not occluded), `min_size_px(a, n)`, `readable(a)` (contrast vs background).
- Solved per frame (or per span) with a small solver: priorities, soft vs hard, and a report of any constraint it could not satisfy.
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
| orphan | Objects that never leave the scene or never appear |
| z\_fight | Coplanar surfaces flickering |
| clutter | Too many simultaneous changes in one block |
| tiny | Objects smaller than N px |
| unsolved | Constraints the solver could not satisfy |

- `phasor lint scene.py` runs it at sampled times plus every marker; output is JSON and human-readable.
- Strict mode fails CI (and AI self-review) on errors.

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
- `visible_fraction` = pixels in `id` pass ÷ pixels in `solo` pass.
- `hidden_by` = IDs found in the `id` pass where the solo mask is set, with the share each one covers.
- `on_screen_fraction` = solo pixels inside the frame ÷ analytic projected area.
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

- Linear HDR pipeline, tone mapping (ACES / AgX), wide-gamut option.
- Physically based bloom and glow with per-object emission strength.
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
- SymPy expressions become kernel code through `sympy.ccode` plus a small complex-math header (`render/gpu/complex.cuh`). One expression feeds the kernel, the NumPy reference and the LaTeX label.

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
| Fractal zoom past about 1e-13 scale | Perturbation: one reference orbit in arbitrary precision on CPU (`gmpy2`), per-pixel deltas in float64 on GPU, glitch detection and rebasing |
| PDE and FDTD fields | float32, with a float64 test mode |

- FDTD stability: enforce dt ≤ dx / (c·√3) in 3D and raise a validation error otherwise.

**Encoding**

- `render/encode.py` probes at startup, in order: PyAV with `hevc_nvenc`/`h264_nvenc`, a system `ffmpeg.exe` fed raw frames over a pipe, then PyAV `libx264`. A probe must encode real frames: PyAV wheels list NVENC codecs even on machines without an NVIDIA GPU.
- Audio is muxed in the same step from the narration blocks.

**Hardware tiers**

| Tier | Hardware | Behavior |
| --- | --- | --- |
| A | NVIDIA with CUDA 12 | Everything |
| B | Any GPU with OpenGL 4.3, no CUDA | Kernels run on NumPy references at preview resolution; final renders allowed but warned as slow |
| C | No usable GPU | Inspector, lint and timeline still work; no rendering |

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

**Choreography verbs (`time/verbs.py`)**

- The AI animates with verbs, not raw tweens: `reveal`, `trace`, `focus_pull`, `bloom_in`, `settle`, `morph`, `orbit_reveal`, `dim_others`, `exit`.
- Each verb expands into state transitions with tuned duration, easing, stagger and overshoot, read from the theme.
- Raw `to()` transitions stay available but lint as `raw_motion` in strict mode.

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

## Interactivity and outputs

One scene exports to several outputs without rewriting it.

- **Video**: MP4/H.264/H.265 (NVENC), ProRes, PNG/EXR sequences, GIF/WebM loops; vertical presets for shorts.
- **Stills**: high-res PNG/EXR posters, SVG for 2D vector content.
- **Interactive desktop player**: the live window with sliders bound to state variables (e.g. `julia.c`, `room.absorption`), drag handles on objects (move a contour, a source, a wall), hover readouts (f(z), |f|, arg f, winding number, pressure).
- **Interactive web export** (later milestone): WebGPU/WebGL build of the scene with the same sliders; GPU kernels transpiled to WGSL where possible, prebaked textures otherwise.
- **Notebook widget**: inline player in Jupyter for exploration.
- Parameters marked `interactive=True` become sliders automatically; constraints and linter keep running live.

## Sketch-driven authoring

Sketches become editable scene objects, not pixels. This works because the scene graph knows what sits under every stroke.

- **Sketch to scene**: draw on a tablet canvas in the GUI, or import a photo of paper. Strokes are classified (circle, line, arrow, dot, curve, handwriting). A vision model plus geometric fitting proposes objects: circle → contour, dot → pole/point, arrow → vector, wavy line → wave, handwriting → LaTeX (via handwriting-to-LaTeX OCR). You confirm or fix each proposal.
- **Freehand paths as math**: draw any closed curve over a plot; it is smoothed to a spline and becomes a real contour (integral computed and animated along it), integration region, boundary condition, ray path, or camera path.
- **Keyframe by drawing**: sketch frame A and frame B; objects are matched by ID/semantics and the engine creates the transitions.
- **Paint-over corrections**: scribble on a preview frame (circle a region, arrow a direction, write "bigger"). `hit_test` turns marks into object IDs, so the AI gets "move `label_res` up 0.3, it overlaps `loop`" instead of a pixel guess.
- **3D camera sketching**: draw a path on a top-down or side view; it becomes a smooth eased camera track that keeps targets framed.
- **Style from an image**: give a reference image; extract a palette, glow strength and background into a reusable theme.
- **Storyboard import**: a sequence of rough sketches + narration lines becomes a first-draft scene with markers.

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
| 7 | Beauty system | Aesthetic lint rules, gallery compare, art-director pass, variants, `choices.jsonl`, two more themes | Critic pass measurably improves a rough scene in 3 rounds | Sonnet |
| 8 | Domains wave 1 | `fourier`, `linalg`, `acoustics`, `Simulation` checkpoints | 3D reverb scene (section above) | Sonnet |
| 9 | Domains wave 2 | `pde`, `algebra`, `calc`, perturbation deep zoom | Heat equation, dihedral group and a 1e-30 zoom | Sonnet; strong for perturbation |
| 10 | GUI editor | Viewport, timeline, inspector, lint panel, MCP attach, variant picker | Edit the reverb scene by hand and by AI in one session | Sonnet |
| 11 | Sketch | Stroke capture, classification, fitting, handwriting to LaTeX, paint-over | Draw a contour; it integrates and animates | Sonnet |
| 12 | Interactive export | Desktop sliders, notebook widget, then web export | Julia explorer with a c slider | Sonnet |
| 13 | Manim import | Converter for common Manim CE scenes | An old Manim reverb scene runs in phasor | Sonnet |

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
- [ ] GUI toolkit: Dear PyGui vs PySide6 (decide at Milestone 10).
- [ ] Graphics backend long-term: stay on ModernGL or move to wgpu-py for WebGPU export.
- [ ] TTS provider default (local vs cloud), or your own voice only.
- [ ] Which GPU you have (VRAM sets FDTD grid sizes and tile sizes).
- [ ] TeX backend: MiKTeX + dvisvgm vs MathJax to SVG.
- [ ] License (MIT/Apache) and whether to publish it.
