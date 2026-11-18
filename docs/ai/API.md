# phasor API (for AI agents)

One canonical example per public name. No deprecated aliases. Every ```python block in this
folder runs in CI (`tests/test_docs.py`), top to bottom in one namespace per file. Mark a
block ```python no-run only when it cannot run in CI, and say why next to it.

Status: Milestone 1 (core + inspector). Layout constraints and the visual linter arrive in
Milestone 2; GPU kernels, bloom and video in Milestone 3.

## Build a scene

```python
from phasor import Dot, Line, Scene, Text, expr

s = Scene("residues")  # 1920x1080, 60 fps, theme "midnight_glow"
s.add(Line(id="axis", points=[(-6, 0), (6, 0)], style={"color": "muted"}))
s.add(Dot(id="pole_i", pos=(0, 1), r=0.15, style={"color": "highlight"}, tags=["key"]))
s.add(Text(id="pole_i_label", text="i", pos=expr("pole_i.pos + vec(0.4, 0, 0)")))
s.set("camera.height", 6)  # world units visible top to bottom
```

- IDs match `[a-z][a-z0-9_]*`, with dots for children (`res.pole_i` needs `res` to exist).
  Name them by meaning (`pole_i`), never by counter (`dot3`).
- Points are world space: x = Re, y = Im, z up. `(x, y)` means z = 0.
- `style` takes role names only. Raw colors and sizes are rejected.
- `tags=["key"]` marks an object other objects are located relative to (see `where`).
- A wrong call raises `DocumentError` at that line and changes nothing.

### Object types

| Type | Parameters | Style keys (role group, default) |
| --- | --- | --- |
| `Dot` | `pos`, `r` (world radius, default 0.1) | `color` (emphasis, primary), `glow` (glow, none) |
| `Line` | `points` (2 or more) | `color` (emphasis, primary), `stroke` (stroke, regular), `glow` (glow, none) |
| `cx.DomainColoring` | `function` (e.g. `"1/(z**2 + 1)"`), `pos` (center), `size` (w, h), `contour` (0-1) | `colormap` (colormap, phase) |
| `cx.Fractal` | `kind` (`mandelbrot`/`julia`), `c` (Julia constant), `pos`, `size`, `max_iter`, `density` | `colormap` (colormap, sequential) |
| `Text` | `text`, `pos`, `anchor` (`center`/`left`/`right`), `leader`, `leader_on` (set by layout) | `color` (emphasis, primary), `size` (text, label), `leader` (stroke, hairline) |

Every type also has `opacity` (0-1; below 0.5 it stops hiding things behind it), `dim`
(0-1, owned by `dim_others`; what is drawn is `opacity` x `dim`) and
`z_index` (higher draws on top among objects at the same depth). Every object takes
`layer="overlay"` to live in screen space: units are frame heights, origin top-left, y down.

Roles: emphasis `primary secondary muted highlight danger`; stroke `hairline regular bold`;
text `title label annotation math`; glow `none soft strong`; colormap `phase sequential`.

`cx` objects are rectangles of the complex plane drawn by GPU kernels: world x is Re z and
world y is Im z, so the camera zooms into them like into anything else. They are
*backdrops*: labels may sit on them (layout does not avoid them; lint checks contrast).
Their semantics list the zeros and poles inside them (for rational functions).

```python
from phasor import Line, Scene, cx

s3 = Scene("poles")
s3.set("camera.height", 4.5)
s3.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8, 4.5)))
s3.add(
    Line(
        id="loop",
        points=[(0.4, 1), (0, 1.4), (-0.4, 1), (0, 0.6), (0.4, 1)],
        style={"color": "highlight", "glow": "strong"},
    )
)
plane = s3.engine.inspect(0, id="plane")["objects"][0]["semantic"]
assert sorted(map(tuple, plane["poles"])) == [(0.0, -1.0), (0.0, 1.0)]
```

`glow` makes a curve emit light: the renderer writes that light to a second target and the
final look blooms it. Plain paint (`glow: none`) never blooms, and colors below the theme's
`knee` (linear 0.9) come out exactly as the theme defines them.

### Expressions

`expr("...")` computes a parameter from other parameters (`<id>.<param>`) and time `t`.
Operators `+ - * / ** %`, indexing `dot.pos[0]`, constants `pi tau e`, functions
`sin cos tan exp log sqrt floor ceil abs min max clamp lerp vec length normalize dot midpoint`.
Scalars broadcast over vectors. A cycle is an error.

## Narrate: words drive time

```python
from phasor import Dot, Scene

s4 = Scene("narrated", passes=None)
s4.add(Dot(id="pole", pos=(0, 1), r=0.2))
s4.narrate(
    "We start with a function that has a pole [[pole_appears]].\n"
    "[[pause 0.5]]\n"
    "Watch it move [[pole_moves]] to the right.",
    voice="estimate",  # or "auto", "espeak[:lang[:wpm]]", "piper:<model.onnx>"
)
s4.at("pole_appears").bloom_in("pole")
s4.at("pole_moves").settle("pole.pos", (2, 1))
s4.pace()  # breathing room after sentences that change a lot; run last
assert s4.engine.lint(strict=True) == []
```

- One sentence is one block; `[[name]]` marks the moment the next word starts (at the end
  of a sentence: when it ends). `[[pause 1.5]]` is a silent block.
- `narrate` writes the markers. Anchor motion to them with `s.at("name")`; never type seconds.
  Re-narrate (another voice, an edited sentence) and everything moves with the words.
- Voices: `auto` = espeak-ng if installed, else `estimate` (timing from the text, no audio).
  Synthesized voices are spoken piece by piece between markers, so markers are exact; clips
  are cached in `<title>.narration/` beside the script. Your own recordings:
  `phasor.voice.from_recordings(script, wavs, aligner, base_dir, theme)` with
  `WhisperX()` or `TimingsFile()`.
- `phasor render` muxes the narration into the video.

## Animate with verbs

| Verb | Does |
| --- | --- |
| `reveal(*ids)` | fade in; hidden until then; drops earlier dimming (it is the new focus) |
| `exit(*ids)` | fade out |
| `trace(*ids)` | draw a `Line` from start to end (`progress` 0 to 1); nothing drawn before |
| `bloom_in(*ids)` | fade in and grow from a point with a small overshoot (objects with `r`) |
| `dim_others(*keep)` | dim everything else to the theme's `dim_level` (backdrops stay) |
| `undim()` | bring dimmed objects back |
| `focus_pull(id, margin=0.15)` | move the camera onto an object |
| `morph(path, value)` | change any parameter smoothly |
| `settle(path, value)` | move like a released spring (overshoot, then rest) |

Durations, easing and stagger come from the theme's `[motion]` table. Chain with `then()`:
`s.at("loop_draw").trace("loop").then().reveal("loop.label")`. Several ids stagger.
`orbit_reveal` arrives with 3D (Milestone 6).

Dimming uses each object's `dim` factor (drawn opacity = `opacity` x `dim`), so dimming and
revealing never fight, whatever order you write them in.

## Animate by hand

```python
s.marker("appear", 1.0)  # a hand-placed marker, in seconds
s.at("appear").to("pole_i.r", 0.3, duration=0.5)  # ease "smooth" by default
s.at("appear", offset=1.0).to("pole_i.pos", (2, 1), duration=2.0, ease="in_out")
s.at(4.0).set("pole_i_label.text", "moved")  # instant change at 4 s
```

Raw `to()` is allowed but lints as `raw_motion` with `engine.lint(strict=True)`: prefer verbs.
Any parameter is animatable by its path, cameras included (`camera.center`, `camera.height`).
Times anchored to a marker move with it. Segments on one parameter must not overlap.
A parameter set by `expr(...)` cannot also have a track; animate what it reads.
Easings: `linear smooth smoother in out in_out`, and `back spring` (these overshoot: not for
opacity). Custom easing functions are not supported: a document must mean the same
everywhere.

## Engine: state, patches, undo

```python
engine = s.engine
state = engine.state_at(3.0)  # pure function of (document, t); immutable
assert state["pole_i"].params.pos == (1.0, 1.0, 0.0)
assert state["pole_i_label"].params.pos == (1.4, 1.0, 0.0)  # the label followed

result = engine.apply({"op": "set_param", "path": "pole_i.r", "value": 0.2})
assert result.ok and engine.version == result.version

bad = engine.apply({"op": "set_param", "path": "pole_i.r", "value": -1})
assert not bad.ok and bad.errors[0].code == "bad_param"  # nothing changed

engine.undo()
engine.redo()
```

`Engine.apply(patch)` is the only way to change a scene. It takes one op, a list of ops, or
`{"ops": [...]}`; applies all or nothing; returns `PatchResult(ok, version, errors,
lint_added, lint_resolved)`. Each error has `code`, `message`, `ids`, `path`, `fix_hint`.

| Op | Fields |
| --- | --- |
| `add_object` | `object: {id, type, params, style, tags, layer}` |
| `remove_object` | `id` (its tracks go too; fails if another object's expression reads it) |
| `set_param` | `path`, `value` (constant or `{"expr": "..."}`) |
| `set_style` | `id`, `key`, `role` |
| `set_tags` | `id`, `tags` |
| `add_segment` | `path`, `segment: {t0, t1, to, ease}`; a time is seconds or `{"marker", "offset"}` |
| `clear_track` | `path` |
| `set_marker` / `remove_marker` | `name`, `t` |
| `set_meta` | `field`, `value` |
| `add_constraint` | `rule`, `priority` (`required`/`strong`/`medium`/`weak`, default `strong`) |
| `set_narration` | `narration` (from `phasor.voice.narrate`) or `null`; replaces the markers it owns |
| `remove_constraint` | `rule` |

On a scene built by a script, `set_param`, `set_style` and `add_constraint` are stored as
tweaks in `name.overrides.json` and survive re-running the script; other ops are refused
with `scripted_scene` (change the script instead).

`lint_added` and `lint_resolved` are worked out when you first read them (or call
`to_dict()`), so building a scene stays fast.

## Layout constraints: say where things go, not coordinates

```python
from phasor import Dot, Scene, Text

s2 = Scene("layout")
s2.add(Dot(id="pole", pos=(-2, 0), r=0.2, tags=["key"]))
s2.add(Text(id="pole.label", text="pole"))  # the label of X is the child object X.label
s2.add(Text(id="title", text="Residues", pos=(0, 3), style={"size": "title"}))
s2.constrain("label(pole) near pole")  # the placer picks a free spot and keeps it there
s2.constrain("on_screen(title)")
s2.at(0).to("pole.pos", (2, 1), duration=2)

assert s2.engine.layout.slots["pole.label"][0] == "n"  # sits above, and follows the pole
assert s2.engine.lint() == []
```

A rule is `SUBJECT RELATION TARGET [gap N]`, or a guard `NAME(...)`; join several with `and`.
The subject is what moves; it needs a `pos` parameter that is not an expression.

| Rule | Meaning |
| --- | --- |
| `a above b`, `below`, `left_of`, `right_of` (`gap N` optional) | touch that side, centered on it |
| `a near b` (`gap N` optional) | label placer: 8 slots around `b` plus 8 leader-line slots further out |
| `a align_x b`, `a align_y b` | same x (stacked) or same y (side by side) |
| `a inside b` | `a`'s box within `b`'s box |
| `on_screen(a)` or `on_screen(a, margin)` | keep `a` inside the frame (moves it when it can) |
| `no_overlap(a, b, ...)`, `min_size_px(a, n)`, `readable(a)` | checked by the linter |

- `label(x)` means the object `x.label`. Numbers are in the object's layer units.
- Layout is solved at every marker, every transition end and 10 times a second, then baked.
  `engine.state_at(t)`, the inspector and the renderer all read the same bake.
- A label changes slot only when another slot is over 20% better, and then it jumps.
- `engine.layout` shows the bake: `times`, `slots` (per label, per key time), `issues`.
- A rule that cannot hold is a lint error `unsolved`, with the times. Nothing is dropped.
- `outside`, `tangent_to`, `attach_to`, `visible_in_3d` arrive with 3D (Milestone 6).

## Lint: find problems before rendering

```python
s2.add(Text(id="note", text="note", pos=(-2, 0.1)))  # dropped right on top of the pole
issues = s2.engine.lint()
found = {(i.code, i.ids) for i in issues}
assert ("overlap", ("note", "pole")) in found  # (with OpenGL, also "contrast": it sits on the dot)
assert issues[0].severity == "warning" and issues[0].times and issues[0].fix_hint
assert s2.engine.apply({"op": "set_param", "path": "note.pos", "value": [-4, -2]}).ok
assert s2.engine.lint() == []
```

`engine.lint()` checks every marker, every transition end and every 0.5 s (`step=`).
Each issue has `code`, `severity`, `message`, `ids`, `times`, `fix_hint`, and `fix` (patch ops).

| Code | Severity | Means |
| --- | --- | --- |
| `overlap` | warning | a label, or an object tagged `key`, overlaps something |
| `offscreen` | warning | a label or key object is cut by the frame |
| `contrast` | warning | text too close in brightness to what is behind it (with OpenGL: measured on the real picture) |
| `missing_glyph` | warning | text uses characters the font cannot draw (`²`, `π`, accents): they show as boxes |
| `tiny` | warning | an object or text too small to read |
| `orphan` | warning | an object that never appears |
| `clutter` | warning | too many objects change at once in one block (between two markers) |
| `unsolved` | error | a constraint or guard does not hold |
| `raw_motion` | warning, strict only | a transition written with raw `to()` instead of a verb |
| `orphan_override` | error | a tweak whose object is gone |

`engine.autofix()` applies every `fix` the linter offers (add `near` for a child label that
overlaps, `on_screen` for something cut by the frame), lints again, and returns
`{"applied": [...], "remaining": [...]}`. Each round is one undo step. Thresholds live in
the theme's `[lint]` table.

## Storyboards: look at the whole scene at once

```python no-run
# no-run: needs OpenGL 4.3
board = engine.storyboard(marks=True)  # start, every marker, end; or pass a list of times
save_png("out/board.png", board["image"])  # one contact sheet, captions show time and marker
board["frames"]  # [{"t", "marker", "objects": brief inspector lines}]
```

`marks=True` draws each object's box and id on the stills, so a vision model can tie what it
sees to object ids.

## Inspect: see what is on screen

```python
snap = engine.inspect(3.0, detail="brief")  # one line per object, at most 40
print(*snap["objects"], sep="\n")

where = engine.where("pole_i", 3.0)  # region, bbox, center, depth, relations
assert where["center"]["px"] == [1140.0, 360.0] and where["center"]["norm"] == [0.594, 0.333]

vis = engine.visible("pole_i", 3.0)  # visible_fraction, on_screen_fraction
assert vis["on_screen_fraction"] == 1.0

engine.occluders("pole_i", 3.0)  # {"hidden_by": [{"id", "share"}, ...]}
assert engine.hit_test(1140, 360, 3.0) == ["pole_i"]  # ids under an output pixel, topmost first
engine.inspect(3.0, id="pole_i", detail="full")  # adds projected feature points
```

- `detail`: `brief`, `standard` (default) or `full`. Start with `brief`.
- `visible_fraction`: share of the object's on-screen silhouette that nothing hides.
- `on_screen_fraction`: share of the whole silhouette that is inside the frame.
- `hidden_by`: who covers it, with shares.
- `relations`: `left_of`, `above`, `in_front_of`, `overlaps` against objects tagged `key`,
  all as seen by the camera.
- `"occlusion": "pixel"` means measured from the id and solo render passes. `"analytic"`
  means no OpenGL here: geometry is still exact, `visible_fraction` and `hidden_by` are `null`.
- `note` is set when an object is too thin to measure at inspector resolution (480x270).

## Stills and files

```python no-run
# no-run: needs OpenGL 4.3 and writes files
from phasor import load
from phasor.render.output import save_png

save_png("out/frame.png", engine.still(3.0, (960, 540)))  # (h, w, 3) uint8 sRGB
picture = engine.frame(3.0, (3840, 2160))  # linear, after bloom and tone map; float32
s.save("residues.phasor.json")  # canonical, diff-friendly
engine = load("residues.phasor.json")  # or load("residues.py")
```

## GPU kernels: per-pixel math

```python
from phasor.render import gpu

f = gpu.ComplexFunction.parse("1/(z**2 + 1)")  # one formula: kernel code, NumPy and LaTeX
assert f.latex() == r"\frac{1}{z^{2} + 1}"

picture = gpu.render("domain", 160, 90, function=f, center=(0, 0), height=4.5)
assert picture.shape == (90, 160, 4)  # linear RGBA float32, row 0 = top

field = gpu.render("escape", 160, 90, center=(-0.5, 0), height=2.5, max_iter=100)
assert field.shape == (90, 160) and (field == -1).any()  # -1 marks points in the set
```

| Kernel | Parameters | Returns |
| --- | --- | --- |
| `domain` | `function`, `center`, `height`, `contour=0.35`, `lut=None`, `precision="f32"` | RGBA: hue = phase, bands = each doubling of the modulus, poles white |
| `escape` | `center`, `height`, `max_iter=256`, `julia=None`, `precision=None` | smooth escape count; Mandelbrot, or the Julia set of `julia=(re, im)` |
| `tonemap` | `image`, `exposure=1.0` | ACES filmic curve, linear in and out (whole image) |
| `gauss` | `image`, `sigma`, `axis` | one pass of a Gaussian blur along x (0) or y (1) (whole image) |

- `center` and `height` describe the view like the 2D camera: the point at the middle of
  the frame and how many units the frame is tall.
- `gpu.render(name, width, height_px, supersample=1, backend="auto", **params)` runs tile by
  tile within the VRAM budget and returns a NumPy array. `gpu.run(name, tile, **params)`
  runs one tile and returns a CuPy array on CUDA.
- With no CUDA the NumPy reference runs instead: same result, slower. `backend="cuda"`
  demands the GPU; `backend="reference"` forces NumPy.
- `escape` picks float64 by itself once `height` is under 1e-5.
- Formulas may use `z`, `i`, `pi`, `e`, numbers, `+ - * / **` and
  `exp log sqrt sin cos tan sinh cosh tanh conj abs`. Anything else is a `FunctionError`.
- A new kernel needs `@kernel(name, signature, reference=numpy_fn)`; registration fails
  without a NumPy reference that takes the same parameters.

## Command line

| Command | Status | Purpose |
| --- | --- | --- |
| `phasor doctor [--json] [--require A\|B\|C]` | M0 | Prove CUDA, NVRTC, OpenGL 4.3, encoders; print tier |
| `phasor inspect SCENE [--t T] [--id ID] [--detail brief\|standard\|full]` | M1 | What is on screen at time t |
| `phasor lint SCENE [--fix] [--strict] [--json] [--step S]` | M2 | Lint; exit 1 on errors (`--strict`: on warnings too). `--fix` applies suggested fixes and saves them |
| `phasor storyboard SCENE [-o PNG] [--t T]... [--marks] [--json]` | M2 | Keyframe contact sheet plus what the inspector sees |
| `phasor render SCENE [-o OUT] [--preset preview\|final\|4k] [--fps F] [--start S] [--end E] [--frame T]` | M3 | Video (`.mp4`; `.mov` is ProRes; a folder or `%05d.png` is a PNG sequence) or one PNG with `--frame` |
| `phasor bench [--width W] [--height H]` | M3 | ms per frame per kernel and peak VRAM; saved in `benchmarks/` |
| `phasor mcp` | M5 | MCP server |
| `phasor gui` | M10 | Editor |

`SCENE` is a `.phasor.json` document or a `.py` script that defines exactly one `Scene`.
Unimplemented commands exit with code 2 and name their milestone.

## Hardware check

```python
from phasor.render.doctor import run_doctor

report = run_doctor()  # never raises; each probe fails on its own
assert report.tier in {"A", "B", "C"}
```

Tiers: **A** CUDA kernels and GL 4.3. **B** GL 4.3 only. **C** no usable GL: the inspector
runs in analytic mode, stills are unavailable.
