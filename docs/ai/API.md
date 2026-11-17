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
| `Dot` | `pos`, `r` (world radius, default 0.1) | `color` (emphasis, primary) |
| `Line` | `points` (2 or more) | `color` (emphasis, primary), `stroke` (stroke, regular) |
| `Text` | `text`, `pos`, `anchor` (`center`/`left`/`right`) | `color` (emphasis, primary), `size` (text, label) |

Every type also has `opacity` (0-1; below 0.5 it stops hiding things behind it) and
`z_index` (higher draws on top among objects at the same depth). Every object takes
`layer="overlay"` to live in screen space: units are frame heights, origin top-left, y down.

Roles: emphasis `primary secondary muted highlight danger`; stroke `hairline regular bold`;
text `title label annotation math`.

### Expressions

`expr("...")` computes a parameter from other parameters (`<id>.<param>`) and time `t`.
Operators `+ - * / ** %`, indexing `dot.pos[0]`, constants `pi tau e`, functions
`sin cos tan exp log sqrt floor ceil abs min max clamp lerp vec length normalize dot midpoint`.
Scalars broadcast over vectors. A cycle is an error.

## Animate

```python
s.marker("appear", 1.0)  # narration marker, in seconds
s.at("appear").to("pole_i.r", 0.3, duration=0.5)  # ease "smooth" by default
s.at("appear", offset=1.0).to("pole_i.pos", (2, 1), duration=2.0, ease="in_out")
s.at(4.0).set("pole_i_label.text", "moved")  # instant change at 4 s
```

Any parameter is animatable by its path, cameras included (`camera.center`, `camera.height`).
Times anchored to a marker move with it. Segments on one parameter must not overlap.
A parameter set by `expr(...)` cannot also have a track; animate what it reads.
Easings: `linear smooth smoother in out in_out`.

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

On a scene built by a script, `set_param` and `set_style` are stored as tweaks in
`name.overrides.json` and survive re-running the script; other ops are refused with
`scripted_scene` (change the script instead).

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
s.save("residues.phasor.json")  # canonical, diff-friendly
engine = load("residues.phasor.json")  # or load("residues.py")
```

## Command line

| Command | Status | Purpose |
| --- | --- | --- |
| `phasor doctor [--json] [--require A\|B\|C]` | M0 | Prove CUDA, NVRTC, OpenGL 4.3, encoders; print tier |
| `phasor inspect SCENE [--t T] [--id ID] [--detail brief\|standard\|full]` | M1 | What is on screen at time t |
| `phasor lint` / `phasor storyboard` | M2 | Linter and keyframe contact sheet |
| `phasor render` / `phasor bench` | M3 | Final render, kernel benchmarks |
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
