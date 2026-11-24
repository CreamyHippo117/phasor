# phasor API (for AI agents)

One canonical example per public name. No deprecated aliases. Every ```python block in this
folder runs in CI (`tests/test_docs.py`), top to bottom in one namespace per file. Mark a
block ```python no-run only when it cannot run in CI, and say why next to it.

Status: through Milestone 9. This file is the Python API; to drive phasor from an AI client
(MCP) read `MCP.md` first.

## Build a scene

```python
from phasor import Dot, Line, Scene, Text, expr

s = Scene("residues")  # 1920x1080, 60 fps, the default look "prism" (look="none": plain)
s.add(Line(id="axis", points=[(-6, 0), (6, 0)], style={"color": "muted"}))
s.add(Dot(id="pole_i", pos=(0, 1), r=0.15, style={"color": "highlight"}, tags=["key"]))
s.add(Text(id="pole_i_label", text="i", pos=expr("pole_i.pos + vec(0.4, 0, 0)")))
s.set("camera.height", 6)  # world units visible top to bottom
```

- IDs match `[a-z][a-z0-9_]*`, with dots for children (`res.pole_i` needs `res` to exist).
  Name them by meaning (`pole_i`), never by counter (`dot3`).
- Points are world space: x = Re, y = Im, z up. `(x, y)` means z = 0.
- `style` takes role names (the default, and what an AI should write). A color, text size
  or stroke key may instead hold an explicit value (`"#58C4DD"`, a size in world units):
  see "Explicit colors and sizes" below.
- `tags=["key"]` marks an object other objects are located relative to (see `where`).
- A wrong call raises `DocumentError` at that line and changes nothing.

### Object types

| Type | Parameters | Style keys (role group, default) |
| --- | --- | --- |
| `Dot` | `pos`, `r` (world radius, default 0.1) | `color` (emphasis, primary), `glow` (glow, none) |
| `Line` | `points` (2 or more), `progress`, `hue_along` (colormap cycles along the length; 0: one color), `hue` (shifts them) | `color` (emphasis, primary), `stroke` (stroke, regular), `glow` (glow, none), `colormap` (colormap, spectrum) |
| `cx.DomainColoring` | `function` (e.g. `"1/(z**2 + 1)"`; it may use the constant `a`), `a` ((re, im), the value of `a`: animate it with `.to("id.a", (re, im))` and the zeros and poles move, nothing recompiles), `pos` (center), `size` (w, h), `contour` (0-1), `softness` (0-1, edge feathering), `hue` (turns the wheel); the prism look: `radius` (> 0 turns it on: zeros and poles glow this far), `ambient` (0-1, light elsewhere), `lines` (glowing rays of phase per turn, plus a ring each time \|f\| doubles) | `colormap` (colormap, phase), `glow` (glow, none; used by the prism look) |
| `cx.Fractal` | `kind` (`mandelbrot`/`julia`), `c` (Julia constant), `pos`, `size`, `max_iter`, `density`, `hue`, `center` (exact text `"re im"`), `zoom` (powers of ten, to 280), `steady` (hold the outer color in a dive); the prism look: `radius` (> 0 turns it on: the edge of the set glows this far, in world units of the rectangle), `ambient` (0-1, light far from the edge) | `colormap` (colormap, sequential), `glow` (glow, none; used by the prism look) |
| `cx.RiemannSurface` | `kind` (`sqrt`/`cbrt`/`log`), `pos` (branch point), `radius`, `height` (`re`/`im`), `scale`, `rings`, `segments` | `colormap` (colormap, phase), `material` (surface, solid) |
| `cloud.PointCloud` | `source` (`roots`/`attractor`/`buddhabrot`/`points`), `degree` and `coefficients` (roots), `kind` (`clifford`/`dejong`/`lorenz`) and `constants` (attractor), `max_iter` (buddhabrot), `points` and `values` (given directly), `count` (at most), `seed`, `pos` (centre and origin of the points), `size` (the rectangle it is seen in), `scale`, `point_size` (splat radius, px), `exposure`, `hue`, `color_by` (`source`/`phase`), `twist` (roots: turns of the last coefficient, animatable), `twinkle` (0-1, life), `white` (0-1, white-hot dense parts) | `colormap` (colormap, spectrum), `glow` (glow, none) |

| `Value` | `value` (a named number to animate; nothing is drawn: see "Your own types") | none |
| `Sound` | `path` (a WAV file in the scene's folder), `start` (seconds into the scene), `gain` (dB, -60 to 24); nothing is drawn: `phasor render` mixes it in (see "Narrate") | none |
| `Image` | `path` (a PNG/JPEG/WebP file, relative to the scene's folder; a GIF shows its first frame; `frames/shot_%04d.png` is a picture sequence played at `fps`, default 24), `pos` (centre), `width`, `height` (0: keep the picture's shape), `frame` (an outline of light), `fit`, `fill`, `backdrop`, and for a sequence `start`, `trim`, `speed`, `loop` (as `Video`; see "Footage and picture sequences") | `glow` (glow, none: bright parts shine), `frame` (emphasis, muted), `stroke` (stroke, hairline). Presets `plain`, `framed`, `luminous`. A missing file is issue `missing_file`, an unreadable one `bad_file`; transparent parts show the background; an AI may only name files inside its folder |
| `Video` | `path` (an `.mp4`, `.mov`, `.m4v`, `.webm` or `.mkv` file in the scene's folder; needs the `encode` extra), `start` (scene seconds where it begins; before it, its first frame), `trim` ((in, out) seconds inside the clip; out 0: its end), `speed` (0-64), `loop` (off: it holds its last frame, lint `clip_ended`), `pos`, `width`, `height` (0: the clip's shape), `fit` (`stretch`, `contain`, `cover`, `none`), `fill` (fill the camera's view: a full-frame background, 2D camera), `backdrop` (labels may sit on it), `frame` | as `Image`. Nothing is heard (use `Sound`). Missing: `missing_file`; not a video: `bad_file`; past a cap: `video_too_big`; without PyAV: `missing_extra`; each draws nothing |
| `Sound` | `path` (a sound file in the scene's folder: WAV, or an mp4's sound, m4a, mp3, ogg, flac with the `encode` extra), `start` (seconds into the scene), `gain` (dB, -60 to 24); nothing is drawn: `phasor render` mixes it in (see "Narrate") | none |
| `Image` | `path` (a PNG/JPEG/WebP file, relative to the scene's folder), `pos` (centre), `width`, `height` (0: keep the picture's shape), `frame` (an outline of light) | `glow` (glow, none: bright parts shine), `frame` (emphasis, muted), `stroke` (stroke, hairline). Presets `plain`, `framed`, `luminous`. A missing file is issue `missing_file`; transparent parts show the background; an AI may only name files inside its folder |
| `Text` | `text` (`$...$` is typeset math), `pos`, `anchor` (`center`/`left`/`right`), `leader`, `leader_on` (set by layout) | `color` (emphasis, primary), `size` (text, label), `leader` (stroke, hairline) |
| `Equation` | `parts` (math or text, set on one baseline), `colors` (an emphasis role per part), `pos` | `color` (emphasis, primary), `size` (text, math), `glow` (glow, none) |

Every type also has `opacity` (0-1; below 0.5 it stops hiding things behind it), `dim`
(0-1, owned by `dim_others`; what is drawn is `opacity` x `dim`) and
`z_index` (higher draws on top among objects at the same depth). Every object takes
`layer="overlay"` to live in screen space: units are frame heights, origin top-left, y down.

Roles: emphasis `primary secondary muted highlight danger`; stroke `hairline regular bold`;
text `title label annotation math`; glow `none soft strong`; colormap `phase sequential`;
surface `solid glass wire emissive` (3D materials; glass is see-through and hides nothing);
fill `none soft solid` (a closed path's fill opacity, from the theme); dash `solid dashed
dotted` (how a path's stroke is cut).

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

**Deep zooms.** `cx.Fractal` shows the plane around `center` magnified `10**zoom` times.
`center` is text, because a float holds 16 digits and a 1e-30 view needs more. Precision
follows the pixel size by itself: float32, float64 under a 1e-5 view, then (Mandelbrot only)
perturbation under 1e-10 per pixel: one orbit computed exactly, every pixel as a float64
difference from it. Semantics report `zoom`, `view_height` and `precision`. Deep views need
`max_iter` in the thousands; `steady=True` keeps a dive's colors from cycling.

```python
from phasor import Scene, cx

dive = Scene("dive")
dive.add(
    cx.Fractal(
        id="set",
        center="-0.77568376800905379746948350393474104 0.13646736829469012473327440961784876",
        pos=(0, 0, 0),
        size=(8, 4.5),
        max_iter=3000,
        steady=True,
    )
)
dive.at(1.0).to("set.zoom", 30.0, duration=10.0)
sem = dive.engine.inspect(11.0, id="set")["objects"][0]["semantic"]
assert sem["precision"] == "perturbation" and sem["view_height"] < 1e-29
```

`glow` makes a curve emit light: the renderer writes that light to a second target and the
final look blooms it. Plain paint (`glow: none`) never blooms, and colors below the theme's
`knee` (linear 0.9) come out exactly as the theme defines them.

### Expressions

`expr("...")` computes a parameter from other parameters (`<id>.<param>`) and time `t`.
Operators `+ - * / ** %`, indexing `dot.pos[0]`, constants `pi tau e`, functions
`sin cos tan exp log sqrt floor ceil abs min max clamp lerp vec length normalize dot midpoint`.
Scalars broadcast over vectors. A cycle is an error.

## Your own types, composites and updaters (ADR 0005)

A type needs two members: `Params` and `draw(ctx, state)`. Its `world_bounds`, `features`
and `semantics` are worked out from what `draw` emits (`core/record.py`), so the inspector,
layout and lint know it like any shipped type:

```python
import numpy as np
from typing import ClassVar

from phasor import Dot, Line, Scene, Text, Value
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3


@register
class Kite(ObjectBase):
    type: ClassVar[str] = "doc_kite"
    Style: ClassVar = {"color": ("emphasis", "highlight")}

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = 0.4

    def draw(self, ctx, state):
        (x, y, z), r = self.me(state).params.pos, self.me(state).params.r
        pts = np.array([[x, y + r, z], [x + r / 2, y, z], [x, y - r, z], [x - r / 2, y, z]])
        ctx.triangles(pts, np.array([[0, 1, 2], [0, 2, 3]]), color=self.me(state).style["color"])


k = Scene("kites")
k.add(Kite(id="kite", pos=(1, 0)))
seen = k.engine.inspect(0.0, id="kite", detail="full")["objects"][0]
assert seen["semantic"]["type"] == "doc_kite" and "top" in seen["features"]
```

- Derived `world_bounds`: the box of everything drawn (a `field`'s quad corners; its pixels
  are never computed). Derived `features`: `center` and the box's compass points (`top`,
  `top_right`, ...), plus `start`, `end` and `curve` when the object draws one stroke through
  `domains/strokes.py`. Derived `semantics`: `{"type", "params"}`.
- Write any of the three yourself when you know more (a pole, a tip, `"kind": "point"`);
  yours wins. `draw(self, ctx, state, pass_)` still works: the pass is optional.
- A recording is cached per (type, id, parameters, style, camera, and each file the type
  names in `Files`: a changed file is drawn again); `draw` may read other objects
  (`state["other"]`), `state.t` and `state.values`, and the recording is made again when what
  it read changes. Its arrays (`curve`, `start`, `end`) are shared and read-only.

**Composites.** `build(self, state)` instead of `draw` returns other objects. Each part
gets the id `<composite>.<name>`, is drawn and inspected as itself (`"part_of"` in the
inspector), fades with its composite and may not reuse a retired id. The composite's bounds
are the union of its parts', its features are the box's plus each part's as `<part>.<name>`:

```python
@register
class Signpost(ObjectBase):
    type: ClassVar[str] = "doc_signpost"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def build(self, state):
        x, y, z = self.me(state).params.pos
        return [
            Line(id="post", points=[(x, y, z), (x, y + 1.2, z)]),
            Kite(id="sign", pos=(x, y + 1.2, z), r=0.3),
        ]


k.add(Signpost(id="post", pos=(-2, -1)))
state = k.engine.state_at(0.0)
assert [o.id for o in state.objects if o.parent == "post"] == ["post.post", "post.sign"]
```

Constraints name document objects; a part is placed through its composite. A part has no
updater (`bad_part`): put it on the composite.

**Updaters.** `obj.always(fn)` with `fn(t, look) -> {param: value}` keeps parameters set
from time and other objects. `look("id")` gives that object's parameters at `t`
(`look("dot").pos`); each one read is an edge in the parameter graph, so a cycle is an
error, and a read the updater did not make at t = 0 is an error too, as is a parameter it
did not return then (read and return the same things at every t). It must be pure:
`lint(strict=True)` calls every updater again, in another order, and reports
`impure_updater` when answers differ (threshold: the theme's `[lint] impure`), and
`bad_updater` with the times when one fails (it tries every sample time and at least the
first two seconds, so a scene with no markers is checked past t = 0). Plain lint reports a
time the scene cannot be evaluated at as an error with its times, instead of raising.
Something that needs the previous frame is a `Simulation`. `look` sees parameters before
living motion: tag an object `exact` when something must sit on it exactly.

**`Value`** is a named number to animate with no drawing (Manim's `ValueTracker`): read
it in an expression (`expr("speed.value")`), in an updater, or show it with a `Counter`.

```python no-run
# no-run: an updater must be defined in a file in the scene's folder (the document names
# it), and this page's text is not one. examples/open_contract.py runs the same thing, and
# tests/test_open_contract.py checks it.
def above(t, look):
    x, y, z = look("ball").pos
    return {"pos": (x, y + 0.5, z)}


def roll(t, look):
    return {"pos": (look("speed").value * t, 0.0, 0.0)}


k.add(Value(id="speed", value=0.5))
k.add(Dot(id="ball", r=0.15, tags=["exact"]).always(roll))
k.add(Text(id="ball.label", text="ball").always(above))
assert k.engine.state_at(2.0)["ball.label"].params.pos == (1.0, 0.5, 0.0)
assert not [i for i in k.engine.lint(strict=True) if i.code == "impure_updater"]
```

**Where the code lives.** A type or updater defined in the scene script, or in a
`*.types.py` file next to it, is named by the document: `meta.modules` lists the file and
`always` lists `"<file>:<function>"`. Opening the document loads those modules only where
scripts may run (`phasor.load("scene.py")`, `Engine(..., allow_scripts=True)`, the MCP
server's `--allow-scripts`, and `--allow-scripts` on `phasor inspect`, `lint`, `storyboard`,
`render`, `web`, `bench --frames` and `sketch`; render workers get exactly the engine's own
`allow_scripts`), only from inside the scene's folder. The desktop app runs a `.py` the
person opens (their own script); a `.phasor.json` it opens runs none of the modules it
names until the person trusts them (`Editor.trust_scene_code()`, which keeps unsaved edits
and starts undo again). An AI attached to the app (`phasor mcp --attach`) never makes code
run: it cannot call that, opens documents without their code, and an edit of its that
changes `meta.modules` or adds an updater is refused (`scene_code`). A missing, refused or
broken module is lint `missing_type`: the object draws as a placeholder box and keeps its
parameters and overrides (saving it changes nothing). An updater defined anywhere else (a
notebook cell, a prompt) is refused: the document could not find it again. Types shipped in phasor still get an entry here, an
example and a storyboard test; scene-local types do not need them.

Scene code belongs to its file. A type or updater a script defines (`python scene.py`,
`load`, `runpy`, or a module the engine loaded) is reached only by a document that lists
that file in its own `meta.modules`, and only where scripts may run, even if this process
already ran it: two scenes may both define `gem` or `scene.py:orbit` and never see each
other's, and a document opened without `allow_scripts` never runs a script's code. A scene's
own type may not take the name of a type shipped in phasor. Code from an imported module (a
package, a test file) and from a notebook cell is shared by the process, as phasor's own
types are. A `Scene(...)` built in a script may run that script's code; one built by imported
code may not unless it passes `allow_scripts=True`, and so does an `Engine` made from a
scene's document (`Engine(s.engine.base, ..., allow_scripts=True)`; `engine.fork()` keeps
the permission). `engine.type_of(name)` gives the class a document uses.

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
- `phasor render` muxes the narration into the video, with any **sound clips**: a `Sound`
  (Manim's `add_sound`) plays a WAV file in the scene's folder from `start` seconds, `gain`
  dB louder, cut where the scene ends. PCM WAV (8 to 32-bit, 8 to 192 kHz, 1 to 16
  channels), or with the `encode` extra the sound of an mp4, m4a, mp3, ogg or flac (decoded
  once into the user's cache; an mp4 with no sound is `bad_file`; without PyAV
  `missing_extra`), mixed down to mono and resampled to the narration's rate, else 48 kHz; at most
  `MAX_CLIP_FRAMES` (2^26) frames of a clip are read, the track covers at most 3 hours, and
  at most `MAX_SOUNDS` (256) clips and `MAX_MIX_SAMPLES` (2^30) clip samples in all are
  mixed. The mix is made a chunk at a time on the scene's own sample grid (a render of part
  of the scene gets exactly that part of the whole mix), before any frame is rendered. A
  missing file is lint `missing_file`; one phasor cannot mix (not PCM WAV, or a header that
  claims more than the file holds) is `bad_file`; a clip past the limits, or whose `start`
  or `gain` is animated (the mix reads them at 0 s), is `sound_limit`. The render leaves
  such clips out and lists them in its warnings; the semantics' `seconds` is what the file
  really holds.

```python
import tempfile
import wave
from pathlib import Path

from phasor import Dot, Scene, Sound

folder = Path(tempfile.mkdtemp())
with wave.open(str(folder / "click.wav"), "wb") as w:  # 0.1 s of a quiet click
    w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000)
    w.writeframes(b"\x00\x10" * 800)
s5 = Scene("clicks", passes=None, base_dir=folder)
s5.add(Dot(id="tick"))
s5.add(Sound(id="click", path="click.wav", start=0.5, gain=-6.0))
s5.marker("end", 2.0)
track, rate = s5.engine.audio_track()
assert rate == 48000 and len(track) == 2 * 48000  # the scene's length, no narration needed
assert track[:24000].max() == 0 and track[24000 : 24000 + 4800].max() > 0
assert s5.engine.inspect(0.0, id="click")["objects"][0]["semantic"]["seconds"] == 0.1
```

### Narration from a recording, and snapping the scene to it

`s.narrate(script, audio="talk.wav")` takes the narration from a recording in the scene's
folder instead of a voice: a WAV, or the sound of an `.mp4`, `.m4a`, `.mp3`, `.ogg`,
`.flac` (decoded once with PyAV, the `encode` extra, into the user's cache, never into the
scene's folder; without PyAV only WAV works and lint says `missing_extra`; a WAV may be
PCM or 32/64-bit float). Speech and pauses
are found in it (pure NumPy: an energy detector with a noise floor that follows the room).
With the script it reads, each sentence is aligned to the speech that says it (by its words,
against the pauses; not equal spacing) and its `[[markers]]` fall on its words; with no
script (`s.narrate(audio="talk.wav")`) each phrase (speech up to a pause of the theme's
`[narration] phrase_pause`) is a block of unknown words (`"…"`). The recording plays from
0 s, cut into one block per sentence (`NarrationBlock.audio_span`: the part of the file it
plays), so `pace()` and re-pacing work on it as on a voice.

`s.snap_to_narration(dry_run=True)` then says how the markers you placed by hand
(`s.marker`, the scene starts) would move onto it: `moves` (marker, from, to, why,
confidence 0..1, kind `sentence`/`phrase`/`between`), `kept`, the detection's `quality` and
`flags`, and `alignment`: how the scene's clock was read against the voice's (`scale`,
`offset`, `residual`, `margin`, `ambiguous`) and, for a recording read with its script, how
the words fitted the speech (`speech`: `rate`, `misfit`, `worst`, `doubtful`). `unreliable`
is true when anything is in doubt: flags `ambiguous_timing` (the timing fits two ways:
nothing is moved; pin markers with `anchors`), `alignment_doubtful` (a sentence break is
probably misplaced: the speaker ran sentences together, or paused as long inside sentences
as between them; confidences are at most 0.3), `unreadable` (the recording is gone: quality
0), `music`, `low_snr`, `no_pauses`. Read `unreliable` before applying.

Without `dry_run` it moves them, as one patch (one undo step); a second snap moves nothing.
The scene may have been laid out on another clock than the voice's: a scale (0.6 to 1.6) and
an offset are fitted so most markers fall on sentence starts, then each marker goes to its
sentence by the steps between markers (a scene that drifts, slower in one half, still lands);
or where speech resumes after a pause; markers left over keep their place between those,
stretched with them. Markers never change order or pass a marker the narration owns, and no
scene gets shorter than the motions anchored at its marker need (at least the theme's
shortest verb) unless `allow_short=True`. `anchors={"poles": 2}` pins a marker to the start
of `narration.blocks[2]`; anchors that would reorder markers or cross one the narration owns
are refused (`ValueError`). Motions anchored to a moved marker follow it. Lint
`narration_drift` lists the markers a snap would move (no `fix`: the snap is the offer). It
works on any narration; on an estimated one the confidence is at most 0.5.

Limits (measured on synthetic speech, not on voices): sentence starts are found within about
40 ms at 12 dB SNR and better; a reader who does not pause between sentences, or a slow one
whose word gaps are as long as sentence pauses, can get sentence starts wrong: that is
flagged `alignment_doubtful`, not hidden. Two markers or few sentences say little about the
scene's clock; very evenly spaced sentences can fit two ways (`ambiguous_timing`). Sound is
decoded to at most 3 hours and at most 64 times the file's size (a tiny file that decodes to
hours of silence is refused); the decode cache is pruned, oldest first, to 4 GiB.

```python
import tempfile
import wave
from pathlib import Path

import numpy as np

from phasor import Dot, Scene

folder = Path(tempfile.mkdtemp())
rate = 8000
x = np.zeros(7 * rate)
for a, b in ((0.5, 1.9), (2.6, 3.8), (4.5, 6.4)):  # three "sentences": tone bursts here
    i, j = int(a * rate), int(b * rate)
    x[i:j] = 0.4 * np.sin(2 * np.pi * 220 * np.arange(j - i) / rate)
with wave.open(str(folder / "talk.wav"), "wb") as w:
    w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
    w.writeframes((x * 32767).astype("<i2").tobytes())

s6 = Scene("recorded", passes=None, base_dir=folder)
s6.add(Dot(id="p"))
s6.marker("start", 0.0)  # laid out before the recording existed
s6.marker("move", 2.0)
s6.at("start").bloom_in("p")
s6.at("move").settle("p.pos", (2, 0))
s6.narrate(
    "One thing appears here. It moves [[moves]] right. And that is all there is.", audio="talk.wav"
)
plan = s6.snap_to_narration(dry_run=True)
assert [m["marker"] for m in plan["moves"]] == ["start", "move"]
s6.snap_to_narration()
markers = s6.engine.document.timeline.markers
assert abs(markers["move"] - 2.6) < 0.08  # sentence 2 starts there (less a 0.03 s pre-roll)
assert s6.snap_to_narration(dry_run=True)["moves"] == []  # nothing left to move
```

## Animate with verbs

| Verb | Does |
| --- | --- |
| `reveal(*ids)` | fade in; hidden until then; drops earlier dimming (it is the new focus) |
| `exit(*ids)` | fade out |
| `trace(*ids)` | draw a `Line` from start to end (`progress` 0 to 1); nothing drawn before |
| `bloom_in(*ids)` | fade in and grow from a point with a small overshoot (objects with `r`) |
| `dim_others(*keep)` | dim everything but the object (or objects) you keep, to the theme's `dim_level` (backdrops stay) |
| `undim()` | bring dimmed objects back |
| `focus_pull(id, margin=0.15)` | move the camera onto an object |
| `morph(path, value)` | change any parameter smoothly (a whole-number one, such as `terms`, in whole steps) |
| `settle(path, value)` | move like a released spring (overshoot, then rest) |
| `transform(a, b)` | `a` becomes `b`. Two equations (as pictures): parts that appear in both glide over (and change color); the rest fade. Anything else with an outline (paths, shapes, text, an equation as shapes, an SVG): the outline morphs (see "Text as shapes and morphing") |
| `transform_matching(a, b)` | `a` becomes `b` piece by piece: equal pieces move to their new place (equations: by TeX and keys; anything else: glyph or path by shape), the rest fade |
| `become(a, b)` | curve `a` turns into the shape of curve `b` (a circle into a square), then `b` takes its place |
| `write(*ids)` | write text or an equation in, left to right (`write` 0 to 1); drawn `as_shapes`, each glyph's outline is drawn, then filled |
| `travel(*ids)` | send `Tracer`s along their curves, trailing light (`progress` 0 to 1) |
| `flash(*ids)` | a quick brightening (every object has `flash`, 0 to 1) |
| `circumscribe(id, shape="circle")` | draw a ring (or `"box"`) around an object, then fade it: the child `<id>.ring` |
| `wiggle(*ids)` | a small side-to-side shake (objects with `pos`) |
| `sweep(path, value, show=True)` | run a number evenly to `value`, with a live readout `<id>.readout` beside it |

Durations, easing and stagger come from the theme's `[motion]` table. Chain with `then()`:
`s.at("loop_draw").trace("loop").then().reveal("loop.label")`. Several ids stagger.
`orbit_reveal` arrives with 3D (Milestone 6).

Dimming uses each object's `dim` factor (drawn opacity = `opacity` x `dim`), so dimming and
revealing never fight, whatever order you write them in.

**Objects made for motion.** `Tracer(path="curve_id", progress=, trail=0.15)` is a glowing
point riding another curve (a `Line`, `calc.Graph`, `geo.ParametricCurve` or
`fourier.PartialSum`), its trail fading behind it (preset `comet`); semantics give its
`position`, or `path_missing`. `Counter(value=expr("comet.progress"), prefix="p = ",
digits=2)` shows a number that follows whatever it reads. `Callout(target=, shape=)` is the
ring `circumscribe` draws. `cx.ComplexMap(function="z**2", amount=0)` is a grid pushed
through f (animate `amount` 0 to 1 with `morph`; `center` is the complex point at the grid's
middle, drawn at `pos`; lines break at poles). `geo.FlowParticles(fp=, fq=, fr=, count=,
speed=, lifetime=, tail=)` streams particles along a field (random starts from
`rng(id, seed)`, a pure function of time; presets `streaks`, `sparks`). `geo.ParametricCurve`
and `geo.ParametricSurface` take `to_fx`/`to_fy`/`to_fz` and `blend` (0 to 1) to flow into
another formula. `examples/motion_gallery.py` shows them all.

```python
from phasor import Counter, Equation, Scene, Tracer, cx, expr, geo

mo = Scene("motion", passes=None)
mo.add(Equation(id="before", parts=("$x$", "$=2$")))
mo.add(Equation(id="after", parts=("$x$", "$+1$", "$=3$")))
mo.add(geo.ParametricCurve(id="loop", fx="cos(t)", fy="sin(2*t)", pos=(3, 0, 0)))
mo.add(Tracer(id="comet", path="loop", preset="comet"))
mo.add(Counter(id="comet.count", value=expr("comet.progress"), prefix="p = ", pos=(3, -1.5, 0)))
mo.add(cx.ComplexMap(id="bend", function="z**2", amount=0.0, extent=(1.6, 1.6), pos=(-3, 0, 0)))
mo.marker("go", 0.5)
mo.at("go").transform("before", "after").travel("comet").morph("bend.amount", 1.0)
mo.at("go").then().flash("after").circumscribe("after")
me = mo.engine
assert me.inspect(1.0, id="after")["objects"][0]["semantic"]["matched"] == 1
assert me.inspect(4.0, id="comet.count")["objects"][0]["semantic"]["text"] == "$p = 1.00$"
assert "after.ring" in [o.id for o in me.document.objects]
```

**Smooth transitions** (`examples/prism_transitions.py`).

- *Solving and rearranging.* Give each part of an `Equation` a name with `keys` (`""` for
  none). In a `transform`, the part with the same key in the source becomes that part
  wherever it was, even when its text changes: `+a` on the left becomes `-a` on the right,
  and `0.6 i` turns into `0.7 i` in place. The text changes while the part moves. Parts
  without keys pair by equal text, in order, as before.
- *Carried parts arc.* A part carried past others travels round them by the theme's
  `[motion] transform_arc` line heights: over them going right, under them going left. The
  parts that keep their order go straight.
- *Semantics.* Mid-transform they report `matched`, `changed` (pairs whose text differs)
  and `carried_over` (pairs that arc).
- *A function changing into another.* `cx.DomainColoring(to="...", blend=0..1)` shows
  w = (1 - blend) f + blend g, a straight line between the two functions. Every frame is
  then a true picture of a function: zeros glide, a pole of g grows out of a point, and
  there are no seams. A blend in log space would cut the plane along the branch cuts of
  log f and log g. At 0 and 1 you get exactly f and g. Both `blend` and `a` are kernel
  arguments, so animating them never recompiles.
- *Slow changes.* `morph(path, value, until="marker")` runs a change until a marker instead
  of the theme's `morph` time: a constant drifting slowly while it is talked about.

```python
from phasor import Equation, Scene, cx

tr = Scene("solve", passes=None)
tr.add(
    cx.DomainColoring(
        id="plane",
        function="(z**5 - 1 - a)/(z**2 + a)",
        to="(z**3 - 1 - a)/(z**2 + a)",
        a=(0.0, 0.6),
    )
)
tr.add(Equation(id="step1", parts=("$z^2$", r"$+\,a$", "$=$", "$0$"), keys=("z", "a", "eq", "")))
tr.add(Equation(id="step2", parts=("$z^2$", "$=$", "$-a$"), keys=("z", "eq", "a")))
for name, t in (("solve", 1.0), ("blend", 3.0), ("done", 9.0)):
    tr.marker(name, t)
tr.at("solve").transform("step1", "step2").morph("plane.a", (0.5, 0.7), until="blend")
tr.at("blend").morph("plane.blend", 1.0, until="done")
mid = tr.engine.inspect(1.5, id="step2")["objects"][0]["semantic"]
assert (mid["matched"], mid["changed"], mid["carried_over"]) == (3, 1, 1)
assert len(tr.engine.inspect(9.0, id="plane")["objects"][0]["semantic"]["zeros"]) == 3
```

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
Easings: `linear smooth smoother sigmoid in out in_out`, and `back spring` (these overshoot:
not for opacity). `sigmoid` is Manim's `smooth` (slower to start; the Manim import uses it). Custom easing functions are not supported: a document must mean the same
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
| `set_narration` | `narration` (from `phasor.voice.narrate` or `phasor.voice.from_audio`) or `null`; replaces the markers it owns |
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
| `a outside b` (`gap N`) | no overlap: leave by whichever side needs the least movement |
| `a tangent_to b` | touch the outside edge of `b` (boxes), lined up with it |
| `a attach_to b at top` (`gap N`) | sit at a named spot of `b`: `top`, `bottom`, `left`, `right`, `center`, `top_left`, ..., or one of its feature points such as an arrow's `tip` |
| `on_screen(a)` or `on_screen(a, margin)` | keep `a` inside the frame (moves it when it can) |
| `no_overlap(a, b, ...)`, `min_size_px(a, n)`, `readable(a)` | checked by the linter |

- `label(x)` means the object `x.label`. Numbers are in the object's layer units.
- Layout is solved at every marker, every transition end and 10 times a second, then baked.
  `engine.state_at(t)`, the inspector and the renderer all read the same bake.
- A label changes slot only when another slot is over 20% better, and then it jumps.
- `engine.layout` shows the bake: `times`, `slots` (per label, per key time), `issues`.
- A rule that cannot hold is a lint error `unsolved`, with the times. Nothing is dropped.
- `visible_in_3d(a)` (not hidden by anything in front of it) is a guard checked by the linter.
- `camera keep_visible(a)` and the other camera goals are in the 3D section.

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
| `missing_file` | warning | a file an object reads (a room's `.obj`) is not in the scene's folder, or is not a regular file (a folder, a FIFO, a device, a link that loops); it draws nothing |
| `missing_extra` | warning | an object needs an optional package that is not installed (`SVG` needs the `svg` extra); it draws nothing. A `Code` without pygments (the `code` extra) is drawn plain instead |
| `fill_crossing` | warning | a filled path whose edges cross: the fill may not follow its rule |
| `fill_unchecked` | warning | a filled path with more edges than are checked for crossings (`path_math.CROSSING_LIMIT`, 20000) |
| `dash_limit` | warning | a dashed or dotted path that would need more than `MAX_DASHES` (2000) dashes: they are drawn longer |
| `glyph_limit` | warning | text or an equation drawn `as_shapes` with more than `glyph_math.MAX_GLYPHS` (400) glyphs, or with glyphs past `path.MAX_COORD` (1e9 world units): those are not drawn |
| `morph_missing` | warning | a morph whose source (`morph_from`) is not in the scene or has no outline: the object draws itself |
| `morph_limit` | warning | a morph between outlines with more than `morph_math.MAX_SUBPATHS` (512) subpaths a side: the rest fade |
| `bad_file` | warning | a file an object reads is not valid (an SVG that is not SVG), or some of its shapes are skipped (points not finite or absurdly far out); what cannot be read draws nothing |
| `sound_limit` | warning | a sound clip the mix leaves out: past `narration.MAX_SOUNDS` (256) clips in one scene; or a clip whose `start` or `gain` is animated (the mix reads them once, at 0 s, so the animation is not heard) |
| `narration_drift` | warning | a marker placed by hand that `snap_to_narration` would move by more than `[lint] narration_drift` seconds (0.3) onto a sentence of a recorded narration; no fix (the snap is offered); `missing_file`, `bad_file` and `missing_extra` also cover the recording |
| `svg_too_complex` | warning | an SVG past a reading limit (size, elements once `<use>` is expanded, nesting, shapes, segments): its drawing is empty or cut short |
| `crowded_ticks` | warning | numbers along an axis (or a chart's bar names) closer **on screen** than `[lint] tick_gap` text heights (every pair, so an axis a 3D camera sees end-on is caught); `path` names the axis (`ax.x`), the hint the smallest nice step that clears them, or to turn the camera when the axis points at it |
| `truncated` | warning | a file a `Table` or `Code` reads is past a limit (rows, columns, lines), or a picture sequence has more than 3000 pictures; only its first part is shown (plays) |
| `unknown_language` | warning | a `Code` language pygments does not know: drawn plain |
| `clamped` | warning | less is drawn than asked: an axis range past 100 ticks (drawn at a wider nice step), a span lost in rounding (no ticks), axis labels past one per axis or that cannot be typeset (left out), bars outside a chart's `y_range` (cut at it) |
| `video_too_big` | warning | a clip or picture sequence past a cap (a frame past 8K, more than 216 000 frames, longer than 4 hours): it is not drawn |
| `clip_ended` | warning | a clip or sequence shown past its end with `loop` off, or trimmed from at or past its end: it holds its last frame (no fix: loop it, slow it, trim less or fade it out) |
| `too_many_clips` | warning | more `Video` objects shown at once than a process keeps decoders open (8, or 4 of one file): each frame is found by seeking, so drawing is much slower |
| `flow_budget` | warning | a flow object past its budget: a set or path whose points x steps pass the theme's `[lint] flow_work`, a field with more arrows than `[lint] flow_arrows`, or work cut at a hard limit (`flow_math.MAX_POINTS`, `MAX_PROBE_STEPS`, `MAX_STEPS`, `flow.MAX_ARROWS`, points x trail) |
| `flow_too_fast` | warning | points of a flow that moved further in one step than the window allows even after 16 sub-steps (a stiff field such as `1e6*x`): they are stopped, not flung across the screen |
| `duplicate_variable` | warning | a `flow.Variable` whose name an earlier `Variable` already has: systems read the first, so this one changes nothing |
| `set_under_resolved` | warning | a `FlowSet` outline that reached `flow_math.MAX_SET_POINTS` (2^17) points: from that flow time it is drawn as points, not a curve |

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

## Footage and picture sequences

`Video` plays a clip from the scene's folder; `Image` plays a picture sequence (a path with
`%04d` in its file name: frames 0, 1, 2, ... or from the first of 0-4 that is there, up to the
first gap, at most 3000). Both place, size and animate like any object, and play from
`start` (scene seconds) the part `trim` keeps, at `speed`, looping or holding their last
frame. The frame shown at t is the last one whose presentation time is at or before the
clip time t maps to: the same t shows the same frame in any order and in any render worker.

`fit` says how the picture sits in its `width` x `height` box: `stretch` (the default, as
`Image` always did), `contain` (inside, keeping its shape), `cover` (filling it, cut to
it), `none` (its own size: 1080 pixels fill the camera's height, cut to the box). `fill`
makes the box the camera's whole view (2D camera; ignored in perspective). `backdrop` (always
with `fill`) lets labels sit on it: layout does not avoid it, lint `overlap` does not count
it, and `contrast` judges text against the picture (without OpenGL, against the mean colour
of the frame shown). Semantics add `frames`, `fps`, `seconds`, `frame` (shown at t),
`clip_time`, `looping`, `trimmed`, `ended`. `start`, `trim`, `speed` (and a sequence's
`fps`) hold one value for the whole scene: a track, an expression or an updater on one is a
validation error (`bad_param`), since clip time is worked out from them at each t (to
change speed part-way, play two objects one after the other).

`s.background(path, **params)` puts a picture, a sequence or a clip behind everything
(`fill`, `fit="cover"`, `z_index` -1000); `Scene.from_video(path, clip={...})` makes a scene
of one, as long as it plays (`meta.duration`, read from the file when the script runs).

```python
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from phasor import Image, Scene, Text

folder = Path(tempfile.mkdtemp())
(folder / "frames").mkdir()
for i in range(6):  # six pictures: frames/shot_0000.png ... shot_0005.png
    PILImage.new("RGB", (32, 18), (40 * i, 60, 120)).save(folder / "frames" / f"shot_{i:04d}.png")
s7 = Scene("stills", passes=None, base_dir=folder)
s7.background("frames/shot_%04d.png", fps=3.0, loop=True)  # the whole frame, playing
s7.add(Image(id="still", path="frames/shot_0002.png", width=2.0, pos=(2.5, 1.2, 0)))
s7.add(Text(id="note", text="on the pictures", pos=(0, -1.5, 0)))
s7.marker("end", 4.0)
sem = s7.engine.inspect(1.0, id="background")["objects"][0]["semantic"]
assert sem["frames"] == 6 and sem["frame"] == 3 and sem["backdrop"]
assert "overlap" not in {i.code for i in s7.engine.lint()}
```

```python
import av  # PyAV: the ``encode`` extra

from phasor import Video

with av.open(str(folder / "clip.mp4"), "w") as c:  # one second, getting lighter
    st = c.add_stream("mpeg4", rate=24)
    st.width, st.height, st.pix_fmt = 64, 36, "yuv420p"
    for i in range(24):
        frame = av.VideoFrame.from_ndarray(np.full((36, 64, 3), 10 * i, np.uint8), "rgb24")
        for packet in st.encode(frame):
            c.mux(packet)
    for packet in st.encode():
        c.mux(packet)
s8 = Scene.from_video("clip.mp4", base_dir=folder, passes=None, clip={"speed": 0.5})
assert s8.engine.document.meta.duration == 2.0  # at half speed it plays two seconds
s8.add(
    Video(id="inset", path="clip.mp4", width=1.6, pos=(2.6, -1.2, 0), trim=(0.25, 0.75), loop=True)
)
sem = s8.engine.inspect(1.0, id="inset")["objects"][0]["semantic"]
assert sem["frame"] == 6 and sem["looping"] and sem["trimmed"]  # 1 s in: looped to 0.25 s
```

Limits (`domains/media_math.py`): a frame past 8K (7680 x 4320), more than 216 000 frames,
longer than 4 hours or a file over 64 GB is `video_too_big`; a file that is not a video, has
no video stream, no frames, frames that do not decode or an absurd size is `bad_file`; a
picture past Pillow's own bomb limit (about 179 million pixels) is `bad_file`. Each draws
nothing. A sequence past 3000 pictures plays its first 3000 (`truncated`). Frames larger
than 4096 pixels are scaled down when decoded; decoded frames are kept in a cache of at most
256 MB per process, stills in one of their own (2 GB). Clips decode forward when played in
order and otherwise seek to the keyframe before the frame (open GOPs and mp4 edit lists
included); a process keeps at most 8 decoders open, 4 of one file (`too_many_clips` past
that).

## Stills and files

```python no-run
# no-run: needs OpenGL 4.3 and writes files
from phasor import load
from phasor.render.output import save_png

save_png("out/frame.png", engine.still(3.0, (960, 540)))  # (h, w, 3) uint8 sRGB
picture = engine.frame(3.0, (3840, 2160))  # linear, after bloom and tone map; float32
bytes8 = engine.frame(3.0, (3840, 2160), output="srgb8")  # (h, w, 3) uint8, encoded on the card
s.save("residues.phasor.json")  # canonical, diff-friendly
engine = load("residues.phasor.json")  # or load("residues.py")
```

## Render presets and the draft render (Milestone 21)

One table (`phasor.render.presets.PRESETS`; `phasor doctor` prints it) says what each
quality draws:

| preset | size | fps | motion blur | look | life, camera drift | still frames | encoder |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `draft` | long side 640, the scene's aspect, even sides, never enlarged | 15 | off | plain | off | repeated, not drawn | fast |
| `preview` | 960 x 540 | 30 | off | full | on | drawn | quality |
| `final` | the scene's own | the scene's | 0.5, 4 samples | full | on | drawn | quality |
| `4k` | 3840 x 2160 | the scene's | 0.5, 4 samples | full | on | drawn | quality |

What the draft skips and keeps, effect by effect:

| effect | where | draft |
| --- | --- | --- |
| bloom (bright pass, blur pyramid) | `render/post.py`, theme `[bloom]` strength, levels | skipped; the emission target is not even read back |
| lens: fringe, vignette | `render/post.py`, theme `[bloom]` fringe, vignette | skipped |
| tone map (exposure, knee) | `render/post.py` | kept: once, as always (contract 4) |
| sRGB encode | `render/post.py`, `render/output.py` | kept: once |
| depth fog ("haze") on lit surfaces | `render/passes.py`, theme `[light]` fog | skipped (`u_fog` 0) |
| lit surfaces: key/fill/rim light, specular, glass, wire grid | `render/passes.py`, theme `[light]` | kept |
| glow (HDR paint brighter than 1) | `render/passes.py`, theme `[glow]` | kept in the picture; its halo is the bloom, skipped |
| antialiasing (4x MSAA) | `render/passes.py` | kept (thin strokes break up without it) |
| hue along strokes | theme `[strokes]` hue_along, scene `palette` | kept (it is the prism look's color) |
| hue flowing along strokes over time | theme `[strokes]` flow (`prism_vivid`) | kept (those frames never repeat) |
| kernel pictures (domain coloring, fractals, heat maps, point clouds) and their glow-weight "haze" | `render/gpu/kernels/` | kept, computed at the draft's size only |
| motion blur | `Engine.frame_blurred` | skipped (shutter 0) |
| life: glow breathing, shimmer, drifting dots and equations, turning field colors, twinkle | theme `[life]`, scene `life` | skipped: the scene is drawn with `life="off"` |
| the theme's slow camera drift (3D) | theme `[camera]` drift | skipped when the camera takes the theme's (`drift` < 0); an explicit `drift` is kept |
| verbs, tracks, expressions, simulations, the flash verb | `[motion]`, the document | kept: everything the document animates moves |
| narration and sound clips | `Engine.audio_track` | kept (muxed into an `.mp4`) |

How it is fast: the picture is about a ninth of the pixels of 1080p and half the frames of
`preview`; no bloom pyramid, no emission readback, no blur samples; kernels compute only the
draft's pixels; the layout is baked once (a draft draws `engine.without_decoration()`, a
fork with life off that bakes once per version of the document and is kept); kernels use
every core without CUDA; and a frame whose drawing is identical to the last one is not
drawn again: its bytes are repeated (`render.job.FrameDrawer`). Whether two frames are
identical is decided from what the objects' `draw` recorded (`GLRenderer.frame_plan`): the
draw calls, draw order, each object's layer, opacity, flash and camera, the light, the
background, the stroke hue and the size. A type that draws from `state.t` with constant
parameters (flow particles) records different calls, so it is drawn every frame. Kernel
pictures are keyed as the renderer caches them (type, parameters, roles, style, quad,
background), not by their pixels. The repeated frames are byte for byte the frames drawing
would make. The encoder runs `ultrafast` (libx264; `p1` on NVENC), no B-frames, a keyframe
every second; a PNG sequence compresses at level 1.

**Where a draft is not much faster.** The savings are in pixels and post effects. A scene
whose time goes into computing its data, not its pixels, gains only the halved frame rate:
`examples/point_cloud.py` solves a million polynomial roots whenever `twist` moves, whatever
the picture's size (measured 2026-11-24, 4-core CPU: 8.1 s a frame as a draft, 6.4 s as a
preview; the draft's whole 3.0-3.4 s window took 48.5 s against 76.2 s, 1.6x). The same holds
for any kernel picture whose parameters change every frame.

```python
from phasor import Dot, Scene
from phasor.render.job import resolve_time
from phasor.render.presets import DRAFT, PRESETS

draft = PRESETS["draft"]
assert draft.size(1920, 1080) == (640, 360) and draft.size(1080, 1920) == (360, 640)
assert draft.fps == 15.0 and draft.effects == DRAFT and not DRAFT.bloom
assert PRESETS["preview"].size(1920, 1080) == (960, 540)  # unchanged

s = Scene("quick", passes=None, look="glow")
s.add(Dot(id="d"))
s.marker("intro", 0.0)
s.marker("proof", 2.0)
plain = s.engine.without_decoration()  # what a draft draws
assert plain.document.meta.life == "off" and s.engine.document.meta.life == "lively"
assert resolve_time(s.engine, "proof") == 2.0  # a window starts at a marker
```

```python no-run
# no-run: needs OpenGL 4.3 and writes files
import threading

s.render("quick.draft.mp4", preset="draft")  # Scene.render: workers=N renders in N processes
engine.render("x.mp4", preset="draft", start="proof", end="outro")  # a window, by markers
stop = threading.Event()  # set it from another thread (a Cancel button) to stop
show = lambda k, n: print(f"{k}/{n}")  # noqa: E731
r = engine.render("x.draft.mp4", preset="draft", cancel=stop, progress=show)
# stopped: RenderCancelled, and the partial file (or the PNGs it wrote, never the folder)
# is removed
r["drawn"], r["repeated"]  # frames drawn, frames repeated
```

`render_file(...)`, `Engine.render(...)` and `Scene.render(...)` take `preset`, `fps`,
`start`/`end`/`frame` (seconds or a marker name), `progress(k, n)` and `cancel` (a
`threading.Event` or a callable, checked between frames; cancelling raises
`render.job.RenderCancelled` and removes what was written, and the folders the render
made if they are empty; Ctrl-C does the same). A bad preset, a frame rate outside 1/1001 to
1000, a time that is not finite, an unknown marker (a marker name is tried first, so a
marker called `inf` works) or a window of more than `render.job.MAX_FRAMES` (1,000,000:
4.6 hours at 60 fps) is a `ValueError` before anything is written. A draft's default file is `<scene>.draft.mp4`, so it
never overwrites the real render. `phasor preview SCENE` is `phasor render SCENE --preset
draft`. Untested on Windows, macOS and CUDA (NVENC's fast settings included).

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
| `domain` | `function`, `center`, `height`, `contour=0.35`, `lut=None`, `precision="f32"`, `a=0j` (the constant `a` in the function) | RGBA: hue = phase, bands = each doubling of the modulus, poles white |
| `domain_glow` | `function`, `center`, `height`, `lut=None`, `radius=0.3`, `ambient=0.08`, `contour=0.15`, `precision="f32"`, `lines=0`, `line_glow=0.6`, `a=0j` | RGBA: hue = phase, lit near zeros and poles (Newton distance \|f/f'\|), rays and rings about a pixel wide; alpha = glow weight |
| `escape` | `center`, `height`, `max_iter=256`, `julia=None`, `precision=None` | smooth escape count; Mandelbrot, or the Julia set of `julia=(re, im)` |
| `fractal_glow` | `field` (from `escape` or `deep`), `lut`, `density=0.35`, `offset=0.0`, `inside=(0,0,0)`, `radius=2.0` (pixels), `ambient=0.08` | RGBA: the palette ramp by sqrt(count), lit along the edge of the set (distance from the count's gradient, 1 / (ln 2 \|grad\|)); inside = `inside`, alpha 0; alpha = glow weight |
| `tonemap` | `image`, `exposure=1.0` | ACES filmic curve, linear in and out (whole image) |
| `gauss` | `image`, `sigma`, `axis` | one pass of a Gaussian blur along x (0) or y (1) (whole image) |
| `srgb8` | `image` | linear display RGBA (0..1) -> (h, w, 3) uint8 sRGB (whole image) |
| `accumulate` | `image`, `weight=1.0`, `acc=None` | float32 `acc + weight * image`; `image` may be float16 (whole image, any channel count; the motion-blur sum) |
| `splat` | `points` ((n, 2) plane coordinates), `colors` ((n, 3) linear RGB, or one), `center`, `height`, `weights=None`, `sigma=1.0` (px), `gain=1.0`, `curve=True`, `white=0.0` | each point adds a Gaussian of light (integer `atomicAdd` in fixed point, so the same bits every run); rgb = mean colour (toward white by white x a^4) x a, alpha a = 1 - exp(-gain x density); `curve=False`: the raw sums |

- `center` and `height` describe the view like the 2D camera: the point at the middle of
  the frame and how many units the frame is tall.
- `gpu.render(name, width, height_px, supersample=1, backend="auto", **params)` runs tile by
  tile within the VRAM budget and returns a NumPy array. `gpu.run(name, tile, **params)`
  runs one tile and returns a CuPy array on CUDA.
- With no CUDA (tier B) the same CUDA source is compiled for the CPU by g++ or clang++
  (found at run time, cached in the user cache) and runs on every core; with no compiler
  either, the NumPy reference runs: same picture, slower. `backend="cuda"` demands the GPU,
  `backend="cpu"` the CPU backend, `backend="reference"` forces NumPy. `gpu.run(...,
  threads=n)`, `phasor.render.cpu.set_threads(n)` or `PHASOR_THREADS` set the CPU threads
  (default: every core); results are the same bits at any thread count.

```python
import numpy as np

from phasor.render import cpu, gpu
from phasor.render.gpu.registry import active_backend

print(active_backend(), cpu.threads(), cpu.find_compiler())  # e.g. "cpu", 4, g++ 13.3
if cpu.find_compiler() is not None:
    lit = np.ones((4, 8, 4), dtype=np.float32)
    one = gpu.run("vignette", gpu.Tile.full(8, 4), image=lit, backend="cpu", threads=1)
    all_cores = gpu.run("vignette", gpu.Tile.full(8, 4), image=lit, backend="cpu")
    assert np.array_equal(one, all_cores)  # the thread count never changes a result
```
- `escape` picks float64 by itself once `height` is under 1e-5.
- Formulas may use `z`, `i`, `pi`, `e`, numbers, `+ - * / **` and
  `exp log sqrt sin cos tan sinh cosh tanh conj abs`. Anything else is a `FunctionError`.
- A new kernel needs `@kernel(name, signature, reference=numpy_fn)`; registration fails
  without a NumPy reference that takes the same parameters.

## Looks, presets, living motion and taste

Start a scene from a **look**: a theme plus a few choices, by name. A new scene gets the
**`prism` look** unless it names another look or a theme (owner, 2026-11-22); `look="none"`
gives a plain scene on the dark `midnight_glow` theme. Everything stays dark by default;
light is opt-in. (The test suite sets `PHASOR_DEFAULT_LOOK=none`, so its own examples here
run plain.)

| Look | Theme | Life | Character |
| --- | --- | --- | --- |
| `calm` | `midnight_glow` | `subtle` | soft contour edges, gentle glow, slow drift |
| `glow` | `midnight_glow` | `lively` | strong bloom, everything breathes |
| `chalk` | `paper` | `off` | light page, ink strokes, no glow, crisp, perfectly still |

Themes: `midnight_glow` (dark blue-black), `aurora` (dark green-black, teal and violet),
`paper` (light). `Scene(title, theme=..., look=..., life=...)`; explicit `theme` and `life`
beat the look's. The document stores the resolved values, so a look changing later never
changes an old scene.

**Object presets** are named starting points: `Dot(id="d", preset="beacon")`,
`cx.DomainColoring(id="p", function="z", preset="soft_phase")`. Precedence: what you set
yourself, then the preset, then the look, then the type's default. Presets: `dot`: `plain`,
`glowing`, `beacon`; `line`: `ink`, `glowing`, `beam`, `hairline`; `domain_coloring`:
`classic` (hard edges), `soft_phase` (restrained wheel, feathered), `vivid`, `prism` (dark, glowing zeros and poles, rays and rings of light; the `prism` looks use it); `fractal`:
`classic`, `ember`, `ocean`, `prism` (the set dark, its edge glowing; the `prism` looks use it); `riemann_surface`: `glass`, `soft_glass`.

**Palettes** are style roles (`colormap`): `phase` (the full wheel: use it where hue means
phase), `soft_phase` (three-hue wheel), `sequential`, `ember`, `ocean` (two-tone ramps),
`spectrum` (a bright even wheel for color running along a curve).
Fewer rainbows does not mean none: domain coloring keeps the full wheel.

**Edges**: `cx.DomainColoring(softness=0..1)` feathers the modulus bands (0.5 by default).

**Prism domain coloring**: `preset="prism"` (or the `prism` looks) draws the plane dark,
lights each zero and pole with the phase round it as colour, and draws rays of constant
phase and rings of constant size as thin light; the glow blooms and hazes like the strokes.

```python
from phasor import Scene, cx

lit = Scene("lit poles", look="prism", passes=None)
lit.add(cx.DomainColoring(id="plane", function="(z**3 - 1)/(z**2 + 0.5)", size=(8, 4.5)))
assert lit.engine.document.object("plane").params["radius"] > 0
```

A kernel picture glows through its alpha channel: alpha is a per-pixel glow weight, and the
object's `glow` role (above 1) turns it into light and bloom. Kernels that only paint leave
alpha at 1 and the role at `none`.

**Prism fractals**: `cx.Fractal(preset="prism")` (or the `prism` looks) keeps the set dark
(the background), colours the outside from the palette by escape time, and lights the edge
of the set out to `radius` world units, with haze. The glow is sized on screen, so it stays
as fine through a deep zoom (the perturbation path glows too). Palettes still apply.

```python
from phasor import Scene, cx

edge = Scene("glowing edge", look="prism", passes=None)
edge.add(cx.Fractal(id="j", kind="julia", c=(-0.7741, 0.1501), size=(8, 4.5)))
assert edge.engine.document.object("j").params["radius"] > 0
```

**Living motion** (`life`: `off`, `subtle`, `lively`): glow breathes, dots drift a fraction
of their radius, color fields turn a few percent of the wheel, all a deterministic function
of (object id, `seed`, t), so the same frame always renders the same. Tag an object
`"exact"` to keep it perfectly still; the geometry of a field (a pole's place) never moves.
The default for plain `Scene(...)` is `off`; looks turn it on.

```python
from phasor import Dot, Scene, cx

calm = Scene("alive", look="calm", passes=None)
calm.set("camera.height", 4.5)
calm.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", preset="soft_phase"))
calm.add(Dot(id="pole", pos=(0, 1), r=0.08, preset="beacon", tags=["key"]))
calm.add(Dot(id="origin", pos=(0, 0), r=0.08, tags=["exact"]))  # never wiggles

meta = calm.engine.document.meta
assert (meta.theme, meta.life, meta.look) == ("midnight_glow", "subtle", "calm")
assert calm.engine.document.object("plane").params["softness"] == 0.8  # from the preset
assert calm.engine.state_at(0.0)["origin"].params.pos == (0.0, 0.0, 0.0)
assert calm.engine.state_at(2.0)["pole"].params.pos != calm.engine.state_at(0.0)["pole"].params.pos
```

**Math.** Any text written as `$...$` is typeset (TeX math through matplotlib's mathtext,
Computer Modern, no TeX install; for a real TeX set `PHASOR_MATH=latex`, or `auto`, with `latex`
and `dvipng` on the PATH (MiKTeX, TeX Live); `mathtext` stays the default so pictures match
across machines): `Text(text=r"$\psi(x) \approx x - \ln 2\pi$")`. Its box can
be taller than a line (sums, fractions) and layout sees the real size. Bad TeX is refused when
the object is added. For terms in different colors, `Equation(parts=(...), colors=(...))` sets
parts side by side on one baseline (a relation at a part's edge gets TeX's thick space). A
color inside a fraction is not possible: split where a part can end.

**Typeset and LaTeX, both ways.** Write maths in either dialect; it renders on either
engine, because every math string passes through `to_mathtext` (mathtext, the default) or
`to_latex` (real TeX) first. `to_mathtext` maps what mathtext lacks to its nearest form
(`\tfrac`, `\le`, `\lvert`, `\textbf`, `\big(`, `\pmod{n}`, `\xrightarrow{f}`) and drops,
keeping the content, what it cannot show (`\color`, `\boxed`, `\underbrace`; listed by
`texconv.mathtext_losses`); environments (`cases`, `pmatrix`) have no mathtext form and are
refused when added. `to_latex` turns Unicode (`π`, `≤`, `ℝ`) and mathtext-only commands
into LaTeX and escapes the text around the maths. Text and Equation report the LaTeX form
as `semantic["latex"]`.

```python
from phasor import Scene, Text, to_latex, to_mathtext

assert to_mathtext(r"$\tfrac{1}{2} \le \lvert z \rvert$") == r"$\frac{1}{2} \leq | z |$"
assert to_latex("50% of $2π r$") == r"50\% of $2\pi r$"
s = Scene("both")
s.add(Text(id="half", text=r"$\tfrac{1}{2} \le x$"))  # LaTeX-only commands, typeset by mathtext
assert (
    s.engine.inspect(0.0, id="half")["objects"][0]["semantic"]["latex"] == r"$\tfrac{1}{2} \le x$"
)
```

**The prism look** (`look="prism"`, the owner's reference style): near-black stage, curves
glowing with a hue running along them (`Line(hue_along=...)`, the `spectrum` colormap) and
warm/cool fringes on the glow (theme `[bloom] fringe`), a vignette (`[bloom] vignette`),
and typeset equations that float (life `float`).
No axes unless the scene adds them. `examples/prism_primes.py` is the reference scene.

**The prism look on everything** (`look="prism_everywhere"`, an option; the default stays
`midnight_glow`): the same lens (fringes, vignette, drifting equations), and *every* glowing
curve runs the spectrum along its length and lets it drift: graphs, Fourier partial sums and
epicycles, parametric curves, contours, the algebra and PDE drawings. Its theme
`prism_vivid` starts from `prism` (`base = "prism"`) and adds `[strokes] hue_along`, which
any theme may set. Strokes that do not glow (axes, grids) keep their color, and objects that
color by meaning (flow lines by speed, fields by strength) keep theirs.

**The prism look on a field** (`look="prism_field"`, for domain coloring scenes) is `prism`
with two changes, and `prism_domain` and `prism_transitions` use it:

- Equations drift less. `Equation(drift=)` is the share of the theme's `[life] float` that
  an equation drifts; the look sets it to 0.6.
- The field is more see-through, with more light. A shorter haze (`radius`), a darker wash
  (`ambient`), brighter rays and rings (`line_glow`, 0.6 by default) and more emitted light
  (`light`, which scales what the glow role adds; 1 by default).

A look file may start from another with `base = "prism"` and change only what it names.
During a `transform`, an equation's drift passes from the source's to the target's, so
nothing jumps.

**Palettes within the prism looks.** The lens and glow stay; the colors are a choice:
`Scene(..., look="prism_everywhere", palette="moonlight")`. `spectrum` is the prism rainbow
(the default), `moonlight` is white with a cool and warm shimmer, then `gold`, `glacier`,
`borealis` (green to violet) and `rose`. The palette is stored in the document
(`meta.palette`); it colors the hue run along every glowing stroke and the lines the look
runs a hue along. On an open scene, `scene_palette("gold")` (the AI's tool, and the editor's
Looks tab) switches it in one undo step. The palettes are colormap roles in every theme, so
any object with a `colormap` may use them too.

```python
from phasor import Line, Scene

lit = Scene("moon", look="prism_everywhere", palette="moonlight", passes=None)
lit.add(Line(id="arc", points=[(0, 0, 0), (1, 1, 0), (2, 0, 0)]))
assert lit.engine.document.meta.palette == "moonlight"
assert lit.engine.document.object("arc").style["colormap"] == "moonlight"
```

```python
from phasor import Equation, Line, Scene

p = Scene("prism", look="prism", passes=None)
p.add(Line(id="wave", points=[(x / 4, (x % 3) / 3, 0) for x in range(-12, 13)]))
p.add(
    Equation(
        id="eq",
        parts=("$E(x) =$", "$\\psi$", "$(x) - x$"),
        colors=("primary", "secondary", "primary"),
        pos=(0, 2, 0),
    )
)
st = p.engine.state_at(3.0)
assert st["wave"].params.hue_along > 0  # the look's default: color runs along curves
assert st["eq"].params.pos != (0.0, 2.0, 0.0)  # equations drift a little (life)
assert p.engine.inspect(0, id="eq")["objects"][0]["semantic"]["colors"][1] == "secondary"
```

**Taste, the art director and variants.** `lint(aesthetic=True)` adds the taste rules:
`focal_point` (one `highlight` per beat), `color_restraint` (three saturated hues at most;
phase coloring exempt), `glow_clip`, `density`, `motion_rhythm`, `margin`, `stroke_mix`,
and for explicit values `off_palette` and `off_scale` (below).
Thresholds are the theme's `[lint]`. `engine.critique()` scores them (0 is nothing to
criticise) with the markers and the fixes; `engine.polish(rounds=3)` applies them, each round
one undo step; `engine.variants(beat)` shows calmer and bolder looks of one beat side by side;
`engine.storyboard(compare="gallery")` holds frames beside the three nearest in your
`gallery/`. `phasor style tune` reads `style/choices.jsonl` (every pick, appended by the
`choose` tool) and says what you favor.

```python
from phasor import Dot, Scene

rough = Scene("rough", passes=None)
rough.set("camera.height", 6)
rough.add(Dot(id="a", pos=(5.2, 0), r=0.2, style={"color": "highlight"}))  # at the edge
rough.add(Dot(id="b", pos=(0, 0), r=0.9, style={"color": "highlight"}))  # two heroes
rough.marker("beat", 1.0)
before = rough.engine.critique()["score"]
report = rough.engine.polish(rounds=3)
assert report["after"] < before and len(report["rounds"]) <= 3
```

**Explicit colors and sizes** (owner, 2026-11-23). Beside a role, a style key may hold an
explicit value; roles stay the default and the AI-friendly path.

| style key group | explicit value |
| --- | --- |
| `emphasis` (`color`, `accent`, ...) | `"#58C4DD"` (sRGB, decoded to linear once) or linear `(r, g, b, a)` in 0..1 |
| `text` (`size`) | text height in world units (positive, at most 1e6) |
| `stroke` (`stroke`, `leader`) | stroke width in world units (positive, at most 1e6) |
| `glow`, `surface`, `colormap` | roles only (`bad_role` otherwise) |

`Scene(..., background="#101820")` (or `set_meta` `background`) gives the scene its own
background; None is the theme's. A background is opaque: a color with alpha below 1 is
refused. A world size is exact in 2D at any zoom; in 3D it is exact at the object's `pos`; on
an overlay-layer object its units are frame heights (the overlay's units). Taste stays checked: `lint(aesthetic=True)` warns with `off_palette`
when an explicit color is farther than the theme's `[lint] off_palette` (an OKLab distance)
from every palette role, or a background from the theme's, and with `off_scale` when a size
is more than `[lint] off_scale` times off the nearest text or stroke role (a composite's
parts are judged too, but not the glyph parts of a text drawn `as_shapes`: the text is). Neither has a
`fix` (the value may be meant): `critique()` lists a `snap` op per issue (the nearest
role), and `polish(snap=True)` applies it, only when asked. A scripted scene's background
is changed in its script.

```python
from phasor import Dot, Line, Scene, Text

own = Scene("own_colors", passes=None, look="none", background="#101820")
own.set("camera.height", 8)
own.add(Dot(id="sun", pos=(0, 0), r=0.5, style={"color": "#83C167"}, tags=["key"]))
own.add(Text(id="name", text="sun", pos=(0, 1.2), style={"size": 0.34}))  # ~ label
own.add(Line(id="orbit", points=[(-3, -2), (3, -2)], style={"stroke": 0.3}))  # very wide
own.marker("beat", 1.0)
st = own.engine.state_at(1.0)
assert abs(st["name"].style["size"] * st.camera.units_per_height() - 0.34) < 1e-9
codes = {i.code: i for i in own.engine.lint(aesthetic=True)}
assert codes["off_palette"].ids == ("sun",) and codes["off_scale"].ids == ("orbit",)
offered = [op for i in own.engine.critique()["issues"] for op in i["snap"]]
assert {"op": "set_style", "id": "orbit", "key": "stroke", "role": "bold"} in offered
own.engine.polish()  # fixes and suggestions only: typed values stay
assert own.engine.document.object("sun").style["color"] == "#83C167"
own.engine.polish(snap=True)  # asked: snapped to the nearest roles
assert own.engine.document.object("orbit").style["stroke"] == "bold"
```

**Style transitions.** A track on `<id>.style.<key>` moves a style key from the role (or
explicit value) it holds to another: `s.at("go").morph("dot.style.color", "highlight")` (the
theme's `morph` timing; strict-lint clean) or a raw `.to("dot.style.color", "#58C4DD",
duration=1)`. Colors (`emphasis` keys: `color`, `fill_color`, ...) mix in linear RGBA; `fill`
(the fill's opacity), `stroke` and text `size` move as numbers, and an explicit world size
follows the camera at every t. Glow, surface, colormap and dash keys cannot be animated
(`bad_style_track`). The document stores a role or a value at each key and in each segment
(a hex target is stored upper case). `state.roles[key]` is the typed value nearer in time
(the start until half-way, then the target); the inspector shows a transition in progress
as `restyling: {key: {from, to, progress}}`; `off_palette` judges an explicit target where
it is reached (its snap is left to the track). A child object named `<id>.style` keeps its
own parameter paths. On a `Line` with `hue_along` the colors come from the colormap, so a
color transition does not show on it.

```python
from phasor import Dot, Scene

fade = Scene("restyle", passes=None, look="none")
fade.marker("go", 1.0)
fade.add(Dot(id="dot", pos=(0, 0), style={"color": "primary"}))
fade.at("go").morph("dot.style.color", "highlight")
st = fade.engine.state_at(1.5)
assert st["dot"].restyling["color"]["to"] == "highlight"
assert fade.engine.document.object("dot").style["color"] == "primary"  # still a role
assert not [i for i in fade.engine.lint(strict=True) if i.code == "raw_motion"]
```

Finals render with motion blur (`phasor render --preset final`; `--motion-blur 0` turns it
off): `engine.frame_blurred(t, fps)` averages sub-frames only where something moves.

## Domains: fourier, linalg, acoustics

Domain modules register their own object types: `from phasor import fourier, linalg, ac`.
They are plain objects: styled by roles, labelled by `near` rules, animated by verbs, and
every parameter (a matrix, a frequency, a radius) can be tracked or tied to another with an
expression.

**`fourier`**: `Epicycles(points=..., terms=N, phase=0..1)` (a closed curve drawn by N rotating
circles; `phase` animates the tip), `PartialSum(kind="square"|"sawtooth"|"triangle", terms=N)`
(the Gibbs overshoot is in the semantics), `Spectrum(frequencies=, amplitudes=, window=)` (bars
per bin; `leakage` in the semantics), `Spectrogram(f0=, f1=)` (a chirp, backdrop). The maths is
`phasor.domains.fourier_math`: `epicycle_coefficients`, `series_partial_sum`, `window`,
`spectrum`, `leakage`, `spectrogram`, `convolve`.

**`linalg`**: `TransformGrid(matrix=((a, b), (c, d)))` (2x2: the grid, the unit square whose
area is the determinant, the images of the basis vectors; 3x3: the unit cube), `EigenLines`,
`Arrow(start=, end=)`, `Matrix(matrix=expr("grid.matrix"))` (entries stay linked). Semantics
give determinant, eigenvalues, singular values and rank. `linalg_math` has `svd` (its
`stages` are rotate, stretch, rotate: morph through them), `null_space`, `gram_schmidt`,
`project`.

**`ac`** (acoustics): `Room(size=(8, 5, 3), material="curtain")` (glass walls: they never hide
the source or listener), `Source`, `Listener`, `Wavefront(center=, radius=expr("343 * t"))` (a
negative radius draws nothing, so `343 * (t - 0.5)` starts at 0.5 s), `ImageSources(order=)`,
`ImpulseResponse(layer="overlay", pos=, extent=)` (arrival spikes; `reveal` and `show_rt60`
animate; semantics give `rt60`, `rt60_theory_eyring` and `rt60_reliable`), `PressureSlice(time=)`
(a slice of the real wave-equation field). Materials and absorption: `ac_math.MATERIALS`.

```python
import numpy as np
from phasor import Scene, ac, expr, fourier, linalg

dom = Scene("domains", passes=None)
dom.set("camera.height", 8)
dom.add(linalg.TransformGrid(id="grid", matrix=((2.0, 1.0), (1.0, 2.0))))
dom.add(linalg.Matrix(id="mat", matrix=expr("grid.matrix"), pos=(-5.5, 3.2, 0)))  # live entries
dom.add(fourier.PartialSum(id="wave", kind="square", terms=9, pos=(0, -3, 0), extent=(5, 1.6)))
dom.marker("go", 1.0)
dom.at("go").morph("grid.matrix", ((1.0, 0.0), (0.0, 3.0)))

e = dom.engine
assert e.state_at(3.0)["mat"].params.matrix == ((1.0, 0.0), (0.0, 3.0))  # the numbers follow
sem = e.inspect(0.0, id="grid")["objects"][0]["semantic"]
assert sem["determinant"] == 3.0 and sem["singular_values"] == [3.0, 1.0]
assert 0.085 < e.inspect(0.0, id="wave")["objects"][0]["semantic"]["gibbs_overshoot"] < 0.1

room = Scene("room", passes=None)
room.add(ac.Room(id="room", size=(8, 5, 3), material="curtain"))
room.add(
    ac.ImpulseResponse(id="ir", material="curtain", order=10, layer="overlay", pos=(0.1, 0.9, 0))
)
ir = room.engine.inspect(0.0, id="ir")["objects"][0]["semantic"]
assert abs(ir["first_arrival_ms"] - 4.0311 / 343 * 1000) < 1e-3 and ir["arrivals"] > 1000
```

**Simulations** (`phasor.core.simulation`): a `Simulation` has a fixed `dt`, `params()` (the
seed included), `initial()` and a pure `step(state, dt)`. `runner_for(sim).state_at(t)` is the
nearest earlier checkpoint plus the remaining steps, cached in memory and under
`.phasor_cache/<hash of the parameters>/` (or `PHASOR_CACHE`), so scrubbing is cheap and the
answer never depends on how you got there. `ac_math.RoomFDTD` is one (it refuses a `dt` above
`dx / (c sqrt(3))`: that is a validation error, not a blow-up).

## Domains, wave 2: pde, algebra, calc

`from phasor import pde, algebra, calc`.

**`pde`**: `Profile(kind="heat"|"wave"|"advection"|"schrodinger", time=...)` (a 1D solution
as a curve, the exact solution drawn beside it), `Field(kind="heat"|"wave"|"reaction")` (a 2D
colour plane, a backdrop) and `Surface(...)` (the same as a lit 3D surface). `time` is
simulation time, usually `expr("t / 10")`; frames are checkpointed simulations, so scrubbing is
cheap. Semantics report the max, `max_error_vs_exact`, the conserved quantity (`total_heat`,
`norm`). An unstable step is a validation error. The numerics are in `domains/pde_math.py`
(`Heat1D`, `Heat2D`, `Wave1D`, `Wave2D`, `Advection1D`, `GrayScott`, `Schrodinger1D`,
`solve_poisson`), all `Simulation`s.

**`algebra`**: `Polygon(n=4, rotation=deg, flip=0..1)` (a dihedral group acting on an n-gon;
the semantics name the group element when it lands on a symmetry), `CayleyTable(group="D4")`
(`D<n>`, `Z<n>`, `S<n>`; coloured by element, a Latin square), `ClockDiagram(n=12, step=4)`
(a cyclic group, a subgroup and its cosets). Groups are in `algebra_math` (`dihedral`, `cyclic`,
`symmetric`: subgroups, cosets, normality, Lagrange).

**`calc`**: `RiemannSum(function="x**2", a=0, b=3, n=8, rule="mid")` (`n` is a number, so
animate it and watch the error fall), `Taylor(function="sin(x)", order=5)` (animate `order`),
`Graph(function=...)` (breaks at poles), `VectorField(p="-y", q="x")` (divergence and curl in the
semantics). Functions are text in `x` (and `y`), read with a whitelist, never evaluated as Python.

```python
from phasor import Scene, algebra, calc, expr, pde

s9 = Scene("wave2", passes=None)
s9.set("camera.height", 10)
s9.add(pde.Profile(id="rod", kind="heat", length=2.0, diffusivity=0.5, modes=(1.0, 0.4),
                   time=expr("t / 10"), pos=(-8, 1, 0)))  # fmt: skip
s9.add(algebra.Polygon(id="square", n=4, pos=(2, 2, 0)))
s9.add(calc.RiemannSum(id="area", function="x**2", a=0.0, b=3.0, n=4.0, pos=(-2, -4, 0)))
s9.marker("go", 1.0)
s9.at("go").morph("square.rotation", 90.0)
s9.at("go").morph("area.n", 64.0)

e9 = s9.engine
rod = e9.inspect(3.0, id="rod")["objects"][0]["semantic"]
assert rod["max_error_vs_exact"] < 3e-3  # the solver sits on the exact solution
assert e9.inspect(3.0, id="square")["objects"][0]["semantic"]["element"] == "r"
assert (
    e9.inspect(0.0, id="area")["objects"][0]["semantic"]["error"]
    > 10 * (e9.inspect(3.0, id="area")["objects"][0]["semantic"]["error"])
)  # more pieces, smaller error
```

**More of each module.** `pde.Profile(kind="heat", boundary="neumann", method="spectral")`:
insulated ends (drawn as walls; held ends are drawn as pins) and a spectral solver that is
exact per mode at any step. `pde.Modes(modes=(1, 0, 0.5), count=7, time=...)`: separation of
variables, each mode drawn faintly and decaying at its own rate, their sum bright (semantics:
`amplitudes`, `decay_rates`). `pde.Slice(axis="z", at=0.5, time=...)`: one plane through a
heating cube (`Heat3D`), inside its outline; animate `at` to sweep (orbit camera for 3D).
`algebra.CayleyGraph(group="D4", generators="r s")`, `CayleyTable(subgroup="r", cosets=1)`
(rows in coset order, colored by coset; semantics give the quotient table when the subgroup is
normal), `Homomorphism(n=12, m=4, a=1)` (Z_n -> Z_m, kernel highlighted), `SubgroupLattice`
(Hasse diagram, normal subgroups highlighted), `GaloisDiagram(primes=(2, 3))` (fields of
Q(sqrt 2, sqrt 3) beside the subgroups that fix them), `Permutation(perm=(1, 2, 0), progress=...)`
(tokens travel; semantics: cycles, sign, order). `calc.Limit(function="sin(x)/x", a=0,
epsilon=...)` (the epsilon band and the delta it forces; `holds`), `calc.LoopIntegral(p=, q=,
measure="flux"|"circulation", progress=...)` (the running line integral around a circle, equal at
the end to the integral of the divergence or curl inside: `theorem_holds`).

```python
from phasor import Scene, algebra, calc, expr, pde

more = Scene("more", passes=None)
more.add(pde.Modes(id="modes", modes=(1.0, 0.0, 0.5), count=5, time=expr("t / 4")))
more.add(algebra.CayleyTable(id="cosets", group="D4", subgroup="r2", cosets=1.0, pos=(8, 0, 0)))
more.add(calc.LoopIntegral(id="green", p="-y", q="x", measure="circulation", pos=(0, 6, 0)))


def info(oid, t=0.0):
    return more.engine.inspect(t, id=oid)["objects"][0]["semantic"]


assert info("modes", 2.0)["amplitudes"][2] < info("modes", 0.0)["amplitudes"][2] / 2
assert info("cosets")["normal"] and info("cosets")["quotient"]["order"] == 4
assert info("green")["theorem_holds"]
```


**Heat maps in the prism look.** `pde.Field`, `pde.Slice`, `acoustics.PressureSlice` and
`fourier.Spectrogram` take `smooth` (bicubic between grid values) and `isolines` (that many
contour levels drawn as thin light), and a `glow` style role: above `none`, strong parts
shine and haze and quiet parts melt into the background. A wave glows both ways, warm for up
and cool for down (`positive` and `negative` roles). `preset="prism"` sets all three; the
`prism` looks use it. Plain pictures are unchanged.

The prism preset also colours by the spectrum: `hue_cycles` > 0 reads the `colormap` role
(`spectrum` in the prism looks, or the scene's `palette`) as a wheel, its hue turning that
many times over the value range, so each contour line has the hue of its level; on a
`Spectrogram` the hue runs up the frequency axis, so a sliding tone's trace runs through the
spectrum like a prism stroke. Brightness then comes from the field's strength alone: the
glow weight raised to `contrast` (above 1, only the strongest parts light up), with
`ambient` the share a quiet pixel keeps (0: the dark ground). `fade` melts the edges into the
background over that share of the shorter side, so there is no square border and no square
halo. `hue` turns the colours; a lively look drifts it a little. All default to the plain
picture; they are taste knobs for looks and presets, not numbers for a scene to type.

```python
from phasor import Scene
from phasor.domains import pde

hot = Scene("hot plate", look="prism", passes=None)
hot.add(pde.Field(id="plate", kind="heat", mode=(0, 0), time=0.1, scale=0.3))
assert hot.engine.document.object("plate").params["isolines"] == 8
plate = hot.engine.document.object("plate")
assert plate.params["hue_cycles"] > 0 and plate.params["fade"] > 0
assert plate.style["colormap"] == "spectrum"
```

The height mode, `pde.Surface`, has a prism preset too (the prism looks use it): with
`hue_cycles` > 0 the surface is dark where the field is quiet and takes the spectrum's hue of
its value where it is strong (`contrast` as above), under a net of `lines` grid lines each
way drawn as glowing strokes in the same hues (`stroke` and `glow` roles). Give the scene an
orbit camera (`examples/prism_drum_3d.py`).

```python
from phasor import Scene
from phasor.domains import pde

drum = Scene("drum", look="prism", passes=None)
drum.use_camera("orbit")
drum.add(pde.Surface(id="skin", kind="wave", mode=(2, 3), scale=1.2))
assert drum.engine.document.object("skin").params["lines"] > 0
```

**Point clouds.** `from phasor import cloud`. `cloud.PointCloud` splats many points (a
million is fine on the GPU) as tiny Gaussians of light that add up, so dense regions glow and
the structure at their edges shows. Sources: `roots` (the roots of every polynomial of
`degree` with coefficients from `coefficients`, the leading one 1; a random sample when
there are more than `count` roots), `attractor` (`kind` clifford, dejong or lorenz, fitted to
[-1, 1]^2), `buddhabrot` (escaping orbits up to `max_iter`) or `points` given directly. A point
sits at world `pos + scale * point`; the cloud is seen inside the `size` rectangle round `pos`
and only the part on screen is splatted, so zooming in keeps it sharp. `exposure` is light
per point against the mean density at the current zoom (resolution does not change it);
`color_by="phase"` colours by each point's angle round `pos`, `source` by what the source
says (the polynomial's place in the list). Presets `prism` (the `prism` looks use it) and
`plain`. Points are cached per source parameters, so animating `hue`, `exposure` or the
camera does not recompute them. Numerics: `phasor.domains.cloud_math`.

Motion: `twist` (roots) turns the last coefficient of the set round the origin by that share
of a turn (b -> b e^(2 pi i twist); the leading coefficient stays), so tweening it with
`.to("id.twist", 0.06)` makes every root glide and the fractal edge tear and morph; the same
polynomials are used at every value, so colours stay with them. A million roots solve in about
half a second (LAPACK on all cores) per new value. `twinkle` flickers each point's light by up
to that share (the roots of one polynomial together), slowly; it is life: `lively` gives the
full amount, `subtle` half, `off` or an `exact` tag none. `white` turns the colour of the
densest parts toward white (by `white * a^4`, a the tone curve's light), never the light
itself. The `prism` preset sets `white` 0.8 and `twinkle` 0.6.

```python
from phasor import Scene, cloud

sky = Scene("roots", look="prism", passes=None)
sky.add(cloud.PointCloud(id="roots", source="roots", degree=8, size=(4.5, 2.6)))
seen = sky.engine.inspect(0.0, id="roots")["objects"][0]["semantic"]
assert seen["points"] == 2**8 * 8 and seen["backdrop"]  # every +-1 polynomial of degree 8
assert sky.engine.document.object("roots").style["glow"] == "strong"
sky.at(0.5).to("roots.twist", 0.05, duration=3.0)  # the roots glide; the ring tears open
assert 0 < sky.engine.state_at(2.0)["roots"].params.twist < 0.05
```

**A 3D room with sound bouncing in it** (`ac`, BEAUTY.md step 7). Give the scene an orbit
camera.
- `ac.Room3D(size=(8, 5, 3), origin=, material=)`: a shoebox, or any closed room read from a
  Wavefront `.obj` file with `path="assets/l_room.obj"` (relative to the scene's folder, and
  inside it: an absolute path or `..` is refused; `scale` turns file units into meters). Drawn
  as faint glowing lines along its edges (for a mesh, its creases: `edge_angle`, so the
  diagonals of flat walls stay hidden) and faint glass walls (`walls`, 0 for lines only;
  `preset="outline"`). A file that is not there is lint `missing_file` and draws nothing.
  Semantics: volume, surface area, walls, Sabine and Eyring RT60.
- `ac.SoundRays(room="room", source=, time=expr("t - 1"), speed=0.01, count=1200,
  bounces=6, trail=1.5)`: rays leaving the source evenly in every direction, reflecting off
  the walls (angle in = angle out). `time` is scene seconds since the sound left, `speed`
  the slow motion (seconds of sound per scene second): each ray is drawn up to
  `343 * speed * time` meters along its path, as a streak `trail` meters long (0: the whole
  path). Brightness is the energy left, (1 - absorption) ** reflections; color comes from the
  `colormap` (the palette) by reflection count, `hue_step` apart. `seed` turns the ray set.
  Semantics: distance, reflections so far, energy left.
- `ac.Wavefront3D(room="room", source=, time=, speed=, order=2, shell=0.12)`: the expanding
  front drawn where it meets the walls, with the reflected fronts (image sources up to
  `order` reflections in a shoebox; one per wall, only where the sound really reaches, in
  any other room). `shell` draws the fronts themselves inside the room, faintly.
- The numerics are `ac_math`: `box_mesh`, `load_obj`, `feature_edges`, `mesh_stats`, `trace`
  and `rays` (specular ray tracing: points, path lengths, energy per bounce),
  `ray_segments` (what has been reached by a distance), `sphere_wall_arcs`, `front_visible`.

An object type that reads files lists the parameters that name them in `Files = ("path",)`
(`phasor.core.files`): validated to stay in the folder, checked by `missing_file`. A type
that reads some parameters as one value for the whole scene lists them in `Fixed` (as
`Image` and `Video` do `start`, `trim`, `speed`, `fps`): a track, an expression or an
updater on one is the validation error `bad_param`.

```python
from phasor import Scene, ac, expr

room = Scene("clap", look="prism", passes=None)
room.use_camera("orbit", target=(4, 2.5, 1.5), distance=15)
room.add(ac.Room3D(id="room", size=(8, 5, 3), material="wood"))
clap = dict(room="room", source=(2, 2, 1.5), time=expr("t - 1"), speed=0.01)
room.add(ac.SoundRays(id="rays", count=500, bounces=8, **clap))
room.add(ac.Wavefront3D(id="front", **clap))
rays = room.engine.inspect(3.0, id="rays")["objects"][0]["semantic"]
assert abs(rays["distance"] - 6.86) < 1e-6 and rays["reflections_so_far"] > 1
assert room.engine.inspect(0.0, id="room")["objects"][0]["semantic"]["volume"] == 120.0
```

## Graphs and networks (`nw`)

`nw.Network` draws a graph: nodes joined by edges, thousands of them in one draw call. Give
one source: `path` (a file next to the scene: an edge list `.csv`/`.tsv`/`.txt` with
`a,b[,weight]` per line, node-link `.json` whose extra node fields become attributes, or
`.graphml`), `edges` (pairs of node names, with optional `nodes` for their order), or a
`generator`: `tree` (`branching`, `depth`), `grid` (`rows`, `cols`), `random` (`n`, `p`),
`small_world` (`n`, `k`, `p`), `hypercube` (`cube_dim`), `tictactoe` (every position of the
game: 765 up to rotation and reflection, or all 5,478 with `symmetric=False`; node
attributes `move` and `outcome` under perfect play).

`layout`: `spring` (force-directed), `radial` (the root in the middle, each move a ring
further out, every node a wedge of angle in proportion to what grows from it: branches fan
out as rays; game trees), `layers` (breadth first from `root`, a level per move, ordered by
their parents), `circle`, `random`;
`dims` 2 or 3 (3D wants an orbit camera); `size` (world half-width), `pos`. `color_by`:
`depth`, `degree`, or a node attribute; `hue` turns the palette. `grow` (0-1) shows that share
of the edges, nearest the root first (morph it to grow the graph); `node_size` draws dots.
`exposure` is the light per edge: a dense graph shares its light out so it builds up where
edges crowd instead of burning out. `cross_edges` (0-1) dims the edges off the breadth-first
spanning tree (in a game: the same position reached by another order of moves); 0 hides them. Styles: `colormap` (spectrum), `edge`, `node`, `glow`,
`stroke`. Presets `plain`, `prism` (the prism looks use it). A missing file is
`missing_file`; an AI may only load files inside its folder.

```python
from phasor import Scene
from phasor.domains import network as nw

g = Scene("graphs", passes=None)
g.add(nw.Network(id="tree", generator="tree", branching=2, depth=5, layout="layers"))
g.add(
    nw.Network(
        id="ring",
        edges=(("a", "b"), ("b", "c"), ("c", "a")),
        layout="circle",
        pos=(5, 0, 0),
        size=1.0,
    )
)
sem = g.engine.inspect(0, id="tree")["objects"][0]["semantic"]
assert (sem["nodes"], sem["edges"], sem["depth"]) == (63, 62, 5)
```

## Paths, fills and shapes (`shapes`)

`from phasor import shapes`. Every outline is a **`shapes.Path`**: subpaths of cubic Béziers
(each subpath is an anchor, then handle, handle, anchor per cubic; closed when it ends where
it began), drawn with a **fill** and a **stroke**. Curves are flattened by their size on
screen, so they stay smooth at any zoom. The shapes below are paths with their own
parameters; they all lie in the plane z = `pos.z` (the screen in 2D). Angles in degrees,
counterclockwise.

| Type | Parameters | Semantics (`kind`, besides `length`, `area`, `closed`, `subpaths`, `rule`, `fill`, `dash`) |
| --- | --- | --- |
| `shapes.Path` | `subpaths`, `rule` (`nonzero`/`evenodd`), `tips` (`none`/`start`/`end`/`both`: arrow tips on the open ends); `Path.through(id, points, closed=False)` builds one of straight segments | `path` |
| `shapes.Circle` | `pos`, `r` | `circle`: `center`, `radius` |
| `shapes.Ellipse` | `pos`, `width`, `height`, `rotation` | `ellipse` |
| `shapes.Arc` | `pos`, `r`, `start`, `angle` (-360 to 360, not 0; negative: clockwise), `tips` | `arc` (open, even a full turn: no area unless filled) |
| `shapes.Sector` | `pos`, `r`, `start`, `angle`, `inner` (> 0: a slice of a ring) | `sector` |
| `shapes.Annulus` | `pos`, `outer`, `inner` | `annulus` (a fill with a hole) |
| `shapes.Polygon` (type `polygon_path`) | `vertices` (3 or more, in order) | `polygon`; feature `vertices` |
| `shapes.RegularPolygon` | `n`, `pos`, `r` (circumradius), `rotation` (first vertex at the top) | `polygon`: `side`; feature `vertices` |
| `shapes.Rectangle` | `pos`, `width`, `height`, `corner_radius`, `rotation` | `rectangle`; feature `vertices` |
| `shapes.Brace` | `start`, `end`, `depth` (left to right it points down) | `brace`: `tip`; feature `tip` (put a label there) |
| `shapes.SVG` | `path` (an .svg file in the scene's folder), `pos` (middle of its view box), `height` | `svg`: `parts` |

Every path and shape also has `progress` (the drawn share of the stroke: `trace` draws it
in and fades its fill in with it), `fill_opacity` and `stroke_opacity` (0-1, times the
role's; `stroke_opacity=0` is a fill with no outline). Style keys: `color` (emphasis,
primary), `stroke` (stroke, regular; `Brace`: bold), `glow` (glow, none), `fill` (fill,
none; `SVG`: solid), `fill_color` (emphasis, primary), `dash` (dash, solid). Presets
`outline`, `filled`, `solid`, `glowing`, `dashed`. Every number of a path (a size, a
position, a point) is within ±1e9 (`MAX_COORD`); beyond that is a validation error when it
is set.

- **Fills** are plain paint: they never glow, and a color below the theme's bloom knee comes
  out exactly as written. `rule="evenodd"`: every nested subpath flips inside and outside;
  `nonzero`: a subpath that runs the other way is a hole. A filled path whose edges cross
  is lint `fill_crossing` (split it, or use a boolean `union`).
- **Dashes** are cut by arc length from each subpath's start, so they stay where they are
  while a path is traced or grows. `dotted` draws round dots. A path draws at most
  `MAX_DASHES` (2000) dashes or dots: a longer one gets longer dashes (lint `dash_limit`).
  A theme's `[dash]` periods are 0 (solid) or at least 0.002 frame heights.
- **Boolean operations**: `shapes.union(a, b, id=...)`, `intersection`, `difference`,
  `exclusion` take two shapes or paths (not yet in a scene, or already added) and return a
  new `Path` of straight segments, measured once when called; it takes `a`'s style unless
  given one. They need the `boolean` extra (shapely): without it they raise an error that
  says how to install it. Input too degenerate for GEOS (a ring with no area beside a tiny
  one) is snapped and tried again; failing that, a `ValueError` names the operation.
- **`SVG`**: each shape of the file becomes a part `<id>.<name>`, named from its `id`
  attribute (lower case, other characters `_`, at most 40 characters; one without an id by
  its kind, `circle`, `circle_2`, in file order, so give shapes ids to keep constraints on
  them). A name the document already uses or retired under the `SVG` (a `logo.label`) gets
  the next free suffix (`label_2`). Filled shapes are filled with `fill_color`, stroked
  ones stroked; the file's own colors are not used. Text, pictures and external references
  are ignored. The file is read again when it changes (a `path` on a track: every file it
  names). A bad or hostile file never makes the scene fail: missing (or not a regular
  file) is lint `missing_file`, not valid SVG is `bad_file`, past a limit (8 MB, 50000
  elements once `<use>` is expanded, nesting 100 deep, 10000 shapes, 200000 segments;
  constants in `path_math`) is `svg_too_complex`, and without the `svg` extra
  (svgelements) it is `missing_extra`; it then draws nothing (or what fits the limits).
- A render worker never re-reads a changed file: one that changes while a scene renders
  stops the render with an error ("render it again"), so no video mixes old and new frames.
- A `Tracer` can travel along any path or shape.

```python
import math

from phasor import Scene, Text, shapes
from phasor.domains import path_math as pm

s = Scene("shapes_demo", passes=None)
s.add(shapes.Circle(id="disc", r=1.2, style={"fill": "soft", "glow": "soft"}))
s.add(
    shapes.Rectangle(
        id="card", pos=(3.5, 0), width=2.5, height=1.5, corner_radius=0.3, style={"dash": "dashed"}
    )
)
s.add(shapes.Arc(id="swing", pos=(-3.5, 0), r=1.0, start=0, angle=200, tips="end"))
s.add(shapes.Brace(id="span", start=(-1.2, -1.5), end=(1.2, -1.5)))
s.add(Text(id="span.label", text="2.4"))
s.constrain("label(span) below span")
# path_math builds subpaths (circle, arc, rectangle, sector, brace, polyline, ...)
outer, inner = pm.circle((0, 2.6, 0), 0.6), pm.circle((0, 2.6, 0), 0.3)
s.add(
    shapes.Path(
        id="ring",
        rule="evenodd",
        style={"fill": "solid"},
        subpaths=[outer.tolist(), inner.tolist()],
    )
)
s.at(0.2).trace("disc", "swing")
e = s.engine
seen = {o["id"]: o["semantic"] for o in e.inspect(2.0)["objects"]}
assert seen["disc"]["kind"] == "circle" and abs(seen["disc"]["area"] - math.pi * 1.44) < 1e-3
assert abs(seen["ring"]["area"] - math.pi * (0.36 - 0.09)) < 1e-3  # even-odd: a hole
assert seen["swing"]["tips"] == "end" and "area" not in seen["swing"]
assert e.lint() == []
```

```python
from phasor import Scene, shapes

b = Scene("booleans", passes=None)
left = shapes.Circle(id="left", pos=(-0.5, 0), style={"fill": "soft"})
right = shapes.Circle(id="right", pos=(0.5, 0))
b.add(shapes.intersection(left, right, id="lens"))  # needs the boolean extra (shapely)
b.add(shapes.exclusion(left, right, id="moon"))
seen = {o["id"]: o["semantic"] for o in b.engine.inspect(0)["objects"]}
assert seen["lens"]["area"] < 1.3 < seen["moon"]["area"]
```

```python no-run
# needs a file: an .svg next to the scene (examples/vector_toolkit.py runs this)
s.add(
    shapes.SVG(
        id="emblem",
        path="assets/emblem.svg",
        pos=(0, -2),
        height=3,
        style={"fill": "soft", "fill_color": "highlight"},
    )
)
# parts: emblem.ring, emblem.star, emblem.badge_1, emblem.wave
```

## Diagrams: number lines, planes, charts, tables, matrices, code (`dg`)

`from phasor import dg`. Each is a composite that draws its own furniture (lines, ticks,
grid, rules, brackets) and builds everything a reader reads as a **part** the inspector,
layout and lint see by itself. Parts are placed by their composite (they take no
constraints: put a `label(x) near x` on a document object instead); numbers along one axis
are judged together by lint `crowded_ticks`, not pair by pair. Sizes and colors come from
roles; explicit values are allowed (`off_palette` / `off_scale` warn).

| Type | Parameters | Parts | Semantics `kind` |
| --- | --- | --- | --- |
| `dg.NumberLine` | `x_range` (min, max, step; step 0: a nice 1-2-5 step), `pos` (middle), `scale` (world units per unit), `angle` (degrees), `tick`, `numbers`, `digits` (-1: as the step needs), `tips` (`none`/`end`/`both`), `label` | `n_2`, `n_m1`, `n_0p5` (a number per tick) | `number_line`; features `start`, `end`, `zero` |
| `dg.NumberPlane` | `x_range`, `y_range`, `pos` (where 0, 0 is), `scale`, `x_scale`, `y_scale` (0: `scale`), `faint` (lighter lines between, 0-4), `numbers`, `digits`, `labels` | `axes` (a `geo.Axes`), whose numbers are `axes.x_2`, `axes.y_m1` | `number_plane`, a backdrop (labels may sit on it) |
| `dg.BarChart` | `values` (1-40), `names`, `colors` (emphasis roles per bar, cycled), `pos` (middle), `width`, `height`, `y_range` ((0, 0, 0): fit the values), `gap`, `grow` (0-1: animate it), `numbers`, `show_values`, `digits` | a `Rectangle` per bar named after its name (`bees`; `bar_1` without names), `bees_name`, `bees_value`, `y_2` | `bar_chart`: `bars`, `largest`, `y_range` |
| `dg.Table` | `cells` (rows of text; maths in `$...$`) **or** `path` (a `.csv` in the scene's folder), `pos`, `header`, `lines` (`none`/`header`/`grid`), `align`, `reveal` (0-1: rows shown) | a `Text` per cell named like a spreadsheet's: `a1`, `b2` | `table`: `rows`, `columns`, `header` |
| `dg.MatrixDisplay` | `entries` (rows of maths: `"a_{11}"`, `"\\frac{1}{2}"`), `bracket` (`square`/`round`/`bar`/`none`), `pos` | `e1_2` (row 1, column 2) | `matrix_display` |
| `dg.Code` | `code` **or** `path` (a source file in the scene's folder), `language` (a pygments name), `pos`, `line_numbers`, `reveal` (0-1: lines shown) | a `CodeLine` per line, `line_3` (blank lines have none unless numbered) | `code`: `lines`, `highlight` (`pygments`, `plain`, `unknown`) |
| `dg.CodeLine` | `text`, `classes` (a letter per character: `p` plain, `k` keyword, `f` name, `s` string, `n` number, `c` comment, `g` line number), `pos` (left end) | | `label` |

- **Axes with numbers and unequal scales** are `geo.Axes(numbers=True, x_scale=..., y_scale=...)`
  (below). A **number that animates** is `Counter` (`group=True` writes 1,234,567, `sign=True`
  a `+`); point its `value` at a `Value` you morph. A live 2x2 or 3x3 matrix of numbers that
  follows another object is `linalg.Matrix`; `MatrixDisplay` is any matrix of maths.
- **Code is monospace by placement**: each character is centered in a cell 0.52 text
  heights wide (wide letters are set a little smaller on the same baseline), in phasor's text
  font. Highlighting needs pygments, the `code` extra (`uv sync --extra code`); without it the
  code is drawn plain and lint says `missing_extra`. A language pygments does not know is
  drawn plain with `unknown_language`. Style keys: `color` (plain), `keyword`, `name`,
  `string`, `number`, `comment`, `size`.
- **Bounds** (`core/ticks.py`, `domains/diagrams_math.py`): a range of a new diagram type
  is finite, within ±1e9, `min < max`, and makes at most `MAX_TICKS` (100) ticks
  (`geo.Axes`, older, only finite and `min < max`: past 100 ticks it is drawn at a wider step,
  lint `clamped`); tables at most 40 rows, 12 columns,
  300 cells of 60 characters; matrices 10 by 10; charts 40 bars; code 120 lines of 120
  characters (12000 in all, no control characters). Past a bound is a validation error when
  it is set. A file read for a table or code is cut to the bounds instead (lint
  `truncated`); one that is not UTF-8 (or not CSV) is `bad_file`; a missing one (or a
  folder, a FIFO) is `missing_file`, and the object draws nothing. A file changed on disk
  is read again.
- Numbers at ticks are written as maths (a true minus sign) in the `numbers` text role and
  named by value: `x_m2` is -2, `x_2p5` is 2.5. Zero is left out where axes cross; a number
  line writes it (`n_0`).

```python
from phasor import Counter, Scene, Value, dg, expr, geo

s = Scene("kit", passes=None)
s.set("camera.height", 12)
s.add(dg.NumberLine(id="line", x_range=(-3, 3, 1), pos=(0, -4.6, 0), tips="end"))
s.add(
    dg.NumberPlane(id="plane", x_range=(-3, 3, 1), y_range=(-2, 2, 1), pos=(-7, 2, 0), y_scale=0.8)
)
s.add(
    geo.Axes(
        id="plot",
        dims=2,
        x_range=(0, 10, 2),
        y_range=(0, 1, 0.5),
        x_scale=0.4,
        y_scale=3,
        pos=(-1.5, 0.5, 0),
        numbers=True,
        labels=("t", "p"),
    )
)
s.add(
    dg.BarChart(
        id="sales",
        values=(3, 5, 2),
        names=("ants", "bees", "cats"),
        pos=(6, 2, 0),
        width=4,
        height=3,
    )
)
s.add(dg.Table(id="t", cells=(("x", "$x^2$"), ("2", "4"), ("3", "9")), pos=(-6, -2.5, 0)))
s.add(dg.MatrixDisplay(id="m", entries=(("1", "2"), ("3", "4")), pos=(-1.5, -2.5, 0)))
s.add(dg.Code(id="code", code="def f(x):\n    return x * x", pos=(5, -2.5, 0)))
s.add(Value(id="n", value=0.0))
s.add(Counter(id="count", value=expr("n.value"), digits=0, group=True, pos=(5, -4.2, 0)))
s.marker("go", 0.5)
s.marker("done", 2.0)
s.at("go").morph("n.value", 25000.0, until="done")
e = s.engine
seen = {o["id"]: o["semantic"] for o in e.inspect(2.0)["objects"]}
assert seen["plot.y_0p5"]["text"] == "$0.5$" and seen["plot"]["axes"]["y"]["scale"] == 3
assert seen["sales"]["largest"] == "bees" and "sales.bees_name" in seen
assert seen["t.b1"]["text"] == "$x^2$" and seen["m.e2_1"]["text"] == "$3$"
assert seen["code.line_2"]["text"] == "return x * x" and seen["count"]["text"] == "$25{,}000$"
assert "plane.axes.x_m3" in seen and seen["plane"]["backdrop"]
assert e.lint() == []
```

A crowded axis is one issue with the step that clears it:

```python
from phasor import Scene, geo

s = Scene("crowded", passes=None)
s.add(geo.Axes(id="ax", dims=2, x_range=(0, 1000, 25), x_scale=0.006, numbers=True))
(issue,) = [i for i in s.engine.lint() if i.code == "crowded_ticks"]
assert issue.path == "ax.x" and "step of at least" in issue.fix_hint
assert not [i for i in s.engine.lint() if i.code == "overlap"]  # not pair by pair
```

### Text as shapes and morphing

**Glyph outlines.** `Text(..., as_shapes=True)` and `Equation(..., as_shapes=True)` draw
their glyphs as outlines (matplotlib's font code: plain text in the picture's font, Aileron;
maths through mathtext in Computer Modern), one `Path` part per glyph named by what it shows:
`word.cap_h`, `word.e`, `word.l`, `word.l_2`, `eq.plus`, `eq.pi`, `eq.bar` (a fraction
bar), clear of the document's ids. They lie in the box the picture would fill. Parameters:
`fill_opacity` (default 1) and `stroke_opacity` (the outline once written, default 0); the outline
is `basic.OUTLINE` (0.04) of the line height wide, so it scales with the text; an
equation's `colors` color each part's glyphs. `write` draws each glyph's outline, then fills
it, glyph after glyph (`basic.WRITE_SPAN` of the write each). A text that is not drawn as
shapes draws as before. At most `glyph_math.MAX_GLYPHS` (400) glyphs (all parts of an
equation together) are drawn, and none past `path.MAX_COORD` world units (lint `glyph_limit`).
With the `latex` math engine the picture comes from TeX but the outlines from mathtext: a text
mathtext cannot read has no outline (a morph from it is `morph_missing`, a morph into it
draws its picture; as shapes it is refused when added).

**`transform` between any two outlines.** Paths, every shape, text and equations can be
morphed into (`morph_from`, `morph`, `morph_kind` parameters, set by the verbs); paths,
shapes, text, equations and SVGs (their parts) can be morphed from. The target draws the
in-between every frame from the source as the state has it at that time (a moving source is
followed; nothing is baked into the document): subpaths are put in a canonical order, the
side with fewer is split (its longest piece at half its length) until the counts agree,
pieces pair by least travel, each pair is resampled by arc length and aligned (two rings keep
the target's winding, so holes stay holes, and turn to the start that moves least; an open
piece runs the shorter way). Colors, widths and fill opacity mix in linear light; the fill
crossfades from the source's layout to the target's, so both ends are exact. Identical
outlines are an identity at every step. Dashes and arrow tips are not drawn in between.
Bounds (`morph_math`): `MAX_SUBPATHS` 512 a side (the shortest beyond fade; lint
`morph_limit`), `MAX_POINTS` 512 points a piece, `MAX_TOTAL` 40 000 in all, `MAX_ASSIGN` 256
pairs by optimal assignment (more pair in canonical order), `SAMPLE_CUBICS` 2048 cubics a
subpath is sampled through; the in-between's fill triangulates at most `morph.MAX_FILL_RING`
(4096) points a ring (a longer ring keeps every k-th point, also at the first frame when one
subpath is split into many pieces; strokes keep all). Plans are
cached by both outlines in the world (pairs are by least travel, so a source that moves or
a camera that zooms makes a plan a frame), in a pool of `morph.PLAN_BYTES` (64 MB) measured
by their arrays. A source that is gone is lint `morph_missing` (it draws itself); a source
whose id is longer than `core.ids.MAX_REF` (200) characters is refused by the verbs.
One object can become itself several times (`morph_from` switches at each verb). The
`morph` verb also animates style keys (see "Style transitions").

**`transform_matching`** pairs pieces: two equations' parts by key, then by equal TeX in
reading order (a part moved across the `=` still pairs); an equal part glides glyph by glyph,
a keyed part whose text changes morphs its outline; anything else pairs glyphs (same font
glyph) or paths (`morph_math.signature`: the same outline at any place and size) in reading
order. Pairs carried past others arc round them (`[motion] transform_arc`, on equations); the
source's other pieces fade out in the first half, the target's fade in in the second. Two
equations drawn `as_shapes` use it for `transform` too (their changed parts morph instead of
crossfading). Inspector semantics while morphing: `morphing_from`, `morph`, `morph_kind`.

```python
from phasor import Equation, Scene, Text, shapes

m = Scene("text_shapes_demo", passes=None, look="none")
m.set("camera.height", 8)
m.marker("write", 0.2)
m.marker("morph", 2.0)
m.add(Text(id="word", text="circle", pos=(-3, 1), as_shapes=True, style={"size": "title"}))
m.add(shapes.Circle(id="disc", pos=(3, 1), r=1.0, style={"fill": "soft"}))
m.add(Equation(id="one", parts=("$a$", "$+$", "$b$", "$=$", "$c$"), pos=(0, -2), as_shapes=True))
m.add(Equation(id="two", parts=("$a$", "$=$", "$c$", "$-$", "$b$"), pos=(0, -2), as_shapes=True))
m.at("write").write("word", "one")
m.at("morph").transform("word", "disc").transform_matching("one", "two")
st = m.engine.state_at(2.3)
assert [o.id for o in m.engine.state_at(1.9).objects if o.parent == "word"][:2] == [
    "word.c",
    "word.i",
]
assert st["disc"].impl.semantics(st)["morphing_from"] == "word"
assert st["two"].params.morph_kind == "matching"
assert m.engine.lint(strict=True) == []
```

## Space: axes, curves, surfaces and fields (`geo`)

`from phasor import geo`. Give the scene an orbit camera (`s.use_camera("orbit")`) to turn
around them: the `orbit` verb, or drag in the editor. Formulas are text in `t` (curves) or `u`,
`v` (surfaces) or `x`, `y`, `z` (fields), read by the whitelist parser, never run as Python.

- `geo.Axes(dims=3, x_range=(-3, 3, 1), z_range=..., labels=("x", "y", "z"))`: axes with ticks
  (ranges are (min, max, tick step), step 0 a nice one; `scale` is world units per unit, and
  `x_scale`, `y_scale`, `z_scale` (0: `scale`) make them unequal). `numbers=True` writes the
  value at every tick but zero as parts (`axes.x_2`, `axes.y_m1`; the `numbers` text role,
  `digits` -1 as the step needs). An axis that points straight at a flat camera (the z axis
  of default `dims=3` axes in a 2D scene) gets no numbers. A range is drawn with at most 100 ticks: a step that would
  make more is drawn at the next nice step that makes fewer, and lint `clamped` says so.
  Axes predate these limits, so a saved document past them still loads: more labels than
  axes, or a label that cannot be typeset, is left out (`clamped`); labels and ranges have
  no other limit.
- `geo.ParametricCurve(fx="cos(t)", fy="sin(t)", fz="0.2*t", t_range=(0, 37.7),
  color_by="t")`: a curve; `color_by` is `none`, `t`, `speed`, `x`, `y` or `z` (through the
  `colormap`); `progress` traces it. Semantics: length, closed, color range.
- `geo.ParametricSurface(fx=..., fy=..., fz=..., u_range, v_range, res, color_by="height")`:
  a surface; `color_by` is `height`, `u`, `v`, `radius`, or `expr` with `color_expr` (a formula
  in `u v x y z`). **Mapping input to output:** `with_map=True` draws the flat (u, v) input
  rectangle at `domain_pos` as a faint grid and `morph` (0 to 1, animate it) carries the
  surface from that flat input to its true shape. Semantics give the `area` (a half sphere of
  radius r reports 2 pi r^2).
- `geo.VectorField3D(fp="-y", fq="x", fr="0", extent=(4, 4, 4), counts=(5, 5, 3))`: arrows
  colored by size; semantics give divergence and curl. `calc.VectorField(color_by_size=True)`
  does the same in a plane. `spacing=0.5` (3D) places arrows that far apart instead of
  `counts`.
- **How field arrows look** (both types): `preset="dense"` is many small thin arrows with
  heads in proportion, lengths eased (`length_by="soft"`: tanh of size, so weak arrows still
  show and strong ones never touch) and color saying how strong (the `magnitude` colormap,
  cool to warm; exempt from `color_restraint`, since the hue is read as a quantity).
  `preset="bold"` is few thick glowing arrows. Without a preset: lengths in proportion to
  size (`length_by="size"`). A dense 3D *box* gets crowded: use a plane (extent z = 0) or a
  larger `spacing`.

```python
from phasor import Scene, calc, geo

s = Scene("fields", passes=None)
s.add(calc.VectorField(id="swirl", p="sin(y)", q="-sin(x)", extent=(12.0, 7.0), preset="dense"))
seen = s.engine.inspect(0.0, id="swirl")["objects"][0]
assert seen["semantic"]["arrows"] > 800  # many small arrows
assert s.engine.document.object("swirl").style["colormap"] == "magnitude"
s.add(geo.VectorField3D(id="plane", fp="-y", fq="x", extent=(6.0, 6.0, 0.0), preset="dense"))
assert s.engine.inspect(0.0, id="plane")["objects"][0]["semantic"]["arrows"] == 14 * 14
```
- `geo.FlowLines(fp=..., fq=..., fr=..., seeds=(5, 5, 1), length=3.0, progress=...)`: where a
  particle dropped at each seed would travel, colored by the field's size along the line.

```python
import math

from phasor import Scene, expr, geo

s = Scene("space", passes=None)
s.use_camera("orbit", azimuth=40.0, elevation=25.0, distance=14.0, drift=0.0)
s.add(geo.Axes(id="axes", labels=("x", "y", "z")))
s.add(
    geo.ParametricCurve(
        id="helix", fx="cos(t)", fy="sin(t)", fz="0.2*t", t_range=(0.0, 6 * math.pi), color_by="t"
    )
)
s.add(
    geo.ParametricSurface(
        id="ball",
        fx="2*sin(u)*cos(v)",
        fy="2*sin(u)*sin(v)",
        fz="2*cos(u)",
        u_range=(0.0, math.pi / 2),
        v_range=(0.0, 2 * math.pi),
        color_by="height",
        with_map=True,
        morph=expr("t / 4"),
        pos=(4, 0, 0),
        domain_pos=(-4, 0, 0),
    )
)
e = s.engine
assert abs(e.inspect(4.0, id="ball")["objects"][0]["semantic"]["area"] - 2 * math.pi * 4) < 0.1
assert e.inspect(0, id="axes")["objects"][0]["semantic"]["axes"]["x"]["ticks"] == 6
```

## Interactive scenes: sliders, the notebook, a web page

A scene marks numbers a viewer may change: `scene.control("set.c", component=0, min=-1.2,
max=1.2, step=0.05, label="Re c")` (a vector parameter needs its `component`; a parameter
animated by a track cannot be a slider; the document keeps the controls, so they travel with
the scene and show in `diff`). Everything interactive builds on `phasor.interactive.Player`,
which moves sliders on a private copy of the scene (the scene itself never changes):

- the **editor**'s Sliders tab previews them in the viewport ("Apply to scene" makes the
  values part of the scene as one undo step); hovering shows f(z), |f| and arg f over a
  domain coloring, and the escape count over a Julia or Mandelbrot set;
- **Jupyter**: `from phasor.interactive import notebook; notebook(scene.engine)` returns
  widgets (needs `uv sync --extra notebook`);
- **a web page**: `phasor web scene.py -o page.html --steps 7` or `export_web(engine, path)`
  bakes a picture for every combination of slider positions into one self-contained HTML file
  (no server, no network; dragging on the picture moves the first two sliders). It refuses
  more than 600 pictures: use fewer `steps`.

`examples/julia_explorer.py` is the exit demo: a Julia set with sliders for Re c, Im c and the
iteration depth.

```python
from phasor import Scene, cx
from phasor.interactive import Player, export_web

s = Scene("explorer", passes=None)
s.add(cx.Fractal(id="set", kind="julia", c=(-0.8, 0.156), max_iter=60))
s.control("set.c", component=0, min=-1.2, max=1.2, step=0.05, label="Re c")
s.control("set.c", component=1, min=-1.2, max=1.2, step=0.05, label="Im c")

player = Player(s.engine)
assert player.set("set.c[0]", 0.33) == 0.35  # clamped to the range, rounded to the step
assert player.state_at(0.0)["set"].params.c[0] == 0.35
assert s.engine.document.object("set").params["c"] == (-0.8, 0.156)  # the scene is untouched
```

## Manim import

`phasor manim lecture.py -o lecture.phasor.json` (or opening the `.py` in `phasor gui`, or
`scene_open` on a server that allows scripts) runs a Manim Community script *unchanged*
against a stand-in for the `manim` module and records the scene it builds: `self.add` and
`self.play` become objects and transitions at Manim's own times (`run_time`, `wait`), in
Manim's frame (8 units tall, y up). The result is a plain phasor scene you can edit, lint and
render. It runs your code: only import scripts you trust. The stand-in is checked against
real Manim CE 0.22.1 (`tests/manim_real`): its clock, rate functions, default sizes and
colors are Manim's own.

| Carries over | How |
| --- | --- |
| `Scene`, `ThreeDScene` (`set_camera_orientation`, ambient rotation) | `construct` runs; 3D uses the orbit camera |
| `MovingCameraScene`: `self.camera.frame` (`.animate.move_to`, `shift`, `scale`, `set(width=)`, `save_state` / `Restore`) | tracks on `camera.center` and `camera.height` (the frame's width is its height x 16/9) |
| `Circle`, `Ellipse`, `Arc`, `ArcBetweenPoints`, `Sector`, `AnnularSector`, `Annulus`, `Polygon`, `RegularPolygon`, `Triangle`, `Rectangle`, `Square`, `RoundedRectangle`, `Brace` (`get_text`, `get_tex`, `get_tip`) | phasor's `shapes` (`circle`, `rectangle`, ...) with their fills (`fill_opacity`, `fill_color`, `set_fill`; `stroke_width=0` is `stroke_opacity` 0); stretched out of its kind (`stretch`, `apply_matrix`, a 3D turn) a shape is a `shapes.Path` |
| `Line`, `DashedLine`, `DashedVMobject`, `VMobject.set_points_as_corners`, `ParametricFunction`, `axes.plot` | a `shapes.Path` (dashed: the `dash` style `dashed`); `Arrow`, `Vector`: `arrow` (Manim's `buff` of 0.25 at both ends kept); `Dot`: `dot` |
| `SVGMobject("logo.svg")` | `shapes.SVG` (a file inside the scene's folder; one outside it is left out, with a warning) |
| `Text`, `Tex`, `MathTex` | `text` (`Tex` is text mode with `$...$` maths, `MathTex` maths); `MathTex("a", "+", "b")` is an `equation` with a part each (`eq[1].set_color` colors that part by the nearest role); a `font_size` the script sets is an explicit size (48 is the `math` role's height), and `scale` scales it (a negative factor is a flip) |
| `Axes` (`c2p`, `p2c`, `i2gp`, `get_axis_labels`, `add_coordinates`, `include_numbers`), `NumberPlane`, `ComplexPlane`, `NumberLine` (`n2p`), `BarChart`, `Table`, `MathTable`, `Matrix`, `Code` (`code_string` or a `code_file` in the scene's folder), `DecimalNumber`, `Integer` | `geo.Axes` (unequal scales; labels and numbers as its own), `dg.NumberPlane` (with its grid), `dg.NumberLine`, `dg.BarChart`, `dg.Table`, `dg.MatrixDisplay`, `dg.Code`, `Counter`; a step that makes more than 100 ticks is widened to a nice one, with a warning |
| `VGroup` and `shift`, `move_to`, `next_to`, `to_edge`, `to_corner`, `scale`, `rotate`, `stretch`, `set_width`, `align_to`, `arrange`, `arrange_in_grid` | positions are computed as Manim would |
| `Create`, `Uncreate`, `Write`, `Unwrite`, `DrawBorderThenFill`, `FadeIn`/`FadeOut` (`shift`, `scale`), `GrowFromCenter`, `GrowFromPoint`, `GrowArrow`, `ShrinkToCenter`, `Rotate`, `Indicate`, `Circumscribe`, `Flash`, `Wiggle`, `MoveToTarget`, `Restore`, `ApplyMethod`, `AnimationGroup`, `LaggedStart`, `Succession`, `.animate` (place, size and color: `set_color`, `set_fill`, `set_opacity`) | tracks with Manim's own run times and rate functions: `smooth` is the `sigmoid` easing, `linear`, `smoothstep` ... exactly; any other rate function (`there_and_back`, `double_smooth`, your own) becomes straight pieces within 0.2% of the motion (at most 64, Douglas-Peucker). Colors move as style tracks (mixed in linear light) |
| `Transform`, `ReplacementTransform`, `TransformMatchingTex`, `TransformMatchingShapes`, `FadeTransform` | the same kind moves its parameters; different outlines (shapes, paths, text, equations) morph (`transform`, `transform_matching`; text becomes `as_shapes`), played in Manim's run time; the rest crossfade |
| `ValueTracker` (`get_value`, `set_value`, `.animate.set_value`) | a `Value` with a track |
| updaters (`add_updater`, with or without `dt`), `always_redraw`, `always`, `f_always` | run by the stand-in at Manim's frames (`config.frame_rate`: 30 a second, Manim's `-qm`, unless the script sets 1 to 120; `dt` 1/30, the first frame of a play with `dt` 0, as Manim does) with every animated mobject where Manim has it; what they change becomes tracks (keys a straight line between does not reproduce within 1e-4 are kept). A baked record: the document never runs the script again, so changing the `Value` later does not move what the updater drove. The work is bounded per import: at most 20000 updater frames, 2 million updater calls and 1 million sampled numbers (a document of at most about 11 MB of keys). Past half of any of them, updaters run at fewer frames a second (down to 2), each at one of Manim's frame times and with `dt` adding up to Manim's; when they are spent, updaters stop and the warning says at what time (what they moved stays where it was). Measured: 200 half-second plays driving an `always_redraw` plot import in 9 s as an 11 MB document (updaters stop at 69 s); 2000 dots with updaters over `wait(30)` in 8 s |
| `self.wait_until(condition, max_time)`, `self.wait(t, stop_condition=)` | the condition is checked after each frame of the wait, with Manim's clock one frame on (as Manim CE 0.22.1 does), and the wait ends at the first frame where it is true; a condition that fails is a `ManimImportError` |
| `self.add_sound("click.wav", time_offset, gain)` | a `media.Sound` (a WAV file inside the scene's folder) mixed in when the scene renders; a start before 0 or a gain outside -60 to 24 dB is clamped, with a warning |
| Colors the script sets (`RED`, `"#58C4DD"`), `stroke_width` / `set_stroke(width=)`, `camera.background_color` | explicit style values (a hex color; a width in world units, 4 is the `regular` role's); a class's own default color is the nearest role. A negative or too large width or font size keeps the role, with a warning; a background's alpha is dropped, with a warning (a background is opaque). `lint(aesthetic=True)` says which are off the theme (`off_palette`, `off_scale`) |

What does not carry over, with a warning in `ManimImport.warnings` (printed by the command):
a turned or stretched text (drawn level), a turned camera frame, a part of a `MathTex` other
than its color, an updater that turns a mobject into another kind of shape, `Axes.get_area`,
a file outside the scene's folder, a color phasor cannot read. A shape phasor cannot draw as
its own kind (a zero radius, a coordinate past 1e9) is drawn as a path or a line, with a
warning. A Manim name phasor lacks raises an error that says so; an updater, rate function
or wait condition that fails, `exit()` in the script, and a script still running after
`MAX_IMPORT_SECONDS` (600 s; checked at each play and updater call) are a
`ManimImportError` naming it. The import applies its document in one patch at the end
(and before each morph, which reads it), so its time grows with what the script makes, not
with the number of plays. `lint(strict=True)` names every moving object of an import in
`raw_motion`, by design: its motion is tracks at Manim's own times, never verbs anchored to
markers (`examples/manim_camera_tour.py` is clean in `lint()` and has only that).

```python
import tempfile
from pathlib import Path

from phasor.importers import import_manim

path = Path(tempfile.mkdtemp()) / "demo.py"
path.write_text(
    "from manim import *\n"
    "class Demo(MovingCameraScene):\n"
    "    def construct(self):\n"
    "        c = Circle(radius=1, color=BLUE, fill_opacity=0.5)\n"
    "        t = ValueTracker(0)\n"
    "        d = Dot().add_updater(lambda m: m.move_to(c.point_at_angle(t.get_value())))\n"
    "        self.play(Create(c), run_time=2)\n"
    "        self.add(d)\n"
    "        self.play(t.animate.set_value(PI), self.camera.frame.animate.scale(0.5), run_time=1)\n"
    "        self.wait(1)\n"
)
result = import_manim(path)
assert result.duration == 4.0 and result.warnings == []
state = result.scene.engine.state_at
assert state(0.0)["circle_1"].params.progress == 0.0 and state(2.0)["circle_1"].params.r == 1.0
assert state(1.0)["circle_1"].params.fill_opacity == 0.5  # a filled shapes.Circle
assert abs(state(3.0)["dot_1"].params.pos[0] + 1.0) < 1e-6  # the updater: half way round
assert abs(state(3.5).values["camera.height"] - 4.0) < 1e-9  # the frame zoomed in
```

## Sketch: draw on the scene (Milestone 11)

Draw on the picture (the editor's **Sketch** tab, or strokes as JSON for `phasor sketch`)
and each stroke becomes a **proposal**: a summary, the patch ops that make it, verbs to run
after them, and what was worked out. Nothing changes until it is accepted (`accept`, one
`Engine.apply` plus the verbs; undo takes it back). Strokes are output pixels (row 0 at the
top), classified by fitting, not by a learned model: `dot`, `line`, `arrow` (a hook at the
end, or a V drawn at the tip), `circle`, `closed`, `scribble`, `curve` (`wavy`), and
`handwriting` (small strokes close together).

| Drawn | Over | Becomes |
| --- | --- | --- |
| a loop | a domain coloring | a glowing contour, traced from the playhead, with its integral of f dz: numeric along the drawn path and exact as 2πi Σ winding × residue (rational f); a label with the value. The direction you draw counts |
| a loop | other objects | they are emphasized (the `highlight` color role) |
| a loop, line, curve | nothing | a Line (a circle is the fitted circle) |
| a dot | a pole or zero (within 14 px) | a Dot snapped onto it |
| an arrow | an object with `pos` | the object moves to the tip; elsewhere, an Arrow |
| a scribble | objects | they are removed (cross it out) |
| handwriting | anywhere | a picture of it (`proposal.image`, PNG) for an AI or the person to read; `transcribe(p, text)` makes the Text. There is no offline handwriting recognizer |

Sketch edits plain scenes. On a scene built by a script, only parameter and style
proposals (move, emphasize) apply; the script's own code can apply the rest with
`engine.apply(p.ops, origin="script")` (see `examples/sketch_contour.py`).

```python
import math

import numpy as np

from phasor import Scene, cx
from phasor.sketch.propose import Sketch

s = Scene("draw")
s.add(cx.DomainColoring(id="plane", function="1/(z**2+1)", size=(8.0, 4.5)))
sk = Sketch(s.engine, t=0.0)
c = sk.to_pixels([[0.0, 1.0, 0.0]])[0]  # the pole at i, in pixels
a = np.linspace(0, 2 * math.pi, 60)
loop = np.stack([c[0] + 80 * np.cos(a), c[1] - 80 * np.sin(a)], axis=1)  # counterclockwise
(p,) = sk.propose([loop])
assert p.kind == "contour" and abs(p.data["exact"] - math.pi) < 1e-9
assert abs(p.data["integral"] - math.pi) < 0.01 and p.data["latex"] == r"\pi"
```

## 3D: orbit camera, surfaces, camera goals

`use_camera("orbit")` makes the scene 3D: a perspective camera on a sphere around
`target`, z up. Parameters: `azimuth` (0 looks from +x, 90 from +y), `elevation`,
`distance`, `fov` (vertical degrees), `drift` (slow sway in degrees; negative = the
theme's, 0 = still). Text and strokes keep their size on screen at any depth; dots and
surfaces get smaller with distance like real things. Surfaces are lit by the theme's
three-point rig and fade into the background with depth (theme `[light]`).

Leave `elevation`, `distance`, `target` (and `azimuth`) unset and give the camera
**goals**; the camera solver picks the rest at every moment, also while you orbit:
`keep_visible(a, b)`, `center(a)`, `thirds(a)`, `avoid_occlusion(a)`, `min_size(a, share)`.

```python
from phasor import Dot, Scene, Text, cx

s3d = Scene("riemann", passes=None)  # passes=None: analytic geometry only, fast
s3d.use_camera("orbit", azimuth=-60)
s3d.add(cx.RiemannSurface(id="sheets", kind="sqrt", radius=2.0, style={"material": "glass"}))
s3d.add(Dot(id="branch", pos=(0, 0, 0), r=0.08, style={"color": "highlight"}, tags=["key"]))
s3d.add(Text(id="branch.label", text="branch point"))
s3d.constrain("label(branch) near branch")  # placed by depth: readable from every side
s3d.constrain("camera keep_visible(sheets)")
s3d.marker("spin", 1.0)
s3d.marker("done", 6.0)
s3d.at("spin").orbit(360, until="done")  # azimuth is animated; the solver works around it

e3 = s3d.engine
assert e3.state_at(6.0).values["camera.azimuth"] == 300
assert set(e3.layout.camera) == {"camera.elevation", "camera.distance", "camera.target"}
sheet = e3.inspect(3.0, id="sheets")["objects"][0]
assert sheet["on_screen_fraction"] == 1.0 and sheet["semantic"]["sheets"] == 2
```

- `orbit(degrees, until=None)` turns the camera (theme `[motion.orbit]` time per turn, or
  until a marker); `orbit_reveal(*ids)` reveals while swinging a little.
- `focus_pull(id)` in 3D aims at the object and comes close enough to fill the frame.
- With camera goals, a camera parameter that is animated but never set is the solver's
  until its first motion: a `focus_pull` or `orbit` at a marker starts from wherever the
  solver had the camera (an orbit turns by its angle from there; a pull glides in), so the
  goals hold before it and nothing jumps when it starts.
- A label (`near` rule) lives at its anchor's depth, nudged toward the camera, and is moved
  only sideways; it avoids only what is **in front** of it, so it may sit on a surface
  behind it. Lint: `occluded_label` when something in front hides it; `visible_in_3d(a)`
  as a rule makes that an error.

### The event log and spatial queries

```python
log = e3.events(hz=2)  # 10 Hz by default: visibility 0.9/0.5/0.1, frame, region, occluders
from phasor.inspect.events import problems

assert problems(log) == []  # nothing important hidden or leaving the frame
assert e3.when_hidden("branch") == []  # spans where less than half of it is visible
free = e3.free_space_near("branch", 3.0)  # empty label-sized boxes, nearest first
assert free and set(free[0]) == {"slot", "box", "distance"}
```

Events carry `t`, the `marker` being spoken, the `id`, `severity` (`warning` = a viewer
would notice) and `fix_hints`.

## Dynamical systems: phase portraits and fixed points

`from phasor import dyn`. A planar system x' = f(x, y), y' = g(x, y) from formulas (read safely,
never run as Python; named numbers go in `constants="mu=1.5, k=2"`). `pos` is the window's
centre and `extent` its size; one unit of (x, y) is one world unit.

- `dyn.PhasePortrait(f=, g=, extent=, seeds=((x, y), ...) or n_seeds=, t_end=, progress=,
  backward=, flow=, nullclines=)`: trajectories from the seeds (RK4), their colour running
  along time; `progress` 0 -> 1 traces them (`s.at("m").sweep("flow.progress", 1.0,
  show=False)`); `flow` adds the small-arrow field and `nullclines` the curves where x' = 0
  and y' = 0. Semantics: the number of trajectories and fixed points.
- `dyn.FixedPoints(f=, g=, extent=, rays=, labels=, reveal=)`: every point where the system rests
  (Newton from a grid), classified from the Jacobian's eigenvalues: `saddle`, `stable node`,
  `unstable node`, `stable spiral`, `unstable spiral`, `centre`, `degenerate` or
  `non-hyperbolic`. Each is a beacon coloured by its type, its real eigen-directions are rays
  (cool where the motion comes in, warm where it goes out), with a label of the eigenvalues.
  Semantics list each point's position, type, stability, eigenvalues, trace and determinant.
  `reveal` 0 -> 1 shows them in order along x.

```python
from phasor import Scene, dyn

s = Scene("pendulum", passes=None)
s.add(dyn.PhasePortrait(id="flow", f="y", g="-sin(x) - 0.3 * y", extent=(9, 6), n_seeds=8))
s.add(dyn.FixedPoints(id="rest", f="y", g="-sin(x) - 0.3 * y", extent=(9, 6)))
kinds = [p["type"] for p in s.engine.inspect(0.0, id="rest")["objects"][0]["semantic"]["points"]]
assert kinds == ["saddle", "stable spiral", "saddle"]
```

Example: `examples/phase_portrait.py`. The prism looks make trajectories glow with the arrow
field and nullclines kept faint.

## Flows: systems, sets and equilibria (`flow`, the Explorer's objects)

`from phasor import flow`. A system x' = F(x) in 2 or 3 dimensions from formulas (Milestone
26). The `System` holds the formulas once; its parts are children with dotted ids
(`lorenz.field`) or name it with `system="lorenz"`. Formulas are read by the whitelist parser
(never run as Python); flow formulas also know `sign`, `floor`, `atan2(y, x)`, `min` and
`max` (other domains do not). Text longer than 2000 characters, deeper than 100 levels, with
a power of numbers too large to compute (`9**9**9`, `(1 + 10**-17)**(10**7)`) or a sum
raised past the 100th power is refused with a reason, in every domain.

- `flow.System(equations=("sigma*(y - x)", ...), constants="sigma=10, rho=28", window=((lo, hi), ...),
  tau=, rate=, dt=, terms=("vortex(pos=(0, 0), strength=2)", ...), pos=, scale=, frame=, label=)`:
  the field. `tau` is the flow time (animate it: `s.at("m").sweep("lorenz.tau", 5.0,
  show=False)`), plus `rate` times the scene time. `dt` is the fixed RK4 step (a seed's path
  never depends on the other seeds). `terms` are sources that add, as text: `uniform(v=)`,
  `source`/`sink(pos=, strength=, core=)`, `vortex(pos=, strength=, core=)`,
  `linear(A=, pos=)`. Every named number needs a value in `constants`; world = `pos + scale *
  (x, y, z)`. Semantics: the equations and LaTeX, the variables and which come from a
  `Variable`, the divergence (formula, or None where it has none in closed form, as with
  `sign`; min and max measured on a 7-per-side grid), linear or not (None when unknown). A
  formula past 120 operations per component gets a numeric Jacobian (central differences).
  `flow.preset("lorenz", rho=27.7)` gives the parameters of a classic
  (`flow_math.PRESETS`: lorenz, rossler, thomas, chua, halvorsen, aizawa, van_der_pol,
  pendulum, duffing, lotka_volterra, brusselator, fitzhugh_nagumo, sir, double_well, hopf,
  saddle_node, pitchfork and the linear zoo).
- `flow.Variable(name=, value=, min=, max=)`: a named number; a System whose `constants`
  name it takes this value (a slider: point a `Control` at `<id>.value`). Draws nothing.
  Two Variables of one name: the first is read, the second is lint `duplicate_variable`.
- `flow.FieldArrows(counts=, shape="auto"|"arrow"|"cone", length_by=)`: the field on a grid,
  flat arrows in 2D and cones in 3D, coloured by size. At most `flow.MAX_ARROWS`.
- `flow.Probe(x0=, trail=, label=)` (not the motion `Tracer`, which rides a curve): one seed's path to the flow time (backward for a
  negative one, stopped at the escape radius), the dot, a label such as `"phi^t(x0)"`.
- `flow.FlowSet(shape="disk"|"ball"|"box"|"segment"|"circle"|"points", center=, size=,
  count=, fill="filled"|"boundary"|"both", trail=, h_max=)`: initial conditions carried as
  a set (a checkpointed `Simulation`). Filled sets are spread evenly without random
  numbers; an outline is refined so neighbours stay closer than `h_max`, every new point
  integrated from flow time 0 (never interpolated). A 3D ball's outline is latitude and
  longitude rings. At most 2*10^5 points, 64 trail samples (recomputed when drawn, not
  stored), `flow_math.MAX_WORK` (1e8) point-steps (an outline counted at its 2^17 cap).
  Checkpoints stay in memory: nothing is written to `.phasor_cache`.
- `flow.Equilibria(grid=, reveal=)`: every equilibrium Newton finds from `grid` starts per
  side in the window, classified from the exact Jacobian: in 2D `saddle`, `stable node`,
  `unstable node`, `stable spiral`, `unstable spiral`, `centre`, `degenerate`,
  `non-hyperbolic` (as `dyn.FixedPoints`); in 3D also `saddle-focus`; `undefined` where
  the field is not finite around the point. A double or triple root (a bifurcation point)
  is one point, not a cloud: Newton runs until its steps stop moving. Semantics carry the
  window and grid searched (none found is not none exist), `non_isolated`, and per point the
  eigenvalues, subspace dimensions (`dims`: s, u, c), trace, determinant.
- `flow.EquilibriumView(near=, radius=, name=, card=)`: the equilibrium nearest `near`
  (never by index: equilibria move and merge as a variable moves), its stable and unstable
  subspaces inside a ring, and a card in plain characters.
- `flow.Manifold(near=, which="both"|"stable"|"unstable", length=, progress=)`: the 1D
  stable and unstable manifolds of a saddle, integrated from its eigen-directions.

```python
from phasor import Scene, flow

s = Scene("lorenz_numbers", passes=None)
s.add(flow.System(id="lorenz", **flow.preset("lorenz", rho=27.7, beta=2.67)))
s.add(flow.Equilibria(id="lorenz.eq"))
s.add(flow.EquilibriumView(id="lorenz.cplus", near=(8, 8, 27), name="C+"))
s.add(flow.FlowSet(id="lorenz.ball", shape="ball", center=(1, 1, 20), size=2, count=500))
s.set("lorenz.tau", 2.0)
seen = {o["id"]: o["semantic"] for o in s.engine.inspect(0.0)["objects"]}
kinds = [p["kind"] for p in seen["lorenz.eq"]["points"]]
assert kinds == ["saddle-focus", "saddle", "saddle-focus"]
assert seen["lorenz.cplus"]["card"][1] == "eigenvalues -13.84, 0.08516 +/- 10.15i"
assert seen["lorenz.ball"]["alive"] == 500
```

Numbers in semantics are rounded to 6 digits. A chaotic path is reproducible on one machine
only (Lorenz parts across machines near flow time 40), so positions of probes and sets are
reported only up to `flow_math.SAFE_TAU` (10). The numerics (`flow_math`: `System`,
`integrate`, `FlowSim`, `equilibria`, `classify`, `manifolds`) are plain NumPy and SymPy;
`dyn` is a 2D front of the same core. Examples: `examples/lorenz_explorer.py` (3D) and `examples/pendulum_flow.py` (2D, a
slider, separatrices, a probe).

### Explorer rows (Milestone 27)

The Explorer's typed rows are a view of the document (`phasor.workspace.explorer`; the AI
reaches them through the `explorer_*` tools in `MCP.md`). Each row is an object that keeps
the text as typed in `source`, its place in `row`, a kind chosen over the guess in `kind`,
and `visible`: `name = number` is a `flow.Variable` with a `Control` on `<id>.value`;
`x' = ...` rows together are one `flow.System` (a line of `source` per row) with
`<id>.field` and `<id>.eq`; `(P, Q)` is a System with its field; `(a, b)` is a `flow.Probe`
on the system of its dimension. Anything else is a `flow.Note(source=, kind=, error=)`: it
draws nothing and keeps the text, with the error in plain words for a row that does not
read (graphs, curves, surfaces, heat maps and implicit curves are read but not drawn yet).
`explorer.plan(document, edits)` gives the ops of one commit (a kind change removes and adds
in the same patch; ids by meaning, never reused); `explorer.rows_of(document)` reads the list
back.

```python
from phasor import Scene
from phasor.workspace import explorer

s = Scene("rows", passes=None)
ops, report = explorer.plan(
    s.engine.document,
    [
        {"text": "x' = y"},
        {"text": "y' = -sin(x) - g y"},
        {"text": "(1, 0.5)"},
        {"text": "y = x^2"},
        {"text": "q = 9^9^9"},
    ],
)
assert s.engine.apply(ops, origin="script").ok  # a Scene is a script
assert report["new_variables"] == ["g"]  # a slider for the free name, at commit time
rows = {r["id"]: r for r in explorer.rows_of(s.engine.document)}
assert rows["system:y"]["text"] == "y' = -sin(x) - g*y"
assert rows["point"]["kind"] == "point" and rows["graph"]["drawn"] is False
assert "too large" in rows["row"]["error"]
assert s.engine.document.controls[0].path == "g.value"
```

## Command line

| Command | Status | Purpose |
| --- | --- | --- |
| `phasor doctor [--json] [--require A\|B\|C]` | M0 | Prove CUDA, NVRTC, OpenGL 4.3, encoders and, without CUDA, the C++ compiler for CPU kernels (name, version, threads); print tier |
| `phasor inspect SCENE [--t T] [--id ID] [--detail brief\|standard\|full] [--allow-scripts]` | M1 | What is on screen at time t |
| `phasor lint SCENE [--fix] [--strict] [--json] [--step S] [--allow-scripts]` | M2 | Lint; exit 1 on errors (`--strict`: on warnings too). `--fix` applies suggested fixes and saves them |
| `phasor storyboard SCENE [-o PNG] [--t T]... [--marks] [--json] [--allow-scripts]` | M2 | Keyframe contact sheet plus what the inspector sees |
| `phasor render SCENE [-o OUT] [--preset draft\|preview\|final\|4k] [--fps F] [--start S] [--end E] [--frame T] [--workers N] [--threads T] [--allow-scripts]` | M3, M21 | Video (`.mp4`; `.mov` is ProRes; a folder or `%05d.png` is a PNG sequence) or one PNG with `--frame`. `--start`, `--end` and `--frame` take seconds or a marker name. `draft` (M21): small, 15 fps, plain look, written to `<scene>.draft.mp4` by default. `--workers`: render processes (default a quarter of the CPU threads, 1 to 4); the result is the same file as one process makes. `--threads`: CPU kernel threads per process without CUDA (default: the cores shared out, workers x threads <= cores) |
| `phasor preview SCENE [...]` | M21 | `phasor render SCENE --preset draft` (the same options): the quick plain look, a window by markers with `--start`/`--end` |
| `phasor kit [FOLDER]` | kit | Set up or update a folder where an AI writes scenes (default `Documents/phasor`): `CLAUDE.md`, the AI docs, the examples sorted by kind, the `/scene` command, `TASTE.md`, and the phasor tools in `.mcp.json`. Your `TASTE.md`, `WISHLIST.md` and `scenes/` are never replaced |
| `phasor bench [--width W] [--height H]` | M3 | ms per frame per kernel and peak VRAM; saved in `benchmarks/` |
| `phasor bench --frames SCENE [--count N] [--preset P] [--allow-scripts]` | GPU_SPEED.md | ms per final frame per stage (state, simulation, dedup, draw, gpu draw, readback, widen, blur sum, post, srgb, encode); saved in `benchmarks/` |
| `phasor mcp [--root DIR] [--allow-scripts] [--attach]` | M5, M10 | MCP server; `--attach` edits the scene open in a running `phasor gui` |
| `phasor gui [SCENE] [--root DIR] [--no-ai] [--install-shortcut]` | M10 | The desktop app (needs `uv sync --extra gui`): a start screen, or the editor on SCENE; also `phasor-gui`; a Manim script opens too |
| `phasor web SCENE [-o PAGE.html] [--steps N] [--allow-scripts]` | M12 | Bake a self-contained interactive web page from the scene's sliders |
| `phasor manim SCRIPT [--scene NAME] [-o OUT] [--code]` | M13 | Import a Manim Community scene as a phasor scene |
| `phasor sketch SCENE STROKES.json [--t T] [--accept [K ...]] [--text K=TEXT]` | M11 | Turn drawn strokes into proposals; `--accept` applies them |

`SCENE` is a `.phasor.json` document or a `.py` script that defines exactly one `Scene`.
`--allow-scripts` (off by default) runs the Python modules a saved document names in
`meta.modules` (its own object types and updaters); without it those objects are
`missing_type` placeholder boxes. A `.py` scene is code you ran, so it needs no flag.
`phasor gui SCENE.phasor.json` (and the app's Open, recent scenes and examples) likewise
opens a document without running its modules; the person trusts them in the editor
(`Editor.trust_scene_code()`).

## The desktop app (`phasor-gui`)

`phasor-gui` (or `phasor gui`) with no scene opens a **start screen**: new scene (2D, 3D,
a complex function), open a file, recent scenes, and the examples (double-click copies one
to `Documents/phasor/examples` and opens the copy). Each scene gets its own window with a
menu bar: **File** (New, Open, Open Example, Open Recent, Save, Save As, Export, Close,
Quit), **Edit** (Undo, Redo), **View** (Play, back to the start), **Settings** (math
typesetting: built-in, the default, or LaTeX; add phasor to the applications menu),
**Help** (these docs, a hardware check, about). The title marks unsaved edits with `•`,
and closing such a window asks first. **Export** makes a picture, a video or a web page
with sliders, with a progress bar, in a separate process (it can be cancelled and the
window keeps working). The packaged app is built with PyInstaller (`packaging/README.md`).

The logic is plain Python in `phasor.gui.app` (no Qt): settings, templates, examples,
exports and the shortcut.

```python
import tempfile
from pathlib import Path

from phasor.gui.app import EXPORTS, TEMPLATES, ExportJob, examples

assert {"2D scene", "3D scene"} <= set(TEMPLATES) and set(EXPORTS) == {"picture", "video", "web"}
assert any(e["name"] == "julia explorer" for e in examples())
job = ExportJob(None, "web", Path(tempfile.mkdtemp()) / "page.html", scene_dir=".")
assert job.command("s.phasor.json")[-5:] == ["s.phasor.json", "-o", str(job.out), "--steps", "5"]
```

## The editor and the AI on one scene (`phasor gui`)

`phasor gui scene.py` opens the editor: the viewport (click selects; drag the selected
object to move it), the timeline (markers, one lane per animated parameter, click or drag to
scrub, Space plays), and tabs for the **inspector** (every parameter as a field: type JSON
or text and press Enter; style roles as menus; what the inspector reads), **lint** (click an
issue to jump to its object and time) and **looks** (switch the whole scene to a look; show
the variants of a beat and pick one: recorded in `style/choices.jsonl` as the person's
choice). Ctrl+Z / Ctrl+Shift+Z / Ctrl+S undo, redo, save. Every edit is one
`Engine.apply`, so a script scene keeps them as overrides. Also: drag empty space to turn the 3D
camera (shift-drag or a 2D scene: slide it), the wheel zooms; the **Sliders** tab previews the
scene's controls; the **Code** tab shows the scene as Python (the script itself, or one
generated from the document) with highlighting: edit and Run (Ctrl+Enter) to rebuild the scene;
the timeline shows narration with its waveform, and lets you drag markers and animation spans
(or their ends) to retime them; the **Who changed what** dock lists your edits and an attached
AI's; hovering shows values under the pointer. `phasor-gui` starts the editor as a program.

While it is open, `phasor mcp --attach --root <same folder>` serves the editor's own live
scene to an AI: its edits show up in the window, the window's edits are what it inspects.
The editor writes `<root>/.phasor/gui.json` (host, port, a token; readable only by you)
and removes it on close; `--no-ai` keeps the AI out. The logic is `phasor.gui.controller.Editor`
(no Qt in it), and the shared tools are `phasor.workspace`:

```python
import tempfile

from phasor.gui.controller import Editor
from phasor.workspace import Workspace
from phasor.workspace.remote import RemoteServer, RemoteWorkspace

root = tempfile.mkdtemp()
ws = Workspace(root)
ws.scene_new("together")
ed = Editor(ws)
server = RemoteServer(ws, ed.lock).start()
server.write_handoff()
ai = RemoteWorkspace.from_handoff(root)  # what `phasor mcp --attach` uses
ai.scene_edit({"op": "add_object", "object": {"id": "p", "type": "dot"}})
assert ed.changed()  # the editor sees the AI's edit
assert ed.set_param("p", "r", "0.3")["ok"]  # and the AI sees the person's
assert ai.inspect(0.0, id="p")["objects"][0]["id"] == "p"
ai.close()
server.stop()
```

## Hardware check

```python
from phasor.render.doctor import run_doctor

report = run_doctor()  # never raises; each probe fails on its own
assert report.tier in {"A", "B", "C"}
```

Tiers: **A** CUDA kernels and GL 4.3. **B** GL 4.3 only. **C** no usable GL: the inspector
runs in analytic mode, stills are unavailable.
