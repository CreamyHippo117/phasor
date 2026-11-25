# Camera and light verbs: follow, ripple, spotlight

Three verbs that point the eye: the camera rides a moving feature, rings of light spread
from a spot, and a pool of light picks out one thing while the rest of the frame (fields
included) goes dark. Timing comes from the theme (`[motion.follow]`, `[motion.ripple]`,
`[motion.spotlight]`); every segment they write is tagged with the verb, so
`lint(strict=True)` accepts them. The objects they add live in `phasor.domains.lighting`
(`from phasor import light`).

## follow

`s.at(marker).follow(object_id, feature="center", index=0, until=None)`

The camera's center (2D) or orbit target (3D) glides onto point `index` of the object's
feature `feature` (any name in `obj.impl.features`: a domain coloring's `"zeros"` or
`"poles"`, a tracer's head, a `"center"`), rides along while it moves, and glides back, over
the theme's `follow` time or until the marker `until`. The verb adds an invisible probe
`<id>.follow` (type `follow`) and animates its weight `on`; the layout bake moves the camera,
so the document's camera center/target stay unset. The point is tracked from frame to frame
(the nearest point of that feature to the last one), so a zero keeps its identity when a
root finder lists zeros in another order.

Rules: the camera parameter it moves must be left to it: not set, not animated (no
`focus_pull` on the same camera) and not chosen by camera goals. Otherwise the document gets
the error `follow_conflict`. One feature per object.

```python
from phasor import Scene, cx

s = Scene("follow_doc", passes=None)
s.add(cx.DomainColoring(id="plane", function="(z - a)*(z + 1)", a=(1.0, 0.0), size=(16, 10)))
s.marker("drift", 1.0)
s.marker("rest", 4.0)
s.at("drift").morph("plane.a", (2.0, 1.0), until="rest")
s.at("drift").follow("plane", feature="zeros", index=1, until="rest")
center = s.engine.state_at(3.0).values["camera.center"]
zeros = s.engine.inspect(3.0, id="plane")["objects"][0]["semantic"]["zeros"]
assert min(abs(x - center[0]) + abs(y - center[1]) for x, y in zeros) < 0.05  # on the zero
```

## ripple

`s.at(marker).ripple(object_id, at="center", index=0, rings=3)`

Rings of light spread from a spot and fade: `at` is a feature name (point `index` of it) or a
point `(x, y)` / `(x, y, z)`. The rings are the child `<id>.ripple` (type `ripple`, made the
first time; later ripples on the same object reuse it and must name the same spot). The spot
is found again at every frame, so the rings ride a zero that moves.

```python
from phasor import Dot, Scene, cx

s = Scene("ripple_doc", passes=None)
s.add(cx.DomainColoring(id="plane", function="z**2 - 1", size=(8, 4.5)))
s.add(Dot(id="p", pos=(0, 1)))
s.marker("here", 0.5)
s.at("here").ripple("plane", at="zeros", index=1).ripple("p", at=(0.5, 1.0), rings=2)
sem = s.engine.inspect(1.0, id="plane.ripple")["objects"][0]["semantic"]
assert sem["rings_visible"] >= 1
```

## spotlight

`s.at(marker).spotlight(*ids, until=None, pad=0.25)`

A soft pool of light over the targets (their bounds, `pad` larger, at least 0.12 of the frame
height), the rest of the frame under a dark veil. The light is the overlay object
`spotlight` (type `spotlight`; params `center`, `radius`, `soft`, `strength`), so it dims
fields and surfaces too, which `dim_others` leaves alone. The first call fades it in where
the target is; later calls glide it to the next target; `until=` fades it out at that
marker, and `spotlight()` with no ids fades it out now. It is a backdrop: labels may sit on
it, it hides nothing from the inspector, and `dim_others` does not touch it.

```python
from phasor import Dot, Scene, cx

s = Scene("spotlight_doc", passes=None)
s.add(cx.DomainColoring(id="plane", function="z**2 - 1", size=(8, 4.5)))
s.add(Dot(id="left", pos=(-1, 0)))
s.add(Dot(id="right", pos=(1, 0)))
s.marker("one", 0.5)
s.marker("two", 2.0)
s.marker("off", 3.5)
s.at("one").spotlight("left")
s.at("two").spotlight("right", until="off")
assert abs(s.engine.state_at(3.0)["spotlight"].params.center[0] - 1.0) < 0.05
assert s.engine.state_at(5.0)["spotlight"].params.opacity == 0.0
```

## MCP

The `animate` tool takes the target ids in `targets` and the keyword options as a dict in
`value`:

- `animate(verb="follow", at="drift", targets=["plane"], value={"feature": "zeros", "index": 1, "until": "rest"})`
- `animate(verb="ripple", at="here", targets=["plane"], value={"at": "zeros", "index": 1})`
- `animate(verb="spotlight", at="one", targets=["left"], value={"until": "off"})`; with
  `targets=[]` it turns the light off.
