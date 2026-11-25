# Scale verbs: grow, shrink, squish, flatten

Four verbs that change the size of any object, not only those with a radius:

| verb | what it does |
|---|---|
| `grow(*ids, by=1.5)` | scale up by `by` about the object's own center |
| `shrink(*ids, by=0.5)` | scale down by `by` (0.5: to half) about its center |
| `squish(*ids, axis="y", to=0.4)` | squash along one axis (`"x"`, `"y"` or `"z"`) to `to` of its size; `to=1.0` restores that axis |
| `flatten(*ids, axis="z")` | squash flat along `axis`; `squish(..., axis=..., to=1.0)` brings it back |

They animate one parameter every object has, `stretch` (x, y, z), a per-axis scale of
everything the object draws. It is owned by these verbs (as `dim` is by `dim_others`), so
it never fights a type's own `r`, `size` or `scale`. Timing comes from the theme's
`[motion.grow]`, `[motion.shrink]`, `[motion.squish]` and `[motion.flatten]`; several ids
stagger by `[motion] stagger`.

- **The center.** An object scales about the center of its unstretched bounds at that
  moment (for an equation, the middle of its box, not its baseline `pos`), so it grows in
  place. An object whose drawing changes while it is stretched (a curve being traced) scales
  about the middle of what is drawn so far.
- **Relative and absolute.** `grow` and `shrink` multiply the stretch the object has when
  the verb starts, on every axis, so a squish is kept: `squish(to=0.5)` then `grow(by=2)` is
  `(2, 1, 2)` along x, y, z. `squish` and `flatten` set one axis and keep the others. Each
  factor is at least 0.001 (a flattened axis stays a sliver, never zero).
- **Everything follows.** The renderer stretches what the object draws (strokes, fills,
  surfaces, text, kernel-drawn fields such as `DomainColoring`), and the inspector, labels,
  `focus_pull` and the camera solver see the stretched bounds and feature points. A stroke
  is stretched with its curve: a squished line is thinner across the squish. Text is
  squished as a picture (its letters get flatter). Semantics (zeros listed by a
  `DomainColoring`, a `Dot`'s `position`) keep the unstretched numbers.
- **In 2D**, everything sits at z = 0, so `flatten("thing")` (axis z) changes nothing you can
  see; flatten along `"x"` or `"y"` instead. In 3D, `flatten(axis="z")` presses a surface
  onto its middle plane.
- A composite (a type with `build`) is not drawn itself: the verbs refuse it and name its
  parts.

```python
from phasor import Dot, Equation, Scene
from phasor.domains.algebra import Polygon

sc = Scene("scale_verbs", passes=None)
sc.add(Dot(id="point", pos=(-3, 0, 0), r=0.4))
sc.add(Equation(id="law", parts=("$e^{i\\pi}$", "$+1=0$"), pos=(0, 1, 0)))
sc.add(Polygon(id="square", n=4, r=1.0, pos=(3, 0, 0)))
for name, t in (("big", 0.5), ("small", 1.5), ("squash", 2.5), ("flat", 3.5), ("back", 5.0)):
    sc.marker(name, t)
sc.at("big").grow("point", by=2)
sc.at("small").shrink("point", by=0.5)
sc.at("squash").squish("law", axis="y", to=0.4)
sc.at("flat").flatten("square", axis="y")
sc.at("back").squish("law", "square", axis="y", to=1.0)

e = sc.engine
b = e.state_at(1.4)["point"].impl.world_bounds(e.state_at(1.4))
assert abs((b.hi[0] - b.lo[0]) - 1.6) < 1e-6  # r = 0.4, grown twice: 1.6 across
st = e.state_at(4.8)
assert st["square"].params.stretch[1] == 0.001  # flat
assert e.state_at(7.0)["square"].params.stretch == (1.0, 1.0, 1.0)  # and back
assert e.document.timeline.tracks["law.stretch"][0].verb == "squish"
```

The same through MCP, with the options as a dict:

```json no-run
{"tool": "animate", "args": {"verb": "squish", "targets": ["law"], "at": "squash",
 "value": {"axis": "y", "to": 0.4}}}
```

`examples/verbs_scale.py` shows all four.
