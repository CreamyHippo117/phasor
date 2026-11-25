# Verbs for space and points: lift, slice, project, dissolve, assemble

Five choreography verbs, used like the others (`s.at("marker").verb(...)`; see API.md
"Choreography verbs"). Each makes the helper objects it needs the first time, as children of
the object it acts on, so nothing else has to be written. Timing comes from the theme
(`[motion.lift]`, `[motion.slice]`, `[motion.project]`, `[motion.dissolve]`,
`[motion.assemble]`). Over MCP, `animate(verb=..., at=..., targets=[...], value={options})`.

The shared scene for the examples below (they run in order in one namespace):

```python
import math

from phasor import Dot, Line, Scene, cx, geo, linalg, pde, space

s = Scene("verbs_space_doc", passes=None)
s.use_camera("orbit")
s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(6, 4)))
s.add(geo.ParametricSurface(id="ball", fx="2*sin(u)*cos(v)", fy="2*sin(u)*sin(v)",
                            fz="2*cos(u)", u_range=(0, math.pi), v_range=(0, 2 * math.pi),
                            pos=(0, 0, 3)))
s.add(linalg.Arrow(id="v", start=(0, 0, 0), end=(2, 1.5, 0)))
s.add(linalg.Arrow(id="axis", start=(0, 0, 0), end=(3, 0, 0)))
s.add(Line(id="star", points=[(0, 1, 0), (0.3, 0.3, 0), (1, 0.2, 0), (0.4, -0.2, 0),
                              (0.6, -1, 0), (0, -0.5, 0), (-0.6, -1, 0), (-0.4, -0.2, 0),
                              (-1, 0.2, 0), (-0.3, 0.3, 0), (0, 1, 0)]))
for name, t in (("rise", 1.0), ("cut", 4.0), ("shadow", 7.0), ("shatter", 10.0), ("rejoin", 13.0)):
    s.marker(name, t)
```

## lift(plane_id, height=0.6)

A flat `DomainColoring` rises into the landscape of its function. The height is
`log(1 + |f|)`, clipped at 3 (a pole becomes a flat-topped tower, never a spike), times
`height` (world units per unit of `log(1 + |f|)`); the color is the phase of f; `lines`
(20) glowing grid lines run over it in the hue under them. While it grows, the flat coloring
fades. The surface is the child `<id>.surface`, a `space.Landscape`, made the first time with
the plane's `function` copied and its `pos`, `size`, `a` and `blend` linked by expressions, so
moving the plane (or its constant `a`) moves the landscape. Use the orbit camera, or you
look straight down on it. The plane's function is read in world coordinates: the poles of
`1/(z**2 + 1)` are at `0 +- i`, so put the plane's `pos` where its poles are.

```python
s.at("rise").lift("plane", height=0.8)
e = s.engine
print(e.inspect(3.5, id="plane.surface")["objects"][0]["semantic"])
```

MCP: `animate(verb="lift", at="rise", targets=["plane"], value={"height": 0.8})`.

## slice(surface_id, axis="x", at=None, start=None)

A see-through cutting plane sweeps through a surface along `axis` ("x", "y" or "z"), from
`start` to `at` (world coordinates; with neither, across the surface's whole extent), and the
curve where it meets the surface glows on it. Works on `geo.ParametricSurface`,
`pde.Surface` and `space.Landscape` (anything with `grid_points(state)`). The plane is the
child `<id>.cut` (a `space.CutPlane`); the curve is recomputed from the surface at every
time, so it follows a surface that moves. A second slice on the same axis starts where the
plane was left; the other axis needs another plane. The default sweep range is the surface's
extent at the anchor.

```python
s.at("cut").slice("ball", axis="z")
print(e.inspect(5.5, id="ball.cut")["objects"][0]["semantic"])  # one closed curve, a circle
```

MCP: `animate(verb="slice", at="cut", targets=["ball"], value={"axis": "z"})`.

## project(v_id, onto, label=None)

An arrow (`linalg.Arrow`) or a point (`Dot`) is projected onto another arrow or a two-point
`Line`. A dashed perpendicular (with a right-angle mark) drops from the tip to the foot, then
the projection arrow grows along the line from its start. Children, all following the arrows
through expressions (move `v` or `axis` afterwards and the whole construction follows):
`<id>.drop` (`space.Dashed`), `<id>.proj` (`linalg.Arrow`) and `<id>.proj.grow` (a `Value`,
what the arrow's growth is driven by). `label` writes that text beside the projection
(`<id>.proj.label`, kept near it by the layout). The projection is onto the line through the
target's start along its direction; both arrows normally start at the same point. Projecting
`v` again re-runs it; onto something else needs a copy of `v`.

```python
s.at("shadow").project("v", onto="axis", label="proj")
print(e.inspect(9.0, id="v.proj")["objects"][0]["semantic"])  # components [2, 0, 0]
```

MCP: `animate(verb="project", at="shadow", targets=["v"], value={"onto": "axis"})`.

## dissolve(*ids, count=900, spread=None, seed=0) and assemble(*ids, ...)

`dissolve`: an object shatters into `count` points of light that drift apart, each along its
own random direction, up to `spread` world units (default about a third of the object's
size), while the object fades. `assemble` is the exact reverse: the points fly back onto the
object, which fades in as they arrive (used alone, the object is hidden until then; after a
`dissolve` it picks up where the points were left).

The points are the child `<id>.dust`, a `cloud.PointCloud(source="points")` whose new
`scatter` parameter (world units, 0 = untouched) the verbs animate. **They are sampled from
the geometry the object draws at the anchor**: along its center lines, in proportion to their
length, else on its drawn vertices, else (text, kernel-drawn pictures) anywhere in its
box; the cloud keeps those points, so an object that moves later does not take them with it.
The cloud lies in the plane of the object (x and y of a 3D one) and the directions come from
`rng(id, seed)`, so the same document draws the same dissolve. Write a dissolve and its
assemble in time order.

```python
s.at("shatter").dissolve("star", count=400)
s.at("rejoin").assemble("star", count=400)
print(e.inspect(11.5, id="star.dust")["objects"][0]["semantic"]["scatter"])  # at full spread
print(e.inspect(14.5, id="star")["objects"][0]["semantic"])  # whole again
```

MCP: `animate(verb="dissolve", at="shatter", targets=["star"], value={"count": 400})`.

## The new objects

- `space.Landscape(function, to, blend, a, pos, size, n, height, cap, hue, lines)`: the
  landscape of a function of z; `height` 0 is flat.
- `space.CutPlane(surface, axis, at, pad)`: the plane and its curve; semantics list the
  curves and their ranges.
- `space.Dashed(start, end, progress, dashes, duty, square, back)`: a dashed line.
- `cloud.PointCloud(scatter=...)`: every point drifts along its own direction by up to that
  distance.
