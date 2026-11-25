# Signal verbs: iterate, decompose, recompose

Three verbs for pictures that come from repeating a step or adding terms. Each is tagged with
its name (so strict lint sees choreographed motion), timed by the theme's `[motion.<verb>]`,
and anchored to a marker. A verb writes the object's own tracks, so one object's verbs are
written in time order (an earlier marker first); verbs on different objects may come in any
order.

## `iterate(orbit_id, n=None)`

Shows an orbit step by step. `orbit_id` is a `dyn.Orbit`: a map applied again and again, its
points joined by glowing segments, the colour running along the steps. The verb reveals the
object and runs its `progress` from 0 to 1 at an even pace (linear). The theme's `iterate`
time is the time of ten steps, so `n` steps take `n / 10` of it; `n` replaces the orbit's
`steps`. An orbit that ends early (it converged, or left `|z| <= bail`) is paced by the steps
it really has.

`dyn.Orbit` parameters: `kind` (`"map"` iterates `function`, `"newton"` iterates Newton's
method on the polynomial `function`, `"cobweb"` iterates a real map and draws the graph, the
diagonal and the staircase), `function` (a formula in `z` and the constant `a`, read by the
same whitelist parser as `DomainColoring`; never run as Python), `a` and `start` (as
`(re, im)`), `steps`, `progress`, `pos` (where z = 0 is), `scale` (world units per unit of z),
`window` (the cobweb's interval of x), `bail` and `tol`. The inspector reports
`steps_total`, `steps_shown`, `converged` (and `converged_at`, `limit`), `escaped` and the
points shown.

```python
from phasor import Scene, dyn

s = Scene("iterate_doc", passes=None)
s.add(
    dyn.Orbit(
        id="newton", kind="newton", function="z**3 - 1", start=(0.45, 0.9), steps=10, scale=2.0
    )
)
s.add(dyn.Orbit(id="mandel", function="z**2 + a", a=(0.3, 0.5), steps=30, scale=2.0, pos=(4, 0, 0)))
s.add(
    dyn.Orbit(
        id="logistic",
        kind="cobweb",
        function="a*z*(1 - z)",
        a=(3.7, 0.0),
        start=(0.2, 0.0),
        steps=20,
        scale=3.0,
        pos=(-3, -4, 0),
    )
)
s.marker("newton", 1.0)
s.marker("logistic", 6.0)
s.at("newton").iterate("newton", n=8)
s.at("logistic").iterate("logistic")
info = s.engine.inspect(2.0, id="newton")["objects"][0]["semantic"]
assert info["steps_total"] == 8 and info["steps_shown"] == 4
```

MCP form: `animate(verb="iterate", targets=["newton"], value={"n": 8}, at="newton")`.

## `decompose(*signal_ids)` and `recompose(*signal_ids)`

`decompose` takes a Fourier sum apart: the object's new `spread` parameter runs from 0 to 1,
and the first `ROWS` (8) harmonics each take a row of their own under the sum, drawn
smaller and in their own colour from the `colormap` style role. The sum itself stays where it
is. `recompose` is the inverse, with its own timing and tag: `spread` returns to 0 and the
rows fold back into the sum. At `spread` 0 the picture is exactly what it was without the
parameter. Works on `fourier.PartialSum` (not `Epicycles`; a `VerbError` says so). While
spread, the semantics also give `spread`, `harmonics_shown` and `harmonic_amplitudes`, and
the bounds grow to hold the rows.

```python
from phasor import Scene, fourier

s = Scene("decompose_doc", passes=None)
s.add(fourier.PartialSum(id="wave", kind="square", terms=5, pos=(0, 2, 0)))
s.marker("apart", 1.0)
s.marker("together", 6.0)
s.at("apart").decompose("wave")
s.at("together").recompose("wave")
e = s.engine
assert e.state_at(5.0)["wave"].params.spread == 1.0
assert e.state_at(9.0)["wave"].params.spread == 0.0
assert e.inspect(5.0, id="wave")["objects"][0]["semantic"]["harmonics_shown"] == 5
```

MCP form: `animate(verb="decompose", targets=["wave"], at="apart")` and
`animate(verb="recompose", targets=["wave"], at="together")`.
