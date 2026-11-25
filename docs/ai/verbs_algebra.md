# Algebra verbs

Six choreography verbs for the algebra and linear-algebra objects. Like every verb they
start at a marker (`s.at("name")`), take their timing from the theme (`[motion.<verb>]`),
return where they end (so `.then()` can follow them), and give the same document whatever
order the script writes them in. Through MCP each is `animate(verb=..., at=..., targets=[id],
value={...options})`, for example `animate(verb="act", at="turn", targets=["square"],
value={"element": "r"})`.

An example scene with all six is `examples/verbs_algebra.py`.

```python
import numpy as np

from phasor import Scene, algebra, linalg
```

## act: a symmetry turns and flips a polygon

`s.at(m).act(polygon_id, element)`. `element` is a dihedral element by label (`"r"`, `"s"`,
`"sr2"`) or number (rotations first, then reflections). The polygon turns and flips on its
own `rotation` and `flip` parameters, so its vertex numbers travel with it. The step acts on
what the polygon already did: after `r` then `s` the polygon shows `s*r` = `sr`. A rotation
takes the short way round. Two acts at the same marker play one after the other in a fixed
order (by element number), so writing them either way round gives the same document.

```python
s = Scene("act_demo", passes=None)
s.add(algebra.Polygon(id="square", n=4))
s.marker("turn", 1.0)
s.marker("flip", 4.0)
s.at("turn").act("square", "r")
s.at("flip").act("square", "s")
e = s.engine

def element(t):
    state = e.state_at(t)
    return state["square"].impl.semantics(state)["element"]

assert element(0.5) == "e"  # before anything
assert element(3.0) == "r"  # a quarter turn counterclockwise
assert element(6.0) == "sr"  # then the mirror
assert 0 < e.state_at(1.6)["square"].params.rotation < 90  # smooth in between
```

## apply: a matrix acts on a transform grid

`s.at(m).apply(grid_id, matrix)`. The grid's matrix becomes `matrix @ current` (the current
matrix at the marker), moving along the straight line between the two matrices, so a rotation
shrinks the grid a little on the way. Works for 2x2 and 3x3 grids (the matrix must match).
Applies at later markers pile up. Writing them out of time order gives the same document
unless a matrix before the inserted one is singular (then the verb says so; write those in
time order). A grid that shows a `flow` ignores its matrix and is refused.

```python
s = Scene("apply_demo", passes=None)
s.add(linalg.TransformGrid(id="grid"))
s.marker("shear", 1.0)
s.marker("turn", 4.0)
s.at("shear").apply("grid", ((1.0, 1.0), (0.0, 1.0)))
s.at("turn").apply("grid", ((0.0, -1.0), (1.0, 0.0)))
e = s.engine

def matrix(t):
    return np.array(e.state_at(t)["grid"].params.matrix)

assert np.allclose(matrix(0.5), np.eye(2))
assert np.allclose(matrix(3.0), ((1, 1), (0, 1)))
assert np.allclose(matrix(7.0), ((0, -1), (1, 0)) @ np.array(((1, 1), (0, 1))))
```

## generate: a Cayley graph grows from the identity

`s.at(m).generate(graph_id, by="r s")`. The graph's `reach` goes from 0 to 1: edges grow
from the identity, breadth first, one generator after another (`by` is the order the
generators join, default the graph's own; the first stage uses only the first generator,
the next adds the second, and so on). Each generator keeps its color. With `reach` at 1,
its default, the graph looks as it always did.

```python
s = Scene("generate_demo", passes=None)
s.add(algebra.CayleyGraph(id="cay", group="D4", generators="r s"))
s.marker("grow", 1.0)
s.at("grow").generate("cay")
e = s.engine

def reach(t):
    return e.state_at(t)["cay"].params.reach

assert reach(0.5) == 0.0 and reach(5.0) == 1.0
assert 0 < reach(2.0) < 1
assert e.state_at(0.5)["cay"].impl.semantics(e.state_at(0.5))["reach"] == 0.0
```

## coset and quotient: G/H

`s.at(m).coset(object_id, subgroup=None)` gathers the elements of a `ClockDiagram` or
`CayleyGraph` into their cosets: each coset glides into a cluster, colored as its coset (the
subgroup keeps the accent color). `s.at(m).quotient(object_id, subgroup=None)` collapses each
coset to a single node, the picture of the quotient group, labelled `H`, `r+H`, ... The
subgroup must be normal. They work with or without each other, in either written order.

`subgroup` is a step (or a list of elements) for a clock, labels such as `"r2"` for a Cayley
graph (which needs one: it has no default). If the diagram has none yet the verb sets it;
a diagram shows one subgroup only.

```python
s = Scene("coset_demo", passes=None)
s.add(algebra.ClockDiagram(id="clock", n=12, step=4))
s.marker("gather", 1.0)
s.marker("collapse", 5.0)
s.at("gather").coset("clock")
s.at("collapse").quotient("clock")
e = s.engine

def points(t):
    state = e.state_at(t)
    return state["clock"].impl.features(state)["points"]

cosets = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
assert not np.allclose(points(0.5), points(4.0))  # the elements moved into clusters
end = points(9.0)
for c in cosets:  # each coset is one node
    assert np.allclose(end[c], end[c[0]])
```

A Cayley graph of D4 modulo its rotation by 180 degrees:

```python
s = Scene("quotient_demo", passes=None)
s.add(algebra.CayleyGraph(id="cay", group="D4", generators="r s"))
s.marker("go", 1.0)
s.at("go").quotient("cay", subgroup="r2")
state = s.engine.state_at(5.0)
sem = state["cay"].impl.semantics(state)
assert sem["subgroup"] == ["e", "r2"] and len(sem["cosets"]) == 4
try:  # a subgroup that is not normal has no quotient
    s.at("go").quotient("cay", subgroup="s")
except ValueError as err:
    assert "not a normal subgroup" in str(err)
else:
    raise AssertionError("expected a refusal")
```

## permute: tokens swap places

`s.at(m).permute(permutation_id, sigma)`. The tokens of a `Permutation` swap places along
arcs, from where they are now to the arrangement `sigma`: token `i` ends in slot `sigma[i]`
(from 0), or write cycles with the numbers on the tokens, `"(1 2 3)(4 5)"`. A second permute
starts where the first left the tokens, with no jump. Tokens keep the color of the cycle they
move in. Two at the same moment are refused (use `.then()`); at different markers the order
written does not matter, because every arrangement is absolute.

```python
s = Scene("permute_demo", passes=None)
s.add(algebra.Permutation(id="p", perm=(0, 1, 2, 3, 4)))
s.marker("swap", 1.0)
s.marker("again", 5.0)
s.at("swap").permute("p", "(1 2 3)(4 5)")
s.at("again").permute("p", [0, 1, 2, 3, 4])
e = s.engine

def tokens(t):
    state = e.state_at(t)
    return state["p"].impl.features(state)["tokens"]

slots = e.state_at(0.0)["p"].impl.features(e.state_at(0.0))["slots"]
assert np.allclose(tokens(0.5), slots)
assert np.allclose(tokens(4.0), slots[[1, 2, 0, 4, 3]])  # token 1 is in slot 2, ...
assert np.allclose(tokens(9.0), slots)  # and back home
assert abs(tokens(1.8)[0][1]) > 0.05  # on an arc, not a straight slide
```
