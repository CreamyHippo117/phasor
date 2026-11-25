"""The algebra verbs, one panel each: a symmetry turns and flips a square (act), a matrix acts
on a grid (apply), the Cayley graph of D4 grows from the identity (generate), a clock gathers
into its cosets and each coset becomes one node, Z12 / <4> (coset, quotient), and tokens swap
places along arcs (permute).

Run:  uv run python examples/verbs_algebra.py      (prints what the inspector reads)
"""

from phasor import Scene, Text, algebra, linalg

s = Scene("verbs_algebra", look="prism_everywhere")
s.set("camera.height", 10.0)

s.add(Text(id="caption", text="algebra verbs", pos=(0, 4.6, 0)))
s.add(algebra.Polygon(id="square", n=4, r=1.3, pos=(-5.5, 1.8, 0)))
s.add(linalg.TransformGrid(id="grid", extent=3, spacing=0.45, pos=(0, 1.8, 0)))
s.add(algebra.CayleyGraph(id="cayley", group="D4", generators="r s", r=1.7, pos=(5.5, 1.8, 0)))
s.add(algebra.ClockDiagram(id="clock", n=12, step=4, r=1.7, pos=(-5.5, -2.6, 0)))
s.add(algebra.Permutation(id="tokens", perm=(0, 1, 2, 3, 4), pos=(0, -2.6, 0)))

for name, t in (("turn", 1.0), ("flip", 3.5), ("gather", 5.5), ("collapse", 8.0), ("end", 11.0)):
    s.marker(name, t)

s.at("turn").act("square", "r").then().act("square", "s")
s.at("turn").apply("grid", ((1.0, 1.0), (0.0, 1.0)))  # a shear
s.at("flip").apply("grid", ((0.0, -1.0), (1.0, 0.0)))  # then a quarter turn
s.at("turn").generate("cayley")
s.at("turn").permute("tokens", "(1 2 3)(4 5)")
s.at("flip").permute("tokens", "(1 3 2)(4 5)")
s.at("gather").coset("clock")
s.at("collapse").quotient("clock")

if __name__ == "__main__":
    e = s.engine
    for oid in ("square", "grid", "cayley", "clock", "tokens"):
        print(oid, e.inspect(10.0, id=oid)["objects"][0]["semantic"])
