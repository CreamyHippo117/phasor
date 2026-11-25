"""The scale verbs, one beat each: a point grows and shrinks back, an equation is squished
and restored, a square is flattened to a line and comes back, and a domain-colored plane is
squeezed sideways (its colors squeeze with it).

Run:  uv run python examples/verbs_scale.py      (prints what the inspector reads)
"""

from phasor import Dot, Equation, Scene, Text, cx
from phasor.domains.algebra import Polygon

s = Scene("verbs_scale")
s.set("camera.height", 6.0)

s.add(Text(id="caption", text="grow, shrink, squish, flatten", pos=(0, 2.5, 0)))
s.add(Dot(id="point", pos=(-3.8, 0.0, 0), r=0.35))
s.add(Equation(id="law", parts=("$e^{i\\pi}$", "$+1=0$"), colors=("primary", "secondary"),
               pos=(0, 1.3, 0), style={"size": "title"}))  # fmt: skip
s.add(Polygon(id="square", n=4, r=0.9, pos=(3.8, 0.0, 0)))
s.add(
    cx.DomainColoring(id="plane", function="(z + 0.9j)**2 - 1", pos=(0, -0.9, 0), size=(3.2, 1.8))
)

for name, t in (("start", 0.0), ("grow", 0.5), ("shrink", 1.7), ("squish", 2.9),
                ("restore", 4.1), ("flatten", 5.3), ("back", 6.7), ("end", 8.0)):  # fmt: skip
    s.marker(name, t)
s.at("grow").grow("point", by=1.8)
s.at("shrink").shrink("point", by=1 / 1.8)
s.at("squish").squish("law", axis="y", to=0.4)
s.at("restore").squish("law", axis="y", to=1.0).squish("plane", axis="x", to=0.5)
s.at("flatten").flatten("square", axis="y")
s.at("back").squish("square", axis="y", to=1.0).squish("plane", axis="x", to=1.0)

if __name__ == "__main__":
    e = s.engine
    for oid in ("point", "law", "square", "plane"):
        print(oid, e.where(oid, 6.0)["bbox"]["norm"])
