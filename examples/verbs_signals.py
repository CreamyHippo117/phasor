"""Iterating and decomposing: Newton's method walks to a root of z^3 - 1 over the plane
coloured by that polynomial, the logistic map climbs its cobweb, and a square wave is taken
apart into its harmonics and put back together.

Run:  uv run python examples/verbs_signals.py      (prints what the inspector reads)
"""

from phasor import Scene, Text, cx, dyn, fourier

s = Scene("verbs_signals", look="prism")
s.set("camera.height", 10.0)

s.add(Text(id="caption", text="iterate and decompose", pos=(-4.5, 4.4, 0)))
s.add(cx.DomainColoring(id="plane", function="z**3 - 1", pos=(-4.4, 0.6, 0), size=(7.0, 6.0),
                        preset="prism"))  # fmt: skip
s.add(dyn.Orbit(id="newton", kind="newton", function="z**3 - 1", start=(0.45, 0.9), steps=10,
                pos=(-4.4, 0.6, 0)))  # fmt: skip
s.add(dyn.Orbit(id="logistic", kind="cobweb", function="a*z*(1 - z)", a=(3.7, 0.0),
                start=(0.2, 0.0), steps=20, pos=(0.4, -4.4, 0), scale=3.0))  # fmt: skip
s.add(fourier.PartialSum(id="wave", kind="square", terms=5, pos=(6.0, 3.0, 0), extent=(4.2, 1.2)))

for name, t in (("start", 0.0), ("newton", 0.5), ("cobweb", 4.0), ("apart", 9.5),
                ("together", 12.5), ("end", 15.0)):  # fmt: skip
    s.marker(name, t)
s.at("start").reveal("plane", "wave")
s.at("newton").iterate("newton")
s.at("cobweb").iterate("logistic")
s.at("apart").decompose("wave")
s.at("together").recompose("wave")

if __name__ == "__main__":
    e = s.engine
    print("newton", e.inspect(4.0, id="newton")["objects"][0]["semantic"])
    print("wave", e.inspect(11.0, id="wave")["objects"][0]["semantic"])
    print([i for i in e.lint(strict=True)])
