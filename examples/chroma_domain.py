"""Domain coloring in the chroma look: a luminous wash on deep indigo.

f(z) = (z^3 - 1) / (z^2 + 0.5 i) has three zeros, the cube roots of 1, and two poles. Phase is
hue from a vivid wheel (red, yellow, green, cyan, violet, magenta); light gathers round the
zeros and poles and runs white-hot there (``white``), falling away to the indigo ground. No
rays or rings: a soft wash. The constant drifts so the zeros and poles move, and the colours
follow them. The look is ``chroma`` (``cx.DomainColoring`` preset ``chroma``).

Run:  uv run python examples/chroma_domain.py      (prints the zeros and poles it shows)
"""

from phasor import Equation, Scene, cx

F = "(z**3 - a)/(z**2 + 0.5j)"

s = Scene("chroma_domain", look="chroma")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(cx.DomainColoring(id="plane", function=F, a=(1.0, 0.0), size=(8.0, 4.5)))
s.add(Equation(id="f_eq", pos=(0.0, -1.85, 0.0),
               parts=(r"$f(z) = \dfrac{z^3 - a}{z^2 + 0.5\,i}$",)))  # fmt: skip

s.marker("appear", 0.3)
s.marker("drift", 2.0)
s.marker("end", 8.0)
s.at("appear").reveal("plane")
s.at("appear", 0.8).write("f_eq")
s.at("drift").morph("plane.a", (0.4, 0.9), until="end")

if __name__ == "__main__":
    sem = s.engine.inspect(0.0, id="plane")["objects"][0]["semantic"]
    print("zeros:", sem["zeros"])
    print("poles:", sem["poles"])
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
