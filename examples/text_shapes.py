"""Text as shapes: a word written outline-first, a word that becomes a circle, a formula that
rearranges piece by piece, and a color that moves between two roles.

- ``Text(..., as_shapes=True)`` draws glyph outlines (one ``Path`` part per glyph), and
  ``write`` draws each glyph's outline, then fills it.
- ``transform`` morphs any outline into any other: here the word "circle" into a circle.
- ``transform_matching`` moves the pieces two formulas share to their new places (by TeX)
  and fades the rest; ``b^2``, carried past the ``=``, arcs round it.
- ``morph("disc.style.fill_color", "danger")`` moves a style key between roles; the color
  mixes in linear light, and the document keeps a role at the key.

Run:  uv run python examples/text_shapes.py        (prints what the inspector sees)
"""

from phasor import Equation, Scene, Text, shapes

s = Scene("text_shapes", look="prism")
s.set("camera.height", 8)

s.add(Text(id="title", text="Shapes", pos=(0, 3.0), as_shapes=True, style={"size": "title"}))
s.add(Text(id="word", text="circle", pos=(-3.6, 1.0), style={"size": "title"}))
s.add(
    shapes.Circle(
        id="disc",
        pos=(3.6, 1.0),
        r=1.1,
        style={"glow": "soft", "color": "highlight", "fill": "soft", "fill_color": "highlight"},
    )
)
s.add(
    Equation(
        id="before",
        parts=("$a^2$", "$+$", "$b^2$", "$=$", "$c^2$"),
        pos=(0, -2.2),
        as_shapes=True,
        style={"size": "title"},
    )
)
s.add(
    Equation(
        id="after",
        parts=("$a^2$", "$=$", "$c^2$", "$-$", "$b^2$"),
        colors=("primary", "primary", "primary", "primary", "secondary"),
        pos=(0, -2.2),
        as_shapes=True,
        style={"size": "title"},
    )
)

s.marker("write", 0.4)
s.marker("morph", 2.6)
s.marker("rearrange", 4.8)
s.marker("recolor", 7.0)
s.marker("end", 9.0)
s.at("write").write("title", "word", "before")
s.at("morph").transform("word", "disc")
s.at("rearrange").transform_matching("before", "after")
s.at("recolor").morph("disc.style.fill_color", "danger")
s.at("recolor").morph("disc.style.color", "danger")

if __name__ == "__main__":
    for line in s.engine.inspect(8.5, detail="brief")["objects"]:
        print(line)
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
