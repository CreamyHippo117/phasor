"""A scene in its own colors: explicit values beside roles.

Two colors, a text size, a stroke width and the background are typed into the scene instead
of named as roles. Roles stay the default (and what an AI should write); a typed value is
allowed, and still checked: ``lint(aesthetic=True)`` warns with ``off_palette`` when a color
is far from every role of the theme's palette, and with ``off_scale`` when a size is off the
theme's type or stroke scale. ``polish(snap=True)`` snaps them to the nearest roles, and
nothing does that unless asked.

Here the cyan of the inner planet and the navy background are near the prism palette (no
warning); the green of the outer planet and the wide orbit are not (one warning each).

Run:  uv run python examples/own_colors.py     (prints the taste issues and the snaps offered)
"""

from phasor import Dot, Line, Scene, Text

s = Scene("Own colors", look="prism", background="#0E1420")
s.set("camera.height", 8)

# An orbit drawn in one typed color (no hue running along it), much wider than "bold".
s.add(Line(id="orbit", points=[(-5.5, -2.2), (5.5, -2.2)], hue_along=0.0,
           style={"color": "#4A5672", "stroke": 0.12}))  # fmt: skip
s.add(Dot(id="inner", pos=(-2.2, 0.2), r=0.42, style={"color": "#58C4DD", "glow": "strong"},
          tags=["key"]))  # fmt: skip
s.add(Dot(id="outer", pos=(2.4, -0.3), r=0.3, style={"color": "#83C167"}))
s.add(Text(id="title", text="Two worlds", pos=(0, 2.9), style={"size": 0.6}))  # = title role
s.add(Text(id="inner.label", text="inner", style={"color": "#58C4DD"}))
s.add(Text(id="outer.label", text="outer", style={"color": "#83C167"}))
s.constrain("label(inner) near inner")
s.constrain("label(outer) near outer")

s.marker("start", 0.3)
s.marker("worlds", 1.4)
s.marker("end", 3.5)
s.at("start").reveal("title", "orbit")
s.at("worlds").reveal("inner", "outer", "inner.label", "outer.label")

if __name__ == "__main__":
    e = s.engine
    for issue in e.lint(aesthetic=True):
        if issue.code in ("off_palette", "off_scale"):
            print("lint:", issue)
    for i in e.critique()["issues"]:
        for op in i["snap"]:
            print("offered (polish(snap=True) applies it):", op)
