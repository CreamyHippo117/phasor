"""Fills, dashes, boolean operations and an SVG drawing (Milestone 16).

- Top row: two discs combined four ways (``shapes.union``, ``intersection``, ``difference``,
  ``exclusion``). The discs themselves are dashed outlines; each result is a filled path.
- Bottom left: ``assets/emblem.svg`` read as ``Path`` parts named from its ids
  (``emblem.ring``, ``emblem.star``, ``emblem.badge_1``, ``emblem.wave``). Edit the file and
  the scene follows it.
- Bottom right: a frame with a hole (two subpaths, the even-odd rule) and a dotted path with
  an arrow tip, traced in.

Booleans need shapely and the drawing svgelements: `uv sync --extra boolean --extra svg`.

Run:  uv run python examples/vector_toolkit.py        (prints what the inspector sees)
"""

from pathlib import Path as FilePath

from phasor import Scene, Text, shapes
from phasor.domains import path_math as pm

s = Scene("vector_toolkit", look="prism", base_dir=FilePath(__file__).parent)
s.set("camera.height", 9)

ops = ("union", "intersection", "difference", "exclusion")
xs = (-5.4, -1.8, 1.8, 5.4)
y = 2.0
for op, x in zip(ops, xs, strict=True):
    a = shapes.Circle(id=f"{op}_a", pos=(x - 0.45, y), r=0.8,
                      style={"color": "muted", "stroke": "hairline", "dash": "dashed"})  # fmt: skip
    b = shapes.Circle(id=f"{op}_b", pos=(x + 0.45, y), r=0.8,
                      style={"color": "muted", "stroke": "hairline", "dash": "dashed"})  # fmt: skip
    s.add(a)
    s.add(b)
    look = {"color": "secondary", "glow": "soft", "fill": "soft", "fill_color": "secondary"}
    s.add(getattr(shapes, op)(a, b, id=op, style=look))
    s.add(Text(id=f"{op}.label", text=op, style={"size": "annotation"}))
    s.constrain(f"label({op}) below {op} gap 0.25")

s.add(shapes.SVG(id="emblem", path="assets/emblem.svg", pos=(-3.0, -2.2), height=3.2,
                 style={"color": "highlight", "fill": "soft", "fill_color": "highlight",
                        "glow": "soft"}))  # fmt: skip

outer = pm.rectangle((3.6, -2.2, 0), 3.0, 2.4, radius=0.25)
inner = pm.rectangle((3.6, -2.2, 0), 1.8, 1.2)
s.add(shapes.Path(id="frame", subpaths=[outer.tolist(), inner.tolist()], rule="evenodd",
                  style={"color": "primary", "fill": "soft", "glow": "soft"}))  # fmt: skip
s.add(shapes.Path.through("route", [(2.3, -3.9), (3.6, -3.6), (4.9, -3.9)], tips="end",
                          style={"color": "danger", "dash": "dotted", "glow": "soft"}))  # fmt: skip

s.marker("operands", 0.3)
s.marker("more_operands", 1.1)
s.marker("results", 2.0)
s.marker("names", 3.8)
s.marker("drawing", 4.6)
s.marker("frame", 6.0)
s.marker("end", 8.4)
s.at("operands").reveal(*(f"{op}_{k}" for op in ops[:2] for k in "ab"))
s.at("more_operands").reveal(*(f"{op}_{k}" for op in ops[2:] for k in "ab"))
s.at("results").trace(*ops)
s.at("names").reveal(*(f"{op}.label" for op in ops))
s.at("drawing").reveal("emblem")
s.at("frame").trace("frame", "route")

if __name__ == "__main__":
    for line in s.engine.inspect(8.0, detail="brief")["objects"]:
        print(line)
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
