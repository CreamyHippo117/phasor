"""A gallery of shapes: every outline phasor draws on ``Path``, traced in one after another.

Each is a path of cubic Béziers with a fill (style ``fill``: none, soft, solid, its opacity
from the theme) and a stroke (``stroke``, ``glow``, ``dash``). ``trace`` draws a shape from
its start, and fades its fill in as it goes.

Run:  uv run python examples/shapes.py        (prints what the inspector sees)
"""

from phasor import Scene, Text, shapes

s = Scene("shapes", look="prism")
s.set("camera.height", 9)

glow = {"glow": "soft"}
row = (2.7, -0.1, -2.8)
col = (-5.2, 0.0, 5.2)
s.add(
    shapes.Circle(
        id="circle",
        pos=(col[0], row[0]),
        r=0.95,
        style={**glow, "color": "secondary", "fill": "soft", "fill_color": "secondary"},
    )
)
s.add(
    shapes.Ellipse(
        id="ellipse",
        pos=(col[1], row[0]),
        width=2.8,
        height=1.5,
        rotation=15,
        style={**glow, "color": "highlight", "dash": "dashed"},
    )
)
s.add(
    shapes.Arc(
        id="arc",
        pos=(col[2], row[0] - 0.5),
        r=1.2,
        start=20,
        angle=140,
        tips="end",
        style={**glow, "color": "primary"},
    )
)
s.add(
    shapes.Sector(
        id="sector",
        pos=(col[0] - 0.5, row[1] - 0.6),
        r=1.4,
        start=0,
        angle=60,
        style={**glow, "color": "highlight", "fill": "soft", "fill_color": "highlight"},
    )
)
s.add(
    shapes.Annulus(
        id="annulus",
        pos=(col[1], row[1]),
        outer=1.0,
        inner=0.55,
        style={**glow, "color": "secondary", "fill": "soft", "fill_color": "secondary"},
    )
)
s.add(
    shapes.Polygon(
        id="polygon",
        vertices=[
            (col[2] - 1.3, row[1] - 0.8),
            (col[2] + 1.2, row[1] - 0.6),
            (col[2] + 0.5, row[1] + 0.9),
            (col[2] - 0.6, row[1] + 0.4),
        ],
        style={**glow, "color": "danger", "dash": "dotted"},
    )
)
s.add(
    shapes.RegularPolygon(
        id="hexagon",
        n=6,
        pos=(col[0], row[2]),
        r=0.95,
        style={**glow, "color": "primary", "fill": "soft"},
    )
)
s.add(
    shapes.Rectangle(
        id="rectangle",
        pos=(col[1], row[2]),
        width=2.6,
        height=1.5,
        corner_radius=0.3,
        style={**glow, "color": "secondary"},
    )
)
s.add(
    shapes.Brace(
        id="brace",
        start=(col[2] - 1.2, row[2] + 0.2),
        end=(col[2] + 1.2, row[2] + 0.2),
        depth=0.45,
        style={**glow, "color": "highlight"},
    )
)

names = ["circle", "ellipse", "arc", "sector", "annulus", "polygon", "hexagon", "rectangle",
         "brace"]  # fmt: skip
for name in names:  # each name sits just under its shape (the layout solver puts it there)
    s.add(Text(id=f"{name}.label", text=name, style={"size": "annotation"}))
    s.constrain(f"label({name}) below {name} gap 0.12")

# One row at a time: the shapes trace in, then their names appear.
for k in range(3):
    s.marker(f"row{k}", 0.4 + 2.4 * k)
    s.marker(f"names{k}", 2.2 + 2.4 * k)
    s.at(f"row{k}").trace(*names[3 * k : 3 * k + 3])
    s.at(f"names{k}").reveal(*(f"{n}.label" for n in names[3 * k : 3 * k + 3]))
s.marker("end", 8.0)

if __name__ == "__main__":
    for line in s.engine.inspect(7.5, detail="brief")["objects"]:
        print(line)
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
