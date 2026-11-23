"""A gallery of the diagram kit: a number line, a number plane with a grid, axes with unequal
scales and labels, a bar chart, a table, a matrix and a block of code, revealed one by one.

Every number at a tick, bar, cell, matrix entry and line of code is a part of its own
(``plane.axes.x_2``, ``chart.bees``, ``table.b2``, ``matrix.e1_2``, ``code.line_3``), so the
inspector and lint see each one. Code is highlighted by pygments (the ``code`` extra) when
it is installed, and drawn plain otherwise.

Run:  uv run python examples/diagram_kit.py        (prints what the inspector sees)
"""

from phasor import Scene, Text, dg, geo

s = Scene("diagram_kit", look="prism")
s.set("camera.height", 13)

s.add(dg.NumberPlane(id="plane", x_range=(-3, 3, 1), y_range=(-2, 2, 1), pos=(-7.4, 2.6, 0),
                     scale=0.8, labels=("x", "y")))  # fmt: skip
s.add(geo.Axes(id="axes", dims=2, x_range=(0, 10, 2), y_range=(0, 1, 0.5), x_scale=0.42,
               y_scale=2.6, pos=(-2.0, 1.1, 0), numbers=True, labels=("t", "p")))  # fmt: skip
s.add(dg.BarChart(id="chart", values=(3, 5, 2, 4), names=("ants", "bees", "cats", "dogs"),
                  pos=(6.2, 2.4, 0), width=5.0, height=3.2, colors=("secondary", "highlight"),
                  show_values=True))  # fmt: skip
s.add(dg.NumberLine(id="line", x_range=(-4, 4, 1), pos=(0, -5.0, 0), scale=1.0, tips="both"))
s.add(dg.Table(id="table", cells=(("n", "$n^2$", "$2^n$"), ("1", "1", "2"), ("2", "4", "4"),
                                  ("3", "9", "8"), ("4", "16", "16")),
               pos=(-7.6, -2.3, 0)))  # fmt: skip
s.add(dg.MatrixDisplay(id="matrix", entries=(("\\cos\\theta", "-\\sin\\theta"),
                                             ("\\sin\\theta", "\\cos\\theta")),
                       pos=(-1.6, -2.2, 0)))  # fmt: skip
s.add(dg.Code(id="code", pos=(5.6, -2.3, 0), line_numbers=True,
              code='def squares(n):\n    """the first n"""\n    return [k * k for k in range(n)]\n'
                   "\nprint(squares(4))  # [0, 1, 4, 9]"))  # fmt: skip
s.add(Text(id="title", text="a diagram kit", pos=(0.5, 0.06), layer="overlay",
           style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("plots", 1.4)
s.marker("data", 3.0)
s.marker("end", 6.0)
s.at("start").reveal("title", "plane")
s.at("plots").reveal("axes", "chart", "line")
s.at("data").reveal("table", "matrix", "code")

if __name__ == "__main__":
    e = s.engine
    for row in e.inspect(5.0, detail="brief")["objects"]:
        print(row)
    for issue in e.lint(strict=True):
        print("lint:", issue)
