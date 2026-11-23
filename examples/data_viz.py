"""Data that moves: bars that grow from their baseline, a total that counts up with them,
and the table the numbers came from, filled in row by row.

``grow`` (bars) and ``reveal`` (rows) are parameters like any other, so the verbs animate
them; the ``Counter`` follows a ``Value`` through an expression, grouping its thousands.

Run:  uv run python examples/data_viz.py        (prints what the inspector sees)
"""

from phasor import Counter, Scene, Text, Value, dg, expr

visits = (1200, 3400, 2600, 4100)
months = ("Jan", "Feb", "Mar", "Apr")

s = Scene("data_viz", look="prism")
s.set("camera.height", 10)
s.add(dg.BarChart(id="visits", values=visits, names=months, pos=(-2.6, 0.2, 0), width=6.4,
                  height=4.6, grow=0.0, colors=("secondary",)))  # fmt: skip
rows = [(m, f"{v:,}") for m, v in zip(months, visits, strict=True)]
s.add(dg.Table(id="source", cells=(("month", "visits"), *rows), pos=(5.2, 0.9, 0), reveal=0.0))
s.add(Value(id="total", value=0.0))
s.add(Counter(id="shown", value=expr("total.value"), digits=0, group=True, prefix="\\Sigma = ",
              pos=(5.2, -2.3, 0)))  # fmt: skip
s.add(Text(id="title", text="visits by month", pos=(0.5, 0.07), layer="overlay",
           style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("rows", 1.0)
s.marker("grow", 2.6)
s.marker("end", 6.0)
s.at("start").reveal("title", "visits", "source", "shown")
s.at("rows").morph("source.reveal", 1.0, until="grow")
s.at("grow").morph("visits.grow", 1.0, until="end")
s.at("grow").morph("total.value", float(sum(visits)), until="end")

if __name__ == "__main__":
    e = s.engine
    for row in e.inspect(5.0, detail="brief")["objects"]:
        print(row)
    for issue in e.lint(strict=True):
        print("lint:", issue)
