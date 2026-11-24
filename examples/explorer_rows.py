"""Explorer rows as a scene: typed lines of maths become document objects.

The rows are what a person types in the Explorer: two ODE rows (a Van der Pol oscillator),
a variable with a slider, a point that follows the flow, a graph (read and kept, not drawn
yet) and a row that does not read (kept with its error, a ``flow.Note``). One commit is one
patch (``explorer.plan``); the list comes back from the document (``explorer.rows_of``).

Run:  uv run python examples/explorer_rows.py      (prints the rows)
"""

from phasor import Scene, Text
from phasor.workspace import explorer

s = Scene("explorer_rows", look="prism")

ops, report = explorer.plan(s.engine.document, [
    {"text": "x' = y"},
    {"text": "y' = mu (1 - x^2) y - x"},
    {"text": "mu = 1.5"},
    {"text": "(2, 0)"},
    {"text": "y = x^2"},
    {"text": "q = sin x"},
])  # fmt: skip
assert s.engine.apply(ops, origin="script").ok, report
s.set("system.rate", 1.0)
s.set("point.trail", 3.0)
s.add(Text(id="title", text="typed rows, drawn", pos=(8 / 9, 0.07), layer="overlay",
           style={"size": "title"}))  # fmt: skip
s.marker("end", 6.0)

if __name__ == "__main__":
    for row in explorer.rows_of(s.engine.document):
        print(f"{row['id']:10} {row['kind']:10} {row['text']:28} {row['error'] or ''}")
