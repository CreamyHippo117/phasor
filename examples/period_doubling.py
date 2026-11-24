"""Period doubling in the Rossler system: a bifurcation diagram (Milestone 30).

x' = -y - z, y' = x + a y, z' = b + z (x - c) with a = b = 0.2 and c swept from 2 to 6. For
each of 96 values of c the flow runs from (1, 1, 0) for 150 time units (the transient), then
every local maximum of x is a dot above that c. One dot per c is a simple cycle; the line
splits in two (the cycle now closes after two loops), in four, ... and dissolves into a
band: chaos, with windows of order inside it (a period-3 window near c = 5.3). The same
picture as the logistic map's, from a flow. The candidates are where the period changes
between neighbouring values (period doubling, the onset of chaos); they are not refined.

On the right: the cycle at c = 3.5, which closes after two loops (``LimitCycle``: period
about 11.5), and its two dots on the plane y = 0 (``Section``).

Not detected by a sweep (and said so in its semantics): global bifurcations (homoclinic
loops, crises) and anything between two swept values.

Run:  uv run python examples/period_doubling.py      (prints the period per value)
"""

from phasor import Scene, Text, flow

s = Scene("period_doubling", look="prism")

s.add(flow.System(id="ro2", scale=0.2, pos=(4.4, -0.6, 0), tau=160.0, frame=False,
                  **flow.preset("rossler", c=3.5)))  # fmt: skip
s.add(flow.BifurcationDiagram(id="cascade", system="ro2", variable="c", range=(2.0, 6.0),
                              count=96, mode="attractor", coord="x", x0=(1, 1, 0),
                              transient=150.0, sample_time=100.0, pos=(-6.6, -3.0, 0),
                              extent=(8.2, 5.6), opacity=0.0))  # fmt: skip
s.add(flow.LimitCycle(id="ro2.cycle", x0=(1, 1, 0), transient=100.0, max_time=200.0,
                      tol=1e-5, progress=0.0))  # fmt: skip
s.add(flow.Section(id="ro2.section", seeds=((1, 1, 0),), normal=(0, 1, 0), direction="down",
                   transient=100.0, radius=0.01, plane=False, opacity=0.0))  # fmt: skip
s.add(Text(id="title", text="period doubling: maxima of x against c", pos=(8 / 9, 0.07),
           layer="overlay", style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("cascade", 1.0)
s.marker("cycle", 3.5)
s.marker("end", 8.0)
s.at("start").reveal("title")
s.at("cascade").reveal("cascade")
s.at("cycle").sweep("ro2.cycle.progress", 1.0, show=False)
s.at("cycle").reveal("ro2.section")

if __name__ == "__main__":
    e = s.engine
    sem = e.inspect(e.duration, id="cascade")["objects"][0]["semantic"]
    for value, period, _ in sem["rows"]:
        print(f"c = {value:.4f}: period {period}")
    for issue in e.lint(strict=True):
        print("lint:", issue)
