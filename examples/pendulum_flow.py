"""A damped pendulum as the Explorer shows it: a 2D flow with a slider.

x is the angle and y the angular speed: x' = y, y' = -sin(x) - gamma y. ``gamma`` is a
``Variable`` with a slider (a ``Control`` on ``gamma.value``): it overrides the constant the
system names. The separatrices are the stable and unstable manifolds of the upright saddles
(``Manifold``); a probe starts at (1.5, 1.5) and spirals into the hanging rest point while
its label follows it; a disk of initial conditions, outline and all, is carried along and
shrinks (the divergence is -gamma, so areas shrink like exp(-gamma tau)).

Run:  uv run python examples/pendulum_flow.py      (prints the rest points)
"""

from phasor import Scene, Text, flow

s = Scene("pendulum_flow", look="prism")

s.add(flow.System(id="pend", equations=("y", "-sin(x) - gamma*y"), constants="gamma=0.25",
                  window=((-6.5, 6.5), (-3.2, 3.2)), dt=0.02, pos=(0, -0.3, 0),
                  scale=0.9))  # fmt: skip
s.add(flow.Variable(id="gamma", name="gamma", value=0.25, min=0.0, max=1.0))
s.control("gamma.value", min=0.0, max=1.0, label="damping")
s.add(flow.FieldArrows(id="pend.field", counts=21, opacity=0.0))
s.add(flow.Equilibria(id="pend.rest", reveal=0.0))
s.add(flow.Manifold(id="pend.sep_right", near=(3.14, 0), length=12.0, progress=0.0))
s.add(flow.Manifold(id="pend.sep_left", near=(-3.14, 0), length=12.0, progress=0.0))
s.add(flow.FlowSet(id="pend.disk", shape="disk", center=(-1.5, 2.0), size=0.6, count=300,
                   fill="both", opacity=0.0))  # fmt: skip
s.add(flow.Probe(id="pend.probe", x0=(1.5, 1.5), label="phi^t(x0)", opacity=0.0))
s.add(Text(id="title", text="a damped pendulum, as a flow", pos=(8 / 9, 0.07),
           layer="overlay", style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("rest", 1.0)
s.marker("seps", 2.0)
s.marker("flow", 5.0)
s.marker("end", 10.5)
s.at("start").reveal("title", "pend.field")
s.at("rest").sweep("pend.rest.reveal", 1.0, show=False)
s.at("seps").sweep("pend.sep_right.progress", 1.0, show=False)
s.at("seps").sweep("pend.sep_left.progress", 1.0, show=False)
s.at("flow").reveal("pend.disk", "pend.probe")
s.at("flow").sweep("pend.tau", 5.0, show=False)

if __name__ == "__main__":
    e = s.engine
    sem = e.inspect(e.duration, id="pend.rest")["objects"][0]["semantic"]
    for pt in sem["points"]:
        print(pt["at"], pt["kind"], "eigenvalues", pt["eigenvalues"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
