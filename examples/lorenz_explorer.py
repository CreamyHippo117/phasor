"""The Lorenz system, as the Explorer shows it (Milestone 26 exit demo).

x' = sigma (y - x), y' = x (rho - z) - y, z' = x y - beta z with sigma = 10, rho = 27.7,
beta = 2.67. The field is a grid of cones coloured by speed; the three equilibria are the red
balls (all unstable: the origin is a saddle, C+ and C- are saddle-foci); a ball of 2000
initial conditions near (1, 1, 20) is carried by the flow from tau = 0 to 5 and folds onto
the two wings. C+ is shown close up: its one stable direction (cool line), its unstable
plane (the warm disc, where the motion spirals out) and a card with the numbers:
eigenvalues -13.84 and 0.0852 +/- 10.15i.

Run:  uv run python examples/lorenz_explorer.py      (prints the equilibria)
"""

from phasor import Scene, Text, flow

s = Scene("lorenz_explorer", look="prism")
s.use_camera("orbit")
s.set("camera.target", (0.0, 0.0, 0.0))
s.set("camera.distance", 15.0)
s.set("camera.elevation", 16.0)
s.set("camera.azimuth", -62.0)

# Coordinates are drawn at 0.12 world units each, the window's middle near the origin.
s.add(flow.System(id="lorenz", scale=0.12, pos=(0, 0, -3.2),
                  **flow.preset("lorenz", rho=27.7, beta=2.67)))  # fmt: skip
s.add(flow.Variable(id="rho", name="rho", value=27.7, min=0, max=60))
s.add(flow.FieldArrows(id="lorenz.field", counts=7, opacity=0.0))
s.add(flow.Equilibria(id="lorenz.eq", reveal=0.0))
s.add(flow.FlowSet(id="lorenz.ball", shape="ball", center=(1, 1, 20), size=2.5, count=2000,
                   opacity=0.0))  # fmt: skip
s.add(flow.EquilibriumView(id="lorenz.cplus", near=(8, 8, 27), radius=5, name="C+",
                           opacity=0.0))  # fmt: skip
s.add(Text(id="title", text="the Lorenz flow", pos=(8 / 9, 0.07), layer="overlay",
           style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("rest", 1.2)
s.marker("ball", 2.4)
s.marker("fold", 3.2)
s.marker("card", 7.2)
s.marker("end", 10.5)
s.at("start").reveal("title", "lorenz.field")
s.at("rest").sweep("lorenz.eq.reveal", 1.0, show=False)  # the equilibria, one by one
s.at("ball").reveal("lorenz.ball")
s.at("fold").sweep("lorenz.tau", 5.0, show=False)  # the flow carries the ball, tau 0 -> 5
s.at("card").reveal("lorenz.cplus")
s.at("card").exit("lorenz.field")  # out of the way of the close-up
s.at("card").focus_pull("lorenz.cplus", 1.2)  # zoom in on C+

if __name__ == "__main__":
    e = s.engine
    sem = e.inspect(e.duration, id="lorenz.eq")["objects"][0]["semantic"]
    for pt in sem["points"]:
        print(pt["at"], pt["kind"], "eigenvalues", pt["eigenvalues"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
