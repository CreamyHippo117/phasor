"""The Rossler attractor cut by a plane (Milestone 30, the Explorer's analysis pack).

x' = -y - z, y' = x + a y, z' = b + z (x - c) with a = b = 0.2, c = 5.7: a band that spirals
out and folds back. A cloud of 64 initial conditions is carried by the flow; every time one
of them passes down through the plane y = 0 a dot is left there (a Poincare section, each
crossing found inside its step), and after a while the dots draw the attractor's thin
cross-section. Two probes start a millionth apart: their distance grows like
exp(lambda t), and the strip in the corner shows log10 of it beside the line of slope
lambda, the largest Lyapunov exponent (Benettin's method, about 0.07 here). The upper strip
is x(t) and z(t) of one probe: the spikes in z are the folds.

The flow time runs to 5 first (the probes still together), then on to 150. A chaotic path is
not reproducible across machines past a few Lyapunov times, so the semantics give counts and
positions only up to flow time 10, and the exponent as a statistic with its spread.

Run:  uv run python examples/rossler_section.py      (prints the exponent)
"""

from phasor import Scene, Text, flow

s = Scene("rossler_section", look="prism")
s.use_camera("orbit")
s.set("camera.target", (0.0, 0.0, 0.0))
s.set("camera.distance", 9.0)
s.set("camera.elevation", 24.0)
s.set("camera.azimuth", -58.0)

s.add(flow.System(id="ro", scale=0.2, pos=(0, 0, -1.6), frame=False, **flow.preset("rossler")))
s.add(flow.FlowSet(id="ro.cloud", shape="ball", center=(1, 1, 0), size=1.0, count=64,
                   radius=0.004, opacity=0.0))  # fmt: skip
s.add(flow.Section(id="ro.section", of="ro.cloud", normal=(0, 1, 0), direction="down",
                   transient=10.0, opacity=0.0))  # fmt: skip
s.add(flow.Probe(id="ro.probe", x0=(1, 1, 0), trail=40.0, opacity=0.0))
s.add(flow.TwinProbes(id="ro.twins", x0=(1, 1, 0), eps=1e-6, lyapunov=300.0, trail=8.0,
                      dot=False, opacity=0.0))  # fmt: skip
s.add(flow.TimeSeries(id="ro.series", source="ro.probe", components="xz", span=60.0,
                      layer="overlay", pos=(0.06, 0.36), extent=(0.62, 0.2),
                      opacity=0.0))  # fmt: skip
s.add(flow.TimeSeries(id="ro.distance", source="ro.twins", layer="overlay",
                      pos=(0.06, 0.94), extent=(0.62, 0.2), opacity=0.0))  # fmt: skip
s.add(Text(id="title", text="Rossler: a section and two probes", pos=(8 / 9, 0.07),
           layer="overlay", style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("parts", 0.8)
s.marker("strips", 1.6)
s.marker("flow", 2.6)
s.marker("chaos", 6.0)
s.marker("end", 12.0)
s.at("start").reveal("title", "ro.section")
s.at("parts").reveal("ro.cloud", "ro.probe")
s.at("strips").reveal("ro.twins", "ro.series", "ro.distance")
s.at("flow").sweep("ro.tau", 5.0, show=False)  # together still: the golden frames sit here
s.at("chaos").sweep("ro.tau", 150.0, show=False)  # the section fills in, the twins part

if __name__ == "__main__":
    e = s.engine
    sem = e.inspect(e.duration, id="ro.twins")["objects"][0]["semantic"]
    print("largest Lyapunov exponent", sem["lyapunov"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
