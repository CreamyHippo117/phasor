"""A continuous shading band beside a plot axis: the modal overlap of a room, M(f), as a smooth
gradient next to a response ridge on a log-frequency axis (30 Hz to 1 kHz).

The band reads the whole array at draw time, so sliding it (the room is scaled) means moving
``pos``, not rebuilding anything. The gradient is the Schroeder frequency made visible: a
gradual change, not a border.

Run:  uv run python examples/band.py      (prints the band's semantics at both ends)
"""

import numpy as np

from phasor import Scene, expr, geo

F = np.geomspace(30.0, 1000.0, 200)
M = 80.0 * (F / 1000.0) ** 3  # modal overlap grows as f^3: ~0.1 at 30 Hz, 80 at 1 kHz

s = Scene("band")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 8.0)
# the ridge on the same axis: y = -4.8 + 2.4 * log10(f); the band shares that mapping
slide = expr("vec(0, -4.8 + 0.3 * clamp(t / 4, 0, 1), 0)")  # the room is scaled: the band moves
s.add(
    geo.Band(
        id="overlap",
        values=M,
        positions=F,
        log=True,
        axis="y",
        extent=(1.5, 2.3),
        scale=2.4,
        pos=slide,
        edge_fade=0.12,
    )
)
s.marker("start", 0.0)
s.marker("end", 4.0)

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 4.0):
        print(t, e.inspect(t, id="overlap")["objects"][0]["semantic"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
