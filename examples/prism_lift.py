"""A spatial transition: a flat picture tilts up and becomes a landscape (the "lift").

A drum skin vibrating in its (2, 3) mode is first seen straight from above, like a heat map:
hue for height, the six cells pulsing. Then, without a cut, the view gains depth and tilts
over, and the picture rises into the landscape it always was.

How it stays seamless (a dolly zoom): the scene uses the 3D orbit camera from the start, but
looking almost straight down with a very narrow field of view from far away, which looks the
same as the flat 2D camera. ``camera.distance`` is an expression of ``camera.fov`` that keeps
the plane through the target exactly ``FRAME`` world units tall on screen, so widening the
field of view brings the camera closer without changing what is framed: perspective fades in
with no jump. Only then does the camera tilt over and the surface rise.

Run:  uv run python examples/prism_lift.py      (prints how well the framing holds)
"""

import math

from phasor import Scene, expr
from phasor.domains import pde

FRAME = 7.5  # world units framed top to bottom at the target, held through the dolly
FLAT_FOV = 1.5  # degrees: at this field of view perspective is invisible

s = Scene("prism_lift", look="prism")
# azimuth -90 and elevation 89: x to the right and y up on screen, as in 2D
s.use_camera("orbit", target=(0.0, 0.0, 0.0), azimuth=-90.0, elevation=89.0,
             fov=FLAT_FOV, drift=0.0)  # fmt: skip
s.set("camera.distance", expr(f"{FRAME} / (2 * tan(camera.fov * pi / 360))"))
s.add(pde.Surface(id="drum", kind="wave", mode=(2, 3), scale=1.2, height=0.01, n=64,
                  time=expr("(t + 3.35) / 4"), size=(6.0, 6.0)))  # fmt: skip
# the phase is shifted so the drum is at full swing when the camera lands (t = 10)

for name, t in (("start", 0.0), ("depth", 3.0), ("tilt", 5.5), ("rise", 6.5),
                ("landed", 10.0), ("end", 16.0)):  # fmt: skip
    s.marker(name, t)

s.at("start").reveal("drum")
s.at("depth").morph("camera.fov", 35.0, until="tilt")  # perspective fades in, framing held
s.at("tilt").morph("camera.elevation", 32.0, until="landed")  # the plane tips over
s.at("rise").morph("drum.height", 1.6, until="landed")  # and the picture rises
s.at("landed").orbit(60.0, until="end")


def framing_error(e, t, width=1920, height=1080):
    """Pixels the drum's corners move from where the flat view puts them (0 before the tilt)."""
    corners = [(-3.0, -3.0, 0.0), (3.0, -3.0, 0.0), (-3.0, 3.0, 0.0), (3.0, 3.0, 0.0)]
    flat = e.state_at(0.0).camera.to_screen(corners, width, height)
    now = e.state_at(t).camera.to_screen(corners, width, height)
    return float(abs(now - flat).max())


if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 3.0, 3.6, 4.25, 4.9, 5.5):
        v = e.state_at(t).values
        print(f"t={t:4.2f}  fov {v['camera.fov']:5.2f}  distance {v['camera.distance']:7.2f}  "
              f"corners off by {framing_error(e, t):.2f} px")  # fmt: skip
    assert math.isclose(e.state_at(9.99).values["camera.elevation"], 32.0, abs_tol=0.5)
    for issue in e.lint(strict=True):
        print("lint:", issue)
