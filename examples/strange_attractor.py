"""The Lorenz trajectory draws two lobes as the camera circles.

Hue follows simulated time along the growing curve. Set system to "rossler"
or "thomas" to trace either of those flows with the same animation.
"""

from phasor import Scene, dyn

s = Scene("strange_attractor", look="prism")
s.use_camera("orbit", azimuth=-95, elevation=12, distance=13.5)
trajectory = s.add(dyn.StrangeAttractor(id="trajectory", seed=7, t_end=30, progress=0.0))
s.marker("start", 0.0)
s.marker("trace", 0.5)
s.marker("end", 7.0)
s.at("trace").trace("trajectory")
s.at("start").orbit(70, until="end")
