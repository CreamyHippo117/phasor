"""Five verbs for space and points: a plane rises into a landscape (lift), a plane slices a
saddle (slice), an arrow casts its shadow on another (project), a star shatters into light
(dissolve) and comes back (assemble).

Top left, a domain coloring with two poles rises into its landscape: height is log(1 + |f|),
colour is the phase, and the poles are flat-topped towers. Top right, a
cutting plane sweeps along a saddle, drawing the curve where they meet. Bottom left, an
arrow is projected onto another: a dashed perpendicular drops, the projection grows. Bottom
right, a star dissolves into points that each drift their own way, then flies back together.

Run:  uv run python examples/verbs_space.py      (prints what the inspector reads)
"""

import math

from phasor import Line, Scene, cx, geo, linalg

s = Scene("verbs_space", look="prism")
s.use_camera("orbit")
s.set("camera.target", (0.0, 0.0, 0.0))
s.set("camera.distance", 16.5)
s.set("camera.elevation", 38.0)
s.set("camera.azimuth", -78.0)

# the plane is where the complex numbers are, so poles at -4.2 + 2.4i -+ i sit in its middle
s.add(cx.DomainColoring(id="plane", function="1/((z + 4.2 - 2.4*i)**2 + 1)", pos=(-4.2, 2.4, 0),
                        size=(5.6, 3.2), radius=0.35, ambient=0.06, contour=0.0,
                        lines=12))  # fmt: skip
s.add(geo.ParametricSurface(id="saddle", fx="u", fy="v", fz="0.16*(u*u - v*v)", color_by="height",
                            u_range=(-2.2, 2.2), v_range=(-2.2, 2.2), pos=(4.2, 2.4, 0),
                            res=(36, 36)))  # fmt: skip
s.add(linalg.Arrow(id="axis", start=(-6.0, -2.8, 0), end=(-2.0, -2.8, 0)))
s.add(linalg.Arrow(id="vec", start=(-6.0, -2.8, 0), end=(-3.2, -0.9, 0)))
star = [(4.2 + (1.6 if k % 2 == 0 else 0.7) * math.cos(math.pi * k / 5 + math.pi / 2),
         -2.4 + (1.6 if k % 2 == 0 else 0.7) * math.sin(math.pi * k / 5 + math.pi / 2), 0.0)
        for k in range(11)]  # fmt: skip
s.add(Line(id="star", points=star + star[:1]))

for name, t in (("start", 0.0), ("rise", 1.0), ("cut", 4.0), ("shadow", 7.0), ("shatter", 10.0),
                ("rejoin", 13.0), ("end", 16.0)):  # fmt: skip
    s.marker(name, t)
s.at("rise").lift("plane", height=0.9)
s.at("cut").slice("saddle", axis="y")
s.at("shadow").project("vec", onto="axis", label="$\\mathrm{proj}$")
s.at("shatter").dissolve("star")
s.at("rejoin").assemble("star")

if __name__ == "__main__":
    e = s.engine
    for oid, t in (("plane.surface", 4.0), ("saddle.cut", 6.0), ("vec.proj", 9.0),
                   ("star.dust", 11.5)):  # fmt: skip
        print(oid, e.inspect(t, id=oid)["objects"][0]["semantic"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
