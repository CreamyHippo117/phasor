"""A term that leaves a 3D object and slides into an overlay equation.

Each term of an equation is built from the object it names: here the volume of a sphere
becomes the first term of ``N = V_sphere + V_cell``, the second term flying in from the
cell. The camera orbits, and the equation lives on the overlay (screen space), so the verb
``lift_term(object, equation, part)`` starts the term where the object is *on the screen*
at that moment (its centre projected through the baked camera) and glides it into its place
in the line. Until its lift starts the term is not shown; its place in the line is reserved.

Run:  uv run python examples/lift_term.py      (prints where a term is mid-flight)
"""

from phasor import Equation, Scene, geo

s = Scene("lift_term", look="prism_everywhere")
s.use_camera("orbit")
s.set("camera.target", (0.0, 0.0, 0.0))
s.set("camera.distance", 16.0)
s.set("camera.elevation", 24.0)
s.set("camera.azimuth", -55.0)

s.add(geo.ParametricSurface(id="sphere", fx="1.6*sin(u)*cos(v)", fy="1.6*sin(u)*sin(v)",
                            fz="1.6*cos(u)", u_range=(0, 3.14159), v_range=(0, 6.28318),
                            res=(24, 48), color_by="height", pos=(-3.0, 0.0, 0.0)))  # fmt: skip
s.add(geo.ParametricSurface(id="cell", fx="1.2*sin(u)*cos(v)", fy="1.2*sin(u)*sin(v)",
                            fz="1.2*cos(u)", u_range=(0, 3.14159), v_range=(0, 6.28318),
                            res=(16, 32), color_by="height", pos=(3.0, 0.0, 0.0)))  # fmt: skip
s.add(Equation(id="count", layer="overlay", pos=(8 / 9, 0.9),
               parts=("$N =$", r"$V_{sphere}$", "$+$", r"$V_{cell}$"),
               colors=("primary", "secondary", "primary", "highlight"),
               keys=("lhs", "sphere", "plus", "cell"), style={"size": "title"}))  # fmt: skip

s.marker("start", 0.0)
s.marker("sphere_term", 2.0)
s.marker("cell_term", 5.0)
s.marker("end", 9.0)
s.at("start").orbit(360.0, until="end").reveal("count")
s.at("sphere_term").lift_term("sphere", "count", "sphere")
s.at("cell_term").lift_term("cell", "count", "cell")

if __name__ == "__main__":
    e = s.engine
    mid = e.inspect(2.6, id="count")["objects"][0]["semantic"]
    print(mid["lifting"])
