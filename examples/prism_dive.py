"""A spatial transition: pull back from a Julia set to its point on the Mandelbrot set, glide,
and dive into another Julia set (the "dive", TRANSITIONS.md T2). The camera never cuts.

Each point c of the Mandelbrot set is a whole Julia set (the Julia set of z^2 + c). The scene
starts close on the Douady rabbit, the Julia set of c = -0.1226 + 0.7449i. Pulling back, the
rabbit shrinks into the point c while the Mandelbrot set fades up around it: the rabbit was
one glowing point of the bigger picture, inside the bulb of period 3. The point glides
across to a second c in the seahorse valley, and the view dives in on it; the Julia set of
the new c, a double spiral, fades up where the point was.

How it stays seamless: the zoom is the fractals' own ``zoom`` (powers of ten), magnifying
both planes about the point c, so it runs at an even pace in scale. One number,
``depth.value`` (about 2 close, 0 far), drives both zooms, and the camera only slides its
center between c and the middle of the set. The Julia set's constant ``julia.c`` is the
only c: the Julia rectangle, the Mandelbrot rectangle (which zooms about it) and the beacon
all read it.

Run:  uv run python examples/prism_dive.py      (prints c and the zoom at each marker)
"""

import math

from phasor import Dot, Scene, Value, cx, expr

FAR = 2.7  # world units on screen top to bottom: the whole Mandelbrot set
NEAR = 0.025  # plane units on screen top to bottom, close in on c
DEEP = math.log10(FAR / NEAR)  # the zoom, in powers of ten, close in
MIDDLE = (-0.62, 0.0, 0.0)  # where the camera looks when pulled back
RABBIT = (-0.1226, 0.7449)  # c of the first Julia set (the Douady rabbit)
SPIRAL = (-0.7453, 0.1127)  # c of the second (in the seahorse valley)
JULIA_VIEW = 3.0  # Julia plane units on screen top to bottom, close in
COVER = (12.0, 6.8)  # both rectangles cover the screen from anywhere c goes

C = "vec(julia.c[0], julia.c[1], 0)"  # the one c, as a world point

s = Scene("prism_dive", look="prism")
s.set("camera.height", FAR)
s.set("camera.center", (RABBIT[0], RABBIT[1], 0.0))
s.add(Value(id="depth", value=DEEP))
s.add(cx.Fractal(id="julia", kind="julia", c=RABBIT, pos=expr(C), center="0 0", size=COVER,
                 zoom=expr(f"depth.value + {math.log10(NEAR / JULIA_VIEW)}"), max_iter=400,
                 radius=0.006, z_index=1))  # fmt: skip
s.add(cx.Fractal(id="mandel", kind="mandelbrot", pos=expr(C), size=COVER,
                 zoom=expr("depth.value"), max_iter=600, radius=0.006, opacity=0.0))  # fmt: skip
s.add(Dot(id="beacon", pos=expr(C), r=0.032, preset="beacon", z_index=2))

for name, t in (("start", 0.3), ("pull", 2.5), ("faded", 5.4), ("wide", 8.0), ("glide", 8.8),
                ("arrive", 12.2), ("dive", 12.8), ("rise", 15.2), ("close", 18.6),
                ("end", 21.0)):  # fmt: skip
    s.marker(name, t)

s.at("start").reveal("julia")
s.at("pull").morph("depth.value", 0.0, until="wide")  # pull back: the rabbit shrinks into c
s.at("pull").morph("camera.center", MIDDLE, until="wide")  # and the whole set comes in
s.at("pull").morph("julia.opacity", 0.0, until="faded")  # the rabbit fades into the set
s.at("pull").morph("mandel.opacity", 1.0, until="faded")
s.at("pull", 2.0).reveal("beacon")
s.at("glide").morph("julia.c", SPIRAL, until="arrive")  # one c: the point and the set move
s.at("dive").morph("depth.value", DEEP, until="close")  # dive back in on the new c
s.at("dive").morph("camera.center", (SPIRAL[0], SPIRAL[1], 0.0), until="close")
s.at("dive").exit("beacon")
s.at("rise").morph("julia.opacity", 1.0, until="close")  # the spiral fades up
s.at("rise").morph("mandel.opacity", 0.0, until="close")

if __name__ == "__main__":
    e = s.engine
    for name in ("start", "pull", "faded", "wide", "glide", "arrive", "dive", "rise", "close"):
        t = e.document.timeline.markers[name]
        st = e.state_at(t)
        center = st.values["camera.center"]
        print(f"{name:>6} t={t:5.2f}  c = {st['julia'].params.c}  "
              f"zoom {st['mandel'].params.zoom:.2f}  center {center}")  # fmt: skip
    for issue in e.lint(strict=True):
        print("lint:", issue)
