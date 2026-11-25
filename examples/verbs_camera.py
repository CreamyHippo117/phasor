"""Camera and light verbs: a ripple marks a zero of f(z) = (z - a)(z + 1), a spotlight
picks out the other zero (dimming the domain coloring, which dim_others cannot), and as
the constant a drifts the camera follows its zero across the plane.

Run:  uv run python examples/verbs_camera.py      (prints what the inspector reads)
"""

from phasor import Dot, Scene, cx

s = Scene("verbs_camera")
s.set("camera.height", 5.0)
s.add(cx.DomainColoring(id="plane", function="(z - a)*(z + 1)", a=(1.0, 0.0),
                        size=(16.0, 10.0), preset="prism"))  # fmt: skip
s.add(Dot(id="pin", pos=(-1.0, 0.0), r=0.12))

for name, t in (("start", 0.0), ("mark", 0.5), ("light", 2.0), ("drift", 4.0), ("again", 6.0),
                ("rest", 8.0), ("end", 9.5)):  # fmt: skip
    s.marker(name, t)
s.at("mark").ripple("plane", at="zeros", index=1)  # the zero at a = 1 (zeros are listed -1, a)
s.at("light").spotlight("pin", until="drift")  # the zero at -1 (dims the field too)
s.at("drift").morph("plane.a", (2.2, 1.2), until="rest")
s.at("drift").follow("plane", feature="zeros", index=1, until="rest")  # the camera rides it
s.at("again").ripple("plane", at="zeros", index=1)  # the rings ride the moving zero

if __name__ == "__main__":
    e = s.engine
    for t in (1.0, 3.0, 6.3, 7.5):
        st = e.state_at(t)
        print(
            f"t={t}: camera center {st.values['camera.center']}, "
            f"zero {e.inspect(t, id='plane.ripple')['objects'][0]['semantic']['spot']}"
        )
