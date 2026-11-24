"""A filter's poles on the z plane, then the pole that makes it unstable.

A low-pass filter has all its poles inside the unit circle. The second filter is built from
its own poles and zeros, with one pole pushed out past the circle (a custom filter may put
them anywhere): any input makes its output grow without bound, the pole is drawn in the
danger colour and ``lint`` says so (``unstable_filter``). For a stable filter seen four ways,
see ``examples/filtering.py``.

Run:  uv run python examples/filter_poles.py      (prints the lint issues)
"""

from phasor import Scene, Text, filt

s = Scene("filter_poles", look="prism")
s.set("camera.height", 9.0)
s.add(filt.PoleZero(id="stable", kind="lowpass", order=4, cutoff=1000.0, fs=8000.0,
                    pos=(-4.0, 0.0, 0.0), size=(6.4, 6.0), scale=2.4))  # fmt: skip
s.add(filt.PoleZero(id="unstable", kind="custom", fs=8000.0,
                    poles=((0.7, 0.5), (0.7, -0.5), (-1.3, 0.0)), zeros=((-1.0, 0.0),),
                    pos=(4.0, 0.0, 0.0), size=(6.4, 6.0), scale=2.4))  # fmt: skip
s.add(Text(id="t_stable", text="stable: poles inside", pos=(-4.0, -3.6)))
s.add(Text(id="t_unstable", text="unstable: a pole outside", pos=(4.0, -3.6)))

s.marker("start", 0.3)
s.at("start").reveal("stable")
s.at("start").reveal("unstable")
s.at("start", 0.4).reveal("t_stable")
s.at("start", 0.4).reveal("t_unstable")

if __name__ == "__main__":
    for issue in s.engine.lint():
        print(issue.code, issue.ids, issue.message)
