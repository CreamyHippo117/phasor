"""A filter's poles and zeros on the z-plane, then the pole that makes it unstable.

A low-pass filter has its poles inside the unit circle. The second filter is the same shape
with one pole pushed out past the circle: it is unstable (any input makes the output grow
without bound), and ``lint`` says so (``unstable_filter``).

Run:  uv run python examples/filter_poles.py      (prints the lint issues)
"""

from phasor import Scene, Text, filt

s = Scene("filter_poles", look="prism")
s.add(filt.PoleZero(id="stable", kind="lowpass", cutoff=0.1, q=0.8, pos=(-3.4, 0, 0), radius=2.2))
s.add(filt.PoleZero(id="unstable", kind="custom", poles=((0.7, 0.5), (0.7, -0.5), (-1.3, 0)),
                    zeros=((-1, 0),), pos=(3.4, 0, 0), radius=2.2))  # fmt: skip
s.add(Text(id="t1", text="stable: poles inside", pos=(-3.4, -3.0)))
s.add(Text(id="t2", text="unstable: a pole outside", pos=(3.4, -3.0)))

if __name__ == "__main__":
    for issue in s.engine.lint():
        print(issue.code, issue.ids, issue.message)
