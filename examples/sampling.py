"""Sampling and aliasing (ANIMATIONS.md S5): the samples of a fast wave, and the slow wave
they also fit.

A 7 Hz wave (cyan) is sampled; every sample is a beacon on a stem. The sampling rate starts
high, so the beacons trace the wave faithfully, then sweeps down. Below twice the wave's
frequency (Nyquist) the beacons also lie exactly on a slower wave (amber): the alias. It is
the only wave a person who sees just the beacons can find, which is the whole trouble.

Run:  uv run python examples/sampling.py      (prints the alias at a few sampling rates)
"""

from phasor import Counter, Scene, Text, expr
from phasor.domains.fourier import Sampling

s = Scene("sampling", look="prism")
s.set("camera.height", 6.2)
s.add(Sampling(id="wave", frequency=7.0, fs=40.0, duration=1.0, size=(9.0, 3.4)))
s.add(Text(id="title", text="sampling a 7 Hz wave", pos=(0.5, 0.06), layer="overlay",
           style={"size": "title"}))  # fmt: skip

s.add(
    Counter(
        id="rate", value=expr("wave.fs"), prefix="fs = ", digits=1, pos=(0.5, 0.15), layer="overlay"
    )
)

s.marker("start", 0.3)
s.marker("sweep", 1.0)
s.marker("end", 5.0)
s.at("start").reveal("title")
s.at("start").reveal("wave")
s.at("start").reveal("rate")
# the rate falls through Nyquist (14 Hz): the alias appears
s.at("sweep").sweep("wave.fs", 9.0, show=False)

if __name__ == "__main__":
    e = s.engine
    for t in (0.3, 2.4, 3.5, 5.0):
        sem = e.inspect(t, id="wave")["objects"][0]["semantic"]
        print(f"t={t:g}: fs={sem['sample_rate_hz']:.1f} Hz, Nyquist {sem['nyquist_hz']:.1f} Hz, "
              f"alias {sem['alias_hz']:.2f} Hz, aliasing: {sem['aliasing']}")  # fmt: skip
    for issue in e.lint(strict=True):
        print("lint:", issue)
