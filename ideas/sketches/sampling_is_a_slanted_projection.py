"""Throwaway still for the idea card `sampling_is_a_slanted_projection` (idea-scout).

Left: the plane of (how much slow wave, how much fast wave). Every signal on one diagonal has
the same samples. Sampling slides a point down its diagonal to the slow axis (the slanted
arrow); a low-pass filter drops it straight down (the plumb arrow). The gap is the alias.
Right: the mix a*slow + b*fast, its samples, and the slow wave (a+b)*slow through the same dots.

Run:  uv run python ideas/sketches/sampling_is_a_slanted_projection.py
"""

from pathlib import Path

import numpy as np

from phasor import Dot, Line, Scene, Text
from phasor.domains.linalg import Arrow

A, B = 0.7, 1.0  # amounts of slow and fast wave
F0, FS = 1.0, 5.0  # slow frequency and sample rate; the fast wave is F0 + FS

s = Scene("sampling_is_a_slanted_projection", look="none")
s.set("camera.height", 8)

# Left: the coefficient plane, origin at ORIGIN, one unit = U.
ORIGIN, U = np.array([-6.2, -1.6]), 2.0


def P(a, b):
    return tuple(ORIGIN + U * np.array([a, b]))


s.add(Line(id="slow_axis", points=[P(-0.1, 0), P(2.2, 0)], style={"color": "muted"}))
s.add(Line(id="fast_axis", points=[P(0, -0.1), P(0, 1.9)], style={"color": "muted"}))
for i, c in enumerate(np.arange(0.4, 2.3, 0.3)):  # a + b = c: one frozen set of samples each
    s.add(
        Line(
            id=f"same_samples_{i}",
            points=[P(max(c - 1.8, 0), min(c, 1.8)), P(c, 0)],
            style={"color": "muted", "stroke": "hairline"},
        )
    )
s.add(Arrow(id="sampling", start=P(A, B), end=P(A + B, 0), style={"color": "highlight"}))
s.add(Arrow(id="filter", start=P(A, B), end=P(A, 0), style={"color": "secondary"}))
s.add(Line(id="alias_gap", points=[P(A, -0.12), P(A + B, -0.12)], style={"color": "highlight"}))
s.add(Dot(id="mix", pos=P(A, B), r=0.1, style={"color": "highlight"}))
s.add(
    Text(
        id="slow_label", text="how much slow wave", pos=P(1.1, -0.45), style={"size": "annotation"}
    )
)
s.add(
    Text(id="fast_label", text="how much fast wave", pos=P(0.1, 2.1), style={"size": "annotation"})
)
s.add(Text(id="sampling_label", text="sampling", pos=P(1.55, 0.75), style={"size": "annotation"}))
s.add(Text(id="filter_label", text="filter", pos=P(0.42, 0.45), style={"size": "annotation"}))
s.add(Text(id="gap_label", text="the alias", pos=P(A + B / 2, -0.3), style={"size": "annotation"}))

# Right: the waveform, its samples, and the slow wave through the same samples.
X0, W, Y0, H = 0.3, 6.4, 0.2, 1.3
t = np.linspace(0, 2, 600)


def wave(tt, a, b):
    return a * np.sin(2 * np.pi * F0 * tt) + b * np.sin(2 * np.pi * (F0 + FS) * tt)


def Q(tt, y):
    return [(float(X0 + W * u / 2), float(Y0 + H * v)) for u, v in zip(tt, y, strict=True)]


s.add(
    Line(
        id="time_axis",
        points=[(X0, Y0), (X0 + W, Y0)],
        style={"color": "muted", "stroke": "hairline"},
    )
)
s.add(Line(id="mixed_wave", points=Q(t, wave(t, A, B)), style={"color": "muted"}))
s.add(Line(id="slow_wave", points=Q(t, wave(t, A + B, 0)), style={"color": "highlight"}))
tn = np.arange(0, 2.0001, 1 / FS)
for i, (x, y) in enumerate(Q(tn, wave(tn, A, B))):
    s.add(Dot(id=f"sample_{i}", pos=(x, y), r=0.07, style={"color": "secondary"}))

if __name__ == "__main__":
    from phasor.render.output import save_png

    save_png(Path(__file__).with_suffix(".png"), s.engine.still(0.0, (1600, 900)))
