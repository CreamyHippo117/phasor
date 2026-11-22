"""Sound: the Fourier transform's winding picture, heard.

A signal made of two cosines (3 and 5.5 cycles per second, the second weaker) is wound
round the origin at a sweeping winding frequency f = 1 + t. The centre of mass of the wound
signal is large exactly when f meets one of the signal's frequencies. A ``Tone`` plays it:
its pitch is the winding frequency (110 Hz per cycle per second), its loudness the length of
the centre of mass. You hear silence, a swell at f = 3, silence, a smaller swell at f = 5.5.

The centre of mass over a window of T seconds is worked out exactly and written as an
``expr``: for g = sum a_k cos(2 pi f_k s) it is
    1/2 sum a_k [S(f_k - f) + S(f_k + f)],   S(d) = exp(i pi d T) sinc(d T),
whose real and imaginary parts are cos/sin(pi d T) sinc(d T). expr has sin, cos, sqrt but no
complex numbers or sinc, so the two parts are spelled out (sinc(x) = sin(pi x) / (pi x),
nudged by 1e-9 so that d = 0 is not a division by zero). The script checks the formula
against a numerical integral.

Run:  uv run python examples/sound_winding.py [out.wav]   (prints the values it plays)
"""

import sys

import numpy as np

from phasor import Scene, expr, sound

SIGNAL = ((3.0, 1.0), (5.5, 0.6))  # (frequency, amplitude)
WINDOW = 4.0  # seconds of signal wound up
SWEEP = "1 + t"  # the winding frequency
PITCH = 110.0  # Hz per winding cycle per second


def com_expr(f: str) -> str:
    """|centre of mass| of the signal wound at frequency ``f`` (an expr string), as an expr."""
    f = f"({f})"
    re, im = [], []
    for freq, amp in SIGNAL:
        for d in (f"({freq} - {f})", f"({freq} + {f})"):
            x = f"(pi_ * {d} * {WINDOW} + 1e-9)"
            sinc = f"(sin({x}) / {x})"
            re.append(f"{amp / 2} * cos(pi_ * {d} * {WINDOW}) * {sinc}")
            im.append(f"{amp / 2} * sin(pi_ * {d} * {WINDOW}) * {sinc}")
    body = f"sqrt(({' + '.join(re)}) ** 2 + ({' + '.join(im)}) ** 2)"
    return body.replace("pi_", "3.141592653589793")


s = Scene("sound_winding", look="calm")
s.add(sound.Tone(id="winding", hz=expr(f"{PITCH} * ({SWEEP})"),
                 gain=expr(f"2 * {com_expr(SWEEP)}")))  # fmt: skip
s.marker("end", 7.0)


def numeric_com(f: float) -> float:
    tau = np.linspace(0.0, WINDOW, 40001)
    g = sum(a * np.cos(2 * np.pi * k * tau) for k, a in SIGNAL)
    return abs(np.trapezoid(g * np.exp(-2j * np.pi * f * tau), tau) / WINDOW)


if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 1.0, 2.0, 3.0, 4.5, 6.0):
        sem = e.inspect(t, id="winding")["objects"][0]["semantic"]
        f = 1.0 + t
        print(f"f = {f:.1f}: plays {sem['hz']:.0f} Hz at gain {sem['gain']:.3f}"
              f"  (|com| = {numeric_com(f):.3f} numerically)")  # fmt: skip
    if len(sys.argv) > 1:
        from phasor.time.narration import write_wav

        track, rate = e.audio_track(sound=True)
        write_wav(sys.argv[1], track, rate)
        print("wrote", sys.argv[1])
