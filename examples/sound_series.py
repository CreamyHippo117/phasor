"""Sound: a Fourier series you can hear. A square wave gains one harmonic every half second:
the picture sharpens toward the square and the tone brightens, the new odd harmonic (at
n x 220 Hz, level 1/n) arriving each time.

Run:  uv run python examples/sound_series.py [out.wav]   (prints the harmonics heard)
"""

import sys

import numpy as np

from phasor import Scene, expr, fourier

s = Scene("sound_series", look="calm")
s.add(fourier.PartialSum(id="wave", kind="square", terms=expr("1 + floor(t * 2)"),
                         hz=220.0, pos=(0.0, 0.0, 0.0), extent=(8.0, 3.0)))  # fmt: skip
s.marker("end", 5.0)


def heard(sound, t0: float, t1: float, rate: int) -> dict[int, float]:
    """The harmonics (multiples of 220 Hz) with real energy in [t0, t1], by FFT."""
    x = sound.samples[round((t0 - sound.t) * rate) : round((t1 - sound.t) * rate)]
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x) * 4
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    top = mag.max()
    return {n: round(float(mag[np.argmin(abs(freqs - 220 * n))] / top), 3)
            for n in range(1, 12) if mag[np.argmin(abs(freqs - 220 * n))] > 0.02 * top}  # fmt: skip


if __name__ == "__main__":
    e = s.engine
    (snd,) = e.sounds()
    for t in (0.25, 1.25, 2.25, 4.25):
        print(f"t = {t}: terms {e.inspect(t, id='wave')['objects'][0]['semantic']['terms']},"
              f" harmonics heard (relative level) {heard(snd, t, t + 0.2, 48000)}")  # fmt: skip
    if len(sys.argv) > 1:
        from phasor.time.narration import write_wav

        track, rate = e.audio_track(sound=True)
        write_wav(sys.argv[1], track, rate)
        print("wrote", sys.argv[1])
