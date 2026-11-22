"""Sound objects (``domains/sound.py``) and the Fourier ``sounds`` methods, on the CPU: every
check reads the synthesized samples from ``Engine.sounds()``; no GL, no audio device."""

from __future__ import annotations

import itertools

import numpy as np

from phasor import Dot, Scene, expr, fourier, sound
from phasor.core import sound as snd

RATE = 48000


def scene(duration: float = 4.0) -> Scene:
    s = Scene("snd", look="none")
    s.add(Dot(id="d", pos=(0, 0, 0)))
    s.marker("end", duration)
    return s


def peak_hz(x: np.ndarray, rate: int = RATE) -> float:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[int(np.argmax(mag))])


def window(sound_: snd.Sound, t0: float, t1: float) -> np.ndarray:
    return sound_.samples[round((t0 - sound_.t) * RATE) : round((t1 - sound_.t) * RATE)]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def levels(x: np.ndarray, hzs) -> list[float]:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x) * 4
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    return [float(mag[np.argmin(abs(f - h))]) for h in hzs]


# ------------------------------------------------------------------ Tone


def test_tone_pitch_is_hz():
    s = scene()
    s.add(sound.Tone(id="a", hz=440.0))
    (snd_,) = s.engine.sounds()
    x = window(snd_, 1.0, 2.0)  # one second: 1 Hz bins
    assert abs(peak_hz(x) - 440.0) <= 1.0


def test_tone_gain_zero_is_silent_then_fades_in():
    s = scene()
    s.add(sound.Tone(id="a", hz=300.0, gain=expr("max(0, t - 1)")))
    (snd_,) = s.engine.sounds()
    assert rms(window(snd_, 0.0, 0.9)) < 1e-6
    assert rms(window(snd_, 3.0, 3.5)) > 20 * max(rms(window(snd_, 0.0, 0.9)), 1e-9)
    assert rms(window(snd_, 3.0, 3.5)) > rms(window(snd_, 1.5, 2.0))  # louder as gain grows


def test_tone_sweep_rises_monotonically_by_zero_crossings():
    s = scene()
    s.add(sound.Tone(id="a", hz=expr("200 + 200 * t"), harmonics=(1.0,)))
    (snd_,) = s.engine.sounds()

    def zero_rate(t0):  # crossings per second over a quarter second: 2 x frequency
        x = window(snd_, t0, t0 + 0.25)
        return np.count_nonzero(np.diff(np.signbit(x))) / 0.25 / 2

    rates = [zero_rate(t) for t in (0.1, 1.0, 2.0, 3.0)]
    assert all(b > a for a, b in itertools.pairwise(rates))
    for t, r in zip((0.1, 1.0, 2.0, 3.0), rates, strict=True):
        assert abs(r - (200 + 200 * (t + 0.125))) < 25  # the mean pitch over the window


def test_tone_gain_is_scaled_by_the_theme_and_pitch_may_be_an_expr():
    s = scene()
    s.add(sound.Tone(id="a", hz=500.0, harmonics=(1.0,)))
    (snd_,) = s.engine.sounds()
    assert abs(np.abs(snd_.samples).max() - 0.4) < 0.01  # [sound] tone_gain (prism: 0.4)


def test_sound_objects_are_not_flagged_by_lint():
    s = scene()
    s.add(sound.Tone(id="a", hz=300.0))
    s.add(sound.Cue(id="c", at=(1.0,)))
    assert [i for i in s.engine.lint() if i.code in ("orphan", "tiny", "offscreen")] == []


# ------------------------------------------------------------------ Cue


def test_cue_onsets_within_5ms():
    s = scene()
    s.add(sound.Cue(id="c", at=(0.5, 1.25, 3.0), kind="click"))
    made = s.engine.sounds()
    assert len(made) == 3
    for want, got in zip((0.5, 1.25, 3.0), made, strict=True):
        x = np.abs(got.samples)
        onset = got.t + int(np.argmax(x > 0.05 * x.max())) / RATE
        assert abs(onset - want) <= 1 / 200


def test_cue_kinds_and_pitch():
    s = scene()
    s.add(sound.Cue(id="p", at=(0.5,), kind="pluck", hz=440.0))
    s.add(sound.Cue(id="b", at=(1.5,), kind="bell", hz=330.0))
    pl, be = s.engine.sounds()
    assert abs(peak_hz(pl.samples[:9600]) - 440.0) < 12
    assert abs(peak_hz(be.samples[:9600]) - 330.0) < 12
    assert pl.label.startswith("pluck") and be.label.startswith("bell")


def test_audio_track_only_with_sound():
    s = scene()
    s.add(sound.Cue(id="c", at=(1.0,)))
    e = s.engine
    assert e.audio_track() is None
    track, rate = e.audio_track(sound=True)
    assert (
        track.dtype == np.int16
        and len(track) == round(e.duration * rate)
        and np.abs(track).max() > 0
    )


def test_scene_without_sound_objects_has_no_sounds():
    s = scene()
    assert s.engine.sounds() == [] and s.engine.audio_track(sound=True) is None


# ------------------------------------------------------------------ PartialSum


def partial(kind: str, terms, **kw) -> Scene:
    s = scene()
    s.add(fourier.PartialSum(id="w", kind=kind, terms=terms, hz=200.0, **kw))
    return s


def test_square_series_spectrum_odd_harmonics_over_n():
    s = partial("square", 7)
    (snd_,) = s.engine.sounds()
    x = window(snd_, 1.0, 3.0)
    lv = levels(x, [200.0 * n for n in range(1, 17)])
    ref = lv[0]
    for n, v in zip(range(1, 17), lv, strict=True):
        if n % 2 == 0 or n > 13:
            assert v < 0.02 * ref  # no even harmonics, nothing beyond the 7th odd one
        else:
            assert abs(v / ref - 1 / n) < 0.02


def test_sawtooth_has_every_harmonic_and_triangle_falls_as_n_squared():
    (saw,) = partial("sawtooth", 4).engine.sounds()
    lv = levels(window(saw, 1.0, 3.0), [200.0 * n for n in (1, 2, 3, 4)])
    assert all(abs(v / lv[0] - 1 / n) < 0.02 for n, v in zip((1, 2, 3, 4), lv, strict=True))
    (tri,) = partial("triangle", 3).engine.sounds()
    lv = levels(window(tri, 1.0, 3.0), [200.0 * n for n in (1, 2, 3, 5)])
    assert lv[1] < 0.02 * lv[0] and abs(lv[2] / lv[0] - 1 / 9) < 0.01
    assert abs(lv[3] / lv[0] - 1 / 25) < 0.01


def test_new_harmonic_appears_when_terms_passes_its_integer():
    s = partial("square", expr("1 + floor(t)"))  # terms 1, 2, 3, 4 on [0,1), [1,2), ...
    (snd_,) = s.engine.sounds()
    third = 600.0  # the 3rd harmonic is the 2nd term: it arrives at t = 1
    before = levels(window(snd_, 0.5, 0.95), [200.0, third])
    after = levels(window(snd_, 1.1, 1.9), [200.0, third])
    assert before[1] < 0.02 * before[0]
    assert abs(after[1] / after[0] - 1 / 3) < 0.03
    fifth = levels(window(snd_, 1.1, 1.9), [1000.0])[0]
    assert fifth < 0.02 * after[0]  # not yet
    assert levels(window(snd_, 2.1, 2.9), [1000.0])[0] > 0.1 * after[0]


def test_partial_sum_sound_follows_terms_no_click_at_the_switch():
    s = partial("square", expr("1 + floor(t)"))
    (snd_,) = s.engine.sounds()
    around = window(snd_, 0.99, 1.01)
    assert np.abs(np.diff(around)).max() < 0.2  # no discontinuity


# ------------------------------------------------------------------ Epicycles and Spectrum


def test_epicycles_partials_at_hz_times_k_weighted_by_radius():
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    z = 2.0 * np.exp(1j * t) + 0.5 * np.exp(3j * t) + 0.25 * np.exp(-2j * t)  # k = 1, 3, -2
    pts = tuple((float(a.real), float(a.imag), 0.0) for a in z)
    s = scene()
    s.add(fourier.Epicycles(id="e", points=pts, terms=3, hz=100.0))
    (snd_,) = s.engine.sounds()
    lv = levels(window(snd_, 1.0, 3.0), [100.0, 200.0, 300.0, 400.0, 150.0])
    assert abs(lv[1] / lv[0] - 0.25 / 2.0) < 0.01  # |k| = 2 (from k = -2)
    assert abs(lv[2] / lv[0] - 0.5 / 2.0) < 0.01
    assert lv[3] < 0.01 * lv[0] and lv[4] < 0.01 * lv[0]


def test_epicycles_circles_come_in_as_terms_grows():
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    z = 2.0 * np.exp(1j * t) + 0.5 * np.exp(3j * t)
    pts = tuple((float(a.real), float(a.imag), 0.0) for a in z)
    s = scene()
    s.add(fourier.Epicycles(id="e", points=pts, terms=expr("1 + floor(t)"), hz=100.0))
    (snd_,) = s.engine.sounds()
    assert (
        levels(window(snd_, 0.2, 0.9), [300.0])[0]
        < 0.01 * levels(window(snd_, 0.2, 0.9), [100.0])[0]
    )
    assert (
        levels(window(snd_, 1.2, 1.9), [300.0])[0]
        > 0.2 * levels(window(snd_, 1.2, 1.9), [100.0])[0]
    )


def test_spectrum_plays_its_sinusoids():
    s = scene()
    s.add(fourier.Spectrum(id="sp", frequencies=(300.0, 700.0), amplitudes=(1.0, 0.5)))
    (snd_,) = s.engine.sounds()
    lv = levels(window(snd_, 1.0, 3.0), [300.0, 700.0, 500.0])
    assert abs(lv[1] / lv[0] - 0.5) < 0.02 and lv[2] < 0.01 * lv[0]
