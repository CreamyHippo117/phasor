"""Sound core (core/sound.py): primitives, finding times, mixing, and that it is opt-in."""

from __future__ import annotations

import numpy as np
import pytest

from phasor.core import sound as snd
from phasor.style.theme import load_theme

RATE = 48000


def _peak_hz(x: np.ndarray, rate: int = RATE) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[spec.argmax()])


def test_theme_has_a_sound_table_in_every_theme():
    for name in ("prism", "paper", "aurora", "midnight_glow", "prism_vivid"):
        t = load_theme(name).sound
        assert t["control_hz"] > 0 and 0 < t["master"] <= 1 and t["floor_db"] < 0


def test_click_starts_at_its_first_sample_and_decays():
    c = snd.click(RATE, hz=2000, tau=0.004)
    assert c[0] == 0.0 and np.abs(c[:30]).max() > 0.2  # onset within 0.6 ms
    assert abs(float(c.max())) <= 1.0 + 1e-6
    assert np.abs(c[-200:]).max() < 0.02 * np.abs(c).max()  # dead by the end


@pytest.mark.parametrize("hz", [220.0, 440.0, 1000.0])
def test_tone_and_pluck_and_bell_have_the_pitch_asked_for(hz):
    assert abs(_peak_hz(snd.tone(RATE, hz, 0.5)) - hz) < 4
    assert abs(_peak_hz(snd.pluck(RATE, hz, 0.4)) - hz) < 6
    assert abs(_peak_hz(snd.bell(RATE, hz, 0.4)) - hz) < 8


def test_chirp_glides_up():
    c = snd.chirp(RATE, 300.0, 1200.0, 1.0)
    assert _peak_hz(c[: RATE // 5]) < _peak_hz(c[-RATE // 5 :])


def test_partials_put_energy_where_asked():
    x = snd.partials(RATE, [200.0, 600.0], [1.0, 0.5], 1.0)
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    a, b = spec[np.argmin(abs(f - 200))], spec[np.argmin(abs(f - 600))]
    assert 1.7 < a / b < 2.3


def test_follow_keeps_phase_continuous_while_gliding():
    times = np.linspace(0, 1, 201)
    s = snd.follow(RATE, times, 200.0 + 400.0 * times, np.ones_like(times), harmonics=(1.0,))
    # a smooth glide has no jumps: sample-to-sample change is bounded by the highest pitch
    assert np.abs(np.diff(s.samples)).max() < 2 * np.pi * 620 / RATE * 1.05
    assert s.t == 0.0 and len(s.samples) == RATE
    assert (
        abs(_peak_hz(s.samples[:4800]) - 220) < 25 and abs(_peak_hz(s.samples[-4800:]) - 580) < 25
    )


def test_follow_is_silent_where_gain_is_zero():
    times = np.linspace(0, 1, 101)
    s = snd.follow(RATE, times, np.full_like(times, 440.0), np.where(times < 0.5, 0.0, 1.0))
    assert np.abs(s.samples[: RATE // 2 - 500]).max() == 0.0
    assert np.abs(s.samples[RATE // 2 + 500 :]).max() > 0.3


def test_crossings_interpolate_and_respect_direction():
    t = np.linspace(0, 4, 5)  # 0..4
    v = np.array([0.0, 2.0, 0.0, -2.0, 0.0])
    assert np.allclose(snd.crossings(t, v, 1.0), [0.5, 1.5])
    assert np.allclose(snd.crossings(t, v, 1.0, rising=True), [0.5])
    assert np.allclose(snd.crossings(t, v, 1.0, rising=False), [1.5])
    assert len(snd.crossings(t, np.zeros(5), 1.0)) == 0
    # a sample that sits exactly on the level is counted once
    assert len(snd.crossings(t, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 2.0)) == 1


def test_mix_places_sounds_on_the_sample_and_scales_by_one_factor():
    a = snd.Sound(0.5, np.full(10, 1.0, np.float32), 1.0)
    b = snd.Sound(0.5, np.full(10, 1.0, np.float32), 1.0)
    out = snd.mix([a, b], 1.0, RATE, ceiling=0.9)
    i = round(0.5 * RATE)
    assert out[i - 1] == 0.0 and out[i] > 0  # starts exactly at 0.5 s
    assert abs(float(np.abs(out).max()) - 0.9) < 1e-6  # 2.0 turned down to the ceiling
    # relative loudness kept: a quiet sound next to a loud one stays in proportion
    loud = snd.Sound(0.0, np.full(5, 1.0, np.float32), 1.0)
    quiet = snd.Sound(0.1, np.full(5, 1.0, np.float32), 0.25)
    m = snd.mix([loud, quiet], 0.5, RATE, ceiling=0.5)
    assert abs(m[round(0.1 * RATE)] / m[0] - 0.25) < 1e-5


def test_mix_cuts_at_the_end_and_ignores_the_past():
    s = snd.Sound(0.9, np.ones(RATE, np.float32), 1.0)
    assert len(snd.mix([s], 1.0, RATE)) == RATE
    assert snd.mix([snd.Sound(-5.0, np.ones(10, np.float32))], 1.0, RATE).max() == 0.0


def test_silent_by_default_and_sound_only_when_asked():
    from phasor import Scene

    s = Scene("quiet")
    s.marker("start", 0.0)
    s.marker("end", 1.0)
    e = s.engine
    assert e.audio_track() is None  # no narration, no flag: nothing to hear
    assert e.sounds() == []
    assert e.audio_track(sound=True) is None  # nothing in the scene makes a sound
