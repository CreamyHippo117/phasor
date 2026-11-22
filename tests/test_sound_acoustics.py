"""Opt-in sound of the acoustics objects: when the clicks happen is read from the document,
and must equal the analytic arrival times (to the 200 Hz control grid)."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Scene, ac, expr
from phasor.core import sound as snd
from phasor.domains import ac_math

TOL = 1 / 200
C = ac_math.C
SIZE, SRC, MIC = (8.0, 5.0, 3.0), (2.2, 2.0, 1.4), (6.0, 3.4, 1.4)
CLAP, SLOW = 1.0, 0.01  # as in examples/room_3d.py


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))


def room_scene(listener="mic", order=2, end=12.0, material="wood"):
    s = Scene("snd")
    s.add(ac.Room3D(id="room", size=SIZE, material=material))
    s.add(ac.Listener(id="mic", pos=MIC))
    s.add(ac.Wavefront3D(id="front", room="room", source=SRC, time=expr(f"t - {CLAP}"),
                         speed=SLOW, order=order, listener=listener))  # fmt: skip
    s.marker("end", end)
    return s


def expected(order, alpha):
    im = ac_math.image_sources(SIZE, SRC, order, alpha)
    d = np.linalg.norm(im["pos"] - np.asarray(MIC), axis=1)
    return d, im["reflections"], im["gain"] / d


def dominant_hz(x, rate):
    spec = np.abs(np.fft.rfft(x, 1 << 16))
    return float(np.argmax(spec)) * rate / (1 << 16)


def test_wavefront3d_clicks_at_analytic_times():
    s = room_scene()
    alpha = s.engine.state_at(0.0)["room"].impl.alpha(s.engine.state_at(0.0))
    d, _, _ = expected(2, alpha)
    made = s.engine.sounds()
    want = np.sort(CLAP + d / (C * SLOW))
    got = np.sort([x.t for x in made])
    # fronts beyond the scene's end are not heard
    want = want[want <= s.engine.duration]
    assert len(got) == len(want)
    assert np.allclose(got, want, atol=TOL)
    direct = min(made, key=lambda x: x.t)
    assert direct.t == pytest.approx(CLAP + np.linalg.norm(np.subtract(SRC, MIC)) / (C * SLOW),
                                     abs=TOL)  # fmt: skip


def test_pitch_gain_order():
    made = room_scene().engine.sounds()
    direct = min(made, key=lambda x: x.t)
    later = [x for x in made if x is not direct]
    assert later
    f0 = dominant_hz(direct.samples, 48000)
    assert f0 == pytest.approx(1800, rel=0.05)
    f1 = dominant_hz(max(later, key=lambda x: x.t).samples, 48000)
    assert f1 < f0
    assert direct.gain == pytest.approx(0.8)  # the strongest: click_gain
    assert all(x.gain < direct.gain for x in later)


def test_gain_ratio_is_reflection_factor_over_distance():
    s = room_scene()
    st = s.engine.state_at(0.0)
    alpha = st["room"].impl.alpha(st)
    _, _, amp = expected(2, alpha)
    made = sorted(s.engine.sounds(), key=lambda x: x.t)
    g = np.array([x.gain for x in made])
    # every heard gain is click_gain * amp / max(amp); the set matches the analytic set
    rel = np.sort(amp / amp.max() * 0.8)
    kept = rel[rel >= 0.8 * 10 ** (-50 / 20)]
    assert np.allclose(np.sort(g), kept[-len(g) :], rtol=1e-6)


@pytest.mark.parametrize("name", ["", "nobody", "room"])
def test_no_listener_no_sound(name):
    s = room_scene(listener=name)
    assert s.engine.sounds() == []
    assert s.engine.audio_track(sound=True) is None


def test_audio_track_is_opt_in_and_onset_is_in_time():
    s = room_scene(order=0)
    assert s.engine.audio_track() is None
    track, rate = s.engine.audio_track(sound=True)
    assert track.dtype == np.int16
    x = np.abs(track.astype(np.float64))
    t_direct = CLAP + np.linalg.norm(np.subtract(SRC, MIC)) / (C * SLOW)
    onset = np.flatnonzero(x > 0.05 * x.max())[0] / rate
    assert onset == pytest.approx(t_direct, abs=1 / 30)
    assert onset >= t_direct - 1 / 200


def test_wavefront_hits_listener_once():
    s = Scene("w")
    s.add(ac.Listener(id="mic", pos=(3.0, 4.0, 0.0)))  # 5 m from the origin
    s.add(ac.Wavefront(id="wave", center=(0, 0, 0), radius=expr("343 * (t - 0.5)"),
                       listener="mic"))  # fmt: skip
    s.marker("end", 2.0)
    made = s.engine.sounds()
    assert len(made) == 1
    assert made[0].t == pytest.approx(0.5 + 5 / 343, abs=TOL)
    s2 = Scene("w2")
    s2.add(ac.Wavefront(id="wave", center=(0, 0, 0), radius=expr("343 * (t - 0.5)")))
    s2.marker("end", 2.0)
    assert s2.engine.sounds() == []


def impulse_scene(reveal, duration=0.2, order=3):
    s = Scene("ir")
    s.add(
        ac.ImpulseResponse(
            id="ir",
            size=SIZE,
            source=SRC,
            listener=MIC,
            material="wood",
            order=order,
            duration=duration,
            layer="overlay",
            reveal=reveal,
        )
    )
    s.marker("end", 5.0)
    return s


def test_impulse_response_follows_reveal():
    dur = 0.2
    s = impulse_scene(expr(f"min(max((t - 1) * 0.1 / {dur}, 0), 1)"), dur)
    made = s.engine.sounds()
    st = s.engine.state_at(0.0)
    arr = st["ir"].impl._arr(st)
    shown = arr["time"][arr["time"] <= dur]
    assert len(shown) > 1
    # reveal * dur = 0.1 * (t - 1): an arrival at time a is heard at t = 1 + a / 0.1
    want = 1 + shown / 0.1
    got = np.sort([x.t for x in made])
    assert len(got) <= len(shown)
    assert np.allclose(got, want[: len(got)], atol=TOL)
    assert got[0] == pytest.approx(1 + arr["time"][0] / 0.1, abs=TOL)


def test_static_impulse_response_plays_in_real_time():
    s = impulse_scene(1.0)
    made = sorted(s.engine.sounds(), key=lambda x: x.t)
    st = s.engine.state_at(0.0)
    arr = st["ir"].impl._arr(st)
    assert made[0].t == pytest.approx(arr["time"][0], abs=TOL)
    assert all(x.t <= 0.2 + TOL for x in made)
    assert made[0].gain == pytest.approx(0.8)
    assert made[-1].gain < made[0].gain


def test_example_room_3d_timings():
    s = room_scene(order=4, end=12.0)
    made = s.engine.sounds()
    t0 = min(x.t for x in made)
    assert t0 == pytest.approx(CLAP + np.linalg.norm(np.subtract(SRC, MIC)) / (C * SLOW), abs=TOL)
    assert all(x.t <= 12.0 for x in made)
    assert snd.mix(made, 12.0, 48000).shape == (12 * 48000,)
