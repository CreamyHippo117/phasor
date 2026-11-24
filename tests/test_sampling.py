"""S5 (ANIMATIONS.md): sampling and aliasing. The folding formula, the fact the whole picture
rests on (the samples of the fast wave and of its alias coincide), no alias below Nyquist,
determinism, and the object through the engine."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, expr
from phasor.domains import fourier_math as fm
from phasor.domains.fourier import Sampling
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


# ---------------------------------------------------------------- the maths


@pytest.mark.parametrize(
    ("f", "fs", "alias"),
    [
        (7.0, 8.0, 1.0),  # just under the rate: a slow wave
        (7.0, 10.0, 3.0),
        (7.0, 12.0, 5.0),
        (7.0, 14.0, 7.0),  # exactly Nyquist: the wave is its own alias
        (7.0, 20.0, 7.0),  # well sampled
        (7.0, 7.0, 0.0),  # sampled once a cycle: looks constant
        (7.0, 3.0, 1.0),  # more than two folds
        (7.0, 4.5, 2.0),
        (100.0, 90.0, 10.0),
        (0.0, 10.0, 0.0),
    ],
)
def test_the_alias_is_the_distance_to_the_nearest_multiple_of_the_rate(f, fs, alias):
    assert fm.alias_frequency(f, fs) == pytest.approx(alias)
    # the definition itself: |f - k fs| minimised over every whole k
    brute = min(abs(f - k * fs) for k in range(-50, 51))
    assert fm.alias_frequency(f, fs) == pytest.approx(brute)


def test_an_alias_never_exceeds_nyquist_and_is_periodic_in_the_rate():
    rng = np.random.default_rng(3)
    for f, fs in rng.uniform(0.1, 300, size=(200, 2)):
        a = fm.alias_frequency(f, fs)
        assert 0 <= a <= fs / 2 + 1e-9
        assert fm.alias_frequency(f + fs, fs) == pytest.approx(a, abs=1e-7)
        assert fm.alias_frequency(-f, fs) == pytest.approx(a, abs=1e-7)


@pytest.mark.parametrize(
    ("f", "fs"),
    [(7.0, 10.0), (7.0, 8.0), (7.0, 12.5), (13.0, 4.0), (7.0, 20.0), (7.0, 7.0), (2.5, 3.0)],
)
@pytest.mark.parametrize("phase", [0.0, 0.7, -2.0])
def test_the_fast_wave_and_its_alias_have_the_same_samples(f, fs, phase):
    """The point of the picture: at the sample times the two waves are the same numbers."""
    t = fm.sample_times(fs, 2.0)
    fast = fm.wave(f, t, phase)
    slow = fm.wave(fm.alias_signed(f, fs), t, phase)
    assert fast == pytest.approx(slow, abs=1e-9)
    assert len(t) == int(2.0 * fs) + 1


def test_between_the_samples_the_two_waves_differ_when_it_aliases():
    t = np.linspace(0, 1, 2001)
    fast, slow = fm.wave(7.0, t, 0.7), fm.wave(fm.alias_signed(7.0, 10.0), t, 0.7)
    assert np.abs(fast - slow).max() > 1.0


def test_below_nyquist_there_is_no_alias():
    assert not fm.aliases(7.0, 20.0)
    assert not fm.aliases(7.0, 14.0)  # exactly Nyquist is the edge, not aliasing
    assert fm.aliases(7.0, 13.9)
    assert fm.alias_signed(7.0, 20.0) == pytest.approx(7.0)
    assert fm.alias_frequency(7.0, 20.0) == pytest.approx(7.0)
    assert fm.nyquist(20.0) == 10.0


def test_the_sign_of_the_alias_tracks_which_side_of_the_rate_the_tone_is():
    assert fm.alias_signed(9.0, 10.0) == pytest.approx(-1.0)  # running backwards
    assert fm.alias_signed(11.0, 10.0) == pytest.approx(1.0)


def test_sample_times_are_exact_and_deterministic():
    a, b = fm.sample_times(16.0, 1.0), fm.sample_times(16.0, 1.0)
    assert np.array_equal(a, b)
    assert a[0] == 0 and a[-1] == pytest.approx(1.0) and len(a) == 17
    assert fm.sample_times(10.0, 0.5).tolist() == pytest.approx([0, 0.1, 0.2, 0.3, 0.4, 0.5])


# ---------------------------------------------------------------- the object


def _scene(**params) -> Scene:
    s = Scene("smp", passes=None)
    s.add(Sampling(id="smp", **params))
    return s


def _obj(s, t=0.0) -> dict:
    return s.engine.inspect(t, id="smp")["objects"][0]


def _sem(s, t=0.0) -> dict:
    return _obj(s, t)["semantic"]


def test_the_semantics_report_the_alias_the_nyquist_rate_and_whether_it_aliases():
    sem = _sem(_scene(frequency=7.0, fs=10.0))
    assert sem["kind"] == "sampling"
    assert sem["backdrop"] is True
    assert sem["frequency_hz"] == 7.0 and sem["sample_rate_hz"] == 10.0
    assert sem["nyquist_hz"] == 5.0
    assert sem["alias_hz"] == pytest.approx(3.0)
    assert sem["aliasing"] is True
    sem = _sem(_scene(frequency=7.0, fs=30.0))
    assert sem["aliasing"] is False
    assert sem["alias_hz"] == pytest.approx(7.0)
    assert sem["nyquist_hz"] == 15.0


def test_the_alias_appears_exactly_when_the_rate_drops_below_twice_the_tone():
    for fs, aliasing in ((40.0, False), (15.0, False), (14.0, False), (13.0, True), (9.0, True)):
        sem = _sem(_scene(frequency=7.0, fs=fs))
        assert sem["aliasing"] is aliasing, fs
        assert sem["alias_hz"] == pytest.approx(fm.alias_frequency(7.0, fs))


def test_fs_can_be_animated_and_the_semantics_follow_the_clock():
    s = _scene(frequency=7.0, fs=expr("lerp(40, 9, clamp(t / 4, 0, 1))"))
    s.marker("end", 4.0)
    assert _sem(s, 0.0)["aliasing"] is False
    late = _sem(s, 4.0)
    assert late["aliasing"] is True
    assert late["alias_hz"] == pytest.approx(2.0)
    assert late["sample_count"] == int(late["duration_s"] * 9.0) + 1


def test_the_bounds_are_the_plot_box_and_the_state_is_pure():
    s = _scene(frequency=3.0, fs=10.0, duration=1.0, size=(6.0, 2.0), pos=(1.0, 2.0, 0.0))
    assert _sem(s)["sample_count"] == 11
    assert _sem(s) == _sem(s)  # deterministic
    st = s.engine.state_at(0.0)
    b = st["smp"].impl.world_bounds(st)
    assert tuple(b.lo) == pytest.approx((-2.0, 1.0, 0.0))
    assert tuple(b.hi) == pytest.approx((4.0, 3.0, 0.0))


def test_progress_hides_the_samples_not_yet_taken():
    half = _sem(_scene(frequency=3.0, fs=10.0, duration=1.0, progress=0.5))
    assert half["sample_count"] == 6  # t = 0, 0.1, ... 0.5
    assert _sem(_scene(frequency=3.0, fs=10.0, duration=1.0, progress=0.0))["sample_count"] == 1


def test_a_silly_setting_is_refused_not_slow():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Sampling.Params(fs=0.0)
    with pytest.raises(ValidationError):
        Sampling.Params(frequency=5.0, fs=500.0, duration=100.0)  # too many samples


@needs_gl
def test_the_picture_has_two_hues_and_beacons_that_glow():
    s = Scene("smp", look="prism")
    s.set("camera.height", 4.0)
    s.add(Sampling(id="smp", frequency=7.0, fs=10.0, duration=1.0, size=(6.0, 2.4)))
    _pic, emit = s.engine.pass_runner().color_and_emission(s.engine.state_at(0.0), (320, 180))
    e = emit[..., :3]
    lit = e[e.sum(axis=-1) > 0.3]
    assert len(lit) > 500
    # the cyan true wave (blue > red) and the amber alias (red > blue) are both there
    assert (lit[:, 2] > 1.3 * lit[:, 0]).sum() > 100
    assert (lit[:, 0] > 1.3 * lit[:, 2]).sum() > 100


@needs_gl
@pytest.mark.slow
def test_sampling_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "sampling.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.5, 3.0, 4.6], size=(320, 180))["image"]
    _golden("sampling", sheet, e.pass_runner().renderer_class())


def test_the_example_runs_and_the_alias_appears_during_the_sweep():
    e = runpy.run_path(str(ROOT / "examples" / "sampling.py"), run_name="x")["s"].engine
    first = e.inspect(0.5, id="wave")["objects"][0]["semantic"]
    last = e.inspect(e.duration, id="wave")["objects"][0]["semantic"]
    assert first["aliasing"] is False and last["aliasing"] is True
    assert last["alias_hz"] == pytest.approx(
        fm.alias_frequency(last["frequency_hz"], last["sample_rate_hz"])
    )
    assert math.isfinite(last["alias_hz"])
