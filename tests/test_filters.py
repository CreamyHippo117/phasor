"""filt.PoleZero: the numerics, the object, and the unstable_filter lint rule."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Scene, filt
from phasor.domains import filters_math as fm

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]  # as tests/test_lint.py


@pytest.mark.parametrize("kind", ["lowpass", "highpass", "bandpass", "notch"])
def test_named_filters_are_stable_and_have_known_gain(kind):
    z, p = fm.biquad(kind, 0.1, 0.8)
    assert fm.is_stable(p) and len(p) == 2
    _f, h = fm.response(z, p, 1.0)
    assert np.all(np.isfinite(h))


def test_biquad_gain_matches_the_cookbook():
    z, p = fm.biquad("lowpass", 0.1, 0.707)
    # the cookbook gain is b0 / a0; the roots drop it, so take it back
    cw = np.cos(2 * np.pi * 0.1)
    alpha = np.sin(2 * np.pi * 0.1) / (2 * 0.707)
    gain = ((1 - cw) / 2) / (1 + alpha)
    _, h = fm.response(z, p, gain)
    assert abs(h[0]) == pytest.approx(1.0, abs=1e-9)  # unity at DC
    assert abs(h[-1]) < 1e-9  # a zero at Nyquist


def test_stability_boundary():
    assert fm.is_stable([0.99]) and fm.is_stable([])
    assert not fm.is_stable([1.0]) and not fm.is_stable([0.5, 1.2j])


def test_named_kind_rejects_given_poles():
    with pytest.raises(Exception, match="custom"):
        Scene("filt_bad").add(filt.PoleZero(id="x", kind="lowpass", poles=((0.5, 0),)))


def _scene(passes, **kw):
    s = Scene("filt", passes=passes)
    s.set("camera.height", 8)
    s.add(filt.PoleZero(id="f", **kw))
    return s


@pytest.mark.parametrize("passes", MODES)
def test_semantics_report_stability(passes):
    s = _scene(passes, kind="custom", poles=((1.5, 0),), zeros=((-1, 0),))
    sem = s.engine.inspect(0.0, id="f")["objects"][0]["semantic"]
    assert sem["stable"] is False and sem["max_pole_radius"] == 1.5


@pytest.mark.parametrize("passes", MODES)
def test_stable_filter_is_clean(passes):
    s = _scene(passes, kind="custom", poles=((0.5, 0.5), (0.5, -0.5)))
    assert [i for i in s.engine.lint() if i.code == "unstable_filter"] == []
    assert [i for i in _scene(passes, kind="notch").engine.lint()
            if i.code == "unstable_filter"] == []  # fmt: skip


@pytest.mark.parametrize("passes", MODES)
def test_unstable_filter_is_flagged_with_times_and_no_fix(passes):
    s = _scene(passes, kind="custom", poles=((1.2, 0.0),))
    (issue,) = [i for i in s.engine.lint() if i.code == "unstable_filter"]
    assert issue.severity == "warning" and issue.ids == ("f",)
    assert issue.times == (0.0,) and "t = 0" in issue.message
    assert issue.fix == () and issue.fix_hint


@pytest.mark.parametrize("passes", MODES)
def test_a_pole_on_the_circle_counts(passes):
    s = _scene(passes, kind="custom", poles=((0.0, 1.0),))
    assert any(i.code == "unstable_filter" for i in s.engine.lint())


@pytest.mark.parametrize("passes", MODES)
def test_instability_that_starts_later_reports_that_time(passes):
    s = _scene(passes, kind="custom", poles=((0.5, 0.0),))
    s.marker("late", 2.0)
    s.at("late").to("f.gain", 2.0)  # keep a track alive so the scene has duration
    s.at("late").set("f.poles", ((1.5, 0.0),))
    issues = [i for i in s.engine.lint() if i.code == "unstable_filter"]
    assert issues and min(issues[0].times) >= 2.0
