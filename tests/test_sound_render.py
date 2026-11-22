"""Sound in a rendered video: absent unless asked for, and on time when asked for."""

from __future__ import annotations

import av
import numpy as np
import pytest

from phasor import Scene, ac, expr
from tests.conftest import gl_available

pytestmark = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _audio(path):
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None, 0
        rate = c.streams.audio[0].rate
        chunks = [f.to_ndarray().reshape(-1) for f in c.decode(audio=0)]
    return np.concatenate(chunks).astype(np.float32), rate


def _scene():
    s = Scene("heard")
    s.add(ac.Room3D(id="room", size=(8.0, 5.0, 3.0)))
    s.add(ac.Listener(id="mic", pos=(6.0, 3.4, 1.4)))
    s.add(ac.Wavefront3D(id="front", room="room", source=(2.2, 2.0, 1.4), listener="mic",
                         time=expr("t - 0.2"), speed=0.01, order=1))  # fmt: skip
    s.marker("start", 0.0)
    s.marker("end", 1.5)
    return s


def test_silent_unless_asked_and_clicks_on_time_when_asked(tmp_path):
    from phasor.render.job import render_file

    e = _scene().engine
    quiet = tmp_path / "quiet.mp4"
    r = render_file(e, quiet, preset="preview", fps=15, sound=False)
    assert r["sound"] is False and _audio(quiet)[0] is None  # no audio stream at all

    loud = tmp_path / "loud.mp4"
    r = render_file(e, loud, preset="preview", fps=15, sound=True)
    assert r["sound"] is True and r["narration"] is False
    x, rate = _audio(loud)
    assert x is not None and abs(len(x) / rate - 1.5) < 0.1
    d = float(np.linalg.norm(np.array([2.2, 2.0, 1.4]) - np.array([6.0, 3.4, 1.4])))
    expected = 0.2 + d / (343.0 * 0.01)
    onset = np.flatnonzero(np.abs(x) > 0.2 * np.abs(x).max())[0] / rate
    assert abs(onset - expected) < 1 / 15  # within one video frame
