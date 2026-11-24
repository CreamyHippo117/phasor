"""The lift (TRANSITIONS.md T1): a flat picture tipping up into 3D with no cut.

``use_camera("orbit", flat=h)`` must show what the 2D camera of height ``h`` shows, and the
first part of ``lift`` (the dolly) must not move anything on the plane. Measured on the
projection, so no OpenGL is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Scene
from phasor.time.verbs import LIFT_TILT, VerbError

W, H = 1920, 1080
CORNERS = [(-3.0, -3.0, 0.0), (3.0, -3.0, 0.0), (-3.0, 3.0, 0.0), (3.0, 3.0, 0.0), (1.0, 0.5, 0.0)]


def _flat_scene(name="lift", frame=7.5):
    s = Scene(name)
    s.use_camera("orbit", flat=frame, target=(0.0, 0.0, 0.0))
    s.marker("go", 1.0)
    s.marker("landed", 6.0)
    return s


def _screen(engine, t):
    return engine.state_at(t).camera.to_screen(CORNERS, W, H)


def test_a_flat_camera_shows_what_the_2d_camera_shows():
    flat = _flat_scene()
    flat2d = Scene("flat2d")
    flat2d.set("camera.center", (0.0, 0.0, 0.0))
    flat2d.set("camera.height", 7.5)
    err = np.abs(_screen(flat.engine, 0.0) - _screen(flat2d.engine, 0.0)).max()
    assert err < 0.5, f"flat 3D camera is {err:.2f} px off the 2D one"


def test_the_dolly_does_not_move_the_plane():
    s = _flat_scene()
    s.at("go").lift()
    e = s.engine
    total = e._snap.compiled.theme.motion["lift"]["duration"]
    start = _screen(e, 1.0)
    for u in np.linspace(0.0, LIFT_TILT, 8):  # up to where the tilt starts
        err = np.abs(_screen(e, 1.0 + u * total) - start).max()
        assert err < 2.5, f"the plane moved {err:.2f} px at {u:.2f} of the lift"


def test_lift_lands_at_the_elevation_and_fov():
    s = _flat_scene()
    s.at("go").lift(elevation=28.0, fov=40.0)
    e = s.engine
    end = 1.0 + e._snap.compiled.theme.motion["lift"]["duration"]
    v = e.state_at(end + 0.01).values
    assert v["camera.elevation"] == pytest.approx(28.0)
    assert v["camera.fov"] == pytest.approx(40.0)
    before = e.state_at(0.5).values
    assert before["camera.elevation"] == pytest.approx(89.0)
    assert before["camera.fov"] == pytest.approx(1.5)


def test_lift_until_a_marker():
    s = _flat_scene()
    s.at("go").lift(until="landed")
    e = s.engine
    assert e.state_at(6.0).values["camera.elevation"] == pytest.approx(32.0)
    assert e.state_at(5.0).values["camera.elevation"] > 33.0  # still tipping


def test_lift_needs_a_flat_camera():
    s = Scene("not_flat")
    s.use_camera("orbit")
    s.marker("go", 1.0)
    with pytest.raises(VerbError, match="flat"):
        s.at("go").lift()
    s2 = Scene("ortho")
    s2.marker("go", 1.0)
    with pytest.raises(VerbError, match="3D camera"):
        s2.at("go").lift()
    with pytest.raises(ValueError, match="orbit"):
        Scene("bad").use_camera("ortho", flat=6.0)


def test_lift_is_the_same_whatever_order_the_script_writes():
    def build(lift_first):
        s = _flat_scene()
        if lift_first:
            s.at("go").lift(until="landed")
            s.at(0.2).set("camera.drift", 0.0)
        else:
            s.at(0.2).set("camera.drift", 0.0)
            s.at("go").lift(until="landed")
        return s.engine

    a, b = build(True), build(False)
    for t in (0.0, 1.5, 3.0, 4.5, 6.0):
        assert np.allclose(_screen(a, t), _screen(b, t))


def test_focus_pull_on_a_lifted_camera_says_how_to_zoom():
    s = _flat_scene()
    s.at("go").lift()
    with pytest.raises(VerbError, match="fov"):
        s.at("landed").focus_pull(_dot(s))


def _dot(s):
    from phasor import Dot

    s.add(Dot(id="d", pos=(1.0, 1.0, 0.0)))
    return "d"
