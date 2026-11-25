"""Group F1 verbs: follow (the camera rides a feature), ripple and spotlight."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Scene, Text, cx
from phasor.core.issues import DocumentError
from phasor.time.verbs import VerbError
from tests.conftest import gl_available

needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
EXAMPLE = Path(__file__).parents[1] / "examples" / "verbs_camera.py"


def _scene() -> Scene:
    s = Scene("f1", passes=None)
    s.add(cx.DomainColoring(id="dc", function="(z - a)*(z + 1)", a=(1.0, 0.0), size=(16, 10)))
    s.add(Dot(id="d", r=0.3, pos=(-1.0, 0.0)))
    for name, t in (("go", 1.0), ("drift", 2.0), ("rest", 5.0), ("end", 7.0)):
        s.marker(name, t)
    return s


def _segments(s: Scene, prefix: str):
    tracks = s.engine.document.timeline.tracks
    return [seg for path, segs in tracks.items() if path.startswith(prefix) for seg in segs]


def _semantic(s: Scene, t: float, object_id: str) -> dict:
    return s.engine.inspect(t, id=object_id)["objects"][0]["semantic"]


# ---------------------------------------------------------------- ripple


def test_ripple_spreads_rings_from_a_zero_and_they_fade():
    s = _scene()
    s.at("go").ripple("dc", at="zeros", index=1)
    segs = _segments(s, "dc.ripple.")
    assert segs and all(seg.verb == "ripple" for seg in segs)
    assert _semantic(s, 0.5, "dc.ripple")["rings_visible"] == 0  # not started
    mid = _semantic(s, 1.5, "dc.ripple")
    assert mid["rings_visible"] >= 2 and mid["spot"] == [1.0, 0.0, 0.0]
    assert _semantic(s, 3.0, "dc.ripple")["rings_visible"] == 0  # all gone


def test_ripple_rides_a_zero_that_moves():
    s = _scene()
    s.at("drift").morph("dc.a", (2.0, 1.0), until="rest")
    s.at("drift").ripple("dc", at="zeros", index=1)
    a = np.array(_semantic(s, 2.5, "dc.ripple")["spot"])
    b = np.array(_semantic(s, 2.9, "dc.ripple")["spot"])
    assert np.linalg.norm(b - a) > 0.05  # the spot moved with the zero


def test_ripple_at_a_point_and_its_errors():
    s = _scene()
    s.at("go").ripple("d", at=(0.5, 0.5))
    assert _semantic(s, 1.3, "d.ripple")["spot"] == [0.5, 0.5, 0.0]
    with pytest.raises(VerbError, match="already ripples"):
        s.at("drift").ripple("d", at=(1.0, 0.0))
    with pytest.raises(VerbError, match="no feature 'tip'"):
        s.at("go").ripple("dc", at="tip")
    with pytest.raises(VerbError, match="index 5"):
        s.at("go").ripple("dc", at="zeros", index=5)
    with pytest.raises(VerbError, match="no object 'nope'"):
        s.at("go").ripple("nope")


# ---------------------------------------------------------------- spotlight


def test_spotlight_fades_in_glides_and_fades_out():
    s = _scene()
    s.add(Dot(id="e", r=0.2, pos=(2.0, 1.0)))
    s.at("go").spotlight("d")
    end = s.at("drift").spotlight("e", until="rest")._end
    assert end == pytest.approx(3.0 + 1.0)  # off at rest, plus the fade
    segs = _segments(s, "spotlight.")
    assert segs and all(seg.verb == "spotlight" for seg in segs)
    e = s.engine
    assert e.state_at(0.5)["spotlight"].params.opacity == 0.0
    assert e.state_at(1.9)["spotlight"].params.opacity == pytest.approx(1.0, abs=0.05)
    assert e.state_at(1.9)["spotlight"].params.center == pytest.approx((-1.0, 0.0, 0.0))
    assert e.state_at(3.5)["spotlight"].params.center == pytest.approx((2.0, 1.0, 0.0))
    assert e.state_at(6.5)["spotlight"].params.opacity == 0.0
    assert e.state_at(1.9)["spotlight"].layer == "overlay"
    sem = _semantic(s, 1.9, "spotlight")
    assert sem["backdrop"] and sem["radius"] > 0


def test_spotlight_leaves_its_target_visible_and_is_left_alone_by_dim_others():
    s = _scene()
    s.at("go").spotlight("d")
    s.at("drift").dim_others("d")
    assert "spotlight.dim" not in s.engine.document.timeline.tracks
    assert s.engine.state_at(3.0)["spotlight"].params.opacity == pytest.approx(1.0)
    assert not [i for i in s.engine.lint(strict=True) if i.code in ("occluded", "orphan")]


def test_spotlight_errors():
    s = _scene()
    with pytest.raises(VerbError, match="none yet"):
        s.at("go").spotlight()
    s.add(Text(id="title", text="hi", pos=(0.2, 0.1, 0), layer="overlay"))
    with pytest.raises(VerbError, match="on the overlay"):
        s.at("go").spotlight("title")
    with pytest.raises(VerbError, match="no marker"):
        s.at("go").spotlight("d", until="never")


# ---------------------------------------------------------------- follow


def test_follow_keeps_the_camera_on_a_moving_zero_and_comes_back():
    s = _scene()
    s.at("drift").morph("dc.a", (2.0, 1.0), until="rest")
    s.at("drift").follow("dc", feature="zeros", index=1, until="rest")
    segs = _segments(s, "dc.follow.")
    assert segs and all(seg.verb == "follow" for seg in segs)
    e = s.engine
    # the document leaves the camera alone: the bake moves it
    assert "camera.center" not in e.document.timeline.tracks
    assert e.state_at(1.0).values["camera.center"] == pytest.approx((0.0, 0.0, 0.0))
    for t in (3.0, 3.8, 4.3):
        zero = np.array(_semantic(s, t, "dc")["zeros"])
        zero = zero[np.argmin(np.linalg.norm(zero - (2.0, 1.0), axis=1))]
        center = np.asarray(e.state_at(t).values["camera.center"])[:2]
        assert np.linalg.norm(center - zero) < 0.05, t
    assert e.state_at(6.5).values["camera.center"] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    # and where the camera is, the inspector sees: the zero is mid-frame
    dc = e.inspect(3.8, id="dc.follow")["objects"][0]
    assert dc["center"]["norm"] == pytest.approx([0.5, 0.5], abs=0.02)


def test_follow_without_until_takes_the_theme_time():
    s = _scene()
    end = s.at("go").follow("d")._end
    assert end == pytest.approx(s.engine._snap.compiled.theme.motion["follow"]["duration"])


def test_follow_in_3d_moves_the_orbit_target():
    s = _scene()
    s.use_camera("orbit")
    s.at("drift").follow("d", until="rest")
    tgt = s.engine.state_at(3.5).values["camera.target"]
    assert tgt == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)


def test_follow_refuses_a_camera_center_the_document_sets_or_animates():
    s = _scene()
    s.at("go").focus_pull("d")
    with pytest.raises(DocumentError, match="follow_conflict|it is animated"):
        s.at("drift").follow("dc", feature="zeros")
    s2 = _scene()
    s2.at("drift").follow("dc", feature="zeros")
    with pytest.raises(DocumentError, match="it is animated"):
        s2.at("go").focus_pull("d")
    s3 = _scene()
    s3.at("go").follow("dc")
    with pytest.raises(VerbError, match="already followed"):
        s3.at("rest").follow("dc", feature="zeros")


# ---------------------------------------------------------------- order


def test_f1_verbs_give_the_same_scene_whatever_order_they_are_written_in():
    def build(order):
        s = _scene()
        s.add(Dot(id="e", r=0.2, pos=(2.0, 1.0)))
        steps = {
            "ripple": lambda: s.at("drift").ripple("dc", at="zeros", index=1),
            "light": lambda: s.at("drift").spotlight("d", until="rest"),
            "follow": lambda: s.at("drift").follow("dc", feature="zeros", index=1, until="rest"),
            "morph": lambda: s.at("drift").morph("dc.a", (2.0, 1.0), until="rest"),
            "flash": lambda: s.at("drift").flash("d"),
            "ring": lambda: s.at("drift").circumscribe("e"),
        }
        for k in order:
            steps[k]()
        return s.engine

    keys = ["ripple", "light", "follow", "morph", "flash", "ring"]
    a, b = build(keys), build(keys[::-1])
    for t in (1.5, 2.4, 3.5, 4.6, 6.0):
        # the children are added in the order the script asks, so compare them as a set
        ia, ib = a.inspect(t, detail="brief"), b.inspect(t, detail="brief")
        assert sorted(ia.pop("objects")) == sorted(ib.pop("objects")) and ia == ib
        assert a.state_at(t).values["camera.center"] == pytest.approx(
            b.state_at(t).values["camera.center"]
        )


def test_a_later_spotlight_glides_from_the_last_one():
    s = _scene()
    s.add(Dot(id="e", r=0.2, pos=(2.0, 1.0)))
    s.at("go").spotlight("d")
    s.at("drift").spotlight("e")
    e = s.engine
    xs = [e.state_at(t)["spotlight"].params.center[0] for t in (2.0, 2.3, 2.6, 3.0)]
    assert xs[0] == pytest.approx(-1.0) and xs[-1] == pytest.approx(2.0)
    assert xs[0] < xs[1] < xs[2] < xs[3]  # a glide, not a jump


# ---------------------------------------------------------------- the example


@needs_gl
def test_spotlight_hides_nothing_in_the_id_pass():
    s = Scene("f1gl")
    s.add(cx.DomainColoring(id="dc", function="z", size=(16, 10)))
    s.add(Dot(id="d", r=0.3, pos=(-1.0, 0.0)))
    s.marker("go", 1.0)
    s.at("go").spotlight("d")
    obj = s.engine.inspect(2.5, id="d")["objects"][0]
    assert obj["visible_fraction"] > 0.9


@needs_gl
def test_the_camera_verbs_example_matches_its_golden():
    import runpy

    from tests.test_beauty import _golden

    e = runpy.run_path(str(EXAMPLE), run_name="x")["s"].engine
    assert not e.lint(strict=True)
    sheet = e.storyboard([0.9, 3.0, 6.4, 7.5], size=(320, 180))["image"]
    _golden("verbs_camera", sheet, e.pass_runner().renderer_class(), threshold=0.97)
