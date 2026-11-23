"""Style transitions: ``.to("id.style.color", "highlight")`` and the same for fill, stroke and
text sizes interpolate the resolved value (linear RGBA for a color) between two roles or
explicit values. The document stores a role or a value at each key (contract 2); the value
at t is pure in (document, t) (contract 3); colors mix in linear light (contract 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text, shapes
from phasor.core.color import hex_to_linear
from phasor.core.issues import DocumentError
from phasor.engine import Engine
from phasor.style.theme import load_theme

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


def _scene(passes=None) -> Scene:
    s = Scene("restyle", passes=passes, look="none")
    s.set("camera.height", 8)
    s.marker("go", 1.0)
    return s


def _theme(s):
    return load_theme(s.engine.document.meta.theme)


def test_a_color_moves_between_two_roles_in_linear_rgba():
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0), style={"color": "primary"}))
    s.at("go").to("d.style.color", "highlight", duration=1.0, ease="linear")
    th = _theme(s)
    a, b = np.array(th.values["emphasis"]["primary"]), np.array(th.values["emphasis"]["highlight"])
    e = s.engine
    assert e.state_at(0.5)["d"].style["color"] == pytest.approx(tuple(a))
    mid = np.array(e.state_at(1.5)["d"].style["color"])
    assert mid == pytest.approx((a + b) / 2)  # linear light, not sRGB
    assert e.state_at(2.5)["d"].style["color"] == pytest.approx(tuple(b))
    # the document still holds a role at the key and the role in the track
    doc = e.document
    assert doc.object("d").style == {"color": "primary"}
    assert doc.timeline.tracks["d.style.color"][0].to == "highlight"
    # what the scene typed, in effect at t: the base, then the target once it is reached
    assert e.state_at(0.5)["d"].roles["color"] == "primary"
    assert e.state_at(1.2)["d"].roles["color"] == "primary"
    assert e.state_at(1.8)["d"].roles["color"] == "highlight"


def test_a_color_moves_to_an_explicit_value_decoded_to_linear_once():
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0)))
    s.at("go").to("d.style.color", "#58c4dd", duration=1.0, ease="linear")
    assert s.engine.document.timeline.tracks["d.style.color"][0].to == "#58C4DD"  # canonical
    end = s.engine.state_at(3.0)["d"].style["color"]
    assert end == pytest.approx(hex_to_linear("#58C4DD"))
    s.at(4.0).to("d.style.color", (0.1, 0.2, 0.3), duration=1.0)
    assert s.engine.state_at(6)["d"].style["color"] == pytest.approx((0.1, 0.2, 0.3, 1.0))


def test_fill_stroke_and_text_size_move_too_and_world_sizes_follow_the_camera():
    s = _scene()
    s.add(shapes.Circle(id="c", r=1.0, style={"fill": "none", "stroke": "hairline"}))
    s.add(Text(id="t", text="x", pos=(0, 2), style={"size": "label"}))
    s.at("go").to("c.style.fill", "solid", duration=1.0, ease="linear")
    s.at("go").to("c.style.fill_color", "danger", duration=1.0, ease="linear")
    s.at("go").to("c.style.stroke", 0.2, duration=1.0, ease="linear")  # world units
    s.at("go").to("t.style.size", "title", duration=1.0, ease="linear")
    th = _theme(s)
    e = s.engine
    c = e.state_at(1.5)["c"]
    assert c.style["fill"] == pytest.approx(0.5 * th.values["fill"]["solid"])
    hair = th.values["stroke"]["hairline"]
    assert c.style["stroke"] == pytest.approx((hair + 0.2 / 8) / 2)  # 0.2 world of an 8 frame
    end = e.state_at(3)["c"].style
    assert end["stroke"] == pytest.approx(0.2 / 8) and end["fill"] == th.values["fill"]["solid"]
    t = e.state_at(1.5)["t"].style["size"]
    assert t == pytest.approx((th.values["text"]["label"] + th.values["text"]["title"]) / 2)
    s.at(4.0).to("camera.height", 16, duration=0.0)
    assert e.state_at(5)["c"].style["stroke"] == pytest.approx(0.2 / 16)


def test_a_style_transition_is_pure_and_order_free():
    def build(order):
        s = _scene()
        s.add(Dot(id="d", pos=(0, 0)))
        steps = [lambda: s.at(3.0).to("d.style.color", "danger", duration=1.0),
                 lambda: s.at("go").to("d.style.color", "highlight", duration=1.0)]  # fmt: skip
        for k in order:
            steps[k]()
        return s.engine

    a, b = build([1, 0]), build([1, 0])
    for t in (0.0, 1.3, 2.0, 3.5, 9.0):
        assert a.state_at(t)["d"].style == b.state_at(t)["d"].style
        assert a.state_at(t)["d"].style == a.state_at(t)["d"].style


def test_overshooting_easing_never_makes_a_color_negative_or_an_opacity_above_one():
    s = _scene()
    s.add(shapes.Circle(id="c", r=1.0, style={"fill": "soft", "color": "muted"}))
    s.at("go").to("c.style.color", "highlight", duration=1.0, ease="back")
    s.at("go").to("c.style.fill", "solid", duration=1.0, ease="spring")
    for t in np.linspace(1.0, 2.0, 41):
        st = s.engine.state_at(float(t))["c"].style
        assert min(st["color"]) >= 0.0 and 0.0 <= st["fill"] <= 1.0


@pytest.mark.parametrize(
    "path, value, code",
    [
        ("d.style.glow", "strong", "bad_style_track"),  # glow is life's: not animated here
        ("d.style.color", "bold", "bad_role"),  # not an emphasis role
        ("d.style.color", 0.3, "bad_role"),  # a size is not a color
        ("d.style.nope", "primary", "unknown_style"),
        ("ghost.style.color", "primary", "unknown_param"),
    ],
)
def test_a_bad_style_track_is_a_clear_document_error(path, value, code):
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0)))
    r = s.engine.apply({"op": "add_segment", "path": path,
                        "segment": {"t0": 0, "t1": 1, "to": value}}, origin="script")  # fmt: skip
    assert not r.ok
    assert code in {i.code for i in r.errors}, [i.message for i in r.errors]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 1e300, 10**400, True])
def test_non_finite_or_huge_style_targets_fail_validation_cleanly(bad):
    s = _scene()
    s.add(Line(id="l", points=[(0, 0), (1, 1)]))
    r = s.engine.apply({"op": "add_segment", "path": "l.style.stroke",
                        "segment": {"t0": 0, "t1": 1, "to": bad}}, origin="script")  # fmt: skip
    assert not r.ok


def test_removing_an_object_removes_its_style_tracks():
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0)))
    s.at("go").to("d.style.color", "highlight")
    r = s.engine.apply({"op": "remove_object", "id": "d"}, origin="script")
    assert r.ok, [i.message for i in r.errors]
    assert "d.style.color" not in s.engine.document.timeline.tracks


def test_a_child_object_named_style_keeps_its_own_parameters():
    """``a.style`` may be a real child object: its parameter paths win (existence first)."""
    s = _scene()
    s.add(Dot(id="a", pos=(0, 0)))
    s.add(Dot(id="a.style", pos=(1, 0), r=0.1))
    s.at("go").to("a.style.r", 0.3, duration=1.0, ease="linear")
    assert s.engine.state_at(3)["a.style"].params.r == pytest.approx(0.3)
    with pytest.raises(DocumentError, match="unknown parameter"):  # the child's, not a's style
        s.at("go").to("a.style.color", "highlight")


def test_the_inspector_shows_a_transition_in_progress():
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0)))
    s.at("go").to("d.style.color", "highlight", duration=1.0, ease="linear")
    (entry,) = [o for o in s.engine.inspect(1.5)["objects"] if o["id"] == "d"]
    assert entry["restyling"] == {"color": {"from": "primary", "to": "highlight",
                                            "progress": 0.5}}  # fmt: skip
    (entry,) = [o for o in s.engine.inspect(3.0)["objects"] if o["id"] == "d"]
    assert "restyling" not in entry and entry["style"]["color"] == "highlight"


def test_the_morph_verb_animates_a_style_with_the_themes_timing_and_strict_lint_is_clean():
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0), style={"color": "highlight"}))
    s.at("go").morph("d.style.color", "secondary")
    seg = s.engine.document.timeline.tracks["d.style.color"][0]
    assert seg.verb == "morph"
    assert "raw_motion" not in {i.code for i in s.engine.lint(strict=True)}


def test_raw_motion_names_the_object_of_a_style_transition():
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0), style={"color": "highlight"}))
    s.at("go").to("d.style.color", "secondary")
    (raw,) = [i for i in s.engine.lint(strict=True) if i.code == "raw_motion"]
    assert raw.ids == ("d",)


@pytest.mark.parametrize("passes", MODES)
def test_off_palette_judges_an_explicit_color_a_transition_reaches(passes):
    s = _scene(passes)
    s.add(Dot(id="d", pos=(0, 0), style={"color": "highlight"}))
    s.at("go").to("d.style.color", "#00FF00", duration=0.5)
    (issue,) = [i for i in s.engine.lint(aesthetic=True) if i.code == "off_palette"]
    assert issue.ids == ("d",) and issue.times and issue.times[0] >= 1.5 - 1e-9
    # a snap would set the base style, not the track's target: not offered
    crit = s.engine.critique()
    assert not any(op.get("op") == "set_style" and op.get("id") == "d"
                   for entry in crit["issues"] for op in entry.get("snap", []))  # fmt: skip


def test_a_saved_scene_with_a_style_track_loads_and_reads_the_same(tmp_path):
    s = _scene()
    s.add(Dot(id="d", pos=(0, 0)))
    s.at("go").to("d.style.color", "highlight", duration=1.0)
    s.engine.save(tmp_path / "x.phasor.json")
    e = Engine.open(tmp_path / "x.phasor.json", passes=None)
    assert e.state_at(1.4)["d"].style == s.engine.state_at(1.4)["d"].style


def test_a_document_error_names_a_track_on_a_roles_only_key_clearly():
    s = _scene()
    s.add(Line(id="l", points=[(0, 0), (1, 1)]))
    with pytest.raises(DocumentError) as e:
        s.at("go").to("l.style.colormap", "ember")
    assert "colormap" in str(e.value)
