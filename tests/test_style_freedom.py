"""Milestone 18: style freedom. Explicit colors, sizes and a background beside roles; the
aesthetic rules ``off_palette`` and ``off_scale``; ``polish`` offers the snap to a role and
applies it only when asked."""

from __future__ import annotations

import json
import math

import pytest

from phasor import Dot, Line, Scene, Text
from phasor.core.color import hex_to_linear
from phasor.core.issues import DocumentError
from phasor.core.migrate import migrate
from phasor.core.schema import SCHEMA_VERSION, Document, Meta, ObjectSpec, Overrides
from phasor.engine import Engine
from phasor.style.theme import load_theme
from tests.conftest import gl_available

MODES = [None, pytest.param("auto", marks=[pytest.mark.gl,
                                          pytest.mark.skipif(not gl_available(), reason="gl")])]  # fmt: skip


def _scene(passes=None, **kw) -> Scene:
    s = Scene("m18", passes=passes, look="none", **kw)
    s.set("camera.height", 8)
    return s


# ---------------------------------------------------------------- 1. the document


def test_style_takes_a_role_or_an_explicit_value_and_stores_it_canonically():
    spec = ObjectSpec(id="a", type="dot", style={"color": "#58c4dd", "glow": "soft"})
    assert spec.style == {"color": "#58C4DD", "glow": "soft"}
    assert ObjectSpec(id="a", type="dot", style={"color": "#abc"}).style["color"] == "#AABBCC"
    assert ObjectSpec(id="a", type="dot", style={"color": [0.1, 0.2, 0.3]}).style["color"] == (
        0.1, 0.2, 0.3, 1.0)  # fmt: skip
    assert ObjectSpec(id="t", type="text", style={"size": 1}).style["size"] == 1.0
    for bad in ("#12345", "#GG0000", True, -0.2, 0.0, math.nan, (0.1, 0.2), (2.0, 0, 0),
                (0.1, 0.2, math.inf)):  # fmt: skip
        with pytest.raises(ValueError):
            ObjectSpec(id="a", type="dot", style={"color": bad})


def test_meta_background_is_a_color_or_none():
    assert Meta(title="x").background is None
    assert Meta(title="x", background="#101820").background == "#101820"
    assert Meta(title="x", background=(0.01, 0.02, 0.03, 1)).background == (0.01, 0.02, 0.03, 1.0)
    for bad in ("midnight", 0.5, (1, 2, 3)):
        with pytest.raises(ValueError):
            Meta(title="x", background=bad)


def test_a_version_8_document_migrates_to_9_and_reads_the_same():
    raw = {"schema_version": 8, "meta": {"title": "old", "palette": None, "modules": []},
           "objects": [{"id": "d", "type": "dot", "params": {"r": 0.2},
                        "style": {"color": "highlight"}, "always": []}]}  # fmt: skip
    doc = Document.model_validate(migrate(json.loads(json.dumps(raw))))
    assert doc.schema_version == SCHEMA_VERSION == 9
    assert doc.meta.background is None and doc.objects[0].style == {"color": "highlight"}
    old = Engine(doc, passes=None).state_at(0)
    assert old.background == load_theme(doc.meta.theme).background
    assert old["d"].roles["color"] == "highlight"


def test_explicit_values_survive_save_and_load_byte_for_byte(tmp_path):
    s = _scene(background="#101820")
    s.add(Dot(id="a", pos=(0, 0), style={"color": "#58C4DD"}))
    s.add(Text(id="t", text="x", pos=(0, 1), style={"size": 0.4, "color": (0.2, 0.3, 0.4)}))
    s.engine.save(tmp_path / "s.phasor.json")
    first = (tmp_path / "s.phasor.json").read_bytes()
    e = Engine.open(tmp_path / "s.phasor.json", passes=None)
    e.save(tmp_path / "t.phasor.json")
    assert (tmp_path / "t.phasor.json").read_bytes() == first


# ---------------------------------------------------------------- 2. resolution


def test_a_hex_color_is_decoded_to_linear_once_and_a_tuple_is_linear_already():
    s = _scene()
    s.add(Dot(id="hex", pos=(0, 0), style={"color": "#58C4DD"}))
    s.add(Dot(id="lin", pos=(1, 0), style={"color": (0.2, 0.3, 0.4, 0.5)}))
    st = s.engine.state_at(0)
    assert st["hex"].style["color"] == pytest.approx(hex_to_linear("#58C4DD"))
    assert st["hex"].style["color"][1] < 0.6  # linear, not sRGB 0.77
    assert st["lin"].style["color"] == (0.2, 0.3, 0.4, 0.5)
    assert st["hex"].roles["color"] == "#58C4DD"  # what the scene says, for lint and the UI


def test_sizes_are_world_units_whatever_the_zoom():
    s = _scene()
    s.add(Text(id="t", text="x", pos=(0, 0), style={"size": 0.5}))
    s.add(Line(id="l", points=[(-1, 0), (1, 0)], style={"stroke": 0.05}))
    s.at(1.0).to("camera.height", 4, duration=1.0)
    e = s.engine
    for t, height in ((0.0, 8.0), (2.0, 4.0)):
        st = e.state_at(t)
        assert st.camera.units_per_height() == pytest.approx(height)
        # the drawing code multiplies the share back by units_per_height: world units
        assert st["t"].style["size"] * height == pytest.approx(0.5)
        assert st["l"].style["stroke"] * height == pytest.approx(0.05)
    # a role stays a share of the frame height: twice as big in the world at half the zoom
    assert e.state_at(0)["t"].roles["leader"] == "hairline"


def test_explicit_sizes_in_3d_are_exact_at_the_objects_position():
    s = Scene("m18_3d", passes=None, look="none")
    s.use_camera("orbit", distance=10, elevation=25, azimuth=30, drift=0)
    s.add(Text(id="t", text="x", pos=(1, 2, 0.5), style={"size": 0.3}))
    st = s.engine.state_at(0)
    assert st["t"].style["size"] * st.camera.units_per_height((1, 2, 0.5)) == pytest.approx(0.3)


def test_roles_only_groups_refuse_explicit_values():
    s = _scene()
    for style in ({"leader": "#FFFFFF"}, {"color": 0.4}, {"size": "#FFFFFF"}):
        with pytest.raises(DocumentError, match="bad_role") as err:
            s.add(Text(id="t", text="x", pos=(0, 0), style=style))
        assert any(w in str(err.value) for w in ("roles only", "not a color", "not a size"))
    assert s.engine.document.object("t") is None
    for glow in (0.5, "#FFFFFF"):  # glow, surface and colormap keys take roles only
        with pytest.raises(DocumentError, match="roles only"):
            s.add(Dot(id="d", pos=(0, 0), style={"glow": glow}))


def test_set_style_takes_an_explicit_value_and_overrides_keep_it(tmp_path):
    s = _scene()
    s.add(Dot(id="a", pos=(0, 0)))
    s.engine.save(tmp_path / "a.phasor.json")
    e = Engine.open(tmp_path / "a.phasor.json", passes=None)
    assert e.apply({"op": "set_style", "id": "a", "key": "color", "role": "#ff8800"}).ok
    assert e.document.object("a").style["color"] == "#FF8800"
    ov = Overrides(styles={"a.color": (0.5, 0.25, 0.0)})
    assert ov.styles["a.color"] == (0.5, 0.25, 0.0, 1.0)


def test_a_scene_background_reaches_the_state_and_the_default_is_the_theme():
    plain = _scene()
    plain.add(Dot(id="a", pos=(0, 0)))
    theme = load_theme("midnight_glow")
    assert plain.engine.state_at(0).background == theme.background
    s = _scene(background="#101820")
    s.add(Dot(id="a", pos=(0, 0)))
    assert s.engine.state_at(0).background == pytest.approx(hex_to_linear("#101820"))
    doc = Document.model_validate({**s.engine.document.model_dump(mode="json"),
                                   "meta": {**s.engine.document.meta.model_dump(mode="json"),
                                            "source": "document"}})  # fmt: skip
    e = Engine(doc, passes=None)
    assert e.apply({"op": "set_meta", "field": "background", "value": "#000000"}).ok
    assert e.document.meta.background == "#000000"
    assert e.apply({"op": "set_meta", "field": "background", "value": None}).ok
    assert e.state_at(0).background == theme.background
