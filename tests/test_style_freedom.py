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


# ---------------------------------------------------------------- 3. taste stays checked


def _aesthetic(e, code):
    return [i for i in e.lint(aesthetic=True) if i.code == code]


@pytest.mark.parametrize("passes", MODES)
def test_off_palette_warns_for_a_color_far_from_every_role(passes):
    s = _scene(passes)
    s.add(Dot(id="near", pos=(-2, 0), r=0.4, style={"color": "#7AA2F7"}))  # = secondary
    s.add(Dot(id="role", pos=(0, 0), r=0.4, style={"color": "highlight"}))
    s.add(Dot(id="green", pos=(2, 0), r=0.4, style={"color": "#83C167"}))  # Manim GREEN
    s.marker("beat", 1.0)
    (issue,) = _aesthetic(s.engine, "off_palette")
    assert issue.ids == ("green",) and issue.path == "green.color"
    assert issue.severity == "warning" and issue.times and not issue.fix
    assert "'#83C167'" in issue.message and "highlight" in issue.message
    assert not [i for i in s.engine.lint() if i.code == "off_palette"]  # aesthetic only


@pytest.mark.parametrize("passes", MODES)
def test_off_palette_judges_a_scene_background_against_the_themes(passes):
    near = _scene(passes, background="#0B0E15")
    near.add(Dot(id="a", pos=(0, 0), r=0.4, style={"color": "highlight"}))
    assert not _aesthetic(near.engine, "off_palette")
    far = _scene(passes, background="#3A2A10")
    far.add(Dot(id="a", pos=(0, 0), r=0.4, style={"color": "highlight"}))
    (issue,) = _aesthetic(far.engine, "off_palette")
    assert issue.path == "meta.background" and issue.ids == () and issue.times


@pytest.mark.parametrize("passes", MODES)
def test_off_scale_warns_for_sizes_off_the_themes_type_and_stroke_scale(passes):
    s = _scene(passes)
    # camera 8 units tall: label is 0.042 x 8 = 0.336 world units; bold is 0.008 x 8 = 0.064
    s.add(Text(id="ok", text="near label", pos=(-3, 2), style={"size": 0.34}))
    s.add(Text(id="huge", text="huge", pos=(0, -1), style={"size": 1.5}))
    s.add(Line(id="wide", points=[(-3, -3), (3, -3)], style={"stroke": 0.3}))
    s.add(Line(id="fine", points=[(-3, 3), (3, 3)], style={"stroke": 0.032}))  # = regular
    s.marker("beat", 1.0)
    issues = {i.ids[0]: i for i in _aesthetic(s.engine, "off_scale")}
    assert set(issues) == {"huge", "wide"}
    assert "'title'" in issues["huge"].message and "'bold'" in issues["wide"].message
    assert all(not i.fix and i.times for i in issues.values())


def test_off_scale_follows_the_zoom_and_reports_when():
    s = _scene()
    s.add(Text(id="t", text="x", pos=(0, 0), style={"size": 0.34}))  # label at height 8
    s.at(2.0).to("camera.height", 2, duration=0.5)  # zoomed in, it is 4 x the label
    s.marker("late", 3.0)
    (issue,) = _aesthetic(s.engine, "off_scale")
    assert issue.times and min(issue.times) >= 2.0


@pytest.mark.parametrize("passes", MODES)
def test_explicit_widths_leave_stroke_mix_alone(passes):
    s = _scene(passes)
    s.add(Line(id="a", points=[(-3, 0), (3, 0)], style={"stroke": "hairline"}))
    s.add(Line(id="b", points=[(-3, 1), (3, 1)], style={"stroke": "bold"}))
    s.add(Line(id="c", points=[(-3, 2), (3, 2)], style={"stroke": 0.5}))
    s.marker("beat", 1.0)
    assert not _aesthetic(s.engine, "stroke_mix")  # two roles; the typed width is not a role
    s.engine.autofix(aesthetic=True)
    assert s.engine.document.object("c").style["stroke"] == 0.5


# ---------------------------------------------------------------- 4. polish offers the snap


def _typed_scene() -> Scene:
    s = _scene(background="#3A2A10")
    s.add(Dot(id="sun", pos=(0, 0), r=0.6, style={"color": "#83C167"}))
    s.add(Text(id="name", text="sun", pos=(0, 1.5), style={"size": 1.5, "color": "#FFFFFF"}))
    s.marker("beat", 1.0)
    return s


def test_critique_offers_the_snap_and_polish_does_not_apply_it():
    e = _typed_scene().engine
    report = e.critique()
    by_code = {i["code"]: i for i in report["issues"]}
    assert by_code["off_palette"]["snap"] and by_code["off_scale"]["snap"]
    offered = [op for i in report["issues"] for op in i["snap"]]
    assert {"op": "set_style", "id": "name", "key": "size", "role": "title"} in offered
    assert all(not i["fix"] for i in report["issues"] if i["code"].startswith("off_"))
    before = e.document
    e.polish()
    for oid in ("sun", "name"):  # nothing typed was replaced, silently or by a suggestion
        assert e.document.object(oid).style == before.object(oid).style
    assert e.document.meta.background == "#3A2A10"
    e.autofix(aesthetic=True)
    assert e.document.object("sun").style["color"] == "#83C167"


def test_polish_snaps_only_when_asked():
    e = _typed_scene().engine
    r = e.polish(snap=True)
    assert e.document.object("sun").style["color"] in ("highlight", "secondary", "primary")
    assert e.document.object("name").style["size"] == "title"
    assert not [i for i in r["remaining"] if i["code"] == "off_scale"]
    # a script owns its meta: the background is reported with no snap (change the script)
    assert e.document.meta.background == "#3A2A10"
    (bg,) = [i for i in r["remaining"] if i["path"] == "meta.background"]
    assert bg["snap"] == []


def test_polish_snaps_a_documents_background_when_asked():
    raw = _typed_scene().engine.document.model_dump(mode="json")
    raw["meta"]["source"] = "document"
    e = Engine(Document.model_validate(raw), passes=None)
    e.polish()
    assert e.document.meta.background == "#3A2A10"
    e.polish(snap=True)
    assert e.document.meta.background is None


# ---------------------------------------------------------------- 5. it renders, the exit demo


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_explicit_colors_and_background_reach_the_pixels_exactly():
    """Paint below the knee leaves the tone map unchanged: a typed hex color comes out as
    itself, linear RGBA is sRGB-encoded once, and the background is the scene's."""
    s = Scene("px", look="none", background="#101820")
    s.set("camera.height", 8)
    s.add(Dot(id="a", pos=(-2, 0), r=1.0, style={"color": "#58C4DD"}))
    s.add(Dot(id="b", pos=(2, 0), r=1.0, style={"color": (0.2, 0.05, 0.4, 1.0)}))
    img = s.engine.still(0.0, (480, 270)).astype(int)
    assert abs(img[135, 180] - (0x58, 0xC4, 0xDD)).max() <= 1
    assert abs(img[135, 300] - (124, 63, 170)).max() <= 1
    assert abs(img[5, 5] - (0x10, 0x18, 0x20)).max() <= 1


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_typed_stroke_color_wins_over_the_prism_looks_hue():
    """The look fills what a scene leaves unset; a color the scene typed is set, so the look
    runs no spectrum along that line (it still gets the look's glow and dim)."""
    s = Scene("prism_red", look="prism")
    s.set("camera.height", 8)
    s.add(Line(id="red", points=[(-3, 0), (3, 0)], style={"color": "#FF0000", "stroke": 0.2}))
    s.add(Line(id="rainbow", points=[(-3, 1), (3, 1)]))
    doc = s.engine.document
    assert "hue_along" not in doc.object("red").params
    assert doc.object("rainbow").params["hue_along"] > 0  # roles: the look as before
    img = s.engine.still(0.0, (480, 270)).astype(int)
    for x in (180, 240, 300):
        assert img[135, x, 0] > 200 and img[135, x, 1] < 40 and img[135, x, 2] < 40


def _own_colors():
    import runpy
    from pathlib import Path

    path = Path(__file__).parent.parent / "examples" / "own_colors.py"
    return runpy.run_path(str(path), run_name="example")["s"].engine


def test_exit_demo_warns_where_it_should_and_offers_the_snaps():
    e = _own_colors()
    assert not [i for i in e.lint() if i.severity == "error"]
    found = {(i.code, i.path) for i in e.lint(aesthetic=True) if i.code.startswith("off_")}
    assert found == {("off_palette", "outer.color"), ("off_palette", "outer.label.color"),
                     ("off_scale", "orbit.stroke")}  # fmt: skip
    offered = [op for i in e.critique()["issues"] for op in i["snap"]]
    assert {"op": "set_style", "id": "orbit", "key": "stroke", "role": "bold"} in offered
    assert e.document.object("inner").style["color"] == "#58C4DD"  # near the palette: quiet


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_exit_demo_golden_storyboard():
    from tests.test_beauty import _golden

    e = _own_colors()
    sheet = e.storyboard([1.0, 1.8, 2.6, 3.4], size=(320, 180))["image"]
    _golden("own_colors", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- audit (2026-11-23)


def _paths(key: str, value, tmp_path):
    """Every way a style value reaches a document: name -> call; a call returns True when
    the value was accepted, and returns False or raises a ValueError when it was refused."""
    from phasor.workspace.tools import Workspace

    typ = {"color": "dot", "stroke": "line", "size": "text"}[key]
    params = {"line": {"points": [[0, 0], [1, 0]]}, "text": {"text": "hi"}, "dot": {}}[typ]
    obj = {"id": "a", "type": typ, "params": params, "style": {key: value}}

    def spec():
        ObjectSpec.model_validate(obj)
        return True

    def apply():
        return _scene().engine.apply({"op": "add_object", "object": obj}).ok

    def set_style():
        s = _scene()
        s.engine.apply({"op": "add_object", "object": {**obj, "style": {}}}).raise_for_errors()
        return s.engine.apply({"op": "set_style", "id": "a", "key": key, "role": value}).ok

    def mcp():
        w = Workspace(tmp_path)
        w.scene_new(title="x", look="none")
        return w.scene_edit({"op": "add_object", "object": obj})["ok"]

    return {"spec": spec, "apply": apply, "set_style": set_style, "mcp": mcp}


def _refused(call) -> bool:
    try:
        return not call()
    except ValueError:  # ValidationError, DocumentError and ToolError are ValueErrors
        return True


@pytest.mark.parametrize("key", ["color", "stroke", "size"])
@pytest.mark.parametrize("value", [10**400, [10**400, 0, 0], 1e308, 1e7], ids=str)
def test_huge_numbers_are_refused_cleanly_on_every_path(key, value, tmp_path):
    """An int too large for a float escaped as OverflowError (pydantic wraps only
    ValueError); a size of 1e308 was accepted and crashed the renderer (round(inf))."""
    for name, call in _paths(key, value, tmp_path).items():
        assert _refused(call), name


def test_the_size_bound_is_finite_and_sizes_up_to_it_are_taken():
    from phasor.core.schema import MAX_SIZE
    from phasor.style import explicit

    assert math.isfinite(MAX_SIZE) and explicit.MAX_SIZE == MAX_SIZE
    assert ObjectSpec(id="t", type="text", style={"size": MAX_SIZE}).style["size"] == MAX_SIZE
    with pytest.raises(ValueError, match="at most"):
        ObjectSpec(id="t", type="text", style={"size": MAX_SIZE * 1.001})


def test_huge_numbers_in_a_background_or_a_file_on_disk_are_validation_errors(tmp_path):
    from phasor.core import io

    with pytest.raises(ValueError):
        Meta(title="x", background=[10**400, 0, 0])
    big = "1" + "0" * 400
    for style in (f'{{"size": {big}}}', f'{{"color": [{big}, 0, 0]}}', '{"size": 1e308}'):
        p = tmp_path / "big.phasor.json"
        obj = f'{{"id": "t", "type": "text", "params": {{"text": "hi"}}, "style": {style}}}'
        p.write_text(f'{{"schema_version": 9, "meta": {{"title": "x"}}, "objects": [{obj}]}}',
                     encoding="utf-8")  # fmt: skip
        with pytest.raises(DocumentError):
            io.load_document(p)
    p.write_text(f'{{"schema_version": 9, "meta": {{"title": "x", "background": [{big}, 0, 0]}}'
                 "}", encoding="utf-8")  # fmt: skip
    with pytest.raises(DocumentError):
        io.load_document(p)


def test_a_background_is_opaque_and_alpha_below_one_is_refused(tmp_path):
    from phasor.workspace.tools import Workspace

    assert Meta(title="x", background="#101820FF").background == "#101820FF"
    assert Meta(title="x", background=(0.1, 0.2, 0.3, 1.0)).background == (0.1, 0.2, 0.3, 1.0)
    w = Workspace(tmp_path)
    w.scene_new(title="x", look="none")
    for bad in ("#00000000", "#FFFFFF80", "#FF000000", (0.1, 0.2, 0.3, 0.5)):
        with pytest.raises(ValueError, match="opaque"):
            Meta(title="x", background=bad)
        with pytest.raises(ValueError):
            _scene(background=bad)
        r = _scene().engine.apply({"op": "set_meta", "field": "background", "value": bad})
        assert not r.ok
        r = w.scene_edit({"op": "set_meta", "field": "background", "value": bad})
        assert not r["ok"] and w.engine.document.meta.background is None
    # an object's color keeps its alpha (it is drawn over something)
    assert ObjectSpec(id="a", type="dot", style={"color": "#FFFFFF80"}).style["color"] == (
        "#FFFFFF80")  # fmt: skip
