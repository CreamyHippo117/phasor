"""Text as shapes: glyph outlines (``domains/glyph_math.py``), ``as_shapes`` on ``Text`` and
``Equation`` (one ``Path`` part per glyph), ``write`` drawing the outline then the fill, the
bounds on glyph counts and lint ``glyph_limit``. A text that is not drawn as shapes draws as
it always did."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Dot, Equation, Scene, Text
from phasor.core import text as text_metrics
from phasor.core.record import record
from phasor.domains import glyph_math as gm
from phasor.domains import path_math as pm

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


def _scene(passes=None) -> Scene:
    s = Scene("glyphs", passes=passes, look="none")
    s.set("camera.height", 8)
    return s


# ---------------------------------------------------------------- 1. outlines


@pytest.mark.parametrize("text", ["Hello, world", "$e^{i\\pi} + 1 = 0$", "$\\frac{a}{b}$",
                                  "x = 2 $y$", "Ünïcödé ñ"])  # fmt: skip
def test_outlines_are_closed_cubic_subpaths_inside_the_texts_box(text):
    o = gm.outline(text)
    assert o.glyphs and o.total == len(o.glyphs)
    width, box = text_metrics.aspect(text), text_metrics.box(text)
    for g in o.glyphs:
        assert g.subpaths and gm.glyph_name("a") == "a"
        for s in g.subpaths:
            pm.check_subpath(s)
            assert pm.is_closed(s)
            assert not s.flags.writeable  # shared by the cache
        pts = np.vstack(g.subpaths)
        # inside the box the picture fills (a little ink may cross its edge)
        assert pts[:, 0].min() > -0.05 * box and pts[:, 0].max() < width + 0.05 * box
        assert pts[:, 1].min() > -0.08 * box and pts[:, 1].max() < box * 1.08


def test_glyphs_are_named_by_what_they_show_and_keyed_by_font_glyph():
    o = gm.outline("Aa1+")
    assert [g.name for g in o.glyphs] == ["cap_a", "a", "one", "plus_sign"]
    math_ = gm.outline("$a + \\pi = 2$")
    assert [g.name for g in math_.glyphs] == ["a", "plus", "pi", "equal", "two"]
    two = gm.outline("aa")
    assert two.glyphs[0].key == two.glyphs[1].key != gm.outline("b").glyphs[0].key
    assert [g.order for g in math_.glyphs] == list(range(5))  # left to right


def test_the_outline_covers_the_picture():
    """The shapes lie where the picture's ink is (plain text: same font, same width)."""
    for text, least in (("Hello", 0.8), ("$x^2 + 1$", 0.55)):
        h = 120
        ras = text_metrics.rasterize(text, h) > 127
        from PIL import Image, ImageDraw

        img = Image.new("1", (ras.shape[1], ras.shape[0]), 0)
        for g in gm.outline(text).glyphs:
            for s in g.subpaths:
                q = pm.flatten(s, 0.002)
                ImageDraw.Draw(img).polygon([(x * h, ras.shape[0] - y * h) for x, y, _ in q],
                                            fill=1)  # fmt: skip
        ink = np.asarray(img, dtype=bool)
        assert (ras & ink).sum() / ras.sum() > least


def test_quadratic_curves_become_equal_cubics():
    from matplotlib.path import Path as MPath

    verts = np.array([[0, 0], [1, 2], [2, 0], [0, 0]], dtype=float)
    codes = np.array([MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.CLOSEPOLY])
    (sub,) = gm.subpaths(verts, codes)
    pm.check_subpath(sub)
    c = pm.cubics(sub)[0]
    u = 0.3
    quad = (1 - u) ** 2 * verts[0] + 2 * u * (1 - u) * verts[1] + u**2 * verts[2]
    cub = (
        (1 - u) ** 3 * c[0] + 3 * (1 - u) ** 2 * u * c[1] + 3 * (1 - u) * u**2 * c[2] + u**3 * c[3]
    )
    assert cub[:2] == pytest.approx(quad)
    assert pm.is_closed(sub)


def test_a_long_text_keeps_a_bounded_number_of_glyphs():
    long = "ab" * 1000
    o = gm.outline(long)
    assert len(o.glyphs) == gm.MAX_GLYPHS and o.total == 2000
    assert gm.outline("   ").glyphs == () and gm.outline("   ").total == 0


# ---------------------------------------------------------------- 2. Text as shapes


def test_text_as_shapes_has_a_path_part_per_glyph_named_by_meaning():
    s = _scene()
    s.add(Text(id="word", text="Hello", pos=(0, 1), as_shapes=True))
    st = s.engine.state_at(0)
    parts = [o for o in st.objects if o.parent == "word"]
    assert [o.id for o in parts] == ["word.cap_h", "word.e", "word.l", "word.l_2", "word.o"]
    assert all(o.type == "path" for o in parts)
    sem = st["word"].impl.semantics(st)
    assert sem["as_shapes"] is True and sem["glyphs"] == 5 and "glyph_limit" not in sem
    (h,) = [o for o in parts if o.id == "word.cap_h"]
    assert h.roles["fill"] == "solid" and h.params.stroke_opacity == 0.0
    assert h.params.fill_opacity == 1.0 and h.params.progress == 1.0
    # the text itself draws no picture; its parts are seen in the inspector
    rec = record(st["word"].impl, st)
    assert "text" not in rec.kinds
    entries = {e["id"]: e for e in s.engine.inspect(0)["objects"]}
    assert entries["word.e"]["part_of"] == "word"


def test_text_that_is_not_drawn_as_shapes_draws_as_before():
    s = _scene()
    s.add(Text(id="word", text="Hello", pos=(0, 1)))
    st = s.engine.state_at(0)
    assert [o.id for o in st.objects] == ["word"]
    assert record(st["word"].impl, st).kinds == ("text",)
    sem = st["word"].impl.semantics(st)
    assert sem == {"kind": "label", "text": "Hello"}


def test_glyph_parts_stay_in_the_texts_box_in_2d_and_3d():
    for orbit in (False, True):
        s = _scene()
        if orbit:
            s.use_camera("orbit")
        s.add(Text(id="word", text="Shapes $x^2$", pos=(0.5, 0.2, 0.3), as_shapes=True))
        st = s.engine.state_at(0)
        box = st["word"].impl.world_bounds(st)
        lo, hi = np.asarray(box.lo), np.asarray(box.hi)
        pad = 0.08 * (hi - lo).max()
        for o in st.objects:
            if o.parent == "word":
                b = o.impl.world_bounds(st)
                assert np.all(np.asarray(b.lo) >= lo - pad), o.id
                assert np.all(np.asarray(b.hi) <= hi + pad), o.id


def test_part_names_never_take_the_documents_ids():
    s = _scene()
    s.add(Text(id="word", text="go", pos=(0, 0), as_shapes=True))
    s.add(Dot(id="word.o", pos=(3, 3)))
    st = s.engine.state_at(0)
    assert {o.id for o in st.objects if o.parent == "word"} == {"word.g", "word.o_2"}


def test_write_draws_each_glyphs_outline_then_fills_it_left_to_right():
    s = _scene()
    s.marker("go", 1.0)
    s.add(Text(id="word", text="abc", as_shapes=True))
    s.at("go").write("word")
    d = s.engine.document.timeline.tracks["word.write"][0]
    t0, t1 = 1.0, 1.0 + (float(d.t1.offset) - float(d.t0.offset))

    def glyphs(t):
        st = s.engine.state_at(t)
        return [st[f"word.{n}"].params for n in "abc"]

    assert all(p.progress == 0 for p in glyphs(t0))  # nothing yet
    early = glyphs(t0 + 0.12 * (t1 - t0))
    assert early[0].progress > 0 and early[0].fill_opacity == 0  # the outline first
    assert early[0].progress >= early[2].progress  # left to right
    late = glyphs(t1)
    assert all(p.progress == 1 and p.fill_opacity == 1 and p.stroke_opacity == 0 for p in late)
    mids = [gm_p for t in np.linspace(t0, t1, 25) for gm_p in glyphs(float(t))]
    # once a glyph fills, its outline is fully drawn
    assert all(p.progress == 1 for p in mids if p.fill_opacity > 0)


def test_write_gives_the_same_scene_in_any_script_order():
    def build(order):
        s = _scene()
        s.marker("go", 1.0)
        s.add(Text(id="a", text="one", pos=(0, 1), as_shapes=True))
        s.add(Text(id="b", text="two", pos=(0, -1), as_shapes=True))
        steps = [lambda: s.at("go").write("a"), lambda: s.at(3.0).write("b")]
        for k in order:
            steps[k]()
        return s.engine

    x, y = build([0, 1]), build([1, 0])
    for t in (1.2, 1.6, 3.4, 5.0):
        sx, sy = x.state_at(t), y.state_at(t)
        assert [o.params for o in sx.objects] == [o.params for o in sy.objects]


def test_an_equation_as_shapes_colors_each_parts_glyphs():
    s = _scene()
    s.add(Equation(id="eq", parts=("$x$", "$= 2$"), colors=("highlight", "primary"),
                   as_shapes=True))  # fmt: skip
    st = s.engine.state_at(0)
    parts = {o.id: o for o in st.objects if o.parent == "eq"}
    assert parts["eq.x"].roles["color"] == "highlight"
    assert parts["eq.two"].roles["color"] == "primary"
    assert st["eq"].impl.semantics(st)["glyphs"] == 3


def test_a_style_transition_on_the_text_reaches_its_glyphs():
    s = _scene()
    s.marker("go", 1.0)
    s.add(Text(id="word", text="hi", as_shapes=True))
    s.at("go").to("word.style.color", "highlight", duration=1.0, ease="linear")
    st = s.engine.state_at(1.5)
    assert st["word.h"].style["color"] == pytest.approx(st["word"].style["color"])
    assert s.engine.state_at(3)["word.h"].roles["color"] == "highlight"


@pytest.mark.parametrize("passes", MODES)
def test_a_text_with_too_many_glyphs_is_glyph_limit(passes):
    s = _scene(passes)
    s.add(Text(id="long", text="x" * (gm.MAX_GLYPHS + 50), as_shapes=True,
               style={"size": "annotation"}))  # fmt: skip
    st = s.engine.state_at(0)
    assert len([o for o in st.objects if o.parent == "long"]) == gm.MAX_GLYPHS
    (issue,) = [i for i in s.engine.lint() if i.code == "glyph_limit"]
    assert issue.ids == ("long",) and issue.times and not issue.fix


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.5, 10**400])
def test_bad_opacities_fail_validation_cleanly(bad):
    s = _scene()
    for name in ("fill_opacity", "stroke_opacity", "morph", "write"):
        r = s.engine.apply({"op": "add_object", "object": {"id": f"t_{name}", "type": "text",
                            "params": {"text": "x", name: bad}}}, origin="script")  # fmt: skip
        assert not r.ok


@pytest.mark.parametrize("bad", ["Not An Id", "x" * 300, "1abc", "a..b"])
def test_morph_from_must_be_an_id(bad):
    s = _scene()
    r = s.engine.apply({"op": "add_object", "object": {"id": "t", "type": "text", "params": {
        "text": "x", "morph_from": bad}}}, origin="script")  # fmt: skip
    assert not r.ok


@pytest.mark.parametrize(
    "text",
    ["a", " ", "  x  ", "$\\sum_{i=0}^{n} i^2$", "☃ é", "$\\sqrt{\\frac{1}{2}}$",
     "tab\there", "x" * 900, "$" + "+".join(["x"] * 300) + "$"],
)  # fmt: skip
def test_any_text_as_shapes_never_breaks_state_inspect_or_lint(text):
    s = _scene()
    s.marker("go", 0.5)
    s.add(Text(id="t", text=text, as_shapes=True, style={"size": "annotation"}))
    s.at("go").write("t")
    for t in (0.0, 0.7, 2.0):
        s.engine.state_at(t)
        s.engine.inspect(t)
    s.engine.lint()
