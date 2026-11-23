"""Text as shapes: glyph outlines (``domains/glyph_math.py``), ``as_shapes`` on ``Text`` and
``Equation`` (one ``Path`` part per glyph), ``write`` drawing the outline then the fill, the
bounds on glyph counts and lint ``glyph_limit``. A text that is not drawn as shapes draws as
it always did."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from phasor import Dot, Equation, Scene, Text
from phasor.core import text as text_metrics
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.record import record
from phasor.core.values import Vec3
from phasor.domains import glyph_math as gm
from phasor.domains import path_math as pm
from phasor.domains import shapes

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


@pytest.mark.parametrize("passes", MODES)
def test_small_glyphs_of_a_label_are_not_tiny_and_the_label_lints_like_a_picture(passes):
    """A comma's outline is small, but it is a glyph of a readable label: the label is
    judged (tiny, contrast), not each glyph."""
    plain, shaped = _scene(passes), _scene(passes)
    plain.add(Text(id="note", text="a, b.", pos=(0, 0)))
    shaped.add(Text(id="note", text="a, b.", pos=(0, 0), as_shapes=True))
    assert shaped.engine.lint() == plain.engine.lint() == []


@pytest.mark.parametrize("passes", MODES)
def test_a_highlighted_text_as_shapes_is_one_focal_point_not_one_per_glyph(passes):
    s = _scene(passes)
    s.add(Dot(id="sun", pos=(2, 0), r=0.5, style={"color": "highlight"}))
    s.add(Text(id="word", text="sun", pos=(-2, 0), as_shapes=True, style={"color": "highlight"}))
    codes = [i for i in s.engine.lint(aesthetic=True) if i.code == "focal_point"]
    assert codes == []


def test_outlines_do_not_depend_on_the_math_engine(monkeypatch):
    """The outline cache is keyed by text: what it holds must not change with the engine."""
    gm.outline.cache_clear()
    one = gm.outline("$x_1^2$")
    gm.outline.cache_clear()
    monkeypatch.setattr(text_metrics, "math_box", lambda t: (9.0, 9.0, 9.0))  # any engine's box
    two = gm.outline("$x_1^2$")
    gm.outline.cache_clear()
    for a, b in zip(one.glyphs, two.glyphs, strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a.subpaths, b.subpaths, strict=True))


def test_a_huge_text_as_shapes_is_bounded_in_time():
    import time

    start = time.perf_counter()
    o = gm.outline("w" * 200_000)
    assert len(o.glyphs) == gm.MAX_GLYPHS and o.total == 200_000
    s = _scene()
    s.add(Equation(id="eq", parts=tuple(f"${k}$" for k in range(600)), as_shapes=True))
    st = s.engine.state_at(0)
    assert len([o for o in st.objects if o.parent == "eq"]) <= gm.MAX_GLYPHS
    assert st["eq"].impl.semantics(st)["glyph_limit"] >= 600
    assert time.perf_counter() - start < 120


@pytest.mark.parametrize("morph", [0.0, 1e-300, 0.5, 1.0])
@pytest.mark.parametrize("kind", ["whole", "matching"])
def test_morph_params_at_their_bounds_never_break_state_inspect_or_lint(morph, kind):
    from phasor.core.objects import ObjectBase  # noqa: F401  (registration)
    from phasor.domains.path import MAX_ID

    s = _scene()
    long_id = "s" * MAX_ID
    s.add(Text(id=long_id, text="ab", pos=(-2, 0)))
    s.add(Equation(id="eq", parts=("$a$", "$b$"), as_shapes=True, morph_from=long_id,
                   morph=morph, morph_kind=kind, arc=10.0))  # fmt: skip
    s.add(Text(id="t", text="x", morph_from="nowhere", morph=morph, morph_kind=kind))
    for t in (0.0, 1.0):
        s.engine.state_at(t)
        s.engine.inspect(t)
    s.engine.lint()
    r = s.engine.apply({"op": "set_param", "path": "eq.arc", "value": 1e300}, origin="script")
    assert not r.ok


# ---------------------------------------------------------------- independent audit


@register
class AuditBadge(ObjectBase):
    """A composite whose part has an explicit color and stroke (scene-local type)."""

    type: ClassVar[str] = "audit17_badge"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def build(self, state):
        return [shapes.Circle("ring", r=1.0, style={"color": "#00FF00", "stroke": 0.9})]


@pytest.mark.parametrize("passes", MODES)
def test_a_composite_part_with_explicit_values_is_judged_but_a_glyph_is_not(passes):
    """Audit 17 #3: ``off_palette``/``off_scale`` skipped every composite part, so a part
    a ``build`` gives an explicit off-palette color and an off-scale stroke was silent.
    Only glyph parts (whose values come from their text, judged there) are skipped."""
    s = _scene(passes)
    s.marker("end", 2.0)
    s.add(AuditBadge(id="b"))
    s.add(Text(id="word", text="green", pos=(0, -2), as_shapes=True,
               style={"color": "#00FF00"}))  # fmt: skip
    found = {(i.code, i.ids) for i in s.engine.lint(aesthetic=True)
             if i.code in ("off_palette", "off_scale")}  # fmt: skip
    assert ("off_palette", ("b.ring",)) in found and ("off_scale", ("b.ring",)) in found
    assert ("off_palette", ("word",)) in found  # the text, once
    assert not [ids for _, ids in found if ids and ids[0].startswith("word.")]


@pytest.fixture
def tex_accepts_anything(monkeypatch):
    """The ``latex`` engine as far as pictures go (no TeX here): every math string
    typesets, so ``\\begin{pmatrix}`` passes ``check_math``; outlines still come from
    mathtext, which cannot read it. The caches it touches are cleared both ways."""

    def caches():
        text_metrics.math_box.cache_clear()
        text_metrics.aspect.cache_clear()
        text_metrics.rasterize.cache_clear()
        gm.outline.cache_clear()

    caches()
    monkeypatch.setattr(text_metrics, "_math", lambda text, em_px, engine=None: (
        np.full((em_px, 3 * em_px), 255, np.uint8), em_px * 0.2))  # fmt: skip
    yield r"$\begin{pmatrix}1\\2\end{pmatrix}$"
    monkeypatch.undo()
    caches()


def test_a_text_mathtext_cannot_outline_is_morph_missing_and_never_raises(tex_accepts_anything):
    """Audit 17 #5: with the latex engine a text mathtext cannot read passed the verb and
    then ``inspect``/``lint`` raised ValueError from the outline. It has no outline: a
    morph from it is ``morph_missing`` (the target draws itself); as shapes it is refused
    when added, saying why."""
    from phasor.core.issues import DocumentError

    tex = tex_accepts_anything
    o = gm.outline(tex)
    assert o.glyphs == () and "mathtext" in o.error
    assert gm.outline("   ").error == ""  # an empty outline is not a failure
    s = _scene()
    s.marker("go", 1.0)
    s.add(Text(id="w", text=tex))
    s.add(shapes.Circle(id="c", r=1))
    s.at("go").transform("w", "c")
    s.engine.state_at(1.3)
    s.engine.inspect(1.3, detail="full")
    assert "morph_missing" in {i.code for i in s.engine.lint()}
    assert len(_points_of(s.engine, "c", 1.3)) > 0  # it draws itself
    # the other way round: a circle into the text, which draws its picture
    s2 = _scene()
    s2.marker("go", 1.0)
    s2.add(shapes.Circle(id="c", r=1))
    s2.add(Text(id="w", text=tex))
    s2.at("go").transform("c", "w")
    s2.engine.inspect(1.3, detail="full")
    s2.engine.lint()
    st = s2.engine.state_at(1.3)
    assert "text" in record(st["w"].impl, st).kinds
    with pytest.raises(DocumentError, match="cannot be drawn as shapes"):
        _scene().add(Text(id="w", text=tex, as_shapes=True))


def _points_of(e, oid, t):
    st = e.state_at(t)
    return np.asarray(record(st[oid].impl, st).points)


def test_outlines_read_matplotlib_before_and_after_3_10(monkeypatch):
    """Audit 17 #4: ``LoadFlags``/``Kerning`` and the 6-entry mathtext glyphs are new in
    matplotlib 3.10; 3.8 and 3.9 (allowed by the pyproject floor) have module constants
    and 5 entries (font, size, char, x, y). Both give the same outlines (also run for real
    on 3.8.4 and 3.9.4)."""
    import warnings

    from matplotlib import ft2font
    from matplotlib.mathtext import MathTextParser

    texts = ("AVa, Wo", "$\\frac{a}{b} + \\sqrt{x}$")
    gm.outline.cache_clear()
    gm._ft_flags.cache_clear()
    new = [gm.outline(t) for t in texts]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_flags = (ft2font.LOAD_NO_HINTING, ft2font.KERNING_DEFAULT)
    monkeypatch.delattr(ft2font, "LoadFlags", raising=False)
    monkeypatch.delattr(ft2font, "Kerning", raising=False)
    parse = MathTextParser.parse

    def five(self, *a, **k):  # the 3.9 layout: no glyph index
        got = parse(self, *a, **k)
        return got._replace(glyphs=[(f, s, c, x, y) for f, s, c, _, x, y in got.glyphs])

    monkeypatch.setattr(MathTextParser, "parse", five)
    gm.outline.cache_clear()
    gm._ft_flags.cache_clear()
    try:
        assert gm._ft_flags() == old_flags
        old = [gm.outline(t) for t in texts]
    finally:
        gm.outline.cache_clear()
        gm._ft_flags.cache_clear()
    for a, b in zip(new, old, strict=True):
        assert a.total == b.total and len(a.glyphs) == len(b.glyphs)
        for ga, gb in zip(a.glyphs, b.glyphs, strict=True):
            assert ga.key == gb.key and ga.name == gb.name
            assert all(np.array_equal(x, y) for x, y in zip(ga.subpaths, gb.subpaths, strict=True))
