"""Objects becoming other objects: the ``transform`` verb between any two outlines (paths,
shapes, text, equations, SVGs) and ``transform_matching`` (equal pieces move, the rest
fade). The in-between is drawn by the target from the source as the state has it at t:
pure in (document, t), exact at both ends, bounded, the same whatever the script order."""

from __future__ import annotations

import time

import numpy as np
import pytest

from phasor import Dot, Equation, Scene, Text, shapes
from phasor.core.record import record
from phasor.domains import morph
from phasor.domains import morph_math as mm
from phasor.engine import Engine
from phasor.time.verbs import VerbError

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


def _scene(passes=None) -> Scene:
    s = Scene("morph", passes=passes, look="none")
    s.set("camera.height", 8)
    s.marker("go", 1.0)
    return s


def _span(s, path):
    """(start, end) in seconds of the last transition on ``path`` (anchored at ``go`` = 1)."""
    seg = s.engine.document.timeline.tracks[path][-1]
    tm = s.engine._snap.compiled.time_map
    return tm.resolve(seg.t0), tm.resolve(seg.t1)


def _points(e, oid, t):
    st = e.state_at(t)
    return np.asarray(record(st[oid].impl, st).points)


# ---------------------------------------------------------------- transform


def test_text_becomes_a_circle_and_the_ends_are_exact():
    s = _scene()
    s.add(Text(id="word", text="ok", pos=(-2, 0), style={"size": "title"}))
    s.add(shapes.Circle(id="disc", pos=(2, 0), r=1.0, style={"fill": "soft"}))
    s.at("go").transform("word", "disc")
    t0, t1 = _span(s, "disc.morph")
    e = s.engine
    st = e.state_at(t0)
    assert st["word"].params.opacity == 0 and st["disc"].params.opacity == 1
    sem = st["disc"].impl.semantics(st)
    assert sem["morphing_from"] == "word" and sem["morph"] == 0 and sem["morph_kind"] == "whole"
    start = _points(e, "disc", t0)  # where the text's glyphs are
    assert start[:, 0].max() < -0.5
    end = _points(e, "disc", t1 + 0.1)  # the circle itself
    assert np.linalg.norm(end[:, :2] - (2, 0), axis=1).max() == pytest.approx(1.0, abs=0.05)
    mid = e.state_at((t0 + t1) / 2)
    b = mid["disc"].impl.world_bounds(mid)
    assert b.lo[0] < 0.5 and b.hi[0] > -1.5  # between the two places
    assert not [i for i in e.lint() if i.severity == "error"]


def test_a_shape_becomes_text_and_then_its_glyph_parts_appear():
    s = _scene()
    s.add(shapes.Rectangle(id="box", width=3, height=1, style={"fill": "solid"}))
    s.add(Text(id="word", text="hi", as_shapes=True, style={"size": "title"}))
    s.at("go").transform("box", "word")
    t0, t1 = _span(s, "word.morph")
    mid = s.engine.state_at((t0 + t1) / 2)
    assert [o.id for o in mid.objects if o.parent == "word"] == []  # the text draws it
    kinds = record(mid["word"].impl, mid).kinds
    assert kinds and "text" not in kinds
    done = s.engine.state_at(t1 + 0.1)
    assert [o.id for o in done.objects if o.parent == "word"] == ["word.h", "word.i"]


def test_identical_outlines_are_an_identity_at_every_step():
    s = _scene()
    s.add(shapes.Circle(id="a", pos=(0, 0), r=1.0))
    s.add(shapes.Circle(id="b", pos=(0, 0), r=1.0))
    s.at("go").transform("a", "b")
    t0, t1 = _span(s, "b.morph")
    own = _points(s.engine, "b", t1 + 0.5)
    for t in (t0, (t0 + t1) / 2, t1 - 1e-3):
        assert np.allclose(_points(s.engine, "b", t), own)


def test_fills_mix_their_opacity_and_colors_mix_in_linear_light():
    s = _scene()
    s.add(shapes.Circle(id="a", r=1.0, style={"fill": "none", "color": "primary"}))
    s.add(shapes.Rectangle(id="b", width=2, height=2, morph_from="a", morph=0.5,
                           style={"fill": "solid", "fill_color": "highlight",
                                  "color": "highlight"}))  # fmt: skip
    st = s.engine.state_at(0)
    (it,) = morph.frame(st, "b")
    a, b = np.array(st["a"].style["color"]), np.array(st["b"].style["color"])
    assert np.array(it.look.color) == pytest.approx((a + b) / 2)
    assert it.look.fill[3] == pytest.approx(0.5 * st["b"].style["fill"])


@pytest.mark.parametrize(
    "src, dst",
    [
        (lambda: shapes.Annulus(id="src", outer=1.0, inner=0.5),
         lambda: shapes.Circle(id="dst", r=1.0)),  # a hole becomes nothing
        (lambda: shapes.Arc(id="src", r=1.0, angle=200),
         lambda: shapes.Circle(id="dst", r=1.0)),  # open into closed
        (lambda: shapes.Circle(id="src", r=1.0),
         lambda: shapes.Brace(id="dst", start=(-1, 0), end=(1, 0))),  # closed into open
        (lambda: shapes.Path.through("src", [(0, 0), (0, 0)]),
         lambda: shapes.Circle(id="dst", r=1.0)),  # zero length
        (lambda: Equation(id="src", parts=("$\\frac{1}{2}$",)),
         lambda: shapes.Annulus(id="dst")),  # a fraction bar and digits into a ring
    ],
)  # fmt: skip
def test_any_two_outlines_morph_without_failing(src, dst):
    s = _scene()
    s.add(src())
    s.add(dst())
    s.at("go").transform("src", "dst")
    t0, t1 = _span(s, "dst.morph")
    for t in np.linspace(t0, t1, 7):
        st = s.engine.state_at(float(t))
        pts = np.asarray(record(st["dst"].impl, st).points)
        assert np.isfinite(pts).all()
        st["dst"].impl.semantics(st)
    s.engine.lint()


def test_a_huge_outline_morphs_in_bounded_time():
    rng = np.random.default_rng(3)
    s = _scene()
    s.add(shapes.Path.through("scribble", rng.uniform(-2, 2, (60_001, 2))))
    s.add(shapes.Circle(id="disc", r=1.0))
    s.at("go").transform("scribble", "disc")
    t0, t1 = _span(s, "disc.morph")
    start = time.perf_counter()
    for t in np.linspace(t0, t1, 5):
        _points(s.engine, "disc", float(t))
    assert time.perf_counter() - start < 60


def test_a_moving_source_is_followed_purely():
    s = _scene()
    s.add(shapes.Circle(id="a", pos=(-3, 0), r=0.5))
    s.add(shapes.Rectangle(id="b", pos=(3, 0), width=1, height=1))
    s.at(0.0).to("a.pos", (-3, 2), duration=3.0, ease="linear")
    s.at("go").transform("a", "b")
    t0, _ = _span(s, "b.morph")
    p1 = _points(s.engine, "b", t0 + 0.01)
    assert p1[:, 1].mean() > 0.5  # it starts where the circle is at t, not where it began
    assert np.array_equal(p1, _points(s.engine, "b", t0 + 0.01))


def test_transform_gives_the_same_document_in_any_script_order():
    """Verbs on different objects may be written in any order (a chain on one object is
    written in time order, as for every verb: its opacity track)."""

    def build(order):
        s = _scene()
        s.add(Text(id="t", text="ab", pos=(0, 2)))
        s.add(shapes.Circle(id="c", r=1.0))
        s.add(Text(id="u", text="cd", pos=(0, -2)))
        s.add(shapes.Rectangle(id="r", width=2, height=1, pos=(0, -2)))
        steps = [lambda: s.at("go").transform("t", "c"),
                 lambda: s.at(4.0).transform_matching("u", "r"),
                 lambda: s.at(6.0).morph("c.style.color", "highlight")]  # fmt: skip
        for k in order:
            steps[k]()
        return s.engine

    x, y = build([0, 1, 2]), build([2, 1, 0])
    assert x.document == y.document
    for t in (1.2, 1.5, 4.3, 6.5):
        assert np.array_equal(_points(x, "c", t), _points(y, "c", t))
        assert np.array_equal(_points(x, "r", t), _points(y, "r", t))


def test_one_object_can_become_itself_twice_from_different_sources():
    s = _scene()
    s.add(Text(id="a", text="a", pos=(-3, 0)))
    s.add(Text(id="b", text="b", pos=(3, 0)))
    s.add(shapes.Circle(id="c", r=1.0))
    s.at("go").transform("a", "c")
    s.at(5.0).transform("b", "c")
    st1, st2 = s.engine.state_at(1.2), s.engine.state_at(5.2)
    assert st1["c"].params.morph_from == "a" and st2["c"].params.morph_from == "b"
    assert 0 < st2["c"].params.morph < 1


def test_the_verbs_refuse_what_cannot_morph():
    s = _scene()
    s.add(Dot(id="dot"))
    s.add(shapes.Circle(id="c", r=1.0))
    with pytest.raises(VerbError, match="no outline"):
        s.at("go").transform("dot", "c")
    with pytest.raises(VerbError, match="cannot be morphed into"):
        s.at("go").transform_matching("c", "dot")
    with pytest.raises(VerbError, match="two objects"):
        s.at("go").transform("c", "c")


def test_equations_drawn_as_pictures_keep_the_parts_transform():
    """The equation transform of before Milestone 17 writes the same ops."""
    s = _scene()
    s.add(Equation(id="x", parts=("$a$", "$=$", "$b$")))
    s.add(Equation(id="y", parts=("$b$", "$=$", "$a$")))
    s.at("go").transform("x", "y")
    paths = set(s.engine.document.timeline.tracks)
    assert "y.morph" in paths and "y.morph_from" not in paths
    assert tuple(s.engine.document.object("y").params["from_parts"]) == ("$a$", "$=$", "$b$")


# ---------------------------------------------------------------- transform_matching


def _formula(s):
    s.add(Equation(id="before", parts=("$a^2$", "$+$", "$b^2$", "$=$", "$c^2$"),
                   as_shapes=True, pos=(0, 0)))  # fmt: skip
    s.add(Equation(id="after", parts=("$a^2$", "$=$", "$c^2$", "$-$", "$b^2$"),
                   as_shapes=True, pos=(0, 0)))  # fmt: skip
    s.at("go").transform_matching("before", "after")


def test_a_formula_rearranges_by_tex_with_transform_matching():
    s = _scene()
    _formula(s)
    t0, t1 = _span(s, "after.morph")
    st = s.engine.state_at(t0)
    src, dst = morph.shape_of(st, "before"), st["after"].impl.morph_shape(st)
    pairs = morph.pair_groups(src, dst)
    names = [([src.groups[i].name for i in a], [dst.groups[j].name for j in b]) for a, b in pairs]
    # a, 2, =, c, 2, b, 2 each find themselves; "+" and "-" pair with nothing
    assert sorted(n for a, _ in names for n in a) == ["a", "b", "c", "equal", "two", "two", "two"]
    assert all(a == b for a, b in names)
    mid = s.engine.state_at((t0 + t1) / 2)
    items = morph.frame(mid, "after", arc=mid["after"].params.arc)
    ys = [float(np.vstack(it.polys)[:, 1].mean()) for it in items if it.alpha == 1.0]
    assert max(abs(y) for y in ys) > 0.1  # b^2, carried past "=" and c^2, arcs round them
    sem = mid["after"].impl.semantics(mid)
    assert sem["morph_kind"] == "matching" and sem["morphing_from"] == "before"
    done = s.engine.state_at(t1 + 0.1)
    assert [o.id for o in done.objects if o.parent == "after"]  # its glyph parts are back


def test_transform_between_equations_as_shapes_matches_and_morphs_a_keyed_part():
    s = _scene()
    s.add(Equation(id="x", parts=("$a$", "$+b$"), keys=("", "k"), as_shapes=True))
    s.add(Equation(id="y", parts=("$-b$", "$a$"), keys=("k", ""), as_shapes=True))
    s.at("go").transform("x", "y")
    st = s.engine.state_at(1.0)
    assert st["y"].params.morph_kind == "matching" and st["y"].params.morph_from == "x"
    src, dst = morph.shape_of(st, "x"), st["y"].impl.morph_shape(st)
    keyed = [p for p in morph.pair_groups(src, dst) if len(p[0]) > 1 or len(p[1]) > 1]
    assert len(keyed) == 1  # "+b" becomes "-b": its outline morphs (no crossfade)


def test_text_glyphs_match_by_shape():
    s = _scene()
    s.add(Text(id="x", text="ab", pos=(0, 0)))
    s.add(Text(id="y", text="ba", pos=(0, 0)))
    s.at("go").transform_matching("x", "y")
    st = s.engine.state_at(1.0)
    src, dst = morph.shape_of(st, "x"), st["y"].impl.morph_shape(st)
    pairs = [(src.groups[a[0]].name, dst.groups[b[0]].name) for a, b in morph.pair_groups(src, dst)]
    assert pairs == [("b", "b"), ("a", "a")]


def test_matching_gives_the_same_pairs_in_any_order_of_equal_glyphs():
    s = _scene()
    s.add(Text(id="x", text="aab", pos=(0, 0)))
    s.add(Text(id="y", text="baa", pos=(0, 0)))
    s.at("go").transform_matching("x", "y")
    st = s.engine.state_at(1.0)
    src, dst = morph.shape_of(st, "x"), st["y"].impl.morph_shape(st)
    one = morph.pair_groups(src, dst)
    assert one == morph.pair_groups(src, dst)
    assert [(a[0], b[0]) for a, b in one] == [(2, 0), (0, 1), (1, 2)]  # reading order


# ---------------------------------------------------------------- lint, files, bounds


@pytest.mark.parametrize("passes", MODES)
def test_a_morph_whose_source_is_gone_is_lint_morph_missing(passes):
    s = _scene(passes)
    s.add(Text(id="word", text="ok", pos=(-2, 0)))
    s.add(shapes.Circle(id="disc", pos=(2, 0), r=1.0))
    s.at("go").transform("word", "disc")
    s.engine.apply({"op": "remove_object", "id": "word"}, origin="script")
    (issue,) = [i for i in s.engine.lint() if i.code == "morph_missing"]
    assert issue.ids == ("disc",) and issue.times and not issue.fix
    t0, _ = _span(s, "disc.morph")
    assert len(_points(s.engine, "disc", t0 + 0.1))  # it draws itself


@pytest.mark.parametrize("passes", MODES)
def test_a_morph_past_the_subpath_limit_is_lint_morph_limit(passes, monkeypatch):
    monkeypatch.setattr(mm, "MAX_SUBPATHS", 3)
    s = _scene(passes)
    s.add(Text(id="word", text="abcdef", pos=(0, 0)))
    s.add(shapes.Circle(id="disc", r=1.0))
    s.at("go").transform("word", "disc")
    (issue,) = [i for i in s.engine.lint() if i.code == "morph_limit"]
    assert issue.ids == ("disc",) and issue.times


def test_a_morph_reads_the_same_after_save_and_load(tmp_path):
    s = _scene()
    s.add(Text(id="word", text="ok", pos=(-2, 0)))
    s.add(shapes.Circle(id="disc", pos=(2, 0), r=1.0))
    s.at("go").transform("word", "disc")
    s.engine.save(tmp_path / "m.phasor.json")
    e = Engine.open(tmp_path / "m.phasor.json", passes=None)
    assert np.array_equal(_points(e, "disc", 1.3), _points(s.engine, "disc", 1.3))


def test_an_svg_source_morphs_into_text(tmp_path):
    pytest.importorskip("svgelements")
    (tmp_path / "d.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<circle id="ring" cx="5" cy="5" r="4"/><rect id="bar" x="1" y="4" width="8" '
        'height="2"/></svg>', encoding="utf-8")  # fmt: skip
    from phasor.domains.svg import SVG

    s = Scene("svg", passes=None, look="none", base_dir=tmp_path)
    s.marker("go", 1.0)
    s.add(SVG(id="logo", path="d.svg", pos=(-2, 0), height=2))
    s.add(Text(id="word", text="ok", pos=(2, 0), style={"size": "title"}))
    s.at("go").transform("logo", "word")
    st = s.engine.state_at(1.0)
    src = morph.shape_of(st, "logo")
    assert src is not None and len(src.groups) == 2
    assert len(_points(s.engine, "word", 1.3))


def test_a_morph_ending_on_a_frame_never_leaves_its_bounds():
    """``smoother`` just below u = 1 rounds above 1; non-overshooting easings are clamped,
    so a bounded parameter (``morph``, ``progress``) is never refused at a frame."""
    from phasor.time.easing import EASINGS, OVERSHOOT

    for name, f in EASINGS.items():
        if name not in OVERSHOOT:
            assert all(0.0 <= f(u) <= 1.0 for u in np.linspace(0, 1, 10001))
            assert f(1 - 1e-16) <= 1.0 and f(1.0) == 1.0
    s = _scene()
    s.add(Text(id="word", text="ok"))
    s.add(shapes.Circle(id="disc", r=1.0))
    s.marker("morph", 2.6)
    s.at("morph").transform("word", "disc")
    for t in np.linspace(3.85, 3.95, 41):
        s.engine.state_at(float(t))


# ---------------------------------------------------------------- the example


def _example():
    import runpy
    from pathlib import Path

    root = Path(__file__).parents[1]
    return runpy.run_path(str(root / "examples" / "text_shapes.py"), run_name="x")["s"].engine


def test_the_example_shows_text_as_shapes_both_transforms_and_a_color_and_lints_clean():
    e = _example()
    verbs = {seg.verb for segs in e.document.timeline.tracks.values() for seg in segs}
    assert {"write", "transform", "transform_matching", "morph"} <= verbs
    assert "disc.style.fill_color" in e.document.timeline.tracks
    st = e.state_at(1.0)
    assert any(o.parent == "title" and o.type == "path" for o in st.objects)
    assert e.lint(strict=True) == []


@pytest.mark.gl
def test_example_golden_storyboard():
    from tests.conftest import gl_available
    from tests.test_beauty import _golden

    if not gl_available():
        pytest.skip("needs OpenGL")
    e = _example()
    sheet = e.storyboard([1.0, 3.0, 3.4, 5.4, 6.4, 8.8], size=(320, 180))["image"]
    _golden("text_shapes", sheet, e.pass_runner().renderer_class())
