"""Milestone 20: what landed in Milestones 15-19 carries over from Manim scripts (shapes and
fills, SVG, glyph morphs, color animations, the diagram kit, ValueTracker, updaters), plus
``MovingCameraScene`` and ``add_sound``. Each was a warning before; now it is phasor's own
feature. Hostile scripts are a clean ``ManimImportError``; nothing that imported before
fails now."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from phasor.importers import import_manim
from phasor.importers.manim import ManimImportError
from tests.test_manim_import import script, state


def doc_of(r):
    return r.scene.engine.document


def kinds(r) -> dict[str, str]:
    return {o.id: o.type for o in doc_of(r).objects}


# ---------------------------------------------------------------- shapes, fills, dashes


def test_shapes_become_phasors_shapes_with_their_fills(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        a = Circle(radius=0.5, color=BLUE, fill_opacity=0.4)
        b = Ellipse(width=2, height=1).rotate(PI / 6)
        c = Arc(radius=1, start_angle=PI / 4, angle=PI / 2)
        d = Sector(radius=1, angle=PI / 3, color=ORANGE)
        e = Annulus(inner_radius=0.3, outer_radius=0.6)
        f = Polygon(ORIGIN, RIGHT, UP, color=GREEN)
        g = RegularPolygon(n=6, radius=0.8)
        h = Rectangle(width=2, height=1).set_fill(RED, opacity=1).set_stroke(width=0)
        i = RoundedRectangle(corner_radius=0.2, width=2, height=1)
        j = DashedLine(LEFT, RIGHT)
        k = Brace(h, DOWN)
        label = k.get_text("width")
        self.add(a, b, c, d, e, f, g, h, i, j, k, label)
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    assert r.warnings == []
    assert kinds(r) == {
        "circle_1": "circle", "ellipse_1": "ellipse", "arc_1": "arc", "sector_1": "sector",
        "annulus_1": "annulus", "polygon_1": "polygon_path", "regularpolygon_1": "regular_polygon",
        "rectangle_1": "rectangle", "roundedrectangle_1": "rectangle", "dashedline_1": "path",
        "brace_1": "brace", "text_1": "text"}  # fmt: skip
    doc = doc_of(r)
    a = doc.object("circle_1")
    assert a.params["fill_opacity"] == 0.4 and a.style["fill_color"] == "#58C4DD"
    assert state(r, 0.5, "ellipse_1").rotation == pytest.approx(30.0)
    assert state(r, 0.5, "arc_1").start == pytest.approx(45.0)
    assert state(r, 0.5, "regularpolygon_1").rotation == pytest.approx(-90.0)  # Manim: vertex right
    h = doc.object("rectangle_1")
    assert h.style["fill_color"] == "#FC6255" and h.params["stroke_opacity"] == 0.0
    assert state(r, 0.5, "roundedrectangle_1").corner_radius == pytest.approx(0.2)
    assert doc.object("dashedline_1").style["dash"] == "dashed"
    tip = state(r, 0.5, "brace_1")
    assert tip.start[1] == pytest.approx(-0.7) and tip.start[0] > tip.end[0]  # under the box
    assert state(r, 0.5, "text_1").pos[1] < -0.7 - 0.25  # below the brace's tip
    assert r.scene.engine.lint() == [] or all(i.severity != "error" for i in r.scene.engine.lint())


def test_a_shape_stretched_out_of_its_kind_is_a_path_and_morphs_there(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        c = Circle(radius=1)
        s = Square().stretch(2, 0)
        self.add(c, s)
        self.play(c.animate.stretch(0.5, 1), run_time=2)
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    k = kinds(r)
    assert k["square_1"] == "polygon_path"  # a stretched square keeps its corners
    assert k["circle_1"] == "circle" and "circle_2" in k and k["circle_2"] == "path"
    assert state(r, 1.0, "circle_2").morph_from == "circle_1"
    assert state(r, 1.0, "circle_2").morph == pytest.approx(0.5, abs=0.05)
    assert state(r, 2.5, "circle_1").opacity == 0.0 and state(r, 2.5, "circle_2").opacity == 1.0


DEGENERATE = {
    "zero_circle": "c = Circle(radius=0)",
    "zero_arc": "c = Arc(angle=0)",
    "two_point_polygon": "c = Polygon(ORIGIN, RIGHT)",
    "zero_square": "c = Square(side_length=0)",
    "two_gon": "c = RegularPolygon(n=2)",
    "far_dot": "c = Dot(RIGHT * 1e12)",
    "far_line": "c = Line(ORIGIN, RIGHT * 1e12)",
    "pole": "ax = Axes(); c = VGroup(ax, ax.plot(lambda x: 1 / x))",
    "negative_rectangle": "c = Rectangle(width=-2, height=1)",
    "negative_circle": "c = Circle(radius=-1)",
    "empty_text": "c = Text('')",
    "empty_group": "c = VGroup()",
    "inner_past_outer": "c = Annulus(inner_radius=2, outer_radius=1)",
    "full_sector": "c = Sector(angle=3 * TAU)",
}


@pytest.mark.parametrize("case", sorted(DEGENERATE))
def test_degenerate_shapes_still_import(tmp_path, case):
    """Lesson 1 of the audits: what imported before Milestone 20 imports now (a shape
    phasor's own kind cannot hold is a path or a line, with a warning)."""
    p = script(tmp_path, f"""
class S(Scene):
    def construct(self):
        {DEGENERATE[case]}
        self.add(c)
        self.play(Create(c))
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    assert r.duration == 2.0
    assert not [i for i in r.scene.engine.lint() if i.severity == "error"]
    assert "manim" not in sys.modules


# ---------------------------------------------------------------- files: SVG, sound, code


SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10">'
       '<rect id="box" x="0" y="0" width="20" height="10"/></svg>')  # fmt: skip


def test_svg_and_sound_and_code_read_files_inside_the_scene_folder_only(tmp_path):
    from tests.test_sound import wav

    (tmp_path / "logo.svg").write_text(SVG, encoding="utf-8")
    wav(tmp_path / "click.wav", np.full(800, 1000))
    (tmp_path / "f.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.svg"
    outside.write_text(SVG, encoding="utf-8")
    p = script(tmp_path, f"""
class S(Scene):
    def construct(self):
        logo = SVGMobject("logo.svg", height=1).to_edge(UP)
        bad = SVGMobject({str(outside)!r})
        code = Code(code_file="f.py", language="python").to_edge(DOWN)
        self.add(logo, bad, code)
        self.add_sound("click.wav", time_offset=0.5, gain=-3)
        self.add_sound("../nope.wav")
        self.wait(2)
""")  # fmt: skip
    r = import_manim(p)
    k = kinds(r)
    assert k == {"svgmobject_1": "svg", "code_1": "code", "sound_1": "sound"}
    assert state(r, 1.0, "svgmobject_1").path == "logo.svg"
    assert state(r, 1.0, "svgmobject_1").pos[1] == pytest.approx(4.0 - 0.5 - 0.5)  # 2:1 box
    assert state(r, 1.0, "code_1").path == "f.py"
    s = state(r, 0.0, "sound_1")
    assert (s.path, s.start, s.gain) == ("click.wav", 0.5, -3.0)
    assert sum("not inside the scene's folder" in w for w in r.warnings) == 2
    track, rate = r.scene.engine.audio_track()
    assert track[: int(0.5 * rate) - 1].max() == 0 and track[int(0.55 * rate)] != 0
    assert "svg_parts" not in r.warnings


# ---------------------------------------------------------------- text, transforms, color


def test_transforms_between_kinds_morph_in_manims_run_time(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        a = Square()
        b = Circle()
        t1 = Text("one")
        t2 = Text("two").shift(DOWN)
        e1 = MathTex("a", "+", "b", "=", "c").shift(UP * 2)
        e2 = MathTex("a", "=", "c", "-", "b").shift(UP * 2)
        self.add(a, t1, e1)
        self.play(Transform(a, b), ReplacementTransform(t1, t2), run_time=2)
        self.play(TransformMatchingTex(e1, e2), run_time=3)
        self.play(a.animate.shift(RIGHT))  # a now stands for the circle
""")  # fmt: skip
    r = import_manim(p)
    assert state(r, 1.0, "circle_1").morph_from == "square_1"
    assert state(r, 1.0, "circle_1").morph == pytest.approx(0.5, abs=0.02)  # sigmoid half way
    assert state(r, 2.0, "circle_1").morph == 1.0
    assert doc_of(r).object("text_1").params["as_shapes"] is True  # text morphs as glyphs
    assert state(r, 1.0, "text_2").morph_from == "text_1"
    eq = state(r, 3.5, "mathtex_2")
    assert eq.morph_kind == "matching" and eq.morph_from == "mathtex_1"
    assert eq.morph == pytest.approx(0.5, abs=0.02)  # linear: Manim's TransformMatchingTex
    assert state(r, 6.0, "circle_1").pos == pytest.approx((1.0, 0.0, 0.0))
    assert r.trace[1]["anims"][0]["rate_func"] == "linear"
    assert not [i for i in r.scene.engine.lint() if i.severity == "error"]


def test_color_and_fill_animate_as_style_tracks(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        a = Square(color=BLUE)
        self.add(a)
        self.play(a.animate.set_color(RED), run_time=2)
        self.play(a.animate.set_fill(GREEN, opacity=0.5))
        self.play(Indicate(a))
""")  # fmt: skip
    r = import_manim(p)
    tracks = doc_of(r).timeline.tracks
    assert "rectangle_1.style.color" in tracks or "square_1.style.color" in tracks
    st = r.scene.engine.state_at
    mid, end = st(1.0)["square_1"].style["color"], st(2.0)["square_1"].style["color"]
    assert mid != end and np.allclose(end[:3], st(2.5)["square_1"].style["color"][:3])
    assert st(3.0)["square_1"].params.fill_opacity == pytest.approx(0.5)
    assert st(2.5)["square_1"].params.fill_opacity == pytest.approx(0.25, abs=0.03)
    # Indicate goes there and back: bigger in the middle, the same size at the end
    assert st(3.5)["square_1"].params.width > 2.2 and st(4.0)["square_1"].params.width == 2.0
    assert r.warnings == []


def test_write_times_like_manim(tmp_path):
    p = script(tmp_path, r"""
class S(Scene):
    def construct(self):
        self.play(Write(Text("short")))
        self.play(Write(Text("a rather longer sentence")))
        self.play(Write(MathTex(r"\frac{a}{b} = c")))
        self.play(DrawBorderThenFill(Square(fill_opacity=1)))
""")  # fmt: skip
    r = import_manim(p)
    assert [p["duration"] for p in r.trace] == [1.0, 2.0, 1.0, 2.0]
    assert [p["anims"][0]["rate_func"] for p in r.trace] == ["linear", "linear", "linear",
                                                            "double_smooth"]  # fmt: skip
    assert state(r, 0.5, "text_1").write == pytest.approx(0.5)


# ---------------------------------------------------------------- the diagram kit


def test_diagrams_become_the_diagram_kit(tmp_path):
    p = script(tmp_path, r"""
class S(Scene):
    def construct(self):
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 2, 1], x_length=4, y_length=2,
                  axis_config={"include_numbers": True}).to_corner(UL)
        labels = ax.get_axis_labels(x_label="t", y_label="v")
        plane = NumberPlane(x_range=[-2, 2, 1], y_range=[-1, 1, 1], x_length=4, y_length=2)
        line = NumberLine(x_range=[0, 5, 1], length=5, include_numbers=True).to_edge(DOWN)
        chart = BarChart([3, 5, 2], bar_names=["a", "b", "c"], x_length=3, y_length=2)
        table = Table([["1", "2"], ["3", "4"]], col_labels=[Text("x"), Text("y")])
        matrix = Matrix([[1, 0], [0, 1]], left_bracket="(")
        number = DecimalNumber(3.14159, num_decimal_places=2)
        self.add(ax, labels, plane, line, chart, table, matrix, number)
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    k = kinds(r)
    assert sorted(k.values()) == sorted(["axes", "number_plane", "number_line", "bar_chart",
                                         "table", "matrix_display", "counter"])  # fmt: skip
    ax = doc_of(r).object("axes_1").params
    assert ax["labels"] == ("$t$", "$v$") and ax["numbers"] is True
    assert state(r, 0.5, "numberline_1").scale == pytest.approx(1.0)
    assert state(r, 0.5, "numberline_1").pos[1] < -3
    assert state(r, 0.5, "table_1").cells[0] == ("x", "y")
    assert state(r, 0.5, "matrix_1").bracket == "round"
    assert state(r, 0.5, "decimalnumber_1").value == pytest.approx(3.14159)
    assert r.warnings == []
    assert not [i for i in r.scene.engine.lint() if i.severity == "error"]


# ---------------------------------------------------------------- trackers and updaters


def test_value_tracker_updaters_and_always_redraw_follow_manims_frames(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        t = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.move_to(RIGHT * t.get_value()))
        tag = Dot(radius=0.05).add_updater(lambda m: m.next_to(dot, UP, buff=0.2))
        arm = always_redraw(lambda: Line(ORIGIN, dot.get_center()))
        spin = Square().add_updater(lambda m, dt: m.rotate(dt))
        n = DecimalNumber(0).add_updater(lambda m: m.set_value(t.get_value()))
        self.add(dot, tag, arm, spin, n)
        self.play(t.animate.set_value(3), run_time=3, rate_func=linear)
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    assert r.warnings == []
    assert kinds(r)["valuetracker_1"] == "value"
    for when in (0.5, 1.5, 2.9):
        x = state(r, when, "dot_1").pos[0]
        assert x == pytest.approx(when, abs=0.04)  # linear, keyed at Manim's frames
        assert state(r, when, "dot_2").pos == pytest.approx((x, 0.08 + 0.2 + 0.05, 0), abs=0.04)
        assert np.asarray(state(r, when, "line_1").subpaths[0])[-1][0] == pytest.approx(x, abs=0.04)
        assert state(r, when, "decimalnumber_1").value == pytest.approx(when, abs=0.05)
    # a dt updater: Manim's frames at 30 a second, the first of each play with dt 0
    turned = math.radians(state(r, 4.0, "square_1").rotation)
    assert turned == pytest.approx((3 * 30 - 1) / 30 + (30 - 1) / 30, abs=1e-6)
    # constant motion keeps two keys, not a key per frame
    segs = doc_of(r).timeline.tracks["dot_1.pos"]
    assert len(segs) < 10


def test_hostile_updaters_and_rate_functions_are_clean_errors(tmp_path):
    for body in (
        "d = Dot().add_updater(lambda m: 1 / 0)\n        self.add(d)\n        self.wait(1)",
        "d = Dot()\n        self.add(d)\n        self.play(d.animate.shift(RIGHT), "
        "rate_func=lambda t: float('nan'))",
        "d = Dot().shift(RIGHT * float('inf'))",
        "d = Dot().add_updater(lambda m: m.shift(RIGHT * float('nan')))\n        self.add(d)\n"
        "        self.wait(1)",
    ):
        p = script(tmp_path, f"""
class S(Scene):
    def construct(self):
        {body}
""")  # fmt: skip
        with pytest.raises(ManimImportError):
            import_manim(p)
        assert "manim" not in sys.modules


def test_updater_frames_are_bounded(tmp_path, monkeypatch):
    from phasor.importers import manim as mi

    monkeypatch.setattr(mi, "MAX_FRAMES", 100)
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        d = Dot().add_updater(lambda m, dt: m.shift(RIGHT * dt * 0.01))
        self.add(d)
        self.wait(20)
""")  # fmt: skip
    r = import_manim(p)
    assert any("frames a second" in w for w in r.warnings)
    assert state(r, 20.0, "dot_1").pos[0] == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------- the camera


def test_moving_camera_frame_animations_become_camera_tracks(tmp_path):
    p = script(tmp_path, """
class S(MovingCameraScene):
    def construct(self):
        sq = Square(side_length=1).shift(RIGHT * 3)
        self.add(sq)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(sq).set(width=sq.width * 4), run_time=2)
        self.play(self.camera.frame.animate.shift(LEFT).scale(2))
        self.play(Restore(self.camera.frame))
""")  # fmt: skip
    r = import_manim(p)
    v = lambda t: r.scene.engine.state_at(t).values  # noqa: E731
    assert v(2.0)["camera.center"] == pytest.approx((3.0, 0.0, 0.0))
    assert v(2.0)["camera.height"] == pytest.approx(4 * 9 / 16)  # width 4, the frame's shape
    assert v(1.0)["camera.center"][0] == pytest.approx(1.5, abs=0.05)
    assert v(3.0)["camera.height"] == pytest.approx(4.5) and v(3.0)["camera.center"][0] == 2.0
    assert v(4.0)["camera.center"] == pytest.approx((0, 0, 0)) and v(4.0)["camera.height"] == 8.0
    plain = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.camera.frame.animate.shift(LEFT)
""", "plain.py")  # fmt: skip
    with pytest.raises(ManimImportError, match="MovingCameraScene"):
        import_manim(plain)


# ---------------------------------------------------------------- ids and permissions


def test_ids_stay_valid_and_distinct_whatever_the_class_names(tmp_path):
    p = script(tmp_path, """
class _3D(Circle):
    pass

class Ünïcode(Square):
    pass

class circle_1(Dot):
    pass

class S(Scene):
    def construct(self):
        self.add(_3D(), Ünïcode(), circle_1(), Circle(), circle_1())
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    ids = [o.id for o in doc_of(r).objects]
    assert len(set(ids)) == 5
    from phasor.core.ids import is_valid_id

    assert all(is_valid_id(i) for i in ids), ids


def test_the_import_never_runs_scripts_itself(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.add(Dot().add_updater(lambda m: m.shift(RIGHT * 0.01)))
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    e = r.scene.engine
    assert e.allow_scripts is False and doc_of(r).meta.modules == ()
    assert all(not o.always for o in doc_of(r).objects)  # tracks, not updaters
    assert "manim" not in sys.modules


@pytest.mark.parametrize("body", [
    "x = 1 / 0",  # at the top of the script
    "class S(Scene):\n    def construct(self):\n        raise ValueError('mine')",
    "class S(Scene):\n    def construct(self):\n        self.play(Wait(run_time=-1))",
    "class S(Scene):\n    def construct(self)\n        pass",  # not Python
    "class S(Scene):\n    def construct(self):\n        self.construct()",  # recursion
])  # fmt: skip
def test_a_script_that_fails_is_a_clean_import_error(tmp_path, body):
    with pytest.raises(ManimImportError) as info:
        import_manim(script(tmp_path, body))
    assert "s.py" in str(info.value) and "Traceback" not in str(info.value)
    assert "manim" not in sys.modules


def test_the_command_brings_named_files_to_where_it_saves(tmp_path, capsys):
    from phasor.cli import main
    from phasor.core import io
    from tests.test_sound import wav

    wav(tmp_path / "click.wav", np.full(800, 1000))
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.add(Dot())
        self.add_sound("click.wav")
        self.wait(1)
""")  # fmt: skip
    out = tmp_path / "elsewhere" / "s.phasor.json"
    assert main(["manim", str(p), "-o", str(out)]) == 0
    assert (out.parent / "click.wav").exists() and "lint error" not in capsys.readouterr().out
    assert io.load_document(out).object("sound_1").params["path"] == "click.wav"


# ---------------------------------------------------------------- the examples (exit demo)


def test_the_camera_tour_imports_with_every_feature_and_no_warning():
    """Milestone 20's exit demo: a Manim script using the camera frame, a filled shape, a
    ValueTracker with an updater, always_redraw, add_sound, a morph, a color animation and
    a number line runs unchanged, with nothing left behind."""
    from tests.test_manim_import import ROOT

    r = import_manim(ROOT / "examples" / "manim_camera_tour.py")
    assert r.warnings == [] and r.duration == pytest.approx(12.0)
    k = kinds(r)
    assert {"value", "sound", "number_line", "circle", "regular_polygon", "path"} <= set(
        k.values())  # fmt: skip
    v = r.scene.engine.state_at
    assert v(4.5).values["camera.height"] == pytest.approx(3.0 * 2.5 * 9 / 16)
    assert v(8.5).values["camera.height"] == pytest.approx(8.0)  # restored
    ride = [v(t)["dot_1"].params.pos for t in (4.5, 5.25, 6.0, 7.5)]
    assert ride[0] == pytest.approx((-1.5, 0, 0)) and ride[2] == pytest.approx(
        (-4.5, 0, 0), abs=1e-6
    )
    # between Manim's frames (1/30 s) the dot moves on a chord: within 1e-3 of the circle
    assert ride[1] == pytest.approx((-3.0, 1.5, 0), abs=1e-3) and ride[3] == pytest.approx(ride[0])
    assert v(4.5)["sound_1"].params.start == 4.5
    assert v(9.25)["regularpolygon_1"].params.morph_from == "square_1"
    assert not [i for i in r.scene.engine.lint() if i.severity == "error"]
    track, rate = r.scene.engine.audio_track()
    assert len(track) == 12 * rate and abs(track[int(4.6 * rate)]) > 0


def test_the_sound_cue_example_mixes_its_clip_at_the_landing():
    import runpy

    from tests.test_manim_import import ROOT

    e = runpy.run_path(str(ROOT / "examples" / "sound_cue.py"), run_name="x")["s"].engine
    track, rate = e.audio_track()
    assert len(track) == 3 * rate
    assert np.abs(track[: int(1.29 * rate)]).max() == 0 and np.abs(track[int(1.32 * rate)]) > 0
    assert e.lint(strict=True) == []


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.parametrize("name, times", [("manim_camera_tour", [1.5, 4.0, 6.0, 9.5]),
                                         ("sound_cue", [0.2, 0.8, 1.4, 2.5])])  # fmt: skip
def test_example_golden_storyboards(name, times):
    import runpy

    from tests.conftest import gl_available
    from tests.test_beauty import _golden
    from tests.test_manim_import import ROOT

    if not gl_available():
        pytest.skip("needs OpenGL")
    path = ROOT / "examples" / f"{name}.py"
    if name.startswith("manim_"):
        e = import_manim(path).scene.engine
    else:
        e = runpy.run_path(str(path), run_name="x")["s"].engine
    sheet = e.storyboard(times, size=(320, 180))["image"]
    _golden(name, sheet, e.pass_runner().renderer_class())
