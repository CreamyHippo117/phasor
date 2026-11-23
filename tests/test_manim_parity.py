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


def test_a_huge_wait_with_updaters_stays_bounded(tmp_path):
    """Lesson 2: whatever the script asks, the import's work is bounded (frames, keys)."""
    import time

    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        d = Dot().add_updater(lambda m, dt: m.move_to(RIGHT * math.sin(self.time)))
        self.add(d)
        for _ in range(3):
            self.wait(1e7)
""")  # fmt: skip
    p.write_text("import math\n" + p.read_text(encoding="utf-8"), encoding="utf-8")
    start = time.perf_counter()
    r = import_manim(p)
    assert time.perf_counter() - start < 60
    assert len(doc_of(r).timeline.tracks["dot_1.pos"]) <= 3 * 20_000


def test_key_simplification_is_linear_for_steady_motion():
    import time

    from phasor.importers.manim import _simplify

    samples = [(k / 30, (k * 0.01, 0.0, 0.0)) for k in range(20_000)]
    start = time.perf_counter()
    assert _simplify(samples, 1e-4) == [samples[0], samples[-1]]
    assert time.perf_counter() - start < 2
    wave = [(k / 30, (math.sin(k / 30), 0.0, 0.0)) for k in range(3000)]
    kept = _simplify(wave, 1e-4)
    assert 10 < len(kept) < 3000
    u = np.interp([s[0] for s in wave], [s[0] for s in kept], [s[1][0] for s in kept])
    assert np.abs(u - [s[1][0] for s in wave]).max() <= 1e-4 + 1e-12


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
    # a plain Scene has no frame in Manim; it imported before (as nothing): now it moves the
    # camera, with a warning (lesson 1: what imported before still imports)
    plain = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.add(Dot())
        self.play(self.camera.frame.animate.shift(LEFT))
""", "plain.py")  # fmt: skip
    r = import_manim(plain)
    assert any("MovingCameraScene" in w for w in r.warnings)
    assert r.scene.engine.state_at(1.0).values["camera.center"] == pytest.approx((-1, 0, 0))


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
    "class S(Scene):\n    def construct(self):\n        self.play(Wait(run_time='x'))",
    "class S(Scene):\n    def construct(self)\n        pass",  # not Python
    "class S(Scene):\n    def construct(self):\n        self.construct()",  # recursion
    "class S(Scene):\n    def construct(self):\n        raise SystemExit(3)",  # exit()
    "class S(Scene):\n    def construct(self):\n        d = Dot().add_updater(lambda m: "
    "__import__('sys').exit(1))\n        self.add(d)\n        self.wait(1)",
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


@pytest.mark.parametrize("body, warned", [
    ("self.play()", "no animation"),
    ("d = Dot()\n        self.add(d)\n        self.play(FadeIn(d), run_time=-1)", "negative"),
    ("g = VGroup(Dot(), Dot())\n        self.add(g)\n        g.target = VGroup(Dot(UP))\n"
     "        self.play(MoveToTarget(g))", "parts"),
])  # fmt: skip
def test_what_imported_before_still_imports_with_a_warning(tmp_path, body, warned):
    """Lesson 1: these imported before Milestone 20 (they did nothing, or played in no
    time); they still import, and now say what happened."""
    r = import_manim(script(tmp_path, f"""
class S(Scene):
    def construct(self):
        {body}
        self.wait(1)
"""))  # fmt: skip
    assert any(warned in w for w in r.warnings), r.warnings


def test_a_redrawn_curve_is_bounded_by_the_numbers_it_samples(tmp_path, monkeypatch):
    """Lesson 2: frames are bounded, and so is what they sample (a 240-point curve redrawn
    every frame is hundreds of numbers a frame)."""
    import json

    from phasor.importers import manim as mi

    monkeypatch.setattr(mi, "MAX_KEY_NUMBERS", 200_000)
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        ax = Axes()
        k = ValueTracker(1)
        g = always_redraw(lambda: ax.plot(lambda x: k.get_value() * np.sin(x)))
        self.add(ax, g)
        self.play(k.animate.set_value(3), run_time=600, rate_func=there_and_back)
""")  # fmt: skip
    r = import_manim(p)
    assert any("numbers" in w for w in r.warnings)
    size = len(json.dumps(doc_of(r).model_dump(mode="json")))
    assert size < 4_000_000


# ---------------------------------------------------------------- bounded work (M20 audit)


def _keyed_numbers(r) -> int:
    from phasor.importers.manim import _numbers

    return sum(_numbers(s.to) for segs in doc_of(r).timeline.tracks.values() for s in segs)


class _Count:
    """Counts calls of ``owner.name`` (still calling it)."""

    def __init__(self, monkeypatch, owner, name):
        self.n, orig = 0, getattr(owner, name)

        def counted(*a, **k):
            self.n += 1
            return orig(*a, **k)

        monkeypatch.setattr(owner, name, counted)


@pytest.mark.parametrize("plays", [50, 200])
def test_many_short_plays_driving_a_redrawn_curve_are_one_patch(tmp_path, monkeypatch, plays):
    """Blocker B: every play's flush recompiled the growing document (50 plays: 228 s,
    3.9 GB, 16 MB; 100 killed at 600 s). Now the scene's document is written a constant
    number of times, whatever the number of plays, and its keys stay within the budget."""
    import json
    import time

    from phasor.importers import manim as mi
    from phasor.scene import Scene as PhasorScene

    applies = _Count(monkeypatch, PhasorScene, "_apply")
    p = script(tmp_path, f"""
class S(Scene):
    def construct(self):
        t = ValueTracker(0)
        ax = Axes(x_range=[0, 10], y_range=[-2, 2])
        g = always_redraw(lambda: ax.plot(lambda x: np.sin(x + t.get_value()),
                                          x_range=[0, 10, 0.04]))
        self.add(ax, g)
        for i in range({plays}):
            self.play(t.animate.set_value(i * 0.1), run_time=0.5)
""")  # fmt: skip
    start = time.perf_counter()
    r = import_manim(p)
    assert time.perf_counter() - start < 180  # about 6 s and 9 s here
    assert applies.n <= 3  # one patch at the end (and the camera set before the script)
    assert _keyed_numbers(r) <= mi.MAX_KEY_NUMBERS
    assert len(json.dumps(doc_of(r).model_dump(mode="json"))) < 13_000_000
    assert r.duration == pytest.approx(plays * 0.5)
    # the curve follows the tracker while the budget lasts
    sub = np.asarray(state(r, 2.0, "graph_1").subpaths[0])
    assert sub[0][1] == pytest.approx(np.sin(0.3) * ax_unit(r), abs=0.05)


def ax_unit(r) -> float:
    """World units per y unit of the test's axes (y from -2 to 2)."""
    from phasor.importers.manim import FRAME_HEIGHT

    del r
    return (FRAME_HEIGHT - 2) / 4  # Manim's Axes: y_length 6 over 4 units


def test_the_updater_budget_is_enforced_and_dt_stays_manims(tmp_path, monkeypatch):
    """S1: past the budget the rate became 0 and stayed there (every later dt updater got
    dt 0 and froze), every later play still ran a frame, and the counts kept growing. Now
    the frames run are bounded, the rate falls gracefully, a dt updater's motion adds up to
    Manim's while it runs, and when updaters stop the warning says when."""
    from phasor.importers import manim as mi

    monkeypatch.setattr(mi, "MAX_FRAMES", 200)
    calls = _Count(monkeypatch, mi, "_call_updater")
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        d = Dot().add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(d)
        for _ in range(300):
            self.wait(0.5)
""")  # fmt: skip
    r = import_manim(p)
    assert calls.n <= 200  # one updater, at most MAX_FRAMES frames
    slow = next(w for w in r.warnings if "fewer frames a second" in w)
    stop = next(w for w in r.warnings if "updaters stop at" in w)
    t_slow = float(slow.split(" from ")[1].split(" s")[0])
    t_stop = float(stop.split("stop at ")[1].split(" s")[0])
    assert 0 < t_slow < t_stop < 150
    # Manim: each wait(0.5) at 30 fps moves the dot by 14/30 (its first frame has dt 0)
    for t in (t_slow, (t_slow + t_stop) / 2 // 0.5 * 0.5, t_stop):
        assert state(r, t, "dot_1").pos[0] == pytest.approx(t / 0.5 * 14 / 30, abs=1e-6)
    assert state(r, 150.0, "dot_1").pos[0] == pytest.approx(t_stop / 0.5 * 14 / 30, abs=1e-6)


def test_the_numbers_budget_bounds_the_keys(tmp_path, monkeypatch):
    from phasor.importers import manim as mi

    monkeypatch.setattr(mi, "MAX_KEY_NUMBERS", 50_000)
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        t = ValueTracker(0)
        ax = Axes()
        g = always_redraw(lambda: ax.plot(lambda x: np.sin(x + t.get_value())))
        self.add(ax, g)
        for i in range(40):
            self.play(t.animate.increment_value(0.1), run_time=0.5)
""")  # fmt: skip
    r = import_manim(p)
    assert _keyed_numbers(r) <= 50_000
    assert any("updaters stop at" in w for w in r.warnings)


def test_many_mobjects_with_updaters_cost_linear_work(tmp_path, monkeypatch):
    """S2: 2000 dots with a dt updater each over wait(30) took 524 s and 2.4 GB: each
    object's add compiled the whole document, and the updater's signature was read on every
    call. Now the scene is written once and the work is bounded by the budget."""
    import inspect
    import time

    from phasor.importers import manim as mi
    from phasor.scene import Scene as PhasorScene

    applies = _Count(monkeypatch, PhasorScene, "_apply")
    signatures = _Count(monkeypatch, inspect, "signature")
    calls = _Count(monkeypatch, mi, "_call_updater")
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        dots = VGroup(*[Dot() for _ in range(2000)])
        for i, d in enumerate(dots):
            d.add_updater(lambda m, dt, i=i: m.shift(RIGHT * dt * 0.001 * (i % 7)))
        self.add(dots)
        self.wait(30)
""")  # fmt: skip
    start = time.perf_counter()
    r = import_manim(p)
    assert time.perf_counter() - start < 180  # about 8 s here
    assert applies.n <= 3
    assert signatures.n <= 2000 + 10  # once per updater
    assert calls.n <= mi.MAX_UPDATER_CALLS
    assert _keyed_numbers(r) <= mi.MAX_KEY_NUMBERS
    # the sixth dot (i = 5) keeps Manim's motion: dt adds up to (30 * 30 - 1) / 30
    assert state(r, 30.0, "dot_6").pos[0] == pytest.approx(0.005 * (30 * 30 - 1) / 30, abs=1e-6)


# ---------------------------------------------------------------- wait_until (M20 audit S5)


def test_wait_until_stops_at_the_first_frame_the_condition_holds(tmp_path):
    """S5: ``wait_until`` waited ``max_time`` whatever the condition said. Manim (CE 0.22.1,
    measured; also ``tests/manim_real/scenes/wait_until.py``) checks the condition after
    each frame of the wait, when its clock is one frame on, and the wait ends there."""
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        d = Dot().add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(d)
        self.wait_until(lambda: d.get_x() >= 0.99, max_time=10)
        self.play(FadeOut(d))
        t = self.time
        self.wait_until(lambda: self.time >= t + 0.49)
        t = self.time
        self.wait(5, stop_condition=lambda: self.time >= t + 0.99)
        self.wait_until(lambda: False, max_time=0.5)
""")  # fmt: skip
    r = import_manim(p)
    assert r.warnings == []
    waits = [p for p in r.trace if p["anims"][0]["cls"] == "Wait"]
    # frames at 30 a second, the first with dt 0: x is 1 at frame 30, when Manim's clock
    # says 31/30 (real Manim: time 1.0333, x 1.0)
    assert waits[0]["duration"] == pytest.approx(31 / 30)
    assert waits[1]["duration"] == pytest.approx(0.5)
    assert waits[2]["duration"] == pytest.approx(1.0)  # stop_condition, as wait_until
    assert waits[3]["duration"] == pytest.approx(0.5)  # never true: the whole max_time
    assert r.duration == pytest.approx(31 / 30 + 1 + 0.5 + 1 + 0.5)


def test_a_failing_wait_until_condition_is_a_clean_error(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.wait_until(lambda: 1 / 0)
""")  # fmt: skip
    with pytest.raises(ManimImportError, match="wait_until condition failed"):
        import_manim(p)
    assert "manim" not in sys.modules


# ---------------------------------------------------------------- the audit's notes


@pytest.mark.parametrize("name", ["slow_into", "wiggle", "there_and_back", "double_smooth"])
def test_a_rate_function_without_a_name_stays_within_its_tolerance(tmp_path, name):
    """Note 1: 16 even pieces missed ``slow_into`` by 8.9%, ``wiggle`` by 3.6%,
    ``there_and_back`` by 1.7% of the motion; the pieces now follow the function."""
    from phasor.importers import manim as mi

    p = script(tmp_path, f"""
class S(Scene):
    def construct(self):
        d = Dot()
        self.add(d)
        self.play(d.animate.shift(RIGHT * 4), run_time=2, rate_func={name})
""")  # fmt: skip
    r = import_manim(p)
    f = getattr(mi.rate_functions, name)
    worst = max(abs(state(r, 2 * u, "dot_1").pos[0] - 4 * f(u))
                for u in np.linspace(0, 1, 801))  # fmt: skip
    assert worst <= 4 * mi.RATE_TOLERANCE + 1e-6
    assert len(doc_of(r).timeline.tracks["dot_1.pos"]) <= mi.MAX_RATE_PIECES + 2


def test_config_frame_rate_is_the_rate_updaters_run_at(tmp_path):
    """Note 5: the stand-in said 60 frames a second and ran updaters at 30. It says 30
    (Manim's -qm) and runs a script's own rate (1 to 120)."""
    body = """
class S(Scene):
    def construct(self):
        config.frame_rate = {rate}
        sq = Square().add_updater(lambda m, dt: m.rotate(dt))
        self.add(sq)
        self.play(Wait(run_time=3))
"""
    r = import_manim(script(tmp_path, body.format(rate="config.frame_rate")))
    turned = math.radians(state(r, 3.0, "square_1").rotation)
    assert turned == pytest.approx((3 * 30 - 1) / 30, abs=1e-6) and r.warnings == []
    r = import_manim(script(tmp_path, body.format(rate=60)))
    turned = math.radians(state(r, 3.0, "square_1").rotation)
    assert turned == pytest.approx((3 * 60 - 1) / 60, abs=1e-6)
    r = import_manim(script(tmp_path, body.format(rate=1e6)))
    assert any("config.frame_rate" in w for w in r.warnings)
    assert math.radians(state(r, 3.0, "square_1").rotation) == pytest.approx(89 / 30, abs=1e-6)


def test_add_sound_says_when_it_clamps(tmp_path):
    """Note 3: a negative ``time_offset`` or a gain past the range was clamped silently."""
    from tests.test_sound import wav

    wav(tmp_path / "a.wav", np.full(800, 1000))
    r = import_manim(script(tmp_path, """
class S(Scene):
    def construct(self):
        self.add_sound("a.wav", time_offset=-2, gain=99)
        self.wait(1)
"""))  # fmt: skip
    assert any("starts at 0 s" in w for w in r.warnings)
    assert any("gain 99 dB is played at 24 dB" in w for w in r.warnings)


def test_a_script_past_the_time_budget_is_a_clean_error(tmp_path, monkeypatch):
    """Note 10: a script that runs on and on stops with a clean error at the next play or
    updater call (a loop that never calls back cannot be stopped this way)."""
    from phasor.importers import manim as mi

    monkeypatch.setattr(mi, "MAX_IMPORT_SECONDS", 0.0)
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.wait(1)
""")  # fmt: skip
    with pytest.raises(ManimImportError, match="ran for more than"):
        import_manim(p)
    assert "manim" not in sys.modules


def test_saving_elsewhere_never_writes_through_a_link(tmp_path):
    """Note 10: bringing a sound along followed a symlink at the destination."""
    import os

    from phasor.importers.manim import import_manim as imp
    from tests.test_sound import wav

    wav(tmp_path / "click.wav", np.full(800, 1000))
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.add_sound("click.wav")
        self.wait(1)
""")  # fmt: skip
    elsewhere, outside = tmp_path / "elsewhere", tmp_path / "outside"
    elsewhere.mkdir()
    outside.mkdir()
    (outside / "victim.wav").write_bytes(b"mine")
    try:
        os.symlink(outside / "victim.wav", elsewhere / "click.wav")
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here (Windows without the privilege)")
    imp(p).scene.engine.save(elsewhere / "s.phasor.json")
    assert (outside / "victim.wav").read_bytes() == b"mine"
    assert (elsewhere / "click.wav").is_symlink()


def test_a_scene_script_that_exits_is_a_tool_error(tmp_path):
    """Note 10: SystemExit escaped the workspace's guard (and the MCP server with it)."""
    from phasor.workspace import ToolError, Workspace

    ws = Workspace(tmp_path)
    with pytest.raises(ToolError, match="exit"):
        ws._guard(lambda: sys.exit(3))
