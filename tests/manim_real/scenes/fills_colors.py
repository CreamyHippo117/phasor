from manim import *


class FillsColors(Scene):
    def construct(self):
        a = Square(side_length=1.5, fill_opacity=0.5, color=BLUE).shift(LEFT * 4)
        b = Circle(radius=0.8).set_fill(YELLOW, opacity=0.8).shift(LEFT)
        c = Ellipse(width=2, height=1, color=GREEN).shift(RIGHT * 2)
        d = Annulus(inner_radius=0.4, outer_radius=0.8, fill_opacity=1).shift(RIGHT * 4.5)
        e = Arc(radius=1, start_angle=0, angle=PI / 2).shift(DOWN * 2)
        f = Sector(radius=1, angle=PI / 3, fill_opacity=0.7, color=ORANGE).shift(
            DOWN * 2 + LEFT * 3
        )
        g = RoundedRectangle(corner_radius=0.3, width=2, height=1).shift(UP * 2.5)
        h = DashedLine(LEFT * 3, RIGHT * 3).shift(DOWN * 3.3)
        self.play(*[FadeIn(m) for m in (a, b, c, d, e, f, g, h)])
        self.play(a.animate.set_color(RED), b.animate.set_fill(GREEN, opacity=0.3), run_time=2)
        self.play(Transform(c, Circle(radius=0.6).shift(RIGHT * 2)))
        self.wait(0.5)
