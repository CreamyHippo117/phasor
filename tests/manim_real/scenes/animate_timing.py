from manim import *


class AnimateTiming(Scene):
    def construct(self):
        d = Dot(LEFT * 3, color=RED)
        sq = Square(side_length=1)
        self.play(Create(sq), run_time=2)
        self.add(d)
        self.play(d.animate.shift(RIGHT * 6), run_time=2)
        self.play(d.animate.shift(UP * 2), rate_func=linear)
        self.play(sq.animate.scale(2).rotate(PI / 4))
        self.play(FadeOut(d), sq.animate.shift(LEFT * 2), run_time=1.5)
        self.play(
            LaggedStart(*[FadeIn(Dot(RIGHT * k)) for k in range(3)], lag_ratio=0.5),
            run_time=2,
        )
        self.play(Succession(Indicate(sq), Rotate(sq, PI / 2)))
        self.wait(0.5)
