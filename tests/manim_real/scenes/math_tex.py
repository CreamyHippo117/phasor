from manim import *


class MathTexScene(Scene):
    def construct(self):
        eq1 = MathTex("a", "+", "b", "=", "c").shift(UP)
        eq2 = MathTex("a", "=", "c", "-", "b").shift(UP)
        single = MathTex(r"e^{i\pi} + 1 = 0").shift(DOWN * 1.5)
        words = Tex("Hello world").to_edge(DOWN)
        self.play(Write(eq1))
        self.play(TransformMatchingTex(eq1, eq2), run_time=1.5)
        self.play(FadeIn(single), Write(words))
        self.wait(1)
