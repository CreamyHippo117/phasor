from manim import *


class TextWrite(Scene):
    def construct(self):
        title = Text("Hello phasor", font_size=60).to_edge(UP)
        body = Text("a second line").next_to(title, DOWN, buff=0.8)
        small = Text("small print", font_size=24, color=GREY).to_corner(DR)
        self.play(Write(title))
        self.play(FadeIn(body), run_time=0.5)
        self.add(small)
        self.play(Transform(body, Text("changed text").next_to(title, DOWN, buff=0.8)))
        self.play(body.animate.scale(0.5).shift(DOWN * 2))
        self.wait(1)
