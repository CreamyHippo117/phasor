from manim import *


class AxesPlot(Scene):
    def construct(self):
        ax = Axes(x_range=[-3, 3, 1], y_range=[-1, 5, 1], x_length=6, y_length=4)
        g = ax.plot(lambda x: x**2 / 2, x_range=[-2, 2], color=YELLOW)
        dot = Dot(ax.c2p(1, 0.5), color=RED)
        line = NumberLine(x_range=[0, 4, 1], length=4).to_edge(DOWN)
        self.play(Create(ax), run_time=2)
        self.play(Create(g), FadeIn(dot))
        self.play(Create(line))
        self.play(dot.animate.move_to(ax.c2p(2, 2)))
        self.wait(1)
