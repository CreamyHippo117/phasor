from manim import *


class ShapesLayout(Scene):
    def construct(self):
        a = Dot(UP * 2 + LEFT * 3)
        b = Square(side_length=2).shift(RIGHT * 4)
        c = Circle(radius=1).to_edge(DOWN, buff=0.5)
        d = Rectangle(width=3, height=1.5, color=GREEN).next_to(b, UP, buff=0.5)
        e = Triangle().scale(0.8).to_corner(UL)
        f = RegularPolygon(n=5, radius=0.7).next_to(c, LEFT, buff=1)
        g = Line(LEFT * 2, RIGHT * 2 + UP).shift(DOWN)
        h = Arrow(LEFT, RIGHT * 2, color=YELLOW).shift(UP * 3)
        group = VGroup(*[Dot() for _ in range(4)]).arrange(RIGHT, buff=0.6).shift(UP * 1.2)
        self.add(a, b, c, d, e, f, g, h, group)
        self.wait(1)
