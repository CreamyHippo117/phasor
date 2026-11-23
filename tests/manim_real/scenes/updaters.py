from manim import *


class Updaters(Scene):
    def construct(self):
        t = ValueTracker(-3)
        dot = Dot(color=YELLOW)
        dot.add_updater(lambda m: m.move_to(RIGHT * t.get_value()))
        label = Dot(color=RED, radius=0.05)
        label.add_updater(lambda m: m.next_to(dot, UP, buff=0.3))
        spinner = Square(side_length=1).shift(DOWN * 2)
        spinner.add_updater(lambda m, dt: m.rotate(dt))
        tail = always_redraw(lambda: Line(ORIGIN + DOWN * 3, dot.get_center()))
        self.add(dot, label, spinner, tail)
        self.play(t.animate.set_value(3), run_time=3, rate_func=linear)
        self.wait(1)
        self.play(t.animate.set_value(0), run_time=1)
