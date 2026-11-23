from manim import *


class WaitUntil(Scene):
    def construct(self):
        dot = Dot(color=YELLOW).shift(LEFT * 3)
        dot.add_updater(lambda m, dt: m.shift(RIGHT * dt * 2))
        self.add(dot)
        self.wait_until(lambda: dot.get_x() >= -1.1, max_time=10)  # ends at 1 s
        box = Square(side_length=1).shift(DOWN * 2)
        self.play(Create(box), run_time=1)
        self.wait(3, stop_condition=lambda: dot.get_x() >= 2.5)
        self.play(box.animate.shift(UP), run_time=0.5)
