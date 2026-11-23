from manim import *


class MovingCamera(MovingCameraScene):
    def construct(self):
        sq = Square(side_length=1).shift(RIGHT * 3)
        dot = Dot(LEFT * 3)
        self.add(sq, dot)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(sq).set(width=sq.width * 4), run_time=2)
        self.play(self.camera.frame.animate.shift(LEFT * 2).scale(1.5))
        self.play(Restore(self.camera.frame))
        self.wait(0.5)
