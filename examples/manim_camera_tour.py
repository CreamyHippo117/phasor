"""A Manim Community scene with a moving camera, updaters and a sound: it runs in phasor
unchanged (``phasor manim examples/manim_camera_tour.py``), and in Manim CE 0.22.1 too.

A moving camera zooms onto a filled shape, a dot rides round a circle on an updater driven
by a ValueTracker, a sound plays when the dot sets off, the camera goes back to where it
began, a square morphs into a hexagon while the circle changes color, and a number line
shows its numbers.
"""

from manim import *


class CameraTour(MovingCameraScene):
    def construct(self):
        ring = Circle(radius=1.5, color=BLUE, fill_opacity=0.15).shift(LEFT * 3)
        angle = ValueTracker(0)
        rider = Dot(color=YELLOW).add_updater(
            lambda m: m.move_to(ring.point_at_angle(angle.get_value()))
        )
        spoke = always_redraw(lambda: Line(ring.get_center(), rider.get_center(), color=GREY))
        box = Square(side_length=2, color=GREEN, fill_opacity=0.5).shift(RIGHT * 3)
        hexagon = RegularPolygon(n=6, radius=1.2, color=ORANGE, fill_opacity=0.5).shift(RIGHT * 3)
        line = NumberLine(x_range=[0, 4, 1], length=6, include_numbers=True).to_edge(DOWN)
        title = Text("a camera tour", font_size=36).to_edge(UP)

        self.play(Write(title), Create(line))
        self.play(DrawBorderThenFill(ring), FadeIn(box, shift=UP))
        self.add(rider, spoke)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(ring).set(width=ring.width * 2.5), run_time=1.5)
        self.add_sound("assets/chime.wav")
        self.play(angle.animate.set_value(TAU), run_time=3, rate_func=linear)
        self.play(Restore(self.camera.frame))
        self.play(Transform(box, hexagon), ring.animate.set_color(PURPLE), run_time=1.5)
        self.play(Indicate(title))
        self.wait(1)
