"""Footage in a scene: a video clip as the whole frame, and again as a framed object on it.

``Scene.from_video`` makes a scene of the clip: it fills the frame and the scene lasts as long
as it plays (here at half speed: 4 s). On top, the same clip again, a second of it looping in
a framed inset, and text that sits on the footage (a backdrop: no overlap; contrast is judged
against the frame). The clip (examples/assets/clip.mp4, 10 KB, made by
examples/assets/make_assets.py) sits next to the scene; the document stores only its path.
Put your own in: ``Scene.from_video("my_clip.mp4")``. Needs PyAV (the ``encode`` extra).

Run:  uv run python examples/video_input.py      (prints what the clip is)
"""

from phasor import Scene, Text, Video

s = Scene.from_video("assets/clip.mp4", title="video input", look="prism",
                     clip={"speed": 0.5})  # fmt: skip
s.set("camera.height", 4.5)
s.add(Video(id="inset", path="assets/clip.mp4", width=2.2, pos=(2.55, -1.15, 0.0),
            trim=(0.5, 1.5), loop=True, preset="framed"))  # fmt: skip
s.add(Text(id="title", text="footage is an object too", pos=(-1.6, 1.75, 0.0)))
s.add(Text(id="note", text="a second of it, looping", pos=(2.55, -0.2, 0.0)))

s.marker("title", 0.3)
s.marker("inset", 1.4)
s.marker("end", 4.0)
s.at("title").reveal("title")
s.at("inset").reveal("inset", "note")

if __name__ == "__main__":
    sem = s.engine.inspect(2.0, id="clip")["objects"][0]["semantic"]
    print(f"clip: {sem['pixels']} pixels, {sem['frames']} frames at {sem['fps']:g} fps; "
          f"at 2 s it shows frame {sem['frame']}")  # fmt: skip
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
