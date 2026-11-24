"""A scene built from pictures: a picture sequence as the whole frame, a still and the same
sequence again as placed objects.

``frames/orbit_%02d.png`` names twelve pictures (examples/assets/frames, made by
examples/assets/make_assets.py); played at ``fps`` they are a clip like any other, with
``start``, ``trim``, ``speed`` and ``loop``. ``Scene.background`` puts one behind everything,
filling the frame; text may sit on it. Put your own in:
``s.background("renders/shot_%04d.png", fps=24)``. Needs only Pillow.

Run:  uv run python examples/picture_scene.py      (prints the sequence's length)
"""

from phasor import Image, Scene, Text

s = Scene("picture scene", look="prism")
s.set("camera.height", 4.5)
s.background("assets/frames/orbit_%02d.png", fps=4.0, loop=True)
s.add(Image(id="sunset", path="assets/sunset.png", width=2.2, pos=(-2.75, -1.25, 0.0),
            preset="framed"))  # fmt: skip
s.add(Image(id="fast", path="assets/frames/orbit_%02d.png", width=1.9, pos=(2.85, 1.25, 0.0),
            fps=12.0, loop=True, preset="framed"))  # fmt: skip
s.add(Text(id="caption", text="twelve pictures, played as a clip", pos=(0.0, -1.95, 0.0)))

s.marker("stills", 0.4)
s.marker("caption", 1.6)
s.marker("end", 4.0)
s.at("stills").reveal("sunset", "fast")
s.at("caption").reveal("caption")

if __name__ == "__main__":
    sem = s.engine.inspect(1.0, id="background")["objects"][0]["semantic"]
    print(f"sequence: {sem['frames']} pictures of {sem['pixels']} at {sem['fps']:g} fps; "
          f"at 1 s it shows picture {sem['frame']}")  # fmt: skip
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
