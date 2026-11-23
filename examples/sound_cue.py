"""A sound on cue: a ``Sound`` clip plays when a ball lands (Milestone 20).

The clip is a WAV file in the scene's folder (``assets/chime.wav``); the document names it,
and ``phasor render`` mixes it into the video's sound at ``start`` seconds, next to any
narration. Nothing is drawn for it: the inspector lists it (``kind: sound``, its length and
rate) and lint says when the file is missing or cannot be mixed.

Run:  uv run python examples/sound_cue.py        (prints the clip and the mixed track)
"""

from phasor import Dot, Line, Scene, Sound, Text

s = Scene("sound_cue")
s.set("camera.height", 6)
s.add(Line(id="floor", points=[(-3, -1.5), (3, -1.5)], style={"color": "muted"}))
s.add(Dot(id="ball", pos=(0, 2), r=0.25, style={"color": "highlight", "glow": "soft"}))
s.add(Text(id="ball.label", text="ding", pos=(0.9, -1.0), style={"size": "label"}))
s.marker("drop", 0.3)
s.marker("land", 1.3)
s.marker("end", 3.0)
s.at("drop").morph("ball.pos", (0, -1.25, 0))
s.at("land").reveal("ball.label")
s.add(Sound(id="chime", path="assets/chime.wav", start=1.3))

if __name__ == "__main__":
    clip = s.engine.inspect(0.0, id="chime")["objects"][0]["semantic"]
    print(f"chime: {clip['seconds']} s at {clip['rate']} Hz, from {clip['start']} s")
    track, rate = s.engine.audio_track()
    print(f"mixed track: {len(track) / rate:g} s at {rate} Hz, loudest {abs(track).max()}")
    print("lint:", [i.code for i in s.engine.lint(strict=True)] or "clean")
