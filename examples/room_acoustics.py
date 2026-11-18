"""Milestone 8: a room, its image sources, its impulse response and a slice of the pressure
field. (The full narrated reverb story is a later film; this is the static kit.)

Run:  uv run python examples/room_acoustics.py       (prints the room's physics)
"""

from phasor import Scene, Text, ac

SIZE, SRC, MIC = (8.0, 5.0, 3.0), (2.0, 2.5, 1.5), (6.0, 2.0, 1.5)

s = Scene("room_acoustics")
s.use_camera("orbit", azimuth=-62, elevation=28, distance=15, target=(4, 2.5, 1.2))
s.add(ac.Room(id="room", size=SIZE, material="curtain"))
s.add(ac.PressureSlice(id="field", size=SIZE, source=SRC, time=0.012, material="curtain",
                       frequency=250.0))  # fmt: skip
s.add(ac.Source(id="src", pos=SRC, tags=["key"]))
s.add(ac.Listener(id="mic", pos=MIC, tags=["key"]))
s.add(ac.ImageSources(id="ghosts", size=SIZE, source=SRC, order=1, material="curtain"))
s.add(ac.ImpulseResponse(id="ir", size=SIZE, source=SRC, listener=MIC, material="curtain", order=14,
                         duration=0.3, show_rt60=1.0, layer="overlay", pos=(0.15, 0.78, 0),
                         extent=(0.9, 0.2)))  # fmt: skip
s.add(Text(id="src.label", text="source"))
s.add(Text(id="mic.label", text="listener"))
s.constrain("label(src) near src")
s.constrain("label(mic) near mic")

if __name__ == "__main__":
    e = s.engine
    for name in ("room", "ghosts", "ir", "field"):
        print(name, e.inspect(0.0, id=name)["objects"][0]["semantic"])
