"""MIDI notes laid over a narration at chosen points (Milestone 23).

The narration is a recording (``assets/narration_demo.wav``, as in
``narrated_from_audio.py``): its sentences are found in the audio and the script's
``[[glide]]`` falls on its word. Two ``Notes`` objects play over it, each from a marker, so
they follow the voice wherever the recording puts it:

- ``partials``: notes made by the script, the first five partials of a square wave (odd
  harmonics 1, 3, 5, 7, 9 of A3, each 1/k as loud), entering one after another from
  ``glide`` on a pure sine voice: you hear the square wave being built. A fractional pitch
  (``57 + 12 log2 k``) plays its exact frequency. ``show=True`` draws them as a piano
  roll, revealed at ``glide``.
- ``motif``: a ``.mid`` file in the scene's folder (``assets/motif.mid``: bells over a drum
  kit, made by ``assets/make_assets.py``), from ``fade``.

``s.export_midi("notes_overlay.mid")`` writes both as one MIDI file at their scene times.

Run:  uv run python examples/notes_overlay.py   (prints the notes, the mix and lint)
"""

import math

from phasor import Dot, Notes, Scene, Text

SCRIPT = "A point appears on the plane. It glides [[glide]] to the right. Then it fades away again."

s = Scene("notes_overlay")
s.set("camera.height", 6)
s.add(Dot(id="point", pos=(-2, 0.6), r=0.25, style={"color": "highlight", "glow": "soft"}))
s.add(Text(id="point.label", text="z", pos=(-2, 1.3), style={"size": "label"}))
s.marker("intro", 0.0)
s.marker("move", 3.0)
s.marker("fade", 5.5)
s.at("intro").bloom_in("point")
s.at("intro", 0.4).reveal("point.label")
s.at("move").settle("point.pos", (2, 0.6))
s.at("move").settle("point.label.pos", (2, 1.3))
s.at("fade").exit("point", "point.label")
s.narrate(SCRIPT, audio="assets/narration_demo.wav")
s.snap_to_narration()

# The square wave's odd partials, quieter as 1/k, entering 0.2 s apart from "glide".
partials = [(57 + 12 * math.log2(k), 0.2 * i, 2.0 - 0.2 * i, max(1, round(110 / k)))
            for i, k in enumerate((1, 3, 5, 7, 9))]  # fmt: skip
s.add(Notes(id="partials", notes=partials, start="glide", instrument="sine", gain=-3.0,
            show=True, pos=(0, -1.6, 0), width=5.0, height=1.4))  # fmt: skip
s.at("glide").reveal("partials")  # the roll appears as its notes start
s.add(Notes(id="motif", path="assets/motif.mid", start="fade", gain=-6.0))

if __name__ == "__main__":
    for name in ("partials", "motif"):
        sem = s.engine.inspect(0.0, id=name)["objects"][0]["semantic"]
        print(f"{name}: {sem['count']} notes from {sem['start']!r}, pitches "
              f"{sem['pitch_range']}, {', '.join(sem['programs'])}")  # fmt: skip
    track, rate = s.engine.audio_track()
    print(f"mixed track: {len(track) / rate:g} s at {rate} Hz, loudest {abs(track).max()}")
    print("lint:", [i.code for i in s.engine.lint(strict=True)] or "clean")
