"""A narrated scene with a sound bed (SOUND.md): ambience under the voice, effects on words.

The bed is examples/assets/drift.wav, a soft drone made for this example (put your own
music or ambience in its place: WAV, or OGG/FLAC/MP3 with ``uv sync --extra sound``). It
loops under the whole scene and dips while a sentence is spoken. A chime lands as the
pole appears, a whoosh just before it moves. The theme sets every level (``[sound]``).

Run:  uv run python examples/sound_bed.py      (prints each lane's level)
      uv run phasor render examples/sound_bed.py -o out/sound_bed.mp4
"""

from phasor import Dot, Scene

s = Scene("sound_bed", look="prism")
s.add(Dot(id="pole", pos=(-1.5, 0.0, 0.0), r=0.18))
s.narrate(
    "Here is a pole [[pole_appears]].\n"
    "Watch it move [[pole_moves]] across the plane.",
    voice="auto",  # espeak-ng when installed; otherwise timing only, and no voice to duck under
)
s.at("pole_appears").bloom_in("pole")
s.at("pole_moves").settle("pole.pos", (1.5, 0.0, 0.0))
s.bed("assets/drift.wav", fade=1.5)
s.cue("pole_appears", "chime")
s.cue("pole_moves", "whoosh", offset=-0.25)
s.pace()

if __name__ == "__main__":
    m = s.engine.sound_mix()
    for lane in m.lanes:
        print(f"{lane.kind:5s} {lane.id:20s} at {lane.start:5.2f} s, {lane.gain_db:+.1f} dB")
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
