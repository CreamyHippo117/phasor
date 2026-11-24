"""Narration from a recording, and the scene snapped onto it (Milestone 22).

The scene was laid out first with markers placed by hand (``intro``, ``move``, ``fade``),
its motions anchored to them with verbs. Then the narration came as a recording,
``assets/narration_demo.wav`` (any WAV; or the sound of an mp4, m4a, mp3, ogg or flac with
the ``encode`` extra). ``narrate(SCRIPT, audio=...)`` finds the speech and pauses in it and
aligns each sentence of the script to the speech that says it, so the script's own
``[[glide]]`` falls on its word. ``snap_to_narration`` then moves the hand-placed markers onto
the sentence starts, and every motion anchored to them follows. ``dry_run=True`` shows the
moves first (marker, from, to, why, confidence).

The recording here is synthetic (formant-shaped bursts, not a voice:
``tests/test_speech.recording(22, [6, 5, 5], rate=8000, snr=40)``); its sentences really
start at 0.6, 3.64 and 6.33 s.

Run:  uv run python examples/narrated_from_audio.py   (prints the moves and the mixed track)
"""

from phasor import Dot, Scene, Text

SCRIPT = "A point appears on the plane. It glides [[glide]] to the right. Then it fades away again."

s = Scene("narrated_from_audio")
s.set("camera.height", 6)
s.add(Dot(id="point", pos=(-2, 0), r=0.25, style={"color": "highlight", "glow": "soft"}))
s.add(Text(id="point.label", text="z", pos=(-2, 0.7), style={"size": "label"}))
# Laid out by hand before there was any narration.
s.marker("intro", 0.0)
s.marker("move", 3.0)
s.marker("fade", 5.5)
s.at("intro").bloom_in("point")
s.at("intro", 0.4).reveal("point.label")
s.at("move").settle("point.pos", (2, 0))
s.at("move").settle("point.label.pos", (2, 0.7))
s.at("fade").exit("point", "point.label")
# The recording arrives: sentences found in it, then the scene snapped onto them.
s.narrate(SCRIPT, audio="assets/narration_demo.wav")
preview = s.snap_to_narration(dry_run=True)
s.snap_to_narration()

if __name__ == "__main__":
    for m in preview["moves"]:
        print(
            f"{m['marker']}: {m['from']:g} s -> {m['to']:g} s  ({m['why']}; sure "
            f"{m['confidence']:.2f})"
        )
    print("markers:", s.engine.document.timeline.markers)
    track, rate = s.engine.audio_track()
    print(f"mixed track: {len(track) / rate:g} s at {rate} Hz")
    print("lint:", [i.code for i in s.engine.lint(strict=True)] or "clean")
