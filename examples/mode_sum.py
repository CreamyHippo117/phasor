"""One room's impulse response two ways at one seat: the echoes (image sources, amber) and
the notes (room modes, lowest first, cyan), with the note count live. As the notes climb the
cyan curve takes on the amber one; the gap between them reads below one millionth at the end.

Run:  uv run python examples/mode_sum.py       (prints the gap at three counts)
"""

from phasor import Counter, Scene, Text, ac, expr

SIZE, SRC, MIC = (3.5, 2.8, 2.4), (0.9, 0.8, 1.1), (2.6, 2.0, 1.4)
NOTES = 60_000

s = Scene("mode_sum", look="prism")
s.add(ac.ModeSum(id="ms", size=SIZE, source=SRC, listener=MIC, notes=1.0, max_notes=NOTES,
                 smoothing=0.3e-3, t_range=(0.0, 0.025), samples=600, layer="overlay",
                 pos=(0.25, 0.8, 0.0), extent=(1.28, 0.5)))  # fmt: skip
s.add(Text(id="title", text="echoes against notes", layer="overlay", pos=(0.89, 0.12, 0),
           style={"size": "title"}))  # fmt: skip
s.add(Counter(id="count", value=expr("ms.notes"), digits=0, math=False, suffix=" notes",
              layer="overlay", pos=(0.89, 0.92, 0)))  # fmt: skip
s.marker("climb", 1.0)
s.marker("full", 6.0)
s.at("climb").morph("ms.notes", float(NOTES), until="full")

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 3.5, 7.0):
        sem = e.inspect(t, id="ms")["objects"][0]["semantic"]
        print(f"t={t}: {sem['notes']} notes, gap {sem['max_rel_diff']:.2e}")
