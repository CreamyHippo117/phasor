"""Three runs of a sound source at constant speed, in slow motion.

The wavefronts crowd ahead of a subsonic source. At Mach 1 their leading
edges meet at the source. Above Mach 1 the rings trail behind, tangent to
the two sides of the Mach cone. Warm hues show higher local pitch.
"""

from phasor import Scene, Text, ac, expr

s = Scene("doppler", look="prism")
s.set("camera.height", 8)

runs = (
    ("subsonic", 0.65, "Subsonic   Mach 0.65"),
    ("sonic", 1.0, "Sonic   Mach 1"),
    ("supersonic", 1.65, "Supersonic   Mach 1.65"),
)
for index, (name, mach, title) in enumerate(runs):
    start = 3.0 * index
    active = expr(f"max(0, 1 - abs(clamp(floor(t / 3), 0, 2) - {index}))")
    clock = expr(f"0.84 * clamp((t - {start}) / 2.65, 0, 1)")
    s.add(ac.Doppler(id=name, speed=mach, frequency=12, time=clock,
                     pos=(-1.1, -0.25, 0), size=3.55,
                     opacity=active))  # fmt: skip
    caption = f"{name}_caption"
    s.add(Text(id=caption, text=title, pos=(8 / 9, 0.08), layer="overlay", opacity=active,
               style={"size": "title"}))  # fmt: skip
    s.marker(name, start)

s.marker("end", 9.0)
