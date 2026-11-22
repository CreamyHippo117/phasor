"""Three polar patterns. The array steers, then the piston narrows at higher frequency."""

from phasor import Scene, Text, ac

s = Scene("directivity", look="prism")
s.set("camera.height", 7)

s.add(ac.Directivity(id="array", elements=5, steer=-30, pos=(-4, 0, 0), size=1.65))
s.add(ac.Directivity(id="piston", kind="piston", frequency=480, pos=(0, 0, 0), size=1.65))
s.add(ac.Directivity(id="cardioid", kind="cardioid", pos=(4, 0, 0), size=1.65))

s.add(Text(id="array_name", text="Linear array", layer="overlay", pos=(0.318, 0.81)))
s.add(Text(id="piston_name", text="Baffled piston", layer="overlay", pos=(0.889, 0.81)))
s.add(Text(id="cardioid_name", text="Cardioid", layer="overlay", pos=(1.46, 0.81)))

s.marker("steer", 1.0)
s.marker("narrow", 2.5)
s.marker("end", 4.5)
s.at("steer").sweep("array.steer", 35.0, show=False)
s.at("narrow").sweep("piston.frequency", 1700.0, show=False)
