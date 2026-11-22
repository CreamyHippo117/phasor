"""Wind two tones clockwise and trace the magnitude of their centroid as frequency rises."""

from phasor import Scene, Text, fourier

s = Scene("fourier_winding", look="prism")
s.add(fourier.Winding(id="signal", pos=(-4.3, 0, 0), size=2.25, fs=64, samples=256,
                      progress=0))  # fmt: skip
s.add(Text(id="winding_title", text="Wound signal", pos=(0.36, 0.14), layer="overlay"))
s.add(Text(id="spectrum_title", text="Centroid magnitude", pos=(1.19, 0.14), layer="overlay"))
s.add(Text(id="frequency_label", text="Test frequency (Hz)", pos=(1.19, 0.87), layer="overlay"))
s.add(Text(id="frequency_start", text="0", pos=(0.77, 0.82), layer="overlay"))
s.add(Text(id="frequency_end", text="8", pos=(1.61, 0.82), layer="overlay"))

s.marker("trace", 0.0)
s.marker("scan", 1.5)
s.marker("resume", 4.75)
s.marker("hold", 7.5)
s.marker("end", 8.5)
s.at("trace").sweep("signal.progress", 1.0, show=False)
s.at("scan").sweep("signal.freq", 3.0, show=False)
s.at("resume").sweep("signal.freq", 8.0, show=False)
