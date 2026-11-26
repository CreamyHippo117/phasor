"""Evidence for pending/phase_wheel.md: is the prism phase wheel continuous end to end?

1. The theme's ``phase`` colormap as loaded, sampled 4096 times through ``lut_colors`` (the
   path PhasorField uses): evenness in OKLab and closure at 2 pi.
2. 144 still hands at angles 2 pi k / 144 drawn with the prism look (glow, bloom, tone map,
   sRGB), each hand read back from the finished still: does the drawn hue follow the angle?

Run (needs OpenGL):  uv run python pending/phase_wheel_probe.py [out.png]
"""

import math
import sys

import numpy as np

from phasor import Scene, filt
from phasor.domains.strokes import lut_colors
from phasor.style.theme import load_theme

M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])  # fmt: skip
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])  # fmt: skip


def oklab(rgb):
    return np.cbrt(np.maximum(rgb @ M1.T, 0)) @ M2.T


def linear(srgb8):
    s = srgb8 / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


# 1. the colormap itself
lut = np.asarray(load_theme("prism").resolve("colormap", "phase"), dtype=np.float64)
lab = oklab(lut_colors(lut, np.linspace(0, 1, 4096, endpoint=False))[:, :3])
step = np.linalg.norm(np.diff(np.r_[lab, lab[:1]], axis=0), axis=1)
hue = np.degrees(np.unwrap(np.arctan2(lab[:, 2], lab[:, 1])))
dh = np.diff(np.r_[hue, hue[0] + 360])
chroma = np.hypot(lab[:, 1], lab[:, 2])
print(
    f"colormap: {len(lut)} entries, {len(np.unique(lut.round(6), axis=0))} distinct; "
    f"L {lab[:, 0].min():.3f}-{lab[:, 0].max():.3f}, C {chroma.min():.3f}-{chroma.max():.3f}"
)
print(
    f"  dE max/median per 1/4096 turn {step.max() / np.median(step):.2f}; hue per step "
    f"{dh.min():.4f}-{dh.max():.4f} deg (even {360 / 4096:.4f}); closes: dE {step[-1]:.5f}"
)

# 2. end to end, from the finished still
N, COLS, HEIGHT, W, H = 144, 12, 13.0, 1200, 1200
s = Scene("wheel_probe", look="prism")
s.set("camera.height", HEIGHT)
res = tuple((math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)) for k in range(N))
s.add(filt.PhasorField(id="w", poles=((1.0, 0.0),) * N, residues=res, layout="grid",
                       order="index", spacing=1.0, rings=0.0))  # fmt: skip
img = s.engine.still(0.0, (W, H))
if len(sys.argv) > 1:
    from PIL import Image

    Image.fromarray(img).save(sys.argv[1])
pic = oklab(linear(img.astype(np.float64)))
ppy, ppx = H / HEIGHT, W / (HEIGHT * 16 / 9)  # the still keeps the scene's 16:9 view
drawn = []
for k in range(N):
    x, y, a = k % COLS - 5.5, 5.5 - k // COLS, 2 * math.pi * k / N
    along = [pic[round(H / 2 - (y + f * math.sin(a)) * ppy),
                 round(W / 2 + (x + f * math.cos(a)) * ppx)]
             for f in np.linspace(0.12, 0.40, 15)]  # fmt: skip
    drawn.append(max(along, key=lambda c: c[0]))  # the brightest pixel on the hand
drawn = np.array(drawn)
h = np.degrees(np.arctan2(drawn[:, 2], drawn[:, 1])) % 360
dh = (np.diff(np.r_[h, h[:1]]) + 180) % 360 - 180
de = np.linalg.norm(np.diff(np.r_[drawn, drawn[:1]], axis=0), axis=1)
c = np.hypot(drawn[:, 1], drawn[:, 2])
print(f"still: L {drawn[:, 0].min():.3f}-{drawn[:, 0].max():.3f}, C {c.min():.3f}-{c.max():.3f}")
print(
    f"  hue step per {360 / N:g} deg of angle: {dh.min():.2f}-{dh.max():.2f} (median "
    f"{np.median(dh):.2f}); dE x100 between neighbours {de.min() * 100:.2f}-{de.max() * 100:.2f}"
)
