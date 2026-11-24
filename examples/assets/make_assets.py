"""Make the small media files the footage examples use (Milestone 24), so they can be
checked and made again: ``clip.mp4`` (2 s of drifting light, 320 x 180, H.264) and
``frames/orbit_00.png`` ... ``orbit_11.png`` (a planet going round, 12 pictures).

Run:  uv run python examples/assets/make_assets.py      (needs the ``encode`` extra)
"""

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

HERE = Path(__file__).parent


def _clip(path: Path, w: int = 320, h: int = 180, fps: int = 24, seconds: float = 2.0) -> None:
    """Bands of teal and violet light drifting over a dark sky, a bright moon rising."""
    import av

    y, x = np.mgrid[0:h, 0:w] / h
    with av.open(str(path), "w", options={"movflags": "faststart"}) as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "30", "preset": "slow", "bf": "2", "g": "24"}
        for i in range(int(fps * seconds)):
            t = i / fps
            band = np.exp(-(((y - 0.45 - 0.12 * np.sin(2.2 * x + 2.4 * t)) / 0.09) ** 2))
            band2 = np.exp(-(((y - 0.7 - 0.08 * np.sin(3.1 * x - 1.7 * t + 1)) / 0.06) ** 2))
            moon = np.exp(-(((x - 1.35) ** 2 + (y - 0.75 + 0.2 * t) ** 2) / 0.004))
            r = 0.04 + 0.35 * band2 + 0.9 * moon
            g = 0.05 + 0.55 * band + 0.1 * band2 + 0.9 * moon
            b = 0.12 + 0.45 * band + 0.6 * band2 + 0.85 * moon
            rgb = np.clip(np.stack([r, g, b], -1) * 255, 0, 255).astype(np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _frames(folder: Path, n: int = 12, w: int = 192, h: int = 108) -> None:
    """A planet going once round a sun, flat colours (so each picture stays small)."""
    folder.mkdir(exist_ok=True)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    for i in range(n):
        a = 2 * np.pi * i / n
        px, py = w / 2 + 0.38 * w * np.cos(a), h / 2 + 0.32 * h * np.sin(a)
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (14, 16, 34)
        ring = np.abs(np.hypot((x - w / 2) / (0.38 * w), (y - h / 2) / (0.32 * h)) - 1) < 0.012
        img[ring] = (60, 64, 110)
        img[np.hypot(x - w / 2, y - h / 2) < 12] = (255, 196, 90)
        img[np.hypot(x - px, y - py) < 6] = (120, 220, 255)
        PILImage.fromarray(img).save(folder / f"orbit_{i:02d}.png", optimize=True)


if __name__ == "__main__":
    _clip(HERE / "clip.mp4")
    _frames(HERE / "frames")
    for p in [HERE / "clip.mp4", *sorted((HERE / "frames").glob("*.png"))]:
        print(p.relative_to(HERE), p.stat().st_size, "bytes")
