"""Render the showcase scenes into showcase/videos/, one video each.

    uv run python showcase/render_all.py                # full quality, 1080p, motion blur
    uv run python showcase/render_all.py --preset 4k    # 3840 x 2160
    uv run python showcase/render_all.py --preset preview   # quick, 960 x 540
    uv run python showcase/render_all.py prism_primes deep_zoom   # only these

On an NVIDIA machine the maths runs on the card and the video goes through its encoder;
without one it still works, just slower. A scene that fails is reported and the rest go on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "videos"

# (example, what it shows). The order is a good watching order.
SCENES = [
    ("prism_primes", "the prism look: primes heard through the zeta zeros, glowing strokes"),
    ("prism_domain", "domain coloring in the prism look: glowing zeros, poles, rays and rings"),
    ("prism_heat", "heat maps in the prism look: a hot spot, a vibrating drum, a chirp"),
    ("residue_theorem", "a narrated short: the residue theorem, every motion on a spoken cue"),
    ("riemann_sqrt", "3D: the Riemann surface of sqrt(z), the camera orbits on its own"),
    ("deep_zoom", "a dive into the Mandelbrot set down to 1e-30 (the GPU's favourite)"),
    ("motion_gallery", "the motion verbs: morphing equations, a tracer, a counter, a warped grid"),
    ("basics_3d", "3D basics: axes, a helix, a surface, a vector field"),
    ("residues", "domain coloring of 1/(z^2 + 1) with a glowing contour"),
    ("fourier_linalg", "Fourier series and linear algebra"),
    ("heat_dihedral", "the heat equation and the dihedral group"),
    ("room_acoustics", "a room, its echoes and the pressure wave"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("only", nargs="*", help="example names (default: all)")
    ap.add_argument("--preset", default="final", choices=["preview", "final", "4k"])
    args = ap.parse_args(argv)
    names = [n for n, _ in SCENES]
    for n in args.only:
        if n not in names:
            ap.error(f"no showcase scene {n!r}; choose from {', '.join(names)}")
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, about in SCENES:
        if args.only and name not in args.only:
            continue
        out = OUT / f"{name}.mp4"
        print(f"{name}: {about}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "phasor", "render",
                            str(ROOT / "examples" / f"{name}.py"), "-o", str(out),
                            "--preset", args.preset, "--quiet"], cwd=ROOT)  # fmt: skip
        if r.returncode == 0:
            print(f"  -> {out.relative_to(ROOT)} ({time.time() - t0:.0f} s)", flush=True)
        else:
            print(f"  FAILED (exit {r.returncode})", flush=True)
            failed.append(name)
    print("done" if not failed else f"done; failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
