"""Why does riemann_sqrt see its branch point hidden on some machines? (TESTING.md B1.)

    uv run python tests/diagnose_riemann.py

Prints the OpenGL renderer, the solved camera and how hidden each goal object is from 18 to
22 s, and saves the picture and the depth buffer at 19.1 s and 20.8 s to examples/out/diag/.
On Linux (Mesa llvmpipe) nothing is hidden at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phasor import load
from phasor.layout.bake import hidden_share
from phasor.render.output import save_png
from phasor.render.passes import default_renderer

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    r = default_renderer()
    print("renderer:", r.on_gl_info("GL_RENDERER"), "|", r.on_gl_info("GL_VERSION"), flush=True)
    print("loading the scene (a few minutes: the camera solver runs)...", flush=True)
    e = load(ROOT / "examples" / "riemann_sqrt.py")
    oracle = e._oracle()
    out = ROOT / "examples" / "out" / "diag"
    out.mkdir(parents=True, exist_ok=True)
    for t in np.round(np.arange(18.0, 22.01, 0.2), 1):
        st = e.state_at(float(t))
        eye = st.camera.eye
        shares = " ".join(
            f"{i}={hidden_share(st, i, oracle):.2f}" for i in ("branch", "upper", "lower")
        )
        print(f"t={t:4.1f} eye=({eye[0]:+.2f}, {eye[1]:+.2f}, {eye[2]:+.2f}) | hidden {shares}",
              flush=True)  # fmt: skip
    for t in (19.1, 20.8):
        st = e.state_at(t)
        save_png(out / f"frame_{t}.png", e.still(t, (640, 360)))
        z = r.depth_buffer(st, (640, 360))
        finite = np.isfinite(z)
        print(f"t={t}: depth min {z[finite].min():.3f} max {z[finite].max():.3f}, "
              f"{(~finite).mean():.0%} empty", flush=True)  # fmt: skip
        img = np.zeros((*z.shape, 3), np.uint8)
        if finite.any():
            lo, hi = z[finite].min(), z[finite].max()
            img[finite] = (255 * (1 - (z[finite] - lo) / max(hi - lo, 1e-9)))[:, None]
        save_png(out / f"depth_{t}.png", img)
    print("saved pictures to", out)
