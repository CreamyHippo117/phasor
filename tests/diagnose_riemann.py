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
    by_index = {o.index: o.id for o in e.state_at(0.0).objects}

    def in_front(st, i: str) -> str:
        """Which objects cover the middle of ``i`` in the id pass (what can hide it)."""
        obj = st[i]
        idb = r.id_buffer(st, (640, 360), force=frozenset({i}))
        ((px, py),) = st.camera_for(obj).to_screen([obj.impl.world_bounds(st).center], 640, 360)
        x, y = int(px), int(py)
        patch = idb[max(y - 4, 0) : y + 5, max(x - 4, 0) : x + 5].ravel()
        names = [
            by_index.get(int(k), "?") for k in np.unique(patch) if k and by_index.get(int(k)) != i
        ]
        return ",".join(names) or "-"

    for t in np.round(np.arange(0.0, e.duration + 0.01, 0.5), 1):
        st = e.state_at(float(t))
        eye = st.camera.eye
        shares = " ".join(
            f"{i}={hidden_share(st, i, oracle):.2f}" for i in ("branch", "upper", "lower")
        )
        front = " ".join(f"{i}<{in_front(st, i)}>" for i in ("branch", "lower"))
        print(f"t={t:4.1f} eye=({eye[0]:+.2f}, {eye[1]:+.2f}, {eye[2]:+.2f}) | hidden {shares}"
              f" | in front: {front}", flush=True)  # fmt: skip
    for t in (19.1, 20.8, 21.0):
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
