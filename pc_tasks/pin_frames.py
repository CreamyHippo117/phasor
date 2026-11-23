"""Hash final-quality frames, to prove a speed-up changed no picture (GPU_SPEED.md).

Run from the repo root before and after, then compare the two files:
    uv run python pc_tasks/pin_frames.py before.json prism_primes:6 motion_gallery:6
Leave lit surfaces (3D surfaces) out: they are not bit-repeatable on the RTX card.
"""

import hashlib
import json
import sys
import time

import numpy as np

from phasor import scene

out, *names = sys.argv[1:]
res = {}
for spec in names:
    name, _, n = spec.partition(":")
    n = int(n or 6)
    eng = scene.load(f"examples/{name}.py")
    w, h = eng.document.meta.width, eng.document.meta.height
    fps = eng.document.meta.fps
    d = eng.duration
    ts = [round(d * (k + 0.5) / n, 4) for k in range(n)]
    hs = []
    t0 = time.perf_counter()
    for t in ts:
        f = eng.frame_blurred(t, fps, shutter=0.5, samples=4, size=(w, h), output="srgb8")
        hs.append(hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()[:16])
    res[name] = {"size": [w, h], "times": ts, "hashes": hs}
    print(name, f"{time.perf_counter() - t0:.1f}s", hs, flush=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
