"""Pin final-quality frames, to prove a speed-up changed no picture visibly (GPU_SPEED.md).

Run from the repo root before and after, then compare the two runs:
    uv run python pc_tasks/pin_frames.py before.json prism_primes:6 motion_gallery:6
    uv run python pc_tasks/pin_frames.py after.json prism_primes:6 motion_gallery:6
    uv run python pc_tasks/pin_frames.py --compare before.json after.json
Each run writes the hashes to the .json and the 8-bit frames beside it (.npz). The compare
passes when every value is within 1 code value of before (owner, 2026-11-26: a final may
differ only where the eye cannot see it); it prints how many values differ and by how much.
Leave lit surfaces (3D surfaces) out: they are not bit-repeatable on the RTX card.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

LIMIT = 1  # code values: the most any 8-bit value of a final may move


def pin(out: str, names: list[str]) -> None:
    from phasor import scene

    res, frames = {}, {}
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
        for k, t in enumerate(ts):
            f = np.ascontiguousarray(
                eng.frame_blurred(t, fps, shutter=0.5, samples=4, size=(w, h), output="srgb8")
            )
            hs.append(hashlib.sha256(f.tobytes()).hexdigest()[:16])
            frames[f"{name}:{k}"] = f
        res[name] = {"size": [w, h], "times": ts, "hashes": hs}
        print(name, f"{time.perf_counter() - t0:.1f}s", hs, flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(Path(out).with_suffix(".npz"), **frames)


def compare(before: str, after: str) -> int:
    a, b = np.load(Path(before).with_suffix(".npz")), np.load(Path(after).with_suffix(".npz"))
    worst = 0
    for key in sorted(set(a.files) | set(b.files)):
        if key not in a.files or key not in b.files:
            print(f"{key}: only in one run")
            worst = max(worst, 255)
            continue
        d = np.abs(a[key].astype(np.int16) - b[key].astype(np.int16))
        worst = max(worst, int(d.max()))
        print(f"{key}: max {int(d.max())} code values, {int((d > 0).sum())} of {d.size} "
              f"values differ")  # fmt: skip
    ok = worst <= LIMIT
    print("PASS" if ok else "FAIL", f"(largest difference {worst}, limit {LIMIT})")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    pin(sys.argv[1], sys.argv[2:])
