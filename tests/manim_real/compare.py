"""Compare phasor's Manim import with REAL Manim Community, scene by scene (Milestone 20).

Real Manim runs in its own Python (``PHASOR_MANIM_PYTHON``, a venv with ``manim``
installed; see ``pc_tasks/10_manim_real.md``) through ``record_manim.py``: every top-level
mobject is measured at 10 frames per second while the scene plays. The same script is
imported by phasor (``import_manim``) and the document is measured at the same times through
the engine (``state_at``): the union of the world bounds of the phasor objects each mobject
became. Compared: the clock (each ``play``'s duration, its animations and their run times and
rate functions), how many mobjects are on screen, their centers and sizes, colors, and the
camera frame of a ``MovingCameraScene``.

Not run by default: the tests in ``test_manim_real.py`` carry the ``manim_real`` marker and
skip without ``PHASOR_MANIM_PYTHON``. By hand::

    PHASOR_MANIM_PYTHON=/path/to/manim_env/bin/python uv run python -m tests.manim_real.compare

prints the table that goes into ``MILESTONE_20.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SCENES = HERE / "scenes"

# Tolerances (world units of Manim's 8-unit frame). Shapes are drawn from the same geometry,
# so they agree closely; text is set in phasor's own font (Aileron / Computer Modern through
# matplotlib) where Manim uses Pango or LaTeX, so its box differs in size, and so does any
# position computed from it (``next_to`` a text).
POS_TOL = 0.05
SIZE_TOL = 0.05
TEXT_POS_TOL = 0.35
TEXT_SIZE_TOL = 0.45  # relative
TEXT_CLASSES = {"Text", "MathTex", "Tex", "MarkupText", "Title", "Paragraph"}
# Composites drawn by phasor's own diagram kit: its furniture (tips, tick lengths, numbers)
# is phasor's, so only the place is compared.
KIT_CLASSES = {"Axes", "NumberPlane", "NumberLine", "BarChart", "Table", "Matrix", "Code"}
# Mobjects real Manim lists that are not the script's: the camera frame of a
# MovingCameraScene, a ValueTracker (a point), and the Group an AnimationGroup or a
# TransformMatching* holds its parts in while it plays.
INTERNAL = {"ScreenRectangle", "ValueTracker", "Group"}


def manim_python() -> str | None:
    exe = os.environ.get("PHASOR_MANIM_PYTHON")
    return exe if exe and Path(exe).exists() else None


def record_real(script: Path, scene: str | None = None) -> dict:
    """Real Manim's record of ``script`` (a subprocess in the Manim venv)."""
    exe = manim_python()
    if exe is None:
        raise RuntimeError("set PHASOR_MANIM_PYTHON to a Python with manim installed")
    from phasor.core.proc import NO_WINDOW

    cmd = [exe, "-W", "ignore", str(HERE / "record_manim.py"), str(script)]
    done = subprocess.run(cmd + ([scene] if scene else []), capture_output=True, text=True,
                          encoding="utf-8", timeout=600, creationflags=NO_WINDOW,
                          cwd=str(script.parent))  # fmt: skip
    if done.returncode != 0:
        raise RuntimeError(f"real Manim failed on {script.name}:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout)


def _measure(state, cls: str, ids: list[str]) -> dict | None:
    lo, hi, opacity = None, None, 0.0
    for oid in ids:
        if oid not in state:
            continue
        o = state[oid]
        opacity = max(opacity, float(getattr(o.params, "opacity", 1.0)))
        try:
            b = o.impl.world_bounds(state)
        except Exception:
            continue
        a, c = np.asarray(b.lo), np.asarray(b.hi)
        lo = a if lo is None else np.minimum(lo, a)
        hi = c if hi is None else np.maximum(hi, c)
    if lo is None:
        return None
    return {"cls": cls, "center": ((lo + hi) / 2).tolist(), "width": float(hi[0] - lo[0]),
            "height": float(hi[1] - lo[1]), "opacity": opacity}  # fmt: skip


def record_phasor(script: Path, times: list[tuple[float, bool]], scene: str | None = None) -> dict:
    """phasor's import of ``script``, measured at ``times`` the way ``record_manim`` does
    (each a time, and whether it is the sample just after a play ended)."""
    from phasor.importers import import_manim

    # the plain look: the prism look's living motion moves objects a little (by design)
    os.environ["PHASOR_DEFAULT_LOOK"] = "none"
    r = import_manim(script, scene)
    engine = r.scene.engine
    plays = [{"t0": p["t0"], "duration": p["duration"], "anims": p["anims"]} for p in r.trace]
    samples = []
    for t, end in times:
        if end:
            play = next((p for p in r.trace if abs(p["t0"] + p["duration"] - t) < 1e-6), None)
            listing = play["after"] if play else []
        else:
            play = next((p for p in r.trace
                         if p["t0"] - 1e-9 <= t < p["t0"] + p["duration"] - 1e-9),
                        r.trace[-1] if r.trace else None)  # fmt: skip
            listing = play["mobjects"] if play else []
        state = engine.state_at(t)
        mobs = []
        for m in listing:
            seen = _measure(state, m["cls"], m["ids"])
            if seen is not None:
                mobs.append(seen)
        sample = {"t": t, "mobjects": mobs}
        cam = state.camera
        if cam is not None and play and play.get("frame"):
            values = state.values
            sample["frame"] = {"center": list(values.get("camera.center", (0, 0, 0))),
                               "height": float(values.get("camera.height", 8.0))}  # fmt: skip
        samples.append(sample)
    return {"scene": r.name, "duration": r.duration, "plays": plays, "samples": samples,
            "warnings": r.warnings}  # fmt: skip


@dataclass
class Finding:
    scene: str
    matched: list[str] = field(default_factory=list)
    differed: list[str] = field(default_factory=list)
    worst: dict[str, float] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)  # the worst sample of each measure


def _pair(real: list[dict], ours: list[dict]) -> list[tuple[dict, dict | None]]:
    """Mobjects paired by class and order of appearance."""
    pool: dict[str, list[dict]] = {}
    for m in ours:
        pool.setdefault(m["cls"], []).append(m)
    return [(m, pool.get(m["cls"], []).pop(0) if pool.get(m["cls"]) else None) for m in real]


def _matching_at(plays: list[dict], sample: dict) -> bool:
    """Whether a TransformMatching* plays at ``sample``: Manim then shows the pieces it
    moves in a VGroup of its own, which is not the script's."""
    t = sample["t"]
    for p in plays:
        if p["t0"] - 1e-9 <= t < p["t0"] + p["duration"] - 1e-9 and not sample.get("end"):
            return any(a["cls"].startswith("TransformMatching") for a in p["anims"])
    return False


def compare(real: dict, ours: dict) -> Finding:
    f = Finding(real["scene"])
    if abs(real["duration"] - ours["duration"]) < 1e-6:
        f.matched.append(f"duration {real['duration']:g} s")
    else:
        f.differed.append(f"duration {ours['duration']:g} s, Manim {real['duration']:g} s")
    rp, op = real["plays"], ours["plays"]
    if len(rp) != len(op):
        f.differed.append(f"{len(op)} plays, Manim {len(rp)}")
    timing_ok = True
    for k, (a, b) in enumerate(zip(rp, op, strict=False)):
        ra = [(x["run_time"], x["rate_func"]) for x in a["anims"]]
        oa = [(x["run_time"], x["rate_func"]) for x in b["anims"]]
        if abs(a["duration"] - b["duration"]) > 1e-6 or ra != oa:
            timing_ok = False
            f.differed.append(f"play {k}: {[x['cls'] for x in b['anims']]} {oa} "
                              f"(Manim {[x['cls'] for x in a['anims']]} {ra})")  # fmt: skip
    if timing_ok and len(rp) == len(op):
        f.matched.append(f"{len(rp)} plays: durations, run times, rate functions")
    pos_err = size_err = text_err = 0.0
    missing: set[str] = set()
    counts_ok = True
    frame_err = 0.0
    where: dict[str, str] = {}
    for rs, os_ in zip(real["samples"], ours["samples"], strict=True):
        internal = INTERNAL | ({"VGroup"} if _matching_at(rp, rs) else set())
        real_mobs = [m for m in rs["mobjects"] if m["cls"] not in internal]
        if len(real_mobs) != len(os_["mobjects"]):
            if counts_ok:
                where["count"] = (f"t={rs['t']:g}: Manim {[m['cls'] for m in real_mobs]}, phasor "
                                  f"{[m['cls'] for m in os_['mobjects']]}")  # fmt: skip
            counts_ok = False
        for m, o in _pair(real_mobs, os_["mobjects"]):
            if o is None:
                missing.add(m["cls"])
                continue
            if max(m["width"], m["height"]) < 1e-3 or max(o["width"], o["height"]) < 1e-3:
                continue  # not drawn yet on one side (the first frame of a Create)
            d = float(np.linalg.norm(np.subtract(m["center"][:2], o["center"][:2])))
            at = (f"t={rs['t']:g} {m['cls']}: Manim {m['center'][:2]} {m['width']:.3f}x"
                  f"{m['height']:.3f}, phasor {[round(v, 3) for v in o['center'][:2]]} "
                  f"{o['width']:.3f}x{o['height']:.3f}")  # fmt: skip
            if m["cls"] in TEXT_CLASSES:
                if d > text_err:
                    text_err, where["text"] = d, at
                continue
            if d > pos_err:
                pos_err, where["position"] = d, at
            if m["cls"] in KIT_CLASSES:
                continue
            for k in ("width", "height"):
                if abs(m[k] - o[k]) > size_err:
                    size_err, where["size"] = abs(m[k] - o[k]), at
        if "frame" in rs and "frame" in os_:
            fr, fo = rs["frame"], os_["frame"]
            frame_err = max(frame_err, float(np.linalg.norm(np.subtract(fr["center"][:2],
                            fo["center"][:2]))), abs(fr["height"] - fo["height"]))  # fmt: skip
        elif "frame" in rs:
            frame_err = float("inf")
    f.worst = {"position": pos_err, "size": size_err, "text": text_err, "frame": frame_err}
    f.details = [f"{k}: {v}" for k, v in where.items()]
    if counts_ok and not missing:
        f.matched.append("mobjects on screen at every sample")
    else:
        f.differed.append("mobject count differs at some samples" + (
            f"; never found: {sorted(missing)}" if missing else ""))  # fmt: skip
    for name, err, tol in (("positions", pos_err, POS_TOL), ("sizes", size_err, SIZE_TOL),
                           ("text positions", text_err, TEXT_POS_TOL)):  # fmt: skip
        (f.matched if err <= tol else f.differed).append(f"{name} within {err:.3f}")
    if "frame" in real["samples"][0]:
        (f.matched if frame_err <= POS_TOL else f.differed).append(
            f"camera frame within {frame_err:.3f}"
        )
    real_colors = {m["cls"]: m["color"] for m in real["samples"][-1]["mobjects"]}
    f.matched.append(f"colors at the end: {sorted(set(real_colors.values()) - {None})}")
    return f


def run_scene(script: Path) -> Finding:
    real = record_real(script)
    times = [(s["t"], bool(s.get("end"))) for s in real["samples"]]
    return compare(real, record_phasor(script, times))


def main() -> int:
    print("| scene | matched | differed |\n| --- | --- | --- |")
    for script in sorted(SCENES.glob("*.py")):
        try:
            f = run_scene(script)
        except Exception as e:  # report and go on: one broken scene is one row
            print(f"| {script.stem} | | error: {type(e).__name__}: {str(e)[:200]} |")
            continue
        print(f"| {script.stem} | {'; '.join(f.matched)} | {'; '.join(f.differed) or '-'} |")
        if "-v" in sys.argv:
            for line in f.details:
                print(f"    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
