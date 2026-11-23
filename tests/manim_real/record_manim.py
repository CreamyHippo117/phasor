"""Run a Manim scene script with REAL Manim Community and print what it does as JSON.

This file runs in a separate Python that has Manim installed (never in phasor's own
environment): ``compare.py`` starts it as a subprocess with ``PHASOR_MANIM_PYTHON``. Nothing
is written to disk (``dry_run``); frames are rendered small (Manim's play loop renders each
frame), at ``SAMPLE_RATE`` frames per second, and at every frame the top-level mobjects are
measured. Usage::

    <manim python> record_manim.py scene.py [SceneName] > out.json

Output: ``{"scene", "duration", "plays": [{"t0", "duration", "anims": [{"cls", "run_time",
"rate_func"}]}], "samples": [{"t", "mobjects": [{"cls", "center", "width", "height", "color",
"fill_opacity", "stroke_opacity", "stroke_width", "leaves"}]}]}``.
"""

from __future__ import annotations

import json
import runpy
import sys
import tempfile

SAMPLE_RATE = 30  # frames a second of the measured run (phasor runs updaters at 30 too)


def _get(m, name: str) -> float:
    try:
        return float(getattr(m, name)())
    except Exception:  # a plain Mobject (no fill or stroke)
        return 0.0


def _measure(mob) -> dict:
    leaves = [m for m in mob.get_family() if m.has_points()] or [mob]
    fill = max((_get(m, "get_fill_opacity") for m in leaves), default=0.0)
    stroke = max((_get(m, "get_stroke_opacity") for m in leaves), default=0.0)
    width = max((_get(m, "get_stroke_width") for m in leaves), default=0.0)
    try:
        color = mob.get_color().to_hex().upper()
    except Exception:
        color = None
    return {
        "cls": type(mob).__name__,
        "center": [round(float(v), 6) for v in mob.get_center()],
        "width": round(float(mob.width), 6),
        "height": round(float(mob.height), 6),
        "color": color,
        "fill_opacity": round(fill, 6),
        "stroke_opacity": round(stroke, 6),
        "stroke_width": round(width, 6),
        "leaves": len(leaves),
    }


def record(path: str, name: str | None = None) -> dict:
    import manim
    from manim import Scene, tempconfig

    ns = runpy.run_path(path, run_name="manim_real_scene")
    classes = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Scene)
               and v.__module__ == "manim_real_scene"]  # fmt: skip
    cls = next(c for c in classes if name is None or c.__name__ == name)
    out: dict = {"scene": cls.__name__, "plays": [], "samples": [], "manim": manim.__version__}
    clock = {"t0": 0.0}

    def snapshot(scene, t: float, end: bool = False) -> None:
        mobs = [
            m
            for m in scene.mobjects
            if m.get_family() and any(x.has_points() for x in m.get_family())
        ]  # fmt: skip  (not a ValueTracker)
        sample = {"t": round(t, 6), "end": end, "mobjects": [_measure(m) for m in mobs]}
        frame = getattr(scene.camera, "frame", None)  # MovingCameraScene
        if frame is not None:
            sample["frame"] = {"center": [round(float(v), 6) for v in frame.get_center()],
                               "height": round(float(frame.height), 6)}  # fmt: skip
        out["samples"].append(sample)

    orig_play, orig_update = Scene.play, Scene.update_to_time

    def play(self, *a, **k):
        t0 = clock["t0"]
        orig_play(self, *a, **k)
        d = float(self.duration)
        anims = [{"cls": type(x).__name__, "run_time": round(float(x.run_time), 6),
                  "rate_func": getattr(x.rate_func, "__name__", "?")}
                 for x in (self.animations or [])]  # fmt: skip
        out["plays"].append({"t0": round(t0, 6), "duration": round(d, 6), "anims": anims})
        clock["t0"] = t0 + d
        snapshot(self, t0 + d, end=True)  # after the play: its removers are gone

    def update_to_time(self, t):
        orig_update(self, t)
        snapshot(self, clock["t0"] + float(t))

    Scene.play, Scene.update_to_time = play, update_to_time
    media = tempfile.mkdtemp(prefix="manim_real_")
    settings = {"dry_run": True, "frame_rate": SAMPLE_RATE, "pixel_width": 160,
                "pixel_height": 90, "disable_caching": True, "verbosity": "ERROR",
                "progress_bar": "none", "media_dir": media}  # fmt: skip
    try:
        with tempconfig(settings):
            scene = cls()
            scene.render()
    finally:
        Scene.play, Scene.update_to_time = orig_play, orig_update
    out["duration"] = round(clock["t0"], 6)
    return out


if __name__ == "__main__":
    print(json.dumps(record(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)))
