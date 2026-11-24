"""What ``Image`` and ``Video`` share (Milestone 24): a flat rectangle showing frames from a
source (``media_math``), placed, sized, fitted, timed and framed the same way.

A picture sits in a box ``width`` x ``height`` world units centred on ``pos`` (``height`` 0:
the picture's own shape), or, with ``fill``, in the camera's whole view (a full-frame
background; the 2D camera only: under a perspective camera ``fill`` is ignored). ``fit``
says how it sits in its box. A sequence or a clip plays from ``start`` (scene seconds), the
part ``trim`` keeps, at ``speed``, looping or holding its last frame. ``backdrop`` (always
with ``fill``) lets labels sit on it: layout does not avoid it and lint ``overlap`` does not
count it; ``contrast`` judges text over it by the rendered picture, or, without OpenGL, by
the mean colour of the frame shown.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import media_math as mm
from phasor.domains.strokes import polyline

MAX_SIZE = 1e9  # world units: ``width`` and ``height``


class PictureParams(ParamsBase):
    pos: Vec3 = (0.0, 0.0, 0.0)  # centre
    width: float = Field(4.0, gt=0, le=MAX_SIZE)
    height: float = Field(0.0, ge=0, le=MAX_SIZE)  # 0: from the picture's aspect ratio
    frame: bool = False  # an outline of light round the picture
    fit: Literal["stretch", "contain", "cover", "none"] = "stretch"  # how it sits in its box
    fill: bool = False  # the box is the camera's whole view (2D): a full-frame background
    backdrop: bool = False  # labels may sit on it (always so with ``fill``)
    start: float = Field(0.0, ge=0, le=mm.MAX_START)  # scene seconds where it begins to play
    trim: tuple[float, float] = (0.0, 0.0)  # (in, out) seconds inside it; out 0: its end
    speed: float = Field(1.0, gt=0, le=mm.MAX_SPEED)
    loop: bool = False  # past its end: start again (off: hold the last frame)

    @field_validator("trim")
    @classmethod
    def _trim(cls, v: tuple[float, float]) -> tuple[float, float]:
        a, b = (float(x) for x in v)
        if not (np.isfinite(a) and np.isfinite(b) and 0 <= a <= mm.MAX_START
                and 0 <= b <= mm.MAX_START):  # fmt: skip
            raise ValueError(f"trim {v!r}: both ends are seconds from 0 to {mm.MAX_START:g}")
        if b and b <= a:
            raise ValueError(f"trim {v!r}: the out point must come after the in point (or be "
                             "0, the end)")  # fmt: skip
        return a, b


def extension(value: str) -> str:
    return value.lower().rsplit(".", 1)[-1] if "." in value else ""


class Picture(ObjectBase):
    """A rectangle of frames from a file (``Image``, ``Video``); not a type of its own."""

    Style: ClassVar = {"glow": ("glow", "none"), "frame": ("emphasis", "muted"),
                       "stroke": ("stroke", "hairline")}  # fmt: skip
    Presets: ClassVar = {
        "plain": {"params": {"frame": False}, "style": {"glow": "none"}},
        "framed": {"params": {"frame": True}, "style": {"frame": "primary"}},
        "luminous": {"params": {"frame": True}, "style": {"glow": "soft", "frame": "highlight"}},
    }
    kind: ClassVar[str] = "picture"

    def source(self, state: SceneState) -> mm.Source:
        """The frames this object shows (``media_math.open_source``)."""
        raise NotImplementedError

    def _probe(self, state: SceneState) -> tuple[mm.Source, mm.Probe]:
        src = self.source(state)
        return src, src.probe()

    # ------------------------------------------------------------ where

    def _fill_view(self, state: SceneState) -> tuple | None:
        """The camera's view at the picture's depth, when it fills it: (centre, right, up,
        width, height); None when it does not (``fill`` off, or a perspective camera)."""
        me = self.me(state)
        if not me.params.fill:
            return None
        cam = state.camera_for(me)
        if cam.perspective:
            return None
        h = float(cam.height)
        w = h * state.aspect
        eye, fwd = np.asarray(cam.eye, dtype=np.float64), np.asarray(cam.forward)
        z = float(me.params.pos[2])
        center = eye + fwd * ((z - eye[2]) / fwd[2]) if abs(fwd[2]) > 1e-12 else eye
        return center, np.asarray(cam.right, dtype=np.float64), np.asarray(cam.up), w, h

    def _place(self, state: SceneState, probe: mm.Probe):
        """corner, right, up (world vectors of the quad drawn) and the part of the picture
        it shows (``fit_box``)."""
        p = self.me(state).params
        view = self._fill_view(state)
        if view is not None:
            center, rv, uv, bw, bh = view
        else:
            center, rv, uv = np.asarray(p.pos, dtype=np.float64), np.array([1.0, 0, 0]), None
            bw = p.width
            if p.height > 0:
                bh = p.height
            elif not probe.ok:  # missing: a 16:9 placeholder rectangle (nothing drawn in it)
                bh = p.width * 9 / 16
            else:
                bh = p.width * probe.height / probe.width
        if probe.ok:
            native = 0.0
            if p.fit == "none":
                native = float(state.camera_for(self.me(state)).units_per_height(center))
                native /= mm.NATIVE_ROWS
            w, h, crop = mm.fit_box(p.fit, (bw, bh), (probe.width, probe.height), native)
        else:
            w, h, crop = bw, bh, mm.FULL
        if uv is None:  # placed by pos: axis-aligned, as Image always was
            corner = center - np.array([w / 2, h / 2, 0.0])
            return corner, np.array([w, 0.0, 0.0]), np.array([0.0, h, 0.0]), crop
        right, up = rv * w, uv * h
        return center - right / 2 - up / 2, right, up, crop

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u, _ = self._place(state, self._probe(state)[1])
        return Bounds.of_points([c, c + r, c + u, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        c, r, u, _ = self._place(state, self._probe(state)[1])
        return {"center": c + (r + u) / 2, "corners": np.array([c, c + r, c + r + u, c + u])}

    # ------------------------------------------------------------ when

    def playing(self, probe: mm.Probe) -> bool:
        """Whether this shows more than one frame over time (a sequence or a clip)."""
        return probe.frames > 1 or self.kind == "video"

    def clip(self, state: SceneState, probe: mm.Probe) -> tuple[int, float, bool]:
        """The frame shown at ``state.t``, the clip time it stands for and whether the clip
        has ended (holding its last frame)."""
        p = self.me(state).params
        at, ended = mm.clip_time(state.t, start=p.start, trim=p.trim, speed=p.speed,
                                 loop=p.loop, seconds=probe.seconds)  # fmt: skip
        return mm.frame_index(probe.times, at, ended), at, ended

    # ------------------------------------------------------------ what

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        probe = self._probe(state)[1]
        _, r, u, _ = self._place(state, probe)
        size = [float(np.linalg.norm(r)), float(np.linalg.norm(u))] if p.fill else \
            [float(r[0]), float(u[1])]  # fmt: skip
        out: dict[str, Any] = {"kind": self.kind, "path": p.path,
                               "pixels": [probe.width, probe.height], "missing": not probe.ok,
                               "size": [round(v, 6) for v in size]}  # fmt: skip
        if probe.problem and probe.problem != "missing":
            out["unreadable"] = probe.problem  # lint bad_file
            out["file"] = p.path
        if probe.too_big:
            out["too_big"] = probe.too_big  # lint video_too_big
            out["file"] = p.path
        if p.backdrop or p.fill:
            out["backdrop"] = True  # labels may sit on it
        if p.fill:
            out["fill"] = True
        if p.fit != "stretch":
            out["fit"] = p.fit
        if probe.ok and self.playing(probe):
            index, at, ended = self.clip(state, probe)
            a, b = mm.clip_range(p.trim, probe.seconds)
            out.update({"frames": probe.frames, "fps": round(probe.fps, 6),
                        "seconds": round(probe.seconds, 6), "frame": index,
                        "clip_time": round(at, 6), "looping": p.loop,
                        "trimmed": a > 0 or b < probe.seconds, "ended": ended})  # fmt: skip
        return out

    def average_color(self, state: SceneState) -> tuple[float, float, float] | None:
        """The mean linear colour of the frame shown, over the background: what text on it
        is judged against without a rendered picture (lint ``contrast``, ``passes=None``).
        None when nothing is drawn."""
        src, probe = self._probe(state)
        if not probe.ok:
            return None
        frame = src.frame(self.clip(state, probe)[0])
        if frame is None:
            return None
        return mm.average(frame, np.asarray(state.background), opaque=probe.opaque)

    # ------------------------------------------------------------ drawing

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        src, probe = self._probe(state)
        corner, right, up, crop = self._place(state, probe)
        if not probe.ok:
            return  # missing or unreadable: lint says so; nothing to draw
        bg = np.asarray(state.background[:3], dtype=np.float32)
        glow = float(me.style.get("glow", 1.0))
        if p.fill:
            quad = (float(np.linalg.norm(right)), float(np.linalg.norm(up)))
        else:
            quad = (float(right[0]), float(up[1]))
        index = self.clip(state, probe)[0] if self.playing(probe) else 0

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            frame = src.frame(index)  # decoded here, when the renderer needs the pixels
            if frame is None:  # a frame a damaged file cannot give: the background
                out = np.empty((ph, pw, 4), dtype=np.float32)
                out[..., :3], out[..., 3] = bg, 0.0
                return out
            return mm.sample(frame, pw, ph, center, height, quad, crop, bg,
                             opaque=probe.opaque, glow=glow > 1)  # fmt: skip

        pixels.cache_key = (src.stamp, index)  # type: ignore[attr-defined]
        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity, glow=glow)
        if p.frame and pass_ == "color":
            cam = state.camera_for(me)
            ring = [corner, corner + right, corner + right + up, corner + up, corner]
            w = me.style["stroke"] * float(cam.units_per_height(corner + (right + up) / 2))
            polyline(ctx, cam, np.asarray(ring), w,
                     color=(*me.style["frame"][:3], p.opacity), glow=2.5)  # fmt: skip
