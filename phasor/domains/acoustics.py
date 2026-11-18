"""Room acoustics (SPEC "Domain modules: acoustics"): objects for the reverb story.

    from phasor import ac

    s.add(ac.Room(id="room", size=(8, 5, 3), material="plaster"))
    s.add(ac.Source(id="src", pos=(2, 2.5, 1.5)))
    s.add(ac.Listener(id="mic", pos=(6, 2, 1.5)))
    s.add(ac.Wavefront(id="wave", center=(2, 2.5, 1.5), radius=expr("343 * t")))
    s.add(ac.ImageSources(id="ghosts", size=(8, 5, 3), source=(2, 2.5, 1.5), order=2))
    s.add(ac.ImpulseResponse(id="ir", size=(8, 5, 3), source=(2, 2.5, 1.5), listener=(6, 2, 1.5),
                             layer="overlay", pos=(0.2, 0.62), extent=(0.8, 0.3)))
    s.add(ac.PressureSlice(id="field", size=(8, 5, 3), source=(2, 2.5, 1.5), time=expr("t")))

Everything is a pure function of its parameters (the numerics live in ``ac_math`` and the
pressure field is a checkpointed ``Simulation``), so the same frame always renders the same.
Walls are glass by default: translucent, so they never hide the source or the listener.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.simulation import runner_for
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import ac_math
from phasor.domains.strokes import circle, disc, polyline, with_alpha

MAX_SHOWN = 800  # image sources drawn at most
CHECKPOINT_EVERY = 50  # FDTD steps between cached checkpoints


def _vec3_positive(v):
    if min(v) <= 0:
        raise ValueError("size must be positive in every direction")
    return v


def _material(material: str, absorption: float) -> float:
    return ac_math.absorption_of(material, absorption)


# ---------------------------------------------------------------- cached numerics


@functools.lru_cache(maxsize=64)
def _arrivals(size, source, listener, order, alpha):
    return ac_math.arrivals(size, source, listener, order, alpha)


@functools.lru_cache(maxsize=32)
def _decay(size, source, listener, order, alpha, fs, length):
    arr = _arrivals(size, source, listener, order, alpha)
    ir = ac_math.impulse_response(arr, fs, length)
    try:
        return ac_math.rt60_fit(ir, fs)
    except ValueError:
        return None


# ---------------------------------------------------------------- Room


@register
class Room(ObjectBase):
    """A shoebox room with its corner at ``origin``: translucent walls and bright edges."""

    type: ClassVar[str] = "room"
    Style: ClassVar = {
        "color": ("emphasis", "muted"),
        "stroke": ("stroke", "hairline"),
        "material": ("surface", "glass"),
    }
    Presets: ClassVar = {
        "glass": {},
        "wireframe": {"style": {"material": "wire"}},
        "solid": {"style": {"material": "solid"}},
    }

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        origin: Vec3 = (0.0, 0.0, 0.0)
        material: str = "plaster"  # one of ``ac_math.MATERIALS``
        absorption: float = -1.0  # 0-1; negative: take it from ``material``

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

        @field_validator("material")
        @classmethod
        def _mat(cls, v):
            if v not in ac_math.MATERIALS:
                raise ValueError(
                    f"unknown material {v!r}; materials are {sorted(ac_math.MATERIALS)}"
                )
            return v

    def _lo_hi(self, state: SceneState):
        p = self.me(state).params
        lo = np.asarray(p.origin, dtype=np.float64)
        return lo, lo + np.asarray(p.size, dtype=np.float64)

    def world_bounds(self, state: SceneState) -> Bounds:
        lo, hi = self._lo_hi(state)
        return Bounds(tuple(lo), tuple(hi))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        lo, hi = self._lo_hi(state)
        return {"center": (lo + hi) / 2, "corners": Bounds(tuple(lo), tuple(hi)).corners()}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        alpha = _material(p.material, p.absorption)
        vol, area = ac_math.room_stats(p.size)
        return {
            "kind": "room", "size": list(p.size), "volume": round(vol, 3),
            "surface_area": round(area, 3), "material": p.material,
            "absorption": round(alpha, 4),
            "rt60_sabine": round(ac_math.sabine_rt60(p.size, alpha), 4),
            "rt60_eyring": round(ac_math.eyring_rt60(p.size, alpha), 4),
        }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        lo, hi = self._lo_hi(state)
        cam = state.camera_for(me)
        color = me.style["color"]
        faces = []  # (4 corners counter-clockwise, outward normal)
        for ax in range(3):
            a, b = [i for i in range(3) if i != ax]
            for side, n in ((lo, -1.0), (hi, 1.0)):
                quad = np.zeros((4, 3))
                quad[:, ax] = side[ax]
                for k, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                    quad[k, a] = (lo, hi)[u][a]
                    quad[k, b] = (lo, hi)[v][b]
                normal = np.zeros(3)
                normal[ax] = n
                faces.append((quad, normal))
        positions = np.vstack([q for q, _ in faces])
        normals = np.vstack([np.tile(n, (4, 1)) for _, n in faces])
        k = 4 * np.arange(6)
        indices = np.vstack([np.c_[k, k + 1, k + 2], np.c_[k, k + 2, k + 3]])
        rgba = np.tile(np.array(color), (len(positions), 1))
        ctx.surface(positions, indices, normals=normals, colors=rgba,
                    uv=np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (6, 1)).astype(float),
                    material=me.roles["material"],
                    opacity=me.params.opacity * float(me.style["material"]))  # fmt: skip
        corners = {(i, j, k): np.array([(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]])
                   for i in (0, 1) for j in (0, 1) for k in (0, 1)}  # fmt: skip
        edge_color = with_alpha(color, me.params.opacity)
        for (i, j, k0), a in corners.items():
            for axis in range(3):
                idx = [i, j, k0]
                if idx[axis] == 0:
                    idx[axis] = 1
                    b = corners[tuple(idx)]
                    mid = (a + b) / 2
                    w = me.style["stroke"] * float(cam.units_per_height(mid))
                    polyline(ctx, cam, [a, b], w, color=edge_color, glow=1.0)


# ---------------------------------------------------------------- Source and Listener


class _Marker(ObjectBase):
    """A small billboard marker at ``pos``."""

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        return Bounds(tuple(c - p.r), tuple(c + p.r))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}


@register
class Source(_Marker):
    """A sound source: a glowing point."""

    type: ClassVar[str] = "source"
    Style: ClassVar = {"color": ("emphasis", "highlight"), "glow": ("glow", "soft")}

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(0.12, gt=0)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "point", "role": "source", "position": list(self.me(state).params.pos)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        color = with_alpha(me.style["color"], me.params.opacity)
        disc(
            ctx,
            state.camera_for(me),
            me.params.pos,
            me.params.r,
            color=color,
            glow=me.style["glow"],
        )


@register
class Listener(_Marker):
    """A listener (microphone): a ring with a dot."""

    type: ClassVar[str] = "listener"
    Style: ClassVar = {"color": ("emphasis", "secondary"), "glow": ("glow", "none"),
                       "stroke": ("stroke", "regular")}  # fmt: skip

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(0.16, gt=0)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "point", "role": "listener", "position": list(self.me(state).params.pos)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        color = with_alpha(me.style["color"], me.params.opacity)
        w = me.style["stroke"] * float(cam.units_per_height(np.asarray(me.params.pos)))
        circle(ctx, cam, me.params.pos, me.params.r, w, color=color, glow=me.style["glow"])
        disc(ctx, cam, me.params.pos, me.params.r * 0.3, color=color, glow=me.style["glow"])


# ---------------------------------------------------------------- Wavefront


@register
class Wavefront(ObjectBase):
    """An expanding spherical wavefront, drawn as a lat-long wire sphere. Its ``radius``
    follows the speed of sound if you give it ``expr("343 * (t - 0.5)")``; before the emission
    time that is negative, and nothing is drawn."""

    type: ClassVar[str] = "wavefront"
    Style: ClassVar = {"color": ("emphasis", "secondary"), "stroke": ("stroke", "hairline"),
                       "glow": ("glow", "soft")}  # fmt: skip

    class Params(ParamsBase):
        center: Vec3 = (0.0, 0.0, 0.0)
        radius: float = 0.0  # a negative radius means "not emitted yet": nothing is drawn
        rings: int = Field(6, ge=2, le=24)  # latitude rings; twice as many meridians

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.center, dtype=np.float64)
        r = max(p.radius, 1e-6)
        return Bounds(tuple(c - r), tuple(c + r))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.center, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "wavefront", "radius": round(max(p.radius, 0.0), 4),
                "center": list(p.center), "time_to_reach_center": 0.0}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if p.radius <= 1e-6:
            return
        cam = state.camera_for(me)
        c = np.asarray(p.center, dtype=np.float64)
        color = with_alpha(me.style["color"], me.params.opacity)
        a = np.linspace(0, 2 * np.pi, 49)
        for k in range(1, p.rings):  # latitude rings
            th = np.pi * k / p.rings
            ring = c + p.radius * np.stack(
                [np.sin(th) * np.cos(a), np.sin(th) * np.sin(a), np.full_like(a, np.cos(th))],
                axis=1,
            )
            w = me.style["stroke"] * float(cam.units_per_height(c))
            polyline(ctx, cam, ring, w, color=color, glow=me.style["glow"])
        t = np.linspace(0, np.pi, 25)
        for k in range(2 * p.rings):  # meridians
            ph = 2 * np.pi * k / (2 * p.rings)
            mer = c + p.radius * np.stack(
                [np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)], axis=1
            )
            polyline(
                ctx,
                cam,
                mer,
                me.style["stroke"] * float(cam.units_per_height(c)),
                color=color,
                glow=me.style["glow"],
            )


# ---------------------------------------------------------------- Image sources


@register
class ImageSources(ObjectBase):
    """Ghost speakers behind the walls: the image-source method made visible. Brightness
    falls with each reflection."""

    type: ClassVar[str] = "image_sources"
    Style: ClassVar = {"color": ("emphasis", "muted"), "glow": ("glow", "none")}

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        source: Vec3 = (2.0, 2.5, 1.5)
        order: int = Field(1, ge=0, le=6)
        material: str = "plaster"
        absorption: float = -1.0
        r: float = Field(0.09, gt=0)

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

    def _images(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return _images(tuple(p.size), tuple(p.source), p.order, _material(p.material, p.absorption))

    def world_bounds(self, state: SceneState) -> Bounds:
        im = self._images(state)
        return Bounds.of_points(im["pos"], pad=self.me(state).params.r)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"points": self._images(state)["pos"][1:], "source": self._images(state)["pos"][:1]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        im = self._images(state)
        p = self.me(state).params
        return {"kind": "image_sources", "count": int(len(im["pos"]) - 1), "order": p.order,
                "weakest_gain": round(float(im["gain"].min()), 4)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        im = self._images(state)
        base = me.style["color"]
        for pos, gain in list(zip(im["pos"][1:], im["gain"][1:], strict=True))[:MAX_SHOWN]:
            faded = with_alpha(base, me.params.opacity * float(gain))
            disc(ctx, cam, pos, me.params.r, color=faded, glow=me.style["glow"])


@functools.lru_cache(maxsize=32)
def _images(size, source, order, alpha):
    return ac_math.image_sources(size, source, order, alpha)


# ---------------------------------------------------------------- Impulse response


@register
class ImpulseResponse(ObjectBase):
    """The room's impulse response as a plot: a spike for each arrival at the listener (the
    direct sound in the accent color), and optionally the Schroeder decay with its RT60 fit.

    A panel: put it on the overlay (``layer="overlay"``, ``pos`` the lower-left corner and
    ``extent`` the (width, height), in frame heights). ``reveal`` (0-1) draws the first share
    of ``duration`` seconds, so you can animate arrivals dropping in; ``show_rt60`` (0-1) fades
    in the decay curve and its fitted line.
    """

    type: ClassVar[str] = "impulse_response"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        source: Vec3 = (2.0, 2.5, 1.5)
        listener: Vec3 = (6.0, 2.0, 1.5)
        order: int = Field(12, ge=0, le=40)
        material: str = "plaster"
        absorption: float = -1.0
        pos: Vec3 = (0.1, 0.1, 0.0)  # lower-left corner of the panel
        extent: tuple[float, float] = (0.8, 0.3)
        duration: float = Field(0.4, gt=0)  # seconds shown across the panel
        reveal: float = Field(1.0, ge=0, le=1)
        show_rt60: float = Field(0.0, ge=0, le=1)

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

    FS = 2000  # Hz, for the decay curve
    DECAY_LENGTH = 1.5  # seconds of IR used for the RT60 fit

    def _alpha(self, state):
        p = self.me(state).params
        return _material(p.material, p.absorption)

    def _arr(self, state):
        p = self.me(state).params
        return _arrivals(tuple(p.size), tuple(p.source), tuple(p.listener), p.order,
                         self._alpha(state))  # fmt: skip

    def _fit(self, state):
        p = self.me(state).params
        return _decay(tuple(p.size), tuple(p.source), tuple(p.listener), p.order,
                      self._alpha(state), self.FS, self.DECAY_LENGTH)  # fmt: skip

    def _ydir(self, state: SceneState) -> float:
        """+1 where screen-up is +y (a world layer), -1 on the overlay (y runs down): the panel
        always grows upward from its lower-left corner."""
        return 1.0 if state.camera_for(self.me(state)).up[1] >= 0 else -1.0

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        y2 = y + self._ydir(state) * p.extent[1]
        return Bounds((x, min(y, y2), z), (x + p.extent[0], max(y, y2), z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        arr = self._arr(state)
        p = self.me(state).params
        x, y, z = p.pos
        w, h = p.extent
        h = h * self._ydir(state)
        shown = arr["time"] <= p.duration
        xs = x + arr["time"][shown] / p.duration * w
        top = np.minimum(arr["amplitude"][shown] / arr["amplitude"][0], 1.0) * h
        return {"spikes": np.stack([xs, y + top, np.full(len(xs), z)], axis=1)[:200]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        arr = self._arr(state)
        fit = self._fit(state)
        p = self.me(state).params
        out: dict[str, Any] = {
            "kind": "impulse_response", "arrivals": len(arr["time"]),
            "first_arrival_ms": round(float(arr["time"][0]) * 1000, 3),
            "direct_distance": round(float(arr["distance"][0]), 4), "order": p.order,
            "rt60": None if fit is None else round(fit["rt60"], 4),
        }  # fmt: skip
        out["rt60_theory_eyring"] = round(ac_math.eyring_rt60(p.size, self._alpha(state)), 4)
        # Image sources up to ``order`` reflections describe the first order * (shortest wall) / c
        # seconds completely; a decay fitted beyond that is cut short, not the room's.
        complete = p.order * min(p.size) / ac_math.C
        out["complete_until"] = round(complete, 4)
        out["rt60_reliable"] = bool(fit is not None and fit["rt60"] <= complete)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        x, y, z = p.pos
        w, h = p.extent
        h = h * self._ydir(state)
        width = me.style["stroke"] * float(cam.units_per_height())
        base = with_alpha(me.style["color"], p.opacity)
        accent = with_alpha(me.style["accent"], p.opacity)
        polyline(ctx, cam, [(x, y, z), (x + w, y, z)], width, color=with_alpha(base, 0.6))  # axis
        arr = self._arr(state)
        shown = (arr["time"] <= p.duration * p.reveal) & (arr["time"] <= p.duration)
        a0 = float(arr["amplitude"][0])
        for k in np.nonzero(shown)[0][:400]:
            sx = x + arr["time"][k] / p.duration * w
            top = min(float(arr["amplitude"][k]) / a0, 1.0) * h
            polyline(ctx, cam, [(sx, y, z), (sx, y + top, z)], width,
                     color=accent if k == 0 else base, glow=me.style["glow"])  # fmt: skip
        if p.show_rt60 > 0:
            fit = self._fit(state)
            if fit is not None:
                tt = np.linspace(0, p.duration, 120)
                curve = np.interp(tt, fit["t"], fit["curve"])  # dB, 0 down to -60+
                ys = y + np.clip(1.0 + curve / 60.0, 0.0, 1.0) * h
                pts = np.stack([x + tt / p.duration * w, ys, np.full_like(tt, z)], axis=1)
                polyline(ctx, cam, pts, width, color=with_alpha(accent, p.show_rt60 * 0.9))
                line = np.interp(tt, fit["t"], fit["fit"])
                ys = y + np.clip(1.0 + line / 60.0, 0.0, 1.0) * h
                fitted = np.stack([x + tt / p.duration * w, ys, np.full_like(tt, z)], axis=1)
                polyline(ctx, cam, fitted, width * 0.8, color=with_alpha(base, p.show_rt60))


# ---------------------------------------------------------------- Pressure slice


@functools.lru_cache(maxsize=8)
def _fdtd(size, source, dx, alpha, frequency, seed):
    return ac_math.RoomFDTD(size, source, dx=dx, alpha=alpha, frequency=frequency, seed=seed)


@register
class PressureSlice(ObjectBase):
    """A horizontal slice of the true pressure field (FDTD) at height ``z``, shown as a plane:
    the source's color for compression, the other for rarefaction. Time is ``time`` (seconds,
    usually ``expr("t")``); the field is a checkpointed simulation, so scrubbing is cheap."""

    type: ClassVar[str] = "pressure_slice"
    Style: ClassVar = {"positive": ("emphasis", "highlight"), "negative": ("emphasis", "secondary")}

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        origin: Vec3 = (0.0, 0.0, 0.0)
        source: Vec3 = (2.0, 2.5, 1.5)
        z: float = 1.5
        time: float = Field(0.0, ge=0)
        dx: float = Field(0.2, gt=0.02)
        material: str = "plaster"
        absorption: float = -1.0
        frequency: float = Field(200.0, gt=1)
        gain: float = Field(1.0, gt=0)  # how strongly pressure turns into color
        seed: int = 0

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

    def _sim(self, state):
        p = self.me(state).params
        # source is given in room coordinates; the simulation lives in them too
        return _fdtd(tuple(p.size), tuple(p.source), p.dx,
                     _material(p.material, p.absorption), p.frequency, p.seed)  # fmt: skip

    def _slab(self, state):
        sim = self._sim(state)
        p = self.me(state).params
        runner = runner_for(sim, every=CHECKPOINT_EVERY)
        return sim.slice_z(runner.state_at(p.time), p.z)

    def _rect(self, state):
        p = self.me(state).params
        o = np.asarray(p.origin, dtype=np.float64)
        return o + np.array([0, 0, p.z]), np.array([p.size[0], 0.0, 0.0]), np.array(
            [0.0, p.size[1], 0.0])  # fmt: skip

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u = self._rect(state)
        return Bounds.of_points([c, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {
            "source": np.asarray(p.source, dtype=np.float64) + np.asarray(p.origin),
            "center": self._rect(state)[0] + (self._rect(state)[1] + self._rect(state)[2]) / 2,
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        slab = self._slab(state)
        p = self.me(state).params
        return {"kind": "pressure_slice", "backdrop": True, "time": round(p.time, 6),
                "peak_pressure": round(float(np.abs(slab).max()), 6),
                "grid": list(slab.shape), "z": p.z}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, right, up = self._rect(state)
        slab = self._slab(state)
        pos, neg = np.asarray(me.style["positive"][:3]), np.asarray(me.style["negative"][:3])
        bg = np.asarray(state.background[:3])
        nx, ny = slab.shape
        sx, sy = p.size[0], p.size[1]

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            half_w = height * pw / ph / 2
            u = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w  # along right (m)
            v = center[1] + half_w * 0 + (0.5 - (np.arange(ph) + 0.5) / ph) * height  # row 0 = top
            gx = np.clip(u / sx * nx - 0.5, 0, nx - 1)
            gy = np.clip(v / sy * ny - 0.5, 0, ny - 1)
            x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
            x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
            fx, fy = (gx - x0)[None, :], (gy - y0)[:, None]
            val = (slab[x0][:, y0].T * (1 - fx) * (1 - fy) + slab[x1][:, y0].T * fx * (1 - fy)
                   + slab[x0][:, y1].T * (1 - fx) * fy + slab[x1][:, y1].T * fx * fy)  # fmt: skip
            level = np.tanh(p.gain * 8.0 * val)  # -1..1
            tint = np.where(level[..., None] >= 0, pos, neg)
            rgb = bg + (tint - bg) * np.abs(level)[..., None]
            out = np.ones((ph, pw, 4), dtype=np.float32)
            out[..., :3] = rgb
            return out

        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity)
