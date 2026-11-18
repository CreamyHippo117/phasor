"""PDE objects (SPEC "Domain modules: pde"): fields, profiles and surfaces of the solvers in
``pde_math``.

    from phasor import pde

    s.add(pde.Profile(id="rod", kind="heat", time=expr("t / 8"), exact=1.0))
    s.add(pde.Field(id="plate", kind="heat", mode=(0, 0), time=expr("t / 20")))
    s.add(pde.Surface(id="drum", kind="wave", mode=(2, 1), time=expr("t / 4")))

``time`` is simulation time; every frame is a pure function of the parameters, computed by a
checkpointed ``Simulation`` so scrubbing is cheap. A solver step that would be unstable is a
validation error.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field as PField
from pydantic import model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.simulation import runner_for
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import pde_math as pm
from phasor.domains.strokes import polyline, with_alpha

CHECKPOINT_EVERY = 100
MAX_STEPS = 400_000  # a frame further into the simulation than this is refused


@functools.lru_cache(maxsize=32)
def _sim(kind: str, key: tuple):
    a = dict(key)
    if kind == "heat1d":
        return pm.Heat1D(k=a["diffusivity"], length=a["length"], n=a["n"], modes=a["modes"])
    if kind == "wave1d":
        return pm.Wave1D(
            c=a["speed"], length=a["length"], n=a["n"], mode=a["modes"][0] if a["modes"] else 0
        )
    if kind == "advection1d":
        return pm.Advection1D(c=a["speed"], length=a["length"], n=a["n"])
    if kind == "schrodinger1d":
        return pm.Schrodinger1D(length=a["length"], n=a["n"], potential=a["potential"], dt=a["dt"])
    if kind == "heat2d":
        return pm.Heat2D(k=a["diffusivity"], size=a["size"], n=a["n"], mode=a["mode"])
    if kind == "wave2d":
        return pm.Wave2D(c=a["speed"], size=a["size"], n=a["n"], mode=a["mode"])
    if kind == "reaction":
        return pm.GrayScott(f=a["feed"], k=a["kill"], n=a["n"], seed=a["seed"])
    raise ValueError(kind)


def _state(sim, time: float):
    runner = runner_for(sim, every=CHECKPOINT_EVERY)
    if runner.step_of(time) > MAX_STEPS:
        raise ValueError(
            f"time {time:g} is {runner.step_of(time):,} steps in (limit {MAX_STEPS:,})"
        )
    return runner.state_at(time)


# ---------------------------------------------------------------- Profile (1D)


@register
class Profile(ObjectBase):
    """A one-dimensional solution as a curve u(x, t): a heating rod, a plucked or standing
    string, a pulse carried along (advection), or a quantum wave packet's density. The exact
    solution, where one exists, is drawn in a second color (``exact`` is its opacity)."""

    type: ClassVar[str] = "pde_profile"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "exact": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        kind: Literal["heat", "wave", "advection", "schrodinger"] = "heat"
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # lower-left corner of the plot
        extent: tuple[float, float] = (6.0, 2.5)
        length: float = PField(1.0, gt=0)
        n: int = PField(100, ge=8, le=2000)
        diffusivity: float = PField(1.0, gt=0)
        speed: float = PField(1.0, gt=0)
        modes: tuple[float, ...] = (1.0,)  # heat: amplitudes of sin(m pi x / L); wave: (m,)
        potential: Literal["free", "harmonic", "barrier"] = "free"
        dt: float = PField(0.01, gt=0)  # quantum solver step
        exact: float = PField(1.0, ge=0, le=1)
        ymax: float = PField(1.2, gt=0)  # the y range shown is [-ymax, ymax] (0 to ymax: density)

        @model_validator(mode="after")
        def _stable(self):
            _sim(self.kind + "1d", _key(self))  # an unstable step raises here
            return self

    def _key_sim(self, p):
        return _sim(p.kind + "1d", _key(p))

    def _data(self, state: SceneState):
        p = self.me(state).params
        sim = self._key_sim(p)
        s = _state(sim, p.time)
        x = sim.x
        if p.kind == "schrodinger":
            u = sim.density(s)
            ex = None
        else:
            u = s["u"]
            ex = sim.exact(p.time)
        return x, u, ex

    def _xy(self, state, p, x, y, density: bool):
        w, h = p.extent
        sx = p.pos[0] + (x - x[0]) / max(x[-1] - x[0], 1e-12) * w
        ysc = (y / p.ymax) * (h if density else h / 2)
        base = p.pos[1] + (0.0 if density else h / 2)
        return np.stack([sx, base + ysc, np.full_like(sx, p.pos[2])], axis=1)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x, y, z), (x + p.extent[0], y + p.extent[1], z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        x, u, _ = self._data(state)
        return {"curve": self._xy(state, p, x, u, p.kind == "schrodinger")}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sim = self._key_sim(p)
        x, u, ex = self._data(state)
        out: dict[str, Any] = {
            "kind": "pde_profile",
            "equation": p.kind,
            "time": round(p.time, 6),
            "max": round(float(u.max()), 6),
            "min": round(float(u.min()), 6),
            "dt": sim.dt,
            "steps": _runner_steps(sim, p.time),
        }
        if ex is not None:
            out["max_error_vs_exact"] = round(float(np.abs(u - ex).max()), 6)
        if p.kind == "schrodinger":
            out["norm"] = round(float(u.sum() * sim.dx), 9)
        if p.kind == "heat":
            out["total_heat"] = round(float(np.trapezoid(u, x)), 6)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        x, u, ex = self._data(state)
        dens = p.kind == "schrodinger"
        w = me.style["stroke"] * float(cam.units_per_height())
        x0, y0, z0 = p.pos
        axis_y = y0 if dens else y0 + p.extent[1] / 2
        polyline(
            ctx,
            cam,
            [(x0, axis_y, z0), (x0 + p.extent[0], axis_y, z0)],
            w * 0.7,
            color=with_alpha(me.style["exact"], 0.5 * p.opacity),
        )
        if ex is not None and p.exact > 0:
            polyline(
                ctx,
                cam,
                self._xy(state, p, x, ex, dens),
                w * 0.8,
                color=with_alpha(me.style["exact"], p.opacity * p.exact),
            )
        polyline(
            ctx,
            cam,
            self._xy(state, p, x, u, dens),
            w,
            color=with_alpha(me.style["color"], p.opacity),
            glow=me.style["glow"],
        )


def _key(p) -> tuple:
    names = (
        "diffusivity",
        "speed",
        "length",
        "n",
        "modes",
        "potential",
        "dt",
        "size",
        "mode",
        "feed",
        "kill",
        "seed",
    )
    return tuple(sorted((k, getattr(p, k)) for k in names if hasattr(p, k)))


def _runner_steps(sim, t: float) -> int:
    return runner_for(sim, every=CHECKPOINT_EVERY).step_of(t)


# ---------------------------------------------------------------- Field and Surface (2D)


class _Plate(ObjectBase):
    """What the 2D objects share: the solver, its state at ``time`` and a value scale."""

    def _sim_of(self, p):
        kind = {"heat": "heat2d", "wave": "wave2d", "reaction": "reaction"}[p.kind]
        return _sim(kind, _key(p))

    def _grid(self, state: SceneState):
        p = self.me(state).params
        sim = self._sim_of(p)
        s = _state(sim, p.time)
        u = s["v"] if p.kind == "reaction" else s["u"]
        return sim, np.asarray(u, dtype=np.float64)

    def _size(self, p) -> tuple[float, float]:
        return p.size

    def _stats(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sim, u = self._grid(state)
        out = {
            "equation": p.kind,
            "time": round(p.time, 6),
            "max": round(float(u.max()), 6),
            "min": round(float(u.min()), 6),
            "grid": list(u.shape),
            "dt": sim.dt,
            "steps": _runner_steps(sim, p.time),
        }
        ex = sim.exact(p.time) if hasattr(sim, "exact") else None
        if ex is not None:
            out["max_error_vs_exact"] = round(float(np.abs(u - ex).max()), 6)
        if p.kind == "heat":
            out["total_heat"] = round(float(u.sum() * sim.dx**2), 6)
        return out


def _plate_params(cls):
    return cls


@register
class Field(_Plate):
    """A 2D solution as a color plane: a plate heating up, a drum vibrating, a reaction-
    diffusion pattern growing. Heat and reaction map ``u / scale`` onto the colormap; a wave
    maps its sign too (zero in the middle). A backdrop: labels may sit on it."""

    type: ClassVar[str] = "pde_field"
    Style: ClassVar = {"colormap": ("colormap", "ember")}
    Presets: ClassVar = {
        "heat": {"style": {"colormap": "ember"}},
        "drum": {"style": {"colormap": "ocean"}},
        "pattern": {"style": {"colormap": "sequential"}},
    }

    class Params(ParamsBase):
        kind: Literal["heat", "wave", "reaction"] = "heat"
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        size: tuple[float, float] = (6.0, 6.0)
        n: int = PField(48, ge=8, le=400)
        diffusivity: float = PField(0.1, gt=0)
        speed: float = PField(1.0, gt=0)
        mode: tuple[int, int] = (1, 1)  # (m, n); heat with (0, 0): a hot spot
        feed: float = PField(0.037, gt=0)
        kill: float = PField(0.06, gt=0)
        seed: int = 0
        scale: float = PField(1.0, gt=0)

        @model_validator(mode="after")
        def _stable(self):
            kind = {"heat": "heat2d", "wave": "wave2d", "reaction": "reaction"}[self.kind]
            _sim(kind, _key(self))
            return self

    def _rect(self, p):
        c = np.asarray(p.pos, dtype=np.float64) - np.array([p.size[0] / 2, p.size[1] / 2, 0.0])
        return c, np.array([p.size[0], 0.0, 0.0]), np.array([0.0, p.size[1], 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([c, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "pde_field", "backdrop": True, **self._stats(state)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        _, u = self._grid(state)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        v = np.clip(u / p.scale, 0, 1) if p.kind != "wave" else 0.5 + 0.5 * np.tanh(u / p.scale)
        nx, ny = v.shape
        corner, right, up = self._rect(p)
        sx, sy = p.size

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            half_w = height * pw / ph / 2
            xs = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w
            ys = center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height
            gx = np.clip(xs / sx * (nx - 1), 0, nx - 1)
            gy = np.clip(ys / sy * (ny - 1), 0, ny - 1)
            x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
            x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
            fx, fy = (gx - x0)[None, :], (gy - y0)[:, None]
            val = (
                v[x0][:, y0].T * (1 - fx) * (1 - fy)
                + v[x1][:, y0].T * fx * (1 - fy)
                + v[x0][:, y1].T * (1 - fx) * fy
                + v[x1][:, y1].T * fx * fy
            )
            idx = np.clip((val * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
            out = np.ones((ph, pw, 4), dtype=np.float32)
            out[..., :3] = lut[idx]
            return out

        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity)


@register
class Surface(_Plate):
    """A 2D solution as a lit 3D surface: the height is the value (times ``height``). Give
    the scene an orbit camera."""

    type: ClassVar[str] = "pde_surface"
    Style: ClassVar = {"colormap": ("colormap", "ember"), "material": ("surface", "solid")}

    class Params(ParamsBase):
        kind: Literal["heat", "wave", "reaction"] = "heat"
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the base
        size: tuple[float, float] = (6.0, 6.0)
        n: int = PField(32, ge=8, le=200)
        diffusivity: float = PField(0.1, gt=0)
        speed: float = PField(1.0, gt=0)
        mode: tuple[int, int] = (1, 1)
        feed: float = PField(0.037, gt=0)
        kill: float = PField(0.06, gt=0)
        seed: int = 0
        scale: float = PField(1.0, gt=0)
        height: float = PField(2.0, gt=0)  # world height of a value of ``scale``

        @model_validator(mode="after")
        def _stable(self):
            kind = {"heat": "heat2d", "wave": "wave2d", "reaction": "reaction"}[self.kind]
            _sim(kind, _key(self))
            return self

    def _mesh(self, state: SceneState):
        p = self.me(state).params
        _, u = self._grid(state)
        nx, ny = u.shape
        xs = p.pos[0] + (np.linspace(0, 1, nx) - 0.5) * p.size[0]
        ys = p.pos[1] + (np.linspace(0, 1, ny) - 0.5) * p.size[1]
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = p.pos[2] + u / p.scale * p.height
        return np.stack([X, Y, Z], axis=-1), u

    def world_bounds(self, state: SceneState) -> Bounds:
        pts, _ = self._mesh(state)
        return Bounds.of_points(pts.reshape(-1, 3))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pts, u = self._mesh(state)
        i = np.unravel_index(int(np.argmax(u)), u.shape)
        return {"peak": pts[i], "center": pts.reshape(-1, 3).mean(axis=0)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "pde_surface", **self._stats(state)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        pts, u = self._mesh(state)
        nx, ny = u.shape
        du, dv = np.gradient(pts, axis=0), np.gradient(pts, axis=1)
        normals = np.cross(du, dv)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        v = np.clip(u / p.scale, 0, 1) if p.kind != "wave" else 0.5 + 0.5 * np.tanh(u / p.scale)
        rgb = lut[np.clip((v * (len(lut) - 1)).astype(int), 0, len(lut) - 1)]
        colors = np.concatenate([rgb, np.ones((*rgb.shape[:-1], 1))], axis=-1)
        k = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
        tris = np.vstack([np.c_[k, k + ny, k + 1], np.c_[k + 1, k + ny, k + ny + 1]])
        uv = np.stack(np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij"), -1)
        ctx.surface(
            pts.reshape(-1, 3),
            tris,
            normals=normals.reshape(-1, 3),
            colors=colors.reshape(-1, 4),
            uv=uv.reshape(-1, 2),
            material=me.roles["material"],
            opacity=me.params.opacity * float(me.style["material"]),
        )
