"""The flow objects (Milestone 26, EXPLORER.md): a dynamical system x' = F(x) in 2 or 3
dimensions and what you look at in it.

    from phasor import flow

    s.add(flow.System(id="lorenz", **flow.preset("lorenz", rho=27.7, beta=2.67)))
    s.add(flow.FieldArrows(id="lorenz.field"))
    s.add(flow.Equilibria(id="lorenz.eq"))
    s.add(flow.FlowSet(id="lorenz.ball", shape="ball", center=(1, 1, 20), size=2, count=2000))

``System`` holds the formulas (``equations``, one per coordinate, plus ``terms``: sources
that add), the named numbers they use (``constants``; a ``Variable`` with the same name
overrides one, so a slider moves it), the window, ``dt`` and the flow time ``tau`` (plus
``rate`` times the scene time). Its children (dotted ids, ``lorenz.field``, or any id with
``system="lorenz"``) read the formulas from it, so they are written once:

- ``FieldArrows``: the field as arrows (2D) or cones (3D) on a grid, coloured by size.
- ``Tracer``: one seed's path up to the flow time, the moving dot and a label.
- ``FlowSet``: a set of initial conditions (disk, ball, box, segment, circle, points)
  advected as a set (a checkpointed ``Simulation``); its boundary is refined so it stays a
  curve, each new point integrated from flow time 0.
- ``Equilibria``: every equilibrium in the window, classified from the exact Jacobian.
- ``EquilibriumView``: one equilibrium close up: its subspaces and a card with the numbers.
- ``Manifold``: the 1D stable and unstable manifolds of a saddle.

Numerics are in ``flow_math``. Everything draws through ``strokes`` for now (world-space
batches are Milestone 28). Semantics are numbers with where they came from, rounded to 6
digits; positions of a path are given only up to flow time ``flow_math.SAFE_TAU`` (a chaotic
path is not reproducible across machines after that).
"""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from typing import Any, ClassVar, Literal

import numpy as np
import sympy as sp
from pydantic import Field, field_validator, model_validator

from phasor.core.ids import parent_of
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.simulation import runner_for
from phasor.core.state import ObjectState, SceneState
from phasor.core.values import Vec3
from phasor.domains import flow_math as fm
from phasor.domains.strokes import (
    arrow_batch,
    circle,
    cone_batch,
    disc_batch,
    lut_colors,
    polyline,
    segment_batch,
    with_alpha,
)

__all__ = ["Equilibria", "EquilibriumView", "FieldArrows", "FlowSet", "Manifold", "System",
           "Tracer", "Variable", "preset"]  # fmt: skip

MAX_ARROWS = 20000  # arrows of one field (a grid is thinned to this)
MAX_DRAWN_POINTS = 50000  # points of a set drawn as dots (a stride past this)
CHECKPOINT_EVERY = 50  # set steps between cached checkpoints
MAX_TAU = 1e5


def _r(v: float, digits: int = 6) -> float:
    """Rounded for semantics; never ``-0.0``."""
    x = round(float(v), digits)
    return 0.0 if x == 0 else x


def _rv(v) -> list[float]:
    return [_r(a) for a in np.asarray(v, dtype=np.float64).ravel()]


def preset(name: str, **variables: float) -> dict[str, Any]:
    """The parameters of a ``System`` for a classic (``flow_math.PRESETS``): equations,
    constants (override any by keyword: ``preset("lorenz", rho=27.7)``), window and dt."""
    if name not in fm.PRESETS:
        raise ValueError(f"no preset {name!r}; presets are {sorted(fm.PRESETS)}")
    p = fm.PRESETS[name]
    unknown = set(variables) - set(p.variables)
    if unknown:
        raise ValueError(f"{name} has no variable {sorted(unknown)}; it has {sorted(p.variables)}")
    vals = {k: float(variables.get(k, v[0])) for k, v in p.variables.items()}
    return {"equations": p.equations, "constants": ", ".join(f"{k}={v!r}" for k, v in vals.items()),
            "window": p.window, "dt": p.dt}  # fmt: skip


# ---------------------------------------------------------------- Variable


@register
class Variable(ObjectBase):
    """A named number the formulas of a ``System`` use (``name``), with its slider range
    (``min``, ``max``). A ``System`` whose ``constants`` name it takes this value instead;
    animate ``value`` or point a ``Control`` at ``<id>.value``. Draws nothing."""

    type: ClassVar[str] = "variable"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        name: str = "a"
        value: float = 1.0
        min: float = -10.0
        max: float = 10.0

        @field_validator("name")
        @classmethod
        def _name(cls, v: str) -> str:
            if not v.isidentifier() or not v.isascii() or v in fm.RESERVED:
                raise ValueError(f"a variable's name is letters, digits and _ and not one of "
                                 f"{sorted(fm.RESERVED)}, not {v!r}")  # fmt: skip
            return v

        @model_validator(mode="after")
        def _range(self):
            if not (math.isfinite(self.value) and math.isfinite(self.min)
                    and math.isfinite(self.max)):  # fmt: skip
                raise ValueError("a variable's value and range must be finite")
            if self.min >= self.max:
                raise ValueError("a variable needs min < max")
            return self

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        used = sorted(o.id for o in state.objects if isinstance(o.impl, System)
                      and p.name in o.impl.resolve(state).values)  # fmt: skip
        return {"kind": "variable", "name": p.name, "value": _r(p.value), "min": _r(p.min),
                "max": _r(p.max), "used_by": used, "drawn": 0.0,
                "outside_range": not (p.min <= p.value <= p.max)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        """Nothing: a variable is only a number."""


# ---------------------------------------------------------------- System


@functools.lru_cache(maxsize=128)
def _build(equations: tuple[str, ...], terms: tuple[str, ...], names: tuple[str, ...]) -> fm.System:
    return fm.build(equations, terms, names)


class Resolved:
    """What a system is at one scene time: the field, the values of its variables, the
    window, the flow time and how far a step may move a seed."""

    def __init__(self, obj: ObjectState, state: SceneState) -> None:
        p = obj.params
        self.id = obj.id
        consts = fm.parse_constants(p.constants)
        self.sys = _build(tuple(p.equations), tuple(p.terms), tuple(sorted(consts)))
        over = {}
        for o in state.objects:  # a Variable of the same name overrides a constant
            if isinstance(o.impl, Variable) and o.params.name in consts:
                over.setdefault(o.params.name, float(o.params.value))
        self.values = {k: over.get(k, v) for k, v in consts.items() if k in self.sys.variables}
        self.from_variable = sorted(k for k in over if k in self.values)
        self.vals = self.sys.values_of(self.values)
        self.dim = self.sys.dim
        self.lo = np.array([w[0] for w in p.window], dtype=np.float64)
        self.hi = np.array([w[1] for w in p.window], dtype=np.float64)
        self.centre = (self.lo + self.hi) / 2
        self.diag = float(np.linalg.norm(self.hi - self.lo))
        self.dt = float(p.dt)
        tau = float(p.tau) + float(p.rate) * float(state.t)
        self.tau = float(np.clip(tau, -MAX_TAU, MAX_TAU))
        self.escape = (self.centre, float(p.escape) if p.escape > 0 else 5.0 * self.diag)
        self.max_step = 0.05 * self.diag
        self.origin = np.asarray(p.pos, dtype=np.float64)
        self.scale = float(p.scale)
        self.key = self.sys.key(self.values)

    def run_key(self) -> tuple:
        """What an integration takes besides the field and the seeds: ``dt``, the escape
        centre and radius, ``max_step`` (memo keys are built from it, so none is missed)."""
        return (self.dt, tuple(map(float, self.escape[0])), float(self.escape[1]), self.max_step)

    def steps(self, tau: float | None = None) -> tuple[int, bool]:
        """Steps to reach flow time ``tau`` (default: now), and whether that was clamped."""
        n = round(abs(self.tau if tau is None else tau) / self.dt)
        return min(n, fm.MAX_STEPS), n > fm.MAX_STEPS

    def world(self, pts) -> np.ndarray:
        """Coordinates (N, dim) to world points (N, 3): ``pos + scale * (x, y[, z])``."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, self.dim)
        if self.dim == 2:
            pts = np.c_[pts, np.zeros(len(pts))]
        return self.origin + self.scale * pts

    def bounds(self) -> Bounds:
        return Bounds.of_points(self.world(np.stack([self.lo, self.hi])))


@register
class System(ObjectBase):
    """x' = F(x): ``equations`` (one formula per coordinate, 2 or 3; ``x, y`` or ``x, y,
    z``) plus ``terms`` (sources that add, as text: ``uniform(v=(1, 0))``,
    ``source(pos=(1, 0), strength=1, core=0.1)``, ``sink``, ``vortex``,
    ``linear(A=((0, 1), (-1, 0)))``). Named numbers in the formulas get a value in
    ``constants`` ("rho=28,
    beta=2.67"); a ``Variable`` of the same name overrides it. ``window`` is (lo, hi) per
    axis; ``pos`` is where the coordinate origin sits in the world and ``scale`` world units
    per unit. The flow time is ``tau + rate * t``; ``dt`` the fixed RK4 step. Draws the
    window's frame (``frame``) and, with ``label``, the equations."""

    type: ClassVar[str] = "flow_system"
    Life: ClassVar = frozenset()
    Style: ClassVar = {
        "frame": ("emphasis", "muted"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "hairline"),
    }

    class Params(ParamsBase):
        equations: tuple[str, ...] = ("y", "-sin(x) - 0.3*y")
        terms: tuple[str, ...] = ()  # "vortex(pos=(0, 0), strength=2)", ...
        constants: str = ""
        window: tuple[tuple[float, float], ...] = ((-4.0, 4.0), (-3.0, 3.0))
        tau: float = Field(0.0, ge=-MAX_TAU, le=MAX_TAU)
        rate: float = Field(0.0, ge=-1e3, le=1e3)
        dt: float = Field(0.01, ge=1e-5, le=1.0)
        escape: float = Field(0.0, ge=0)  # 0: five window diagonals from the centre
        pos: Vec3 = (0.0, 0.0, 0.0)
        scale: float = Field(1.0, gt=0, le=1e6)
        frame: bool = True
        label: bool = False

        @model_validator(mode="after")
        def _formulas(self):
            n = len(self.equations)
            if n not in (2, 3):
                raise ValueError(f"a system has 2 or 3 equations (x' and y', and z'), not {n}")
            if len(self.window) != n:
                raise ValueError(f"the window needs {n} (lo, hi) pairs, one per coordinate")
            for lo, hi in self.window:
                if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                    raise ValueError(f"a window side must be finite with lo < hi, not {(lo, hi)}")
            try:
                consts = fm.parse_constants(self.constants)
                bad = set(consts) & fm.RESERVED
                if bad:
                    raise fm.FunctionError(f"{sorted(bad)} are coordinates or constants, "
                                           "not names to give values")  # fmt: skip
                _build(tuple(self.equations), tuple(self.terms), tuple(sorted(consts)))
            except fm.FunctionError as e:
                msg = str(e)
                if "unknown name" in msg:
                    msg += "; give every named number a value in constants ('rho=28')"
                raise ValueError(msg) from None
            return self

    def resolve(self, state: SceneState) -> Resolved:
        return Resolved(self.me(state), state)

    def world_bounds(self, state: SceneState) -> Bounds:
        return self.resolve(state).bounds()

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        r = self.resolve(state)
        return {"center": r.world(r.centre[None])[0]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        r = self.resolve(state)
        div = r.sys.divergence()
        names = r.sys.variables
        sample = fm.grid_starts(r.lo, r.hi, 7)

        xs = [sp.Symbol(c, real=True) for c in fm.COORDS[: r.dim]]
        dfn = sp.lambdify(xs + [sp.Symbol(n, real=True) for n in names], div, "numpy")
        with np.errstate(all="ignore"):
            dv = np.broadcast_to(np.asarray(dfn(*sample.T, *r.vals), dtype=np.float64),
                                 (len(sample),))  # fmt: skip
        dv = dv[np.isfinite(dv)]
        return {
            "kind": "flow_system",
            "dim": r.dim,
            "equations": [f"{c}' = {e}" for c, e in zip(fm.COORDS, r.sys.exprs, strict=False)],
            "latex": r.sys.latex(),
            "variables": {k: _r(v) for k, v in r.values.items()},
            "from_variables": r.from_variable,
            "divergence": {
                "formula": str(div),
                "min": _r(dv.min()) if len(dv) else None,
                "max": _r(dv.max()) if len(dv) else None,
                "sampled": "7 per side of the window",
            },
            "linear": all(
                e.is_polynomial(*xs) and sp.Poly(e, *xs).total_degree() <= 1 for e in r.sys.exprs
            ),
            "window": [[_r(a), _r(b)] for a, b in zip(r.lo, r.hi, strict=True)],
            "tau": _r(r.tau),
            "dt": _r(r.dt),
            "rate": _r(self.me(state).params.rate),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        r = self.resolve(state)
        cam = state.camera_for(me)
        if p.frame:
            corners = fm.grid_starts(r.lo, r.hi, 2)
            pairs = [(i, j) for i in range(len(corners)) for j in range(i + 1, len(corners))
                     if (corners[i] != corners[j]).sum() == 1]  # fmt: skip
            a = r.world(corners[[i for i, _ in pairs]])
            b = r.world(corners[[j for _, j in pairs]])
            segment_batch(ctx, cam, a, b, me.style["stroke"],
                          colors=with_alpha(me.style["frame"], p.opacity * 0.5))  # fmt: skip
        if p.label:
            unit = float(cam.units_per_height(r.world(r.centre[None])[0]))
            size = me.style["size"] * unit
            top = r.world(np.r_[r.lo[0], r.hi[1:]][None])[0]
            for k, (c, src) in enumerate(zip(fm.COORDS, p.equations, strict=False)):
                ctx.text(f"{c}' = {src}", top - cam.up * size * (1.4 * k + 1.0), size,
                         color=with_alpha(me.style["text"], p.opacity), anchor="left")  # fmt: skip


# ---------------------------------------------------------------- children


class _ChildParams(ParamsBase):
    system: str = ""  # the System this draws; empty: the dotted parent (``lorenz.field``)


class _Child(ObjectBase):
    Life: ClassVar = frozenset()

    def _sys(self, state: SceneState) -> Resolved | None:
        sid = self.me(state).params.system or parent_of(self.id)
        if sid and sid in state and isinstance(state[sid].impl, System):
            return state[sid].impl.resolve(state)
        return None

    def world_bounds(self, state: SceneState) -> Bounds:
        r = self._sys(state)
        return r.bounds() if r is not None else Bounds((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        r = self._sys(state)
        return {"center": np.zeros(3) if r is None else r.world(r.centre[None])[0]}

    def _none(self, kind: str) -> dict[str, Any]:
        return {"kind": kind, "system": None, "drawn": 0.0,
                "why": "no System: give a dotted id under one or set 'system'"}  # fmt: skip

    @staticmethod
    def _unit(state: SceneState, me: ObjectState, r: Resolved) -> float:
        cam = state.camera_for(me)
        return float(cam.units_per_height(r.world(r.centre[None])[0]))


def _budget(**kw) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


# ---------------------------------------------------------------- FieldArrows


@register
class FieldArrows(_Child):
    """The field of its system as arrows on a grid (``counts`` per side; 0: 15 in 2D and 7
    in 3D), flat arrows in 2D and cones in 3D (``shape``), coloured by size (the
    ``colormap``). Lengths follow ``length_by`` (``soft``: weak arrows still show). A grid
    of more than ``MAX_ARROWS`` is thinned and linted ``flow_budget``."""

    type: ClassVar[str] = "flow_arrows"
    Style: ClassVar = {
        "colormap": ("colormap", "magnitude"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(_ChildParams):
        counts: int = Field(0, ge=0, le=200)
        shape: Literal["auto", "arrow", "cone"] = "auto"
        length_by: Literal["soft", "size"] = "soft"
        size: float = Field(0.85, gt=0, le=3)  # the longest arrow, as a share of the spacing

    def _grid(self, p, r: Resolved):
        n = p.counts or (15 if r.dim == 2 else 7)
        want = n**r.dim
        n = min(n, int(MAX_ARROWS ** (1 / r.dim) + 1e-9))
        axes = [np.linspace(a + (b - a) / (2 * n), b - (b - a) / (2 * n), n)
                for a, b in zip(r.lo, r.hi, strict=True)]  # fmt: skip
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, r.dim)
        spacing = float(min((b - a) / n for a, b in zip(r.lo, r.hi, strict=True)))
        return pts, spacing, want

    def semantics(self, state: SceneState) -> dict[str, Any]:
        r = self._sys(state)
        if r is None:
            return self._none("flow_arrows")
        p = self.me(state).params
        pts, _, want = self._grid(p, r)
        v = r.sys.fn.velocity(pts, r.vals)
        mag = np.linalg.norm(v, axis=1)
        fin = mag[np.isfinite(mag)]
        return {"kind": "flow_arrows", "system": r.id, "arrows": len(pts),
                "strongest": _r(fin.max()) if len(fin) else None,
                "weakest": _r(fin.min()) if len(fin) else None,
                "not_finite": int((~np.isfinite(mag)).sum()),
                "budget": _budget(arrows=len(pts),
                                  clamped=["arrows"] if want > len(pts) else [])}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        r = self._sys(state)
        if r is None:
            return
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        pts, spacing, _ = self._grid(p, r)
        v = r.sys.fn.velocity(pts, r.vals)
        mag = np.linalg.norm(v, axis=1)
        ok = np.isfinite(mag) & (mag > 1e-12)
        if not ok.any():
            return
        pts, v, mag = pts[ok], v[ok], mag[ok]
        if p.length_by == "soft":
            length = spacing * p.size * np.tanh(mag / max(1.5 * float(np.median(mag)), 1e-12))
        else:
            length = spacing * p.size * mag / float(mag.max())
        d = v / mag[:, None] * length[:, None]
        a = r.world(pts - d / 2)
        b = r.world(pts + d / 2)
        lo, hi = float(np.log1p(mag.min())), float(np.log1p(mag.max()))
        u = (np.log1p(mag) - lo) / max(hi - lo, 1e-12)
        cols = np.c_[lut_colors(me.style["colormap"], u)[:, :3], np.full(len(u), p.opacity)]
        shape = p.shape if p.shape != "auto" else ("cone" if r.dim == 3 else "arrow")
        if shape == "cone":
            cone_batch(ctx, a, b, length * r.scale * 0.22, colors=cols, glow=me.style["glow"])
        else:
            w = me.style["stroke"] * self._unit(state, me, r)
            arrow_batch(ctx, cam, a, b, w, colors=cols, glow=me.style["glow"])


# ---------------------------------------------------------------- Tracer

_PATHS: OrderedDict[tuple, fm.Path] = OrderedDict()


def _path(r: Resolved, x0: tuple[float, ...], n: int, direction: float) -> fm.Path:
    """The stored path of one seed to step ``n``: the longest one computed is kept and
    extended (the same steps, so the answer does not depend on what was asked before)."""
    key = (r.key, tuple(map(float, x0)), direction, *r.run_key())
    have = _PATHS.get(key)
    if have is None or have.points.shape[1] - 1 < n:
        more = n - (0 if have is None else have.points.shape[1] - 1)
        have = fm.integrate(r.sys.fn, r.vals, np.asarray([x0], dtype=np.float64), more,
                            direction * r.dt, escape=r.escape, max_step=r.max_step,
                            start=have)  # fmt: skip
        _PATHS[key] = have
        while len(_PATHS) > 64:
            _PATHS.popitem(last=False)
    _PATHS.move_to_end(key)
    return fm.Path(have.points[:, : n + 1], np.where(have.at <= n, have.code, 0),
                   np.where(have.at <= n, have.at, -1))  # fmt: skip


@register
class Tracer(_Child):
    """One seed ``x0`` and its path under the flow up to the system's flow time (backward
    for a negative one, bounded by the escape radius): the path (the last ``trail`` of flow
    time, 0: all of it), the moving dot and a ``label`` beside it (``"phi^t(x0)"``). A path
    stores at most ``flow_math.MAX_PROBE_STEPS`` steps (lint ``flow_budget`` past that)."""

    type: ClassVar[str] = "flow_tracer"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "dot": ("emphasis", "primary"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams):
        x0: tuple[float, ...] = (1.0, 0.0)
        trail: float = Field(0.0, ge=0)
        dot: bool = True
        label: str = ""

        @field_validator("x0")
        @classmethod
        def _x0(cls, v):
            if len(v) not in (2, 3) or not all(math.isfinite(a) for a in v):
                raise ValueError("x0 is 2 or 3 finite numbers")
            return v

    def _run(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None
        p = self.me(state).params
        x0 = tuple(p.x0[: r.dim]) + (0.0,) * (r.dim - len(p.x0))
        n, clamped = r.steps()
        n = min(n, fm.MAX_PROBE_STEPS)
        clamped = clamped or round(abs(r.tau) / r.dt) > fm.MAX_PROBE_STEPS
        path = _path(r, x0, n, -1.0 if r.tau < 0 else 1.0)
        return r, x0, n, clamped, path

    def semantics(self, state: SceneState) -> dict[str, Any]:
        run = self._run(state)
        if run is None:
            return self._none("flow_tracer")
        r, x0, n, clamped, path = run
        pts = path.points[0]
        fin = pts[np.isfinite(pts).all(axis=1)]
        stopped = {fm.NONFINITE: "not finite", fm.TOO_FAST: "too fast",
                   fm.ESCAPED: "escaped"}.get(int(path.code[0]))  # fmt: skip
        out = {"kind": "flow_tracer", "system": r.id, "x0": _rv(x0), "tau": _r(r.tau),
               "steps": n, "stopped": stopped,
               "stopped_at_tau": _r(path.at[0] * r.dt) if path.at[0] >= 0 else None,
               "too_fast": int(path.code[0] == fm.TOO_FAST),
               "budget": _budget(points=1, steps=n,
                                 clamped=["steps"] if clamped else [])}  # fmt: skip
        if abs(r.tau) <= fm.SAFE_TAU and len(fin):
            out["at"] = _rv(fin[-1])
            v = r.sys.fn.velocity(fin[-1:], r.vals)[0]
            out["speed"] = _r(np.linalg.norm(v))
        else:
            out["at"] = None
            out["note"] = (f"positions are given up to flow time {fm.SAFE_TAU:g} only (a "
                           "chaotic path is not reproducible across machines)")  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        run = self._run(state)
        if run is None:
            return
        r, _, _n, _, path = run
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        pts = path.points[0]
        if p.trail > 0:
            pts = pts[max(0, len(pts) - 1 - round(p.trail / r.dt)) :]
        w = me.style["stroke"] * self._unit(state, me, r)
        for run_ in _finite_runs(pts):
            polyline(ctx, cam, r.world(run_), w, color=with_alpha(me.style["color"], p.opacity),
                     glow=me.style["glow"])  # fmt: skip
        fin = pts[np.isfinite(pts).all(axis=1)]
        if not len(fin):
            return
        at = r.world(fin[-1:])[0]
        unit = self._unit(state, me, r)
        if p.dot:
            disc_batch(ctx, cam, at[None], unit * 0.012,
                       colors=with_alpha(me.style["dot"], p.opacity), glow=2.0)  # fmt: skip
        if p.label:
            size = me.style["size"] * unit
            ctx.text(p.label, at + (cam.right + cam.up) * size * 0.8, size,
                     color=with_alpha(me.style["text"], p.opacity), anchor="left")  # fmt: skip


def _finite_runs(pts: np.ndarray) -> list[np.ndarray]:
    ok = np.isfinite(pts).all(axis=1)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return []
    return [pts[run] for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            if len(run) >= 2]  # fmt: skip


# ---------------------------------------------------------------- FlowSet


@register
class FlowSet(_Child):
    """A set of initial conditions advected as a set: ``shape`` (``disk`` or ``ball``: radius
    ``size``; ``box``: half-side ``size``; ``segment``, ``circle``; ``points``: the
    ``points`` given) around ``center``. ``fill``: ``filled`` (``count`` points spread evenly,
    no random numbers), ``boundary`` (the outline, refined so neighbours stay closer than
    ``h_max``: each new point integrated from flow time 0) or ``both``. ``trail`` keeps the
    last few positions of the points. A checkpointed ``Simulation``: scrubbing costs at most
    ``CHECKPOINT_EVERY`` steps. Lint: ``flow_budget`` (more than the theme's work, or
    clamped), ``flow_too_fast`` (points stopped for speed), ``set_under_resolved`` (the
    outline reached ``flow_math.MAX_SET_POINTS`` and is drawn as points from then on)."""

    type: ClassVar[str] = "flow_set"
    Style: ClassVar = {
        "colormap": ("colormap", "spectrum"),
        "outline": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams):
        shape: Literal["disk", "ball", "box", "segment", "circle", "points"] = "disk"
        center: tuple[float, ...] = (1.0, 0.0)
        size: float = Field(0.5, gt=0)
        points: tuple[tuple[float, ...], ...] = ()
        count: int = Field(400, ge=0, le=fm.MAX_POINTS)
        fill: Literal["filled", "boundary", "both"] = "filled"
        trail: int = Field(0, ge=0, le=fm.MAX_TRAIL)
        samples: int = Field(64, ge=4, le=4096)  # outline points before refining
        h_max: float = Field(0.0, ge=0)  # 0: 1% of the window's diagonal
        rings: int = Field(3, ge=1, le=8)  # a 3D ball's outline: latitude and longitude rings
        radius: float = Field(0.006, gt=0, le=0.1)  # dot radius, a share of the frame height

        @field_validator("center")
        @classmethod
        def _center(cls, v):
            if len(v) not in (2, 3) or not all(math.isfinite(a) for a in v):
                raise ValueError("center is 2 or 3 finite numbers")
            return v

        @field_validator("points")
        @classmethod
        def _points(cls, v):
            if len(v) > fm.MAX_POINTS:
                raise ValueError(f"at most {fm.MAX_POINTS} points")
            if any(len(q) not in (2, 3) or not all(math.isfinite(a) for a in q) for q in v):
                raise ValueError("each point is 2 or 3 finite numbers")
            return v

    def _sim(self, p, r: Resolved) -> tuple[fm.FlowSim, list[str]]:
        d = r.dim
        c = np.asarray(tuple(p.center[:d]) + (0.0,) * (d - len(p.center)), dtype=np.float64)
        clamped = []
        trail = p.trail
        if p.shape == "points":
            pts = np.array([tuple(q[:d]) + (0.0,) * (d - len(q)) for q in p.points],
                           dtype=np.float64).reshape(-1, d)  # fmt: skip
            loops: tuple = ()
        else:
            filled = p.fill in ("filled", "both") and p.shape not in ("segment", "circle")
            pts = fm.fill(p.shape, c, p.size, p.count, d) if filled else np.zeros((0, d))
            outline = p.fill in ("boundary", "both") or p.shape in ("segment", "circle")
            loops = fm.curves(p.shape, c, p.size, d, p.rings) if outline else ()
        if trail and len(pts) * trail > fm.MAX_POINTS_X_TRAIL:
            trail = max(1, fm.MAX_POINTS_X_TRAIL // max(len(pts), 1))
            clamped.append("trail")
        sim = fm.FlowSim(r.sys, r.vals, r.dt, points=pts, loops=loops, samples=p.samples,
                         h_max=p.h_max if p.h_max > 0 else 0.01 * r.diag, refine_every=10,
                         cap=fm.MAX_SET_POINTS,
                         max_step=r.max_step, escape=(tuple(map(float, r.escape[0])),
                                                      r.escape[1]), trail=trail,
                         direction=-1.0 if r.tau < 0 else 1.0)  # fmt: skip
        return sim, clamped

    def _run(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None
        p = self.me(state).params
        sim, clamped = self._sim(p, r)
        n, over = r.steps()
        if over:
            clamped.append("steps")
        # the work is bounded before it starts: points x steps, an outline counted at its cap
        size = len(sim.points) + (sim.cap if sim.loops else 0)
        most = int(fm.MAX_WORK // max(size, 1))
        if n > most:
            n = most
            clamped.append("work")
        runner = runner_for(sim, every=CHECKPOINT_EVERY)
        return r, sim, n, clamped, runner.state_at(n * sim.dt)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        run = self._run(state)
        if run is None:
            return self._none("flow_set")
        r, _sim, n, clamped, st = run
        p = self.me(state).params
        codes = np.concatenate([st["pc"], st["qc"]])
        out = {"kind": "flow_set", "system": r.id, "shape": p.shape, "fill": p.fill,
               "points": len(st["p"]), "outline": len(st["q"]),
               "alive": int((codes == fm.ALIVE).sum()),
               "escaped": int((codes == fm.ESCAPED).sum()),
               "not_finite": int((codes == fm.NONFINITE).sum()),
               "too_fast": int((codes == fm.TOO_FAST).sum()),
               "tau": _r(r.tau), "steps": n,
               "budget": _budget(points=len(codes), steps=n, clamped=clamped)}  # fmt: skip
        under = int(st["under"])
        if under >= 0:
            out["under_resolved_since"] = _r(under * r.dt)
        if abs(r.tau) <= fm.SAFE_TAU:
            both = np.concatenate([st["p"], st["q"]])
            fin = both[np.isfinite(both).all(axis=1)]
            if len(fin):
                out["box"] = [_rv(fin.min(axis=0)), _rv(fin.max(axis=0))]
        else:  # counts that depend on where chaotic points went: not reproducible
            for k in ("outline", "alive", "escaped", "not_finite"):
                out[k] = None
            out["note"] = (f"counts and positions are given up to flow time {fm.SAFE_TAU:g} "
                           "only (a chaotic set is not reproducible across machines)")  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        run = self._run(state)
        if run is None:
            return
        r, sim, _n, _, st = run
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = self._unit(state, me, r)
        lut = me.style["colormap"]
        pts = st["p"]
        if len(pts):
            start = np.asarray(sim.points)
            u = (start[:, 0] - start[:, 0].min()) / max(float(np.ptp(start[:, 0])), 1e-12)
            ok = np.isfinite(pts).all(axis=1)
            stride = max(1, -(-int(ok.sum()) // MAX_DRAWN_POINTS))
            idx = np.flatnonzero(ok)[::stride]
            cols = np.c_[lut_colors(lut, u[idx])[:, :3], np.full(len(idx), p.opacity)]
            if "trail" in st and st["trail"].shape[0] > 1 and len(idx):
                tr = st["trail"][:, idx].astype(np.float64)  # (T, K, dim)
                good = np.isfinite(tr).all(axis=(0, 2))
                a = r.world(tr[:-1, good].reshape(-1, r.dim))
                b = r.world(tr[1:, good].reshape(-1, r.dim))
                fade = np.repeat(np.linspace(0.1, 0.7, tr.shape[0] - 1), int(good.sum()))
                tc = np.tile(cols[good], (tr.shape[0] - 1, 1))
                tc[:, 3] *= fade
                segment_batch(ctx, cam, a, b, me.style["stroke"] * 0.6, colors=tc)
            disc_batch(ctx, cam, r.world(pts[idx]), unit * p.radius, colors=cols,
                       glow=me.style["glow"])  # fmt: skip
        if len(st["q"]):
            col = with_alpha(me.style["outline"], p.opacity)
            under = int(st["under"]) >= 0
            for k, c in enumerate(sim.loops):
                q = st["q"][st["loop"] == k]
                if under:
                    fin = q[np.isfinite(q).all(axis=1)]
                    disc_batch(ctx, cam, r.world(fin), unit * p.radius * 0.6, colors=col)
                    continue
                if c[0] == "circle" and len(q) > 2:
                    q = np.vstack([q, q[:1]])
                w = me.style["stroke"] * unit
                for run_ in _finite_runs(q):
                    polyline(ctx, cam, r.world(run_), w, color=col, glow=me.style["glow"])


# ---------------------------------------------------------------- Equilibria

_EQ: OrderedDict[tuple, list[dict]] = OrderedDict()


def _equilibria(r: Resolved, grid: int) -> tuple[list[dict], dict]:
    key = (r.key, tuple(r.lo), tuple(r.hi), grid)
    if key not in _EQ:
        found = fm.equilibria(r.sys, r.vals, r.lo, r.hi, grid)
        pts = []
        for at in found["points"]:
            c = fm.classify(r.sys.jacobian_exact(at, r.values))
            c["at"] = at
            pts.append(c)
        meta = {k: found[k] for k in ("window", "grid", "starts", "found", "non_isolated")}
        _EQ[key] = (pts, meta)
        while len(_EQ) > 64:
            _EQ.popitem(last=False)
    _EQ.move_to_end(key)
    return _EQ[key]


def _eq_role(c: dict) -> str:
    if c["stable"] is True:
        return "stable"
    if c["stable"] is False:
        return "unstable"
    return "neutral"


def _eq_semantic(c: dict) -> dict[str, Any]:
    return {"at": _rv(c["at"]), "kind": c["kind"], "stable": c["stable"],
            "eigenvalues": [[_r(a), _r(b)] for a, b in c["eigenvalues"]],
            "dims": dict(c["dims"]), "trace": _r(c["trace"]), "det": _r(c["det"]),
            "hyperbolic": c["dims"]["c"] == 0,
            "tolerance": float(f"{c['tolerance']:.3g}")}  # fmt: skip


def _nearest(pts: list[dict], near) -> dict | None:
    if not pts:
        return None
    q = np.asarray(near, dtype=np.float64)
    d = [float(np.linalg.norm(np.asarray(c["at"])[: len(q)] - q[: len(c["at"])])) for c in pts]
    return pts[int(np.argmin(d))]


@register
class Equilibria(_Child):
    """Every equilibrium of its system in the window (Newton from a ``grid`` of starts per
    side, at most ``flow_math.MAX_EQUILIBRIA``), each classified from the exact Jacobian and
    drawn as a ball coloured by stability (``unstable`` red: saddles too). Semantics: the
    window and grid searched (none found is not none exist), ``non_isolated`` (a line of
    them), and per point its position, kind, eigenvalues, subspace dimensions, trace and
    determinant. ``reveal`` 0 -> 1 shows them in order."""

    type: ClassVar[str] = "flow_equilibria"
    Style: ClassVar = {
        "stable": ("emphasis", "secondary"),
        "unstable": ("emphasis", "danger"),
        "neutral": ("emphasis", "primary"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams):
        grid: int = Field(9, ge=2, le=64)
        radius: float = Field(0.014, gt=0, le=0.1)  # a share of the frame height
        reveal: float = Field(1.0, ge=0, le=1)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        r = self._sys(state)
        if r is None:
            return self._none("flow_equilibria")
        pts, meta = _equilibria(r, self.me(state).params.grid)
        return {"kind": "flow_equilibria", "system": r.id, "count": len(pts), **meta,
                "points": [_eq_semantic(c) for c in pts]}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        r = self._sys(state)
        if r is None:
            return
        me = self.me(state)
        p = me.params
        pts, _ = _equilibria(r, p.grid)
        k = int(np.floor(p.reveal * len(pts) + 1e-9))
        if not k:
            return
        cam = state.camera_for(me)
        at = r.world(np.array([c["at"] for c in pts[:k]]))
        cols = np.array([with_alpha(me.style[_eq_role(c)], p.opacity) for c in pts[:k]])
        disc_batch(ctx, cam, at, self._unit(state, me, r) * p.radius, colors=cols,
                   glow=me.style["glow"])  # fmt: skip


# ---------------------------------------------------------------- EquilibriumView


def _card(name: str, c: dict) -> list[str]:
    at = ", ".join(f"{v:.4g}" for v in c["at"])
    ev = []
    done = set()
    for k, (a, b) in enumerate(c["eigenvalues"]):
        if k in done:
            continue
        if abs(b) > c["tolerance"]:
            ev.append(f"{a:.4g} +/- {abs(b):.4g}i")
            tol = c["tolerance"]
            done.update(j for j, (a2, b2) in enumerate(c["eigenvalues"])
                        if j != k and abs(a2 - a) <= tol and abs(b2 + b) <= tol)  # fmt: skip
        else:
            ev.append(f"{a:.4g}")
    d = c["dims"]
    dims = f"dim Es = {d['s']}, dim Eu = {d['u']}" + (f", dim Ec = {d['c']}" if d["c"] else "")
    head = f"{name} = ({at})" if name else f"({at})"
    return [f"{head}  {c['kind']}", "eigenvalues " + ", ".join(ev), dims]


@register
class EquilibriumView(_Child):
    """One equilibrium close up: the one nearest ``near`` (an equilibrium is found again
    whenever the system changes; it is never named by index), its subspaces drawn inside a
    ring of ``radius`` (a line for a 1D one, a translucent disc for a 2D one; stable cool,
    unstable warm) and a ``card`` with its position, kind, eigenvalues and subspace
    dimensions in plain characters. ``name`` is shown on the card (``"C+"``); it is not an
    id."""

    type: ClassVar[str] = "flow_equilibrium_view"
    Style: ClassVar = {
        "stable": ("emphasis", "secondary"),
        "unstable": ("emphasis", "danger"),
        "ring": ("emphasis", "muted"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(_ChildParams):
        near: tuple[float, ...] = (0.0, 0.0)
        radius: float = Field(1.0, gt=0)
        name: str = ""
        card: bool = True
        grid: int = Field(9, ge=2, le=64)

    def _point(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None, None
        p = self.me(state).params
        pts, _ = _equilibria(r, p.grid)
        return r, _nearest(pts, p.near)

    def world_bounds(self, state: SceneState) -> Bounds:
        r, c = self._point(state)
        if r is None or c is None:
            return super().world_bounds(state)
        rad = self.me(state).params.radius * r.scale
        return Bounds.of_points(r.world(np.asarray(c["at"])[None]), pad=rad)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        r, c = self._point(state)
        if r is None:
            return self._none("flow_equilibrium_view")
        p = self.me(state).params
        if c is None:
            return {"kind": "flow_equilibrium_view", "system": r.id, "point": None,
                    "near": _rv(p.near), "drawn": 0.0}  # fmt: skip
        return {"kind": "flow_equilibrium_view", "system": r.id, "near": _rv(p.near),
                "name": p.name, "point": _eq_semantic(c),
                "subspaces": {k: [_rv(v) for v in c[f"E{k}"]] for k in ("s", "u", "c")},
                "card": _card(p.name, c) if p.card else []}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        r, c = self._point(state)
        if r is None or c is None:
            return
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = self._unit(state, me, r)
        w = me.style["stroke"] * unit
        at = np.asarray(c["at"], dtype=np.float64)
        centre = r.world(at[None])[0]
        rad = p.radius * r.scale
        circle(ctx, cam, centre, rad, w * 0.6, color=with_alpha(me.style["ring"], p.opacity * 0.6))
        for key, role in (("Es", "stable"), ("Eu", "unstable")):
            basis = c[key]
            col = with_alpha(me.style[role], p.opacity)
            if len(basis) == 1:
                v = np.asarray(basis[0])
                polyline(ctx, cam, r.world(np.stack([at - p.radius * v, at + p.radius * v])), w,
                         color=col, glow=me.style["glow"])  # fmt: skip
            elif len(basis) == 2 and r.dim == 3:
                e1, e2 = (np.asarray(b) for b in basis)
                ang = np.linspace(0, 2 * np.pi, 49)
                rim = at + p.radius * 0.9 * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
                ring = r.world(rim)
                n = len(ring) - 1
                tris = np.stack([np.full(n, n + 1), np.arange(n), np.arange(1, n + 1)], axis=1)
                ctx.triangles(np.vstack([ring, centre[None]]), tris,
                              color=with_alpha(me.style[role], p.opacity * 0.3))  # fmt: skip
                polyline(ctx, cam, ring, w * 0.7, color=col, glow=me.style["glow"])
        if p.card:
            size = me.style["size"] * unit
            corner = centre + cam.right * (rad * 1.6 + size) + cam.up * rad * 0.6
            for k, line in enumerate(_card(p.name, c)):
                ctx.text(line, corner - cam.up * size * 1.35 * k, size,
                         color=with_alpha(me.style["text"], p.opacity), anchor="left")  # fmt: skip


# ---------------------------------------------------------------- Manifold

_MAN: OrderedDict[tuple, list[dict]] = OrderedDict()


@register
class Manifold(_Child):
    """The 1D stable and/or unstable manifold (``which``) of the equilibrium nearest
    ``near``: each branch integrated from just off the point along its eigen-direction
    (forward for the unstable one, backward for the stable one) for flow time ``length``,
    stopped at the escape radius. ``progress`` 0 -> 1 draws them out. An equilibrium
    without a one-dimensional E^s or E^u has none to draw (semantics say so)."""

    type: ClassVar[str] = "flow_manifold"
    Style: ClassVar = {
        "stable": ("emphasis", "secondary"),
        "unstable": ("emphasis", "danger"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams):
        near: tuple[float, ...] = (0.0, 0.0)
        which: Literal["both", "stable", "unstable"] = "both"
        length: float = Field(20.0, gt=0, le=1e4)
        progress: float = Field(1.0, ge=0, le=1)
        grid: int = Field(9, ge=2, le=64)

    def _branches(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None, None, []
        p = self.me(state).params
        pts, _ = _equilibria(r, p.grid)
        c = _nearest(pts, p.near)
        if c is None:
            return r, None, []
        steps = round(p.length / r.dt)
        # every argument of ``manifolds`` below is in the key
        key = (r.key, tuple(map(float, c["at"])), p.which, steps, 1e-3 * r.diag, *r.run_key())
        if key not in _MAN:
            _MAN[key] = fm.manifolds(r.sys, r.vals, c["at"], which=p.which, dt=r.dt,
                                     steps=min(steps, fm.MAX_PROBE_STEPS), escape=r.escape,
                                     eps=1e-3 * r.diag,
                                     max_step=r.max_step)  # fmt: skip
            while len(_MAN) > 64:
                _MAN.popitem(last=False)
        return r, c, _MAN[key]

    def semantics(self, state: SceneState) -> dict[str, Any]:
        r, c, br = self._branches(state)
        if r is None:
            return self._none("flow_manifold")
        p = self.me(state).params
        steps = round(p.length / r.dt)
        out = {"kind": "flow_manifold", "system": r.id, "which": p.which,
               "equilibrium": None if c is None else _rv(c["at"]),
               "equilibrium_kind": None if c is None else c["kind"],
               "branches": [{"which": b["which"], "sign": b["sign"], "ends": b["ends"],
                             "length": _r(b["length"], 4) if p.length <= fm.SAFE_TAU else None}
                            for b in br],
               "budget": _budget(points=2 * len(br), steps=min(steps, fm.MAX_PROBE_STEPS),
                                 clamped=["steps"] if steps > fm.MAX_PROBE_STEPS
                                 else [])}  # fmt: skip
        if c is not None and not br:
            out["why"] = "this equilibrium has no one-dimensional stable or unstable subspace"
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        r, _, br = self._branches(state)
        if r is None or not br:
            return
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        w = me.style["stroke"] * self._unit(state, me, r)
        for b in br:
            pts = b["points"]
            k = max(2, round(p.progress * len(pts)))
            if p.progress <= 0:
                continue
            polyline(ctx, cam, r.world(pts[:k]), w,
                     color=with_alpha(me.style[b["which"]], p.opacity),
                     glow=me.style["glow"])  # fmt: skip
