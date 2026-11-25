"""The Explorer's analysis objects (Milestone 30, EXPLORER.md; advice section 2.6 rows 5,
7-12). Children of a ``flow.System`` like the Milestone 26 objects (a dotted id or
``system=``), reached as ``flow.Section`` and so on:

- ``Section``: a Poincare section, the crossings of a plane by seeds (its own, or a
  ``Probe``'s or a ``FlowSet``'s), each found by linear interpolation inside a step.
- ``TimeSeries``: x(t), y(t), z(t) of a probe as a panel (or the log of the twins' distance).
- ``TwinProbes``: two probes ``eps`` apart; their distance's growth rate beside the largest
  Lyapunov exponent (Benettin).
- ``DivergenceMap``: div F as a heat map over the window (a 2D system, or a 3D one at
  height ``z``), and a disk carried by the flow: its area against A0 exp(integral of div).
- ``BifurcationDiagram``: a variable swept over a range, as a panel: the equilibria
  (stable and unstable) with candidate bifurcations refined and labelled, or the samples of
  the attractor after a transient (a period-doubling cascade).
- ``LimitCycle``: the attracting cycle reached from a point: period, a point and its path.

``lyapunov(...)`` is the function tools call (a document's System by id, or a
``flow_math.System``). Numerics are in ``flow_analysis_math``; everything draws through
``strokes``. Semantics are numbers with where they came from: positions from a path only up
to ``flow_math.SAFE_TAU``; numbers from long runs as statistics rounded to their spread
(``flow_analysis_math.statistic``).
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ParamsBase, Pass, register
from phasor.core.simulation import runner_for
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import flow_analysis_math as fa
from phasor.domains import flow_math as fm
from phasor.domains.flow import (
    CHECKPOINT_EVERY,
    MAX_DRAWN_POINTS,
    FlowSet,
    Probe,
    Resolved,
    System,
    _budget,
    _Child,
    _ChildParams,
    _finite_runs,
    _path,
    _r,
    _rv,
)
from phasor.domains.heatmap import HeatParams, heat_pixels, wheel_colour
from phasor.domains.strokes import disc_batch, polyline, with_alpha

__all__ = [
    "BifurcationDiagram",
    "DivergenceMap",
    "LimitCycle",
    "Section",
    "TimeSeries",
    "TwinProbes",
    "lyapunov",
]  # fmt: skip

NOT_DETECTED = ("global bifurcations (homoclinic and heteroclinic loops, saddle-node of "
                "cycles, crises); bifurcations of cycles other than the period changes the "
                "samples show; anything outside the window, between Newton's grid of starts "
                "or between two swept values")  # fmt: skip
MEMO = 32  # entries per analysis memo
_LOCK = threading.RLock()  # the memos are shared by worker threads (advice 5.23)


def _memo(table: OrderedDict, key: tuple, make):
    """A value from a memo keyed by every input; computed outside the lock (two threads may
    both compute it: the same answer), stored and read under it."""
    with _LOCK:
        if key in table:
            table.move_to_end(key)
            return table[key]
    value = make()
    with _LOCK:
        table.setdefault(key, value)
        table.move_to_end(key)
        while len(table) > MEMO:
            table.popitem(last=False)
        return table[key]


def clear_memos() -> None:
    """Empty every analysis memo and the flow objects' (tests compare a warm engine with a
    cold one this way)."""
    from phasor.domains import flow as fl

    with _LOCK, fl._LOCK:
        for memo in (_SEC, _LYA, _AREA, _SWP, _CYC, fl._PATHS, fl._MAN, fl._EQ):
            memo.clear()


def _ref(state: SceneState, rid: str, kinds: tuple, r: Resolved | None, what: str):
    """The object ``rid`` names if it is one of ``kinds`` and of the same System, else
    (None, why) for semantics and the lint ``flow_ref``."""
    if rid not in state:
        return None, f"{what} names {rid!r}, which is not in the scene"
    other = state[rid]
    if not isinstance(other.impl, kinds):
        return None, (f"{what} names {rid!r}, a {other.impl.type}, not one of "
                      f"{', '.join(k.type for k in kinds)}")  # fmt: skip
    theirs = other.impl._sys(state)
    if r is not None and (theirs is None or theirs.id != r.id):
        return None, (f"{what} names {rid!r}, which belongs to "
                      f"{None if theirs is None else theirs.id!r}, not {r.id!r}")  # fmt: skip
    return other, None


def _stat_list(v, spread: float) -> list[float | None]:
    return [fa.statistic(float(a), spread) for a in np.asarray(v, dtype=np.float64).ravel()]


def _seed(v, d: int) -> tuple[float, ...]:
    return tuple(float(a) for a in tuple(v[:d]) + (0.0,) * (d - len(v)))


def _finite_vec(v, what: str):
    if len(v) not in (2, 3) or not all(math.isfinite(a) for a in v):
        raise ValueError(f"{what} is 2 or 3 finite numbers")
    return v


# ---------------------------------------------------------------- panels


class _PanelParams(ParamsBase):
    pos: Vec3 = (-3.6, -3.7, 0.0)  # lower-left corner (on the overlay y runs down: (0.05, 0.95))
    extent: tuple[float, float] = (7.2, 2.0)  # (width, height)

    @field_validator("extent")
    @classmethod
    def _extent(cls, v):
        if not all(math.isfinite(a) and a > 0 for a in v):
            raise ValueError("a panel's extent is two positive numbers")
        return v


class _Panel:
    """A plot in a rectangle (as ``acoustics.ImpulseResponse``): ``pos`` the lower-left corner
    and ``extent`` (width, height) in world units, or in frame heights on the overlay."""

    def _ydir(self, state: SceneState) -> float:
        return 1.0 if state.camera_for(self.me(state)).up[1] >= 0 else -1.0

    def _rect(self, state: SceneState):
        p = self.me(state).params
        x, y, z = p.pos
        return x, y, z, p.extent[0], p.extent[1] * self._ydir(state)

    def world_bounds(self, state: SceneState) -> Bounds:
        x, y, z, w, h = self._rect(state)
        return Bounds((x, min(y, y + h), z), (x + w, max(y, y + h), z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        x, y, z, w, h = self._rect(state)
        return {"center": np.array([x + w / 2, y + h / 2, z])}

    def _axes(self, ctx, state, width: float, color) -> None:
        x, y, z, w, h = self._rect(state)
        cam = state.camera_for(self.me(state))
        polyline(ctx, cam, [(x, y + h, z), (x, y, z), (x + w, y, z)], width, color=color)

    @staticmethod
    def _place(rect, u, v) -> np.ndarray:
        """Shares (0..1) across and up the panel to points."""
        x, y, z, w, h = rect
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        return np.stack([x + u * w, y + v * h, np.full(u.shape, z)], axis=-1)


def _share(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.asarray(v, dtype=np.float64) - lo) / max(hi - lo, 1e-300)


def _limits(v: np.ndarray) -> tuple[float, float]:
    fin = v[np.isfinite(v)]
    if not len(fin):
        return 0.0, 1.0
    lo, hi = float(fin.min()), float(fin.max())
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(lo), 1.0) * 0.5
    return lo - pad, hi + pad


# ---------------------------------------------------------------- Section

_SEC: OrderedDict[tuple, fa.SectionRun] = OrderedDict()
_DIRS = {"up": 1, "down": -1, "both": 0}


def _plane_polygon(normal: np.ndarray, offset: float, lo, hi) -> np.ndarray:
    """Where the plane ``normal . p = offset`` meets the window box: a polygon (3D, sorted
    round its centre) or a segment (2D); empty when it misses."""
    d = len(lo)
    corners = fm.grid_starts(lo, hi, 2)
    g = corners @ normal - offset
    pts = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if (corners[i] != corners[j]).sum() != 1:
                continue
            a, b = g[i], g[j]
            if a == 0:
                pts.append(corners[i])
            elif a * b < 0:
                pts.append(corners[i] + a / (a - b) * (corners[j] - corners[i]))
    if not pts:
        return np.zeros((0, d))
    pts = np.unique(np.round(np.array(pts), 12), axis=0)
    if d == 3 and len(pts) > 2:
        c = pts.mean(axis=0)
        e1 = pts[0] - c
        e1 = e1 / max(np.linalg.norm(e1), 1e-300)
        e2 = np.cross(normal, e1)
        ang = np.arctan2((pts - c) @ e2, (pts - c) @ e1)
        pts = pts[np.argsort(ang)]
    return pts


@register
class Section(_Child):
    """A Poincare section: where seeds cross the plane ``normal . p = offset`` (a line in
    2D) up to the system's flow time, each crossing bracketed by a step and refined to
    rounding (``flow_analysis_math.refine_crossings``). ``direction``: ``up`` (along
    ``normal``), ``down`` or ``both``. The seeds are
    ``seeds``, or those of the ``Probe`` or ``FlowSet`` named by ``of``; crossings before
    ``transient`` flow time are left out. Drawn as dots (ready for a splat) on a faint plane
    clipped to the window. At most ``MAX_SECTION_SEEDS`` seeds and ``MAX_SECTION_WORK``
    seed-steps (clamped, lint ``flow_budget``); the first ``MAX_CROSSINGS`` are kept. An
    ``of`` that names nothing, another type or another System's object is lint ``flow_ref``
    (and no seeds)."""

    type: ClassVar[str] = "flow_section"
    Style: ClassVar = {
        "points": ("emphasis", "highlight"),
        "plane": ("emphasis", "muted"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams):
        normal: tuple[float, ...] = (1.0, 0.0, 0.0)
        offset: float = 0.0
        direction: Literal["up", "down", "both"] = "up"
        seeds: tuple[tuple[float, ...], ...] = ((1.0, 0.5),)
        of: str = ""  # a Probe or FlowSet whose seeds to use instead
        transient: float = Field(0.0, ge=0, le=fm.MAX_STEPS)
        plane: bool = True
        radius: float = Field(0.006, gt=0, le=0.1)  # dot radius, a share of the frame height

        @field_validator("normal")
        @classmethod
        def _normal(cls, v):
            _finite_vec(v, "normal")
            if not any(v):
                raise ValueError("a section's normal must not be zero")
            return v

        @field_validator("seeds")
        @classmethod
        def _seeds(cls, v):
            if len(v) > fa.MAX_SECTION_SEEDS:
                raise ValueError(f"at most {fa.MAX_SECTION_SEEDS} seeds")
            for q in v:
                _finite_vec(q, "each seed")
            return v

    def _seeds(self, state: SceneState, r: Resolved) -> tuple[np.ndarray, str | None]:
        p = self.me(state).params
        if p.of:
            other, why = _ref(state, p.of, (Probe, FlowSet), r, "of")
            if other is None:
                return np.zeros((0, r.dim)), why
            if isinstance(other.impl, Probe):
                return np.array([_seed(other.params.x0, r.dim)]), None
            sim, _, _ = other.impl._sim(other.params, r)
            return np.asarray(sim.points, dtype=np.float64).reshape(-1, r.dim), None
        return (np.array([_seed(q, r.dim) for q in p.seeds],
                         dtype=np.float64).reshape(-1, r.dim), None)  # fmt: skip

    def _run(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None
        p = self.me(state).params
        normal = np.asarray(_seed(p.normal, r.dim))
        if not np.linalg.norm(normal):
            return r, None, 0, [], np.zeros((0, r.dim))
        sampler = fa.Sampler.plane(normal, p.offset, _DIRS[p.direction])
        seeds, why = self._seeds(state, r)
        clamped = []
        if len(seeds) > fa.MAX_SECTION_SEEDS:
            seeds, clamped = seeds[: fa.MAX_SECTION_SEEDS], ["seeds"]
        n, over = r.steps()
        if over:
            clamped.append("steps")
        most = int(fa.MAX_SECTION_WORK // max(len(seeds), 1))
        if n > most:
            n = most
            clamped.append("work")
        direction = -1.0 if r.tau < 0 else 1.0
        blob = hashlib.sha256(np.ascontiguousarray(seeds).tobytes()).hexdigest()
        key = (r.key, blob, seeds.shape, sampler.key(), direction, *r.run_key())
        run = _memo(_SEC, key, lambda: fa.SectionRun(r.sys.fn, r.vals, seeds, direction * r.dt,
                                                    sampler, escape=r.escape,
                                                    max_step=r.max_step))  # fmt: skip
        run.advance(n)
        cr = run.crossings(upto=n)  # "full": anything cut at or before step n (not later)
        keep = np.abs(cr["times"]) >= p.transient
        cr = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in cr.items()}
        cr["bad_ref"] = why
        return r, sampler, n, clamped, seeds, cr

    def semantics(self, state: SceneState) -> dict[str, Any]:
        run = self._run(state)
        if run is None:
            return self._none("flow_section")
        r = run[0]
        p = self.me(state).params
        if run[1] is None:
            return {"kind": "flow_section", "system": r.id, "crossings": 0, "drawn": 0.0,
                    "why": "the normal has no part in this system's coordinates"}  # fmt: skip
        _, sampler, n, clamped, seeds, cr = run
        safe = np.abs(cr["times"]) <= fm.SAFE_TAU
        out = {"kind": "flow_section", "system": r.id, "normal": _rv(sampler.normal[: r.dim]),
               "offset": _r(sampler.offset), "direction": p.direction, "seeds": len(seeds),
               "transient": _r(p.transient), "tau": _r(r.tau), "steps": n,
               "crossings_to_safe_tau": int(safe.sum()),
               "first": [{"at": _rv(q), "tau": _r(t), "seed": int(s)} for q, t, s in
                         zip(cr["points"][safe][:8], cr["times"][safe][:8], cr["seed"][safe][:8],
                             strict=True)],
               "full": bool(cr["full"]) if abs(r.tau) <= fm.SAFE_TAU else None,
               "bad_ref": cr["bad_ref"],
               "budget": _budget(points=max(len(seeds), fa.LOOP_POINTS) if len(seeds) else 0,
                                 steps=n, clamped=clamped)}  # fmt: skip
        if abs(r.tau) <= fm.SAFE_TAU:
            out["crossings"] = len(cr["times"])
        else:
            out["crossings"] = None
            out["note"] = (f"crossings are counted and placed up to flow time {fm.SAFE_TAU:g} "
                           "only (a chaotic path is not reproducible across machines)")  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        run = self._run(state)
        if run is None or run[1] is None:
            return
        r, sampler, _n, _, _, cr = run
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = self._unit(state, me, r)
        if p.plane:
            n = np.asarray(sampler.normal[: r.dim])
            poly = _plane_polygon(n, sampler.offset, r.lo, r.hi)
            col = with_alpha(me.style["plane"], p.opacity * 0.6)
            if len(poly) >= 2:
                w = me.style["stroke"] * unit
                ring = np.vstack([poly, poly[:1]]) if len(poly) > 2 else poly
                polyline(ctx, cam, r.world(ring), w, color=col)
            if len(poly) > 2:
                pts = r.world(poly)
                c = pts.mean(axis=0)
                k = len(pts)
                tris = np.stack([np.full(k, k), np.arange(k), (np.arange(k) + 1) % k], axis=1)
                ctx.triangles(np.vstack([pts, c[None]]), tris,
                              color=with_alpha(me.style["plane"], p.opacity * 0.05))  # fmt: skip
        pts = cr["points"]
        if len(pts):
            stride = max(1, -(-len(pts) // MAX_DRAWN_POINTS))
            disc_batch(ctx, cam, r.world(pts[::stride]), unit * p.radius,
                       colors=with_alpha(me.style["points"], p.opacity),
                       glow=me.style["glow"])  # fmt: skip


# ---------------------------------------------------------------- TwinProbes

_LYA: OrderedDict[tuple, dict] = OrderedDict()


def _lyapunov(r: Resolved, x0: tuple[float, ...], T: float, transient: float) -> dict:
    key = (r.key, x0, float(T), float(transient), *r.run_key())
    return _memo(_LYA, key, lambda: fa.lyapunov(r.sys, np.asarray(x0), T, values=r.vals,
                                                dt=r.dt, transient=transient, escape=r.escape,
                                                max_step=r.max_step))  # fmt: skip


def _lyapunov_semantic(res: dict) -> dict[str, Any]:
    """Rounded no finer than the spread (a statistic agrees across machines within it, and a
    last digit that flips with a change of x0 in its last bit is not reported)."""
    spread = res["spread"]
    beyond = res["chaotic_beyond"]
    coarse = None if spread is None else max(spread, 1e-6) * 10  # one digit coarser
    return {"value": None if res["value"] is None else fa.statistic(res["value"], coarse),
            "spread": None if spread is None else float(f"{spread:.1g}"),
            "trend": None if res["trend"] is None else fa.statistic(res["trend"], coarse),
            "converged": res["converged"], "T": _r(res["T"]) if res["T"] else None,
            "transient": _r(res["transient"]), "stopped": res["stopped"],
            "chaotic_beyond": None if beyond is None else float(f"{beyond:.1g}"),
            "method": "Benettin, two trajectories, renormalised every "
                      f"{res['renorm']:g}; spread: standard error over blocks; converged: "
                      "small spread and no trend across blocks; chaotic_beyond only for a "
                      f"converged value above {fa.CHAOS_FLOOR:g} and three spreads",
            "clamped": res["clamped"],
            "statistic": True}  # fmt: skip


@register
class TwinProbes(_Child):
    """Two probes ``eps`` apart (sensitive dependence): ``x0`` and ``x0 + eps u``, ``u`` a
    fixed direction from a hash of the system and the start (no random numbers). Both paths
    to the flow time, their dots, and in semantics the distance, its growth ``rate`` (a fit of
    log distance against flow time before it saturates, up to ``flow_math.SAFE_TAU``) beside
    the largest Lyapunov exponent from ``x0`` over ``lyapunov`` flow time (0: not computed;
    Benettin, a statistic with its spread). Show the distance with a ``TimeSeries`` whose
    ``source`` is this object (a log strip)."""

    type: ClassVar[str] = "flow_twins"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "twin": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams):
        x0: tuple[float, ...] = (1.0, 0.0)
        eps: float = Field(1e-6, gt=0, le=1.0)
        lyapunov: float = Field(40.0, ge=0, le=1e4)  # flow time of the estimate (0: off)
        transient: float = Field(10.0, ge=0, le=1e4)  # before the estimate starts
        trail: float = Field(0.0, ge=0)
        dot: bool = True

        @field_validator("x0")
        @classmethod
        def _x0(cls, v):
            return _finite_vec(v, "x0")

    def _run(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None
        p = self.me(state).params
        a0 = _seed(p.x0, r.dim)
        u = fa.separation(r.sys, r.vals, np.asarray(a0))
        b0 = tuple(float(v) for v in np.asarray(a0) + p.eps * u)
        n, clamped = r.steps()
        n = min(n, fm.MAX_PROBE_STEPS)
        clamped = clamped or round(abs(r.tau) / r.dt) > fm.MAX_PROBE_STEPS
        sgn = -1.0 if r.tau < 0 else 1.0
        pa = _path(r, a0, n, sgn).points[0]
        pb = _path(r, b0, n, sgn).points[0]
        with np.errstate(invalid="ignore"):
            dist = np.linalg.norm(pa - pb, axis=1)
        return r, a0, u, n, clamped, pa, pb, dist

    def growth(self, r: Resolved, dist: np.ndarray) -> tuple[float | None, float | None]:
        """(rate, saturated at): the slope of log distance before it reaches 1% of the
        window's diagonal, fitted up to ``SAFE_TAU``."""
        t = np.arange(len(dist)) * r.dt
        ok = np.isfinite(dist) & (dist > 0) & (t <= fm.SAFE_TAU)
        sat = np.flatnonzero(ok & (dist >= 0.01 * r.diag))
        end = sat[0] if len(sat) else len(dist)
        use = ok & (np.arange(len(dist)) < end)
        rate = None
        if use.sum() >= 3:
            rate = float(np.polyfit(t[use], np.log(dist[use]), 1)[0])
        return rate, (float(t[end]) if len(sat) else None)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        run = self._run(state)
        if run is None:
            return self._none("flow_twins")
        r, a0, u, n, clamped, _pa, _pb, dist = run
        p = self.me(state).params
        rate, sat = self.growth(r, dist)
        steps = n
        out = {"kind": "flow_twins", "system": r.id, "x0": _rv(a0), "eps": _r(p.eps, 12),
               "direction": _rv(u), "tau": _r(r.tau), "steps": n,
               "rate": _r(rate) if rate is not None else None,
               "rate_fitted_to": _r(min(abs(r.tau), fm.SAFE_TAU)),
               "saturated_at": _r(sat) if sat is not None else None}  # fmt: skip
        if abs(r.tau) <= fm.SAFE_TAU and np.isfinite(dist[-1]):
            out["distance"] = _r(dist[-1], 12)
        else:
            out["distance"] = None
        if p.lyapunov > 0:
            res = _lyapunov(r, a0, p.lyapunov, p.transient)
            out["lyapunov"] = _lyapunov_semantic(res)
            steps += res["steps"]
            clamped = clamped or bool(res["clamped"])
        # steps of one or two points in a Python loop: weighed as LOOP_POINTS points each
        out["budget"] = _budget(points=fa.LOOP_POINTS, steps=steps,
                                clamped=["steps"] if clamped else [])  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        run = self._run(state)
        if run is None:
            return
        r, *_, pa, pb, _ = run
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = self._unit(state, me, r)
        w = me.style["stroke"] * unit
        for pts, role in ((pa, "color"), (pb, "twin")):
            if p.trail > 0:
                pts = pts[max(0, len(pts) - 1 - round(p.trail / r.dt)) :]
            col = with_alpha(me.style[role], p.opacity)
            for run_ in _finite_runs(pts):
                polyline(ctx, cam, r.world(run_), w, color=col, glow=me.style["glow"])
            fin = pts[np.isfinite(pts).all(axis=1)]
            if p.dot and len(fin):
                disc_batch(ctx, cam, r.world(fin[-1:]), unit * 0.012, colors=col, glow=2.0)


# ---------------------------------------------------------------- TimeSeries


@register
class TimeSeries(_Panel, _Child):
    """A strip of x(t), y(t), z(t) (``components``) over flow time, in a panel (``pos``,
    ``extent``; on the overlay give ``layer="overlay"`` and ``pos`` in frame heights, y
    down: ``(0.05, 0.95)``). ``source``: a ``Probe`` (its path) or ``TwinProbes`` (log10 of
    their distance, with the line of slope lambda, the Lyapunov estimate, from ``eps``); empty:
    its own probe from ``x0`` in ``system``. ``span``: the flow time shown (0: all)."""

    type: ClassVar[str] = "flow_time_series"
    Style: ClassVar = {
        "x": ("emphasis", "highlight"),
        "y": ("emphasis", "secondary"),
        "z": ("emphasis", "primary"),
        "axis": ("emphasis", "muted"),
        "text": ("emphasis", "muted"),
        "size": ("text", "label"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(_ChildParams, _PanelParams):
        source: str = ""
        x0: tuple[float, ...] = (1.0, 0.0)
        components: str = "xyz"
        span: float = Field(0.0, ge=0)
        labels: bool = True

        @field_validator("components")
        @classmethod
        def _components(cls, v):
            if not v or set(v) - set("xyz") or len(set(v)) != len(v):
                raise ValueError("components is some of 'xyz', each once, such as 'xz'")
            return v

        @field_validator("x0")
        @classmethod
        def _x0(cls, v):
            return _finite_vec(v, "x0")

    def _series(self, state: SceneState):
        """(resolved system, times, {name: values}, log scale, reference line or None, what)"""
        p = self.me(state).params
        src, why = None, None
        if p.source:
            src, why = _ref(state, p.source, (Probe, TwinProbes), None, "source")
            if src is None:
                r = self._sys(state)
                return None if r is None else (r, np.zeros(0), {}, False, None, None, why)
        if src is not None and isinstance(src.impl, TwinProbes):
            run = src.impl._run(state)
            if run is None:
                return None
            r, *_, n, _, _, _, dist = run
            t = np.arange(len(dist)) * r.dt * (-1 if r.tau < 0 else 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                logd = np.where(dist > 0, np.log10(dist), np.nan)
            ref = None
            if src.params.lyapunov > 0:
                res = _lyapunov(r, _seed(src.params.x0, r.dim), src.params.lyapunov,
                                src.params.transient)  # fmt: skip
                if res["value"] is not None:
                    ref = (
                        math.log10(src.params.eps),
                        res["value"] / math.log(10),
                        res["spread"] / math.log(10),
                    )
            return r, t, {"log10 d": logd}, True, ref, src.id, None
        if src is not None and isinstance(src.impl, Probe):
            run = src.impl._run(state)
            if run is None:
                return None
            r, _, _, _, path = run
            pts = path.points[0]
            what = src.id
        else:
            r = self._sys(state)
            if r is None:
                return None
            n, _ = r.steps()
            n = min(n, fm.MAX_PROBE_STEPS)
            pts = _path(r, _seed(p.x0, r.dim), n, -1.0 if r.tau < 0 else 1.0).points[0]
            what = "x0"
        t = np.arange(len(pts)) * r.dt * (-1 if r.tau < 0 else 1)
        comps = {c: pts[:, fm.COORDS.index(c)] for c in p.components
                 if fm.COORDS.index(c) < r.dim}  # fmt: skip
        return r, t, comps, False, None, what, None

    def _shown(self, state, t):
        span = self.me(state).params.span
        if span <= 0 or not len(t):
            return np.ones(len(t), dtype=bool)
        return np.abs(t) >= abs(t[-1]) - span

    def semantics(self, state: SceneState) -> dict[str, Any]:
        got = self._series(state)
        if got is None:
            return self._none("flow_time_series")
        r, t, comps, log, ref, what, why = got
        keep = self._shown(state, t)
        out = {"kind": "flow_time_series", "system": r.id, "source": what,
               "components": list(comps), "log": log, "samples": int(keep.sum()),
               "span": [_r(t[keep][0]), _r(t[keep][-1])] if keep.any() else None,
               "bad_ref": why}  # fmt: skip
        if abs(r.tau) <= fm.SAFE_TAU:
            out["range"] = {k: [_r(np.nanmin(v[keep])), _r(np.nanmax(v[keep]))]
                            if np.isfinite(v[keep]).any() else None
                            for k, v in comps.items()}  # fmt: skip
        else:
            out["range"] = None
            out["note"] = (f"values are given up to flow time {fm.SAFE_TAU:g} only (a chaotic "
                           "path is not reproducible across machines)")  # fmt: skip
        if ref is not None:  # no finer than the exponent's spread (one digit coarser)
            out["reference_slope"] = fa.statistic(ref[1], 10 * max(ref[2], 1e-7))
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height())
        w = me.style["stroke"] * unit
        self._axes(ctx, state, w * 0.7, with_alpha(me.style["axis"], p.opacity * 0.7))
        got = self._series(state)
        if got is None:
            return
        _, t, comps, _log, ref, _, _ = got
        keep = self._shown(state, t)
        t = t[keep]
        if len(t) < 2:
            return
        rect = self._rect(state)
        t0, t1 = float(t[0]), float(t[-1])
        stride = max(1, len(t) // 2000)
        vals = np.concatenate([v[keep] for v in comps.values()])
        lo, hi = _limits(vals)
        size = me.style["size"] * unit
        for name, v in comps.items():
            role = name if name in "xyz" else "x"
            col = with_alpha(me.style[role], p.opacity)
            pts = self._place(rect, _share(t, t0, t1), _share(v[keep], lo, hi))[::stride]
            for run_ in _finite_runs(pts):
                polyline(ctx, cam, run_, w, color=col, glow=me.style["glow"])
            fin = pts[np.isfinite(pts).all(axis=1)]
            if p.labels and len(fin):
                ctx.text(name, fin[-1] + cam.right * size * 0.6, size, color=col, anchor="left")
        if ref is not None:  # log10 d = log10 eps + lambda t / ln 10
            line = ref[0] + ref[1] * np.array([t0, t1])
            ends = self._place(rect, np.array([0.0, 1.0]), _share(np.clip(line, lo, hi), lo, hi))
            polyline(ctx, cam, ends, w * 0.6, color=with_alpha(me.style["text"], p.opacity * 0.8))


# ---------------------------------------------------------------- DivergenceMap

_AREA: OrderedDict[tuple, dict] = OrderedDict()
DISK_CAP = 4096  # outline points of the advected disk
DISK_STEPS = 2000  # steps it is carried (flow time 20 at dt 0.01): clamped, flow_budget


@register
class DivergenceMap(_Child):
    """div F as a heat map over the system's window (a 2D system; a 3D one at height
    ``z``): warm where the flow spreads areas out, cool where it squeezes them, dark where it
    keeps them (``counts`` grid nodes per side). With ``disk_radius`` > 0 (2D), a disk
    around ``disk_center`` is carried to the flow time as a refined outline, and semantics
    give its area against A0 exp(integral of div along its centre's path) (equal when the
    divergence is constant: Liouville). A backdrop: labels may sit on it."""

    type: ClassVar[str] = "flow_divergence"
    Style: ClassVar = {
        "positive": ("emphasis", "danger"),
        "negative": ("emphasis", "secondary"),
        "colormap": ("colormap", "spectrum"),
        "outline": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(_ChildParams, HeatParams):
        counts: int = Field(96, ge=8, le=256)
        z: float = 0.0
        disk_center: tuple[float, ...] = (1.0, 0.0)
        disk_radius: float = Field(0.0, ge=0)

        @field_validator("disk_center")
        @classmethod
        def _centre(cls, v):
            return _finite_vec(v, "disk_center")

    def _grid(self, r: Resolved, p) -> np.ndarray:
        return fa.divergence_grid(r.sys, r.vals, r.lo, r.hi, p.counts, p.z)

    def _quad(self, r: Resolved, p):
        z = [p.z] if r.dim == 3 else []
        corner = r.world(np.array([[r.lo[0], r.lo[1], *z]]))[0]
        right = r.world(np.array([[r.hi[0], r.lo[1], *z]]))[0] - corner
        up = r.world(np.array([[r.lo[0], r.hi[1], *z]]))[0] - corner
        return corner, right, up

    def world_bounds(self, state: SceneState) -> Bounds:
        r = self._sys(state)
        if r is None:
            return super().world_bounds(state)
        c, a, b = self._quad(r, self.me(state).params)
        return Bounds.of_points([c, c + a + b])

    def _disk(self, state: SceneState, r: Resolved):
        p = self.me(state).params
        if p.disk_radius <= 0 or r.dim != 2:
            return None
        c = np.asarray(_seed(p.disk_center, 2))
        loops = fm.curves("disk", c, p.disk_radius, 2)
        sim = fm.FlowSim(r.sys, r.vals, r.dt, points=np.zeros((0, 2)), loops=loops,
                         samples=64, h_max=0.01 * r.diag, refine_every=10, cap=DISK_CAP,
                         max_step=r.max_step, escape=(tuple(map(float, r.escape[0])),
                                                      r.escape[1]),
                         direction=-1.0 if r.tau < 0 else 1.0)  # fmt: skip
        n, over = r.steps()
        clamped = ["steps"] if over else []
        if n > DISK_STEPS:  # outline x steps, refinement included, stays under 1e7
            n, clamped = DISK_STEPS, [*clamped, "work"]
        st = runner_for(sim, every=CHECKPOINT_EVERY, cache_dir=None).state_at(n * sim.dt)
        sgn = -1.0 if r.tau < 0 else 1.0
        key = (r.key, tuple(c), n, sgn, *r.run_key())

        def predicted():
            path = _path(r, tuple(c), n, sgn).points[0]
            fin = path[np.isfinite(path).all(axis=1)]
            with np.errstate(all="ignore"):
                div = np.trace(r.sys.fn.jacobians(fin, r.vals), axis1=1, axis2=2)
            if len(fin) != len(path) or not np.isfinite(div).all():
                return None
            return float(np.trapezoid(div, dx=r.dt) * sgn) if len(div) > 1 else 0.0

        integral = _memo(_AREA, key, lambda: {"v": predicted()})["v"]
        return sim, st, n, clamped, integral

    def semantics(self, state: SceneState) -> dict[str, Any]:
        r = self._sys(state)
        if r is None:
            return self._none("flow_divergence")
        p = self.me(state).params
        g = self._grid(r, p)
        fin = g[np.isfinite(g)]
        div = r.sys.divergence()
        out = {"kind": "flow_divergence", "backdrop": True, "system": r.id,
               "formula": None if div is None else str(div), "counts": p.counts,
               "z": _r(p.z) if r.dim == 3 else None,
               "min": _r(fin.min()) if len(fin) else None,
               "max": _r(fin.max()) if len(fin) else None,
               "mean": _r(fin.mean()) if len(fin) else None,
               "not_finite": int((~np.isfinite(g)).sum())}  # fmt: skip
        got = self._disk(state, r)
        if got is not None:
            _, st, n, clamped, integral = got
            # A0: the polygon of the same outline parameters at flow time 0 (so the ratio
            # compares like with like: the circle's own area is a little larger)
            loop = fm.curves("disk", np.asarray(_seed(p.disk_center, 2)), p.disk_radius, 2)[0]
            a0 = fa.polygon_area(fm._on_curve(loop, st["s"]))
            disk: dict[str, Any] = {"center": _rv(_seed(p.disk_center, 2)),
                                    "radius": _r(p.disk_radius),
                                    "area0": _r(math.pi * p.disk_radius**2),
                                    "polygon0": _r(a0), "tau": _r(r.tau)}  # fmt: skip
            q = st["q"]
            whole = bool((st["qc"] == fm.ALIVE).all()) and int(st["under"]) < 0
            if abs(r.tau) <= fm.SAFE_TAU and whole:
                area = fa.polygon_area(q)
                disk["area"] = _r(area)
                disk["predicted"] = _r(a0 * math.exp(integral)) if integral is not None else None
                disk["ratio"] = (_r(area / (a0 * math.exp(integral)))
                                 if integral is not None else None)  # fmt: skip
                disk["outline"] = len(q)
                disk["predicted_by"] = ("polygon0 exp(integral of div along the centre's "
                                        "path): exact when the divergence is constant")  # fmt: skip
            else:
                disk["area"] = None
                disk["why"] = ("past flow time 10 or the outline lost points (escaped, too "
                               "fast, under-resolved): no area")  # fmt: skip
            out["disk"] = disk
            out["budget"] = _budget(points=len(st["q"]), steps=n, clamped=clamped)
        elif p.disk_radius > 0:
            out["disk"] = None
            out["why"] = "the advected disk is for 2D systems"
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        r = self._sys(state)
        if r is None:
            return
        me = self.me(state)
        p = me.params
        g = self._grid(r, p)
        fin = np.abs(g[np.isfinite(g)])
        s = float(fin.max()) if len(fin) and fin.max() > 0 else 1.0
        v = np.where(np.isfinite(g), 0.5 + 0.5 * g / s, 0.5)
        pos, neg = (np.asarray(me.style[k][:3], dtype=np.float64) for k in ("positive", "negative"))

        def colour(x):
            return np.where((x >= 0.5)[..., None], pos, neg)

        if p.hue_cycles > 0:
            colour = wheel_colour(me.style["colormap"], p)
        corner, right, up = self._quad(r, p)
        size = (float(np.linalg.norm(right)), float(np.linalg.norm(up)))
        glow = float(me.style.get("glow", 1.0))
        pixels = heat_pixels(v, size, colour, how="smooth" if p.smooth else "linear",
                             weight=lambda x: np.abs(2 * x - 1), background=state.background,
                             **p.drawing())  # fmt: skip
        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity, glow=glow)
        got = self._disk(state, r)
        if got is not None:
            _, st, _, _, _ = got
            q = st["q"]
            if len(q) > 2:
                cam = state.camera_for(me)
                w = me.style["stroke"] * self._unit(state, me, r)
                ring = np.vstack([q, q[:1]])
                for run_ in _finite_runs(ring):
                    polyline(ctx, cam, r.world(run_), w,
                             color=with_alpha(me.style["outline"], p.opacity))  # fmt: skip


# ---------------------------------------------------------------- BifurcationDiagram

_SWP: OrderedDict[tuple, dict] = OrderedDict()
_SHORT = {
    "onset of chaos or a long period": "chaos",
    "period doubling": "doubling",
    "period halving": "halving",
    "enters or leaves the window": "edge",
    "a periodic window": "window",
}  # fmt: skip  (drawn labels; semantics keep the words)
GRID_2D = 27  # Newton's grid per side in a sweep: 729 starts
GRID_3D = 9  # 729 starts


@register
class BifurcationDiagram(_Panel, _Child):
    """A variable of the system (``variable``) swept over ``range`` in ``count`` values, as
    a panel (``pos``, ``extent``). ``mode="equilibria"``: every equilibrium's ``coord`` at
    each value, stable ones in the stable colour, and the candidate bifurcations (where the
    count changes or an eigenvalue crosses the imaginary axis) refined by bisection and
    labelled by ``explore_math.sweep`` (the one equilibria sweep): ``saddle-node``,
    ``pitchfork``, ``transcritical``, ``Hopf``, ``enters or leaves the window`` or ``not
    named``; at most ``explore_math.MAX_SWEEP`` values and a Newton grid of at most
    ``GRID_2D`` (``GRID_3D`` in 3D) per side (clamped, ``flow_budget``).
    ``mode="attractor"``: from ``x0``, after ``transient`` flow time, the samples of
    ``sample_time`` (the maxima of ``coord``, or crossings of the plane ``normal``/``offset``
    with ``sample="section"``): a period-doubling cascade, with its period and state per
    value (``periodic``, ``not settled``, ``chaos or a long period``). Not detected: global
    bifurcations (homoclinic, heteroclinic, crises) and anything between swept values
    (``flow_analysis.NOT_DETECTED``). At most ``flow_analysis_math.MAX_SWEEP`` values and
    ``MAX_SWEEP_WORK`` value-steps. A variable the system does not use draws the panel and
    says why."""

    type: ClassVar[str] = "flow_bifurcation"
    Style: ClassVar = {
        "stable": ("emphasis", "secondary"),
        "unstable": ("emphasis", "danger"),
        "neutral": ("emphasis", "primary"),
        "points": ("emphasis", "highlight"),
        "axis": ("emphasis", "muted"),
        "text": ("emphasis", "muted"),
        "size": ("text", "label"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams, _PanelParams):
        variable: str = ""
        range: tuple[float, float] = (0.0, 1.0)
        count: int = Field(48, ge=2, le=fa.MAX_SWEEP)
        mode: Literal["equilibria", "attractor"] = "equilibria"
        coord: Literal["x", "y", "z"] = "x"
        sample: Literal["maxima", "section"] = "maxima"
        normal: tuple[float, ...] = (0.0, 1.0, 0.0)
        offset: float = 0.0
        x0: tuple[float, ...] = (0.1, 0.1)
        transient: float = Field(50.0, ge=0, le=1e4)
        sample_time: float = Field(30.0, gt=0, le=1e4)
        grid: int = Field(9, ge=2, le=64)
        marks: bool = True
        radius: float = Field(0.004, gt=0, le=0.05)

        @model_validator(mode="after")
        def _check(self):
            lo, hi = self.range
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(f"range is (lo, hi), finite with lo < hi, not {self.range}")
            if not math.isfinite(hi - lo):
                raise ValueError(f"the range's span hi - lo must be finite, not {self.range}")
            _finite_vec(self.x0, "x0")
            _finite_vec(self.normal, "normal")
            if not any(self.normal):
                raise ValueError("a section's normal must not be zero")
            return self

    def _sweep(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None
        p = self.me(state).params
        if p.variable not in r.values:
            return (
                r,
                None,
                0,
                f"the system has no variable {p.variable!r}; it has {sorted(r.values)}",
            )
        k = fm.COORDS.index(p.coord)
        if k >= r.dim:
            k = 0
        others = {n: v for n, v in r.values.items() if n != p.variable}
        if p.mode == "attractor":
            normal = np.asarray(_seed(p.normal, r.dim))
            sampler = (fa.Sampler.maxima(k) if p.sample == "maxima" or not normal.any()
                       else fa.Sampler.plane(normal, p.offset, +1))  # fmt: skip
            extra = (sampler.key(), _seed(p.x0, r.dim), p.transient, p.sample_time)
        else:
            sampler = None
            extra = (tuple(r.lo), tuple(r.hi), self._grid(p, r)[0])
        base = r.sys.key({**others, p.variable: 0.0})  # the exprs and every other value
        key = (base, p.variable, tuple(p.range), p.count, p.mode, *extra, *r.run_key(), r.diag)

        def make():
            try:
                return fa.sweep(r.sys, {**others, p.variable: 0.0}, p.variable, p.range[0],
                                p.range[1], p.count, mode=p.mode, lo=r.lo, hi=r.hi,
                                grid=self._grid(p, r)[0], x0=_seed(p.x0, r.dim), dt=r.dt,
                                transient=p.transient, sample_time=p.sample_time,
                                sampler=sampler, escape=r.escape, max_step=r.max_step,
                                scale=r.diag)  # fmt: skip
            except fm.FunctionError as e:
                return {"why": str(e)}

        sw = _memo(_SWP, key, make)
        if "why" in sw:
            return r, None, k, sw["why"]
        return r, sw, k, None

    @staticmethod
    def _grid(p, r: Resolved) -> tuple[int, bool]:
        """Newton's grid per side, at most GRID_2D in 2D and GRID_3D in 3D (a sweep runs it
        at every value and at every bisection step)."""
        most = GRID_3D if r.dim == 3 else GRID_2D
        return min(p.grid, most), p.grid > most

    def semantics(self, state: SceneState) -> dict[str, Any]:
        got = self._sweep(state)
        if got is None:
            return self._none("flow_bifurcation")
        r, sw, _, why = got
        p = self.me(state).params
        out: dict[str, Any] = {"kind": "flow_bifurcation", "system": r.id,
                               "variable": p.variable, "range": _rv(p.range),
                               "count": p.count, "mode": p.mode, "coord": p.coord,
                               "not_detected": NOT_DETECTED}  # fmt: skip
        if sw is None:
            out.update(rows=[], candidates=[], drawn=0.0, why=why)
            return out
        clamped = list(sw["clamped"])
        if sw["mode"] == "equilibria":
            grid, cut = self._grid(p, r)
            if cut:
                clamped.append("grid")
            out["columns"] = ["value", "count", "stable", "max_real", "non_isolated"]
            out["rows"] = [[_r(row["value"]), row["count"], row["stable"],
                            None if row["max_real"] is None else _r(row["max_real"]),
                            bool(row["non_isolated"])] for row in sw["rows"]]  # fmt: skip
            out["grid"] = grid
            # each value and each bisection step is a Newton run from grid^dim starts
            points, steps = len(sw["rows"]), grid**r.dim * (1 + 40 * len(sw["candidates"]))
        else:
            out["columns"] = ["value", "period", "state", "samples"]
            # a chaotic row's count of samples depends on the machine: None with its period
            out["rows"] = [[_r(row["value"]), row["period"], row["state"],
                            None if row["period"] is None else row["samples"]]
                           for row in sw["rows"]]  # fmt: skip
            out["transient"] = _r(sw["transient"])
            out["periods_note"] = ("a period is the number of windings after which the samples "
                                   "repeat within 1e-5 of the window (flow_analysis_math."
                                   "returns_repeat); not settled: still closing in (near a "
                                   "doubling, a longer transient helps)")  # fmt: skip
            points, steps = max(len(sw["rows"]), fa.LOOP_POINTS), sw["steps"]
        width = (p.range[1] - p.range[0]) * 1e-6
        out["candidates"] = [
            {"label": c["label"], "change": c.get("change"), "converged": c.get("converged"),
             "value": None if c["value"] is None else fa.statistic(c["value"], width),
             "between": [fa.statistic(c["between"][0], width),
                         fa.statistic(c["between"][1], width)]}
            for c in sw["candidates"]]  # fmt: skip
        out["budget"] = _budget(points=points, steps=steps, clamped=clamped)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height())
        w = me.style["stroke"] * unit
        self._axes(ctx, state, w * 0.7, with_alpha(me.style["axis"], p.opacity * 0.7))
        got = self._sweep(state)
        if got is None or got[1] is None:
            return
        r, sw, k, _ = got
        rect = self._rect(state)
        lo_v, hi_v = p.range
        dot = unit * p.radius
        if sw["mode"] == "equilibria":
            lo, hi = float(r.lo[k]), float(r.hi[k])
            xs, ys, roles = [], [], []
            for row in sw["rows"]:
                for e in row["equilibria"]:
                    s = fa._stable_kind(e["kind"])
                    if e["at"][k] is None:
                        continue
                    xs.append(row["value"])
                    ys.append(e["at"][k])
                    roles.append("stable" if s is True else "unstable" if s is False else "neutral")
            if xs:
                pts = self._place(rect, _share(xs, lo_v, hi_v), _share(ys, lo, hi))
                cols = np.array([with_alpha(me.style[ro], p.opacity) for ro in roles])
                disc_batch(ctx, cam, pts, dot * 1.6, colors=cols, glow=me.style["glow"])
        else:
            smp = sw["samples"]
            lo, hi = _limits(smp)
            vv = np.repeat(sw["values"], smp.shape[1])
            ok = np.isfinite(smp.ravel())
            if ok.any():
                pts = self._place(rect, _share(vv[ok], lo_v, hi_v), _share(smp.ravel()[ok], lo, hi))
                disc_batch(ctx, cam, pts, dot, colors=with_alpha(me.style["points"], p.opacity),
                           glow=me.style["glow"])  # fmt: skip
        if p.marks:
            size = me.style["size"] * unit
            col = with_alpha(me.style["text"], p.opacity * 0.8)
            last = -1.0  # a label only where it has room (the lines are all drawn)
            room = 1.2 * size * 10 / max(abs(rect[3]), 1e-12)  # about ten characters
            for c in sorted(sw["candidates"], key=lambda c: c["between"][0]):
                at = c["value"] if c["value"] is not None else 0.5 * sum(c["between"])
                u = float(_share(at, lo_v, hi_v))
                ends = self._place(rect, np.array([u, u]), np.array([0.0, 1.0]))
                polyline(ctx, cam, ends, w * 0.5, color=col)
                if u - last >= room:
                    ctx.text(_SHORT.get(c["label"], c["label"]), ends[1] + cam.up * size * 0.3,
                             size, color=col)  # fmt: skip
                    last = u


# ---------------------------------------------------------------- LimitCycle

_CYC: OrderedDict[tuple, dict] = OrderedDict()


@register
class LimitCycle(_Child):
    """The attracting cycle reached from ``x0``: after ``transient`` flow time, returns to
    the plane through the point across the flow until two in a row agree within ``tol``
    (times the window's diagonal) and so do their times; at most ``max_time`` of flow time.
    Draws one period of the cycle (``progress`` 0 -> 1 traces it) and the point on the plane;
    a rest point it settles on instead is drawn as a dot. Semantics: ``found`` (``cycle``,
    ``equilibrium``, ``stopped`` or ``none``), the period, the point, ``converged`` and the
    returns; numbers rounded to ``tol`` (an attracting cycle forgets rounding, so they are
    reproducible to it). A centre's closed orbits also return: whether the cycle is isolated
    is not checked."""

    type: ClassVar[str] = "flow_limit_cycle"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "dot": ("emphasis", "primary"),
        "stroke": ("stroke", "bold"),
        "glow": ("glow", "soft"),
    }

    class Params(_ChildParams):
        x0: tuple[float, ...] = (1.0, 0.0)
        transient: float = Field(50.0, ge=0, le=1e4)
        max_time: float = Field(100.0, gt=0, le=1e4)
        tol: float = Field(1e-6, gt=0, le=1e-5)  # looser: chaos passes for a cycle
        progress: float = Field(1.0, ge=0, le=1)

        @field_validator("x0")
        @classmethod
        def _x0(cls, v):
            return _finite_vec(v, "x0")

    def _cycle(self, state: SceneState):
        r = self._sys(state)
        if r is None:
            return None
        p = self.me(state).params
        x0 = _seed(p.x0, r.dim)
        key = (r.key, x0, p.transient, p.max_time, p.tol, r.diag, *r.run_key())
        return r, _memo(_CYC, key, lambda: fa.limit_cycle(
            r.sys, np.asarray(x0), values=r.vals, dt=r.dt, transient=p.transient,
            max_time=p.max_time, tol=p.tol, scale=r.diag, escape=r.escape,
            max_step=r.max_step))  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        got = self._cycle(state)
        if got is None:
            return self._none("flow_limit_cycle")
        r, c = got
        p = self.me(state).params
        # fixed-step RK4: the period and the points are accurate to about dt^2 relative (the
        # tolerance only says the returns repeat), so no digit finer than that is shown
        acc = max(p.tol, r.dt * r.dt)
        spread = acc * r.diag
        out = {"kind": "flow_limit_cycle", "system": r.id, "x0": _rv(_seed(p.x0, r.dim)),
               "found": c["kind"], "converged": c["converged"], "loops": c["loops"],
               "note": c["note"],
               # a count of returns that never settled depends on the chaos: not reported
               "returns": c["returns"] if c["kind"] in ("cycle", "equilibrium") else None,
               "period": None if c["period"] is None else
               fa.statistic(c["period"], acc * max(c["period"], 1.0)),
               "period_accuracy": (f"about {acc:.0e} relative: limited by dt "
                                   f"({r.dt:g}, fixed-step RK4) and tol; halve dt to check"),
               "point": None if c["point"] is None else _stat_list(c["point"], spread),
               "box": None if c["box"] is None else [_stat_list(b, spread) for b in c["box"]],
               "isolated": "not checked (a drift of the returns is: a centre is not a cycle)",
               "budget": _budget(points=fa.LOOP_POINTS, steps=c["steps"],
                                 clamped=c["clamped"])}  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        got = self._cycle(state)
        if got is None:
            return
        r, c = got
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = self._unit(state, me, r)
        if c["path"] is not None and p.progress > 0:
            pts = c["path"]
            k = max(2, round(p.progress * len(pts)))
            col = with_alpha(me.style["color"], p.opacity)
            polyline(ctx, cam, r.world(pts[:k]), me.style["stroke"] * unit, color=col,
                     glow=me.style["glow"])  # fmt: skip
        if c["point"] is not None:
            disc_batch(ctx, cam, r.world(np.asarray(c["point"])[None]), unit * 0.012,
                       colors=with_alpha(me.style["dot"], p.opacity), glow=2.0)  # fmt: skip


# ---------------------------------------------------------------- the tools' function


def lyapunov(system, x0, T: float = 100.0, *, state: SceneState | None = None, values=(),
             transient: float = 10.0, dt: float | None = None, **kw) -> dict[str, Any]:  # fmt: skip
    """The largest Lyapunov exponent from ``x0`` over ``T`` flow time (Benettin): a
    ``flow_math.System`` with its ``values``, or the id of a document's ``System`` with the
    ``state`` (``Engine.state_at(t)``), whose variables, ``dt``, escape radius and step bound
    are used. Returns plain data: ``value`` and ``spread`` (a statistic, rounded to its
    spread), ``T``, ``converged``, ``chaotic_beyond`` (ln(1e16) / value: the flow time after
    which a path is not reproducible across machines; None unless positive and converged),
    ``stopped``."""
    if isinstance(system, str):
        if state is None or system not in state or not isinstance(state[system].impl, System):
            raise ValueError(f"no System {system!r} in the state")
        r = state[system].impl.resolve(state)
        res = fa.lyapunov(r.sys, np.asarray(x0, dtype=np.float64), T, values=r.vals,
                          dt=r.dt if dt is None else dt, transient=transient, escape=r.escape,
                          max_step=r.max_step, **kw)  # fmt: skip
    else:
        res = fa.lyapunov(system, np.asarray(x0, dtype=np.float64), T, values=values,
                          dt=0.01 if dt is None else dt, transient=transient, **kw)  # fmt: skip
    return _lyapunov_semantic(res)
