"""The Explorer's analysis pack (Milestone 30, EXPLORER.md; advice section 2.6 rows 5, 7-12):
numbers about a flow, from the flow core (``flow_math``). Pure NumPy, no drawing.

- ``Sampler``: where a path is sampled: crossings of a plane ``n . p = offset`` (a Poincare
  section) or the local maxima of one coordinate (its velocity crossing 0 downward). A
  crossing inside a step is found by linear interpolation of the sampler's value between
  the two ends of the step.
- ``SectionRun``: the crossings of seeds advected together (any number, batch independent),
  continued step by step, so a longer flow time extends the answer instead of redoing it.
- ``lyapunov``: the largest Lyapunov exponent by Benettin's two-trajectory method (a second
  point ``d0`` away, renormalised every ``renorm`` flow time), its spread over blocks and a
  ``converged`` flag; ``chaotic_beyond`` = ln(1e16) / lambda, the flow time after which a
  path is not reproducible across machines. Deterministic: the first separation direction is
  a hash of the system and the start, never a random number.
- ``limit_cycle``: settle for ``transient``, then return to the plane through the point,
  across the flow, until two returns agree within ``tol``: the period, a point of the cycle
  and one period of path. An equilibrium (no motion) and an escape are reported as such.
- ``sweep``: one variable over a range (at most ``MAX_SWEEP`` values). ``equilibria``: per
  value the count, the largest real part of an eigenvalue and the stable count; candidates
  where they change, refined by bisection and labelled by the local test (saddle-node,
  pitchfork, transcritical, Hopf, or a window edge). ``attractor``: every value advanced at
  once (a variable may be an array, one value per point), sampled after a transient; the
  period is the number of clusters of the samples, None for chaos or a long period.

Not detected (say so wherever a sweep is shown): global bifurcations (homoclinic and
heteroclinic loops, saddle-node of cycles, crises), bifurcations of cycles except the
period changes the samples show (never refined), anything outside the window or that falls
between the grid of Newton starts or between two swept values.

Determinism: as the flow core. Numbers after flow time ``flow_math.SAFE_TAU`` from a chaotic
path are statistics (a Lyapunov exponent agrees across machines within its spread, not
bit for bit); an attracting cycle forgets rounding, so its period is reproducible to its
tolerance.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phasor.domains import flow_math as fm

__all__ = [
    "Sampler",
    "SectionRun",
    "divergence_grid",
    "limit_cycle",
    "lyapunov",
    "polygon_area",
    "sweep",
]  # fmt: skip

# Budgets (clamped and linted ``flow_budget``, never refused)
MAX_SECTION_SEEDS = 4096
MAX_SECTION_WORK = 2e7  # seeds x steps of one section
MAX_CROSSINGS = 200_000  # crossings kept by one section (the first ones)
MAX_LYAPUNOV_STEPS = 50_000  # steps of the two Benettin trajectories (transient included)
MAX_CYCLE_STEPS = 100_000
MAX_RETURNS = 64
MAX_SWEEP = 256  # values of a sweep
MAX_SWEEP_WORK = 1e7  # values x steps of an attractor sweep
MAX_SAMPLES = 64  # samples kept per value (the last ones)
MAX_CANDIDATES = 8  # candidate bifurcations refined per sweep
BISECT_ITERS = 24
LN_1E16 = math.log(1e16)  # about 36.8: e-foldings from rounding (1e-16) to order one


# ------------------------------------------------------------------ samplers


@dataclass(frozen=True)
class Sampler:
    """Where a path is sampled. ``kind="plane"``: the value is ``normal . p - offset`` (a
    Poincare section); ``kind="maxima"``: the value is the velocity of coordinate ``coord``,
    whose downward crossings are the local maxima of that coordinate. ``direction``: +1
    counts upward crossings only, -1 downward only, 0 both."""

    kind: str = "plane"
    normal: tuple[float, ...] = (1.0, 0.0, 0.0)
    offset: float = 0.0
    coord: int = 0
    direction: int = 1

    @classmethod
    def plane(cls, normal, offset: float = 0.0, direction: int = 1) -> Sampler:
        n = np.asarray(normal, dtype=np.float64).ravel()
        size = float(np.linalg.norm(n))
        if not (np.isfinite(n).all() and size > 0):
            raise ValueError("a section's normal must be finite and not zero")
        return cls("plane", tuple(float(v) for v in n / size), float(offset) / size, 0,
                   int(np.sign(direction)))  # fmt: skip

    @classmethod
    def maxima(cls, coord: int) -> Sampler:
        return cls("maxima", (), 0.0, int(coord), -1)

    def values(self, fn: fm.Compiled, vals, p: np.ndarray) -> np.ndarray:
        """The sampler's value at points (N, dim)."""
        if self.kind == "maxima":
            with np.errstate(all="ignore"):
                return fn.velocity(p, vals)[:, self.coord]
        n = np.asarray(self.normal[: p.shape[1]], dtype=np.float64)
        n = np.r_[n, np.zeros(p.shape[1] - len(n))]
        with np.errstate(all="ignore"):
            return p @ n - self.offset

    def key(self) -> tuple:
        return (self.kind, self.normal, self.offset, self.coord, self.direction)


def crossed(g0: np.ndarray, g1: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Which steps cross zero (from ``g0`` to ``g1``) in ``direction`` and where inside the
    step (the share 0..1, by linear interpolation of the value). A step that starts on the
    plane does not count again (``g0 < 0 <= g1`` upward, ``g0 > 0 >= g1`` downward)."""
    with np.errstate(invalid="ignore"):
        up = (g0 < 0) & (g1 >= 0)
        down = (g0 > 0) & (g1 <= 0)
    hit = up if direction > 0 else down if direction < 0 else up | down
    with np.errstate(all="ignore"):
        f = np.where(hit, g0 / np.where(hit, g0 - g1, 1.0), 0.0)
    return hit, np.clip(f, 0.0, 1.0)


# ------------------------------------------------------------------ sections


@dataclass
class SectionRun:
    """Crossings of ``sampler`` by seeds (S, dim) advected together, from step 0 on.
    ``advance(n)`` continues to step ``n`` (the same steps whatever was asked before, so the
    answer is a function of the inputs). Kept: ``points`` (K, dim) where each crossing is,
    ``times`` (step + share of a step, times ``dt`` for flow time), ``seed`` which seed
    crossed; at most ``MAX_CROSSINGS`` (the first ones, then ``full``)."""

    fn: fm.Compiled
    vals: tuple
    seeds: np.ndarray
    dt: float
    sampler: Sampler
    escape: tuple | None = None
    max_step: float | None = None
    n: int = 0
    p: np.ndarray = field(default=None)  # type: ignore[assignment]
    code: np.ndarray = field(default=None)  # type: ignore[assignment]
    g: np.ndarray = field(default=None)  # type: ignore[assignment]
    found: list = field(default_factory=list)  # (points, times, seed) chunks
    count: int = 0
    full: bool = False

    def __post_init__(self) -> None:
        self.seeds = np.asarray(self.seeds, dtype=np.float64).reshape(len(self.seeds), -1)
        if self.p is None:
            self.p = self.seeds.copy()
            self.code = np.zeros(len(self.seeds), dtype=np.int8)
            self.g = self.sampler.values(self.fn, self.vals, self.p)

    def advance(self, n: int) -> SectionRun:
        while self.n < n and (self.code == fm.ALIVE).any():
            alive = self.code == fm.ALIVE
            q, c = _advance(self.fn, self.vals, np.where(alive[:, None], self.p, 0.0), self.dt,
                            self.escape, self.max_step)  # fmt: skip
            c = np.where(alive, c, self.code).astype(np.int8)
            ok = alive & (c == fm.ALIVE)
            safe = np.where(ok[:, None], q, 0.0)
            g1 = np.where(ok, self.sampler.values(self.fn, self.vals, safe), np.nan)
            hit, f = crossed(self.g, g1, self.sampler.direction)
            hit &= ok
            if hit.any() and not self.full:
                idx = np.flatnonzero(hit)
                fr = f[idx][:, None]
                pts = self.p[idx] + fr * (q[idx] - self.p[idx])
                room = MAX_CROSSINGS - self.count
                if len(idx) > room:
                    idx, pts, fr, self.full = idx[:room], pts[:room], fr[:room], True
                self.found.append((pts, self.n + fr[:, 0], idx))
                self.count += len(idx)
            self.p = np.where(ok[:, None], q, self.p)
            self.code = c
            self.g = g1
            self.n += 1
        if self.n < n:  # every seed stopped: nothing more can cross
            self.n = n
        return self

    def crossings(self, upto: int | None = None) -> dict[str, np.ndarray]:
        """The crossings so far (or up to step ``upto``), in the order they happened."""
        d = self.seeds.shape[1]
        if not self.found:
            return {"points": np.zeros((0, d)), "times": np.zeros(0),
                    "seed": np.zeros(0, dtype=np.int64)}  # fmt: skip
        pts = np.concatenate([a for a, _, _ in self.found])
        tim = np.concatenate([b for _, b, _ in self.found])
        sid = np.concatenate([c for _, _, c in self.found])
        if upto is not None:
            keep = tim <= upto
            pts, tim, sid = pts[keep], tim[keep], sid[keep]
        return {"points": pts, "times": tim * self.dt, "seed": sid}


# ------------------------------------------------------------------ Lyapunov


def _advance(fn: fm.Compiled, vals, p: np.ndarray, dt: float, escape, max_step):
    """One step of every point (``flow_math.step``) and the escape radius: (q, code)."""
    q, c = fm.step(fn, vals, p, dt, max_step)
    if escape is not None:
        with np.errstate(invalid="ignore"):
            far = np.linalg.norm(q - np.asarray(escape[0]), axis=1) > escape[1]
        c = np.where((c == fm.ALIVE) & far, fm.ESCAPED, c)
    return q, c


def _direction(key: str, dim: int) -> np.ndarray:
    """A unit vector from a hash (no random numbers): the first separation of the twins."""
    h = hashlib.sha256(key.encode()).digest()
    v = np.array([int.from_bytes(h[4 * k : 4 * k + 4], "little") / 2**32 - 0.5
                  for k in range(dim)])  # fmt: skip
    if np.linalg.norm(v) < 1e-6:
        v = np.ones(dim)
    return v / np.linalg.norm(v)


def separation(system: fm.System, values, x0) -> np.ndarray:
    """The deterministic first separation direction of two twins at ``x0``."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    key = repr(([str(e) for e in system.exprs], [float(v) for v in vals],
                [float(v) for v in np.ravel(x0)]))  # fmt: skip
    return _direction(key, system.dim)


def statistic(value: float, spread: float) -> float | None:
    """``value`` rounded to the decade of its ``spread`` (one digit more): what a statistic
    from a long chaotic run can promise across machines. None when not finite."""
    if value is None or not math.isfinite(value):
        return None
    if not (spread is not None and math.isfinite(spread) and spread > 0):
        return round(float(value), 6)
    digits = max(0, min(6, -math.floor(math.log10(spread)) + 1))
    out = round(float(value), digits)
    return 0.0 if out == 0 else out


def lyapunov(system: fm.System, x0, T: float = 100.0, *, values=(), dt: float = 0.01,
             transient: float = 10.0, renorm: float = 0.5, blocks: int = 8,
             d0: float | None = None, escape=None, max_step: float | None = None,
             atol: float = 0.01, rtol: float = 0.05) -> dict[str, Any]:  # fmt: skip
    """The largest Lyapunov exponent along the path from ``x0`` (Benettin), the system's
    variables at ``values`` (a name -> value table or the values in order): after
    ``transient`` flow time, a twin ``d0`` away (default ``1e-8 * max(1, |x|)``) in a
    direction from a hash; every ``renorm`` flow time the log of the stretch is added and the
    twin is put back at ``d0`` along the separation. ``T`` flow time is split into ``blocks``
    of equal length; ``value`` is the mean rate, ``spread`` the standard error over blocks,
    ``converged`` when the spread is under ``max(atol, rtol * |value|)``. ``chaotic_beyond``:
    ln(1e16) / value for a positive converged value (else None). A path that stops (escape,
    not finite, too fast) gives ``value`` None and says why (``stopped``)."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    fn = system.fn
    dt = abs(float(dt))
    every = max(1, round(renorm / dt))
    nb = max(2, int(blocks))
    per = max(1, round(T / (dt * every * nb)))  # renormalisations per block
    n_tr = max(0, round(transient / dt))
    clamped = []
    total = n_tr + nb * per * every
    if total > MAX_LYAPUNOV_STEPS:  # the transient takes at most half, the blocks the rest
        n_tr = min(n_tr, MAX_LYAPUNOV_STEPS // 2)
        per = max(1, (MAX_LYAPUNOV_STEPS - n_tr) // (nb * every))
        clamped.append("steps")
    p = np.asarray(x0, dtype=np.float64).reshape(1, -1)
    out: dict[str, Any] = {"value": None, "spread": None, "converged": False, "T": None,
                           "blocks": [], "chaotic_beyond": None, "stopped": None,
                           "steps": n_tr + nb * per * every, "clamped": clamped,
                           "renorm": every * dt, "transient": n_tr * dt}  # fmt: skip

    def _stop(code) -> bool:
        if (code != fm.ALIVE).any():
            out["stopped"] = {fm.NONFINITE: "not finite", fm.TOO_FAST: "too fast",
                              fm.ESCAPED: "escaped"}[int(code[code != fm.ALIVE][0])]  # fmt: skip
            return True
        return False

    def _go(q):
        return _advance(fn, vals, q, dt, escape, max_step)

    def _size(at: np.ndarray) -> float:  # relative to where the path is (a path that grows
        # must not lose its twin below rounding)
        return d0 if d0 is not None else 1e-8 * max(1.0, float(np.linalg.norm(at)))

    # The twin rides along through the transient too (renormalised, not counted), so its
    # separation has turned to the fastest-growing direction before counting starts.
    x = p[0]
    u = separation(system, vals, np.asarray(x0, dtype=np.float64))
    pair = np.stack([x, x + _size(x) * u])
    start = float(np.linalg.norm(pair[1] - pair[0]))  # the separation as it is in floats
    sums = np.zeros(nb)
    warm = -(-n_tr // every)  # renormalisations in the transient
    for k in range(warm + nb * per):
        for _ in range(every):
            pair, c = _go(pair)
            if _stop(c):
                return out
        sep = pair[1] - pair[0]
        d = float(np.linalg.norm(sep))
        if not (math.isfinite(d) and d > 0 and start > 0):
            out["stopped"] = "the twins met"  # a separation of exactly 0: no direction
            return out
        if k >= warm:
            sums[(k - warm) // per] += math.log(d / start)
        pair[1] = pair[0] + sep * (_size(pair[0]) / d)
        start = float(np.linalg.norm(pair[1] - pair[0]))
    span = per * every * dt
    rates = sums / span
    value = float(rates.mean())
    spread = float(rates.std(ddof=1) / math.sqrt(nb))
    out.update(value=value, spread=spread, T=nb * span, blocks=[float(r) for r in rates],
               converged=bool(spread <= max(atol, rtol * abs(value))))  # fmt: skip
    if out["converged"] and value > atol:
        out["chaotic_beyond"] = LN_1E16 / value
    return out


# ------------------------------------------------------------------ limit cycles


MAX_LOOPS = 8  # a cycle that winds this many times before it closes (period doubling)


def _loops(rets: list[tuple[np.ndarray, float]], tol: float, scale: float) -> int:
    """The fewest loops m (1..MAX_LOOPS) after which the returns repeat: the last two returns
    each within ``tol * scale`` of the one m before, the two periods within ``tol``; 0 if
    none yet."""
    for m in range(1, MAX_LOOPS + 1):
        if len(rets) < m + 2:
            break
        (q1, t1), (q0, t0) = rets[-1], rets[-1 - m]
        (p1, s1), (p0, s0) = rets[-2], rets[-2 - m]
        if (float(np.linalg.norm(q1 - q0)) <= tol * scale
                and float(np.linalg.norm(p1 - p0)) <= tol * scale
                and abs((t1 - t0) - (s1 - s0)) <= tol * max(t1 - t0, 1e-12)):  # fmt: skip
            return m
    return 0


def _cross(fn, vals, p: np.ndarray, n: np.ndarray, off: float, dt: float, g0: float,
           g1: float) -> tuple[float, np.ndarray]:  # fmt: skip
    """Where a step from ``p`` crosses the plane, to rounding: the share ``h`` of the step
    with ``n . RK4(p, h dt) = off`` (secant steps from the linear guess, kept inside the
    bracket), and the point. A cycle's returns are then compared to ``tol``, which a linear
    guess (wrong by about dt^2) would not allow."""

    def gap(h: float) -> tuple[float, np.ndarray]:
        q = fm._rk4(fn, vals, p[None], h * dt)[0] if h > 0 else p
        return float(q @ n) - off, q

    a, b, ga, gb = 0.0, 1.0, g0, g1
    h = ga / (ga - gb)
    q = p
    for _ in range(30):
        gh, q = gap(h)
        if abs(gh) <= 1e-14 * max(1.0, abs(off)):
            break
        if gh < 0:
            a, ga = h, gh
        else:
            b, gb = h, gh
        h = a + (b - a) * ga / (ga - gb) if ga != gb else 0.5 * (a + b)
        if b - a < 1e-15:
            break
    return h, q


def limit_cycle(system: fm.System, x0, *, values=(), dt: float = 0.01, transient: float = 50.0,
                max_time: float = 100.0, tol: float = 1e-6, scale: float = 1.0,
                escape=None, max_step: float | None = None) -> dict[str, Any]:  # fmt: skip
    """An attracting cycle reached from ``x0``: after ``transient``, the plane through the
    point across the flow; returns to it (upward, within a quarter of ``scale`` of the
    point) until the returns repeat (``_loops``: after ``loops`` windings, 2 or more past a
    period doubling) within ``tol * scale``, their times within ``tol`` of a period.
    ``kind``: ``cycle`` (``period`` (all its loops), ``point`` on the plane, ``path`` one
    period, ``box`` its extent), ``equilibrium`` (the path stopped moving: ``point``),
    ``stopped`` (escape, not finite, too fast) or ``none`` (no settled return in
    ``max_time``: chaos, a torus, a slow approach). A centre's closed orbits also return:
    whether the cycle is isolated is not checked."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    fn = system.fn
    dt = abs(float(dt))
    n_tr = min(round(transient / dt), MAX_CYCLE_STEPS)
    n_max = min(round(max_time / dt), MAX_CYCLE_STEPS - n_tr)
    clamped = [] if round(transient / dt) + round(max_time / dt) <= MAX_CYCLE_STEPS else ["steps"]
    out: dict[str, Any] = {"kind": "none", "period": None, "point": None, "converged": False,
                           "returns": 0, "path": None, "box": None, "gap": None, "loops": None,
                           "steps": n_tr + n_max, "clamped": clamped}  # fmt: skip
    p, code = np.asarray(x0, dtype=np.float64).reshape(1, -1), np.zeros(1, np.int8)
    for _ in range(n_tr):
        p, code = _advance(fn, vals, p, dt, escape, max_step)
        if code[0] != fm.ALIVE:
            out["kind"] = "stopped"
            return out
    p0 = p[0]
    v0 = fn.velocity(p0[None], vals)[0]
    speed = float(np.linalg.norm(v0))
    big = max(1.0, float(np.linalg.norm(p0)))
    if not math.isfinite(speed) or speed < 1e-9 * big:
        out.update(kind="equilibrium", point=p0, converged=True)
        return out
    n = v0 / speed
    off = float(np.dot(n, p0))
    near = 0.25 * scale
    rets: list[tuple[np.ndarray, float]] = []
    g = 0.0
    for k in range(n_max):
        q, code = _advance(fn, vals, p, dt, escape, max_step)
        if code[0] != fm.ALIVE:
            out["kind"] = "stopped"
            return out
        g1 = float(q[0] @ n) - off
        if g < 0 <= g1 and float(np.linalg.norm(q[0] - p0)) <= near:
            h, at = _cross(fn, vals, p[0], n, off, dt, g, g1)
            rets.append((at, (k + h) * dt))
            m = _loops(rets, tol, scale)
            if m:
                out.update(converged=True, loops=m)
                break
            if len(rets) >= MAX_RETURNS:
                break
        p, g = q, g1
    out["returns"] = len(rets)
    if len(rets) < 2 or not out["converged"]:
        end = p[0]  # settling onto a rest point (a slow spiral returns, shrinking)
        if float(np.linalg.norm(fn.velocity(end[None], vals)[0])) < 1e-5 * big:
            out.update(kind="equilibrium", point=end, converged=True)
            return out
    m = out.get("loops") or 1
    if len(rets) > m:
        out["gap"] = float(np.linalg.norm(rets[-1][0] - rets[-1 - m][0]))
    if not out["converged"]:  # returns that do not settle: chaos, a torus, a slow approach
        return out
    (_, tb), (qc, tc) = rets[-1 - m], rets[-1]
    period = tc - tb
    one = fm.integrate(fn, vals, qc[None], min(max(2, round(period / dt)), fm.MAX_PROBE_STEPS),
                       dt, escape=escape, max_step=max_step).points[0]  # fmt: skip
    one = one[np.isfinite(one).all(axis=1)]
    out.update(kind="cycle", period=period, point=qc, path=one,
               box=[one.min(axis=0), one.max(axis=0)])  # fmt: skip
    return out


# ------------------------------------------------------------------ divergence and areas


def divergence_grid(system: fm.System, vals, lo, hi, counts: int, z: float = 0.0) -> np.ndarray:
    """div F on a ``counts`` x ``counts`` grid of nodes over the window's x-y rectangle
    (a 3D system at height ``z``): the trace of the Jacobian (numeric where it must be),
    ``[ix, iy]``, NaN where it is not finite."""
    xs = np.linspace(lo[0], hi[0], counts)
    ys = np.linspace(lo[1], hi[1], counts)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cols = [X.ravel(), Y.ravel()] + ([np.full(X.size, float(z))] if system.dim == 3 else [])
    with np.errstate(all="ignore"):
        J = system.fn.jacobians(np.stack(cols, axis=1), vals)
        div = np.trace(J, axis1=1, axis2=2)
    div = np.where(np.isfinite(div), div, np.nan)
    return div.reshape(counts, counts)


def polygon_area(pts: np.ndarray) -> float:
    """The area inside a closed curve (M, 2) by the shoelace formula (orientation dropped)."""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


# ------------------------------------------------------------------ sweeps


def _with(system: fm.System, values: dict, name: str, v) -> tuple:
    vals = dict(values)
    vals[name] = v
    return tuple(vals[k] for k in system.variables)


def _eigs(system: fm.System, vals: tuple, pts: np.ndarray) -> np.ndarray:
    """Eigenvalues (K, dim) of the Jacobian at each point (NaN where not finite)."""
    if not len(pts):
        return np.zeros((0, system.dim), dtype=complex)
    J = system.fn.jacobians(pts, vals)
    ok = np.isfinite(J).all(axis=(1, 2))
    out = np.full((len(pts), system.dim), np.nan + 0j)
    if ok.any():
        out[ok] = np.linalg.eigvals(J[ok])
    return out


def _tol(ev: np.ndarray) -> float:
    fin = np.abs(ev[np.isfinite(ev)])
    return 1e-7 * max(1.0, float(fin.max()) if len(fin) else 1.0)


def _row(system, vals, lo, hi, grid) -> dict[str, Any]:
    found = fm.equilibria(system, vals, lo, hi, grid)
    pts = found["points"]
    ev = _eigs(system, vals, pts)
    tol = _tol(ev)
    re = ev.real
    with np.errstate(invalid="ignore"):
        stable = int((np.nanmax(re, axis=1) < -tol).sum()) if len(pts) else 0
    return {"points": pts, "eigs": ev, "count": len(pts), "stable": stable,
            "max_real": float(np.nanmax(re)) if len(pts) and np.isfinite(re).any() else None,
            "non_isolated": found["non_isolated"]}  # fmt: skip


def _track(system, vals, near: np.ndarray, cap: float) -> np.ndarray | None:
    """The equilibrium Newton reaches from ``near`` (one start), or None."""
    with np.errstate(all="ignore"):
        pts, ok = fm.newton(system.fn, vals, near[None], cap=cap)
    return pts[0] if ok[0] else None


def _crossing_parts(ev: np.ndarray, tol: float) -> tuple[float, float]:
    """(largest real part of a real eigenvalue, of a complex pair) of one point; -inf none."""
    real = ev[np.abs(ev.imag) <= tol].real
    cplx = ev[np.abs(ev.imag) > tol].real
    return (float(real.max()) if len(real) else -np.inf,
            float(cplx.max()) if len(cplx) else -np.inf)  # fmt: skip


def _stable(ev: np.ndarray) -> bool | None:
    """True: every real part negative; False: one positive; None: undecided (on the axis)."""
    re = ev.real[np.isfinite(ev)]
    if not len(re):
        return None
    tol = _tol(ev)
    if re.max() < -tol:
        return True
    return False if re.max() > tol else None


def _partner(q: np.ndarray, there: np.ndarray, here: np.ndarray) -> int | None:
    """The point of ``there`` that is the same equilibrium as ``q`` (of ``here``): each the
    other's nearest (two equilibria that pass each other are not matched), else None."""
    if not len(there):
        return None
    j = int(np.argmin(np.linalg.norm(there - q, axis=1)))
    back = int(np.argmin(np.linalg.norm(here - there[j], axis=1)))
    return j if np.array_equal(here[back], q) else None


def _merged(cands: list[dict], width: float) -> list[dict]:
    """One candidate per label and value (Lorenz's C+ and C- lose stability together)."""
    out: list[dict] = []
    for c in cands:
        same = next((o for o in out if o["label"] == c["label"] and c["value"] is not None
                     and o["value"] is not None and abs(o["value"] - c["value"]) <= 4 * width),
                    None)  # fmt: skip
        if same is None:
            out.append(c)
        elif "near" in c:
            same["also_near"] = [*same.get("also_near", []), c["near"]]
    return out


def _bisect(fa_left, a: float, b: float, test, iters: int) -> tuple[float, float, int]:
    """Shrink [a, b] keeping ``test(a) == fa_left`` and ``test(b) != fa_left``."""
    k = 0
    for k in range(iters):  # noqa: B007
        m = 0.5 * (a + b)
        if m in (a, b):
            break
        if test(m) == fa_left:
            a = m
        else:
            b = m
    return a, b, k + 1


def sweep(system: fm.System, values: dict, name: str, lo_v: float, hi_v: float, n: int, *,
          mode: str = "equilibria", lo=None, hi=None, grid: int = 9, x0=None,
          dt: float = 0.01, transient: float = 100.0, sample_time: float = 50.0,
          sampler: Sampler | None = None, escape=None, max_step: float | None = None,
          refine: bool = True) -> dict[str, Any]:  # fmt: skip
    """Sweep the variable ``name`` over ``n`` values from ``lo_v`` to ``hi_v`` (at most
    ``MAX_SWEEP``).

    ``mode="equilibria"``: per value every equilibrium in the window (``lo``, ``hi``, Newton
    from ``grid`` starts per side) with its eigenvalues; ``candidates`` between neighbouring
    values where the count changes (bisection on "the count equals the left one"), or where,
    the count the same, a tracked equilibrium's real eigenvalue (``transcritical``) or complex
    pair (``Hopf``) crosses the imaginary axis (bisection on the sign). Labels: a change by 2
    with an equilibrium that persists and changes stability is a ``pitchfork``, else
    ``saddle-node``; a change by 1 with a point at the window's edge is a ``window edge`` (not
    a bifurcation). At most ``MAX_CANDIDATES``, each in ``BISECT_ITERS`` halvings.

    ``mode="attractor"``: every value advanced at once from ``x0`` for ``transient``, then
    sampled (``sampler``, default: the maxima of x) for ``sample_time``; per value the last
    ``MAX_SAMPLES`` samples (``samples``, NaN padded), the ``period`` (clusters of the samples
    within ``1e-3`` of their range, None past a quarter of the samples: chaos or a long
    period) and where the period changes (``candidates``, between swept values only)."""
    clamped = ["values"] if int(n) > MAX_SWEEP else []
    n = int(min(max(n, 2), MAX_SWEEP))
    vs = np.linspace(float(lo_v), float(hi_v), n)
    if name not in system.variables:
        return {"mode": mode, "values": vs, "rows": [], "candidates": [], "clamped": clamped,
                "why": f"the system does not use {name!r}", "steps": 0}  # fmt: skip
    if mode == "attractor":
        return _sweep_attractor(system, values, name, vs, x0=x0, dt=dt, transient=transient,
                                sample_time=sample_time, sampler=sampler or Sampler.maxima(0),
                                escape=escape, max_step=max_step, clamped=clamped)  # fmt: skip
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    span = float((hi - lo).max())
    rows = [_row(system, _with(system, values, name, v), lo, hi, grid) for v in vs]
    cands: list[dict[str, Any]] = []
    width = 1e-6 * max(abs(hi_v - lo_v), 1e-12)
    iters = max(1, min(BISECT_ITERS, math.ceil(math.log2(max((vs[1] - vs[0]) / width, 2)))))

    def count_at(v):
        return _row(system, _with(system, values, name, v), lo, hi, grid)["count"]

    for k in range(n - 1):
        if len(cands) >= MAX_CANDIDATES:
            break
        ra, rb = rows[k], rows[k + 1]
        a, b = float(vs[k]), float(vs[k + 1])
        if ra["non_isolated"] or rb["non_isolated"]:
            continue
        if ra["count"] != rb["count"]:
            # the local test, read at the two swept values (clear of the degenerate point)
            label = "count change"
            if abs(ra["count"] - rb["count"]) == 2:
                few, many = (ra, rb) if ra["count"] < rb["count"] else (rb, ra)
                label = "saddle-node"
                for i, q in enumerate(few["points"]):
                    j = _partner(q, many["points"], few["points"])
                    if j is not None and _stable(few["eigs"][i]) != _stable(many["eigs"][j]):
                        label = "pitchfork"  # it persists and changes stability
                        break
            if refine:
                a, b, _ = _bisect(ra["count"], a, b, count_at, iters)
            if abs(ra["count"] - rb["count"]) == 1:  # one point crossing the window's side?
                ends = [_row(system, _with(system, values, name, v), lo, hi, grid)
                        for v in (a, b)]  # fmt: skip
                more = max(ends, key=lambda row: row["count"])["points"]
                edge = (1e-3 if refine else 0.05) * (hi - lo)
                if len(more) and bool(((more <= lo + edge) | (more >= hi - edge)).any()):
                    label = "window edge"
            cands.append({"label": label, "between": [a, b], "value": 0.5 * (a + b),
                          "change": [ra["count"], rb["count"]]})  # fmt: skip
            continue
        # the same count: follow each equilibrium and watch its eigenvalues cross
        for i, q in enumerate(ra["points"]):
            if len(cands) >= MAX_CANDIDATES:
                break
            j = _partner(q, rb["points"], ra["points"])
            if j is None:
                continue
            tol = max(_tol(ra["eigs"][i]), _tol(rb["eigs"][j]))
            real_a, cx_a = _crossing_parts(ra["eigs"][i], tol)
            real_b, cx_b = _crossing_parts(rb["eigs"][j], tol)
            for label, fa_, fb_, part in (("Hopf", cx_a, cx_b, 1),
                                          ("transcritical", real_a, real_b, 0)):  # fmt: skip
                if not (np.isfinite(fa_) and np.isfinite(fb_)) or (fa_ < 0) == (fb_ < 0):
                    continue
                state = {"at": q.copy()}

                def sign_at(v, part=part, state=state):
                    vals = _with(system, values, name, v)
                    at = _track(system, vals, state["at"], span)
                    if at is None:
                        return None
                    state["at"] = at
                    ev = _eigs(system, vals, at[None])[0]
                    x = _crossing_parts(ev, _tol(ev))[part]
                    return bool(x < 0) if np.isfinite(x) else None

                lo_b, hi_b = a, b
                if refine:
                    lo_b, hi_b, _ = _bisect(bool(fa_ < 0), a, b, sign_at, iters)
                cands.append({"label": label, "between": [lo_b, hi_b],
                              "value": 0.5 * (lo_b + hi_b), "change": [ra["count"], rb["count"]],
                              "near": [float(c) for c in q]})  # fmt: skip
    cands = _merged(cands, width)
    table = [{"value": float(v), "count": r["count"], "stable": r["stable"],
              "max_real": r["max_real"], "non_isolated": r["non_isolated"],
              "points": r["points"], "eigs": r["eigs"]}
             for v, r in zip(vs, rows, strict=True)]  # fmt: skip
    return {"mode": "equilibria", "values": vs, "rows": table, "candidates": cands,
            "clamped": clamped, "steps": 0, "grid": grid}  # fmt: skip


def periods(samples: np.ndarray, scale: float) -> list[int | None]:
    """Per row of samples (NaN padded), the number of clusters within ``1e-3 * scale``;
    None past a quarter of the samples (chaos or a long period) or with fewer than 8."""
    out: list[int | None] = []
    tol = 1e-3 * max(scale, 1e-12)
    for row in samples:
        s = np.sort(row[np.isfinite(row)])
        if len(s) < 8:
            out.append(None)
            continue
        k = 1 + int((np.diff(s) > tol).sum())
        out.append(k if k <= len(s) // 4 else None)
    return out


def _sweep_attractor(system, values, name, vs, *, x0, dt, transient, sample_time, sampler,
                     escape, max_step, clamped) -> dict[str, Any]:  # fmt: skip
    n = len(vs)
    d = system.dim
    x0 = np.zeros(d) + 0.1 if x0 is None else np.asarray(x0, dtype=np.float64)[:d]
    n_tr = max(0, round(transient / dt))
    n_s = max(1, round(sample_time / dt))
    if n * (n_tr + n_s) > MAX_SWEEP_WORK:
        scale = MAX_SWEEP_WORK / (n * (n_tr + n_s))
        n_tr, n_s = int(n_tr * scale), max(1, int(n_s * scale))
        clamped = [*clamped, "work"]
    others = {k: float(v) for k, v in values.items()}
    vals = tuple(np.asarray(vs, dtype=np.float64) if k == name else others[k]
                 for k in system.variables)  # fmt: skip
    run = SectionRun(system.fn, vals, np.tile(x0, (n, 1)), dt, sampler, escape=escape,
                     max_step=max_step)  # fmt: skip
    run.advance(n_tr + n_s)
    cr = run.crossings()
    keep = cr["times"] >= n_tr * dt
    pts, sid = cr["points"][keep], cr["seed"][keep]
    coord = sampler.coord if sampler.kind == "maxima" else int(np.argmax(np.abs(sampler.normal)))
    if sampler.kind == "plane":  # plot a coordinate the plane does not fix
        coord = next(k for k in range(d) if k != coord)
    samples = np.full((n, MAX_SAMPLES), np.nan)
    for i in range(n):
        mine = pts[sid == i, coord][-MAX_SAMPLES:]
        samples[i, : len(mine)] = mine
    fin = samples[np.isfinite(samples)]
    scale = float(np.ptp(fin)) if len(fin) else 1.0
    per = periods(samples, max(scale, 1e-9))
    codes = run.code
    rows = [{"value": float(v), "period": per[i], "samples": int(np.isfinite(samples[i]).sum()),
             "stopped": None if codes[i] == fm.ALIVE else
             {fm.NONFINITE: "not finite", fm.TOO_FAST: "too fast",
              fm.ESCAPED: "escaped"}.get(int(codes[i]), "stopped")}
            for i, v in enumerate(vs)]  # fmt: skip
    cands = []
    for k in range(n - 1):
        pa, pb = per[k], per[k + 1]
        if pa is not None and pb is not None and pb == 2 * pa:
            label = "period doubling"
        elif pa is not None and pb is not None and pa == 2 * pb:
            label = "period halving"
        elif (pa is None) != (pb is None) and (pa or pb):
            label = "onset of chaos or a long period"
        else:
            continue
        if len(cands) < MAX_CANDIDATES:
            cands.append({"label": label, "between": [float(vs[k]), float(vs[k + 1])],
                          "value": None, "change": [pa, pb]})  # fmt: skip
    return {"mode": "attractor", "values": vs, "rows": rows, "samples": samples,
            "coord": coord, "candidates": cands, "clamped": clamped,
            "steps": n_tr + n_s, "transient": n_tr * dt}  # fmt: skip
