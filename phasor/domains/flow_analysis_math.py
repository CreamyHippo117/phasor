"""The Explorer's analysis pack (Milestone 30, EXPLORER.md; advice section 2.6 rows 5, 7-12):
numbers about a flow, from the flow core (``flow_math``). Pure NumPy, no drawing.

One implementation of each analysis (CLAUDE.md, "one way to do each thing"), shared with the
Milestone 27 tools (``explore_math``, ``workspace/explorer.py``):

- ``Sampler`` / ``refine_crossings``: where a path is sampled: crossings of a plane
  ``n . p = offset`` (a Poincare section) or the local maxima of one coordinate (its
  velocity crossing 0 downward). A crossing is bracketed by a step, guessed by linear
  interpolation and then refined to rounding (regula falsi on a partial RK4 step from the
  step's start): every section, return and maximum here goes through it.
- ``SectionRun``: the crossings of seeds advected together (any number, batch independent),
  continued step by step under a lock, so a longer flow time extends the answer instead of
  redoing it and threads may share it.
- ``benettin`` / ``lyapunov``: the largest Lyapunov exponent (Benettin's two-trajectory
  method), batched over seeds; ``judge`` decides ``converged`` and whether a value is
  clearly chaotic (``chaotic_beyond``): the spread over blocks, a trend test across them (a
  separation that grows like a power of t, as in a shear, falls like ln T / T and is not a
  positive exponent), at least three spreads above zero and above ``CHAOS_FLOOR``.
- ``returns_repeat``: the one test for periods: the fewest windings after which returns to
  a section repeat (persisting, and not drifting); ``limit_cycle``, the attractor sweep and
  ``explore_math.probe`` all use it.
- ``sweep``: one variable over a range. ``equilibria`` is ``explore_math.sweep`` (the
  Milestone 27 tool's, audited: continuation, window edges, degenerate roots); ``attractor``
  advances every value at once (a variable may be an array, one value per point), samples
  after a transient and reads each value's period with ``returns_repeat``.

Not detected (say so wherever a sweep is shown): global bifurcations (homoclinic and
heteroclinic loops, saddle-node of cycles, crises), bifurcations of cycles except the
period changes the samples show (never refined), anything outside the window or that falls
between the grid of Newton starts or between two swept values.

Determinism: as the flow core. Numbers after flow time ``flow_math.SAFE_TAU`` from a chaotic
path are statistics (a Lyapunov exponent agrees across machines within its spread, not
bit for bit); an attracting cycle forgets rounding, so its period is reproducible to its
tolerance (and accurate only to what ``dt`` allows).
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phasor.domains import flow_math as fm
from phasor.domains.flow_math import FunctionError

__all__ = [
    "Sampler",
    "SectionRun",
    "benettin",
    "divergence_grid",
    "limit_cycle",
    "lyapunov",
    "polygon_area",
    "returns_repeat",
    "sweep",
]  # fmt: skip

# Budgets (clamped and linted ``flow_budget``, never refused). A step taken in a Python loop
# for one or two points costs about as much as stepping LOOP_POINTS points at once, so the
# work the lint counts is LOOP_POINTS x steps for those (measured here: about 0.3 ms a step).
LOOP_POINTS = 1000
MAX_SECTION_SEEDS = 4096
MAX_SECTION_WORK = 2e7  # seeds x steps of one section
MAX_CROSSINGS = 200_000  # crossings kept by one section (the first ones)
MAX_LYAPUNOV_STEPS = 25_000  # steps of the two Benettin trajectories (transient included)
MAX_CYCLE_STEPS = 20_000
MAX_RETURNS = 64
MAX_SWEEP = 256  # values of an attractor sweep (an equilibria sweep: explore_math.MAX_SWEEP)
MAX_SWEEP_WORK = 4e6  # values x steps of an attractor sweep (about 20 s here at its maximum)
MAX_SAMPLES = 64  # samples kept per value (the last ones)
MAX_CANDIDATES = 8  # candidates of an attractor sweep
MAX_BLOCKS = 64
MAX_LOOPS = 16  # a cycle that winds this many times before it closes (period doubling)
CHAOS_FLOOR = 0.02  # a positive exponent below this (per unit flow time) is not called chaos
LN_1E16 = math.log(1e16)  # about 36.8: e-foldings from rounding (1e-16) to order one
SETTLED = 1e-5  # returns repeat within this share of the window (an attracting orbit)


def _finite(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise FunctionError(f"{what} must be a number, not {x!r}") from None
    if not math.isfinite(v):
        raise FunctionError(f"{what} must be finite, not {v}")
    return v


def _point(system: fm.System, x0, what: str = "x0") -> np.ndarray:
    p = np.asarray(x0, dtype=np.float64).ravel()
    if p.shape != (system.dim,) or not np.isfinite(p).all():
        raise FunctionError(f"{what} is {system.dim} finite numbers here, not {list(np.ravel(x0))}")
    return p


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
        if not (np.isfinite(n).all() and size > 0 and math.isfinite(float(offset))):
            raise FunctionError("a section's normal and offset must be finite, the normal not 0")
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
    """Which steps cross zero (from ``g0`` to ``g1``) in ``direction`` and the linear guess of
    where inside the step (the share 0..1). A step that starts on the plane does not count
    again (``g0 < 0 <= g1`` upward, ``g0 > 0 >= g1`` downward)."""
    with np.errstate(invalid="ignore"):
        up = (g0 < 0) & (g1 >= 0)
        down = (g0 > 0) & (g1 <= 0)
    hit = up if direction > 0 else down if direction < 0 else up | down
    with np.errstate(all="ignore"):
        f = np.where(hit, g0 / np.where(hit, g0 - g1, 1.0), 0.0)
    return hit, np.clip(f, 0.0, 1.0)


def refine_crossings(fn: fm.Compiled, vals, sampler: Sampler, p: np.ndarray, dt: float,
                     g0: np.ndarray, g1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Where steps of ``dt`` from the points ``p`` (K, dim) cross the sampler, to rounding:
    ``(share of the step, point)``. Regula falsi (Illinois) on ``h`` with
    ``value(RK4(p, h dt))``, from the linear guess and inside the bracket [0, 1]. ``vals``
    are the values of these K points (arrays are per point). A linear guess alone is wrong
    by about dt^2 (0.04 on Lorenz at dt 0.01); this is wrong by rounding."""
    k = len(p)
    a, b = np.zeros(k), np.ones(k)
    ga, gb = np.asarray(g0, dtype=np.float64).copy(), np.asarray(g1, dtype=np.float64).copy()
    side = np.zeros(k)
    active = np.ones(k, dtype=bool)  # each point stops on its own (batch independence)
    with np.errstate(all="ignore"):
        lin = np.clip(ga / (ga - gb), 0.0, 1.0)
        h = lin.copy()
        for _ in range(30):
            q = fm._rk4(fn, vals, p, (h * dt)[:, None])
            gh = sampler.values(fn, vals, q)
            scale = 1.0 + np.abs(np.where(np.isfinite(q), q, 0.0)).max(axis=1)
            active &= np.isfinite(gh) & ~(np.abs(gh) <= 1e-12 * scale) & ((b - a) > 1e-13)
            if not active.any():
                break
            same = np.sign(gh) == np.sign(ga)  # the root is in [h, b]: move a
            na, nb = np.where(same, h, a), np.where(same, b, h)
            nga, ngb = np.where(same, gh, ga), np.where(same, gb, gh)
            ngb = np.where(same & (side == 1), 0.5 * ngb, ngb)  # Illinois: one end twice
            nga = np.where(~same & (side == -1), 0.5 * nga, nga)
            den = ngb - nga
            nxt = np.where(den != 0, (na * ngb - nb * nga) / np.where(den != 0, den, 1.0),
                           0.5 * (na + nb))  # fmt: skip
            ok = np.isfinite(nxt) & (nxt >= na) & (nxt <= nb)
            nh = np.where(ok, nxt, 0.5 * (na + nb))
            a, b = np.where(active, na, a), np.where(active, nb, b)
            ga, gb = np.where(active, nga, ga), np.where(active, ngb, gb)
            side = np.where(active, np.where(same, 1, -1), side)
            h = np.where(active, nh, h)
        h = np.where(np.isfinite(h), h, lin)
        q = fm._rk4(fn, vals, p, (h * dt)[:, None])
    bad = ~np.isfinite(q).all(axis=1)
    if bad.any():  # keep the linear guess where the refinement leaves the finite numbers
        h = np.where(bad, lin, h)
    return h, q


# ------------------------------------------------------------------ sections


@dataclass
class SectionRun:
    """Crossings of ``sampler`` by seeds (S, dim) advected together, from step 0 on.
    ``advance(n)`` continues to step ``n`` (the same steps whatever was asked before, so the
    answer is a function of the inputs); a lock makes it safe to share between threads.
    Kept: ``points`` (K, dim) where each crossing is, ``times`` (step + share of a step,
    times ``dt`` for flow time), ``seed`` which seed crossed; at most ``MAX_CROSSINGS`` (the
    first ones; ``cut_at`` is the step where one was first left out)."""

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
    cut_at: int | None = None
    lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.seeds = np.atleast_2d(np.asarray(self.seeds, dtype=np.float64))
        if self.p is None:
            self.p = self.seeds.copy()
            self.code = np.zeros(len(self.seeds), dtype=np.int8)
            self.g = self.sampler.values(self.fn, self.vals, self.p)

    def advance(self, n: int) -> SectionRun:
        with self.lock:
            self._advance(n)
        return self

    def _advance(self, n: int) -> None:
        while self.n < n and (self.code == fm.ALIVE).any():
            alive = self.code == fm.ALIVE
            q, c = _advance(self.fn, self.vals, np.where(alive[:, None], self.p, 0.0), self.dt,
                            self.escape, self.max_step)  # fmt: skip
            c = np.where(alive, c, self.code).astype(np.int8)
            ok = alive & (c == fm.ALIVE)
            safe = np.where(ok[:, None], q, 0.0)
            g1 = np.where(ok, self.sampler.values(self.fn, self.vals, safe), np.nan)
            hit, _ = crossed(self.g, g1, self.sampler.direction)
            hit &= ok
            if hit.any() and self.cut_at is None:
                idx = np.flatnonzero(hit)
                h, pts = refine_crossings(self.fn, fm._take(self.vals, idx), self.sampler,
                                          self.p[idx], self.dt, self.g[idx], g1[idx])  # fmt: skip
                room = MAX_CROSSINGS - self.count
                if len(idx) > room:
                    idx, pts, h = idx[:room], pts[:room], h[:room]
                    self.cut_at = self.n + 1
                self.found.append((pts, self.n + h, idx))
                self.count += len(idx)
            self.p = np.where(ok[:, None], q, self.p)
            self.code = c
            self.g = g1
            self.n += 1
        if self.n < n:  # every seed stopped: nothing more can cross
            self.n = n

    def crossings(self, upto: int | None = None) -> dict[str, Any]:
        """The crossings so far (or up to step ``upto``), in the order they happened, and
        ``full``: whether one was left out at or before that step (the cap)."""
        with self.lock:
            d = self.seeds.shape[1]
            full = self.cut_at is not None and (upto is None or self.cut_at <= upto)
            if not self.found:
                return {"points": np.zeros((0, d)), "times": np.zeros(0),
                        "seed": np.zeros(0, dtype=np.int64), "full": full}  # fmt: skip
            pts = np.concatenate([a for a, _, _ in self.found])
            tim = np.concatenate([b for _, b, _ in self.found])
            sid = np.concatenate([c for _, _, c in self.found])
        if upto is not None:
            keep = tim <= upto
            pts, tim, sid = pts[keep], tim[keep], sid[keep]
        return {"points": pts, "times": tim * self.dt, "seed": sid, "full": full}


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


_STOP = {fm.NONFINITE: "not finite", fm.TOO_FAST: "too fast", fm.ESCAPED: "escaped"}


def benettin(fn: fm.Compiled, vals, seeds, directions, *, dt: float, transient: int,
             every: int, blocks: int, per: int, escape=None, max_step: float | None = None,
             d0: float | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """The block rates of Benettin's method for each seed (S, dim): a twin at
    ``d0`` (default ``1e-8 max(1, |x|)``, relative to where the path is, so a growing path
    keeps its twin above rounding) along ``directions`` (S, dim), renormalised every
    ``every`` steps of ``dt``; the first ``transient`` steps (rounded up to renormalisations)
    turn the separation to the fastest direction and are not counted; then ``blocks`` blocks
    of ``per`` renormalisations each. Per seed: ``rates`` (or None) and ``stopped``."""
    seeds = np.atleast_2d(np.asarray(seeds, dtype=np.float64))
    s = len(seeds)

    def size(at: np.ndarray) -> np.ndarray:
        if d0 is not None:
            return np.full(len(at), float(d0))
        return 1e-8 * np.maximum(1.0, np.linalg.norm(at, axis=1))

    dirs = np.asarray(directions, dtype=np.float64).reshape(s, -1)
    pair = np.vstack([seeds, seeds + size(seeds)[:, None] * dirs])
    start = np.linalg.norm(pair[s:] - pair[:s], axis=1)
    sums = np.zeros((blocks, s))
    stopped: list[str | None] = [None] * s
    alive = np.ones(s, dtype=bool)
    warm = -(-int(transient) // every)
    with np.errstate(all="ignore"):
        for k in range(warm + blocks * per):
            for _ in range(every):
                pair, code = _advance(fn, _twice(vals), pair, dt, escape, max_step)
                bad = alive & ((code[:s] != fm.ALIVE) | (code[s:] != fm.ALIVE))
                for i in np.flatnonzero(bad):
                    c = int(code[i]) if code[i] != fm.ALIVE else int(code[s + i])
                    stopped[i] = _STOP.get(c, "stopped")
                alive &= ~bad
                pair = np.where(np.isfinite(pair), pair, 0.0)  # stopped seeds: finite arithmetic
            if not alive.any():
                break
            sep = pair[s:] - pair[:s]
            dist = np.linalg.norm(sep, axis=1)
            ok = alive & np.isfinite(dist) & (dist > 0) & (start > 0)
            for i in np.flatnonzero(alive & ~ok):
                stopped[i] = "not finite" if not np.isfinite(dist[i]) else "the twins met"
            alive &= ok
            if k >= warm:
                sums[(k - warm) // per, ok] += np.log(dist[ok] / start[ok])
            new = size(pair[:s])
            moved = pair[:s] + sep * (new / np.where(ok, dist, 1.0))[:, None]
            pair[s:] = np.where(ok[:, None], moved,
                                pair[s:])  # fmt: skip
            start = np.linalg.norm(pair[s:] - pair[:s], axis=1)
    span = per * every * abs(dt)
    return [{"rates": (sums[:, i] / span) if alive[i] else None, "stopped": stopped[i]}
            for i in range(s)]  # fmt: skip


def _twice(vals) -> tuple:
    """Per-point values for a run of seeds and their twins (arrays repeated)."""
    return tuple(np.concatenate([v, v]) if isinstance(v, np.ndarray) else v for v in vals)


def judge(rates, atol: float = 0.01, rtol: float = 0.05) -> dict[str, Any]:
    """What the block rates say. ``value`` the mean, ``spread`` the standard error over the
    blocks, ``trend`` the change across them (a least-squares line); ``converged`` only when
    the spread is under ``max(atol, rtol |value|)`` and there is no significant trend (a
    separation growing like a power of t gives rates falling like ln T / T: a shear, an
    undamped pendulum, a neutral direction); ``chaotic`` only when converged, above
    ``CHAOS_FLOOR`` and above three spreads."""
    r = np.asarray(rates, dtype=np.float64)
    nb = len(r)
    value = float(r.mean())
    spread = float(r.std(ddof=1) / math.sqrt(nb)) if nb > 1 else float("inf")
    idx = np.arange(nb) - (nb - 1) / 2
    slope = float((idx * (r - value)).sum() / max((idx * idx).sum(), 1e-300))
    resid = r - value - slope * idx
    se_slope = float(math.sqrt((resid**2).sum() / max(nb - 2, 1) / max((idx * idx).sum(), 1e-300)))
    change = abs(slope) * (nb - 1)  # across the run
    trend = change > max(atol, 0.2 * abs(value)) and abs(slope) > 3 * se_slope
    converged = bool(spread <= max(atol, rtol * abs(value)) and not trend)
    chaotic = bool(converged and value > max(CHAOS_FLOOR, 3 * spread))
    return {"value": value, "spread": spread, "trend": -slope * (nb - 1),
            "converged": converged, "chaotic": chaotic}  # fmt: skip


def lyapunov(system: fm.System, x0, T: float = 100.0, *, values=(), dt: float = 0.01,
             transient: float = 10.0, renorm: float = 0.5, blocks: int = 8,
             d0: float | None = None, escape=None, max_step: float | None = None,
             atol: float = 0.01, rtol: float = 0.05) -> dict[str, Any]:  # fmt: skip
    """The largest Lyapunov exponent along the path from ``x0`` (``benettin``, judged by
    ``judge``): after ``transient`` flow time, ``T`` flow time in ``blocks`` blocks, a twin
    renormalised every ``renorm``. ``value``, ``spread``, ``trend``, ``converged``;
    ``chaotic_beyond`` = ln(1e16) / value only for a clearly chaotic value (else None). A
    path that stops gives ``value`` None and says why (``stopped``). Every factor is bounded
    by ``MAX_LYAPUNOV_STEPS`` (``clamped``); nonsense (a time or step that is not a finite
    positive number, a point of the wrong size) is a ``FunctionError`` with the reason."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    p = _point(system, x0)
    T, dt = _finite(T, "T"), _finite(dt, "dt")
    transient, renorm = _finite(transient, "transient"), _finite(renorm, "renorm")
    if T <= 0 or dt == 0 or transient < 0 or renorm <= 0:
        raise FunctionError("T, dt and renorm must be positive and transient at least 0")
    dt = abs(dt)
    clamped: list[str] = []
    nb = int(min(max(int(blocks), 2), MAX_BLOCKS))
    if nb != int(blocks):
        clamped.append("blocks")
    every = max(1, round(renorm / dt))
    if every * nb * 2 > MAX_LYAPUNOV_STEPS:  # each block at least one renormalisation
        every = max(1, MAX_LYAPUNOV_STEPS // (2 * nb))
        clamped.append("renorm")
    per = max(1, round(T / (dt * every * nb)))
    n_tr = max(0, round(transient / dt))
    if n_tr + nb * per * every > MAX_LYAPUNOV_STEPS:  # the transient at most half
        n_tr = min(n_tr, MAX_LYAPUNOV_STEPS // 2)
        per = max(1, (MAX_LYAPUNOV_STEPS - n_tr) // (nb * every))
        clamped.append("steps")
    out: dict[str, Any] = {"value": None, "spread": None, "trend": None, "converged": False,
                           "T": None, "blocks": [], "chaotic_beyond": None, "stopped": None,
                           "steps": n_tr + nb * per * every, "clamped": clamped,
                           "renorm": every * dt, "transient": n_tr * dt}  # fmt: skip
    (res,) = benettin(system.fn, vals, p[None], separation(system, vals, p)[None], dt=dt,
                      transient=n_tr, every=every, blocks=nb, per=per, escape=escape,
                      max_step=max_step, d0=d0)  # fmt: skip
    if res["rates"] is None:
        out["stopped"] = res["stopped"]
        return out
    j = judge(res["rates"], atol, rtol)
    out.update(value=j["value"], spread=j["spread"], trend=j["trend"], converged=j["converged"],
               T=nb * per * every * dt, blocks=[float(r) for r in res["rates"]])  # fmt: skip
    if j["chaotic"]:
        out["chaotic_beyond"] = LN_1E16 / j["value"]
    return out


# ------------------------------------------------------------------ periods


def returns_repeat(rets: list[tuple[np.ndarray, float]], tol: float, scale: float, *,
                   persist: int = 3, drift: bool = True) -> tuple[int, str | None]:  # fmt: skip
    """The one period test. Returns (points, times) to a section, in order: the fewest
    windings m (1..MAX_LOOPS) after which they repeat: each of the last ``persist`` returns
    within ``tol * scale`` of the one m before, and their times (periods) within ``tol`` of
    each other. For m > 1 the windings must be distinct (further apart than 100 x the
    tolerance) and not still closing in (a period-one orbit near a doubling alternates with
    a shrinking amplitude: that is "not settled", not period two). With ``drift``, returns
    that keep moving by a steady amount (a centre, a weak focus, a slow drift: the gap m
    returns ago not 10% larger) are not a cycle. ``(m, None)``; ``(0, None)``: not yet;
    ``(0, why)``: what was seen instead."""
    tol_d = tol * scale
    q = [np.asarray(r[0], dtype=np.float64) for r in rets]
    t = [float(r[1]) for r in rets]
    for m in range(1, MAX_LOOPS + 1):
        need = m + persist + 1 + (m if drift else 0)
        if len(q) < need:
            return 0, None
        if not all(np.linalg.norm(q[-j] - q[-j - m]) <= tol_d for j in range(1, persist + 1)):
            continue
        per = [t[-j] - t[-j - m] for j in range(1, persist + 1)]
        if max(per) - min(per) > tol * max(max(per), 1e-12):
            continue
        if m > 1:
            d1 = [float(np.linalg.norm(q[-j] - q[-j - 1])) for j in range(1, m + 1)]
            if min(d1) <= 100 * tol_d:
                return 0, "not settled: the windings are not yet apart"
            for k in (d for d in range(1, m) if m % d == 0):  # a simpler orbit still closing?
                dk = [float(np.linalg.norm(q[-j] - q[-j - k])) for j in range(1, len(q) - k)]
                if len(dk) >= 2 * m and np.mean(dk[:m]) < 0.98 * np.mean(dk[-m:]):
                    return 0, "not settled: closing in on a simpler orbit"
        if drift:
            g1 = float(np.linalg.norm(q[-1] - q[-1 - m]))
            g0 = float(np.linalg.norm(q[-1 - m] - q[-1 - 2 * m]))
            if g1 > 1e-11 * max(scale, 1e-300) and g1 > 0.9 * g0:
                return 0, ("the returns drift by a steady amount: a centre, a weak focus or a "
                           "slow drift, not an isolated attracting cycle")  # fmt: skip
        return m, None
    return 0, None


def _check_run(system: fm.System, x0, dt, transient, max_time) -> tuple:
    p = _point(system, x0)
    dt, transient, max_time = (_finite(dt, "dt"), _finite(transient, "transient"),
                               _finite(max_time, "max_time"))  # fmt: skip
    if dt == 0 or transient < 0 or max_time <= 0:
        raise FunctionError("dt must not be 0, transient at least 0 and max_time positive")
    return p, abs(dt), transient, max_time


def limit_cycle(system: fm.System, x0, *, values=(), dt: float = 0.01, transient: float = 50.0,
                max_time: float = 100.0, tol: float = 1e-6, scale: float = 1.0,
                escape=None, max_step: float | None = None) -> dict[str, Any]:  # fmt: skip
    """An attracting cycle reached from ``x0``: after ``transient``, the plane through the
    point across the flow; returns to it (upward, within a quarter of ``scale`` of the
    point, each refined by ``refine_crossings``) until ``returns_repeat`` says they repeat
    after ``loops`` windings. ``kind``: ``cycle`` (``period`` (all its loops), ``point`` on
    the plane, ``path`` one period, ``box`` its extent), ``equilibrium`` (the path stopped
    moving: ``point``), ``stopped`` (escape, not finite, too fast) or ``none`` with a
    ``note`` (chaos, a torus, a slow approach, a centre or a drift). The period is accurate
    to what ``dt`` allows (fixed-step RK4), not to ``tol``."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    p0, dt, transient, max_time = _check_run(system, x0, dt, transient, max_time)
    fn = system.fn
    n_tr = min(round(transient / dt), MAX_CYCLE_STEPS // 2)
    n_max = min(round(max_time / dt), MAX_CYCLE_STEPS - n_tr)
    clamped = [] if round(transient / dt) + round(max_time / dt) <= MAX_CYCLE_STEPS else ["steps"]
    out: dict[str, Any] = {"kind": "none", "period": None, "point": None, "converged": False,
                           "returns": 0, "path": None, "box": None, "gap": None, "loops": None,
                           "note": None, "steps": n_tr + n_max, "clamped": clamped}  # fmt: skip
    p, code = p0.reshape(1, -1), np.zeros(1, np.int8)
    for _ in range(n_tr):
        p, code = _advance(fn, vals, p, dt, escape, max_step)
        if code[0] != fm.ALIVE:
            out["kind"] = "stopped"
            return out
    a = p[0]
    v0 = fn.velocity(a[None], vals)[0]
    speed = float(np.linalg.norm(v0))
    big = max(1.0, float(np.linalg.norm(a)))
    if not math.isfinite(speed) or speed < 1e-9 * big:
        out.update(kind="equilibrium", point=a, converged=True)
        return out
    sampler = Sampler.plane(v0, float(np.dot(v0, a)), +1)
    near = 0.25 * scale
    rets: list[tuple[np.ndarray, float]] = []
    g = 0.0
    for k in range(n_max):
        q, code = _advance(fn, vals, p, dt, escape, max_step)
        if code[0] != fm.ALIVE:
            out["kind"] = "stopped"
            return out
        g1 = float(sampler.values(fn, vals, q)[0])
        if g < 0 <= g1 and float(np.linalg.norm(q[0] - a)) <= near:
            h, at = refine_crossings(fn, vals, sampler, p, dt, np.array([g]), np.array([g1]))
            rets.append((at[0], (k + float(h[0])) * dt))
            m, why = returns_repeat(rets, tol, scale)
            if m or why:
                out.update(converged=bool(m), loops=m or None, note=why)
                break
            if len(rets) >= MAX_RETURNS:
                break
        p, g = q, g1
    out["returns"] = len(rets)
    if not out["converged"]:
        end = p[0]  # settling onto a rest point (a slow spiral returns, shrinking)
        if float(np.linalg.norm(fn.velocity(end[None], vals)[0])) < 1e-5 * big:
            out.update(kind="equilibrium", point=end, converged=True, note=None)
            return out
        if out["note"] is None:
            out["note"] = "no settled return in max_time: chaos, a torus or a slow approach"
        return out
    m = out["loops"]
    out["gap"] = float(np.linalg.norm(rets[-1][0] - rets[-1 - m][0]))
    (_, tb), (qc, tc) = rets[-1 - m], rets[-1]
    period = tc - tb
    # Isolated and attracting? A neighbour on the plane, 1e-3 of the window off, must come
    # back: after three periods its return is within half its offset (a centre's or a weak
    # focus's neighbours stay where they are).
    n = np.asarray(sampler.normal)
    e = np.eye(len(n))[int(np.argmin(np.abs(n)))]
    e = e - (e @ n) * n
    e /= np.linalg.norm(e)
    off = 1e-3 * scale
    q2, g2, back, k2 = (qc + off * e)[None], 0.0, [], 0
    limit = min(round(3.5 * period / dt) + 2, MAX_CYCLE_STEPS)
    while k2 < limit and len(back) < 3 * m:
        nq, code = _advance(fn, vals, q2, dt, escape, max_step)
        if code[0] != fm.ALIVE:
            break
        g3 = float(sampler.values(fn, vals, nq)[0])
        if g2 < 0 <= g3 and float(np.linalg.norm(nq[0] - a)) <= near:
            _, at = refine_crossings(fn, vals, sampler, q2, dt, np.array([g2]), np.array([g3]))
            back.append(at[0])
        q2, g2, k2 = nq, g3, k2 + 1
    out["steps"] += k2
    if len(back) < 3 * m or float(np.linalg.norm(back[-1] - qc)) > 0.5 * off:
        out.update(converged=False, loops=None, gap=None,
                   note="the returns repeat, but a neighbouring orbit does not come back: a "
                        "centre, a weak focus or a very weakly attracting cycle, not an "
                        "isolated attracting cycle")  # fmt: skip
        return out
    one = fm.integrate(fn, vals, qc[None], min(max(2, round(period / dt)), fm.MAX_PROBE_STEPS),
                       dt, escape=escape, max_step=max_step).points[0]  # fmt: skip
    one = one[np.isfinite(one).all(axis=1)]
    out.update(kind="cycle", period=period, point=qc, path=one,
               box=[one.min(axis=0), one.max(axis=0)])  # fmt: skip
    return out


def path_returns(fn: fm.Compiled, vals, pts: np.ndarray, dt: float, start: int,
                 near: float) -> list[tuple[np.ndarray, float]]:  # fmt: skip
    """Returns of a stored path (N, dim, steps of ``dt``) to the plane through ``pts[start]``
    across the flow (upward, within ``near`` of that point), each refined from the step's
    start (``refine_crossings``): for ``returns_repeat`` on a path already computed (the
    probe tool)."""
    a = pts[start]
    v = fn.velocity(a[None], vals)[0]
    if not (np.isfinite(v).all() and np.linalg.norm(v) > 0):
        return []
    sampler = Sampler.plane(v, float(np.dot(v, a)) / 1.0, +1)
    g = sampler.values(fn, vals, pts)
    hit, _ = crossed(g[start:-1], g[start + 1 :], 1 if dt > 0 else -1)
    idx = np.flatnonzero(hit) + start
    idx = idx[np.linalg.norm(pts[idx + 1] - a, axis=1) <= near]
    if not len(idx):
        return []
    h, at = refine_crossings(fn, vals, sampler, pts[idx], dt, g[idx], g[idx + 1])
    return [(at[i], (idx[i] + h[i]) * abs(dt)) for i in range(len(idx))]


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


def _range(lo_v, hi_v) -> tuple[float, float]:
    lo_v, hi_v = _finite(lo_v, "the range's start"), _finite(hi_v, "the range's end")
    if not (lo_v < hi_v and math.isfinite(hi_v - lo_v)):
        raise FunctionError(f"a sweep needs lo < hi with a finite span, not ({lo_v}, {hi_v})")
    return lo_v, hi_v


def _stable_kind(kind: str) -> bool | None:
    if kind.startswith("stable"):
        return True
    if kind in ("saddle", "saddle-focus") or kind.startswith("unstable"):
        return False
    return None


def sweep(system: fm.System, values: dict, name: str, lo_v: float, hi_v: float, n: int, *,
          mode: str = "equilibria", lo=None, hi=None, grid: int = 9, x0=None,
          dt: float = 0.01, transient: float = 100.0, sample_time: float = 50.0,
          sampler: Sampler | None = None, escape=None, max_step: float | None = None,
          scale: float | None = None) -> dict[str, Any]:  # fmt: skip
    """Sweep the variable ``name`` over ``n`` values from ``lo_v`` to ``hi_v``.

    ``mode="equilibria"``: ``explore_math.sweep`` (at most ``explore_math.MAX_SWEEP``
    values), read into rows (value, count, stable count, largest real part, non-isolated)
    and candidates (``label``: saddle-node, pitchfork, transcritical, Hopf, enters or
    leaves the window, not named; refined by bisection, with ``converged``).

    ``mode="attractor"``: every value advanced at once from ``x0`` for ``transient``, then
    sampled (``sampler``, default: the maxima of x, each refined) for ``sample_time``; per
    value the last ``MAX_SAMPLES`` samples (time order), the ``period`` (``returns_repeat``
    on the samples, within ``SETTLED`` of ``scale``) and its ``state``: ``periodic``, ``not
    settled`` (still closing in, as near a doubling), ``chaos or a long period``,
    ``stopped`` or ``few samples``; candidates where the period of two settled values in a
    row changes (between them, not refined).

    A name the system does not use, or a range that is not finite with lo < hi, is a
    ``FunctionError``."""
    if name not in system.variables:
        raise FunctionError(f"the system does not use {name!r}; it uses "
                            f"{', '.join(system.variables) or 'no variables'}")  # fmt: skip
    lo_v, hi_v = _range(lo_v, hi_v)
    if mode == "attractor":
        clamped = ["values"] if not 2 <= int(n) <= MAX_SWEEP else []
        n = int(min(max(int(n), 2), MAX_SWEEP))
        vs = np.linspace(lo_v, hi_v, n)
        return _sweep_attractor(system, values, name, vs, x0=x0, dt=dt, transient=transient,
                                sample_time=sample_time, sampler=sampler or Sampler.maxima(0),
                                escape=escape, max_step=max_step, clamped=clamped,
                                scale=scale)  # fmt: skip
    from phasor.domains import explore_math as em  # the one equilibria sweep

    clamped = ["values"] if not 2 <= int(n) <= em.MAX_SWEEP else []
    n = int(min(max(int(n), 2), em.MAX_SWEEP))
    vals = {k: float(v) for k, v in values.items() if k != name}
    r = em.sweep(system, {**vals, name: lo_v}, name, lo_v, hi_v, n, lo, hi, grid)
    rows = []
    for row in r["table"]:
        st = [_stable_kind(e["kind"]) for e in row["equilibria"]]
        rows.append({"value": row["value"], "count": row["count"],
                     "stable": sum(1 for s in st if s is True), "max_real": row["max_real"],
                     "non_isolated": row["non_isolated"],
                     "equilibria": row["equilibria"]})  # fmt: skip
    cands = [{"label": c["kind"], "value": c["value"], "between": c["bracket"],
              "converged": c["converged"], "test": c["test"]} for c in r["candidates"]]  # fmt: skip
    return {"mode": "equilibria", "values": np.linspace(lo_v, hi_v, n), "rows": rows,
            "candidates": cands, "clamped": clamped, "steps": 0, "grid": r["grid"],
            "not_detected": r["not_detected"]}  # fmt: skip


def _sweep_attractor(system, values, name, vs, *, x0, dt, transient, sample_time, sampler,
                     escape, max_step, clamped, scale) -> dict[str, Any]:  # fmt: skip
    n = len(vs)
    d = system.dim
    dt = _finite(dt, "dt")
    transient, sample_time = _finite(transient, "transient"), _finite(sample_time, "sample_time")
    if dt == 0 or transient < 0 or sample_time <= 0:
        raise FunctionError("dt must not be 0, transient at least 0, sample_time positive")
    dt = abs(dt)
    x0 = np.zeros(d) + 0.1 if x0 is None else _point(system, np.asarray(x0)[:d])
    n_tr = max(0, round(transient / dt))
    n_s = max(1, round(sample_time / dt))
    if n * (n_tr + n_s) > MAX_SWEEP_WORK:
        shrink = MAX_SWEEP_WORK / (n * (n_tr + n_s))
        n_tr, n_s = int(n_tr * shrink), max(1, int(n_s * shrink))
        clamped = [*clamped, "work"]
    others = {k: float(v) for k, v in values.items()}
    vals = tuple(np.asarray(vs, dtype=np.float64) if k == name else others[k]
                 for k in system.variables)  # fmt: skip
    run = SectionRun(system.fn, vals, np.tile(x0, (n, 1)), dt, sampler, escape=escape,
                     max_step=max_step)  # fmt: skip
    run.advance(n_tr + n_s)
    cr = run.crossings()
    keep = cr["times"] >= n_tr * dt
    pts, sid, tim = cr["points"][keep], cr["seed"][keep], cr["times"][keep]
    coord = sampler.coord if sampler.kind == "maxima" else int(np.argmax(np.abs(sampler.normal)))
    if sampler.kind == "plane":  # plot a coordinate the plane does not fix
        coord = next(k for k in range(d) if k != coord)
    samples = np.full((n, MAX_SAMPLES), np.nan)
    size = float(scale) if scale else None
    rows = []
    codes = run.code
    for i, v in enumerate(vs):
        mine = sid == i
        p_i, t_i = pts[mine][-MAX_SAMPLES:], tim[mine][-MAX_SAMPLES:]
        samples[i, : len(p_i)] = p_i[:, coord]
        ref = size or max(1.0, float(np.abs(p_i).max()) if len(p_i) else 1.0)  # orbit size
        state, period = "few samples", None
        if codes[i] != fm.ALIVE:
            state = "stopped"
        elif len(p_i) >= 8:
            rets = list(zip(p_i, t_i, strict=True))
            m, why = returns_repeat(rets, SETTLED, ref, drift=False)
            if m:
                state, period = "periodic", m
            elif why or _closing(p_i[:, coord]):
                state = "not settled"
            else:
                state = "chaos or a long period"
        rows.append(
            {
                "value": float(v),
                "period": period,
                "state": state,
                "samples": int(np.isfinite(samples[i]).sum()),
                "stopped": None if codes[i] == fm.ALIVE else _STOP.get(int(codes[i]), "stopped"),
            }
        )
    cands = []
    last = None
    for row in rows:
        if row["state"] not in ("periodic", "chaos or a long period"):
            continue
        if last is not None:
            pa, pb = last["period"], row["period"]
            label = None
            if pa and pb and pb == 2 * pa:
                label = "period doubling"
            elif pa and pb and pa == 2 * pb:
                label = "period halving"
            elif pa and pb and pa != pb:
                label = "period change"
            elif (pa is None) != (pb is None):
                label = "onset of chaos or a long period" if pb is None else "a periodic window"
            if label and len(cands) < MAX_CANDIDATES:
                cands.append({"label": label, "between": [last["value"], row["value"]],
                              "value": None, "change": [pa, pb]})  # fmt: skip
        last = row
    return {"mode": "attractor", "values": vs, "rows": rows, "samples": samples,
            "coord": coord, "candidates": cands, "clamped": clamped,
            "steps": n_tr + n_s, "transient": n_tr * dt}  # fmt: skip


def _closing(s: np.ndarray) -> bool:
    """Whether a sequence of samples (time order) is near a repeat of some period k <= 16 and
    still closing in: its latest k-differences under 5% of the samples' spread and not
    growing (chaos keeps differences comparable to its spread for every k)."""
    s = s[np.isfinite(s)]
    spread = max(float(np.ptp(s)), 1e-12)
    for k in range(1, min(MAX_LOOPS, len(s) // 3) + 1):
        d = np.abs(s[k:] - s[:-k])
        third = max(len(d) // 3, 1)
        if d[-third:].max() < 0.05 * spread and d[-third:].mean() <= d[:third].mean():
            return True
    return False
