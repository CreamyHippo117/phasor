"""Numbers for the Explorer's AI tools (Milestone 27: ``find_equilibria``, ``flow_probe``,
``sweep_parameter`` in ``workspace/explorer.py``). Pure NumPy on the flow core
(``flow_math``); nothing here draws or writes. Every answer says what it was computed from
and how far it can be trusted: a tolerance is a number, never an adjective.

- ``lyapunov``: the largest Lyapunov exponent (Benettin) with the spread of two halves;
- ``probe``: what a seed does over a flow time: positions (only while they are
  reproducible and ``dt/2`` agrees), escape, a bounding box, a period, the dt check;
- ``sweep``: equilibria along one variable, and candidate bifurcations, refined by
  bisection and named by the local test that found them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from phasor.domains.flow_math import (
    ALIVE,
    ESCAPED,
    MERGE,
    NONFINITE,
    TOO_FAST,
    Compiled,
    FunctionError,
    System,
    classify,
    equilibria,
    integrate,
    newton,
    step,
)

MAX_SAMPLES = 64  # positions one probe reports
MAX_SEEDS = 16  # seeds of one probe call
MAX_RUN = 4000  # steps of one probe run (the dt/2 run takes twice as many; ~1 s a seed here)
MIN_COMPARED = 10  # steps dt and dt/2 must share before they can be said to agree
MAX_SWEEP = 64  # values of one sweep
MAX_CANDIDATES = 12  # bifurcation candidates refined by bisection in one sweep
BISECT_ITERS = 40
CONTINUATION_STEPS = 8  # parameter sub-steps an equilibrium is followed by between two values
EDGE = 1e-3  # a point this share of the window from a side is on the window's edge
NEWTON_TOL = 1e-9  # the residual Newton stops at (``flow_math.newton``)
CHAOS_TOL = 0.05  # a largest Lyapunov exponent above this (per unit flow time): chaotic
DT_AGREE = 1e-3  # dt and dt/2 agree when their paths differ by less than this share of the window
DT_LYAPUNOV_TIMES = 5.0  # ... compared over this many Lyapunov times on a chaotic path
GROWTH = math.log(1e-3 / 1e-12)  # an error of 1e-12 of the window grown to 1e-3

KINDS = ("saddle-node", "pitchfork", "Hopf", "transcritical", "enters or leaves the window",
         "not named")  # fmt: skip


def sig(v, digits: int = 6) -> float | None:
    """A float for an answer: ``digits`` significant digits (so 1e-9 stays 1e-9), None when
    it is not finite."""
    v = float(v)
    return float(f"{v:.{digits}g}") if math.isfinite(v) else None


def _sv(v) -> list[float | None]:
    return [sig(a) for a in np.asarray(v, dtype=np.float64).ravel()]


# ---------------------------------------------------------------- Lyapunov and probes


def lyapunov(fn: Compiled, vals, seeds, steps: int, dt: float, *, renorm: int = 10,
             max_step: float | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """The largest Lyapunov exponent of each seed (Benettin: a twin ``1e-8`` away,
    renormalised every ``renorm`` steps; the first fifth of the run is a transient and not
    counted). ``spread`` is the difference between the estimates of the two halves counted.
    ``converged`` only when the spread is under 10% of the value and the value is clear of 0
    by twice the spread (a limit cycle's or a centre's 0 is never 'converged')."""
    seeds = np.atleast_2d(np.asarray(seeds, dtype=np.float64))
    s, d = seeds.shape
    d0 = 1e-8
    e = np.ones(d) / math.sqrt(d)
    p = np.vstack([seeds, seeds + d0 * e])
    blocks = max(int(steps) // renorm, 0)
    logs = np.full((blocks, s), np.nan)
    alive = np.ones(s, dtype=bool)
    with np.errstate(all="ignore"):
        for b in range(blocks):
            for _ in range(renorm):
                p, code = step(fn, vals, p, dt, max_step)
                alive &= (code[:s] == ALIVE) & (code[s:] == ALIVE)
            if not alive.any():
                break
            gap = p[s:] - p[:s]
            dist = np.linalg.norm(gap, axis=1)
            ok = alive & np.isfinite(dist) & (dist > 0)
            logs[b, ok] = np.log(dist[ok] / d0)
            alive &= ok
            scale = (d0 / np.where(ok, dist, 1.0))[:, None]
            p[s:] = np.where(ok[:, None], p[:s] + gap * scale, p[s:])
            p = np.where(np.isfinite(p), p, 0.0)  # a stopped seed keeps finite arithmetic
    out = []
    skip = blocks // 5
    span = renorm * abs(dt)
    for k in range(s):
        used = logs[skip:, k]
        if not alive[k] or len(used) < 4 or not np.isfinite(used).all():
            out.append({"value": None, "spread": None, "T": sig(len(used) * span),
                        "converged": False})  # fmt: skip
            continue
        half = len(used) // 2
        lam = float(used.sum() / (len(used) * span))
        a = float(used[:half].sum() / (half * span))
        b = float(used[half:].sum() / ((len(used) - half) * span))
        spread = abs(a - b)
        ok = spread <= 0.1 * abs(lam) and abs(lam) > 2 * spread
        out.append({"value": sig(lam), "spread": sig(spread), "T": sig(len(used) * span),
                    "converged": bool(ok)})  # fmt: skip
    return out


def _period(pts: np.ndarray, dt: float, size: float) -> dict[str, Any]:
    """A period estimate from the second half of a path: the times it comes back within 1%
    of its size to where that half started (at least three returns at steady intervals)."""
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = len(pts)
    if n < 20 or size <= 0:
        return {"period": None, "returns": 0}
    ref = pts[n // 2]
    dist = np.linalg.norm(pts[n // 2 :] - ref, axis=1)
    if dist.max() < 1e-9 * max(size, 1.0):
        return {"period": None, "returns": 0, "period_note": "at rest"}
    tol = 0.01 * size
    k = np.arange(1, len(dist) - 1)
    mins = k[(dist[k] < tol) & (dist[k] <= dist[k - 1]) & (dist[k] <= dist[k + 1])]
    away = np.flatnonzero(dist > 2 * tol)
    if not len(away):
        return {"period": None, "returns": 0, "period_note": "stays within 2% of its size"}
    mins = mins[mins > away[0]]
    keep: list[int] = []
    for m in mins:  # one return is one dip: it must leave between two
        if not keep or (dist[keep[-1] : m] > 2 * tol).any():
            keep.append(int(m))
    if len(keep) < 3:
        return {"period": None, "returns": len(keep)}
    gaps = np.diff(np.r_[0, keep]) * abs(dt)
    if gaps.std() > 0.05 * gaps.mean():
        return {"period": None, "returns": len(keep), "period_note": "returns at uneven intervals"}
    return {"period": sig(gaps.mean()), "returns": len(keep)}


def probe(system: System, values, seeds, T: float, dt: float, *, samples: int = 16,
          size: float = 1.0, escape=None,
          max_step: float | None = None) -> dict[str, Any]:  # fmt: skip
    """Integrate ``seeds`` (S, dim) for flow time ``T`` (negative: backward) and say what
    happened, with the numbers it was computed from (``size``: the window's diagonal):

    - ``samples`` positions at even times, only up to ``positions_until``: the smaller of
      ``chaotic_beyond`` (an error of 1e-12 of the window grown to 1e-3 at the largest
      Lyapunov exponent) and the time the ``dt`` and ``dt/2`` paths part; statistics after;
    - ``escaped`` and ``when``; the bounding box; the final speed; a period estimate;
    - the dt check: ``agree`` only when at least ``MIN_COMPARED`` steps were compared and the
      seed was not stopped as too fast or not finite.

    At most ``MAX_RUN`` steps (``clamped``)."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    seeds = np.atleast_2d(np.asarray(seeds, dtype=np.float64))
    if seeds.ndim != 2 or seeds.shape[1] != system.dim:
        raise FunctionError(f"a seed has {system.dim} coordinates here")
    if len(seeds) > MAX_SEEDS:
        raise FunctionError(f"at most {MAX_SEEDS} seeds in one probe")
    if not np.isfinite(seeds).all() or not math.isfinite(T):
        raise FunctionError("seeds and T must be finite numbers")
    samples = int(min(max(samples, 2), MAX_SAMPLES))
    sign = -1.0 if T < 0 else 1.0
    h = abs(float(dt)) * sign
    want = round(abs(T) / abs(h))
    n = int(min(want, MAX_RUN))
    fn = system.fn
    with np.errstate(all="ignore"):
        path = integrate(fn, vals, seeds, n, h, escape=escape, max_step=max_step)
        half = integrate(fn, vals, seeds, 2 * n, h / 2, escape=escape, max_step=max_step)
        lyap = lyapunov(fn, vals, seeds, n, h, max_step=max_step)
    times = np.unique(np.linspace(0, n, samples).round().astype(int))
    tol = DT_AGREE * max(size, 1e-9)
    out = []
    for k in range(len(seeds)):
        pts = path.points[k]
        fin = pts[np.isfinite(pts).all(axis=1)]
        lam = lyap[k]["value"]
        beyond = GROWTH / lam if lam is not None and lam > CHAOS_TOL else None
        limit = n if beyond is None else min(n, int(beyond / abs(h)))
        # dt against dt/2, over the run (or five Lyapunov times on a chaotic path, past which
        # any truncation error has grown); ``agree_until``: where they part over the whole run
        check = limit if beyond is None else min(limit, int(DT_LYAPUNOV_TIMES / lam / abs(h)))
        a = path.points[k]
        b = half.points[k, ::2]
        both = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        gaps = np.where(both, np.linalg.norm(np.where(both[:, None], a - b, 0.0), axis=1), np.nan)
        within = gaps[: check + 1][both[: check + 1]]
        compared = max(len(within) - 1, 0)
        diff = float(within.max()) if len(within) else None
        over = np.flatnonzero(both & (gaps > tol))
        code = int(path.code[k])
        why = None
        if code in (NONFINITE, TOO_FAST):
            why = "the seed was stopped (" + ("not finite" if code == NONFINITE else "too fast") \
                + "): the step is too large for this field here"  # fmt: skip
        elif compared < MIN_COMPARED:
            why = f"only {compared} step(s) compared (at least {MIN_COMPARED} are needed)"
        elif diff is None or diff > tol:
            why = "dt and dt/2 differ by more than the tolerance"
        agree = why is None
        # positions only while reproducible and while dt and dt/2 agree
        shown = limit if not len(over) else min(limit, int(over[0]) - 1)
        shown = max(shown, 0)
        reason = None
        if shown < n:
            reason = ("dt and dt/2 part" if len(over) and shown < limit
                      else "chaotic_beyond")  # fmt: skip
        box = (fin.min(axis=0), fin.max(axis=0)) if len(fin) else None
        speed = None
        if len(fin) and shown == n:
            with np.errstate(all="ignore"):
                speed = sig(np.linalg.norm(fn.velocity(fin[-1:], vals)[0]))
        rec: dict[str, Any] = {
            "x0": _sv(seeds[k]),
            "samples": [{"t": sig(i * h), "at": _sv(pts[i])}
                        for i in times if i <= shown and np.isfinite(pts[i]).all()],
            "positions_until": sig(shown * h), "positions_cut_by": reason,
            "escaped": code == ESCAPED,
            "stopped": {NONFINITE: "not finite", TOO_FAST: "too fast",
                        ESCAPED: "escaped"}.get(code),
            "when": sig(path.at[k] * h) if path.at[k] >= 0 else None,
            "bounding_box": None if box is None else [_sv(box[0]), _sv(box[1])],
            "final_speed": speed,
            "largest_lyapunov": lyap[k],
            "chaotic_beyond": None if beyond is None else sig(beyond),
            "dt_check": {"dt": sig(abs(h)), "against": sig(abs(h) / 2),
                         "compared_up_to": sig(check * h), "steps_compared": compared,
                         "max_difference": None if diff is None else sig(diff, 3),
                         "tolerance": sig(tol), "agree": agree, "why": why,
                         "agree_until": sig(over[0] * h) if len(over) else None},
            "converged": agree,
            **_period(fin, h, float(np.linalg.norm(box[1] - box[0])) if box else 0.0),
        }  # fmt: skip
        later = pts[shown:]
        later = later[np.isfinite(later).all(axis=1)]
        if shown < n and len(later):
            rec["after_positions"] = {
                "mean": _sv(later.mean(axis=0)), "spread": _sv(later.std(axis=0)),
                "note": f"positions are not given past {sig(shown * h)} ({reason}); "
                        "statistics only",
            }  # fmt: skip
        out.append(rec)
    return {"seeds": out, "T": sig(n * h), "dt": sig(abs(h)), "steps": n,
            "clamped": bool(want > n), "max_steps": MAX_RUN,
            "tolerance": {"dt_agree": sig(tol), "chaos": CHAOS_TOL,
                          "min_steps_compared": MIN_COMPARED}}  # fmt: skip


# ---------------------------------------------------------------- equilibria and sweeps


def classified_equilibria(system: System, values: dict[str, float], lo, hi,
                          grid: int = 9) -> dict[str, Any]:  # fmt: skip
    """``flow_math.equilibria`` with each point classified from its exact Jacobian
    (``classify``, plus ``at``), and the search's window, grid and ``non_isolated``."""
    found = equilibria(system, values, lo, hi, grid)
    pts = []
    for at in found["points"]:
        c = classify(system.jacobian_exact(at, values))
        c["at"] = at
        pts.append(c)
    return {"points": pts, **{k: found[k] for k in ("non_isolated", "grid", "starts", "window")}}


def _complex_real(c: dict) -> float | None:
    """The largest real part among the complex eigenvalues (None: there are none)."""
    re = [a for a, b in c["eigenvalues"] if abs(b) > c["tolerance"]]
    return max(re) if re else None


def _real_real(c: dict) -> float | None:
    re = [a for a, b in c["eigenvalues"] if abs(b) <= c["tolerance"]]
    return max(re) if re else None


def _max_real(c: dict) -> float | None:
    re = [a for a, _ in c["eigenvalues"]]
    return max(re) if re else None


def _near(c: dict, others: list[dict], reach: float) -> bool:
    return any(np.linalg.norm(np.asarray(c["at"]) - np.asarray(o["at"])) < reach for o in others)


def _newton_at(system: System, values: dict[str, float], at, size: float) -> dict | None:
    """The equilibrium near ``at`` at these values (Newton from ``at``), classified."""
    vals = system.values_of(values)
    with np.errstate(all="ignore"):
        p, ok = newton(system.fn, vals, np.asarray([at], dtype=np.float64), cap=0.05 * size)
    if not ok[0] or np.linalg.norm(p[0] - np.asarray(at)) > 0.05 * size:
        return None
    c = classify(system.jacobian_exact(p[0], values))
    c["at"] = p[0]
    return c


def _follow(system, values, name, c: dict, v0: float, v1: float, size: float) -> dict | None:
    """An equilibrium followed from ``v0`` to ``v1`` in small steps of the variable
    (continuation), so two branches that cross are not swapped as a nearest match would."""
    here = c
    for k in range(1, CONTINUATION_STEPS + 1):
        v = v0 + (v1 - v0) * k / CONTINUATION_STEPS
        here = _newton_at(system, {**values, name: v}, here["at"], size)
        if here is None:
            return None
    return here


def _unmatched(more: list[dict], fewer: list[dict]) -> list[dict]:
    """The points of ``more`` left once each point of ``fewer`` has taken its nearest
    (closest pairs first): the ones that appeared."""
    if not fewer:
        return list(more)
    pairs = sorted((float(np.linalg.norm(np.asarray(m["at"]) - np.asarray(f["at"]))), i, j)
                   for i, m in enumerate(more) for j, f in enumerate(fewer))  # fmt: skip
    used_m: set[int] = set()
    used_f: set[int] = set()
    for _, i, j in pairs:
        if i not in used_m and j not in used_f:
            used_m.add(i)
            used_f.add(j)
    return [m for i, m in enumerate(more) if i not in used_m]


def _on_edge(at, lo: np.ndarray, hi: np.ndarray) -> bool:
    at = np.asarray(at)
    room = EDGE * (hi - lo)
    return bool(((at - lo) <= room).any() or ((hi - at) <= room).any())


def sweep(system: System, values: dict[str, float], name: str, lo: float, hi: float,
          n: int, wlo, whi, grid: int = 9) -> dict[str, Any]:  # fmt: skip
    """Equilibria along ``name`` from ``lo`` to ``hi`` (``n`` values, at most ``MAX_SWEEP``),
    with candidate bifurcations between neighbouring values, refined by bisection
    (``BISECT_ITERS`` halvings) and named by the local test that found them:

    - the number of equilibria changes inside the window: ``saddle-node`` (a pair appears
      or disappears away from the other points) or ``pitchfork`` (two appear beside one
      that stays); new points on the window's edge are ``enters or leaves the window``, and
      a side with non-isolated equilibria (a line of them) is ``not named``;
    - an equilibrium followed by continuation has a complex pair crossing the imaginary
      axis: ``Hopf``; a real eigenvalue crossing zero with the count unchanged:
      ``transcritical`` (two branches exchange stability).

    ``converged`` per candidate: the count was the same on each side all through the
    bisection (or the crossing was followed all the way) and the bracket is under its
    tolerance. ``precision`` says what limits the value besides the bracket. Global
    bifurcations (homoclinic orbits, crises) are not detected by any of these."""
    if name not in system.variables:
        raise FunctionError(f"the system does not use {name!r}; it uses "
                            f"{', '.join(system.variables) or 'no variables'}")  # fmt: skip
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and math.isfinite(hi - lo)):
        raise FunctionError("a sweep needs finite lo < hi")
    n = int(min(max(n, 2), MAX_SWEEP))
    wlo, whi = np.asarray(wlo, dtype=np.float64), np.asarray(whi, dtype=np.float64)
    size = float(np.linalg.norm(whi - wlo))
    merge = MERGE * float((whi - wlo).max())
    vs = np.linspace(lo, hi, n)

    def at(v: float) -> dict[str, Any]:
        return classified_equilibria(system, {**values, name: float(v)}, wlo, whi, grid)

    tables = [at(v) for v in vs]
    width = (hi - lo) / (n - 1) / 2**BISECT_ITERS
    step_v = (hi - lo) / (n - 1)
    tight = 4 * width + 1e-12 * max(abs(lo), abs(hi), 1.0)
    precision = {"merge_distance": merge, "newton_residual": NEWTON_TOL,
                 "note": "two roots closer than merge_distance count once, so a fold or "
                 "pitchfork is seen a little early or late: about (merge_distance / 2)**2 "
                 "times the curvature, more than the bracket"}  # fmt: skip
    # ---- count changes: bisected between neighbours, named afterwards
    events: list[dict[str, Any]] = []
    unnamed: list[dict[str, Any]] = []
    for i in range(n - 1):
        ta, tb = tables[i], tables[i + 1]
        na, nb = len(ta["points"]), len(tb["points"])
        if na == nb or len(events) + len(unnamed) >= MAX_CANDIDATES:
            continue
        if ta["non_isolated"] or tb["non_isolated"]:
            unnamed.append({"kind": "not named", "value": float((vs[i] + vs[i + 1]) / 2),
                            "bracket": [float(vs[i]), float(vs[i + 1])],
                            "test": f"the number of equilibria changes from {na} to {nb}",
                            "at": [], "converged": False,
                            "precision": {**precision, "bracket": float(step_v)},
                            "why": "equilibria are not isolated on one side (a line or plane of "
                                   "them): no local test applies"})  # fmt: skip
            continue
        x0, x1, monotone = float(vs[i]), float(vs[i + 1]), True
        for _ in range(BISECT_ITERS):
            mid = (x0 + x1) / 2
            m = at(mid)
            k = len(m["points"])
            if not m["non_isolated"] and min(na, nb) < k < max(na, nb):
                x0 = x1 = mid  # the degenerate root itself, counted once: found exactly
                break
            if m["non_isolated"] or k not in (na, nb):
                monotone = False  # a third count between: two events, or the grid missed points
                break
            x0, x1 = (mid, x1) if k == na else (x0, mid)
        events.append({"i": i, "x0": x0, "x1": x1, "counts": [na, nb], "monotone": monotone})
    # A value of the sweep on the event itself counts its degenerate root once: the change
    # shows as two, k -> k' at that sample and k' -> k'' after it. Join them; when the counts
    # on both sides are equal (a transcritical, two roots meeting) there is no count change.
    artifact: set[int] = set()
    joined: list[dict[str, Any]] = []
    for ev in events:
        last = joined[-1] if joined else None
        j = ev["i"]
        near = 1e-3 * step_v  # both changes at the sample, within the merge scale of it
        if (last is not None and last["i_end"] == j - 1 and abs(last["x1"] - vs[j]) <= near
                and abs(ev["x0"] - vs[j]) <= near):  # fmt: skip
            last["counts"] = [last["counts"][0], ev["counts"][1]]
            last["x1"], last["i_end"] = ev["x1"], j
            last["monotone"] = last["monotone"] and ev["monotone"]
            last["at_sample"] = float(vs[j])
        else:
            joined.append({**ev, "i_end": j, "at_sample": None})
    cands: list[dict[str, Any]] = list(unnamed)
    for ev in joined:
        k0, k1 = ev["counts"]
        if k0 == k1:
            artifact.update(range(ev["i"], ev["i_end"] + 1))
            continue
        x0, x1 = ev["x0"], ev["x1"]
        v = ev["at_sample"] if ev["at_sample"] is not None else (x0 + x1) / 2
        d = 1e-4 * step_v  # a little either side: the points that appear are apart there
        sides = (at(v - d), at(v + d))
        fewer, more = sorted(sides, key=lambda s: len(s["points"]))
        new = _unmatched(more["points"], fewer["points"])
        ends = (at(x0)["points"], at(x1)["points"])
        e_few, e_more = sorted(ends, key=len)
        edge = any(_on_edge(c["at"], wlo, whi) for c in _unmatched(e_more, e_few))
        cand = {"value": float(v), "bracket": [float(x0), float(x1)],
                "test": f"the number of equilibria changes from {k0} to {k1}",
                "at": [c["at"] for c in new],
                "converged": ev["monotone"] and ((x1 - x0) <= 3 * tight
                                                 or ev["at_sample"] is not None),
                "precision": {**precision, "bracket": float(x1 - x0)}}  # fmt: skip
        if edge:
            cand["kind"] = "enters or leaves the window"
        elif (abs(k1 - k0) == 2 and len(new) == 2 and not more["non_isolated"]
              and float(np.linalg.norm(np.asarray(new[0]["at"]) - np.asarray(new[1]["at"])))
              < 0.05 * size):  # fmt: skip
            mid_pt = (np.asarray(new[0]["at"]) + np.asarray(new[1]["at"])) / 2
            beside = any(np.linalg.norm(np.asarray(c["at"]) - mid_pt) < 0.01 * size
                         for c in fewer["points"])  # fmt: skip
            cand["kind"] = "pitchfork" if beside else "saddle-node"
        else:
            cand["kind"] = "not named"
            cand["converged"] = False
            cand["why"] = ("the points that appear are not born together as a pair (the grid "
                           "of starts found or missed them): no local test applies")  # fmt: skip
        cands.append(cand)
    # ---- crossings, following each equilibrium by continuation
    for i in range(n - 1):
        a, b = tables[i]["points"], tables[i + 1]["points"]
        for ca in a:
            if len(cands) >= MAX_CANDIDATES:
                break
            cb = _follow(system, values, name, ca, float(vs[i]), float(vs[i + 1]), size)
            if cb is None:
                continue
            for kind, part, test in (
                ("Hopf", _complex_real, "a complex pair of eigenvalues crosses the imaginary axis"),
                ("transcritical", _real_real,
                 "a real eigenvalue crosses zero with the number of equilibria unchanged"),
            ):  # fmt: skip
                ra, rb = part(ca), part(cb)
                if ra is None or rb is None or (ra < 0) == (rb < 0):
                    continue
                if kind != "Hopf" and len(a) != len(b) and i not in artifact:
                    continue  # the count change names it
                x0, x1, here, ok = float(vs[i]), float(vs[i + 1]), ca, True
                for _ in range(BISECT_ITERS):
                    mid = (x0 + x1) / 2
                    c = _newton_at(system, {**values, name: mid}, here["at"], size)
                    r = None if c is None else part(c)
                    if r is None:
                        ok = False
                        break
                    if (r < 0) == (ra < 0):
                        x0, here = mid, c
                    else:
                        x1 = mid
                cand = {"kind": kind, "value": (x0 + x1) / 2, "bracket": [x0, x1], "test": test,
                        "at": [here["at"]], "converged": ok and (x1 - x0) <= tight,
                        "precision": {"bracket": x1 - x0, "newton_residual": NEWTON_TOL,
                                      "eigenvalue_tolerance": here["tolerance"]}}  # fmt: skip
                if kind == "Hopf":
                    ims = [abs(b) for _, b in here["eigenvalues"] if abs(b) > here["tolerance"]]
                    cand["frequency"] = sig(max(ims)) if ims else None
                cands.append(cand)
    # a point that stays through a pitchfork crosses zero there too: the pitchfork names it
    counted = [c for c in cands if c["kind"] in ("saddle-node", "pitchfork")]
    cands = [c for c in cands if not (c["kind"] == "transcritical" and any(
        abs(c["value"] - j["value"]) <= 1.01 * step_v for j in counted))]  # fmt: skip
    merged: list[dict[str, Any]] = []
    for c in sorted(cands, key=lambda c: c["value"]):
        near = max(tight, 1e-9 * abs(c["value"]))
        same = next((m for m in merged if m["kind"] == c["kind"] in ("Hopf", "transcritical")
                     and abs(m["value"] - c["value"]) <= near), None)  # fmt: skip
        if same is None:
            merged.append(c)
        else:
            same["at"] += c["at"]
            same["converged"] = same["converged"] and c["converged"]
    for c in merged:
        c["value"] = float(c["value"])
        c["bracket"] = [float(x) for x in c["bracket"]]
        c["at"] = [_sv(p) for p in c["at"]]
        c.pop("counts", None)
        c.pop("beside", None)
        c["precision"] = {k: (sig(v) if isinstance(v, float) else v)
                          for k, v in c["precision"].items()}  # fmt: skip
    table = []
    for v, t in zip(vs, tables, strict=True):
        reals = [m for m in (_max_real(c) for c in t["points"]) if m is not None]
        table.append({
            "value": sig(v), "count": len(t["points"]),
            "kinds": [c["kind"] for c in t["points"]],
            "max_real": sig(max(reals)) if reals else None,
            "non_isolated": t["non_isolated"],
            "equilibria": [{"at": _sv(c["at"]), "kind": c["kind"],
                            "max_real": None if _max_real(c) is None else sig(_max_real(c))}
                           for c in t["points"]],
        })  # fmt: skip
    return {"name": name, "lo": float(lo), "hi": float(hi), "n": n, "table": table,
            "candidates": merged,
            "bisection": {"iterations": BISECT_ITERS, "width": width},
            "window": [[float(a), float(b)] for a, b in zip(wlo, whi, strict=True)],
            "grid": tables[0]["grid"],
            "not_detected": "global bifurcations (homoclinic, heteroclinic orbits, crises), "
                            "and events between two values with no local trace"}  # fmt: skip
