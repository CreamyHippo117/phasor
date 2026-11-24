"""Point sources for ``PointCloud`` (pure NumPy, no drawing).

Each source returns ``(points, value)``: points as complex numbers (x = Re, y = Im) and a
value in [0, 1) per point that the object turns into colour. Randomness only through the
generator handed in, which the object makes with ``rng(object_id, seed)`` (contract 3).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.random import Generator  # the type only; generators come from core.rng

from phasor.domains.flow_math import PRESETS  # one table of the classic constants

ROOT_CHUNK = 2048  # polynomials per batched eigenvalue solve (one thread's share)
# LAPACK lets go of the GIL, so the chunks solve side by side: a million complex roots
# (2^16 polynomials of degree 16) in about 0.5 s on 16 threads instead of 4.4 s on one
ROOT_THREADS = max(1, min(16, os.cpu_count() or 1))
ATTRACTORS = {  # kind -> the classic constants
    "clifford": (-1.4, 1.6, 1.0, 0.7),
    "dejong": (-2.0, -2.0, -1.2, 2.0),
    "lorenz": tuple(v[0] for v in PRESETS["lorenz"].variables.values()),  # sigma, rho, beta
}
ORBITS = 2048  # orbits an attractor runs side by side
TRANSIENT = 64  # first iterates of each orbit, dropped (not yet on the attractor)


def polynomial_roots(
    degree: int,
    coefficients: tuple[float, ...] = (-1.0, 1.0),
    count: int | None = None,
    gen: Generator | None = None,
    twist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The roots of polynomials of ``degree`` whose lower coefficients all come from
    ``coefficients``; the leading one is 1 if the set has 1, else its first nonzero entry
    (a sign flip of the whole polynomial changes no root). ``count``: how many polynomials,
    a random sample of them (with ``gen``); None or more than there are: all of them, in
    order. ``value`` is the polynomial's place in that list, so neighbours share a colour.

    ``twist`` turns the last entry of the set round the origin by that share of a full
    turn (b -> b e^(2 pi i twist)) in the lower coefficients; the leading one stays. The
    roots move continuously with it: the sample of polynomials does not depend on it, and
    the roots of polynomial k are always ``roots[k * degree : (k + 1) * degree]`` (in no
    particular order inside that run)."""
    if degree < 1:
        raise ValueError("degree must be at least 1")
    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.ndim != 1 or len(coeffs) < 1:
        raise ValueError("coefficients must be a non-empty list of numbers")
    nonzero = coeffs[coeffs != 0]
    if not len(nonzero):
        raise ValueError("coefficients must include a nonzero number")
    lead = 1.0 if (coeffs == 1).any() else float(nonzero[0])
    k = len(coeffs)
    total = k**degree
    if count is None or count >= total:
        if total > 1 << 26:
            raise ValueError(f"{total} polynomials is too many; give a count to sample them")
        index = np.arange(total, dtype=np.int64)
    else:
        if gen is None:
            raise ValueError("sampling polynomials needs a random generator")
        if total > 1 << 62:
            raise ValueError(f"{total} polynomials is too many to sample from")
        index = np.sort(gen.choice(total, size=int(count), replace=False))
    if twist % 1.0:
        coeffs = coeffs.astype(np.complex128)
        coeffs[-1] *= np.exp(2j * np.pi * float(twist))

    def solve(start: int) -> np.ndarray:
        idx = index[start : start + ROOT_CHUNK]
        digits = np.empty((len(idx), degree), dtype=np.int64)  # base-k digits, low first
        rest = idx.copy()
        for d in range(degree):
            digits[:, d] = rest % k
            rest //= k
        low = coeffs[digits]  # (m, degree): a_0 .. a_{degree-1}
        companion = np.zeros((len(idx), degree, degree), dtype=coeffs.dtype)
        if degree > 1:
            companion[:, np.arange(1, degree), np.arange(degree - 1)] = 1.0
        companion[:, :, -1] = -low / lead
        return np.linalg.eigvals(companion).reshape(-1).astype(np.complex128)

    starts = range(0, len(index), ROOT_CHUNK)
    if len(starts) > 1 and ROOT_THREADS > 1:
        with ThreadPoolExecutor(ROOT_THREADS) as pool:
            roots = list(pool.map(solve, starts))
    else:
        roots = [solve(s) for s in starts]
    values = np.repeat(index.astype(np.float64) / total, degree)
    return np.concatenate(roots), values


def attractor(
    kind: str,
    n: int,
    gen: Generator,
    constants: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points on a strange attractor, centred and scaled to fit the square
    [-1, 1] x [-1, 1]: ``clifford`` and ``dejong`` (2D maps), ``lorenz`` (the flow, seen
    from the side: x across, z up). Many orbits run side by side from random starts; their
    first iterates are dropped. ``value`` is how far each step jumped, as a rank (0-1), so
    the fast and slow parts of the attractor take different colours."""
    if kind not in ATTRACTORS:
        raise ValueError(f"kind must be one of {sorted(ATTRACTORS)}, got {kind!r}")
    c = tuple(constants) or ATTRACTORS[kind]
    if len(c) != len(ATTRACTORS[kind]):
        raise ValueError(f"{kind} takes {len(ATTRACTORS[kind])} constants, got {len(c)}")
    orbits = max(1, min(ORBITS, n))
    steps = -(-n // orbits)
    if kind == "lorenz":
        sigma, rho, beta = c
        p = gen.normal(0.0, 1.0, (orbits, 3)) + np.array([0.0, 0.0, 25.0])
        dt = 0.004

        def f(p):
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=1)

        def step(p):  # classic Runge-Kutta: a fixed dt
            k1 = f(p)
            k2 = f(p + 0.5 * dt * k1)
            k3 = f(p + 0.5 * dt * k2)
            k4 = f(p + dt * k3)
            return p + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        for _ in range(TRANSIENT * 8):
            p = step(p)
        out = np.empty((steps, orbits, 2))
        for s in range(steps):
            q = step(p)
            out[s] = q[:, [0, 2]]
            p = q
    else:
        a, b, cc, d = c
        x, y = gen.uniform(-1.0, 1.0, orbits), gen.uniform(-1.0, 1.0, orbits)

        def step(x, y):
            if kind == "clifford":
                return np.sin(a * y) + cc * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)
            return np.sin(a * y) - np.cos(b * x), np.sin(cc * x) - np.cos(d * y)

        for _ in range(TRANSIENT):
            x, y = step(x, y)
        out = np.empty((steps, orbits, 2))
        for s in range(steps):
            x, y = step(x, y)
            out[s, :, 0], out[s, :, 1] = x, y
    jump = np.zeros((steps, orbits))
    jump[1:] = np.hypot(*(out[1:] - out[:-1]).transpose(2, 0, 1))
    jump[0] = jump[1] if steps > 1 else 0.0
    pts = out.reshape(-1, 2)[:n]
    jump = jump.reshape(-1)[:n]
    lo, hi = pts.min(0), pts.max(0)
    mid, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2, 1e-12)
    pts = (pts - mid) / half
    rank = np.empty(len(jump))
    rank[np.argsort(jump, kind="stable")] = np.arange(len(jump)) / max(len(jump), 1)
    return pts[:, 0] + 1j * pts[:, 1], rank


def buddhabrot(
    n: int,
    max_iter: int,
    gen: Generator,
    min_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points of the Buddhabrot: the orbits z -> z^2 + c (from z = 0) of random c in
    the disc |c| < 2 that escape within ``max_iter`` steps (and not before ``min_iter``),
    every point of every such orbit. Points inside the main cardioid and the period-2 bulb
    never escape and are skipped at once. ``value`` is the orbit's escape time as
    log(time) / log(max_iter): short orbits and long ones take different colours, the
    "nebulabrot" colouring."""
    if max_iter < 2:
        raise ValueError("max_iter must be at least 2")
    batch = max(1024, (1 << 22) // max_iter)  # orbits per pass: about 64 MB of them
    pts: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    have = 0
    for _ in range(10_000):
        if have >= n:
            break
        r = 2.0 * np.sqrt(gen.random(batch))
        c = r * np.exp(2j * np.pi * gen.random(batch))
        q = (c.real - 0.25) ** 2 + c.imag**2
        inside = (q * (q + (c.real - 0.25)) <= 0.25 * c.imag**2) | (
            (c.real + 1) ** 2 + c.imag**2 <= 1 / 16
        )
        c = c[~inside]
        z = np.zeros_like(c)
        orbit = np.empty((max_iter, len(c)), dtype=np.complex128)
        escaped = np.full(len(c), -1)
        alive = np.ones(len(c), dtype=bool)
        for k in range(max_iter):
            z = np.where(alive, z * z + c, z)
            orbit[k] = z
            gone = alive & (z.real * z.real + z.imag * z.imag > 4.0)
            escaped[gone] = k + 1
            alive &= ~gone
            if not alive.any():
                break
        keep = escaped >= min_iter
        if not keep.any():
            continue
        e = escaped[keep]
        o = orbit[:, keep]
        mask = np.arange(max_iter)[:, None] < e[None, :]  # each orbit up to its escape
        pts.append(o.T[mask.T])
        vals.append(np.repeat(np.log(e) / np.log(max_iter), e))
        have += int(e.sum())
    if not pts:
        return np.zeros(0, dtype=np.complex128), np.zeros(0)
    return np.concatenate(pts)[:n], np.minimum(np.concatenate(vals)[:n], 0.999999)


def every_polynomial_count(degree: int, coefficients: tuple[float, ...]) -> int:
    """How many polynomials ``polynomial_roots`` enumerates (before any sampling)."""
    return len(coefficients) ** degree
