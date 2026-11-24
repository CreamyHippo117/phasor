"""Dynamical-systems numerics: a planar system x' = f(x, y), y' = g(x, y) from formulas,
trajectories (RK4), nullclines, fixed points (Newton) and their classification from the
Jacobian's eigenvalues. Pure NumPy and SymPy, no drawing.

A thin 2D front of the flow core (``flow_math``, Milestone 26): the field, its Jacobian,
the stepping, Newton and the classification all live there; this module keeps the planar
names and results the ``dyn`` objects have always used.

Formulas are read with the whitelist parser (never evaluated as Python). Named constants
(``constants={"mu": 1.5}``) may appear in them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from phasor.domains import flow_math as fm
from phasor.domains.flow_math import KINDS, FunctionError, classify, parse_constants
from phasor.render.gpu.codegen import parse_expression

__all__ = [
    "KINDS",
    "FunctionError",
    "System",
    "classify",
    "fixed_points",
    "nullcline_segments",
    "parse_constants",
    "trajectories",
]  # fmt: skip


@dataclass(frozen=True)
class System:
    """x' = f(x, y), y' = g(x, y)."""

    f_source: str
    g_source: str
    f: sp.Expr
    g: sp.Expr

    @classmethod
    def parse(cls, f: str, g: str, constants: dict[str, float] | str | None = None) -> System:
        consts = parse_constants(constants) if isinstance(constants, str) else dict(constants or {})
        names = ("x", "y", *consts)
        out = []
        for src in (f, g):
            e = parse_expression(src, names)
            e = e.subs({sp.Symbol(k, real=True): float(v) for k, v in consts.items()})
            left = e.free_symbols - set(_xy())
            if left:
                raise FunctionError(f"{src!r} uses {sorted(map(str, left))}: not x or y")
            out.append(e)
        return cls(f, g, out[0], out[1])

    @property
    def flow(self) -> fm.System:
        """The same field as a ``flow_math.System`` (constants already substituted)."""
        return fm.System.from_exprs((self.f, self.g), (self.f_source, self.g_source))

    def _fn(self):
        fields = self.flow.fn.fields

        def call(px, py):
            return tuple(fields([px, py]))

        return call

    def __call__(self, x, y):
        return self._fn()(x, y)

    def jacobian_at(self, x: float, y: float) -> np.ndarray:
        return self.flow.jacobian_exact((x, y))

    def jacobian(self):
        """A function ``(x, y) -> 2x2 matrix`` (evaluated, vectorised over scalars)."""
        jac = self.flow.fn.jacobians
        return lambda x, y: jac(np.array([[float(x), float(y)]]))[0]


def _xy():
    return sp.Symbol("x", real=True), sp.Symbol("y", real=True)


# ------------------------------------------------------------------ trajectories


def trajectories(system: System, seeds, t_end: float, steps: int = 400,
                 bounds=(-1e3, 1e3, -1e3, 1e3), direction: float = 1.0) -> np.ndarray:  # fmt: skip
    """RK4 paths from each seed: an array (S, steps + 1, 2). A path that leaves ``bounds``
    (or blows up) stays NaN from there on, so a drawing can cut it. ``direction=-1`` runs
    time backwards (toward where the flow came from)."""
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    x0, x1, y0, y1 = bounds
    h = direction * t_end / steps
    return fm.integrate(system.flow.fn, (), seeds, steps, h,
                        box=((x0, y0), (x1, y1))).points  # fmt: skip


# ------------------------------------------------------------------ nullclines


def nullcline_segments(system: System, which: str, bounds, n: int = 120) -> np.ndarray:
    """Line segments (M, 2, 2) tracing f = 0 (``which="x"``: where x does not change) or
    g = 0 (``"y"``) inside ``bounds = (x0, x1, y0, y1)``: marching squares on an n x n grid,
    crossings placed by linear interpolation."""
    x0, x1, y0, y1 = bounds
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    a, b = system(X, Y)
    z = a if which == "x" else b
    z = np.where(np.isfinite(z), z, np.nan)
    segs = []
    # cell corners: 00 (i,j), 10 (i+1,j), 11 (i+1,j+1), 01 (i,j+1)
    for i in range(n - 1):
        for j in range(n - 1):
            v = (z[i, j], z[i + 1, j], z[i + 1, j + 1], z[i, j + 1])
            if not np.isfinite(v).all():
                continue
            pts_xy = ((xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]),
                      (xs[i], ys[j + 1]))  # fmt: skip
            cross = []
            for e in range(4):
                va, vb = v[e], v[(e + 1) % 4]
                if (va < 0) != (vb < 0):
                    t = va / (va - vb)
                    pa, pb = pts_xy[e], pts_xy[(e + 1) % 4]
                    cross.append((pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])))
            if len(cross) == 2:
                segs.append(cross)
            elif len(cross) == 4:  # a saddle cell: join by the centre value's sign
                segs.append([cross[0], cross[1]])
                segs.append([cross[2], cross[3]])
    return np.array(segs, dtype=np.float64).reshape(-1, 2, 2)


# ------------------------------------------------------------------ fixed points


def fixed_points(system: System, bounds, grid: int = 9, tol: float = 1e-9) -> np.ndarray:
    """Points inside ``bounds`` where f = g = 0, by Newton's method from a grid of starts,
    sorted by x then y. Returns (K, 2)."""
    x0, x1, y0, y1 = bounds
    return fm.equilibria(system.flow, (), (x0, y0), (x1, y1), grid, singular="drop")["points"]


def eigenvalue_label(c: dict) -> str:
    """A short label: ``saddle (1.5, -2)`` or ``stable spiral (-0.5 +/- 1.2i)``: the
    eigenvalues, in plain characters every font draws."""
    (a, b), (c2, _) = c["eigenvalues"]
    if abs(b) > 1e-9:
        return f"{c['kind']} ({a:.3g} +/- {abs(b):.3g}i)"
    return f"{c['kind']} ({a:.3g}, {c2:.3g})"


def hash_seeds(n: int, extent, seed: int = 0) -> np.ndarray:
    """``n`` start points spread evenly over the window (a grid with a small fixed wobble)."""
    w, h = extent
    cols = max(1, round(math.sqrt(n * w / h)))
    rows = max(1, math.ceil(n / cols))
    gx = (np.arange(cols) + 0.5) / cols
    gy = (np.arange(rows) + 0.5) / rows
    X, Y = np.meshgrid(gx, gy)
    pts = np.c_[X.ravel(), Y.ravel()][:n]
    # a small, fixed wobble (golden-ratio sequence, shifted by ``seed``): no two seeds line up,
    # and no random numbers are needed
    k = np.arange(len(pts)) + 1 + seed
    wobble = np.c_[(k * 0.6180339887) % 1.0, (k * 0.7548776662) % 1.0] - 0.5
    pts = pts + wobble * np.array([0.6 / cols, 0.6 / rows])
    return (pts - 0.5) * np.array([w, h])
