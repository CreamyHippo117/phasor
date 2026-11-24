"""The flow core (Milestone 26, EXPLORER.md): an autonomous system x' = F(x) in 2 or 3
dimensions from formulas, and everything the Explorer asks of it. Pure NumPy and SymPy, no
drawing.

- ``System``: the components as SymPy expressions in the coordinates (``x, y`` or
  ``x, y, z``) and named variables. A formula is read once per text (``compiled``); variable
  values are *arguments*, so moving a slider never re-parses or re-compiles anything.
- ``step`` / ``integrate``: classic Runge-Kutta with a fixed ``dt`` (a seed's path never
  depends on the other seeds in the batch). ``max_step`` bounds how far one sub-step may move
  a seed (a share of the window): a fast seed is sub-stepped up to ``MAX_SUBSTEPS`` times,
  then stopped and counted as too fast (lint ``flow_too_fast``). A non-finite velocity stops
  only that seed (NaN from there), never raises.
- ``FlowSim``: a set of seeds advected as a ``core.simulation.Simulation`` (checkpoints),
  with closed and open curves whose sampling is refined on a fixed grid of steps: a new point
  is always integrated from flow time 0 (never interpolated), so every drawn point is a true
  image of an initial point, and the answer depends only on the parameters.
- ``equilibria``: batched Newton from a grid of starts, clustered and sorted, at most
  ``MAX_EQUILIBRIA``; ``classify``: from the Jacobian's eigenvalues, in 2D with the old
  ``KINDS`` strings and in 3D with ``KINDS3``; ``manifolds``: the 1D stable and unstable
  manifolds of a saddle.
- ``PRESETS``: the classic systems as data (rows, variables and ranges, window, ``dt``).

Determinism: the same document on the same machine gives the same bits. A chaotic path is
*not* reproducible across machines beyond a few Lyapunov times (Lorenz: about flow time 40),
so semantics never report positions past ``SAFE_TAU``.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from phasor.render.gpu.codegen import FunctionError, parse_expression

__all__ = [
    "KINDS",
    "KINDS3",
    "PRESETS",
    "FlowSim",
    "FunctionError",
    "System",
    "build",
    "classify",
    "equilibria",
    "integrate",
    "manifolds",
    "parse_constants",
]  # fmt: skip

COORDS = ("x", "y", "z")
RESERVED = frozenset({"x", "y", "z", "t", "pi", "e", "i", "I", "j"})
KINDS = (
    "stable node",
    "unstable node",
    "saddle",
    "stable spiral",
    "unstable spiral",
    "centre",
    "degenerate",
    "non-hyperbolic",
)  # fmt: skip  (2D, unchanged from dynamics_math)
KINDS3 = ("stable node", "unstable node", "saddle", "saddle-focus", "stable spiral",
          "unstable spiral", "degenerate", "non-hyperbolic")  # fmt: skip

# Budgets (advice section 1.3): clamped and linted (``flow_budget``), never refused.
MAX_PROBE_STEPS = 20000  # steps of one stored path (a tracer, a manifold branch)
MAX_POINTS = 200_000  # seeds of one set on the CPU
MAX_TRAIL = 64  # trail samples per seed
MAX_POINTS_X_TRAIL = 4_000_000
MAX_SET_POINTS = 2**17  # a refined curve set stops growing here (lint set_under_resolved)
MAX_STEPS = 200_000  # steps a set is run to (flow time / dt)
MAX_WORK = 1e8  # point-steps one set may compute in all (about 25 s here); then it stops
MAX_EQUILIBRIA = 64
MAX_STARTS = 4096  # Newton starts (grid ** dim)
NEWTON_ITERS = 60
POLISH_ITERS = 150  # the flow core's Newton: room to creep into a double or triple root
XTOL = 1e-12  # ... and converged once a step moves a point less than this (relative)
MERGE = 1e-4  # singular-Jacobian points closer than this share of the window are one root
MAX_SUBSTEPS = 16
SAFE_TAU = 10.0  # semantics report positions only up to this flow time (chaos, see above)

# dead codes of a seed
ALIVE, NONFINITE, TOO_FAST, ESCAPED, OUTSIDE = 0, 1, 2, 3, 4


# ------------------------------------------------------------------ formulas


def parse_constants(text: str) -> dict[str, float]:
    """``"mu=1.5, k=2"`` as ``{"mu": 1.5, "k": 2.0}`` (commas or semicolons between)."""
    out: dict[str, float] = {}
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        name, eq, value = part.partition("=")
        name = name.strip()
        if not eq or not name.isidentifier() or name in ("x", "y"):
            raise FunctionError(f"cannot read the constant {part.strip()!r}: write name=number")
        try:
            out[name] = float(value)
        except ValueError:
            raise FunctionError(
                f"cannot read the constant {part.strip()!r}: write name=number"
            ) from None
        if not math.isfinite(out[name]):
            raise FunctionError(f"the constant {name!r} is not a finite number")
    return out


def free_names(source: str) -> set[str]:
    """The names a formula uses that are not functions or the constants ``pi``, ``e``,
    ``i`` (empty for text that does not read: the parser says why)."""
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name)}  # fmt: skip
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called - {
        "pi", "e", "i", "I", "j"}  # fmt: skip


@functools.lru_cache(maxsize=512)
def _parse(source: str, names: tuple[str, ...]) -> sp.Expr:
    return parse_expression(source, names, extras=True)


def _sym(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


JAC_OPS = 120  # a field past this many operations per component is differentiated numerically
JAC_DEPTH = 24  # ... or nested deeper than this (towers of powers grow fast when differentiated)


def _depth(e: sp.Expr) -> int:
    best, stack = 0, [(e, 1)]
    while stack:
        node, d = stack.pop()
        best = max(best, d)
        stack.extend((a, d + 1) for a in node.args)
    return best


def _small(exprs) -> bool:
    """Whether the exact Jacobian of ``exprs`` is cheap to build (advice 5.3: nothing that
    runs while someone types may take seconds)."""
    return all(sp.count_ops(e) <= JAC_OPS and _depth(e) <= JAC_DEPTH for e in exprs)


class Compiled:
    """NumPy functions of one system: the field and its Jacobian, with the coordinates and
    then the variables as arguments (compiled once per expression; ``compiled``)."""

    def __init__(self, exprs: tuple[sp.Expr, ...], variables: tuple[str, ...]) -> None:
        self.dim = len(exprs)
        self.variables = variables
        syms = [_sym(c) for c in COORDS[: self.dim]] + [_sym(v) for v in variables]
        self._syms = syms
        self._fns = [sp.lambdify(syms, e, "numpy") for e in exprs]
        self.jac_expr: sp.Matrix | None = None
        self._jac: list | None = None
        if _small(exprs):  # a large formula: central differences, so typing never stalls
            jac = sp.Matrix(exprs).jacobian(syms[: self.dim])
            if not any(e.has(sp.Derivative, sp.DiracDelta, sp.Heaviside, sp.Subs) for e in jac):
                self.jac_expr = jac  # not differentiable in closed form: central differences
                try:
                    self._jac = [sp.lambdify(syms, e, "numpy") for e in jac]
                except Exception:
                    self._jac = None

    def fields(self, coords, values=()) -> list[np.ndarray]:
        """Each component at the points ``coords`` (one array per coordinate, any shape)."""
        coords = [np.asarray(c, dtype=np.float64) for c in coords]
        shape = np.broadcast(*coords).shape
        vals = [float(v) for v in values]
        with np.errstate(all="ignore"):
            return [np.broadcast_to(np.asarray(fn(*coords, *vals), dtype=np.float64), shape)
                    for fn in self._fns]  # fmt: skip

    def velocity(self, pts: np.ndarray, values=()) -> np.ndarray:
        """The field at points (N, dim) as (N, dim)."""
        pts = np.asarray(pts, dtype=np.float64)
        out = np.empty_like(pts)
        for k, comp in enumerate(self.fields([pts[:, i] for i in range(self.dim)], values)):
            out[:, k] = comp
        return out

    def jacobians(self, pts: np.ndarray, values=()) -> np.ndarray:
        """The Jacobian at points (N, dim) as (N, dim, dim): the exact derivative where it
        is finite, central differences where it is not (``abs`` at 0, ``sign``)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, self.dim)
        d = self.dim
        out = np.full((len(pts), d, d), np.nan)
        if self._jac is not None:
            cols = [pts[:, i] for i in range(d)]
            vals = [float(v) for v in values]
            with np.errstate(all="ignore"):
                for k, fn in enumerate(self._jac):
                    out[:, k // d, k % d] = np.broadcast_to(
                        np.asarray(fn(*cols, *vals), dtype=np.float64), (len(pts),)
                    )
        bad = ~np.isfinite(out).all(axis=(1, 2))
        if bad.any():
            out[bad] = self._central(pts[bad], values)
        return out

    def _central(self, pts: np.ndarray, values) -> np.ndarray:
        d = self.dim
        h = 1e-6 * np.maximum(1.0, np.abs(pts))
        out = np.empty((len(pts), d, d))
        for j in range(d):
            e = np.zeros_like(pts)
            e[:, j] = h[:, j]
            out[:, :, j] = (self.velocity(pts + e, values) - self.velocity(pts - e, values)) / (
                2 * h[:, j : j + 1])  # fmt: skip
        return out


@functools.lru_cache(maxsize=128)
def compiled(exprs: tuple[sp.Expr, ...], variables: tuple[str, ...]) -> Compiled:
    return Compiled(exprs, variables)


@functools.lru_cache(maxsize=256)
def _sreprs(exprs: tuple[sp.Expr, ...]) -> tuple[str, ...]:
    """The expressions' exact text (``srepr``), once per system: a key is asked for at every
    draw, and printing is the slow part of it (Milestone 28, measured)."""
    return tuple(sp.srepr(e) for e in exprs)


@dataclass(frozen=True)
class System:
    """x' = F(x) in ``dim`` (2 or 3) coordinates: ``exprs`` in the coordinates and the
    named ``variables`` (sorted), and the text it was read from (``sources``)."""

    sources: tuple[str, ...]
    exprs: tuple[sp.Expr, ...]
    variables: tuple[str, ...] = ()

    @property
    def dim(self) -> int:
        return len(self.exprs)

    @classmethod
    def parse(cls, components, variables=None) -> System:
        """One formula per coordinate. ``variables``: the names the formulas may use besides
        the coordinates (None: any name that is not reserved)."""
        comps = tuple(str(c) for c in components)
        if len(comps) not in (2, 3):
            raise FunctionError(f"a system has 2 or 3 components, not {len(comps)}")
        coords = COORDS[: len(comps)]
        if variables is None:
            found = set().union(*(free_names(c) for c in comps)) - set(COORDS) - RESERVED
            variables = tuple(sorted(found))
        names = (*coords, *sorted(variables))
        exprs = tuple(_parse(c, names) for c in comps)
        for c, e in zip(comps, exprs, strict=True):
            for s in e.free_symbols:
                if s.name in COORDS and s.name not in coords:
                    raise FunctionError(f"{c!r} uses {s.name}, but the system has only "
                                        f"{', '.join(coords)}")  # fmt: skip
        used = sorted({s.name for e in exprs for s in e.free_symbols} - set(COORDS))
        return cls(comps, exprs, tuple(used))

    @classmethod
    def from_exprs(cls, exprs, sources=None) -> System:
        exprs = tuple(sp.sympify(e) for e in exprs)
        used = sorted({s.name for e in exprs for s in e.free_symbols} - set(COORDS))
        return cls(tuple(sources or (str(e) for e in exprs)), exprs, tuple(used))

    def plus(self, other: System) -> System:
        """The sum of two fields (sources add: advice section 2.2)."""
        if other.dim != self.dim:
            raise FunctionError("only fields of the same dimension add")
        exprs = tuple(a + b for a, b in zip(self.exprs, other.exprs, strict=True))
        srcs = tuple(f"{a} + ({b})" for a, b in zip(self.sources, other.sources, strict=True))
        return System.from_exprs(exprs, srcs)

    @property
    def fn(self) -> Compiled:
        return compiled(self.exprs, self.variables)

    def values_of(self, values: dict[str, float] | None) -> tuple[float, ...]:
        """The values of ``self.variables`` from a name -> value table (missing: an error)."""
        values = values or {}
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise FunctionError(f"no value for {', '.join(missing)}")
        return tuple(float(values[v]) for v in self.variables)

    def key(self, values: dict[str, float] | None = None) -> str:
        """A hash of what decides the field: the expressions and the values of the
        variables the system actually uses (a slider it does not use changes nothing)."""
        blob = repr((list(_sreprs(self.exprs)), self.values_of(values)))
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    def jacobian_exact(self, point, values=None) -> np.ndarray:
        """The Jacobian at one point, from the symbolic derivative (central differences
        where that is not finite)."""
        vals = self.values_of(values)
        sub = {_sym(c): float(v) for c, v in zip(COORDS, point, strict=False)}
        sub.update({_sym(n): v for n, v in zip(self.variables, vals, strict=True)})
        if self.fn.jac_expr is not None:
            try:
                J = np.array(sp.N(self.fn.jac_expr.subs(sub)), dtype=np.float64)
                if J.shape == (self.dim, self.dim) and np.isfinite(J).all():
                    return J
            except (TypeError, ValueError):
                pass
        return self.fn.jacobians(np.asarray(point, dtype=np.float64)[None], vals)[0]

    def divergence(self) -> sp.Expr | None:
        """The divergence as a formula, or None where it has none in closed form (``sign``,
        ``floor``) or the field is too large to differentiate quickly."""
        J = self.fn.jac_expr
        if J is None:
            return None
        return sum((J[k, k] for k in range(self.dim)), sp.Integer(0))

    def linear(self) -> bool | None:
        """Whether the field is affine in the coordinates: its Jacobian holds no coordinate
        (decided without expanding anything); None when the Jacobian is not symbolic."""
        J = self.fn.jac_expr
        if J is None:
            return None
        coords = {_sym(c) for c in COORDS[: self.dim]}
        return not any(e.free_symbols & coords for e in J)

    def latex(self) -> list[str]:
        return [sp.latex(e) for e in self.exprs]


# ------------------------------------------------------------------ sources that add

TERM_KINDS = ("uniform", "source", "sink", "vortex", "linear")
MAX_TERMS = 16
MAX_TERM_VALUE = 1e6


def _numbers(term: dict, key: str, n: int | None, default=None) -> list[float]:
    v = term.get(key, default)
    if v is None:
        raise FunctionError(f"a {term.get('kind')!r} term needs {key!r}")
    flat = np.asarray(v, dtype=np.float64).ravel()
    if n is not None and flat.size != n:
        raise FunctionError(f"{term.get('kind')!r}: {key!r} needs {n} numbers, not {flat.size}")
    if not np.isfinite(flat).all() or np.abs(flat).max(initial=0) > MAX_TERM_VALUE:
        raise FunctionError(f"{term.get('kind')!r}: {key!r} must be finite and at most "
                            f"{MAX_TERM_VALUE:g} in size")  # fmt: skip
    return [float(a) for a in flat]


def parse_term(text: str) -> dict[str, Any]:
    """A source written as text, ``vortex(pos=(0, 0), strength=2)``, as a table: the kind
    and its keyword values (numbers and tuples of numbers only; read with the AST, never
    run)."""
    if not isinstance(text, str) or len(text) > 500:
        raise FunctionError("a term is text of at most 500 characters, "
                            "such as 'vortex(pos=(0, 0), strength=2)'")  # fmt: skip
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise FunctionError(f"cannot read the term {text!r}") from None
    if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and not tree.args):
        raise FunctionError(f"write a term as kind(name=value, ...), not {text!r}; kinds: "
                            f"{', '.join(TERM_KINDS)}")  # fmt: skip
    out: dict[str, Any] = {"kind": tree.func.id}
    for kw in tree.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
            raise FunctionError(f"{text!r}: {kw.arg} must be a number or numbers") from None
    return out


def term_exprs(term: dict | str, dim: int) -> tuple[sp.Expr, ...]:
    """One source of a field, to be added to the others (advice section 2.2):
    ``uniform(v=(1, 0))``; ``source`` / ``sink`` ``(pos=, strength=, core=)`` (radial, 1/r in
    2D and 1/r^2 in 3D, softened by the core radius so it is finite); ``vortex`` (about the
    z axis through ``pos``); ``linear(A=((0, 1), (-1, 0)), pos=)`` (A (x - pos)). Text
    (``parse_term``) or the same as a table."""
    if isinstance(term, str):
        term = parse_term(term)
    if not isinstance(term, dict):
        raise FunctionError(f"a term is a table such as {{'kind': 'vortex', ...}}, not {term!r}")
    kind = term.get("kind")
    if kind not in TERM_KINDS:
        raise FunctionError(f"unknown term kind {kind!r}; use {', '.join(TERM_KINDS)}")
    X = [_sym(c) for c in COORDS[:dim]]
    if kind == "uniform":
        return tuple(sp.Float(v) for v in _numbers(term, "v", dim))
    pos = _numbers(term, "pos", dim, [0.0] * dim)
    r = [x - sp.Float(p) for x, p in zip(X, pos, strict=True)]
    if kind == "linear":
        A = np.asarray(_numbers(term, "A", dim * dim)).reshape(dim, dim)
        return tuple(sum((sp.Float(A[i, j]) * r[j] for j in range(dim)), sp.Integer(0))
                     for i in range(dim))  # fmt: skip
    s = sp.Float(_numbers(term, "strength", 1, [1.0])[0])
    core = _numbers(term, "core", 1, [0.1])[0]
    if core <= 0:
        raise FunctionError(f"{kind!r}: 'core' must be positive (the field is infinite at 0)")
    c2 = sp.Float(core * core)
    if kind in ("source", "sink"):
        sgn = -1 if kind == "sink" else 1
        r2 = sum((a * a for a in r), sp.Integer(0)) + c2
        den = r2 if dim == 2 else r2 ** sp.Rational(3, 2)
        return tuple(sgn * s * a / den for a in r)
    r2 = r[0] ** 2 + r[1] ** 2 + c2  # vortex
    out = (-s * r[1] / r2, s * r[0] / r2)
    return out if dim == 2 else (*out, sp.Integer(0))


def build(equations, terms=(), variables: tuple[str, ...] | None = None) -> System:
    """The system of ``equations`` (one formula per coordinate) plus every source in
    ``terms``, summed symbolically: the Jacobian, the LaTeX and the code all come from the
    sum."""
    if len(terms) > MAX_TERMS:
        raise FunctionError(f"at most {MAX_TERMS} terms in one system")
    sysm = System.parse(equations, variables)
    for term in terms:
        sysm = sysm.plus(System.from_exprs(term_exprs(term, sysm.dim)))
    return sysm


# ------------------------------------------------------------------ stepping


def _rk4(fn: Compiled, vals, p: np.ndarray, h) -> np.ndarray:
    k1 = fn.velocity(p, vals)
    k2 = fn.velocity(p + h / 2 * k1, vals)
    k3 = fn.velocity(p + h / 2 * k2, vals)
    k4 = fn.velocity(p + h * k3, vals)
    return p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def step(fn: Compiled, vals, p: np.ndarray, dt: float, max_step: float | None = None):
    """One fixed step of every seed (N, dim): ``(new positions, code)``, code per seed
    ``ALIVE``, ``NONFINITE`` or ``TOO_FAST``. A seed whose step would move it further than
    ``max_step`` is sub-stepped (each seed by its own count, so it does not depend on its
    neighbours), at most ``MAX_SUBSTEPS`` times; past that it is stopped."""
    code = np.zeros(len(p), dtype=np.int8)
    if max_step is None or len(p) == 0:
        q = _rk4(fn, vals, p, dt)
    else:
        with np.errstate(all="ignore"):
            need = np.linalg.norm(fn.velocity(p, vals), axis=1) * abs(dt) / max_step
        fast = ~(need <= MAX_SUBSTEPS)  # NaN counts as too fast here; sorted out below
        n = np.where(fast, 1, np.maximum(1, np.ceil(np.where(fast, 1, need)))).astype(np.int64)
        if (n == 1).all():
            q = _rk4(fn, vals, p, dt)
        else:
            q = p.copy()
            for k in range(int(n.max())):
                sel = n > k
                h = (dt / n[sel])[:, None]
                q[sel] = _rk4(fn, vals, q[sel], h)
        code[fast & np.isfinite(need)] = TOO_FAST
    code[(code == ALIVE) & ~np.isfinite(q).all(axis=1)] = NONFINITE
    return q, code


@dataclass
class Path:
    """Stored paths: ``points`` (S, steps + 1, dim), NaN once a seed stops; ``code`` why each
    stopped (``ALIVE`` if it did not) and ``at`` the step it stopped at (-1: did not)."""

    points: np.ndarray
    code: np.ndarray
    at: np.ndarray


def integrate(fn: Compiled, vals, seeds, steps: int, dt: float, *, box=None, escape=None,
              max_step: float | None = None, start=None) -> Path:  # fmt: skip
    """Paths of ``steps`` fixed RK4 steps from ``seeds`` (S, dim). ``dt`` < 0 runs backward.
    ``box = (lo, hi)``: a seed leaving it stops (``OUTSIDE``); ``escape = (centre, radius)``:
    a seed further out stops (``ESCAPED``). Both bound a backward run, which for a
    dissipative system blows up (Lorenz backward is NaN within a few time units).
    ``start``: a ``Path`` to continue (its last positions; the result has its steps too)."""
    steps = int(min(max(steps, 0), MAX_PROBE_STEPS))
    seeds = np.asarray(seeds, dtype=np.float64).reshape(len(np.atleast_2d(seeds)), -1)
    n0 = 0 if start is None else start.points.shape[1] - 1
    out = np.full((len(seeds), n0 + steps + 1, seeds.shape[1]), np.nan)
    code = np.zeros(len(seeds), dtype=np.int8)
    at = np.full(len(seeds), -1, dtype=np.int64)
    if start is None:
        out[:, 0] = seeds
        p = seeds.copy()
    else:
        out[:, : n0 + 1] = start.points
        code[:], at[:] = start.code, start.at
        p = np.where(code[:, None] == ALIVE, start.points[:, -1], 0.0)
    alive = code == ALIVE
    p = np.where(alive[:, None], p, 0.0)
    lo = hi = None
    if box is not None:
        lo, hi = (np.asarray(b, dtype=np.float64) for b in box)
    for k in range(n0, n0 + steps):
        if not alive.any():
            break
        q, c = step(fn, vals, p, dt, max_step)
        new = alive & (c != ALIVE)
        if lo is not None:
            with np.errstate(invalid="ignore"):
                out_box = alive & (c == ALIVE) & ~((q >= lo) & (q <= hi)).all(axis=1)
            new |= out_box
            c = np.where(out_box, OUTSIDE, c)
        if escape is not None:
            with np.errstate(invalid="ignore"):
                far = np.linalg.norm(q - np.asarray(escape[0]), axis=1) > escape[1]
            gone = alive & (c == ALIVE) & far
            new |= gone
            c = np.where(gone, ESCAPED, c)
        code[new], at[new] = c[new], k + 1
        alive &= ~new
        p = np.where(alive[:, None], q, 0.0)  # a stopped seed keeps finite arithmetic
        out[alive, k + 1] = p[alive]
    return Path(out, code, at)


# ------------------------------------------------------------------ sets as a Simulation


def _halton(n: int, base: int) -> np.ndarray:
    out = np.zeros(n)
    k = np.arange(1, n + 1)
    f = 1.0
    while (k > 0).any():
        f /= base
        out += f * (k % base)
        k //= base
    return out


def fill(shape: str, centre, size: float, count: int, dim: int) -> np.ndarray:
    """``count`` points filling a ``disk`` / ``ball`` (radius ``size``) or a ``box``
    (half-side ``size``) evenly and without random numbers (a Halton sequence)."""
    c = np.asarray(centre, dtype=np.float64)[:dim]
    count = int(max(0, count))
    if count == 0:
        return np.zeros((0, dim))
    u = np.stack([_halton(count, b) for b in (2, 3, 5)[:dim]], axis=1)
    if shape == "box":
        return c + (2 * u - 1) * size
    if dim == 2:  # disk: area-uniform
        r = size * np.sqrt(u[:, 0])
        a = 2 * np.pi * u[:, 1]
        return c + np.c_[r * np.cos(a), r * np.sin(a)]
    r = size * np.cbrt(u[:, 0])
    cz = 2 * u[:, 1] - 1
    a = 2 * np.pi * u[:, 2]
    sz = np.sqrt(np.maximum(0.0, 1 - cz * cz))
    return c + r[:, None] * np.c_[sz * np.cos(a), sz * np.sin(a), cz]


def curves(shape: str, centre, size: float, dim: int, rings: int = 3) -> tuple[tuple, ...]:
    """The boundary of a shape as curves the set refines: ``("circle", centre, u, w, r)``
    (closed: x(s) = centre + r (cos 2 pi s u + sin 2 pi s w)) and ``("segment", a, b)``
    (open: x(s) = a + s (b - a)). A 3D ball is ``rings`` latitude and ``rings`` longitude
    circles (advice section 2.4: no refined mesh in v1)."""
    c = tuple(float(v) for v in np.asarray(centre, dtype=np.float64)[:dim])
    ex = tuple(1.0 if i == 0 else 0.0 for i in range(dim))
    ey = tuple(1.0 if i == 1 else 0.0 for i in range(dim))
    if shape in ("disk", "circle") or (shape == "ball" and dim == 2):
        return (("circle", c, ex, ey, float(size)),)
    if shape == "ball":
        out = []
        for k in range(rings):  # latitudes
            h = -1 + 2 * (k + 1) / (rings + 1)
            out.append(("circle", (c[0], c[1], c[2] + size * h), ex, ey,
                        float(size * math.sqrt(1 - h * h))))  # fmt: skip
        for k in range(rings):  # longitudes, through the poles
            a = math.pi * k / rings
            out.append(("circle", c, (math.cos(a), math.sin(a), 0.0), (0.0, 0.0, 1.0),
                        float(size)))  # fmt: skip
        return tuple(out)
    if shape == "box":
        corners = np.array(np.meshgrid(*[[-1, 1]] * dim, indexing="ij")).reshape(dim, -1).T
        edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1 :]
                 if np.abs(a - b).sum() == 2]  # fmt: skip
        cc = np.asarray(c)
        return tuple(("segment", tuple(map(float, cc + size * a)),
                      tuple(map(float, cc + size * b))) for a, b in edges)  # fmt: skip
    if shape == "segment":
        a = np.asarray(c) - size * np.asarray(ex)
        b = np.asarray(c) + size * np.asarray(ex)
        return (("segment", tuple(map(float, a)), tuple(map(float, b))),)
    return ()


def _on_curve(curve: tuple, s: np.ndarray) -> np.ndarray:
    if curve[0] == "circle":
        _, c, u, w, r = curve
        a = 2 * np.pi * s
        return (np.asarray(c) + r * (np.cos(a)[:, None] * np.asarray(u)
                                     + np.sin(a)[:, None] * np.asarray(w)))  # fmt: skip
    _, a, b = curve
    return np.asarray(a) + s[:, None] * (np.asarray(b) - np.asarray(a))


@dataclass
class FlowSim:
    """A set of seeds advected by a system, as a ``Simulation`` (core/simulation.py).

    ``points`` (N, dim) are advected as they are; ``loops`` are curves (``curves``) sampled
    at ``samples`` parameters each and refined every ``refine_every`` steps where two
    neighbours are further apart than ``h_max``: the new parameter's point is integrated
    from flow time 0 (``_from_start``), never interpolated. Past ``cap`` curve points the
    set stops growing and records the step (``under_resolved``). A trail is not part of
    the state (``trail_from``): checkpoints hold positions only."""

    system: System
    values: tuple[float, ...]
    dt: float
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    loops: tuple = ()
    samples: int = 64
    h_max: float = 0.0
    refine_every: int = 10
    cap: int = MAX_SET_POINTS
    max_step: float | None = None
    escape: tuple | None = None  # (centre, radius)
    direction: float = 1.0  # -1: backward (bounded by ``escape``)
    max_work: float = MAX_WORK  # point-steps in all, refinement included; then it stops

    def params(self) -> dict[str, Any]:
        pts = np.ascontiguousarray(self.points, dtype=np.float64)
        return {
            "dim": self.system.dim, "exprs": [sp.srepr(e) for e in self.system.exprs],
            "variables": list(self.system.variables), "values": list(self.values),
            "points": hashlib.sha256(pts.tobytes()).hexdigest() + str(pts.shape),
            "loops": [list(map(str, c)) for c in self.loops], "samples": self.samples,
            "h_max": self.h_max, "refine_every": self.refine_every, "cap": self.cap,
            "max_step": self.max_step, "escape": None if self.escape is None else
            [list(map(float, self.escape[0])), float(self.escape[1])],
            "direction": self.direction, "max_work": self.max_work,
            "sympy": sp.__version__,
            "method": "rk4",
            "dtype": "float64",
        }  # fmt: skip

    def initial(self) -> dict[str, np.ndarray]:
        d = self.system.dim
        pts = np.asarray(self.points, dtype=np.float64).reshape(-1, d)
        s = np.concatenate([(np.arange(self.samples) / self.samples) if c[0] == "circle"
                            else np.linspace(0.0, 1.0, self.samples) for c in self.loops]
                           ) if self.loops else np.zeros(0)  # fmt: skip
        loop = np.repeat(np.arange(len(self.loops)), self.samples) if self.loops else np.zeros(
            0, dtype=np.int64)  # fmt: skip
        q = np.concatenate([_on_curve(c, s[loop == k]) for k, c in enumerate(self.loops)]
                           ) if self.loops else np.zeros((0, d))  # fmt: skip
        out = {"p": pts.copy(), "pc": np.zeros(len(pts), np.int8), "s": s, "loop": loop,
               "q": q, "qc": np.zeros(len(q), np.int8), "n": np.array(0),
               "under": np.array(-1), "work": np.array(0.0),
               "halted": np.array(-1)}  # fmt: skip
        return out

    def _advance(self, p: np.ndarray, code: np.ndarray, dt: float):
        alive = code == ALIVE
        q, c = step(self.system.fn, self.values, np.where(alive[:, None], p, 0.0), dt,
                    self.max_step)  # fmt: skip
        if self.escape is not None:
            with np.errstate(invalid="ignore"):
                far = np.linalg.norm(q - np.asarray(self.escape[0]), axis=1) > self.escape[1]
            c = np.where((c == ALIVE) & far, ESCAPED, c)
        code = np.where(alive, c, code).astype(np.int8)
        return np.where(code[:, None] == ALIVE, q, np.nan), code

    def trail_from(self, state, steps: int) -> np.ndarray:
        """The positions of the points over ``steps`` more steps from ``state`` (steps + 1,
        N, dim): the same steps the set takes (a point moves the same alone or in any
        batch), computed when drawn instead of stored in every checkpoint."""
        p, code = np.array(state["p"]), np.array(state["pc"])
        out = [p]
        for _ in range(steps):
            p, code = self._advance(p, code, self.dt * self.direction)
            out.append(p)
        return np.stack(out)

    def _from_start(self, x0: np.ndarray, n: int, dt: float):
        p, code = x0.copy(), np.zeros(len(x0), np.int8)
        for _ in range(n):
            p, code = self._advance(p, code, dt)
        return p, code

    def step(self, state, dt: float):
        """One step of the whole set. Past ``max_work`` point-steps the set stops where it is
        (``halted``: the step it stopped at), so no flow time or count can hang a frame."""
        dt = dt * self.direction
        n = int(state["n"]) + 1
        if int(state["halted"]) >= 0:
            return {**state, "n": np.array(n)}
        work = float(state["work"]) + len(state["p"]) + len(state["q"])
        if work > self.max_work:
            return {**state, "n": np.array(n), "halted": np.array(n - 1)}
        both = np.concatenate([state["p"], state["q"]])
        codes = np.concatenate([state["pc"], state["qc"]])
        moved, codes = self._advance(both, codes, dt)
        k = len(state["p"])
        out = dict(state)
        out["p"], out["pc"] = moved[:k], codes[:k]
        out["q"], out["qc"] = moved[k:], codes[k:]
        out["n"], out["work"] = np.array(n), np.array(work)
        if self.loops and self.h_max > 0 and n % self.refine_every == 0:
            out.update(self._refine(out, n, dt))
        return out

    def _refine(self, st, n: int, dt: float) -> dict[str, np.ndarray]:
        """Insert a parameter between neighbours further apart than ``h_max`` (both alive),
        integrated from flow time 0 to step ``n``; deterministic (a function of the state)."""
        s, loop, q, qc = st["s"], st["loop"], st["q"], st["qc"]
        # room under the cap, and under the work left (each new point costs n steps)
        room = min(self.cap - len(s), int((self.max_work - float(st["work"])) // max(n, 1)))
        new_s, new_loop = [], []
        for k, c in enumerate(self.loops):
            idx = np.flatnonzero(loop == k)
            if len(idx) < 2:
                continue
            closed = c[0] == "circle"
            a = idx if closed else idx[:-1]
            b = np.roll(idx, -1) if closed else idx[1:]
            ok = (qc[a] == ALIVE) & (qc[b] == ALIVE)
            gap = np.linalg.norm(q[b] - q[a], axis=1)
            far = ok & (gap > self.h_max)
            sb = np.where(s[b] <= s[a], s[b] + 1.0, s[b]) if closed else s[b]
            # as many new parameters as the gap needs, evenly between the two (at most 64)
            k_new = np.minimum(np.ceil(gap[far] / self.h_max) - 1, 64).astype(np.int64)
            k_new = np.maximum(k_new, 1)
            sa, sbf = s[a][far], sb[far]
            rep = np.repeat(np.arange(len(k_new)), k_new)
            j = np.arange(len(rep)) - np.repeat(np.cumsum(k_new) - k_new, k_new) + 1
            mid = sa[rep] + (sbf[rep] - sa[rep]) * j / (k_new[rep] + 1)
            mid = mid % 1.0 if closed else mid
            new_s.append(mid)
            new_loop.append(np.full(len(mid), k))
        if not new_s:
            return {}
        ns, nl = np.concatenate(new_s), np.concatenate(new_loop)
        out: dict[str, np.ndarray] = {}
        if len(ns) > room:
            if int(st["under"]) < 0:
                out["under"] = np.array(n)
            ns, nl = ns[: max(room, 0)], nl[: max(room, 0)]
        if not len(ns):
            return out
        x0 = np.concatenate([_on_curve(self.loops[k], ns[nl == k])
                             for k in range(len(self.loops)) if (nl == k).any()])  # fmt: skip
        order_in = np.concatenate([np.flatnonzero(nl == k) for k in range(len(self.loops))
                                   if (nl == k).any()])  # fmt: skip
        ns, nl = ns[order_in], nl[order_in]
        pq, pc = self._from_start(x0, n, dt)
        out["work"] = np.array(float(st["work"]) + len(x0) * n)
        s2 = np.concatenate([s, ns])
        l2 = np.concatenate([loop, nl])
        order = np.lexsort((s2, l2))
        out.update({"s": s2[order], "loop": l2[order],
                    "q": np.concatenate([q, pq])[order],
                    "qc": np.concatenate([qc, pc])[order]})  # fmt: skip
        return out


# ------------------------------------------------------------------ equilibria


def newton(fn: Compiled, vals, starts: np.ndarray, *, tol: float = 1e-9, cap: float,
           singular: str = "pinv") -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Newton's method from every start at once: ``(points, converged)``. A step longer
    than ``cap`` is shortened.

    ``singular="pinv"`` (the flow core): a singular Jacobian takes the least-squares step
    (finds lines of equilibria), and a start counts as converged only once the residual is
    under ``tol`` *and* its steps have stopped moving it (up to ``POLISH_ITERS``): near a
    double or triple root the residual is tiny long before the point is there.
    ``"drop"`` (``dynamics_math``, unchanged): a singular start gives up, and the residual
    alone decides, in ``NEWTON_ITERS`` steps."""
    p = np.asarray(starts, dtype=np.float64).copy()
    n, d = p.shape
    polish = singular == "pinv"
    done = np.zeros(n, dtype=bool)
    live = np.ones(n, dtype=bool)
    last = np.full(n, np.inf)  # the length of each start's last step
    small = np.zeros(n, dtype=bool)  # residual under tol
    for _ in range(POLISH_ITERS if polish else NEWTON_ITERS):
        act = np.flatnonzero(live & ~done)
        if not len(act):
            break
        r = fn.velocity(p[act], vals)
        fin = np.isfinite(r).all(axis=1)
        live[act[~fin]] = False
        under = fin & (np.abs(np.where(np.isfinite(r), r, 0)).max(axis=1) < tol)
        small[act] = under
        if polish:
            still = last[act] <= XTOL * (1.0 + np.abs(p[act]).max(axis=1))
            conv = under & (still | (np.abs(np.where(fin[:, None], r, 1)).max(axis=1) == 0))
        else:
            conv = under
        done[act[conv]] = True
        act, r = act[fin & ~conv], r[fin & ~conv]
        if not len(act):
            continue
        J = fn.jacobians(p[act], vals)
        okJ = np.isfinite(J).all(axis=(1, 2))
        with np.errstate(all="ignore"):
            det = np.where(okJ, np.linalg.det(np.where(okJ[:, None, None], J, 0.0)), 0.0)
        scale = np.abs(np.where(okJ[:, None, None], J, 0.0)).max(axis=(1, 2))
        if singular == "drop":  # as dynamics_math always did: give up only on an exact zero
            reg = okJ & (det != 0)
        else:
            reg = okJ & (np.abs(det) > 1e-14 * np.maximum(scale, 1e-300) ** d)
        stepv = np.full_like(r, np.nan)
        if reg.any():
            try:
                stepv[reg] = np.linalg.solve(J[reg], r[reg][..., None])[..., 0]
            except np.linalg.LinAlgError:
                for i in np.flatnonzero(reg):
                    with contextlib.suppress(np.linalg.LinAlgError):
                        stepv[i] = np.linalg.solve(J[i], r[i])
        sing = okJ & ~reg
        if singular == "pinv" and sing.any():
            stepv[sing] = (np.linalg.pinv(J[sing]) @ r[sing][..., None])[..., 0]
        good = np.isfinite(stepv).all(axis=1)
        live[act[~good]] = False
        act, stepv = act[good], stepv[good]
        norm = np.linalg.norm(stepv, axis=1)
        stepv = np.where((norm > cap)[:, None], stepv * (cap / np.maximum(norm, 1e-300))[:, None],
                         stepv)  # fmt: skip
        last[act] = np.minimum(norm, cap)
        p[act] = p[act] - stepv
    if polish:  # out of steps with the residual under tol: converged as far as floats go
        done |= live & small
    return p, done


def grid_starts(lo, hi, grid: int) -> np.ndarray:
    axes = [np.linspace(a, b, grid) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def _singular(J: np.ndarray) -> np.ndarray:
    """Which Jacobians (N, d, d) are singular relative to their size."""
    d = J.shape[-1]
    sc = np.maximum(np.abs(J).max(axis=(1, 2)), 1e-300)
    with np.errstate(all="ignore"):
        return ~(np.abs(np.linalg.det(J)) > 1e-9 * sc**d)


def equilibria(system: System, values, lo, hi, grid: int = 9, *,
               singular: str = "pinv") -> dict[str, Any]:  # fmt: skip
    """Every equilibrium Newton finds from a ``grid`` of starts in the window ``lo..hi``,
    clustered and sorted along x, then y, then z. At most ``MAX_EQUILIBRIA``; more, or three
    or more separate points with a singular Jacobian (a line or plane of them), sets
    ``non_isolated``. Points with a singular Jacobian closer than ``MERGE`` of the window
    are one degenerate root (a saddle-node or pitchfork at its bifurcation, where Newton
    only creeps in), not a line. The window and grid are part of the answer: "none found"
    is not "none exist"."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    d = system.dim
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    grid = int(max(2, min(grid, round(MAX_STARTS ** (1 / d)))))
    span = hi - lo
    fn = system.fn
    with np.errstate(all="ignore"):
        pts, ok = newton(fn, vals, grid_starts(lo, hi, grid), cap=float(span.max()),
                         singular=singular)  # fmt: skip
    pad = span * 1e-6
    cand = [q for q in pts[ok] if ((q >= lo - pad) & (q <= hi + pad)).all()]
    found: list[np.ndarray] = []
    if singular == "pinv" and cand:
        cand_a = np.array(cand)
        with np.errstate(all="ignore"):
            res = np.abs(fn.velocity(cand_a, vals)).max(axis=1)
            sing = _singular(fn.jacobians(cand_a, vals))
        merge = MERGE * float(span.max())
        reps: list[tuple[np.ndarray, bool, float]] = []
        for q, sg, rq in zip(cand_a, sing, res, strict=True):
            near = 1e-6 * max(1.0, float(np.linalg.norm(q)))
            for k, (o, so, ro) in enumerate(reps):
                gap = float(np.linalg.norm(q - o))
                if gap < near or (sg and so and gap < merge):
                    if rq < ro:  # keep the best of a cluster
                        reps[k] = (q, sg, float(rq))
                    break
            else:
                reps.append((q, bool(sg), float(rq)))
        found = [q for q, _, _ in reps]
    else:
        for q in cand:
            if not any(np.linalg.norm(q - o) < 1e-6 * max(1.0, np.linalg.norm(o)) for o in found):
                found.append(q)
    out = np.array(found, dtype=np.float64).reshape(-1, d)
    if len(out):
        keys = tuple(np.round(out[:, k], 7) for k in reversed(range(d)))
        out = out[np.lexsort(keys)]
    non_isolated = len(out) > MAX_EQUILIBRIA
    if not non_isolated and len(out) >= 3:
        non_isolated = int(_singular(fn.jacobians(out, vals)).sum()) >= 3
    return {"points": out[:MAX_EQUILIBRIA], "non_isolated": bool(non_isolated),
            "window": [[float(a), float(b)] for a, b in zip(lo, hi, strict=True)],
            "grid": grid, "starts": grid**d, "found": len(out)}  # fmt: skip


# ------------------------------------------------------------------ classification


def _sign_fixed(v: np.ndarray) -> np.ndarray:
    """A direction with a fixed sign: its first clearly nonzero entry is positive."""
    for x in v:
        if abs(x) > 1e-12:
            return -v if x < 0 else v
    return v


def _classify2(J: np.ndarray) -> dict:
    tr, det = float(np.trace(J)), float(np.linalg.det(J))
    vals, vecs = np.linalg.eig(J)
    scale = max(float(np.abs(J).max()), 1.0)
    tol = 1e-7 * scale
    re, im = vals.real, vals.imag
    if np.abs(vals).min() < tol:
        kind, stable = "non-hyperbolic", None
    elif abs(im[0]) > tol:  # a complex pair
        if abs(re[0]) <= tol:
            kind, stable = "centre", None
        else:
            kind = "stable spiral" if re[0] < 0 else "unstable spiral"
            stable = bool(re[0] < 0)
    elif re[0] * re[1] < 0:
        kind, stable = "saddle", False
    else:
        stable = bool(re.max() < 0)
        defective = abs(re[0] - re[1]) < tol and np.linalg.matrix_rank(J - re[0] * np.eye(2),
                                                                      tol=tol) > 0  # fmt: skip
        kind = "degenerate" if defective else ("stable node" if stable else "unstable node")
    real_dirs = []
    for k in range(2):
        if abs(im[k]) <= tol:
            v = vecs[:, k].real
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            if v[0] < -1e-12 or (abs(v[0]) <= 1e-12 and v[1] < 0):
                v = -v
            real_dirs.append((float(re[k]), v))
    if kind == "degenerate" and len(real_dirs) == 2:  # one direction, found twice
        real_dirs = real_dirs[:1]
    return {
        "kind": kind,
        "stable": stable,
        "eigenvalues": [(float(a), float(b)) for a, b in zip(re, im, strict=True)],
        "trace": tr,
        "det": det,
        "directions": real_dirs,
        "tolerance": tol,
        **_subspaces(vals, vecs, tol),
    }


def _subspaces(vals, vecs, tol: float) -> dict:
    """Orthonormal bases of the stable, unstable and centre subspaces: real eigenvectors,
    and the real and imaginary parts of one vector of each complex pair (a plane)."""
    bases: dict[str, list[np.ndarray]] = {"s": [], "u": [], "c": []}
    done = set()
    for k, lam in enumerate(vals):
        if k in done:
            continue
        which = "s" if lam.real < -tol else "u" if lam.real > tol else "c"
        v = vecs[:, k]
        if abs(lam.imag) > tol:
            bases[which] += [v.real, v.imag]
            for j in range(k + 1, len(vals)):  # its conjugate
                if j not in done and abs(vals[j] - np.conj(lam)) <= 10 * tol:
                    done.add(j)
                    break
        else:
            bases[which].append(v.real)
        done.add(k)
    out = {}
    for key, vs in bases.items():
        if not vs:
            out[f"E{key}"] = []
            continue
        Q, _ = np.linalg.qr(np.array(vs, dtype=np.float64).T)
        out[f"E{key}"] = [_sign_fixed(Q[:, i]) for i in range(len(vs))]
    out["dims"] = {k: len(out[f"E{k}"]) for k in ("s", "u", "c")}
    return out


def classify(J) -> dict:
    """The type of an equilibrium from its Jacobian (2x2 or 3x3): ``kind`` (``KINDS`` in
    2D, ``KINDS3`` in 3D), ``stable`` (True: all real parts negative; False: some positive;
    None: the linear part does not decide), ``eigenvalues`` as (re, im), ``trace``, ``det``,
    real eigen-``directions``, the subspaces ``Es``, ``Eu``, ``Ec`` (orthonormal bases) and
    their ``dims``. Tolerances are relative to the Jacobian's size (``tolerance``)."""
    J = np.asarray(J, dtype=np.float64)
    if J.shape in ((2, 2), (3, 3)) and not np.isfinite(J).all():
        d = J.shape[0]  # the field is not finite around the point: nothing to say
        return {"kind": "undefined", "stable": None, "eigenvalues": [], "trace": float("nan"),
                "det": float("nan"), "directions": [], "tolerance": 0.0, "Es": [], "Eu": [],
                "Ec": [], "dims": {"s": 0, "u": 0, "c": d}}  # fmt: skip
    if J.shape == (2, 2):
        return _classify2(J)
    if J.shape != (3, 3):
        raise ValueError(f"a 2x2 or 3x3 Jacobian, not {J.shape}")
    vals, vecs = np.linalg.eig(J)
    order = np.lexsort((-vals.imag, vals.real))  # by real part, then +imag first
    vals, vecs = vals[order], vecs[:, order]
    scale = float(np.abs(J).max())
    tol = 1e-7 * scale if scale > 0 else 1e-300
    sub = _subspaces(vals, vecs, tol)
    s, u, c = sub["dims"]["s"], sub["dims"]["u"], sub["dims"]["c"]
    cplx = bool((np.abs(vals.imag) > tol).any())
    if c:
        kind, stable = "non-hyperbolic", (False if u else None)
    elif s == 3 or u == 3:
        stable = s == 3
        re = vals.real
        repeated = any(abs(re[i] - re[j]) < tol for i in range(3) for j in range(i + 1, 3))
        defective = (not cplx and repeated
                     and np.linalg.matrix_rank(vecs, tol=1e-6) < 3)  # fmt: skip
        word = "stable" if stable else "unstable"
        kind = "degenerate" if defective else f"{word} {'spiral' if cplx else 'node'}"
    else:
        kind, stable = ("saddle-focus" if cplx else "saddle"), False
    dirs = [
        (
            float(lam.real),
            _sign_fixed(vecs[:, k].real / max(np.linalg.norm(vecs[:, k].real), 1e-12)),
        )
        for k, lam in enumerate(vals)
        if abs(lam.imag) <= tol
    ]
    return {
        "kind": kind, "stable": stable,
        "eigenvalues": [(float(a), float(b)) for a, b in zip(vals.real, vals.imag, strict=True)],
        "trace": float(np.trace(J)), "det": float(np.linalg.det(J)),
        "directions": dirs, "tolerance": tol, **sub,
    }  # fmt: skip


# ------------------------------------------------------------------ manifolds


def manifolds(system: System, values, point, J=None, *, which: str = "both", dt: float = 0.01,
              steps: int = 2000, eps: float | None = None, escape=None,
              max_step: float | None = None) -> list[dict]:  # fmt: skip
    """The 1D stable and unstable manifolds of the equilibrium at ``point``: from
    ``point +/- eps v`` (v the eigen-direction of a one-dimensional E^s or E^u), integrated
    forward for the unstable one and backward for the stable one, each branch at most
    ``steps`` steps and stopped at ``escape``. Each branch: ``which`` (stable/unstable),
    ``sign``, ``points`` (M, dim, NaN-free), ``ends`` (``escaped``, ``steps``,
    ``equilibrium``, ``nonfinite``, ``too_fast``) and ``length`` (arc length)."""
    vals = system.values_of(values) if isinstance(values, dict) else tuple(values)
    point = np.asarray(point, dtype=np.float64)
    J = system.jacobian_exact(point, dict(zip(system.variables, vals, strict=True))) \
        if J is None else np.asarray(J)  # fmt: skip
    c = classify(J)
    scale = max(1.0, float(np.linalg.norm(point)))
    eps = 1e-5 * scale if eps is None else eps
    out = []
    for name, key, sgn_dt in (("unstable", "Eu", 1.0), ("stable", "Es", -1.0)):
        if which not in ("both", name) or len(c[key]) != 1:
            continue
        v = np.asarray(c[key][0])
        seeds = np.stack([point + eps * v, point - eps * v])
        path = integrate(system.fn, vals, seeds, steps, sgn_dt * abs(dt), escape=escape,
                         max_step=max_step)  # fmt: skip
        for i, sign in enumerate((1, -1)):
            pts = path.points[i]
            pts = pts[np.isfinite(pts).all(axis=1)]
            end = {ALIVE: "steps", NONFINITE: "nonfinite", TOO_FAST: "too_fast",
                   ESCAPED: "escaped", OUTSIDE: "escaped"}[int(path.code[i])]  # fmt: skip
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if end == "steps" and len(pts) > 2 and seg[-1] < 1e-9 * scale:
                end = "equilibrium"
            out.append({"which": name, "sign": sign, "points": np.vstack([point[None], pts]),
                        "ends": end, "length": float(seg.sum() + eps)})  # fmt: skip
    return out


# ------------------------------------------------------------------ presets


@dataclass(frozen=True)
class Preset:
    """A classic system as data: one formula per coordinate, its variables (name ->
    (value, min, max)), the window (lo, hi per axis), a ``dt``, a camera hint and one
    sentence."""

    equations: tuple[str, ...]
    variables: dict[str, tuple[float, float, float]]
    window: tuple[tuple[float, float], ...]
    dt: float
    note: str
    camera: tuple[float, float] = (-50.0, 20.0)  # azimuth, elevation (3D)

    @property
    def dim(self) -> int:
        return len(self.equations)

    def constants(self) -> str:
        return ", ".join(f"{k}={v[0]!r}" for k, v in self.variables.items())


def _P(eqs, var, win, dt, note, camera=(-50.0, 20.0)) -> Preset:
    return Preset(tuple(eqs), dict(var), tuple(tuple(w) for w in win), dt, note, camera)


PRESETS: dict[str, Preset] = {
    # ---- 3D
    "lorenz": _P(("sigma*(y - x)", "x*(rho - z) - y", "x*y - beta*z"),
                 {"sigma": (10.0, 0.0, 30.0), "rho": (28.0, 0.0, 60.0),
                  "beta": (8.0 / 3.0, 0.0, 10.0)},
                 ((-25, 25), (-30, 30), (-2, 55)), 0.01,
                 "Lorenz's convection model: two wings around two saddle-foci."),
    "rossler": _P(("-y - z", "x + a*y", "b + z*(x - c)"),
                  {"a": (0.2, 0.0, 0.5), "b": (0.2, 0.0, 2.0), "c": (5.7, 0.0, 20.0)},
                  ((-12, 12), (-12, 12), (-1, 25)), 0.02,
                  "Rossler's band: a spiral that folds back on itself."),
    "thomas": _P(("sin(y) - b*x", "sin(z) - b*y", "sin(x) - b*z"),
                 {"b": (0.208186, 0.0, 1.0)}, ((-5, 5), (-5, 5), (-5, 5)), 0.05,
                 "Thomas' cyclically symmetric attractor: a frictional walk through a lattice."),
    "chua": _P(("alpha*(y - x - (m1*x + (m0 - m1)/2*(abs(x + 1) - abs(x - 1))))",
                "x - y + z", "-gamma*y"),
               {"alpha": (15.6, 0.0, 30.0), "gamma": (28.0, 0.0, 50.0),
                "m0": (-1.143, -3.0, 0.0), "m1": (-0.714, -3.0, 0.0)},
               ((-3, 3), (-0.6, 0.6), (-5, 5)), 0.005,
               "Chua's circuit: a double scroll from a piecewise-linear resistor."),
    "halvorsen": _P(("-a*x - 4*y - 4*z - y**2", "-a*y - 4*z - 4*x - z**2",
                     "-a*z - 4*x - 4*y - x**2"),
                    {"a": (1.89, 0.0, 4.0)}, ((-12, 8), (-12, 8), (-12, 8)), 0.005,
                    "Halvorsen's cyclically symmetric attractor."),
    "aizawa": _P(("(z - b)*x - d*y", "d*x + (z - b)*y",
                  "c + a*z - z**3/3 - (x**2 + y**2)*(1 + k*z) + f*z*x**3"),
                 {"a": (0.95, 0.0, 2.0), "b": (0.7, 0.0, 2.0), "c": (0.6, 0.0, 2.0),
                  "d": (3.5, 0.0, 6.0), "k": (0.25, 0.0, 1.0), "f": (0.1, 0.0, 1.0)},
                 ((-1.6, 1.6), (-1.6, 1.6), (-0.6, 2.0)), 0.01,
                 "Aizawa's attractor: a sphere-like shell with a tube through it."),
    # ---- 2D
    "van_der_pol": _P(("y", "mu*(1 - x**2)*y - x"), {"mu": (1.0, 0.0, 5.0)},
                      ((-4, 4), (-5, 5)), 0.01, "Van der Pol's oscillator: one limit cycle."),
    "pendulum": _P(("y", "-sin(x) - gamma*y"), {"gamma": (0.3, 0.0, 2.0)},
                   ((-7, 7), (-4, 4)), 0.02, "A damped pendulum: spirals and saddles."),
    "duffing": _P(("y", "-delta*y - alpha*x - beta*x**3"),
                  {"delta": (0.3, 0.0, 2.0), "alpha": (-1.0, -2.0, 2.0),
                   "beta": (1.0, 0.0, 3.0)}, ((-2.5, 2.5), (-2, 2)), 0.01,
                  "Duffing's unforced double well: two sinks and a saddle."),
    "lotka_volterra": _P(("a*x - b*x*y", "d*x*y - c*y"),
                         {"a": (2.0 / 3.0, 0.0, 2.0), "b": (4.0 / 3.0, 0.0, 2.0),
                          "c": (1.0, 0.0, 2.0), "d": (1.0, 0.0, 2.0)},
                         ((-0.2, 3), (-0.2, 2)), 0.02, "Predators and prey: closed orbits."),
    "brusselator": _P(("a - (b + 1)*x + x**2*y", "b*x - x**2*y"),
                      {"a": (1.0, 0.0, 3.0), "b": (3.0, 0.0, 5.0)}, ((0, 4), (0, 5)), 0.01,
                      "The Brusselator: a chemical clock (a limit cycle for b > 1 + a^2)."),
    "fitzhugh_nagumo": _P(("x - x**3/3 - y + drive", "eps*(x + a - b*y)"),
                          {"drive": (0.5, 0.0, 2.0), "a": (0.7, 0.0, 2.0),
                           "b": (0.8, 0.0, 2.0), "eps": (0.08, 0.0, 1.0)},
                          ((-2.5, 2.5), (-1, 2)), 0.05, "FitzHugh-Nagumo: a nerve's spike."),
    "sir": _P(("-beta*x*y", "beta*x*y - gamma*y"),
              {"beta": (0.5, 0.0, 2.0), "gamma": (0.1, 0.0, 1.0)},
              ((0, 1), (0, 1)), 0.05, "An epidemic: susceptible x and infected y."),
    "double_well": _P(("y", "x - x**3 - delta*y"), {"delta": (0.2, 0.0, 2.0)},
                      ((-2, 2), (-1.5, 1.5)), 0.02, "A particle in a double well, with friction."),
    "hopf": _P(("mu*x - y - x*(x**2 + y**2)", "x + mu*y - y*(x**2 + y**2)"),
               {"mu": (0.5, -1.0, 1.0)}, ((-2, 2), (-2, 2)), 0.02,
               "The Hopf normal form: a cycle of radius sqrt(mu) is born at mu = 0."),
    "saddle_node": _P(("mu - x**2", "-y"), {"mu": (1.0, -1.0, 2.0)}, ((-2, 2), (-1.5, 1.5)),
                      0.02, "The saddle-node normal form: two equilibria meet at mu = 0."),
    "pitchfork": _P(("mu*x - x**3", "-y"), {"mu": (1.0, -1.0, 2.0)}, ((-2, 2), (-1.5, 1.5)),
                    0.02, "The pitchfork normal form: one equilibrium splits in three."),
    "linear_saddle": _P(("x", "-y"), {}, ((-2, 2), (-2, 2)), 0.02, "A saddle."),
    "linear_node": _P(("-x", "-2*y"), {}, ((-2, 2), (-2, 2)), 0.02, "A stable node."),
    "linear_focus": _P(("-0.3*x - y", "x - 0.3*y"), {}, ((-2, 2), (-2, 2)), 0.02,
                       "A stable focus (spiral)."),
    "linear_centre": _P(("y", "-x"), {}, ((-2, 2), (-2, 2)), 0.02, "A centre: closed orbits."),
    "linear_degenerate": _P(("-x + y", "-y"), {}, ((-2, 2), (-2, 2)), 0.02,
                            "A degenerate node: one eigen-direction."),
}  # fmt: skip
