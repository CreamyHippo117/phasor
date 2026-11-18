"""PDE numerics (SPEC "Domain modules: pde"): pure NumPy finite-difference and spectral
solvers, each a ``Simulation`` (``core/simulation.py``): a fixed ``dt``, ``params()`` with the
seed, ``initial()`` and a pure ``step``.

- ``Heat1D`` / ``Heat2D``: u_t = k laplacian(u), explicit, zero boundary values.
- ``Wave1D`` / ``Wave2D``: u_tt = c^2 laplacian(u), leapfrog, fixed edges (a string, a drum).
- ``Advection1D``: u_t + c u_x = 0, Lax-Wendroff, periodic.
- ``GrayScott``: a reaction-diffusion pattern former, periodic.
- ``Schrodinger1D``: i psi_t = -psi_xx / 2 + V psi, split-step Fourier (unitary).
- ``solve_poisson``: laplacian(u) = f with Dirichlet boundary values (sparse direct solve).

A step size that would be unstable is a ``ValueError`` when the simulation is built (SPEC:
stability is a validation error, never a blow-up). Exact solutions are provided where they
exist, so a picture can show the numbers beside the truth.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from phasor.core.rng import rng

Array = np.ndarray


def _grid(length: float, n: int) -> Array:
    """``n`` cells, boundaries at both ends: n + 1 points from 0 to length."""
    return np.linspace(0.0, length, n + 1)


class _Sim:
    dt: float

    def params(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


# ---------------------------------------------------------------- heat


class Heat1D(_Sim):
    """A rod of length ``length``, ends held at 0, started as sine modes ``modes`` (amplitudes
    of sin(m pi x / L), m = 1, 2, ...) or a Gaussian ``bump`` when ``modes`` is empty."""

    def __init__(self, *, k=1.0, length=1.0, n=100, modes=(1.0,), dt=None) -> None:
        self.k, self.length, self.n = float(k), float(length), int(n)
        self.modes = tuple(float(m) for m in modes)
        self.dx = self.length / self.n
        limit = 0.5 * self.dx**2 / self.k
        self.dt = 0.8 * limit if dt is None else float(dt)
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(f"unstable: dt = {self.dt:g} is above dx^2 / (2 k) = {limit:g}")
        self.x = _grid(self.length, self.n)

    def initial(self) -> dict[str, Array]:
        return {"u": self.profile(0.0), "n": np.zeros(1)}

    def profile(self, t: float) -> Array:
        """The exact solution at time t (for the mode initial condition)."""
        if not self.modes:
            return np.exp(-(((self.x - self.length / 2) / (0.1 * self.length)) ** 2))
        u = np.zeros_like(self.x)
        for m, a in enumerate(self.modes, start=1):
            w = m * math.pi / self.length
            u += a * np.sin(w * self.x) * math.exp(-self.k * w * w * t)
        return u

    def exact(self, t: float) -> Array | None:
        return self.profile(t) if self.modes else None

    def step(self, s: dict[str, Array], dt: float) -> dict[str, Array]:
        u = s["u"]
        new = u.copy()
        new[1:-1] = u[1:-1] + self.k * dt / self.dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        new[0] = new[-1] = 0.0
        return {"u": new, "n": s["n"] + 1}


class Heat2D(_Sim):
    """A plate ``size`` = (Lx, Ly), edges held at 0. ``mode`` = (m, n) starts as the single
    mode sin(m pi x / Lx) sin(n pi y / Ly); ``(0, 0)`` starts as a hot Gaussian spot."""

    def __init__(self, *, k=1.0, size=(1.0, 1.0), n=48, mode=(1, 1), dt=None) -> None:
        self.k, self.size, self.n = float(k), (float(size[0]), float(size[1])), int(n)
        self.mode = (int(mode[0]), int(mode[1]))
        self.dx = max(self.size) / self.n
        self.shape = (round(self.size[0] / self.dx) + 1, round(self.size[1] / self.dx) + 1)
        limit = 0.25 * self.dx**2 / self.k
        self.dt = 0.8 * limit if dt is None else float(dt)
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(f"unstable: dt = {self.dt:g} is above dx^2 / (4 k) = {limit:g}")
        self.x = np.linspace(0.0, self.size[0], self.shape[0])
        self.y = np.linspace(0.0, self.size[1], self.shape[1])

    def initial(self) -> dict[str, Array]:
        return {"u": self.profile(0.0), "n": np.zeros(1)}

    def profile(self, t: float) -> Array:
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        m, n = self.mode
        if (m, n) == (0, 0):
            r2 = ((X - self.size[0] / 2) ** 2 + (Y - self.size[1] / 2) ** 2) / (
                0.08 * max(self.size)
            ) ** 2
            return np.exp(-r2)
        lam = (m * math.pi / self.size[0]) ** 2 + (n * math.pi / self.size[1]) ** 2
        return (
            np.sin(m * math.pi * X / self.size[0])
            * np.sin(n * math.pi * Y / self.size[1])
            * math.exp(-self.k * lam * t)
        )

    def exact(self, t: float) -> Array | None:
        return self.profile(t) if self.mode != (0, 0) else None

    def step(self, s: dict[str, Array], dt: float) -> dict[str, Array]:
        u = s["u"]
        new = u.copy()
        lap = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
        new[1:-1, 1:-1] = u[1:-1, 1:-1] + self.k * dt / self.dx**2 * lap
        new[0, :] = new[-1, :] = new[:, 0] = new[:, -1] = 0.0
        return {"u": new, "n": s["n"] + 1}


# ---------------------------------------------------------------- waves


class Wave1D(_Sim):
    """A string of length ``length`` fixed at both ends: standing wave ``mode`` (sin(m pi x / L))
    or, with ``mode = 0``, a plucked triangle."""

    def __init__(self, *, c=1.0, length=1.0, n=200, mode=1, dt=None) -> None:
        self.c, self.length, self.n, self.mode = float(c), float(length), int(n), int(mode)
        self.dx = self.length / self.n
        limit = self.dx / self.c
        self.dt = 0.9 * limit if dt is None else float(dt)
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(f"unstable: dt = {self.dt:g} is above dx / c = {limit:g}")
        self.x = _grid(self.length, self.n)

    def _u0(self) -> Array:
        if self.mode:
            return np.sin(self.mode * math.pi * self.x / self.length)
        peak = 0.3 * self.length
        return np.where(self.x < peak, self.x / peak, (self.length - self.x) / (self.length - peak))

    def initial(self) -> dict[str, Array]:
        u0 = self._u0()
        lap = np.zeros_like(u0)
        lap[1:-1] = u0[2:] - 2 * u0[1:-1] + u0[:-2]
        # Zero initial velocity makes the motion symmetric in time, so the step before t = 0
        # is the step after it: u(-dt) = u(dt) = u0 + (c dt)^2 u_xx / 2. State n is time n dt.
        u1 = u0 + 0.5 * (self.c * self.dt / self.dx) ** 2 * lap
        return {"u": u0, "prev": u1, "n": np.zeros(1)}

    def exact(self, t: float) -> Array | None:
        if not self.mode:
            return None
        w = self.mode * math.pi / self.length
        return np.sin(w * self.x) * math.cos(self.c * w * (t))

    def step(self, s: dict[str, Array], dt: float) -> dict[str, Array]:
        u, prev = s["u"], s["prev"]
        new = np.zeros_like(u)
        new[1:-1] = (
            2 * u[1:-1] - prev[1:-1] + (self.c * dt / self.dx) ** 2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        )
        return {"u": new, "prev": u, "n": s["n"] + 1}


class Wave2D(_Sim):
    """A drum of ``size`` fixed at its edge, in the standing-wave ``mode`` = (m, n)."""

    def __init__(self, *, c=1.0, size=(1.0, 1.0), n=64, mode=(1, 1), dt=None) -> None:
        self.c, self.size, self.n = float(c), (float(size[0]), float(size[1])), int(n)
        self.mode = (int(mode[0]), int(mode[1]))
        self.dx = max(self.size) / self.n
        self.shape = (round(self.size[0] / self.dx) + 1, round(self.size[1] / self.dx) + 1)
        limit = self.dx / (self.c * math.sqrt(2.0))
        self.dt = 0.9 * limit if dt is None else float(dt)
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(f"unstable: dt = {self.dt:g} is above dx / (c sqrt(2)) = {limit:g}")
        self.x = np.linspace(0.0, self.size[0], self.shape[0])
        self.y = np.linspace(0.0, self.size[1], self.shape[1])

    def omega(self) -> float:
        m, n = self.mode
        return self.c * math.pi * math.hypot(m / self.size[0], n / self.size[1])

    def _shape_fn(self) -> Array:
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        m, n = self.mode
        return np.sin(m * math.pi * X / self.size[0]) * np.sin(n * math.pi * Y / self.size[1])

    def initial(self) -> dict[str, Array]:
        u0 = self._shape_fn()
        lap = np.zeros_like(u0)
        lap[1:-1, 1:-1] = (
            u0[2:, 1:-1] + u0[:-2, 1:-1] + u0[1:-1, 2:] + u0[1:-1, :-2] - 4 * u0[1:-1, 1:-1]
        )
        u1 = u0 + 0.5 * (self.c * self.dt / self.dx) ** 2 * lap  # u(-dt) = u(dt), as in Wave1D
        return {"u": u0, "prev": u1, "n": np.zeros(1)}

    def exact(self, t: float) -> Array | None:
        return self._shape_fn() * math.cos(self.omega() * t)

    def step(self, s: dict[str, Array], dt: float) -> dict[str, Array]:
        u, prev = s["u"], s["prev"]
        new = np.zeros_like(u)
        lap = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
        new[1:-1, 1:-1] = 2 * u[1:-1, 1:-1] - prev[1:-1, 1:-1] + (self.c * dt / self.dx) ** 2 * lap
        return {"u": new, "prev": u, "n": s["n"] + 1}


# ---------------------------------------------------------------- advection


class Advection1D(_Sim):
    """u_t + c u_x = 0 on a periodic domain: a Gaussian pulse carried at speed ``c`` (Lax-Wendroff,
    second order; the pulse keeps its shape up to a little numerical dispersion)."""

    def __init__(self, *, c=1.0, length=1.0, n=200, dt=None) -> None:
        self.c, self.length, self.n = float(c), float(length), int(n)
        self.dx = self.length / self.n
        limit = self.dx / abs(self.c)
        self.dt = 0.8 * limit if dt is None else float(dt)
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(f"unstable: dt = {self.dt:g} is above dx / |c| = {limit:g}")
        self.x = np.arange(self.n) * self.dx

    def _pulse(self, shift: float) -> Array:
        d = (self.x - 0.3 * self.length - shift + self.length / 2) % self.length - self.length / 2
        return np.exp(-((d / (0.05 * self.length)) ** 2))

    def initial(self) -> dict[str, Array]:
        return {"u": self._pulse(0.0), "n": np.zeros(1)}

    def exact(self, t: float) -> Array:
        return self._pulse(self.c * t)

    def step(self, s: dict[str, Array], dt: float) -> dict[str, Array]:
        u, nu = s["u"], self.c * dt / self.dx
        new = (
            u
            - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1))
            + 0.5 * nu**2 * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
        )
        return {"u": new, "n": s["n"] + 1}


# ---------------------------------------------------------------- reaction-diffusion


class GrayScott(_Sim):
    """The Gray-Scott model on a periodic grid: two chemicals that diffuse and react; with the
    right feed ``f`` and kill ``k`` it grows spots and stripes from a seeded disturbance."""

    def __init__(self, *, f=0.037, k=0.06, du=0.16, dv=0.08, n=64, dt=1.0, seed=0) -> None:
        self.f, self.k, self.du, self.dv, self.n = float(f), float(k), float(du), float(dv), int(n)
        self.dt, self.seed = float(dt), int(seed)
        limit = 0.25 / max(self.du, self.dv)  # dx = 1
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(f"unstable: dt = {self.dt:g} is above 1 / (4 max(du, dv)) = {limit:g}")

    def initial(self) -> dict[str, Array]:
        g = rng("gray_scott", self.seed)
        u = np.ones((self.n, self.n))
        v = np.zeros((self.n, self.n))
        a, b = self.n // 2 - self.n // 8, self.n // 2 + self.n // 8
        u[a:b, a:b], v[a:b, a:b] = 0.5, 0.25
        v += 0.02 * g.random((self.n, self.n))
        return {"u": u, "v": v, "n": np.zeros(1)}

    def step(self, s: dict[str, Array], dt: float) -> dict[str, Array]:
        u, v = s["u"], s["v"]

        def lap(a):
            return (
                np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a
            )

        uvv = u * v * v
        nu = u + dt * (self.du * lap(u) - uvv + self.f * (1 - u))
        nv = v + dt * (self.dv * lap(v) + uvv - (self.f + self.k) * v)
        return {"u": np.clip(nu, 0.0, 1.5), "v": np.clip(nv, 0.0, 1.5), "n": s["n"] + 1}


# ---------------------------------------------------------------- Schrodinger


class Schrodinger1D(_Sim):
    """A wave packet in a potential (hbar = m = 1): i psi_t = -psi_xx / 2 + V(x) psi on a periodic
    grid of length ``length``, by split-step Fourier, which keeps the norm exactly.
    ``potential``: ``"free"``, ``"harmonic"`` (V = x^2 / 2 about the centre) or ``"barrier"``."""

    def __init__(
        self, *, length=20.0, n=256, x0=-4.0, p0=2.0, width=1.0, potential="free", dt=0.01
    ) -> None:
        self.length, self.n, self.x0, self.p0, self.width = (
            float(length),
            int(n),
            float(x0),
            float(p0),
            float(width),
        )
        self.potential, self.dt = potential, float(dt)
        if potential not in ("free", "harmonic", "barrier"):
            raise ValueError("potential is 'free', 'harmonic' or 'barrier'")
        self.dx = self.length / self.n
        self.x = (np.arange(self.n) - self.n / 2) * self.dx
        self.k = 2 * math.pi * np.fft.fftfreq(self.n, d=self.dx)

    def V(self) -> Array:
        if self.potential == "harmonic":
            return 0.5 * self.x**2
        if self.potential == "barrier":
            return 8.0 * (np.abs(self.x) < 0.5)
        return np.zeros_like(self.x)

    def initial(self) -> dict[str, Array]:
        psi = np.exp(-((self.x - self.x0) ** 2) / (4 * self.width**2) + 1j * self.p0 * self.x)
        psi /= math.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)
        return {"re": psi.real.copy(), "im": psi.imag.copy(), "n": np.zeros(1)}

    @staticmethod
    def psi(s: dict[str, Array]) -> Array:
        return s["re"] + 1j * s["im"]

    def density(self, s: dict[str, Array]) -> Array:
        return np.abs(self.psi(s)) ** 2

    def step(self, s: dict[str, Array], dt: float) -> dict[str, Array]:
        psi = self.psi(s)
        half = np.exp(-0.5j * self.V() * dt)
        psi = half * psi
        psi = np.fft.ifft(np.exp(-0.5j * self.k**2 * dt) * np.fft.fft(psi))
        psi = half * psi
        return {"re": psi.real.copy(), "im": psi.imag.copy(), "n": s["n"] + 1}


# ---------------------------------------------------------------- Poisson / Laplace


def solve_poisson(f: Array, h: float, boundary: Array) -> Array:
    """Solve laplacian(u) = f on a grid with spacing ``h`` and Dirichlet values given by the
    outer ring of ``boundary`` (same shape as ``f``). Five-point stencil, sparse direct solve;
    exact for polynomials up to degree 3."""
    f, boundary = np.asarray(f, dtype=np.float64), np.asarray(boundary, dtype=np.float64)
    if f.shape != boundary.shape or min(f.shape) < 3:
        raise ValueError("f and boundary must have the same shape, at least 3x3")
    nx, ny = f.shape
    mx, my = nx - 2, ny - 2
    ix = sps.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(mx, mx))
    iy = sps.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(my, my))
    lap = (sps.kron(ix, sps.identity(my)) + sps.kron(sps.identity(mx), iy)) / h**2
    rhs = f[1:-1, 1:-1].copy()
    rhs[0, :] -= boundary[0, 1:-1] / h**2
    rhs[-1, :] -= boundary[-1, 1:-1] / h**2
    rhs[:, 0] -= boundary[1:-1, 0] / h**2
    rhs[:, -1] -= boundary[1:-1, -1] / h**2
    u = boundary.copy()
    u[1:-1, 1:-1] = spla.spsolve(lap.tocsc(), rhs.ravel()).reshape(mx, my)
    return u
