"""Room acoustics numerics (SPEC "Domain modules: acoustics"): pure NumPy, no drawing.

- ``image_sources``: the image-source method for a shoebox room (exact for a rectangular room
  with frequency-independent walls).
- ``impulse_response``: the arrivals (time, amplitude) and the sampled IR.
- ``rt60_fit`` (Schroeder backward integration), ``sabine_rt60`` and ``eyring_rt60`` (theory).
- ``convolve`` (FFT convolution reverb).
- ``RoomFDTD``: a 3D finite-difference pressure field as a ``Simulation`` (fixed dt, seed,
  checkpointed), with the stability limit enforced.

Units: meters, seconds, speed of sound ``C`` (343 m/s) unless said otherwise.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

C = 343.0
MATERIALS = {  # broadband absorption coefficients (energy absorbed per reflection)
    "concrete": 0.02,
    "glass": 0.04,
    "plaster": 0.04,
    "brick": 0.05,
    "wood": 0.10,
    "carpet": 0.30,
    "curtain": 0.50,
    "foam": 0.80,
    "anechoic": 0.99,
}
MAX_IMAGES = 2_000_000


def absorption_of(material: str, absorption: float = -1.0) -> float:
    """``absorption`` if it is given (0-1), else the material's coefficient."""
    if absorption >= 0:
        return float(min(max(absorption, 0.0), 1.0))
    if material not in MATERIALS:
        raise ValueError(f"unknown material {material!r}; materials are {sorted(MATERIALS)}")
    return MATERIALS[material]


def room_stats(size) -> tuple[float, float]:
    """(volume, total wall area) of a shoebox room."""
    x, y, z = (float(v) for v in size)
    return x * y * z, 2 * (x * y + y * z + x * z)


def sabine_rt60(size, alpha: float, c: float = C) -> float:
    """Time for sound to decay 60 dB, Sabine's formula: 0.161 V / (S alpha)."""
    v, s = room_stats(size)
    return 0.161 * v / (s * alpha) if alpha > 0 else math.inf


def eyring_rt60(size, alpha: float) -> float:
    """Eyring's formula, better than Sabine when walls absorb a lot."""
    v, s = room_stats(size)
    return 0.161 * v / (-s * math.log(1.0 - alpha)) if 0 < alpha < 1 else (
        math.inf if alpha <= 0 else 0.0)  # fmt: skip


def image_sources(size, source, order: int, alpha: float) -> dict[str, np.ndarray]:
    """Image sources of ``source`` in a shoebox room (corner at the origin) with up to
    ``order`` wall reflections. Returns arrays: ``pos`` (N, 3), ``reflections`` (N,), and
    ``gain`` (N,) = (sqrt(1 - alpha)) ** reflections (pressure reflection factor). The
    source itself (order 0) is first. Deterministic order: by reflections, then position."""
    size = np.asarray(size, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    if order < 0:
        raise ValueError("order must be 0 or more")
    q = np.arange(-order, order + 1)
    axes = []
    for L, s in zip(size, src, strict=True):
        pos = np.concatenate([2 * L * q + s, 2 * L * q - s])
        refl = np.concatenate([np.abs(2 * q), np.abs(2 * q - 1)])
        keep = refl <= order
        axes.append((pos[keep], refl[keep]))
    n = [len(a[0]) for a in axes]
    if n[0] * n[1] * n[2] > MAX_IMAGES:
        raise ValueError(f"order {order} needs {n[0] * n[1] * n[2]:,} image sources "
                         f"(limit {MAX_IMAGES:,}); lower the order")  # fmt: skip
    px, py, pz = np.meshgrid(axes[0][0], axes[1][0], axes[2][0], indexing="ij")
    rx, ry, rz = np.meshgrid(axes[0][1], axes[1][1], axes[2][1], indexing="ij")
    refl = (rx + ry + rz).ravel()
    keep = refl <= order
    pos = np.stack([px.ravel(), py.ravel(), pz.ravel()], axis=1)[keep]
    refl = refl[keep]
    key = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0], refl))
    pos, refl = pos[key], refl[key]
    return {"pos": pos, "reflections": refl, "gain": np.sqrt(1.0 - alpha) ** refl}


def arrivals(size, source, mic, order: int, alpha: float, c: float = C) -> dict[str, np.ndarray]:
    """What reaches the listener: for every image source its ``time`` (s), ``amplitude``
    (pressure: gain / distance, 1/r spreading), ``reflections`` and ``distance``; sorted by
    time. The first entry is the direct sound."""
    im = image_sources(size, source, order, alpha)
    d = np.linalg.norm(im["pos"] - np.asarray(mic, dtype=np.float64), axis=1)
    d = np.maximum(d, 1e-3)
    idx = np.argsort(d, kind="stable")
    return {
        "time": d[idx] / c,
        "amplitude": (im["gain"] / d)[idx],
        "reflections": im["reflections"][idx],
        "distance": d[idx],
        "pos": im["pos"][idx],
    }


def impulse_response(arr: dict[str, np.ndarray], fs: int, length: float) -> np.ndarray:
    """The sampled impulse response: each arrival is split between the two nearest samples
    (linear fractional delay) so timing is not rounded to a whole sample."""
    n = math.ceil(length * fs)
    ir = np.zeros(n + 2)
    pos = arr["time"] * fs
    i = np.floor(pos).astype(np.int64)
    frac = pos - i
    ok = i < n
    np.add.at(ir, i[ok], arr["amplitude"][ok] * (1 - frac[ok]))
    np.add.at(ir, i[ok] + 1, arr["amplitude"][ok] * frac[ok])
    return ir[:n]


def schroeder_db(ir: np.ndarray) -> np.ndarray:
    """The Schroeder decay curve in dB (0 dB at the start): backward-integrated energy."""
    e = np.cumsum((ir.astype(np.float64) ** 2)[::-1])[::-1]
    e = e / max(e[0], 1e-300)
    return 10.0 * np.log10(np.maximum(e, 1e-300))


def rt60_fit(ir: np.ndarray, fs: int, lo: float = -5.0, hi: float = -25.0) -> dict[str, Any]:
    """Fit a line to the Schroeder curve between ``lo`` and ``hi`` dB (T20 by default) and
    extrapolate to 60 dB. Returns ``rt60``, the ``slope`` (dB/s), the curve and the fitted line."""
    curve = schroeder_db(ir)
    t = np.arange(len(curve)) / fs
    sel = (curve <= lo) & (curve >= hi)
    if sel.sum() < 3:
        raise ValueError("the impulse response is too short to fit a decay")
    slope, intercept = np.polyfit(t[sel], curve[sel], 1)
    if slope >= 0:
        raise ValueError("no decay found in the impulse response")
    return {"rt60": float(-60.0 / slope), "slope": float(slope), "intercept": float(intercept),
            "curve": curve, "t": t, "fit": slope * t + intercept}  # fmt: skip


def convolve(dry: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolution reverb by FFT: the dry signal as it would sound in the room."""
    n = len(dry) + len(ir) - 1
    size = 1 << (n - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(dry, size) * np.fft.rfft(ir, size), size)[:n]
    return wet


# ---------------------------------------------------------------- FDTD pressure field


class RoomFDTD:
    """A 3D wave-equation pressure field in a shoebox room, leapfrog finite differences.

    ``p_next = 2 p - p_prev + (c dt / dx)^2 laplacian(p) + source``, rigid walls, plus a thin
    damping shell at the walls whose strength comes from ``alpha`` (a model of absorption,
    not an exact boundary condition). Stability needs ``dt <= dx / (c sqrt(3))``; asking for
    more raises ``ValueError`` (SPEC: FDTD stability is a validation error).
    """

    def __init__(
        self,
        size,
        source,
        *,
        dx: float = 0.2,
        c: float = C,
        alpha: float = 0.1,
        dt: float | None = None,
        frequency: float = 120.0,
        seed: int = 0,
    ) -> None:
        self.size = tuple(float(v) for v in size)
        self.source = tuple(float(v) for v in source)
        self.dx, self.c, self.alpha = float(dx), float(c), float(alpha)
        self.frequency, self.seed = float(frequency), int(seed)
        limit = self.dx / (self.c * math.sqrt(3.0))
        self.dt = 0.95 * limit if dt is None else float(dt)
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(
                f"unstable: dt = {self.dt:g} s is above the limit dx / (c sqrt(3)) = {limit:g} s; "
                "use a smaller dt or a larger dx"
            )
        self.shape = tuple(max(3, round(v / self.dx)) for v in self.size)
        self.k = (self.c * self.dt / self.dx) ** 2
        # damping shell: 2 cells, strength from the absorption coefficient
        sigma = np.ones(self.shape)
        edge = 2
        for ax, n in enumerate(self.shape):
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)
            shell = np.where(d < edge, 1.0 - self.alpha * (edge - d) / edge * 0.5, 1.0)
            view = [1, 1, 1]
            view[ax] = n
            sigma = sigma * shell.reshape(view)
        self.sigma = sigma
        self.cell = tuple(min(max(int(v / self.dx), 0), n - 1)
                          for v, n in zip(self.source, self.shape, strict=True))  # fmt: skip

    # the Simulation protocol (core/simulation.py)
    def params(self) -> dict[str, Any]:
        return {"size": self.size, "source": self.source, "dx": self.dx, "c": self.c,
                "alpha": self.alpha, "frequency": self.frequency, "seed": self.seed,
                "dt": self.dt}  # fmt: skip

    def initial(self) -> dict[str, np.ndarray]:
        z = np.zeros(self.shape)
        return {"p": z, "prev": z.copy(), "n": np.zeros(1)}

    def pulse(self, n: int) -> float:
        """The source: one cycle of a Gaussian-windowed sine at ``frequency`` Hz, then silence."""
        t = n * self.dt
        t0 = 1.5 / self.frequency
        return math.sin(2 * math.pi * self.frequency * (t - t0)) * math.exp(
            -(((t - t0) * self.frequency / 0.6) ** 2)
        )

    def step(self, state: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
        p, prev = state["p"], state["prev"]
        n = int(state["n"][0])
        padded = np.pad(p, 1, mode="edge")  # rigid walls: zero normal gradient
        lap = (
            padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
            + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2] - 6.0 * p
        )  # fmt: skip
        new = 2.0 * p - prev + self.k * lap
        new[self.cell] += self.pulse(n) * self.k * 40.0
        new = new * self.sigma
        return {"p": new, "prev": p, "n": state["n"] + 1}

    def slice_z(self, state: dict[str, np.ndarray], z: float) -> np.ndarray:
        """The pressure on a horizontal plane at height ``z`` (m), shape (nx, ny)."""
        k = min(max(int(z / self.dx), 0), self.shape[2] - 1)
        return state["p"][:, :, k]

    def at(self, state: dict[str, np.ndarray], point) -> float:
        """The pressure at a point (nearest cell)."""
        i = [min(max(int(v / self.dx), 0), n - 1) for v, n in zip(point, self.shape, strict=True)]
        return float(state["p"][tuple(i)])
