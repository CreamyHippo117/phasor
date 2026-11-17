"""Easing curves: functions from progress u in [0, 1] to eased progress, with f(0)=0, f(1)=1."""

from __future__ import annotations

from collections.abc import Callable

EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda u: u,
    "smooth": lambda u: u * u * (3 - 2 * u),
    "smoother": lambda u: u * u * u * (u * (6 * u - 15) + 10),
    "in": lambda u: u * u * u,
    "out": lambda u: 1 - (1 - u) ** 3,
    "in_out": lambda u: 4 * u * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2,
}
