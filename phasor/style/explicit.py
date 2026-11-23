"""Explicit style values beside roles (Milestone 18, owner 2026-11-23).

A style key holds a role name (``"highlight"``, ``"bold"``: the default and the AI-friendly
path) or, where its role group allows, an explicit value:

| group      | explicit value                                                          |
| emphasis   | a color: ``"#58C4DD"`` (sRGB, decoded to linear once, here) or linear RGBA |
| text       | a text height in world units                                            |
| stroke     | a stroke width in world units                                           |

A size is positive and at most ``MAX_SIZE`` (1e6 world units; checked by the document schema,
``core/schema.py``, which cannot import this layer). On an overlay-layer object the units are
the overlay's: frame heights. A scene background (``meta.background``) is an opaque color.

Glow, surface and colormap keys take roles only. Resolution is pure: a color resolves when
the document compiles; a world size is turned into the share of the frame height the
drawing code expects when the scene is evaluated at ``t``, through the camera of the
object's layer at its ``pos`` (exact for the 2D camera; in 3D exact at ``pos``).

Taste stays checked: the aesthetic rules ``off_palette`` and ``off_scale``
(``inspect/aesthetic.py``) warn about values far from the theme, and ``polish(snap=True)``
snaps them to the nearest role (``nearest_color_role``, ``nearest_size_role``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from phasor.core.color import hex_to_linear, linear_to_oklab
from phasor.core.schema import MAX_SIZE as MAX_SIZE  # re-exported: the largest size
from phasor.style.theme import ROLE_GROUPS, Theme, ThemeError

# role group -> what an explicit value means there
EXPLICIT: dict[str, str] = {"emphasis": "color", "text": "size", "stroke": "size"}


def is_role(value: Any) -> bool:
    """A role name (as opposed to an explicit value)."""
    return isinstance(value, str) and not value.startswith("#")


def to_linear(value: str | tuple[float, ...]) -> tuple[float, float, float, float]:
    """An explicit color as linear RGBA: hex is sRGB, decoded here (contract 4); a tuple is
    linear already."""
    if isinstance(value, str):
        return hex_to_linear(value)
    r, g, b, a = (float(x) for x in (*value, 1.0)[:4])
    return (r, g, b, a)


def resolve(theme: Theme, group: str, value: Any) -> tuple[Any, float | None]:
    """``(resolved, world)`` for one style key: the theme's value for a role, the linear
    color for an explicit color, or ``(None, size)`` for an explicit world size (converted
    when the scene is evaluated, ``to_share``). Raises ``ThemeError`` with what is wrong."""
    if is_role(value):
        return theme.resolve(group, value), None
    kind = EXPLICIT.get(group)
    if kind is None:
        raise ThemeError(f"{value!r} is not a {group} role; {group} takes roles only: "
                         f"{ROLE_GROUPS.get(group)}")  # fmt: skip
    if kind == "color":
        if isinstance(value, float):
            raise ThemeError(f"{value!r} is not a color: write a role {ROLE_GROUPS[group]}, "
                             "'#rrggbb' or linear (r, g, b, a)")  # fmt: skip
        return to_linear(value), None
    if not isinstance(value, float):
        raise ThemeError(f"{value!r} is not a size: write a role {ROLE_GROUPS[group]} or a "
                         "number of world units")  # fmt: skip
    return None, value


def to_share(world: float, units_per_height: float) -> float:
    """A world size as the share of the frame height the drawing code multiplies back."""
    return float(world) / max(float(units_per_height), 1e-12)


def oklab_distance(a, b) -> float:
    """Perceptual distance of two linear colors (their RGB), in OKLab units."""
    la = linear_to_oklab(np.clip(np.asarray(a[:3], dtype=np.float64), 0.0, None))
    lb = linear_to_oklab(np.clip(np.asarray(b[:3], dtype=np.float64), 0.0, None))
    return float(np.linalg.norm(la - lb))


def nearest_color_role(theme: Theme, rgba) -> tuple[str, float]:
    """The emphasis role whose color is nearest (OKLab) and how far it is."""
    roles = ROLE_GROUPS["emphasis"]
    dists = [(oklab_distance(rgba, theme.values["emphasis"][r]), r) for r in roles]
    d, role = min(dists)
    return role, d


def nearest_size_role(theme: Theme, group: str, share: float) -> tuple[str, float]:
    """The role of ``group`` (text or stroke) whose size is nearest in ratio, and that ratio
    (>= 1: how many times larger or smaller the value is than the role)."""
    best = min(
        (abs(math.log(share / theme.values[group][r])), r)
        for r in ROLE_GROUPS[group]
        if theme.values[group][r] > 0
    )
    return best[1], math.exp(best[0])
