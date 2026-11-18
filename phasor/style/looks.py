"""Look presets (SPEC "Presets"): a theme plus a handful of choices, by name.

``Scene("x", look="calm")`` starts from a look: it picks the theme and the scene's ``life``
and supplies defaults for object parameters and style roles. A look is only named defaults:
anything a script or patch sets explicitly wins, and the document stores the resolved
values (``meta.theme``, ``meta.life`` and each object's own params), so it renders the same
later even if the look file changes. Files live in ``style/looks/*.toml``.
"""

from __future__ import annotations

import functools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phasor.core.life import LEVELS
from phasor.style.theme import ThemeError, available_themes

LOOK_DIR = Path(__file__).parent / "looks"
DEFAULT_THEME = "midnight_glow"  # the house look is dark, always


class LookError(ValueError):
    pass


@dataclass(frozen=True)
class Look:
    name: str
    theme: str
    life: str
    description: str = ""
    # object type -> {"params": {...}, "style": {...}}; filled in only where a scene says nothing
    defaults: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def fill(self, type_name: str, params: dict[str, Any], style: dict[str, str]):
        """``params`` and ``style`` with this look's defaults for ``type_name`` added where
        the object did not set them."""
        d = self.defaults.get(type_name, {})
        return {**d.get("params", {}), **params}, {**d.get("style", {}), **style}


def available_looks() -> list[str]:
    return sorted(p.stem for p in LOOK_DIR.glob("*.toml"))


@functools.cache
def load_look(name: str) -> Look:
    path = LOOK_DIR / f"{name}.toml"
    if not path.exists():
        raise LookError(f"unknown look {name!r}; available: {available_looks()}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    theme, life = raw.get("theme", DEFAULT_THEME), raw.get("life", "off")
    if theme not in available_themes():
        raise ThemeError(f"look {name!r} names unknown theme {theme!r}")
    if life not in LEVELS:
        raise LookError(f"look {name!r}: life must be one of {LEVELS}")
    defaults = {
        t: {"params": dict(v.get("params", {})), "style": dict(v.get("style", {}))}
        for t, v in raw.get("defaults", {}).items()
    }
    return Look(name, theme, life, raw.get("description", ""), defaults)
