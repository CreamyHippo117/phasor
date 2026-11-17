"""Themes and style roles (SPEC "Beauty system").

Scenes name roles (``highlight``, ``bold``, ``label``); the theme decides what they look
like. Raw hex codes and pixel sizes never appear in a scene.
"""

from __future__ import annotations

import functools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phasor.core.color import hex_to_linear

THEME_DIR = Path(__file__).parent / "themes"

# role group -> role names. "glow" and "surface" get theme values in Milestones 3 and 6.
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "emphasis": ("primary", "secondary", "muted", "highlight", "danger"),
    "stroke": ("hairline", "regular", "bold"),
    "glow": ("none", "soft", "strong"),
    "text": ("title", "label", "annotation", "math"),
    "surface": ("solid", "glass", "wire", "emissive"),
}
_SECTION = {"emphasis": "palette", "stroke": "stroke", "text": "text"}


_TABLES = {
    "layout": ("gap", "margin", "leader"),
    "lint": ("overlap", "contrast", "tiny", "text_min", "clutter", "crowded"),
}


class ThemeError(ValueError):
    pass


@dataclass(frozen=True)
class Theme:
    name: str
    background: tuple[float, float, float, float]
    values: dict[str, dict[str, Any]]  # group -> role -> resolved value
    layout: dict[str, float] = field(default_factory=dict)  # fractions of the frame height
    lint: dict[str, float] = field(default_factory=dict)  # linter thresholds

    def resolve(self, group: str, role: str) -> Any:
        if role not in ROLE_GROUPS.get(group, ()):
            raise ThemeError(f"{role!r} is not a {group} role; use one of {ROLE_GROUPS.get(group)}")
        try:
            return self.values[group][role]
        except KeyError:
            raise ThemeError(f"theme {self.name!r} defines no {group} role {role!r}") from None


def available_themes() -> list[str]:
    return sorted(p.stem for p in THEME_DIR.glob("*.toml"))


@functools.cache
def load_theme(name: str) -> Theme:
    path = THEME_DIR / f"{name}.toml"
    if not path.exists():
        raise ThemeError(f"unknown theme {name!r}; available: {available_themes()}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    values: dict[str, dict[str, Any]] = {}
    for group, section in _SECTION.items():
        data = raw.get(section, {})
        missing = [r for r in ROLE_GROUPS[group] if r not in data]
        if missing:
            raise ThemeError(f"theme {name!r} is missing [{section}] roles {missing}")
        convert = hex_to_linear if group == "emphasis" else float
        values[group] = {r: convert(data[r]) for r in ROLE_GROUPS[group]}
    extra = {}
    for section, required in _TABLES.items():
        data = raw.get(section, {})
        missing = [k for k in required if k not in data]
        if missing:
            raise ThemeError(f"theme {name!r} is missing [{section}] keys {missing}")
        extra[section] = {k: float(v) for k, v in data.items()}
    return Theme(name, hex_to_linear(raw["palette"]["background"]), values, **extra)
