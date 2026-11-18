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

import numpy as np

from phasor.core.color import hex_to_linear, phase_lut

THEME_DIR = Path(__file__).parent / "themes"

# role group -> role names. "surface" values are opacities until 3D materials (Milestone 6).
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "emphasis": ("primary", "secondary", "muted", "highlight", "danger"),
    "stroke": ("hairline", "regular", "bold"),
    "glow": ("none", "soft", "strong"),
    "text": ("title", "label", "annotation", "math"),
    "surface": ("solid", "glass", "wire", "emissive"),
    "colormap": ("phase", "sequential"),
}
_SECTION = {
    "emphasis": "palette",
    "stroke": "stroke",
    "text": "text",
    "glow": "glow",
    "surface": "surface",
    "colormap": "colormap",
}
LUT_SIZE = 256


_TABLES = {
    "layout": ("gap", "margin", "leader"),
    "lint": ("overlap", "contrast", "tiny", "text_min", "clutter", "crowded"),
    "bloom": ("threshold", "strength", "levels", "exposure", "knee"),
    "narration": ("words_per_second", "lead", "gap", "pause_per_change", "pause_max"),
}


def _colormap(theme: str, role: str, spec: Any) -> tuple[tuple[float, float, float], ...]:
    """A cyclic color table as a tuple (hashable, frozen): ``"oklch L C"`` for the even
    phase wheel at OKLCH lightness L and chroma C (``"oklch"`` alone: 0.74 0.125), or a list
    of hex stops blended in linear light and wrapped back to the first."""
    if isinstance(spec, str) and spec.split()[:1] == ["oklch"]:
        try:
            lc = [float(v) for v in spec.split()[1:]]
        except ValueError:
            lc = []
        if len(lc) not in (0, 2):
            raise ThemeError(f"theme {theme!r}: colormap {role!r} must be 'oklch' or 'oklch L C'")
        lut = phase_lut(LUT_SIZE, *lc)
        return tuple(tuple(float(v) for v in row) for row in lut)
    if not isinstance(spec, list) or len(spec) < 2:
        raise ThemeError(f"theme {theme!r}: colormap {role!r} must be 'oklch' or 2+ hex stops")
    stops = np.array([hex_to_linear(h)[:3] for h in spec] + [hex_to_linear(spec[0])[:3]])
    x = np.linspace(0, len(spec), LUT_SIZE, endpoint=False)
    k = np.minimum(x.astype(int), len(spec) - 1)
    u = (x - k)[:, None]
    lut = stops[k] * (1 - u) + stops[k + 1] * u
    return tuple(tuple(float(v) for v in row) for row in lut)


class ThemeError(ValueError):
    pass


@dataclass(frozen=True)
class Theme:
    name: str
    background: tuple[float, float, float, float]
    values: dict[str, dict[str, Any]]  # group -> role -> resolved value
    layout: dict[str, float] = field(default_factory=dict)  # fractions of the frame height
    lint: dict[str, float] = field(default_factory=dict)  # linter thresholds
    bloom: dict[str, float] = field(default_factory=dict)  # final-look glow settings
    narration: dict[str, float] = field(default_factory=dict)  # speaking rate, pacing
    motion: dict[str, Any] = field(default_factory=dict)  # verb -> {duration, ease}, + scalars

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
        if group == "emphasis":
            values[group] = {r: hex_to_linear(data[r]) for r in ROLE_GROUPS[group]}
        elif group == "colormap":
            values[group] = {r: _colormap(name, r, data[r]) for r in ROLE_GROUPS[group]}
        else:
            values[group] = {r: float(data[r]) for r in ROLE_GROUPS[group]}
    extra = {}
    for section, required in _TABLES.items():
        data = raw.get(section, {})
        missing = [k for k in required if k not in data]
        if missing:
            raise ThemeError(f"theme {name!r} is missing [{section}] keys {missing}")
        extra[section] = {k: float(v) for k, v in data.items()}
    motion = _motion(name, raw.get("motion", {}))
    return Theme(name, hex_to_linear(raw["palette"]["background"]), values, **extra,
                 motion=motion)  # fmt: skip


VERBS = ("reveal", "exit", "trace", "bloom_in", "dim", "focus_pull", "morph", "settle")


def _motion(theme: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("stagger", "dim_level"):
        if key not in data:
            raise ThemeError(f"theme {theme!r} is missing [motion] {key}")
        out[key] = float(data[key])
    for verb in VERBS:
        spec = data.get(verb)
        if not isinstance(spec, dict) or "duration" not in spec or "ease" not in spec:
            raise ThemeError(f"theme {theme!r} needs [motion.{verb}] with duration and ease")
        out[verb] = {"duration": float(spec["duration"]), "ease": str(spec["ease"])}
    return out
