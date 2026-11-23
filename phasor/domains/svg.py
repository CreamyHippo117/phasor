"""``SVG``: a drawing read from an .svg file next to the scene (Milestone 16).

It is a composite (ADR 0005): each shape of the file becomes a ``Path`` part,
``<svg id>.<shape id>``, named from the shape's ``id`` attribute (lower case, other
characters as ``_``; a shape without one is named by its kind: ``circle``, ``circle_2``).
The file is read only from inside the scene's folder (contract 3: ``Files``), and read again
when it changes; a missing file is lint ``missing_file``, and without svgelements (the
``svg`` extra) lint ``missing_extra``. Either way the drawing is empty, never an error.

The drawing's view box is centered on ``pos`` and scaled to ``height`` world units, with
the file's y axis (down) turned up. Shapes take the roles of the ``SVG`` object's style:
a shape the file fills is filled with ``fill_color`` at the ``fill`` role's opacity, one it
strokes is stroked; the file's own colors are not used (roles only). Text, pictures and
external references in the file are ignored.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core import files
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import path_math as pm
from phasor.domains.path import STYLE, Path

EXTRA = {"svgelements": "svg"}  # module -> the extra that installs it


def part_names(shapes) -> list[str]:
    """A valid, unique part name per shape, from its ``id`` (or its kind), in file order."""
    used: set[str] = set()
    out = []
    for sh in shapes:
        base = re.sub(r"[^a-z0-9_]+", "_", sh.id.lower()).strip("_")
        if not base:
            base = sh.tag
        elif not base[0].isalpha():
            base = f"{sh.tag}_{base}"
        name, n = base, 2
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)
        out.append(name)
    return out


@register
class SVG(ObjectBase):
    """The shapes of an SVG file, as ``Path`` parts."""

    type: ClassVar[str] = "svg"
    Files: ClassVar = ("path",)
    Requires: ClassVar = EXTRA  # optional packages it needs (lint missing_extra)
    Style: ClassVar = {**STYLE, "fill": ("fill", "solid")}
    Presets: ClassVar = {
        "solid": {},
        "outline": {"style": {"fill": "none"}},
        "glowing": {"style": {"glow": "soft", "fill": "soft"}},
    }

    class Params(ParamsBase):
        path: str = ""  # an .svg file in the scene's folder
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the middle of its view box goes
        height: float = Field(2.0, gt=0)  # world height of its view box

        @field_validator("path")
        @classmethod
        def _path(cls, v):
            return files.relative_path(v) if v else v

    def drawing(self, state: SceneState) -> pm.SvgDrawing | None:
        """The file's shapes, or None while it is missing or cannot be read (or svgelements
        is not installed)."""
        p = self.me(state).params
        if not p.path or not pm.extra_available("svgelements"):
            return None
        stamp = files.stamp(state.base_dir, p.path)
        if stamp[1] is None:
            return None
        try:
            return pm.read_svg(stamp[0], stamp)
        except (OSError, ValueError, pm.MissingExtra):
            return None
        except Exception:  # a malformed file: nothing to draw (lint says the file is bad)
            return None

    def build(self, state: SceneState) -> list[ObjectBase]:
        me = self.me(state)
        p = me.params
        d = self.drawing(state)
        if d is None:
            return []
        x0, y0, w, h = d.box
        s = p.height / (h if h > 0 else 1.0)
        mid = np.array([x0 + w / 2, y0 + h / 2, 0.0])
        flip = np.array([s, -s, 1.0])
        origin = np.asarray(p.pos, dtype=np.float64)
        style = {k: me.roles[k] for k in STYLE if k in me.roles}
        parts = []
        for name, sh in zip(part_names(d.shapes), d.shapes, strict=True):
            subs = [np.round((sub - mid) * flip + origin, 9).tolist() for sub in sh.subpaths]
            parts.append(Path(name, subpaths=subs, rule=sh.rule, style=style,
                              fill_opacity=1.0 if sh.filled else 0.0,
                              stroke_opacity=1.0 if sh.stroked else 0.0))  # fmt: skip
        return parts

    def semantics(self, state: SceneState) -> dict[str, Any]:
        from phasor.core.record import parts

        me = self.me(state)
        p = me.params
        out: dict[str, Any] = {"kind": "svg", "file": p.path,
                               "parts": [o.id[len(self.id) + 1 :] for o in parts(state, self.id)]}  # fmt: skip
        if not pm.extra_available("svgelements"):
            out["missing_extra"] = "svg"
        elif self.drawing(state) is None:
            there = p.path and files.exists(state.base_dir, p.path)
            out["unreadable" if there else "missing_file"] = p.path
        return out
