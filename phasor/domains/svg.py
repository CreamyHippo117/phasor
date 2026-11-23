"""``SVG``: a drawing read from an .svg file next to the scene (Milestone 16).

It is a composite (ADR 0005): each shape of the file becomes a ``Path`` part,
``<svg id>.<shape id>``, named from the shape's ``id`` attribute (lower case, other
characters as ``_``, at most ``MAX_NAME`` characters; a shape without one is named by its
kind: ``circle``, ``circle_2``). A name the document already uses or has retired under the
``SVG`` (``logo.label``, say) gets the next free suffix (``label_2``), so a file can never
take an id that belongs to the document. Names follow file order: a shape without an id
added early in the file renumbers the later unnamed shapes of its kind (give shapes ids
to keep constraints and overrides on them).

The file is read only from inside the scene's folder (contract 3: ``Files``), and read again
when it changes. A bad or hostile file never makes the scene fail; the drawing is empty (or
cut short) and lint says why: a missing file (or a folder, a FIFO, a device, a link that
loops) is ``missing_file``; a file that is not valid SVG is ``bad_file``, as are shapes
skipped for points that are not finite or absurdly far out; a file past one of the limits
in ``path_math`` (size, elements once ``<use>`` is expanded, nesting, shapes, segments) is
``svg_too_complex``; without svgelements (the ``svg`` extra) it is ``missing_extra``.

The drawing's view box is centered on ``pos`` and scaled to ``height`` world units, with
the file's y axis (down) turned up. Shapes take the roles of the ``SVG`` object's style:
a shape the file fills is filled with ``fill_color`` at the ``fill`` role's opacity, one it
strokes is stroked; the file's own colors are not used (roles only). Text, pictures and
external references in the file are ignored.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core import extras, files
from phasor.core.ids import children, name_from, unique_names
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import path_math as pm
from phasor.domains.path import MAX_COORD, STYLE, Path

MAX_NAME = 40  # characters of a part's name taken from the file (before any suffix)
MAX_WORLD = 1e6  # a part with a point this far from ``pos`` (in heights) is skipped


def part_names(shapes, reserved=()) -> list[str]:
    """A valid, unique part name per shape, from its ``id`` (or its kind), in file order,
    none of them in ``reserved`` (names the document holds under the ``SVG``)."""
    return unique_names([name_from(sh.id, sh.tag, MAX_NAME) for sh in shapes], reserved)


@functools.lru_cache(maxsize=16)
def _placed(stamp: tuple, pos: tuple, height: float, reserved: frozenset[str]):
    """The parts of the file ``stamp`` names, placed with its view box centred on ``pos``,
    ``height`` tall, named clear of ``reserved``: ((name, subpaths, shape), ...), skipped."""
    d = pm.read_svg(stamp[0], stamp)
    if not d.shapes:
        return (), 0
    x0, y0, w, h = d.box
    s = height / h
    mid = np.array([x0 + w / 2, y0 + h / 2, 0.0])
    flip = np.array([s, -s, 1.0])
    origin = np.asarray(pos, dtype=np.float64)
    far = MAX_WORLD * max(height, 1.0)
    keep, skipped = [], 0
    with np.errstate(over="ignore", invalid="ignore"):
        for sh in d.shapes:
            subs = [np.round((sub - mid) * flip + origin, 9) for sub in sh.subpaths]
            if all(np.isfinite(a).all() and np.abs(a - origin).max() <= far
                   and np.abs(a).max() <= MAX_COORD for a in subs):  # fmt: skip
                keep.append((sh, subs))
            else:
                skipped += 1
    names = part_names([sh for sh, _ in keep], reserved)
    parts = tuple((n, tuple(tuple(map(tuple, a.tolist())) for a in subs), sh)
                  for n, (sh, subs) in zip(names, keep, strict=True))  # fmt: skip
    return parts, skipped


@register
class SVG(ObjectBase):
    """The shapes of an SVG file, as ``Path`` parts."""

    type: ClassVar[str] = "svg"
    Files: ClassVar = ("path",)
    Requires: ClassVar = extras.requires("svgelements")  # (lint missing_extra)
    Style: ClassVar = {**STYLE, "fill": ("fill", "solid")}
    Presets: ClassVar = {
        "solid": {},
        "outline": {"style": {"fill": "none"}},
        "glowing": {"style": {"glow": "soft", "fill": "soft"}},
    }

    class Params(ParamsBase):
        path: str = ""  # an .svg file in the scene's folder
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the middle of its view box goes
        height: float = Field(2.0, gt=0, le=MAX_COORD)  # world height of its view box

        @field_validator("path")
        @classmethod
        def _path(cls, v):
            return files.relative_path(v) if v else v

    def drawing(self, state: SceneState) -> pm.SvgDrawing | None:
        """The file's shapes (empty, with ``problem`` or ``limited``, when the file is bad or
        too complex), or None while it is missing or svgelements is not installed."""
        p = self.me(state).params
        if not p.path or not extras.available("svgelements"):
            return None
        stamp = files.stamp(state.base_dir, p.path)
        if stamp[1] is None:
            return None
        try:
            return pm.read_svg(stamp[0], stamp)
        except pm.MissingExtra:
            return None

    def _parts(self, state: SceneState) -> tuple[tuple[tuple[str, tuple, Any], ...], int]:
        """(name, world subpaths, shape) per shape drawn, and how many were skipped because
        a point is not finite or absurdly far out once placed. Cached by the file's stamp
        and the placement, so a new frame of an unchanged drawing does not place it again."""
        p = self.me(state).params
        if self.drawing(state) is None:
            return (), 0
        reserved = children(state.reserved, self.id)
        stamp = files.stamp(state.base_dir, p.path)
        return _placed(stamp, tuple(p.pos), p.height, reserved)

    def build(self, state: SceneState) -> list[ObjectBase]:
        me = self.me(state)
        style = {k: me.roles[k] for k in STYLE if k in me.roles}
        return [Path(name, subpaths=subs, rule=sh.rule, style=style,
                     fill_opacity=1.0 if sh.filled else 0.0,
                     stroke_opacity=1.0 if sh.stroked else 0.0)
                for name, subs, sh in self._parts(state)[0]]  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        from phasor.core.record import parts

        me = self.me(state)
        p = me.params
        names = [o.id[len(self.id) + 1 :] for o in parts(state, self.id)]
        out: dict[str, Any] = {"kind": "svg", "file": p.path, "parts": names}
        if not extras.available("svgelements"):
            out["missing_extra"] = "svg"
            return out
        d = self.drawing(state)
        if d is None:
            out["missing_file"] = p.path
            return out
        if d.problem:
            out["unreadable"] = d.problem  # lint bad_file
        if d.limited:
            out["too_complex"] = d.limited  # lint svg_too_complex
        skipped = d.skipped + self._parts(state)[1]
        if skipped:
            out["skipped"] = skipped  # lint bad_file
        return out
