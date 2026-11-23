"""The person's helper module, beside their scene script (imported, so "imported code")."""

import pathlib
from typing import ClassVar

import numpy as np

from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3

MARK = pathlib.Path(__file__).parent / "ran_gemlib_draw.txt"


@register
class Gem2(ObjectBase):
    type: ClassVar[str] = "gem2"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def draw(self, ctx, state):
        MARK.write_text("drawn for a document that may not run scripts", encoding="utf-8")
        x, y, z = self.me(state).params.pos
        tri = np.array([[x, y, z], [x + 1, y, z], [x, y + 1, z]])
        ctx.triangles(tri, np.array([[0, 1, 2]]), color=(1, 1, 1, 1))
