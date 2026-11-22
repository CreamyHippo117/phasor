"""Digital filters on the z-plane (ANIMATIONS.md S3).

    from phasor import filt

    s.add(filt.PoleZero(id="lp", kind="lowpass", cutoff=0.1, q=0.8))
    s.add(filt.PoleZero(id="mine", kind="custom", poles=((0.5, 0.5), (0.5, -0.5))))

``PoleZero`` draws the unit circle, the poles as crosses and the zeros as rings. A named
``kind`` (``lowpass``, ``highpass``, ``bandpass``, ``notch``) is a second-order section from
``cutoff`` (a share of the sampling rate, 0 to 0.5) and ``q``; ``custom`` takes ``poles`` and
``zeros`` as (re, im) points, and may put them anywhere, outside the unit circle too: that
filter is unstable and the linter says so (``unstable_filter``). One unit of the plane is
``radius`` world units. Numerics are in ``filters_math``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import filters_math as fm
from phasor.domains.strokes import circle, disc_batch, polyline, with_alpha


@register
class PoleZero(ObjectBase):
    """A filter's poles and zeros on the z-plane, inside the unit circle. A pole outside it
    (only ``kind="custom"`` can) is drawn in the danger colour."""

    type: ClassVar[str] = "pole_zero"
    Style: ClassVar = {
        "pole": ("emphasis", "primary"),
        "zero": ("emphasis", "secondary"),
        "unstable": ("emphasis", "danger"),
        "circle": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        kind: Literal["custom", "lowpass", "highpass", "bandpass", "notch"] = "custom"
        poles: tuple[Vec3, ...] = ()  # (re, im), for kind="custom"
        zeros: tuple[Vec3, ...] = ()
        cutoff: float = Field(0.1, gt=0, lt=0.5)  # share of the sampling rate
        q: float = Field(0.707, gt=0)
        gain: float = 1.0
        pos: Vec3 = (0.0, 0.0, 0.0)
        radius: float = Field(2.0, gt=0)  # world units per unit of the plane

        @model_validator(mode="after")
        def _named_or_custom(self):
            if self.kind != "custom" and (self.poles or self.zeros):
                raise ValueError(f"kind={self.kind!r} sets its own poles and zeros; "
                                 "use kind='custom' to give them")  # fmt: skip
            return self

    def _pz(self, p) -> tuple[np.ndarray, np.ndarray]:
        if p.kind == "custom":
            to_c = lambda pts: np.array([complex(x, y) for x, y, _ in pts], dtype=np.complex128)  # noqa: E731
            return to_c(p.zeros), to_c(p.poles)
        return fm.biquad(p.kind, p.cutoff, p.q)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        zs, ps = self._pz(p)
        r = max([1.0, *np.abs(zs), *np.abs(ps)]) * p.radius * 1.1
        c = np.asarray(p.pos, dtype=np.float64)
        return Bounds(tuple(c - np.array([r, r, 0.0])), tuple(c + np.array([r, r, 0.0])))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        zs, ps = self._pz(p)
        o = np.asarray(p.pos, dtype=np.float64)
        pts = [o + np.r_[z.real, z.imag, 0.0] * p.radius for z in (*ps, *zs)]
        return {"center": o, "points": np.array(pts, dtype=np.float64).reshape(-1, 3)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        zs, ps = self._pz(p)
        pair = lambda a: [[round(float(z.real), 6), round(float(z.imag), 6)] for z in a]  # noqa: E731
        return {
            "kind": "pole_zero",
            "filter": p.kind,
            "poles": pair(ps),
            "zeros": pair(zs),
            "max_pole_radius": round(fm.max_pole_radius(ps), 6),
            "stable": fm.is_stable(ps),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        o = np.asarray(p.pos, dtype=np.float64)
        unit = float(cam.units_per_height(o))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        zs, ps = self._pz(p)
        rim = with_alpha(me.style["circle"], p.opacity * 0.7)
        circle(ctx, cam, o, p.radius, w * 0.6, color=rim, glow=glow)
        at = lambda z: o + np.r_[z.real, z.imag, 0.0] * p.radius  # noqa: E731
        for z in zs:
            circle(ctx, cam, at(z), unit * 0.012, w, color=with_alpha(me.style["zero"], p.opacity),
                   glow=glow)  # fmt: skip
        arm = unit * 0.014
        for z in ps:
            role = "pole" if abs(z) < 1.0 else "unstable"
            col = with_alpha(me.style[role], p.opacity)
            c = at(z)
            for d in (cam.right + cam.up, cam.right - cam.up):
                polyline(ctx, cam, [c - d * arm, c + d * arm], w, color=col, glow=glow)
        if len(ps):
            disc_batch(ctx, cam, np.array([at(z) for z in ps]), unit * 0.003,
                       colors=with_alpha(me.style["pole"], p.opacity), glow=glow)  # fmt: skip
