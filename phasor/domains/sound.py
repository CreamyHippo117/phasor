"""Sound objects (SPEC "Sound"): invisible objects that make noise, so any scene can have sound.

    from phasor import expr, sound

    s.add(sound.Tone(id="hum", hz=expr("220 + 110 * t"), gain=expr("min(t / 2, 1)")))
    s.add(sound.Cue(id="ticks", at=(1.0, 2.5), kind="pluck", hz=440))

Both draw nothing: their semantics say ``drawn`` 0 and ``unseen`` (made to be heard), so
lint's ``orphan`` rule and the inspector's event log leave them alone. Sound is off unless
a render or export asks for it (``render --sound``, ``phasor sound``), so neither has an
on/off switch.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from phasor.core import sound as snd
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState

_ORIGIN = Bounds((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@register
class Tone(ObjectBase):
    """A continuous tone. ``hz`` (pitch) and ``gain`` (loudness, 0 is silent) are numbers that
    may be ``expr(...)`` of ``t``, so any value can be heard: a sweeping frequency, a
    converging sum. ``harmonics`` are the relative levels of the fundamental and its
    overtones. The pitch is followed on the theme's control grid; loudness is scaled by the
    theme's ``tone_gain``."""

    type: ClassVar[str] = "tone"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        hz: float = Field(gt=0)
        gain: float = Field(1.0, ge=0)
        harmonics: tuple[float, ...] = Field((1.0, 0.25), min_length=1)

    def world_bounds(self, state: SceneState) -> Bounds:
        return _ORIGIN

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {
            "kind": "tone",
            "drawn": 0.0,
            "unseen": True,
            "hz": round(p.hz, 4),
            "gain": round(p.gain, 4),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        return  # heard, not seen

    def sounds(self, ctx: snd.SoundContext) -> list[snd.Sound]:
        pick = lambda f: ctx.series(lambda s: f(s[self.id].params))  # noqa: E731
        hz = np.clip(pick(lambda p: p.hz), 1e-3, None)
        gain = np.clip(pick(lambda p: p.gain), 0.0, None) * ctx.theme["tone_gain"]
        harmonics = self.me(ctx.state_at(0.0)).params.harmonics
        return [snd.follow(ctx.rate, ctx.times(), hz, gain, harmonics)]


@register
class Cue(ObjectBase):
    """One-off sounds at the scene times in ``at`` (seconds). ``kind``: a ``click`` (a tick), a
    ``pluck`` or a ``bell``; ``hz`` is the pitch (0: the theme's ``click_hz``), ``gain`` scales
    the theme's cue loudness."""

    type: ClassVar[str] = "cue"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        at: tuple[float, ...] = Field((), max_length=10000)
        kind: Literal["click", "pluck", "bell"] = "click"
        hz: float = Field(0.0, ge=0)
        gain: float = Field(1.0, ge=0)

    def world_bounds(self, state: SceneState) -> Bounds:
        return _ORIGIN

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "cue", "drawn": 0.0, "unseen": True, "sound": p.kind, "count": len(p.at),
                "at": [round(t, 4) for t in p.at[:16]]}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        return

    def sounds(self, ctx: snd.SoundContext) -> list[snd.Sound]:
        p = self.me(ctx.state_at(0.0)).params
        th = ctx.theme
        hz = p.hz or th["click_hz"]
        make = {
            "click": lambda: snd.click(ctx.rate, hz, th["click_tau"]),
            "pluck": lambda: snd.pluck(ctx.rate, hz),
            "bell": lambda: snd.bell(ctx.rate, hz),
        }[p.kind]
        samples = make()
        loud = p.gain * (th["click_gain"] if p.kind == "click" else th["cue_gain"])
        return [snd.Sound(float(t), samples, loud, f"{p.kind} {i}") for i, t in enumerate(p.at)]
