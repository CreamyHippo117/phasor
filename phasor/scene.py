"""The Python authoring API: a thin builder over ``Engine`` (SPEC contract 1).

A script builds the base document. Every call here is one ``Engine.apply`` with
``origin="script"``; an invalid call raises at the line that made it.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from phasor.core import io
from phasor.core.objects import ObjectBase
from phasor.core.patch import AddConstraint, AddObject, AddSegment, SetMarker, SetParam
from phasor.core.schema import Document, ExprSpec, MarkerRef, Meta, Segment
from phasor.engine import Engine


def expr(source: str) -> ExprSpec:
    """A parameter computed from other parameters and ``t``, e.g. ``expr("2 * dot.r + sin(t)")``."""
    return ExprSpec(expr=source)


class At:
    """A point on the timeline, returned by ``Scene.at``."""

    def __init__(self, scene: Scene, when: str | float, offset: float) -> None:
        self._scene, self._when, self._offset = scene, when, offset

    def _ref(self, extra: float) -> float | MarkerRef:
        if isinstance(self._when, str):
            return MarkerRef(marker=self._when, offset=self._offset + extra)
        return float(self._when) + self._offset + extra

    def to(self, path: str, value: Any, *, duration: float = 1.0, ease: str = "smooth") -> At:
        """Move parameter ``path`` to ``value`` over ``duration`` seconds, starting here."""
        seg = Segment(t0=self._ref(0.0), t1=self._ref(duration), to=value, ease=ease)
        self._scene._apply(AddSegment(path=path, segment=seg))
        return self

    def set(self, path: str, value: Any) -> At:
        """Jump parameter ``path`` to ``value`` at this instant."""
        return self.to(path, value, duration=0.0, ease="linear")


class Scene:
    def __init__(
        self,
        title: str,
        *,
        theme: str = "midnight_glow",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        **engine_kwargs: Any,
    ) -> None:
        meta = Meta(title=title, theme=theme, width=width, height=height, fps=fps, source="script")
        self.engine = Engine(Document(meta=meta), **engine_kwargs)

    def _apply(self, op: Any) -> None:
        self.engine.apply(op, origin="script").raise_for_errors()

    def add(self, obj: ObjectBase) -> ObjectBase:
        """Add an object: ``s.add(Dot(id="dot", pos=(1, 2)))``. Returns it for chaining."""
        self._apply(AddObject(object=obj.to_spec()))
        return obj

    def set(self, path: str, value: Any) -> None:
        """Set a parameter (objects and cameras): ``s.set("camera.height", 6)``."""
        self._apply(SetParam(path=path, value=value))

    def constrain(self, rule: str, *, priority: str = "strong") -> None:
        """Add a layout rule: ``s.constrain("label(pole) near pole")``. The engine solves it
        at every key time; a rule it cannot satisfy shows up in ``engine.lint()``."""
        self._apply(AddConstraint(rule=rule, priority=priority))

    def marker(self, name: str, t: float) -> None:
        """Place a narration marker at ``t`` seconds."""
        self._apply(SetMarker(name=name, t=t))

    def at(self, when: str | float, offset: float = 0.0) -> At:
        """A time to animate from: a marker name or seconds, plus an optional offset."""
        return At(self, when, offset)

    def save(self, path: str | Path) -> None:
        self.engine.save(path)


def load(path: str | Path, **engine_kwargs: Any) -> Engine:
    """Open a scene for any client: a ``.phasor.json`` document, or a ``.py`` script that
    builds a ``Scene``. Tweaks in ``name.overrides.json`` beside it are applied on top."""
    path = Path(path)
    if path.suffix != ".py":
        return Engine.open(path, **engine_kwargs)
    scenes = [v for v in runpy.run_path(str(path), run_name="__phasor__").values()
              if isinstance(v, Scene)]  # fmt: skip
    if len(scenes) != 1:
        raise ValueError(
            f"{path} must define exactly one Scene at module level, found {len(scenes)}"
        )
    overrides = io.load_overrides(path.with_name(path.stem + io.OVERRIDES_SUFFIX))
    return Engine(scenes[0].engine.base, overrides=overrides, **engine_kwargs)
