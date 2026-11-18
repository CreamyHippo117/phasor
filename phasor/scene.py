"""The Python authoring API: a thin builder over ``Engine`` (SPEC contract 1).

A script builds the base document. Every call here is one ``Engine.apply`` with
``origin="script"``; an invalid call raises at the line that made it.

    s = Scene("residues")
    s.narrate(SCRIPT)                                  # markers come from the voice
    s.at("poles_appear").reveal("pole_a", "pole_b")    # animate by verb, at a word
    s.at("loop_draw").trace("loop").then().reveal("loop.label")
    s.pace()                                           # breathing room after busy sentences
"""

from __future__ import annotations

import inspect
import runpy
from pathlib import Path
from typing import Any

from phasor.core import io
from phasor.core.objects import ObjectBase
from phasor.core.patch import (
    AddConstraint,
    AddObject,
    AddSegment,
    SetCamera,
    SetMarker,
    SetNarration,
    SetParam,
)
from phasor.core.schema import Document, ExprSpec, MarkerRef, Meta, Segment
from phasor.engine import Engine
from phasor.time import verbs as _verbs
from phasor.time.narration import block_of, repace


def expr(source: str) -> ExprSpec:
    """A parameter computed from other parameters and ``t``, e.g. ``expr("2 * dot.r + sin(t)")``."""
    return ExprSpec(expr=source)


class At:
    """A point on the timeline, returned by ``Scene.at``. Verbs and ``to`` start here."""

    def __init__(self, scene: Scene, when: str | float, offset: float) -> None:
        self._scene, self._when, self._offset = scene, when, offset
        self._end = 0.0  # where the last verb started from here ended, for ``then``

    def _ref(self, extra: float) -> float | MarkerRef:
        if isinstance(self._when, str):
            return MarkerRef(marker=self._when, offset=self._offset + extra)
        return float(self._when) + self._offset + extra

    def to(self, path: str, value: Any, *, duration: float = 1.0, ease: str = "smooth") -> At:
        """Move parameter ``path`` to ``value`` over ``duration`` seconds, starting here.
        A raw transition: prefer a verb, whose timing comes from the theme."""
        seg = Segment(t0=self._ref(0.0), t1=self._ref(duration), to=value, ease=ease)
        self._scene._apply(AddSegment(path=path, segment=seg))
        self._end = max(self._end, duration)
        return self

    def set(self, path: str, value: Any) -> At:
        """Jump parameter ``path`` to ``value`` at this instant."""
        return self.to(path, value, duration=0.0, ease="linear")

    def then(self, gap: float = 0.0) -> At:
        """The moment everything started here has finished (plus ``gap`` seconds)."""
        return At(self._scene, self._when, self._offset + self._end + gap)

    def _verb(self, name: str, *args: Any, **kwargs: Any) -> At:
        engine = self._scene.engine
        at = self._ref(0.0)
        compiled = engine._snap.compiled
        try:
            t = compiled.time_map.resolve(at)
        except KeyError as e:
            raise _verbs.VerbError(f"no marker {e.args[0]!r}; markers are "
                                   f"{sorted(compiled.time_map.markers)}") from None  # fmt: skip
        motion = getattr(_verbs.Verbs(compiled, engine.state_at(t), at), name)(*args, **kwargs)
        if motion.ops:
            self._scene._apply(list(motion.ops))
        self._end = max(self._end, motion.end)
        return self

    def reveal(self, *ids: str) -> At:
        """Fade objects in (hidden until now), staggered."""
        return self._verb("reveal", *ids)

    def exit(self, *ids: str) -> At:
        """Fade objects out."""
        return self._verb("exit", *ids)

    def trace(self, *ids: str) -> At:
        """Draw curves from start to end (nothing drawn until now)."""
        return self._verb("trace", *ids)

    def bloom_in(self, *ids: str) -> At:
        """Fade in while growing from a point, with a small overshoot."""
        return self._verb("bloom_in", *ids)

    def dim_others(self, *keep: str) -> At:
        """Dim everything but ``keep`` (backdrops stay)."""
        return self._verb("dim_others", *keep)

    def undim(self) -> At:
        """Bring dimmed objects back."""
        return self._verb("undim")

    def focus_pull(self, object_id: str, margin: float = _verbs.FOCUS_MARGIN) -> At:
        """Move the camera onto ``object_id``."""
        return self._verb("focus_pull", object_id, margin)

    def morph(self, path: str, value: Any) -> At:
        """Change any parameter smoothly, with the theme's morph timing."""
        return self._verb("morph", path, value)

    def settle(self, path: str, value: Any) -> At:
        """Move a parameter like a released spring."""
        return self._verb("settle", path, value)

    def orbit(self, degrees: float = 360.0, until: str | None = None) -> At:
        """Swing the 3D camera ``degrees`` around its target, over the theme's time per turn
        or until the marker ``until``."""
        return self._verb("orbit", degrees, until)

    def orbit_reveal(self, *ids: str) -> At:
        """Reveal objects while the 3D camera swings a little around them."""
        return self._verb("orbit_reveal", *ids)


def _caller_dir() -> Path:
    """The folder of the script that called ``Scene(...)``, or the working folder."""
    for frame in inspect.stack()[2:]:
        path = Path(frame.filename)
        if path.suffix == ".py" and path.exists() and "phasor" not in path.parts[-2:-1]:
            return path.parent
    return Path.cwd()


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
        engine_kwargs.setdefault("base_dir", _caller_dir())
        self.engine = Engine(Document(meta=meta), **engine_kwargs)

    @classmethod
    def of(cls, engine: Engine) -> Scene:
        """A builder around an engine that already exists (the MCP server uses this, so a
        client gets the same verbs as a script)."""
        scene = cls.__new__(cls)
        scene.engine = engine
        return scene

    def _apply(self, op: Any) -> None:
        self.engine.apply(op, origin="script").raise_for_errors()

    def add(self, obj: ObjectBase) -> ObjectBase:
        """Add an object: ``s.add(Dot(id="dot", pos=(1, 2)))``. Returns it for chaining."""
        self._apply(AddObject(object=obj.to_spec()))
        return obj

    def use_camera(self, kind: str = "orbit", **params: Any) -> None:
        """Choose the camera: ``"orbit"`` (3D perspective around ``target``; ``azimuth``,
        ``elevation``, ``distance``, ``fov``, ``drift``) or ``"ortho"`` (2D, the default).
        Parameters you leave out are free for the camera solver when you give it goals
        (``s.constrain("camera keep_visible(surface)")``)."""
        self._apply(SetCamera(kind=kind, params=params))

    def set(self, path: str, value: Any) -> None:
        """Set a parameter (objects and cameras): ``s.set("camera.height", 6)``."""
        self._apply(SetParam(path=path, value=value))

    def constrain(self, rule: str, *, priority: str = "strong") -> None:
        """Add a layout rule: ``s.constrain("label(pole) near pole")``. The engine solves it
        at every key time; a rule it cannot satisfy shows up in ``engine.lint()``."""
        self._apply(AddConstraint(rule=rule, priority=priority))

    def narrate(self, script: str, *, voice: str = "auto") -> None:
        """Speak ``script`` and take its ``[[markers]]`` as the timeline (SPEC "Narration").

        ``voice``: ``auto`` (espeak-ng when installed, else estimated timing), ``estimate``,
        ``espeak[:lang[:wpm]]`` or ``piper:<model.onnx>``. Audio clips are cached in
        ``<title>.narration/`` beside the script. Calling it again (another voice, an edited
        script) moves every marker, and everything anchored to them follows.
        """
        from phasor.voice import narrate

        folder = self.engine.base_dir / f"{self.engine.document.meta.title}.narration"
        theme = self.engine._snap.compiled.theme
        self._apply(SetNarration(narration=narrate(script, voice, folder, theme)))

    def pace(self) -> None:
        """Give each sentence breathing room in proportion to how much it changes on screen:
        after a sentence whose transitions touch N objects, wait N x ``pause_per_change``
        seconds more (at most ``pause_max``; theme ``[narration]``). Run it last; running it
        again changes nothing."""
        doc = self.engine.document
        if doc.narration is None:
            raise ValueError("pace() needs narration: call narrate() first")
        compiled = self.engine._snap.compiled
        busy: dict[int, set[str]] = {}
        for path, segs in doc.timeline.tracks.items():
            for seg in segs:
                k = block_of(doc.narration, compiled.time_map.resolve(seg.t0))
                if k is not None:
                    busy.setdefault(k, set()).add(path.rpartition(".")[0])
        n = compiled.theme.narration
        extra = [min(len(busy.get(k, ())) * n["pause_per_change"], n["pause_max"])
                 for k in range(len(doc.narration.blocks))]  # fmt: skip
        self._apply(SetNarration(narration=repace(doc.narration, extra, n["gap"], n["lead"])))

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
    engine_kwargs.setdefault("base_dir", path.parent)
    return Engine(scenes[0].engine.base, overrides=overrides, **engine_kwargs)
