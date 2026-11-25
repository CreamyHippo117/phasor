"""Things brought in from outside: pictures (BEAUTY.md item 5).

``Image`` puts a PNG or JPEG file in the scene, as a flat rectangle placed, sized and
animated like any object (it fades, moves, flashes, sits in 3D). The file's path is relative
to the scene's folder; the document stores the path, never the pixels, and compiling checks
the file is there (issue ``missing_file``). A picture sequence plays like a clip (Milestone
24); what it shares with ``Video`` lives in ``frames.py``, the decoding in ``media_math.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core import files
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, register
from phasor.core.state import SceneState
from phasor.domains import media_math as mm
from phasor.domains.frames import Picture, PictureParams, extension

MAX_SOURCE_PX = mm.MAX_SOURCE_PX  # larger pictures are scaled down once, when read


def picture(state: SceneState, path: str) -> np.ndarray | None:
    """The picture ``path`` names in the scene's folder (core/files.py) as linear RGBA float32
    (h, w, 4), premultiplied, row 0 = top; re-read when the file changes; None while it is
    missing or unreadable (lint ``missing_file``, ``bad_file``: draws nothing)."""
    return mm.open_source(state.base_dir, path).frame(0)


@register
class Image(Picture):
    """A picture from a file (PNG, JPEG, anything Pillow reads; a GIF's first frame),
    ``width`` world units wide (``height`` 0: keep the picture's shape), centred on ``pos``.
    Transparent parts show the scene's background colour. The ``glow`` style role makes it
    shine (bright parts most); ``frame`` draws a thin outline of light round it.

    A path with ``%04d`` in its name (``frames/shot_%04d.png``) is a picture sequence played
    at ``fps`` like a clip (``start``, ``trim``, ``speed``, ``loop``; see ``Video``). ``fit``,
    ``fill`` and ``backdrop`` place it as ``domains/frames.py`` says."""

    type: ClassVar[str] = "image"
    kind: ClassVar[str] = "image"
    Files: ClassVar = ("path",)  # in the scene's folder (core/files.py); missing: lint

    class Params(PictureParams):
        path: str = Field(min_length=1)  # relative to the scene's folder, and inside it
        fps: float = Field(24.0, gt=0, le=mm.MAX_FPS)  # a picture sequence's frames a second

        @field_validator("path")
        @classmethod
        def _image_file(cls, v: str) -> str:
            if extension(v) not in mm.PICTURE_EXT:
                raise ValueError(f"{v!r} is not a picture file (png, jpg, webp, ...)")
            return files.relative_path(v)

    def source(self, state: SceneState) -> mm.Source:
        p = self.me(state).params
        return mm.open_source(state.base_dir, p.path, fps=p.fps)


# ---------------------------------------------------------------- sound clips

MIN_GAIN, MAX_GAIN = -60.0, 24.0  # dB
MAX_START = 1e6  # seconds


@register
class Sound(ObjectBase):
    """A sound clip (Manim's ``add_sound``): a sound file in the scene's folder that plays
    from ``start`` (seconds, or a marker's name: it then follows the marker, Milestone 25),
    ``gain`` dB louder (negative: quieter). Nothing is drawn:
    ``phasor render`` mixes it into the video's sound, next to the narration, cut where the
    scene ends. A WAV is read as it is (PCM, 8 to 32-bit, 8 to 192 kHz, 1 to 16 channels);
    any other sound file (the sound of an ``.mp4``, ``.m4a``, ``.mp3``, ``.ogg``, ``.flac``,
    M22) is decoded once with PyAV, the ``encode`` extra (``domains.audio``). A missing file
    is lint ``missing_file``; one that cannot be read (an mp4 with no sound) is ``bad_file``;
    a non-WAV file without PyAV is ``missing_extra``; the render leaves such clips out and
    says so."""

    type: ClassVar[str] = "sound"
    Files: ClassVar = ("path",)  # in the scene's folder (core/files.py)
    Life: ClassVar = frozenset()  # a time and a level: life never touches them

    class Params(ParamsBase):
        path: str = Field(min_length=1)  # relative to the scene's folder, and inside it
        start: float | str = 0.0  # seconds into the scene, or a marker's name
        gain: float = Field(0.0, ge=MIN_GAIN, le=MAX_GAIN)  # dB

        @field_validator("start", mode="before")
        @classmethod
        def _start(cls, v: Any) -> float | str:
            import math

            from phasor.core.ids import is_valid_name

            if isinstance(v, str):
                if is_valid_name(v):  # a marker's name first ("inf" and "nan" are names too)
                    return v
                try:
                    v = float(v)  # "1.5": seconds, as a float field always took it
                except ValueError:
                    raise ValueError(f"start {v!r}: seconds, or a marker's name") from None
            if isinstance(v, bool) or not isinstance(v, int | float):
                raise ValueError(f"start {v!r}: seconds, or a marker's name")
            if not (math.isfinite(v) and 0 <= v <= MAX_START):
                raise ValueError(f"start {v} s (0 to {MAX_START:g})")
            return float(v)

        @field_validator("path")
        @classmethod
        def _sound_file(cls, v: str) -> str:
            from phasor.domains.audio import AUDIO_EXTS

            if not v.lower().endswith(AUDIO_EXTS):
                raise ValueError(f"{v!r} is not a sound file ({', '.join(AUDIO_EXTS)})")
            return files.relative_path(v)

    def _header(self, state: SceneState):
        """What the file holds once read (a ``WavInfo``), a reason it cannot be mixed (str),
        or None (missing); cached per file version (``domains.audio.info``)."""
        from phasor.domains.audio import info

        return info(state.base_dir, self.me(state).params.path)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        from phasor.domains.audio import decoder_available, needs_decoder

        out: dict[str, Any] = {"kind": "sound", "path": p.path, "start": p.start,
                               "gain": p.gain, "missing": False, "drawn": 0.0,
                               "file": p.path}  # fmt: skip
        if needs_decoder(p.path) and not decoder_available():
            out["missing"] = not files.exists(state.base_dir, p.path)
            out["needs"] = "av"  # lint missing_extra: not heard until PyAV is installed
            return out
        head = self._header(state)
        out["missing"] = head is None
        if isinstance(head, str):
            out["unreadable"] = head
        elif head is not None:
            out["seconds"] = round(head.seconds, 6)
            out["rate"] = head.rate
        return out

    def world_bounds(self, state: SceneState) -> Bounds:
        return Bounds((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def audio_clip(self, state: SceneState):
        """What ``Engine.audio_track`` mixes: this clip (``time.narration.Clip``)."""
        from phasor.time.narration import Clip

        p = self.me(state).params
        if isinstance(p.start, str):  # at a marker: the engine places it (``_at_markers``)
            return Clip(path=p.path, start=0.0, gain=p.gain, marker=p.start)
        return Clip(path=p.path, start=p.start, gain=p.gain)

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        """Nothing: a sound is heard, not seen."""
