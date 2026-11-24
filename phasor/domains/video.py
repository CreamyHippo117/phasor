"""Footage in a scene (Milestone 24): ``Video`` plays a clip from a file in the scene's folder.

The clip is an mp4 (or mov, m4v, webm, mkv: whatever PyAV, the ``encode`` extra, decodes),
placed, sized and animated like ``Image`` (``domains/frames.py``), and played from ``start``
seconds into the scene: the part ``trim`` keeps, at ``speed``, looping or holding its last
frame. The frame shown at t is the last one whose presentation time is at or before the
clip time t maps to, so the same t always shows the same frame, in any order and in any
render worker (each opens the file itself). Sound in the file is not played (``Sound``).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, field_validator

from phasor.core import extras, files
from phasor.core.objects import register
from phasor.core.state import SceneState
from phasor.domains import media_math as mm
from phasor.domains.frames import Picture, PictureParams, extension


@register
class Video(Picture):
    """A video clip from a file (``.mp4``, ``.mov``, ``.m4v``, ``.webm``, ``.mkv``) in the
    scene's folder, ``width`` world units wide (``height`` 0: keep the clip's shape), centred
    on ``pos``; ``fit``, ``fill`` and ``backdrop`` as for ``Image``. It plays from ``start``
    (scene seconds), the part ``trim`` = (in, out) keeps (out 0: the end), at ``speed``
    (0-64 times), looping when ``loop``, else holding its last frame (lint ``clip_ended``).
    Before ``start`` it shows its first frame. A missing file is ``missing_file``, one that is
    not a video ``bad_file``, one past a size cap ``video_too_big``; each draws nothing."""

    type: ClassVar[str] = "video"
    kind: ClassVar[str] = "video"
    Files: ClassVar = ("path",)  # in the scene's folder (core/files.py); missing: lint
    Requires: ClassVar = extras.requires("av")

    class Params(PictureParams):
        path: str = Field(min_length=1)  # relative to the scene's folder, and inside it

        @field_validator("path")
        @classmethod
        def _video_file(cls, v: str) -> str:
            if extension(v) not in mm.VIDEO_EXT:
                raise ValueError(f"{v!r} is not a video file (mp4, mov, m4v, webm, mkv)")
            return files.relative_path(v)

    def source(self, state: SceneState) -> mm.Source:
        return mm.open_source(state.base_dir, self.me(state).params.path, video=True)
