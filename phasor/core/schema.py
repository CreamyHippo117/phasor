"""The scene document (SPEC contract 2): pydantic v2 models, frozen, extra keys forbidden.

This module checks *shape* only (ids, uniqueness, references that need no registry).
Meaning (object types, parameter types, expressions, tracks, cycles) is checked by
``phasor.time.compile.compile_document``, which every write path runs.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainValidator, model_validator

from phasor.core.ids import is_valid_id, is_valid_name, parent_of, split_path
from phasor.core.values import normalize

SCHEMA_VERSION = 1


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExprSpec(Model):
    """A parameter computed from ``t`` and other parameters, e.g. ``dot.pos + vec(0, 0.4, 0)``."""

    expr: str


def _parse_param(v: Any) -> Any:
    if isinstance(v, ExprSpec):
        return v
    if isinstance(v, dict):
        return ExprSpec.model_validate(v)
    return normalize(v)


# A parameter is a constant value or {"expr": "..."}. Animation lives in timeline.tracks.
ParamSpec = Annotated[Any, PlainValidator(_parse_param)]
ConstValue = Annotated[Any, PlainValidator(normalize)]


class MarkerRef(Model):
    """A time anchored to a narration marker; re-recording moves the marker, not the track."""

    marker: str
    offset: float = 0.0


TimeRef = float | MarkerRef


class Segment(Model):
    """Move a parameter to ``to`` between ``t0`` and ``t1``. It holds its value in between."""

    t0: TimeRef
    t1: TimeRef
    to: ConstValue
    ease: str = "smooth"


class ObjectSpec(Model):
    id: str
    type: str
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    style: dict[str, str] = Field(default_factory=dict)  # style key -> role name
    tags: tuple[str, ...] = ()
    layer: Literal["world", "overlay"] = "world"


class CameraSpec(Model):
    id: str
    kind: Literal["ortho"] = "ortho"
    params: dict[str, ParamSpec] = Field(default_factory=dict)


class Timeline(Model):
    markers: dict[str, float] = Field(default_factory=dict)  # the TimeMap (contract 3)
    tracks: dict[str, tuple[Segment, ...]] = Field(default_factory=dict)  # "<id>.<param>" -> segs
    camera: str = "camera"


class Meta(Model):
    title: str
    width: int = Field(1920, gt=0)
    height: int = Field(1080, gt=0)
    fps: float = Field(60.0, gt=0)
    theme: str = "midnight_glow"
    color_space: Literal["srgb"] = "srgb"
    duration: float | None = Field(None, ge=0)  # None: derived from tracks and markers
    # "script": built by a Python script; later tweaks go to name.overrides.json.
    source: Literal["document", "script"] = "document"


class Document(Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    meta: Meta
    objects: tuple[ObjectSpec, ...] = ()
    cameras: tuple[CameraSpec, ...] = (CameraSpec(id="camera"),)
    timeline: Timeline = Field(default_factory=Timeline)

    @model_validator(mode="after")
    def _check_shape(self) -> Document:
        seen: set[str] = set()
        for thing in (*self.objects, *self.cameras):
            if not is_valid_id(thing.id):
                raise ValueError(f"invalid id {thing.id!r}: must match [a-z][a-z0-9_]* with dots")
            if thing.id in seen:
                raise ValueError(f"duplicate id {thing.id!r}")
            seen.add(thing.id)
        object_ids = {o.id for o in self.objects}
        for o in self.objects:
            parent = parent_of(o.id)
            if parent is not None and parent not in object_ids:
                raise ValueError(f"object {o.id!r} needs its parent {parent!r} to exist")
        if self.timeline.camera not in {c.id for c in self.cameras}:
            raise ValueError(f"timeline.camera {self.timeline.camera!r} is not a camera id")
        for name in self.timeline.markers:
            if not is_valid_name(name):
                raise ValueError(f"invalid marker name {name!r}")
        for path in self.timeline.tracks:
            split_path(path)
        return self

    def object(self, object_id: str) -> ObjectSpec | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def camera(self, camera_id: str) -> CameraSpec | None:
        return next((c for c in self.cameras if c.id == camera_id), None)


class Overrides(Model):
    """GUI/AI tweaks to a scripted scene, keyed by ``<id>.<param>`` / ``<id>.<style key>``."""

    schema_version: Literal[1] = SCHEMA_VERSION
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    styles: dict[str, str] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.params and not self.styles


def json_schema() -> dict[str, Any]:
    """JSON Schema for the document, exported for MCP validation and docs."""
    return Document.model_json_schema()
