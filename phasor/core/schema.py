"""The scene document (SPEC contract 2): pydantic v2 models, frozen, extra keys forbidden.

This module checks *shape* only (ids, uniqueness, references that need no registry).
Meaning (object types, parameter types, expressions, tracks, cycles) is checked by
``phasor.time.compile.compile_document``, which every write path runs.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainValidator, model_validator

from phasor.core.constraints import PRIORITIES, RuleError, normalize_rule, parse_rule
from phasor.core.ids import is_valid_id, is_valid_name, parent_of, split_path
from phasor.core.values import normalize

SCHEMA_VERSION = 2  # 2: added `constraints` (Milestone 2)
OVERRIDES_VERSION = 1


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


def _check_rule(rule: str) -> str:
    rule = normalize_rule(rule)
    try:
        parse_rule(rule)
    except RuleError as e:
        raise ValueError(str(e)) from e
    return rule


class ConstraintSpec(Model):
    """One layout rule, e.g. ``label(pole) near pole``. The rule text is its identity."""

    rule: Annotated[str, PlainValidator(_check_rule)]
    priority: Literal["required", "strong", "medium", "weak"] = "strong"


assert ConstraintSpec.model_fields["priority"].annotation.__args__ == PRIORITIES


def _check_unique_rules(constraints: tuple[ConstraintSpec, ...]) -> None:
    rules = [c.rule for c in constraints]
    for r in rules:
        if rules.count(r) > 1:
            raise ValueError(f"duplicate constraint {r!r}")


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
    schema_version: Literal[2] = SCHEMA_VERSION
    meta: Meta
    objects: tuple[ObjectSpec, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
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
        _check_unique_rules(self.constraints)
        return self

    def object(self, object_id: str) -> ObjectSpec | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def camera(self, camera_id: str) -> CameraSpec | None:
        return next((c for c in self.cameras if c.id == camera_id), None)


class Overrides(Model):
    """GUI/AI tweaks to a scripted scene, keyed by ``<id>.<param>`` / ``<id>.<style key>``,
    plus layout rules added on top of the script's own."""

    schema_version: Literal[1] = OVERRIDES_VERSION
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    styles: dict[str, str] = Field(default_factory=dict)
    constraints: tuple[ConstraintSpec, ...] = ()  # rules added on top of the script's own

    @model_validator(mode="after")
    def _check_shape(self) -> Overrides:
        _check_unique_rules(self.constraints)
        return self

    def is_empty(self) -> bool:
        return not self.params and not self.styles and not self.constraints


def json_schema() -> dict[str, Any]:
    """JSON Schema for the document, exported for MCP validation and docs."""
    return Document.model_json_schema()
