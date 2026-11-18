"""Schema migrations. Each step upgrades a raw dict by exactly one version."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from phasor.core.issues import DocumentError, Issue
from phasor.core.schema import SCHEMA_VERSION


# from_version -> function returning the document at from_version + 1
def _v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 2 added ``constraints`` (layout rules)."""
    raw.setdefault("constraints", [])
    return raw


def _v2_to_v3(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 3 added ``retired_ids`` (ids of removed objects, never to be reused)."""
    raw.setdefault("retired_ids", [])
    return raw


def _v3_to_v4(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 4 added ``narration`` and ``Segment.verb``; both default to none."""
    raw.setdefault("narration", None)
    return raw


MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _v1_to_v2,
    2: _v2_to_v3,
    3: _v3_to_v4,
}


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    version = raw.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise DocumentError(Issue("schema_version", f"missing or bad schema_version: {version!r}"))
    if version > SCHEMA_VERSION:
        raise DocumentError(
            Issue(
                "schema_version",
                f"document is version {version}, this phasor reads up to {SCHEMA_VERSION}",
                fix_hint="upgrade phasor",
            )
        )
    while version < SCHEMA_VERSION:
        raw = MIGRATIONS[version](dict(raw))
        version += 1
        raw["schema_version"] = version
    return raw
