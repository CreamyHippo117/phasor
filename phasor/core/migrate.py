"""Schema migrations. Each step upgrades a raw dict by exactly one version."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from phasor.core.issues import DocumentError, Issue
from phasor.core.schema import SCHEMA_VERSION

# from_version -> function returning the document at from_version + 1
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


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
