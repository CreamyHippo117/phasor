"""Object IDs (SPEC contract 2): ``[a-z][a-z0-9_]*``, dots for children, never renumbered.

A parameter path is ``<id>.<param>``; the parameter is always the last segment, so
``res.pole_i.pos`` is parameter ``pos`` of object ``res.pole_i``.
"""

from __future__ import annotations

import re

_SEGMENT = r"[a-z][a-z0-9_]*"
ID_RE = re.compile(rf"{_SEGMENT}(\.{_SEGMENT})*")
NAME_RE = re.compile(_SEGMENT)


def is_valid_id(s: str) -> bool:
    return isinstance(s, str) and ID_RE.fullmatch(s) is not None


def is_valid_name(s: str) -> bool:
    return isinstance(s, str) and NAME_RE.fullmatch(s) is not None


def parent_of(object_id: str) -> str | None:
    """``res.pole_i`` -> ``res``; top-level ids have no parent."""
    head, _, _ = object_id.rpartition(".")
    return head or None


def split_path(path: str) -> tuple[str, str]:
    """``loop.r`` -> (``loop``, ``r``). Raises ValueError when there is no parameter part."""
    object_id, _, param = path.rpartition(".")
    if not object_id or not is_valid_id(object_id) or not is_valid_name(param):
        raise ValueError(f"not a parameter path: {path!r} (expected <id>.<param>)")
    return object_id, param
