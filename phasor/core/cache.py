"""Caches bounded by the memory they hold (least recently used out).

``nbytes`` measures a value by its arrays, however deep they sit in tuples, lists, dicts and
dataclasses (a morph plan, a recording); ``ByteCache`` keeps values while their sizes fit in
a budget, and at most ``max_entries`` of them. The path geometry cache, the morph plans and
the recordings of ``draw`` each have one (Milestone 17 independent audit: a plan was counted
as 64 bytes whatever its size, so a moving morph grew memory without bound).
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

ENTRY = 256  # bytes counted for each entry besides its value (key, bookkeeping)
CONTAINER = 64  # bytes counted for each container (tuple, list, dict, dataclass)


def nbytes(value: Any) -> int:
    """About how much memory a value holds: its arrays, plus a little per container."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, tuple | list):
        return CONTAINER + sum(nbytes(v) for v in value)
    if isinstance(value, dict):
        return CONTAINER + sum(nbytes(v) for v in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return CONTAINER + sum(nbytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    return CONTAINER


class ByteCache:
    """Values kept while they fit in ``budget()`` bytes and ``max_entries`` entries, safe
    across threads. ``budget`` is read at each store, so a test can lower the module
    constant it names."""

    def __init__(self, budget: Callable[[], int], max_entries: int | None = None) -> None:
        self.budget = budget
        self.max_entries = max_entries
        self.entries: OrderedDict[Any, tuple[Any, int]] = OrderedDict()  # key -> (value, bytes)
        self.used = 0
        self.lock = threading.Lock()

    def peek(self, key: Any, default: Any = None) -> Any:
        """The value at ``key`` (marked as just used), or ``default``."""
        with self.lock:
            hit = self.entries.get(key)
            if hit is None:
                return default
            self.entries.move_to_end(key)
            return hit[0]

    def put(self, key: Any, value: Any) -> None:
        size = ENTRY + nbytes(value)
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None:
                self.used -= old[1]
            self.entries[key] = (value, size)
            self.used += size
            budget, most = self.budget(), self.max_entries
            while len(self.entries) > 1 and (
                self.used > budget or (most is not None and len(self.entries) > most)
            ):
                self.used -= self.entries.popitem(last=False)[1][1]

    def get(self, key: Any, make: Callable[[], Any]) -> Any:
        """The value at ``key``, made (outside the lock) and kept when it is not there."""
        missing = object()
        hit = self.peek(key, missing)
        if hit is not missing:
            return hit
        value = make()
        self.put(key, value)
        return value
