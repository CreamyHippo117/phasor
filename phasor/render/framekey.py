"""A digest of everything a frame's drawing depends on (Milestone 21: frame de-duplication).

Two frames with the same digest draw the same pixels, so a render may repeat the first
instead of drawing the second. The digest covers values exactly (bytes of every number), so
it errs only one way: something it cannot read exactly (an unknown object) gets a token
that never repeats, and the frame is drawn. Never the other way round.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel

_UNIQUE = itertools.count()  # unknown values: a token no other value ever gets


def _feed(h: Any, x: Any) -> None:
    u = h.update
    if x is None or isinstance(x, bool):
        u(b"N" if x is None else (b"T" if x else b"F"))
    elif isinstance(x, int):
        u(b"i" + str(x).encode() + b";")
    elif isinstance(x, float):
        u(b"f" + struct.pack("<d", x))
    elif isinstance(x, str):
        b = x.encode("utf-8", "surrogatepass")
        u(b"s" + str(len(b)).encode() + b":" + b)
    elif isinstance(x, np.ndarray):
        if x.dtype == object:
            u(b"O" + str(x.shape).encode())
            for item in x.ravel():
                _feed(h, item)
        else:
            u(b"a" + x.dtype.str.encode() + str(x.shape).encode())
            u(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, np.generic):
        _feed(h, x.item())
    elif isinstance(x, (tuple, list)):
        u(b"(" + str(len(x)).encode() + b";")
        for item in x:
            _feed(h, item)
        u(b")")
    elif isinstance(x, Mapping):
        items = sorted(x.items(), key=lambda kv: repr(kv[0]))
        u(b"{" + str(len(items)).encode() + b";")
        for k, v in items:
            _feed(h, k)
            _feed(h, v)
        u(b"}")
    elif isinstance(x, (set, frozenset)):
        _feed(h, sorted(x, key=repr))
    elif isinstance(x, BaseModel):
        u(b"M" + type(x).__qualname__.encode() + b";")
        for name in type(x).model_fields:
            _feed(h, name)
            _feed(h, getattr(x, name))
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        u(b"D" + type(x).__qualname__.encode() + b";")
        for f in dataclasses.fields(x):
            _feed(h, f.name)
            _feed(h, getattr(x, f.name))
    else:  # a callable, an object of unknown make: never equal to anything else
        u(b"?" + str(next(_UNIQUE)).encode() + b";")


def digest(*parts: Any) -> bytes:
    """The digest of ``parts`` (numbers, strings, arrays, sequences, mappings, pydantic models
    and dataclasses, nested). Equal values give equal digests in one process."""
    h = hashlib.blake2b(digest_size=32)
    for p in parts:
        _feed(h, p)
    return h.digest()
