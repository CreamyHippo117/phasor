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


def name_from(text: str, fallback: str, max_len: int = 40) -> str:
    """A valid name made from free text (a file's shape id, a bar's category): lower case,
    other characters as ``_``, at most ``max_len`` characters; ``fallback`` when nothing is
    left, and in front when it would start with a digit (``rect_1st``)."""
    base = re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")
    if not base:
        base = fallback
    elif not base[0].isalpha():
        base = f"{fallback}_{base}"
    return base[:max_len].rstrip("_")


def unique_names(wanted, reserved=()) -> list[str]:
    """``wanted`` names, in order, each made unique and kept out of ``reserved`` by the next
    free suffix (``label``, ``label_2``)."""
    used: set[str] = set(reserved)
    out = []
    for base in wanted:
        name, n = base, 2
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)
        out.append(name)
    return out


def children(ids, parent: str) -> frozenset[str]:
    """The first names under ``parent`` among ``ids`` (``plot.label.x`` -> ``label``): what
    a composite's parts must keep clear of."""
    prefix = f"{parent}."
    return frozenset(i[len(prefix) :].split(".")[0] for i in ids if i.startswith(prefix))


STYLE = "style"  # ``<id>.style.<key>``: a track on a style key (a style transition)


def split_style_path(path: str) -> tuple[str, str] | None:
    """``dot.style.color`` -> (``dot``, ``color``); None for any other path. Whether it
    names a style key or a parameter of a child object called ``dot.style`` is decided by
    the document: a real object ``<id>.style`` keeps its own parameter paths."""
    head, _, key = path.rpartition(".")
    object_id, _, word = head.rpartition(".")
    if word != STYLE or not object_id or not is_valid_id(object_id) or not is_valid_name(key):
        return None
    return object_id, key


_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def file_stem(title: str) -> str:
    """A scene title made safe to use as a file name: letters, digits, ``_ . -`` only, no
    leading dots, never empty. Titles are free text; a file name made from one must never
    climb out of its folder (``../x``) or become absolute (``/tmp/x``)."""
    stem = _UNSAFE.sub("_", title).lstrip(".") or "scene"
    return stem[:80]
