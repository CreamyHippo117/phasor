"""Optional packages some object types need (Milestone 16), and the extra that installs each.

The one place that says whether one can be imported: the types that need one (their
``Requires``), lint ``missing_extra`` and ``phasor doctor`` all ask here.
"""

from __future__ import annotations

# module -> (the extra that installs it, what it is for)
EXTRAS: dict[str, tuple[str, str]] = {
    "shapely": ("boolean", "boolean operations on shapes"),
    "svgelements": ("svg", "SVG drawings"),
    "pygments": ("code", "highlighting code"),
    "av": (
        "encode",
        "video files, and sound from mp4, m4a, mp3, ogg and flac files (WAV needs nothing)",
    ),
}


def available(module: str) -> bool:
    """Whether an optional package can be imported (without importing it)."""
    import importlib.util
    import sys

    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def requires(*modules: str) -> dict[str, str]:
    """A type's ``Requires``: each module it needs -> the extra that installs it."""
    return {m: EXTRAS[m][0] for m in modules}
