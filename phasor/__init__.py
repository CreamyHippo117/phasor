"""phasor: GPU-rendered math animations that an AI can see, check and fix.

This module stays import-light on purpose: importing ``phasor`` must never pull in
CuPy, ModernGL or PyAV, and must never import a layer as a side effect (that would
hide layer violations from import-linter and create import cycles). Public names
are added here lazily as milestones land.
"""

from typing import Any

__version__ = "0.0.1"

# name -> module that defines it; resolved on first access.
_LAZY = {
    "Scene": "phasor.scene",
    "expr": "phasor.scene",
    "load": "phasor.scene",
    "Engine": "phasor.engine",
    "Dot": "phasor.domains.basic",
    "Line": "phasor.domains.basic",
    "Text": "phasor.domains.basic",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'phasor' has no attribute {name!r}")
