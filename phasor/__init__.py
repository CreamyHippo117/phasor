"""phasor: GPU-rendered math animations that an AI can see, check and fix.

This module stays import-light on purpose: importing ``phasor`` must never pull in
CuPy, ModernGL or PyAV, and must never import a layer as a side effect (that would
hide layer violations from import-linter and create import cycles). Public names
are added here lazily as milestones land.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
