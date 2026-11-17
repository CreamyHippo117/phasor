"""The single way phasor creates a headless OpenGL context.

Windows and desktop Linux use the platform default (WGL / GLX). Headless Linux
(CI, cloud containers) has no X display, so we fall back to EGL, which works with
Mesa's llvmpipe software renderer. Nothing else in phasor may call
``moderngl.create_context`` directly.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import moderngl

REQUIRED_GL = 430  # OpenGL 4.3: compute shaders, SSBOs, integer textures


class GLUnavailableError(RuntimeError):
    """No OpenGL 4.3 context could be created on this machine."""


def _backends() -> list[str | None]:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return ["egl", None]
    if sys.platform.startswith("linux"):
        return [None, "egl"]
    return [None]


def create_standalone_context(require: int = REQUIRED_GL) -> moderngl.Context:
    """Create an offscreen context with at least GL ``require`` (e.g. 430).

    Raises GLUnavailableError listing every backend that was tried.
    """
    try:
        import moderngl
    except ImportError as e:  # pragma: no cover - depends on install
        raise GLUnavailableError("moderngl is not installed (pip install phasor[gl])") from e

    errors: list[str] = []
    for backend in _backends():
        kwargs: dict[str, object] = {"standalone": True, "require": require}
        if backend:
            kwargs["backend"] = backend
        try:
            return moderngl.create_context(**kwargs)
        except Exception as e:
            errors.append(f"{backend or 'default'}: {e}")
    raise GLUnavailableError("; ".join(errors))
