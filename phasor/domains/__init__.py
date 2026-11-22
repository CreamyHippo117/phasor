"""phasor.domains: object types, grouped by field. Importing this package registers the
built-in types; Engine does that once at startup."""

from phasor.domains import (
    acoustics,
    algebra,
    basic,
    calc,
    cx,
    fourier,
    geometry,
    linalg,
    media,
    pde,
)  # fmt: skip

__all__ = ["acoustics", "algebra", "basic", "calc", "cx", "fourier", "geometry", "linalg",
           "media", "network", "pde"]  # fmt: skip
