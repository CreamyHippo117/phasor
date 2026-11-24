"""phasor.domains: object types, grouped by field. Importing this package registers the
built-in types; Engine does that once at startup."""

from phasor.domains import (
    acoustics,
    algebra,
    basic,
    calc,
    cloud,
    cx,
    diagrams,
    dynamics,
    flow,
    fourier,
    geometry,
    linalg,
    media,
    network,
    pde,
    shapes,
)  # fmt: skip

__all__ = ["acoustics", "algebra", "basic", "calc", "cloud", "cx", "diagrams", "dynamics",
           "flow", "fourier",
           "geometry", "linalg", "media", "network", "pde",
           "shapes"]  # fmt: skip
