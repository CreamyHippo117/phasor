"""phasor.domains: object types, grouped by field. Importing this package registers the
built-in types; Engine does that once at startup."""

from phasor.domains import acoustics, basic, cx, fourier, linalg

__all__ = ["acoustics", "basic", "cx", "fourier", "linalg"]
