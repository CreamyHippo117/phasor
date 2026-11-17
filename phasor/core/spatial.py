"""The one spatial pipeline (SPEC contract 4).

World space is right-handed with z up; the complex plane is x = Re, y = Im. A 2D scene is
an orthographic camera looking down -z at the plane z = 0. Screen space is pixels with the
origin at the top-left, plus normalized 0-1 coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from phasor.core.values import Vec3

ORTHO_DEPTH = 100.0  # the ortho camera sits this far above its center; depth range is 2x this
REGIONS = (
    ("upper-left", "upper-center", "upper-right"),
    ("middle-left", "center", "middle-right"),
    ("lower-left", "lower-center", "lower-right"),
)


class OrthoCameraParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    center: Vec3 = (0.0, 0.0, 0.0)
    height: float = Field(8.0, gt=0)  # world units visible from top to bottom of the frame


def _frozen(a) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class Camera:
    """A resolved camera: matrices plus the basis vectors objects need to face the viewer."""

    kind: str
    view_proj: np.ndarray  # 4x4, world -> clip (row-major; multiply as M @ [x, y, z, 1])
    eye: np.ndarray
    right: np.ndarray
    up: np.ndarray  # the world direction that appears as "up" on screen
    forward: np.ndarray
    height: float = field(default=1.0)  # ortho: world units per frame height

    def units_per_height(self, point=None) -> float:
        """World units spanned by the full frame height at ``point`` (constant for ortho)."""
        return self.height

    def project(self, points) -> np.ndarray:
        """World points (N, 3) -> NDC (N, 3); x, y in [-1, 1] inside the frame, y up."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        clip = np.c_[p, np.ones(len(p))] @ self.view_proj.T
        return clip[:, :3] / clip[:, 3:4]

    def to_screen(self, points, width: int, height: int) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2), origin top-left."""
        ndc = self.project(points)
        return np.c_[(ndc[:, 0] + 1) * 0.5 * width, (1 - ndc[:, 1]) * 0.5 * height]

    def depth(self, points) -> np.ndarray:
        """Distance from the camera along its viewing direction."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return (p - self.eye) @ self.forward


def ortho_camera(params: OrthoCameraParams, aspect: float) -> Camera:
    cx, cy, cz = params.center
    half_h = params.height / 2
    half_w = half_h * aspect
    m = np.array(
        [
            [1 / half_w, 0, 0, -cx / half_w],
            [0, 1 / half_h, 0, -cy / half_h],
            [0, 0, -1 / ORTHO_DEPTH, cz / ORTHO_DEPTH],  # nearer (higher z) -> smaller NDC z
            [0, 0, 0, 1],
        ]
    )
    return Camera(
        kind="ortho",
        view_proj=_frozen(m),
        eye=_frozen([cx, cy, cz + ORTHO_DEPTH]),
        right=_frozen([1, 0, 0]),
        up=_frozen([0, 1, 0]),
        forward=_frozen([0, 0, -1]),
        height=params.height,
    )


def overlay_camera(aspect: float) -> Camera:
    """The screen-space overlay layer: units are frame heights, origin top-left, y down.

    So x runs 0..aspect and y runs 0..1, and sizes mean the same as in world layers.
    """
    m = np.array(
        [
            [2 / aspect, 0, 0, -1],
            [0, -2, 0, 1],
            [0, 0, -1 / ORTHO_DEPTH, 0],
            [0, 0, 0, 1],
        ]
    )
    return Camera(
        kind="overlay",
        view_proj=_frozen(m),
        eye=_frozen([0, 0, ORTHO_DEPTH]),
        right=_frozen([1, 0, 0]),
        up=_frozen([0, -1, 0]),
        forward=_frozen([0, 0, -1]),
        height=1.0,
    )


def region_name(nx: float, ny: float) -> str:
    """3x3 grid name for a normalized screen point (clamped into the frame)."""
    col = min(2, max(0, int(nx * 3)))
    row = min(2, max(0, int(ny * 3)))
    return REGIONS[row][col]
