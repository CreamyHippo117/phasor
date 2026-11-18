"""The object contract (SPEC contract 5) and the type registry.

Every object type implements ``Params``, ``world_bounds``, ``features``, ``semantics`` and
``draw``. ``register`` refuses a type that is missing any of them.

An instance is bound to one object id and holds no scene state: everything it needs comes
from the ``SceneState`` it is handed. The same instance doubles as the Python builder
(``Dot(id="dot", pos=(1, 2))``), so there is one class per type, not two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from phasor.core.ids import is_valid_name
from phasor.core.schema import ObjectSpec
from phasor.core.state import ObjectState, SceneState

Pass = Literal["color", "id", "solo"]
PASSES: tuple[str, ...] = ("color", "id", "solo")
REQUIRED_MEMBERS = ("Params", "world_bounds", "features", "semantics", "draw")


class ParamsBase(BaseModel):
    """Parameters every object has."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    opacity: float = Field(1.0, ge=0, le=1)
    # A second opacity factor, owned by dim_others/undim so that dimming and revealing never
    # fight over one track. In a SceneState, ``opacity`` already includes it.
    dim: float = Field(1.0, ge=0, le=1)
    z_index: int = 0  # draw order among coplanar objects; higher draws on top


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned box in world space, with its bounding sphere."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.lo) + np.asarray(self.hi)) / 2

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(np.asarray(self.hi) - np.asarray(self.lo)) / 2)

    def corners(self) -> np.ndarray:
        (x0, y0, z0), (x1, y1, z1) = self.lo, self.hi
        return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])

    @classmethod
    def of_points(cls, points, pad: float = 0.0) -> Bounds:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return cls(tuple(float(v) for v in p.min(0) - pad), tuple(float(v) for v in p.max(0) + pad))


class DrawContext(Protocol):
    """What an object may ask a renderer to draw. Renderer-agnostic and pass-agnostic: the
    renderer decides what a command means in the ``color``, ``id`` and ``solo`` passes."""

    def triangles(
        self,
        positions: np.ndarray,  # (N, 3) world space
        indices: np.ndarray | None = None,  # (M, 3) or None for a plain triangle list
        *,
        color: tuple[float, float, float, float],  # linear RGBA, not premultiplied
        glow: float = 1.0,  # emission: the color pass multiplies rgb by this (theme glow role)
    ) -> None: ...

    def field(
        self,
        corner: np.ndarray,  # world position of the quad's lower-left corner
        right: np.ndarray,  # world vector along the quad's bottom edge
        up: np.ndarray,  # world vector along its left edge
        *,
        pixels: Callable[[int, int, tuple[float, float], float], np.ndarray],
        opacity: float = 1.0,
    ) -> None:
        """A quad whose color is computed per pixel (domain coloring, fractals). It is one
        object in the id and solo passes. For the color pass the renderer calls
        ``pixels(width_px, height_px, center, height)`` with the visible part of the quad as
        a view in quad coordinates (``center`` and ``height`` measured along ``right`` and
        ``up`` from ``corner``, in world units), and expects linear RGBA (h, w, 4) float32,
        row 0 = top."""
        ...

    def text(
        self,
        text: str,
        origin: np.ndarray,  # world position of the anchor
        height: float,  # world units, full line height
        *,
        color: tuple[float, float, float, float],
        anchor: str = "center",  # "center" | "left" | "right"
    ) -> None: ...


@runtime_checkable
class SceneObject(Protocol):
    type: ClassVar[str]
    Params: ClassVar[type[BaseModel]]
    id: str

    def world_bounds(self, state: SceneState) -> Bounds: ...
    def features(self, state: SceneState) -> dict[str, np.ndarray]: ...
    def semantics(self, state: SceneState) -> dict[str, Any]: ...
    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None: ...


class ObjectBase:
    """Convenience base. It deliberately implements none of the contract members."""

    type: ClassVar[str]
    # style key -> (role group, default role). Scenes may only use roles, never raw values.
    Style: ClassVar[dict[str, tuple[str, str]]] = {}

    def __init__(
        self,
        id: str,
        *,
        style: dict[str, str] | None = None,
        tags: tuple[str, ...] | list[str] = (),
        layer: str = "world",
        **params: Any,
    ) -> None:
        self.id = id
        self._spec_fields = {"style": style or {}, "tags": tuple(tags), "layer": layer}
        self._params = params

    def to_spec(self) -> ObjectSpec:
        return ObjectSpec(id=self.id, type=self.type, params=self._params, **self._spec_fields)

    def me(self, state: SceneState) -> ObjectState:
        return state[self.id]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"


_REGISTRY: dict[str, type] = {}


class RegistrationError(TypeError):
    pass


def register(cls: type) -> type:
    """Class decorator: add an object type, or fail loudly if it breaks the contract."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not is_valid_name(name):
        raise RegistrationError(f"{cls.__name__}.type must be a name like 'dot', got {name!r}")
    missing = [m for m in REQUIRED_MEMBERS if not hasattr(cls, m)]
    if missing:
        raise RegistrationError(f"{cls.__name__} breaks the object contract: missing {missing}")
    if not (isinstance(cls.Params, type) and issubclass(cls.Params, ParamsBase)):
        raise RegistrationError(f"{cls.__name__}.Params must subclass ParamsBase")
    for m in REQUIRED_MEMBERS[1:]:
        if not callable(getattr(cls, m)):
            raise RegistrationError(f"{cls.__name__}.{m} must be a method")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise RegistrationError(f"object type {name!r} is already registered")
    _REGISTRY[name] = cls
    return cls


def get_type(name: str) -> type | None:
    return _REGISTRY.get(name)


def registered_types() -> dict[str, type]:
    return dict(_REGISTRY)
