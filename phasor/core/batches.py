"""World-space batches (Milestone 28): many points, lines or arrows given as plain world
arrays, which the renderer turns into camera-facing triangles.

``ctx.points3d``, ``ctx.lines3d`` and ``ctx.arrows3d`` (``core.objects.DrawContext``) take
world positions, a colour per item (or one) and a size in one of three ``UNITS``:

- ``"world"``: world units;
- ``"height"``: a share of the frame height (what stroke roles are), so the size on screen
  does not change with zoom or depth;
- ``"px"``: pixels of the rendered picture (a draft at 640 pixels draws them relatively
  bolder than a final at 1080).

Screen units are measured at each item (``anchor=None``) or at one world point for the whole
batch (``anchor``: a size measured at the middle of a field, as ``strokes.py`` callers did).

What a batch keeps does not depend on the camera: the GL renderer uploads its arrays once and
expands them in the vertex shader with the camera as a uniform, so orbiting re-uploads
nothing (``render/passes.py``). ``expand`` here is the same expansion in NumPy: the reference
the shader is tested against (exactly, through transform feedback) and what the renderer
draws instead when told to expand on the CPU. The geometry is that of ``strokes.py``:
``disc_batch`` (12-gon dots), ``polylines`` (mitered strips, miter limit 4),
``arrow_batch`` (flat arrows) and ``cone_batch`` (8-sided cones).

Input is never trusted (``make``): wrong shapes or dtypes, non-finite rows, negative sizes and
more than ``MAX_ITEMS`` items are dropped, cut or clamped, and each such repair is noted in
``Batch.issues`` (the lint rule ``batch_input`` reports them); nothing raises.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phasor.core.spatial import Camera

KINDS = ("points", "lines", "arrows")
UNITS = ("world", "height", "px")
ARROW_SHAPES = ("flat", "cone")
MAX_ITEMS = 4_200_000  # points, line points or arrows in one batch (points x trail is 4e6)
DISC_SEGMENTS = 12  # as strokes.disc_batch
CONE_SEGMENTS = 8  # as strokes.cone_batch
MITER_MIN = 0.25  # miter limit 4x, as strokes.polylines
HEAD = 0.35  # an arrow's head, a share of its length (strokes.arrow_batch)
CONE_SHADE = (0.55,) * CONE_SEGMENTS + (1.0, 0.4)  # ring, tip, base centre


@dataclass(eq=False)
class Batch:
    """One sanitised batch: float32, C-contiguous, finite arrays of ``n`` items.

    ``positions`` (n, 3): dot centres, line points, arrow tails. ``tips`` (n, 3): arrow tips.
    ``strips`` (n,) int32: the strip each line point belongs to (consecutive points of one
    strip are joined); None: all one strip. ``colors`` (n, 4) linear RGBA, not
    premultiplied, or (1, 4): one for all. ``sizes`` (n,) or (1,): a dot's radius, a line's
    or an arrow's width, a cone's base diameter, in ``units``. One colour or size is kept
    once, so a batch whose item count changes (a growing path) uploads only its positions
    (and strips).
    ``tint``: the colours were one colour (the flash verb brightens them, as it does a plain
    mesh). ``opacity`` multiplies every alpha (a uniform: a fade uploads nothing)."""

    kind: str
    shape: str
    positions: np.ndarray
    colors: np.ndarray
    sizes: np.ndarray
    units: str
    anchor: tuple[float, float, float] | None
    glow: float
    opacity: float
    tint: bool
    tips: np.ndarray | None = None
    strips: np.ndarray | None = None
    head: float = HEAD
    issues: tuple[str, ...] = ()
    _digests: dict[str, bytes] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.positions)

    @property
    def each(self) -> int:
        """Which arrays have one row per item: 1 the colours, 2 the sizes (bits)."""
        return int(len(self.colors) == self.n and self.n > 1) + 2 * int(
            len(self.sizes) == self.n and self.n > 1
        )

    def per_item(self) -> tuple[np.ndarray, np.ndarray]:
        """Colours (n, 4) and sizes (n,), one value repeated where one was given."""
        return (np.broadcast_to(self.colors, (self.n, 4)),
                np.broadcast_to(self.sizes, (self.n,)))  # fmt: skip

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays the renderer uploads, by name, in binding order."""
        out = {"positions": self.positions, "colors": self.colors, "sizes": self.sizes}
        if self.strips is not None:
            out["strips"] = self.strips
        if self.tips is not None:
            out["tips"] = self.tips
        return out

    def digest(self, name: str) -> bytes:
        """A 128-bit hash of one array's bytes (with its dtype and shape), computed once."""
        d = self._digests.get(name)
        if d is None:
            a = self.arrays()[name]
            h = hashlib.blake2b(digest_size=16)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(memoryview(a).cast("B"))
            d = self._digests[name] = h.digest()
        return d

    def frame_key(self) -> tuple:
        """Everything this batch draws, for ``render.framekey`` (arrays by their hash)."""
        arrays = tuple((k, self.digest(k)) for k in self.arrays())
        return (self.kind, self.shape, self.units, self.anchor, self.glow, self.opacity,
                self.tint, self.head, arrays)  # fmt: skip


# ---------------------------------------------------------------- sanitising


def _numbers(x: Any, name: str, issues: list[str]) -> np.ndarray | None:
    """``x`` as float64, or None (noted) when it is not an array of real numbers."""
    try:
        if np.iscomplexobj(x):
            raise TypeError
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        issues.append(f"{name} is not an array of real numbers: nothing drawn")
        return None
    return a


def _points(x: Any, name: str, issues: list[str]) -> np.ndarray | None:
    """(..., 3) or (..., 2) (z = 0) world points flattened to (n, 3), or None (noted)."""
    a = _numbers(x, name, issues)
    if a is None:
        return None
    if a.ndim == 0 or a.shape[-1] not in (2, 3) or (a.ndim == 1 and a.size not in (2, 3)):
        issues.append(f"{name} must be (n, 3) world points, not shape {a.shape}: nothing drawn")
        return None
    a = a.reshape(-1, a.shape[-1])
    if a.shape[1] == 2:
        a = np.c_[a, np.zeros(len(a))]
    return a


def _per_item(x: Any, n: int, name: str, width: int, issues: list[str],
              default: np.ndarray) -> tuple[np.ndarray, bool]:  # fmt: skip
    """``x`` as (n, width), or (1, width) when one value is given (then True): one value is
    kept once, so it does not change when the number of items does."""
    a = _numbers(x, name, issues)
    if a is None:
        return default.reshape(1, width), True
    if width == 1:
        if a.size == 1:
            return a.reshape(1, 1), True
        if a.size == n:
            return a.reshape(n, 1), False
    else:
        if a.shape == (width,):
            return a.reshape(1, width), True
        if a.size == n * width and a.shape[-1] == width:
            return a.reshape(n, width), False
    issues.append(f"{name} has shape {a.shape}, not one value or one per item ({n}): "
                  "the default is used")  # fmt: skip
    return default.reshape(1, width), True


def _scalar(x: Any, name: str, default: float, lo: float, hi: float, issues: list[str]) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        issues.append(f"{name} {x!r} is not a number: {default:g} is used")
        return default
    if v != v:
        issues.append(f"{name} is not a number: {default:g} is used")
        return default
    if not lo <= v <= hi:
        issues.append(f"{name} {v:g} is outside {lo:g} to {hi:g}: clamped")
        return float(min(max(v, lo), hi))
    return v


def _common(units: Any, anchor: Any, glow: Any, opacity: Any, issues: list[str]):
    if units not in UNITS:
        issues.append(f"units {units!r} is not one of {', '.join(UNITS)}: 'height' is used")
        units = "height"
    if anchor is not None:
        a = _numbers(anchor, "anchor", issues)
        if a is None or a.shape != (3,) or not np.isfinite(a).all():
            if a is not None:
                issues.append(f"anchor must be one finite world point, not {anchor!r}: sizes "
                              "are measured at each item")  # fmt: skip
            anchor = None
        else:
            anchor = (float(a[0]), float(a[1]), float(a[2]))
    glow = _scalar(glow, "glow", 1.0, 0.0, 1e4, issues)
    opacity = _scalar(opacity, "opacity", 1.0, 0.0, 1.0, issues)
    return units, anchor, glow, opacity


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a).all(axis=1)
    return ok


def _cut(n: int, issues: list[str], what: str) -> int:
    if n > MAX_ITEMS:
        issues.append(f"{n:,} {what} is more than {MAX_ITEMS:,}: the first {MAX_ITEMS:,} are "
                      "drawn")  # fmt: skip
        return MAX_ITEMS
    return n


def _colors_sizes(colors, sizes, n, ok, issues, size_name):
    """Colours (clamped to finite, >= 0, alpha <= 1) and sizes (finite, >= 0): one per item
    (the items ``ok`` keeps), or one for all (1 row). Also whether the colour was one."""
    cols, one = _per_item(colors, n, "colors", 4, issues, np.ones(4))
    cols = np.array(cols if one else cols[ok], dtype=np.float64)
    if not np.isfinite(cols).all() or (cols < 0).any() or (cols[:, 3] > 1).any():
        issues.append("colors had values that are not finite, negative, or an alpha above 1: "
                      "clamped")  # fmt: skip
        cols = np.clip(np.nan_to_num(cols, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1e4)
        cols[:, 3] = np.minimum(cols[:, 3], 1.0)
    siz, one_size = _per_item(sizes, n, size_name, 1, issues, np.zeros(1))
    siz = np.array(siz if one_size else siz[ok], dtype=np.float64).reshape(-1)
    if not np.isfinite(siz).all() or (siz < 0).any() or (siz > 1e9).any():
        issues.append(f"{size_name} had values that are not finite, negative or above 1e9: "
                      "clamped")  # fmt: skip
        siz = np.clip(np.nan_to_num(siz, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1e9)
    return (np.ascontiguousarray(cols, dtype=np.float32), np.ascontiguousarray(siz, np.float32),
            one)  # fmt: skip


def _skipped(ok: np.ndarray, issues: list[str], what: str) -> None:
    bad = int((~ok).sum())
    if bad:
        issues.append(f"{bad:,} {what} not finite: skipped")


def make_points(centers, *, radius, units="height", colors=(1.0, 1.0, 1.0, 1.0), opacity=1.0,
                glow=1.0, anchor=None) -> tuple[Batch | None, list[str]]:  # fmt: skip
    """A batch of round dots facing the camera, ``radius`` each; (batch or None, issues)."""
    issues: list[str] = []
    units, anchor, glow, opacity = _common(units, anchor, glow, opacity, issues)
    p = _points(centers, "centers", issues)
    if p is None or not len(p):
        return None, issues
    total = len(p)
    n = _cut(total, issues, "points")
    p = p[:n]
    ok = _finite_rows(p)
    _skipped(ok, issues, "points")
    cols, siz, one = _colors_sizes(_head(colors, n, total), _head(radius, n, total), n, ok,
                                   issues, "radius")  # fmt: skip
    p = p[ok]
    if not len(p):
        return None, issues
    return Batch("points", "disc", np.ascontiguousarray(p, np.float32), cols, siz, units, anchor,
                 glow, opacity, one, issues=tuple(issues)), issues  # fmt: skip


def make_lines(points, *, width, units="height", colors=(1.0, 1.0, 1.0, 1.0), opacity=1.0,
               glow=1.0, anchor=None, lengths=None) -> tuple[Batch | None, list[str]]:  # fmt: skip
    """Polylines with mitered joins: ``points`` (n, 3) is one strip, (s, l, 3) is ``s`` strips
    of ``l`` points; ``lengths`` cuts (n, 3) into strips of those lengths (summing to n). A
    non-finite point ends its strip (noted). ``width`` is one or one per point."""
    issues: list[str] = []
    units, anchor, glow, opacity = _common(units, anchor, glow, opacity, issues)
    raw = _numbers(points, "points", issues)
    if raw is None:
        return None, issues
    p = _points(raw, "points", issues)
    if p is None or not len(p):
        return None, issues
    total = len(p)
    if lengths is not None:
        ln = _numbers(lengths, "lengths", issues)
        if ln is None or ln.ndim != 1 or (ln < 0).any() or not np.isfinite(ln).all() \
                or (ln != np.round(ln)).any() or int(ln.sum()) != total:  # fmt: skip
            if ln is not None:
                issues.append(f"lengths must be whole numbers summing to the {total} points: "
                              "the points are one strip")  # fmt: skip
            strip = np.zeros(total, dtype=np.int64)
        else:
            strip = np.repeat(np.arange(len(ln)), ln.astype(np.int64))
    elif raw.ndim == 3:
        strip = np.repeat(np.arange(raw.shape[0]), raw.shape[1])
    else:
        strip = np.zeros(total, dtype=np.int64)
    n = _cut(total, issues, "line points")
    p, strip = p[:n], strip[:n]
    ok = _finite_rows(p)
    _skipped(ok, issues, "line points")
    # A non-finite point ends its strip: the points after it start a new one.
    strip = strip + np.cumsum(~ok)
    cols, siz, one = _colors_sizes(_head(colors, n, total), _head(width, n, total), n, ok,
                                   issues, "width")  # fmt: skip
    p, strip = p[ok], strip[ok]
    if len(p) < 2:
        return None, issues
    # renumbered 0, 1, 2... in order (int32 for the card); one strip needs none (a path
    # that grows then uploads only its positions)
    strip = np.concatenate([[0], np.cumsum(strip[1:] != strip[:-1])]).astype(np.int32)
    strips = np.ascontiguousarray(strip) if strip[-1] > 0 else None
    return Batch("lines", "strip", np.ascontiguousarray(p, np.float32), cols, siz, units, anchor,
                 glow, opacity, one, strips=strips,
                 issues=tuple(issues)), issues  # fmt: skip


def make_arrows(tails, tips, *, width, units="height", colors=(1.0, 1.0, 1.0, 1.0),
                opacity=1.0, glow=1.0, anchor=None, shape="flat",
                head=HEAD) -> tuple[Batch | None, list[str]]:  # fmt: skip
    """Arrows from ``tails`` to ``tips`` (n, 3): ``shape="flat"`` (facing the camera, shaft
    ``width``, a head ``head`` of the length) or ``"cone"`` (a solid cone, base diameter
    ``width``)."""
    issues: list[str] = []
    units, anchor, glow, opacity = _common(units, anchor, glow, opacity, issues)
    if shape not in ARROW_SHAPES:
        issues.append(f"shape {shape!r} is not one of {', '.join(ARROW_SHAPES)}: 'flat' is used")
        shape = "flat"
    head = _scalar(head, "head", HEAD, 0.0, 1.0, issues)
    a = _points(tails, "tails", issues)
    b = _points(tips, "tips", issues)
    if a is None or b is None or not len(a):
        return None, issues
    if a.shape != b.shape:
        issues.append(f"tails {a.shape} and tips {b.shape} differ: nothing drawn")
        return None, issues
    total = len(a)
    n = _cut(total, issues, "arrows")
    a, b = a[:n], b[:n]
    ok = _finite_rows(a, b)
    _skipped(ok, issues, "arrows")
    cols, siz, one = _colors_sizes(_head(colors, n, total), _head(width, n, total), n, ok,
                                   issues, "width")  # fmt: skip
    a, b = a[ok], b[ok]
    if not len(a):
        return None, issues
    return Batch("arrows", shape, np.ascontiguousarray(a, np.float32), cols, siz, units, anchor,
                 glow, opacity, one, tips=np.ascontiguousarray(b, np.float32), head=head,
                 issues=tuple(issues)), issues  # fmt: skip


def _head(x: Any, n: int, total: int) -> Any:
    """A per-item array of ``total`` items cut to the first ``n`` (past ``MAX_ITEMS``)."""
    if n < total and isinstance(x, np.ndarray) and x.ndim >= 1 and len(x) == total:
        return x[:n]
    return x


# ---------------------------------------------------------------- the expansion (CPU)


def world_sizes(cam: Camera, at: np.ndarray, sizes: np.ndarray, units: str, frame_px: int,
                anchor) -> np.ndarray:  # fmt: skip
    """Sizes in world units at the points ``at`` (n, 3): the shader's ``world_size``."""
    sizes = np.asarray(sizes, dtype=np.float64)
    if units == "world":
        return sizes
    share = sizes if units == "height" else sizes / float(max(frame_px, 1))
    if not cam.perspective:
        return share * float(cam.height)
    ref = np.asarray(at, dtype=np.float64) if anchor is None else np.asarray(anchor)[None]
    depth = (ref - np.asarray(cam.eye, dtype=np.float64)) @ np.asarray(cam.forward, np.float64)
    return share * np.maximum(depth, 1e-6) * 2.0 * float(cam.tan_half_fov)


def _camera(cam: Camera):
    return (np.asarray(cam.right, np.float64), np.asarray(cam.up, np.float64),
            np.asarray(cam.forward, np.float64), np.asarray(cam.eye, np.float64))  # fmt: skip


def _unit_rows(v: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.sqrt((v * v).sum(axis=-1))
    return v / np.maximum(n, eps)[..., None], n


def _points_instances(b: Batch, cam, frame_px):
    right, up, _, _ = _camera(cam)
    c = b.positions.astype(np.float64)
    r = world_sizes(cam, c, b.per_item()[1], b.units, frame_px, b.anchor)
    ang = 2.0 * np.pi * np.arange(DISC_SEGMENTS) / DISC_SEGMENTS
    corner = np.r_[[[0.0, 0.0]], np.c_[np.cos(ang), np.sin(ang)]]  # (13, 2)
    off = corner[:, 0, None] * right + corner[:, 1, None] * up  # (13, 3)
    verts = c[:, None, :] + r[:, None, None] * off[None]
    return verts, np.ones(len(c), dtype=bool), None


def _lines_instances(b: Batch, cam, frame_px):
    """Instance j is the segment from point j to j + 1 (valid when both are in one strip);
    its four vertices are point j's two sides, then point j + 1's (strokes.polylines)."""
    right, up, _, _ = _camera(cam)
    p = b.positions.astype(np.float64)
    n = len(p)
    s = b.strips if b.strips is not None else np.zeros(n, dtype=np.int32)
    flat = np.c_[p @ right, p @ up]
    d = flat[1:] - flat[:-1]
    norm = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
    u = d / np.where(norm == 0, 1.0, norm)[:, None]
    perp = np.c_[-u[:, 1], u[:, 0]]  # (n - 1, 2): the normal of segment i -> i + 1
    has_prev = np.r_[False, s[1:] == s[:-1]]
    has_next = np.r_[s[1:] == s[:-1], False]
    zero = np.zeros((1, 2))
    before = np.r_[zero, perp]  # the normal of the segment ending at i
    after = np.r_[perp, zero]  # ... starting at i
    n_in = np.where(has_prev[:, None], before, after)
    n_out = np.where(has_next[:, None], after, before)
    miter = n_in + n_out
    miter = miter / np.maximum(np.sqrt((miter * miter).sum(axis=1)), 1e-9)[:, None]
    scale = 1.0 / np.maximum((miter * n_in).sum(axis=1), MITER_MIN)
    half = world_sizes(cam, p, b.per_item()[1], b.units, frame_px, b.anchor) / 2.0
    off = miter * (scale * half)[:, None]
    off3 = off[:, 0, None] * right + off[:, 1, None] * up
    plus, minus = p + off3, p - off3
    j = np.arange(n - 1)
    verts = np.stack([plus[j], minus[j], plus[j + 1], minus[j + 1]], axis=1)
    return verts, s[1:] == s[:-1], None


def _flat_instances(b: Batch, cam, frame_px):
    right, _, forward, eye = _camera(cam)
    a = b.positions.astype(np.float64)
    tip = b.tips.astype(np.float64)
    d = tip - a
    length = np.sqrt((d * d).sum(axis=1))
    mid = (a + tip) / 2.0
    view = (mid - eye) if cam.perspective else np.broadcast_to(forward, a.shape)
    side = np.cross(d, view)
    sn = np.sqrt((side * side).sum(axis=1))
    side = np.where((sn > 1e-12)[:, None], side / np.maximum(sn, 1e-12)[:, None], right)
    w = world_sizes(cam, mid, b.per_item()[1], b.units, frame_px, b.anchor)[:, None] / 2.0
    hw = np.maximum(w * 2.2, length[:, None] * b.head * 0.45)
    base = tip - d * b.head
    verts = np.stack([a - side * w, a + side * w, base + side * w, base - side * w,
                      base - side * hw, base + side * hw, tip], axis=1)  # fmt: skip
    return verts, np.ones(len(a), dtype=bool), None


def _cone_instances(b: Batch, cam, frame_px):
    a = b.positions.astype(np.float64)
    tip = b.tips.astype(np.float64)
    d, _ = _unit_rows(tip - a, 1e-12)
    pick = np.where((np.abs(d[:, 0]) < 0.9)[:, None], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u, _ = _unit_rows(np.cross(d, pick), 1e-12)
    v = np.cross(d, u)
    r = world_sizes(cam, a, b.per_item()[1], b.units, frame_px, b.anchor)[:, None, None] / 2.0
    ang = 2.0 * np.pi * np.arange(CONE_SEGMENTS) / CONE_SEGMENTS
    ring = a[:, None, :] + r * (np.cos(ang)[None, :, None] * u[:, None, :]
                                + np.sin(ang)[None, :, None] * v[:, None, :])  # fmt: skip
    verts = np.concatenate([ring, tip[:, None, :], a[:, None, :]], axis=1)
    return verts, np.ones(len(a), dtype=bool), np.asarray(CONE_SHADE)


def _template(kind: str, shape: str) -> np.ndarray:
    """The triangles of one instance, as indices into its vertices (the renderer draws the
    same list, instanced)."""
    if kind == "points":
        k = np.arange(DISC_SEGMENTS)
        return np.stack([np.zeros(DISC_SEGMENTS, int), 1 + k, 1 + (k + 1) % DISC_SEGMENTS], 1)
    if kind == "lines":
        return np.array([[0, 1, 2], [1, 3, 2]])
    if shape == "flat":
        return np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    k = np.arange(CONE_SEGMENTS)
    side = np.stack([np.full(CONE_SEGMENTS, CONE_SEGMENTS), k, (k + 1) % CONE_SEGMENTS], 1)
    cap = np.stack([np.full(CONE_SEGMENTS, CONE_SEGMENTS + 1), (k + 1) % CONE_SEGMENTS, k], 1)
    return np.concatenate([side, cap])


TEMPLATES = {("points", "disc"): _template("points", "disc"),
             ("lines", "strip"): _template("lines", "strip"),
             ("arrows", "flat"): _template("arrows", "flat"),
             ("arrows", "cone"): _template("arrows", "cone")}  # fmt: skip
VERTICES = {("points", "disc"): DISC_SEGMENTS + 1, ("lines", "strip"): 4,
            ("arrows", "flat"): 7, ("arrows", "cone"): CONE_SEGMENTS + 2}  # fmt: skip


def instances(b: Batch, cam: Camera, frame_px: int):
    """Every instance's vertices (k, v, 3) in world units, their colours (k, v, 4) and which
    instances draw (k,): exactly what the vertex shader computes, instance by instance."""
    fn = {"points": _points_instances, "lines": _lines_instances}.get(b.kind)
    if fn is None:
        fn = _cone_instances if b.shape == "cone" else _flat_instances
    verts, valid, shade = fn(b, cam, frame_px)
    v = verts.shape[1]
    if b.kind == "lines":
        c = b.per_item()[0].astype(np.float64)
        cols = np.stack([c[:-1], c[:-1], c[1:], c[1:]], axis=1)
    else:
        cols = np.repeat(b.per_item()[0].astype(np.float64)[:, None, :], v, axis=1)
    if shade is not None:
        cols = cols.copy()
        cols[..., :3] *= shade[None, :, None]
    return verts, cols, valid


def expand(b: Batch, cam: Camera, frame_px: int):
    """The batch as plain triangles: (positions (m, 3), indices (k, 3), colors (m, 4)), all
    float64 world units / linear RGBA, instance after instance in the shader's order."""
    verts, cols, valid = instances(b, cam, frame_px)
    verts, cols = verts[valid], cols[valid]
    tri = TEMPLATES[(b.kind, b.shape)]
    v = verts.shape[1]
    idx = (np.arange(len(verts))[:, None, None] * v + tri[None]).reshape(-1, 3)
    return verts.reshape(-1, 3), idx, cols.reshape(-1, 4)
