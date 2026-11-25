"""World-space batches (Milestone 28): ``ctx.points3d``, ``lines3d`` and ``arrows3d``.

- the CPU expansion (``core.batches.instances``) equals a scalar per-vertex reference written
  the way the vertex shader is, and the geometry ``strokes.py`` drew before;
- the vertex shader equals the CPU expansion (transform feedback), and pictures drawn both
  ways agree (``GLRenderer.expand_on_cpu``), in the color and id passes;
- hostile arrays are repaired and linted (``batch_input``), never raised, in both MODES;
- buffers are keyed by content: orbiting the M26 Lorenz scene uploads nothing after the first
  frame, advancing the flow time uploads one positions array per moving batch;
- ``quality="interactive"`` thins a field and strides a set, and only the color pass.
"""

from __future__ import annotations

import math
import runpy
import time
import tracemalloc
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from phasor import Scene, flow
from phasor.core import batches as bt
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.spatial import Camera
from phasor.domains import strokes
from phasor.render.passes import _Collector
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))  # nothing in the repo


def _ortho() -> Camera:
    return Scene("o").engine.state_at(0.0).camera


def _persp() -> Camera:
    s = Scene("p")
    s.use_camera("orbit")
    s.set("camera.distance", 9.0)
    s.set("camera.elevation", 25.0)
    s.set("camera.azimuth", -40.0)
    return s.engine.state_at(0.0).camera


CAMERAS = {"ortho": _ortho, "perspective": _persp}


def _spiral(n=40, z=0.0):
    t = np.linspace(0.0, 6.0, n)
    return np.c_[np.cos(t) * t / 3, np.sin(t) * t / 3, z + 0.2 * np.sin(3 * t)]


# ---------------------------------------------------------------- the scalar reference
# One vertex at a time, in plain Python floats, written as the shader is (``_VS_BATCH``).


def _uph(cam, at):
    if not cam.perspective:
        return float(cam.height)
    depth = sum((at[i] - cam.eye[i]) * cam.forward[i] for i in range(3))
    return max(depth, 1e-6) * 2.0 * float(cam.tan_half_fov)


def _size(b, cam, s, at, frame_px):
    if b.units == "world":
        return float(s)
    share = float(s) if b.units == "height" else float(s) / frame_px
    return share * _uph(cam, b.anchor if b.anchor is not None else at)


def _v(a):
    return [float(x) for x in a]


def _add(a, b, k=1.0):
    return [a[i] + k * b[i] for i in range(3)]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(sum(x * x for x in a))


def _ref_vertex(b, cam, frame_px, j, k):
    right, up, fwd, eye = _v(cam.right), _v(cam.up), _v(cam.forward), _v(cam.eye)
    P = [_v(p) for p in b.positions]
    if b.kind == "points":
        c = P[j]
        r = _size(b, cam, np.broadcast_to(b.sizes, (b.n,))[j], c, frame_px)
        cx, cy = (0.0, 0.0) if k == 0 else (math.cos(2 * math.pi * (k - 1) / 12),
                                             math.sin(2 * math.pi * (k - 1) / 12))  # fmt: skip
        return _add(c, [cx * right[i] + cy * up[i] for i in range(3)], r)
    if b.kind == "lines":
        s = [0] * b.n if b.strips is None else [int(x) for x in b.strips]
        i = j + k // 2
        side = 1.0 if k % 2 == 0 else -1.0

        def flat(p):
            return (sum(p[m] * right[m] for m in range(3)), sum(p[m] * up[m] for m in range(3)))

        def normal(q):
            a, c = flat(P[q]), flat(P[q + 1])
            d = (c[0] - a[0], c[1] - a[1])
            n = math.sqrt(d[0] * d[0] + d[1] * d[1])
            n = 1.0 if n == 0 else n
            return (-d[1] / n, d[0] / n)

        has_prev = i > 0 and s[i - 1] == s[i]
        has_next = i + 1 < len(P) and s[i + 1] == s[i]
        before = normal(i - 1) if has_prev else (0.0, 0.0)
        after = normal(i) if has_next else (0.0, 0.0)
        n_in = before if has_prev else after
        n_out = after if has_next else before
        m = (n_in[0] + n_out[0], n_in[1] + n_out[1])
        mn = max(math.sqrt(m[0] ** 2 + m[1] ** 2), 1e-9)
        m = (m[0] / mn, m[1] / mn)
        scale = 1.0 / max(m[0] * n_in[0] + m[1] * n_in[1], 0.25)
        half = _size(b, cam, np.broadcast_to(b.sizes, (b.n,))[i], P[i], frame_px) / 2
        off = (m[0] * scale * half, m[1] * scale * half)
        return _add(P[i], [off[0] * right[m_] + off[1] * up[m_] for m_ in range(3)], side)
    a, tip = P[j], _v(b.tips[j])
    d = [tip[i] - a[i] for i in range(3)]
    if b.shape == "flat":
        length = _norm(d)
        mid = [(a[i] + tip[i]) / 2 for i in range(3)]
        view = [mid[i] - eye[i] for i in range(3)] if cam.perspective else fwd
        side = _cross(d, view)
        sn = _norm(side)
        side = [x / max(sn, 1e-12) for x in side] if sn > 1e-12 else right
        w = _size(b, cam, np.broadcast_to(b.sizes, (b.n,))[j], mid, frame_px) / 2
        hw = max(w * 2.2, length * b.head * 0.45)
        base = [tip[i] - d[i] * b.head for i in range(3)]
        return [_add(a, side, -w), _add(a, side, w), _add(base, side, w), _add(base, side, -w),
                _add(base, side, -hw), _add(base, side, hw), tip][k]  # fmt: skip
    dn = max(_norm(d), 1e-12)
    d = [x / dn for x in d]
    pick = [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = _cross(d, pick)
    un = max(_norm(u), 1e-12)
    u = [x / un for x in u]
    v = _cross(d, u)
    r = _size(b, cam, np.broadcast_to(b.sizes, (b.n,))[j], a, frame_px) / 2
    if k < 8:
        t = 2 * math.pi * k / 8
        return [a[i] + r * (math.cos(t) * u[i] + math.sin(t) * v[i]) for i in range(3)]
    return tip if k == 8 else a


def _batches(units="height", anchor=None):
    sp = _spiral()
    rng = np.random.default_rng(7)
    sizes = 0.004 + 0.01 * rng.random(len(sp))
    scale = {"world": 10.0, "height": 1.0, "px": 1000.0}[units]
    kw = {"units": units, "anchor": anchor}
    pts, _ = bt.make_points(sp, radius=sizes * scale, colors=rng.random((len(sp), 4)), **kw)
    lines, _ = bt.make_lines(np.r_[sp, [[np.nan] * 3], sp[::-1] * 0.5], width=0.006 * scale,
                             **kw)  # fmt: skip
    flat, _ = bt.make_arrows(
        sp[::3], sp[::3] + np.array([0.3, 0.1, 0.05]), width=0.005 * scale, **kw
    )
    tips = sp[::3] + rng.normal(size=(len(sp[::3]), 3)) * 0.3
    tips[0] = sp[0] + [1.0, 0.0, 0.0]  # along x: the other side vector is picked
    cone, _ = bt.make_arrows(sp[::3], tips, width=0.02 * scale, shape="cone", **kw)
    return {"points": pts, "lines": lines, "flat": flat, "cone": cone}


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize("cam_name", sorted(CAMERAS))
@pytest.mark.parametrize(
    ("units", "anchor"),
    [("world", None), ("height", None), ("px", None), ("height", (0.3, -0.2, 0.1))],
)
def test_the_cpu_expansion_equals_the_shader_written_out_by_hand(cam_name, units, anchor):
    cam = CAMERAS[cam_name]()
    for name, b in _batches(units, anchor).items():
        verts, _, valid = bt.instances(b, cam, 360)
        for j in range(len(verts)):
            if not valid[j]:
                continue
            for k in range(verts.shape[1]):
                ref = _ref_vertex(b, cam, 360, j, k)
                assert np.allclose(verts[j, k], ref, rtol=1e-12, atol=1e-12), (name, j, k)


class _Tris:
    stroke_hue = None

    def __init__(self):
        self.calls = []

    def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None):
        self.calls.append((np.asarray(positions), np.asarray(indices)))


def _same_triangles(a, b):
    """The same set of triangles (as vertex triples), whatever the order."""
    pa, ia = a
    pb, ib = b
    ta = np.sort(np.round(pa[ia].reshape(len(ia), -1), 9), axis=0)
    tb = np.sort(np.round(pb[ib].reshape(len(ib), -1), 9), axis=0)
    return ta.shape == tb.shape and np.allclose(ta, tb, atol=1e-9)


@pytest.mark.parametrize("cam_name", sorted(CAMERAS))
def test_the_geometry_is_what_strokes_drew_before(cam_name):
    """Migrating a flow object changes nothing but who expands it: world units, the same
    triangles as ``strokes.polylines``, ``disc_batch``, ``arrow_batch`` and ``cone_batch``."""
    cam = CAMERAS[cam_name]()
    sp = _spiral().astype(np.float32).astype(np.float64)
    tips = (sp[::3] + np.array([0.3, 0.1, 0.05])).astype(np.float32).astype(np.float64)
    cases = [
        (bt.make_lines(sp, width=0.03, units="world")[0],
         lambda c: strokes.polyline(c, cam, sp, 0.03, color=(1, 1, 1, 1))),
        (bt.make_points(sp, radius=0.02, units="world")[0],
         lambda c: strokes.disc_batch(c, cam, sp, 0.02, colors=(1, 1, 1, 1))),
        (bt.make_arrows(sp[::3], tips, width=0.02, units="world")[0],
         lambda c: strokes.arrow_batch(c, cam, sp[::3], tips, 0.02, colors=(1, 1, 1, 1))),
        (bt.make_arrows(sp[::3], tips, width=0.04, units="world", shape="cone")[0],
         lambda c: strokes.cone_batch(c, sp[::3], tips, 0.02, colors=(1, 1, 1, 1))),
    ]  # fmt: skip
    for b, old in cases:
        ctx = _Tris()
        old(ctx)
        pos, idx, _ = bt.expand(b, cam, 360)
        assert _same_triangles((pos, idx), ctx.calls[0]), b.kind


# ---------------------------------------------------------------- hostile input

HOSTILE = [
    ("points3d", {"centers": [[np.nan, 0, 0], [0, 0, 0], [np.inf, 1, 1]], "radius": 0.01}),
    ("points3d", {"centers": "not numbers", "radius": 0.01}),
    ("points3d", {"centers": np.zeros((5, 4)), "radius": 0.01}),
    ("points3d", {"centers": np.array([[1 + 2j, 0, 0]]), "radius": 0.01}),
    ("points3d", {"centers": np.array([[1, 0, 0]], dtype=object), "radius": [0.01, 0.02]}),
    ("points3d", {"centers": np.zeros((3, 3)), "radius": -1.0, "colors": [[np.nan] * 4] * 3}),
    ("points3d", {"centers": np.zeros((3, 3)), "radius": 0.01, "units": "furlongs"}),
    ("points3d", {"centers": np.zeros((3, 3)), "radius": 0.01, "anchor": (np.nan, 0, 0)}),
    ("points3d", {"centers": np.zeros((3, 3)), "radius": 0.01, "opacity": 7.0, "glow": "x"}),
    ("lines3d", {"points": np.zeros((6, 3)), "width": 0.01, "lengths": [2, 2]}),
    ("lines3d", {"points": np.ones((2, 3))[:, ::-1], "width": [np.inf, -3]}),
    ("lines3d", {"points": np.zeros(7), "width": 0.01}),
    ("arrows3d", {"tails": np.zeros((3, 3)), "tips": np.ones((2, 3)), "width": 0.01}),
    ("arrows3d", {"tails": np.zeros((3, 3)), "tips": np.ones((3, 3)), "width": 0.01,
                  "shape": "teapot", "head": 9.0}),
]  # fmt: skip


@pytest.mark.parametrize(("method", "kw"), HOSTILE)
def test_hostile_arrays_are_repaired_and_noted_never_raised(method, kw):
    make = {"points3d": bt.make_points, "lines3d": bt.make_lines, "arrows3d": bt.make_arrows}
    kw = dict(kw)  # (the table is shared)
    args = [kw.pop(k) for k in ("centers", "points", "tails", "tips") if k in kw]
    b, issues = make[method](*args, **kw)
    assert issues, (method, kw)
    if b is not None:
        for a in b.arrays().values():
            assert a.flags.c_contiguous and np.isfinite(a).all()
        assert b.colors.dtype == np.float32 and (b.sizes >= 0).all() and 0 <= b.opacity <= 1


def test_a_huge_batch_is_cut_and_noted(monkeypatch):
    monkeypatch.setattr(bt, "MAX_ITEMS", 1000)
    b, issues = bt.make_lines(np.zeros((5000, 3)) + np.arange(5000)[:, None],
                              width=np.full(5000, 0.01), colors=np.ones((5000, 4)))  # fmt: skip
    assert b.n == 1000 and any("first 1,000" in i for i in issues)


@register
class _Hostile(ObjectBase):
    """A scene-local type that hands its batches whatever it is given."""

    type: ClassVar[str] = "m28_hostile"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        case: int = 0

    def draw(self, ctx, state):
        method, kw = HOSTILE[self.me(state).params.case]
        kw = dict(kw)
        args = [kw.pop(k) for k in ("centers", "points", "tails", "tips") if k in kw]
        getattr(ctx, method)(*args, **kw)
        ctx.points3d(np.zeros((1, 3)), radius=0.01)  # something on screen


@pytest.mark.parametrize("passes", MODES)
def test_a_type_drawing_hostile_batches_lints_and_renders(passes):
    s = Scene("hostile", passes=passes)
    for k in range(len(HOSTILE)):
        s.add(_Hostile(id=f"h{k}", case=k))
    e = s.engine
    found = {i.ids[0] for i in e.lint() if i.code == "batch_input"}
    assert found == {f"h{k}" for k in range(len(HOSTILE))}
    sem = e.inspect(0.0, id="h0")["objects"][0]["semantic"]
    assert any("not finite" in x for x in sem["batch_input"])
    if passes is not None and gl_available():
        assert e.still(0.0, (160, 90)).shape == (90, 160, 3)


@register
class _Curve(ObjectBase):
    type: ClassVar[str] = "m28_curve"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        kind: str = "lines"

    def draw(self, ctx, state):
        sp = _spiral()
        k = self.me(state).params.kind
        if k == "lines":
            ctx.lines3d(sp, width=0.008, colors=(1.0, 0.6, 0.2, 1.0), glow=2.5)
        elif k == "points":
            ctx.points3d(sp, radius=0.008, colors=np.c_[np.linspace(0, 1, len(sp)),
                                                         np.full((len(sp), 3), 0.7)])  # fmt: skip
        elif k == "flat":
            ctx.arrows3d(sp[::3], sp[::3] + np.array([0.3, 0.1, 0.0]), width=0.006,
                         colors=(0.4, 0.8, 1.0, 1.0))  # fmt: skip
        else:
            ctx.arrows3d(sp[::3], sp[::3] + np.array([0.3, 0.1, 0.2]), width=0.12, units="world",
                         shape="cone", colors=(0.9, 0.3, 0.9, 1.0))  # fmt: skip


def test_bounds_and_features_come_from_what_the_batch_records():
    s = Scene("rec", passes=None)
    s.add(_Curve(id="c"))
    e = s.engine
    state = e.state_at(0.0)
    impl = state["c"].impl
    box = impl.world_bounds(state)
    sp = _spiral()
    # the points' box padded by half the line's width (0.008 of the frame height) on screen
    half = 0.008 * state.camera.units_per_height() / 2
    pad = np.array([half, half, 0.0])
    assert np.allclose(box.lo, sp.min(axis=0) - pad, atol=1e-6)
    assert np.allclose(box.hi, sp.max(axis=0) + pad, atol=1e-6)
    f = impl.features(state)
    assert np.allclose(f["start"], sp[0], atol=1e-6) and np.allclose(f["end"], sp[-1], atol=1e-6)
    assert "batch_input" not in e.inspect(0.0, id="c")["objects"][0]["semantic"]


# ---------------------------------------------------------------- the GL path


def _scene_of(kind: str, cam: str) -> Scene:
    s = Scene(f"b_{kind}_{cam}")
    if cam == "perspective":
        s.use_camera("orbit")
        s.set("camera.distance", 6.0)
        s.set("camera.elevation", 25.0)
        s.set("camera.azimuth", -40.0)
    s.add(_Curve(id="c", kind=kind))
    return s


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize("cam_name", sorted(CAMERAS))
@pytest.mark.parametrize(
    ("units", "anchor"),
    [("height", None), ("px", None), ("world", None), ("height", (0.3, -0.2, 0.1))],
)
def test_the_vertex_shader_equals_the_cpu_expansion(cam_name, units, anchor):
    from phasor.render.passes import default_renderer

    r = default_renderer()
    cam = CAMERAS[cam_name]()
    for name, b in _batches(units, anchor).items():
        gl = r.batch_vertices(b, cam, (640, 360))
        ref, _, valid = bt.instances(b, cam, 360)
        assert gl.shape == ref.shape, name
        scale = float(np.abs(ref[valid]).max())
        assert np.abs(gl[valid] - ref[valid]).max() <= 2e-6 * max(scale, 1.0), name


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize("cam_name", sorted(CAMERAS))
@pytest.mark.parametrize("kind", ["lines", "points", "flat", "cone"])
def test_gl_and_cpu_expanded_pictures_agree(kind, cam_name):
    e = _scene_of(kind, cam_name).engine
    r = e.pass_runner()
    try:
        r.expand_on_cpu = False
        a = e.still(0.0, (320, 180)).astype(int)
        ida = r.id_buffer(e.state_at(0.0), (320, 180)).copy()
        r.expand_on_cpu = True
        b = e.still(0.0, (320, 180)).astype(int)
        idb = r._render(e.state_at(0.0), "id", (320, 180))
    finally:
        r.expand_on_cpu = False
    lit = a.max(axis=-1) > 24
    assert lit.sum() > 50  # it drew something
    d = np.abs(a - b).max(axis=-1)
    assert d.max() <= 24 and (d > 8).mean() < 0.002, (d.max(), (d > 8).sum())
    assert (ida == idb).mean() > 0.998


def _lorenz():
    s = runpy.run_path(str(ROOT / "examples" / "lorenz_explorer.py"), run_name="x")["s"]
    return s.engine


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_orbiting_uploads_nothing_and_flow_time_uploads_one_positions_array():
    """Advice 1.4, measured: the M26 Lorenz scene. After the first frame an orbit (twelve
    azimuths) uploads no batch buffer at all; advancing the flow time uploads one array per
    batch that moves (the ball's positions), never its colours or sizes."""
    e = _lorenz()
    r = e.pass_runner()
    t = 4.0  # field, equilibria and the ball shown; the flow time is moving
    e.still(t, (320, 180))
    before = dict(r.uploads)
    for az in np.linspace(-62.0, 298.0, 12, endpoint=False)[1:]:
        f = e.fork()
        assert f.apply({"op": "set_param", "path": "camera.azimuth", "value": float(az)},
                       origin="script").ok  # fmt: skip
        f.still(t, (320, 180))
    assert r.uploads["batch"] == before["batch"]
    plan = r.frame_plan(e.state_at(t + 0.1), (320, 180))
    moving = [c[1]["batch"] for cmds in plan.commands.values() for c in cmds
              if c[0] == "batch"]  # fmt: skip
    e.still(t + 0.1, (320, 180))
    assert r.uploads["batch"] - before["batch"] == 1
    assert sum(b.kind == "points" and b.n == 2000 for b in moving) == 1


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_growing_path_uploads_only_its_positions():
    """A probe's path gains points as the flow time runs: one colour and one width are kept
    once and one strip needs no strip array, so each new frame uploads one array per batch
    (the path's points and the moving dot's)."""
    s = Scene("grow")
    s.add(flow.System(id="p", equations=("y", "-sin(x) - 0.3*y"), window=((-4, 4), (-3, 3)),
                      rate=1.0, frame=False))  # fmt: skip
    s.add(flow.Probe(id="p.probe", x0=(1.0, 0.5)))
    e = s.engine
    r = e.pass_runner()
    e.still(0.5, (160, 90))
    for t in (0.6, 0.7, 0.8):
        before = r.uploads["batch"]
        e.still(t, (160, 90))
        assert r.uploads["batch"] - before == 2, t


# ---------------------------------------------------------------- quality while dragging


def _big() -> Scene:
    s = Scene("big", passes=None)
    s.add(flow.System(id="p", equations=("y", "-sin(x)"), window=((-4, 4), (-3, 3)), tau=0.5))
    s.add(flow.FieldArrows(id="p.field", counts=120))
    s.add(flow.FlowSet(id="p.set", count=20000, size=1.0))
    return s


def test_interactive_quality_thins_the_field_and_strides_the_set():
    e = _big().engine
    state = e.state_at(0.0)
    got = {}
    for q in ("full", "interactive"):
        for oid in ("p.field", "p.set"):
            c = _Collector(quality=q)
            state[oid].impl.draw(c, state, "color")
            got[(q, oid)] = max(cmd["batch"].n for _, cmd in c.commands)
    assert got[("full", "p.field")] > flow.INTERACTIVE_ARROWS >= got[("interactive", "p.field")]
    assert got[("interactive", "p.field")] > 0.8 * flow.INTERACTIVE_ARROWS
    assert got[("full", "p.set")] == 20000
    assert got[("interactive", "p.set")] <= flow.INTERACTIVE_POINTS
    # a hint only: the document, its semantics and lint are those of the full drawing
    sem = e.inspect(0.0, id="p.field")["objects"][0]["semantic"]
    assert sem["arrows"] == 120 * 120


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_interactive_quality_is_checked_and_keys_the_frame_plan():
    from phasor.engine import Engine

    eng = Engine(_big().engine.document, passes="auto")
    r = eng.pass_runner()
    st = eng.state_at(0.0)
    full = r.frame_plan(st, (320, 180))
    fast = r.frame_plan(st, (320, 180), quality="interactive")
    assert full.key != fast.key
    with pytest.raises(ValueError, match="quality"):
        eng.still(0.0, (64, 36), quality="fastest")
    with pytest.raises(ValueError, match="quality"):
        r.start_color_raw(st, (320, 180), plan=full, quality="interactive")
    a = eng.still(0.0, (320, 180), quality="interactive")
    b = eng.still(0.0, (320, 180))
    assert a.shape == b.shape and (a != b).any()  # fewer arrows drawn while dragging
    # the inspector's passes ignore the hint
    assert (r.id_buffer(st, (320, 180)) == r._render(st, "id", (320, 180))).all()


# ---------------------------------------------------------------- bounds at the maxima


def _peak(fn):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        out = fn()
        return out, time.perf_counter() - start, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.slow
def test_batches_at_the_documented_maxima_are_bounded():
    """Points x trail at its cap (4e6 line points) and a field of ``MAX_ARROWS``: making the
    batches is bounded in memory (generous bounds against a blow-up, not timers)."""
    n = 4_000_000
    pts = np.random.default_rng(1).random((n // 64, 64, 3))
    cols = np.ones((n // 64, 64, 4))
    b, took, peak = _peak(lambda: bt.make_lines(pts, width=0.002, colors=cols)[0])
    assert b.n == n and took < 120 and peak < 1_000_000_000, (took, peak)
    arrows = np.random.default_rng(2).random((flow.MAX_ARROWS, 3))
    a, took, peak = _peak(lambda: bt.make_arrows(arrows, arrows + 0.01, width=0.003)[0])
    assert a.n == flow.MAX_ARROWS and took < 30 and peak < 50_000_000, (took, peak)


def test_a_batch_keeps_nothing_that_depends_on_the_camera():
    """The arrays a flow object hands over are the same from any camera (the property the
    buffer cache relies on)."""
    keys = []
    for az in (-62.0, 40.0):
        s = runpy.run_path(str(ROOT / "examples" / "lorenz_explorer.py"), run_name="x")["s"]
        s.set("camera.azimuth", az)
        st = s.engine.state_at(4.0)
        got = []
        for oid in ("lorenz.field", "lorenz.eq", "lorenz.ball", "lorenz"):
            c = _Collector()
            st[oid].impl.draw(c, st, "color")
            got += [cmd["batch"].frame_key() for kind, cmd in c.commands if kind == "batch"]
        keys.append(got)
    assert keys[0] == keys[1] and len(keys[0]) >= 4


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_live_view_bench_reports_frame_rates_and_uploads():
    """``phasor bench --explorer``'s measurement on a small case: an orbit uploads nothing,
    flow frames upload the moving points; the CPU-expanded run uploads no batch at all."""
    from phasor.cli import _explorer_cases
    from phasor.render.bench import run_view_bench

    assert len(_explorer_cases()) == 4
    s = Scene("bench_small")
    s.use_camera("orbit")
    s.add(flow.System(id="lo", scale=0.12, rate=1.0, **flow.preset("lorenz")))
    s.add(flow.FieldArrows(id="lo.field", counts=5))
    s.add(flow.FlowSet(id="lo.set", shape="ball", center=(1, 1, 20), size=3, count=300))
    e = s.engine
    r = run_view_bench(e, "small", 0.2, ("orbit", "flow"), width=160, height=90, count=3)
    assert r["motions"]["orbit"]["batch_uploads_per_frame"] == 0
    assert r["motions"]["flow"]["batch_uploads_per_frame"] == 1
    assert r["motions"]["orbit"]["fps"] > 0 and r["expand"] == "shader"
    c = run_view_bench(e, "small", 0.5, ("flow",), width=160, height=90, count=2,
                       cpu_expand=True)  # fmt: skip
    assert c["motions"]["flow"]["batch_uploads_per_frame"] == 0 and c["expand"] == "cpu"
    with pytest.raises(ValueError, match="motion"):
        run_view_bench(e, "small", 0.2, ("spin",), count=1)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_long_flow_keeps_few_buffers_on_the_card():
    """Each flow-time frame makes new positions; the old ones are dead and let go of after
    ``BATCH_KEEP_RENDERS`` renders, so a long render keeps a bounded handful of buffers (not
    every frame's positions), and an orbit afterwards still uploads nothing."""
    from phasor.render.passes import BATCH_KEEP_RENDERS, BATCH_SPARE, GLRenderer

    s = Scene("long_flow")
    s.use_camera("orbit")
    s.add(flow.System(id="lo", scale=0.12, rate=1.0, **flow.preset("lorenz")))
    s.add(flow.FieldArrows(id="lo.field", counts=4))
    s.add(flow.FlowSet(id="lo.set", shape="ball", center=(1, 1, 20), size=3, count=200))
    e = s.engine
    r = GLRenderer()  # its own cache, on the shared context
    t_end = 0.1 + 119 / 30
    sizes = []
    for k in range(120):
        r.start_color_raw(e.state_at(0.1 + k / 30), (96, 54)).collect()
        sizes.append(r.batch_memory()["buffers"])
    assert max(sizes[60:]) <= BATCH_KEEP_RENDERS + 12, max(sizes)
    assert r.batch_memory()["spares"] <= BATCH_SPARE
    before = r.uploads["batch"]
    for az in (10.0, 50.0, 90.0):
        f = e.fork()
        assert f.apply({"op": "set_param", "path": "camera.azimuth", "value": az},
                       origin="script").ok  # fmt: skip
        r.start_color_raw(f.state_at(t_end), (96, 54)).collect()
    assert r.uploads["batch"] == before
