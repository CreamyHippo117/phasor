"""Milestone 28 audit findings (2026-11-24), one regression test each: every test here failed
on the audited commit and passes after the fix.

B1 a batch's recorded bounds include its sizes (not only item centres);
S1 values that overflow float32 are repaired and noted, and lint sees repairs of every type;
S2 spare storage buffers shrink and count against the byte cap;
S3 a line receding along the view agrees between the shader and the CPU;
S5 storage-buffer limits are read, and a batch too large for them is drawn another way;
nits: 4-D line input, colours cut with the points, ``ctx.quality`` in the protocol.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from phasor import Scene
from phasor.core import batches as bt
from phasor.core.objects import DrawContext, ObjectBase, ParamsBase, register
from phasor.domains import strokes
from tests.conftest import gl_available

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]
GL = [pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))


@register
class _Dots(ObjectBase):
    type: ClassVar[str] = "m28a_dots"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        case: str = "bigdot"

    def draw(self, ctx, state):
        k = self.me(state).params.case
        cam = state.camera_for(self.me(state))
        if k == "bigdot":
            ctx.points3d(np.zeros((1, 3)), radius=0.3, colors=(0.2, 0.9, 0.3, 1.0))
        elif k == "bigdot_strokes":  # the control: the stroke path's disc
            strokes.disc_batch(ctx, cam, np.zeros((1, 3)), 0.3 * cam.units_per_height(),
                               colors=(0.2, 0.9, 0.3, 1.0))  # fmt: skip
        elif k == "row":
            x = np.linspace(-3, 3, 50)
            ctx.points3d(np.c_[x, 0 * x, 0 * x], radius=0.03, colors=(1.0, 0.5, 0.2, 1.0))
        elif k == "edge":  # a big dot centred on the right edge of the frame
            x = cam.height * 16 / 9 / 2
            ctx.points3d(np.array([[x, 0.0, 0.0]]), radius=0.3, colors=(0.2, 0.9, 0.3, 1.0))
        elif k == "line":
            x = np.linspace(-3, 3, 50)
            ctx.lines3d(np.c_[x, 0 * x, 0 * x], width=0.05)
        elif k == "arrow":
            ctx.arrows3d(np.zeros((1, 3)), np.array([[2.0, 0.0, 0.0]]), width=0.05)


def _seen(case, passes):
    s = Scene(f"b1_{case}", passes=passes)
    s.add(_Dots(id="probe", case=case))
    e = s.engine
    return e, e.inspect(0.0, id="probe")["objects"][0]


@pytest.mark.parametrize("passes", MODES)
def test_b1_a_dot_drawn_as_a_batch_has_its_size_in_its_bounds(passes):
    e, o = _seen("bigdot", passes)
    _, ctl = _seen("bigdot_strokes", passes)
    x0, y0, x1, y1 = o["bbox"]["px"]
    c0, d0, c1, d1 = ctl["bbox"]["px"]
    assert x1 - x0 == pytest.approx(c1 - c0, rel=0.05) and y1 - y0 == pytest.approx(
        d1 - d0, rel=0.05
    )
    assert o["on_screen_fraction"] == pytest.approx(1.0, abs=0.01)
    assert "orphan" not in {i.code for i in e.lint()}


@pytest.mark.parametrize("passes", MODES)
@pytest.mark.parametrize("case", ["row", "line", "arrow"])
def test_b1_rows_lines_and_arrows_are_seen(case, passes):
    e, o = _seen(case, passes)
    x0, y0, x1, y1 = o["bbox"]["px"]
    assert y1 - y0 > 2 and x1 - x0 > 2
    assert o["on_screen_fraction"] > 0.9
    assert "orphan" not in {i.code for i in e.lint()}


@pytest.mark.parametrize("passes", MODES)
def test_b1_a_dot_on_the_frame_edge_is_half_on_screen(passes):
    _, o = _seen("edge", passes)
    assert 0.35 < o["on_screen_fraction"] < 0.65


# ---------------------------------------------------------------- S1


def test_s1_values_that_overflow_float32_are_repaired_and_noted():
    b, issues = bt.make_points([[1e39, 0, 0], [0, 0, 0]], radius=0.01)
    assert b.n == 1 and np.isfinite(b.positions).all() and issues
    b, issues = bt.make_lines([[0, 0, 0], [1e39, 0, 0], [1, 1, 0], [2, 2, 0]], width=0.01)
    assert np.isfinite(b.positions).all() and issues
    b, issues = bt.make_arrows([[0, 0, 0], [1, 0, 0]], [[1e38, 0, 0], [2, 0, 0]], width=0.01)
    assert b.n == 1 and issues  # a length the card cannot square is not drawn
    b, issues = bt.make_points([[0, 0, 0]], radius=1e39, colors=(1e39, 0, 0, 1))
    assert np.isfinite(b.sizes).all() and np.isfinite(b.colors).all() and issues


@register
class _OwnSemantics(ObjectBase):
    """A type that writes its own semantics (as the flow types do) and hands a batch bad
    numbers: lint must still say so."""

    type: ClassVar[str] = "m28a_own"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        pass

    def semantics(self, state):
        return {"kind": "own"}

    def draw(self, ctx, state):
        ctx.points3d(np.array([[np.nan, 0, 0], [0, 0, 0], [1e39, 0, 0]]), radius=0.02)


@pytest.mark.parametrize("passes", MODES)
def test_s1_lint_sees_batch_repairs_whatever_the_semantics(passes):
    s = Scene("own", passes=passes)
    s.add(_OwnSemantics(id="own"))
    assert "batch_input" in {i.code for i in s.engine.lint()}


# ---------------------------------------------------------------- S2


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_s2_spare_buffers_shrink_and_count_against_the_cap():
    from phasor.render import passes as P
    from phasor.render.glcontext import on_gl_thread

    r = P.GLRenderer()

    def up(b):
        def run():
            with r.ctx:
                r._batch_used.clear()
                r._renders += 1
                for name in b.arrays():
                    r._batch_buffer(b, name)

        on_gl_thread(run)

    big, _ = bt.make_points(np.random.default_rng(0).standard_normal((1_500_000, 3)), radius=0.01)
    up(big)
    for k in range(P.BATCH_KEEP_RENDERS + 8):
        up(bt.make_points(np.full((3, 3), float(k)), radius=0.01 + k * 1e-4)[0])
    largest = max(x.size for x in r._batch_cache.values())
    assert largest < 2**20, largest  # no 16-byte array in a buffer of megabytes
    mem = r.batch_memory()
    assert mem["bytes"] < 4 * 2**20, mem  # the spares count, and do not keep the big one


# ---------------------------------------------------------------- S3


@register
class _Receding(ObjectBase):
    type: ClassVar[str] = "m28a_receding"
    Life: ClassVar = frozenset()

    class Params(ParamsBase):
        pass

    def draw(self, ctx, state):
        cam = state.camera_for(self.me(state))
        t = np.linspace(-3, 20, 60)[:, None]
        line = cam.eye + cam.forward * t + cam.right * 0.3 + cam.up * 0.1
        ctx.lines3d(line, width=0.01, colors=(1, 0.5, 0.2, 1))


def _receding_scene():
    s = Scene("receding")
    s.use_camera("orbit")
    s.set("camera.distance", 9.0)
    s.set("camera.elevation", 20.0)
    s.add(_Receding(id="n"))
    return s


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_s3_a_line_along_the_view_agrees_between_shader_and_cpu():
    from phasor.render.passes import _Collector

    e = _receding_scene().engine
    r = e.pass_runner()
    st = e.state_at(0.0)
    c = _Collector()
    st["n"].impl.draw(c, st, "color")
    b = c.commands[0][1]["batch"]
    cam = st.camera
    gl = r.batch_vertices(b, cam, (320, 180))
    ref, _, valid = bt.instances(b, cam, 180)
    depth = (ref[..., :] - cam.eye) @ cam.forward
    front = valid & (depth.min(axis=1) > 0.5)  # in front of the eye
    scale = max(float(np.abs(ref[front]).max()), 1.0)
    assert np.abs(gl[front] - ref[front]).max() <= 2e-5 * scale
    try:
        a = e.still(0.0, (320, 180)).astype(int)
        r.expand_on_cpu = True
        cpu = e.still(0.0, (320, 180)).astype(int)
    finally:
        r.expand_on_cpu = False
    d = np.abs(a - cpu).max(-1)
    assert (d > 8).sum() <= 20, int((d > 8).sum())


# ---------------------------------------------------------------- S5


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_s5_storage_limits_are_read_and_a_batch_past_them_draws_on_the_cpu(monkeypatch):
    from phasor.render import passes as P

    r = P.GLRenderer()
    lim = r.storage_limits()
    assert lim["block_bytes"] >= 2**24 and lim["vertex_blocks"] >= 4
    s = Scene("s5")
    s.add(_Dots(id="probe", case="row"))
    e = s.engine
    state = e.state_at(0.0)
    want = r.start_color_raw(state, (160, 90)).collect()[0].copy()
    monkeypatch.setattr(r, "_limits", {"block_bytes": 256, "vertex_blocks": 4})
    before = r.uploads["batch"]
    got = r.start_color_raw(state, (160, 90)).collect()[0]
    assert r.uploads["batch"] == before  # nothing went to storage buffers
    assert np.abs(np.asarray(got, float) - np.asarray(want, float)).max() < 1e-2
    monkeypatch.setattr(r, "_limits", {"block_bytes": 2**30, "vertex_blocks": 0})
    got = r.start_color_raw(state, (160, 90)).collect()[0]
    assert np.abs(np.asarray(got, float) - np.asarray(want, float)).max() < 1e-2


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_s5_doctor_reports_the_storage_limits():
    from phasor.render import doctor

    gl = doctor.check_opengl()
    assert gl.data["storage_block_bytes"] >= 2**24 and gl.data["vertex_storage_blocks"] >= 4
    line = doctor.check_batches(gl)
    assert line.name == "batches" and line.ok and "MB" in line.detail
    small = doctor.Check("opengl", True, "", {**gl.data, "vertex_storage_blocks": 0})
    assert doctor.check_batches(small).ok is False


# ---------------------------------------------------------------- nits


def test_nit_lines_of_more_dimensions_are_strips_of_the_last_but_one():
    pts = np.random.default_rng(0).random((2, 3, 5, 3))
    b, _ = bt.make_lines(pts, width=0.01)
    assert b.strips is not None and int(b.strips.max()) == 5


def test_nit_colours_are_cut_with_the_points(monkeypatch):
    monkeypatch.setattr(bt, "MAX_ITEMS", 10)
    cols = [[0.5, 0.25, 0.125, 1.0]] * 20
    b, issues = bt.make_points(np.zeros((20, 3)) + np.arange(20)[:, None], radius=0.01,
                               colors=cols)  # fmt: skip
    assert b.n == 10 and np.allclose(b.colors[:, 0], 0.5)
    assert not [i for i in issues if "default is used" in i]
    b, issues = bt.make_lines(np.zeros((4, 5, 3)) + np.arange(5)[None, :, None], width=0.01,
                              colors=np.full((4, 5, 4), 0.5))  # fmt: skip
    assert np.allclose(b.colors[:, 0], 0.5) and not [i for i in issues if "default" in i]


def test_nit_the_protocol_names_the_quality_hint():
    assert "quality" in DrawContext.__annotations__
