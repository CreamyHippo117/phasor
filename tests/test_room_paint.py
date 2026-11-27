"""A scalar painted on a room's walls (``Room3D(paint=)``): the formula reader, the wall
kernel's maths, the object, the pictures, and the example's golden storyboard. (The kernel is
also in ``CASES`` of ``tests/test_kernels.py``, which runs it three ways.)"""

from __future__ import annotations

import itertools
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac
from phasor.render.gpu import FunctionError, Tile
from phasor.render.gpu.kernels import wall
from phasor.render.passes import _Collector
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
NOTE = "(cos(pi*l*x/Lx) * cos(pi*m*y/Ly) * cos(pi*n*z/Lz))^2"
SIZE = (5.0, 4.0, 3.0)
ZERO, EX, EY, EZ = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _room(paint=NOTE, passes=None, **kw):
    s = Scene("r", look="prism", passes=passes)
    s.use_camera("orbit", azimuth=-52, elevation=24, distance=12.5, target=(2.5, 2, 1.5))
    s.add(ac.Room3D(id="room", size=SIZE, paint=paint, **kw))
    return s


def _commands(s, t=0.0):
    st = s.engine.state_at(t)
    c = _Collector()
    st["room"].impl.draw(c, st, "color")
    return [cmd for kind, cmd in c.commands if kind == "field"], st


# ---------------------------------------------------------------- the formula reader


def test_a_power_is_written_with_a_caret_and_the_names_are_the_rooms():
    f = wall.parse("x^2 + y^2 + z^2 + l*m*n + Lx/Ly - Lz")
    got = f.numpy()(1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0)
    assert got == pytest.approx(1 + 4 + 9 + 24 + 1.25 - 3)


@pytest.mark.parametrize("bad", ["x + w", "sin(x, y)", "i * x", "__import__('os')", "1/0"])
def test_a_formula_outside_the_whitelist_is_refused(bad):
    with pytest.raises(FunctionError):
        wall.parse(bad)


def test_the_room_refuses_a_bad_formula_naming_the_problem():
    for bad, why in ((dict(paint="cos(w)"), "unknown name 'w'"),
                     (dict(paint_to="x"), "paint_to needs paint"),
                     (dict(paint="x", path="assets/l_room.obj"), "shoebox")):  # fmt: skip
        with pytest.raises(Exception, match=why):
            Scene("r", passes=None).add(ac.Room3D(id="room", **bad))


# ---------------------------------------------------------------- the kernel's maths


def _wall(tile, origin, right, up, view, function=NOTE, **kw):
    return wall.wall_field_reference(tile, wall.parse(function), origin, right, up, *view, **kw)


def test_the_axial_note_is_one_on_the_end_walls_and_a_cosine_squared_along_the_sides():
    view = ((2.5, 2.0), 4.0)  # the floor, 5 x 4: 80 x 64 pixels is a pixel every 1/16 m
    floor = _wall(Tile.full(80, 64), ZERO, EX, EY, view, mode=(1, 0, 0), size=SIZE)
    x = (np.arange(80) + 0.5) / 16.0
    assert floor.shape == (64, 80) and floor.dtype == np.float32
    np.testing.assert_allclose(floor[32], np.cos(np.pi * x / 5.0) ** 2, atol=1e-6)
    assert np.ptp(floor, axis=0).max() < 1e-6  # m = 0: the same all the way up
    end = _wall(Tile.full(40, 30), ZERO, EY, EZ, ((2.0, 1.5), 3.0), mode=(1, 0, 0), size=SIZE)
    np.testing.assert_allclose(end, 1.0, atol=1e-6)  # the wall x = 0


def test_scale_divides_and_blend_is_a_straight_line_between_two_formulas():
    view, tile = ((2.5, 2.0), 4.0), Tile.full(20, 16)
    a = _wall(tile, ZERO, EX, EY, view, mode=(1, 1, 0), size=SIZE)
    b = _wall(tile, ZERO, EX, EY, view, function="x/Lx", mode=(1, 1, 0), size=SIZE)
    half = _wall(tile, ZERO, EX, EY, view, mode=(1, 1, 0), size=SIZE, scale=0.5)
    np.testing.assert_allclose(half, 2 * a, atol=1e-6)
    for k in (0.0, 0.25, 1.0):
        mix = _wall(tile, ZERO, EX, EY, view, mode=(1, 1, 0), size=SIZE, blend=k,
                    to=wall.parse("x/Lx"))  # fmt: skip
        np.testing.assert_allclose(mix, (1 - k) * a + k * b, atol=1e-6)


def test_the_kernel_agrees_with_its_reference_whatever_backend_runs_it():
    from phasor.render import gpu

    args = {"function": wall.parse(NOTE), "origin": (0.0, 4.0, 0.0), "right": EX, "up": EZ,
            "center": (2.5, 1.5), "height": 3.2, "mode": (2, 1, 1), "size": SIZE}  # fmt: skip
    tile = Tile(8, 4, 48, 40, 64, 48)
    want = gpu.run("wall_field", tile, backend="reference", **args)
    np.testing.assert_allclose(np.asarray(gpu.run("wall_field", tile, **args)), want, atol=1e-4)


# ---------------------------------------------------------------- the object


def test_without_paint_nothing_extra_is_drawn_or_said():
    fields, st = _commands(_room(paint=""))
    assert fields == []
    assert "paint" not in st["room"].impl.semantics(st)


def test_far_paints_the_walls_behind_and_all_paints_six():
    far, _ = _commands(_room())
    assert len(far) == 3  # an outside view: the three walls whose inside faces the camera
    every, _ = _commands(_room(paint_faces="all"))
    assert len(every) == 6
    sizes = sorted(tuple(sorted(np.linalg.norm([f["right"], f["up"]], axis=1))) for f in every)
    assert sizes == sorted([(3, 4), (3, 4), (3, 5), (3, 5), (4, 5), (4, 5)])


def test_the_inspector_says_where_a_note_is_strong():
    st = _room(mode=(1, 0, 0)).engine.state_at(0.0)
    mean = st["room"].impl.semantics(st)["paint"]["mean"]
    # the end walls (x_min, x_max) carry the full note; the four side walls, half of it
    assert mean["x_min"] == mean["x_max"] == pytest.approx(1.0)
    assert {mean[k] for k in ("y_min", "y_max", "z_min", "z_max")} == {0.5}
    st = _room(mode=(1, 1, 1)).engine.state_at(0.0)
    assert set(st["room"].impl.semantics(st)["paint"]["mean"].values()) == {0.25}


def test_a_pixel_is_dark_where_the_note_is_quiet_and_light_where_it_is_strong():
    fields, _ = _commands(_room(paint="x/Lx", mode=(0, 0, 0), paint_faces="all"))
    near = next(f for f in fields if np.allclose(f["corner"], 0) and f["right"][1] > 0)
    far = next(f for f in fields if np.allclose(f["corner"], (5, 0, 0)))
    # the wall x = 0 (right = y, up = z) has value 0 everywhere, the wall x = 5 value 1
    dark = np.asarray(near["pixels"](48, 36, (2.0, 1.5), 3.0))
    light = np.asarray(far["pixels"](48, 36, (2.0, 1.5), 3.0))
    assert dark.shape == (36, 48, 4) and dark[..., 3].max() < 1e-6
    assert light[..., 3].min() > 0.99 and light[..., :3].mean() > 3 * dark[..., :3].mean()


def _pixels(s):
    fields, _ = _commands(s)
    return np.asarray(fields[0]["pixels"](96, 72, (2.0, 1.5), 3.0))[..., :3]


def _worst_step(make, n):
    frames = [_pixels(make(k)) for k in np.linspace(0.0, 1.0, n + 1)]
    return max(float(np.abs(b - a).max()) for a, b in itertools.pairwise(frames))


def test_changing_the_note_never_flickers():
    """Mode numbers animate continuously and a blend between two formulas is a straight line,
    so a picture changes by an amount that shrinks with the step: no pixel jumps."""
    other = "(cos(pi*x/Lx)*cos(pi*y/Ly)*cos(pi*z/Lz))^2"
    for make in (lambda k: _room(mode=(1, k, k)),
                 lambda k: _room(paint_to=other, mode=(1, 0, 0), paint_blend=k)):  # fmt: skip
        coarse, fine = _worst_step(make, 20), _worst_step(make, 40)
        assert fine < 0.7 * coarse < 0.35  # halving the step about halves the change
        assert np.abs(_pixels(make(1.0)) - _pixels(make(0.0))).max() > 0.3  # and it does change


def test_the_example_builds_animates_and_lints_without_errors():
    e = runpy.run_path(str(ROOT / "examples" / "room_modes_paint.py"),
                       run_name="room_modes_paint")["s"].engine  # fmt: skip
    assert not [i for i in e.lint() if i.severity == "error"]
    assert e.state_at(0.0)["room"].params.mode == (1.0, 0.0, 0.0)
    assert e.state_at(6.0)["room"].params.mode == pytest.approx((1.0, 1.0, 1.0))
    assert 0.0 < e.state_at(3.0)["room"].params.mode[1] < 1.0


# ---------------------------------------------------------------- the pictures (OpenGL)

needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


@pytest.mark.gl
@needs_gl
def test_paint_changes_the_picture_and_the_six_walls_are_six_pictures():
    from phasor.render.passes import default_renderer

    plain = _room(paint="", passes="auto").engine.still(0.0, (240, 135)).astype(int)
    painted = _room(passes="auto").engine.still(0.0, (240, 135)).astype(int)
    assert np.abs(painted - plain).max() > 60  # bright bands where there were glass walls
    every = _room(paint_faces="all", passes="auto").engine
    r = default_renderer()
    r._fields.clear()
    every.still(0.0, (240, 135))
    assert len(r._fields) >= 6  # two same-sized walls are not mistaken for one another


@pytest.mark.gl
@needs_gl
def test_the_example_matches_its_golden():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = runpy.run_path(str(ROOT / "examples" / "room_modes_paint.py"),
                       run_name="room_modes_paint")["s"].engine  # fmt: skip
    path = GOLDEN / e.pass_runner().renderer_class() / "room_modes_paint.png"
    sheet = e.storyboard([0.0, 3.0, 6.0, 9.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
