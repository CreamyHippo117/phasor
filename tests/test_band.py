"""``geo.Band``: a continuous shading strip along a plot axis whose colour follows a 1D array."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, expr, geo
from tests.conftest import gl_available
from tests.test_beauty import _golden

ROOT = Path(__file__).resolve().parent.parent
F = np.geomspace(30.0, 1000.0, 120)
M = 80.0 * (F / 1000.0) ** 3
MODES = [None, pytest.param("auto", marks=[pytest.mark.gl, pytest.mark.skipif(
    not gl_available(), reason="needs OpenGL")])]  # fmt: skip


def _scene(passes=None, **kw):
    s = Scene("b", passes=passes, look="none")
    kw = {"extent": (1.0, 2.0), "scale": 2.0, **kw}
    s.add(geo.Band(id="m", values=M, positions=F, log=True, **kw))
    return s


def _bounds(e, t=0.0):
    state = e.state_at(t)
    b = state.objects[0].impl.world_bounds(state)
    return np.asarray(b.lo), np.asarray(b.hi)


def _quad(e, t=0.0):
    """What the band's ``draw`` hands the renderer: the quad and its pixels function."""
    state = e.state_at(t)
    seen = {}

    class Ctx:
        def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0, clamp=False):
            seen.update(corner=corner, right=right, up=up, pixels=pixels, glow=glow)

    state.objects[0].impl.draw(Ctx(), state, "color")
    return seen


def test_semantics_bounds_and_features_follow_the_mapping():
    e = _scene().engine
    sem = e.inspect(0.0, id="m")["objects"][0]["semantic"]
    assert sem["kind"] == "band" and sem["backdrop"] is True
    assert sem["values"] == pytest.approx([M.min(), 80.0])
    lo, hi = _bounds(e)
    assert lo[1] == pytest.approx(np.log10(F[0]) * 2.0)
    assert hi[1] == pytest.approx(np.log10(F[-1]) * 2.0)
    assert lo[0] == pytest.approx(1.0) and hi[0] == pytest.approx(2.0)


def test_offset_slides_the_band_and_costs_nothing():
    e = _scene(pos=expr("vec(0, 0.5 * t, 0)")).engine
    (a_lo, a_hi), (b_lo, b_hi) = _bounds(e, 0.0), _bounds(e, 2.0)
    assert b_lo[1] - a_lo[1] == pytest.approx(1.0)
    assert b_hi[1] - a_hi[1] == pytest.approx(1.0)
    assert e.lint() == []


def test_validation_refuses_bad_arrays():
    bad = [dict(values=[1.0]), dict(values=[1, 2], positions=[1.0]),
           dict(values=[1, 2], positions=[2.0, 1.0]),
           dict(values=[1, 2], positions=[0.0, 1.0], log=True), dict(values=[1, 2], log=True),
           dict(values=[1, 2], extent=(1.0, 1.0)), dict(values=[1, 2], value_range=(2.0, 1.0))]  # fmt: skip
    for kw in bad:
        with pytest.raises(ValueError):
            geo.Band.Params(**kw)


@pytest.mark.parametrize("axis", ["y", "x"])
def test_the_gradient_is_smooth_and_follows_the_array(axis):
    """No steps: neighbouring pixels along the band differ by little, and the brightness
    rises from the quiet end to the strong end with the values."""
    e = _scene(axis=axis, extent=(0.0, 1.0)).engine
    q = _quad(e)
    n, length = 400, (q["up"][1] if axis == "y" else q["right"][0])
    if axis == "y":
        px = q["pixels"](8, n, (0.5, length / 2), length)
        line = px[::-1, 4, :3].sum(-1)  # row 0 is the top: reverse to run low to high
    else:
        px = q["pixels"](n, 8, (length / 2, 0.5), length * 8 / n)
        line = px[4, :, :3].sum(-1)
    strength = line - np.asarray(e.state_at(0.0).background)[:3].sum()
    assert np.abs(np.diff(strength)).max() < 0.04 * np.abs(strength).max() + 1e-9  # smooth
    assert strength[-1] > 20 * max(strength[0], 1e-6)  # near 0 at 30 Hz, strong at 1 kHz
    assert np.all(np.diff(strength) >= -1e-6)  # M rises with f


def test_sliding_the_offset_slides_the_quad_and_keeps_the_picture():
    a, b = (_quad(_scene(pos=(0.0, d, 0.0)).engine) for d in (0.0, 3.0))
    np.testing.assert_allclose(b["corner"] - a["corner"], [0, 3, 0])
    np.testing.assert_allclose(b["up"], a["up"])
    view = ((0.5, a["up"][1] / 2), a["up"][1])
    np.testing.assert_array_equal(a["pixels"](16, 64, *view), b["pixels"](16, 64, *view))


def test_colormap_mode_and_edge_fade():
    e = _scene(mode="colormap", edge_fade=0.25, extent=(0.0, 1.0)).engine
    q = _quad(e)
    h = q["up"][1]
    px = q["pixels"](33, 33, (0.5, h / 2), h)
    bg = np.asarray(e.state_at(0.0).background)[:3]
    np.testing.assert_allclose(px[5, 0, :3], bg, atol=1e-6)  # the edge melts into the background
    assert not np.allclose(px[5, 16, :3], bg)
    assert px.dtype == np.float32 and px.shape == (33, 33, 4)


def test_the_example_lints_clean():
    e = runpy.run_path(str(ROOT / "examples" / "band.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize("passes", MODES[1:])
def test_band_golden_storyboard(passes):
    e = runpy.run_path(str(ROOT / "examples" / "band.py"), run_name="x")["s"].engine
    sheet = e.storyboard([0.0, 2.0, 4.0], size=(320, 180))["image"]
    _golden("band", sheet, e.pass_runner().renderer_class())
