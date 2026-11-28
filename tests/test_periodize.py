"""Poisson summation: calc.Periodize. The numerics against the closed form, the pieces against f,
the object in a flat scene, a 3D scene and on the overlay, and a storyboard."""

from __future__ import annotations

import math
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, calc, expr
from phasor.domains import calc_math as cm
from tests.conftest import gl_available

BUMP = "exp(-(x-1.3)^2/9.68)"  # a Gaussian of variance 4.84 centred at 1.3
P = 10.0
COPIES = (-2, 3)


def F(src=BUMP):
    return cm.RealFunction.parse(src)


def sem(e, oid="p", t=0.0):
    return e.inspect(t, id=oid)["objects"][0]["semantic"]


def gaussian_ft(xi, s=9.68, a=1.3):
    """The Fourier transform of exp(-(x-a)^2/s): sqrt(pi s) exp(-pi^2 s xi^2) exp(-2 pi i a xi)."""
    return math.sqrt(math.pi * s) * np.exp(-(math.pi**2) * s * xi**2) * np.exp(-2j * np.pi * a * xi)


def test_the_coefficients_of_the_periodized_bump_are_samples_of_its_transform():
    """Poisson summation: c_k of g = sum_j f(x - jP) is fhat(k/P) / P."""
    ks = np.arange(-6, 7)
    ck = cm.fourier_coefficients(F(), P, COPIES, 0.0, ks)
    fk = cm.fourier_transform(F(), P, COPIES, 0.0, ks / P) / P
    assert np.max(np.abs(ck - fk)) < 1e-12
    # and both are the closed form (the five copies leave out a negligible tail of the bump)
    assert np.max(np.abs(ck - gaussian_ft(ks / P) / P)) < 1e-9


def test_the_identity_holds_for_other_cells_periods_and_asymmetric_copies():
    f = F("exp(-x^2) * cos(3 * x) + 0.5 * exp(-(x - 2)^2)")
    for period, copies, cell in [(6.0, (-3, 4), -1.5), (8.0, (0, 5), 2.0), (7.0, (-4, 1), 0.0)]:
        ks = np.arange(-5, 6)
        ck = cm.fourier_coefficients(f, period, copies, cell, ks)
        fk = cm.fourier_transform(f, period, copies, cell, ks / period) / period
        assert np.max(np.abs(ck - fk)) < 1e-12, (period, copies, cell)


def test_pieces_slid_into_the_cell_sum_to_g_and_at_fold_1_they_are_f():
    f = F()
    ms = cm.piece_range(COPIES)
    assert ms == [-2, -1, 0, 1, 2]
    x = np.linspace(0.0, P, 200, endpoint=False)
    g = cm.periodic_sum(f, P, COPIES, x)
    slid = 0.0
    for m in ms:
        xs, ys = cm.fold_piece(f, P, m, 0.0, 0.0)
        assert xs[0] == pytest.approx(0.0) and xs[-1] == pytest.approx(P)  # inside the cell
        slid = slid + np.interp(x, xs, ys)
    assert np.max(np.abs(slid - g)) < 5e-4  # the interpolation of the drawn pieces
    # fold = 1: every piece is f itself over its own cell
    for m in ms:
        xs, ys = cm.fold_piece(f, P, m, 0.0, 1.0)
        assert xs[0] == pytest.approx(m * P) and xs[-1] == pytest.approx((m + 1) * P)
        assert np.array_equal(ys, f(xs - 0.0 * m))  # the same values: exactly f
    # in between the pieces sit part of the way
    xs, _ = cm.fold_piece(f, P, 2, 0.0, 0.25)
    assert xs[0] == pytest.approx(2 * P - 0.75 * 2 * P)


def test_the_object_reports_equal_coefficients_and_a_tiny_tail():
    s = Scene("poisson", passes=None)
    s.add(calc.Periodize(id="p", f=BUMP, period=P, copies=COPIES, harmonics=3))
    info = sem(s.engine)
    assert info["k"] == [-3, -2, -1, 0, 1, 2, 3]
    assert info["max_mismatch"] < 1e-12
    assert info["c_k"] == info["fhat_over_P"]
    c = [complex(*v) for v in info["c_k"]]
    assert c[3] == pytest.approx(math.sqrt(math.pi * 9.68) / P, rel=1e-6)  # c_0 = mean of g
    assert c[2] == pytest.approx(c[4].conjugate())  # f is real: c_{-k} = conj(c_k)
    assert info["window"] == [-20.0, 30.0] and info["pieces"] == 5
    assert info["folded_sum_error"] < 1e-9
    assert info["tail"] < 1e-3
    assert info["series_error"] < 1e-2  # g is its seven lowest harmonics, nearly


def test_fold_animates_and_changes_nothing_in_the_numbers():
    s = Scene("poisson", passes=None)
    s.add(calc.Periodize(id="p", f=BUMP, fold=expr("t / 4")))
    e = s.engine
    a, b = sem(e, t=0.0), sem(e, t=4.0)
    assert (a["fold"], b["fold"]) == (0.0, 1.0)
    assert a["c_k"] == b["c_k"]


def test_a_bad_function_or_copy_range_is_refused():
    s = Scene("bad", passes=None)
    with pytest.raises(Exception, match="cannot use|unknown name"):
        s.add(calc.Periodize(id="p", f="__import__('os')"))
    with pytest.raises(Exception, match="between 1 and 15"):
        s.add(calc.Periodize(id="q", copies=(0, 40)))


def test_without_the_spectrum_the_picture_is_one_panel_and_the_numbers_stay():
    s = Scene("one", passes=None)
    s.add(calc.Periodize(id="p", f=BUMP, spectrum=False))
    e = s.engine
    assert sem(e)["spectrum"] is False and sem(e)["max_mismatch"] < 1e-12
    o = e.inspect(0.0, id="p")["objects"][0]
    assert o["on_screen_fraction"] == 1.0  # the figure is its extent, spectrum or not


def _flat(layer="world", **kw):
    s = Scene("poisson", look="none")
    if layer == "overlay":
        s.add(calc.Periodize(id="p", f=BUMP, layer="overlay", pos=(0.9, 0.5, 0.0),
                             extent=(1.5, 0.8), **kw))  # fmt: skip
    else:
        s.add(calc.Periodize(id="p", f=BUMP, extent=(13.0, 7.0), **kw))
        s.set("camera.height", 8)
    return s


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize("variant", ["world", "overlay", "orbit"])
def test_it_draws_in_a_flat_scene_on_the_overlay_and_in_3d(variant):
    s = _flat("overlay" if variant == "overlay" else "world")
    if variant == "orbit":
        s.use_camera("orbit", azimuth=-50, elevation=30, distance=14, drift=0)
    e = s.engine
    img = np.asarray(e.storyboard([0.0], size=(320, 180))["image"])
    assert img.std() > 5  # something was drawn
    assert e.lint() == []


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_example_lints_clean_and_matches_its_golden():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    root = Path(__file__).parents[1]
    e = runpy.run_path(str(root / "examples" / "periodize.py"), run_name="x")["s"].engine
    assert e.lint() == [] and e.critique()["issues"] == []
    path = GOLDEN / e.pass_runner().renderer_class() / "periodize.png"
    sheet = e.storyboard([0.0, 2.0, 4.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.98
