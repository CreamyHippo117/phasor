"""The beauty plan (BEAUTY.md): the prism look carried to more of phasor."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cx
from phasor.render.gpu import ComplexFunction, Tile, get
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _golden(name: str, sheet: np.ndarray, klass: str, threshold: float = 0.98) -> None:
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    path = GOLDEN / klass / f"{name}.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= threshold


# ---------------------------------------------------------------- 1. prism domain coloring


def test_the_glow_weight_is_one_at_a_zero_and_half_at_the_radius():
    """For f(z) = z the Newton step |f / f'| is exactly the distance to the zero, so the
    weight is 1 / (1 + (d / r)^2): 1 at the zero, 1/2 at d = r, small far away."""
    k = get("domain_glow").reference
    f = ComplexFunction.parse("z")
    tile = Tile.full(201, 201)  # odd: the middle pixel sits on the zero
    out = k(tile, f, (0.0, 0.0), 4.0, radius=0.5, ambient=0.0, contour=0.0)
    w = out[..., 3]
    px = 4.0 / 201
    assert w[100, 100] == pytest.approx(1.0)
    col = 100 + round(0.5 / px)  # a pixel about r to the right
    d = (col - 100) * px
    assert w[100, col] == pytest.approx(1 / (1 + (d / 0.5) ** 2), rel=1e-3)
    assert w[100, 200] < 0.07
    assert out[..., :3].max() <= 1.0  # paint stays in range; the light comes from the glow


def test_a_pole_glows_like_a_zero():
    """1/f has the same Newton step as f, so a pole lights up just like a zero."""
    k = get("domain_glow").reference
    tile = Tile.full(101, 101)
    zero = k(tile, ComplexFunction.parse("z"), (0.0, 0.0), 2.0)[..., 3]
    pole = k(tile, ComplexFunction.parse("1/z"), (0.0, 0.0), 2.0)[..., 3]
    ring = np.ones_like(zero, dtype=bool)
    ring[45:56, 45:56] = False  # away from the centre pixel itself
    np.testing.assert_allclose(pole[ring], zero[ring], rtol=5e-3)  # finite differences


def test_rays_and_rings_are_a_pixel_wide_at_any_zoom():
    k = get("domain_glow").reference
    f = ComplexFunction.parse("z")
    for height in (4.0, 0.01):
        out = k(Tile.full(240, 240), f, (height / 3, height / 5), height, radius=1e-9,
                ambient=0.0, contour=0.0, lines=8)  # fmt: skip
        lit = out[..., 3] > 0.3
        assert 0.01 < lit.mean() < 0.15, height  # thin lines, not a wash or nothing


def test_the_prism_preset_and_looks_pick_the_glowing_picture():
    s = Scene("p", passes=None)
    s.add(cx.DomainColoring(id="a", function="z", preset="prism"))
    s.add(cx.DomainColoring(id="b", function="z"))
    doc = {o.id: o for o in s.engine.document.objects}
    assert doc["a"].params["radius"] > 0 and doc["a"].params["lines"] > 0
    assert doc["b"].params.get("radius", 0) == 0  # the classic picture stays the default
    for look in ("prism", "prism_everywhere"):
        t = Scene("q", look=look, passes=None)
        t.add(cx.DomainColoring(id="c", function="z"))
        assert t.engine.document.object("c").params["radius"] > 0, look


@needs_gl
def test_the_prism_picture_emits_light_and_the_classic_one_does_not():
    def emission(**kw):
        s = Scene("e")
        s.set("camera.height", 4.5)
        s.add(cx.DomainColoring(id="p", function="z**2 - 1", size=(8, 4.5), **kw))
        r = s.engine.pass_runner()
        _, emit = r.color_and_emission(s.engine.state_at(0.0), (160, 90))
        return emit[..., :3]

    assert emission().max() == 0.0  # as before: kernel pictures emit nothing by default
    glow = emission(preset="prism")
    assert glow.max() > 0.5
    hot = np.argwhere(glow.sum(-1) > glow.sum(-1).max() * 0.5)[:, 1]
    assert ((hot > 50) & (hot < 110)).all()  # the light sits round the zeros at +-1


@needs_gl
@pytest.mark.slow
def test_prism_domain_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "prism_domain.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.6, 2.0, 6.0], size=(320, 180))["image"]
    _golden("prism_domain", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- 2. heat maps


def _old_heatmap(v, lut, size, pw, ph, center, height):
    """The plain picture as it was drawn before the shared helper (kept to check it)."""
    nx, ny = v.shape
    sx, sy = size
    half_w = height * pw / ph / 2
    xs = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w
    ys = center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height
    gx = np.clip(xs / sx * (nx - 1), 0, nx - 1)
    gy = np.clip(ys / sy * (ny - 1), 0, ny - 1)
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
    fx, fy = (gx - x0)[None, :], (gy - y0)[:, None]
    val = (v[x0][:, y0].T * (1 - fx) * (1 - fy) + v[x1][:, y0].T * fx * (1 - fy)
           + v[x0][:, y1].T * (1 - fx) * fy + v[x1][:, y1].T * fx * fy)  # fmt: skip
    return lut[np.clip((val * (len(lut) - 1)).astype(int), 0, len(lut) - 1)]


def test_the_plain_heat_map_is_exactly_what_it_was():
    from phasor.domains.heatmap import heat_pixels, lut_colour

    rng = np.random.default_rng(3)
    v = rng.random((17, 11))
    lut = rng.random((64, 3)).astype(np.float32)
    px = heat_pixels(v, (4.0, 3.0), lut_colour(lut))(80, 60, (2.0, 1.5), 3.0)
    np.testing.assert_array_equal(px[..., :3], _old_heatmap(v, lut, (4.0, 3.0), 80, 60,
                                                            (2.0, 1.5), 3.0))  # fmt: skip
    assert (px[..., 3] == 1).all()


def test_smooth_sampling_passes_through_the_grid_and_has_no_kinks():
    from phasor.domains.heatmap import sample

    x = np.linspace(0, 1, 9)
    grid = np.sin(3 * x)[:, None] * np.ones((1, 4))
    g = np.arange(9.0)
    np.testing.assert_allclose(sample(grid, g, np.zeros(1), "smooth")[0], grid[:, 0])
    fine = np.linspace(0, 8, 400)
    smooth = sample(grid, fine, np.zeros(1), "smooth")[0]
    linear = sample(grid, fine, np.zeros(1), "linear")[0]
    kinks = lambda y: np.abs(np.diff(y, 2)).max()  # noqa: E731
    assert kinks(smooth) < kinks(linear) / 5  # no corners at the grid points
    inner = (fine > 1) & (fine < 7)  # the ends repeat the end value: less exact there
    truth = np.sin(3 * fine / 8)
    assert np.abs(smooth - truth)[inner].max() < np.abs(linear - truth)[inner].max()


def test_isolines_are_about_a_pixel_wide_at_any_scale():
    from phasor.domains.heatmap import isoline_weight

    for n in (100, 400):
        ramp_ = np.tile(np.linspace(0, 1, n), (8, 1))  # 0..1 across n pixels
        lines = isoline_weight(ramp_, 4)[4]
        lit = lines > 0.5
        assert lit.sum() == pytest.approx(3 * 2, abs=3)  # 3 inner levels, about 2 px each
        assert lines[n // 4 - 2 : n // 4 + 3].max() > 0.9


def test_the_prism_heat_preset_and_looks():
    from phasor.domains import pde

    s = Scene("h", passes=None)
    s.add(pde.Field(id="a", kind="heat", preset="prism"))
    s.add(pde.Field(id="b", kind="heat"))
    assert s.engine.document.object("a").params["isolines"] > 0
    assert not s.engine.document.object("b").params.get("smooth", False)
    t = Scene("look", look="prism", passes=None)
    t.add(pde.Field(id="c", kind="wave"))
    c = t.engine.document.object("c")
    assert c.params["smooth"] and c.style["glow"] == "soft"


@needs_gl
def test_a_prism_heat_map_glows_where_hot_and_melts_into_the_background():
    from phasor.domains import pde

    def frame(**kw):
        s = Scene("glow")
        s.set("camera.height", 4.0)
        s.add(pde.Field(id="f", kind="heat", mode=(0, 0), time=0.02, scale=0.3,
                        size=(3.0, 3.0), **kw))  # fmt: skip
        r = s.engine.pass_runner()
        pic, emit = r.color_and_emission(s.engine.state_at(0.0), (120, 90))
        return s, pic[..., :3], emit[..., :3]

    s, _, emit = frame()
    assert emit.max() == 0.0
    s, pic, emit = frame(preset="prism")
    assert emit[45, 60].sum() > 0.1  # the hot middle shines
    corner = pic[10, 30]  # inside the plate, far from the heat
    assert np.abs(corner - np.asarray(s.engine.state_at(0.0).background[:3])).max() < 0.05


@needs_gl
@pytest.mark.slow
def test_prism_heat_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "prism_heat.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.5, 3.0, 7.0], size=(320, 180))["image"]
    _golden("prism_heat", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- 5. pictures


def _png(path: Path, w: int = 40, h: int = 20, alpha: bool = False) -> Path:
    from PIL import Image as PILImage

    a = np.zeros((h, w, 4), dtype=np.uint8)
    a[..., 0] = 255  # red on the left half, blue on the right
    a[:, w // 2 :, 0], a[:, w // 2 :, 2] = 0, 255
    a[..., 3] = 255
    if alpha:
        a[: h // 2, :, 3] = 0  # the top half transparent
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(a, "RGBA").save(path)
    return path


def test_a_picture_is_found_next_to_the_scene_and_keeps_its_shape(tmp_path):
    from phasor import Engine, Image
    from phasor.core.schema import Document, Meta

    _png(tmp_path / "art" / "flag.png", 40, 20)
    s = Scene("pic", passes=None)
    s.engine = Engine(Document(meta=Meta(title="pic")), passes=None, base_dir=tmp_path)
    s.add(Image(id="flag", path="art/flag.png", width=4.0))
    sem = s.engine.inspect(0.0, id="flag")["objects"][0]["semantic"]
    assert sem["pixels"] == [40, 20] and sem["size"] == [4.0, 2.0]
    b = s.engine.state_at(0.0)["flag"].impl.world_bounds(s.engine.state_at(0.0))
    np.testing.assert_allclose([b.lo, b.hi], [(-2, -1, 0), (2, 1, 0)])


def test_a_missing_picture_is_refused_with_where_it_looked(tmp_path):
    from phasor import Engine, Image
    from phasor.core.issues import DocumentError
    from phasor.core.schema import Document, Meta

    s = Scene("pic", passes=None)
    s.engine = Engine(Document(meta=Meta(title="pic")), passes=None, base_dir=tmp_path)
    with pytest.raises(DocumentError, match=r"missing_file.*no file 'nope.png'"):
        s.add(Image(id="x", path="nope.png"))
    with pytest.raises(DocumentError, match="notes.txt"):
        s.add(Image(id="y", path="notes.txt"))


@needs_gl
def test_a_picture_draws_its_pixels_and_transparency_shows_the_background(tmp_path):
    from phasor import Engine, Image
    from phasor.core.schema import Document, Meta

    _png(tmp_path / "half.png", 40, 20, alpha=True)
    s = Scene("draw")
    s.engine = Engine(Document(meta=Meta(title="draw")), base_dir=tmp_path)
    s.set("camera.height", 2.0)
    s.add(Image(id="p", path="half.png", width=4.0, height=2.0))
    pic = s.engine.pass_runner().color(s.engine.state_at(0.0), (160, 90))[..., :3]
    bg = np.asarray(s.engine.state_at(0.0).background[:3])
    left, right = pic[70, 40], pic[70, 120]  # the opaque bottom half
    assert left[0] > 0.8 and left[2] < 0.1 and right[2] > 0.8 and right[0] < 0.1
    np.testing.assert_allclose(pic[20, 40], bg, atol=0.02)  # transparent: the background


def test_the_ai_cannot_show_a_picture_from_outside_its_folder(tmp_path):
    from phasor.workspace import ToolError, Workspace

    _png(tmp_path / "outside.png")
    root = tmp_path / "project"
    _png(root / "inside.png")
    ws = Workspace(root)
    ws.scene_new("pics")
    ok = ws.scene_edit({"op": "add_object", "object": {"id": "a", "type": "image",
                        "params": {"path": "inside.png"}}})  # fmt: skip
    assert ok["ok"]
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.scene_edit({"op": "add_object", "object": {"id": "b", "type": "image",
                       "params": {"path": "../outside.png"}}})  # fmt: skip
    assert [o.id for o in ws.engine.document.objects] == ["a"]  # the edit was taken back


@needs_gl
@pytest.mark.slow
def test_picture_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "picture.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.8, 2.6, 5.0], size=(320, 180))["image"]
    _golden("picture", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- the prism default


def test_new_scenes_get_the_prism_look_unless_they_say_otherwise(monkeypatch, tmp_path):
    from phasor.workspace import Workspace

    monkeypatch.delenv("PHASOR_DEFAULT_LOOK", raising=False)  # as outside the test suite
    assert Scene("a", passes=None).engine.document.meta.look == "prism"
    assert Scene("b", look="calm", passes=None).engine.document.meta.look == "calm"
    plain = Scene("c", look="none", passes=None).engine.document.meta
    assert plain.look is None and plain.theme == "midnight_glow"
    themed = Scene("d", theme="paper", passes=None).engine.document.meta
    assert themed.look is None and themed.theme == "paper"  # a chosen theme is kept as is
    ws = Workspace(tmp_path)
    ws.scene_new("e")
    assert ws.engine.document.meta.look == "prism"
    ws.scene_new("f", look="none")
    assert ws.engine.document.meta.look is None


def test_palettes_are_smooth_and_pass_through_their_colours():
    from phasor.core.color import hex_to_linear, linear_to_oklab
    from phasor.style.theme import load_theme

    theme = load_theme("prism")
    for name, stops in (("gold", ["#ffe3a3", "#ffc061", "#ff9a5c", "#ffd27a"]),
                        ("rose", ["#ffb3c7", "#ff7aa2", "#c38bff", "#ff9a8a"])):  # fmt: skip
        lut = np.asarray(theme.values["colormap"][name])
        lab = linear_to_oklab(lut)
        bend = np.abs(np.diff(np.vstack([lab, lab[:2]]), 2, axis=0)).max()
        assert bend < 1e-3, name  # no corners anywhere, the seam included (straight: ~4e-3)
        for k, h in enumerate(stops):  # it passes through every stop
            np.testing.assert_allclose(lut[k * len(lut) // len(stops)], hex_to_linear(h)[:3],
                                       atol=2e-3)  # fmt: skip
