"""The chroma look (owner's references, 2026-11-25): candy hues on deep indigo, domain
coloring as a luminous wash running white-hot, strokes coloured by place so a lattice wears
one rainbow, and the 3D lattice of ``TransformGrid``."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cx, linalg
from phasor.domains.strokes import DISC_SEGMENTS, lut_colors
from phasor.render.passes import _Collector, _stroke_hue
from phasor.style.theme import ThemeError, _strokes, load_theme
from tests.conftest import gl_available
from tests.test_beauty import _golden

ROOT = Path(__file__).resolve().parents[1]
CAGE = {"matrix": ((1.0, 0.3, 0.0), (0.0, 1.0, 0.4), (0.2, 0.0, 1.0)), "extent": 1,
        "lattice": True, "square": 0.0, "basis": 0.0}  # fmt: skip


def _draw(scene: Scene, oid: str, t: float = 0.0):
    st = scene.engine.state_at(t)
    c = _Collector(_stroke_hue(st, "color", st[oid]))
    st[oid].impl.draw(c, st, "color")
    return [cmd for kind, cmd in c.commands if kind == "mesh"]


def test_the_theme_starts_from_prism_and_colors_strokes_by_place():
    chroma, prism = load_theme("chroma"), load_theme("prism")
    assert chroma.background[2] > 2 * chroma.background[0]  # indigo, not black
    assert chroma.values["glow"]["soft"] == prism.values["glow"]["soft"]  # the rest inherited
    axis = np.asarray(chroma.strokes["axis"])
    assert np.isclose(np.linalg.norm(axis), 1.0) and chroma.strokes["hue_along"] > 0
    assert chroma.strokes["only"] == ()  # every glowing stroke, where prism lights only
    assert "transform_grid" in prism.strokes["only"]  # linear algebra by place
    for name in ("midnight_glow", "prism_vivid", "paper", "aurora"):
        assert "axis" not in load_theme(name).strokes  # by length, or off: unchanged


def test_prism_lights_only_the_types_it_names_so_its_lines_keep_their_own_hue():
    from phasor import Line

    s = Scene("p", look="prism", passes=None)
    s.add(Line(id="ring", points=[(0, 0, 0), (1, 1, 0), (2, 0, 0)]))
    s.add(linalg.TransformGrid(id="g", matrix=((1.0, 0.5), (0.0, 1.0))))
    st = s.engine.state_at(0.0)
    assert _stroke_hue(st, "color", st["ring"]) is None  # prism_primes' lines: unchanged
    assert len(_stroke_hue(st, "color", st["g"])) == 4  # the grid: hue by place
    assert st["g"].params.solid  # the unit square is a solid fill


def test_a_bad_strokes_mode_or_axis_is_refused():
    maps = {"spectrum": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))}
    with pytest.raises(ThemeError, match="by must be"):
        _strokes("t", {"hue_along": 0.1, "by": "time"}, maps)
    with pytest.raises(ThemeError, match="axis"):
        _strokes("t", {"hue_along": 0.1, "by": "space", "axis": [0, 0, 0]}, maps)


def test_strokes_that_meet_share_their_hue_and_nodes_match_them():
    """By place, the colour is a function of where a point is: the lattice lines through one
    crossing all have the same colour there, and so does the node drawn on it."""
    s = Scene("c", look="chroma", passes=None)
    s.add(linalg.TransformGrid(id="cage", **CAGE))
    meshes = _draw(s, "cage")
    lines = [m for m in meshes if m["colors"] is not None]
    assert lines, "the lattice lines run a hue"
    pos = np.concatenate([m["positions"] for m in lines])
    col = np.concatenate([m["colors"] for m in lines])[:, :3]
    st = s.engine.state_at(0.0)
    lut, cycles, shift, axis = _stroke_hue(st, "color", st["cage"])
    # the strip's two vertices of a point straddle it: their midpoint is the lattice point
    mid = (pos[0::2] + pos[1::2]) / 2
    want = lut_colors(lut, mid @ axis * cycles + shift)[:, :3]
    assert np.allclose(col[0::2], want, atol=1e-9)
    assert np.ptp(col, axis=0).max() > 0.3  # a rainbow across the cage, not one colour
    nodes = [m for m in meshes if m["colors"] is None]
    assert len(nodes) == 27  # (2 * extent + 1)^3 nodes
    c0 = nodes[0]["positions"][0]  # a disc's first vertex is its centre
    assert np.allclose(nodes[0]["color"][:3], lut_colors(lut, [c0 @ axis * cycles + shift])[0, :3])


def test_the_lattice_is_three_families_of_lines_and_only_in_3d():
    s = Scene("c", passes=None)
    s.add(linalg.TransformGrid(id="cage", **CAGE))
    s.add(linalg.TransformGrid(id="flat", matrix=((1.0, 0.0), (0.0, 1.0)), lattice=True))
    meshes = _draw(s, "cage")
    strips = [m for m in meshes if len(m["positions"]) != DISC_SEGMENTS + 1]
    # 3 families x 9 lines of 3 points, 2 vertices each: 24 light + 3 heavy lines
    assert sum(len(m["positions"]) for m in strips) == 27 * 3 * 2
    plane = Scene("p", passes=None)
    plane.add(linalg.TransformGrid(id="flat", matrix=((1.0, 0.0), (0.0, 1.0))))
    assert len(_draw(s, "flat")) == len(_draw(plane, "flat"))  # 2x2 ignores it
    assert s.engine.inspect(0.0, id="cage")["objects"][0]["semantic"]["dimension"] == 3


def test_the_look_gives_domain_coloring_its_white_hot_wash_and_prism_keeps_its_own():
    lit = Scene("c", look="chroma", passes=None)
    lit.add(cx.DomainColoring(id="plane", function="(z**3 - 1)/z"))
    p = lit.engine.state_at(0.0)["plane"].params
    assert p.white > 0 and p.lines == 0 and p.radius > 0
    plain = Scene("p", look="prism", passes=None)
    plain.add(cx.DomainColoring(id="plane", function="(z**3 - 1)/z"))
    assert plain.engine.state_at(0.0)["plane"].params.white == 0.0
    preset = Scene("q", passes=None)
    preset.add(cx.DomainColoring(id="plane", function="z", preset="chroma"))
    assert preset.engine.state_at(0.0)["plane"].params.white == p.white  # the look = the preset


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize(("name", "times"), [("chroma_algebra", [1.0, 3.5, 7.5]),
                                             ("chroma_domain", [1.5, 5.0, 7.5])])  # fmt: skip
def test_examples_lint_clean_and_match_their_golden(name, times):
    e = runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard(times, size=(320, 180))["image"]
    _golden(name, sheet, e.pass_runner().renderer_class())
