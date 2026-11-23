"""Milestone 20: phasor's Manim import against REAL Manim Community, scene by scene.

Skipped unless ``PHASOR_MANIM_PYTHON`` names a Python with Manim installed (see
``pc_tasks/10_manim_real.md``); the comparison itself is ``compare.py``. What must hold,
within the tolerances documented there: Manim's clock (every play's duration, run times and
rate functions) exactly; every mobject on screen at every sample; shapes, dots, lines and
the camera frame where Manim has them. Text is set in phasor's own fonts, so its boxes (and
what is placed next to them) are compared with a wider tolerance; the differences that are
by design are listed in ``MILESTONE_20.md``.
"""

from __future__ import annotations

import pytest

from tests.manim_real import compare

pytestmark = [
    pytest.mark.manim_real,
    pytest.mark.skipif(compare.manim_python() is None,
                       reason="needs PHASOR_MANIM_PYTHON (a Python with manim installed)"),
]  # fmt: skip

SCENES = {p.stem: p for p in compare.SCRIPTS}
# Where phasor differs by design (MILESTONE_20.md, "Real Manim"): the measure is skipped.
BY_DESIGN = {
    # `.animate.rotate`: Manim mixes the points in a straight line (the square shrinks on the
    # way); phasor turns it (rotation is a parameter). Sizes differ mid-way (0.10), not after.
    "animate_timing": {"size"},
    "shapes_layout": {"size"},  # Manim's arrow tip is wider (0.35 against phasor's 0.1)
    # text set at phasor's role sizes is narrower than Manim's (owner decision, MILESTONE_20):
    # a small text put in a corner by its edge lands that much further out
    "text_write": {"text"},
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_the_import_matches_real_manim(name):
    f = compare.run_scene(SCENES[name])
    differed = " ".join(f.differed)
    assert "duration" not in differed and "play " not in differed, f.differed
    assert "never found" not in differed and "colors at the end" not in differed, f.differed
    skip = BY_DESIGN.get(name, set())
    if "position" not in skip:
        assert f.worst["position"] <= compare.POS_TOL, f.details
    if "size" not in skip:
        assert f.worst["size"] <= compare.SIZE_TOL, f.details
    if "text" not in skip:
        assert f.worst["text"] <= compare.TEXT_POS_TOL, f.details
    assert f.worst["frame"] in (0.0,) or f.worst["frame"] <= compare.POS_TOL, f.details
