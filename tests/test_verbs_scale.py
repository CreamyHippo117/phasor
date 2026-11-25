"""Group S verbs: grow, shrink, squish, flatten, and the ``stretch`` every object has."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Equation, Line, Scene, Text, cx
from phasor.core.objects import (
    STRETCH_FLOOR,
    ObjectBase,
    ParamsBase,
    register,
    stretch_points,
    unregister,
)
from phasor.domains.algebra import Polygon
from phasor.domains.linalg import TransformGrid
from phasor.inspect.passes import inspect_size
from phasor.render.passes import _stretched_pixels
from phasor.time.verbs import VerbError
from tests.conftest import gl_available
from tests.test_beauty import _golden

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
IDS = ("d", "l", "p", "e", "g", "dc")


def _scene(passes=None) -> Scene:
    s = Scene("scale", passes=passes)
    s.add(Dot(id="d", pos=(-3, 1, 0), r=0.4))
    s.add(Line(id="l", points=[(-1, 1, 0), (1, 2, 0)]))
    s.add(Polygon(id="p", r=1.0, pos=(3, 1, 0)))
    s.add(Equation(id="e", parts=("$x$", "$=2$"), pos=(0, -2, 0)))
    s.add(TransformGrid(id="g", pos=(-3, -2, 0), extent=1))
    s.add(cx.DomainColoring(id="dc", function="z", pos=(3, -2, 0), size=(2, 1.5)))
    s.marker("go", 1.0)
    return s


def _box(e, oid, t):
    st = e.state_at(t)
    b = st[oid].impl.world_bounds(st)
    return np.asarray(b.lo), np.asarray(b.hi)


def _tracks(e, verb):
    return {p: segs for p, segs in e.document.timeline.tracks.items() if p.endswith(".stretch")
            and any(s.verb == verb for s in segs)}  # fmt: skip


# ---------------------------------------------------------------- the verbs


def test_every_segment_carries_its_verb_and_each_id_is_staggered():
    s = _scene()
    s.marker("g2", 3.0)
    s.marker("f", 5.0)
    s.at("go").grow("d", "l")
    s.at("g2").shrink("d")
    s.at("g2").squish("p", axis="x", to=0.5)
    s.at("f").flatten("e")
    tracks = s.engine.document.timeline.tracks
    assert [seg.verb for seg in tracks["d.stretch"]] == ["grow", "shrink"]
    assert tracks["p.stretch"][0].verb == "squish" and tracks["e.stretch"][0].verb == "flatten"
    assert tracks["l.stretch"][0].t0.offset > tracks["d.stretch"][0].t0.offset  # staggered
    assert not [i for i in s.engine.lint(strict=True) if i.code == "raw_motion"]


def test_end_is_the_theme_duration_plus_the_stagger():
    s = _scene()
    at = s.at("go").grow("d", "l", "p")
    m = s.engine._snap.compiled.theme.motion
    assert at._end == pytest.approx(2 * m["stagger"] + m["grow"]["duration"])


@pytest.mark.parametrize("oid", IDS)
def test_squish_changes_the_inspected_bounds_about_the_center(oid):
    s = _scene()
    s.at("go").squish(oid, axis="y", to=0.4)
    e = s.engine
    lo0, hi0 = _box(e, oid, 0.5)
    lo1, hi1 = _box(e, oid, 3.0)
    mid = _box(e, oid, 1.45)  # part way through
    c0 = (lo0 + hi0) / 2
    assert np.allclose((lo1 + hi1) / 2, c0, atol=1e-9)  # it stays where it is
    assert hi1[0] - lo1[0] == pytest.approx(hi0[0] - lo0[0])  # x untouched
    assert hi1[1] - lo1[1] == pytest.approx(0.4 * (hi0[1] - lo0[1]))
    assert 0.4 * (hi0[1] - lo0[1]) < mid[1][1] - mid[0][1] < hi0[1] - lo0[1]
    # what ``inspect`` reports is the same box
    bbox = e.where(oid, 3.0)["bbox"]["norm"]
    h = e.where(oid, 0.5)["bbox"]["norm"]
    assert bbox[3] - bbox[1] == pytest.approx(0.4 * (h[3] - h[1]), abs=2e-3)


def test_grow_and_shrink_keep_a_squish_and_round_trip():
    s = _scene()
    for name, t in (("a", 2.0), ("b", 3.0), ("c", 4.0), ("z", 5.0)):
        s.marker(name, t)
    s.at("go").squish("d", axis="x", to=0.5)
    s.at("a").grow("d", by=2)
    s.at("b").shrink("d", by=0.5)
    s.at("c").squish("d", axis="x", to=1.0)
    s.at("z").flatten("g").then().squish("g", axis="z", to=1.0)
    e = s.engine
    assert e.state_at(2.95)["d"].params.stretch == pytest.approx((1.0, 2.0, 2.0))
    assert e.state_at(3.95)["d"].params.stretch == pytest.approx((0.5, 1.0, 1.0))
    assert e.state_at(9.0)["d"].params.stretch == (1.0, 1.0, 1.0)
    assert _box(e, "d", 9.0)[0] == pytest.approx(_box(e, "d", 0.5)[0])
    flat = e.state_at(6.0)["g"].params.stretch
    assert flat[2] == STRETCH_FLOOR and e.state_at(9.0)["g"].params.stretch == (1.0, 1.0, 1.0)


def test_flatten_leaves_a_sliver_not_nothing():
    s = _scene()
    s.at("go").flatten("p", axis="y")
    lo, hi = _box(s.engine, "p", 3.0)
    assert 0 < hi[1] - lo[1] < 0.01


def test_the_verbs_give_the_same_scene_whatever_order_they_are_written_in():
    def build(order):
        s = _scene()
        s.marker("later", 2.5)
        steps = {
            "grow": lambda: s.at("go").grow("d"),
            "squish": lambda: s.at("go").squish("e", "dc", axis="x", to=0.5),
            "flatten": lambda: s.at("later").flatten("p", axis="y"),
            "flash": lambda: s.at("go").flash("d"),
        }
        for k in order:
            steps[k]()
        return s.engine

    a = build(["grow", "squish", "flatten", "flash"])
    b = build(["flash", "flatten", "squish", "grow"])
    for t in (0.5, 1.3, 2.9, 4.0):
        assert a.inspect(t, detail="brief") == b.inspect(t, detail="brief")


def test_bad_calls_say_what_is_wrong():
    s = _scene()
    with pytest.raises(VerbError, match="no object 'nope'"):
        s.at("go").grow("nope")
    with pytest.raises(VerbError, match="positive"):
        s.at("go").shrink("d", by=0)
    with pytest.raises(VerbError, match="'x', 'y' or 'z'"):
        s.at("go").squish("d", axis="w")
    with pytest.raises(VerbError, match="'x', 'y' or 'z'"):
        s.at("go").flatten("d", axis=2)
    with pytest.raises(VerbError, match="to="):
        s.at("go").squish("d", to=-1)
    with pytest.raises(VerbError, match="at least one"):
        s.at("go").grow()


def test_a_composite_is_refused_and_its_parts_are_named():
    @register
    class Pair(ObjectBase):
        type = "scale_test_pair"

        class Params(ParamsBase):
            pass

        def build(self, state):
            return [Dot(id="a", pos=(0, 0, 0)), Dot(id="b", pos=(1, 0, 0))]

    try:
        s = Scene("comp", passes=None)
        s.add(Pair(id="pair"))
        s.marker("go", 1.0)
        with pytest.raises(VerbError, match="pair.a"):
            s.at("go").grow("pair")
    finally:
        unregister("scale_test_pair")


# ---------------------------------------------------------------- applied centrally


def test_a_type_with_its_own_bounds_is_not_stretched_twice():
    """Bounds that read the object's own features (or a ``super()`` call) are measured
    unstretched inside the wrapper, then stretched once."""

    @register
    class Box(ObjectBase):
        type = "scale_test_box"

        class Params(ParamsBase):
            pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

        def features(self, state):
            x, y, z = self.me(state).params.pos
            return {"center": np.array([x, y, z]), "corner": np.array([x + 1, y + 1, z]),
                    "label": "not a point"}  # fmt: skip

        def world_bounds(self, state):
            from phasor.core.objects import Bounds

            f = self.features(state)  # its own (wrapped) features
            return Bounds.of_points([2 * f["center"] - f["corner"], f["corner"]])

        def draw(self, ctx, state):
            b = self.world_bounds(state)  # its own bounds, inside draw
            lo, hi = np.asarray(b.lo), np.asarray(b.hi)
            ctx.triangles(np.array([lo, [hi[0], lo[1], 0], hi]), color=(1, 1, 1, 1))

    @register
    class Box2(Box):
        type = "scale_test_box2"

        def world_bounds(self, state):
            return super().world_bounds(state)

    try:
        s = Scene("own", passes=None)
        s.add(Box(id="b", pos=(2, 0, 0)))
        s.add(Box2(id="c", pos=(-2, 0, 0)))
        s.marker("go", 1.0)
        s.at("go").squish("b", "c", axis="x", to=0.5)
        st = s.engine.state_at(5.0)
        for oid, cx_ in (("b", 2.0), ("c", -2.0)):
            b = st[oid].impl.world_bounds(st)
            assert b.lo[0] == pytest.approx(cx_ - 0.5) and b.hi[0] == pytest.approx(cx_ + 0.5)
            assert b.lo[1] == pytest.approx(-1) and b.hi[1] == pytest.approx(1)
            f = st[oid].impl.features(st)
            assert f["corner"] == pytest.approx([cx_ + 0.5, 1, 0]) and f["label"] == "not a point"
        assert type(Box2.world_bounds).__name__ == "function"
        assert getattr(Box2.features, "stretch_wrapped", False)  # inherited, not wrapped twice
        assert Box2.features is Box.features
    finally:
        unregister("scale_test_box")
        unregister("scale_test_box2")


def test_identity_stretch_changes_no_bounds_or_features_in_the_examples():
    """With stretch (1, 1, 1) the wrapped members return exactly what the type's own do."""
    for name in ("prism_primes", "motion_gallery"):
        e = runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name="x")["s"].engine
        for t in (0.5, 6.0):
            st = e.state_at(t)
            for o in st.objects:
                cls = type(o.impl)
                if not hasattr(cls.world_bounds, "__wrapped__"):
                    continue  # a composite
                assert o.impl.world_bounds(st) == cls.world_bounds.__wrapped__(o.impl, st)
                got, raw = o.impl.features(st), cls.features.__wrapped__(o.impl, st)
                assert got.keys() == raw.keys()
                for k in got:
                    assert np.array_equal(np.asarray(got[k]), np.asarray(raw[k]))


def test_a_squished_field_shows_the_same_place_in_each_share_of_its_quad():
    """The ``pixels`` callback of a stretched field sees the unstretched quad: each pixel of
    the stretched picture shows what the same share of the unstretched quad showed."""

    def where(pw, ph, center, height):  # each pixel's quad coordinates (x, y)
        width = height * pw / ph
        xs = center[0] - width / 2 + (np.arange(pw) + 0.5) / pw * width
        ys = center[1] + height / 2 - (np.arange(ph) + 0.5) / ph * height
        out = np.zeros((ph, pw, 4))
        out[..., 0], out[..., 1] = xs[None, :], ys[:, None]
        return out

    # a 4 x 2 quad squished to 4 x 1 (ku = 0.5), shown whole at 80 x 20 pixels: each pixel
    # shows the unstretched place at the same share, to within one unstretched pixel
    img = _stretched_pixels(where, 1.0, 0.5)(80, 20, (2.0, 0.5), 1.0)
    assert img.shape == (20, 80, 4)
    share_x = (np.arange(80) + 0.5) / 80
    share_y = (np.arange(20) + 0.5) / 20
    assert np.abs(img[0, :, 0] - 4 * share_x).max() <= 0.05 + 1e-9
    assert np.abs(img[:, 0, 1] - 2 * (1 - share_y)).max() <= 0.05 + 1e-9
    # squished across by 0.5 (kr): the same, the other way
    img = _stretched_pixels(where, 0.5, 1.0)(40, 40, (1.0, 1.0), 2.0)
    assert np.abs(img[0, :, 0] - 4 * (np.arange(40) + 0.5) / 40).max() <= 0.05 + 1e-9
    # flattened: the callback is asked for a bounded number of pixels
    asked = []

    def counted(pw, ph, center, height):
        asked.append(pw * ph)
        return where(pw, ph, center, height)

    flat = _stretched_pixels(counted, 1.0, STRETCH_FLOOR)(80, 2, (2.0, 0.001), 0.002)
    assert flat.shape == (2, 80, 4) and asked[0] <= 16 * 80 * 2 + 200


def test_stretch_points_scale_about_the_pivot():
    p = stretch_points([[2.0, 3.0, 1.0]], np.array([0.5, 2.0, 1.0]), np.array([1.0, 1.0, 0.0]))
    assert p.tolist() == [[1.5, 5.0, 1.0]]


# ---------------------------------------------------------------- pictures


@needs_gl
@pytest.mark.gl
def test_the_picture_agrees_with_the_inspected_bounds():
    s = _scene(passes="auto")
    s.at("go").squish(*IDS, axis="y", to=0.4)
    s.add(Text(id="t", text="hello", pos=(0, 2.5, 0)))
    e = s.engine
    st = e.state_at(4.0)
    size = inspect_size(st)
    runner = e.pass_runner()
    for oid in ("d", "l", "p", "g", "dc"):
        ys, xs = np.nonzero(runner.solo_mask(st, size, oid))
        x0, y0, x1, y1 = e.where(oid, 4.0)["bbox"]["norm"]
        assert xs.min() / size[0] == pytest.approx(x0, abs=0.02)
        assert (xs.max() + 1) / size[0] == pytest.approx(x1, abs=0.02)
        assert ys.min() / size[1] == pytest.approx(y0, abs=0.03)
        assert (ys.max() + 1) / size[1] == pytest.approx(y1, abs=0.03)


@needs_gl
@pytest.mark.gl
def test_identity_stretch_draws_the_same_pixels():
    """Setting stretch to (1, 1, 1) by hand draws exactly the unstretched picture, and a
    stretch that is not the identity changes it."""
    base = _scene(passes="auto").engine.still(0.5, (240, 135))
    s = _scene(passes="auto")
    for oid in IDS:
        s.set(f"{oid}.stretch", (1.0, 1.0, 1.0))
    assert np.array_equal(s.engine.still(0.5, (240, 135)), base)
    s.at("go").grow(*IDS, by=1.2)
    assert not np.array_equal(s.engine.still(4.0, (240, 135)), base)


@needs_gl
@pytest.mark.gl
def test_the_scale_example_matches_its_golden():
    e = runpy.run_path(str(ROOT / "examples" / "verbs_scale.py"), run_name="x")["s"].engine
    assert not e.lint(strict=True)
    sheet = e.storyboard([1.4, 3.6, 5.0, 6.0, 7.9], size=(320, 180))["image"]
    _golden("verbs_scale", np.asarray(sheet), e.pass_runner().renderer_class())
