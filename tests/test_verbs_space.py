"""The space and point verbs (group F2b): lift, slice, project, dissolve and assemble, and the
objects they grow (Landscape, CutPlane, Dashed) and PointCloud's ``scatter``."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Line, Scene, cx, geo, linalg, pde, space
from phasor.domains import cloud
from phasor.domains.space import cut_curves
from phasor.time.verbs import VerbError
from tests.conftest import gl_available
from tests.test_beauty import _golden

ROOT = Path(__file__).parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def sem(e, i, t=0.0):
    return e.inspect(t, id=i)["objects"][0]["semantic"]


def scene(**kw):
    s = Scene("space", passes=None, **kw)
    s.use_camera("orbit")
    s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(6, 4)))
    s.add(geo.ParametricSurface(id="ball", fx="2*sin(u)*cos(v)", fy="2*sin(u)*sin(v)",
                                fz="2*cos(u)", u_range=(0, math.pi), v_range=(0, 2 * math.pi),
                                pos=(0, 0, 3)))  # fmt: skip
    s.add(linalg.Arrow(id="v", start=(0, 0, 0), end=(2, 1.5, 0)))
    s.add(linalg.Arrow(id="w", start=(0, 0, 0), end=(3, 0, 0)))
    s.add(Line(id="ring", points=[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (1, 0, 0)]))
    for name, t in (("a", 1.0), ("b", 4.0), ("c", 7.0), ("d", 10.0), ("e", 13.0)):
        s.marker(name, t)
    return s


def tagged(e):
    """Every segment the verbs wrote carries its verb's name (strict lint: ``raw_motion``)."""
    segs = [sg for track in e.document.timeline.tracks.values() for sg in track]
    return {sg.verb for sg in segs}


# ---------------------------------------------------------------- lift


def test_lift_raises_the_plane_into_its_landscape_and_fades_the_flat_colors():
    s = scene()
    end = s.at("a").lift("plane")._end
    e = s.engine
    assert end == pytest.approx(2.0)  # the theme's lift duration
    assert "plane.surface" in e.state_at(0.0)
    s0, s1, s2 = (sem(e, "plane.surface", t) for t in (0.5, 2.0, 3.5))
    assert s0["height"] == 0.0 and s0["peak"] == 0.0
    assert 0.0 < s1["height"] < 0.6 and s1["peak"] > 0.0
    assert s2["height"] == pytest.approx(0.6) and s2["peak"] > 1.0
    opacity = [e.state_at(t)["plane"].params.opacity for t in (0.5, 1.2, 3.5)]
    assert opacity[0] == 1.0 and opacity[2] == 0.0
    assert e.state_at(0.5)["plane.surface"].params.opacity == 0.0  # hidden until it grows
    assert e.state_at(3.5)["plane.surface"].params.opacity == 1.0
    assert "lift" in tagged(e)


def test_the_landscape_stays_linked_to_the_plane_and_poles_are_clipped():
    s = scene()
    s.at("a").lift("plane", height=1.0)
    e = s.engine
    s.set("plane.a", (0.0, 0.0))
    s.set("plane.pos", (1.0, 0.0, 0.0))
    st = e.state_at(4.0)
    land = st["plane.surface"].params
    assert land.pos == (1.0, 0.0, 0.0) and land.size == (6.0, 4.0)  # expressions follow
    pole = np.asarray(st["plane.surface"].impl.grid_points(st))
    assert np.isfinite(pole).all()
    assert pole[..., 2].max() <= 3.0 * 1.0 + 1e-9  # the clip: cap 3 times the height
    # a pole right on the grid: |f| is huge, the tower stops at the cap
    s.add(space.Landscape(id="tower", function="1/z", size=(2, 2), n=9, height=1.0))
    tower = sem(s.engine, "tower")
    assert tower["clipped"] > 0 and tower["peak"] == pytest.approx(3.0)


def test_lift_needs_a_domain_coloring_and_a_sensible_height():
    s = scene()
    with pytest.raises(
        VerbError, match="lift needs domain_coloring; 'ball' is a parametric_surface"
    ):
        s.at("a").lift("ball")
    with pytest.raises(VerbError, match="height above 0"):
        s.at("a").lift("plane", height=0)
    with pytest.raises(VerbError, match="no object 'nope'"):
        s.at("a").lift("nope")


# ---------------------------------------------------------------- slice


def test_slice_sweeps_a_plane_through_a_sphere_and_draws_the_curve():
    s = scene()
    s.at("b").slice("ball", axis="z")
    e = s.engine
    # the sphere has radius 2 about (0, 0, 3): z runs 1 to 5
    lo, mid, hi = (sem(e, "ball.cut", t) for t in (4.0, 4.8, 6.0))
    assert lo["at"] == pytest.approx(1.08) and hi["at"] == pytest.approx(4.92)
    assert lo["curves"] == hi["curves"] == 1
    r_mid = max(abs(v) for v in mid["x_range"])
    assert 1.0 < r_mid <= 2.0  # a circle: its radius at height ``at``
    z = hi["at"] - 3.0
    assert max(hi["x_range"]) == pytest.approx(math.sqrt(4 - z * z), abs=0.08)
    assert e.state_at(3.0)["ball.cut"].params.opacity == 0.0
    assert e.state_at(6.0)["ball.cut"].params.opacity == 1.0
    assert "slice" in tagged(e)


def test_slice_goes_from_start_to_at_and_the_next_slice_starts_where_the_plane_is():
    s = scene()
    s.at("b").slice("ball", axis="x", start=-1.0, at=0.5)
    s.at("d").slice("ball", axis="x", at=-0.5)
    e = s.engine
    assert sem(e, "ball.cut", 4.0)["at"] == pytest.approx(-1.0)
    assert sem(e, "ball.cut", 9.0)["at"] == pytest.approx(0.5)
    assert sem(e, "ball.cut", 12.0)["at"] == pytest.approx(-0.5)
    with pytest.raises(VerbError, match="cuts along 'x'"):
        s.at("e").slice("ball", axis="y")


def test_the_cut_curve_of_a_cylinder_and_a_saddle():
    uu, vv = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41), indexing="ij")
    saddle = np.stack([uu, vv, 0.5 * (uu**2 - vv**2)], axis=-1)
    (curve,) = cut_curves(saddle, 1, 1.0)  # y = 1: z = 0.5 x^2 - 0.5
    assert np.allclose(curve[:, 1], 1.0)
    assert np.allclose(curve[:, 2], 0.5 * curve[:, 0] ** 2 - 0.5, atol=0.03)
    assert curve[:, 0].min() == pytest.approx(-2, abs=0.01)  # the whole width
    assert cut_curves(saddle, 1, 5.0) == []  # a plane that misses it
    # a plane through the middle of a torus cuts two loops
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(0, 2 * np.pi, 49),
                         indexing="ij")  # fmt: skip
    torus = np.stack([(3 + np.cos(ph)) * np.cos(th), (3 + np.cos(ph)) * np.sin(th),
                      np.sin(ph)], axis=-1)  # fmt: skip
    loops = cut_curves(torus, 0, 0.0)
    assert len(loops) == 2 and all(np.allclose(c[0], c[-1]) for c in loops)


def test_slice_cuts_a_landscape_and_a_pde_surface_and_refuses_other_things():
    s = scene()
    s.add(space.Landscape(id="land", function="1/(z**2 + 1)"))
    s.add(pde.Surface(id="heat", kind="heat", time=0.05, n=16))
    s.at("a").slice("land", axis="x", at=0.0)
    s.at("b").slice("heat", axis="y", at=0.0)
    e = s.engine
    assert sem(e, "land.cut", 3.5)["curves"] == 1
    assert sem(e, "heat.cut", 6.0)["curves"] == 1
    with pytest.raises(VerbError, match="slice cuts a ParametricSurface"):
        s.at("c").slice("v")
    with pytest.raises(VerbError, match="'x', 'y' or 'z'"):
        s.at("c").slice("ball", axis="w")


def test_a_cut_plane_with_no_surface_says_so():
    s = Scene("cut", passes=None)
    s.add(space.CutPlane(id="lonely", surface="nothing"))
    assert sem(s.engine, "lonely")["missing_surface"] is True


# ---------------------------------------------------------------- project


def test_project_drops_a_perpendicular_to_the_foot_and_grows_the_projection():
    s = scene()
    end = s.at("c").project("v", onto="w")._end
    e = s.engine
    assert end == pytest.approx(1.4)  # the theme's project duration
    foot = (2.0, 0.0, 0.0)  # v = (2, 1.5) onto the x axis
    drop = e.state_at(9.0)["v.drop"].params
    assert drop.start == pytest.approx((2.0, 1.5, 0.0)) and drop.end == pytest.approx(foot)
    assert drop.square > 0
    assert sem(e, "v.drop", 7.0)["drawn"] == 0.0  # nothing before the anchor
    mid = sem(e, "v.drop", 7.35)
    assert 0 < mid["drawn"] < 1
    proj = [sem(e, "v.proj", t)["components"] for t in (7.0, 7.7, 8.0, 9.0)]
    assert proj[0] == [0.0, 0.0, 0.0] and proj[1] == [0.0, 0.0, 0.0]  # after the drop
    assert 0 < proj[2][0] < 2.0 and proj[3] == pytest.approx([2.0, 0.0, 0.0])
    assert e.state_at(7.0)["v.proj"].params.opacity == 0.0
    assert e.state_at(9.0)["v.proj"].params.opacity == 1.0
    assert "project" in tagged(e)


def test_project_follows_the_arrows_when_they_move():
    s = scene()
    s.at("c").project("v", onto="w")
    s.set("w.end", (0.0, 3.0, 0.0))  # now the y axis
    s.set("v.end", (2.0, 1.5, 0.0))
    e = s.engine
    assert e.state_at(9.0)["v.drop"].params.end == pytest.approx((0.0, 1.5, 0.0))
    assert sem(e, "v.proj", 9.0)["components"] == pytest.approx([0.0, 1.5, 0.0])


def test_project_a_point_onto_a_line_with_a_label():
    s = scene()
    s.add(Dot(id="p", pos=(1.0, 2.0, 0.0)))
    s.add(Line(id="axis", points=[(0, 0, 0), (4, 4, 0)]))
    s.at("c").project("p", onto="axis", label="shadow")
    e = s.engine
    assert e.state_at(9.0)["p.drop"].params.end == pytest.approx((1.5, 1.5, 0.0))
    assert sem(e, "p.proj.label", 9.0)["kind"] == "label"
    assert e.state_at(7.0)["p.proj.label"].params.opacity == 0.0
    assert e.state_at(9.0)["p.proj.label"].params.opacity == 1.0


def test_project_error_messages():
    s = scene()
    s.add(Line(id="bent", points=[(0, 0, 0), (1, 0, 0), (1, 1, 0)]))
    with pytest.raises(VerbError, match="project needs arrow or dot; 'ball' is a parametric"):
        s.at("c").project("ball", onto="w")
    with pytest.raises(VerbError, match="project needs arrow or line; 'ball'"):
        s.at("c").project("v", onto="ball")
    with pytest.raises(VerbError, match="two-point line; 'bent' has 3"):
        s.at("c").project("v", onto="bent")
    with pytest.raises(TypeError):
        s.at("c").project("v")  # onto is required
    s.at("c").project("v", onto="w")
    with pytest.raises(VerbError, match="already projects onto something else"):
        s.at("d").project("v", onto="v")


def test_project_twice_starts_over():
    s = scene()
    s.at("c").project("v", onto="w")
    s.at("e").project("v", onto="w")
    e = s.engine
    assert sem(e, "v.drop", 13.0)["drawn"] == 0.0  # reset at the second anchor
    assert sem(e, "v.proj", 15.0)["components"] == pytest.approx([2.0, 0.0, 0.0])


# ---------------------------------------------------------------- dissolve / assemble


def test_dissolve_turns_an_object_into_points_that_drift_and_assemble_brings_it_back():
    s = scene()
    d = s.at("d").dissolve("ring", count=300)._end
    e = s.engine
    assert d == pytest.approx(1.5)
    dust = e.state_at(10.0)["ring.dust"].params
    assert dust.source == "points" and len(dust.points) == 300 and dust.scatter == 0.0
    pts = np.asarray(dust.points) + np.asarray(dust.pos[:2])
    r = np.abs(pts[:, 0]) + np.abs(pts[:, 1])
    assert np.allclose(r, 1.0, atol=1e-3)  # on the diamond the ring draws
    before, after = e.state_at(10.0), e.state_at(11.6)
    assert before["ring"].params.opacity > 0.5 and after["ring"].params.opacity == 0.0
    assert before["ring.dust"].params.opacity == 0.0 and after["ring.dust"].params.opacity == 1.0
    scat = [e.state_at(t)["ring.dust"].params.scatter for t in (10.0, 10.5, 11.0, 11.6)]
    assert scat[0] == 0.0 and scat[1] < scat[2] < scat[3] == pytest.approx(scat[3])
    assert scat[3] > 0.3
    # every point moves along its own direction: the spread of the cloud grows
    ext = [sem(e, "ring.dust", t)["extent"] for t in (10.0, 11.6)]
    assert ext[1][1][0] - ext[1][0][0] > ext[0][1][0] - ext[0][0][0] + 0.3
    assert "dissolve" in tagged(e)


def test_assemble_is_dissolve_in_reverse():
    s = scene()
    s.at("d").dissolve("ring", count=200)
    s.at("e").assemble("ring", count=200)
    e = s.engine
    d_end = e.state_at(11.5)["ring.dust"].params.scatter
    assert e.state_at(13.0)["ring.dust"].params.scatter == pytest.approx(d_end)  # where it was left
    assert e.state_at(14.5)["ring.dust"].params.scatter == pytest.approx(0.0)
    for t in (10.4, 10.9):  # the same scatter on the way out as on the way back (mirrored)
        out = e.state_at(t)["ring.dust"].params.scatter
        assert 0 < out < d_end
    assert e.state_at(14.6)["ring"].params.opacity > 0.5
    assert e.state_at(14.6)["ring.dust"].params.opacity == 0.0
    assert e.state_at(13.0)["ring"].params.opacity == 0.0
    assert "assemble" in tagged(e)


def test_assemble_on_its_own_hides_the_object_until_the_points_arrive():
    s = scene()
    s.at("c").assemble("ring", count=100, spread=1.5)
    e = s.engine
    assert e.state_at(0.0)["ring"].params.opacity == 0.0
    assert e.state_at(7.0)["ring.dust"].params.scatter == pytest.approx(1.5)
    assert e.state_at(8.6)["ring"].params.opacity > 0.5


def test_the_points_are_pure_in_the_document_and_the_seed():
    def dust(seed):
        s = scene()
        s.at("d").dissolve("ring", count=64, seed=seed)
        return s.engine.state_at(11.0)["ring.dust"].params

    a, b, c = dust(0), dust(0), dust(3)
    assert a == b and a.points != c.points
    # same state at the same time, whatever came before
    s = scene()
    s.at("d").dissolve("ring", count=64)
    e = s.engine
    first = e.state_at(10.8)["ring.dust"].params.scatter
    e.state_at(14.0)
    assert e.state_at(10.8)["ring.dust"].params.scatter == first


def test_dissolve_samples_a_surface_and_a_filled_object():
    s = scene()
    s.at("d").dissolve("ball", count=100)
    e = s.engine
    assert sem(e, "ball.dust", 11.6)["points"] == 100
    s2 = scene()
    s2.at("d").dissolve("plane", count=50)  # a kernel-drawn field: points fill its box
    box = e.state_at(0.0)["plane"].impl.world_bounds(e.state_at(0.0))
    dust = s2.engine.state_at(11.0)["plane.dust"].params
    pts = np.asarray(dust.points) + np.asarray(dust.pos[:2])
    assert pts[:, 0].min() >= box.lo[0] - 1e-3 and pts[:, 0].max() <= box.hi[0] + 1e-3


def test_dissolve_and_assemble_error_messages():
    s = scene()
    with pytest.raises(VerbError, match="needs at least one object id"):
        s.at("d").dissolve()
    with pytest.raises(VerbError, match="no object 'nope'"):
        s.at("d").assemble("nope")
    with pytest.raises(VerbError, match="count is between 12 and 20000"):
        s.at("d").dissolve("ring", count=3)
    with pytest.raises(VerbError, match="spread must be above 0"):
        s.at("d").dissolve("ring", spread=0)
    s.at("d").dissolve("ring", spread=0.5)
    with pytest.raises(VerbError, match="too small for a spread"):
        s.at("e").assemble("ring", spread=8.0)


def test_several_objects_dissolve_in_a_stagger():
    s = scene()
    s.add(Dot(id="d1", pos=(2, 2)))
    s.add(Dot(id="d2", pos=(-2, 2)))
    end = s.at("d").dissolve("d1", "d2", count=40)._end
    assert end > 1.5  # the second starts a beat later
    e = s.engine
    assert e.state_at(10.3)["d1.dust"].params.opacity > e.state_at(10.3)["d2.dust"].params.opacity


# ---------------------------------------------------------------- PointCloud scatter


def test_scatter_zero_changes_nothing_and_scatter_moves_every_point_its_own_way():
    base = dict(source="points", points=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    s = Scene("cloud", passes=None)
    s.add(cloud.PointCloud(id="still", **base))
    s.add(cloud.PointCloud(id="moved", scatter=0.5, **base))
    s.add(cloud.PointCloud(id="again", scatter=0.5, **base))
    st = s.engine.state_at(0.0)
    p0, _ = st["still"].impl.cloud(st)
    p1, _ = st["moved"].impl.cloud(st)
    assert np.allclose(p0, [[0, 0], [1, 0], [0, 1], [1, 1]])
    moved = np.linalg.norm(p1 - p0, axis=1)
    assert np.all(moved <= 0.5 + 1e-9) and np.all(moved > 0.1)
    assert len({tuple(np.round(d, 6)) for d in p1 - p0}) == 4  # four different directions
    assert "scatter" not in st["still"].impl.semantics(st)
    assert st["moved"].impl.semantics(st)["scatter"] == 0.5
    q, _ = st["again"].impl.cloud(st)  # another id: another set of directions
    assert not np.allclose(q, p1)


# ---------------------------------------------------------------- the objects


def test_dashed_draws_dashes_up_to_its_progress():
    s = Scene("dash", passes=None)
    s.add(space.Dashed(id="d", start=(0, 0, 0), end=(4, 0, 0), dashes=4, progress=0.5, square=0.3,
                       back=(-1, 0, 0)))  # fmt: skip
    e = s.engine
    assert sem(e, "d")["drawn"] == 0.5
    st = e.state_at(0.0)
    assert np.allclose(st["d"].impl.features(st)["end"], (4, 0, 0))


def test_landscape_height_zero_is_flat_and_the_peak_scales_with_height():
    s = Scene("land", passes=None)
    s.use_camera("orbit")
    s.add(space.Landscape(id="flat", height=0.0))
    s.add(space.Landscape(id="tall", height=1.0))
    assert sem(s.engine, "flat")["peak"] == 0.0
    assert sem(s.engine, "tall")["peak"] > 0.5


# ---------------------------------------------------------------- order, MCP, example


def test_verbs_give_the_same_scene_whatever_order_they_are_written_in():
    def build(order):
        s = scene()
        s.add(Dot(id="d1", pos=(2, 2)))
        steps = {
            "lift": lambda: s.at("a").lift("plane"),
            "slice": lambda: s.at("a").slice("ball", axis="z"),
            "project": lambda: s.at("a").project("v", onto="w"),
            "dissolve": lambda: s.at("a").dissolve("ring", count=64),
            "assemble": lambda: s.at("a").assemble("d1", count=64),
        }
        for k in order:
            steps[k]()
        return s.engine

    names = ["lift", "slice", "project", "dissolve", "assemble"]
    ref = build(names)
    for order in (names[::-1], names[2:] + names[:2], ["project", "assemble", "lift", "dissolve",
                                                     "slice"]):  # fmt: skip
        other = build(order)
        for t in (0.5, 1.4, 2.0, 2.4, 3.0):
            a, b = (x.inspect(t, detail="brief") for x in (ref, other))
            assert sorted(a["objects"]) == sorted(b["objects"])  # (children come in any order)
            for oid in ("plane.surface", "ball.cut", "v.drop", "v.proj", "ring.dust", "d1.dust"):
                assert sem(ref, oid, t) == sem(other, oid, t)


def test_the_ai_tool_takes_the_space_verbs(tmp_path):
    from phasor.workspace import Workspace
    from phasor.workspace.tools import ToolError

    ws = Workspace(tmp_path)
    ws.scene_new("ai")
    ws.scene_edit([
        {"op": "set_camera", "kind": "orbit", "params": {}},
        {"op": "add_object", "object": {"id": "plane", "type": "domain_coloring", "params": {"function": "1/(z**2+1)"}}},
        {"op": "add_object", "object": {"id": "sheet", "type": "parametric_surface", "params": {"fz": "u*v"}}},
        {"op": "add_object", "object": {"id": "a", "type": "arrow", "params": {"end": [2, 1, 0]}}},
        {"op": "add_object", "object": {"id": "b", "type": "arrow", "params": {"end": [3, 0, 0]}}},
        {"op": "add_object", "object": {"id": "star", "type": "line", "params": {"points": [[0, 0, 0], [1, 1, 0], [2, 0, 0]]}}},
        {"op": "set_marker", "name": "go", "t": 1.0},
        {"op": "set_marker", "name": "later", "t": 5.0},
    ])  # fmt: skip
    assert ws.animate("lift", "go", targets=["plane"], value={"height": 0.9})[
        "end"
    ] == pytest.approx(3.0)
    assert ws.animate("slice", "go", targets=["sheet"], value={"axis": "y", "at": 1.0})["end"] > 1.0
    assert ws.animate("project", "go", targets=["a"], value={"onto": "b"})["end"] > 1.0
    assert ws.animate("dissolve", "go", targets=["star"], value={"count": 50})["end"] > 1.0
    assert ws.animate("assemble", "later", targets=["star"], value={"count": 50})["end"] > 1.0
    with pytest.raises(ToolError, match="project needs arrow or dot"):
        ws.animate("project", "go", targets=["sheet"], value={"onto": "b"})


@needs_gl
@pytest.mark.slow
def test_the_verbs_space_example_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "verbs_space.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([2.0, 5.5, 9.0, 11.5], size=(320, 180))["image"]
    _golden("verbs_space", sheet, e.pass_runner().renderer_class())
