"""The algebra verbs: act, apply, generate, coset, quotient, permute."""

import math

import numpy as np
import pytest

from phasor import Scene, algebra, linalg
from phasor.time.verbs import VerbError
from tests.conftest import gl_available

needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _at(e, t, object_id):
    state = e.state_at(t)
    return state[object_id], state


def _tags(e, prefix):
    segs = [s for path, ss in e.document.timeline.tracks.items() if path.startswith(prefix)
            for s in ss]  # fmt: skip
    return {s.verb for s in segs}


def _same_document(build, orders, times):
    """Build the scene with each order of its steps: the documents (and so every state) match."""
    docs = [build(o) for o in orders]
    ref = docs[0].document
    for e in docs[1:]:
        assert e.document.timeline.tracks == ref.timeline.tracks
        for t in times:
            assert e.inspect(t, detail="brief") == docs[0].inspect(t, detail="brief")
    return docs


# ---------------------------------------------------------------- act


def _polygon(n=4):
    s = Scene("act", passes=None)
    s.add(algebra.Polygon(id="sq", n=n))
    s.marker("a", 1.0)
    s.marker("b", 4.0)
    s.marker("c", 7.0)
    return s


def test_act_turns_the_polygon_and_the_vertices_travel_with_it():
    s = _polygon()
    s.at("a").act("sq", "r")
    e = s.engine
    assert _tags(e, "sq.") == {"act"}
    start = e.state_at(0.5)["sq"].impl.features(e.state_at(0.5))["first_vertex"]
    end = e.state_at(3.0)["sq"].impl.features(e.state_at(3.0))["first_vertex"]
    assert np.allclose(start, (0, 1.5, 0), atol=1e-9)  # vertex 1 at the top
    assert np.allclose(end, (-1.5, 0, 0), atol=1e-9)  # a quarter turn counterclockwise
    mid = e.state_at(1.6)["sq"].params.rotation
    assert 0 < mid < 90  # smooth, not a jump
    assert e.inspect(3.0)["objects"][0]["semantic"]["element"] == "r"


def test_act_names_elements_by_label_or_number_and_reflections_flip():
    s = _polygon()
    s.at("a").act("sq", "s")
    e = s.engine
    assert e.state_at(0.5)["sq"].params.flip == 0.0
    assert 0.0 < e.state_at(1.6)["sq"].params.flip < 1.0
    sem = e.state_at(3.0)["sq"].impl.semantics(e.state_at(3.0))
    assert sem["element"] == "s" and sem["flipped"]
    t = _polygon()
    t.at("a").act("sq", 4)  # the same element, by number
    assert t.engine.document.timeline.tracks == e.document.timeline.tracks


def test_act_takes_the_short_way_round():
    s = _polygon()
    s.at("a").act("sq", "r3")  # three quarter turns is one back
    assert s.engine.state_at(3.0)["sq"].params.rotation == -90.0


def test_act_composes_onto_what_the_polygon_already_did():
    s = _polygon()
    s.at("a").act("sq", "r")
    s.at("b").act("sq", "s")
    e = s.engine
    sem = e.state_at(6.0)["sq"].impl.semantics(e.state_at(6.0))
    assert sem["element"] == "sr"  # s after r is s*r, not r*s
    s.at("c").act("sq", "r")  # r after s*r: a reflection again
    sem = e.state_at(9.0)["sq"].impl.semantics(e.state_at(9.0))
    group = algebra.am.dihedral(4)
    assert sem["element_index"] == group.mul(group.labels.index("r"), group.labels.index("sr"))


@pytest.mark.parametrize(
    "marks",
    [("a", "b"), ("a", "a")],
    ids=["different markers", "the same marker"],
)
def test_act_gives_the_same_document_whatever_order_it_is_written_in(marks):
    def build(order):
        s = _polygon()
        steps = {"r": lambda: s.at(marks[0]).act("sq", "r"),
                 "s": lambda: s.at(marks[1]).act("sq", "s")}  # fmt: skip
        for k in order:
            steps[k]()
        return s.engine

    a, b = _same_document(build, (["r", "s"], ["s", "r"]), (0.5, 1.6, 3.0, 5.0, 8.0))
    sem = a.state_at(9.0)["sq"].impl.semantics(a.state_at(9.0))
    assert sem["element"] in ("sr", "sr3")  # a fixed order: the same whichever way written
    assert a.lint() == b.lint() == []  # no overlapping or out-of-order segments


def test_act_at_one_marker_plays_the_steps_one_after_another():
    s = _polygon()
    s.at("a").act("sq", "r")
    s.at("a").act("sq", "r")
    e = s.engine
    segs = e.document.timeline.tracks["sq.rotation"]
    assert len(segs) == 2
    time_map = e._snap.compiled.time_map
    t0 = [time_map.resolve(x.t0) for x in segs]
    t1 = [time_map.resolve(x.t1) for x in segs]
    assert t0[1] == pytest.approx(t1[0])  # the second follows the first
    assert e.state_at(t1[1] + 0.1)["sq"].params.rotation == 180.0


def test_act_errors_name_the_problem():
    s = _polygon()
    s.add(algebra.ClockDiagram(id="clock"))
    with pytest.raises(VerbError, match="needs polygon; 'clock' is a clock_diagram"):
        s.at("a").act("clock", "r")
    with pytest.raises(VerbError, match="no element 'x'"):
        s.at("a").act("sq", "x")
    with pytest.raises(VerbError, match="elements 0 to 7"):
        s.at("a").act("sq", 8)
    with pytest.raises(VerbError, match="no object 'nope'"):
        s.at("a").act("nope", "r")
    t = Scene("odd", passes=None)
    t.add(algebra.Polygon(id="sq", n=4, rotation=30.0))
    t.marker("a", 1.0)
    with pytest.raises(VerbError, match="on a symmetry"):
        t.at("a").act("sq", "r")


def test_act_end_is_the_theme_duration():
    s = _polygon()
    s.at("a").act("sq", "r")
    d = s.engine._snap.compiled.theme.motion["act"]["duration"]
    assert s.at("a").act("sq", "s")._end == pytest.approx(2 * d)  # pushed behind the first


# ---------------------------------------------------------------- apply


def _grid(matrix=((1.0, 0.0), (0.0, 1.0))):
    s = Scene("apply", passes=None)
    s.add(linalg.TransformGrid(id="grid", matrix=matrix))
    s.marker("a", 1.0)
    s.marker("b", 4.0)
    return s


def _matrix(e, t):
    return np.array(e.state_at(t)["grid"].params.matrix)


def test_apply_composes_onto_the_current_matrix_smoothly():
    s = _grid(((2.0, 0.0), (0.0, 1.0)))
    shear = ((1.0, 1.0), (0.0, 1.0))
    s.at("a").apply("grid", shear)
    e = s.engine
    assert _tags(e, "grid.") == {"apply"}
    assert np.allclose(_matrix(e, 0.5), ((2, 0), (0, 1)))
    assert np.allclose(_matrix(e, 3.0), np.array(shear) @ np.array(((2, 0), (0, 1))))
    mid = _matrix(e, 1.7)
    assert not np.allclose(mid, _matrix(e, 0.5)) and not np.allclose(mid, _matrix(e, 3.0))
    assert e.inspect(3.0)["objects"][0]["semantic"]["determinant"] == pytest.approx(2.0)


def test_apply_two_steps_compose_in_time_order():
    s = _grid()
    rot = ((0.0, -1.0), (1.0, 0.0))
    stretch = ((3.0, 0.0), (0.0, 1.0))
    s.at("a").apply("grid", rot)
    s.at("b").apply("grid", stretch)
    assert np.allclose(_matrix(s.engine, 8.0), np.array(stretch) @ np.array(rot))


def test_apply_works_on_3x3():
    eye3 = tuple(tuple(float(i == j) for j in range(3)) for i in range(3))
    s = _grid(eye3)
    z90 = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    s.at("a").apply("grid", z90)
    assert np.allclose(_matrix(s.engine, 3.0), z90)


@pytest.mark.parametrize("marks", [("a", "b"), ("a", "a")], ids=["different", "same marker"])
def test_apply_gives_the_same_document_whatever_order_it_is_written_in(marks):
    rot = ((0.0, -1.0), (1.0, 0.0))
    shear = ((1.0, 2.0), (0.0, 1.0))

    def build(order):
        s = _grid()
        steps = {"rot": lambda: s.at(marks[0]).apply("grid", rot),
                 "shear": lambda: s.at(marks[1]).apply("grid", shear)}  # fmt: skip
        for k in order:
            steps[k]()
        return s.engine

    a, b = _same_document(build, (["rot", "shear"], ["shear", "rot"]), (0.5, 1.7, 3.0, 5.0, 9.0))
    assert a.lint() == b.lint() == []


def test_apply_errors():
    s = _grid()
    s.add(algebra.Polygon(id="sq"))
    with pytest.raises(VerbError, match="needs transform_grid"):
        s.at("a").apply("sq", ((1, 0), (0, 1)))
    with pytest.raises(VerbError, match="2x2; apply needs a 2x2 matrix"):
        s.at("a").apply("grid", np.eye(3))
    with pytest.raises(VerbError, match="matrix of numbers"):
        s.at("a").apply("grid", "identity")
    f = Scene("flow", passes=None)
    f.add(linalg.TransformGrid(id="grid", flow=((0.0, -1.0), (1.0, 0.0))))
    f.marker("a", 1.0)
    with pytest.raises(VerbError, match="shows a flow"):
        f.at("a").apply("grid", ((1, 0), (0, 1)))


def test_apply_can_follow_a_singular_matrix_but_not_be_put_before_one():
    s = _grid()
    s.at("a").apply("grid", ((1.0, 0.0), (0.0, 0.0)))  # a projection
    s.at("b").apply("grid", ((0.0, 1.0), (1.0, 0.0)))  # in time order: fine
    assert np.allclose(_matrix(s.engine, 8.0), ((0, 0), (1, 0)))
    t = _grid()
    t.at("b").apply("grid", ((1.0, 0.0), (0.0, 0.0)))
    t.at("a").apply("grid", ((0.0, 1.0), (1.0, 0.0)))  # goes before the projection
    assert np.allclose(_matrix(t.engine, 8.0), ((0, 1), (0, 0)))
    u = _grid()
    u.at("a").apply("grid", ((1.0, 0.0), (0.0, 0.0)))
    u.at("b").apply("grid", ((0.0, 1.0), (1.0, 0.0)))
    with pytest.raises(VerbError, match="singular"):  # between them: needs the lost matrix
        u.marker("m", 2.5)
        u.at("m").apply("grid", ((2.0, 0.0), (0.0, 2.0)))


# ---------------------------------------------------------------- generate


def _cayley(**kw):
    s = Scene("gen", passes=None)
    s.add(algebra.CayleyGraph(id="cay", group="D4", generators="r s", **kw))
    s.marker("a", 1.0)
    s.marker("b", 5.0)
    return s


def _growth(e, t):
    state = e.state_at(t)
    return state["cay"].params.reach


def test_generate_grows_the_graph_from_the_identity():
    s = _cayley()
    s.at("a").generate("cay")
    e = s.engine
    assert _tags(e, "cay.") == {"generate"}
    assert _growth(e, 0.5) == 0.0 and _growth(e, 8.0) == 1.0
    assert 0.0 < _growth(e, 2.0) < 1.0
    sem0 = e.state_at(0.5)["cay"].impl.semantics(e.state_at(0.5))
    assert sem0["reach"] == 0.0
    assert "reach" not in e.state_at(8.0)["cay"].impl.semantics(e.state_at(8.0))


def test_generate_orders_the_edges_breadth_first_one_generator_after_another():
    g = algebra.group_named("D4")
    gens = [g.labels.index("r"), g.labels.index("s")]
    order = algebra.am.growth_order(g, gens)
    assert len(order) == 16 and len({(x, s) for x, s, _ in order}) == 16  # each edge once
    r = gens[0]
    first_stage = [e for e in order[:4]]
    assert all(s == r for _, s, _ in first_stage)  # r alone goes round the ring first
    assert order[0][0] == 0  # from the identity
    flipped = algebra.am.growth_order(g, gens, by=[gens[1], gens[0]])
    assert flipped[0][1] == gens[1] and flipped[0] != order[0]


def test_generate_by_sets_the_order_and_checks_the_generators():
    s = _cayley()
    s.at("a").generate("cay", by="s r")
    assert s.engine.state_at(0.5)["cay"].params.order == "s r"
    with pytest.raises(VerbError, match="'t' is not a generator of 'cay'; its generators are"):
        s.at("b").generate("cay", by="t")
    with pytest.raises(VerbError, match="already grows in the order 's r'"):
        s.at("b").generate("cay", by="r s")


def test_generate_shows_the_edges_as_they_grow_and_all_when_done():
    s = _cayley()
    s.at("a").generate("cay")
    e = s.engine
    state, _ = _at(e, 8.0, "cay")[1], None
    full = e.state_at(8.0)["cay"].impl.features(state)["nodes"]
    assert len(full) == 8
    # the plain graph (no verb) has the same nodes and semantics as the grown one
    plain = _cayley().engine
    assert plain.state_at(8.0)["cay"].impl.semantics(plain.state_at(8.0)) == \
        e.state_at(8.0)["cay"].impl.semantics(e.state_at(8.0))  # fmt: skip


def test_generate_gives_the_same_document_in_either_order():
    def build(order):
        s = _cayley()
        s.add(algebra.CayleyGraph(id="cay2", group="Z6", generators="1", pos=(5, 0, 0)))
        steps = {"d": lambda: s.at("b").generate("cay"), "z": lambda: s.at("a").generate("cay2")}
        for k in order:
            steps[k]()
        return s.engine

    _same_document(build, (["d", "z"], ["z", "d"]), (0.5, 2.0, 6.0, 9.0))


def test_generate_errors():
    s = _cayley()
    s.add(algebra.ClockDiagram(id="clock"))
    with pytest.raises(VerbError, match="needs cayley_graph; 'clock' is a clock_diagram"):
        s.at("a").generate("clock")


# ---------------------------------------------------------------- coset and quotient


def _clock(**kw):
    s = Scene("coset", passes=None)
    s.add(algebra.ClockDiagram(id="clock", n=12, step=4, **kw))
    s.marker("a", 1.0)
    s.marker("b", 5.0)
    return s


def _points(e, t, object_id="clock"):
    state = e.state_at(t)
    return state[object_id].impl.features(state)["points" if object_id == "clock" else "nodes"]


def test_coset_gathers_each_coset_into_a_cluster_keeping_its_color_roles():
    s = _clock()
    s.at("a").coset("clock")
    e = s.engine
    assert _tags(e, "clock.") == {"coset"}
    before, after = _points(e, 0.5), _points(e, 5.0)
    assert not np.allclose(before, after)
    cosets = e.state_at(5.0)["clock"].impl.semantics(e.state_at(5.0))["cosets"]
    assert cosets == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
    centres = [after[c].mean(axis=0) for c in cosets]
    for c, mid in zip(cosets, centres, strict=True):  # a cluster is tight
        assert max(np.linalg.norm(after[k] - mid) for k in c) < 1.0
    spread = min(np.linalg.norm(a - b) for i, a in enumerate(centres) for b in centres[i + 1:])
    assert spread > 1.0  # and the clusters are apart
    assert np.allclose(before, _points(e, 0.0))  # nothing moves before the verb
    assert 0 < e.state_at(1.8)["clock"].params.gather < 1


def test_quotient_collapses_each_coset_to_one_node():
    s = _clock()
    s.at("a").coset("clock")
    s.at("b").quotient("clock")
    e = s.engine
    assert _tags(e, "clock.") == {"coset", "quotient"}
    end = _points(e, 9.0)
    cosets = e.state_at(9.0)["clock"].impl.semantics(e.state_at(9.0))["cosets"]
    for c in cosets:
        assert np.allclose(end[c], end[c[0]])  # all at one place
    assert len({tuple(np.round(end[c[0]], 6)) for c in cosets}) == 4  # four distinct nodes
    mid = _points(e, 6.0)
    assert not np.allclose(mid[0], mid[4])  # still collapsing halfway


def test_quotient_alone_glides_straight_to_the_coset_nodes():
    s = _clock()
    s.at("a").quotient("clock")
    end = _points(s.engine, 9.0)
    assert np.allclose(end[[0, 4, 8]], end[0])


@pytest.mark.parametrize("which", ["clock", "cayley"])
def test_coset_and_quotient_give_the_same_document_in_either_order(which):
    def build(order):
        s = Scene("two", passes=None)
        if which == "clock":
            s.add(algebra.ClockDiagram(id="d", n=12, step=4))
        else:
            s.add(algebra.CayleyGraph(id="d", group="D4", generators="r s", subgroup="r2"))
        s.marker("a", 1.0)
        s.marker("b", 5.0)
        steps = {"c": lambda: s.at("a").coset("d"), "q": lambda: s.at("b").quotient("d")}
        for k in order:
            steps[k]()
        return s.engine

    a, b = _same_document(build, (["c", "q"], ["q", "c"]), (0.5, 2.0, 6.0, 9.0))
    assert a.lint() == b.lint() == []


def test_coset_sets_the_subgroup_when_none_is_set_and_refuses_a_second_one():
    s = _clock()
    s.at("a").coset("clock", subgroup=6)  # the multiples of 6: {0, 6}
    e = s.engine
    assert e.state_at(0.5)["clock"].params.step == 6
    assert e.state_at(8.0)["clock"].impl.semantics(e.state_at(8.0))["subgroup"] == [0, 6]
    with pytest.raises(VerbError, match="different subgroup"):
        s.at("b").quotient("clock", subgroup=4)
    s.at("b").quotient("clock", subgroup=(0, 6))  # the same one, written differently
    c = Scene("g", passes=None)
    c.add(algebra.CayleyGraph(id="d", group="D4", generators="r s"))
    c.marker("a", 1.0)
    with pytest.raises(VerbError, match="needs a subgroup"):
        c.at("a").coset("d")
    c.at("a").coset("d", subgroup="r2")
    assert c.engine.state_at(0.5)["d"].params.subgroup == "r2"


def test_quotient_refuses_a_subgroup_that_is_not_normal():
    s = Scene("nn", passes=None)
    s.add(algebra.CayleyGraph(id="d", group="D4", generators="r s"))
    s.marker("a", 1.0)
    with pytest.raises(VerbError, match="not a normal subgroup of D4"):
        s.at("a").quotient("d", subgroup="s")
    s.at("a").coset("d", subgroup="s")  # cosets of a non-normal one are fine


def test_coset_and_quotient_on_a_cayley_graph_move_the_nodes_and_keep_generator_colors():
    s = Scene("cay", passes=None)
    s.add(algebra.CayleyGraph(id="d", group="D4", generators="r s", subgroup="r2"))
    s.marker("a", 1.0)
    s.marker("b", 5.0)
    s.at("a").coset("d")
    s.at("b").quotient("d")
    e = s.engine
    sem = e.state_at(9.0)["d"].impl.semantics(e.state_at(9.0))
    assert sem["subgroup"] == ["e", "r2"] and len(sem["cosets"]) == 4
    end = _points(e, 9.0, "d")
    g = algebra.group_named("D4")
    for c in e.state_at(9.0)["d"].impl.semantics(e.state_at(9.0))["cosets"]:
        idx = [g.labels.index(x) for x in c]
        assert np.allclose(end[idx], end[idx[0]])
    assert not np.allclose(_points(e, 0.5, "d"), end)


def test_coset_errors():
    s = _clock()
    s.add(algebra.Polygon(id="sq"))
    with pytest.raises(VerbError, match="needs clock_diagram or cayley_graph"):
        s.at("a").coset("sq")
    with pytest.raises(VerbError, match="Z12 has no element 99"):
        s.at("a").coset("clock", subgroup=99)
    t = Scene("g", passes=None)
    t.add(algebra.CayleyGraph(id="d", group="D4", generators="r s"))
    t.marker("a", 1.0)
    with pytest.raises(VerbError, match="no element 'zz'"):
        t.at("a").coset("d", subgroup="zz")


# ---------------------------------------------------------------- permute


def _perm(**kw):
    s = Scene("perm", passes=None)
    s.add(algebra.Permutation(id="p", perm=(0, 1, 2, 3, 4), **kw))
    s.marker("a", 1.0)
    s.marker("b", 5.0)
    return s


def _tokens(e, t):
    state = e.state_at(t)
    return state["p"].impl.features(state)["tokens"]


def _slot_x(e, slot):
    state = e.state_at(0.0)
    return state["p"].impl.features(state)["slots"][slot]


def test_permute_swaps_tokens_along_arcs_to_the_new_arrangement():
    s = _perm()
    s.at("a").permute("p", [1, 0, 2, 4, 3])
    e = s.engine
    assert _tags(e, "p.") == {"permute"}
    start, mid, end = _tokens(e, 0.5), _tokens(e, 1.7), _tokens(e, 4.0)
    assert np.allclose(start, [_slot_x(e, k) for k in range(5)])
    assert np.allclose(end, [_slot_x(e, k) for k in (1, 0, 2, 4, 3)])
    assert abs(mid[0][1]) > 0.05  # an arc, not a straight slide
    assert e.state_at(4.0)["p"].impl.semantics(e.state_at(4.0))["done"]


def test_permute_accepts_cycles_written_with_the_numbers_on_the_tokens():
    s = _perm()
    s.at("a").permute("p", "(1 2 3)(4 5)")
    e = s.engine
    assert np.allclose(_tokens(e, 4.0), [_slot_x(e, k) for k in (1, 2, 0, 4, 3)])
    sem = e.state_at(4.0)["p"].impl.semantics(e.state_at(4.0))
    assert sem["cycles"] == "(1 2 3)(4 5)"


def test_a_second_permute_starts_from_where_the_first_left_the_tokens():
    s = _perm()
    s.at("a").permute("p", [1, 0, 2, 3, 4])
    s.at("b").permute("p", [0, 1, 2, 4, 3])
    e = s.engine
    before, after = _tokens(e, 4.99), _tokens(e, 5.0)
    assert np.allclose(before, after)  # no jump where the second begins
    assert np.allclose(_tokens(e, 9.0), [_slot_x(e, k) for k in (0, 1, 2, 4, 3)])
    assert np.allclose(_tokens(e, 4.0), [_slot_x(e, k) for k in (1, 0, 2, 3, 4)])
    assert e.lint() == []


def test_permute_gives_the_same_document_whatever_order_it_is_written_in():
    def build(order):
        s = _perm()
        steps = {"x": lambda: s.at("a").permute("p", [1, 0, 2, 3, 4]),
                 "y": lambda: s.at("b").permute("p", [4, 3, 2, 1, 0])}  # fmt: skip
        for k in order:
            steps[k]()
        return s.engine

    a, b = _same_document(build, (["x", "y"], ["y", "x"]), (0.5, 2.0, 4.0, 6.0, 9.0))
    assert a.lint() == b.lint() == []
    assert np.allclose(_tokens(a, 9.0), [_slot_x(a, k) for k in (4, 3, 2, 1, 0)])


def test_permute_keeps_cycle_colors_and_errors():
    s = _perm()
    with pytest.raises(VerbError, match="sigma has 3 places; the permutation has 5 tokens"):
        s.at("a").permute("p", [0, 1, 2])
    with pytest.raises(VerbError, match="not a permutation"):
        s.at("a").permute("p", [0, 0, 1, 2, 3])
    with pytest.raises(VerbError, match="not a cycle of the tokens 1..5"):
        s.at("a").permute("p", "(1 9)")
    s.add(algebra.Polygon(id="sq"))
    with pytest.raises(VerbError, match="needs permutation"):
        s.at("a").permute("sq", [0])
    s.at("a").permute("p", [1, 0, 2, 3, 4])
    with pytest.raises(VerbError, match="already permutes at this moment"):
        s.at("a").permute("p", [0, 1, 2, 3, 4])


# ---------------------------------------------------------------- strict lint and golden


def test_the_example_is_strict_clean_and_uses_every_verb():
    import runpy
    from pathlib import Path

    s = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "verbs_algebra.py"),
                       run_name="x")["s"]  # fmt: skip
    e = s.engine
    assert not [i for i in e.lint(strict=True) if i.severity == "error"]
    used = {seg.verb for ss in e.document.timeline.tracks.values() for seg in ss}
    assert {"act", "apply", "generate", "coset", "quotient", "permute"} <= used


@needs_gl
@pytest.mark.gl
def test_the_algebra_verbs_example_matches_its_golden():
    import runpy
    from pathlib import Path

    from tests.test_beauty import _golden

    e = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "verbs_algebra.py"),
                       run_name="x")["s"].engine  # fmt: skip
    times = sorted(e.document.timeline.markers.values())
    sheet = e.storyboard([t + 1.8 for t in times], size=(320, 180))["image"]
    _golden("verbs_algebra", sheet, e.pass_runner().renderer_class())


def test_math_helpers_are_consistent():
    assert math.isclose(sum(algebra.am.growth_progress([(0, 0, 0)] * 4, 0.5)), 2.0)
