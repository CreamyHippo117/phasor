"""Milestone 15 (ADR 0005): the open object contract. Two required members (``Params`` and
``draw``); bounds, features and semantics derived from a recorded draw; composites; types
defined in a scene; pure updaters and ``Value``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from phasor import Dot, Scene
from phasor.core import record
from phasor.core.objects import (
    CONTRACT_MEMBERS,
    ObjectBase,
    ParamsBase,
    RegistrationError,
    get_type,
    register,
    registered_types,
)
from phasor.core.values import Vec3


def _never(*_):
    raise AssertionError("the recorder must never compute a field's pixels")


@register
class Square(ObjectBase):
    """The short form: a new shape in a few lines."""

    type: ClassVar[str] = "m15_square"
    Style: ClassVar = {"color": ("emphasis", "primary")}

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        side: float = 1.0

    def draw(self, ctx, state):
        p = self.me(state).params
        h = p.side / 2
        x, y, z = p.pos
        quad = np.array(
            [[x - h, y - h, z], [x + h, y - h, z], [x + h, y + h, z], [x - h, y + h, z]]
        )
        ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=self.me(state).style["color"])


@register
class Everything(ObjectBase):
    """Draws one of each command, so every kind is recorded."""

    type: ClassVar[str] = "m15_everything"
    draws: ClassVar[int] = 0

    class Params(ParamsBase):
        shift: float = 0.0

    def draw(self, ctx, state):
        type(self).draws += 1
        s = self.me(state).params.shift
        ctx.triangles(np.array([[s, 0, 0], [s + 1, 0, 0], [s, 1, 0]]), color=(1, 1, 1, 1))
        ctx.field(np.array([-3.0, -1, 0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                  pixels=_never)  # fmt: skip
        ctx.surface(np.array([[0, 0, -2.0], [0, 1, -2], [1, 0, -2]]), np.array([[0, 1, 2]]),
                    normals=np.zeros((3, 3)), colors=np.ones((3, 4)))  # fmt: skip
        ctx.text("hi", np.array([0.0, 3, 0]), 0.5, color=(1, 1, 1, 1))


@register
class Stroke(ObjectBase):
    type: ClassVar[str] = "m15_stroke"
    Style: ClassVar = {"stroke": ("stroke", "regular")}

    class Params(ParamsBase):
        a: Vec3 = (0.0, 0.0, 0.0)
        b: Vec3 = (2.0, 1.0, 0.0)

    def draw(self, ctx, state):
        from phasor.domains.strokes import polyline

        me = self.me(state)
        cam = state.camera_for(me)
        polyline(ctx, cam, [me.params.a, me.params.b], me.style["stroke"] * cam.height,
                 color=(1, 1, 1, 1))  # fmt: skip


def _scene(*objs, height=8, passes=None):
    s = Scene("m15", passes=passes)
    s.set("camera.height", height)
    for o in objs:
        s.add(o)
    return s


# ---------------------------------------------------------------- 1. the recorder


def test_the_recorder_keeps_every_command_and_never_computes_pixels():
    s = _scene(Everything(id="all"))
    state = s.engine.state_at(0)
    rec = record.record(state["all"].impl, state)
    assert rec.kinds == ("triangles", "field", "surface", "text")
    b = state["all"].impl.world_bounds(state)
    assert b.lo[0] == pytest.approx(-3) and b.lo[1] == pytest.approx(-1)  # the field's corner
    assert b.lo[2] == pytest.approx(-2)  # the surface
    assert b.hi[1] > 3  # the text box reaches above its anchor


def test_derived_bounds_features_and_semantics():
    s = _scene(Square(id="sq", pos=(1, 2), side=2))
    state = s.engine.state_at(0)
    impl = state["sq"].impl
    b = impl.world_bounds(state)
    assert b.lo == (0.0, 1.0, 0.0) and b.hi == (2.0, 3.0, 0.0)
    f = impl.features(state)
    assert sorted(f) == sorted(["center", *record.COMPASS])
    np.testing.assert_allclose(f["center"], [1, 2, 0])
    np.testing.assert_allclose(f["top"], [1, 3, 0])
    np.testing.assert_allclose(f["bottom_left"], [0, 1, 0])
    assert impl.semantics(state) == {
        "type": "m15_square",
        "params": {"pos": [1.0, 2.0, 0.0], "side": 2.0, "opacity": 1.0, "dim": 1.0,
                   "z_index": 0, "flash": 0.0},
    }  # fmt: skip


def test_an_object_that_draws_one_stroke_has_a_start_and_an_end():
    s = _scene(Stroke(id="st"))
    state = s.engine.state_at(0)
    f = state["st"].impl.features(state)
    np.testing.assert_allclose(f["start"], [0, 0, 0])
    np.testing.assert_allclose(f["end"], [2, 1, 0])
    assert f["curve"].shape == (2, 3)
    # The box covers the stroke's width too, so it reaches a little past both ends.
    b = state["st"].impl.world_bounds(state)
    assert b.lo[0] < 0 and b.hi[0] > 2


def test_a_recording_is_cached_until_the_params_change():
    Everything.draws = 0
    s = _scene(Everything(id="cached"))
    state = s.engine.state_at(0)
    impl = state["cached"].impl
    for _ in range(3):
        impl.world_bounds(state), impl.features(state)
    assert Everything.draws == 1
    s.set("cached.shift", 5)
    state = s.engine.state_at(0)
    assert state["cached"].impl.world_bounds(state).hi[0] == pytest.approx(6)
    assert Everything.draws == 2


@register
class Follower(ObjectBase):
    """Draws at another object's position: reads that object's state while drawing."""

    type: ClassVar[str] = "m15_follower"
    draws: ClassVar[int] = 0

    class Params(ParamsBase):
        target: str = "dot"

    def draw(self, ctx, state):
        type(self).draws += 1
        x, y, z = state[self.me(state).params.target].params.pos
        ctx.triangles(np.array([[x, y, z], [x + 1, y, z], [x, y + 1, z]]), color=(1, 1, 1, 1))


def test_a_recording_is_made_again_when_an_object_it_read_changes():
    Follower.draws = 0
    s = _scene(Dot(id="dot", pos=(1, 1)), Follower(id="fol"))
    state = s.engine.state_at(0)
    assert state["fol"].impl.world_bounds(state).lo[:2] == (1.0, 1.0)
    state["fol"].impl.world_bounds(state)
    assert Follower.draws == 1
    s.set("dot.pos", (-2, 0))
    state = s.engine.state_at(0)
    assert state["fol"].impl.world_bounds(state).lo[:2] == (-2.0, 0.0)
    assert Follower.draws == 2


# ---------------------------------------------------------------- 2. registration


def test_two_members_are_enough_and_the_rest_are_filled_in():
    assert all(hasattr(Square, m) for m in CONTRACT_MEMBERS)
    assert Square.world_bounds is record.derived_world_bounds


def test_registration_needs_params_and_draw():
    with pytest.raises(RegistrationError, match="Params"):
        register(type("NoParams", (ObjectBase,), {"type": "m15_no_params",
                                                  "draw": lambda self, ctx, state: None}))  # fmt: skip
    with pytest.raises(RegistrationError, match="draw"):
        register(type("NoDraw", (ObjectBase,), {"type": "m15_no_draw",
                                                "Params": type("P", (ParamsBase,), {})}))  # fmt: skip
    assert get_type("m15_no_params") is None and get_type("m15_no_draw") is None


def test_a_draw_that_takes_the_pass_still_gets_it():
    seen = []

    @register
    class Passy(ObjectBase):
        type: ClassVar[str] = "m15_passy"

        class Params(ParamsBase):
            pass

        def draw(self, ctx, state, pass_):
            seen.append(pass_)

    class Ctx:
        def triangles(self, *a, **k):
            pass

    s = _scene(Passy(id="p"))
    state = s.engine.state_at(0)
    state["p"].impl.draw(Ctx(), state, "solo")
    assert seen == ["solo"]
    # The short form is wrapped once at registration, and its wrapper drops the pass.
    assert getattr(Square.draw, "short_form", False)
    assert not getattr(Passy.draw, "short_form", False)


def test_a_type_defined_twice_by_the_same_file_replaces_itself_but_builtins_never_do():
    def make():
        class Again(ObjectBase):
            type: ClassVar[str] = "m15_again"

            class Params(ParamsBase):
                pass

            def draw(self, ctx, state):
                pass

        return Again

    first, second = register(make()), register(make())
    assert get_type("m15_again") is second and first is not second

    class FakeDot(ObjectBase):
        type: ClassVar[str] = "dot"

        class Params(ParamsBase):
            pass

        def draw(self, ctx, state):
            pass

    with pytest.raises(RegistrationError, match="already registered"):
        register(FakeDot)


def test_every_registered_type_has_all_five_members():
    for name, cls in registered_types().items():
        assert all(callable(getattr(cls, m, None)) or m == "Params" for m in CONTRACT_MEMBERS), name


# ---------------------------------------------------------------- 2b. ported built-in types

PORTED = ("dot", "line", "domain_coloring", "fractal")  # short-form draw; see MILESTONE_15
PORT_GOLDEN = Path(__file__).parent / "golden" / "contract" / "ported_inspector.json"


def _ported_scene(camera: str, passes=None):
    from phasor import Line, Text, cx

    s = Scene("ported", passes=passes)
    if camera == "orbit":
        s.use_camera("orbit", azimuth=30, elevation=25, distance=12)
    else:
        s.set("camera.height", 8)
    s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(6, 3), pos=(0, 0, -0.1)))
    s.add(cx.Fractal(id="set", pos=(-4.5, 2.5, 0), size=(2.4, 1.35)))
    s.add(Dot(id="pole", pos=(0, 1, 0), r=0.25, tags=["key"]))
    s.add(Line(id="path", points=[(-3, -1), (0, 2), (3, -1)], progress=0.7, tags=["key"]))
    s.add(Text(id="pole.label", text="pole", pos=(0.6, 1.4)))
    s.at(0.0).to("pole.pos", (1, -1, 0.5), duration=2)
    return s


def _ported_snapshots(passes=None) -> dict:
    out = {}
    for camera in ("flat", "orbit"):
        e = _ported_scene(camera, passes).engine
        out[camera] = {f"{t:g}": e.inspect(t, detail="full") for t in (0.0, 1.0, 2.0)}
    return json.loads(json.dumps(out, sort_keys=True))


def test_ported_types_are_short_form():
    for name in PORTED:
        assert getattr(get_type(name).draw, "short_form", False), name


def test_the_inspector_output_of_ported_types_is_byte_identical():
    """Frozen before the port (analytic mode: geometry only, the same on every machine)."""
    got = json.dumps(_ported_snapshots(), sort_keys=True, indent=1)
    if os.environ.get("PHASOR_UPDATE_GOLDEN") or not PORT_GOLDEN.exists():
        PORT_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        PORT_GOLDEN.write_text(got + "\n", encoding="utf-8")
    assert got + "\n" == PORT_GOLDEN.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 3. composites


@register
class Pin(ObjectBase):
    """A composite: a stem and a head, built from other types."""

    type: ClassVar[str] = "m15_pin"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        length: float = 1.0

    def build(self, state):
        from phasor import Line

        x, y, z = self.me(state).params.pos
        top = (x, y + self.me(state).params.length, z)
        return [Line(id="stem", points=[(x, y, z), top]), Dot(id="head", pos=top, r=0.2)]


MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


@pytest.mark.parametrize("passes", MODES)
def test_a_composite_is_made_of_parts_inspected_as_themselves(passes):
    s = _scene(Pin(id="pin", pos=(1, 0), length=2), passes=passes)
    state = s.engine.state_at(0)
    assert [o.id for o in state.objects] == ["pin", "pin.stem", "pin.head"]
    assert [o.index for o in state.objects] == [1, 2, 3]
    assert state["pin.head"].parent == "pin" and state["pin"].parent is None
    b = state["pin"].impl.world_bounds(state)
    head = state["pin.head"].impl.world_bounds(state)
    assert b.hi[1] == pytest.approx(head.hi[1]) and b.lo[1] < 1e-6
    f = state["pin"].impl.features(state)
    np.testing.assert_allclose(f["head.center"], [1, 2, 0])
    np.testing.assert_allclose(f["stem.start"], [1, 0, 0])
    snap = s.engine.inspect(0, detail="full")
    entries = {e["id"]: e for e in snap["objects"]}
    assert entries["pin.head"]["semantic"]["kind"] == "point"  # a Dot, inspected as a Dot
    assert entries["pin.head"]["part_of"] == "pin" and "part_of" not in entries["pin"]
    assert entries["pin"]["semantic"]["parts"] == ["stem", "head"]
    if passes is not None:  # the composite is seen where its parts are
        assert entries["pin"]["visible_fraction"] == pytest.approx(1.0)
        assert entries["pin"]["note"] is None
    assert s.engine.lint() == []


def test_parts_fade_with_their_composite():
    s = _scene(Pin(id="pin"))
    s.set("pin.opacity", 0.5)
    state = s.engine.state_at(0)
    assert state["pin.head"].params.opacity == pytest.approx(0.5)


@register
class Named(ObjectBase):
    """A composite whose one part is named by a parameter."""

    type: ClassVar[str] = "m15_named"

    class Params(ParamsBase):
        part: str = "a"

    def build(self, state):
        return [Dot(id=self.me(state).params.part)]


def test_part_ids_follow_the_id_rules():
    from phasor.core.issues import DocumentError

    s = _scene(Named(id="c"))
    assert "c.a" in s.engine.state_at(0)
    s.add(Dot(id="c.x"))  # a document child of the composite, then removed: retired
    s.engine.apply([{"op": "remove_object", "id": "c.x"}], origin="script").raise_for_errors()
    r = s.engine.apply([{"op": "set_param", "path": "c.part", "value": "x"}], origin="script")
    assert not r.ok and "retired" in r.errors[0].message
    s.add(Dot(id="c.y"))
    r = s.engine.apply([{"op": "set_param", "path": "c.part", "value": "y"}], origin="script")
    assert not r.ok and "already taken" in r.errors[0].message
    r = s.engine.apply([{"op": "set_param", "path": "c.part", "value": "Bad"}], origin="script")
    assert not r.ok and r.errors[0].code == "bad_part"
    with pytest.raises(DocumentError):
        s.set("c.part", "x")
