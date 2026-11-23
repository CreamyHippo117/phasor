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
from tests.conftest import gl_available


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


def test_a_recording_follows_the_camera():
    """A stroke's width is a share of the frame: zooming out widens its box."""
    s = _scene(Stroke(id="st"))
    state = s.engine.state_at(0)
    near = state["st"].impl.world_bounds(state)
    s.set("camera.height", 16)
    state = s.engine.state_at(0)
    far = state["st"].impl.world_bounds(state)
    assert far.lo[1] < near.lo[1] and far.hi[1] > near.hi[1]


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


# ---------------------------------------------------------------- 4. types defined in a scene

STAR_MODULE = """
from typing import ClassVar

import numpy as np

from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3
from tests.conftest import gl_available

with open(__file__[: -len(".types.py")] + ".ran", "a", encoding="utf-8") as f:
    f.write("ran\\n")  # how a test knows the module ran


@register
class Star(ObjectBase):
    type: ClassVar[str] = "{name}"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = 0.5

    def draw(self, ctx, state):
        p = self.me(state).params
        a = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        k = np.where(np.arange(10) % 2 == 0, p.r, p.r / 2)
        rim = np.asarray(p.pos) + np.c_[k * np.cos(a), k * np.sin(a), 0 * a]
        pts = np.vstack([p.pos, rim])
        tri = [[0, 1 + i, 1 + (i + 1) % 10] for i in range(10)]
        ctx.triangles(pts, np.array(tri), color=(1, 1, 1, p.opacity))
"""


@pytest.fixture
def star_scene(tmp_path):
    """A folder with ``shapes.types.py`` (a type nobody registered yet) and a document that
    uses it, with a tweak in its overrides file."""
    from phasor.core import io
    from phasor.core.objects import unregister
    from phasor.core.schema import Document, Meta, ObjectSpec, Overrides

    name = f"m15_star_{tmp_path.name[-12:].lower().replace('-', '_')}"
    name = "m15_star_" + "".join(ch for ch in name if ch.isalnum() or ch == "_")[-8:]
    (tmp_path / "shapes.types.py").write_text(STAR_MODULE.replace("{name}", name), "utf-8")
    doc = Document(meta=Meta(title="stars", modules=("shapes.types.py",)),
                   objects=(ObjectSpec(id="star", type=name, params={"pos": (1, 0, 0)}),))  # fmt: skip
    path = tmp_path / "stars.phasor.json"
    io.save(path, doc)
    io.save(io.overrides_path(path), Overrides(params={"star.r": 0.8}))
    yield path, name
    unregister(name)


def _star_ran(path):
    return (path.parent / "shapes.ran").exists()


def _star_type(engine):
    """The class the engine draws ``star`` with (a placeholder when its code is not here)."""
    state = engine.state_at(0)
    return type(state["star"].impl)


def _files(path):
    from phasor.core import io

    return path.read_bytes(), io.overrides_path(path).read_bytes()


def test_a_refused_module_is_missing_type_and_nothing_is_lost(star_scene, tmp_path):
    from phasor.engine import Engine

    path, name = star_scene
    before = _files(path)
    e = Engine.open(path, passes=None)  # scripts may not run: the module is refused
    (issue,) = [i for i in e.lint() if i.code == "missing_type"]
    assert issue.ids == ("star",) and "refused" in issue.message and issue.times == (0.0,)
    assert not _star_ran(path)  # nothing ran
    state = e.state_at(0)
    assert (
        state["star"].type == name and state["star"].impl.semantics(state)["kind"] == "placeholder"
    )
    assert state["star"].impl.world_bounds(state).center[0] == pytest.approx(1)
    assert e.document.object("star").params["r"] == 0.8  # the override is still applied
    e.save(path)
    assert _files(path) == before  # byte for byte


def test_a_module_loads_where_scripts_may_run(star_scene):
    from phasor.engine import Engine

    path, name = star_scene
    e = Engine.open(path, passes=None, allow_scripts=True)
    assert _star_ran(path) and _star_type(e).type == name and e.type_of(name) is _star_type(e)
    assert get_type(name) is None  # the scene's own: the process does not share it
    assert [i.code for i in e.lint()] == []
    snap = e.inspect(0, "star", detail="full")["objects"][0]
    assert snap["semantic"]["type"] == name and snap["semantic"]["params"]["r"] == 0.8
    assert snap["features"]["center"]["world"][0] == pytest.approx([1.0764, 0.0, 0.0])  # its box


def test_a_render_worker_loads_the_scene_code_its_parent_ran(star_scene):
    """A worker is a fresh process (ADR 0003): it has none of the scene's types until it
    loads them, and it may when its parent could."""
    from phasor import engine as engine_mod
    from phasor.core.objects import unregister
    from phasor.engine import Engine

    path, name = star_scene
    parent = Engine.open(path, passes=None, allow_scripts=True)
    payload = parent.worker_payload()
    assert payload["allow_scripts"] is True
    unregister(name)
    engine_mod._LOADED.clear()  # as in a new process
    worker = Engine.from_payload(payload, passes=None)
    assert worker.lint() == [] and _star_type(worker).type == name
    unregister(name)  # a parent that may not run the code sends no permission on
    refused = Engine.open(path, passes=None).worker_payload()
    assert refused["allow_scripts"] is False


def _cli_json(argv, capsys):
    from phasor.cli import main

    code = main(argv)
    return code, capsys.readouterr().out


def test_the_cli_refuses_scene_code_unless_asked(star_scene, capsys):
    """Deviation 9 of Milestone 15: a saved document that names modules opens through the
    CLI with the placeholder, and with the real type under `--allow-scripts`."""
    path, name = star_scene
    base = ["inspect", str(path), "--id", "star", "--detail", "full"]
    code, out = _cli_json(base, capsys)
    assert code == 0 and json.loads(out)["objects"][0]["semantic"]["kind"] == "placeholder"
    assert not _star_ran(path)  # nothing ran
    code, out = _cli_json([*base, "--allow-scripts"], capsys)
    snap = json.loads(out)["objects"][0]
    assert code == 0 and snap["semantic"]["type"] == name
    assert snap["semantic"].get("kind") != "placeholder"


def test_cli_lint_reports_missing_type_until_allowed(star_scene, capsys):
    path, _ = star_scene
    code, out = _cli_json(["lint", str(path), "--json"], capsys)
    assert "missing_type" in out
    code, out = _cli_json(["lint", str(path), "--json", "--allow-scripts"], capsys)
    assert code == 0 and "missing_type" not in out


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_cli_render_draws_the_real_type_only_when_allowed(star_scene, tmp_path):
    from phasor.cli import main

    path, _ = star_scene
    plain, real = tmp_path / "plain.png", tmp_path / "real.png"
    assert main(["render", str(path), "--frame", "0", "-o", str(plain), "--quiet"]) == 0
    assert main(["render", str(path), "--frame", "0", "-o", str(real), "--quiet",
                 "--allow-scripts"]) == 0  # fmt: skip
    assert plain.read_bytes() != real.read_bytes()


def test_the_cli_flag_reaches_the_render_workers(star_scene, tmp_path, monkeypatch):
    """`parallel_frames` builds the worker payload from the engine the CLI opened."""
    import phasor.scene as scene_mod
    from phasor.cli import main

    path, _ = star_scene
    seen = []

    def fake(engine, workers, **kw):
        seen.append(engine.worker_payload()["allow_scripts"])
        return None

    monkeypatch.setattr(scene_mod, "parallel_frames", fake)
    for flags in ([], ["--allow-scripts"]):
        out = tmp_path / "f.png"
        main(["render", str(path), "--frame", "0", "-o", str(out), "--quiet",
              "--workers", "2", *flags])  # fmt: skip
    assert seen == [False, True]


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_spawned_render_workers_draw_the_scene_type_not_a_placeholder(star_scene, tmp_path):
    from phasor.engine import Engine
    from phasor.render.job import render_file
    from phasor.scene import parallel_frames

    path, _ = star_scene
    e = Engine.open(path, allow_scripts=True)
    e.apply([{"op": "set_marker", "name": "end", "t": 0.5},
             {"op": "add_segment", "path": "star.pos", "segment": {
                 "t0": 0.0, "t1": 0.5, "to": [-1, 0, 0]}}]).raise_for_errors()  # fmt: skip
    kw = {"preset": "preview", "fps": 12.0}
    serial = render_file(e, tmp_path / "serial", **kw)
    many = render_file(e, tmp_path / "many", frame_source=parallel_frames(e, 2, chunk=2), **kw)
    a = [p.read_bytes() for p in sorted((tmp_path / "serial").glob("frame_*.png"))]
    b = [p.read_bytes() for p in sorted((tmp_path / "many").glob("frame_*.png"))]
    assert serial["frames"] == many["frames"] == len(a) == len(b) > 2
    assert a == b  # the workers loaded shapes.types.py: no placeholder boxes


def test_a_changed_module_is_read_again(star_scene, tmp_path):
    """Contract 3: files the document names are re-read when they change."""
    import os

    from phasor.engine import Engine

    path, name = star_scene
    e = Engine.open(path, passes=None, allow_scripts=True)
    before = e.state_at(0)["star"].impl.world_bounds(e.state_at(0))
    module = tmp_path / "shapes.types.py"
    module.write_text(STAR_MODULE.replace("{name}", name).replace("p.r / 2", "p.r * 2"), "utf-8")
    st = module.stat()
    os.utime(module, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # a clearly newer file
    again = Engine.open(path, passes=None, allow_scripts=True)
    after = again.state_at(0)["star"].impl.world_bounds(again.state_at(0))
    assert after.hi[1] > before.hi[1] + 0.5  # the inner points now reach further out


def test_a_missing_module_and_a_broken_one(star_scene, tmp_path):
    import sys

    from phasor.engine import Engine

    path, name = star_scene
    (tmp_path / "shapes.types.py").unlink()
    e = Engine.open(path, passes=None, allow_scripts=True)
    assert "missing" in next(i for i in e.lint() if i.code == "missing_type").message
    # A module that registers its type and then fails takes the type with it.
    broken = STAR_MODULE.replace("{name}", name) + "\nraise RuntimeError('half written')\n"
    (tmp_path / "shapes.types.py").write_text(broken, "utf-8")
    modules = set(sys.modules)
    e = Engine.open(path, passes=None, allow_scripts=True)
    issue = next(i for i in e.lint() if i.code == "missing_type")
    assert "RuntimeError: half written" in issue.message
    assert e.type_of(name) is None and set(sys.modules) == modules
    from phasor.core.objects import _SCENE_TYPES

    assert not [k for k in _SCENE_TYPES if k[1] == name]  # rolled back


def test_modules_stay_in_the_scene_folder_and_unknown_types_stay_errors():
    from phasor.core.issues import DocumentError
    from phasor.core.schema import Document, Meta, ObjectSpec

    for bad in ("../elsewhere.py", "/abs/types.py", "notes.txt"):
        with pytest.raises(ValueError):
            Meta(title="x", modules=(bad,))
    from phasor.engine import Engine

    doc = Document(meta=Meta(title="x"), objects=(ObjectSpec(id="a", type="m15_no_such"),))
    with pytest.raises(DocumentError, match="unknown object type"):
        Engine(doc, passes=None)


def test_a_version_7_document_migrates_to_8():
    from phasor.core.migrate import migrate
    from phasor.core.schema import SCHEMA_VERSION, Document

    raw = {"schema_version": 7, "meta": {"title": "old", "palette": None},
           "objects": [{"id": "d", "type": "dot", "params": {"r": 0.2}}]}  # fmt: skip
    doc = Document.model_validate(migrate(raw))
    assert SCHEMA_VERSION == 8 and doc.schema_version == 8
    assert doc.meta.modules == () and doc.objects[0].always == ()


SCRIPT = """
from typing import ClassVar

import numpy as np

from phasor import Scene
from phasor.core.objects import ObjectBase, ParamsBase, register


@register
class Tri(ObjectBase):
    type: ClassVar[str] = "m15_script_tri"

    class Params(ParamsBase):
        size: float = 1.0

    def draw(self, ctx, state):
        s = self.me(state).params.size
        ctx.triangles(np.array([[0, 0, 0], [s, 0, 0], [0, s, 0]]), color=(1, 1, 1, 1))


s = Scene("tri", passes=None)
s.add(Tri(id="tri", size=2))
"""


def test_a_type_in_the_scene_script_is_named_by_the_document(tmp_path):
    from phasor import load
    from phasor.core.objects import unregister
    from phasor.engine import Engine

    script = tmp_path / "tri_scene.py"
    script.write_text(SCRIPT, "utf-8")
    try:
        e = load(script, passes=None)
        assert e.document.meta.modules == ("tri_scene.py",)
        load(script, passes=None)  # run twice in one process: the type replaces itself
        e.save(tmp_path / "tri.phasor.json")
        unregister("m15_script_tri")
        opened = Engine.open(tmp_path / "tri.phasor.json", passes=None, allow_scripts=True)
        assert opened.lint() == [] and opened.type_of("m15_script_tri") is not None
        state = opened.state_at(0)
        assert state["tri"].impl.world_bounds(state).hi[:2] == (2.0, 2.0)
    finally:
        unregister("m15_script_tri")


# ---------------------------------------------------------------- 5. updaters and Value


def follow_dot(t, look):
    """Keep a label just above the dot."""
    x, y, z = look("dot").pos
    return {"pos": (x, y + 0.6, z)}


def circle_dot(t, look):
    return {"pos": (2 * np.cos(t), 2 * np.sin(t), 0.0)}


def _follow_scene(passes=None, dot_moves_by="track"):
    from phasor import Text

    dot = Dot(id="dot", pos=(-2, 0), r=0.15)
    if dot_moves_by == "updater":
        dot.always(circle_dot)
    s = _scene(dot, passes=passes)
    s.add(Text(id="dot.label", text="z", pos=(0, 0)).always(follow_dot))
    if dot_moves_by == "track":
        s.at(0).to("dot.pos", (2, 1, 0), duration=2)
    return s


@pytest.mark.parametrize("passes", MODES)
def test_an_updater_keeps_a_label_on_a_moving_dot(passes):
    s = _follow_scene(passes)
    doc = s.engine.document
    assert doc.object("dot.label").always == ("test_open_contract.py:follow_dot",)
    assert "test_open_contract.py" in doc.meta.modules
    for t in (0.0, 1.0, 2.0):
        state = s.engine.state_at(t)
        x, y, z = state["dot"].params.pos
        assert state["dot.label"].params.pos == pytest.approx((x, y + 0.6, z))
    assert state.values["dot.label.pos"] == state["dot.label"].params.pos
    assert not any(k.startswith("#") for k in state.values)  # no working notes left behind
    assert s.engine.lint() == [] and s.engine.lint(strict=True) == [
        i for i in s.engine.lint(strict=True) if i.code == "raw_motion"]  # fmt: skip


def test_an_updater_can_read_another_updaters_output():
    s = _follow_scene(dot_moves_by="updater")
    state = s.engine.state_at(np.pi / 2)
    assert state["dot"].params.pos == pytest.approx((0, 2, 0), abs=1e-12)
    assert state["dot.label"].params.pos == pytest.approx((0, 2.6, 0), abs=1e-12)
    assert [i for i in s.engine.lint(strict=True) if i.code == "impure_updater"] == []


def reads_q(t, look):
    return {"r": look("q").r}


def reads_p(t, look):
    return {"r": look("p").r}


def reads_late(t, look):
    return {"r": look("a").r if t > 1 else 0.1}


def sets_nothing_real(t, look):
    return {"radius": 1.0}


def test_updater_reads_are_edges_and_a_cycle_is_an_error():
    from phasor.core.issues import DocumentError

    s = _scene(Dot(id="a"))
    here = s.engine.base_dir
    pair = [{"op": "add_object", "object": Dot(id=i).always(f).to_spec(here).model_dump()}
            for i, f in (("p", reads_q), ("q", reads_p))]  # fmt: skip
    names = {"op": "set_meta", "field": "modules", "value": ["test_open_contract.py"]}
    r = s.engine.apply([names, *pair], origin="script")
    assert not r.ok and r.errors[0].code == "cycle" and r.errors[0].ids == ("p", "q")
    with pytest.raises(DocumentError, match="no parameter"):
        s.add(Dot(id="c").always(sets_nothing_real))
    s.add(Dot(id="late").always(reads_late))  # reads a only after t = 1: not an edge it knew
    with pytest.raises(DocumentError, match="did not read when the scene was compiled"):
        s.engine.state_at(2.0)


_calls = {"n": 0}


def counting(t, look):
    _calls["n"] += 1
    return {"r": 0.1 + 0.001 * _calls["n"]}


_shared = {"x": 0.0}


def writer(t, look):
    _shared["x"] = t
    return {"r": 0.2}


def reader(t, look):
    return {"r": 0.1 + 0.01 * _shared["x"]}


@pytest.mark.parametrize("passes", MODES)
def test_impure_updaters_are_caught_by_strict_lint(passes):
    s = _scene(Dot(id="tick", pos=(-2, 0)).always(counting), passes=passes)
    s.add(Dot(id="w", pos=(0, 2)).always(writer))
    s.add(Dot(id="reads", pos=(2, 0)).always(reader))
    s.marker("end", 2.0)
    assert [i.code for i in s.engine.lint()] == []  # only the strict lint checks purity
    found = {i.ids[0]: i for i in s.engine.lint(strict=True) if i.code == "impure_updater"}
    assert sorted(found) == ["reads", "tick"]  # the writer itself is pure
    issue = found["tick"]
    assert issue.severity == "error" and issue.fix == () and issue.times[0] == 0.0
    assert "Simulation" in issue.fix_hint


def test_a_value_is_a_named_number_with_no_drawing():
    from phasor import Counter, Value, expr

    s = _scene(Value(id="speed", value=0.0))
    s.add(Counter(id="speed_readout", value=expr("speed.value"), pos=(0, 2)))
    s.add(Dot(id="runner").always(lambda t, look: {"pos": (look("speed").value, 0, 0)}))
    s.at(0).to("speed.value", 3.0, duration=2)
    state = s.engine.state_at(2.0)
    assert state["runner"].params.pos == (3.0, 0.0, 0.0)
    assert state["speed_readout"].params.value == 3.0
    entry = s.engine.inspect(2.0, "speed")["objects"][0]
    assert entry["semantic"] == {"kind": "value", "value": 3.0, "drawn": 0.0}
    assert s.engine.lint() == [] and s.engine.lint(aesthetic=True) == [
        i for i in s.engine.lint(aesthetic=True) if "speed" not in i.ids]  # fmt: skip
    (ref,) = s.engine.document.object("runner").always
    assert ref.startswith("test_open_contract.py:") and "<lambda>@" in ref


def test_an_updater_that_is_not_here_is_missing_type_and_the_object_keeps_its_value(tmp_path):
    from phasor.core import objects
    from phasor.engine import Engine

    s = _follow_scene()
    path = tmp_path / "follow.phasor.json"
    s.engine.save(path)
    assert any(k[1] == "follow_dot" for k in objects._UPDATERS)  # this process has it
    # The module is the test file, outside this folder: not there, so nothing runs.
    e = Engine.open(path, passes=None, allow_scripts=True)
    (issue,) = [i for i in e.lint() if i.code == "missing_type"]
    assert "follow_dot" in issue.message and "missing" in issue.message
    assert e.state_at(1.0)["dot.label"].params.pos == (0.0, 0.0, 0.0)  # its own value


def test_an_updater_from_outside_the_scene_folder_is_refused():
    """Contract 3: the document must be able to name the file an updater lives in."""
    from phasor.core.schema import ObjectSpec

    ns: dict = {}
    exec(compile("def lift(t, look):\n    return {'pos': (0.0, t, 0.0)}\n", "notebook.ipynb",
                 "exec"), ns)  # fmt: skip
    s = _scene()
    with pytest.raises(ValueError, match="not defined in a file in the scene's folder"):
        s.add(Dot(id="d").always(ns["lift"]))
    with pytest.raises(ValueError, match="must be a .py file"):
        ObjectSpec(id="d", type="dot", always=("<notebook.ipynb>:lift",))


# ---------------------------------------------------------------- 6. the exit demo

EXAMPLE = Path(__file__).parents[1] / "examples" / "open_contract.py"


def _example(passes="auto"):
    import runpy

    s = runpy.run_path(str(EXAMPLE), run_name="x")["s"]
    if passes is None:
        from phasor.engine import Engine

        e = s.engine  # the example is a script: its engine may run its own code, and so this
        return Engine(e.base, overrides=e.overrides, base_dir=e.base_dir, passes=None,
                      allow_scripts=e.allow_scripts)  # fmt: skip
    return s.engine


@pytest.mark.parametrize("passes", MODES)
def test_exit_demo_a_scene_with_its_own_types_through_inspector_layout_and_lint(passes):
    e = _example(passes)
    assert e.document.meta.modules == ("open_contract.py",)
    assert e.document.object("planet").always == ("open_contract.py:orbit",)
    # Inspector: the ten-line type, the composite and its parts, each as itself.
    snap = {o["id"]: o for o in e.inspect(3.0, detail="full")["objects"]}
    assert snap["sun"]["semantic"] == {
        "type": "diamond",
        "params": snap["sun"]["semantic"]["params"],
    }
    assert {"center", "top", "bottom_left"} <= set(snap["sun"]["features"])
    assert snap["flag"]["semantic"]["parts"] == ["pole", "tip"]
    assert snap["flag.tip"]["type"] == "diamond" and snap["flag.tip"]["part_of"] == "flag"
    if passes is not None:
        assert snap["sun"]["visible_fraction"] == 1.0 and snap["flag"]["visible_fraction"] == 1.0
    # The updaters: the planet goes round with ``turn``, its label stays above it.
    for t in (1.0, 3.0, 5.5):
        st = e.state_at(t)
        x, y, z = st["planet"].params.pos
        assert st["planet.label"].params.pos == pytest.approx((x, y + 0.5, z))
    assert e.state_at(1.0)["planet"].params.pos != e.state_at(3.0)["planet"].params.pos
    # Layout: the composite's label was placed beside it, by its box.
    bake = e.layout
    assert "flag.label" in bake.slots and not bake.issues
    # Lint, strict included (purity of both updaters): nothing to say.
    assert e.lint(strict=True) == []


@pytest.mark.gl
@pytest.mark.slow
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_exit_demo_golden_storyboard():
    from tests.test_beauty import _golden

    e = _example()
    sheet = e.storyboard([0.8, 2.5, 4.0, 5.8], size=(320, 180))["image"]
    _golden("open_contract", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- 7. the independent audit
# Findings of the independent audit of Milestone 15 (2026-11-23), repro scripts x1, x2, x4,
# x9, x10 and x11, each a test here. Scene code (a script run as ``python x.py``, by ``load``
# or by the engine) belongs to its own file: a document uses it only through its own
# ``meta.modules`` and only where scripts may run.

AUDIT_SCENE = """
import pathlib
from typing import ClassVar

import numpy as np

from phasor import Dot, Scene
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3

HERE = pathlib.Path(__file__).parent
with open(HERE / "ran.txt", "a", encoding="utf-8") as f:
    f.write("ran\\n")


@register
class Gem(ObjectBase):
    type: ClassVar[str] = "m15_audit_gem"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def draw(self, ctx, state):
        (HERE / "drew.txt").write_text("drew", encoding="utf-8")
        x, y, z = self.me(state).params.pos
        r = SIZE
        ctx.triangles(np.array([[x, y, z], [x + r, y, z], [x, y + r, z]]), color=(1, 1, 1, 1))


SIZE = {size}


def orbit(t, look):
    (HERE / "drew.txt").write_text("updated", encoding="utf-8")
    return {"pos": ({x}, 0.0, 0.0)}


s = Scene("gem", passes=None)
s.add(Gem(id="g"))
s.add(Dot(id="d", pos=(0, 0, 0)).always(orbit))
"""


def _audit_folder(root, name, size=1.0, x=1.0):
    folder = root / name
    folder.mkdir()
    script = folder / "scene.py"
    script.write_text(AUDIT_SCENE.replace("{size}", str(size)).replace("{x}", str(x)), "utf-8")
    return script


def _ran(folder):
    """What the scene code in ``folder`` did: ran (the module), drew (its draw or updater)."""
    return {n: (folder / f"{n}.txt").exists() for n in ("ran", "drew")}


def _forget(folder):
    for n in ("ran", "drew"):
        (folder / f"{n}.txt").unlink(missing_ok=True)


def _missing(engine):
    return sorted(i.ids[0] for i in engine.lint() if i.code == "missing_type")


def _gem_hi(engine):
    st = engine.state_at(0)
    return st["g"].impl.world_bounds(st).hi[0]


def test_audit_x1_a_document_that_may_not_run_scripts_never_runs_code_from_elsewhere(tmp_path):
    """Finding 1 (x1, x9): code a script registered earlier in the process is not used by a
    document opened without scripts, in another folder or in the script's own."""
    import shutil

    from phasor import load
    from phasor.engine import Engine

    a = _audit_folder(tmp_path, "a")
    e = load(a, passes=None)  # the person ran a/scene.py: its type and updater exist now
    assert e.state_at(0.5)["d"].params.pos == (1.0, 0.0, 0.0)
    e.save(a.parent / "a.phasor.json")
    other = tmp_path / "c"
    other.mkdir()  # the same document in a folder that has no scene.py
    shutil.copy(a.parent / "a.phasor.json", other / "c.phasor.json")
    for path in (other / "c.phasor.json", a.parent / "a.phasor.json"):
        _forget(a.parent)
        doc = Engine.open(path, passes=None)  # allow_scripts=False
        assert _missing(doc) == ["d", "g"]
        st = doc.state_at(0.5)
        assert st["d"].params.pos == (0.0, 0.0, 0.0)  # its own value, not the updater's
        assert st["g"].impl.semantics(st)["kind"] == "placeholder"
        doc.inspect(0.5, detail="full")
        assert _ran(a.parent) == {"ran": False, "drew": False}
        assert doc.worker_payload()["allow_scripts"] is False


def test_audit_x1_an_engine_that_loaded_a_module_lends_nothing_to_another(tmp_path):
    """The variant through the engine: folder p's module loaded for an allowed document;
    folder q's document, not allowed, names the same type and updater."""
    from phasor import load
    from phasor.core.objects import unregister
    from phasor.engine import Engine

    for name in ("p", "q"):
        load(_audit_folder(tmp_path, name), passes=None).save(tmp_path / name / "s.phasor.json")
    unregister("m15_audit_gem")
    p = Engine.open(tmp_path / "p" / "s.phasor.json", passes=None, allow_scripts=True)
    assert _missing(p) == [] and _gem_hi(p) == 1.0
    _forget(tmp_path / "q")
    q = Engine.open(tmp_path / "q" / "s.phasor.json", passes=None)
    assert _missing(q) == ["d", "g"]
    q.inspect(0.0, detail="full")
    assert _ran(tmp_path / "p")["drew"] is True  # p's own lint drew
    _forget(tmp_path / "p")
    q.lint(), q.state_at(1.0)
    assert _ran(tmp_path / "p") == _ran(tmp_path / "q") == {"ran": False, "drew": False}


def test_audit_x11_a_worker_never_gets_more_permission_than_its_parent(tmp_path):
    """Finding 2 (x11): the worker payload carries the engine's own permission only; a fresh
    spawned worker of a document that may not run scripts runs none of its folder's code."""
    import pickle
    import subprocess
    import sys

    from phasor import load
    from phasor.engine import Engine

    load(_audit_folder(tmp_path, "h1"), passes=None)  # registers the type and updater
    u = _audit_folder(tmp_path, "u")
    load(u, passes=None).save(u.parent / "u.phasor.json")
    _forget(u.parent)
    parent = Engine.open(u.parent / "u.phasor.json", passes=None)
    payload = parent.worker_payload()
    assert payload["allow_scripts"] is False
    blob = tmp_path / "payload.pkl"
    blob.write_bytes(pickle.dumps(payload))
    code = ("import pickle, sys; from phasor.engine import Engine; "
            "e = Engine.from_payload(pickle.load(open(sys.argv[1], 'rb')), passes=None); "
            "print(sorted(i.code for i in e.lint()))")  # fmt: skip
    r = subprocess.run([sys.executable, "-c", code, str(blob)], capture_output=True, text=True,
                       cwd=Path(__file__).parents[1], timeout=300)  # fmt: skip
    assert r.returncode == 0, r.stderr[-400:]
    assert "missing_type" in r.stdout
    assert _ran(u.parent) == {"ran": False, "drew": False}
    allowed = Engine.open(u.parent / "u.phasor.json", passes=None, allow_scripts=True)
    assert allowed.worker_payload()["allow_scripts"] is True


def test_audit_x1_two_scenes_with_the_same_names_never_see_each_others_code(tmp_path):
    """Finding 3 (x1, x9): two scene folders that both define type ``m15_audit_gem`` and an
    updater ``scene.py:orbit``; alternating between them, each sees only its own."""
    from phasor import load

    ea = load(_audit_folder(tmp_path, "a", size=1.0, x=1.0), passes=None)
    eb = load(_audit_folder(tmp_path, "b", size=4.0, x=-5.0), passes=None)  # no clash
    for _ in range(2):
        assert ea.state_at(0.5)["d"].params.pos == (1.0, 0.0, 0.0)
        assert eb.state_at(0.5)["d"].params.pos == (-5.0, 0.0, 0.0)
        assert _gem_hi(ea) == 1.0 and _gem_hi(eb) == 4.0
        assert ea.apply({"op": "set_param", "path": "d.r", "value": 0.3}).ok
        assert ea.state_at(0.7)["d"].params.pos == (1.0, 0.0, 0.0)
        assert eb.apply({"op": "set_param", "path": "d.r", "value": 0.3}).ok
        assert eb.state_at(0.7)["d"].params.pos == (-5.0, 0.0, 0.0)
        assert _gem_hi(ea.fork()) == 1.0 and _gem_hi(eb.fork()) == 4.0


def test_audit_scene_code_cannot_take_a_shipped_name(tmp_path):
    from phasor import load

    script = tmp_path / "fake.py"
    script.write_text(AUDIT_SCENE.replace("m15_audit_gem", "dot").replace("{size}", "1")
                      .replace("{x}", "1"), "utf-8")  # fmt: skip
    with pytest.raises(RegistrationError, match="already registered"):
        load(script, passes=None)
    assert get_type("dot").__module__ == "phasor.domains.basic"


def test_audit_where_code_comes_from(tmp_path):
    """The line between scene code (scoped to its file, needs permission) and imported code
    (phasor, packages, test modules: shared by the process)."""
    import runpy
    import subprocess
    import sys

    from phasor.core.objects import script_of

    assert script_of(globals()) is None  # a test module is imported
    assert script_of(vars(np)) is None
    ns: dict = {"__name__": "__main__"}  # a notebook cell or a prompt: no file
    exec("def f(t, look):\n    return {}\n", ns)
    assert script_of(ns["f"].__globals__) is None
    path = tmp_path / "run.py"
    path.write_text("def f():\n    pass\n", "utf-8")
    for name in ("__phasor__", "x", "__main__"):  # runpy, any run_name
        f = runpy.run_path(str(path), run_name=name)["f"]
        assert script_of(f.__globals__) == str(path.resolve())
    probe = tmp_path / "probe.py"  # ``python probe.py``
    probe.write_text("from phasor.core.objects import script_of\nprint(script_of(globals()))\n",
                     "utf-8")  # fmt: skip
    out = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True,
                         cwd=Path(__file__).parents[1], timeout=300).stdout.strip()  # fmt: skip
    assert out == str(probe.resolve())


def test_audit_a_short_form_draw_is_still_scene_code(tmp_path):
    """``register`` wraps a two-argument ``draw``; the wrapper lives in phasor, the type does
    not: it is still the script's, and the process does not share it."""
    from phasor import load

    a = _audit_folder(tmp_path, "a")
    e = load(a, passes=None)
    assert get_type("m15_audit_gem") is None  # not shared with the process
    assert e.type_of("m15_audit_gem")._phasor_script == str(a.resolve())


def test_audit_a_scene_built_by_phasor_or_a_test_may_not_run_scripts(tmp_path):
    """``Scene()`` lets scripts run only when a script built it (the person ran it): not when
    an imported module did, such as an importer inside a server."""
    import runpy

    assert Scene("t", passes=None).engine.allow_scripts is False  # built by a test module
    script = tmp_path / "built.py"
    script.write_text("from phasor import Scene\ns = Scene('b', passes=None)\n", "utf-8")
    assert runpy.run_path(str(script))["s"].engine.allow_scripts is True


AUDIT_LATE = """
from phasor import Dot, Scene


def late(t, look):
    out = {"pos": (1.0, 0.0, 0.0)}
    if t > 1.0:
        out["r"] = 0.9
    return out


def branch(t, look):
    if t < 1.0:
        return {"r": look("a").r}
    return {"r": look("b").r + 0.2}


s = Scene("d", passes=None)
s.add(Dot(id="a", r=0.1, tags=["exact"]))
s.add(Dot(id="b", r=0.5, tags=["exact"]))
s.add(Dot(id="d", pos=(0, 0, 0), r=0.2, tags=["exact"]).always({which}))
"""


def test_audit_x2_an_updater_must_set_the_same_parameters_at_every_t(tmp_path):
    """Finding 4 (x2): a parameter an updater returns only later is an error naming the
    parameter and the time, never silently dropped."""
    from phasor import load
    from phasor.core.issues import DocumentError

    script = tmp_path / "late.py"
    script.write_text(AUDIT_LATE.replace("{which}", "late"), "utf-8")
    e = load(script, passes=None)
    assert e.state_at(0.5)["d"].params.r == 0.2
    with pytest.raises(DocumentError, match=r"'r'.*t=2"):
        e.state_at(2.0)


@pytest.mark.parametrize("passes", MODES)
@pytest.mark.parametrize("which", ["late", "branch"])
def test_audit_x2_strict_lint_probes_updaters_beyond_t0(tmp_path, which, passes):
    """The scene has no markers (only t = 0 is a sample); strict lint still tries later
    times and reports the updater that fails there. Plain lint does not crash."""
    from phasor import load

    script = tmp_path / f"{which}.py"
    script.write_text(AUDIT_LATE.replace("{which}", which), "utf-8")
    e = load(script, passes=passes)
    assert [i.code for i in e.lint()] == []
    (issue,) = [i for i in e.lint(strict=True) if i.code == "bad_updater"]
    assert issue.ids == ("d",) and issue.severity == "error" and issue.times
    assert all(t >= 1.0 for t in issue.times) and which in issue.message


@pytest.mark.parametrize("passes", MODES)
@pytest.mark.parametrize("which", ["late", "branch"])
def test_audit_x2_a_scene_that_fails_at_a_sample_is_linted_not_crashed(tmp_path, which, passes):
    """With a marker at 3 s the samples reach the failing times: lint reports the failure
    (an error with its times) instead of raising, and the times that work still evaluate."""
    from phasor import load

    script = tmp_path / f"{which}.py"
    script.write_text(AUDIT_LATE.replace("{which}", which) + 's.marker("end", 3)\n', "utf-8")
    e = load(script, passes=passes)
    assert e.state_at(0.5)["d"].params.r in (0.1, 0.2)
    (issue,) = [i for i in e.lint() if i.ids == ("d",)]
    assert issue.severity == "error" and issue.times and min(issue.times) >= 1.0
    assert which in issue.message
    strict = {i.code for i in e.lint(strict=True) if i.ids == ("d",)}
    assert "bad_updater" in strict


AUDIT_FILES = """
from typing import ClassVar

import numpy as np

from phasor import Scene
from phasor.core import files
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3


@register
class FileBox(ObjectBase):
    type: ClassVar[str] = "m15_filebox"
    Files: ClassVar = ("path",)

    class Params(ParamsBase):
        path: str = "size.txt"

    def draw(self, ctx, state):
        r = float(files.resolve(state.base_dir, self.me(state).params.path).read_text("utf-8"))
        ctx.triangles(np.array([[0, 0, 0], [r, 0, 0], [0, r, 0]]), color=(1, 1, 1, 1))


@register
class TimeBox(ObjectBase):
    type: ClassVar[str] = "m15_timebox"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def draw(self, ctx, state):
        r = 1.0 + state.t
        ctx.triangles(np.array([[0, 0, 0], [r, 0, 0], [0, r, 0]]), color=(1, 1, 1, 1))


@register
class ValueBox(ObjectBase):
    type: ClassVar[str] = "m15_valuebox"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def draw(self, ctx, state):
        r = state.values["fb.opacity"] * 2
        ctx.triangles(np.array([[0, 0, 0], [r, 0, 0], [0, r, 0]]), color=(1, 1, 1, 1))


s = Scene("f", passes=None)
s.add(FileBox(id="fb"))
s.add(TimeBox(id="tb"))
s.add(ValueBox(id="vb"))
s.marker("end", 5)
"""


def test_audit_x4_a_recording_follows_the_files_and_the_time_a_draw_reads(tmp_path):
    """Finding 5 (x4): derived bounds are made again when a file the type names changes, and
    for a draw that reads ``state.t`` or ``state.values``, when what it read changes."""
    from phasor import load

    (tmp_path / "size.txt").write_text("1.0", "utf-8")
    script = tmp_path / "boxes.py"
    script.write_text(AUDIT_FILES, "utf-8")
    e = load(script, passes=None)

    def hi(i, t=0.0):
        st = e.state_at(t)
        return st[i].impl.world_bounds(st).hi[0]

    assert hi("fb") == 1.0
    size = tmp_path / "size.txt"
    size.write_text("3.0", "utf-8")
    st = size.stat()
    os.utime(size, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert hi("fb") == 3.0
    assert (hi("tb", 0.0), hi("tb", 2.0), hi("tb", 0.0)) == (1.0, 3.0, 1.0)
    assert hi("vb") == 2.0
    e.apply({"op": "set_param", "path": "fb.opacity", "value": 0.5}).raise_for_errors()
    assert hi("vb") == 1.0


def test_audit_x10_cached_geometry_is_read_only():
    """Finding 6 (x10): what a recording keeps is shared, so it cannot be written to."""
    s = _scene(Stroke(id="st"))
    state = s.engine.state_at(0)
    f = state["st"].impl.features(state)
    # Every derived feature is read-only (re-audit): the stroke's points are the recording's
    # own (shared), and the box's points are made read-only too, so nobody writes to one.
    assert {"start", "end", "curve", "center", "top", "bottom_right"} <= set(f)
    for name in f:
        assert not f[name].flags.writeable, name
        with pytest.raises(ValueError):
            f[name][...] = 99
    again = state["st"].impl.features(state)
    np.testing.assert_allclose(again["end"], [2, 1, 0])
    np.testing.assert_allclose(again["center"], f["center"])


def _lifted(t, look):
    return {"pos": (0.0, t, 0.0)}


@register
class Lifted(ObjectBase):
    """A composite whose part has an updater: parts have none (finding 7)."""

    type: ClassVar[str] = "m15_lifted"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def build(self, state):
        return [Dot(id="d").always(_lifted)]


def test_audit_a_part_with_an_updater_is_an_error_not_silently_still():
    from phasor.core.issues import DocumentError

    with pytest.raises(DocumentError, match="updater"):
        _scene(Lifted(id="lift"))


def test_audit_only_plain_names_are_adopted_as_updaters(tmp_path):
    """Finding 9: a document names an updater in a module by a plain function name; a dotted
    name (an attribute of something in the module) is not looked up."""
    from phasor.core import io
    from phasor.core.schema import Document, Meta, ObjectSpec
    from phasor.engine import Engine

    (tmp_path / "ups.types.py").write_text(
        "def plain(t, look):\n    return {'r': 0.3}\n\n\nclass Holder:\n"
        "    @staticmethod\n    def lift(t, look):\n        return {'r': 0.4}\n", "utf-8")  # fmt: skip
    for ref, missing in (("ups.types.py:plain", []), ("ups.types.py:Holder.lift", ["d"])):
        doc = Document(meta=Meta(title="u", modules=("ups.types.py",)),
                       objects=(ObjectSpec(id="d", type="dot", always=(ref,)),))  # fmt: skip
        io.save(tmp_path / "u.phasor.json", doc)
        e = Engine.open(tmp_path / "u.phasor.json", passes=None, allow_scripts=True)
        assert _missing(e) == missing


# ---------------------------------------------------------------- 8. the re-audit
# Re-audit of the allow_scripts fix (2026-11-23), repro scripts r1, r3b and r5. Code runs only
# on the person's request: the desktop editor runs a .py the person opened (their own script)
# but opens a .phasor.json without running the modules it names, until the person trusts
# them (``Editor.trust_scene_code``); an attached AI never makes new code run. Files named
# ``ran_*.txt`` prove what ran.

TRUST_MODULE = """
import pathlib
from typing import ClassVar

import numpy as np

from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3

HERE = pathlib.Path(__file__).parent
(HERE / "ran_{name}.txt").write_text("ran", encoding="utf-8")


@register
class Gem(ObjectBase):
    type: ClassVar[str] = "m15_trust_{name}"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = 0.5

    def draw(self, ctx, state):
        x, y, z = self.me(state).params.pos
        r = self.me(state).params.r
        ctx.triangles(np.array([[x, y, z], [x + r, y, z], [x, y + r, z]]), color=(1, 1, 1, 1))


def spin(t, look):
    return {"pos": (t, 0.0, 0.0)}
"""


def _trust_folder(root):
    """A folder with the person's plain document ``plain.phasor.json`` (no code), a document
    ``gem.phasor.json`` that names ``gem.py``, and ``other.py`` that nobody asked to run."""
    from phasor.core import io
    from phasor.core.schema import Document, Meta, ObjectSpec

    folder = root / "w"
    folder.mkdir()
    for name in ("gem", "other"):
        (folder / f"{name}.py").write_text(TRUST_MODULE.replace("{name}", name), "utf-8")
    io.save(folder / "plain.phasor.json",
            Document(meta=Meta(title="plain"), objects=(ObjectSpec(id="d", type="dot"),)))  # fmt: skip
    io.save(folder / "gem.phasor.json", Document(
        meta=Meta(title="gem", modules=("gem.py",)),
        objects=(ObjectSpec(id="g", type="m15_trust_gem"),
                 ObjectSpec(id="d", type="dot", always=("gem.py:spin",)))))  # fmt: skip
    return folder


def _code_ran(folder):
    return sorted(p.stem.removeprefix("ran_") for p in folder.glob("ran_*.txt"))


def _attach(ws):
    """An AI attached to the editor's workspace, as ``phasor mcp --attach`` is."""
    from phasor.workspace.remote import RemoteServer, RemoteWorkspace

    server = RemoteServer(ws).start()
    server.write_handoff()
    return server, RemoteWorkspace.from_handoff(ws.root)


@pytest.mark.parametrize("name", ["gem.phasor.json", "plain.phasor.json"])
def test_reaudit_b1_the_editor_opens_a_document_without_running_its_code(tmp_path, name):
    """r1: the desktop app (Open dialog, recent files, examples, a command-line path) opens a
    document through ``open_workspace``; the modules it names do not run."""
    from phasor.gui.controller import open_workspace

    folder = _trust_folder(tmp_path)
    ws = open_workspace(str(folder / name))
    assert ws.allow_scripts is False and ws.engine.allow_scripts is False
    assert ws.engine.fork().allow_scripts is False
    if name == "gem.phasor.json":
        issues = [i for i in ws.engine.lint() if i.code == "missing_type"]
        assert sorted(i.ids[0] for i in issues) == ["d", "g"]
        assert all("trust_scene_code" in i.message + (i.fix_hint or "") for i in issues)
        st = ws.engine.state_at(1.0)
        assert st["g"].impl.semantics(st)["kind"] == "placeholder"
    assert _code_ran(folder) == []


def test_reaudit_b1_a_script_the_person_opens_runs_and_an_ai_may_only_tweak_it(tmp_path):
    """The person's own .py runs (it is their code); the AI is limited to tweaks."""
    from phasor.gui.controller import open_workspace
    from phasor.workspace.tools import ToolError

    folder = _trust_folder(tmp_path)
    (folder / "mine.py").write_text(
        "from phasor import Dot, Scene\ns = Scene('mine', look='none')\ns.add(Dot(id='d'))\n",
        "utf-8")  # fmt: skip
    ws = open_workspace(str(folder / "mine.py"))
    assert ws.engine.scripted and ws.engine.allow_scripts and ws.allow_scripts is False
    server, ai = _attach(ws)
    try:
        for ops in (
            [{"op": "set_meta", "field": "modules", "value": ["other.py"]}],
            [{"op": "add_object", "object": {"id": "q", "type": "m15_trust_other"}}],
        ):
            try:
                r = ai.scene_edit(ops)
            except ToolError:
                continue
            assert r["ok"] is False
        assert ai.scene_edit({"op": "set_param", "path": "d.r", "value": 0.2})["ok"]
    finally:
        server.stop()
    assert _code_ran(folder) == []


@pytest.mark.parametrize("trusted", [False, True])
def test_reaudit_b1_an_attached_ai_never_makes_new_code_run(tmp_path, trusted):
    """r3b and its variants: whatever the person allowed, the AI cannot name a module, add an
    updater, open a document with its code or a script, or trust code itself."""
    from phasor.gui.controller import Editor, open_workspace
    from phasor.workspace.tools import ToolError

    folder = _trust_folder(tmp_path)
    ws = open_workspace(str(folder / ("gem.phasor.json" if trusted else "plain.phasor.json")))
    ed = Editor(ws)
    if trusted:
        ed.trust_scene_code()
        assert _code_ran(folder) == ["gem"]
        (folder / "ran_gem.txt").unlink()
    server, ai = _attach(ws)
    try:
        attacks = [
            [{"op": "set_meta", "field": "modules", "value": ["other.py"]},
             {"op": "add_object", "object": {"id": "q", "type": "m15_trust_other"}}],
            {"ops": [{"op": "set_meta", "field": "modules",
                      "value": ["gem.py", "other.py"]}]},
        ]  # fmt: skip
        for patch in attacks:
            with pytest.raises(ToolError, match="may not make new code run"):
                ai.scene_edit(patch)
        # A new updater: refused by the guard where the module is trusted, by the schema
        # (its module is not named) where it is not.
        new_updater = {"op": "add_object", "object": {"id": "u", "type": "dot",
                                                      "always": ["gem.py:lift"]}}  # fmt: skip
        if trusted:
            with pytest.raises(ToolError, match=r"may not make new code run \(updaters"):
                ai.scene_edit(new_updater)
        else:
            assert ai.scene_edit(new_updater)["ok"] is False
        assert ws.engine.document.meta.modules == (("gem.py",) if trusted else ())
        if trusted:  # an object of a type the person's trusted module defines is fine
            assert ai.scene_edit(
                {"op": "add_object", "object": {"id": "g2", "type": "m15_trust_gem"}}
            )["ok"]
        with pytest.raises(ToolError, match="no tool"):
            ai.call("trust_scene_code")
        with pytest.raises(ToolError, match="no tool"):
            ai.call("_trust_code")
        with pytest.raises(ToolError, match="does not run scripts"):
            ai.scene_open("other.py")
        ai.scene_open("gem.phasor.json")  # opens, but without its code
        assert ws.engine.allow_scripts is False and ws.engine.fork().allow_scripts is False
        assert _missing(ws.engine) == ["d", "g"]
        ai.scene_edit({"op": "set_param", "path": "d.r", "value": 0.2})
    finally:
        server.stop()
    assert _code_ran(folder) == []
    assert ws.allow_scripts is False


def test_reaudit_b1_trusting_the_scene_code_is_the_persons_choice(tmp_path):
    """The person may trust the modules a document names: they run, the missing_type issues
    go, unsaved edits are kept; the person may also name a module themselves."""
    from phasor.gui.controller import Editor, open_workspace

    folder = _trust_folder(tmp_path)
    ed = Editor(open_workspace(str(folder / "gem.phasor.json")))
    assert sorted(i["ids"][0] for i in ed.lint_items() if i["code"] == "missing_type") == ["d", "g"]
    ed.set_param("g", "r", "0.25")
    assert ed.dirty
    r = ed.trust_scene_code()
    assert r["modules"] == ["gem.py"] and _code_ran(folder) == ["gem"]
    assert ed.engine.allow_scripts and ed.ws.allow_scripts is False
    assert [i for i in ed.lint_items() if i["code"] == "missing_type"] == []
    assert ed.dirty and ed.engine.document.object("g").params["r"] == 0.25
    assert ed.engine.state_at(1.0)["d"].params.pos == (1.0, 0.0, 0.0)  # the updater runs
    ed._edit({"op": "set_meta", "field": "modules", "value": ["gem.py", "other.py"]}, "modules")
    assert ed.engine.document.meta.modules == ("gem.py", "other.py")  # the person may


def test_reaudit_b1_running_code_never_lends_the_permission_to_the_workspace(tmp_path):
    """``Editor.run_code`` runs the person's code without raising the workspace's own
    permission (an attached AI calling ``scene_open`` meanwhile still may not run scripts)."""
    from phasor.gui.controller import Editor, open_workspace

    folder = _trust_folder(tmp_path)
    ws = open_workspace(str(folder / "plain.phasor.json"))
    seen = []
    real = ws._open

    def spy(path, *, allow_scripts):
        seen.append(ws.allow_scripts)
        return real(path, allow_scripts=allow_scripts)

    ws._open = spy
    r = Editor(ws).run_code("from phasor import Dot, Scene\ns = Scene('x', look='none')\n"
                            "s.add(Dot(id='d'))\n")  # fmt: skip
    assert r["ok"] and seen == [False] and ws.allow_scripts is False


CONTROL_SCENE = """
from typing import ClassVar

import numpy as np

from phasor import Scene
from phasor.core.objects import ObjectBase, ParamsBase, register


@register
class Blob(ObjectBase):
    type: ClassVar[str] = "m15_blob"

    class Params(ParamsBase):
        r: float = 0.5

    def draw(self, ctx, state):
        r = self.me(state).params.r
        ctx.triangles(np.array([[0, 0, 0], [r, 0, 0], [0, r, 0]]), color=(1, 1, 1, 1))


s = Scene("k", look="none", passes=None)
s.add(Blob(id="b"))
s.control("b.r", min=0.1, max=2.0)
"""


def test_reaudit_s2_a_slider_on_a_scene_types_default_parameter(tmp_path):
    """r5: a control on a parameter a scene-local type leaves at its default is accepted,
    is not an orphan, and the player and the editor read it; without the code it is the
    missing type that lint reports, not a false orphan, and nothing raises."""
    from phasor import load
    from phasor.engine import Engine
    from phasor.gui.controller import Editor
    from phasor.interactive import Player
    from phasor.workspace import Workspace

    (tmp_path / "scene.py").write_text(CONTROL_SCENE, "utf-8")
    e = load(tmp_path / "scene.py", passes=None)  # s.control("b.r") is not refused
    assert [c.key for c in e.document.controls] == ["b.r"]
    e.save(tmp_path / "k.phasor.json")
    for allow in (True, False):
        e2 = Engine.open(tmp_path / "k.phasor.json", passes=None, allow_scripts=allow)
        codes = [i.code for i in e2.lint()]
        assert "orphan_control" not in codes
        assert ("missing_type" in codes) is (not allow)
        p = Player(e2)
        assert p.values == ({"b.r": 0.5} if allow else {})
        ws = Workspace(tmp_path)
        ws._engine = e2
        ed = Editor(ws)
        assert [c["value"] for c in ed.controls()] == ([0.5] if allow else [])
        if allow:
            assert p.set("b.r", 1.0) == 1.0
            ed.set_control("b.r", 1.5)
            assert ed.apply_controls()["ok"]
            assert e2.document.object("b").params["r"] == 1.5
