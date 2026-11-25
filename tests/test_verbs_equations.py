"""Group E verbs: substitute, cancel, factor, expand (equations), ignite, pulse (emphasis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Equation, Line, Scene, Text
from phasor.time.verbs import VerbError
from tests.test_beauty import _golden, needs_gl

KEYS = ("y", "eq", "p", "m")
PARTS = ("$y$", "$=a$", "$+b$", "$-b$")


def _scene() -> Scene:
    s = Scene("eqs", passes=None)
    s.add(Equation(id="f", parts=PARTS, keys=KEYS, pos=(0, 2, 0)))
    s.add(Text(id="src", text="a = 3", pos=(-4, -3, 0)))
    s.add(Line(id="ln", points=[(-3, -1, 0), (0, 0, 0), (3, -1, 0)]))
    s.add(Dot(id="d", pos=(3, -3, 0)))
    s.add(Equation(id="g1", parts=("$x$", "$+1$", "$=3$"), keys=("x", "p", "r")))
    s.add(Equation(id="g2", parts=("$x$", "$=2$"), keys=("x", "r")))
    for name, t in (("a", 1.0), ("b", 4.0)):
        s.marker(name, t)
    return s


def _motion(e):
    return e._snap.compiled.theme.motion


def _sem(e, oid, t):
    return e.inspect(t, id=oid)["objects"][0]["semantic"]


def _tags(e, path):
    return {seg.verb for seg in e.document.timeline.tracks[path]}


# ------------------------------------------------------------------ substitute


def test_substitute_swaps_one_part_and_flies_in_from_its_source():
    s = _scene()
    s.at("a").substitute("f", "eq", "$=3$", source="src")
    e = s.engine
    assert _tags(e, "f.swap") == {"substitute"}
    assert _sem(e, "f", 0.5)["parts"][1] == "$=a$"  # before: the old text
    mid = _sem(e, "f", 1.3)
    assert mid["substituting"] == "eq" and mid["replaces"] == "$=a$" and 0 < mid["swap"] < 1
    done = _sem(e, "f", 3.0)
    assert done["parts"] == ["$y$", "$=3$", "$+b$", "$-b$"] and "substituting" not in done
    # a number is typeset
    s2 = _scene()
    s2.at("a").substitute("f", "eq", 3)
    assert _sem(s2.engine, "f", 3.0)["parts"][1] == "$3$"


def test_substitute_part_travels_from_the_source_mid_way():
    s = _scene()
    s.at("a").substitute("f", "eq", "$=3$", source="src")
    s2 = _scene()
    s2.at("a").substitute("f", "eq", "$=3$")  # no source: fades in where it stands
    commands = {}
    for name, sc in (("fly", s), ("fade", s2)):
        st = sc.engine.state_at(1.5)
        ctx = _Recorder()
        st["f"].impl.draw(ctx, st, None)
        commands[name] = ctx.texts
    new = {k: next(c for c in v if c[0] == "$=3$")[1] for k, v in commands.items()}
    # flying text is nearer the source (left, below) than the fading one at the same time
    assert new["fly"][0] < new["fade"][0] and new["fly"][1] < new["fade"][1]


class _Recorder:
    def __init__(self):
        self.texts, self.lines = [], []

    def text(self, text, origin, height, **kw):
        self.texts.append((text, np.asarray(origin), kw.get("color")))

    def triangles(self, positions, *a, **kw):
        self.lines.append(np.asarray(positions))

    def __getattr__(self, name):
        return lambda *a, **kw: None


def test_substitute_errors():
    s = _scene()
    with pytest.raises(VerbError, match="keys"):
        s.at("a").substitute("f", "nope", "$3$")
    with pytest.raises(VerbError, match="needs equation"):
        s.at("a").substitute("src", "eq", "$3$")
    with pytest.raises(VerbError, match="already reads"):
        s.at("a").substitute("f", "eq", "$=a$")
    with pytest.raises(VerbError, match="needs equation|counter|text"):
        s.at("a").substitute("f", "eq", "$3$", source="d")
    with pytest.raises(VerbError, match="no object"):
        s.at("a").substitute("f", "eq", "$3$", source="ghost")
    s.at("a").substitute("f", "eq", "$=3$")
    with pytest.raises(VerbError, match="already has a substitution"):
        s.at("b").substitute("f", "p", "$+c$")
    t = _scene()
    with pytest.raises(VerbError, match="cannot typeset"):
        t.at("a").substitute("f", "eq", "$\\notacommand{$")


# ------------------------------------------------------------------ cancel


def test_cancel_strikes_then_closes_the_gap():
    s = _scene()
    s.at("a").cancel("f", "m", "p")  # whatever order the keys come in
    e = s.engine
    assert _tags(e, "f.strike") == {"cancel"} and _tags(e, "f.close") == {"cancel"}
    assert e.document.object("f").params["struck"] == ("p", "m")
    assert "struck" not in _sem(e, "f", 0.5)
    first = _sem(e, "f", 1.3)  # struck, not yet gone
    assert first["strike"] > 0 and first["close"] == 0 and len(first["parts"]) == 4
    done = _sem(e, "f", 3.0)
    assert done["parts"] == ["$y$", "$=a$"] and done["strike"] == 1.0


def test_cancel_draws_a_stroke_and_the_gap_closes():
    s = _scene()
    s.at("a").cancel("f", "p", "m")
    st0, st1, st2 = (s.engine.state_at(t) for t in (0.5, 1.3, 3.0))
    sizes = []
    for st in (st0, st1, st2):
        sizes.append(st["f"].impl.world_bounds(st))
    assert sizes[0].hi[0] - sizes[0].lo[0] > sizes[2].hi[0] - sizes[2].lo[0] + 0.1
    r0, r1 = _Recorder(), _Recorder()
    st0["f"].impl.draw(r0, st0, None)
    st1["f"].impl.draw(r1, st1, None)
    assert len(r0.lines) == 0 and len(r1.lines) >= 1  # the strike appears with the verb


def test_cancel_errors():
    s = _scene()
    with pytest.raises(VerbError, match="two or more"):
        s.at("a").cancel("f", "p")
    with pytest.raises(VerbError, match="two or more"):
        s.at("a").cancel("f", "p", "p")
    with pytest.raises(VerbError, match="keys"):
        s.at("a").cancel("f", "p", "zzz")
    with pytest.raises(VerbError, match="needs equation"):
        s.at("a").cancel("src", "p", "m")
    s.at("a").cancel("f", "p", "m")
    with pytest.raises(VerbError, match="already has terms crossed out"):
        s.at("b").cancel("f", "y", "eq")


# ------------------------------------------------------------------ factor, expand


@pytest.mark.parametrize("verb", ["factor", "expand"])
def test_factor_and_expand_are_transforms_with_their_own_tag_timing_and_arc(verb):
    s = _scene()
    getattr(s.at("a"), verb)("g1", "g2")
    e = s.engine
    d = _motion(e)[verb]["duration"]
    segs = e.document.timeline.tracks["g2.morph"]
    assert {seg.verb for seg in segs} == {verb}
    assert len(segs) == 1
    assert e._snap.compiled.time_map.resolve(segs[0].t1) == pytest.approx(1.0 + d)
    p = e.document.object("g2").params
    assert p["arc_mode"] == verb and p["from_id"] == "g1"
    mid = _sem(e, "g2", 1.0 + d / 2)
    assert mid["matched"] == 2 and "transforming_from" in mid
    assert _sem(e, "g1", 3.0) is not None and e.state_at(3.0)["g1"].params.opacity == 0.0


def test_factor_and_expand_look_different_from_each_other_and_from_transform():
    def centers(verb):
        s = _scene()
        s.at("a").transform("g1", "g2") if verb == "transform" else getattr(s.at("a"), verb)(
            "g1", "g2"
        )
        d = _motion(s.engine)[verb]["duration"]
        st = s.engine.state_at(1.0 + d / 2)
        r = _Recorder()
        st["g2"].impl.draw(r, st, None)
        return sorted((t, tuple(np.round(o, 3))) for t, o, _ in r.texts), len(r.lines)

    t, f, x = centers("transform"), centers("factor"), centers("expand")
    assert f[0] != x[0] and f[0] != t[0] and x[0] != t[0]
    assert f[1] >= 1 and t[1] == 0 and x[1] == 0  # factor draws a bracket


def test_factor_and_expand_errors():
    s = _scene()
    with pytest.raises(VerbError, match="needs equation"):
        s.at("a").factor("src", "g2")
    with pytest.raises(VerbError, match="needs equation"):
        s.at("a").expand("g1", "ln")


# ------------------------------------------------------------------ ignite


def test_ignite_runs_a_fuse_along_the_stroke_then_settles_it_bright():
    s = _scene()
    s.at("a").ignite("ln")
    e = s.engine
    assert e.document.object("ln.fuse") is not None
    for path in ("ln.fuse.progress", "ln.fuse.opacity", "ln.flash"):
        assert _tags(e, path) == {"ignite"}
    d = _motion(e)["ignite"]["duration"]
    assert _sem(e, "ln.fuse", 0.5)["progress"] == 0.0
    mid = _sem(e, "ln.fuse", 1.0 + 0.35 * d)
    assert 0 < mid["progress"] < 1 and e.state_at(1.0 + 0.35 * d)["ln.fuse"].params.opacity == 1
    end = e.state_at(1.0 + d + 0.2)
    assert end["ln.fuse"].params.opacity == 0.0 and end["ln"].params.flash == pytest.approx(0.3)
    assert end["ln.fuse"].params.lit == 1.0 and end["ln.fuse"].params.trail == 1.0
    assert e.state_at(0.5)["ln.fuse"].params.opacity == 0.0  # not visible before


def test_ignite_trail_stays_lit_behind_the_head():
    s = _scene()
    s.at("a").ignite("ln")
    st = s.engine.state_at(1.0 + 0.5 * _motion(s.engine)["ignite"]["duration"])
    r = _Recorder()
    st["ln.fuse"].impl.draw(r, st, None)
    assert r.lines  # the lit trail is drawn from the start up to the head


def test_ignite_can_run_again_and_checks_what_it_lights():
    s = _scene()
    s.at("a").ignite("ln")
    s.at("b").ignite("ln", settle=0.5)
    assert s.engine.state_at(7.0)["ln"].params.flash == pytest.approx(0.5)
    with pytest.raises(VerbError, match="needs a stroke with a path"):
        s.at("a").ignite("src")
    with pytest.raises(VerbError, match="settle"):
        s.at("a").ignite("ln", settle=2.0)
    with pytest.raises(VerbError, match="at least one"):
        s.at("a").ignite()


# ------------------------------------------------------------------ pulse


def test_pulse_flashes_n_times_evenly():
    s = _scene()
    s.at("a").pulse("d", n=3)
    e = s.engine
    assert _tags(e, "d.flash") == {"pulse"}
    beat = _motion(e)["pulse"]["duration"] / 2
    peaks = [e.state_at(1.0 + k * beat + 0.35 * beat)["d"].params.flash for k in range(3)]
    valleys = [e.state_at(1.0 + k * beat)["d"].params.flash for k in range(1, 3)]
    assert peaks == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert max(valleys) < 0.05
    assert e.state_at(1.0 + 3 * beat + 0.1)["d"].params.flash == 0.0
    assert len(e.document.timeline.tracks["d.flash"]) == 6


def test_pulse_default_staggers_and_checks_n():
    s = _scene()
    s.at("a").pulse("d", "ln")
    assert len(s.engine.document.timeline.tracks["d.flash"]) == 4
    stagger = _motion(s.engine)["stagger"]
    assert s.engine.state_at(1.0 + stagger)["ln"].params.flash == 0.0  # starts after d
    assert s.engine.state_at(1.0 + stagger + 0.35 * 0.4)["ln"].params.flash > 0.9
    for bad in (0, 13, 1.5, True):
        with pytest.raises(VerbError, match="beats"):
            s.at("a").pulse("d", n=bad)
    with pytest.raises(VerbError, match="no object"):
        s.at("a").pulse("ghost")


# ------------------------------------------------------------------ order, lint, off-state


def test_verbs_give_the_same_scene_whatever_order_they_are_written_in():
    def build(order):
        s = _scene()
        s.marker("c", 7.0)
        steps = {
            "sub": lambda: s.at("a").substitute("f", "eq", "$=3$", source="src"),
            "cancel": lambda: s.at("b").cancel("f", "p", "m"),
            "factor": lambda: s.at("c").factor("g1", "g2"),
            "ignite": lambda: s.at("a").ignite("ln"),
            "pulse": lambda: s.at("b").pulse("d", n=3),
        }
        for k in order:
            steps[k]()
        return s.engine

    a = build(["sub", "cancel", "factor", "ignite", "pulse"])
    b = build(["pulse", "ignite", "factor", "cancel", "sub"])
    assert a.document.timeline.model_dump() == b.document.timeline.model_dump()
    objs = lambda e: {o.id: o.model_dump() for o in e.document.objects}  # noqa: E731
    assert objs(a) == objs(b)
    for t in (0.5, 1.4, 2.2, 4.3, 5.0, 8.0):
        assert a.inspect(t, detail="brief") == b.inspect(t, detail="brief")


def test_the_new_equation_params_leave_the_old_picture_alone():
    s = Scene("old", passes=None)
    s.add(Equation(id="e", parts=("$a$", "$+b$", "$=c$"), keys=("a", "b", "c")))
    st = s.engine.state_at(0.0)
    r = _Recorder()
    st["e"].impl.draw(r, st, None)
    assert [t for t, _, _ in r.texts] == ["$a$", "$+b$", "$=c$"] and not r.lines
    sem = _sem(s.engine, "e", 0.0)
    assert set(sem) == {"kind", "text", "parts", "colors", "latex"}


def test_narrated_lint_is_clean_for_the_verbs():
    s = _scene()
    s.at("a").substitute("f", "eq", "$=3$", source="src").then().ignite("ln")
    s.at("b").cancel("g1", "p", "r")
    s.at("b").pulse("d")
    issues = [i for i in s.engine.lint(strict=True) if i.code == "raw_motion"]
    assert issues == []


# ------------------------------------------------------------------ the example


@needs_gl
def test_the_equations_example_matches_its_golden():
    import runpy

    e = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "verbs_equations.py"),
                       run_name="x")["s"].engine  # fmt: skip
    assert not [i for i in e.lint(strict=True) if i.severity == "error"]
    times = sorted(e.document.timeline.markers.values())[1:7]
    sheet = e.storyboard([t + 0.6 for t in times], size=(320, 180))["image"]
    _golden("verbs_equations", sheet, e.pass_runner().renderer_class(), threshold=0.97)
