"""Updaters defined interactively and saved as a copy (ADR 0006).

A notebook cell is imitated the way IPython does it: the cell's text goes into ``linecache``
under the cell's file name and is compiled with that name, so ``inspect.getsource`` finds it.
"""

from __future__ import annotations

import itertools
import json
import linecache
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Scene, Text
from phasor.core import capture
from phasor.core.capture import UpdaterCaptureError
from phasor.core.proc import NO_WINDOW

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]
_cells = itertools.count(1)
HERE = Path(__file__).parent


def cell(source: str, **given):
    """Run ``source`` as a notebook cell: returns its namespace (like ``__main__``)."""
    name = f"<ipython-input-{next(_cells)}-test>"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    ns = {"__name__": "__main__", "np": np, **given}
    exec(compile(source, name, "exec"), ns)
    return ns


ORBIT = "def orbit(t, look):\n    return {'pos': (SPEED * np.cos(t), np.sin(t), 0.0)}\n"
TIMES = (0.0, 0.7, 1.0, 2.5)


def _scene(folder: Path, passes=None, **kw):
    s = Scene("nb", passes=passes, base_dir=folder, **kw)
    s.set("camera.height", 6)
    return s


def _held(s: Scene, obj):
    with pytest.warns(capture.HeldUpdaterWarning):
        s.add(obj)


# ---------------------------------------------------------------- 1. what is captured


def test_a_cell_function_is_captured_with_what_it_needs():
    ns = cell(
        "from math import sqrt\nimport numpy\nWAYS = (1, 2.5, 'x', None)\nK = 3\n"
        "def f(t, look, k=K):\n    return {'pos': (sqrt(2) * numpy.sin(t) * WAYS[0] * k, 0, 0)}\n",
        SPEED=2.0,
    )
    c = capture.capture(ns["f"])
    assert c.name == "f" and c.source.startswith("def f(t, look, k=K):")
    assert dict(c.imports) == {"sqrt": "from math import sqrt", "numpy": "import numpy"}
    assert dict(c.constants) == {"WAYS": "(1, 2.5, 'x', None)", "K": "3"}
    text = capture.module_text([c])
    assert text.index("import numpy") < text.index("WAYS =") < text.index("def f")
    assert text == capture.module_text([c, c])  # the same function twice is written once


def test_a_function_only_reading_builtins_and_its_arguments_needs_nothing():
    c = capture.capture(cell("def f(t, look):\n    return {'r': abs(min(t, 1.0))}\n")["f"])
    assert c.imports == () and c.constants == ()


def test_names_read_inside_comprehensions_and_recursion_are_found():
    ns = cell(
        "def f(t, look):\n    g = [SPEED * i for i in range(3)]\n    return {'r': sum(g)}\n"
        "def h(t, look):\n    return {'r': 1.0} if t <= 0 else h(t - 1, look)\n",
        SPEED=2.0,
    )
    assert dict(capture.capture(ns["f"]).constants) == {"SPEED": "2.0"}
    assert capture.capture(ns["h"]).constants == ()


@pytest.mark.parametrize(
    ("source", "given", "why"),
    [
        ("f = lambda t, look: {'r': t}\n", {}, "lambda"),
        (
            "def make():\n    k = 2\n    def f(t, look):\n        return {'r': k}\n    return f\n"
            "f = make()\n",
            {},
            "inside",
        ),
        (
            "def deco(fn):\n    return fn\n@deco\ndef f(t, look):\n    return {'r': t}\n",
            {},
            "decorator",
        ),
        ("def f(t, look):\n    global COUNT\n    COUNT = 1\n    return {}\n", {}, "global"),
        ("def f(t, look):\n    return {'r': NOWHERE}\n", {}, "'NOWHERE', which is not defined"),
        (
            "def f(t, look):\n    return {'r': WEIGHTS[0]}\n",
            {"WEIGHTS": [1.0, 2.0]},
            "'WEIGHTS' is a list",
        ),
        ("def f(t, look):\n    return {'r': A.sum()}\n", {"A": np.ones(3)}, "ndarray"),
        ("def f(t, look):\n    return {'r': X}\n", {"X": float("nan")}, "'X' is a float"),
        (
            "def f(t, look):\n    return {'r': np.float64(1) * N}\n",
            {"N": np.float64(2)},
            "float64",
        ),
    ],
)
def test_what_cannot_be_written_is_refused_with_a_reason_and_a_fix(source, given, why):
    ns = cell(source, **given)
    fn = ns.get("f")
    with pytest.raises(UpdaterCaptureError, match=why) as e:
        capture.capture(fn)
    assert "cannot save the updater" in str(e.value) and len(str(e.value)) > 80


def test_a_helper_defined_in_the_session_is_refused():
    ns = cell("def helper(x):\n    return x * 2\ndef f(t, look):\n    return {'r': helper(t)}\n")
    with pytest.raises(UpdaterCaptureError, match="'helper' is a function defined in the session"):
        capture.capture(ns["f"])


def test_a_class_or_module_that_only_exists_in_the_session_is_refused():
    import types

    ghost = types.ModuleType("ghost_module_nobody_installed")
    ns = cell("def f(t, look):\n    return {'r': ghost.x}\n", ghost=ghost)
    with pytest.raises(UpdaterCaptureError, match="cannot be imported by name"):
        capture.capture(ns["f"])
    ns = cell("class K:\n    x = 1\ndef f(t, look):\n    return {'r': K.x}\n")
    with pytest.raises(UpdaterCaptureError, match="'K' is a class defined in the session"):
        capture.capture(ns["f"])


def test_a_prompt_that_keeps_no_source_is_refused_with_the_fix():
    # What the plain Python prompt does: compiled under "<stdin>", nothing in linecache.
    ns: dict = {}
    exec(compile("def f(t, look):\n    return {'r': t}\n", "<stdin>", "exec"), ns)
    assert capture.is_interactive(ns["f"])
    with pytest.raises(UpdaterCaptureError, match="kept no source") as e:
        capture.capture(ns["f"])
    assert "IPython or Jupyter" in str(e.value) and ".py file in the scene's folder" in str(e.value)


def test_a_function_from_a_file_is_not_interactive():
    assert not capture.is_interactive(capture.capture)
    assert not capture.is_interactive(lambda t, look: {})  # defined in this file


def test_two_different_functions_or_names_cannot_share_a_name():
    a = capture.capture(cell("def f(t, look):\n    return {'r': 1.0}\n")["f"])
    b = capture.capture(cell("def f(t, look):\n    return {'r': 2.0}\n")["f"])
    with pytest.raises(UpdaterCaptureError, match="two different functions are named 'f'"):
        capture.module_text([a, b])
    c1 = capture.capture(cell("def f(t, look):\n    return {'r': K}\n", K=1)["f"])
    c2 = capture.capture(cell("def g(t, look):\n    return {'r': K}\n", K=2)["g"])
    with pytest.raises(UpdaterCaptureError, match="'K' for two different constants"):
        capture.module_text([c1, c2])


# ---------------------------------------------------------------- 2. the open scene


def test_add_holds_the_updater_and_the_open_scene_does_not_change(tmp_path):
    ns = cell(ORBIT, SPEED=2.0)
    s = _scene(tmp_path)
    before = s.engine.version
    with pytest.warns(capture.HeldUpdaterWarning, match="does not run them"):
        s.add(Dot(id="dot", pos=(-1, 0)).always(ns["orbit"]))
    doc = s.engine.document
    assert doc.object("dot").always == () and doc.meta.modules == ()  # nothing named
    assert s.engine.state_at(1.0)["dot"].params.pos == (-1.0, 0.0, 0.0)  # not run here
    assert s.engine.version == before + 1  # the add, nothing else
    s.save_copy(tmp_path / "copy" / "nb.phasor.json")
    assert s.engine.document == doc and s.engine.version == before + 1
    assert list(tmp_path.glob("*.py")) == []  # nothing written beside the open scene


def test_save_refuses_to_drop_held_updaters_silently(tmp_path):
    s = _scene(tmp_path)
    _held(s, Dot(id="dot").always(cell(ORBIT, SPEED=1.0)["orbit"]))
    with pytest.raises(ValueError, match="save_copy"):
        s.save(tmp_path / "plain.phasor.json")
    assert not (tmp_path / "plain.phasor.json").exists()
    s2 = _scene(tmp_path)  # a scene without any saves as always
    s2.add(Dot(id="dot"))
    s2.save(tmp_path / "plain.phasor.json")


def test_an_uncapturable_updater_is_refused_at_add_and_nothing_is_added(tmp_path):
    ns = cell("def f(t, look):\n    return {'pos': (W[0], 0, 0)}\n", W=[1.0])
    s = _scene(tmp_path)
    with pytest.raises(UpdaterCaptureError, match="'W' is a list"):
        s.add(Dot(id="dot").always(ns["f"]))
    assert s.engine.document.object("dot") is None and s._held == {}


def test_a_function_in_a_file_outside_the_folder_is_still_refused(tmp_path):
    from tests.test_open_contract import follow_dot

    with pytest.raises(ValueError, match="not defined in a file in the scene's folder"):
        _scene(tmp_path).add(Dot(id="dot").always(follow_dot))


# ---------------------------------------------------------------- 3. the saved copy


def _fresh_positions(path: Path) -> list:
    """The dot's position in a new process that opens the copy: nothing of this session."""
    code = (
        "import json, sys\nfrom phasor.engine import Engine\n"
        "e = Engine.open(sys.argv[1], passes=None, allow_scripts=True)\n"
        f"print(json.dumps([e.state_at(t)['dot'].params.pos for t in {list(TIMES)}]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                         text=True, check=True, creationflags=NO_WINDOW)  # fmt: skip
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("passes", MODES)
def test_round_trip_the_copy_is_clean_and_reproducible(tmp_path, passes):
    ns = cell(ORBIT, SPEED=2.0)
    s = _scene(tmp_path, passes=passes)
    _held(s, Dot(id="dot", pos=(-1, 0), r=0.2).always(ns["orbit"]))
    s.add(Text(id="dot.label", text="z", pos=(0, 0)))
    path = tmp_path / "out" / "nb.phasor.json"
    copy = s.save_copy(path)

    module = tmp_path / "out" / "nb.updaters.py"
    assert module.is_file() and "SPEED = 2.0" in module.read_text(encoding="utf-8")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["meta"]["modules"] == ["nb.updaters.py"]
    assert next(o for o in doc["objects"] if o["id"] == "dot")["always"] == ["nb.updaters.py:orbit"]

    expected = [(2.0 * np.cos(t), np.sin(t), 0.0) for t in TIMES]
    for t, want in zip(TIMES, expected, strict=True):
        assert copy.engine.state_at(t)["dot"].params.pos == pytest.approx(want)
    assert copy.engine.lint(strict=True) == []  # impure_updater included
    from phasor.engine import Engine

    again = Engine.open(path, passes=passes, allow_scripts=True)
    assert [again.state_at(t)["dot"].params.pos for t in TIMES] == [
        copy.engine.state_at(t)["dot"].params.pos for t in TIMES
    ]
    assert _fresh_positions(path) == [list(map(float, p)) for p in
                                      (copy.engine.state_at(t)["dot"].params.pos for t in TIMES)]  # fmt: skip
    # Where scripts may not run (a new process, nothing registered) the module is not loaded:
    # missing_type, and the object keeps its own position.
    code = (
        "import sys\nfrom phasor.engine import Engine\n"
        "e = Engine.open(sys.argv[1], passes=None)\n"
        "print([i.code for i in e.lint()], e.state_at(1.0)['dot'].params.pos)"
    )
    closed = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                            text=True, check=True, creationflags=NO_WINDOW)  # fmt: skip
    assert "missing_type" in closed.stdout and "(-1.0, 0.0, 0.0)" in closed.stdout


def test_the_copy_keeps_tracks_overrides_and_the_order_of_updaters(tmp_path):
    local = tmp_path / "local.py"
    local.write_text("def lift(t, look):\n    return {'r': 0.3 + 0.0 * t}\n", encoding="utf-8")
    import importlib.util

    spec = importlib.util.spec_from_file_location("local_mod_for_test", local)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    loose = cell("def grow(t, look):\n    return {'r': 0.1 + 0.1 * t}\n")["grow"]
    s = _scene(tmp_path)
    _held(s, Dot(id="dot", pos=(0, 0)).always(mod.lift).always(loose))
    s.at(0).to("dot.pos", (2, 0, 0), duration=1)
    copy = s.save_copy(tmp_path / "c" / "nb.phasor.json")
    assert copy.engine.document.object("dot").always == ("local.py:lift", "nb.updaters.py:grow")
    assert copy.engine.document.meta.modules == ("local.py", "nb.updaters.py")
    assert (tmp_path / "c" / "local.py").read_bytes() == local.read_bytes()  # code goes along
    st = copy.engine.state_at(1.0)["dot"].params
    assert st.pos == (2.0, 0.0, 0.0) and st.r == pytest.approx(0.2)  # the later updater wins


def test_saving_twice_reuses_the_file_and_changed_code_gets_a_new_one(tmp_path):
    out = tmp_path / "out" / "nb.phasor.json"
    s = _scene(tmp_path)
    _held(s, Dot(id="dot").always(cell(ORBIT, SPEED=2.0)["orbit"]))
    s.save_copy(out)
    stamp = (out.parent / "nb.updaters.py").stat().st_mtime_ns
    s.save_copy(out)
    assert sorted(p.name for p in out.parent.glob("*.py")) == ["nb.updaters.py"]
    assert (out.parent / "nb.updaters.py").stat().st_mtime_ns == stamp  # not rewritten
    # The cell is edited and run again, the scene saved again to the same place.
    s2 = _scene(tmp_path)
    _held(s2, Dot(id="dot").always(cell(ORBIT, SPEED=3.0)["orbit"]))
    copy = s2.save_copy(out)
    assert sorted(p.name for p in out.parent.glob("*.py")) == ["nb.updaters.py", "nb.updaters2.py"]
    assert "SPEED = 2.0" in (out.parent / "nb.updaters.py").read_text(encoding="utf-8")
    assert copy.engine.document.object("dot").always == ("nb.updaters2.py:orbit",)
    assert copy.engine.state_at(0.0)["dot"].params.pos == (3.0, 0.0, 0.0)


def test_a_file_of_the_users_is_never_overwritten(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    mine = out / "nb.updaters.py"
    mine.write_text("# mine, not generated\nX = 1\n", encoding="utf-8")
    s = _scene(tmp_path)
    _held(s, Dot(id="dot").always(cell(ORBIT, SPEED=2.0)["orbit"]))
    s.save_copy(out / "nb.phasor.json")
    assert mine.read_text(encoding="utf-8") == "# mine, not generated\nX = 1\n"
    assert (out / "nb.updaters2.py").is_file()
    # A module the document already names that differs in the new folder stops the copy.
    local = tmp_path / "local.py"
    local.write_text("def lift(t, look):\n    return {}\n", encoding="utf-8")
    clash = tmp_path / "other"
    clash.mkdir()
    (clash / "local.py").write_text("# yours\n", encoding="utf-8")
    s3 = _scene(tmp_path)
    s3.engine.apply({"op": "set_meta", "field": "modules", "value": ["local.py"]}, origin="script")
    s3.add(Dot(id="dot"))
    with pytest.raises(FileExistsError, match="never overwrites"):
        s3.save_copy(clash / "nb.phasor.json")
    assert (clash / "local.py").read_text(encoding="utf-8") == "# yours\n"
    assert not (clash / "nb.phasor.json").exists()


def test_nothing_is_written_when_one_updater_cannot_be_saved(tmp_path):
    s = _scene(tmp_path)
    _held(s, Dot(id="a").always(cell(ORBIT, SPEED=2.0)["orbit"]))
    _held(s, Dot(id="b").always(cell(ORBIT.replace("np.sin", "np.cos"), SPEED=2.0)["orbit"]))
    with pytest.raises(UpdaterCaptureError, match="two different functions are named 'orbit'"):
        s.save_copy(tmp_path / "out" / "nb.phasor.json")
    assert not (tmp_path / "out").exists()


def test_non_ascii_source_is_written_as_utf8(tmp_path):
    ns = cell("def f(t, look):\n    # \u03c0 per second\n    return {'pos': (3.14159 * t, 0, 0)}\n")
    s = _scene(tmp_path)
    _held(s, Dot(id="dot").always(ns["f"]))
    copy = s.save_copy(tmp_path / "nb.phasor.json")
    assert "\u03c0 per second" in (tmp_path / "nb.updaters.py").read_bytes().decode("utf-8")
    assert copy.engine.state_at(1.0)["dot"].params.pos[0] == pytest.approx(3.14159)


@pytest.mark.parametrize("passes", MODES)
def test_an_impure_interactive_updater_is_caught_in_the_copy(tmp_path, passes):
    ns = cell("calls = []\ndef f(t, look):\n    return {'pos': (t, 0, 0)}\n")
    # A function that keeps state in a default argument: pure for capture, not for lint.
    impure = cell("def tick(t, look, seen=[]):\n    seen.append(t)\n"
                  "    return {'pos': (len(seen), 0, 0)}\n")  # fmt: skip
    s = _scene(tmp_path, passes=passes)
    _held(s, Dot(id="ok").always(ns["f"]))
    _held(s, Dot(id="bad").always(impure["tick"]))
    copy = s.save_copy(tmp_path / "nb.phasor.json")
    flagged = {i.ids[0] for i in copy.engine.lint(strict=True) if i.code == "impure_updater"}
    assert flagged == {"bad"}


def test_held_updaters_of_a_removed_object_are_left_out(tmp_path):
    s = _scene(tmp_path)
    _held(s, Dot(id="gone").always(cell(ORBIT, SPEED=2.0)["orbit"]))
    s.engine.apply({"op": "remove_object", "id": "gone"}, origin="script").raise_for_errors()
    copy = s.save_copy(tmp_path / "nb.phasor.json")
    assert copy.engine.document.meta.modules == () and not list(tmp_path.glob("*.updaters*.py"))


def test_a_scene_with_nothing_held_saves_a_plain_copy(tmp_path):
    s = _scene(tmp_path)
    s.add(Dot(id="dot"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        copy = s.save_copy(tmp_path / "x" / "nb.phasor.json")
    assert copy.engine.document == s.engine.document and not list((tmp_path / "x").glob("*.py"))


# ---------------------------------------------------------------- 4. the set_always op


def test_set_always_is_a_patch_op_and_checks_the_module_list(tmp_path):
    s = _scene(tmp_path)
    s.add(Dot(id="dot"))
    bad = s.engine.apply({"op": "set_always", "id": "dot", "always": ["m.py:f"]}, origin="script")
    assert not bad.ok and "meta.modules" in str(bad.errors)
    assert not s.engine.apply({"op": "set_always", "id": "nope", "always": []}, origin="script").ok
    (tmp_path / "m.py").write_text("def f(t, look):\n    return {}\n", encoding="utf-8")
    ok = s.engine.apply([{"op": "set_meta", "field": "modules", "value": ["m.py"]},
                         {"op": "set_always", "id": "dot", "always": ["m.py:f"]}], origin="script")  # fmt: skip
    assert ok.ok and s.engine.document.object("dot").always == ("m.py:f",)
    s.engine.undo()
    assert s.engine.document.object("dot").always == ()
