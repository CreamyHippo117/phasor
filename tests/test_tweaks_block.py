"""Milestone 25, phase 1 (ADR 0004): a script scene keeps its tweaks in a managed block at
the end of the script. The block is the overrides written as code, one line per tweak in a
canonical order; the code outside it is never touched; an old ``name.overrides.json`` is
folded in once and removed."""

from __future__ import annotations

import json

import pytest

from phasor import load
from phasor.core import io
from phasor.core.issues import DocumentError
from phasor.core.pycode import (
    TWEAKS_END,
    TWEAKS_HEADER,
    find_tweaks,
    insert_code,
    read_tweaks,
    tweak_lines,
    with_tweaks,
)
from phasor.core.schema import ConstraintSpec, Overrides
from phasor.workspace import ToolError, Workspace

SCRIPT = """from phasor import Dot, Scene, Text

s = Scene("tweaked", passes=None)
s.add(Dot(id="pole", pos=(1, 0, 0), r=0.2))
s.add(Text(id="note", text="hi", pos=(0, 1, 0)))
"""


def _doc(engine) -> str:
    return io.dumps(engine.document)


def test_the_block_is_canonical_and_reads_back():
    o = Overrides(params={"pole.r": 0.3, "camera.height": 6.0, "note.pos": (1.0, 2.0, 0.0)},
                  styles={"pole.color": "highlight", "note.size": 0.4},
                  constraints=(ConstraintSpec(rule="note near pole"),
                               ConstraintSpec(rule="note above pole", priority="weak")))  # fmt: skip
    lines = tweak_lines(o, "s")
    assert lines == [
        "s.tweak('camera.height', 6.0)",
        "s.tweak('note.pos', [1.0, 2.0, 0.0])",
        "s.tweak('pole.r', 0.3)",
        "s.tweak_style('note', 'size', 0.4)",
        "s.tweak_style('pole', 'color', 'highlight')",
        "s.tweak_rule('note near pole')",
        "s.tweak_rule('note above pole', priority='weak')",
    ]
    text = with_tweaks(SCRIPT, o)
    assert text.startswith(SCRIPT)  # the code above the block is byte for byte the same
    assert TWEAKS_HEADER in text and text.rstrip().endswith(TWEAKS_END)
    assert io.dumps(read_tweaks(text)) == io.dumps(o)
    assert with_tweaks(text, o) == text  # writing the same tweaks again changes nothing
    assert with_tweaks(with_tweaks(text, Overrides()), o) == text
    emptied = with_tweaks(text, Overrides())
    assert emptied.startswith(SCRIPT) and find_tweaks(emptied) is not None  # markers stay
    assert with_tweaks(SCRIPT, Overrides()) == SCRIPT  # no tweaks, no block


def test_the_block_keeps_line_endings_and_goes_before_a_main_guard():
    crlf = SCRIPT.replace("\n", "\r\n") + 'if __name__ == "__main__":\r\n    s.render("x.mp4")\r\n'
    o = Overrides(params={"pole.r": 0.3})
    out = with_tweaks(crlf, o)
    assert "\n" not in out.replace("\r\n", "")  # only CRLF line ends
    a, b = find_tweaks(out)
    assert out[:a] == SCRIPT.replace("\n", "\r\n")  # before the guard, the rest untouched
    assert out[b:].startswith("\r\nif __name__")
    added = insert_code(out, ["s.add(Dot(id='more'))"])
    assert added.index("s.add(Dot(id='more'))") < added.index(TWEAKS_HEADER)
    plain = insert_code(SCRIPT, ["s.marker('go', 1.0)"])
    assert plain == SCRIPT + "s.marker('go', 1.0)\n"


def test_scene_tweaks_land_in_the_overrides_and_orphans_are_kept(tmp_path):
    path = tmp_path / "s.py"
    path.write_text(SCRIPT + "s.tweak('pole.r', 0.35)\ns.tweak_style('pole', 'color', 'highlight')\n"
                    "s.tweak('gone.r', 0.1)\ns.tweak_rule('note near pole')\n",
                    encoding="utf-8")  # fmt: skip
    e = load(path, passes=None)
    assert e.base.object("pole").params["r"] == 0.2  # the script's own value...
    assert e.document.object("pole").params["r"] == 0.35  # ...with the tweak laid over it
    assert e.overrides.params["gone.r"] == 0.1  # a tweak whose object is gone is kept
    assert any(i.code == "orphan_override" for i in e.lint())
    assert e.document.object("pole").style["color"] == "highlight"
    assert [c.rule for c in e.overrides.constraints] == ["note near pole"]
    path.write_text(SCRIPT + "s.tweak('pole.r', -4)\n", encoding="utf-8")
    with pytest.raises(DocumentError):  # a tweak with a bad value is an error at its line
        load(path, passes=None)


def test_saving_writes_the_block_and_a_fresh_run_equals_the_live_scene(tmp_path):
    (tmp_path / "s.py").write_text(SCRIPT, encoding="utf-8")
    ws = Workspace(tmp_path, allow_scripts=True)
    ws.scene_open("s.py")
    assert ws.script_var == "s" and ws.script_lines["pole"] == 4
    assert ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.31})["ok"]
    assert ws.scene_edit({"op": "set_style", "id": "note", "key": "color", "role": "muted"})["ok"]
    assert not ws.diff("saved", "current")["same"]  # the block on disk does not have them yet
    r = ws.scene_save()
    assert r["saved"] == str(tmp_path / "s.py")
    text = (tmp_path / "s.py").read_text(encoding="utf-8")
    assert text.startswith(SCRIPT) and "s.tweak('pole.r', 0.31)" in text
    assert (
        not (tmp_path / "s.overrides.json").exists() and not (tmp_path / "s.phasor.json").exists()
    )
    assert _doc(load(tmp_path / "s.py", passes=None)) == _doc(ws.engine)
    assert ws.diff("saved", "current")["same"]
    ws.undo()  # the style
    ws.scene_save()
    assert "tweak_style" not in (tmp_path / "s.py").read_text(encoding="utf-8")
    ws.undo()  # and the radius
    ws.scene_save()
    assert "pole.r" not in (tmp_path / "s.py").read_text(encoding="utf-8")
    assert _doc(load(tmp_path / "s.py", passes=None)) == _doc(ws.engine)
    with pytest.raises(ToolError, match="only to the script"):
        ws.scene_save("other.py")
    copy = ws.scene_save("copy.phasor.json")  # a copy as a document, tweaks included
    assert copy["saved"].endswith("copy.phasor.json")


def test_an_old_overrides_file_is_folded_into_the_block_once(tmp_path):
    """The migration test: a scene saved before ADR 0004 (script + overrides file) opens
    with the same document; its tweaks are now in the script and the file is gone."""
    script = tmp_path / "old.py"
    script.write_text(SCRIPT, encoding="utf-8")
    before_bytes = script.read_bytes()
    io.save(tmp_path / "old.overrides.json",
            Overrides(params={"pole.r": 0.5, "lost.r": 0.2}, styles={"pole.color": "danger"},
                      constraints=(ConstraintSpec(rule="note near pole"),)))  # fmt: skip
    expected = _doc(load(script, passes=None))  # what the old scene looked like
    ws = Workspace(tmp_path, allow_scripts=True)
    r = ws.scene_open("old.py")
    assert r["migrated"].endswith("old.overrides.json")
    assert not (tmp_path / "old.overrides.json").exists()
    text = script.read_text(encoding="utf-8")
    assert script.read_bytes().startswith(before_bytes)  # the person's code untouched
    assert "s.tweak('lost.r', 0.2)" in text  # an orphan is kept, never dropped
    assert _doc(ws.engine) == expected
    assert _doc(load(script, passes=None)) == expected  # a fresh run, without the file
    again = Workspace(tmp_path, allow_scripts=True).scene_open("old.py")
    assert "migrated" not in again and script.read_text(encoding="utf-8") == text


def test_a_document_keeps_its_overrides_file(tmp_path):
    """ADR 0004 changes script scenes only: a .phasor.json scene saves as before."""
    ws = Workspace(tmp_path)
    ws.scene_new("doc")
    ws.scene_edit({"op": "add_object", "object": {"id": "d", "type": "dot"}})
    ws.scene_save("doc.phasor.json")
    assert json.loads((tmp_path / "doc.phasor.json").read_text(encoding="utf-8"))["objects"]
