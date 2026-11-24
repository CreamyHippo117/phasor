"""Cue sheet (``phasor cues``): every marker and every verb gesture of a scene as one row of
a CSV, so music and sound design can be lined up with the transitions in an audio editor."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from phasor import Dot, Equation, Scene
from phasor.workspace.cues import COLUMNS, cue_rows, cues_csv, timecode

ROOT = Path(__file__).resolve().parents[1]


def _scene() -> Scene:
    s = Scene("cues", passes=None)
    s.add(Dot(id="a", pos=(0.0, 0.0, 0.0)))
    s.add(Dot(id="b", pos=(1.0, 0.0, 0.0)))
    s.add(Equation(id="eq_1", pos=(0.0, 1.0, 0.0), parts=("$x$", "$= 1$")))
    s.add(Equation(id="eq_2", pos=(0.0, 1.0, 0.0), parts=("$x$", "$= 2$")))
    for name, t in (("start", 0.0), ("go", 1.0), ("swap", 2.0), ("slide", 4.0),
                    ("again", 5.0), ("end", 7.0)):  # fmt: skip
        s.marker(name, t)
    # written out of time order on purpose: the sheet is sorted by time
    s.at("swap").transform("eq_1", "eq_2")  # two objects, an instant and a span
    s.at("start").reveal("a", "b")  # staggered: one gesture
    s.at("go").flash("a")  # two touching spans on one track: one gesture
    s.at("slide").morph("a.pos", (1.0, 0.0, 0.0), until="again")  # ends where the next starts
    s.at("again").morph("a.pos", (2.0, 0.0, 0.0), until="end")
    return s


def _names(rows):
    return [(r["kind"], r["name"]) for r in rows]


def test_rows_are_markers_and_gestures_in_time_order():
    rows = cue_rows(_scene().engine)
    assert _names(rows) == [
        ("marker", "start"),
        ("motion", "reveal a b"),
        ("marker", "go"),
        ("motion", "flash a"),
        ("marker", "swap"),
        ("motion", "transform eq_1 eq_2"),
        ("marker", "slide"),
        ("motion", "morph a"),
        ("marker", "again"),
        ("motion", "morph a"),
        ("marker", "end"),
    ]
    by = {(r["kind"], r["name"], r["start"]): r for r in rows}
    reveal = by[("motion", "reveal a b", 0.0)]
    assert reveal["end"] == pytest.approx(0.12 + 0.6) and reveal["time"] == 0.0
    flash = by[("motion", "flash a", 1.0)]
    assert flash["end"] == pytest.approx(1.7) and flash["duration"] == pytest.approx(0.7)
    swap = by[("motion", "transform eq_1 eq_2", 2.0)]
    assert swap["end"] == pytest.approx(3.3)
    # two morphs of one parameter that touch at a marker stay two cues
    assert by[("motion", "morph a", 4.0)]["end"] == pytest.approx(5.0)
    assert by[("motion", "morph a", 5.0)]["end"] == pytest.approx(7.0)
    marker = by[("marker", "swap", None)]
    assert marker["time"] == 2.0 and marker["end"] is None and marker["duration"] is None


def test_every_animated_span_is_in_exactly_one_cue():
    e = _scene().engine
    motions = [r for r in cue_rows(e) if r["kind"] == "motion"]
    for path, spans in e.timeline()["lanes"].items():
        owner = path.rpartition(".")[0]
        for t0, t1 in spans:
            hits = [m for m in motions if owner in m["ids"]
                    and m["start"] - 1e-9 <= t0 and t1 <= m["end"] + 1e-9]  # fmt: skip
            assert len(hits) == 1, (path, t0, t1, hits)


def test_raw_transitions_are_cues_too():
    s = Scene("raw", passes=None)
    s.add(Dot(id="a", pos=(0.0, 0.0, 0.0)))
    s.marker("go", 1.0)
    s.at("go").to("a.r", 0.3, duration=1.5)
    rows = cue_rows(s.engine)
    assert _names(rows) == [("marker", "go"), ("motion", "raw a")]
    assert rows[1]["duration"] == pytest.approx(1.5)


def test_csv_columns_and_text():
    text = cues_csv(cue_rows(_scene().engine))
    lines = text.splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert lines[1] == "0.000,00:00:00.000,marker,start,,,"
    assert lines[2] == "0.000,00:00:00.000,motion,reveal a b,0.000,0.720,0.720"
    assert lines[6] == "2.000,00:00:02.000,motion,transform eq_1 eq_2,2.000,3.300,1.300"
    assert text.endswith("\n") and "\r" not in text
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert len(parsed) == 11 and parsed[3]["name"] == "flash a"


def test_timecode_rounds_to_whole_milliseconds():
    assert timecode(0.0) == "00:00:00.000"
    assert timecode(59.9996) == "00:01:00.000"  # never ":60.000"
    assert timecode(3725.5) == "01:02:05.500"
    assert timecode(1.0005) == "00:00:01.000" or timecode(1.0005) == "00:00:01.001"


def test_a_script_and_its_saved_document_give_the_same_sheet(tmp_path):
    from phasor.scene import load

    s = _scene()
    saved = tmp_path / "cues.phasor.json"
    s.save(saved)
    assert cues_csv(cue_rows(load(saved))) == cues_csv(cue_rows(s.engine))


def test_cli_cues_to_stdout_and_to_a_file(tmp_path, capsys):
    from phasor.cli import main

    scene = ROOT / "examples" / "prism_lift.py"
    assert main(["cues", str(scene)]) == 0
    out = capsys.readouterr().out
    target = tmp_path / "cues.csv"
    assert main(["cues", str(scene), "-o", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == out
    rows = list(csv.DictReader(io.StringIO(out)))
    motions = [(r["name"], r["start"], r["end"]) for r in rows if r["kind"] == "motion"]
    assert motions == [
        ("reveal drum", "0.000", "0.600"),
        ("lift camera", "3.000", "10.000"),  # depth fades in, then the camera tips over
        ("morph drum", "6.500", "10.000"),
        ("orbit camera", "10.000", "16.000"),
    ]
    assert main(["cues", "does_not_exist.phasor.json"]) == 1


def test_ids_are_named_in_the_order_they_move():
    s = Scene("order", passes=None)
    for i in ("z_dot", "a_dot"):
        s.add(Dot(id=i, pos=(0.0, 0.0, 0.0)))
    s.add(Equation(id="new", pos=(0.0, 1.0, 0.0), parts=("$x$", "$= 2$")))
    s.add(Equation(id="old", pos=(0.0, 1.0, 0.0), parts=("$x$", "$= 1$")))
    s.marker("go", 1.0)
    s.at(0.0).reveal("z_dot", "a_dot")  # the stagger order, not the alphabet
    s.at("go").transform("old", "new")  # the source first
    names = [r["name"] for r in cue_rows(s.engine) if r["kind"] == "motion"]
    assert names == ["reveal z_dot a_dot", "transform old new"]
