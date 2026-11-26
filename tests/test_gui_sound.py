"""Sound in the editor (SOUND.md step 4): add music and effects, move them and the narration
on the timeline, master, listen. The logic only (``gui/controller.py``, no Qt); the window
that draws it is tested in ``test_gui_window.py``."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from phasor.gui.audio import Player, backend
from phasor.gui.controller import Editor
from phasor.time.narration import write_wav
from phasor.workspace import Workspace
from tests.test_sound_bed import _narration, _tone


@pytest.fixture
def ed(tmp_path):
    ws = Workspace(tmp_path / "scenes")
    ws.scene_new("with sound")
    folder = ws.engine.base_dir
    _tone(folder / "v.wav", 220, 1.5, amp=12000)
    e = Editor(ws)
    nar = _narration((1.0, 5.0)).model_dump(mode="json")
    assert e.ws.scene_edit([{"op": "set_narration", "narration": nar},
                            {"op": "set_marker", "name": "end", "t": 8.0}])["ok"]  # fmt: skip
    return e


@pytest.fixture
def ambient(tmp_path):
    """A music file somewhere outside the scene's folder, as a person would pick it."""
    path = tmp_path / "elsewhere" / "Night Drift.wav"
    path.parent.mkdir()
    _tone(path, 110, 2.0, amp=6000)
    return path


def test_adding_music_copies_it_into_the_scene_folder(ed, ambient):
    r = ed.add_bed(ambient)
    assert r["ok"]
    (bed,) = ed.engine.document.sound.beds
    assert bed.id == "night_drift" and bed.file == "audio/Night Drift.wav"
    assert (ed.engine.base_dir / "audio" / "Night Drift.wav").read_bytes() == ambient.read_bytes()
    assert ed.add_bed(ambient)["ok"]  # the same file again: a second bed, no second copy
    assert [b.id for b in ed.engine.document.sound.beds] == ["night_drift", "night_drift_2"]
    assert len(list((ed.engine.base_dir / "audio").iterdir())) == 1
    other = ambient.with_name("x") / ambient.name
    other.parent.mkdir()
    _tone(other, 330, 1.0)
    ed.add_bed(other)  # same name, other sound: kept apart
    assert ed.engine.document.sound.beds[-1].file == "audio/Night Drift_1.wav"


def test_an_effect_at_the_playhead_ties_itself_to_the_nearest_marker(ed):
    ed.set_time(5.3)  # just after the second sentence's marker (5.1 s)
    assert ed.add_cue("chime")["ok"]
    (cue,) = ed.engine.document.sound.cues
    assert cue.id == "chime_m1" and cue.at.marker == "m1"
    assert cue.at.offset == pytest.approx(0.2)
    assert ed.add_cue("pop", marker="m0", offset=-0.1)["ok"]
    assert ed.engine.document.sound.cues[-1].at.marker == "m0"


def test_dragging_a_cue_snaps_it_to_a_marker_and_it_follows_the_voice(ed):
    ed.add_cue("tick", marker="m0")
    assert ed.move_cue("tick_m0", 4.8)["ok"]
    cue = ed.engine.document.sound.item("tick_m0")
    assert cue.at.marker == "m1" and cue.at.offset == pytest.approx(-0.3)
    assert ed.move_sentence(1, 6.0)["ok"]  # move the second sentence: its marker and the cue
    (t,) = [c["t"] for c in ed.timeline()["sound"]["cues"]]
    assert t == pytest.approx(5.8)
    ed.undo()
    assert ed.timeline()["sound"]["cues"][0]["t"] == pytest.approx(4.8)


def test_a_sentence_moves_only_between_its_neighbours(ed):
    assert ed.move_sentence(1, 0.0)["ok"]  # cannot go before the first one ends
    blocks = ed.engine.document.narration.blocks
    assert blocks[1].start == pytest.approx(blocks[0].start + blocks[0].duration)
    assert ed.timeline()["markers"]["m1"] == pytest.approx(blocks[1].start + 0.1)
    assert not ed.move_sentence(7, 1.0)["ok"]


def test_sliding_a_bed_keeps_its_length(ed, ambient):
    ed.add_bed(ambient, start={"marker": "m0"}, end={"marker": "m1", "offset": 1.0})
    before = ed.timeline()["sound"]["beds"][0]
    assert ed.move_bed("night_drift", before["start"] + 0.5)["ok"]
    after = ed.timeline()["sound"]["beds"][0]
    assert after["start"] == pytest.approx(before["start"] + 0.5)
    assert after["end"] - after["start"] == pytest.approx(before["end"] - before["start"])


def test_gain_and_removal_are_one_undo_step_each(ed, ambient):
    ed.add_bed(ambient)
    v = ed.version
    assert ed.update_sound("night_drift", gain=-4.0)["ok"] and ed.version == v + 1
    assert ed.engine.document.sound.beds[0].gain == -4.0
    assert not ed.update_sound("nothing", gain=1.0)["ok"]
    assert ed.remove_sound("night_drift")["ok"] and not ed.engine.document.sound.beds
    ed.undo()
    assert ed.engine.document.sound.beds[0].gain == -4.0
    assert [a["who"] for a in ed.activity()][-2:] == ["you", "you"]


def test_master_turns_down_what_covers_the_voice(ed, ambient):
    ed.add_bed(ambient, gain=20.0)
    ed.add_cue("pop", marker="m0", gain=14.0)
    report = ed.sound()
    assert {"bed_masks_voice", "cue_on_word"} <= {i["code"] for i in report["issues"]}
    r = ed.master()
    assert r["ok"] and r["rounds"] and r["before"]["issues"]
    assert not {"bed_masks_voice", "cue_on_word"} & set(r["after"]["issues"])
    assert min(r["after"]["margins"]) >= 12.0
    assert ed.engine.document.sound.beds[0].gain < 20.0
    for _ in r["rounds"]:
        ed.undo()
    assert ed.engine.document.sound.beds[0].gain == 20.0


def test_master_on_a_clipping_mix_brings_it_under(ed, ambient):
    for k in range(4):
        ed.add_bed(ambient, gain=30.0, duck=False, id=f"layer{k}")
    assert "audio_clip" in {i["code"] for i in ed.sound()["issues"]}
    r = ed.master()
    assert "audio_clip" not in r["after"]["issues"]


def test_the_sound_tab_reads_lanes_levels_and_switches(ed, ambient):
    ed.add_bed(ambient)
    ed.add_cue("chime", marker="m0")
    s = ed.sound()
    assert [x["id"] for x in s["lanes"]] == ["voice", "night_drift", "chime_m0"]
    assert s["beds"][0]["file"] == "audio/Night Drift.wav" and s["cues"][0]["sound"] == "chime"
    assert not s["scripted"] and s["muted"] == [] and len(s["sentences"]) == 2
    ed.toggle_mute("night_drift")
    ed.toggle_solo("voice")
    s = ed.sound()
    assert s["muted"] == ["night_drift"] and s["solo"] == ["voice"]
    picture = ed.sound_picture()
    assert picture and picture.endswith(".png")


def _read(path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)
        return data, w.getframerate()


def test_listening_plays_the_mix_from_the_playhead_with_mutes(ed, ambient):
    ed.add_bed(ambient, fade_in=0.0)
    full, rate = _read(ed.listen_file(0.0))
    track, rate2 = ed.engine.audio_track()
    assert rate == rate2 and (full == track).all()  # what you hear is what gets encoded
    later, _ = _read(ed.listen_file(2.0))
    assert len(later) == len(full) - 2 * rate
    ed.toggle_solo("voice")  # only the voice: silent before the first sentence
    only_voice, _ = _read(ed.listen_file(0.0))
    assert not only_voice[: rate // 2].any() and full[: rate // 2].any()
    ed.toggle_solo("voice")
    ed.toggle_mute("voice")
    ed.toggle_mute("night_drift")
    silent, _ = _read(ed.listen_file(0.0))
    assert not silent.any()


def test_a_silent_scene_has_nothing_to_listen_to(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("quiet")
    e = Editor(ws)
    assert e.listen_file() is None and e.sound()["silent"]
    assert e.master() == {"ok": True, "silent": True}


def test_narrating_from_the_editor(ed):
    r = ed.narrate("One sentence [[first]]. Then another [[second]].", voice="estimate")
    assert r["ok"] and set(r["markers"]) >= {"first", "second"}
    assert ed.activity()[-1]["what"] == "narration (estimate)"


def test_a_script_scene_says_where_sound_goes(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("from phasor import Scene\ns = Scene('scripted')\ns.marker('go', 1.0)\n",
                      encoding="utf-8")  # fmt: skip
    ws = Workspace(tmp_path)
    ws.allow_scripts = True
    ws.scene_open(str(script))
    e = Editor(ws)
    assert e.sound()["scripted"]
    r = e.add_cue("pop", marker="go")
    assert not r["ok"] and r["errors"][0]["code"] == "scripted_scene"
    assert not e.master()["ok"] and not e.narrate("x")["ok"]


def test_a_player_without_a_backend_stays_silent(tmp_path):
    p = Player(kind=None)
    assert not p.available and not p.play(tmp_path / "x.wav")
    p.stop()
    assert backend() in (None, "winsound", "afplay", "paplay", "aplay", "ffplay")


@pytest.mark.skipif(backend() not in ("paplay", "aplay", "ffplay"), reason="no player here")
def test_a_player_plays_and_stops(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, (3000 * np.sin(np.arange(8000) * 0.1)).astype(np.int16), 8000)
    p = Player()
    assert p.play(path)
    p.stop()
    assert not p.playing
