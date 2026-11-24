"""Milestone 23, independent audit (2026-11-24): one regression test per finding, each
failing on the audited head (`6b5a4b8`) and passing after the fix."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Notes
from phasor.core.schema import Narration, NarrationBlock
from phasor.domains import midi_math as mm
from phasor.time import narration as nar
from tests.test_midi import scene, smf


def _clear():
    from phasor.domains import midi

    midi._cache.clear()
    midi._read_song.cache_clear() if hasattr(midi._read_song, "cache_clear") else None
    getattr(midi, "_songs", {}).clear()


# ---------------------------------------------------------------- B1 export overwrites


def test_b1_export_never_overwrites_a_file_the_scene_reads(tmp_path):
    from phasor.workspace.tools import ToolError, Workspace

    ws = Workspace(tmp_path)
    ws.scene_new("tune")
    (tmp_path / "src.mid").write_bytes(mm.write_smf([("a", [mm.Note(60, 0, 0.5, 100)])]))
    ws.scene_edit({"op": "add_object", "object": {"id": "m", "type": "notes",
                   "params": {"path": "src.mid", "start": 1.0}}})  # fmt: skip
    before = (tmp_path / "src.mid").read_bytes()
    with pytest.raises(ToolError, match="reads"):
        ws.export_midi("src.mid")
    with pytest.raises(ToolError, match="reads"):
        ws.export_midi("src.mid", overwrite=True)  # never, even when asked
    assert (tmp_path / "src.mid").read_bytes() == before
    assert ws.export_midi("x.mid")["seconds"] == 1.5  # not shifted twice


def test_b1_export_replaces_an_existing_file_only_when_asked_and_atomically(tmp_path):
    s = scene(tmp_path)
    s.add(Notes(id="n", notes=[(60, 0, 0.5, 100)]))
    out = tmp_path / "out.mid"
    out.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        s.export_midi(out)
    assert out.read_bytes() == b"keep me"
    r = s.export_midi(out, overwrite=True)
    assert r["notes"] == 1 and mm.read_smf(out.read_bytes()).error is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mid"]  # no temp file left
    with pytest.raises(ValueError, match="reads"):
        s.add(Notes(id="m", path="out.mid"))
        s.engine.export_midi(out, overwrite=True)


# ---------------------------------------------------------------- S1 clipping, gain


def _peak(tmp_path, notes, instrument, gain=0.0):
    _clear()
    s = scene(tmp_path, 2.0)
    s.add(Notes(id="n", notes=notes, instrument=instrument, gain=gain))
    track, _ = s.engine.audio_track()
    return track.astype(np.int64)


CHORDS = {
    "organ": [(48 + 4 * i, 0, 1, 127) for i in range(8)],
    "sine": [(48 + 4 * i, 0, 1, 127) for i in range(8)],
    "pluck": [(48 + 4 * i, 0, 1, 127) for i in range(8)],
    "lead": [(48 + 4 * i, 0, 1, 127) for i in range(8)],
    "bell": [(60 + 5 * i, 0, 1, 127) for i in range(4)],
    "auto": [(36, 0, 0.5, 127, 9), (38, 0, 0.5, 127, 9), (42, 0, 0.5, 127, 9),
             (49, 0, 0.5, 127, 9), (45, 0, 0.5, 127, 9)],
}  # fmt: skip


@pytest.mark.parametrize("instrument", sorted(CHORDS))
def test_s1_ordinary_chords_do_not_clip(tmp_path, instrument):
    x = _peak(tmp_path, CHORDS[instrument], instrument)
    peak = np.abs(x).max()
    assert 0 < peak < 32767
    assert np.count_nonzero(np.abs(x) == peak) <= 4  # a peak, not a plateau


def test_s1_gain_is_applied_before_any_limiting(tmp_path):
    loud = _peak(tmp_path, CHORDS["organ"], "organ", gain=-12.0)
    quiet = _peak(tmp_path, CHORDS["organ"], "organ", gain=-24.0)
    ratio = np.abs(loud).max() / np.abs(quiet).max()
    assert abs(ratio - 10 ** (12 / 20)) < 0.05 * 10 ** (12 / 20)  # linear: no clipping left


# ---------------------------------------------------------------- S2 aliasing


@pytest.mark.parametrize("rate", [8000, 16000])
def test_s2_notes_under_a_low_rate_narration_are_not_aliased(tmp_path, rate):
    from tests.mix_baseline import _wav

    a = _wav(tmp_path / "say.wav", np.zeros(rate // 10, np.int16) + 1, rate)
    said = Narration(script="A.", voice="file", blocks=(
        NarrationBlock(text="A.", start=0.0, duration=0.1, audio=a),))  # fmt: skip
    _clear()
    s = scene(tmp_path, 2.0)
    s.add(Notes(id="a4", notes=[(69, 0.0, 1.5, 100)], instrument="lead", start=0.3))
    st = s.engine.state_at(0.0)
    track, got = nar.mix_track(said, tmp_path, 2.0, [st["a4"].impl.audio_clip(st)])
    assert got == rate
    x = track[int(0.8 * rate) : int(1.6 * rate)].astype(np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / rate)
    near = np.zeros(len(f), bool)
    for k in range(1, int(rate / 2 / 440) + 1):
        near |= np.abs(f - 440 * k) < 25
    aliased = spec[~near].sum() / spec.sum()
    assert 10 * np.log10(aliased) < -40


# ---------------------------------------------------------------- S3 programs in export


def test_s3_each_objects_program_survives_export(tmp_path):
    s = scene(tmp_path)
    s.add(Notes(id="piano", notes=[(60, 0, 1, 100), (64, 1, 1, 100)], program=0))
    s.add(Notes(id="violin", notes=[(67, 0.5, 2, 100)], program=40))
    s.add(Notes(id="bells", notes=[(72, 0, 1, 100)], instrument="bell"))
    s.export_midi(tmp_path / "o.mid")
    song = mm.read_smf((tmp_path / "o.mid").read_bytes())
    got = {(n.pitch, n.start): mm.timbre_of(n.program, n.channel) for n in song.notes}
    progs = {(n.pitch, n.start): n.program for n in song.notes}
    assert progs[(60, 0.0)] == 0 and progs[(64, 1.0)] == 0 and progs[(67, 0.5)] == 40
    assert got[(72, 0.0)] == "bell"


def test_s3_the_writer_sets_programs_in_time_order():
    data = mm.write_smf([("a", [mm.Note(60, 1, 0.5, 100, 0, 5), mm.Note(62, 0, 0.5, 100, 0, 5)])])
    assert {n.program for n in mm.read_smf(data).notes} == {5}


# ---------------------------------------------------------------- S4 the file cache


def test_s4_many_files_are_each_read_once(tmp_path, monkeypatch):
    from phasor.domains import midi

    _clear()
    s = scene(tmp_path)
    for k in range(6):
        notes = [mm.Note(40 + i % 40, 0.01 * i, 0.05, 90) for i in range(2000)]
        (tmp_path / f"f{k}.mid").write_bytes(mm.write_smf([("a", notes)]))
        s.add(Notes(id=f"f{k}", path=f"f{k}.mid"))
    calls = []
    real = midi.mm.read_smf
    monkeypatch.setattr(midi.mm, "read_smf", lambda d: calls.append(1) or real(d))
    _clear()
    s.engine.lint()
    s.engine.inspect(1.0)
    assert len(calls) == 6


# ---------------------------------------------------------------- S5 zero-length notes


def test_s5_zero_length_drums_play_and_other_zero_notes_are_said():
    drums = bytes.fromhex("00992464 00892400 64992664 00892600 00FF2F00")
    song = mm.read_smf(smf(drums, fmt=0, division=100))
    assert [n.pitch for n in song.notes] == [36, 38]
    melodic = bytes.fromhex("00903C64 00803C00 00FF2F00")
    song = mm.read_smf(smf(melodic, fmt=0, division=100))
    assert song.notes == () and "no length" in song.cut


# ---------------------------------------------------------------- S6 drum tails


@pytest.mark.parametrize("pitch", [35, 36, 38, 42, 45, 46, 49, 50, 51, 60])
def test_s6_a_drum_hit_dies_away_to_silence(pitch):
    syn = mm.Synth([mm.Note(pitch, 0, 0.1, 127, 9)], noise=mm.drum_noise(np.random.default_rng(1)))
    x = syn.read(0, syn.frames)
    assert np.abs(x[-50:]).max() < 1e-3 and abs(x[-1]) < 1e-4


# ---------------------------------------------------------------- S7 long deltas


def test_s7_the_writer_never_writes_what_the_reader_refuses(tmp_path):
    with pytest.raises(ValueError, match="after"):
        mm.write_smf([("a", [mm.Note(60, 0, 1, 100), mm.Note(62, 200000.0, 1, 100)])])
    s = scene(tmp_path)
    s.add(Notes(id="n", notes=[(60, 0, 1, 100), (62, 200000.0, 1, 100)]))
    r = s.export_midi(tmp_path / "far.mid")
    assert r["notes"] == 1 and r["left_out"] == 1
    assert len(mm.read_smf((tmp_path / "far.mid").read_bytes()).notes) == 1


# ---------------------------------------------------------------- S8 running status


def test_s8_running_status_survives_a_meta_event():
    track = b"\x00\x90\x3c\x40" + b"\x00\xff\x01\x01x" + b"\x60\x3c\x00" + b"\x00\xff\x2f\x00"
    song = mm.read_smf(smf(track, fmt=0))
    assert len(song.notes) == 1 and song.notes[0].length == 0.5 and song.cut is None


# ---------------------------------------------------------------- nits


def test_nit_a_note_struck_again_while_held_round_trips():
    notes = [mm.Note(60, 0, 2, 100), mm.Note(60, 0.5, 0.5, 100)]
    back = mm.read_smf(mm.write_smf([("a", notes)])).notes
    assert [(n.start, n.length) for n in back] == [(0.0, 0.5), (0.5, 0.5)]


def test_nit_titles_and_names_keep_their_letters():
    song = mm.read_smf(mm.write_smf([("Grüße ♪", [mm.Note(60, 0, 1, 100)])], title="Étude π"))
    assert song.names == ("Étude π", "Grüße ♪")


def test_nit_a_read_before_the_start_is_silence():
    syn = mm.Synth([mm.Note(69, 0, 0.5, 100)], instrument="sine")
    x = syn.read(-5, 10)
    assert len(x) == 10 and not x[:5].any() and np.array_equal(x[5:], syn.read(0, 5))
