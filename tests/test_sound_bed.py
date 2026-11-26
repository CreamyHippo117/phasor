"""Beds, cues and ducking in the document (SOUND.md steps 2 and 3)."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from phasor import Scene
from phasor.core.issues import DocumentError
from phasor.core.migrate import migrate
from phasor.core.schema import Bed, Cue, Document, Narration, NarrationBlock, Word
from phasor.time import sound as snd
from phasor.time.narration import write_wav
from phasor.time.synth import builtin

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]
RATE = 22_050


def _tone(path, freq, seconds, amp=8000, rate=RATE):
    t = np.arange(round(seconds * rate)) / rate
    write_wav(path, (amp * np.sin(2 * math.pi * freq * t)).astype(np.int16), rate)


def _narration(starts, length=1.5, audio="v.wav"):
    """One sentence per start, each one word long, a marker on each word."""
    return Narration(script="x", voice="test", blocks=tuple(
        NarrationBlock(text=f"w{k}", start=s, duration=length, audio=audio,
                       words=(Word(text=f"w{k}", start=0.1, end=length - 0.1),),
                       markers={f"m{k}": 0.1})
        for k, s in enumerate(starts)))  # fmt: skip


def _scene(tmp_path, passes=None, voice=(1.0, 5.0), **kw):
    _tone(tmp_path / "v.wav", 220, 1.5, amp=12000)
    _tone(tmp_path / "bed.wav", 110, 2.0, amp=6000)
    s = Scene("snd", passes=passes, base_dir=tmp_path, look="none", **kw)
    if voice:
        s.engine.apply(
            {"op": "set_narration", "narration": _narration(voice).model_dump()}, origin="script"
        )
    s.marker("end", 8.0)
    return s


def _rms_db(x):
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12)


# ---------------------------------------------------------------- the document


def test_cue_sounds_are_built_ins_or_files_in_the_folder():
    assert Cue(id="c", sound="chime", at=0.0).sound == "chime"
    assert Cue(id="c", sound="audio\\whoosh.wav", at=0.0).sound == "audio/whoosh.wav"
    with pytest.raises(ValueError, match="no built-in sound 'gong'"):
        Cue(id="c", sound="gong", at=0.0)
    for bad in ("/etc/x.wav", "../x.wav"):
        with pytest.raises(ValueError):
            Cue(id="c", sound=bad, at=0.0)
        with pytest.raises(ValueError):
            Bed(id="b", file=bad)


def test_sound_ids_are_unique_among_beds_cues_and_objects(tmp_path):
    s = _scene(tmp_path, voice=None)
    from phasor import Dot

    s.add(Dot(id="hum"))
    with pytest.raises(DocumentError, match="duplicate id 'hum'"):
        s.bed("bed.wav", id="hum")
    s.bed("bed.wav")
    with pytest.raises(DocumentError, match="duplicate id 'bed'"):
        s.cue("end", "chime", id="bed")


def test_a_version_8_document_migrates_to_silence():
    raw = {"schema_version": 8, "meta": {"title": "old", "palette": None, "modules": []}}
    doc = Document.model_validate(migrate(raw))
    assert doc.sound.beds == () and doc.sound.cues == ()


def test_set_bed_replaces_in_place_and_remove_sound_removes(tmp_path):
    s = _scene(tmp_path, voice=None)
    s.bed("bed.wav", id="one")
    s.bed("bed.wav", id="two")
    s.bed("bed.wav", id="one", gain=-3.0)
    beds = s.engine.document.sound.beds
    assert [b.id for b in beds] == ["one", "two"] and beds[0].gain == -3.0
    assert s.engine.apply({"op": "remove_sound", "id": "one"}, origin="script").ok
    assert not s.engine.apply({"op": "remove_sound", "id": "one"}, origin="script").ok
    s.engine.undo()
    assert [b.id for b in s.engine.document.sound.beds] == ["one", "two"]


def test_a_cue_on_a_missing_marker_is_refused(tmp_path):
    s = _scene(tmp_path, voice=None)
    r = s.engine.apply({"op": "set_cue", "cue": {"id": "c", "sound": "pop",
                                                 "at": {"marker": "nowhere"}}},
                       origin="script")  # fmt: skip
    assert not r.ok and r.errors[0].code == "unknown_marker"


def test_cue_ids_are_named_after_the_sound_and_the_marker(tmp_path):
    s = _scene(tmp_path)
    assert s.cue("m0", "chime") == "chime_m0"
    assert s.cue("m1", "Whoosh Loud.wav", offset=-0.2) == "whoosh_loud_m1"
    cue = s.engine.document.sound.item("whoosh_loud_m1")
    assert cue.at.marker == "m1" and cue.at.offset == pytest.approx(-0.2)
    assert s.bed("audio/Drift-01.ogg") == "drift_01"


def test_the_generated_script_rebuilds_the_sound(tmp_path):
    from phasor.core.pycode import document_to_python

    s = _scene(tmp_path, voice=None)
    s.bed("bed.wav", start=("end", -6.0), end="end", loop=False, gain=-2.0, fade=0.5)
    s.cue("end", "chime", offset=-1.0, pan=0.5)
    code = document_to_python(s.engine.document)
    assert "s.bed('bed.wav'" in code and "s.cue(('end', -1.0), 'chime'" in code
    scope = {"BASE": tmp_path}
    head, rest = code.split("s = Scene(", 1)
    call, tail = rest.split("\n", 1)
    exec(f"{head}s = Scene({call[:-1]}, passes=None, base_dir=BASE)\n{tail}", scope)
    rebuilt = scope["s"]
    assert rebuilt.engine.document.sound == s.engine.document.sound


# ---------------------------------------------------------------- levels and ducking


def test_the_voice_is_set_to_the_theme_level_and_beds_and_cues_under_it(tmp_path):
    s = _scene(tmp_path)
    s.bed("bed.wav", duck=False)
    s.cue("m0", "chime")
    m = s.engine.sound_mix()
    lv = s.engine._snap.compiled.theme.sound
    voice = snd.mix([snd.Placed(x.start, x.samples, m.rate) for x in m.lanes
                     if x.kind == "voice"], m.duration)[0]  # fmt: skip
    assert snd.loudness(voice, m.rate) == pytest.approx(lv["voice"], abs=0.2)
    bed = m.lane("bed")
    steady = bed.samples[round(lv["fade"] * m.rate) : -round(lv["fade"] * m.rate)]
    assert snd.loudness(steady, m.rate) == pytest.approx(lv["voice"] + lv["bed"], abs=0.3)
    chime = m.lane("chime_m0")
    assert snd.momentary_max(chime.samples, m.rate) == pytest.approx(lv["voice"] + lv["cue"],
                                                                     abs=0.5)  # fmt: skip


def test_without_narration_a_bed_plays_at_the_voice_level(tmp_path):
    s = _scene(tmp_path, voice=None)
    s.bed("bed.wav", fade=0.0)
    m = s.engine.sound_mix()
    assert snd.loudness(m.lane("bed").samples, m.rate) == pytest.approx(-16.0, abs=0.3)


def test_beds_dip_while_the_voice_speaks(tmp_path):
    s = _scene(tmp_path, voice=(2.0,))
    s.bed("bed.wav", fade=0.0)
    m = s.engine.sound_mix()
    bed, r = m.lane("bed").samples, m.rate
    duck = s.engine._snap.compiled.theme.sound["duck"]
    (a, b), = m.speech  # fmt: skip
    assert (a, b) == pytest.approx((2.1, 3.4))
    before = _rms_db(bed[round(0.2 * r) : round(1.0 * r)])
    during = _rms_db(bed[round((a + 0.2) * r) : round((b - 0.2) * r)])
    after = _rms_db(bed[round(5.0 * r) : round(6.0 * r)])
    assert before - during == pytest.approx(duck, abs=0.2)
    assert after == pytest.approx(before, abs=0.2)


def test_duck_envelope_ramps_in_before_and_out_after_speech():
    t = np.array([0.0, 0.85, 1.0, 1.5, 2.0, 2.4, 3.0])
    env = snd.duck_envelope(t, [(1.0, 2.0)], 8.0, attack=0.3, release=0.8)
    np.testing.assert_allclose(env, [0, -4, -8, -8, -8, -4, 0], atol=1e-9)


def test_a_bed_fades_in_loops_and_stops_at_its_end(tmp_path):
    s = _scene(tmp_path, voice=None)
    s.bed("bed.wav", start=1.0, end=6.0, fade=1.0)
    s.bed("bed.wav", id="once", loop=False, fade=0.0)
    m = s.engine.sound_mix()
    lane = m.lane("bed")
    assert lane.start == 1.0 and len(lane.samples) == 5 * m.rate
    assert np.abs(lane.samples[:10]).max() < 1e-3  # faded in from silence
    assert np.abs(lane.samples[-10:]).max() < 1e-3  # and out
    assert _rms_db(lane.samples[3 * m.rate : 4 * m.rate]) > -40  # looped past the 2 s file
    assert len(m.lane("once").samples) == 2 * m.rate  # played once


def test_pan_moves_a_cue_to_one_side():
    left = snd._balance(-1.0)
    assert left.tolist() == [1.0, 0.0] and snd._balance(0.0).tolist() == [1.0, 1.0]


def test_cues_follow_their_markers_when_the_narration_is_re_recorded(tmp_path):
    s = _scene(tmp_path, voice=(1.0, 5.0))
    s.cue("m1", "pop", offset=-0.1)
    assert s.engine.sound_mix().lane("pop_m1").start == pytest.approx(5.0)
    s.engine.apply(
        {"op": "set_narration", "narration": _narration((1.0, 7.0)).model_dump()}, origin="script"
    )
    assert s.engine.sound_mix().lane("pop_m1").start == pytest.approx(7.0)


def test_the_mix_is_reread_when_a_file_changes(tmp_path):
    import os

    s = _scene(tmp_path, voice=None)
    s.bed("bed.wav", fade=0.0, loop=False)
    first = s.engine.audio_track()[0]
    _tone(tmp_path / "bed.wav", 330, 1.0)
    st = os.stat(tmp_path / "bed.wav")
    os.utime(tmp_path / "bed.wav", (st.st_atime, st.st_mtime + 5))
    second = s.engine.audio_track()[0]
    assert first.tobytes() != second.tobytes()


def test_the_rendered_sound_is_the_same_every_time(tmp_path):
    s = _scene(tmp_path)
    s.bed("bed.wav")
    s.cue("m0", "whoosh")
    a = s.engine.audio_track()
    b = Scene.of(s.engine.fork()).engine.audio_track()
    assert a[1] == b[1] and a[0].tobytes() == b[0].tobytes()


# ---------------------------------------------------------------- built-in effects


@pytest.mark.parametrize("name", ["chime", "pop", "swell", "tick", "whoosh"])
def test_built_in_sounds_are_pure_functions_of_name_id_and_seed(name):
    x, rate = builtin(name, "cue", 0)
    assert rate == 48_000 and x.dtype == np.float32 and not x.flags.writeable
    assert 0.25 < np.abs(x).max() <= 0.5 + 1e-6
    assert abs(x[0]) < 1e-3 and abs(x[-1]) < 1e-3  # no clicks at the ends
    from phasor.time import synth

    synth._make.cache_clear()
    y, _ = builtin(name, "cue", 0)
    assert hashlib.sha256(x.tobytes()).digest() == hashlib.sha256(y.tobytes()).digest()
    assert builtin(name, "other", 0)[0].tobytes() != x.tobytes()
    assert builtin(name, "cue", 1)[0].tobytes() != x.tobytes()


# ---------------------------------------------------------------- lint


def _codes(engine, **kw):
    return sorted({i.code for i in engine.lint(**kw)})


@pytest.mark.parametrize("passes", MODES)
def test_a_missing_sound_file_is_reported_and_silent(tmp_path, passes):
    s = _scene(tmp_path, passes, voice=None)
    s.bed("gone.ogg")
    (issue,) = [i for i in s.engine.lint() if i.code == "missing_file"]
    assert issue.ids == ("gone",) and "gone.ogg" in issue.message
    assert s.engine.audio_track() is None


@pytest.mark.parametrize("passes", MODES)
def test_a_bed_too_loud_under_the_voice_is_reported_and_the_fix_clears_it(tmp_path, passes):
    s = _scene(tmp_path, passes)
    s.bed("bed.wav")
    assert "bed_masks_voice" not in _codes(s.engine)
    s.bed("bed.wav", gain=20.0)
    (issue,) = [i for i in s.engine.lint() if i.code == "bed_masks_voice"]
    assert issue.ids == ("bed",) and len(issue.times) == 2
    assert s.engine.apply(list(issue.fix), origin="script").ok
    assert "bed_masks_voice" not in _codes(s.engine)
    assert s.engine.document.sound.beds[0].gain < 20.0


@pytest.mark.parametrize("passes", MODES)
def test_a_mix_far_over_full_scale_is_reported(tmp_path, passes):
    s = _scene(tmp_path, passes, voice=None)
    for k in range(4):
        s.bed("bed.wav", id=f"layer{k}", gain=12.0)
    assert "audio_clip" in _codes(s.engine)
    track, _ = s.engine.audio_track()
    assert np.abs(track.astype(np.int32)).max() <= round(32768 * 10 ** (-1 / 20))


@pytest.mark.parametrize("passes", MODES)
def test_strict_lint_wants_cues_on_markers(tmp_path, passes):
    s = _scene(tmp_path, passes, voice=None)
    s.cue(2.5, "tick")
    assert "cue_unanchored" not in _codes(s.engine)
    assert "cue_unanchored" in _codes(s.engine, strict=True)
    s.engine.apply({"op": "remove_sound", "id": "tick_at_2500ms"}, origin="script")
    s.cue("end", "tick")
    assert "cue_unanchored" not in _codes(s.engine, strict=True)


@pytest.mark.parametrize("passes", MODES)
def test_a_loud_cue_on_a_word_is_a_taste_issue(tmp_path, passes):
    s = _scene(tmp_path, passes)
    s.cue(("m0", 0.0), "pop")  # the word starts at the marker
    assert "cue_on_word" not in _codes(s.engine, aesthetic=True)  # quiet: fine
    s.cue(("m0", 0.0), "pop", gain=12.0)
    assert "cue_on_word" in _codes(s.engine, aesthetic=True)
    assert "cue_on_word" not in _codes(s.engine)
