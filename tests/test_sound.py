"""Sound events (SPEC "Sound events"): notes and WAV files in the document, timed to markers,
mixed sample-exactly with the narration under a headroom rule."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Scene
from phasor.core import io
from phasor.core.issues import DocumentError
from phasor.core.migrate import migrate
from phasor.core.schema import SCHEMA_VERSION, Document, Meta, Narration, NarrationBlock, Note
from phasor.time import sound as snd
from phasor.time.narration import write_wav

FULL = 32768.0


def scene(tmp_path, **kw):
    return Scene("sounds", passes=None, base_dir=tmp_path, **kw)


# ---------------------------------------------------------------- the document


def test_sounds_round_trip_through_the_saved_document(tmp_path):
    s = scene(tmp_path)
    s.marker("hit", 1.0)
    s.note(69, "hit", amp=0.25, dur=0.4, voice="pluck")
    s.note(60.5, 0.2)
    s.sound("clap.wav", ("hit", 0.125), gain=0.5)
    doc = s.engine.document
    assert [x.id for x in doc.sounds] == ["note_a4_hit", "note_c4_200ms", "sound_clap_hit"]
    path = tmp_path / "s.phasor.json"
    io.save(path, doc)
    again = io.load_document(path)
    assert again == doc and again.schema_version == SCHEMA_VERSION == 9
    assert Document.model_validate(doc.model_dump(mode="json")) == doc


def test_a_version_8_document_migrates_to_no_sounds():
    raw = migrate({"schema_version": 8, "meta": {"title": "old"}, "objects": []})
    assert raw["schema_version"] == SCHEMA_VERSION and raw["sounds"] == []
    assert io.parse_document({"schema_version": 8, "meta": {"title": "old"}}).sounds == ()


def test_sound_files_stay_inside_the_folder_and_are_wav():
    for bad in ("/etc/x.wav", "../x.wav", "x.mp3"):
        with pytest.raises(ValueError):
            Document(meta=Meta(title="t"), sounds=({"kind": "file", "id": "s", "at": 0,
                                                    "path": bad},))  # fmt: skip


def test_sound_ids_share_the_id_space_and_are_retired_when_removed(tmp_path):
    from phasor.engine import Engine

    e = Engine(Document(meta=Meta(title="doc")), passes=None, base_dir=tmp_path)  # not a script
    s = Scene.of(e)
    assert e.apply({"op": "add_object", "object": {"id": "dot", "type": "dot"}}).ok
    clash = e.apply({"op": "add_sound", "sound": {"kind": "note", "id": "dot", "at": 0,
                                                  "midi": 60}})  # fmt: skip
    assert not clash.ok and "duplicate id" in clash.errors[0].message
    assert e.apply({"op": "add_sound", "sound": {"kind": "note", "id": "ping", "at": 0.5,
                                                 "midi": 72}}).ok  # fmt: skip
    moved = {"kind": "note", "id": "ping", "at": 0.75, "midi": 74}
    assert e.apply({"op": "set_sound", "sound": moved}).ok
    assert e.document.sound("ping").at == 0.75 and e.document.sound("ping").midi == 74
    assert e.apply({"op": "remove_sound", "id": "ping"}).ok
    assert "ping" in e.document.retired_ids and e.document.sounds == ()
    back = e.apply({"op": "add_sound", "sound": moved})
    assert not back.ok and back.errors[0].code == "id_retired"
    e.undo()  # undo the removal: the sound is back, its id no longer retired
    assert e.document.sound("ping") is not None and "ping" not in e.document.retired_ids
    assert s.note(74, 0.75) == "note_d5_750ms"  # a fresh id, never a retired one


def test_a_sound_at_an_unknown_marker_is_refused(tmp_path):
    s = scene(tmp_path)
    with pytest.raises(DocumentError, match="no marker 'nowhere'"):
        s.note(60, "nowhere")
    assert s.engine.document.sounds == ()


def test_diff_and_generated_script_carry_sounds(tmp_path):
    import runpy

    from phasor.core.diff import diff_documents
    from phasor.core.pycode import document_to_python

    s = scene(tmp_path)
    before = s.engine.document
    s.marker("go", 0.5)
    s.note(64, ("go", 0.25), amp=0.2, voice="bell")
    s.sound("drum.wav", 1.0, gain=2.0)
    kinds = [c["kind"] for c in diff_documents(before, s.engine.document)["changes"]]
    assert kinds.count("sound_added") == 2
    path = tmp_path / "again.py"
    path.write_text(document_to_python(s.engine.document), encoding="utf-8")
    again = runpy.run_path(str(path), run_name="x")["s"].engine.document
    assert again.sounds == s.engine.document.sounds


# ---------------------------------------------------------------- timing


def test_notes_follow_their_marker_when_the_narration_is_retimed(tmp_path):
    s = scene(tmp_path)
    s.narrate("A first sentence. Then the bell [[ring]] sounds here.", voice="estimate")
    sid = s.note(69, ("ring", 0.1))
    e = s.engine
    t = next(p.t for p in e._snap.compiled.sounds if p.id == sid)
    assert t == pytest.approx(e.document.timeline.markers["ring"] + 0.1)
    s.narrate("A much, much longer first sentence than before, to push everything later. "
              "Then the bell [[ring]] sounds here.", voice="estimate")  # fmt: skip
    later = next(p.t for p in e._snap.compiled.sounds if p.id == sid)
    assert later == pytest.approx(e.document.timeline.markers["ring"] + 0.1) and later > t + 1


def test_a_note_holds_the_scene_open_until_it_ends(tmp_path):
    s = scene(tmp_path)
    s.marker("end", 2.0)
    s.note(60, ("end", 0.5), dur=1.25)
    assert s.engine.duration == pytest.approx(3.75)


# ---------------------------------------------------------------- the mix


def test_a_note_starts_on_its_sample_with_its_peak(tmp_path):
    s = scene(tmp_path)
    s.note(69, 0.5, amp=0.25, dur=0.2)  # no narration: the mix runs at 48 kHz
    samples, rate = s.engine.audio_track()
    assert rate == snd.RATE and len(samples) == round(0.7 * rate)
    i0 = round(0.5 * rate)
    nz = np.flatnonzero(samples)
    assert nz[0] == i0 + 1 and nz[-1] < i0 + round(0.2 * rate)  # attack starts at 0, ends at 0
    assert np.abs(samples).max() == round(0.25 * FULL)
    seg = samples[i0 : i0 + round(0.2 * rate)].astype(float)
    hz = np.fft.rfftfreq(len(seg), 1 / rate)[np.argmax(np.abs(np.fft.rfft(seg)))]
    assert hz == pytest.approx(440, abs=5)  # A4


@pytest.mark.parametrize("voice", ["sine", "soft", "pluck", "bell"])
def test_every_voice_is_deterministic_and_peaks_at_amp(voice):
    note = Note(id="n", at=0.0, midi=57, amp=0.5, dur=0.3, voice=voice)
    a, b = snd.synth(note, 48000), snd.synth(note, 48000)
    assert np.array_equal(a, b) and a[0] == 0 and a[-1] == 0
    assert np.abs(a).max() == pytest.approx(0.5, rel=1e-12)


def test_a_sound_file_is_placed_sample_exactly_with_its_gain(tmp_path):
    clip = np.zeros(200, np.int16)
    clip[0], clip[100] = 16000, -8000
    write_wav(tmp_path / "click.wav", clip, snd.RATE)
    s = scene(tmp_path)
    s.marker("hit", 1.0)
    s.marker("end", 2.0)
    s.sound("click.wav", ("hit", 0.01), gain=0.5)
    samples, _ = s.engine.audio_track()
    i0 = round(1.01 * snd.RATE)
    assert samples[i0] == 8000 and samples[i0 + 100] == -4000
    assert np.count_nonzero(samples) == 2


def test_sounds_mix_at_the_narrations_rate_and_leave_the_voice_untouched(tmp_path):
    rate = 16000
    voice = (0.5 * FULL * np.sin(np.arange(rate) * 2 * np.pi * 200 / rate)).astype(np.int16)
    write_wav(tmp_path / "v.wav", voice, rate)
    clip = np.zeros(10, np.int16)
    clip[0] = 1000
    write_wav(tmp_path / "tick.wav", clip, 48000)  # another rate: resampled to the voice's
    s = scene(tmp_path)
    s._apply({"op": "set_narration", "narration": Narration(script="x", voice="test", blocks=(
        NarrationBlock(text="x", start=0.5, duration=1.0, audio="v.wav", markers={"go": 0.0}),
    ))})  # fmt: skip
    s.note(81, ("go", 0.25), amp=0.2, dur=0.1)
    s.sound("tick.wav", 2.0)
    samples, r = s.engine.audio_track()
    assert r == rate and len(samples) == 2 * rate  # the file's start holds the scene open
    i0 = round(0.75 * rate)
    from phasor.time.narration import mix_track

    alone, _ = mix_track(s.engine.document.narration, tmp_path, s.engine.duration)
    assert np.array_equal(samples[: i0 + 1], alone[: i0 + 1])  # before the note: the voice alone
    assert np.array_equal(samples[i0 + round(0.1 * rate) :], alone[i0 + round(0.1 * rate) :])
    assert not np.array_equal(samples[i0 : i0 + 100], alone[i0 : i0 + 100])


def test_the_limiter_never_clips_and_only_turns_the_sounds_down(tmp_path):
    rate = 48000
    voice = np.zeros(rate, np.int16)
    voice[rate // 2 : rate // 2 + 4800] = 30000  # a loud voice, above the ceiling on its own
    write_wav(tmp_path / "v.wav", voice, rate)
    s = scene(tmp_path)
    s._apply({"op": "set_narration", "narration": Narration(script="x", voice="test", blocks=(
        NarrationBlock(text="x", start=0.0, duration=1.0, audio="v.wav", markers={"go": 0.0}),
    ))})  # fmt: skip
    for k in range(4):  # four loud notes on top of each other, then over the voice
        s.note(60 + 4 * k, 0.1, amp=0.9, dur=0.2, id=f"chord_{k}")
    s.note(72, 0.45, amp=0.9, dur=0.2, id="over_voice")
    samples, _ = s.engine.audio_track()
    x = samples.astype(float) / FULL
    ceiling = np.maximum(snd.CEILING, np.abs(voice / FULL))
    assert np.all(np.abs(x) <= ceiling + 1 / FULL)
    assert np.abs(x[: rate // 2]).max() == pytest.approx(snd.CEILING, abs=2 / FULL)
    assert not samples[: round(0.1 * rate) + 1].any()  # silence before stays silence
    # The gain moves smoothly: no step larger than the signal itself makes
    bus_only = x[round(0.1 * rate) : round(0.3 * rate)]
    assert np.abs(np.diff(bus_only)).max() < 0.5


def test_duck_gain_is_one_below_the_ceiling_and_safe_above():
    rng = np.random.default_rng(0)
    voice = rng.uniform(-0.9, 0.9, 5000)
    bus = rng.uniform(-1.5, 1.5, 5000)
    g = snd.duck_gain(voice, bus, 48000)
    assert np.all((g >= 0) & (g <= 1))
    assert np.all(np.abs(voice + g * bus) <= np.maximum(snd.CEILING, np.abs(voice)) + 1e-12)
    quiet = snd.duck_gain(np.zeros(100), np.full(100, 0.5), 48000)
    assert np.array_equal(quiet, np.ones(100))


def test_the_mix_is_deterministic(tmp_path):
    def build():
        s = scene(tmp_path)
        s.note(57, 0.1, voice="pluck", amp=0.8)
        s.note(64, 0.15, voice="bell", amp=0.8)
        return s.engine.audio_track()[0]

    assert build().tobytes() == build().tobytes()


def test_no_sounds_and_no_narration_is_silence(tmp_path):
    assert scene(tmp_path).engine.audio_track() is None


# ---------------------------------------------------------------- lint


def test_missing_unreadable_and_cut_files_are_lint_warnings(tmp_path):
    s = scene(tmp_path)
    s.marker("end", 1.0)
    s.sound("nope.wav", 0.5)
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    s.sound("bad.wav", 0.25)
    write_wav(tmp_path / "long.wav", np.full(48000, 100, np.int16), 48000)
    s.sound("long.wav", 0.5)
    by_code = {i.code: i for i in s.engine.lint() if i.ids and i.ids[0].startswith("sound_")}
    assert by_code["missing_file"].ids == ("sound_nope_500ms",)
    assert by_code["missing_file"].times == (0.5,)
    assert by_code["bad_sound_file"].ids == ("sound_bad_250ms",)
    assert by_code["sound_cut"].ids == ("sound_long_500ms",) and "1.500" in by_code[
        "sound_cut"].fix_hint  # fmt: skip
    samples, _ = s.engine.audio_track()  # the broken ones are silent; the good one plays
    assert samples[round(0.5 * 48000)] == 100 and len(samples) == 48000


# ---------------------------------------------------------------- the MCP tool


def test_the_sound_tool_adds_notes_and_files_and_undo_takes_them_back(tmp_path):
    from phasor.workspace.tools import ToolError, Workspace

    ws = Workspace(tmp_path)
    ws.scene_new("tool")
    ws.narrate("Listen [[ping]] now.", voice="estimate", pace=False)
    r = ws.sound("ping", midi=76, offset=0.1, voice="soft")
    assert r["id"] == "note_e5_ping" and r["t"] == pytest.approx(
        ws.engine.document.timeline.markers["ping"] + 0.1)  # fmt: skip
    write_wav(tmp_path / "pop.wav", np.full(10, 50, np.int16), 48000)
    assert ws.sound(0.2, path="pop.wav")["id"] == "sound_pop_200ms"
    with pytest.raises(ToolError, match="either midi"):
        ws.sound(0.2)
    with pytest.raises(ToolError):
        ws.sound(0.2, path="../outside.wav")
    assert ws.summary()["sounds"] == ["note_e5_ping", "sound_pop_200ms"]
    ws.undo()
    assert ws.summary()["sounds"] == ["note_e5_ping"]
    assert ws.scene_edit({"op": "remove_sound", "id": "note_e5_ping"})["ok"]
    assert ws.engine.document.sounds == ()


# ---------------------------------------------------------------- the rendered video


@pytest.mark.gl
def test_a_rendered_video_carries_the_notes(tmp_path):
    from tests.conftest import gl_available

    if not gl_available():
        pytest.skip("needs OpenGL")
    av = pytest.importorskip("av")
    from phasor import Dot
    from phasor.render.job import render_file

    s = Scene("beep", base_dir=tmp_path)
    s.add(Dot(id="dot", pos=(0, 0)))
    s.note(69, 0.25, amp=0.5, dur=0.25)
    out = tmp_path / "beep.mp4"
    r = render_file(s.engine, out, preset="draft", end=1.0)
    assert r["audio"] and r["sounds"] == 1 and not r["narration"]
    with av.open(str(out)) as c:
        assert [st.type for st in c.streams] == ["video", "audio"]
        pcm = np.concatenate([f.to_ndarray().ravel() for f in c.decode(audio=0)])
    rate = 48000
    loud = np.flatnonzero(np.abs(pcm) > 0.25)  # AAC is lossy: look for where it is loud
    assert loud.size and abs(loud[0] / rate - 0.25) < 0.05 and loud[-1] / rate < 0.6
