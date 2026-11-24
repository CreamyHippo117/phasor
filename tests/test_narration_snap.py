"""Milestone 22: narration from a recording (``Scene.narrate(script, audio=...)``) and
``snap_to_narration``, which moves the markers placed by hand onto the sentences of it.

Recordings are synthetic (``tests.test_speech.recording``: formant-shaped word bursts whose
sentence starts are known), written as WAV, or as an mp4 through PyAV when it is installed."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Sound, Text
from phasor.core import extras
from phasor.domains import audio
from phasor.time.narration import PHRASE, block_of, repace
from phasor.voice import VoiceError
from tests.test_audio_inputs import encode
from tests.test_lint import MODES
from tests.test_speech import recording

RATE = 16000
SCRIPT = ("We start with a function that has two poles. "
          "Draw a small loop around [[loop_word]] the one at i. "
          "Only that pole contributes to the integral. "
          "The residue theorem says the rest is zero, and that is all.")  # fmt: skip
WORDS = [8, 10, 7, 12]
TOLERANCE = 0.05  # seconds: a snapped marker is this close to where its sentence's voice starts


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_AUDIO_CACHE", str(tmp_path / "_cache"))
    for f in (audio._decoded, audio._info, audio._speech):
        f.cache_clear()


def talk(folder: Path, seed: int = 5, snr: float = 30.0, name: str = "talk.wav", words=WORDS):
    x, starts = recording(seed, words, rate=RATE, snr=snr)
    if name.endswith(".wav"):
        with wave.open(str(folder / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    else:
        encode(folder / name, np.clip(x, -1, 1), RATE)
    return x, starts


def scene(folder: Path, passes=None, order: int = 0) -> Scene:
    """Laid out by hand before the recording existed: four scenes with motions on them."""
    s = Scene("snap", passes=passes, base_dir=folder)
    s.add(Dot(id="a", pos=(-1, 0)))
    s.add(Dot(id="b", pos=(1, 0)))
    s.add(Text(id="a.label", text="a", pos=(-1, 0.6)))
    for name, t in (("intro", 0.0), ("poles", 4.0), ("loop", 9.0), ("residue", 14.0)):
        s.marker(name, t)
    verbs = [lambda: s.at("intro").reveal("a"), lambda: s.at("poles").reveal("b", "a.label"),
             lambda: s.at("loop").settle("a.pos", (0, 1)),
             lambda: s.at("residue").settle("b.pos", (2, 2))]  # fmt: skip
    for v in verbs if order == 0 else verbs[::-1]:
        v()
    return s


# ------------------------------------------------------------------ narration from a file


def test_a_recording_with_its_script_puts_each_sentence_where_it_is_said(tmp_path):
    x, starts = talk(tmp_path)
    s = scene(tmp_path)
    s.narrate(SCRIPT, audio="talk.wav")
    nar = s.engine.document.narration
    assert nar.voice == "file:talk.wav" and nar.script == SCRIPT
    spoken = [b for b in nar.blocks if b.text]
    assert len(spoken) == 4 and all(b.audio == "talk.wav" and b.audio_span for b in spoken)
    for b, t in zip(spoken, starts, strict=True):
        assert abs(b.start - t) <= TOLERANCE  # (the 0.03 s pre-roll included)
        assert b.audio_span[0] == b.start  # the timeline is the recording's
    # the [[marker]] inside sentence 2 lands inside it, and the timeline owns it
    loop_word = s.engine.document.timeline.markers["loop_word"]
    assert spoken[1].start < loop_word < spoken[1].start + spoken[1].duration
    assert all(w.end <= b.duration + 1e-6 for b in spoken for w in b.words)
    # the mix is the recording itself, where the blocks are
    track, rate = s.engine.audio_track()
    assert rate == RATE
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    for b in spoken:
        i, j = round(b.start * rate), round((b.start + b.duration) * rate)
        assert np.array_equal(track[i:j], pcm[i:j])


def test_an_mp4_recording_narrates_and_mixes(tmp_path):
    pytest.importorskip("av")
    _, starts = talk(tmp_path, name="talk.mp4")
    s = scene(tmp_path)
    s.narrate(SCRIPT, audio="talk.mp4")
    spoken = [b for b in s.engine.document.narration.blocks if b.text]
    assert max(abs(b.start - t) for b, t in zip(spoken, starts, strict=True)) <= 0.06
    track, rate = s.engine.audio_track()
    assert rate == RATE and np.abs(track[round(spoken[0].start * rate) :][:4000]).max() > 1000
    assert not list((tmp_path).glob("*.wav"))  # decoded into the cache, not the scene folder


def test_a_recording_without_a_script_is_cut_into_phrases(tmp_path):
    talk(tmp_path)
    s = scene(tmp_path)
    s.narrate(audio="talk.wav")
    nar = s.engine.document.narration
    assert nar.script == "" and len(nar.blocks) >= 4
    assert all(b.text == PHRASE and b.audio_span for b in nar.blocks)
    lanes = s.engine.timeline()["narration"]
    assert len(lanes) == len(nar.blocks)  # the editor's timeline shows them
    r = s.snap_to_narration(dry_run=True)
    assert {m["kind"] for m in r["moves"]} <= {"phrase", "between"}


def test_what_cannot_be_narrated_is_said(tmp_path):
    s = scene(tmp_path)
    with pytest.raises(VoiceError, match="not there"):
        s.narrate(SCRIPT, audio="gone.wav")
    with pytest.raises((VoiceError, ValueError)):
        s.narrate(SCRIPT, audio="../outside.wav")
    with wave.open(str(tmp_path / "quiet.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000)
        w.writeframes(bytes(16000))
    with pytest.raises(VoiceError, match="no speech"):
        s.narrate(SCRIPT, audio="quiet.wav")
    (tmp_path / "bad.wav").write_bytes(b"RIFF....WAVEjunk")
    with pytest.raises(VoiceError, match="bad.wav"):
        s.narrate(SCRIPT, audio="bad.wav")
    assert s.engine.document.narration is None  # nothing changed


def test_repace_and_block_of_work_on_a_recording(tmp_path):
    talk(tmp_path)
    s = scene(tmp_path)
    s.narrate(SCRIPT, audio="talk.wav")
    nar = s.engine.document.narration
    wider = repace(nar, [1.0] * len(nar.blocks), 0.3, 0.5)
    for a, b in zip(nar.blocks, wider.blocks, strict=True):
        assert a.audio_span == b.audio_span and a.duration == b.duration
    assert block_of(wider, wider.blocks[2].start + 0.01) == 2
    s.pace()  # the scene's own pacing runs on it unchanged
    track, rate = s.engine.audio_track()
    moved = s.engine.document.narration.blocks
    k = next(i for i, b in enumerate(moved) if b.text)
    i = round(moved[k].start * rate)
    assert np.abs(track[i : i + 2000]).max() > 1000


# ------------------------------------------------------------------ snapping


def test_snap_previews_then_moves_markers_onto_sentence_starts(tmp_path):
    _, starts = talk(tmp_path)
    s = scene(tmp_path)
    s.narrate(SCRIPT, audio="talk.wav")
    before = dict(s.engine.document.timeline.markers)
    preview = s.snap_to_narration(dry_run=True)
    assert s.engine.document.timeline.markers == before  # a preview changes nothing
    moves = {m["marker"]: m for m in preview["moves"]}
    assert set(moves) == {"intro", "poles", "loop", "residue"}
    for name, t in zip(("intro", "poles", "loop", "residue"), starts, strict=True):
        m = moves[name]
        assert m["from"] == before[name] and abs(m["to"] - t) <= TOLERANCE
        assert m["kind"] == "sentence" and "sentence" in m["why"] and 0 < m["confidence"] <= 1
    assert any(k["kind"] == "owned" for k in preview["kept"])  # [[loop_word]] stays
    version = s.engine.version
    done = s.snap_to_narration()
    assert done["applied"] and s.engine.version == version + 1  # one patch
    after = s.engine.document.timeline.markers
    assert all(after[k] == v["to"] for k, v in moves.items())
    assert s.snap_to_narration(dry_run=True)["moves"] == []  # idempotent
    s.engine.undo()
    assert s.engine.document.timeline.markers == before  # one undo step


def test_motions_follow_the_markers_whatever_order_the_script_wrote_them(tmp_path):
    talk(tmp_path)
    docs, states = [], []
    for order in (0, 1):
        s = scene(tmp_path, order=order)
        s.narrate(SCRIPT, audio="talk.wav")
        s.snap_to_narration()
        e = s.engine
        poles = e.document.timeline.markers["poles"]
        assert e.state_at(poles - 0.01)["b"].params.opacity < 0.05  # not yet
        assert e.state_at(poles + 1.0)["b"].params.opacity > 0.95  # from the new time
        docs.append(e.document.timeline.markers)
        states.append([e.state_at(t)["a"].params.pos for t in (5, 10, 12, 15)])
    assert docs[0] == docs[1] and states[0] == states[1]


def test_a_scene_is_never_made_shorter_than_its_motions_need(tmp_path):
    """``trace`` takes 1.4 s and the scene after ``one`` was 1.2 s; the voice says the next
    sentence under a second after the first. Snapping both would leave the trace 0.97 s: one
    of them stays, unless ``allow_short`` (or an anchor) says otherwise."""
    talk(tmp_path, words=[1, 3, 10])
    s = Scene("short", passes=None, base_dir=tmp_path)
    s.add(Line(id="a", points=[(-1, 0), (1, 0)]))
    s.add(Dot(id="b", pos=(1, 0)))
    s.marker("one", 0.0)
    s.marker("two", 1.2)
    s.at("one").trace("a")
    s.at("two").reveal("b")
    s.narrate("Look. Look at that. And then the rest of it follows from here on.",
              audio="talk.wav")  # fmt: skip
    starts = [b.start for b in s.engine.document.narration.blocks if b.text]
    assert starts[1] - starts[0] < 1.2  # the voice leaves the trace too little time

    def plan(**kw):
        r = s.snap_to_narration(dry_run=True, **kw)
        return {m["marker"]: m["to"] for m in r["moves"] + r["kept"]}

    to = plan()
    assert to["two"] - to["one"] >= 1.2 - 1e-6
    assert plan(allow_short=True) == {"one": starts[0], "two": starts[1]}
    assert plan(anchors={"one": 0, "two": 1}) == {"one": starts[0], "two": starts[1]}


def test_bad_anchors_and_no_narration_are_clean_errors(tmp_path):
    talk(tmp_path)
    s = scene(tmp_path)
    with pytest.raises(ValueError, match="narrate first"):
        s.snap_to_narration()
    s.narrate(SCRIPT, audio="talk.wav")
    for anchors, msg in (
        ({"loop_word": 1}, "belongs to the narration"),
        ({"nope": 1}, "no marker"),
        ({"poles": 99}, "not the index"),
        ({"poles": True}, "not the index"),
        ({"poles": "1"}, "not the index"),
    ):
        with pytest.raises(ValueError, match=msg):
            s.snap_to_narration(anchors=anchors)


def test_a_tweak_on_a_scripted_scene_is_refused_not_half_done(tmp_path):
    talk(tmp_path)
    s = scene(tmp_path)
    s.narrate(SCRIPT, audio="talk.wav")
    s.engine.apply({"op": "set_meta", "field": "source", "value": "script"}, origin="script")
    before = dict(s.engine.document.timeline.markers)
    r = s.engine.snap_to_narration()  # origin "tweak": a scripted scene keeps timing in code
    assert not r["applied"] and r["errors"][0]["code"] == "scripted_scene"
    assert s.engine.document.timeline.markers == before


def test_snap_works_on_a_synthesized_or_estimated_narration_too(tmp_path):
    s = scene(tmp_path)
    s.narrate(SCRIPT, voice="estimate")
    r = s.snap_to_narration(dry_run=True)
    assert r["moves"] and all(m["confidence"] <= 0.5 for m in r["moves"])  # a guess


@pytest.mark.parametrize("snr", [30.0, 15.0])
def test_snapped_markers_land_within_the_stated_tolerance_with_noise(tmp_path, snr):
    errors = []
    for seed in (5, 6, 7):
        _, starts = talk(tmp_path, seed=seed, snr=snr, name=f"t{seed}.wav")
        s = scene(tmp_path)
        s.narrate(SCRIPT, audio=f"t{seed}.wav")
        s.snap_to_narration()
        m = s.engine.document.timeline.markers
        errors += [m[k] - t for k, t in zip(("intro", "poles", "loop", "residue"), starts,
                                            strict=True)]  # fmt: skip
    assert max(abs(e) for e in errors) <= TOLERANCE, errors


# ------------------------------------------------------------------ lint


@pytest.mark.parametrize("passes", MODES)
def test_narration_drift_says_what_the_snap_would_move_and_then_nothing(tmp_path, passes):
    talk(tmp_path)
    s = scene(tmp_path, passes)
    s.narrate(SCRIPT, audio="talk.wav")
    drift = [i for i in s.engine.lint() if i.code == "narration_drift"]
    assert {i.path for i in drift} == {"markers.poles", "markers.loop", "markers.intro"}
    poles = next(i for i in drift if i.path == "markers.poles")
    assert poles.times == (4.0,) and poles.severity == "warning" and not poles.fix
    assert "snap_to_narration" in poles.fix_hint
    s.snap_to_narration()
    assert not [i for i in s.engine.lint() if i.code == "narration_drift"]


@pytest.mark.parametrize("passes", MODES)
def test_a_missing_bad_or_undecodable_recording_is_a_lint_issue(tmp_path, passes, monkeypatch):
    talk(tmp_path)
    s = scene(tmp_path, passes)
    s.narrate(SCRIPT, audio="talk.wav")
    s.snap_to_narration()
    good = (tmp_path / "talk.wav").read_bytes()
    (tmp_path / "talk.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEnope")
    assert "bad_file" in {i.code for i in s.engine.lint()}
    problems: list[str] = []
    s.engine.audio_track(problems)
    assert any("talk.wav" in p for p in problems)  # the render says it, never raises
    (tmp_path / "talk.wav").unlink()
    issue = next(i for i in s.engine.lint() if i.code == "missing_file")
    assert issue.path == "narration"
    (tmp_path / "talk.wav").write_bytes(good)
    assert not {"bad_file", "missing_file"} & {i.code for i in s.engine.lint()}


def test_a_recording_that_needs_pyav_without_it_is_missing_extra(tmp_path, monkeypatch):
    pytest.importorskip("av")
    talk(tmp_path, name="talk.mp4")
    s = scene(tmp_path)
    s.narrate(SCRIPT, audio="talk.mp4")
    monkeypatch.setattr(extras, "available", lambda m: m != "av")
    audio._info.cache_clear()
    s.engine.apply({"op": "set_marker", "name": "extra", "t": 1.0})  # a new version: lint again
    assert "missing_extra" in {i.code for i in s.engine.lint()}


# ------------------------------------------------------------------ sound clips


@pytest.mark.parametrize("passes", MODES)
def test_a_sound_clip_from_an_mp4_mixes_and_one_without_sound_is_bad_file(tmp_path, passes):
    pytest.importorskip("av")
    encode(tmp_path / "chime.m4a", 0.3 * np.sin(np.arange(22050) / 4).astype(np.float32), 22050)
    encode(tmp_path / "silent.mp4", None, 0, video=True, audio_stream=False)
    s = Scene("clips", passes=passes, base_dir=tmp_path)
    s.add(Dot(id="d"))
    s.add(Sound(id="chime", path="chime.m4a", start=0.5))
    s.marker("end", 2.0)
    sem = s.engine.inspect(0.0, id="chime")["objects"][0]["semantic"]
    assert sem["rate"] == 22050 and abs(sem["seconds"] - 1.0) < 0.1 and not sem["missing"]
    track, rate = s.engine.audio_track()
    assert rate == 48000 and track[: rate // 2].max() == 0 and np.abs(track[rate:]).max() > 0
    s.add(Sound(id="mute", path="silent.mp4"))
    issue = next(i for i in s.engine.lint() if i.code == "bad_file")
    assert issue.ids == ("mute",) and "no audio stream" in issue.message
    assert "sound track" in issue.fix_hint


@pytest.mark.parametrize("passes", MODES)
def test_without_pyav_a_non_wav_sound_is_missing_extra_and_wav_still_plays(tmp_path, passes,
                                                                          monkeypatch):  # fmt: skip
    from tests.test_sound import wav

    wav(tmp_path / "a.wav", np.full(8000, 9000))
    (tmp_path / "b.mp3").write_bytes(b"\xff\xfb" + bytes(500))
    monkeypatch.setattr(extras, "available", lambda m: m != "av")
    s = Scene("noav", passes=passes, base_dir=tmp_path)
    s.add(Dot(id="d"))
    s.add(Sound(id="a", path="a.wav"))
    s.add(Sound(id="b", path="b.mp3", start=0.2))
    s.marker("end", 2.0)
    codes = {(i.code, i.ids) for i in s.engine.lint()}
    assert ("missing_extra", ("b",)) in codes and not any(c == "bad_file" for c, _ in codes)
    problems: list[str] = []
    track, _ = s.engine.audio_track(problems)
    assert track.max() > 0 and any("'av' package" in p for p in problems)


# ------------------------------------------------------------------ the example


def test_the_example_snaps_and_lints_clean():
    import runpy

    root = Path(__file__).resolve().parents[1]
    e = runpy.run_path(str(root / "examples" / "narrated_from_audio.py"), run_name="x")["s"].engine
    m = e.document.timeline.markers
    for name, t in (("intro", 0.6), ("move", 3.6375), ("fade", 6.3298)):  # true starts
        assert abs(m[name] - t) <= TOLERANCE
    assert e.lint(strict=True) == []
    track, rate = e.audio_track()
    assert rate == 8000 and abs(len(track) / rate - e.duration) < 0.01


@pytest.mark.gl
@pytest.mark.slow
def test_example_golden_storyboard():
    import runpy

    from tests.conftest import gl_available
    from tests.test_beauty import _golden

    if not gl_available():
        pytest.skip("needs OpenGL")
    root = Path(__file__).resolve().parents[1]
    e = runpy.run_path(str(root / "examples" / "narrated_from_audio.py"), run_name="x")["s"].engine
    m = e.document.timeline.markers
    times = [m["intro"] + 0.8, m["move"] + 0.5, m["glide"] + 0.6, m["fade"] + 0.8]
    sheet = e.storyboard(times, size=(320, 180))["image"]
    _golden("narrated_from_audio", sheet, e.pass_runner().renderer_class())
