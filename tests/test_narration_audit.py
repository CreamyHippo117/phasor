"""Milestone 22, independent audit (2026-11-24): a regression test per finding.

Principle the audit set: a confident wrong answer must never look like a right one. Each
test failed on the audited head (``bf178df``) and passes now."""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Scene
from phasor.core.issues import DocumentError
from phasor.core.schema import Narration, NarrationBlock
from phasor.domains import audio
from phasor.time import narration as nar
from phasor.time import speech as sp
from tests.test_speech import _word

RATE = 16000


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_AUDIO_CACHE", str(tmp_path / "_cache"))
    for f in (audio._decoded, audio._info, audio._speech, audio._aligned):
        f.cache_clear()


def speech(seed, sentences, *, rate=RATE, snr=30.0, word=(0.18, 0.45), gap=(0.04, 0.3),
           pause=(0.35, 0.8), lead=0.6, midpause=None):  # fmt: skip
    """Synthetic speech with known sentence starts; ``midpause``: {sentence: (after word,
    seconds)}, a long pause inside a sentence."""
    rng = np.random.default_rng(seed)
    parts, t, starts = [np.zeros(int(lead * rate))], lead, []
    for si, words in enumerate(sentences):
        starts.append(t)
        for i in range(words):
            parts.append(0.5 * _word(rng, rate, rng.uniform(*word), rng.uniform(100, 160)))
            t += len(parts[-1]) / rate
            if i < words - 1:
                g = rng.uniform(*gap)
                if midpause and si in midpause and midpause[si][0] == i:
                    g = midpause[si][1]
                parts.append(np.zeros(int(g * rate)))
                t += len(parts[-1]) / rate
        parts.append(np.zeros(int(rng.uniform(*pause) * rate)))
        t += len(parts[-1]) / rate
    x = np.concatenate(parts)
    level = np.sqrt(np.mean(x[x != 0] ** 2))
    x = x + rng.normal(0, level * 10 ** (-snr / 20), x.size)
    return x.astype(np.float32), starts


def write(path: Path, x, rate=RATE):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def script_for(words):
    return " ".join(" ".join(["word"] * n).capitalize() + "." for n in words)


# ------------------------------------------------------------------ B1: the scene's clock


def _clock_scene(folder, words, starts, clock, motions_to=None, every=2):
    s = Scene("r", base_dir=folder, passes=None)
    idx = list(range(0, len(words), every))
    for i in idx:
        s.add(Dot(id=f"d{i}"))
        s.marker(f"m{i}", round(clock(starts[i]), 3))
    if motions_to is not None:
        names = [f"m{i}" for i in idx]
        for k, i in enumerate(idx):
            end = names[k + 1] if k + 1 < len(names) else round(clock(motions_to), 3)
            s.segment(f"d{i}.pos", f"m{i}", end, (1.0, 0.0))
    s.narrate(script_for(words), audio="t.wav")
    return s, {f"m{i}": starts[i] for i in idx}


@pytest.mark.parametrize("name", ["x0.8", "x0.9", "x1.1", "x1.25", "offset +2 s",
                                  "drift x1.1", "drift x0.9"])  # fmt: skip
def test_b1_markers_land_on_their_own_sentence_when_the_scene_runs_on_another_clock(
        tmp_path, name):  # fmt: skip
    """Was: x0.9 put 10 of 12 markers more than 1 s off (the wrong sentence), with
    unreliable False and confidence 0.6."""
    words = list(np.random.default_rng(5).integers(6, 12, 24))
    x, starts = speech(24, words, rate=8000)
    write(tmp_path / "t.wav", x, 8000)
    end = len(x) / 8000
    half = end / 2
    clock = {
        "x0.8": lambda t: 0.8 * t, "x0.9": lambda t: 0.9 * t, "x1.1": lambda t: 1.1 * t,
        "x1.25": lambda t: 1.25 * t, "offset +2 s": lambda t: t + 2.0,
        "drift x1.1": lambda t: t if t < half else half + 1.1 * (t - half),
        "drift x0.9": lambda t: t if t < half else half + 0.9 * (t - half),
    }[name]  # fmt: skip
    s, truth = _clock_scene(tmp_path, words, starts, clock, motions_to=end)
    r = s.snap_to_narration(dry_run=True)
    to = {m["marker"]: m["to"] for m in r["moves"] + r["kept"]}
    wrong = [k for k, t in truth.items() if abs(to[k] + 0.03 - t) > 0.25]
    assert not wrong, (wrong, r["alignment"])
    assert not r["unreliable"] and not r["alignment"]["ambiguous"]


def test_b1_when_the_timing_fits_two_ways_nothing_moves_and_it_says_so():
    """Two markers 1 s apart over evenly spaced sentences: any neighbouring pair fits."""
    bounds = [sp.Boundary(float(t), "sentence", "s") for t in np.arange(0.5, 30, 1.0)]
    plan = sp.plan_snap({"a": 10.3, "b": 11.3}, {}, bounds, visual_end=30, narration_end=30,
                        reach=0.1, need={})  # fmt: skip
    if plan.fit["ambiguous"]:
        assert all(not m.moves and m.confidence == 0.0 for m in plan)
    else:  # it chose the nearest pair: its own clock
        assert {m.marker: m.t1 for m in plan} == {"a": 10.5, "b": 11.5}


def test_b1_an_ambiguous_engine_snap_is_unreliable_and_moves_nothing(tmp_path, monkeypatch):
    words = [8, 8, 8, 8, 8, 8]
    x, _ = speech(3, words)
    write(tmp_path / "t.wav", x)
    s = Scene("r", base_dir=tmp_path, passes=None)
    s.add(Dot(id="d"))
    s.marker("m", 7.0)
    s.narrate(script_for(words), audio="t.wav")
    monkeypatch.setattr(sp, "_AMBIGUOUS", 10.0)  # every second reading counts as close
    r = s.snap_to_narration(dry_run=True)
    if r["alignment"]["ambiguous"]:
        assert r["unreliable"] and "ambiguous_timing" in r["flags"] and not r["moves"]


# ------------------------------------------------------------------ B2: no speech dropped


@pytest.mark.parametrize("words", [[100], [40, 40, 40], [91]])
def test_b2_every_detected_speech_sample_is_in_some_block(tmp_path, words):
    """Was: a 100-word sentence lost its first 7.5 s and last 2 s (48 segments at most per
    sentence, ends skipped); a script without full stops lost 10.6 s."""
    x, _ = speech(1, words)
    write(tmp_path / "t.wav", x)
    s = Scene("r", base_dir=tmp_path, passes=None)
    s.narrate(script_for(words), audio="t.wav")
    blocks = [b for b in s.engine.document.narration.blocks if b.text]
    heard = audio.speech(tmp_path, "t.wav")
    for a, b in heard.segments:
        assert any(blk.audio_span[0] <= a + 1e-6 and b <= blk.audio_span[1] + 1e-6
                   for blk in blocks), (a, b)  # fmt: skip
    track, _ = s.engine.audio_track()
    ref = (np.clip(x, -1, 1) * 32767).astype(np.int16).astype(float)
    kept = np.sum(track.astype(float) ** 2) / np.sum(ref[: len(track)] ** 2)
    assert kept > 0.99


# ------------------------------------------------------------------ B3: harder readings


CASES = {
    "two mid-sentence pauses of 1.2 s": dict(midpause={1: (4, 1.2), 3: (5, 1.2)}),
    "two mid-sentence pauses of 0.9 s": dict(midpause={1: (4, 0.9), 3: (5, 0.9)}),
    "pauses 1.5 and 1.2 s mid-sentence": dict(midpause={1: (4, 1.5), 3: (5, 1.2)}),
    "one 70-word sentence": dict(words=[70], gap=(0.16, 0.3)),
    "varied lengths": dict(words=[1, 20, 2, 15, 1, 3]),
    "fast speaker": dict(word=(0.10, 0.22), gap=(0.02, 0.08), pause=(0.25, 0.5)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_b3_sentences_are_found_where_long_pauses_mislead(case):
    """Was: two 1.2 s mid-sentence pauses put 16 of 30 sentence starts more than 250 ms off
    (worst 4.6 s) with confidence 0.6-0.85."""
    kw = dict(CASES[case])
    words = kw.pop("words", [8, 10, 7, 12, 6, 9])
    for seed in range(3):
        x, starts = speech(seed, words, **kw)
        placed = sp.align(sp.detect_samples(x, RATE), [sp.text_length(" ".join(["w"] * n))
                                                       for n in words])  # fmt: skip
        err = [abs(p.start - t) for p, t in zip(placed, starts, strict=True)]
        assert max(err) < 0.25, (seed, err)
        assert not placed.doubtful


@pytest.mark.parametrize("case", ["no pauses between sentences", "slow speaker"])
def test_b3_what_still_fails_is_flagged_doubtful(case):
    """Run-on reading (0.05-0.12 s between sentences, word gaps up to 0.3 s) and a slow
    speaker (word gaps as long as sentence pauses) cannot be split by pauses: where a start
    is wrong, the alignment says it doubts (and caps confidence at 0.3)."""
    kw = {"no pauses between sentences": dict(pause=(0.05, 0.12)),
          "slow speaker": dict(word=(0.3, 0.6), gap=(0.15, 0.6), pause=(0.4, 0.9))}[case]  # fmt: skip
    words = [8, 10, 7, 12, 6, 9]
    for seed in range(5):
        x, starts = speech(seed, words, **kw)
        placed = sp.align(sp.detect_samples(x, RATE), [float(n) for n in words])
        wrong = any(abs(p.start - t) > 0.25 for p, t in zip(placed, starts, strict=True))
        if wrong:
            assert placed.doubtful and all(p.confidence <= 0.3 for p in placed)


def test_b3_doubt_reaches_the_snap_result(tmp_path):
    words = [8, 10, 7, 12, 6, 9]
    x, _ = speech(0, words, pause=(0.05, 0.12))
    write(tmp_path / "t.wav", x)
    s = Scene("r", base_dir=tmp_path, passes=None)
    s.add(Dot(id="d"))
    s.marker("m", 3.0)
    s.narrate(script_for(words), audio="t.wav")
    r = s.snap_to_narration(dry_run=True)
    assert r["alignment"]["speech"]["doubtful"]
    assert r["unreliable"] and "alignment_doubtful" in r["flags"]


# ------------------------------------------------------------------ should-fix


def _snapx(tmp_path):
    words = [8, 10, 7, 12]
    x, _ = speech(5, words)
    write(tmp_path / "t.wav", x)
    s = Scene("r", base_dir=tmp_path, passes=None)
    s.add(Dot(id="a"))
    for k, t in {"intro": 0.0, "poles": 4.0, "loop": 9.0, "residue": 14.0}.items():
        s.marker(k, t)
    s.narrate("One two three four five six seven eight. A b c d [[w]] e f g h i j. "
              "K l m n o p q. R s t u v w x y z aa bb cc.", audio="t.wav")  # fmt: skip
    return s


def test_s1_anchors_that_reorder_or_cross_an_owned_marker_are_refused(tmp_path):
    s = _snapx(tmp_path)
    before = dict(s.engine.document.timeline.markers)
    with pytest.raises(ValueError, match="swap their order"):
        s.snap_to_narration(anchors={"intro": 3, "residue": 0})
    with pytest.raises(ValueError, match="would cross the marker 'w'"):
        s.snap_to_narration(anchors={"poles": 3})
    assert s.engine.document.timeline.markers == before


def test_s2_a_refused_scene_snap_raises(tmp_path):
    s = _snapx(tmp_path)
    s.engine.apply({"op": "set_meta", "field": "source", "value": "script"}, origin="script")
    s.engine.apply = lambda *a, **k: type(
        "R",
        (),
        {
            "ok": False,
            "errors": [
                __import__("phasor.core.issues", fromlist=["Issue"]).Issue("scripted_scene", "no")
            ],
        },
    )()
    with pytest.raises(DocumentError, match="scripted_scene"):
        s.snap_to_narration()


@pytest.mark.parametrize("name", ["dc 0.1", "dc 0.5 (half scale)", "hum 50 Hz 0.08",
                                  "hum 60 Hz 0.3"])  # fmt: skip
def test_s3_dc_and_hum_are_removed_before_detection(name):
    """Was: DC 0.1 found no speech; DC 0.02 and hum 0.03 were flagged music."""
    x, _ = speech(5, [8, 10, 7, 12], snr=40)
    t = np.arange(len(x)) / RATE
    y = {"dc 0.1": x + 0.1, "dc 0.5 (half scale)": 0.5 * x + 0.5,
         "hum 50 Hz 0.08": x + 0.08 * np.sin(2 * np.pi * 50 * t),
         "hum 60 Hz 0.3": x + 0.3 * np.sin(2 * np.pi * 60 * t)}[name]  # fmt: skip
    clean = sp.detect_samples(x, RATE)
    got = sp.detect_samples(y.astype(np.float32), RATE)
    assert len(got.segments) == len(clean.segments) and not got.unreliable
    assert "music" not in got.flags


def _riff(tag, bits, data, guid=None):
    ba = (bits + 7) // 8
    fmt = struct.pack("<HHIIHH", tag, 1, RATE, RATE * ba, ba, bits)
    if tag == 0xFFFE:
        fmt += struct.pack("<HHI", 22, bits, 4) + guid
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_s4_float_wavs_are_read(tmp_path):
    """Was: refused as "format 3 (compressed)"."""
    x = (0.5 * np.sin(np.arange(RATE) / 5)).astype("<f4")
    guid = b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq"
    (tmp_path / "f.wav").write_bytes(_riff(3, 32, x.tobytes()))
    (tmp_path / "e.wav").write_bytes(_riff(0xFFFE, 32, x.tobytes(), guid))
    (tmp_path / "d.wav").write_bytes(_riff(3, 64, x.astype("<f8").tobytes()))
    for name in ("f.wav", "e.wav", "d.wav"):
        y, rate = nar.read_sound(tmp_path / name)
        assert rate == RATE and np.allclose(y, x, atol=1e-6), name
    (tmp_path / "h.wav").write_bytes(_riff(3, 16, bytes(100)))  # 16-bit float: no
    with pytest.raises(ValueError, match="float"):
        nar.wav_info(tmp_path / "h.wav")
    (tmp_path / "c.wav").write_bytes(_riff(2, 16, bytes(100)))  # ADPCM: still refused
    with pytest.raises(ValueError, match="compressed"):
        nar.wav_info(tmp_path / "c.wav")
    with pytest.raises(ValueError, match="16-bit"):  # narration's int16 reader: unchanged
        nar.read_wav(tmp_path / "f.wav")


@pytest.mark.parametrize("rate", [48000, 192000])
def test_s5_a_block_far_past_the_scene_allocates_nothing(tmp_path, rate):
    """Was: a block at 10700 s in a 5 s scene made a 979 MiB track (MemoryError at
    192 kHz)."""
    nar.write_wav(tmp_path / "tiny.wav", np.ones(rate // 100, np.int16) * 1000, rate)
    b = NarrationBlock(text="Hi.", start=10700.0, duration=0.01, audio="tiny.wav",
                       audio_span=(0.0, 0.01))  # fmt: skip
    problems: list[str] = []
    out = nar.mix_track(Narration(script="Hi.", voice="file:tiny.wav", blocks=(b,)), tmp_path,
                        5.0, problems=problems)  # fmt: skip
    assert out is None and "past the scene's end" in problems[0]


def test_s5_the_track_is_bounded_in_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(nar, "MAX_TRACK_SAMPLES", 1000)
    nar.write_wav(tmp_path / "a.wav", np.ones(8000, np.int16) * 1000, 8000)
    b = NarrationBlock(text="Hi.", start=0.0, duration=1.0, audio="a.wav", audio_span=(0, 1))
    problems: list[str] = []
    track, _ = nar.mix_track(Narration(script="Hi.", voice="file:a.wav", blocks=(b,)),
                             tmp_path, 1.0, problems=problems)  # fmt: skip
    assert len(track) == 1000 and "cut" in problems[0]


def test_s6_a_decode_bomb_is_refused_and_the_cache_is_pruned(tmp_path, monkeypatch):
    from tests.test_audio_inputs import encode, tone

    pytest.importorskip("av")
    src = encode(tmp_path / "quiet.ogg", np.zeros(48000 * 20, np.float32), 48000, "opus")
    monkeypatch.setattr(audio, "_MIN_DECODED", 1000)  # (a 20 s file, not an hour)
    with pytest.raises(ValueError, match="would decode to more than"):
        audio.readable_wav(src)
    monkeypatch.setattr(audio, "_MIN_DECODED", 256 << 20)
    cache = tmp_path / "_cache"
    made = []
    for k in range(4):
        f = encode(tmp_path / f"s{k}.m4a", tone(1.0, 22050), 22050)
        made.append(audio.readable_wav(f))
        os.utime(made[-1], ns=(k * 10**9, k * 10**9))  # oldest first
    one = made[0].stat().st_size
    monkeypatch.setattr(audio, "MAX_CACHE_BYTES", int(2.5 * one))
    audio._prune(cache, keep=made[-1])
    left = sorted(p.name for p in cache.glob("*.wav"))
    assert made[-1].name in left and made[0].name not in left and len(left) <= 2


def test_s7_a_missing_recording_is_quality_0_and_unreliable(tmp_path):
    s = _snapx(tmp_path)
    (tmp_path / "t.wav").unlink()
    r = s.snap_to_narration(dry_run=True)
    assert r["quality"] == 0.0 and r["unreliable"] and "unreadable" in r["flags"]


def test_s8_the_alignment_is_computed_once(tmp_path, monkeypatch):
    s = _snapx(tmp_path)
    calls = []
    real = sp.align
    monkeypatch.setattr(sp, "align", lambda *a, **k: calls.append(1) or real(*a, **k))
    for _ in range(3):
        s.snap_to_narration(dry_run=True)
        s.engine.apply({"op": "set_marker", "name": f"x{len(calls)}_{_}", "t": 1.0})
        s.engine.lint()
    assert calls == []  # narrate computed it; lint and snap read the cache


def test_s9_a_file_held_by_another_process_is_not_an_error_once_it_exists(tmp_path,
                                                                          monkeypatch):  # fmt: skip
    from tests.test_audio_inputs import encode, tone

    pytest.importorskip("av")
    src = encode(tmp_path / "a.m4a", tone(1.0, 22050), 22050)
    real = os.replace

    def busy(a, b):  # another process decoded the same file first and holds it open
        real(a, b)
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", busy)
    out = audio.readable_wav(src)
    assert out.is_file() and nar.wav_info(out).frames > 0


def test_s9_a_passing_cache_error_is_not_cached(tmp_path, monkeypatch):
    from tests.test_audio_inputs import encode, tone

    pytest.importorskip("av")
    src = encode(tmp_path / "a.m4a", tone(1.0, 22050), 22050)
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the cache folder should be", encoding="utf-8")
    monkeypatch.setenv("PHASOR_AUDIO_CACHE", str(blocked / "cache"))
    with pytest.raises(ValueError, match="PHASOR_AUDIO_CACHE"):
        audio.readable_wav(src)
    monkeypatch.setenv("PHASOR_AUDIO_CACHE", str(tmp_path / "ok"))
    assert audio.readable_wav(src).is_file()


def test_nits_pyav_errors_are_worded_and_a_cut_decode_is_flagged(tmp_path, monkeypatch):
    from tests.test_audio_inputs import encode

    pytest.importorskip("av")
    (tmp_path / "bad.mp4").write_bytes(b"nope" * 1000)
    with pytest.raises(ValueError) as e:
        audio.readable_wav(tmp_path / "bad.mp4")
    assert str(e.value).startswith("cannot be decoded:") and "Errno" not in str(e.value)
    x, _ = speech(2, [6, 6], rate=22050)
    encode(tmp_path / "talk.m4a", x, 22050)
    monkeypatch.setattr(audio, "MAX_DECODE_SECONDS", 3.0)
    assert "cut" in audio.speech(tmp_path, "talk.m4a").flags
