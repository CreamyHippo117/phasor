"""Milestone 22: sound from files phasor does not read itself (``domains.audio``: an mp4's
audio track, m4a, mp3, ogg, flac, decoded once with PyAV into the user's cache) and narration
blocks that play part of a recording (``NarrationBlock.audio_span``), mixed a chunk at a time.
Hostile files are a clean ``ValueError`` (lint ``bad_file``), never a traceback or a hang."""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from phasor.core import extras, io
from phasor.core.schema import Narration, NarrationBlock
from phasor.domains import audio
from phasor.time import narration as nar
from tests.test_sound import wav

RATE = 8000


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    """Decoded sound goes to a cache of the test's own (never the user's)."""
    monkeypatch.setenv("PHASOR_AUDIO_CACHE", str(tmp_path / "_cache"))
    audio._decoded.cache_clear()
    audio._info.cache_clear()
    audio._speech.cache_clear()


def encode(path: Path, x, rate: int, codec: str = "aac", *, video: bool = False,
           audio_stream: bool = True, channels: int = 1) -> Path:  # fmt: skip
    """A sound (and/or video) file written with PyAV."""
    av = pytest.importorskip("av")
    layout = "mono" if channels == 1 else "stereo"
    with av.open(str(path), "w") as c:
        v = c.add_stream("mpeg4", rate=10) if video else None
        if v is not None:
            v.width = v.height = 32
            v.pix_fmt = "yuv420p"
        if audio_stream:
            a = c.add_stream(codec, rate=rate, layout=layout)
            fmt = a.codec_context.format.name
            n = a.codec_context.frame_size or 1024
            data = np.asarray(x, np.float32)
            data = data if data.ndim == 2 else data[:, None]
            for i in range(0, len(data), n):
                chunk = data[i : i + n].T
                if fmt.startswith("s16"):
                    chunk = (chunk * 32767).astype(np.int16)
                elif fmt.startswith("s32"):
                    chunk = (chunk * 2.1e9).astype(np.int32)
                if not fmt.endswith("p"):
                    chunk = chunk.T.reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format=fmt,
                                                   layout=layout)  # fmt: skip
                frame.sample_rate = rate
                frame.pts = i
                for p in a.encode(frame):
                    c.mux(p)
            for p in a.encode(None):
                c.mux(p)
        if v is not None:
            for _ in range(5):
                f = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
                for p in v.encode(f):
                    c.mux(p)
            for p in v.encode(None):
                c.mux(p)
    return path


def tone(seconds: float, rate: int, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate)).astype(
        np.float32
    )


# ------------------------------------------------------------------ decoding


@pytest.mark.parametrize(("name", "codec", "rate"), [
    ("talk.mp4", "aac", 44100), ("talk.m4a", "aac", 22050), ("talk.mp3", "mp3", 44100),
    ("talk.flac", "flac", 96000), ("talk.ogg", "opus", 48000),
])  # fmt: skip
def test_other_formats_decode_once_to_a_wav_in_the_cache(tmp_path, name, codec, rate):
    src = encode(tmp_path / name, tone(1.0, rate), rate, codec)
    out = audio.readable_wav(src)
    assert out.parent == tmp_path / "_cache" and out.suffix == ".wav"
    info = nar.wav_info(out)
    assert info.channels == 1 and info.width == 2 and info.rate == min(rate, 48000)
    assert abs(info.seconds - 1.0) < 0.1
    x, _ = nar.read_sound(out)
    mid = x[len(x) // 4 : 3 * len(x) // 4]
    assert abs(np.sqrt(np.mean(mid**2)) - 0.3 / np.sqrt(2)) < 0.03  # the tone came through
    stamp = out.stat().st_mtime_ns
    audio._decoded.cache_clear()
    assert audio.readable_wav(src) == out and out.stat().st_mtime_ns == stamp  # not again
    assert not list(out.parent.glob("*.tmp"))


def test_a_wav_is_read_as_it_is_without_a_decoder(tmp_path, monkeypatch):
    f = wav(tmp_path / "a.wav", np.zeros(800, np.int16))
    monkeypatch.setitem(sys.modules, "av", None)  # PyAV cannot be imported
    monkeypatch.setattr(extras, "available", lambda m: m != "av")
    assert audio.readable_wav(f) == f
    with pytest.raises(ValueError, match="needs the 'av' package"):
        audio.readable_wav(tmp_path / "a.mp4")


def test_stereo_is_mixed_down_and_the_cache_follows_the_file(tmp_path):
    x = tone(1.0, 44100)
    src = encode(tmp_path / "st.m4a", np.stack([x, x], 1), 44100, channels=2)
    first = audio.readable_wav(src)
    y, _ = nar.read_sound(first)
    assert np.abs(y[1000:-1000]).max() > 0.2
    encode(src, tone(2.0, 44100, 220.0), 44100)  # a new recording under the same name
    os.utime(src, ns=(time.time_ns(), time.time_ns() + 10**9))
    second = audio.readable_wav(src)
    assert second != first and abs(nar.wav_info(second).seconds - 2.0) < 0.1


def test_two_files_with_the_same_name_in_two_folders_do_not_share_the_cache(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = encode(tmp_path / "a" / "s.m4a", tone(1.0, 22050), 22050)
    two = encode(tmp_path / "b" / "s.m4a", tone(1.5, 22050), 22050)
    os.utime(two, ns=(one.stat().st_atime_ns, one.stat().st_mtime_ns))
    assert audio.readable_wav(one) != audio.readable_wav(two)


def test_an_mp4_with_no_audio_is_a_reason_not_a_crash(tmp_path):
    src = encode(tmp_path / "v.mp4", None, 0, video=True, audio_stream=False)
    with pytest.raises(ValueError, match="no audio stream"):
        audio.readable_wav(src)
    both = encode(tmp_path / "va.mp4", tone(1.0, 44100), 44100, video=True)
    assert abs(nar.wav_info(audio.readable_wav(both)).seconds - 1.0) < 0.1


def test_hostile_files_are_clean_value_errors(tmp_path):
    pytest.importorskip("av")
    good = encode(tmp_path / "good.mp4", tone(1.0, 44100), 44100).read_bytes()
    cases = {
        "empty.mp4": b"",
        "noise.mp4": np.random.default_rng(0).integers(0, 256, 50000, np.uint8).tobytes(),
        "cut.mp4": good[: len(good) // 3],
        "head.mp4": good[:64],
        "text.mp3": b"not a sound at all\n" * 100,
        "zeros.flac": bytes(4096),
    }
    for name, data in cases.items():
        (tmp_path / name).write_bytes(data)
        t0 = time.perf_counter()
        try:
            out = audio.readable_wav(tmp_path / name)
        except ValueError as e:
            assert str(e), name
        else:  # a cut file may hold some sound: then a valid WAV of it
            assert nar.wav_info(out).frames > 0, name
        assert time.perf_counter() - t0 < 20, name
    with pytest.raises(ValueError, match="not a sound file"):
        audio.readable_wav(tmp_path / "x.txt")


@pytest.mark.skipif(sys.platform == "win32", reason="a sparse file needs a POSIX filesystem")
def test_a_2gb_sparse_file_costs_little(tmp_path):
    pytest.importorskip("av")
    big = tmp_path / "big.mp4"
    with open(big, "wb") as f:
        f.truncate(2 << 30)
    tracemalloc.start()
    t0 = time.perf_counter()
    with pytest.raises(ValueError):
        audio.readable_wav(big)
    took = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert took < 30 and peak < 64 << 20, (took, peak)


def test_a_file_past_the_size_bound_is_refused(tmp_path, monkeypatch):
    src = encode(tmp_path / "a.m4a", tone(1.0, 22050), 22050)
    monkeypatch.setattr(audio, "MAX_DECODE_BYTES", 100)
    with pytest.raises(ValueError, match="larger than"):
        audio.readable_wav(src)


def test_decoding_stops_at_the_sound_bound(tmp_path, monkeypatch):
    src = encode(tmp_path / "a.m4a", tone(3.0, 22050), 22050)
    monkeypatch.setattr(audio, "MAX_DECODE_SECONDS", 1.0)
    assert abs(nar.wav_info(audio.readable_wav(src)).seconds - 1.0) < 0.05


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_fifo_never_blocks(tmp_path):
    pytest.importorskip("av")
    os.mkfifo(tmp_path / "pipe.mp4")
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="regular file"):
        audio.readable_wav(tmp_path / "pipe.mp4")
    assert time.perf_counter() - t0 < 5
    assert audio.info(tmp_path, "pipe.mp4") is None  # not a regular file: not there


def test_info_and_speech_say_what_a_file_holds(tmp_path):
    x = np.concatenate([np.zeros(11025, np.float32), tone(1.0, 22050), np.zeros(11025)])
    src = encode(tmp_path / "a.m4a", x, 22050)
    head = audio.info(tmp_path, "a.m4a")
    assert head.rate == 22050 and abs(head.seconds - 2.0) < 0.1
    assert audio.info(tmp_path, "gone.m4a") is None
    (tmp_path / "bad.mp3").write_bytes(b"x" * 100)
    assert isinstance(audio.info(tmp_path, "bad.mp3"), str)
    assert len(audio.speech(tmp_path, "a.m4a").segments) == 1
    with pytest.raises(ValueError):
        audio.speech(tmp_path, "gone.m4a")
    del src


# ------------------------------------------------------------------ narration spans


def _recording(folder: Path) -> np.ndarray:
    """2 s of three tone bursts at 8 kHz: 0.2-0.6, 0.9-1.3, 1.5-1.9 s."""
    x = np.zeros(2 * RATE, np.int16)
    for a, b in ((0.2, 0.6), (0.9, 1.3), (1.5, 1.9)):
        i = int(a * RATE)
        x[i : int(b * RATE)] = (8000 * np.sin(np.arange(int(b * RATE) - i) / 3)).astype(np.int16)
    wav(folder / "talk.wav", x)
    return x


def _span_narration(spans, starts, audio_name="talk.wav"):
    pairs = enumerate(zip(spans, starts, strict=True))
    blocks = [NarrationBlock(text=f"s{k}", start=t, duration=b - a, audio=audio_name,
                             audio_span=(a, b)) for k, ((a, b), t) in pairs]  # fmt: skip
    return Narration(script="", voice="file:talk.wav", blocks=tuple(blocks))


def test_span_blocks_play_their_part_of_the_recording(tmp_path):
    x = _recording(tmp_path)
    spans = [(0.2, 0.6), (0.9, 1.3), (1.5, 1.9)]
    n = _span_narration(spans, [1.0, 2.0, 2.4])  # moved apart, as repace does
    track, rate = nar.mix_track(n, tmp_path, 3.0)
    assert rate == RATE and len(track) == 3 * RATE
    for (a, b), t in zip(spans, [1.0, 2.0, 2.4], strict=True):
        got = track[int(t * RATE) : int(t * RATE) + int((b - a) * RATE)]
        assert np.array_equal(got, x[int(a * RATE) : int(a * RATE) + len(got)])
    assert track[:RATE].max() == 0  # nothing before the first block
    whole = track
    part, _ = nar.mix_track(n, tmp_path, 3.0, start=1.1, end=2.2)
    assert np.array_equal(part, whole[int(1.1 * RATE) : int(2.2 * RATE)])


def test_a_missing_or_bad_recording_is_said_not_raised(tmp_path):
    n = _span_narration([(0.0, 1.0)], [0.0], "gone.wav")
    problems: list[str] = []
    assert nar.mix_track(n, tmp_path, 2.0, problems=problems) is None
    assert "narration audio 'gone.wav' left out" in problems[0]
    (tmp_path / "bad.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    problems.clear()
    assert nar.mix_track(_span_narration([(0, 1)], [0], "bad.wav"), tmp_path, 2.0,
                         problems=problems) is None  # fmt: skip
    assert "left out" in problems[0]


def test_many_overlapping_blocks_on_one_long_file_are_bounded(tmp_path, monkeypatch):
    wav(tmp_path / "talk.wav", np.full(10 * RATE, 1000, np.int16))
    spans = [(0.0, 10.0)] * 400
    n = _span_narration(spans, [0.0] * 400)
    monkeypatch.setattr(nar, "MAX_MIX_SAMPLES", 50 * RATE)
    problems: list[str] = []
    tracemalloc.start()
    track, _ = nar.mix_track(n, tmp_path, 10.0, problems=problems)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert int(track.max()) == 5 * 1000  # five blocks' worth fit the budget
    assert any("clip samples in all" in p for p in problems) and peak < 64 << 20


def test_a_span_past_the_clip_bound_is_still_read(tmp_path, monkeypatch):
    """Narration spans are bounded by the track (3 hours), not by a clip's
    ``MAX_CLIP_FRAMES``: a sentence 25 minutes into a recording plays."""
    x = _recording(tmp_path)
    monkeypatch.setattr(nar, "MAX_CLIP_FRAMES", RATE)  # as if 1 s were a long recording
    track, _ = nar.mix_track(_span_narration([(1.5, 1.9)], [0.0]), tmp_path, 1.0)
    assert np.array_equal(track[: int(0.4 * RATE)], x[int(1.5 * RATE) : int(1.9 * RATE)])


def test_an_mp4_recording_mixes_through_the_resolver(tmp_path):
    src = tone(2.0, 44100)
    encode(tmp_path / "talk.mp4", src, 44100)
    n = _span_narration([(0.5, 1.0)], [0.0], "talk.mp4")
    problems: list[str] = []
    assert nar.mix_track(n, tmp_path, 1.0, problems=problems) is None  # no resolver: said
    assert "not a RIFF" in problems[0] or "not a PCM WAV" in problems[0]
    track, rate = nar.mix_track(n, tmp_path, 1.0, resolve=audio.readable_wav)
    assert rate == 44100 and np.abs(track[: int(0.5 * rate)]).max() > 5000


def test_schema_v9_narration_loads_and_mixes_as_before(tmp_path):
    x = (3000 * np.sin(np.arange(RATE) / 5)).astype(np.int16)
    (tmp_path / "s.narration").mkdir()
    wav(tmp_path / "s.narration" / "a.wav", x)
    block = {"text": "Hello there.", "start": 0.5, "duration": 1.0, "audio": "s.narration/a.wav",
             "words": [], "markers": {"hi": 0.0}}  # fmt: skip
    raw = {"schema_version": 9, "meta": {"title": "old"},
           "narration": {"script": "Hello [[hi]] there.", "voice": "espeak", "blocks": [block]},
           "timeline": {"camera": "camera", "markers": {"hi": 0.5}, "tracks": {}}}  # fmt: skip
    doc = io.parse_document(json.loads(json.dumps(raw)))
    assert doc.schema_version == 10 and doc.narration.blocks[0].audio_span is None
    track, _ = nar.mix_track(doc.narration, tmp_path, 2.0)
    assert np.array_equal(track[RATE // 2 : RATE // 2 + RATE], x)
    with pytest.raises(ValueError):
        NarrationBlock(text="x", start=0, duration=1, audio_span=(0.0, 1.0))  # no audio
    for bad in ((1.0, 0.5), (-1.0, 1.0), (0.0, float("nan")), (0.0, 1e9)):
        with pytest.raises(ValueError):
            NarrationBlock(text="x", start=0, duration=1, audio="a.wav", audio_span=bad)
