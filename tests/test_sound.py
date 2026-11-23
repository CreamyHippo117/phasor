"""Milestone 20: ``media.Sound``, a sound clip (Manim's ``add_sound``) mixed into the video's
audio next to the narration. The document names a WAV file in the scene's folder (contract
3: read from there, read again when it changes); the mixing is bounded whatever the file
says; a missing or bad file is a lint issue and a render warning, never a crash."""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Scene, Sound
from phasor.core.issues import DocumentError
from phasor.time import narration as nar
from tests.test_lint import MODES

RATE = 8000


def wav(path: Path, samples, rate: int = RATE, channels: int = 1, width: int = 2) -> Path:
    """A PCM WAV written with the stdlib ``wave`` module."""
    a = np.asarray(samples)
    if width == 1:
        raw = (a.astype(np.int16) // 256 + 128).astype(np.uint8).tobytes()
    elif width == 2:
        raw = a.astype("<i2").tobytes()
    elif width == 3:
        raw = b"".join(struct.pack("<i", int(v) * 256)[:3] for v in a.ravel())
    else:
        raw = (a.astype(np.int64) * 65536).astype("<i4").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(raw)
    return path


def scene(folder: Path, length: float = 2.0, passes=None) -> Scene:
    s = Scene("sounds", passes=passes, base_dir=folder)
    s.add(Dot(id="tick"))
    s.marker("end", length)
    return s


# ---------------------------------------------------------------- the type


def test_a_sound_is_a_drawless_object_with_semantics_and_bounds(tmp_path):
    wav(tmp_path / "a.wav", np.full(4000, 1000))
    s = scene(tmp_path)
    s.add(Sound(id="beep", path="a.wav", start=0.25, gain=-3.0))
    seen = s.engine.inspect(0.0, id="beep")["objects"][0]["semantic"]
    assert seen == {"kind": "sound", "path": "a.wav", "start": 0.25, "gain": -3.0,
                    "missing": False, "drawn": 0.0, "file": "a.wav", "seconds": 0.5,
                    "rate": RATE}  # fmt: skip
    st = s.engine.state_at(0.0)
    assert st["beep"].impl.world_bounds(st).lo == (0.0, 0.0, 0.0)
    assert s.engine.lint() == []


@pytest.mark.parametrize("params", [
    {"path": "a.mp3"}, {"path": "../a.wav"}, {"path": "/etc/a.wav"}, {"path": ""},
    {"path": "a.wav", "start": -1}, {"path": "a.wav", "start": 2e6},
    {"path": "a.wav", "gain": 30}, {"path": "a.wav", "gain": float("nan")},
    {"path": "a.wav", "start": float("inf")},
])  # fmt: skip
def test_bad_parameters_are_refused_when_set(tmp_path, params):
    s = scene(tmp_path)
    with pytest.raises((DocumentError, ValueError)):
        s.add(Sound(id="beep", **params))
    r = s.engine.apply({"op": "add_object", "object": {"id": "b", "type": "sound",
                                                       "params": params}}, origin="script")  # fmt: skip
    assert not r.ok  # refused through Engine.apply too, never raised


@pytest.mark.parametrize("passes", MODES)
def test_a_missing_or_bad_file_is_a_lint_issue(tmp_path, passes):
    s = scene(tmp_path, passes=passes)
    s.add(Sound(id="gone", path="gone.wav"))
    (tmp_path / "bad.wav").write_bytes(b"RIFF....WAVEjunk")
    s.add(Sound(id="bad", path="bad.wav"))
    issues = {(i.code, i.ids[0]) for i in s.engine.lint()}
    assert ("missing_file", "gone") in issues and ("bad_file", "bad") in issues
    bad = next(i for i in s.engine.lint() if i.code == "bad_file")
    assert "PCM WAV" in bad.fix_hint and bad.times


def test_a_file_changed_on_disk_is_read_again(tmp_path):
    p = wav(tmp_path / "a.wav", np.full(800, 1000))
    s = scene(tmp_path)
    s.add(Sound(id="beep", path="a.wav"))
    assert s.engine.inspect(0.0, id="beep")["objects"][0]["semantic"]["seconds"] == 0.1
    wav(p, np.full(1600, 1000))
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # a new version, even if fast
    assert s.engine.inspect(0.0, id="beep")["objects"][0]["semantic"]["seconds"] == 0.2


# ---------------------------------------------------------------- the mix


def test_clips_mix_at_their_start_with_their_gain_and_stop_at_the_end(tmp_path):
    wav(tmp_path / "a.wav", np.full(8000, 10000))  # 1 s
    s = scene(tmp_path, length=2.0)
    s.add(Sound(id="one", path="a.wav", start=0.5))
    s.add(Sound(id="two", path="a.wav", start=1.0, gain=-6.0))
    track, rate = s.engine.audio_track()
    assert rate == nar.SOUND_RATE and len(track) == 2 * rate  # cut where the scene ends
    at = lambda t: int(track[round(t * rate)])  # noqa: E731
    assert at(0.25) == 0 and at(0.75) == 10000
    assert at(1.25) == pytest.approx(10000 + 10000 * 10 ** (-6 / 20), abs=2)  # both, mixed
    assert at(1.75) == pytest.approx(10000 * 10 ** (-6 / 20), abs=2)
    assert track.dtype == np.int16


def test_narration_and_sounds_share_the_narrations_rate(tmp_path):
    from phasor.core.schema import Narration, NarrationBlock

    wav(tmp_path / "voice.wav", np.full(1000, 7), rate=1000)
    wav(tmp_path / "fx.wav", np.full(16000, 100), rate=16000)  # resampled to 1 kHz
    n = Narration(script="x", voice="test", blocks=(
        NarrationBlock(text="a", start=0.0, duration=1.0, audio="voice.wav"),))  # fmt: skip
    track, rate = nar.mix_track(n, tmp_path, 3.0, [nar.Clip("fx.wav", 2.0)])
    assert rate == 1000 and len(track) == 3000
    assert track[500] == 7 and track[1500] == 0 and track[2500] == 100


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("channels", [1, 2])
def test_pcm_widths_and_channels_read_the_same_level(tmp_path, width, channels):
    frames = np.full((800, channels), 8192)
    p = wav(tmp_path / "a.wav", frames, channels=channels, width=width)
    x, rate = nar.read_sound(p)
    assert rate == RATE and len(x) == 800
    assert x.mean() == pytest.approx(0.25, abs=0.01)


def test_hostile_files_are_bounded_or_left_out(tmp_path):
    problems: list[str] = []
    (tmp_path / "junk.wav").write_bytes(os.urandom(4096))
    (tmp_path / "empty.wav").write_bytes(b"")
    wav(tmp_path / "fast.wav", np.zeros(10), rate=400000)
    os.mkdir(tmp_path / "dir.wav")
    clips = [nar.Clip(p, 0.0) for p in ("junk.wav", "empty.wav", "fast.wav", "dir.wav",
                                        "missing.wav", "../outside.wav")]  # fmt: skip
    assert nar.mix_track(None, tmp_path, 1.0, clips, problems) is None
    assert len(problems) == len(clips)
    # a header that claims far more frames than the file has: only what is there is read
    p = wav(tmp_path / "liar.wav", np.full(100, 5))
    data = bytearray(p.read_bytes())
    data[40:44] = struct.pack("<I", 0x7FFFFFF0)  # the data chunk's size
    p.write_bytes(bytes(data))
    x, _ = nar.read_sound(p)
    assert len(x) == 100
    # a start far past the scene, or a huge one, costs nothing
    assert nar.mix_track(None, tmp_path, 1.0, [nar.Clip("liar.wav", 1e6)]) is None


def test_the_track_never_covers_more_than_its_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(nar, "MAX_TRACK_SECONDS", 2.0)
    wav(tmp_path / "a.wav", np.full(800, 3))
    track, rate = nar.mix_track(None, tmp_path, 1e6, [nar.Clip("a.wav", 0.0)])
    assert len(track) == 2 * rate


def test_a_render_mixes_sounds_into_the_video(tmp_path):
    """The encoder (PyAV, the ``encode`` extra) muxes the mixed track into the video; the
    frames are made here, so no OpenGL is needed."""
    av = pytest.importorskip("av")
    from phasor.render.encode import EncodeError, mux_audio, open_writer

    wav(tmp_path / "a.wav", np.full(8000, 12000))
    s = scene(tmp_path, length=1.0)
    s.add(Sound(id="beep", path="a.wav", start=0.5))
    out = tmp_path / "v.mp4"
    try:
        w = open_writer(out, 64, 64, 10)
    except EncodeError as e:
        pytest.skip(f"no video encoder here: {e}")
    for _ in range(10):
        w.write(np.zeros((64, 64, 3), np.uint8))
    w.close()
    track, rate = s.engine.audio_track()
    mux_audio(out, track, rate)
    with av.open(str(out)) as f:
        (audio,) = f.streams.audio
        assert audio.codec_context.name == "aac" and audio.rate == rate
        pcm = np.concatenate([fr.to_ndarray().ravel() for fr in f.decode(audio)])
    loud = np.abs(pcm.astype(float))
    assert loud[: int(0.4 * rate)].max() < 0.02 * loud.max()  # silent before the clip starts
    assert loud[int(0.6 * rate) : int(0.9 * rate)].mean() > 0.2 * loud.max()


def test_a_lying_header_costs_only_what_the_file_holds(tmp_path):
    """Lesson 2: a header that claims 2^31 frames allocates for what is really there."""
    import tracemalloc

    p = wav(tmp_path / "liar.wav", np.full(100, 5))
    data = bytearray(p.read_bytes())
    data[40:44] = struct.pack("<I", 0x7FFFFFF0)
    p.write_bytes(bytes(data))
    tracemalloc.start()
    x, _ = nar.read_sound(p)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(x) == 100 and peak < 5_000_000
    tracemalloc.start()
    out = nar.mix_track(None, tmp_path, 10.0, [nar.Clip("liar.wav", 0.0)] * 10)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert out is not None and peak < 50_000_000  # ten clips, a 10 s track: a few MB


def test_only_the_rendered_window_is_mixed(tmp_path):
    """A render of part of a long scene allocates and reads only that part."""
    wav(tmp_path / "a.wav", np.arange(16000) % 1000)  # 2 s, a ramp to see where it was cut
    whole, rate = nar.mix_track(None, tmp_path, 3600.0, [nar.Clip("a.wav", 1.0)])
    part, _ = nar.mix_track(None, tmp_path, 3600.0, [nar.Clip("a.wav", 1.0)], start=1.5,
                            end=2.5)  # fmt: skip
    assert len(part) == rate and len(whole) == 3600 * rate
    assert np.abs(part.astype(int) - whole[int(1.5 * rate) : int(2.5 * rate)]).max() <= 1
    assert nar.mix_track(None, tmp_path, 3600.0, [nar.Clip("a.wav", 1.0)], start=10,
                         end=11) is None  # fmt: skip


# ---------------------------------------------------------------- hostile headers (M20 audit)


def riff(fmt_size: int, riff_len: int | None, data_len: int, data: bytes = b"",
         rate: int = 48000) -> bytes:  # fmt: skip
    """A hand-made 16-bit mono WAV whose sizes may lie."""
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", fmt_size) + fmt + b"data"
            + struct.pack("<I", data_len) + data)  # fmt: skip
    return b"RIFF" + struct.pack("<I", len(body) if riff_len is None else riff_len) + body


HOSTILE = {
    "fmt_lie.wav": riff(0xFFFFFFF0, None, 200, b"\0" * 100),  # a fmt chunk past the file
    "riff_short.wav": riff(16, 36, 192000, b"\x10\x00" * 96000),  # RIFF smaller than data
    "no_data.wav": riff(16, None, 0)[:36],
    "zero_rate.wav": riff(16, None, 4, b"\0" * 4, rate=0),
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_a_hostile_header_is_bad_file_in_lint_inspect_and_every_mix(tmp_path, name):
    """Blocker A: Python's ``wave`` raised a bare RuntimeError (a chunk seek) out of lint,
    inspect and the mix; a RIFF size smaller than the data was dropped silently."""
    (tmp_path / name).write_bytes(HOSTILE[name])
    s = scene(tmp_path, length=3.0)
    s.add(Sound(id="clip", path=name))
    assert [i.code for i in s.engine.lint() if "clip" in i.ids] == ["bad_file"]
    sem = s.engine.inspect(0.0, id="clip")["objects"][0]["semantic"]
    assert sem["unreadable"] and "seconds" not in sem
    for window in ((0.0, 3.0), (0.5, 1.5)):  # a whole render and ``--start 0.5``
        problems: list[str] = []
        assert s.engine.audio_track(problems, *window) is None
        assert len(problems) == 1 and name in problems[0]
    with pytest.raises(ValueError):
        nar.read_wav(tmp_path / name)  # narration: a ValueError that says why


def test_semantics_report_what_the_file_really_holds(tmp_path):
    """A data chunk that claims 10416 s of a 500-frame file: the semantics say 500 frames."""
    (tmp_path / "short.wav").write_bytes(riff(16, 0xFFFFFFFF, 0x7FFFFFF0, b"\x10\x00" * 500))
    s = scene(tmp_path)
    s.add(Sound(id="clip", path="short.wav"))
    sem = s.engine.inspect(0.0, id="clip")["objects"][0]["semantic"]
    assert sem["seconds"] == round(500 / 48000, 6)
    assert s.engine.lint() == []


def _stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return data, w.getframerate()


def test_narration_reads_wavs_exactly_as_the_stdlib_did(tmp_path):
    """The narration reader shares the header walk; its samples are the same as before."""
    rng = np.random.default_rng(1)
    mono = wav(tmp_path / "mono.wav", rng.integers(-32768, 32767, 999), rate=22050)
    wav(tmp_path / "stereo.wav", rng.integers(-32768, 32767, (777, 2)), rate=4000, channels=2)
    # an odd-sized LIST chunk (padded) before the data
    raw = mono.read_bytes()
    listed = raw[:36] + b"LIST" + struct.pack("<I", 5) + b"INFOx\0" + raw[36:]
    listed = listed[:4] + struct.pack("<I", len(listed) - 8) + listed[8:]
    (tmp_path / "list.wav").write_bytes(listed)
    # WAVE_FORMAT_EXTENSIBLE with the PCM sub-format
    ext = struct.pack("<HHIIHHHHI16s", 0xFFFE, 1, 22050, 44100, 2, 16, 22, 16, 4,
                      b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq")  # fmt: skip
    body = b"WAVEfmt " + struct.pack("<I", len(ext)) + ext + raw[36:]
    (tmp_path / "ext.wav").write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    for name in ("mono.wav", "stereo.wav", "list.wav", "ext.wav"):
        mine, rate = nar.read_wav(tmp_path / name)
        theirs, their_rate = _stdlib(tmp_path / name)
        assert rate == their_rate and mine.dtype == np.int16
        assert np.array_equal(mine, theirs), name


def test_a_render_mixes_the_sound_before_any_frame(tmp_path, monkeypatch):
    """A clip that cannot be mixed is said before the frames render, never after them."""
    from types import SimpleNamespace

    from phasor.render import job

    order: list[str] = []

    class Writer:
        path, encoder = str(tmp_path / "v.mp4"), "test"

        def write(self, frame):
            order.append("frame")

        def close(self):
            pass

    class FakeEngine:
        document = SimpleNamespace(meta=SimpleNamespace(width=8, height=8, fps=10))
        duration = 0.3  # three frames at 10 a second

        def frame(self, t, size, output):
            order.append("render")
            return np.zeros((8, 8, 3), np.uint8)

        def audio_track(self, problems, start, end):
            order.append("sound")
            problems.append("sound 'x.wav' left out: a test")
            return None

    monkeypatch.setattr(job, "run_doctor", lambda: SimpleNamespace(tier="A"))
    monkeypatch.setattr(job, "open_writer", lambda *a, **k: Writer())
    out = job.render_file(FakeEngine(), tmp_path / "v.mp4", preset="preview", fps=10)
    assert order[0] == "sound" and order.count("render") == 3
    assert "sound 'x.wav' left out: a test" in out["warnings"]


# ---------------------------------------------------------------- bounded mixing (M20 audit)


def test_a_window_is_the_same_samples_as_that_slice_of_the_whole_mix(tmp_path):
    """Note 2: a window that does not start on a source sample (an 8 kHz clip in a 48 kHz
    track, a window from 25/30 s) was off by a fraction of a sample (up to 30810 on a saw)."""
    wav(tmp_path / "saw.wav", (np.arange(8000 * 4) % 400) * 80 - 16000)  # 8 kHz
    wav(tmp_path / "tone.wav", (np.sin(np.arange(44100) * 0.03) * 9000).astype(int),
        rate=44100)  # fmt: skip
    clips = [nar.Clip("saw.wav", 0.1, -3.0), nar.Clip("tone.wav", 1.2345678)]
    from phasor.core.schema import Narration, NarrationBlock

    wav(tmp_path / "voice.wav", (np.arange(48000) % 300) * 50, rate=48000)
    voice = Narration(script="x", voice="test", blocks=(
        NarrationBlock(text="a", start=0.123457, duration=1.0, audio="voice.wav"),))  # fmt: skip
    for n in (None, voice):
        whole, rate = nar.mix_track(n, tmp_path, 4.0, clips)
        for lo, hi in ((25 / 30, 2.0), (0.5 + 1 / 7, 3.3), (1.23456, 1.5), (3.9, 4.0)):
            part, _ = nar.mix_track(n, tmp_path, 4.0, clips, start=lo, end=hi)
            g0 = round(lo * rate)
            assert np.array_equal(part, whole[g0 : g0 + len(part)]), (n is None, lo)


def test_mixing_memory_does_not_grow_with_the_number_of_clips(tmp_path):
    """S3: every resampled clip was held before summing (300 clips of a 10-minute WAV in a
    60 s scene: 3.4 GB). Now each clip is read a chunk at a time into the track."""
    import tracemalloc

    wav(tmp_path / "long.wav", np.full(8000 * 120, 100))  # 2 minutes at 8 kHz
    peaks = []
    for count in (1, 60):
        tracemalloc.start()
        track, rate = nar.mix_track(None, tmp_path, 20.0, [nar.Clip("long.wav", 0.0)] * count)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        assert track[rate] == min(100 * count, 32767)
        del track
    # the track itself (20 s at 48 kHz: 1.9 MB) and one chunk of each kind: never per clip
    assert peaks[1] < peaks[0] * 1.5 + 1_000_000, peaks
    assert peaks[1] < 40_000_000


def test_the_number_of_clips_and_the_samples_mixed_are_bounded(tmp_path, monkeypatch):
    wav(tmp_path / "a.wav", np.full(8000, 1000))
    monkeypatch.setattr(nar, "MAX_SOUNDS", 3)
    problems: list[str] = []
    track, rate = nar.mix_track(None, tmp_path, 1.0, [nar.Clip("a.wav", 0.0)] * 5, problems)
    assert track[100] == 3000
    assert len(problems) == 2 and all("at most 3 sound clips" in p for p in problems)
    monkeypatch.setattr(nar, "MAX_MIX_SAMPLES", rate * 2)
    problems = []
    track, _ = nar.mix_track(None, tmp_path, 1.0, [nar.Clip("a.wav", 0.0)] * 3, problems)
    assert track[100] == 2000 and len(problems) == 1 and "left out" in problems[0]
    monkeypatch.setattr(nar, "MAX_MIX_SAMPLES", rate * 5 // 2)  # the third clip: half of it
    problems = []
    track, _ = nar.mix_track(None, tmp_path, 1.0, [nar.Clip("a.wav", 0.0)] * 3, problems)
    assert track[100] == 3000 and track[rate * 3 // 4] == 2000
    assert len(problems) == 1 and "cut after 0.5 s" in problems[0]


@pytest.mark.parametrize("passes", MODES)
def test_sound_limit_lints_clips_the_mix_leaves_out_or_does_not_follow(tmp_path, monkeypatch,
                                                                       passes):  # fmt: skip
    """A clip past MAX_SOUNDS is left out; an animated start or gain is not heard (the mix
    reads them at 0 s): both said, never silent."""
    wav(tmp_path / "a.wav", np.full(800, 1000))
    monkeypatch.setattr(nar, "MAX_SOUNDS", 2)
    s = scene(tmp_path, passes=passes)
    for k in range(3):
        s.add(Sound(id=f"s{k}", path="a.wav", start=0.1 * k))
    s.at(0.0).to("s0.start", 1.0, duration=1.0)
    found = {(i.ids[0], i.path) for i in s.engine.lint() if i.code == "sound_limit"}
    assert found == {("s2", None), ("s0", "s0.start")}
