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
