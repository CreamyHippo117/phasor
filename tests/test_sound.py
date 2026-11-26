"""The sound bus (SOUND.md step 1): reading, mixing, loudness, the limiter, stereo mux."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor.core.schema import Narration, NarrationBlock
from phasor.time import sound
from phasor.time.narration import write_wav
from phasor.time.sound import Placed, loudness, master, mix, mixdown, true_peak


def _sine(freq, dbfs, seconds, rate=48_000, channels=2):
    t = np.arange(round(seconds * rate)) / rate
    x = sound.db_to_gain(dbfs) * np.sin(2 * math.pi * freq * t)
    return np.repeat(x[:, None], channels, axis=1).astype(np.float32)


# ---------------------------------------------------------------- loudness (BS.1770 / R128)


def test_k_weighting_at_48k_is_the_standards_filters():
    (b1, a1), (b2, a2) = sound._k_weighting(48_000)
    np.testing.assert_allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285],
                               atol=1e-10)  # fmt: skip
    np.testing.assert_allclose(a1, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-10)
    np.testing.assert_allclose(b2, [1.0, -2.0, 1.0])
    np.testing.assert_allclose(a2, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-10)


@pytest.mark.parametrize("rate", [48_000, 44_100])
@pytest.mark.parametrize("dbfs", [-23.0, -33.0])
def test_ebu_3341_steady_sine_reads_its_level(rate, dbfs):
    # EBU Tech 3341 cases 1 and 2: a 1 kHz stereo sine at -23 / -33 dBFS is -23 / -33 LUFS.
    assert loudness(_sine(1000, dbfs, 20, rate), rate) == pytest.approx(dbfs, abs=0.1)


def test_ebu_3341_relative_gate_drops_the_quiet_parts():
    # Case 3: 10 s at -36, 60 s at -23, 10 s at -36 dBFS reads -23 LUFS.
    x = np.concatenate([_sine(1000, -36, 10), _sine(1000, -23, 60), _sine(1000, -36, 10)])
    assert loudness(x, 48_000) == pytest.approx(-23.0, abs=0.1)


def test_ebu_3341_absolute_gate_drops_near_silence():
    # Case 4: -72, -36, -23, -36, -72 dBFS for 10, 10, 60, 10, 10 s reads -23 LUFS.
    parts = [(-72, 10), (-36, 10), (-23, 60), (-36, 10), (-72, 10)]
    x = np.concatenate([_sine(1000, db, s) for db, s in parts])
    assert loudness(x, 48_000) == pytest.approx(-23.0, abs=0.1)


def test_silence_and_short_sound_have_no_loudness():
    assert loudness(np.zeros((48_000, 2), np.float32), 48_000) == -math.inf
    assert loudness(_sine(1000, -20, 0.1), 48_000) == -math.inf


def test_true_peak_finds_the_peak_between_samples():
    # A sine at a quarter of the rate, 45 degrees off: every sample is at 0.707 of the peak.
    rate = 48_000
    n = np.arange(rate)
    x = (0.5 * np.sin(2 * math.pi * n / 4 + math.pi / 4)).astype(np.float32)
    assert 20 * math.log10(np.abs(x).max()) == pytest.approx(-9.03, abs=0.05)
    assert true_peak(x[:, None], rate) == pytest.approx(-6.02, abs=0.3)


# ---------------------------------------------------------------- the limiter


def test_master_leaves_a_mix_under_the_ceiling_untouched():
    x = _sine(440, -3, 1)
    assert master(x, 48_000) is x


def test_master_holds_peaks_at_the_ceiling_smoothly():
    rate = 48_000
    x = _sine(440, -20, 2, rate)
    x[rate : rate + 480] *= 20.0  # a 10 ms burst at +6 dBFS in the middle
    y = master(x, rate)
    ceiling = sound.db_to_gain(sound.CEILING_DB)
    assert np.abs(y).max() <= ceiling + 1e-7
    gain = np.divide(y[:, 0], x[:, 0], out=np.ones(len(x)), where=np.abs(x[:, 0]) > 1e-4)
    assert np.allclose(y[: rate // 2], x[: rate // 2])  # far from the burst: untouched
    assert np.allclose(y[-rate // 2 :], x[-rate // 2 :])
    # The gain ramps over tens of milliseconds instead of jumping on the burst.
    near = gain[rate - 2400 : rate + 2880]
    assert np.abs(np.diff(near[np.abs(x[rate - 2400 : rate + 2880, 0]) > 1e-3])).max() < 0.05


# ---------------------------------------------------------------- the bus


def test_mix_places_clips_with_their_gain_and_centres_mono():
    rate = 1000
    one = np.full(500, 0.5, np.float32)
    bus, r = mix([Placed(0.25, one, rate), Placed(2.0, one, rate, gain_db=-6.0)], duration=3.0)
    assert r == rate and bus.shape == (3000, 2)
    assert bus[249].tolist() == [0, 0] and bus[250].tolist() == [0.5, 0.5]
    assert bus[749].tolist() == [0.5, 0.5] and bus[750].tolist() == [0, 0]
    assert bus[2000, 0] == pytest.approx(0.5 * 10 ** (-6 / 20))
    assert mix([], 1.0) is None


def test_mix_runs_at_the_common_rate_or_48k():
    a = Placed(0, np.zeros(441, np.float32), 44_100)
    b = Placed(0, np.zeros(220, np.float32), 22_050)
    assert mix([a, a], 0.1)[1] == 44_100
    bus, rate = mix([a, b], 0.1)
    assert rate == 48_000 and len(bus) == 4800


def test_a_clip_running_past_the_scene_lengthens_the_bus_and_one_before_it_is_cut():
    rate = 100
    bus, _ = mix([Placed(0.5, np.ones(100, np.float32), rate)], duration=1.0)
    assert len(bus) == 150
    bus, _ = mix([Placed(-0.5, np.ones(100, np.float32), rate)], duration=1.0)
    assert bus[:50, 0].tolist() == [1.0] * 50 and not bus[50:].any()


# ---------------------------------------------------------------- the scene's sound


def _narration(*starts):
    return Narration(script="x", voice="test", blocks=tuple(
        NarrationBlock(text="a", start=s, duration=0.5, audio="a.wav") for s in starts))  # fmt: skip


def test_mixdown_keeps_a_quiet_voice_bit_for_bit_in_both_channels(tmp_path):
    rate = 22_050
    rng = np.random.default_rng(0)
    voice = rng.integers(-20_000, 20_000, rate // 2).astype(np.int16)  # peaks under -1 dBFS
    write_wav(tmp_path / "a.wav", voice, rate)
    track, r = mixdown(_narration(0.25, 2.0), tmp_path, duration=3.0)
    assert r == rate and track.dtype == np.int16 and track.shape == (3 * rate, 2)
    i = round(0.25 * rate)
    assert (track[:, 0] == track[:, 1]).all()
    assert (track[i : i + len(voice), 0] == voice).all() and not track[:i].any()
    assert (track[round(2.0 * rate) :][: len(voice), 0] == voice).all()


def test_mixdown_is_deterministic_and_limits_overlapping_loud_clips(tmp_path):
    rate = 8000
    write_wav(tmp_path / "a.wav", np.full(4000, 30_000, np.int16), rate)
    first = mixdown(_narration(0.0, 0.25), tmp_path, 1.0)[0]
    again = mixdown(_narration(0.0, 0.25), tmp_path, 1.0)[0]
    assert first.tobytes() == again.tobytes()
    assert np.abs(first.astype(np.int32)).max() <= round(32768 * 10 ** (-1 / 20))


def test_mixdown_is_none_without_audio_and_refuses_files_outside(tmp_path):
    est = Narration(script="x", voice="estimate",
                    blocks=(NarrationBlock(text="a", start=0, duration=1),))  # fmt: skip
    assert mixdown(est, tmp_path, 1.0) is None and mixdown(None, tmp_path, 1.0) is None
    out = Narration(script="x", voice="test", blocks=(
        NarrationBlock(text="a", start=0, duration=1, audio="../secret.wav"),))  # fmt: skip
    with pytest.raises(ValueError, match="scene's folder"):
        mixdown(out, tmp_path, 1.0)


# ---------------------------------------------------------------- reading files


def test_read_audio_reads_wav_of_any_width_as_stereo_float(tmp_path):
    from scipy.io import wavfile

    x = 0.5 * np.sin(np.arange(1000) * 0.1)
    wavfile.write(tmp_path / "f.wav", 8000, np.stack([x, -x], axis=1).astype(np.float32))
    wavfile.write(tmp_path / "i32.wav", 8000, (x * 2**31).astype(np.int32))
    wavfile.write(tmp_path / "u8.wav", 8000, (x * 127 + 128).astype(np.uint8))
    s, rate = sound.read_audio(tmp_path / "f.wav")
    assert rate == 8000 and s.shape == (1000, 2) and s.dtype == np.float32
    np.testing.assert_allclose(s[:, 1], -x, atol=1e-6)
    for name, tol in [("i32.wav", 1e-6), ("u8.wav", 2e-2)]:
        s, _ = sound.read_audio(tmp_path / name)
        np.testing.assert_allclose(s[:, 0], x, atol=tol)
        assert (s[:, 0] == s[:, 1]).all()
    assert not s.flags.writeable


def test_read_audio_reads_compressed_files_through_soundfile(tmp_path):
    sf = pytest.importorskip("soundfile")
    x = _sine(440, -6, 0.5, 44_100)
    sf.write(tmp_path / "bed.flac", x, 44_100)
    s, rate = sound.read_audio(tmp_path / "bed.flac")
    assert rate == 44_100 and s.shape == x.shape
    np.testing.assert_allclose(s, x, atol=1e-4)


# ---------------------------------------------------------------- the video file


def test_mux_audio_writes_a_stereo_track(tmp_path):
    av = pytest.importorskip("av")
    from phasor.render.encode import mux_audio

    video = tmp_path / "v.mp4"
    with av.open(str(video), mode="w") as c:
        st = c.add_stream("mpeg4", rate=10)
        st.width, st.height, st.pix_fmt = 32, 32, "yuv420p"
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            for p in st.encode(frame):
                c.mux(p)
        for p in st.encode(None):
            c.mux(p)
    left = (8000 * np.sin(np.arange(48_000) * 0.05)).astype(np.int16)
    mux_audio(video, np.stack([left, np.zeros_like(left)], axis=1), 48_000)
    with av.open(str(video)) as c:
        a = c.streams.audio[0]
        assert a.codec_context.channels == 2 and a.rate == 48_000
        pcm = np.concatenate([f.to_ndarray() for f in c.decode(a)], axis=1)
    assert np.abs(pcm[0]).max() > 0.1 and np.abs(pcm[1]).max() < 0.01  # left loud, right quiet
