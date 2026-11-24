"""Milestone 23: MIDI. One type, ``Notes``, with an import end (a ``.mid`` in the scene's
folder, or notes a script makes), a play end (a deterministic built-in synthesizer mixed
like a ``Sound`` clip) and an export end (``export_midi``). The SMF reader is self-written:
hostile files are lint and play nothing, never a traceback, a hang or a memory blow-up."""

from __future__ import annotations

import hashlib
import math
import struct
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Notes, Scene
from phasor.core.issues import DocumentError
from phasor.domains import midi_math as mm
from phasor.time import narration as nar
from tests.test_lint import MODES

ROOT = Path(__file__).resolve().parents[1]

# The example file of the SMF 1.0 specification (format 0, 96 ticks a quarter note, 120 bpm),
# with what the specification says it holds: channel 1 (from 0: 0) program 5, channel 2
# program 46, channel 3 program 70; C3 and C4 on channel 3 at 0, G4 on channel 2 a quarter
# later (velocity 64), E5 on channel 1 at the half (velocity 32), all released together at
# the end of the bar (two seconds).
SPEC = bytes.fromhex(
    "4D546864 00000006 0000 0001 0060"
    "4D54726B 0000003B"
    "00FF5804 04021808" "00FF5103 07A120" "00C005" "00C12E" "00C246"
    "00923060" "003C60" "60914340" "60904C20"
    "81408230 40" "003C40" "00814340" "00804C40" "00FF2F00".replace(" ", "")
)  # fmt: skip


def smf(*tracks: bytes, fmt: int = 1, division: int = 96, ntrk: int | None = None) -> bytes:
    head = b"MThd" + struct.pack(">IHHH", 6, fmt, len(tracks) if ntrk is None else ntrk,
                                 division)  # fmt: skip
    return head + b"".join(b"MTrk" + struct.pack(">I", len(t)) + t for t in tracks)


def scene(folder: Path, length: float = 3.0, passes=None) -> Scene:
    s = Scene("notes", passes=passes, base_dir=folder)
    s.add(Dot(id="tick"))
    s.marker("end", length)
    return s


# ---------------------------------------------------------------- reading


def test_the_specification_example_reads_as_the_specification_says():
    song = mm.read_smf(SPEC)
    assert song.error is None and song.cut is None
    assert (song.format, song.tracks, song.bpm, song.time_signature) == (0, 1, 120.0, (4, 4))
    assert song.notes == (
        mm.Note(48, 0.0, 2.0, 96, 2, 70),
        mm.Note(60, 0.0, 2.0, 96, 2, 70),
        mm.Note(67, 0.5, 1.5, 64, 1, 46),
        mm.Note(76, 1.0, 1.0, 32, 0, 5),
    )


def test_the_tempo_map_turns_ticks_into_seconds():
    # 96 ticks a quarter: 120 bpm for one quarter (0.5 s), then 60 bpm (a quarter is 1 s).
    conductor = (
        b"\x00\xff\x51\x03\x07\xa1\x20" + b"\x60\xff\x51\x03\x0f\x42\x40" + b"\x00\xff\x2f\x00"
    )
    notes = (b"\x00\x90\x3c\x40" + b"\x81\x40\x80\x3c\x40"  # on at 0, off at 192 ticks
             b"\x00\x90\x3e\x40" + b"\x60\x3e\x00" + b"\x00\xff\x2f\x00")  # fmt: skip
    song = mm.read_smf(smf(conductor, notes))
    assert song.error is None and song.cut is None and song.tempo_changes == 2
    (a, b) = song.notes
    assert (a.pitch, a.start, a.length) == (60, 0.0, 1.5)  # 0.5 s at 120 + 1 s at 60
    assert (b.pitch, b.start, b.length) == (62, 1.5, 1.0)  # note-on velocity 0 ends it


def test_smpte_time_counts_frames_not_beats():
    # 25 frames a second, 40 ticks a frame: a tick is 1 ms whatever the tempo says.
    division = ((256 - 25) << 8) | 40
    track = b"\x00\xff\x51\x03\x0f\x42\x40\x00\x90\x3c\x40\x87\x68\x80\x3c\x40\x00\xff\x2f\x00"
    song = mm.read_smf(smf(track, fmt=0, division=division))
    assert song.error is None and song.division == "SMPTE 25 fps x 40"
    assert song.notes[0].start == 0.0 and abs(song.notes[0].length - 1.0) < 1e-12


def test_overlapping_unclosed_and_empty_notes():
    track = (
        b"\x00\x90\x3c\x40"
        b"\x10\x3c\x50"  # the same key twice: first in, first out
        b"\x10\x80\x3c\x00"
        b"\x10\x3c\x00"
        b"\x00\x90\x40\x40\x00\x80\x40\x00"  # on and off at one tick: no sound, dropped
        b"\x00\x90\x43\x30"
        b"\x20\xff\x2f\x00"
    )  # never let go: ends with the track
    song = mm.read_smf(smf(track, fmt=0))
    got = [(n.pitch, round(n.start * 192), round(n.length * 192), n.velocity) for n in song.notes]
    assert got == [(60, 0, 32, 64), (60, 16, 32, 80), (67, 48, 32, 48)]


HOSTILE = {
    "empty": (b"", "not a Standard MIDI File"),
    "garbage": (bytes(range(256)) * 4, "not a Standard MIDI File"),
    "cut header": (SPEC[:12], "not a Standard MIDI File"),
    "short header length": (b"MThd\0\0\0\x04\0\0\0\x01\0\x60" + SPEC[14:], "at least 6"),
    "lying header length": (b"MThd\xff\xff\xff\xff" + SPEC[8:], "the file has"),
    "format 2": (smf(b"\x00\xff\x2f\x00", fmt=2), "format 2"),
    "format 7": (smf(b"\x00\xff\x2f\x00", fmt=7), "format 7"),
    "no tracks": (smf(ntrk=0), "no tracks"),
    "no track chunk": (b"MThd\0\0\0\x06\0\0\0\x01\0\x60XTrk\0\0\0\0", "no track"),
    "zero division": (smf(b"\x00\xff\x2f\x00", division=0), "0 ticks"),
    "SMPTE 23 fps": (smf(b"\x00\xff\x2f\x00", division=((256 - 23) << 8) | 4), "SMPTE"),
    "SMPTE 0 ticks": (smf(b"\x00\xff\x2f\x00", division=((256 - 25) << 8)), "SMPTE"),
    "too large": (b"MThd" + bytes(mm.MAX_FILE_BYTES), "larger than"),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_files_that_are_not_midi_are_an_error_with_a_reason(name):
    data, said = HOSTILE[name]
    song = mm.read_smf(data)
    assert song.notes == () and said in song.error


ON = b"\x00\x90\x3c\x40\x60\x80\x3c\x40"  # one quarter-note C4

CUT = {
    "track length 0xFFFFFFFF": (SPEC[:18] + b"\xff\xff\xff\xff" + SPEC[22:], "says 4294967295"),
    "track cut short": (SPEC[:-9], "the file has"),
    "a delta longer than 4 bytes": (smf(ON + b"\xff\xff\xff\xff\x7f\x90\x3c\x40"), "longer than"),
    "a data byte past 127": (smf(ON + b"\x00\x90\x3c\xc0"), "past 127"),
    "running status with no status": (smf(b"\x00\x3c\x40" + ON), "no status"),
    "a real-time status in a file": (smf(ON + b"\x00\xf8"), "does not belong"),
    "a meta event past the track": (smf(ON + b"\x00\xff\x01\x7f\x00"), "meta event says"),
    "a sysex past the track": (smf(ON + b"\x00\xf0\x8f\xff\xff\x7f"), "system exclusive"),
    "a tempo of 0": (smf(b"\x00\xff\x51\x03\x00\x00\x00" + ON), "is ignored"),
    "a tempo of 2 bytes": (smf(b"\x00\xff\x51\x02\x07\xa1" + ON), "tempo event of 2 bytes"),
    "more tracks said than there": (smf(ON, ntrk=3), "says 3 tracks"),
    "a note held for days": (smf(b"\x00\x90\x3c\x40\x8f\xff\xff\x7f\xff\x2f\x00"), "cut to it"),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(CUT))
def test_files_with_a_fault_keep_what_comes_before_it(name):
    data, said = CUT[name]
    song = mm.read_smf(data)
    assert song.error is None and said in song.cut
    assert all(n.length <= mm.MAX_NOTE_SECONDS for n in song.notes)
    if name not in ("running status with no status", "track length 0xFFFFFFFF"):
        assert any(n.pitch == 60 for n in song.notes)  # what came before the fault is kept


def test_the_header_may_be_longer_than_six_bytes():
    data = b"MThd\0\0\0\x08\0\0\0\x01\0\x60\xab\xcd" + SPEC[14:]
    assert mm.read_smf(data).notes == mm.read_smf(SPEC).notes


def test_unknown_chunks_are_skipped():
    data = SPEC[:14] + b"XFIH\0\0\0\x03abc" + SPEC[14:]
    assert mm.read_smf(data).notes == mm.read_smf(SPEC).notes


def _timed(data: bytes) -> tuple[mm.Song, float, float]:
    tracemalloc.start()
    t = time.perf_counter()
    try:
        song = mm.read_smf(data)
        return song, time.perf_counter() - t, tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


def test_a_maximal_file_is_bounded_in_time_and_memory():
    """4 MiB of 3-byte running-status note events: stopped at MAX_NOTES (measured here:
    0.5 s without tracemalloc, 6 s and 40 MiB with it); the bounds are generous for slow
    machines and Windows' coarse clock."""
    body = b"\x00\x90\x3c\x40" + b"\x01\x3c\x00\x01\x3c\x40" * 700_000
    data = smf(body)[: mm.MAX_FILE_BYTES]
    song, seconds, mib = _timed(data)
    assert len(song.notes) == mm.MAX_NOTES and "notes" in song.cut
    assert seconds < 60 and mib < 256


def test_events_tempo_changes_and_tracks_are_bounded():
    meta = b"\x00\xff\x01\x00" * (mm.MAX_EVENTS + 10)  # empty text events
    song = mm.read_smf(smf(meta + ON))
    assert f"more than {mm.MAX_EVENTS} events" in song.cut
    tempos = b"\x00\xff\x51\x03\x07\xa1\x20" * (mm.MAX_TEMPOS + 1)
    assert "tempo changes" in mm.read_smf(smf(tempos + ON)).cut
    many = smf(*([ON] * 300), ntrk=300)
    song = mm.read_smf(many)
    assert song.tracks == mm.MAX_TRACKS and "300 tracks" in song.cut
    assert len(song.notes) == mm.MAX_TRACKS


def test_mutated_files_never_raise():
    """3000 random corruptions of real files (bytes flipped, cut, lengths rewritten): every
    one is a Song, and none takes long."""
    gen = np.random.default_rng(23)
    good = [SPEC, mm.write_smf([("a", [mm.Note(60 + i % 12, i * 0.1, 0.2, 90, i % 16, i % 128)
                                       for i in range(200)])])]  # fmt: skip
    t = time.perf_counter()
    for k in range(3000):
        data = bytearray(good[k % 2])
        for _ in range(int(gen.integers(1, 6))):
            what = gen.integers(0, 3)
            i = int(gen.integers(0, len(data)))
            if what == 0:
                data[i] = int(gen.integers(0, 256))
            elif what == 1:
                data = data[: i + 1]
            else:
                data[i : i + 4] = b"\xff\xff\xff\xff"
        song = mm.read_smf(bytes(data))
        assert isinstance(song, mm.Song)
        assert all(0 < n.length <= mm.MAX_NOTE_SECONDS for n in song.notes)
    assert time.perf_counter() - t < 120


# ---------------------------------------------------------------- writing


def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """The chunks of a file, walked by their stated lengths (which must add up exactly)."""
    out, pos = [], 0
    while pos < len(data):
        kind, n = data[pos : pos + 4], struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        out.append((kind, data[pos + 8 : pos + 8 + n]))
        pos += 8 + n
    assert pos == len(data)
    return out


def test_written_files_are_valid_and_read_back_the_same():
    gen = np.random.default_rng(5)
    raw = sorted((mm.Note(int(gen.integers(21, 109)), round(float(gen.uniform(0, 30)), 3),
                            round(float(gen.uniform(0.05, 3)), 3), int(gen.integers(1, 128)),
                            int(gen.integers(0, 16)), int(gen.integers(0, 128)))
                  for _ in range(500)), key=lambda n: (n.start, n.channel, n.pitch))  # fmt: skip
    # one key of one channel never overlaps itself (MIDI cannot say which note-off ends which)
    held: dict[tuple[int, float], float] = {}
    notes = []
    for n in raw:
        if held.get((n.channel, n.pitch), -1.0) < n.start - 0.01:
            held[(n.channel, n.pitch)] = n.start + n.length
            notes.append(n)
    half = len(notes) // 2
    data = mm.write_smf([("melody", notes[:half]), ("rest", notes[half:])], title="song")
    chunks = _chunks(data)
    assert [k for k, _ in chunks] == [b"MThd", b"MTrk", b"MTrk", b"MTrk"]
    assert struct.unpack(">HHH", chunks[0][1]) == (1, 3, mm.PPQ)
    assert all(body.endswith(b"\x00\xff\x2f\x00") for _, body in chunks[1:])
    song = mm.read_smf(data)
    assert song.error is None and song.cut is None and song.names == ("song", "melody", "rest")
    tick = 1 / 1920
    back = sorted(song.notes, key=lambda n: (round(n.start / tick), n.channel, n.pitch,
                                             round(n.length / tick)))  # fmt: skip
    want = sorted(notes, key=lambda n: (round(n.start / tick), n.channel, n.pitch,
                                        round(n.length / tick)))  # fmt: skip
    for a, b in zip(back, want, strict=True):
        assert (a.pitch, a.velocity, a.channel) == (b.pitch, b.velocity, b.channel)
        assert abs(a.start - b.start) <= tick and abs(a.length - b.length) <= 2 * tick


def test_programs_are_written_where_they_change():
    notes = [mm.Note(60, 0, 0.5, 90, 0, 14), mm.Note(62, 1, 0.5, 90, 0, 40),
             mm.Note(36, 0, 0.1, 90, 9, 0)]  # fmt: skip
    song = mm.read_smf(mm.write_smf([("a", notes)]))
    assert {(n.pitch, n.program) for n in song.notes} == {(60, 14), (62, 40), (36, 0)}


def test_an_empty_scene_writes_a_valid_file():
    data = mm.write_smf([])
    assert [k for k, _ in _chunks(data)] == [b"MThd", b"MTrk"]
    song = mm.read_smf(data)
    assert song.error is None and song.notes == ()


# ---------------------------------------------------------------- the synthesizer


def _synth(notes, **kw) -> mm.Synth:
    return mm.Synth(notes, noise=mm.drum_noise(np.random.default_rng(3)), **kw)


MIXED = [mm.Note(48 + (i * 7) % 36, 0.13 * i, 0.4, 40 + i % 80, i % 3, (i * 9) % 128)
         for i in range(60)] + [mm.Note(35 + i % 25, 0.25 * i, 0.1, 100, 9) for i in range(30)]  # fmt: skip


def test_any_window_is_that_slice_of_the_whole():
    for instrument in mm.INSTRUMENTS:
        syn = _synth(MIXED, instrument=instrument)
        whole = syn.read(0, syn.frames)
        assert whole.dtype == np.float32 and np.abs(whole).max() <= 1.0
        assert np.abs(whole).max() > 0.01
        for a, n in ((0, 1), (1000, 777), (65535, 3), (70000, 200001), (syn.frames - 5, 50)):
            assert np.array_equal(syn.read(a, n), whole[a : a + n]), (instrument, a, n)


def test_timing_is_sample_exact():
    syn = _synth([mm.Note(69, 0.5, 0.25, 127, 0)], instrument="organ")
    x = syn.read(0, syn.frames)
    first = int(np.flatnonzero(x)[0])
    assert first == 24001  # the attack starts from 0 at sample 24000
    assert syn.frames == 24000 + 12000 + round(mm.TIMBRES["organ"].release * mm.RATE)
    assert abs(x[-1]) < 1e-5 and x[syn.frames - 1 :].size == 1  # the release ends at 0


def test_pitch_and_velocity():
    syn = _synth([mm.Note(69, 0.0, 1.0, 127)], instrument="sine")
    x = syn.read(0, mm.RATE).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x[12000:36000] * np.hanning(24000)))
    assert abs(np.argmax(spectrum) * mm.RATE / 24000 - 440.0) < 2.5
    quiet = _synth([mm.Note(69, 0.0, 1.0, 32)], instrument="sine").read(0, mm.RATE)
    assert 0.05 < np.abs(quiet).max() / np.abs(x).max() < 0.08  # (32/127)^2


def test_polyphony_is_capped():
    chord = [mm.Note(40 + i % 60, 0.0, 2.0, 100) for i in range(mm.MAX_VOICES + 10)]
    syn = _synth(chord)
    assert len(syn.voices) == mm.MAX_VOICES and syn.dropped == 10 and syn.dropped_at == 0.0


def _digest(notes) -> str:
    syn = _synth(notes)
    return hashlib.sha256(syn.read(0, syn.frames).tobytes()).hexdigest()


def test_the_same_samples_in_another_process():
    """Bit for bit across processes (a render worker mixes nothing, but a reader in another
    process must hear what this one hears)."""
    here = _digest(MIXED)
    code = "from tests.test_midi import _digest, MIXED; print(_digest(MIXED))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=300, encoding="utf-8")  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == here


def test_a_thousand_notes_over_ten_minutes():
    """Measured on this machine: about 2 s and well under 100 MiB for the synthesizer, and
    the whole 48 kHz mix (its int16 track is 55 MiB) a few seconds more."""
    notes = [mm.Note(36 + (i * 5) % 60, i * 0.6, 0.5 + (i % 4) * 0.25, 60 + i % 60, i % 2,
                     (i * 11) % 128) for i in range(1000)]  # fmt: skip
    tracemalloc.start()
    t = time.perf_counter()
    syn = _synth(notes)
    for a in range(0, syn.frames, 1 << 20):  # as the mixer reads it
        syn.read(a, 1 << 20)
    seconds = time.perf_counter() - t
    mib = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    assert syn.frames / mm.RATE > 600 and seconds < 120 and mib < 200


# ---------------------------------------------------------------- the mix

BASELINE = {  # mixes before Milestone 23 (tests/mix_baseline.py on the commit it started from)
    "tts_whole_files": "115180b69f7f24666a65d467acebad1f0877d9627ce615de2877182e19605174",
    "sound_clips_48k": "983355564f9cf6bd60f04cb96909828384377aeb21dab092d3306115215adb7c",
    "tts_and_clips": "3741d060776c8d1f7ab411459dcb6e35f8fe3b08a0a76c579d193124e129bf5f",
    "recording_spans": "96721a308ab2006f46d7c854f4f6f8f23c661c4c7ed630f63713966b907df4ea",
    "window": "c7b851d432d7b3e6c66ee9ef51db5ce3ba6ac9f81267ef057f2814c95e7e05a8",
}


def test_existing_mixes_are_bit_identical(tmp_path):
    from tests.mix_baseline import mixes

    assert mixes(tmp_path) == BASELINE


def test_notes_mix_like_a_sound_clip(tmp_path):
    s = scene(tmp_path)
    s.add(Notes(id="arp", notes=[(60, 0.0, 0.3, 100), (64, 0.3, 0.3, 100)], start=1.0))
    track, rate = s.engine.audio_track()
    assert rate == nar.SOUND_RATE and len(track) == 3 * rate
    assert not track[:48001].any() and track[48001] != 0
    st = s.engine.state_at(0.0)
    alone = st["arp"].impl.placed(st).synth
    want = np.clip(np.round(alone.read(0, alone.frames) * 32768.0), -32768, 32767)
    assert np.array_equal(track[rate : rate + len(want)], want.astype(np.int16))
    assert not track[rate + len(want) :].any()


def test_notes_resample_to_a_narration_rate_and_windows_match(tmp_path):
    """A narration at 8 kHz sets the track's rate: the synthesizer (48 kHz) is resampled
    by the mixer as a WAV clip is, and a window of the mix is that slice of the whole."""
    from phasor.core.schema import Narration, NarrationBlock
    from tests.mix_baseline import _tone, _wav

    a = _wav(tmp_path / "say.wav", _tone(8000, 8000, 200.0), 8000)
    said = Narration(script="A.", voice="file", blocks=(
        NarrationBlock(text="A.", start=0.0, duration=1.0, audio=a),))  # fmt: skip
    syn = mm.Synth([mm.Note(72 + i, 0.2 * i, 0.4, 90) for i in range(8)], "bell")
    clip = nar.Clip(path="bells", start=0.5, source=syn)
    whole, rate = nar.mix_track(said, tmp_path, 3.0, [clip])
    assert rate == 8000 and len(whole) == 3 * rate
    for lo, hi in ((1.0, 2.0), (0.3, 2.9), (0.55, 0.6)):
        part, _ = nar.mix_track(said, tmp_path, 3.0, [clip], start=lo, end=hi)
        assert np.array_equal(part, whole[round(lo * rate) : round(lo * rate) + len(part)])
    quiet = nar.mix_track(said, tmp_path, 3.0)[0]
    assert np.array_equal(whole[:4000], quiet[:4000])  # nothing before the notes start
    assert not np.array_equal(whole[4001:], quiet[4001:])


def test_a_marker_start_follows_the_marker(tmp_path):
    s = scene(tmp_path)
    s.marker("beat", 0.75)
    s.add(Notes(id="n", notes=[(60, 0.0, 0.2, 100)], start="beat"))
    first = int(np.flatnonzero(s.engine.audio_track()[0])[0])
    s.marker("beat", 1.25)
    again = int(np.flatnonzero(s.engine.audio_track()[0])[0])
    assert (first, again) == (36001, 60001)
    s.engine.apply({"op": "remove_marker", "name": "beat"}, origin="script")
    problems: list[str] = []
    assert s.engine.audio_track(problems) is None
    assert any("marker 'beat'" in p for p in problems)


def test_transpose_tempo_channels_repeat_and_gain(tmp_path):
    s = scene(tmp_path, 10.0)
    base = [(60, 0.0, 0.5, 100, 0), (64, 0.5, 0.5, 100, 1), (36, 0.0, 0.1, 100, 9)]
    s.add(Notes(id="n", notes=base, transpose=12, tempo_scale=2.0, channels=(0, 9), repeat=3))
    sem = s.engine.inspect(0.0, id="n")["objects"][0]["semantic"]
    assert sem["count"] == 6 and sem["channels"] == [0, 9]
    assert sem["pitch_range"] == [36.0, 72.0]  # drums are never transposed
    assert sem["span"] == [0.0, 0.75]  # two copies of 0.25 s, then the last
    loud = Notes(id="m", notes=base, gain=6.0)
    s.add(loud)
    st = s.engine.state_at(0.0)
    assert st["m"].impl.audio_clip(st).gain == 6.0


def test_drum_noise_comes_from_rng_and_the_seed(tmp_path):
    s = scene(tmp_path)
    hit = [(38, 0.0, 0.1, 120, 9)]
    s.add(Notes(id="a", notes=hit))
    s.add(Notes(id="b", notes=hit, seed=1))
    st = s.engine.state_at(0.0)
    xa, xb = (st[i].impl.placed(st).synth.read(0, 4800) for i in ("a", "b"))
    assert not np.array_equal(xa, xb)
    from phasor.core.rng import rng

    assert np.array_equal(st["a"].impl.placed(st).synth.noise, mm.drum_noise(rng("a", 0)))


# ---------------------------------------------------------------- the object


def test_notes_are_a_drawless_object_with_semantics_and_bounds(tmp_path):
    (tmp_path / "song.mid").write_bytes(SPEC)
    s = scene(tmp_path)
    s.add(Notes(id="song", path="song.mid", start=0.5))
    sem = s.engine.inspect(0.0, id="song")["objects"][0]["semantic"]
    assert sem == {"kind": "notes", "source": "file", "start": 0.5, "gain": 0.0, "count": 4,
                   "instrument": "auto", "drawn": 0.0, "span": [0.0, 2.0],
                   "pitch_range": [48.0, 76.0], "channels": [0, 1, 2],
                   "programs": ["Bassoon", "Electric Piano 2", "Orchestral Harp"],
                   "voices": ["lead", "pad", "pluck"], "file": "song.mid", "missing": False,
                   "format": 0, "tracks": 1, "bpm": 120.0,
                   "time_signature": [4, 4]}  # fmt: skip
    st = s.engine.state_at(0.0)
    assert st["song"].impl.world_bounds(st).lo == (0.0, 0.0, 0.0)
    assert s.engine.lint() == []


@pytest.mark.parametrize("params", [
    {"path": "a.wav"}, {"path": "../a.mid"}, {"path": "/a.mid"},
    {"path": "a.mid", "notes": [(60, 0, 1, 100)]},
    {"notes": [(60, 0, 1)]}, {"notes": [(128, 0, 1, 100)]}, {"notes": [(60, -1, 1, 100)]},
    {"notes": [(60, 0, 0, 100)]}, {"notes": [(60, 0, 61, 100)]}, {"notes": [(60, 0, 1, 0)]},
    {"notes": [(60, 0, 1, 100, 16)]}, {"notes": [(60, 0, 1, 100.5)]},
    {"notes": [(float("nan"), 0, 1, 100)]}, {"notes": "C4"},
    {"notes": [(60, 0, 1, 100)] * 20_001},
    {"start": "not a name!"}, {"start": -1}, {"start": float("inf")},
    {"gain": 30}, {"transpose": 49}, {"tempo_scale": 0}, {"channels": (16,)},
    {"repeat": 0}, {"repeat": 65}, {"instrument": "kazoo"}, {"program": 128},
])  # fmt: skip
def test_bad_parameters_are_refused_when_set(tmp_path, params):
    s = scene(tmp_path)
    with pytest.raises((DocumentError, ValueError)):
        s.add(Notes(id="n", **params))
    r = s.engine.apply({"op": "add_object", "object": {"id": "m", "type": "notes",
                                                       "params": params}}, origin="script")  # fmt: skip
    assert not r.ok


def test_the_audio_parameters_hold_one_value(tmp_path):
    s = scene(tmp_path)
    s.add(Notes(id="n", notes=[(60, 0, 1, 100)]))
    r = s.engine.apply({"op": "add_segment", "path": "n.gain",
                        "segment": {"t0": 0, "t1": 1, "to": -6}}, origin="script")  # fmt: skip
    assert not r.ok and "bad_param" in str(r.errors)


@pytest.mark.parametrize("passes", MODES)
def test_missing_bad_and_truncated_files_are_lint(tmp_path, passes):
    s = scene(tmp_path, passes=passes)
    s.add(Notes(id="gone", path="gone.mid"))
    (tmp_path / "junk.mid").write_bytes(b"MThd nothing here")
    s.add(Notes(id="junk", path="junk.mid"))
    (tmp_path / "cut.mid").write_bytes(SPEC[:-9])
    s.add(Notes(id="cut", path="cut.mid"))
    issues = {(i.code, i.ids) for i in s.engine.lint()}
    assert ("missing_file", ("gone",)) in issues
    assert ("bad_file", ("junk",)) in issues and ("truncated", ("cut",)) in issues
    bad = next(i for i in s.engine.lint() if i.code == "bad_file")
    assert "MIDI" in bad.fix_hint
    problems: list[str] = []
    track, _ = s.engine.audio_track(problems)  # the cut file still plays what it has
    assert track.any() and problems == []


def test_a_file_changed_on_disk_is_read_again(tmp_path):
    import os

    path = tmp_path / "song.mid"
    path.write_bytes(SPEC)
    s = scene(tmp_path)
    s.add(Notes(id="song", path="song.mid"))
    assert s.engine.inspect(0.0, id="song")["objects"][0]["semantic"]["count"] == 4
    path.write_bytes(mm.write_smf([("x", [mm.Note(60, 0, 1, 100)])]))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert s.engine.inspect(0.0, id="song")["objects"][0]["semantic"]["count"] == 1


@pytest.mark.parametrize("passes", MODES)
def test_notes_limit(tmp_path, passes):
    s = scene(tmp_path, 2.0, passes=passes)
    s.add(Notes(id="late", notes=[(60, 0.0, 0.5, 100), (62, 1.5, 0.5, 100)], start=1.0))
    s.add(Notes(id="lost", notes=[(60, 0.0, 0.5, 100)], start="nowhere"))
    s.add(Notes(id="crowd", notes=[(30 + i, 0.25, 1.0, 100) for i in range(mm.MAX_VOICES + 1)]))
    got = {i.ids: i for i in s.engine.lint() if i.code == "notes_limit"}
    assert set(got) == {("late",), ("lost",), ("crowd",)}
    assert "1 of its 2 notes" in got[("late",)].message and got[("late",)].times == (2.0,)
    assert "'nowhere'" in got[("lost",)].message
    assert "1 notes are dropped" in got[("crowd",)].message and got[("crowd",)].times == (0.25,)


def test_a_mid_file_needs_no_pyav(tmp_path, monkeypatch):
    from phasor.core import extras

    monkeypatch.setattr(extras, "available", lambda m: m != "av")
    (tmp_path / "song.mid").write_bytes(SPEC)
    s = scene(tmp_path)
    s.add(Notes(id="song", path="song.mid"))
    assert s.engine.lint() == []


# ---------------------------------------------------------------- export


def test_export_writes_the_notes_as_heard_at_their_scene_times(tmp_path):
    s = scene(tmp_path, 3.0)
    s.marker("beat", 0.5)
    s.add(Notes(id="arp", notes=[(60, 0.0, 0.25, 90), (64, 0.25, 0.25, 90), (67, 3.0, 1, 90)],
                start="beat", transpose=2))  # fmt: skip
    s.add(Notes(id="hit", notes=[(36, 2.75, 0.5, 120, 9)]))
    r = s.export_midi(tmp_path / "out" / "scene.mid")
    assert r["tracks"] == 2 and r["notes"] == 4 and r["seconds"] == 4.5
    song = mm.read_smf((tmp_path / "out" / "scene.mid").read_bytes())
    assert song.error is None and song.names == ("notes", "arp", "hit")
    got = [(n.pitch, n.start, round(n.length, 4), n.channel) for n in song.notes]
    # the score, also what is past the scene's end (lint notes_limit says it is not heard)
    assert got == [(62, 0.5, 0.25, 0), (66, 0.75, 0.25, 0), (36, 2.75, 0.5, 9), (69, 3.5, 1.0, 0)]
    # and played back from the file, it is the same notes
    s.add(Notes(id="again", path="out/scene.mid"))
    sem = s.engine.inspect(0.0, id="again")["objects"][0]["semantic"]
    assert sem["count"] == 4 and sem["programs"] == ["Acoustic Grand Piano", "Drum kit"]


def test_the_workspace_tool_writes_inside_the_workspace(tmp_path):
    from phasor.workspace.tools import ToolError, Workspace

    ws = Workspace(tmp_path)
    ws.scene_new("tune")
    r = ws.export_midi()  # no notes: still a valid file
    assert r["notes"] == 0 and mm.read_smf(Path(r["path"]).read_bytes()).error is None
    ws.scene_edit({"op": "add_object", "object": {"id": "tune", "type": "notes",
                   "params": {"notes": [[60, 0, 0.5, 100], [67, 0.5, 0.5, 100]]}}})  # fmt: skip
    r = ws.export_midi("out/tune.mid")
    assert r["notes"] == 2 and (tmp_path / "out" / "tune.mid").is_file()
    with pytest.raises(ToolError):
        ws.export_midi("../elsewhere.mid")
    with pytest.raises(ToolError):
        ws.export_midi("tune.txt")


# ---------------------------------------------------------------- the code view, the example


def test_the_generated_script_rebuilds_notes(tmp_path):
    from phasor.core.diff import diff_documents
    from tests.test_pycode import plain, rebuild

    (tmp_path / "song.mid").write_bytes(SPEC)
    s = scene(tmp_path)
    s.marker("beat", 1.0)
    s.add(Notes(id="arp", notes=[(57 + 12 * math.log2(k), 0.1 * k, 0.5, 90) for k in (1, 3)],
                start="beat", instrument="sine", show=True, pos=(0, -1, 0)))  # fmt: skip
    s.add(Notes(id="song", path="song.mid", start=0.25, channels=(1, 2), repeat=2))
    again = rebuild(s.engine.document, tmp_path)
    d = diff_documents(plain(s.engine.document), plain(again))
    assert d["same"], d["changes"][:3]


def test_the_example_plays_its_notes_at_its_markers():
    import runpy

    e = runpy.run_path(str(ROOT / "examples" / "notes_overlay.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    m = e.document.timeline.markers
    problems: list[str] = []
    track, rate = e.audio_track(problems)
    assert rate == 8000 and problems == []
    st = e.state_at(0.0)
    for name in ("partials", "motif"):
        sem = e.inspect(0.0, id=name)["objects"][0]["semantic"]
        assert sem["count"] > 0 and "truncated" not in sem
        assert st[name].impl.audio_clip(st).marker in m
    quiet = e.fork()
    quiet.apply([{"op": "remove_object", "id": "partials"}, {"op": "remove_object", "id": "motif"}],
                origin="script")  # fmt: skip
    plain, _ = quiet.audio_track()
    differ = np.flatnonzero(track != plain)
    assert abs(differ[0] / rate - m["glide"]) < 0.01  # the notes come in at the marker


@pytest.mark.gl
@pytest.mark.slow
def test_example_golden_storyboard():
    import runpy

    from tests.conftest import gl_available
    from tests.test_beauty import _golden

    if not gl_available():
        pytest.skip("needs OpenGL")
    e = runpy.run_path(str(ROOT / "examples" / "notes_overlay.py"), run_name="x")["s"].engine
    m = e.document.timeline.markers
    times = [m["intro"] + 0.8, m["glide"] + 0.3, m["glide"] + 1.2, m["fade"] + 0.8]
    sheet = e.storyboard(times, size=(320, 180))["image"]
    _golden("notes_overlay", sheet, e.pass_runner().renderer_class())
