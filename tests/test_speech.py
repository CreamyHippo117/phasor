"""Milestone 22: speech and pause detection, sentence alignment and the marker-snap plan
(``phasor.time.speech``): pure NumPy, tested on synthetic signals whose truth is known."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from phasor.time import speech as sp

RATE = 16000


def _word(rng, rate: int, dur: float, f0: float) -> np.ndarray:
    """A voiced, formant-shaped burst with syllable-rate loudness changes: speech-like."""
    t = np.arange(int(dur * rate)) / rate
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / rate
    x = np.zeros_like(t)
    for h in range(1, 25):
        if h * f0 > rate / 2 - 200:
            break
        amp = sum(np.exp(-(((h * f0 - f) / bw) ** 2))
                  for f, bw in ((700, 150), (1200, 200), (2600, 300))) + 0.05  # fmt: skip
        x += amp * np.sin(h * phase) / h**0.3
    x *= 0.55 - 0.45 * np.cos(2 * np.pi * t / dur * max(1, round(dur / 0.22)))
    x *= np.minimum(1, np.minimum(t, dur - t) / 0.02)
    return x / np.abs(x).max()


def recording(seed: int, sentences, rate: int = RATE, snr: float = 30.0,
              gap=(0.04, 0.3), pause=(0.35, 0.8), lead: float = 0.6):  # fmt: skip
    """Sentences of ``sentences[i]`` words: words with short (or comma-long) gaps, sentences
    with longer pauses, white noise ``snr`` dB under the speech. Returns the samples and
    where each sentence really starts."""
    rng = np.random.default_rng(seed)
    parts, t, starts = [np.zeros(int(lead * rate))], lead, []
    for words in sentences:
        starts.append(t)
        for i in range(words):
            parts.append(0.5 * _word(rng, rate, rng.uniform(0.18, 0.45), rng.uniform(100, 160)))
            t += len(parts[-1]) / rate
            if i < words - 1:
                parts.append(np.zeros(int(rng.uniform(*gap) * rate)))
                t += len(parts[-1]) / rate
        parts.append(np.zeros(int(rng.uniform(*pause) * rate)))
        t += len(parts[-1]) / rate
    x = np.concatenate(parts)
    level = np.sqrt(np.mean(x[x != 0] ** 2))
    x = x + rng.normal(0, level * 10 ** (-snr / 20), x.size)
    return x.astype(np.float32), starts


def bursts(rate: int, spans, total: float, freq: float = 300.0, amp: float = 0.4):
    """Tone bursts at known (start, end) seconds."""
    x = np.zeros(int(total * rate), np.float32)
    for a, b in spans:
        i, j = int(a * rate), int(b * rate)
        x[i:j] = amp * np.sin(2 * np.pi * freq * np.arange(j - i) / rate)
    return x


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_tone_bursts_with_gaps_are_found_where_they_are(rate):
    spans = [(0.5, 1.2), (1.6, 2.0), (2.9, 3.7)]
    s = sp.detect_samples(bursts(rate, spans, 4.5), rate)
    assert len(s.segments) == 3
    for (a, b), (x, y) in zip(spans, s.segments, strict=True):
        assert abs(a - x) <= 0.02 and abs(b - y) <= 0.02
    assert s.pauses[0] == (0.0, s.segments[0][0]) and s.quality > 0.9 and not s.unreliable


def test_short_pauses_are_closed_and_clicks_dropped():
    spans = [(0.5, 1.0), (1.08, 1.5), (2.0, 2.03), (2.5, 3.0)]  # an 80 ms gap; a 30 ms click
    s = sp.detect_samples(bursts(RATE, spans, 3.5), RATE)
    assert len(s.segments) == 2
    assert abs(s.segments[0][0] - 0.5) < 0.02 and abs(s.segments[0][1] - 1.5) < 0.02


def test_silence_and_constant_noise_find_no_speech_and_say_so():
    quiet = sp.detect_samples(np.zeros(RATE * 3, np.float32), RATE)
    assert quiet.segments == () and "silence" in quiet.flags and quiet.quality == 0.0
    noise = np.random.default_rng(0).normal(0, 0.1, RATE * 5).astype(np.float32)
    s = sp.detect_samples(noise, RATE)
    assert s.segments == () and "constant_noise" in s.flags and s.unreliable
    hum = 0.3 * np.sin(2 * np.pi * 50 * np.arange(RATE * 5) / RATE)
    assert sp.detect_samples(hum, RATE).segments == ()


def test_clipping_is_flagged_and_lowers_the_quality():
    x = bursts(RATE, [(0.5, 1.5), (2.0, 3.0)], 3.5, amp=3.0)
    s = sp.detect_samples(np.clip(x, -1.0, 1.0), RATE)
    assert len(s.segments) == 2 and "clipping" in s.flags and s.quality <= 0.7


def test_very_short_and_empty_clips_are_answered_not_raised():
    for n in (0, 1, 50, 159):
        s = sp.detect_samples(np.ones(n, np.float32) * 0.1, RATE)
        assert s.segments == () and "short" in s.flags
    s = sp.detect_samples(bursts(RATE, [(0.1, 0.3)], 0.4), RATE)
    assert "short" in s.flags and s.quality <= 0.5


def test_nan_and_inf_samples_do_not_poison_the_answer():
    x = bursts(RATE, [(0.5, 1.5)], 2.0)
    x[100:200] = np.nan
    x[5000] = np.inf
    s = sp.detect_samples(x, RATE)
    assert len(s.segments) == 1 and all(np.isfinite(v) for seg in s.segments for v in seg)


def test_stereo_is_mixed_down_and_int16_is_scaled():
    x = bursts(RATE, [(0.5, 1.5)], 2.0)
    stereo = np.stack([x, x * 0.5], axis=1)
    assert sp.detect_samples(stereo, RATE).segments == sp.detect_samples(x, RATE).segments
    ints = (x * 20000).astype(np.int16)
    assert sp.detect_samples(ints, RATE).segments == sp.detect_samples(x, RATE).segments


def test_a_bad_rate_is_a_clean_error():
    for rate in (0, -5, 10, 10**9):
        with pytest.raises(ValueError, match="sample rate"):
            sp.detect(lambda a, b: np.zeros(0, np.float32), 100, rate)


def test_music_under_speech_is_flagged_unreliable():
    x, _ = recording(4, [5, 6, 4, 7], snr=60)
    t = np.arange(x.size) / RATE
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6)) * 0.05
    s = sp.detect_samples(x + chord.astype(np.float32), RATE)
    assert "music" in s.flags and s.unreliable and s.quality < 0.5
    assert "music" not in sp.detect_samples(x, RATE).flags


def test_a_long_recording_is_read_a_chunk_at_a_time_and_bounded():
    asked = []
    rng = np.random.default_rng(0)
    one = bursts(8000, [(0.2, 0.9)], 1.5)

    def read(skip, count):
        asked.append(count)
        idx = np.arange(skip, skip + count) % one.size
        return one[idx] + rng.normal(0, 1e-3, count).astype(np.float32)

    frames = 3600 * 8000  # an hour
    t0 = time.perf_counter()
    s = sp.detect(read, frames, 8000)
    took = time.perf_counter() - t0
    assert max(asked) <= sp._CHUNK + 2048 and len(s.segments) == 2400
    assert took < 60, took
    asked.clear()
    s = sp.detect(read, int((sp.MAX_SECONDS + 100) * 8000), 8000)  # past the bound: cut
    assert "cut" in s.flags and s.seconds <= sp.MAX_SECONDS


def test_detection_is_deterministic():
    x, _ = recording(7, [4, 5, 6])
    assert sp.detect_samples(x, RATE) == sp.detect_samples(x.copy(), RATE)


def test_the_floor_follows_a_recording_whose_noise_changes():
    x = bursts(RATE, [(1, 2), (4, 5), (21, 22), (24, 25)], 26)
    rng = np.random.default_rng(1)
    x[: 13 * RATE] += rng.normal(0, 0.002, 13 * RATE).astype(np.float32)
    x[13 * RATE :] += rng.normal(0, 0.03, x.size - 13 * RATE).astype(np.float32)
    s = sp.detect_samples(x, RATE)
    found = [seg for seg in s.segments if not 13.0 <= seg[0] < 14.5]
    assert [round(a) for a, _ in found] == [1, 4, 21, 24]
    # the louder room is followed within 1.5 s of the change (a lag, then pauses again)
    assert all(b <= 14.5 for a, b in s.segments if 13.0 <= a < 14.5)


# ------------------------------------------------------------------ sentences


def _errors_ms(seed, snr, sentences=None):
    rng = np.random.default_rng(seed)
    sentences = sentences or [int(rng.integers(3, 12)) for _ in range(12)]
    x, starts = recording(seed, sentences, snr=snr)
    placed = sp.align(sp.detect_samples(x, RATE), [float(w) for w in sentences])
    return np.array([(p.start - t) * 1000 for p, t in zip(placed, starts, strict=True)]), placed


@pytest.mark.parametrize(("snr", "median_ms", "max_ms"), [(40, 10, 20), (20, 25, 40), (10, 45, 70)])
def test_sentences_align_to_where_they_start(snr, median_ms, max_ms):
    """Comma pauses (up to 0.3 s) overlap sentence pauses (0.35 s on): the alignment uses
    the words of each sentence, not the longest pauses or equal spacing."""
    for seed in (1, 2, 3):
        err, placed = _errors_ms(seed, snr)
        assert np.median(np.abs(err)) <= median_ms and np.abs(err).max() <= max_ms, err
        assert all(0 < p.confidence <= 1 for p in placed)
        assert all(a.end <= b.start for a, b in itertools.pairwise(placed))


def test_a_sentence_run_into_the_next_is_cut_at_a_quiet_moment():
    """Fewer pauses than sentences: the longest stretch is split, and it says it is unsure."""
    x = bursts(RATE, [(0.5, 2.5), (3.0, 4.0)], 4.5)
    x[int(1.5 * RATE) : int(1.53 * RATE)] *= 0.01  # a dip, not a pause
    placed = sp.align(sp.detect_samples(x, RATE), [1.0, 1.0, 1.0])
    assert len(placed) == 3 and abs(placed[1].start - 1.5) < 0.05
    assert placed[1].confidence < placed[0].confidence


def test_alignment_refuses_what_it_cannot_do():
    with pytest.raises(ValueError, match="no speech"):
        sp.align(sp.detect_samples(np.zeros(RATE, np.float32), RATE), [1.0])
    s = sp.detect_samples(bursts(RATE, [(0.5, 1.5)], 2), RATE)
    with pytest.raises(ValueError, match="at most"):
        sp.align(s, [1.0] * (sp.MAX_SENTENCES + 1))
    assert sp.align(s, []) == []


def test_phrases_and_pieces():
    s = sp.detect_samples(bursts(RATE, [(0.5, 1.0), (1.2, 1.6), (2.4, 3.0)], 3.5), RATE)
    ph = sp.phrases(s, 0.35)
    assert len(ph) == 2 and abs(ph[0].end - 1.6) < 0.02
    p = sp.Placed(1.0, 3.0, 1.0, onsets=(2.05,))
    starts = sp.pieces(p, [1.0, 1.0])
    assert starts == [1.0, 2.05]  # the break moved onto the onset near the middle
    assert sp.pieces(sp.Placed(1.0, 3.0, 1.0), [1.0, 3.0]) == [1.0, 1.5]


# ------------------------------------------------------------------ snap plan


def _b(t, kind="sentence", i=None):
    return sp.Boundary(t, kind, f"sentence at {t}", 1.0, i)


def _plan(hand, bounds, owned=None, **kw):
    kw.setdefault("visual_end", 25.0)
    kw.setdefault("narration_end", 30.0)
    kw.setdefault("reach", 0.1)
    kw.setdefault("need", {})
    return {m.marker: m for m in sp.plan_snap(hand, owned or {}, bounds, **kw)}


def test_markers_snap_to_sentence_starts_by_their_place_in_the_scene():
    hand = {"intro": 0.0, "poles": 5.0, "loop": 12.0, "residue": 20.0}
    bounds = [_b(0.3), _b(3.0), _b(6.1), _b(9.0), _b(14.8), _b(18.0), _b(24.0)]
    m = _plan(hand, bounds)
    assert [m[k].t1 for k in hand] == [0.3, 6.1, 14.8, 24.0]
    assert all(m[k].kind == "sentence" and m[k].confidence > 0.5 for k in hand)


def test_unmatched_markers_keep_their_place_and_order_between_anchors():
    hand = {"a": 2.0, "mid": 4.0, "b": 6.0}
    m = _plan(hand, [_b(2.4), _b(7.2)], visual_end=10, narration_end=12)
    assert (m["a"].t1, m["b"].t1) == (2.4, 7.2)
    assert m["mid"].kind == "between" and m["mid"].t1 == pytest.approx(4.8)


def test_owned_markers_never_move_and_bound_the_others():
    m = _plan({"h": 5.0}, [_b(5.5)], owned={"w": 5.2}, visual_end=10, narration_end=10)
    assert m["w"].kind == "owned" and m["w"].t1 == 5.2
    assert m["h"].t1 == 5.0  # 5.5 would cross the owned marker at 5.2


def test_a_span_never_becomes_shorter_than_its_motions_need():
    hand = {"a": 1.0, "b": 2.5}
    bounds = [_b(1.0), _b(1.6)]
    m = _plan(hand, bounds, need={"a": 1.4}, visual_end=5, narration_end=5, reach=0.2)
    assert m["b"].t1 - m["a"].t1 >= 1.4 - 1e-9
    m = _plan(hand, bounds, need={"a": 1.4}, visual_end=5, narration_end=5, reach=0.2,
              allow_short=True)  # fmt: skip
    assert m["b"].t1 == 1.6


def test_snapping_twice_changes_nothing():
    hand = {"intro": 0.0, "poles": 5.0, "x": 8.0, "loop": 12.0, "residue": 20.0}
    bounds = [_b(0.3), _b(3.0), _b(6.1), _b(9.0), _b(14.8), _b(18.0), _b(24.0),
              _b(11.0, "phrase")]  # fmt: skip
    first = _plan(hand, bounds, need={k: 0.6 for k in hand})
    again = _plan({k: v.t1 for k, v in first.items()}, bounds, need={k: 0.6 for k in hand},
                  visual_end=first["residue"].t1 + 5)  # fmt: skip
    assert all(not v.moves for v in again.values()), [v for v in again.values() if v.moves]


def test_a_pinned_marker_goes_where_it_is_told():
    hand = {"a": 1.0, "b": 9.0}
    m = _plan(hand, [_b(1.1), _b(9.3)], anchors={"b": _b(5.0, i=2)}, visual_end=10,
              narration_end=10)  # fmt: skip
    assert m["b"].t1 == 5.0 and m["b"].why.startswith("pinned") and m["a"].t1 == 1.1


def test_nothing_to_snap_to_keeps_everything():
    m = _plan({"a": 1.0, "b": 2.0}, [])
    assert all(v.kind == "kept" and not v.moves for v in m.values())
