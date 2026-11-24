"""Mixes whose bytes are pinned (Milestone 23): the TTS-style whole-file narration, sound
clips at three rates and a recording cut into spans, built from fixed synthetic signals.
``tests/test_midi.py`` checks their sha256 against values taken on the commit before the
MIDI work, so adding the synthesizer to the mixer provably changed none of them."""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

from phasor.core.schema import Narration, NarrationBlock
from phasor.time import narration as nar


def _tone(n: int, rate: int, f: float, level: int = 9000) -> np.ndarray:
    """A decaying triangle wave in integer arithmetic only (the same bytes on every system)."""
    k = np.arange(n, dtype=np.int64)
    period = max(2, int(rate // f))
    tri = np.abs((k * 4 * level // period) % (4 * level) - 2 * level) - level
    return (tri * (n - k) // n).astype(np.int16)


def _wav(path: Path, samples: np.ndarray, rate: int) -> str:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())
    return path.name


def _digest(mixed) -> str:
    track, rate = mixed
    return hashlib.sha256(np.asarray(track, "<i2").tobytes() + str(rate).encode()).hexdigest()


def mixes(folder: Path) -> dict[str, str]:
    """name -> sha256 of the mixed track (and its rate)."""
    out = {}
    a = _wav(folder / "say_a.wav", _tone(11025, 22050, 220.0), 22050)
    b = _wav(folder / "say_b.wav", _tone(6000, 22050, 330.0), 22050)
    whole = Narration(script="A. B.", voice="file", blocks=(
        NarrationBlock(text="A.", start=0.2, duration=0.5, audio=a),
        NarrationBlock(text="B.", start=1.1, duration=0.3, audio=b)))  # fmt: skip
    out["tts_whole_files"] = _digest(nar.mix_track(whole, folder, 2.0))
    clips = []
    for rate, f in ((8000, 440.0), (22050, 550.0), (48000, 660.0)):
        name = _wav(folder / f"clip_{rate}.wav", _tone(rate // 2, rate, f), rate)
        clips.append(nar.Clip(path=name, start=0.1 + rate / 1e5, gain=0.0))  # 10**0: exact
    out["sound_clips_48k"] = _digest(nar.mix_track(None, folder, 2.0, clips))
    out["tts_and_clips"] = _digest(nar.mix_track(whole, folder, 2.0, clips))
    rec = _wav(folder / "rec.wav", _tone(16000 * 3, 16000, 200.0), 16000)
    cut = Narration(script="A. B.", voice="file:rec.wav", blocks=(
        NarrationBlock(text="A.", start=0.0, duration=1.0, audio=rec, audio_span=(0.0, 1.0)),
        NarrationBlock(text="B.", start=1.5, duration=1.2, audio=rec,
                       audio_span=(1.5, 2.7))))  # fmt: skip
    out["recording_spans"] = _digest(nar.mix_track(cut, folder, 3.0, clips[:1]))
    out["window"] = _digest(nar.mix_track(whole, folder, 2.0, clips, start=0.5, end=1.5))
    return out


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for k, v in mixes(Path(d)).items():
            print(f'    "{k}": "{v}",')
