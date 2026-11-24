"""Sound from files phasor does not read itself (Milestone 22).

A WAV is read as it is (``time.narration``: no decoder, no PyAV). Any other sound file a
scene names (the audio track of an ``.mp4``, ``.m4a``, ``.mp3``, ``.ogg``, ``.flac``, ...) is
decoded once with PyAV (the ``encode`` extra) into a 16-bit mono WAV in the user's cache
(``files.user_cache("audio")``, ``PHASOR_AUDIO_CACHE`` to move it), never into the scene's
folder. The cached file is named by everything it depends on: the file's resolved path,
modification time, size, a digest of its first and last 64 KiB, and the decoder's version; a
changed file is a different name, so it is decoded again (contract 3). An edit that keeps
a file's size and modification time and changes only its middle is not seen (the digest
covers the ends). The cache is pruned, oldest first, to ``MAX_CACHE_BYTES``; a file may not
decode to more than ``MAX_GROWTH`` times its size (and 256 MiB). The mixer
(``narration.mix_track``) and the narration builder read the cached WAV through
``readable_wav``, which the engine hands to the lower layers (``core``, ``style``, ``time``
and ``layout`` import no video library).

Hostile files: the file is opened non-blocking and checked to be a regular file before
PyAV sees it (a FIFO never blocks), files larger than ``MAX_DECODE_BYTES`` are refused,
decoding stops after ``MAX_DECODE_SECONDS`` of sound and ``MAX_DECODE_WALL`` seconds of work,
and whatever PyAV raises becomes a ``ValueError`` with the reason (lint ``bad_file``). A
decode that fails or is killed leaves no cached file (it is written under a temporary name,
then renamed).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import os
import stat as _stat
import threading
import time
import wave
from pathlib import Path

import numpy as np

from phasor.core import extras, files
from phasor.time import narration as nar
from phasor.time import speech as sp

WAV_EXTS = (".wav", ".wave")
DECODED_EXTS = (".mp4", ".m4a", ".m4v", ".mov", ".mp3", ".aac", ".ogg", ".oga", ".opus",
                ".flac", ".webm", ".mkv")  # fmt: skip
AUDIO_EXTS = WAV_EXTS + DECODED_EXTS
MAX_DECODE_BYTES = 4 << 30  # larger files are not decoded (a video's sound is its small part)
MAX_DECODE_SECONDS = nar.MAX_TRACK_SECONDS  # sound decoded at most (the track's own bound)
MAX_DECODE_WALL = 600.0  # seconds a decode may take before it is given up
DECODE_RATES = (8000, 48000)  # decoded sound is resampled into this range
_VERSION = 1  # of the decoder's output: a change makes every cached file new
_DIGEST = 1 << 16


def is_wav(name: str | Path) -> bool:
    return str(name).lower().endswith(WAV_EXTS)


def needs_decoder(name: str | Path) -> bool:
    """Whether reading ``name`` needs PyAV (any sound file that is not a WAV)."""
    return str(name).lower().endswith(DECODED_EXTS)


def decoder_available() -> bool:
    return extras.available("av")


def cache_dir() -> Path:
    return files.user_cache("audio", "PHASOR_AUDIO_CACHE")


def readable_wav(path: str | Path) -> Path:
    """A PCM WAV holding the sound of ``path`` (a regular file inside a scene's folder,
    already checked: ``files.regular_file``): the file itself when it is a WAV, else its
    decoded copy in the cache. ``ValueError`` with the reason when it cannot be."""
    path = Path(path)
    if is_wav(path):
        return path
    if not needs_decoder(path):
        raise ValueError(f"{path.name!r} is not a sound file phasor reads (WAV, or with the "
                         f"encode extra {', '.join(DECODED_EXTS)})")  # fmt: skip
    if not decoder_available():
        raise ValueError(f"{path.name!r} needs the 'av' package to be decoded (install the "
                         "encode extra: uv sync --extra encode); WAV needs nothing")  # fmt: skip
    try:
        st = path.stat()
    except OSError as e:
        raise ValueError(f"cannot be read ({e})") from None
    args = (str(path.resolve()), st.st_mtime_ns, st.st_size, str(cache_dir()))
    try:
        done = _decoded(*args)
        if not done.startswith("error:") and not Path(done).is_file():  # pruned meanwhile
            done = _decoded.__wrapped__(*args)
    except OSError as e:  # passing (and never cached): said, not raised as another type
        raise ValueError(str(e)) from None
    if done.startswith("error:"):
        raise ValueError(done[6:])
    return Path(done)


_LOCK = threading.Lock()


@functools.lru_cache(maxsize=64)
def _decoded(path: str, mtime_ns: int, size: int, cache: str) -> str:
    """The cached WAV's path, or ``"error:<why>"`` (cached too: a bad file is not decoded
    again until it changes). A passing trouble (``OSError``: the cache folder, another
    process holding a file) is raised, so it is never cached."""
    try:
        key = _key(Path(path), mtime_ns, size)
    except OSError:
        raise
    except Exception as e:
        return "error:" + (str(e) or type(e).__name__)
    folder = Path(cache)
    for name in (f"{key}.wav", f"{key}.cut.wav"):
        if (folder / name).is_file():
            return str(folder / name)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"the audio cache folder {folder} cannot be made ({e}); set "
                      "PHASOR_AUDIO_CACHE to a folder you can write") from None  # fmt: skip
    try:
        with _LOCK:
            out = _decode(Path(path), folder, key)
        _prune(folder, keep=out)
        return str(out)
    except OSError:
        raise
    except Exception as e:  # whatever a hostile file provokes: a reason, never a traceback
        return "error:" + (str(e) or type(e).__name__)


MAX_CACHE_BYTES = 4 << 30  # the audio cache is pruned, oldest first, down to this
MAX_GROWTH = 64  # a decoded file may be at most this many times its source (and 256 MiB)
_MIN_DECODED = 256 << 20


def _prune(folder: Path, keep: Path) -> None:
    """Delete the least recently made decoded files until the cache holds at most
    ``MAX_CACHE_BYTES`` (``keep`` stays); files in use elsewhere are skipped."""
    try:
        found = [(p.stat().st_mtime_ns, p.stat().st_size, p) for p in folder.glob("*.wav")]
    except OSError:
        return
    total = sum(size for _, size, _ in found)
    for _, size, p in sorted(found):
        if total <= MAX_CACHE_BYTES:
            break
        if p == keep:
            continue
        try:
            p.unlink()
            total -= size
        except OSError:  # open in another process (Windows): left for next time
            pass


def _key(path: Path, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    h.update(f"{path}|{mtime_ns}|{size}|{_VERSION}|{DECODE_RATES}".encode())
    with _open_regular(path) as f:
        h.update(f.read(_DIGEST))
        if size > _DIGEST:
            f.seek(max(_DIGEST, size - _DIGEST))
            h.update(f.read(_DIGEST))
    return h.hexdigest()[:32]


def _open_regular(path: Path):
    """``path`` opened for reading without ever blocking (a FIFO swapped in after the check
    is refused, not waited on)."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    f = os.fdopen(fd, "rb")
    if not _stat.S_ISREG(os.fstat(fd).st_mode):
        f.close()
        raise ValueError("it is not a regular file")
    return f


def _decode(path: Path, folder: Path, key: str) -> Path:
    """Decode ``path`` into ``folder`` as ``<key>.wav`` (``<key>.cut.wav`` when it was cut at
    ``MAX_DECODE_SECONDS``) and return that path."""
    import av

    tmp = folder / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with _open_regular(path) as f:
            size = os.fstat(f.fileno()).st_size
            if size > MAX_DECODE_BYTES:
                raise ValueError(f"it is larger than {MAX_DECODE_BYTES >> 30} GiB")
            cut = _decode_stream(av, f, tmp, max(_MIN_DECODED, MAX_GROWTH * size))
        out = folder / (f"{key}.cut.wav" if cut else f"{key}.wav")
        try:
            os.replace(tmp, out)
        except PermissionError:  # Windows: another process made and holds the same file
            if not out.is_file():
                raise
        return out
    except av.FFmpegError as e:  # PyAV's own errors (some are ValueError or OSError too):
        why = getattr(e, "strerror", None) or str(e)  # their reason, not an errno and a path
        raise ValueError(f"cannot be decoded: {why}") from None
    except (ValueError, OSError):
        raise
    except Exception as e:  # anything else a hostile file provokes
        raise ValueError(f"cannot be decoded ({type(e).__name__}: {e})") from None
    finally:
        if tmp.exists():
            tmp.unlink()


def _decode_stream(av, f, tmp: Path, max_bytes: int) -> bool:
    """Decode the first audio stream of ``f`` into a 16-bit mono WAV at ``tmp``; whether it
    was cut at ``MAX_DECODE_SECONDS``. ``ValueError`` past ``max_bytes`` of output (a tiny
    file that decodes to hours of silence) or ``MAX_DECODE_WALL`` seconds of work."""
    start = time.monotonic()
    with av.open(f, mode="r") as c:
        streams = c.streams.audio
        if not streams:
            raise ValueError("it has no audio stream")
        stream = streams[0]
        rate_in = int(stream.codec_context.sample_rate or stream.rate or 0)
        if rate_in <= 0:
            raise ValueError(f"its audio stream says a sample rate of {rate_in} Hz")
        rate = min(max(rate_in, DECODE_RATES[0]), DECODE_RATES[1])
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        limit = int(MAX_DECODE_SECONDS * rate)
        written = 0
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)

            def put(frames) -> None:
                nonlocal written
                for o in frames:
                    a = np.asarray(o.to_ndarray()).reshape(-1)[: limit - written]
                    w.writeframes(a.astype("<i2").tobytes())
                    written += a.size
                if 2 * written > max_bytes:
                    raise ValueError(f"it would decode to more than {max_bytes >> 20} MiB "
                                     f"({MAX_GROWTH} times its size): refused")  # fmt: skip

            for frame in c.decode(stream):
                put(resampler.resample(frame))
                if written >= limit:
                    break
                if time.monotonic() - start > MAX_DECODE_WALL:
                    raise ValueError(f"decoding took longer than {MAX_DECODE_WALL:g} s")
            if written < limit:
                put(resampler.resample(None))
        if written == 0:
            raise ValueError("its audio stream holds no sound")
    return written >= limit


# ------------------------------------------------------------------ what a file holds


@functools.lru_cache(maxsize=64)
def _info(path: str, mtime_ns: int, size: int, cache: str):
    try:
        return nar.wav_info(readable_wav(path))
    except Exception as e:  # a hostile file is lint bad_file, never a crash
        return str(e) or type(e).__name__


def info(base_dir: str | Path, name: str):
    """The ``WavInfo`` of the sound ``name`` (in the scene's folder) holds once read or
    decoded, the reason it cannot be mixed (a str), or None when the file is not there.
    Cached per file version."""
    where, mtime, size = files.stamp(base_dir, name)
    if mtime is None:
        return None
    return _info(where, mtime, size, str(cache_dir()))


@functools.lru_cache(maxsize=16)
def _speech(path: str, mtime_ns: int, size: int, cache: str):
    try:
        wav = readable_wav(path)
        head = nar.wav_info(wav)
        src = nar.WavSource(wav, head, int(sp.MAX_SECONDS * head.rate) + 1)
        heard = sp.detect(src.read, src.frames, head.rate)
        if wav.name.endswith(".cut.wav") and "cut" not in heard.flags:  # decoding stopped
            heard = dataclasses.replace(heard, flags=(*heard.flags, "cut"))
        return heard
    except Exception as e:
        return str(e) or type(e).__name__


def speech(base_dir: str | Path, name: str) -> sp.Speech:
    """Speech and pauses in the sound file ``name`` (``time.speech.detect``), cached per file
    version. ``ValueError`` when it is missing or cannot be read."""
    where, mtime, size = files.stamp(base_dir, name)
    if mtime is None:
        raise ValueError(files.problem(base_dir, name) or f"no file {name!r}")
    got = _speech(where, mtime, size, str(cache_dir()))
    if isinstance(got, str):
        raise ValueError(got)
    return got


@functools.lru_cache(maxsize=16)
def _aligned(path: str, mtime_ns: int, size: int, cache: str, script: str):
    from phasor.time.narration import parse_script

    heard = _speech(path, mtime_ns, size, cache)
    if isinstance(heard, str):
        return heard
    try:
        blocks = parse_script(script)
        before, pause = [], 0.0
        for b in blocks:
            if b.pause is not None:
                pause += b.pause
            else:
                before.append(pause)
                pause = 0.0
        spoken = [b for b in blocks if b.pause is None]
        return sp.align(heard, [sp.text_length(b.text) for b in spoken], before)
    except ValueError as e:
        return str(e)


def aligned(base_dir: str | Path, name: str, script: str) -> sp.Alignment:
    """The sentences of ``script`` found in the recording ``name`` (``time.speech.align`` on
    its cached detection), cached per file version and script: narrating and every later lint
    and snap read the same alignment, computed once. Do not change what it returns.
    ``ValueError`` when the file is missing or cannot be read or aligned."""
    where, mtime, size = files.stamp(base_dir, name)
    if mtime is None:
        raise ValueError(files.problem(base_dir, name) or f"no file {name!r}")
    got = _aligned(where, mtime, size, str(cache_dir()), script)
    if isinstance(got, str):
        raise ValueError(got)
    return got


def samples(base_dir: str | Path, name: str, span: tuple[float, float] | None = None
            ) -> tuple[np.ndarray, int]:  # fmt: skip
    """Mono float32 samples of the sound ``name`` (seconds ``span`` of it, else at most
    ``MAX_CLIP_FRAMES``) and the rate: for cutting and drawing, never the whole of a long
    recording."""
    wav = readable_wav(files.regular_file(base_dir, name))
    head = nar.wav_info(wav)
    if span is None:
        return nar.read_sound(wav)
    a = round(span[0] * head.rate)
    n = max(0, min(round((span[1] - span[0]) * head.rate), nar.MAX_CLIP_FRAMES))
    src = nar.WavSource(wav, head, int(sp.MAX_SECONDS * head.rate) + 1)
    return src.read(a, n), head.rate


# ------------------------------------------------------------------ where sentences start

MAX_BOUNDARIES = 20000


def narration_boundaries(
    narration, base_dir: str | Path, words_per_second: float, phrase_pause: float
) -> tuple[list[sp.Boundary], float, tuple[str, ...], dict | None]:
    """Where a marker may snap to on the narration: every spoken block's start (a sentence,
    or a phrase of a recording read without a script) and, inside blocks that have audio,
    where speech resumes after a pause of at least ``phrase_pause`` seconds. With how sure
    each is: a synthesized block's start is exact (1), an estimated one is a guess (0.5), a
    recording's comes from the alignment (``time.speech.align``, re-run on the cached
    detection). Also the detection's quality (the lowest of the files; 0 when a file could
    not be read), its flags (``unreadable`` then) and, for a recording read with its script,
    how well the text fitted the speech (``Alignment.to_dict``: rate, misfit, worst,
    doubtful), else None."""
    from phasor.time.narration import PHRASE, PRE_ROLL

    spoken = [(k, b) for k, b in enumerate(narration.blocks) if b.text]
    heard: dict[str, sp.Speech | None] = {}
    flags: set[str] = set()
    quality = 1.0
    for _, b in spoken:
        if b.audio and b.audio not in heard:
            try:
                heard[b.audio] = speech(base_dir, b.audio)
                quality = min(quality, heard[b.audio].quality)
                flags.update(heard[b.audio].flags)
            except ValueError:
                heard[b.audio] = None
                flags.add("unreadable")
    sure: dict[int, float] = {}
    fit: dict | None = None
    if narration.voice.startswith("file:") and narration.script.strip():
        try:
            found = aligned(base_dir, narration.voice[5:], narration.script)
            fit = found.to_dict()
            if len(found) == len(spoken):
                sure = {i: p.confidence for i, p in enumerate(found)}
            else:
                fit["doubtful"] = True  # the narration no longer matches its script
        except ValueError:
            flags.add("unreadable")
    if "unreadable" in flags:
        quality = 0.0
    out: list[sp.Boundary] = []
    for i, (k, b) in enumerate(spoken):
        phrase = b.text == PHRASE
        words = " ".join(b.text.split()[:5])
        label = (f"phrase {i + 1} starts" if phrase else
                 f"sentence {i + 1} ({words!r}{'...' if len(b.text.split()) > 5 else ''}) "
                 "starts")  # fmt: skip
        found = heard.get(b.audio) if b.audio else None
        if not b.audio:
            conf = 0.5  # timed from the text: a guess
        elif b.audio_span is None:
            conf = 1.0  # a synthesized clip per block: exact
        else:
            conf = sure.get(i, found.quality if found is not None else 0.0)
        out.append(sp.Boundary(round(b.start, 4), "phrase" if phrase else "sentence", label,
                               round(conf, 3), k))  # fmt: skip
        if found is None:
            continue
        a, z = b.audio_span if b.audio_span is not None else (0.0, found.seconds)
        prev = None
        for s0, s1 in found.segments:
            if prev is not None and a + PRE_ROLL < s0 < z and s0 - prev >= phrase_pause:
                t = b.start + s0 - PRE_ROLL - a
                if b.start < t < b.start + b.duration:
                    out.append(sp.Boundary(round(t, 4), "phrase", f"speech resumes after a "
                                           f"{s0 - prev:.2f} s pause in block {i + 1}",
                                           round(0.8 * found.quality, 3), k))  # fmt: skip
            prev = s1
        if len(out) > MAX_BOUNDARIES:
            break
    return out[:MAX_BOUNDARIES], round(quality, 3), tuple(sorted(flags)), fit
