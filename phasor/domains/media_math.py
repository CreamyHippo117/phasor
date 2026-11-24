"""Pictures, picture sequences and video clips as frames by index (Milestone 24). No drawing.

One frame source serves ``Image`` (a still, or a sequence ``frames/shot_%04d.png``) and
``Video`` (an mp4, mov, webm or mkv clip, through PyAV, the ``encode`` extra):

- ``open_source(base_dir, path, ...)`` names the source by its file stamp
  (``core/files.py``); ``Source.probe()`` says what it is (size, the time of every frame,
  or why it cannot be shown) without decoding more than its first frame, cached by stamp;
  ``Source.frame(i)`` gives frame ``i``, cached by (stamp, i) in a cache bounded by bytes.
- ``clip_time`` maps scene time to clip time (``start``, ``trim``, ``speed``, ``loop``) and
  ``frame_index`` picks the frame shown: the last one whose presentation time is at or
  before it. A pure function of (document, t, file): the same t gives the same frame in any
  order, in any process.
- ``fit_box`` places a picture in its box (``contain``, ``cover``, ``stretch``, ``none``);
  ``sample`` turns the visible part of a frame into linear RGBA pixels.

Colour: files are sRGB 8-bit with straight alpha. A still is converted to linear float when
read (as ``Image`` always did); sequence and video frames are kept as 8-bit and converted to
linear only for the part a draw samples, through a lookup table: once either way. Pictures
with transparency are sampled premultiplied (no dark fringes), then laid over the
background.

Decoding seeks to the keyframe whose time is at or before the wanted frame (from the
probe's list of keyframes, so an open GOP's leading frames decode as in a decode from the
start) and decodes forward. Cursors are kept open (``PER_FILE`` per file, ``MAX_OPEN`` in
all, per process, each behind a lock) and a frame goes to the cursor just behind it, so
clips played in order decode each frame once and never seek. Only an exact frame is taken
from a cursor: the same frame comes out whichever way the decoder got there. Every limit is a constant here: a file past one is
``bad_file`` (not a picture or video, or an absurd size) or ``video_too_big`` (past a cap)
and draws nothing; it never raises, hangs or fills memory.
"""

from __future__ import annotations

import contextlib
import os
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core import files
from phasor.core.cache import ByteCache
from phasor.core.color import srgb_to_linear

# ---------------------------------------------------------------- limits

MAX_SOURCE_PX = 8192  # a still larger than this is scaled down once, when read (as before)
STORE_SIDE = 4096  # sequence and video frames larger than this are scaled down when decoded
MAX_SIDE = 16384  # wider or taller than this is absurd: bad_file
MAX_FRAME_PIXELS = 7680 * 4320  # a frame past 8K: video_too_big
MAX_FRAMES = 216_000  # frames in a clip (an hour at 60 fps): more is video_too_big
MAX_SECONDS = 4 * 3600.0  # a clip longer than this: video_too_big
MAX_FPS = 1000.0  # a sequence's ``fps``; a clip whose frames come faster is bad_file
MAX_SPEED = 64.0  # ``speed``: a clip plays at most this many times faster
MAX_START = 1e6  # seconds: ``start`` and the ends of ``trim``
MAX_VIDEO_BYTES = 64 * 2**30  # a larger file is not opened: video_too_big
FIT = ("stretch", "contain", "cover", "none")
VIDEO_EXT, PICTURE_EXT = files.VIDEO_EXT, files.PICTURE_EXT
NATIVE_ROWS = 1080  # fit "none": a frame 1080 pixels tall fills the camera's height
EPS = 1e-6  # seconds: a time this close before a frame already shows it

FRAME_CACHE_BYTES = 256 * 2**20  # decoded frames kept, per process
FRAME_CACHE_ENTRIES = 256
PROBE_CACHE_BYTES = 64 * 2**20
MAX_OPEN = 16  # decoders (cursors in video files) kept open, per process
PER_FILE = 4  # cursors in one video file (a clip shown at several times at once)
BAND_ROWS = 256  # a frame is sampled this many output rows at a time
FORWARD = 90  # decode forward instead of seeking when the frame is at most this far ahead


def picture_pixels_cap() -> int:
    """The largest picture read: Pillow's own bomb limit (twice ``MAX_IMAGE_PIXELS``, where
    it refuses), so every picture that read before still does."""
    from PIL import Image as PILImage

    cap = PILImage.MAX_IMAGE_PIXELS
    return int(2 * cap) if cap else 2 * 89_478_485


# sRGB byte -> linear float32, exactly as ``srgb_to_linear(byte / 255)`` in float32
_LUT = np.asarray(srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0), dtype=np.float32)


# ---------------------------------------------------------------- what a source is


@dataclass(frozen=True)
class Probe:
    """What a source holds: its size, the time of each frame (seconds from the first, rising)
    and how long one frame lasts; or why it cannot be shown. ``problem``: lint ``bad_file``;
    ``too_big``: lint ``video_too_big``; ``needs``: an optional package that is not
    installed (lint ``missing_extra``). Any of them: nothing is drawn."""

    width: int = 0
    height: int = 0
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frame_seconds: float = 0.0
    problem: str = ""
    too_big: str = ""
    needs: str = ""
    opaque: bool = True
    pts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # video
    # video: (pts, seek time) of each keyframe, by pts: where decoding a frame starts
    keys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    store: tuple[int, int] = (0, 0)  # video: the size frames are kept at

    @property
    def frames(self) -> int:
        return len(self.times)

    @property
    def seconds(self) -> float:
        """How long the clip plays: the last frame's time plus one frame."""
        return float(self.times[-1]) + self.frame_seconds if self.frames else 0.0

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_seconds if self.frame_seconds > 0 else 0.0

    @property
    def ok(self) -> bool:
        return not (self.problem or self.too_big or self.needs) and self.frames > 0


class Source:
    """A picture, sequence or clip, named by its stamp (``files.stamp``): the same stamp is
    the same pictures. ``missing``: nothing is there (lint ``missing_file``)."""

    kind = "none"

    def __init__(self, stamp: tuple) -> None:
        self.stamp = stamp

    @property
    def missing(self) -> bool:
        return self.stamp[1] is None

    def probe(self) -> Probe:
        if self.missing:
            return Probe(problem="missing")
        return _PROBES.get((self.kind, self.stamp, *self._probe_key()), self._safe_probe)

    def frame(self, index: int) -> np.ndarray | None:
        """Frame ``index`` (clamped to the frames there are): (h, w, 4), row 0 = top,
        read-only. A still is linear float32, premultiplied; sequence and video frames are
        sRGB uint8 with straight alpha. None when it cannot be read."""
        probe = self.probe()
        if not probe.ok:
            return None
        i = min(max(int(index), 0), probe.frames - 1)
        key = (self.kind, self.stamp, i)
        hit = _FRAMES.peek(key)
        if hit is not None:
            return hit
        try:
            img = self._read(i, probe)
        except Exception:  # a frame a hostile file breaks draws nothing, never a crash
            img = None
        if img is not None:
            img.flags.writeable = False
            _FRAMES.put(key, img)
        return img

    def _probe_key(self) -> tuple:
        return ()

    def _safe_probe(self) -> Probe:
        try:
            return self._probe()
        except MemoryError:
            return Probe(too_big="too large to read in memory")
        except Exception as e:  # a hostile file is lint bad_file, never a crash
            return Probe(problem=_why(e))

    def _probe(self) -> Probe:
        raise NotImplementedError

    def _read(self, index: int, probe: Probe) -> np.ndarray | None:
        raise NotImplementedError


def _why(e: BaseException) -> str:
    """Why, in one short line (an FFmpeg error says only its message)."""
    text = getattr(e, "strerror", None) or str(e).strip() or type(e).__name__
    return str(text).splitlines()[0][:200]


_PROBES = ByteCache(lambda: PROBE_CACHE_BYTES, 1024)
_FRAMES = ByteCache(lambda: FRAME_CACHE_BYTES, FRAME_CACHE_ENTRIES)


# ---------------------------------------------------------------- pictures (Pillow)


def _open_picture(path: Path, stamp: tuple | None, max_pixels: int):
    """The picture at ``path`` opened lazily by Pillow (the header only), from a file opened
    as a regular file; ``ValueError`` when its size is absurd or past ``max_pixels``."""
    from PIL import Image as PILImage

    expect = stamp[1:] if stamp is not None and isinstance(stamp[1], int) else None
    f = files.open_regular(path, expect)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)
            im = PILImage.open(f)
        w, h = im.size
        if w <= 0 or h <= 0:
            raise ValueError(f"the picture claims an absurd size ({w} x {h} pixels)")
        if w * h > max_pixels:
            raise PictureTooBig(f"{w} x {h} pixels is more than {max_pixels:,} pixels")
    except BaseException:
        f.close()
        raise
    return im, f


class PictureTooBig(ValueError):
    pass


def _decode_picture(path: Path, stamp: tuple | None, max_pixels: int, side: int):
    """The picture as RGBA uint8 (h, w, 4), scaled down to ``side`` pixels at most."""
    from PIL import Image as PILImage

    im, f = _open_picture(path, stamp, max_pixels)
    with f, im:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)
            im.seek(0)  # a GIF or an animated WebP: its first frame
            rgba = im.convert("RGBA")
        if max(rgba.size) > side:
            rgba.thumbnail((side, side))
        return np.asarray(rgba, dtype=np.uint8)


class StillSource(Source):
    """One picture (``Image`` with a single file): frame 0 is it, as linear float32
    premultiplied by its alpha (exactly what ``Image`` drew before Milestone 24)."""

    kind = "still"

    def __init__(self, stamp: tuple, path: Path) -> None:
        super().__init__(stamp)
        self.path = path

    def _probe(self) -> Probe:
        try:
            img = self._linear()
        except PictureTooBig as e:  # a still: bad_file (video_too_big is for clips)
            return Probe(problem=f"the picture is {e}")
        key = (self.kind, self.stamp, 0)
        img.flags.writeable = False
        _FRAMES.put(key, img)
        h, w = img.shape[:2]
        opaque = bool(img[..., 3].min() >= 1.0)
        return Probe(width=w, height=h, times=np.zeros(1), frame_seconds=1.0, opaque=opaque)

    def _linear(self) -> np.ndarray:
        a = _decode_picture(self.path, self.stamp, picture_pixels_cap(), MAX_SOURCE_PX)
        a = a.astype(np.float32) / 255.0
        rgb = np.asarray(srgb_to_linear(a[..., :3]), dtype=np.float32)
        return np.concatenate([rgb * a[..., 3:], a[..., 3:]], axis=-1)

    def _read(self, index: int, probe: Probe) -> np.ndarray | None:
        return self._linear()


class SequenceSource(Source):
    """Pictures ``frames/shot_%04d.png`` played at ``fps``: frame i is the i-th file. Every
    frame is read when shown (each a regular file inside the folder, the stamp covering all of
    them); one that cannot be read draws nothing. Frames of another size than the first are
    stretched to its box."""

    kind = "sequence"

    def __init__(self, stamp: tuple, frames: tuple[Path, ...], fps: float) -> None:
        super().__init__(stamp)
        self.paths = frames
        self.fps = fps

    def _probe_key(self) -> tuple:
        return (self.fps,)

    def _probe(self) -> Probe:
        if not (np.isfinite(self.fps) and 0 < self.fps <= MAX_FPS):
            return Probe(problem=f"fps {self.fps!r} is not between 0 and {MAX_FPS:g}")
        if len(self.paths) > files.MAX_SEQUENCE:
            return Probe(too_big=f"more than {files.MAX_SEQUENCE} frames")
        try:
            first = _decode_picture(self.paths[0], None, MAX_FRAME_PIXELS, STORE_SIDE)
        except PictureTooBig as e:
            return Probe(too_big=f"frame 0 is {e}")
        h, w = first.shape[:2]
        first.flags.writeable = False
        _FRAMES.put((self.kind, self.stamp, 0), first)
        n = len(self.paths)
        return Probe(width=w, height=h, times=np.arange(n) / self.fps,
                     frame_seconds=1.0 / self.fps, opaque=False)  # fmt: skip

    def _read(self, index: int, probe: Probe) -> np.ndarray | None:
        return _decode_picture(self.paths[index], None, MAX_FRAME_PIXELS, STORE_SIDE)


# ---------------------------------------------------------------- video (PyAV)


class VideoSource(Source):
    """A clip read by PyAV: frame i is the i-th by presentation time (every packet's time,
    read when probed, without decoding). Frames come out as sRGB RGBA uint8."""

    kind = "video"

    def __init__(self, stamp: tuple, path: Path) -> None:
        super().__init__(stamp)
        self.path = path

    def _probe(self) -> Probe:
        try:
            import av  # noqa: F401
        except ImportError:
            return Probe(needs="av")
        if self.stamp[2] > MAX_VIDEO_BYTES:
            return Probe(too_big=f"the file is larger than {MAX_VIDEO_BYTES // 2**30} GB")
        with _Container(self.path, self.stamp) as box:
            s = box.stream
            w, h = int(s.codec_context.width or 0), int(s.codec_context.height or 0)
            if w <= 0 or h <= 0:
                return Probe(problem="its frames have no size")
            if w > MAX_SIDE or h > MAX_SIDE:
                return Probe(problem=f"it claims an absurd frame size ({w} x {h} pixels)")
            if w * h > MAX_FRAME_PIXELS:
                return Probe(too_big=f"its frames are {w} x {h} pixels, past 8K "
                             f"({MAX_FRAME_PIXELS:,} pixels)")  # fmt: skip
            pts, keys = [], []
            for packet in box.container.demux(s):
                t = packet.pts if packet.pts is not None else packet.dts
                if t is None or packet.size == 0:
                    continue
                if packet.is_keyframe:  # where decoding can start (discarded ones too)
                    d = packet.dts if packet.dts is not None else t
                    keys.append((int(t), int(min(t, d))))
                if packet.is_discard:  # cut by an edit list: never shown
                    continue
                pts.append(int(t))
                if len(pts) > MAX_FRAMES:
                    return Probe(too_big=f"more than {MAX_FRAMES:,} frames")
            if not pts:
                return Probe(problem="it has no frames")
            pts = np.unique(np.asarray(pts, dtype=np.int64))  # the first shown is time 0
            keys = np.unique(np.asarray(keys, dtype=np.int64).reshape(-1, 2), axis=0)
            base = Fraction(s.time_base) if s.time_base else None
            if not base or base <= 0:
                return Probe(problem="its frames have no time base")
            times = (pts - pts[0]).astype(np.float64) * base.numerator / base.denominator
            if not np.all(np.isfinite(times)) or times[-1] > MAX_SECONDS:
                return Probe(too_big=f"it is longer than {MAX_SECONDS / 3600:g} hours")
            step = _frame_seconds(times, s)
            if not (np.isfinite(step) and step >= 1.0 / MAX_FPS):
                return Probe(problem=f"its frames come faster than {MAX_FPS:g} per second")
            store = _store_size(w, h)
            try:
                first = box.decode_at(int(pts[0]), store, keys)
            except Exception as e:
                return Probe(problem=f"its first frame cannot be decoded ({_why(e)})")
            if first is None:
                return Probe(problem="its first frame cannot be decoded")
        first.flags.writeable = False
        _FRAMES.put((self.kind, self.stamp, 0), first)
        times.flags.writeable = pts.flags.writeable = keys.flags.writeable = False
        return Probe(width=w, height=h, times=times, frame_seconds=step, opaque=True, pts=pts,
                     store=store, keys=keys)  # fmt: skip

    def _read(self, index: int, probe: Probe) -> np.ndarray | None:
        for _ in range(3):  # a decoder another thread closed meanwhile: take another
            out = _decoder(self.path, self.stamp, index, probe.pts).frame(index, probe)
            if out is not _CLOSED:
                return out
        with _Container(self.path, self.stamp) as box:  # still contended: one of its own
            return box.decode_at(int(probe.pts[index]), probe.store, probe.keys)


def _frame_seconds(times: np.ndarray, stream: Any) -> float:
    """How long one frame lasts: the median gap between frames, or the stream's rate for a
    single frame (1/25 s when it gives none)."""
    if len(times) > 1:
        return float(np.median(np.diff(times)))
    rate = stream.average_rate or stream.guessed_rate
    return float(1 / Fraction(rate)) if rate else 1 / 25


def _store_size(w: int, h: int) -> tuple[int, int]:
    """The size a frame is kept at: at most ``STORE_SIDE`` on its longer side, even."""
    if max(w, h) <= STORE_SIDE:
        return w, h
    s = STORE_SIDE / max(w, h)
    return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)


class _Container:
    """A video file opened for PyAV from a regular file this process opened itself (PyAV
    never gets a path: a FIFO swapped in after the check would block it), checked against
    its stamp. Frames are converted by its own single-threaded scaler (swscale's thread pool
    deadlocks in a process forked after the parent used it)."""

    def __init__(self, path: Path, stamp: tuple) -> None:
        import av
        from av.video.reformatter import VideoReformatter

        self.file = files.open_regular(path, stamp[1:], nofollow=True)
        try:
            try:
                self.container = av.open(self.file, mode="r")
            except Exception as e:  # FFmpeg's error, without its errno and file number
                raise ValueError(f"it is not a video PyAV can read ({_why(e)})") from None
            streams = self.container.streams.video
            if not streams:
                raise ValueError("it has no video stream")
            self.stream = streams[0]
        except BaseException:
            self.close()
            raise
        self.scaler = VideoReformatter()
        self.it = None  # the decode iterator, positioned after ``last``
        self.last: int | None = None  # pts of the frame decoded last

    def __enter__(self) -> _Container:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        c = getattr(self, "container", None)
        if c is not None:
            with contextlib.suppress(Exception):
                c.close()
            self.container = None
        with contextlib.suppress(Exception):
            self.file.close()

    def decode_at(self, target: int, size: tuple[int, int], keys: np.ndarray,
                  prev_pts: int | None = None):  # fmt: skip
        """The frame whose time is ``target`` (pts) as sRGB RGBA uint8 at ``size``.

        Decoded on from where the decoder is when it is just behind (``prev_pts``: the frame
        before it was the last decoded), else from the keyframe (``keys``) with the latest
        time at or before it. A frame shown before its keyframe (an open GOP's leading
        B-frames: decoded after the keyframe but referring to the GOP before it) decodes
        there, as in a decode from the start, whatever the demuxer seeks by. If that does
        not give the frame, from the keyframe before. Only an exact hit is taken from the
        first two ways, so the frame does not depend on how the decoder got there; when no
        way gives it exactly (a damaged clip), the last frame before it from the earliest
        start, else None."""
        ahead = (self.it is not None and self.last is not None and prev_pts is not None
                 and self.last <= prev_pts < target)  # fmt: skip
        if ahead:
            hit = self._decode_to(target)
            if hit is not None and hit[0] == target:
                return self._rgba(hit[1], size)
        best = None
        for start in _seek_points(keys, target):
            self.container.seek(start, stream=self.stream, backward=True, any_frame=False)
            self.it = self.container.decode(self.stream)
            self.last = None
            best = self._decode_to(target)
            if best is not None and best[0] == target:
                break
        return None if best is None else self._rgba(best[1], size)

    def _decode_to(self, target: int):
        """Decode on to ``target``: (pts, frame) of it, or of the last frame before it when
        the decode passes it (the decoder is then no longer just behind anything)."""
        best = None
        for frame in self.it:
            p = frame.pts
            if p is None:
                continue
            if p > target:
                self.it, self.last = None, None
                return best
            self.last = p
            if best is None or p >= best[0]:
                best = (p, frame)
            if p == target:
                return best
        self.it, self.last = None, None
        return best

    def _rgba(self, frame: Any, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        return self.scaler.reformat(frame, width=w, height=h, format="rgba",
                                    threads=1).to_ndarray()  # fmt: skip


def _seek_points(keys: np.ndarray, target: int) -> list[int]:
    """Where to seek for the frame at ``target``: the keyframe with the latest time at or
    before it, then the one before that (its seek time: the smaller of its pts and dts, so
    a demuxer that seeks by either lands on it or earlier). Just ``target`` without keys."""
    if len(keys) == 0:
        return [int(target)]
    j = int(np.searchsorted(keys[:, 0], target, side="right")) - 1
    return [int(keys[i, 1]) for i in dict.fromkeys((max(j, 0), max(j - 1, 0)))]


_CLOSED = object()  # what a decoder closed by another thread gives: ask for another


class _Decoder:
    """A cursor in a file kept open for decoding, with a lock. A file has up to
    ``PER_FILE`` of them per process (one per place it is played from)."""

    def __init__(self, path: Path, stamp: tuple) -> None:
        self.box = _Container(path, stamp)
        self.lock = threading.Lock()

    def behind(self, i: int, pts: np.ndarray) -> int | None:
        """How many frames this cursor is behind frame ``i`` (1 to ``FORWARD``), or None."""
        last = self.box.last
        if last is None:
            return None
        j = int(np.searchsorted(pts, last))
        if j < len(pts) and pts[j] == last and j < i <= j + FORWARD:
            return i - j
        return None

    def frame(self, index: int, probe: Probe):
        pts = probe.pts
        i = min(max(index, 0), len(pts) - 1)
        with self.lock:
            if self.box.container is None:
                return _CLOSED
            prev = int(pts[i - 1]) if self.behind(i, pts) else None  # decode on, no seek
            return self.box.decode_at(int(pts[i]), probe.store, probe.keys, prev)

    def close(self) -> None:
        with self.lock:
            self.box.close()


_DECODERS: OrderedDict[tuple, _Decoder] = OrderedDict()  # (stamp, n) -> cursor
_DECODERS_LOCK = threading.Lock()
_PID = os.getpid()


def _decoder(path: Path, stamp: tuple, index: int, pts: np.ndarray) -> _Decoder:
    """A cursor in this file version for frame ``index``, in this process (a render worker
    opens its own; one inherited across a fork is dropped, never shared): the one closest
    behind it (it decodes on); else a new one while the file has fewer than ``PER_FILE``;
    else its least recently used. At most ``MAX_OPEN`` in all: the least recently used is
    closed (on Windows an open file cannot be replaced), as is any of an older version."""
    global _PID
    with _DECODERS_LOCK:
        if os.getpid() != _PID:  # a forked child: the parent's handles are not ours
            _DECODERS.clear()
            _PID = os.getpid()
        for old in [k for k in _DECODERS if k[0][0] == stamp[0] and k[0] != stamp]:
            _DECODERS.pop(old).close()  # an older version of it
        mine = [k for k in _DECODERS if k[0] == stamp]
        near = [(gap, k) for k in mine if (gap := _DECODERS[k].behind(index, pts))]
        if near:
            key = min(near)[1]
        elif len(mine) >= PER_FILE:
            key = mine[0]  # least recently used first
        else:
            n = next(n for n in range(PER_FILE) if (stamp, n) not in _DECODERS)
            key = (stamp, n)
            _DECODERS[key] = _Decoder(path, stamp)
            while len(_DECODERS) > MAX_OPEN:
                _DECODERS.popitem(last=False)[1].close()
        _DECODERS.move_to_end(key)
        return _DECODERS[key]


def close_all() -> None:
    """Close every open video file and forget every decoded frame and probe (tests; a file
    about to be replaced on Windows)."""
    with _DECODERS_LOCK:
        while _DECODERS:
            _DECODERS.popitem()[1].close()
    for cache in (_FRAMES, _PROBES):
        with cache.lock:
            cache.entries.clear()
            cache.used = 0


# ---------------------------------------------------------------- naming a source


def open_source(base_dir: str | Path, value: str, *, video: bool = False,
                fps: float = 24.0) -> Source:  # fmt: skip
    """The source ``value`` names in the scene's folder: a clip (``video``), a picture
    sequence (``files.is_sequence``) played at ``fps``, or one picture. A source with
    nothing there is ``missing``."""
    stamp = files.stamp(base_dir, value)
    if stamp[1] is None:
        return Source(stamp)
    if files.is_sequence(value):
        return SequenceSource(stamp, files.sequence(base_dir, value), float(fps))
    path = Path(stamp[0])
    return VideoSource(stamp, path) if video else StillSource(stamp, path)


# ---------------------------------------------------------------- time


def clip_time(t: float, *, start: float, trim: tuple[float, float], speed: float, loop: bool,
              seconds: float) -> tuple[float, bool]:  # fmt: skip
    """Seconds into the clip shown at scene time ``t``, and whether the clip has ended (it
    holds its last frame: ``loop`` off). The clip begins at ``start``; ``trim`` = (in, out)
    seconds inside it (out 0: its end); ``speed`` scales time. Before ``start`` it holds its
    first frame. Anything not finite holds the first frame."""
    vals = (t, start, trim[0], trim[1], speed, seconds)
    if not all(np.isfinite(v) for v in vals) or speed <= 0 or seconds <= 0:
        return max(0.0, float(trim[0])) if np.isfinite(trim[0]) else 0.0, False
    a, b = clip_range(trim, seconds)
    local = (t - start) * speed
    span = b - a
    if local <= 0 or span <= 0:
        return a, False
    if loop:
        return a + float(np.fmod(local, span)), False
    # at its very end it has just finished, not "ended" (within EPS of scene time: times
    # rounded to the microsecond, as lint samples them, are scaled by ``speed`` here)
    if local > span + EPS * max(1.0, speed):
        return b, True
    return min(a + local, b), False


def clip_range(trim: tuple[float, float], seconds: float) -> tuple[float, float]:
    """The part of a clip ``trim`` keeps, in seconds: (in, out), inside the clip."""
    b = seconds if trim[1] <= 0 else min(float(trim[1]), seconds)
    a = min(max(float(trim[0]), 0.0), b)
    return a, b


def frame_index(times: np.ndarray, at: float, ended: bool = False) -> int:
    """The frame shown at ``at`` seconds into the clip: the last whose time is at or before
    it (within ``EPS``). An ended clip shows the last frame before its out point."""
    if len(times) == 0:
        return 0
    probe = at - EPS if ended else at + EPS
    i = int(np.searchsorted(times, probe, side="right")) - 1
    return min(max(i, 0), len(times) - 1)


# ---------------------------------------------------------------- placing and sampling

FULL = (0.0, 0.0, 1.0, 1.0)


def fit_box(fit: str, box: tuple[float, float], src: tuple[int, int],
            native: float = 0.0) -> tuple[float, float, tuple]:  # fmt: skip
    """How a picture ``src`` = (w, h) pixels sits in a box (w, h) in world units: the size of
    the quad drawn and the part of the picture it shows (u0, v0, u1, v1, fractions of its
    width and height, v from the top). ``stretch`` fills the box; ``contain`` fits inside it
    keeping its shape; ``cover`` fills it keeping its shape, cut to the box; ``none`` keeps
    its own size (``native`` world units per pixel), cut to the box."""
    bw, bh = box
    sw, sh = max(src[0], 1), max(src[1], 1)
    if fit == "contain":
        s = min(bw / sw, bh / sh)
        return sw * s, sh * s, FULL
    if fit == "cover":
        s = max(bw / sw, bh / sh)
        fx, fy = min(bw / (sw * s), 1.0), min(bh / (sh * s), 1.0)
        return bw, bh, ((1 - fx) / 2, (1 - fy) / 2, (1 + fx) / 2, (1 + fy) / 2)
    if fit == "none" and native > 0:
        w, h = sw * native, sh * native
        fx, fy = min(bw / w, 1.0), min(bh / h, 1.0)
        return min(w, bw), min(h, bh), ((1 - fx) / 2, (1 - fy) / 2, (1 + fx) / 2, (1 + fy) / 2)
    return bw, bh, FULL


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def sample(src: np.ndarray, pw: int, ph: int, center: tuple[float, float], height: float,
           quad: tuple[float, float], crop: tuple[float, float, float, float],
           bg: np.ndarray, *, opaque: bool, glow: bool) -> np.ndarray:  # fmt: skip
    """The visible part of a frame as linear RGBA (ph, pw, 4) float32, row 0 = top: the view
    ``center``/``height`` in quad coordinates (``DrawContext.field``) of a quad ``quad`` =
    (w, h) world units showing the part ``crop`` of ``src``. Transparent parts show ``bg``;
    alpha is the glow weight (luminance when ``glow``, else 1)."""
    sh, sw = src.shape[:2]
    rw, uh = quad
    half_w = height * pw / ph / 2
    if crop == FULL:  # the arithmetic Image always used
        x0, x1 = (center[0] - half_w) / rw * sw, (center[0] + half_w) / rw * sw
        y0, y1 = (
            (1 - (center[1] + height / 2) / uh) * sh,
            (1 - (center[1] - height / 2) / uh) * sh,
        )
    else:
        u0, v0, u1, v1 = crop
        cw, ch = (u1 - u0) * sw, (v1 - v0) * sh
        x0 = u0 * sw + (center[0] - half_w) / rw * cw
        x1 = u0 * sw + (center[0] + half_w) / rw * cw
        y0 = v0 * sh + (1 - (center[1] + height / 2) / uh) * ch
        y1 = v0 * sh + (1 - (center[1] - height / 2) / uh) * ch
    xs = np.clip(x0 + (np.arange(pw) + 0.5) / pw * (x1 - x0) - 0.5, 0, sw - 1)
    ys = np.clip(y0 + (np.arange(ph) + 0.5) / ph * (y1 - y0) - 0.5, 0, sh - 1)
    sx, sy = (x1 - x0) / pw, (y1 - y0) / ph
    if src.dtype == np.uint8:  # in bands of rows, so a big frame never sits in float
        img = np.empty((ph, pw, 4), dtype=np.float32)
        for r in range(0, ph, BAND_ROWS):
            img[r : r + BAND_ROWS] = area_sample_bytes(src, xs, ys[r : r + BAND_ROWS], sx, sy,
                                                       premultiply=not opaque)  # fmt: skip
    else:
        img = area_sample(src, xs, ys, sx, sy)
    if src.dtype == np.uint8 and opaque:  # every pixel opaque: nothing shows through
        out = img
    else:
        a = img[..., 3:]
        # transparency shows the background; premultiplied, the colour already carries alpha
        rgb = img[..., :3] * a + bg * (1 - a) if opaque else img[..., :3] + bg * (1 - a)
        out = np.empty((ph, pw, 4), dtype=np.float32)
        out[..., :3] = rgb
    out[..., 3] = out[..., :3] @ _LUMA if glow else 1.0
    return out


def area_sample(src: np.ndarray, xs: np.ndarray, ys: np.ndarray, sx: float, sy: float):
    """Bilinear samples of ``src`` at ``xs`` x ``ys``; when a screen pixel covers several
    picture pixels (shrinking), the picture is first box-averaged down to about that scale, so
    a large photo does not shimmer or alias."""
    k = int(max(1, np.floor(min(abs(sx), abs(sy)))))
    if k > 1:
        h, w = (src.shape[0] // k) * k, (src.shape[1] // k) * k
        src = src[:h, :w].reshape(h // k, k, w // k, k, 4).mean(axis=(1, 3))
        xs, ys = (xs + 0.5) / k - 0.5, (ys + 0.5) / k - 0.5
        xs, ys = np.clip(xs, 0, src.shape[1] - 1), np.clip(ys, 0, src.shape[0] - 1)
    return _bilinear(src, xs, ys)


def _bilinear(src: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, src.shape[1] - 1), np.minimum(y0 + 1, src.shape[0] - 1)
    fx, fy = (xs - x0)[None, :, None], (ys - y0)[:, None, None]
    return (src[y0][:, x0] * (1 - fx) * (1 - fy) + src[y0][:, x1] * fx * (1 - fy)
            + src[y1][:, x0] * (1 - fx) * fy + src[y1][:, x1] * fx * fy)  # fmt: skip


def linear(frame: np.ndarray, premultiply: bool) -> np.ndarray:
    """An sRGB uint8 RGBA frame (or part of one) as linear float32, premultiplied if asked."""
    out = _LUT[frame]
    out[..., 3] = frame[..., 3].astype(np.float32) / 255.0
    if premultiply:
        out[..., :3] *= out[..., 3:]
    return out


def area_sample_bytes(src: np.ndarray, xs: np.ndarray, ys: np.ndarray, sx: float, sy: float,
                      *, premultiply: bool) -> np.ndarray:  # fmt: skip
    """``area_sample`` of an sRGB uint8 frame, converting only the part the samples read
    (aligned to the box filter's blocks, so the result is that of converting it all)."""
    k = int(max(1, np.floor(min(abs(sx), abs(sy)))))
    H, W = src.shape[:2]
    if k > 1:
        Hk, Wk = H // k, W // k
        xr = np.clip((xs + 0.5) / k - 0.5, 0, Wk - 1)
        yr = np.clip((ys + 0.5) / k - 0.5, 0, Hk - 1)
    else:
        Hk, Wk, xr, yr = H, W, xs, ys
    c0, r0 = int(np.floor(xr.min())), int(np.floor(yr.min()))
    c1, r1 = min(int(np.floor(xr.max())) + 1, Wk - 1), min(int(np.floor(yr.max())) + 1, Hk - 1)
    part = linear(src[r0 * k : (r1 + 1) * k, c0 * k : (c1 + 1) * k], premultiply)
    if k > 1:
        part = part.reshape(r1 - r0 + 1, k, c1 - c0 + 1, k, 4).mean(axis=(1, 3), dtype=np.float32)
    return _bilinear32(part, xr - c0, yr - r0)


def _bilinear32(src: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``_bilinear`` in float32, separably: rows first (whole-row copies), then columns
    gathered as one 16-byte element a pixel (several times faster on a full frame)."""
    x0, y0 = np.floor(xs).astype(np.intp), np.floor(ys).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, src.shape[1] - 1), np.minimum(y0 + 1, src.shape[0] - 1)
    fx = (xs - x0).astype(np.float32)[None, :, None]
    fy = (ys - y0).astype(np.float32)[:, None, None]
    rows = np.take(src, y0, axis=0)
    rows += (np.take(src, y1, axis=0) - rows) * fy
    packed = np.ascontiguousarray(rows).view(np.complex128)[..., 0]  # (ph, w): RGBA each
    left = np.take(packed, x0, axis=1).view(np.float32).reshape(len(ys), len(xs), 4)
    right = np.take(packed, x1, axis=1).view(np.float32).reshape(len(ys), len(xs), 4)
    left += (right - left) * fx
    return left


def average(src: np.ndarray, bg: np.ndarray, *, opaque: bool) -> tuple[float, float, float]:
    """The mean linear colour of a frame laid over ``bg`` (about 64 x 64 samples): what text
    over it is judged against when there is no rendered picture (lint ``contrast``)."""
    step = max(1, max(src.shape[:2]) // 64)
    part = src[::step, ::step]
    if part.dtype == np.uint8:
        part = linear(part, premultiply=True)
    elif opaque:
        part = part.astype(np.float32)
    mean = part.reshape(-1, 4).mean(axis=0)
    rgb = mean[:3] + np.asarray(bg[:3], dtype=np.float64) * (1 - mean[3])
    return tuple(float(v) for v in rgb)
