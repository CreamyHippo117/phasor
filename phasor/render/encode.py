"""Video output (SPEC "Encoding").

``open_writer(path, width, height, fps)`` returns a writer with ``write(frame)`` and
``close()``; frames are (h, w, 3) uint8 sRGB from ``render.output.encode_srgb8``.

- ``*.png`` with a ``%d`` pattern, or a directory: a PNG sequence (for editing).
- ``*.mov``: ProRes 4444 through a system ffmpeg (for editing).
- anything else (``*.mp4``): the best working encoder, probed in SPEC order: PyAV NVENC
  (hevc, then h264), a system ffmpeg fed raw frames over a pipe (NVENC first), then PyAV
  libx264. A probe counts only when it really encoded frames (``render.doctor``).
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np

from phasor.render.doctor import check_ffmpeg, check_pyav, pick_encoder


class EncodeError(RuntimeError):
    pass


class Writer(Protocol):
    path: Path
    encoder: str

    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


@functools.cache
def best_encoder() -> str | None:
    """``"pyav:<codec>"``, ``"ffmpeg:<codec>"`` or None. Probed once per process."""
    return pick_encoder(check_pyav(), check_ffmpeg())


def _rate(fps: float) -> Fraction:
    return Fraction(fps).limit_denominator(1001)


def _check(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
        raise EncodeError(
            f"frames must be ({height}, {width}, 3) uint8, got {frame.dtype} {frame.shape}"
        )
    return np.ascontiguousarray(frame)


class PngSequence:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.pattern = path / "frame_%05d.png" if "%" not in path.name else path
        self.pattern.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.encoder = path, "png"
        self.size, self.count = (width, height), 0

    def write(self, frame: np.ndarray) -> None:
        from PIL import Image

        frame = _check(frame, *self.size)
        Image.fromarray(frame, "RGB").save(str(self.pattern) % self.count)
        self.count += 1

    def close(self) -> None:
        pass


class PyAVWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, codec: str) -> None:
        import av

        self.path, self.encoder, self.size = path, f"pyav:{codec}", (width, height)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream(codec, rate=_rate(fps))
        self._stream.width, self._stream.height = width, height
        self._stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            self._stream.options = {"crf": "16", "preset": "medium"}
        self._frame_type = av.VideoFrame

    def write(self, frame: np.ndarray) -> None:
        vf = self._frame_type.from_ndarray(_check(frame, *self.size), format="rgb24")
        for packet in self._stream.encode(vf):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class FFmpegWriter:
    def __init__(
        self, path: Path, width: int, height: int, fps: float, codec: str, pix_fmt: str
    ) -> None:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise EncodeError("no ffmpeg on PATH")
        self.path, self.encoder, self.size = path, f"ffmpeg:{codec}", (width, height)
        rate = _rate(fps)
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
               "-c:v", codec, "-pix_fmt", pix_fmt, str(path)]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(_check(frame, *self.size).tobytes())
        except BrokenPipeError as e:
            raise EncodeError(self._proc.stderr.read().decode(errors="replace")) from e

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise EncodeError(self._proc.stderr.read().decode(errors="replace"))


def open_writer(path: str | Path, width: int, height: int, fps: float) -> Writer:
    path = Path(path)
    if path.suffix == "" or (path.suffix == ".png" and "%" in path.name):
        return PngSequence(path, width, height)
    if width % 2 or height % 2:
        raise EncodeError(f"video needs an even width and height, got {width}x{height}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mov":
        return FFmpegWriter(path, width, height, fps, "prores_ks", "yuva444p10le")
    choice = best_encoder()
    if choice is None:
        raise EncodeError("no working video encoder (run `phasor doctor`); write PNGs instead")
    backend, codec = choice.split(":", 1)
    if backend == "pyav":
        return PyAVWriter(path, width, height, fps, codec)
    return FFmpegWriter(path, width, height, fps, codec, "yuv420p")


def mux_audio(video: str | Path, samples: np.ndarray, rate: int) -> None:
    """Add a mono int16 narration track to a finished video file, in place. The video
    stream is copied, not re-encoded; the audio becomes AAC (MP4) or PCM (MOV)."""
    import av

    video = Path(video)
    tmp = video.with_name(video.stem + ".mux" + video.suffix)
    with av.open(str(video)) as src, av.open(str(tmp), mode="w") as dst:
        vin = src.streams.video[0]
        vout = dst.add_stream_from_template(vin)
        codec = "pcm_s16le" if video.suffix == ".mov" else "aac"
        aout = dst.add_stream(codec, rate=rate, layout="mono")
        for packet in src.demux(vin):
            if packet.dts is None:
                continue
            packet.stream = vout
            dst.mux(packet)
        pcm = np.ascontiguousarray(samples, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = rate
        for packet in aout.encode(frame):
            dst.mux(packet)
        for packet in aout.encode(None):
            dst.mux(packet)
    tmp.replace(video)
