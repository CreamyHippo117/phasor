"""Files an object reads (a mesh, a picture), named relative to the scene's folder.

A type lists the parameters that hold such paths in its ``Files`` class variable
(``Files = ("path",)``). Every path is relative and stays inside the folder: an absolute
path or a ``..`` part is a validation error, so a scene (or the AI writing one) can never
read a file from anywhere else. A path naming a file that is not there is the lint issue
``missing_file``; the object then draws nothing instead of failing.

Only a regular file counts as there: a folder, a FIFO, a device (``/dev/zero``, Windows'
``CON``) or a link that loops is treated as missing, so reading one can never hang or raise
(Milestone 16 audit). ``read_bytes`` reads one with a size cap, refusing anything that turned
into something else between the check and the read.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat as _stat
from pathlib import Path, PurePosixPath, PureWindowsPath

# A picture sequence (Milestone 24): a file name with one ``%d`` or ``%0Nd`` in it
# (``frames/shot_%04d.png``) names the frames 0, 1, 2, ... (or from 1 to 4, like ffmpeg: the
# first of 0-4 that is there), up to the first one missing, at most ``MAX_SEQUENCE``.
_SEQUENCE = re.compile(r"%(0[1-9])?d")
FIRST_FRAMES = 5  # the first frame is the first of 0..4 that is there
MAX_SEQUENCE = 3000  # frames; more are not read (lint video_too_big)
# The file name endings of pictures (Pillow) and clips (PyAV), lower case.
PICTURE_EXT = ("png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff")
VIDEO_EXT = ("mp4", "m4v", "mov", "webm", "mkv")


def is_sequence(value: str) -> bool:
    """Whether ``value`` names a picture sequence: exactly one ``%d``/``%0Nd`` (N 1-9) in
    its file name and no other ``%`` anywhere in it."""
    if not isinstance(value, str) or value.count("%") != 1:
        return False
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    return len(_SEQUENCE.findall(name)) == 1


def sequence(base_dir: str | Path, value: str) -> tuple[Path, ...]:
    """The frames of the sequence ``value`` names, in order: regular files inside the folder
    (each checked like ``regular_file``, so a link out of the folder or a FIFO ends the
    sequence), from the first of 0-4 that is there up to the first gap; at most
    ``MAX_SEQUENCE + 1`` (one more than is read, to tell that there are more). Empty when
    none is there."""
    return tuple(path for _, path, _ in _scan(base_dir, value))


def members(base_dir: str | Path, value: str) -> tuple[str, ...]:
    """The names (relative, as written) of the files a ``Files`` value reads: itself, or each
    frame of a sequence (see ``sequence``)."""
    if not is_sequence(value):
        return (value,)
    return tuple(name for name, _, _ in _scan(base_dir, value))


def _scan(base_dir: str | Path, value: str) -> list[tuple[str, Path, os.stat_result]]:
    """(name, path, stat) of each frame of a sequence. The folder is resolved (and checked to
    be inside) once; a frame that is a plain regular file in it (``lstat``: not a link) is
    inside too, so a long sequence costs one ``lstat`` a frame; anything else (a link) goes
    through ``regular_file``."""
    if not is_sequence(value):
        return []
    try:
        folder = resolve(base_dir, value).parent
    except ValueError:
        return []
    rel_folder = relative_path(value).rpartition("/")[0]
    out: list[tuple[str, Path, os.stat_result]] = []
    for i in range(FIRST_FRAMES + MAX_SEQUENCE + 1):
        name = value % i
        leaf = name.replace("\\", "/").rpartition("/")[2]
        try:
            st = os.lstat(folder / leaf)
            if _stat.S_ISREG(st.st_mode):
                path = folder / leaf
            else:  # a link (it may point out of the folder) or not a file
                path = regular_file(base_dir, f"{rel_folder}/{leaf}" if rel_folder else leaf)
                st = path.stat()
            out.append((name, path, st))
        except (OSError, ValueError, TypeError):
            if out or i + 1 >= FIRST_FRAMES:
                break
            continue
        if len(out) > MAX_SEQUENCE:
            break
    return out


def _sequence_stamp(base_dir: str | Path, value: str) -> tuple:
    """A sequence's stamp: the pattern's path, a digest of every frame's name, modification
    time and size (``hashlib``: the same in every process, so a render worker can compare
    it), and the number of frames; ``None`` for both when no frame is there."""
    try:
        where = str(resolve(base_dir, value))
    except ValueError:
        return (value, None, None)
    frames = _scan(base_dir, value)
    if not frames:
        return (where, None, None)
    digest = hashlib.sha256()
    for _, path, st in frames:
        digest.update(f"{path.name}\0{st.st_mtime_ns}\0{st.st_size}\n".encode())
    return (where, "seq:" + digest.hexdigest(), len(frames))


def relative_path(value: str) -> str:
    """``value`` if it is a relative path that stays in its folder, else ``ValueError``.
    Written with forward slashes, so a document reads the same on every system."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a file path must be a non-empty string")
    win, posix = PureWindowsPath(value), PurePosixPath(value.replace("\\", "/"))
    if win.is_absolute() or win.drive or posix.is_absolute():
        raise ValueError(f"{value!r} is absolute; name files relative to the scene's folder")
    if ".." in posix.parts:
        raise ValueError(f"{value!r} leaves the scene's folder; files must stay inside it")
    return posix.as_posix()


def resolve(base_dir: str | Path, value: str) -> Path:
    """The file ``value`` names, inside ``base_dir`` (the current folder when empty).
    ``ValueError`` if it resolves outside the folder (a link pointing out, say)."""
    try:
        root = Path(base_dir or ".").resolve()
        path = (root / relative_path(value)).resolve()
    except (RuntimeError, OSError) as e:  # a link that loops (RuntimeError before 3.13)
        raise ValueError(f"{value!r} cannot be followed: {e}") from None
    if root != path and root not in path.parents:
        raise ValueError(f"{value!r} is outside the scene's folder")
    return path


def regular_file(base_dir: str | Path, value: str) -> Path:
    """The regular file ``value`` names inside ``base_dir``, or ``ValueError`` (outside the
    folder, missing, a folder, a FIFO or device, a link that loops). Every type that reads a
    file goes through this (or ``stamp``/``read_bytes``)."""
    path = resolve(base_dir, value)
    try:
        st = path.stat()
    except (OSError, ValueError) as e:
        raise ValueError(f"{value!r} is not there ({e})") from None
    if not _stat.S_ISREG(st.st_mode):
        raise ValueError(f"{value!r} is not a regular file")
    return path


def stamp(base_dir: str | Path, value: str) -> tuple:
    """What the file ``value`` names is now, to tell when it changed (contract 3: a changed
    file is read again): its path, modification time and size; ``None`` for the last two
    when it is not there (or is not a regular file). For a picture sequence
    (``is_sequence``) the second is a digest of all its frames and the third their number."""
    if is_sequence(value):
        return _sequence_stamp(base_dir, value)
    try:
        path = resolve(base_dir, value)
        st = path.stat()
    except (ValueError, OSError):
        return (value, None, None)
    if not _stat.S_ISREG(st.st_mode):
        return (str(path), None, None)
    return (str(path), st.st_mtime_ns, st.st_size)


def exists(base_dir: str | Path, value: str) -> bool:
    """Whether ``value`` names a regular file inside ``base_dir``."""
    return stamp(base_dir, value)[1] is not None


def problem(base_dir: str | Path, value: str) -> str | None:
    """Why ``value`` cannot be read as a file of the scene (for lint ``missing_file``), or
    None when it is a regular file inside the folder (for a sequence: when its first frame
    is)."""
    if is_sequence(value):
        try:
            resolve(base_dir, value)
        except ValueError as e:
            return str(e)
        if not sequence(base_dir, value):
            first = ", ".join(repr(value % i) for i in range(2))
            return f"no frames of {value!r} in the scene's folder (looked for {first}, ...)"
        return None
    try:
        regular_file(base_dir, value)
    except ValueError as e:
        msg = str(e)
        if "is not there" in msg:
            return f"no file {value!r} in the scene's folder"
        return msg
    return None


class TooLarge(ValueError):
    """A file is larger than a reader takes."""


def read_bytes(path: str | Path, limit: int) -> bytes:
    """The bytes of the regular file at ``path`` (a path ``stamp`` or ``regular_file``
    returned), at most ``limit`` of them: ``TooLarge`` (a ``ValueError``) when it is
    larger, ``ValueError`` when it is not a regular file when opened (it never blocks on a
    FIFO: opened non-blocking, then checked)."""
    with open_regular(path) as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise TooLarge(f"{path}: larger than {limit} bytes")
    return data


def open_regular(path: str | Path, expect: tuple | None = None):
    """The regular file at ``path`` opened for reading (binary), never blocking on a FIFO
    (opened non-blocking, then checked); ``ValueError`` when it is not a regular file, or
    when ``expect`` (a ``stamp``'s modification time and size) no longer matches: the file
    changed since it was stamped. For readers that stream a file (a video decoder)."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise ValueError(f"{path}: cannot be opened ({e})") from None
    f = os.fdopen(fd, "rb")
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise ValueError(f"{path}: not a regular file")
        if expect is not None and (st.st_mtime_ns, st.st_size) != tuple(expect):
            raise ValueError(f"{path}: changed since it was stamped")
    except BaseException:
        f.close()
        raise
    return f
