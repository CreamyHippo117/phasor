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

import os
import stat as _stat
from pathlib import Path, PurePosixPath, PureWindowsPath


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
    when it is not there (or is not a regular file)."""
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
    None when it is a regular file inside the folder."""
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
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise ValueError(f"{path}: cannot be opened ({e})") from None
    with os.fdopen(fd, "rb") as f:
        if not _stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{path}: not a regular file")
        data = f.read(limit + 1)
    if len(data) > limit:
        raise TooLarge(f"{path}: larger than {limit} bytes")
    return data


def user_cache(name: str, env: str | None = None) -> Path:
    """phasor's ``name`` folder in the user's cache (``%LOCALAPPDATA%``, ``~/Library/Caches``
    or ``$XDG_CACHE_HOME``): what phasor builds once and keeps (built kernels, decoded
    sound). The environment variable ``env``, when set, names the folder instead."""
    import sys

    if env and os.environ.get(env):
        return Path(os.environ[env])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "phasor" / name
