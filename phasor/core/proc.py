"""Starting other programs (TeX, ffmpeg, nvidia-smi, speech engines, PowerShell).

Pass ``creationflags=NO_WINDOW`` to every ``subprocess`` call. On Windows the desktop app has
no console, and a console program started from it without this flag opens a visible console
window of its own: a black flash on every typeset formula (checked in the packaged app,
2026-11-20). Elsewhere it is 0 and changes nothing.
"""

from __future__ import annotations

import subprocess
import sys

NO_WINDOW: int = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def ascii_path(path: object) -> str:
    """``path`` as a string a compiler can open. NVRTC and MinGW's assembler and linker misread
    a path outside the ASCII (NVRTC) or ANSI (MinGW) code page, as under a non-English user
    name (``C:/Users/Jürgen``, ``C:/Users/Ψ``). On Windows such a path is given by its
    8.3 short name (``JRGENO~1``) when the volume keeps them; the path must exist. Elsewhere,
    or when there is no short name, it is returned unchanged."""
    s = str(path)
    if s.isascii() or sys.platform != "win32":
        return s
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, len(buf))  # type: ignore[attr-defined]
    return buf.value if 0 < n < len(buf) and buf.value.isascii() else s


def ascii_temp_env() -> dict[str, str] | None:
    """The environment for a compiler subprocess, with ``TEMP``/``TMP`` given as ASCII short
    paths (see ``ascii_path``); None (inherit) when they already are ASCII."""
    import os
    import tempfile

    tmp = tempfile.gettempdir()
    short = ascii_path(tmp)
    if short == tmp:
        return None
    return {**os.environ, "TEMP": short, "TMP": short}


def ascii_tempdir() -> None:
    """Point Python's temp folder at its ASCII short path, for libraries that hand temp files
    to a compiler themselves (CuPy writes NVRTC's ``.cu`` files there)."""
    import tempfile

    tmp = tempfile.gettempdir()
    short = ascii_path(tmp)
    if short != tmp:
        tempfile.tempdir = short
