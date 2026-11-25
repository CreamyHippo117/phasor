"""Render jobs: a render that runs on a thread of its own, with progress and a cancel that
really stops it (Milestone 25; ``render.job.render_file``'s ``progress`` and ``cancel`` from
Milestone 21). The app's Draft preview and Render buttons and the MCP tools ``render_start``,
``render_status`` and ``render_cancel`` share them (``Workspace``).

- The job draws a **snapshot** of the scene taken when it starts (``Engine.snapshot``): edits
  made meanwhile, by the person or an AI, never reach it, and the live engine is never read
  from the job's thread.
- It writes to a **partial file** of its own beside the target (hidden, with a random part:
  ``.a.draft.3f9c01ab.partial.mp4``, created exclusively, so it is never a file of yours) and
  moves it onto the target only when the render finished: a cancelled or failed render leaves
  nothing behind, and an earlier file of that name stays as it was until then. When the
  target cannot be replaced (Windows: a player has it open) the render goes to the next free
  name (``a.draft (2).mp4``) and ``moved_to`` says so.
- GL work goes through the renderer, which runs it on phasor's one GL thread
  (``glcontext.on_gl_thread``); the job thread itself never touches GL.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from pathlib import Path
from typing import Any

MAX_RUNNING = 2  # renders at once (each shares the one GL thread with the editor)
KEEP_FINISHED = 16  # finished jobs whose status can still be read
CANCEL_WAIT = 5.0  # seconds ``cancel`` waits for the render to stop and clean up


def _render(engine: Any, out: Path, **kwargs: Any) -> dict[str, Any]:
    """The render a job runs (a seam: tests put a fake here, so jobs are tested without
    OpenGL). One process: ``scene.render`` with one worker, its kernels on every core."""
    from phasor.scene import render

    return render(engine, out, workers=1, **kwargs)


PARTIAL = ".partial"  # in every partial file's name (the real extension stays last)


def partial_path(target: Path) -> Path:
    """A new partial file for a render to ``target``, created here (exclusively: never an
    existing file): hidden, beside it, ``.a.draft.<random>.partial.mp4`` (the real extension
    last, so the encoder still knows the format)."""
    import secrets

    for _ in range(100):
        name = target.with_name(f".{target.stem}.{secrets.token_hex(4)}{PARTIAL}{target.suffix}")
        try:
            fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return name
    raise OSError(f"no free name for a partial file beside {target}")


def _free_name(target: Path, k: int) -> Path:
    return target.with_name(f"{target.name.split('.')[0]} ({k}){''.join(target.suffixes)}")


def _move_into_place(partial: Path, target: Path, *, keep: bool = False) -> Path:
    """``os.replace`` the finished file onto ``target``; when that is refused (Windows: the
    file is open in a player), or ``keep`` (a file there must stay), onto the next free
    ``name (2).ext``. Returns where it went."""
    try:
        if keep:
            raise PermissionError("kept")
        os.replace(partial, target)
        return target
    except PermissionError:
        for k in range(2, 100):
            other = _free_name(target, k)
            if other.exists():
                continue
            try:
                os.replace(partial, other)
                return other
            except PermissionError:
                continue
        raise


class RenderJob:
    def __init__(self, engine: Any, target: Path, *, preset: str, start: Any = None,
                 end: Any = None, fps: float | None = None, frame: Any = None,
                 overwrite: bool = False) -> None:  # fmt: skip
        self.id = ""  # given by ``Jobs.start``: render_1, render_2, ... in each workspace
        self.engine, self.target, self.preset = engine, Path(target), preset
        self.kwargs = {"preset": preset, "start": start, "end": end, "fps": fps, "frame": frame}
        self.partial: Path | None = None  # made when it starts (``partial_path``)
        # Whether it may replace a file at the target: checked again at the end, since a
        # file the person made there meanwhile is theirs (it then goes to the next free name).
        self.overwrite = overwrite
        self.moved_to: Path | None = None  # set when the target could not be replaced
        self.state = "running"  # running, done, failed, cancelled
        self.frame, self.frames = 0, 0
        self.result: dict[str, Any] | None = None
        self.error = ""
        self.started = time.monotonic()
        self.finished: float | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, job_id: str) -> RenderJob:
        self.id = job_id
        if self.partial is None:
            self.partial = partial_path(self.target)
        self._thread = threading.Thread(target=self._run, name=f"phasor-{job_id}", daemon=True)
        self._thread.start()
        return self

    def _progress(self, k: int, n: int) -> None:
        self.frame, self.frames = k, n

    def _run(self) -> None:
        from phasor.render.job import RenderCancelled

        try:
            r = _render(self.engine, self.partial, progress=self._progress, cancel=self._cancel,
                        **{k: v for k, v in self.kwargs.items() if v is not None})  # fmt: skip
            if self._cancel.is_set():  # cancelled as it finished: the person said stop
                raise RenderCancelled("render cancelled")
            if self.target.exists() and not self.overwrite:  # made meanwhile: theirs
                where = _move_into_place(self.partial, self.target, keep=True)
            else:
                where = _move_into_place(self.partial, self.target)
            if where != self.target:
                self.moved_to = where
            self.result = {**r, "path": str(where)}
            self.frame = self.frames = int(r.get("frames", self.frames) or 1)
            self.state = "done"
        except RenderCancelled:
            self.state = "cancelled"
        except BaseException as e:  # no OpenGL, no encoder, a file gone: said, never raised
            self.error = str(e) or type(e).__name__
            self.state = "failed"
        finally:
            if self.state != "done" and self.partial is not None:
                self.partial.unlink(missing_ok=True)  # ours (made exclusively): never yours
            self.finished = time.monotonic()
            self.engine = None  # the snapshot is not needed any more

    @property
    def running(self) -> bool:
        return self.state == "running"

    def cancel(self, wait: float = CANCEL_WAIT) -> None:
        """Stop the render (between two frames) and wait for it to clean up."""
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(wait)

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> dict[str, Any]:
        end = self.finished if self.finished is not None else time.monotonic()
        out: dict[str, Any] = {
            "job": self.id, "state": self.state, "preset": self.preset,
            "path": str(self.moved_to or self.target),
            "frame": self.frame, "frames": self.frames,
            "progress": round(self.frame / self.frames, 4) if self.frames else 0.0,
            "seconds": round(end - self.started, 2),
        }  # fmt: skip
        if self.state == "running" and self._cancel.is_set():
            out["state"] = "cancelling"
        if self.error:
            out["error"] = self.error
        if self.moved_to is not None:
            out["moved_to"] = str(self.moved_to)
            out["note"] = (f"{self.target.name} could not be replaced (open in another program?):"
                           f" the render is {self.moved_to.name}")  # fmt: skip
        if self.result is not None:
            out["result"] = {k: self.result[k] for k in ("frames", "fps", "size", "drawn",
                                                         "repeated", "warnings", "narration")
                             if k in self.result}  # fmt: skip
        return out


class Jobs:
    """The render jobs of one workspace."""

    def __init__(self) -> None:
        self._jobs: dict[str, RenderJob] = {}
        self._next = itertools.count(1)

    def start(self, job: RenderJob) -> RenderJob:
        running = [j for j in self._jobs.values() if j.running]
        if len(running) >= MAX_RUNNING:
            names = ", ".join(j.id for j in running)
            raise ValueError(f"{len(running)} renders are running ({names}); wait for one or "
                             "cancel it (render_cancel)")  # fmt: skip
        if any(j.running and j.target == job.target for j in running):
            raise ValueError(f"{job.target.name} is being rendered already")
        done = [k for k, j in self._jobs.items() if not j.running]
        for k in done[: max(0, len(done) - KEEP_FINISHED + 1)]:
            del self._jobs[k]
        job.partial = partial_path(job.target)  # first: an OSError registers nothing
        job_id = f"render_{next(self._next)}"
        self._jobs[job_id] = job
        return job.start(job_id)

    def get(self, job_id: str) -> RenderJob:
        if job_id not in self._jobs:
            known = ", ".join(self._jobs) or "none"
            raise KeyError(f"no render job {job_id!r} (jobs: {known})")
        return self._jobs[job_id]

    def all(self) -> list[RenderJob]:
        return list(self._jobs.values())

    def cancel_all(self) -> None:
        for j in self._jobs.values():
            if j.running:
                j.cancel()
