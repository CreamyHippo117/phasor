"""Render jobs: a render that runs on a thread of its own, with progress and a cancel that
really stops it (Milestone 25; ``render.job.render_file``'s ``progress`` and ``cancel`` from
Milestone 21). The app's Draft preview and Render buttons and the MCP tools ``render_start``,
``render_status`` and ``render_cancel`` share them (``Workspace``).

- The job draws a **snapshot** of the scene taken when it starts (``Engine.snapshot``): edits
  made meanwhile, by the person or an AI, never reach it, and the live engine is never read
  from the job's thread.
- It writes to a **partial file** beside the target (``a.draft.partial.mp4``) and moves it
  onto the target only when the render finished: a cancelled or failed render leaves nothing
  behind, and an earlier file of that name stays as it was until then.
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

_ids = itertools.count(1)


def _render(engine: Any, out: Path, **kwargs: Any) -> dict[str, Any]:
    """The render a job runs (a seam: tests put a fake here, so jobs are tested without
    OpenGL). One process: ``scene.render`` with one worker, its kernels on every core."""
    from phasor.scene import render

    return render(engine, out, workers=1, **kwargs)


def partial_path(target: Path) -> Path:
    """Where a job writes before it is done: ``a.draft.mp4`` -> ``a.draft.partial.mp4`` (the
    real extension last, so the encoder still knows the format)."""
    return target.with_name(f"{target.stem}.partial{target.suffix}")


class RenderJob:
    def __init__(self, engine: Any, target: Path, *, preset: str, start: Any = None,
                 end: Any = None, fps: float | None = None, frame: Any = None) -> None:  # fmt: skip
        self.id = f"render_{next(_ids)}"
        self.engine, self.target, self.preset = engine, Path(target), preset
        self.kwargs = {"preset": preset, "start": start, "end": end, "fps": fps, "frame": frame}
        self.partial = partial_path(self.target)
        self.state = "running"  # running, done, failed, cancelled
        self.frame, self.frames = 0, 0
        self.result: dict[str, Any] | None = None
        self.error = ""
        self.started = time.monotonic()
        self.finished: float | None = None
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"phasor-{self.id}", daemon=True)

    def start(self) -> RenderJob:
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
            os.replace(self.partial, self.target)
            self.result = {**r, "path": str(self.target)}
            self.frame = self.frames = int(r.get("frames", self.frames) or 1)
            self.state = "done"
        except RenderCancelled:
            self.state = "cancelled"
        except BaseException as e:  # no OpenGL, no encoder, a file gone: said, never raised
            self.error = str(e) or type(e).__name__
            self.state = "failed"
        finally:
            if self.state != "done":
                self.partial.unlink(missing_ok=True)  # render_file removes it too; be sure
            self.finished = time.monotonic()
            self.engine = None  # the snapshot is not needed any more

    @property
    def running(self) -> bool:
        return self.state == "running"

    def cancel(self, wait: float = CANCEL_WAIT) -> None:
        """Stop the render (between two frames) and wait for it to clean up."""
        self._cancel.set()
        self._thread.join(wait)

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def status(self) -> dict[str, Any]:
        end = self.finished if self.finished is not None else time.monotonic()
        out: dict[str, Any] = {
            "job": self.id, "state": self.state, "preset": self.preset, "path": str(self.target),
            "frame": self.frame, "frames": self.frames,
            "progress": round(self.frame / self.frames, 4) if self.frames else 0.0,
            "seconds": round(end - self.started, 2),
        }  # fmt: skip
        if self.state == "running" and self._cancel.is_set():
            out["state"] = "cancelling"
        if self.error:
            out["error"] = self.error
        if self.result is not None:
            out["result"] = {k: self.result[k] for k in ("frames", "fps", "size", "drawn",
                                                         "repeated", "warnings", "narration")
                             if k in self.result}  # fmt: skip
        return out


class Jobs:
    """The render jobs of one workspace."""

    def __init__(self) -> None:
        self._jobs: dict[str, RenderJob] = {}

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
        self._jobs[job.id] = job
        return job.start()

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
