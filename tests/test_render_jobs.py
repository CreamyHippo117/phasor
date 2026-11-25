"""Milestone 25: render jobs (``Workspace.render_start`` / ``render_status`` /
``render_cancel``), shared by the app's Draft preview and Render buttons and the MCP server.
A job renders a snapshot on its own thread, reports progress, and its cancel really stops it
and leaves nothing behind. The machinery is tested with a fake render (no OpenGL); one real
draft runs where there is OpenGL."""

from __future__ import annotations

import threading
import time

import pytest

from phasor.render.job import RenderCancelled
from phasor.workspace import ToolError, Workspace
from phasor.workspace import jobs as jobmod
from tests.conftest import gl_available

DOT = {"op": "add_object", "object": {"id": "d", "type": "dot", "params": {"r": 0.2}}}


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.scene_new("jobs")
    w.scene_edit(DOT)
    w.scene_edit({"op": "set_marker", "name": "end", "t": 1.0})
    yield w
    w._jobs.cancel_all()


class FakeRender:
    """Writes a file a frame at a time like render_file: progress after each frame, cancel
    checked between frames (removing what it wrote), a gate to hold it mid-way."""

    def __init__(self, frames: int = 20, fail_at: int | None = None) -> None:
        self.frames, self.fail_at = frames, fail_at
        self.gate = threading.Event()
        self.seen: list = []

    def __call__(self, engine, out, *, progress, cancel, preset, **kw):
        self.seen.append((engine, out, preset, kw))
        with open(out, "wb") as f:
            for k in range(self.frames):
                if k == self.frames // 2:
                    self.gate.wait(10)
                if k == self.fail_at:
                    raise RuntimeError("no OpenGL here (tier C); nothing can be rendered")
                f.write(b"x")
                progress(k + 1, self.frames)
                if cancel.is_set():
                    f.close()
                    out.unlink()
                    raise RenderCancelled("render cancelled")
        return {"path": str(out), "frames": self.frames, "fps": 15.0, "size": (64, 36)}


def _wait(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_a_job_reports_progress_and_finishes_into_its_file(ws, monkeypatch):
    fake = FakeRender()
    monkeypatch.setattr(jobmod, "_render", fake)
    st = ws.render_start()  # the draft, to <scene>.draft.mp4
    assert st["state"] == "running" and st["path"].endswith("jobs.draft.mp4")
    _wait(lambda: ws.render_status(st["job"])["frame"] >= 10)
    mid = ws.render_status(st["job"])
    assert 0 < mid["progress"] < 1 and mid["frames"] == 20
    out = ws.out_dir / "jobs.draft.mp4"
    assert not out.exists()  # it writes beside the target until it is done
    fake.gate.set()
    _wait(lambda: ws.render_status(st["job"])["state"] == "done")
    done = ws.render_status(st["job"])
    assert done["progress"] == 1.0 and done["result"]["frames"] == 20
    assert out.read_bytes() == b"x" * 20
    assert not list(out.parent.glob(".*.partial.*"))
    engine, written, preset, _ = fake.seen[0]
    assert (
        preset == "draft"
        and written.parent == out.parent
        and written.name.startswith(".jobs.draft.")
    )
    assert written.name.endswith(".partial.mp4")
    assert engine is not ws.engine  # a snapshot: later edits never reach the job
    assert [j["job"] for j in ws.render_status()["jobs"]] == [st["job"]]


def test_cancel_stops_it_and_leaves_the_old_file_as_it_was(ws, monkeypatch):
    fake = FakeRender()
    monkeypatch.setattr(jobmod, "_render", fake)
    final = ws.out_dir / "final.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"the real render")
    with pytest.raises(ToolError, match="exists"):  # never over a file without being asked
        ws.render_start("final", out=str(final))
    st = ws.render_start("final", out=str(final), overwrite=True)
    _wait(lambda: ws.render_status(st["job"])["frame"] >= 10)
    threading.Timer(0.2, fake.gate.set).start()
    r = ws.render_cancel(st["job"])
    assert r["state"] == "cancelled"
    assert final.read_bytes() == b"the real render"  # untouched
    assert not list(final.parent.glob(".*.partial.*"))  # nothing half-written left behind
    assert ws.render_cancel(st["job"])["state"] == "cancelled"  # again: nothing to do


def test_a_failed_job_says_why_and_cleans_up(ws, monkeypatch):
    monkeypatch.setattr(jobmod, "_render", FakeRender(fail_at=3))
    st = ws.render_start("preview", out="p.mp4")
    _wait(lambda: ws.render_status(st["job"])["state"] != "running")
    r = ws.render_status(st["job"])
    assert r["state"] == "failed" and "OpenGL" in r["error"]
    assert not (ws.root / "p.mp4").exists() and not (ws.root / "p.partial.mp4").exists()


def test_bad_requests_are_refused_before_anything_starts(ws, monkeypatch):
    fake = FakeRender()
    monkeypatch.setattr(jobmod, "_render", fake)
    for kwargs, msg in (({"preset": "ultra"}, "preset is one of"),
                        ({"start": "nowhere"}, "neither seconds nor a marker"),
                        ({"fps": 0.0001}, "fps"),
                        ({"out": "../far.mp4"}, "outside"),
                        ({"out": "a.gif"}, "writes a .mp4"),
                        ({"out": "a.png"}, "one frame")):  # fmt: skip
        with pytest.raises(ToolError, match=msg):
            ws.render_start(**kwargs)
    with pytest.raises(ToolError, match="no render job"):
        ws.render_status("render_999999")
    assert fake.seen == [] and ws.render_status() == {"jobs": []}
    a = ws.render_start(out="a.mp4")
    with pytest.raises(ToolError, match="being rendered"):
        ws.render_start(out="a.mp4", overwrite=True)
    ws.render_start(out="b.mp4")
    with pytest.raises(ToolError, match="renders are running"):
        ws.render_start(out="c.mp4")
    fake.gate.set()
    _wait(lambda: ws.render_status(a["job"])["state"] == "done")


def test_the_tools_are_served(ws):
    from phasor.mcp.server import TOOL_NAMES
    from phasor.workspace.remote import tool_names

    for name in ("render_start", "render_status", "render_cancel"):
        assert name in TOOL_NAMES and name in tool_names()


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_real_draft_job_renders_and_a_cancelled_one_leaves_nothing(ws):
    st = ws.render_start(end=0.4)
    _wait(lambda: ws.render_status(st["job"])["state"] != "running", timeout=300)
    r = ws.render_status(st["job"])
    assert r["state"] == "done", r
    assert (ws.out_dir / "jobs.draft.mp4").stat().st_size > 0
    ws.scene_edit({"op": "set_marker", "name": "late", "t": 30.0})  # long enough to cancel
    st = ws.render_start("preview", out="long.mp4")
    ws.render_cancel(st["job"])
    _wait(
        lambda: ws.render_status(st["job"])["state"] not in ("running", "cancelling"), timeout=120
    )
    assert ws.render_status(st["job"])["state"] == "cancelled", ws.render_status(st["job"])
    assert not (ws.root / "long.mp4").exists()
    assert not list(ws.root.glob(".*.partial.*"))
