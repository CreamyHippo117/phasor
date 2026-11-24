"""Milestone 21: the draft render.

A ``draft`` preset below ``preview``: long side 640 at the scene's aspect, 15 fps, a plain look
(no bloom, lens or depth fog; tone map and sRGB still once), the scene without its decorative
motion (life, the theme's camera drift), frames whose drawing repeats the last one repeated
rather than drawn, a fast encoder, progress and cancel, and a window by markers.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Scene, Text, load
from phasor.cli import build_parser, main
from phasor.engine import Engine
from phasor.render import framekey
from phasor.render.presets import DRAFT, FULL, PRESETS, check_fps, preset_named
from tests.conftest import gl_available

ROOT = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "golden"
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _scene(title="draft", look="glow"):
    """A dot that moves for one second then rests for two, a label, living glow."""
    s = Scene(title, look=look)
    s.set("camera.height", 4)
    s.add(Dot(id="p", pos=(-2.0, -0.5)))
    s.add(Text(id="t", text="draft", pos=(0.0, 1.5)))
    s.marker("go", 0.0)
    s.marker("rest", 1.0)
    s.at("go").to("p.pos", (2.0, 0.5, 0.0), duration=1.0)
    s.marker("end", 3.0)
    return s


# ---------------------------------------------------------------- the preset table


def test_the_presets_are_one_table_and_the_old_ones_are_unchanged():
    assert list(PRESETS) == ["draft", "preview", "final", "4k"]
    pv, fi, k4 = PRESETS["preview"], PRESETS["final"], PRESETS["4k"]
    assert (pv.size(1920, 1080), pv.fps, pv.shutter, pv.blur_samples) == ((960, 540), 30.0, 0, 1)
    assert (fi.size(1920, 1080), fi.fps, fi.shutter, fi.blur_samples) == (
        (1920, 1080),
        None,
        0.5,
        4,
    )
    assert (k4.size(1280, 720), k4.fps) == ((3840, 2160), None)
    for p in (pv, fi, k4):
        assert p.effects == FULL and p.life and not p.dedup and p.encoder == "quality"
    d = PRESETS["draft"]
    assert d.fps == 15.0 and d.effects == DRAFT and not d.life and d.dedup and d.encoder == "fast"
    assert (d.shutter, d.blur_samples) == (0.0, 1)
    # the job's motion-blur table is read from the presets
    from phasor.render.job import BLUR

    assert BLUR["preview"] == (0.0, 1) and BLUR["final"] == (0.5, 4) == BLUR["4k"]


@pytest.mark.parametrize(
    ("scene", "size"),
    [((1920, 1080), (640, 360)), ((1080, 1920), (360, 640)), ((1000, 1000), (640, 640)),
     ((320, 180), (320, 180)), ((1001, 501), (640, 320)), ((8000, 2), (640, 2)),
     ((2, 2), (2, 2))],
)  # fmt: skip
def test_a_draft_keeps_the_scenes_aspect_with_even_sides(scene, size):
    w, h = PRESETS["draft"].size(*scene)
    assert (w, h) == size and w % 2 == 0 and h % 2 == 0 and max(w, h) <= 640


@pytest.mark.parametrize("bad", ["huge", "", "Draft", None, 3, ["draft"]])
def test_a_bad_preset_name_is_a_clear_error(bad):
    with pytest.raises(ValueError, match="preset must be one of"):
        preset_named(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, -0.5, 1000.5, 1e9, "x", None])
def test_a_bad_frame_rate_is_a_clear_error(bad):
    with pytest.raises(ValueError, match="fps"):
        check_fps(bad)
    assert check_fps(15) == 15.0 and check_fps(1000) == 1000.0 and check_fps(0.5) == 0.5


def test_doctor_lists_the_presets():
    from phasor.render.doctor import DoctorReport, format_report

    r = DoctorReport([], "C", None)
    text = format_report(r)
    for name in PRESETS:
        assert f"    {name} " in text
    assert "draft    long side 640" in text and "look: plain" in text
    assert [p["name"] for p in r.to_dict()["presets"]] == list(PRESETS)


# ---------------------------------------------------------------- the scene a draft draws


def test_a_draft_draws_the_scene_without_its_decoration():
    e = _scene().engine
    assert e.document.meta.life == "lively"
    plain = e.without_decoration()
    assert plain is not e and plain.document.meta.life == "off"
    assert e.document.meta.life == "lively"  # the open scene is untouched
    assert e.without_decoration() is plain  # made once per version of the document
    # life off: a resting dot is the same at every moment, with life it breathes
    assert plain.state_at(1.5)["p"].style == plain.state_at(2.5)["p"].style
    assert e.state_at(1.5)["p"].style != e.state_at(2.5)["p"].style
    # what the document animates still moves
    assert plain.state_at(0.5)["p"].params.pos != plain.state_at(1.5)["p"].params.pos
    assert plain.without_decoration() is plain  # nothing left to strip
    e.apply({"op": "set_param", "path": "t.text", "value": "changed"})
    again = e.without_decoration()
    assert again is not plain and again.document.object("t").params["text"] == "changed"


def test_the_themes_camera_drift_is_decoration_an_explicit_one_is_not():
    s = Scene("orbit", look="none", passes=None)
    s.use_camera("orbit", azimuth=30.0)
    s.add(Dot(id="d"))
    cam = s.engine.document.timeline.camera
    plain = s.engine.without_decoration()
    assert plain.document.camera(cam).params["drift"] == 0.0
    assert plain.state_at(2.0).camera.eye.tolist() == plain.state_at(7.0).camera.eye.tolist()
    kept = Scene("orbit2", look="none", passes=None)
    kept.use_camera("orbit", azimuth=30.0, drift=5.0)
    kept.add(Dot(id="d"))
    assert kept.engine.without_decoration().document.camera(cam).params["drift"] == 5.0


def test_a_scene_with_nothing_to_strip_is_drawn_as_it_is():
    s = Scene("still", look="none", passes=None, life="off")
    s.add(Dot(id="d"))
    assert s.engine.without_decoration() is s.engine


def test_a_saved_scene_with_tweaks_keeps_them_in_its_draft(tmp_path):
    s = _scene()
    s.save(tmp_path / "x.phasor.json")
    e = load(tmp_path / "x.phasor.json", passes=None)
    e.apply({"op": "set_param", "path": "p.r", "value": 0.4})
    assert e.without_decoration().state_at(2.0)["p"].params.r == 0.4


# ---------------------------------------------------------------- the frame key


def test_the_digest_compares_values_exactly():
    a = {"x": np.arange(6.0).reshape(2, 3), "y": (1, 2.5, "s")}
    assert framekey.digest(a) == framekey.digest(
        {"y": (1, 2.5, "s"), "x": np.arange(6.0).reshape(2, 3)}
    )
    assert framekey.digest(a) != framekey.digest({**a, "x": np.arange(6.0)})  # shape counts
    assert framekey.digest(np.zeros(3, np.float32)) != framekey.digest(np.zeros(3, np.float64))
    assert framekey.digest(0.0) != framekey.digest(-0.0)
    assert framekey.digest(1) != framekey.digest(1.0) != framekey.digest("1")
    assert framekey.digest((1, 2)) != framekey.digest(((1, 2),))
    assert framekey.digest("ab", "c") != framekey.digest("a", "bc")

    def f():
        return 1

    assert framekey.digest(f) != framekey.digest(f)  # unknown: never equal, so drawn again


def _plans(engine, times, size=(160, 90), fog=False):
    """Frame keys without OpenGL: ``frame_plan`` only records draw calls."""
    from phasor.render.passes import GLRenderer

    r = GLRenderer.__new__(GLRenderer)
    return [r.frame_plan(engine.state_at(t), size, fog=fog).key for t in times]


def test_a_still_moment_has_one_key_and_motion_changes_it():
    plain = _scene().engine.without_decoration()
    k = _plans(plain, [0.2, 0.6, 1.2, 1.8, 2.6])
    assert k[0] != k[1] != k[2]  # moving
    assert k[2] == k[3] == k[4]  # at rest
    a, b = _plans(plain, [2.0], (160, 90)), _plans(plain, [2.0], (320, 180))
    assert a != b  # the size is in the key
    with_fog = _plans(plain, [2.0], fog=True)
    assert with_fog != a


def test_a_type_that_draws_from_the_clock_is_never_repeated():
    """``FlowParticles`` reads ``state.t`` in ``draw`` with constant parameters: hashing the
    state without the clock would freeze it. The key is what ``draw`` recorded, so it moves."""
    from phasor import geo

    s = Scene("flow", look="none", passes=None, life="off")
    s.add(geo.FlowParticles(id="f", fp="-y", fq="x", count=40))
    k = _plans(s.engine, [1.0, 1.1, 1.2])
    assert len(set(k)) == 3


def test_living_glow_is_never_repeated():
    e = _scene().engine  # life on (what preview and final draw)
    k = _plans(e, [1.5, 1.6, 1.7])
    assert len(set(k)) == 3


# ---------------------------------------------------------------- encoder settings


def test_the_fast_encoder_settings_and_the_quality_ones_unchanged(monkeypatch, tmp_path):
    from phasor.render import encode

    assert encode.fast_options("libx264", 15) == {"preset": "ultrafast", "crf": "26", "bf": "0",
                                                   "g": "15"}  # fmt: skip
    assert encode.fast_options("h264_nvenc", 29.97)["g"] == "30"
    assert encode.fast_options("prores_ks", 15) == {}
    seen = []

    class Proc:
        stdin = None

        def __init__(self, cmd, **kw):
            seen.append((cmd, kw))

    monkeypatch.setattr(encode.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(encode.subprocess, "Popen", Proc)
    encode.FFmpegWriter(tmp_path / "a.mp4", 64, 48, 30.0, "libx264", "yuv420p")
    encode.FFmpegWriter(tmp_path / "a.mp4", 64, 48, 15.0, "libx264", "yuv420p", "fast")
    quality, fast = seen[0][0], seen[1][0]
    assert quality == ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", "64x48", "-r", "30/1", "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "a.mp4")]  # fmt: skip
    assert fast[fast.index("-c:v") + 2:fast.index("-pix_fmt", 9)] == [
        "-preset", "ultrafast", "-crf", "26", "-bf", "0", "-g", "15"]  # fmt: skip
    assert all(kw["creationflags"] == encode.NO_WINDOW for _, kw in seen)
    with pytest.raises(ValueError, match="speed"):
        encode.open_writer(tmp_path / "x.mp4", 64, 48, 15.0, speed="turbo")


def test_a_fast_video_has_no_b_frames_and_a_keyframe_a_second(tmp_path):
    av = pytest.importorskip("av")
    from phasor.render.encode import PyAVWriter

    w = PyAVWriter(tmp_path / "f.mp4", 64, 48, 15.0, "libx264", "fast")
    for k in range(40):
        w.write(np.full((48, 64, 3), 5 * k, np.uint8))
    w.close()
    with av.open(str(tmp_path / "f.mp4")) as c:
        frames = [(str(f.pict_type).rsplit(".", 1)[-1], f.key_frame) for f in c.decode(video=0)]
    assert len(frames) == 40 and not [t for t, _ in frames if t.upper() == "B"]
    assert [k for k, (_, key) in enumerate(frames) if key] == [0, 15, 30]


# ---------------------------------------------------------------- surfaces (no OpenGL needed)


def test_the_cli_has_a_draft_preset_and_a_preview_command():
    p = build_parser()
    a = p.parse_args(["render", "x.py", "--preset", "draft", "--start", "intro", "--end", "2.5"])
    assert (a.preset, a.start, a.end) == ("draft", "intro", "2.5")
    a = p.parse_args(["preview", "x.py"])
    assert a.preset == "draft" and a.command == "preview"
    assert p.parse_args(["render", "x.py"]).preset == "preview"  # as before
    a = p.parse_args(["bench", "--frames", "x", "--preset", "draft"])
    assert a.preset == "draft"
    with pytest.raises(SystemExit):
        p.parse_args(["render", "x.py", "--preset", "huge"])


def test_a_draft_never_overwrites_the_real_render_by_default():
    from phasor.cli import default_out

    assert default_out("a/intro.py", "draft", False) == "intro.draft.mp4"
    assert default_out("a/intro.phasor.json", "draft", True) == "intro.draft.png"
    assert default_out("a/intro.py", "preview", False) == "intro.mp4"
    assert default_out("a/intro.py", "final", True) == "intro.png"


def test_a_render_window_takes_seconds_or_markers():
    from phasor.render.job import resolve_time

    e = _scene().engine
    assert resolve_time(e, None) is None
    assert resolve_time(e, "rest") == 1.0 and resolve_time(e, "1.25") == 1.25
    assert resolve_time(e, 2) == 2.0 and resolve_time(e, -5.0) == -5.0  # kept: it always rendered
    for bad in ["nowhere", "nan", "inf", float("nan"), float("-inf"), True, [1.0], {}]:
        with pytest.raises(ValueError):
            resolve_time(e, bad)
    with pytest.raises(ValueError, match="markers: end, go, rest"):
        resolve_time(e, "nowhere", "start")


@pytest.mark.parametrize("kw", [{"preset": "huge"}, {"fps": -3}, {"fps": float("nan")},
                                {"start": "nowhere"}, {"end": float("inf")},
                                {"frame": "nowhere"}])  # fmt: skip
def test_bad_requests_fail_before_anything_is_written(tmp_path, kw):
    e = _scene().engine
    out = tmp_path / "x.mp4"
    with pytest.raises(ValueError):
        e.render(out, **{"preset": "draft", **kw})
    assert not out.exists()


def test_the_workspace_render_tool_knows_draft(tmp_path):
    from phasor.workspace.tools import ToolError, Workspace

    ws = Workspace(tmp_path)
    ws.scene_new("tool draft")
    for bad in ["huge", None, 3]:
        with pytest.raises(ToolError, match="preset"):
            ws.render(preset=bad)
    with pytest.raises(ToolError, match="nowhere"):
        ws.render(preset="draft", start="nowhere")
    assert "draft" in Workspace.render.__doc__


def test_scene_render_checks_its_arguments_first(tmp_path):
    s = _scene()
    for bad in [2.5, True, "2", [2]]:
        with pytest.raises(ValueError, match="workers"):
            s.render(tmp_path / "x.mp4", preset="draft", workers=bad)
    with pytest.raises(ValueError, match="preset"):
        s.render(tmp_path / "x.mp4", preset="nope")


# ---------------------------------------------------------------- rendering (OpenGL)


def _frames(folder: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(folder.glob("frame_*.png"))]


@pytest.mark.gl
@needs_gl
def test_a_draft_is_small_quick_and_plain(tmp_path):
    e = _scene().engine
    r = e.render(tmp_path / "d", preset="draft")
    assert r["size"] == (640, 360) and r["fps"] == 15.0 and r["frames"] == 45
    assert not r["motion_blur"] and r["preset"] == "draft"
    assert r["repeated"] >= 25 and r["drawn"] + r["repeated"] == 45  # two still seconds
    assert len(_frames(tmp_path / "d")) == 45
    # the plain look: no halo around the glowing dot, the paint the same
    plain = e.without_decoration()
    full = plain.frame(2.0, (640, 360), output="srgb8")
    draft = plain.frame(2.0, (640, 360), output="srgb8", effects=DRAFT)
    assert draft.shape == full.shape == (360, 640, 3)
    assert draft.sum() < full.sum()  # no bloom, no lens
    # paint is tone-mapped once and unchanged: the corner is the background exactly (no
    # vignette darkens it), as the theme defines it
    from phasor.core.color import linear_to_srgb

    bg = np.round(linear_to_srgb(np.asarray(plain.state_at(2.0).background[:3])) * 255)
    assert draft[0, 0].tolist() == bg.astype(int).tolist()


@pytest.mark.gl
@needs_gl
def test_repeated_frames_are_byte_identical_to_drawing_every_frame(tmp_path):
    from phasor.render.job import serial_frames

    plain = _scene().engine.without_decoration()
    times = [k / 15 for k in range(45)]
    counts: dict[str, int] = {}
    fast = list(serial_frames(plain, times, (320, 180), 15.0, 0.0, 1, effects=DRAFT, dedup=True,
                              counts=counts))  # fmt: skip
    slow = list(serial_frames(plain, times, (320, 180), 15.0, 0.0, 1, effects=DRAFT, dedup=False))
    assert counts["repeated"] >= 25  # the test means something only if frames were repeated
    assert all(a.tobytes() == b.tobytes() for a, b in zip(fast, slow, strict=True))


@pytest.mark.gl
@needs_gl
def test_a_clock_driven_type_is_drawn_every_frame(tmp_path):
    from phasor import geo
    from phasor.render.job import serial_frames

    s = Scene("flow", look="none", life="off")
    s.add(geo.FlowParticles(id="f", fp="-y", fq="x", count=60))
    s.marker("end", 1.0)
    counts: dict[str, int] = {}
    out = list(serial_frames(s.engine, [0.5, 0.6, 0.7], (160, 90), 15.0, 0.0, 1, effects=DRAFT,
                             dedup=True, counts=counts))  # fmt: skip
    assert counts == {"drawn": 3, "repeated": 0}
    assert len({f.tobytes() for f in out}) == 3


@pytest.mark.gl
@needs_gl
def test_a_parallel_draft_equals_a_serial_one(tmp_path):
    from phasor.scene import render

    e = _scene().engine
    serial = render(e, tmp_path / "one", preset="draft", workers=1)
    many = render(e, tmp_path / "two", preset="draft", workers=2)
    assert serial["frames"] == many["frames"] == 45
    assert many["repeated"] == serial["repeated"] >= 25
    a, b = _frames(tmp_path / "one"), _frames(tmp_path / "two")
    assert len(a) == 45 and a == b


@pytest.mark.gl
@needs_gl
def test_a_draft_starts_at_a_marker(tmp_path):
    e = _scene().engine
    r = e.render(tmp_path / "w", preset="draft", start="rest", end="end")
    assert r["frames"] == 30 and r["drawn"] == 1 and r["repeated"] == 29  # all at rest
    r = e.render(tmp_path / "still.png", preset="draft", frame="rest")
    assert r["size"] == (640, 360) and r["t"] == 1.0 and (tmp_path / "still.png").exists()


@pytest.mark.gl
@needs_gl
@pytest.mark.parametrize("target", ["v.mp4", "seq"])
def test_a_cancelled_draft_reports_progress_and_removes_what_it_wrote(tmp_path, target):
    from phasor.render.job import RenderCancelled

    e = _scene().engine
    stop, seen = threading.Event(), []
    (tmp_path / "seq").mkdir()
    keep = tmp_path / "seq" / "mine.txt"
    keep.write_text("not the render's", encoding="utf-8")

    def progress(k, n):
        seen.append((k, n))
        if k == 5:
            stop.set()

    with pytest.raises(RenderCancelled, match="5 of 45"):
        e.render(tmp_path / target, preset="draft", progress=progress, cancel=stop)
    assert seen == [(k, 45) for k in range(1, 6)]
    if target == "seq":
        assert not list((tmp_path / "seq").glob("frame_*.png")) and keep.exists()
    else:
        assert not (tmp_path / target).exists()
    # a callable works too, and a render cancelled before it starts writes nothing
    with pytest.raises(RenderCancelled):
        e.render(tmp_path / "never.mp4", preset="draft", cancel=lambda: True)
    assert not (tmp_path / "never.mp4").exists()


@pytest.mark.gl
@needs_gl
def test_preview_and_final_are_not_touched_by_the_draft_path(tmp_path):
    """A frame rendered the old way (``engine.frame``) and through the job are the same bytes,
    and the preview still has its halo (bloom), lens and life."""
    from phasor.render.job import render_frame

    e = _scene().engine
    old = e.frame(1.5, (320, 180), output="srgb8")
    assert render_frame(e, 1.5, 30.0, (320, 180), 0.0, 1).tobytes() == old.tobytes()
    assert e.frame(1.5, (320, 180), output="srgb8", effects=FULL).tobytes() == old.tobytes()
    r = e.render(tmp_path / "p", preset="preview", fps=6.0)
    assert r["repeated"] == 0 and r["drawn"] == r["frames"] == 18


DRAFT_SCENES = [("prism_primes", 9.0), ("basics_3d", 6.0), ("prism_domain", 4.8)]


@pytest.mark.gl
@needs_gl
def test_a_draft_frame_is_recognisably_the_final_frame():
    """The draft frame against the final frame downscaled to the draft's size: structurally
    similar (SSIM over a generous threshold). Prism frames are mostly near-black, where any
    two frames look alike, so the same is checked over the windows with content (``busy``):
    there the draft must be far more like its final than a blank frame or another scene's
    final frame is (measured 2026-11-24: 0.44-0.69 against at most 0.30)."""
    from PIL import Image

    from phasor.render.job import engine_for, render_frame
    from tests.ssim import ssim

    p = PRESETS["draft"]
    drafts, finals = {}, {}
    for name, t in DRAFT_SCENES:
        e = load(ROOT / "examples" / f"{name}.py")
        m = e.document.meta
        size = p.size(m.width, m.height)
        drafts[name] = render_frame(engine_for(e, "draft"), t, 15.0, size, 0.0, 1, p.effects)
        final = e.frame(t, (m.width, m.height), output="srgb8")
        finals[name] = np.asarray(Image.fromarray(final).resize(size, Image.BOX))
    for name, _ in DRAFT_SCENES:
        d, f = drafts[name], finals[name]
        assert ssim(d, f) >= 0.75, (name, ssim(d, f))
        blank = np.empty_like(d)
        blank[:] = d[0, 0]
        same = ssim(d, f, busy=20.0)
        others = [ssim(d, finals[o], busy=20.0) for o, _ in DRAFT_SCENES if o != name]
        assert same >= 0.35, (name, same)
        assert same >= ssim(blank, f, busy=20.0) + 0.2, (name, same, ssim(blank, f, busy=20.0))
        assert same >= max(others) + 0.2, (name, same, others)


@pytest.mark.gl
@needs_gl
def test_golden_draft_still():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = load(ROOT / "examples" / "prism_primes.py")
    path = GOLDEN / e.pass_runner().renderer_class() / "draft_prism_primes.png"
    still = e.render(path.with_suffix(".new.png"), preset="draft", frame=9.0)
    got = np.asarray(Image.open(still["path"]).convert("RGB"))
    Path(still["path"]).unlink()
    assert got.shape == (360, 640, 3)
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, got)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(got, np.asarray(Image.open(path).convert("RGB"))) >= 0.99


@pytest.mark.gl
@needs_gl
def test_golden_draft_storyboard():
    """Storyboard snapshot: draft frames of the example scene on one contact sheet, against a
    golden; and the inspector agrees with the draft scene (the dot rests, no life)."""
    from PIL import Image

    from phasor.render.job import engine_for, render_frame
    from phasor.render.output import save_png
    from phasor.render.storyboard import contact_sheet
    from tests.ssim import ssim

    e = engine_for(_scene().engine, "draft")
    cells = [(f"t = {t:g} s", render_frame(e, t, 15.0, (320, 180), 0.0, 1, DRAFT))
             for t in (0.0, 0.5, 1.0, 2.5)]  # fmt: skip
    sheet = contact_sheet(cells, 2)
    path = GOLDEN / e.pass_runner().renderer_class() / "draft_storyboard.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.99
    assert cells[2][1].tobytes() == render_frame(e, 2.5, 15.0, (320, 180), 0.0, 1, DRAFT).tobytes()
    assert e.where("p", 2.5)["center"] == e.where("p", 1.0)["center"]  # at rest, no life
    assert e.where("p", 0.5)["center"] != e.where("p", 1.0)["center"]


@pytest.mark.gl
@needs_gl
def test_old_worker_counts_still_render_and_ctrl_c_cleans_up(tmp_path, capsys):
    """``--workers 0`` rendered in this process before Milestone 21 and still does; Ctrl-C
    (KeyboardInterrupt) removes the partial file like a cancel."""
    scene = tmp_path / "tiny.py"
    scene.write_text("from phasor import Dot, Scene\ns = Scene('tiny')\ns.add(Dot(id='d'))\n"
                     "s.marker('end', 0.4)\n", encoding="utf-8")  # fmt: skip
    out = tmp_path / "w0.mp4"
    assert main(["preview", str(scene), "--workers", "0", "-o", str(out), "--quiet"]) == 0
    assert out.exists() and "6 frames" in capsys.readouterr().out

    def interrupt(k, n):
        if k == 2:
            raise KeyboardInterrupt

    e = _scene().engine
    with pytest.raises(KeyboardInterrupt):
        e.render(tmp_path / "ctrl_c.mp4", preset="draft", progress=interrupt)
    assert not (tmp_path / "ctrl_c.mp4").exists()


def test_the_cli_reports_a_bad_window_cleanly(tmp_path, capsys):
    scene = tmp_path / "tiny.py"
    scene.write_text("from phasor import Dot, Scene\ns = Scene('tiny')\ns.add(Dot(id='d'))\n",
                     encoding="utf-8")  # fmt: skip
    assert main(["preview", str(scene), "--start", "nowhere", "-o", str(tmp_path / "x.mp4")]) == 1
    assert (
        "phasor preview: start 'nowhere' is neither seconds nor a marker" in capsys.readouterr().err
    )


def test_engine_render_is_the_job(monkeypatch, tmp_path):
    import phasor.render.job as job

    calls = []
    monkeypatch.setattr(job, "render_file", lambda e, out, **kw: calls.append((e, out, kw)) or {})
    e = Engine(passes=None)
    e.render(tmp_path / "a.mp4", preset="draft", start="x")
    assert calls == [(e, tmp_path / "a.mp4", {"preset": "draft", "start": "x"})]
