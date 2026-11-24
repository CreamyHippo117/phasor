"""Milestone 24: video and picture inputs (STUDIO.md).

``Video`` plays a clip from the scene's folder; ``Image`` also plays a picture sequence
(``frames/shot_%04d.png``); ``Scene.background`` and ``Scene.from_video`` make either the
whole frame. One frame source (``domains/media_math.py``) serves both. The frame shown is a
pure function of (document, t, file): any order, any process, the same picture. Hostile files
are lint issues that draw nothing, never a traceback, hang or memory blow-up.

Clips are made here with PyAV (the ``encode`` extra; those tests skip without it) and
pictures with Pillow, in ``tmp_path``.
"""

from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from phasor import Image, Scene, Text
from phasor.core import files
from phasor.core.issues import DocumentError
from phasor.domains import media_math as mm
from tests.conftest import gl_available

ROOT = Path(__file__).parents[1]
MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]
FPS = 24


@pytest.fixture(autouse=True)
def _fresh():
    """Every test starts with no open decoder and nothing cached (and leaves none open: on
    Windows an open file cannot be removed with ``tmp_path``)."""
    mm.close_all()
    yield
    mm.close_all()


def _color(i: int) -> tuple[int, int, int]:
    """Frame ``i``'s colour: red counts frames in steps of 10, so a picture says its frame."""
    return (10 * i % 250, 200 - (7 * i % 150), 90)


def _which(rgb) -> int:
    """The frame a decoded colour came from (``_color``)."""
    return round(float(rgb[0]) / 10)


def _clip(path: Path, n: int = 24, fps: int = FPS, w: int = 64, h: int = 36,
          faststart: bool = False) -> Path:  # fmt: skip
    """A clip of ``n`` solid frames (``_color``), H.264 with B-frames where PyAV has x264."""
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = {"movflags": "faststart"} if faststart else {}
    with av.open(str(path), "w", options=opts) as c:
        for codec in ("libx264", "mpeg4"):
            try:
                s = c.add_stream(codec, rate=fps)
                break
            except Exception:
                continue
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        if s.codec_context.name in ("libx264", "h264"):
            s.options = {"bf": "2", "g": "8", "crf": "8"}
        for i in range(n):
            a = np.zeros((h, w, 3), np.uint8)
            a[:] = _color(i)
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def _png(path: Path, rgba=(255, 0, 0, 255), w: int = 8, h: int = 6) -> Path:
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.zeros((h, w, 4), np.uint8)
    a[:] = rgba
    PILImage.fromarray(a, "RGBA").save(path)
    return path


def _frames(folder: Path, n: int, first: int = 0, pattern: str = "f_%03d.png") -> str:
    for i in range(first, first + n):
        _png(folder / (pattern % i), (*_color(i), 255))
    return f"{folder.name}/{pattern}"


def _scene(tmp_path, passes=None, **kw) -> Scene:
    s = Scene("footage", passes=passes, base_dir=tmp_path, **kw)
    s.set("camera.height", 4.5)
    return s


def _sem(s: Scene, t: float, oid: str = "v") -> dict:
    return s.engine.inspect(t, id=oid)["objects"][0]["semantic"]


def _codes(s: Scene) -> set[str]:
    return {i.code for i in s.engine.lint()}


# ---------------------------------------------------------------- 1. files: sequences


def test_a_sequence_names_its_frames_from_the_first_of_0_to_4_to_the_first_gap(tmp_path):
    assert files.is_sequence("frames/shot_%04d.png") and files.is_sequence("a_%d.png")
    for not_one in ("a.png", "100%.png", "a_%04d_%04d.png", "%04d/a.png", "a_%s.png"):
        assert not files.is_sequence(not_one), not_one
    _frames(tmp_path / "fr", 4, first=2)
    _png(tmp_path / "fr" / "f_007.png")  # after a gap: not part of it
    names = files.members(tmp_path, "fr/f_%03d.png")
    assert names == ("fr/f_002.png", "fr/f_003.png", "fr/f_004.png", "fr/f_005.png")
    assert [p.name for p in files.sequence(tmp_path, "fr/f_%03d.png")] == [
        "f_002.png", "f_003.png", "f_004.png", "f_005.png"]  # fmt: skip
    assert files.exists(tmp_path, "fr/f_%03d.png")
    assert files.problem(tmp_path, "fr/f_%03d.png") is None
    assert "no frames" in files.problem(tmp_path, "fr/g_%03d.png")
    assert files.members(tmp_path, "plain.png") == ("plain.png",)


def test_a_sequence_stamp_changes_with_any_frame_and_is_the_same_in_every_process(tmp_path):
    """The digest is hashlib's (Python's ``hash`` of text changes per process: a render
    worker comparing stamps would refuse every payload)."""
    import subprocess

    from phasor.core.proc import NO_WINDOW

    _frames(tmp_path / "fr", 3)
    a = files.stamp(tmp_path, "fr/f_%03d.png")
    assert a[1].startswith("seq:") and a[2] == 3
    code = ("import sys; from phasor.core import files; "
            "print(repr(files.stamp(sys.argv[1], 'fr/f_%03d.png')))")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True,
                         text=True, check=True, creationflags=NO_WINDOW,
                         encoding="utf-8")  # fmt: skip
    assert out.stdout.strip() == repr(a)
    f = tmp_path / "fr" / "f_001.png"
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert files.stamp(tmp_path, "fr/f_%03d.png") != a
    _png(tmp_path / "fr" / "f_003.png")  # one more frame
    assert files.stamp(tmp_path, "fr/f_%03d.png")[2] == 4


def test_a_link_out_of_the_folder_ends_a_sequence_and_is_never_read(tmp_path):
    outside = _png(tmp_path / "secret.png", (0, 255, 0, 255))
    root = tmp_path / "scene"
    _frames(root / "fr", 2)
    try:
        os.symlink(outside, root / "fr" / "f_002.png")
    except (OSError, NotImplementedError):
        pytest.skip("this system cannot make symbolic links")
    assert len(files.sequence(root, "fr/f_%03d.png")) == 2


def test_a_long_sequence_stops_at_the_cap_and_stamping_it_stays_cheap(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "MAX_SEQUENCE", 40)
    folder = tmp_path / "fr"
    folder.mkdir()
    src = _png(tmp_path / "one.png")
    for i in range(60):
        (folder / f"f_{i:03d}.png").write_bytes(src.read_bytes())
    assert len(files.sequence(tmp_path, "fr/f_%03d.png")) == 41  # one past the cap: "more"
    s = _scene(tmp_path)
    s.add(Image(id="v", path="fr/f_%03d.png", width=2.0))
    assert "too_big" in _sem(s, 0.0) and "video_too_big" in _codes(s)


# ---------------------------------------------------------------- 2. time and placement


def test_clip_time_follows_start_trim_speed_and_loop():
    kw = {"start": 1.0, "trim": (0.0, 0.0), "speed": 1.0, "loop": False, "seconds": 2.0}
    assert mm.clip_time(0.5, **kw) == (0.0, False)  # before start: the first frame
    assert mm.clip_time(1.5, **kw) == (0.5, False)
    assert mm.clip_time(3.5, **kw) == (2.0, True)  # past the end: held, ended
    assert mm.clip_time(3.5, **{**kw, "loop": True}) == pytest.approx((0.5, False))
    assert mm.clip_time(1.5, **{**kw, "speed": 2.0}) == (1.0, False)
    trimmed = {**kw, "trim": (0.5, 1.5)}
    assert mm.clip_time(1.25, **trimmed) == (0.75, False)
    assert mm.clip_time(9.0, **trimmed) == (1.5, True)
    assert mm.clip_time(9.0, **{**trimmed, "loop": True})[0] == pytest.approx(0.5 + 8.0 % 1.0)
    assert mm.clip_range((0.5, 99.0), 2.0) == (0.5, 2.0)
    for bad in (float("nan"), float("inf")):  # from a track or an expression: held, no raise
        assert mm.clip_time(bad, **kw) == (0.0, False)
        assert mm.clip_time(1.0, **{**kw, "speed": bad}) == (0.0, False)


def test_the_frame_shown_is_the_last_one_at_or_before_the_time():
    times = np.arange(48) / 24
    for k in range(48):  # t = k / fps shows frame k exactly, in float
        assert mm.frame_index(times, k / 24) == k
        assert mm.frame_index(times, k / 24 + 0.5 / 24) == k
    assert mm.frame_index(times, -1.0) == 0 and mm.frame_index(times, 99.0) == 47
    assert mm.frame_index(times, 1.0, ended=True) == 23  # an out point at frame 24: frame 23
    ntsc = np.arange(30) * 1001 / 30000
    assert all(mm.frame_index(ntsc, k * 1001 / 30000) == k for k in range(30))


def test_fit_contains_covers_stretches_or_keeps_the_pixel_size():
    box, src = (4.0, 2.0), (100, 100)
    assert mm.fit_box("stretch", box, src) == (4.0, 2.0, mm.FULL)
    assert mm.fit_box("contain", box, src) == (2.0, 2.0, mm.FULL)
    w, h, crop = mm.fit_box("cover", box, src)
    assert (w, h) == box and crop == pytest.approx((0.0, 0.25, 1.0, 0.75))
    w, h, crop = mm.fit_box("none", box, src, native=0.01)  # 1 x 1 world units: fits
    assert (w, h, crop) == (1.0, 1.0, mm.FULL)
    w, h, crop = mm.fit_box("none", (0.5, 2.0), src, native=0.01)  # cut to the box
    assert (w, h) == (0.5, 1.0) and crop == pytest.approx((0.25, 0.0, 0.75, 1.0))


def test_sampling_bytes_converts_only_what_it_reads_and_matches_a_full_conversion():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (90, 160, 4), dtype=np.uint8)
    frame[..., 3] = 255
    full = mm.linear(frame, premultiply=False)
    for pw, ph, center, height in ((40, 20, (2.0, 1.0), 1.0), (8, 4, (1.0, 0.6), 0.5),
                                   (300, 150, (2.0, 1.125), 2.25)):  # fmt: skip
        quad = (4.0, 2.25)
        args = (pw, ph, center, height, quad, mm.FULL, np.zeros(3, np.float32))
        a = mm.sample(frame, *args, opaque=True, glow=False)
        b = mm.sample(full, *args, opaque=True, glow=False)
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_straight_alpha_is_premultiplied_so_edges_take_no_colour_from_clear_pixels():
    """Half the picture clear with red stored under it, half opaque blue: the edge between
    them is blue over the background, never purple (straight-alpha filtering bleeds red)."""
    frame = np.zeros((4, 8, 4), np.uint8)
    frame[:, :4] = (255, 0, 0, 0)  # clear, but red underneath
    frame[:, 4:] = (0, 0, 255, 255)
    bg = np.zeros(3, np.float32)
    out = mm.sample(frame, 16, 8, (2.0, 1.0), 2.0, (4.0, 2.0), mm.FULL, bg, opaque=False,
                    glow=False)  # fmt: skip
    assert out[..., 0].max() < 1e-6  # no red anywhere
    assert 0.1 < out[4, 7, 2] < 0.9  # the edge pixel: some blue, over the black background
    assert out[4, 12, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------- 3. decoding a clip


def test_a_clip_probes_its_frames_without_decoding_them_all(tmp_path):
    _clip(tmp_path / "c.mp4", n=30, fps=24)
    probe = mm.open_source(tmp_path, "c.mp4", video=True).probe()
    assert probe.ok and probe.frames == 30 and (probe.width, probe.height) == (64, 36)
    assert probe.fps == pytest.approx(24) and probe.seconds == pytest.approx(30 / 24)


def test_every_frame_is_the_same_decoded_in_order_out_of_order_or_afresh(tmp_path):
    """Nearest-previous by presentation time, whatever way the decoder got there: in order
    (decoding on, no seek), shuffled (seeking), and in a new decoder."""
    _clip(tmp_path / "c.mp4", n=24)
    src = mm.open_source(tmp_path, "c.mp4", video=True)
    in_order = [np.array(src.frame(i)) for i in range(24)]
    assert [_which(f[5, 5]) for f in in_order] == list(range(24))
    for _ in range(2):
        mm.close_all()
        src = mm.open_source(tmp_path, "c.mp4", video=True)
        for i in np.random.default_rng(len(in_order)).permutation(24):
            assert np.array_equal(src.frame(int(i)), in_order[i]), i
    r, g, b = in_order[10][5, 5, :3]  # colours come through within a few codes
    assert abs(int(r) - 100) <= 4 and abs(int(g) - 130) <= 4 and abs(int(b) - 90) <= 4


def _open_gop(path: Path, n: int = 40) -> Path:
    """An open-GOP H.264 clip (``open-gop=1``, a keyframe every 8, 3 B-frames, pyramid): after
    each keyframe but the first come B-frames shown before it that refer to the GOP before
    (leading frames). A square moves so no two frames are alike. Skips without libx264."""
    av = pytest.importorskip("av")
    try:
        av.codec.Codec("libx264", "w")
    except Exception:
        pytest.skip("PyAV here has no libx264 (open GOPs are an x264 option)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = 64, 36, "yuv420p"
        s.options = {"crf": "8", "x264-params": "open-gop=1:keyint=8:min-keyint=8:scenecut=0:"
                     "bframes=3:b-adapt=0:b-pyramid=normal"}  # fmt: skip
        for i in range(n):
            a = np.zeros((36, 64, 3), np.uint8)
            a[:] = _color(i)
            a[4:12, (3 * i) % 56 : (3 * i) % 56 + 8] = 255
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    with av.open(str(path)) as c:
        packets = [(p.pts, p.is_keyframe) for p in c.demux(video=0) if p.size]
    leading = [q for k, (pts, key) in enumerate(packets) if key and k
               for q, _ in packets[k + 1 : k + 4] if q < pts]  # fmt: skip
    assert leading, "the encoder made no leading frames: the clip does not test open GOPs"
    return path


def _full_decode(path: Path) -> dict[int, np.ndarray]:
    """Every frame of a clip decoded from the start, in order, by pts (the reference)."""
    import av

    with av.open(str(path)) as c:
        return {f.pts: f.to_ndarray(format="rgba") for f in c.decode(video=0)}


def test_an_open_gop_clip_gives_every_frame_the_same_in_any_order(tmp_path):
    """Leading B-frames of an open GOP are shown before their keyframe but decoded after it:
    seeking to the keyframe the demuxer finds for them (mp4 seeks by decode time) gave none,
    or another frame. Every frame now equals a decode from the start, in order, shuffled
    and afresh (Milestone 24 audit, B1)."""
    _open_gop(tmp_path / "og.mp4")
    ref = _full_decode(tmp_path / "og.mp4")
    src = mm.open_source(tmp_path, "og.mp4", video=True)
    probe = src.probe()
    assert probe.frames == 40 == len(ref)
    want = [ref[int(p)] for p in probe.pts]
    for i in range(40):
        assert np.array_equal(src.frame(i), want[i]), i
    for seed in range(3):
        mm.close_all()
        src = mm.open_source(tmp_path, "og.mp4", video=True)
        for i in np.random.default_rng(seed).permutation(40):
            got = src.frame(int(i))
            assert got is not None and np.array_equal(got, want[i]), i


def _edit_list(src: Path, dst: Path, cut: int) -> Path:
    """``src`` remuxed with its first ``cut`` frames before time 0: the mp4 gets an edit
    list, and those frames come out of the demuxer as discarded packets with negative times
    (what ``ffmpeg -ss 0.37 -i a.mp4 -c copy b.mp4`` makes)."""
    import av

    with av.open(str(src)) as i, av.open(str(dst), "w") as o:
        s = i.streams.video[0]
        t = o.add_stream_from_template(s)
        shift = cut * int(1 / (s.time_base * FPS))
        for p in i.demux(s):
            if p.dts is None:
                continue
            p.pts, p.dts, p.stream = p.pts - shift, p.dts - shift, t
            o.mux(p)
    with av.open(str(dst)) as c:
        assert any(p.is_discard for p in c.demux(video=0)), "no edit list was written"
    return dst


def test_a_clip_with_an_edit_list_starts_at_its_first_shown_frame(tmp_path):
    """Packets an mp4 edit list cuts are never shown: they are not frames of the clip, and
    its time 0 is the first frame shown (they made the clip ``bad_file``: audit, B2)."""
    _open_gop(tmp_path / "og.mp4")
    _edit_list(tmp_path / "og.mp4", tmp_path / "cut.mp4", 9)
    src = mm.open_source(tmp_path, "cut.mp4", video=True)
    probe = src.probe()
    assert probe.ok, probe.problem
    assert probe.frames == 31 and probe.times[0] == 0.0
    ref = _full_decode(tmp_path / "cut.mp4")
    assert _which(src.frame(0)[30, 5]) == 9  # the tenth frame encoded is the first shown
    for i in np.random.default_rng(0).permutation(31):
        assert np.array_equal(src.frame(int(i)), ref[int(probe.pts[i])]), i


def test_a_decoder_closed_by_another_thread_is_replaced_not_a_missing_frame(tmp_path,
                                                                            monkeypatch):  # fmt: skip
    """Another thread may close a decoder (too many open) between this thread taking it and
    locking it: the frame comes from another decoder, never None (audit, S2)."""
    _clip(tmp_path / "c.mp4", n=8)
    src = mm.open_source(tmp_path, "c.mp4", video=True)
    src.probe()
    real, calls = mm._decoder, []

    def evicted(*a, **k):
        d = real(*a, **k)
        if not calls:
            d.close()  # what an eviction does to it meanwhile
        calls.append(d)
        return d

    monkeypatch.setattr(mm, "_decoder", evicted)
    got = src.frame(5)
    assert got is not None and _which(got[5, 5]) == 5


def test_five_clips_played_together_each_decode_on_without_seeking(tmp_path, monkeypatch):
    """More clips than the old four decoders, and one file shown at two times: each keeps
    its own cursor, so playing on never seeks (each frame was a seek: audit, S1)."""
    for k in range(5):
        _clip(tmp_path / f"c{k}.mp4", n=48)
    srcs = [mm.open_source(tmp_path, f"c{k}.mp4", video=True) for k in range(5)]
    for s in srcs:
        s.probe()
    seeks = []
    real = mm._seek_points
    monkeypatch.setattr(mm, "_seek_points", lambda keys, t: seeks.append(t) or real(keys, t))
    monkeypatch.setattr(mm, "FRAME_CACHE_BYTES", 0)
    for i in range(1, 12):
        for s in srcs:
            s.frame(i)
        srcs[0].frame(i + 30)  # the same file 30 frames later: a second cursor
    assert len(seeks) == 6, seeks  # one to start each cursor
    assert len(mm._DECODERS) == 6 <= mm.MAX_OPEN


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this system")
def test_a_forked_process_decodes_after_its_parent_did(tmp_path):
    """A render worker forked after the parent decoded: swscale's thread pool, inherited
    mid-use, deadlocked the child's first conversion (audit, S4). Each decoder now converts
    with its own single-threaded scaler."""
    import multiprocessing

    _clip(tmp_path / "c.mp4", n=24, w=320, h=180)
    src = mm.open_source(tmp_path, "c.mp4", video=True)
    want = [np.array(src.frame(i)) for i in (3, 17)]
    mm.close_all()

    def child(q):
        s = mm.open_source(tmp_path, "c.mp4", video=True)
        q.put([np.array(s.frame(i)) for i in (3, 17)])

    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=child, args=(q,))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads: the point
        proc.start()
    try:
        got = q.get(timeout=60)
    finally:
        proc.join(5)
        if proc.is_alive():
            proc.kill()
    assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))


def test_playing_in_order_never_seeks_and_jumping_does(tmp_path, monkeypatch):
    _clip(tmp_path / "c.mp4", n=24)
    src = mm.open_source(tmp_path, "c.mp4", video=True)
    src.probe()
    seeks = []
    real = mm._seek_points  # called exactly when the decoder seeks

    def counting(keys, target):
        seeks.append(target)
        return real(keys, target)

    monkeypatch.setattr(mm, "_seek_points", counting)
    for i in range(1, 24):
        src.frame(i)
    assert len(seeks) == 1  # one seek to start, then straight on
    seeks.clear()
    mm._FRAMES.entries.clear()
    src.frame(3)
    assert len(seeks) == 1  # going back: a seek


def test_the_decoded_frames_cache_is_bounded_by_bytes(tmp_path, monkeypatch):
    _clip(tmp_path / "c.mp4", n=24, w=128, h=72)
    monkeypatch.setattr(mm, "FRAME_CACHE_BYTES", 3 * 128 * 72 * 4)
    src = mm.open_source(tmp_path, "c.mp4", video=True)
    for i in range(24):
        src.frame(i)
    assert mm._FRAMES.used <= 3 * 128 * 72 * 4 + 4 * 1024 and len(mm._FRAMES.entries) <= 3


def test_decoders_are_bounded_and_a_changed_file_closes_the_old_one(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MAX_OPEN", 2)
    for k in range(3):
        _clip(tmp_path / f"c{k}.mp4", n=4)
        mm.open_source(tmp_path, f"c{k}.mp4", video=True).frame(2)
    assert len(mm._DECODERS) == 2
    f = tmp_path / "c0.mp4"
    before = mm.open_source(tmp_path, "c0.mp4", video=True)
    before.frame(3)
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    after = mm.open_source(tmp_path, "c0.mp4", video=True)
    assert after.stamp != before.stamp and after.frame(3) is not None
    assert [k[0] for k in mm._DECODERS if k[0][0] == before.stamp[0]] == [after.stamp]


# ---------------------------------------------------------------- 4. the Video object


def test_a_video_plays_its_frames_against_the_scene_clock(tmp_path):
    from phasor import Video

    _clip(tmp_path / "c.mp4", n=24)
    s = _scene(tmp_path)
    s.add(Video(id="v", path="c.mp4", width=4.0, start=1.0, speed=2.0))
    s.marker("end", 4.0)
    sem = _sem(s, 1.25)
    assert sem["kind"] == "video" and sem["pixels"] == [64, 36] and sem["frames"] == 24
    assert sem["fps"] == pytest.approx(24) and sem["seconds"] == pytest.approx(1.0)
    assert sem["frame"] == 12 and sem["clip_time"] == pytest.approx(0.5)
    assert sem["looping"] is False and sem["trimmed"] is False and sem["ended"] is False
    assert sem["size"] == [4.0, 2.25] and not sem["missing"]
    assert _sem(s, 0.0)["frame"] == 0 and _sem(s, 3.0)["ended"]
    b = s.engine.state_at(0.0)["v"].impl.world_bounds(s.engine.state_at(0.0))
    np.testing.assert_allclose([b.lo, b.hi], [(-2, -1.125, 0), (2, 1.125, 0)])


def test_the_frame_at_t_is_the_same_in_any_order(tmp_path):
    from phasor import Video

    _clip(tmp_path / "c.mp4", n=24)
    s = _scene(tmp_path)
    s.add(Video(id="v", path="c.mp4", width=4.0, loop=True, trim=(0.25, 0.75)))
    s.marker("end", 3.0)
    times = np.linspace(0, 3, 37)
    first = {t: _sem(s, t)["frame"] for t in times}
    assert set(first.values()) <= set(range(6, 18))
    mm.close_all()
    s2 = _scene(tmp_path)
    s2.add(Video(id="v", path="c.mp4", width=4.0, loop=True, trim=(0.25, 0.75)))
    s2.marker("end", 3.0)
    for t in np.random.default_rng(0).permutation(times):
        assert _sem(s2, t)["frame"] == first[t]


def test_video_parameters_are_bounded(tmp_path):
    from phasor import Video

    s = _scene(tmp_path)
    bad = [{"width": float("nan")}, {"width": 1e300}, {"speed": 0.0}, {"speed": 1e6},
           {"speed": float("nan")}, {"start": -1.0}, {"start": float("inf")},
           {"trim": (2.0, 1.0)}, {"trim": (float("nan"), 0.0)}, {"trim": (0.0, 1e12)},
           {"fit": "zoom"}]  # fmt: skip
    for i, kw in enumerate(bad):  # a non-finite number is refused already by the document
        with pytest.raises((DocumentError, ValueError)):
            s.add(Video(id=f"v{i}", path="c.mp4", **kw))
    with pytest.raises(DocumentError, match="not a video"):
        s.add(Video(id="w", path="c.png"))
    for outside in ("../c.mp4", str(tmp_path / "c.mp4")):
        with pytest.raises(DocumentError, match="folder|absolute"):
            s.add(Video(id="x", path=outside))
    for fps in (float("nan"), 0.0, 1e6):
        with pytest.raises((DocumentError, ValueError)):
            s.add(Image(id="y", path="a_%03d.png", fps=fps))


def test_fill_follows_the_camera_and_makes_a_backdrop(tmp_path):
    from phasor import Video

    _clip(tmp_path / "c.mp4", n=4, w=64, h=64)  # square: cover crops it to 16:9
    s = _scene(tmp_path)
    s.background("c.mp4", id="v")
    s.set("camera.center", (1.0, 2.0, 0.0))
    sem = _sem(s, 0.0)
    assert sem["backdrop"] and sem["fill"] and sem["fit"] == "cover"
    w, h = sem["size"]
    assert h == pytest.approx(4.5) and w == pytest.approx(4.5 * 16 / 9)
    st = s.engine.state_at(0.0)
    b = st["v"].impl.world_bounds(st)
    np.testing.assert_allclose([b.lo[:2], b.hi[:2]], [(1 - w / 2, 2 - 2.25), (1 + w / 2, 4.25)])
    assert isinstance(st["v"].impl, Video) and st["v"].params.z_index < 0


# ---------------------------------------------------------------- 5. picture sequences


def test_a_picture_sequence_plays_like_a_clip(tmp_path):
    path = _frames(tmp_path / "fr", 6)
    s = _scene(tmp_path)
    s.add(Image(id="v", path=path, width=3.0, fps=2.0, loop=True))
    s.marker("end", 6.0)
    sem = _sem(s, 1.6)
    assert sem["kind"] == "image" and sem["frames"] == 6 and sem["frame"] == 3
    assert sem["seconds"] == pytest.approx(3.0) and sem["pixels"] == [8, 6]
    assert _sem(s, 3.2)["frame"] == 0  # looped
    src = s.engine.state_at(0.0)["v"].impl.source(s.engine.state_at(0.0))
    assert _which(src.frame(4)[0, 0]) == 4


def test_one_still_keeps_its_exact_semantics(tmp_path):
    """A plain picture's inspector output has the keys it always had (nothing new)."""
    _png(tmp_path / "p.png", w=40, h=20)
    s = _scene(tmp_path)
    s.add(Image(id="v", path="p.png", width=4.0))
    assert _sem(s, 0.0) == {"kind": "image", "path": "p.png", "pixels": [40, 20],
                            "missing": False, "size": [4.0, 2.0]}  # fmt: skip


def test_a_one_frame_sequence_draws_as_its_still(tmp_path):
    _png(tmp_path / "p.png", (200, 120, 40, 255), w=50, h=30)
    _png(tmp_path / "s" / "p_0.png", (200, 120, 40, 255), w=50, h=30)
    still = mm.open_source(tmp_path, "p.png")
    seq = mm.open_source(tmp_path, "s/p_%d.png")
    bg = np.zeros(3, np.float32)
    args = (40, 24, (2.0, 1.2), 2.4, (4.0, 2.4), mm.FULL, bg)
    a = mm.sample(still.frame(0), *args, opaque=still.probe().opaque, glow=False)
    b = mm.sample(seq.frame(0), *args, opaque=seq.probe().opaque, glow=False)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_stills_are_decoded_once_whatever_the_frame_cache_holds(tmp_path, monkeypatch):
    """Stills live in a cache of their own: two photos larger than the frame cache were
    decoded again for every draw (eleven times slower than before Milestone 24: audit, B3)."""
    monkeypatch.setattr(mm, "FRAME_CACHE_BYTES", 1024)  # holds no picture
    for name in ("a.png", "b.png"):
        _png(tmp_path / name, (200, 120, 40, 255), w=64, h=48)
    reads = []
    real = mm._decode_picture
    monkeypatch.setattr(mm, "_decode_picture", lambda *a: reads.append(a[0]) or real(*a))
    srcs = [mm.open_source(tmp_path, n) for n in ("a.png", "b.png")]
    for _ in range(5):
        for s in srcs:
            assert s.frame(0) is not None
    assert len(reads) == 2


def test_a_still_reads_as_image_always_did_bit_for_bit(tmp_path):
    """Linear premultiplied float32 exactly as ``srgb_to_linear(byte / 255)`` gave it (now
    through the lookup table, without float64 temporaries: a 9000 x 9000 picture peaked at
    7.5 GB, now 1.9 GB: audit, S6)."""
    from PIL import Image as PILImage

    from phasor.core.color import srgb_to_linear

    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
    PILImage.fromarray(a, "RGBA").save(tmp_path / "n.png")
    f = a.astype(np.float32) / 255.0
    rgb = np.asarray(srgb_to_linear(f[..., :3]), dtype=np.float32)
    want = np.concatenate([rgb * f[..., 3:], f[..., 3:]], axis=-1)
    assert np.array_equal(mm.open_source(tmp_path, "n.png").frame(0), want)


def test_a_gif_shows_its_first_frame(tmp_path):
    from PIL import Image as PILImage

    frames = [PILImage.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 0, 255))]
    frames[0].save(tmp_path / "a.gif", save_all=True, append_images=frames[1:])
    src = mm.open_source(tmp_path, "a.gif")
    px = src.frame(0)[0, 0]
    assert px[0] > 0.9 and px[2] < 0.05


# ---------------------------------------------------------------- 6. lint


@pytest.mark.parametrize("passes", MODES)
def test_lint_reports_missing_bad_and_ended_clips(tmp_path, passes):
    pytest.importorskip("av")
    from phasor import Video

    _clip(tmp_path / "c.mp4", n=12)  # half a second
    (tmp_path / "notes.mp4").write_text("not a video", encoding="utf-8")
    s = _scene(tmp_path, passes)
    s.add(Video(id="gone", path="gone.mp4", width=1.0, pos=(-3, 1.5, 0)))
    s.add(Video(id="junk", path="notes.mp4", width=1.0, pos=(-1, 1.5, 0)))
    s.add(Video(id="v", path="c.mp4", width=1.0, pos=(1, 1.5, 0)))
    s.add(Image(id="seq", path="fr/f_%03d.png", width=1.0, pos=(3, 1.5, 0)))
    s.marker("end", 2.0)
    issues = s.engine.lint()
    by = {(i.code, i.ids) for i in issues}
    assert ("missing_file", ("gone",)) in by and ("missing_file", ("seq",)) in by
    assert ("bad_file", ("junk",)) in by and ("clip_ended", ("v",)) in by
    (ended,) = [i for i in issues if i.code == "clip_ended"]
    assert ended.times[0] >= 0.5 and not ended.fix and "loop" in ended.fix_hint
    (junk,) = [i for i in issues if i.code == "bad_file"]
    assert "mp4" in junk.fix_hint
    s.engine.apply({"op": "set_param", "path": "v.loop", "value": True})
    assert "clip_ended" not in _codes(s)


@pytest.mark.parametrize("passes", MODES)
def test_lint_reports_a_clip_past_a_cap(tmp_path, passes, monkeypatch):
    pytest.importorskip("av")
    from phasor import Video

    _clip(tmp_path / "c.mp4", n=6)
    monkeypatch.setattr(mm, "MAX_FRAME_PIXELS", 32 * 32)
    s = _scene(tmp_path, passes)
    s.add(Video(id="v", path="c.mp4", width=2.0))
    (issue,) = [i for i in s.engine.lint() if i.code == "video_too_big"]
    assert "8K" in issue.message and _sem(s, 0.0)["missing"]
    monkeypatch.setattr(mm, "MAX_FRAME_PIXELS", 7680 * 4320)
    monkeypatch.setattr(mm, "MAX_FRAMES", 3)
    mm.close_all()
    s.engine.apply({"op": "set_param", "path": "v.width", "value": 2.5})
    assert "frames" in next(i for i in s.engine.lint() if i.code == "video_too_big").message


def test_without_pyav_a_video_is_missing_extra_and_draws_nothing(tmp_path, monkeypatch):
    import builtins

    from phasor import Video
    from phasor.core import extras

    real_import = builtins.__import__

    def no_av(name, *a, **k):
        if name == "av" or name.startswith("av."):
            raise ImportError("no av")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_av)
    monkeypatch.setattr(extras, "available", lambda m: m != "av")
    (tmp_path / "c.mp4").write_bytes(b"\0" * 64)
    s = _scene(tmp_path)
    s.add(Video(id="v", path="c.mp4", width=2.0))
    codes = _codes(s)
    assert "missing_extra" in codes and "bad_file" not in codes
    assert _sem(s, 0.0)["missing"]


@pytest.mark.parametrize("passes", MODES)
def test_text_on_footage_is_no_overlap_and_contrast_judges_the_frame(tmp_path, passes):
    """A label on a backdrop clip is not an overlap. Its contrast is judged against the
    frame: dark text on bright footage is flagged, light text is not (without OpenGL by the
    mean colour of the frame shown, ``average_color``)."""
    pytest.importorskip("av")
    av = pytest.importorskip("av")
    path = tmp_path / "bright.mp4"
    with av.open(str(path), "w") as c:
        s_ = c.add_stream("mpeg4", rate=24)
        s_.width, s_.height, s_.pix_fmt = 64, 36, "yuv420p"
        for _ in range(4):
            a = np.full((36, 64, 3), 235, np.uint8)
            for p in s_.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s_.encode():
            c.mux(p)
    s = _scene(tmp_path, passes)
    s.background("bright.mp4", id="v")
    s.add(Text(id="note", text="on the footage", pos=(0, 0, 0),
               style={"color": "#e8e8e8"}))  # fmt: skip
    codes = _codes(s)
    assert "overlap" not in codes and "contrast" in codes
    s.engine.apply({"op": "set_param", "path": "v.opacity", "value": 0.0})
    assert "contrast" not in _codes(s)


def test_hostile_files_are_lint_and_draw_nothing_quickly(tmp_path):
    """Truncated, not a video, no video stream, a folder, a 1 KB clip claiming 10^9
    frames, a picture claiming 900 million pixels or none: each is a lint issue, nothing is
    drawn, and nothing raises or takes long."""
    av = pytest.importorskip("av")
    from phasor import Video

    good = _clip(tmp_path / "good.mp4", n=24).read_bytes()
    (tmp_path / "cut.mp4").write_bytes(good[: len(good) // 2])  # moov at the end: gone
    fast = _clip(tmp_path / "fast.mp4", n=24, faststart=True).read_bytes()
    (tmp_path / "cut_fast.mp4").write_bytes(fast[: len(fast) * 2 // 3])  # frames cut off
    (tmp_path / "zeros.mkv").write_bytes(b"\0" * 1024)
    (tmp_path / "dir.mp4").mkdir()
    with av.open(str(tmp_path / "sound.mp4"), "w") as c:  # sound only
        a = c.add_stream("aac", rate=48000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp",
                                           layout="mono")  # fmt: skip
        frame.sample_rate = 48000
        for p in [*a.encode(frame), *a.encode()]:
            c.mux(p)
    tiny = _clip(tmp_path / "tiny.mp4", n=1, w=16, h=16).read_bytes()
    claim = bytearray(tiny)  # stts: one entry, sample_count = 10^9
    i = claim.find(b"stts")
    assert i > 0
    claim[i + 12 : i + 16] = (10**9).to_bytes(4, "big")
    (tmp_path / "claims.mp4").write_bytes(bytes(claim))
    _png(tmp_path / "bomb.png", w=1, h=1)
    _patch_png_size(tmp_path / "bomb.png", 30000, 30000)
    _png(tmp_path / "zero.png", w=1, h=1)
    _patch_png_size(tmp_path / "zero.png", 0, 0)
    (tmp_path / "trunc.png").write_bytes(_png(tmp_path / "t.png", w=64, h=64).read_bytes()[:60])
    s = _scene(tmp_path)
    clips = ["cut.mp4", "cut_fast.mp4", "zeros.mkv", "dir.mp4", "sound.mp4", "claims.mp4"]
    pics = ["bomb.png", "zero.png", "trunc.png"]
    for k, name in enumerate(clips):
        s.add(Video(id=f"c{k}", path=name, width=1.0))
    for k, name in enumerate(pics):
        s.add(Image(id=f"p{k}", path=name, width=1.0))
    s.marker("end", 1.0)
    t0 = time.perf_counter()
    issues = s.engine.lint()
    sems = {o: _sem(s, 0.5, o) for o in [f"c{k}" for k in range(6)] + ["p0", "p1", "p2"]}
    took = time.perf_counter() - t0
    flagged = {i.ids[0] for i in issues if i.code in ("bad_file", "missing_file",
                                                      "video_too_big")}  # fmt: skip
    expect = {"c0", "c2", "c3", "c4", "p0", "p1", "p2"}
    assert expect <= flagged, (flagged, sems)
    for o in expect:
        assert sems[o]["missing"], (o, sems[o])
    assert "no video stream" in sems["c4"]["unreadable"]
    assert "pixels" in sems["p0"]["unreadable"]
    # the cut-short clip shows what it has; the 10^9 claim shows its one frame (or is bad)
    assert sems["c5"]["missing"] or sems["c5"]["frames"] < 10
    assert took < 60, took
    for t in (0.0, 0.5, 1.0):  # drawing (the pixels too) never raises
        st = s.engine.state_at(t)
        for o in sems:
            _draw_pixels(st, o)


def _draw_pixels(state, oid):
    """Run an object's draw with a context that also asks for every field's pixels."""
    calls = []

    class Ctx:
        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k))

        def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0):
            out = pixels(16, 9, (float(np.linalg.norm(right)) / 2,
                                 float(np.linalg.norm(up)) / 2), float(np.linalg.norm(up)))  # fmt: skip
            assert out.shape == (9, 16, 4) and np.isfinite(out).all()

    state[oid].impl.draw(Ctx(), state, "color")


def _patch_png_size(path: Path, w: int, h: int) -> None:
    """Rewrite a PNG's IHDR to claim ``w`` x ``h`` pixels (with a valid CRC)."""
    import zlib

    data = bytearray(path.read_bytes())
    i = data.find(b"IHDR")
    data[i + 4 : i + 8] = w.to_bytes(4, "big")
    data[i + 8 : i + 12] = h.to_bytes(4, "big")
    data[i + 17 : i + 21] = zlib.crc32(bytes(data[i : i + 17])).to_bytes(4, "big")
    path.write_bytes(bytes(data))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_fifo_named_like_a_clip_is_never_opened(tmp_path):
    from phasor import Video

    os.mkfifo(tmp_path / "pipe.mp4")
    os.mkfifo(tmp_path / "pipe.png")
    s = _scene(tmp_path)
    s.add(Video(id="v", path="pipe.mp4", width=1.0))
    s.add(Image(id="p", path="pipe.png", width=1.0))
    codes = {(i.code, i.ids[0]) for i in s.engine.lint()}
    assert ("missing_file", "v") in codes and ("missing_file", "p") in codes


def test_a_link_out_of_the_folder_is_not_shown(tmp_path):
    from phasor import Video

    _png(tmp_path / "secret.png")
    root = tmp_path / "scene"
    root.mkdir()
    try:
        os.symlink(tmp_path / "secret.png", root / "pic.png")
        os.symlink(tmp_path / "secret.png", root / "clip.mp4")
    except (OSError, NotImplementedError):
        pytest.skip("this system cannot make symbolic links")
    s = _scene(root)
    s.add(Image(id="p", path="pic.png", width=1.0))
    s.add(Video(id="v", path="clip.mp4", width=1.0))
    issues = [i for i in s.engine.lint() if i.code == "missing_file"]
    assert {i.ids[0] for i in issues} == {"p", "v"}
    assert all("outside" in i.message for i in issues)


# ---------------------------------------------------------------- 7. scenes made of footage


def test_from_video_makes_a_scene_as_long_as_the_clip(tmp_path):
    _clip(tmp_path / "c.mp4", n=36)  # 1.5 s
    s = Scene.from_video("c.mp4", base_dir=tmp_path, passes=None,
                         clip={"speed": 0.5, "start": 0.25})  # fmt: skip
    assert s.engine.document.meta.duration == pytest.approx(0.25 + 3.0)
    sem = _sem(s, 1.0, "clip")
    assert sem["fill"] and sem["backdrop"] and sem["frame"] == 9
    s.add(Text(id="title", text="over the clip", pos=(0, 1.5, 0)))
    assert "overlap" not in _codes(s)
    seq = _frames(tmp_path / "fr", 10)
    s2 = Scene.from_video(seq, base_dir=tmp_path, passes=None, clip={"fps": 5.0})
    assert s2.engine.document.meta.duration == pytest.approx(2.0)


def test_a_fast_clip_scene_ends_exactly_with_its_clip(tmp_path):
    _clip(tmp_path / "c.mp4", n=36)
    for speed in (7.3, 64.0):
        s = Scene.from_video("c.mp4", base_dir=tmp_path, passes=None, clip={"speed": speed})
        assert "clip_ended" not in _codes(s), speed


def test_a_still_past_the_pixel_cap_is_bad_file(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    _png(tmp_path / "p.png", w=40, h=30)
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", None)  # Pillow's own check off
    monkeypatch.setattr(mm, "picture_pixels_cap", lambda: 100)
    s = _scene(tmp_path)
    s.add(Image(id="v", path="p.png", width=2.0))
    codes = _codes(s)
    assert "bad_file" in codes and "video_too_big" not in codes
    assert "pixels" in _sem(s, 0.0)["unreadable"]


def test_saving_a_scene_elsewhere_brings_every_frame_of_its_sequence(tmp_path):
    path = _frames(tmp_path / "fr", 3)
    s = _scene(tmp_path)
    s.add(Image(id="v", path=path, width=2.0))
    out = tmp_path / "copy"
    out.mkdir()
    s.engine.save(out / "scene.phasor.json")
    assert sorted(p.name for p in (out / "fr").iterdir()) == ["f_000.png", "f_001.png",
                                                              "f_002.png"]  # fmt: skip


def test_a_worker_refuses_a_sequence_changed_since_the_bake(tmp_path):
    from phasor.engine import Engine

    path = _frames(tmp_path / "fr", 3)
    s = _scene(tmp_path)
    s.add(Image(id="v", path=path, width=2.0))
    payload = s.engine.worker_payload()
    w = Engine.from_payload(payload, passes=None)
    assert _sem(Scene.of(w), 0.0)["frames"] == 3
    _png(tmp_path / "fr" / "f_003.png")
    with pytest.raises(RuntimeError, match="changed while"):
        w.state_at(0.1)
    with pytest.raises(ValueError, match="changed since"):
        Engine.from_payload(payload, passes=None)


# ---------------------------------------------------------------- 8. drawn (OpenGL)


def _gl_scene(tmp_path):
    from phasor import Video

    _clip(tmp_path / "c.mp4", n=24)
    s = Scene("draw", base_dir=tmp_path)
    s.set("camera.height", 2.25)
    s.add(Video(id="v", path="c.mp4", width=4.0))
    s.marker("end", 1.0)
    return s


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_video_draws_the_frame_of_each_time(tmp_path):
    from phasor.core.color import linear_to_srgb

    s = _gl_scene(tmp_path)
    runner = s.engine.pass_runner()
    seen = []
    for k in (0, 5, 17, 5):
        pic = runner.color(s.engine.state_at(k / FPS), (64, 36))[..., :3]
        srgb = np.asarray(linear_to_srgb(pic[18, 32])) * 255
        seen.append(_which(srgb))
    assert seen == [0, 5, 17, 5]


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.parametrize("background", ["png_sequence", "mp4", "open_gop"])
def test_a_parallel_render_of_footage_equals_the_serial_render(tmp_path, background):
    from phasor.render.job import render_file
    from phasor.scene import parallel_frames

    fps, n, chunk = 12.0, 12, 3
    if background == "open_gop":
        # the second worker starts at scene frame 47 = clip frame 94, a leading B-frame of the
        # keyframe at 96, too far ahead to decode on to: it seeks (audit, B1)
        _open_gop(tmp_path / "og.mp4", n=200)
        s = Scene.from_video("og.mp4", base_dir=tmp_path)
        n, chunk = 100, 47
    elif background == "mp4":
        from phasor import Video

        _clip(tmp_path / "c.mp4", n=24)
        s = Scene.from_video("c.mp4", base_dir=tmp_path)
        s.set("camera.height", 2.25)
        s.add(Image(id="seq", path=_frames(tmp_path / "fr", 6), fps=6.0, width=1.0,
                    loop=True))  # fmt: skip
        s.add(Video(id="v", path="c.mp4", width=1.0, pos=(1.2, 0, 0), speed=2.0, loop=True))
    else:
        s = _gl_scene(tmp_path)
        s.background(_frames(tmp_path / "fr", 6), id="back", fps=6.0, opacity=0.5)
    e = s.engine
    serial = render_file(e, tmp_path / "serial", preset="preview", fps=fps)
    many = render_file(e, tmp_path / "many", preset="preview", fps=fps,
                       frame_source=parallel_frames(e, 2, chunk=chunk))  # fmt: skip
    assert serial["frames"] == many["frames"] == n
    a = [p.read_bytes() for p in sorted((tmp_path / "serial").glob("frame_*.png"))]
    b = [p.read_bytes() for p in sorted((tmp_path / "many").glob("frame_*.png"))]
    assert len(a) == n and a == b
    assert len(set(a)) > n // 2  # the clip plays: the frames differ


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_frame_a_decoder_failed_to_give_is_drawn_again_when_it_can(tmp_path, monkeypatch):
    """A frame that could not be had once (a decoder closed meanwhile, a passing read error)
    draws the background that time, and is not kept: the next draw at that time shows the
    frame (the field cache kept the stand-in: audit, B1/S2)."""
    s = _gl_scene(tmp_path)
    runner = s.engine.pass_runner()
    real = mm.VideoSource.frame
    monkeypatch.setattr(mm.VideoSource, "frame", lambda self, i: None)
    blank = runner.color(s.engine.state_at(5 / FPS), (64, 36))
    monkeypatch.setattr(mm.VideoSource, "frame", real)
    shown = runner.color(s.engine.state_at(5 / FPS), (64, 36))
    assert not np.array_equal(blank, shown)
    assert np.array_equal(shown, runner.color(s.engine.state_at(5 / FPS), (64, 36)))


# ---------------------------------------------------------------- 9. examples


def _example(name: str, passes):
    if passes is None:  # Windows CI has no OpenGL: the analytic lint alone
        import phasor.render.passes as p

        real = p.default_renderer
        p.default_renderer = lambda *a, **k: None
        try:
            return runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name="x")["s"]
        finally:
            p.default_renderer = real
    return runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name="x")["s"]


@pytest.mark.parametrize("passes", MODES)
def test_the_examples_lint_clean(passes):
    pytest.importorskip("av")
    for name in ("video_input", "picture_scene"):
        s = _example(name, passes)
        assert s.engine.lint(strict=True) == [], name


def _golden(name: str, sheet: np.ndarray, klass: str) -> None:
    from tests.test_beauty import _golden as golden

    golden(name, sheet, klass)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_video_input_golden_storyboard():
    pytest.importorskip("av")
    e = _example("video_input", "auto").engine
    sheet = e.storyboard([0.2, 1.0, 2.5, 3.9], size=(320, 180))["image"]
    _golden("video_input", sheet, e.pass_runner().renderer_class())


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_picture_scene_golden_storyboard():
    e = _example("picture_scene", "auto").engine
    sheet = e.storyboard([0.2, 1.0, 2.5, 3.9], size=(320, 180))["image"]
    _golden("picture_scene", sheet, e.pass_runner().renderer_class())
