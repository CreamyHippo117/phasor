# Milestone 24: video and picture inputs

From `STUDIO.md` (owner, 2026-11-24: "mp4/png input scenes"). Footage and pictures as scenes
or as parts of scenes. Exit condition: **an mp4 and a png sequence, as backgrounds and as
placed objects, render correctly and deterministically, in parallel workers too, with hostile
files handled as lint** (`examples/video_input.py`, `examples/picture_scene.py`).

Status: **built on Linux (llvmpipe OpenGL 4.5, no CUDA), branch `m24-video-inputs`; gate run
by the authoring model (not independent).** Untested on Windows, macOS and CUDA.

## Built

| Area | Where | What |
| --- | --- | --- |
| One frame source | `phasor/domains/media_math.py` (new; no drawing) | `open_source(base_dir, path, video=, fps=)` names a `StillSource`, `SequenceSource` or `VideoSource` by its file stamp; `Source.probe()` (size, the presentation time of every frame, why it cannot be shown) is cached by stamp and decodes only the first frame; `Source.frame(i)` is cached by (kind, stamp, i) in a `ByteCache` (256 MB, 256 entries). `clip_time` (start, trim, speed, loop; NaN/inf hold the first frame), `clip_range`, `frame_index` (nearest previous by presentation time, 1 µs tolerance), `fit_box` (`stretch`, `contain`, `cover`, `none`), `sample` (the visible part of a frame to linear RGBA), `average` (mean colour, for lint) |
| Video decoding | `media_math.VideoSource`, `_Container`, `_Decoder` | PyAV, lazily imported (the `encode` extra). The probe demuxes every packet (no decoding) for the exact pts list; a frame is decoded by seeking to the keyframe at or before it and decoding forward to the frame with that pts, or, when the decoder is just behind it (within `FORWARD` = 90 frames), by decoding on without a seek. One open decoder per (process, stamp), each with a lock, at most `MAX_OPEN` = 4 per process (least recently used closed; an older version of a file closed when it changes); a pid guard drops handles inherited across a fork. PyAV never gets a path: the file is opened by `files.open_regular` (non-blocking, regular file, checked against its stamp) and the file object handed over. Frames larger than 4096 px are scaled down when decoded (sws) and kept as sRGB uint8 |
| Picture sequences | `core/files.py` | a `Files` value with one `%d`/`%0Nd` in its file name (`frames/shot_%04d.png`) is a sequence: frames from the first of 0-4 that is there to the first gap, at most `MAX_SEQUENCE` = 3000 (`is_sequence`, `sequence`, `members`). Each frame is a plain regular file in the folder (one `lstat` each; a link goes through `regular_file`, so a link out of the folder ends the sequence). `stamp` of a sequence is (pattern path, a sha256 of every frame's name, mtime and size, count): the same in every process, so a render worker compares it; `problem` says when no frame is there (`missing_file`). `open_regular` (new) is what `read_bytes` uses too. `PICTURE_EXT`, `VIDEO_EXT` |
| What Image and Video share | `phasor/domains/frames.py` (new) | `PictureParams` (`pos`, `width` (≤ 1e9), `height`, `frame`, `fit`, `fill`, `backdrop`, `start`, `trim`, `speed` (0-64], `loop`) and `Picture` (bounds, features, semantics, `average_color`, draw). `fill` makes the box the camera's view (ortho cameras; ignored in perspective); `backdrop` (always with `fill`) lets labels sit on it. Draw records only the quad; the frame is decoded inside the `pixels` callback, when the renderer needs it |
| `Image` extended | `phasor/domains/media.py` | now a `Picture`; new parameters `fps` (24, a sequence's rate) and the shared ones above, all defaulting to what Image did (`fit="stretch"`, no fill, no backdrop). A still is read exactly as before (linear float32, scaled to 8192 px), now through `StillSource`, stored premultiplied (identical for opaque pictures), in the bounded byte cache instead of an `lru_cache(16)` of up to 1 GB arrays; a GIF shows its first frame. `examples/picture.py` renders byte-identical (checked against the pre-change code) |
| `Video` | `phasor/domains/video.py` (new) | `path` (`.mp4`, `.m4v`, `.mov`, `.webm`, `.mkv`) plus the shared parameters; `Requires = {"av": "encode"}`. Semantics: `kind`, `path`, `pixels`, `missing`, `size`, `frames`, `fps`, `seconds`, `frame` (at t), `clip_time`, `looping`, `trimmed`, `ended`, `backdrop`, `fill`, `fit`, and `unreadable`/`too_big` when it cannot be shown |
| Renderer | `render/passes.py` (one line), `core/objects.py` (docstring) | the field-pixel cache key also takes the `pixels` function's `cache_key` attribute (a clip's stamp and frame index): without it a playing clip with constant parameters would be served one frame |
| Backgrounds and clip scenes | `phasor/scene.py` | `Scene.background(path, id="background", **params)` (Video or Image by extension; `fill`, `fit="cover"`, `z_index` -1000); `Scene.from_video(path, clip={...}, **scene_kwargs)` sets `meta.duration` to start + trimmed length / speed (capped at `MAX_SECONDS`), read when the script runs |
| Lint | `inspect/lint.py` | `video_too_big` (semantics `too_big`) and `clip_ended` (`ended` while shown, no fix); `bad_file` hints for clips and pictures; `contrast` without OpenGL judges text on a *backdrop* picture or clip against `average_color` (the mean of the frame shown, over the background); with OpenGL it is measured on the rendered picture as before. `missing_extra` for a Video without PyAV comes from `Requires` |
| Files beside a saved scene | `engine.py` `_bring_files` | copies every frame of a sequence (`files.members`) |
| Code view | `core/pycode.py` | `Video` imports from `phasor`; a document's `meta.duration` (set by `from_video`) round-trips through the generated script (it was never written before) |
| Docs | `docs/ai/API.md` | `Image` row updated, `Video` row, "Footage and picture sequences" (two python blocks that run in CI), lint rows; `llms.txt` regenerated |
| Examples | `examples/video_input.py`, `examples/picture_scene.py`, `examples/assets/make_assets.py`, `examples/assets/clip.mp4` (10 KB), `examples/assets/frames/orbit_00..11.png` (12 x ~760 B) | both sorted into `kit/examples.toml` (basics, technique) |

## Verification (Linux, 4 cores, llvmpipe OpenGL 4.5, no CUDA)

Tests: `tests/test_video.py` (new, 41 tests: sequences and stamps, clip time, frame choice
at k/fps including 30000/1001, fit, premultiplied sampling, crop-conversion equals full
conversion, a one-frame sequence equals its still, decode order independence (in order,
shuffled, fresh decoder: byte-identical), no seek when playing in order, byte-bounded
cache, bounded decoders, Video semantics and bounds, fill following the camera, picture
sequences, lint in both modes (`passes=None` and OpenGL) for `missing_file`, `bad_file`,
`clip_ended`, `video_too_big`, `missing_extra`, overlap/contrast on footage, hostile files,
FIFO, symlinks, `from_video`, saving a sequence scene elsewhere, a worker refusing a changed
sequence, GL frame-per-time, parallel render = serial render byte for byte, examples lint
clean in both modes, two golden storyboards); `tests/test_final.py` gains `video` in
`REQUIRED`/`NEEDS`.

Measurements (`/tmp` scripts, this box):

| What | Result |
| --- | --- |
| 1920x1080 H.264 (GOP 60, B-frames), 300 frames | probe 0.3 s; sequential 120-146 frames/s; random access 5.5-6 frames/s (seek + decode up to a GOP) |
| 640x360, same | probe 23-67 ms; sequential 880-1000 frames/s; random 50 frames/s |
| sampling a 1080p frame into a 1080p field | 86 ms (banded, float32) |
| 7680x4320 clip (3 frames) | probe + 3 frames 0.35 s, stored at 4096x2304; peak RSS +260 MB; sampled to 4K +210 MB |
| frame cache at its cap | 253 MB used of 256 MB playing 300 HD frames; process peak 538 MB |
| stamp of a 3000-picture sequence | 60 ms (one lstat a frame) |
| hostile files (one process each) | truncated mp4 (moov lost; or frames cut, faststart), 1 KB of zeros as .mkv, 200 KB of noise as .mp4, a 1 KB mp4 whose stts claims 10^9 frames (shows its 1 real frame), the same with stsz claiming 10^9 (`no frames`), a PNG claiming 30000x30000 (Pillow's bomb error) or 12000x12000 with no data: each 5-75 ms, no memory growth, `bad_file` or drawn as what it really holds |

## Deviations

- **The frame-source class is `media_math.Source`, not `FrameSource`**: `render/job.py`
  already has a `FrameSource` (a render's frame producer).
- **A path with one `%d`/`%0Nd` in its file name is a sequence for every `Files` type**
  (`files.stamp`, `problem`), not only `Image`. A file actually named like `a%04d.png`
  cannot be shown as a still any more; no such name was found anywhere.
- **Lists of paths as a sequence are deferred** (the `Files` machinery is one string per
  value; a pattern covers the use).
- **`fit` defaults to `stretch` for both types** (what `Image` always did with `width` and
  `height` set); `Scene.background` uses `cover`.
- **`video_too_big` draws nothing** (a cap is a hard limit); absurd values are `bad_file`.
  No theme `[lint]` thresholds were added: these are safety caps, not taste (constants in
  `media_math`).
- **A still past the bomb limit is `bad_file`**, not `video_too_big`.
- **Before `start` a clip shows its first frame** (fade it in with `reveal` to hide it).
- **Transparent pictures are sampled premultiplied** (no fringe of the colour stored under
  clear pixels). Opaque pictures draw exactly as before.
- **`Image` semantics for a picture that cannot be read add `unreadable`** (lint
  `bad_file`; before, it only said `missing`). A plain still's semantics are otherwise
  unchanged (tested).
- **`passes.py` field cache key**: one new element (`cache_key` of the pixels function,
  `None` for every existing type).

## Deferred

- A list of paths as a sequence; per-frame sizes in a sequence (frames of another size are
  stretched to the first frame's box).
- `fill` under a perspective camera (falls back to `pos`/`width`).
- Clips with alpha (VP9/ProRes 4444): decoded opaque.
- Colour matrices: frames go through PyAV's default `rgba` conversion; BT.709 vs BT.601 and
  full/limited range of unusual sources are not checked (H.264 test clips come back within
  4 codes). HDR clips are not tone mapped.
- Variable-frame-rate clips: `fps` reported is the median frame gap; the frame shown is still
  the exact nearest previous by presentation time.
- Sound of a clip is not played (use `Sound`); a damaged frame in the middle of a good clip
  draws the background for that frame (not reported per frame).
- Faster sampling (GPU texture upload of the frame instead of CPU sampling).
- The `kit` copies examples without `assets/` (as for `picture.py` before).

## Untested

Windows (CI will run the non-GL tests; libx264 may be absent from PyAV's Windows wheels: the
tests fall back to `mpeg4`), macOS, CUDA, GPU goldens, FIFO/symlink tests on Windows (they
skip there).

## Shared files touched (merge notes: Milestones 21 and 22 run in parallel)

`docs/ai/API.md`, `llms.txt` (regenerate after merging), `inspect/lint.py` (two codes, a
hint map, the analytic contrast branch), `kit/examples.toml`, `phasor/__init__.py`
(`Video`), `phasor/domains/__init__.py`, `phasor/domains/media.py` (only the `Image` part;
`Sound` untouched), `core/files.py`, `core/extras.py` (`av`), `core/objects.py` (docstring),
`core/pycode.py`, `engine.py` (`_bring_files`), `render/passes.py` (one line), `scene.py`
(two methods, `BACKGROUND_Z`), `tests/test_final.py`.
