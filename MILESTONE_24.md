# Milestone 24: video and picture inputs

From `STUDIO.md` (owner, 2026-11-24: "mp4/png input scenes"). Footage and pictures as scenes
or as parts of scenes. Exit condition: **an mp4 and a png sequence, as backgrounds and as
placed objects, render correctly and deterministically, in parallel workers too, with hostile
files handled as lint** (`examples/video_input.py`, `examples/picture_scene.py`).

Status: **built on Linux (llvmpipe OpenGL 4.5, no CUDA), branch `m24-video-inputs`; audited
independently (2026-11-24): 3 blockers + 6 should-fix fixed, with N1, N2, N3, N5 (see
"Independent audit").** Untested on Windows, macOS and CUDA.

## Built

| Area | Where | What |
| --- | --- | --- |
| One frame source | `phasor/domains/media_math.py` (new; no drawing) | `open_source(base_dir, path, video=, fps=)` names a `StillSource`, `SequenceSource` or `VideoSource` by its file stamp; `Source.probe()` (size, the presentation time of every frame, why it cannot be shown) is cached by stamp and decodes only the first frame; `Source.frame(i)` is cached by (kind, stamp, i) in a `ByteCache` (256 MB, 256 entries). `clip_time` (start, trim, speed, loop; NaN/inf hold the first frame), `clip_range`, `frame_index` (nearest previous by presentation time, 1 µs tolerance), `fit_box` (`stretch`, `contain`, `cover`, `none`), `sample` (the visible part of a frame to linear RGBA), `average` (mean colour, for lint) |
| Video decoding | `media_math.VideoSource`, `_Container`, `_Decoder` | PyAV, lazily imported (the `encode` extra). The probe demuxes every packet (no decoding) for the exact pts list of the frames shown (packets an edit list discards left out; time 0 is the first shown) and the keyframes (pts, and the smaller of pts and dts to seek by). A frame is decoded by seeking to the keyframe whose pts is at or before it (an open GOP's leading B-frames decode as from the start; if the frame does not come out exactly, from the keyframe before), or, when a cursor is just behind it (within `FORWARD` = 90 frames), by decoding on without a seek; a cursor's frame is taken only on an exact pts hit, so the way there never changes the frame. Up to `PER_FILE` = 4 cursors per file and `MAX_OPEN` = 8 in all per process, each with a lock (least recently used closed; an older version of a file closed when it changes; a decoder closed by another thread meanwhile is replaced, then a private one is used); a pid guard drops handles inherited across a fork. PyAV never gets a path: the file is opened by `files.open_regular` (non-blocking, no link, regular file, checked against its stamp) and the file object handed over. Frames larger than 4096 px are scaled down when decoded by the container's own single-threaded `VideoReformatter` and kept as sRGB uint8 |
| Picture sequences | `core/files.py` | a `Files` value with one `%d`/`%0Nd` in its file name (`frames/shot_%04d.png`) is a sequence: frames from the first of 0-4 that is there to the first gap; at most `MAX_SEQUENCE` = 3000 play (more: lint `truncated`) (`is_sequence`, `sequence`, `sequence_frames`, `members`). Each frame is opened without following a link and checked against the time and size it was stamped with. Each frame is a plain regular file in the folder (one `lstat` each; a link goes through `regular_file`, so a link out of the folder ends the sequence). `stamp` of a sequence is (pattern path, a sha256 of every frame's name, mtime and size, count): the same in every process, so a render worker compares it; `problem` says when no frame is there (`missing_file`). `open_regular` (new) is what `read_bytes` uses too. `PICTURE_EXT`, `VIDEO_EXT` |
| What Image and Video share | `phasor/domains/frames.py` (new) | `PictureParams` (`pos`, `width` (≤ 1e9), `height`, `frame`, `fit`, `fill`, `backdrop`, `start`, `trim`, `speed` (0-64], `loop`) and `Picture` (bounds, features, semantics, `average_color`, draw). `fill` makes the box the camera's view (ortho cameras; ignored in perspective); `backdrop` (always with `fill`) lets labels sit on it. Draw records only the quad; the frame is decoded inside the `pixels` callback, when the renderer needs it |
| `Image` extended | `phasor/domains/media.py` | now a `Picture`; new parameters `fps` (24, a sequence's rate) and the shared ones above, all defaulting to what Image did (`fit="stretch"`, no fill, no backdrop). A still is read exactly as before (linear float32, scaled to 8192 px, bit for bit), now through `StillSource` and the lookup table (no float64 temporaries), stored premultiplied (identical for opaque pictures), in a byte-bounded cache of its own (`STILL_CACHE_BYTES` = 2 GB, 64 entries) instead of an `lru_cache(16)` of up to 1 GB arrays; a GIF shows its first frame. `examples/picture.py` renders byte-identical (checked against the pre-change code) |
| `Video` | `phasor/domains/video.py` (new) | `path` (`.mp4`, `.m4v`, `.mov`, `.webm`, `.mkv`) plus the shared parameters; `Requires = {"av": "encode"}`. Semantics: `kind`, `path`, `pixels`, `missing`, `size`, `frames`, `fps`, `seconds`, `frame` (at t), `clip_time`, `looping`, `trimmed`, `ended`, `backdrop`, `fill`, `fit`, and `unreadable`/`too_big` when it cannot be shown |
| Renderer | `render/passes.py`, `core/objects.py` | the field-pixel cache key also takes the `pixels` function's `cache_key` attribute (a clip's stamp and frame index): without it a playing clip with constant parameters would be served one frame; pixels from a call that marks its function `uncached` (a frame that could not be had that time) are drawn but never kept. `ObjectBase.Fixed` (optional): parameters that hold one value for the whole scene |
| Backgrounds and clip scenes | `phasor/scene.py` | `Scene.background(path, id="background", **params)` (Video or Image by extension; `fill`, `fit="cover"`, `z_index` -1000); `Scene.from_video(path, clip={...}, **scene_kwargs)` sets `meta.duration` to start + trimmed length / speed (capped at `MAX_SECONDS`), read when the script runs |
| Lint | `inspect/lint.py` | `video_too_big` (semantics `too_big`), `clip_ended` (`ended` while shown, no fix; its own message for a trim in-point at or past the clip's end), `too_many_clips` (more `Video` objects shown at once than `MAX_OPEN`, or `PER_FILE` of one file: each frame then seeks), `truncated` for a sequence past 3000; `bad_file` hints for clips and pictures; `contrast` without OpenGL judges text on a *backdrop* picture or clip against `average_color` (the mean of the frame shown, over the background); with OpenGL it is measured on the rendered picture as before. `missing_extra` for a Video without PyAV comes from `Requires` |
| Files beside a saved scene | `engine.py` `_bring_files` | copies every frame of a sequence (`files.members`) |
| Code view | `core/pycode.py` | `Video` imports from `phasor`; a document's `meta.duration` (set by `from_video`) round-trips through the generated script (it was never written before) |
| Docs | `docs/ai/API.md` | `Image` row updated, `Video` row, "Footage and picture sequences" (two python blocks that run in CI), lint rows; `llms.txt` regenerated |
| Examples | `examples/video_input.py`, `examples/picture_scene.py`, `examples/assets/make_assets.py`, `examples/assets/clip.mp4` (10 KB), `examples/assets/frames/orbit_00..11.png` (12 x ~760 B) | both sorted into `kit/examples.toml` (basics, technique) |

## Verification (Linux, 4 cores, llvmpipe OpenGL 4.5, no CUDA)

Tests: `tests/test_video.py` (new, 50 test functions, 57 cases, after the audit: sequences and stamps, clip time, frame choice
at k/fps including 30000/1001, fit, premultiplied sampling, crop-conversion equals full
conversion, a one-frame sequence equals its still, decode order independence (in order,
shuffled, fresh decoder: byte-identical), no seek when playing in order, byte-bounded
cache, bounded decoders, Video semantics and bounds, fill following the camera, picture
sequences, lint in both modes (`passes=None` and OpenGL) for `missing_file`, `bad_file`,
`clip_ended`, `video_too_big`, `missing_extra`, overlap/contrast on footage, hostile files,
FIFO, symlinks, `from_video`, saving a sequence scene elsewhere, a worker refusing a changed
sequence, GL frame-per-time, parallel render = serial render byte for byte (an mp4 background with a sequence and a clip placed on it, and a sequence background with a clip placed), a fast clip scene (speed 7.3 and 64) ending without `clip_ended`, an over-cap still as `bad_file`, examples lint
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
| a 3000-picture sequence as the background (the cap) | `state_at` 46-56 ms, `inspect` 177 ms, a 480x270 frame 234 ms: the engine, the draw cache and the object each stamp the sequence (once per state for the object, memoized); proportionally less for shorter sequences (a 120-picture one: about 2 ms a stamp) |
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
- **A still past the pixel cap (Pillow's bomb limit) is `bad_file`**, not `video_too_big`
  (tested with Pillow's own check off).
- **Before `start` a clip shows its first frame** (fade it in with `reveal` to hide it).
- **Transparent pictures are sampled premultiplied** (no fringe of the colour stored under
  clear pixels). Opaque pictures draw exactly as before.
- **`Image` semantics for a picture that cannot be read add `unreadable`** (lint
  `bad_file`; before, it only said `missing`). A plain still's semantics are otherwise
  unchanged (tested).
- **`passes.py` field cache key**: one new element (`cache_key` of the pixels function,
  `None` for every existing type); and pixels whose function was marked `uncached` during
  the call are not kept (audit).
- **`ObjectBase.Fixed`** (audit, S3): an optional class variable beside `Files` and `Life`
  (not one of contract 5's required members) naming parameters that hold one value for the
  whole scene; `compile` refuses a track, expression or updater on one (`bad_param`).
  `Image`/`Video` list `start`, `trim`, `speed`, `fps`: clip time is (t - start) x speed, so
  an animated speed played the clip backwards as it slowed. Refusing keeps clip time a
  plain pure function of (document, t); integrating speed over time was the alternative.
  `from_video` with such arguments is a `DocumentError` too (N5).
- **A sequence past 3000 pictures plays its first 3000** (lint `truncated`, as a long table
  shows its first rows), instead of drawing nothing (`video_too_big`) (audit, N2).
- **`too_many_clips`** (audit, S1): a new lint code. Its limits (`MAX_OPEN` = 8 decoders a
  process, `PER_FILE` = 4 of one file) are constants in `media_math` carried in a clip's
  semantics (`decoders`; `inspect` may not import `domains`), not theme `[lint]` values:
  they are what the decoder keeps, not taste.
- **Stills keep their own cache, 2 GB a process** (audit, B3): as linear float32, the
  arithmetic `Image` always used (16 bytes a pixel). Keeping them as sRGB bytes was tried and
  measured: the full-picture conversion on every draw that shrinks it (the box filter
  needs it) made `photos.py` 14-16 s against 5.3 s before Milestone 24, so the float picture
  is kept, bounded by bytes, decoded once.
- **The pixel cap of stills stays Pillow's bomb limit** (audit, S6, not lowered to 64 Mpx):
  the 7.5 GB peak came from the float64 temporaries of the colour conversion, which the
  lookup table removes (same bits): a 9000 x 9000 PNG now peaks at 1.9 GB, an 8000 x 8000
  one at 1.8 GB and a 13000 x 13000 one (169 Mpx, just under the cap) at 2.7 GB. Most of
  that is the 1 GB float picture an 8192 x 8192 still is, so a 64 Mpx cap would save little
  and would refuse pictures that drew before.

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
- Cheaper stamping of long sequences (each `state_at` stamps every picture once, about 20 µs
  a picture here).
- The `kit` copies examples without `assets/` (as for `picture.py` before).

## Untested

Windows (CI will run the non-GL tests; libx264 may be absent from PyAV's Windows wheels: the
tests fall back to `mpeg4`, checked here with x264 hidden, and the two open-GOP tests skip;
Windows has no `O_NOFOLLOW`, so a sequence frame there is guarded by its stamp alone; the
fork test skips), macOS, CUDA, GPU goldens, FIFO/symlink tests on Windows (they
skip there).

## Independent audit (2026-11-24) and what was fixed

Reproducers were the auditor's (`/tmp/claude-0/audit24/scripts`, media in `.../w`); each
blocker has a regression test that fails on 61e65a4 and passes now (checked by running the
new `tests/test_video.py` against the old package).

| Finding | Fix | Before -> after (this box) | Test |
| --- | --- | --- | --- |
| B1 open GOP: seeking landed after a leading B-frame's keyframe (mp4 seeks by dts) | keyframes from the probe, seek by the one whose pts is at or before (by min(pts, dts)); exact hits only from a cursor; retry from the keyframe before; stand-in pixels never cached | `det.py`: frames 59, 89 differ (None afresh) -> 0 of 120 differ from a full decode (also bf, vfr, vp9 webm, hevc mkv, edit list, HD); `par.py` and the new parallel test: frame 47 differed -> equal | `test_an_open_gop_clip_gives_every_frame_the_same_in_any_order`, `test_a_parallel_render_of_footage_equals_the_serial_render[open_gop]`, `test_a_frame_a_decoder_failed_to_give_is_drawn_again_when_it_can` |
| B2 edit list: discarded packets with negative pts made the first frame undecodable | `is_discard` packets are not frames (their keyframes still seek); time 0 is the first shown | `editlist.mp4`: `bad_file` -> 108 frames, equal to a full decode | `test_a_clip_with_an_edit_list_starts_at_its_first_shown_frame` |
| B3 large stills 11x slower: 16 B/px past the frame cache, decoded every draw | stills in their own byte-bounded cache (see Deviations) | `photos.py` (two 5000 x 3500 JPEGs, 8 frames): 52.7 s -> 5.1-5.8 s (5.3-5.5 s before Milestone 24); `picture.py` inspector output and the picture golden identical; still sampling bit-identical to 61e65a4 over 160 random views | `test_stills_are_decoded_once_whatever_the_frame_cache_holds`, `test_a_still_reads_as_image_always_did_bit_for_bit` |
| S1 decoder thrash: one per file, 4 in all | cursors per file (4) and in all (8); the cursor just behind a frame decodes on; lint `too_many_clips` past that | `thrash.py` 4 / 5 clips: 21.8 / 1.7 -> 20.3 / 18.8 scene frames/s; `same.py` one file at two times: 2.4 -> 31.4 frames/s (one object 85 -> 111). Six HD decoders: +190 MB, 4 threads each | `test_five_clips_played_together_each_decode_on_without_seeking`, `test_lint_reports_a_trim_from_past_the_end_and_too_many_clips` |
| S2 race: a decoder evicted between the registry and its lock gave None | `_CLOSED` sentinel: take another decoder (3 tries), then a private one | `race.py`: 4 of 720 None -> 0 | `test_a_decoder_closed_by_another_thread_is_replaced_not_a_missing_frame` |
| S3 animated speed played backwards | `Fixed` (see Deviations) | - | `test_a_clips_timing_cannot_be_animated` |
| S4 forked child deadlocked in swscale | each container converts with its own `VideoReformatter`, `threads=1` (bytes equal to `to_ndarray`) | `fds.py`: child hung -> frames equal the parent's; 20 clips leave 8 files open | `test_a_forked_process_decodes_after_its_parent_did` |
| S5 mpeg4 fallback in the test helper shifted colours | mpeg4 at 20 Mbit/s, qmin 1, qmax 2 | `PYTHONPATH=.../plug pytest -p nox264 tests/test_video.py -m "not gl"`: 40 passed, 2 skipped (open GOP needs x264) | the whole file without x264 |
| S6 a 9000 x 9000 PNG peaked at 7.5 GB | conversion through the lookup table | 7482 MB -> 1919 MB peak RSS (see Deviations for the cap) | `test_a_still_reads_as_image_always_did_bit_for_bit` |
| N1 trim in-point past the end: silent | `clip_ended` with its own message (semantics `trim_past_end`) | - | `test_lint_reports_a_trim_from_past_the_end_and_too_many_clips` |
| N2 sequence past 3000: nothing drawn | first 3000 play, `truncated` | - | `test_a_long_sequence_plays_its_first_frames_up_to_the_cap` |
| N3 sequence frames read unchecked | per frame: `O_NOFOLLOW`, stamped time and size | - | `test_a_sequence_frame_changed_or_swapped_for_a_link_is_not_read` |
| N5 `from_video` with an expression raised a bare `float()` error | follows from S3 (`DocumentError`) | - | `test_a_clips_timing_cannot_be_animated` |

Not done: N4 and N6 (skipped as asked: not trivial).

## Shared files touched (merge notes: Milestones 21 and 22 run in parallel)

`docs/ai/API.md`, `llms.txt` (regenerate after merging), `inspect/lint.py` (two codes, a
hint map, the analytic contrast branch), `kit/examples.toml`, `phasor/__init__.py`
(`Video`), `phasor/domains/__init__.py`, `phasor/domains/media.py` (only the `Image` part;
`Sound` untouched), `core/files.py`, `core/extras.py` (`av`), `core/objects.py` (docstring),
`core/pycode.py` (it now writes `meta.duration` for any document that sets one, not only
footage scenes), `engine.py` (`_bring_files`), `render/passes.py` (the field cache key, and
uncached stand-in pixels), `time/compile.py` (the `Fixed` check), `core/objects.py`
(`Fixed`), `scene.py`
(two methods, `BACKGROUND_Z`), `tests/test_final.py`, `STUDIO.md` (row 24). `ruff format`
reformatted the python blocks in `docs/ai/API.md` (expect noise there).

## After the audit (2026-11-24)

Audited independently: 3 blockers + 6 should-fix fixed (and nits N1, N2, N3, N5; N4, N6
not done). ruff check, ruff format --check, lint-imports (2 contracts kept): pass. Focused
suites, each run on its own: test_video 57 passed (GL included); test_lint 32; test_engine
30; test_workers 9; test_render 22 (1 skipped: CUDA); test_beauty 31; test_docs 5;
test_layers 9; test_open_contract 71; test_pycode 5; test_final `-k "not slow"` 426 passed,
4 skipped. Full suite not run (shared box). Untested on Windows, macOS, CUDA.

## Gate (phasor-milestone-gate, 2026-11-24, by the authoring model, not independent)

| Step | Result | Evidence |
| --- | --- | --- |
| ruff check, ruff format --check, lint-imports | pass | all checks passed; 2 contracts kept |
| focused tests | pass | test_video (44), and test_beauty, test_lint, test_docs, test_layers, test_open_contract, test_engine, test_workers, test_render, test_kit, test_mcp, test_pycode, test_sound, test_shapes, test_final `-k "not slow"`: 841 passed, 5 skipped (CUDA; drawless types) before the last fixes; the affected suites re-run after them |
| full suite | not run here | as instructed; CI runs it |
| phasor doctor | pass | tier B (OpenGL llvmpipe, no CUDA), PyAV 19.0.2 with libx264 |
| exit demo | pass on Linux | `examples/video_input.py` (an mp4 as the background, a placed looping inset) and `examples/picture_scene.py` (a png sequence as the background, a still and the sequence placed): storyboards looked at, goldens `tests/golden/software/{video_input,picture_scene}.png`; parallel = serial byte for byte with each as the background; hostile files lint and draw nothing |
| contract audit | done by the authoring model (no tool to start a fresh reviewer) | one write path (`from_video` uses `Engine.apply`); state depends on (document, t, files named in `Files`); `av` only in `domains`; `Video` registers with Params and draw; render workers open their own decoders and refuse changed files (existing frozen stamps); colour linear once |
| docs and artifacts | pass | API.md rows and section (blocks run), examples, goldens (software only; no GPU goldens), this file |

MILESTONE 24: DONE (on Linux, gate by the authoring model, not independent; untested on
Windows, macOS, CUDA).
