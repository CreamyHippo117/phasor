# Milestone 22: audio inputs and narration snap

Plan: `STUDIO.md`, row 22. Owner (2026-11-24): "mp4/wav input and option for scenes to snap to
narration smartly". Built 2026-11-24 in a worktree branched from `claude/eloquent-mayer-jiqxs9`
(`fa4232d`). Exit demo: **a script with narration audio from a WAV or the audio of an mp4;
`snap_to_narration` previews and applies marker moves that land on the sentence starts within a
stated tolerance (50 ms); motions follow; the audio mixes into the render**
(`examples/narrated_from_audio.py`, `tests/test_narration_snap.py`).

Status: **built; audited independently (2026-11-24): 3 blockers + 9 should-fix items and the
nits fixed, each with a regression test that fails on the audited head (`bf178df`) and
passes now (`tests/test_narration_audit.py`); focused gates green on Linux.** Windows, macOS
and CUDA untested; nobody has listened to the audio (its samples and timing were checked);
no real human voice was used (synthetic speech-like signals only).

## Built

| Area | Where | What |
| --- | --- | --- |
| Speech and pauses | `phasor/time/speech.py` (pure NumPy) | `detect(read, frames, rate)`: 10 ms frames read a chunk at a time (2^18 samples), a noise floor that follows the room (10th percentile per 1.5 s window, median of 3 windows), start/stop thresholds above it (hysteresis), pauses under 0.15 s closed, speech under 0.08 s dropped; `quality` 0..1 and `flags` (`silence`, `constant_noise`, `low_snr`, `clipping`, `music` (a tonal floor), `no_pauses`, `short`, `cut`, `unreliable`); at most `MAX_SECONDS` (3 h) looked at; NaN/inf samples ignored; any rate 1-384 kHz, stereo mixed down |
| Sentence alignment | `speech.align` | dynamic programming over the detected segments: sentence *i* covers a run of segments whose speech length matches its text (`estimate_duration`) at the speaking rate measured over the whole recording (log-ratio cost), breaks are rewarded on long gaps and long gaps inside a sentence cost; a `[[pause]]` asks for a long gap; leading/trailing noise segments may be skipped; fewer segments than sentences: the longest is split at a quiet dip. Per-sentence confidence. `phrases` (no script), `pieces` (where `[[markers]]` inside a sentence fall: in proportion to their text, moved onto a speech onset within a quarter of the sentence) |
| Snap plan | `speech.plan_snap` | an order-keeping DP matching of hand-placed markers to boundaries (cost: distance in the markers' own seconds; the scene stretched to the narration's length is tried too and used only when it fits clearly better (half the cost); at most `[narration] snap_reach` = 0.1 of the narration's length; phrase boundaries cost a quarter reach more); unmatched markers interpolated piecewise linearly between anchors; owned (`[[...]]`) markers fixed and never crossed; spans never shorter than what the motions anchored at the marker need (at least the theme's shortest verb) unless `allow_short`, else the least sure anchor of the offending span is dropped; repeated on its own result to a fixed point (idempotent); `anchors` pin |
| Schema | `core/schema.py`, `core/migrate.py` | `NarrationBlock.audio_span: (a, b) | None`: the part of `audio` a block plays (None: the whole file, as before); finite, 0 <= a <= b <= 3 h; `SCHEMA_VERSION` 10, migration 9 -> 10 (blocks get `None`) |
| Mixing | `time/narration.py` | the `Source` protocol (rate, frames, `read(skip, count)`) and `WavSource`; `Clip.source` (Milestone 23's synthesizer plugs in here); `_Planned` reads through a source with a `skip`; `mix_track(..., resolve=)`: span blocks are mixed a chunk at a time like clips (their samples count toward `MAX_MIX_SAMPLES`, capped by the track's 3 h, not `MAX_CLIP_FRAMES`); a missing/bad recording is said in `problems`, not raised; the whole-file (TTS) branch is untouched; `place_recorded` builds the narration from found sentences (0.03 s pre-roll, 0.05 s post-roll, never overlapping) |
| Decoding | `phasor/domains/audio.py` | `readable_wav`: a WAV as it is (PyAV never imported); `.mp4 .m4a .m4v .mov .mp3 .aac .ogg .oga .opus .flac .webm .mkv` decoded once with PyAV to 16-bit mono WAV (8-48 kHz) in `files.user_cache("audio")` (`PHASOR_AUDIO_CACHE`), named by resolved path + mtime + size + a digest of the first and last 64 KiB + decoder version; opened non-blocking and checked regular (a FIFO never blocks); bounded by 4 GiB, 3 h of sound, 600 s of work; written to a temp name and renamed; any PyAV error a `ValueError`; `info`, `speech` (cached per file version), `narration_boundaries` |
| Narration from a file | `voice.from_audio`, `Scene.narrate(script="", *, voice, audio=None)` | with a script: sentences aligned, `[[markers]]` on their words; without: phrases (`[narration] phrase_pause` = 0.35 s) as blocks of text `"…"`; voice `file:<path>`; the recording plays from 0 s |
| Snap | `Engine.snap_to_narration(dry_run, anchors, allow_short, origin)`, `Scene.snap_to_narration`, `Workspace.snap_to_narration` (MCP; `dry_run` default true) | returns `moves`/`kept` (marker, from, to, why, confidence, kind), `quality`, `flags`, `unreliable`; applies as one `apply([SetMarker...])` (one undo step); on a scripted scene a tweak is refused (`scripted_scene`) and the tool says to put it in the script |
| Lint | `inspect/lint.py` `narration_rules` | `missing_file` / `bad_file` / `missing_extra` for the recording (a probe injected by the engine: this layer decodes nothing); `narration_drift`: a marker the snap plan (injected, the same function) would move by more than `[lint] narration_drift` = 0.3 s; times = the marker's; no fix (hint: the snap). Sound: `missing_extra` for a non-WAV clip without PyAV; `bad_file`'s hint for sound files |
| Sound | `domains/media.py` | `Sound.path` takes any of the sound extensions; semantics from `domains.audio.info`; without PyAV a non-WAV clip reports `needs: "av"` (no `bad_file`); `extras.EXTRAS["av"]` |
| Engine | `engine.py` | `audio_track` passes `resolve=readable_wav`; `file_stamps` includes the recordings narration is cut from (lint re-reads on change) |
| Editor, code view | `gui/controller.py`, `gui/window.py`, `engine.timeline`, `core/pycode.py` | the timeline's narration blocks carry `span`; `Editor.waveform(audio, bins, span)` reads only that part (any format `domains.audio` reads; a hostile file is an empty list), so a recording is drawn block by block, never whole per block; the generated script writes a recorded narration as `s.narrate(script, audio=...)` (it rebuilds the same document) |
| Mux fix | `render/encode.py` | `mux_audio` gives the AAC frame a time (`pts` 0), so the encoder's 1024-sample priming is written as a skip; before, an 8 kHz narration was heard 0.13 s late (found by this milestone's end-to-end render; measured with ffmpeg and PyAV) |
| Theme | 4 theme files, `style/theme.py` | `[narration] phrase_pause`, `snap_reach`; `[lint] narration_drift` |
| Docs, example | `docs/ai/API.md`, `docs/ai/MCP.md`, `llms.txt`, `SPEC.md`, `kit/examples.toml` | "Narration from a recording, and snapping the scene to it" (a running block); `snap_to_narration` tool; `examples/narrated_from_audio.py` with `examples/assets/narration_demo.wav` (150 KB, 9.4 s, 8 kHz, synthetic) |

## Accuracy (synthetic ground truth, this machine; after the audit)

Formant-shaped word bursts (`tests/test_speech.recording`, `tests/test_narration_audit.speech`),
words 0.18-0.45 s, word gaps 0.04-0.3 s, sentence pauses 0.35-0.8 s, white noise at the SNR;
variants below. Only what is listed was measured.

- Sentence starts (`align`, 6 sentences x 5 seeds unless said; > 250 ms counts as wrong):
  baseline, fast speaker, 2-word sentences, lengths 1 to 20 words, 3 x 40 words, one 70-word
  sentence, one or two mid-sentence pauses of 0.6 to 1.5 s, 12 dB SNR: 0 wrong, max 37 ms.
  Still wrong and **flagged** (`alignment_doubtful`, confidence at most 0.3): a reader who
  does not pause between sentences (0.05-0.12 s, word gaps up to 0.3 s): 25 of 30 wrong, all
  5 runs flagged; a slow speaker (word gaps 0.15-0.6 s as long as sentence pauses): 2 of 30
  wrong, flagged (with 2 false alarms in 5 correct runs).
- End to end (`narrate(script, audio=)` + `snap_to_narration()`, 4 hand markers, 10 seeds):
  40 dB median 28 ms, max 33 ms; 30 dB 21 / 27 ms; 20 dB 15 / 22 ms; 10 dB 5 / 20 ms
  (flagged `low_snr`, `unreliable`). These include the deliberate 0.03 s pre-roll.
  Stated tolerance: 50 ms, tested at 30 and 15 dB.
- The scene on another clock (24 sentences, a marker on every second): scale x0.8 to x1.25,
  an offset of +2 s, drift x1.1 in the second half: 12 of 12 on their sentence, not flagged.
  Drift x0.9 in the second half, and x0.8 with only 3 markers on 6 sentences: two readings
  fit almost equally, so the result is `ambiguous_timing`, `unreliable`, nothing moves.
  Sentences all of one length (a pathological script) with markers 0.5-2 s off: ambiguous,
  nothing moves (before: confident, markers 12 s off).

## Bounds measured (wall clock, `tracemalloc` peak)

- `detect` on a 60-minute 16 kHz WAV streamed from disk: 1.1 s, 24 MiB.
- `align` of 1000 sentences to 4691 segments: 4.0 s, 19 MiB (was 21.5 s before the per-width
  costs were hoisted).
- `plan_snap` with 1000 markers and 20000 boundaries: 13 s under tracemalloc (7.6 s without),
  21 MiB (was 56 s, 459 MiB: the cost matrix is now made a row at a time and the
  interpolation is vectorized).
- Mixing 1000 span blocks of a one-hour file: 0.7 s, 119 MiB (the int16 track itself).
- Decoding a 10-minute m4a: 0.85 s; again (cached): 0 s.
- A 2 GiB sparse `.mp4`: a clean `ValueError` in well under 30 s, under 64 MiB (test).
- 400 overlapping blocks over one file: bounded by `MAX_MIX_SAMPLES`, said, under 64 MiB (test).

## Verification

After the audit fixes (2026-11-24), each suite on its own (`-m "not slow"`): test_speech 28,
test_audio_inputs 21, test_narration_snap 31, test_narration_audit 38, test_sound 40,
test_narration 23 (2 skipped: espeak-ng), test_style_freedom 42, test_docs 5, test_layers 9,
test_lint 32, test_engine 30, test_mcp 39, test_render 22 (1 skipped: CUDA),
test_gui_controller 20, test_pycode 5, test_final 419 (4 skipped) passed, 0 failed; the slow
golden passed; ruff, ruff format and lint-imports clean. The full suite was not run (CI).

Gate: **audited independently, 3 blockers + 9 should-fix (and the nits) fixed**; the
pre-audit gate table, run by the authoring model:

| Step | Result | Evidence |
| --- | --- | --- |
| ruff check, ruff format --check, lint-imports | pass | clean; 2 contracts kept |
| Focused tests | pass | 23 files `-m "not slow"`: 628 passed, 7 skipped (espeak-ng, CUDA, Windows-only); after the last changes the touched suites again: 274 passed, 3 skipped; `test_final -k "not slow"`: 418 passed + the one failure it found (the generated script dropped a recorded narration) fixed and rerun; slow golden 1 passed |
| Full suite | not verified here | not run (over an hour; CI runs it) |
| `phasor doctor` | pass | tier B (OpenGL llvmpipe, no CUDA), PyAV 19.0.2, ffmpeg 6.1.1, all extras |
| Exit demo | pass on Linux | WAV: `examples/narrated_from_audio.py` (markers 30 ms before the true starts, lint clean), rendered with 2 workers "with sound", track onset 0.60 s; mp4: a scene narrated from `talk.mp4` (a video file with a sound track) rendered with 2 workers, onsets in the video 0.62 / 3.65 / 6.35 s vs true 0.60 / 3.64 / 6.33 s |
| Contract audit | not verified here | no independent model was available; the author's own review: writes only through `Engine.apply` (`snap_to_narration` applies one `SetMarker` list); `state_at` unchanged; decoding only in `domains` (lint and mix receive injected functions); `Sound` keeps every member; one WAV reader (`parse_wav`), one way to read sound files (`domains.audio.readable_wav`) |
| Docs and artifacts | pass | API.md (running block), MCP.md (`json calls` block runs), llms.txt regenerated, example + asset, golden looked at, this file |

- `uv run ruff check .`, `ruff format --check .`, `lint-imports` (2 contracts kept): clean.
- New tests: `tests/test_speech.py` (28), `tests/test_audio_inputs.py` (21),
  `tests/test_narration_snap.py` (29, one `gl`+`slow` golden).
- Focused suites (`-m "not slow"`): see the report of the gate below.
- `phasor render examples/narrated_from_audio.py --preset preview --fps 10 --workers 2`: 88
  frames "with sound"; the decoded track (ffmpeg) starts at 0.60 s after the mux fix (0.75 s
  before it), as the mixed track does.
- Golden (looked at): `tests/golden/software/narrated_from_audio.png`.

## Deviations

1. **Schema 10** (`NarrationBlock.audio_span`). A recording is cut into one block per
   sentence by span, not copied into per-sentence WAVs in the scene's folder: the document
   names the file, the decoded copy lives only in the user's cache, and `repace` keeps each
   sentence whole. Older documents migrate (their blocks play whole files, as before); the
   TTS mixing branch is unchanged. `tests/test_style_freedom.py` asserted version 9; now 10.
2. `media.Sound` accepted only `.wav`; it now accepts the sound extensions above (the M20 test
   that `a.mp3` is refused now uses `a.txt`).
3. Phrase blocks (no script) have the text `"…"`: an empty text means a pause to `repace` and
   the timeline.
4. `narration_drift` runs only for a narration cut from a recording (blocks with spans), so no
   existing scene gains an issue; its threshold is the snap plan's own move.
5. The snap moves only markers placed by hand; markers the narration owns (`[[...]]`) are
   already on their words and are fixed points. A marker sitting on a phrase onset is a valid
   snap target too ("speech resumes after a pause").
6. `render/encode.py` (not this milestone's area): the AAC priming fix above, with a test at 8,
   22.05 and 48 kHz. Every render with sound now starts its track 1024 samples earlier than
   before (21 ms at 48 kHz): correct per ffmpeg and PyAV, not heard in a player.
7. A render worker's file check (`file_stamps`) now includes the recording a narration is cut
   from: re-recording it while a render runs stops the render (as for any file the scene
   names).
8. `render/cpu/backend.cache_dir` now uses `core/files.user_cache` (factored out, same paths).
9. Audit: the high-pass before detection is zero-phase, so a sudden onset in digital silence
   is found up to about 20 ms early; the tone-burst tests allow 25 ms (was 20) and the
   API.md example 80 ms (was 50). Real rooms have noise and do not show it.
10. Audit: internal return types grew (`align` -> `Alignment`, `plan_snap` -> `SnapPlan`,
    `narration_boundaries` 4 values, `_plan_once` 2); all are lists or tuples as before plus
    attributes, and none is a documented public API outside this milestone.

## Audit (independent, 2026-11-24): findings and fixes

Principle: a confident wrong answer must never look like a right one. "Before" is the
audited head `bf178df`; every row has a test in `tests/test_narration_audit.py` (37 tests; 36
fail on the audited head, the 37th, `[40, 40, 40]` coverage, already passed there).

| Finding | Before | Fix | After |
| --- | --- | --- | --- |
| B1 snap under clock mismatch | x0.9: 10/12 markers on the wrong sentence, x1.1 10/12, drift 4/12, +2 s offset 7/12; unreliable False, confidence >= 0.57 | `speech.fit_clock`: scale (0.6-1.6) and offset by a vote over offsets per scale, scored on every marker, a mild prior for no change (offset in sentence spacings); markers then matched by a Viterbi chain over the steps between markers (`_chain`), so drift lands; a close second reading that moves markers elsewhere makes the plan `ambiguous`: nothing moves, flag `ambiguous_timing`, `unreliable`; `SnapPlan.fit` -> result `alignment` | 0/12 at x0.8, x0.9, x1.05, x1.1, x1.25, +2 s, drift x1.1; drift x0.9 ambiguous (flagged, kept) |
| B2 speech dropped | a 100-word sentence lost 7.5 s + 2 s; a script without full stops lost 10.6 s | no skipping of segments at the ends; the width a sentence may span grows with the longest sentence | every detected segment inside a block; 100% of the energy kept |
| B3 misleading pauses | two 1.2 s mid-sentence pauses: 16/30 starts > 250 ms off (4.6 s), confidence 0.6-0.85; run-on 25/30; slow 4/30 | sentence length counted in words only (punctuation pauses are silence); length weighted 2x against gaps; long segments cut at quiet dips (a break there costs more); `Alignment.doubtful` when a sentence is 1.8x off its text, the median misfit is large, or sentence breaks are hardly longer than gaps inside sentences; confidences then <= 0.3; flag `alignment_doubtful` | mid-sentence pauses 0/30; run-on and slow: still wrong, all flagged (accuracy above) |
| S1 anchors reorder | `{"poles": 3}` crossed owned markers | `engine._check_pins`: refused with a clear `ValueError` | refused |
| S2 Scene snap refused silently | returned `applied: False` | `Scene.snap_to_narration` raises `DocumentError` | raises |
| S3 DC and hum | DC 0.1: no speech; DC 0.02, hum 0.03: music | `_highpass` (mean removed, 60-100 Hz linear step, zero-phase FFT with 4096-sample overlap), music flatness above 100 Hz, floor never below -80 dBFS | DC 0.1/0.5, hum 0.08/0.3: same segments, no flags |
| S4 float WAV | "format 3 (compressed)" | `parse_wav` reads 32/64-bit float (plain and extensible); 16-bit float, ADPCM still refused | read exactly |
| S5 track size | block at 10700 s in a 5 s scene: 979 MiB; 192 kHz MemoryError | recording blocks more than `SPAN_MARGIN` (60 s) past the scene are left out and said (lint `sound_limit`); the track at most `MAX_TRACK_SAMPLES` (3 h at 48 kHz) | nothing allocated |
| S6 decode bomb, cache | 1.3 MB Opus -> 330 MB; never pruned | output at most 64x the file (and 256 MiB); cache pruned oldest first to 4 GiB | refused in 7 s, 149 MiB peak |
| S7 doubt in the result | only detection quality; missing file quality 1.0 | result `alignment` (clock fit + `speech`: rate, misfit, worst, doubtful); missing recording: quality 0, `unreadable`, unreliable | |
| S8 lint re-aligned | 1000 sentences 11 s per lint | `domains.audio.aligned` cached by file stamp + script | a 25-min, 300-sentence lint: narration rules no longer visible in the cost |
| S9 Windows replace | PermissionError, cached | a held destination that exists is success; `OSError` never cached | (by reading; untested on Windows) |
| Nits | | docstring fixed; PyAV errors "cannot be decoded: <reason>"; the cache folder error names `PHASOR_AUDIO_CACHE`; a decode cut at 3 h is `<key>.cut.wav` and detection flags `cut`; same-size same-mtime middle edits documented | |

Also changed by the audit work: `place_recorded` / `from_audio` use the cached alignment;
`narration_boundaries` returns the alignment too (4 values); `align` returns an `Alignment`
(a list of `Placed` with `rate`, `misfit`, `worst`, `doubtful`); `plan_snap` returns a
`SnapPlan` (a list of `Move` with `fit`). Measured cost: `align` of 2000 sentences over
16000 segments 34.5 s (once per recording and script, then cached).

## Deferred

1. (Done in the audit: the decode cache is pruned.)
2. `Engine._bring_files` does not copy narration audio (recordings or TTS clips) when a scene
   is saved into another folder (pre-existing for TTS).
3. The app button (Milestone 25); MIDI clips through `Source` (Milestone 23).
4. Word-level alignment inside a sentence is proportional to the text (refined to pauses), not
   phonetic; mid-sentence `[[markers]]` are less exact than sentence starts.
5. The music detector is a heuristic (a tonal floor); speech over loud broadband music may not
   be flagged.
6. Associating hand markers with sentences by the narration they were laid out against (an
   estimated narration of the same script, replaced by the recording: block k is the same
   sentence), instead of by time alone; until then scenes whose timing fits two ways are
   flagged ambiguous and need `anchors`.
7. The in-process caches (`domains.audio._info`, `_speech`) are keyed by path, mtime and size
   only (the on-disk decode cache adds a content digest): a file rewritten with the same size
   and modification time is served stale until the process restarts.

## Untested

- Windows and macOS (tests avoid POSIX-only calls; the FIFO, symlink and sparse-file tests skip
  where unsupported; espeak-ng is absent here as on CI), CUDA (nothing here touches kernels).
- Real recordings of a human voice (only synthetic speech-like signals were used).
- Listening: no one has heard the narration or the mixed video; samples and timing were
  checked numerically.
- An independent audit and the contract review by a model that did not write the code.
