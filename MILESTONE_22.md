# Milestone 22: audio inputs and narration snap

Plan: `STUDIO.md`, row 22. Owner (2026-11-24): "mp4/wav input and option for scenes to snap to
narration smartly". Built 2026-11-24 in a worktree branched from `claude/eloquent-mayer-jiqxs9`
(`fa4232d`). Exit demo: **a script with narration audio from a WAV or the audio of an mp4;
`snap_to_narration` previews and applies marker moves that land on the sentence starts within a
stated tolerance (50 ms); motions follow; the audio mixes into the render**
(`examples/narrated_from_audio.py`, `tests/test_narration_snap.py`).

Status: **built; focused gates green on Linux; no independent audit yet.** Windows, macOS and
CUDA untested; nobody has listened to the audio (its samples and timing were checked).

## Built

| Area | Where | What |
| --- | --- | --- |
| Speech and pauses | `phasor/time/speech.py` (pure NumPy) | `detect(read, frames, rate)`: 10 ms frames read a chunk at a time (2^18 samples), a noise floor that follows the room (10th percentile per 1.5 s window, median of 3 windows), start/stop thresholds above it (hysteresis), pauses under 0.15 s closed, speech under 0.08 s dropped; `quality` 0..1 and `flags` (`silence`, `constant_noise`, `low_snr`, `clipping`, `music` (a tonal floor), `no_pauses`, `short`, `cut`, `unreliable`); at most `MAX_SECONDS` (3 h) looked at; NaN/inf samples ignored; any rate 1-384 kHz, stereo mixed down |
| Sentence alignment | `speech.align` | dynamic programming over the detected segments: sentence *i* covers a run of segments whose speech length matches its text (`estimate_duration`) at the speaking rate measured over the whole recording (log-ratio cost), breaks are rewarded on long gaps and long gaps inside a sentence cost; a `[[pause]]` asks for a long gap; leading/trailing noise segments may be skipped; fewer segments than sentences: the longest is split at a quiet dip. Per-sentence confidence. `phrases` (no script), `pieces` (where `[[markers]]` inside a sentence fall: in proportion to their text, moved onto a speech onset within a quarter of the sentence) |
| Snap plan | `speech.plan_snap` | an order-keeping DP matching of hand-placed markers to boundaries (cost: the smaller of the distance in place-in-scene vs place-in-narration, and plain distance; at most `[narration] snap_reach` = 0.1 of the narration's length; phrase boundaries cost a quarter reach more); unmatched markers interpolated piecewise linearly between anchors; owned (`[[...]]`) markers fixed and never crossed; spans never shorter than what the motions anchored at the marker need (at least the theme's shortest verb) unless `allow_short`, else the least sure anchor of the offending span is dropped; repeated on its own result to a fixed point (idempotent); `anchors` pin |
| Schema | `core/schema.py`, `core/migrate.py` | `NarrationBlock.audio_span: (a, b) | None`: the part of `audio` a block plays (None: the whole file, as before); finite, 0 <= a <= b <= 3 h; `SCHEMA_VERSION` 10, migration 9 -> 10 (blocks get `None`) |
| Mixing | `time/narration.py` | the `Source` protocol (rate, frames, `read(skip, count)`) and `WavSource`; `Clip.source` (Milestone 23's synthesizer plugs in here); `_Planned` reads through a source with a `skip`; `mix_track(..., resolve=)`: span blocks are mixed a chunk at a time like clips (their samples count toward `MAX_MIX_SAMPLES`, capped by the track's 3 h, not `MAX_CLIP_FRAMES`); a missing/bad recording is said in `problems`, not raised; the whole-file (TTS) branch is untouched; `place_recorded` builds the narration from found sentences (0.03 s pre-roll, 0.05 s post-roll, never overlapping) |
| Decoding | `phasor/domains/audio.py` | `readable_wav`: a WAV as it is (PyAV never imported); `.mp4 .m4a .m4v .mov .mp3 .aac .ogg .oga .opus .flac .webm .mkv` decoded once with PyAV to 16-bit mono WAV (8-48 kHz) in `files.user_cache("audio")` (`PHASOR_AUDIO_CACHE`), named by resolved path + mtime + size + a digest of the first and last 64 KiB + decoder version; opened non-blocking and checked regular (a FIFO never blocks); bounded by 4 GiB, 3 h of sound, 600 s of work; written to a temp name and renamed; any PyAV error a `ValueError`; `info`, `speech` (cached per file version), `narration_boundaries` |
| Narration from a file | `voice.from_audio`, `Scene.narrate(script="", *, voice, audio=None)` | with a script: sentences aligned, `[[markers]]` on their words; without: phrases (`[narration] phrase_pause` = 0.35 s) as blocks of text `"…"`; voice `file:<path>`; the recording plays from 0 s |
| Snap | `Engine.snap_to_narration(dry_run, anchors, allow_short, origin)`, `Scene.snap_to_narration`, `Workspace.snap_to_narration` (MCP; `dry_run` default true) | returns `moves`/`kept` (marker, from, to, why, confidence, kind), `quality`, `flags`, `unreliable`; applies as one `apply([SetMarker...])` (one undo step); on a scripted scene a tweak is refused (`scripted_scene`) and the tool says to put it in the script |
| Lint | `inspect/lint.py` `narration_rules` | `missing_file` / `bad_file` / `missing_extra` for the recording (a probe injected by the engine: this layer decodes nothing); `narration_drift`: a marker the snap plan (injected, the same function) would move by more than `[lint] narration_drift` = 0.3 s; times = the marker's; no fix (hint: the snap). Sound: `missing_extra` for a non-WAV clip without PyAV; `bad_file`'s hint for sound files |
| Sound | `domains/media.py` | `Sound.path` takes any of the sound extensions; semantics from `domains.audio.info`; without PyAV a non-WAV clip reports `needs: "av"` (no `bad_file`); `extras.EXTRAS["av"]` |
| Engine | `engine.py` | `audio_track` passes `resolve=readable_wav`; `file_stamps` includes the recordings narration is cut from (lint re-reads on change) |
| Mux fix | `render/encode.py` | `mux_audio` gives the AAC frame a time (`pts` 0), so the encoder's 1024-sample priming is written as a skip; before, an 8 kHz narration was heard 0.13 s late (found by this milestone's end-to-end render; measured with ffmpeg and PyAV) |
| Theme | 4 theme files, `style/theme.py` | `[narration] phrase_pause`, `snap_reach`; `[lint] narration_drift` |
| Docs, example | `docs/ai/API.md`, `docs/ai/MCP.md`, `llms.txt`, `SPEC.md`, `kit/examples.toml` | "Narration from a recording, and snapping the scene to it" (a running block); `snap_to_narration` tool; `examples/narrated_from_audio.py` with `examples/assets/narration_demo.wav` (150 KB, 9.4 s, 8 kHz, synthetic) |

## Accuracy (synthetic ground truth, this machine)

Formant-shaped word bursts (`tests/test_speech.recording`), words 0.18-0.45 s with gaps of
0.04-0.3 s (comma-length gaps overlap sentence pauses of 0.35-0.8 s), white noise at the SNR.

- `align` alone (sentence start found vs true, 12 sentences, 3 seeds): 40 dB median 2 ms,
  max 8 ms; 20 dB median 14 ms, max 24 ms; 10 dB median 30 ms, max 47 ms (flagged
  `low_snr`, `unreliable`).
- End to end (`narrate(script, audio=)` + `snap_to_narration()`, 4 hand markers, 10 seeds):
  40 dB median 28 ms, max 33 ms; 30 dB median 21 ms, max 27 ms; 20 dB median 15 ms, max
  22 ms; 10 dB median 5 ms but one marker of 40 off by 1.44 s (the run said `unreliable`).
  These include the deliberate 0.03 s pre-roll (markers land 30 ms before the voice, as the
  TTS clips do); detection lateness at low SNR partly cancels it. Stated tolerance: 50 ms,
  tested at 30 and 15 dB (`test_snapped_markers_land_within_the_stated_tolerance_with_noise`).

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
   22.05 and 48 kHz.
7. `render/cpu/backend.cache_dir` now uses `core/files.user_cache` (factored out, same paths).

## Deferred

1. The decode cache is never pruned (each version of each file stays until removed by hand).
2. `Engine._bring_files` does not copy narration audio (recordings or TTS clips) when a scene
   is saved into another folder (pre-existing for TTS).
3. The app button (Milestone 25); MIDI clips through `Source` (Milestone 23).
4. Word-level alignment inside a sentence is proportional to the text (refined to pauses), not
   phonetic; mid-sentence `[[markers]]` are less exact than sentence starts.
5. The music detector is a heuristic (a tonal floor); speech over loud broadband music may not
   be flagged.

## Untested

- Windows and macOS (tests avoid POSIX-only calls; the FIFO, symlink and sparse-file tests skip
  where unsupported; espeak-ng is absent here as on CI), CUDA (nothing here touches kernels).
- Real recordings of a human voice (only synthetic speech-like signals were used).
- Listening: no one has heard the narration or the mixed video; samples and timing were
  checked numerically.
- An independent audit and the contract review by a model that did not write the code.
