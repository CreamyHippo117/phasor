# Milestone 23: MIDI

Plan: `STUDIO.md`, row 23. Owner (2026-11-24): "midi output ... a midi input option for you to
overlay with the audio at certain points", read there as **one type with an import end, a
play end and an export end**. Built 2026-11-24 in a worktree branched from
`claude/eloquent-mayer-jiqxs9` (merged at `9fa24f7`: Milestones 21, 22, 24, 26, 27).
Exit demo: **a narrated scene (a recording, M22) with notes a script makes and a `.mid` file
played over it from markers, heard in the mixed track where the markers are, and the notes
written back to a `.mid` that reads back the same** (`examples/notes_overlay.py`,
`tests/test_midi.py`).

Status: **built; audited independently (2026-11-24): 1 blocker + 8 should-fix items and the
nits fixed, each with a regression test that fails on the audited head (`6b5a4b8`) and passes
now (`tests/test_midi_audit.py`: 30 tests, 29 failed on the head; the 30th, gain linearity,
is a guard that already held); focused gates green on Linux.** Windows, macOS and CUDA
untested; **nobody has listened to the synthesizer** (pitch, timing, levels, aliasing and
determinism are checked numerically).

## Built

| Area | Where | What |
| --- | --- | --- |
| SMF reader | `domains/midi_math.py` `read_smf` | format 0 and 1; PPQ and SMPTE (24, 25, 29.97, 30 fps) time; running status; note on/off (velocity-0 note-on is an off; one key of one channel pairs first in, first out; a note still held at the end of its track ends there); program change (each note gets its channel's program at its tick, across tracks); meta tempo (a tempo map over all tracks), time signature, track name; sysex and other metas skipped; unknown chunks skipped; a header longer than 6 bytes skipped past. Returns a `Song`: `notes` (`Note(pitch, start s, length s, velocity, channel, program)`, sorted by start, channel, pitch), `format`, `tracks`, `names`, `division`, `bpm`, `tempo_changes`, `time_signature`, `error` (nothing read) or `cut` (read up to a fault or a bound) |
| Bounds | `midi_math` | file 4 MiB, 256 tracks, 400 000 events, 100 000 notes (checked before any array is built), 10 000 tempo changes, 1024 chunks, 2^40 ticks a track, notes after 3 h left out, notes longer than 60 s cut to it, tempos outside 1-1000 bpm ignored (said), a VLQ of more than 4 bytes, a data byte past 127, running status with no status, a real-time status in a file, a meta/sysex/track length past what is there: that track stops there (said). Format 2 is refused with a reason ("save it as format 0 or 1"); format 7, 0 tracks, 0 division, SMPTE with another frame rate or 0 ticks a frame: refused. Never raises (any other exception is said as `error`) |
| SMF writer | `midi_math.write_smf` | format 1: a conductor track (title, 120 bpm, 4/4), then one named track per part; 960 ticks a quarter (1/1920 s); running status; note-offs before program changes before note-ons at one tick; programs set in time order before every note that needs another one (never on channel 10); a note struck again on its key while held ends there; names Latin-1, or UTF-8 when they need it; pitches rounded and clamped to 0-127, velocities to 1-127, a note at least one tick; a note after 3 h, or any time past 2^28 ticks, is a `ValueError` (never written for the reader to refuse) |
| Synthesizer | `midi_math.Synth` (a `time.narration.Source`) | 48 kHz, or the track's rate (`at_rate`, asked for by the mixer: no resampling, partials band-limited to 0.45 of it); `gain` scales each voice before the sum, which goes through a soft limiter (`KNEE` 0.5: unchanged below, bending towards 1 above, never reaching it); voices `sine`, `pluck` (24 harmonics, decaying, low notes ring longer), `organ` (drawbars), `bell` (inharmonic partials 1, 2.76, 5.4, 8.93, each dying faster), `pad` (a detuned second oscillator, slow attack), `lead` (saw-like); General MIDI program families map onto them (`timbre_of`), channel 10 is a drum kit (kick: a falling sine; snare: noise and a tone; hats and cymbals: differenced noise; toms: tuned sines; the rest a noise tick) whose noise is `rng(object_id, seed)`; ADSR (attack up from 0, exponential decay to sustain, a release that ends at 0); velocity as (v/127)^2 x 0.18 (`LEVEL`; a bell's partials normalized to peak where any voice does); drum hits fade linearly over their last quarter to 0; at most `MAX_VOICES` = 64 notes sounding at once (with their release; a note starting when 64 sound is dropped, counted, its time kept); harmonics above 0.45 x 48 kHz are left out (band-limited tables) |
| Determinism | `Synth.read` | every sample is a pure function of the notes, the seed and its index: integer phases (32-bit), fixed 4096-entry waveform and exp tables with linear interpolation (only +, -, x, /, floor and indexing on varying arrays), voices summed in one order in float64, cast to float32 once, made 2^16 samples at a time inside a read. Tested: any window equals that slice of the whole (every instrument, odd offsets), the same sha256 in another process, the mixer's window equals the slice of the whole mix on the resampled path |
| Mixing | `time/narration.py`, `engine.py` | the synthesizer is a `Clip.source` (M22's protocol): the mixer asks a source that has `at_rate` for its track's rate (a WAV has none: its path is untouched), then reads, budgets (`MAX_SOUNDS`, `MAX_MIX_SAMPLES`) and cuts it exactly as a WAV clip. `Clip.marker` (new, default None): a clip that starts at a marker; `Engine._at_markers` resolves it before mixing (an unknown marker: left out and said in `problems`). TTS, Sound and M22 recording mixes are **bit-identical** to before (`tests/mix_baseline.py`: five sha256 values taken on the merged commit before any M23 change, integer-only test signals so they hold on any system) |
| `Notes` | `domains/midi.py` | `path` (a `.mid`/`.midi`/`.smf` in the scene's folder, through `core/files.py` (`stamp`, `read_bytes`: regular files only, never blocking on a FIFO), cached by stamp so it is read again when it changes) or `notes` (the script's `(pitch, start, length, velocity[, channel])`, at most 20 000, each checked; a fractional pitch plays its exact frequency), never both; `start` (seconds or a marker's name), `gain`, `transpose` (never drums), `tempo_scale`, `channels`, `repeat` (1-64, bounded by `MAX_NOTES`), `instrument`, `program`, `seed`; `show` with `pos`, `width`, `height`. The audio parameters are `Fixed` (a track on one is `bad_param`); `Life` none. Semantics: `kind`, `source`, `start`, `gain`, `count`, `instrument`, `drawn`, `span`, `pitch_range`, `channels`, `programs` (GM names, "Drum kit"), `voices`, `voices_dropped`, `out_of_range`, and for a file `file`, `missing`, `format`, `tracks`, `bpm`, `names`, `time_signature`, `unreadable`, `truncated`. Bounds: a point, or the roll's box |
| Piano roll | `Notes.draw` | with `show`: a frame and one bar per note (time across, pitch up) with `segment_batch`; with `start` in seconds, notes sounding at `t` take the `played` role and a playhead runs. Golden `tests/golden/software/notes_overlay.png` (looked at) |
| Export | `Engine.export_midi(path, overwrite=False)`, `Scene.export_midi`, `Workspace.export_midi(out, overwrite)` (MCP tool `export_midi`) | every `Notes` (the notes as the synthesizer plays them) at its times in the scene, one track per object named by its id, the scene's title on the conductor track; a channel per (object, program) (drums stay on 10; past 15 pairs shared, with program changes); `instrument` written as its program (`PROGRAM_OF`); an object at a marker the scene lacks is left out, notes after 3 h too (`left_out`). Never over a file the scene reads (`ValueError`, even with `overwrite`), nor over an existing file without `overwrite` (`FileExistsError`); written to a temporary file beside it and `os.replace`d. Returns `path`, `tracks`, `notes`, `left_out`, `seconds`. The workspace tool writes inside the workspace only, `.mid`/`.midi` only, default `<scene>.mid` in `out/` |
| Lint | `inspect/lint.py` | `notes_limit` (new, `notes_rules`, injected state like `narration_rules`): notes starting at or after the scene's end (times: the end), notes dropped past the voices (times: the first drop), a `start` marker the scene does not have. `missing_file` (from `Files`), `bad_file` (`unreadable`; hint "export it again as a Standard MIDI File (format 0 or 1)"), `truncated` (`cut`). `"notes"` is a drawless kind (never `orphan`). `_sound_rules`' PyAV check now looks only at `sound` objects (a `.mid` never needs PyAV) |
| Docs, example | `docs/ai/API.md`, `docs/ai/MCP.md`, `llms.txt`, `SPEC.md`, `kit/examples.toml` | an object-table row, "MIDI notes: played over the narration, and exported" (a running block), the lint row; the MCP tool row with a `json patch` and a `json calls` block (both run in CI); `examples/notes_overlay.py` (a square wave's partials from `glide` on a sine voice, shown as a roll revealed there; `assets/motif.mid`, 251 bytes, from `fade`), the asset made by `examples/assets/make_assets.py motif` |

## Measured (this machine, a shared 4-core box, with another test run going)

- 1000 notes over 10 minutes (60 voices of pluck/organ/pad/lead/bell mixed, two channels):
  the synthesizer alone, read as the mixer reads it (2^20 samples at a time): 4.0 s, 17 MiB
  peak (tracemalloc; 5.3 s under it). The whole `audio_track` of the 602 s scene at 48 kHz:
  4.0 s, 75 MiB peak (the int16 track itself is 55 MiB). Test bounds: 120 s and 200 MiB.
- A 4 MiB file of 3-byte running-status note events (the worst case for the pure-Python
  parser): 0.5 s, stopped at 100 000 notes; under tracemalloc 6 s and 40 MiB. Test bounds:
  60 s and 256 MiB.
- 3000 random corruptions of two real files (bytes, cuts, 0xFFFFFFFF lengths): every one a
  `Song`, about 1 s in all.
- One `Notes` at the note bound (1563 script notes x `repeat=64`, cut at 98 469 notes, 78 885
  dropped past the voices) in a 60 s scene: placing them the first time 4.5 s; a whole lint
  2 s (it was 68 s before the summary (span, ranges, programs, voices, the roll's table) was
  made once per placement instead of at every lint sample); its `audio_track` 17 s (64
  voices for 60 s); 60 MiB peak. Eight such objects: 39 s to place, lint 15 s, the mix 122 s,
  313 MiB peak. The synthesizers' work is bounded: `MAX_SYNTH_WORK` = 2^31 samples, per
  synthesizer and for the scene's notes together (about 11 million samples a second on one
  core here, so at most a few minutes), and placed notes are cached while they hold at most
  `CACHED_NOTES` = 400 000 notes in all (each ~0.4 KiB with its voice).

## Verification

Before the audit fixes, each suite on its own (`-m "not slow"`): test_midi 90 (+1 `gl`+`slow`
golden, passed against the stored picture without `PHASOR_UPDATE_GOLDEN`), test_docs 5,
test_lint + test_sound + test_mcp 111, test_engine + test_proc + test_windows_robust 38
(2 skipped: Windows only), test_narration + test_narration_snap + test_narration_audit +
test_audio_inputs + test_style_freedom + test_pycode + test_kit + test_layers +
test_contracts 188 (2 skipped: espeak-ng), test_final `-k "notes and not slow"` 7. Before the
last change: test_final `-k "not slow"` 475 passed, 8 skipped, and the 2 failures it found
(`notes` drew nothing in the type sweep: it now gets `show=True` there) fixed. ruff check,
ruff format --check and lint-imports (2 contracts kept) clean. The full suite was not run
(CI).

Checked against something other than phasor's own writer: the SMF 1.0 specification's
example file reads as the specification describes it (programs, notes, velocities, the
quarter-note timing at 120 bpm). After the audit, `mido` (in a throwaway venv outside the
repository; not a dependency) read 1500 random notes on channels 0, 1, 10 and 16 (same-key
overlaps included) and an exported scene of four objects (piano, violin, bells, drums)
exactly as phasor's reader does: 0 mismatches in pitch, start, length, velocity, channel and
program; track names read as written.

## Deviations

1. `time.narration.Clip` gained `marker: str | None = None` (the mix never reads it; the
   engine resolves it). Nothing else in the mixer changed; the bit-identity test pins it.
2. `Notes.start` is `float | str` (a marker's name), not a `MarkerRef`: a dict-shaped
   parameter would be read by the parameter compiler. There is no offset after the marker:
   place another marker.
3. The piano roll's playhead and lit notes need `start` in seconds: an object's `draw`
   sees only the state at `t`, which holds no markers (contract 3). With a marker `start`
   the roll is drawn still; the example reveals it at its marker.
4. (Audit S2.) The synthesizer is made at the track's rate: the mixer asks a source with an
   `at_rate` method for its rate (a small addition to the mixer; WAV sources have no such
   method, so their path and bytes are unchanged). Aliased energy of a lead A4: -112.7 dB at
   8 kHz, -109.9 dB at 16 kHz (was -14 and -25 dB through the linear resampler).
   `MAX_SYNTH_WORK` is counted at 48 kHz whatever the rate.
5. `export_midi` writes every note, also those past the scene's end (the file is the
   score); lint `notes_limit` says which are not heard. Notes are written at 120 bpm, not
   the source file's tempo map (times in seconds are kept; bars are not).
6. Notes count towards `MAX_SOUNDS` (256 clips in one track) like `Sound`.
7. `inspect/lint.py` `_sound_rules`: the PyAV check skips non-`sound` objects (a `.mid` had
   no business there; it would have warned `missing_extra` where PyAV is absent).
8. No theme `[lint]` threshold: every `notes_limit` case is a hard limit of the mix.
9. Two bounds the plan did not name: `MAX_SYNTH_WORK` (samples the synthesizers make; past
   it notes, then whole objects, are left out and lint says so) and `CACHED_NOTES` (the
   placed-notes cache is bounded by notes held, not by entries). A scene of many objects
   each at `MAX_NOTES` still costs about 4.5 s and 40 MiB per object when first placed (and
   again when evicted); a scene-wide note bound is deferred.
10. (Audit S1.) A `Notes` clip's `gain` is applied inside the synthesizer, before its
   limiter; the mix gets the clip at 0 dB. A `Sound`'s `gain` is the mix's, as before.
11. (Audit.) General MIDI family 9 (pipes: flute, recorder, ocarina) now plays the `sine`
   voice (was `lead`), so `instrument="sine"` can be written as a program (73, flute) that
   reads back as itself. The `pad` voice's second oscillator 7 cents up makes its pitch read
   about 3.5 cents sharp: a chorus, by design.
12. (Audit S8.) Running status is kept across meta and system exclusive events (the
   standard says they cancel it; files in the wild rely on it, and mido accepts it).

## Deferred

1. Pitch bend, controllers (sustain pedal, volume, pan), aftertouch, key signature, lyrics
   and markers in a `.mid` are read past and not played.
2. Stereo: the track is mono, as for every sound in phasor.
3. Format 2 files (refused, with the reason).
4. The app's MIDI panel (Milestone 25), a timeline lane for notes.
5. A check of written files in a DAW on the owner's PC (`mido` agreed, see Verification).
6. A scene-wide bound on notes placed (many objects each at `MAX_NOTES` are slow to lint).

## Untested

- **Hearing**: no one has listened to the synthesizer, the drum kit, or a mixed render with
  notes. Pitch (440 Hz within 2.5 Hz), velocity (level ratio), sample-exact start, the
  release reaching 0 and determinism are checked numerically.
- Windows and macOS (no POSIX-only calls in the tests; `os.utime(ns=...)` only; the
  subprocess determinism test uses `sys.executable`), CUDA (nothing here touches kernels).
- (Checked, not untested:) `s.render(..., preset="draft")` of the example: 132 frames "with
  sound"; the mp4's AAC track decoded with PyAV lines up with `audio_track` at lag 0 (8 kHz),
  and the notes alone (decoded minus the narration-only mix) are silent 0.15 s before
  `glide` (0.0005) and sound from it (0.08). Heard by no one.

## Audit (independent, 2026-11-24): findings and fixes

"Before" is the audited head `6b5a4b8`; every row has a test in `tests/test_midi_audit.py`.

| Finding | Before | Fix | After |
| --- | --- | --- | --- |
| B1 export overwrote the `.mid` a `Notes` reads, not atomically | `export_midi("src.mid")` replaced the scene's own file; re-read, its notes were shifted twice (1.5 s became 2.5 s); the sound changed without `Engine.apply` | `Engine._files_read` (every `Files` value, resolved): such a path is refused always; any other existing file only with `overwrite`; written to a temp file beside it, then `os.replace` | refused (`ValueError` / `ToolError`); `overwrite=True` replaces other files, no temp file left |
| S1 hard clip before `gain` | 8 organ notes at v127: 948 samples at full scale, the same at `gain=-12` | gain inside the synthesizer, `LEVEL` 0.18, bells normalized, a soft limiter (`KNEE` 0.5) | sine, pluck, organ, lead 8-note chords, 4 bells, 5 drums at once: peaks 0.55-0.81 of full scale, no plateau; -12 vs -24 dB exactly 12 dB apart |
| S2 aliasing at 8 and 16 kHz | -14 dB (8 kHz), -25 dB (16 kHz) | `Synth.at_rate`, asked for by the mixer | -112.7 dB, -109.9 dB (test: < -40 dB at both) |
| S3 programs crossed in export | two objects on channel 0, programs 0 and 40: the piano's second note read as 40; programs tracked in input order | a channel per (object, program); program changes in time order; `instrument` written as `PROGRAM_OF` | programs and voices read back as written; mido agrees |
| S4 file cache thrashed | `lru_cache(4)`: six files, 0 hits, lint 1.85 s to 16.3 s | `_songs`: kept while at most `CACHED_NOTES` notes and 256 files | six files read once each |
| S5 zero-length notes dropped silently | drum on/off at one tick: 0 notes, no `cut` | drums keep `DRUM_HIT` (0.01 s); melodic ones left out and said (`truncated`) | 2 drum hits; "1 notes with no length" in `cut` |
| S6 drum tails ended on a step | toms 0.031, kick 0.018 at the last sample | a linear fade over each hit's last quarter | last 50 samples < 1e-3, last < 1e-4 (10 drum keys) |
| S7 writer wrote 5-byte VLQs | a note at 200 000 s: the reader then refused the track (1 note read) | `write_smf` raises for a note after 3 h or a delta past 2^28 ticks; `export_midi` leaves such notes out (`left_out`) | never written; the export reads back whole |
| S8 running status after a meta event | the rest of the track dropped | the last channel status kept | read |
| Nits | a key struck again while held did not round-trip; non-Latin-1 names became `?`; `read(-5, 10)` returned samples 0-9; the cache key hashed up to 20 000 rows at every call | the outer note ends at the new strike; Latin-1 or UTF-8 names; negative skips are silence; script notes digested once per tuple (`_listed`); pad's +3.5 cents documented | tested |
