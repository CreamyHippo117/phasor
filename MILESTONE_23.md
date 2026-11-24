# Milestone 23: MIDI

Plan: `STUDIO.md`, row 23. Owner (2026-11-24): "midi output ... a midi input option for you to
overlay with the audio at certain points", read there as **one type with an import end, a
play end and an export end**. Built 2026-11-24 in a worktree branched from
`claude/eloquent-mayer-jiqxs9` (merged at `9fa24f7`: Milestones 21, 22, 24, 26, 27).
Exit demo: **a narrated scene (a recording, M22) with notes a script makes and a `.mid` file
played over it from markers, heard in the mixed track where the markers are, and the notes
written back to a `.mid` that reads back the same** (`examples/notes_overlay.py`,
`tests/test_midi.py`).

Status: **built; focused gates green on Linux; not independently audited.** Windows, macOS and
CUDA untested; **nobody has listened to the synthesizer** (pitch, timing, levels and
determinism are checked numerically).

## Built

| Area | Where | What |
| --- | --- | --- |
| SMF reader | `domains/midi_math.py` `read_smf` | format 0 and 1; PPQ and SMPTE (24, 25, 29.97, 30 fps) time; running status; note on/off (velocity-0 note-on is an off; one key of one channel pairs first in, first out; a note still held at the end of its track ends there); program change (each note gets its channel's program at its tick, across tracks); meta tempo (a tempo map over all tracks), time signature, track name; sysex and other metas skipped; unknown chunks skipped; a header longer than 6 bytes skipped past. Returns a `Song`: `notes` (`Note(pitch, start s, length s, velocity, channel, program)`, sorted by start, channel, pitch), `format`, `tracks`, `names`, `division`, `bpm`, `tempo_changes`, `time_signature`, `error` (nothing read) or `cut` (read up to a fault or a bound) |
| Bounds | `midi_math` | file 4 MiB, 256 tracks, 400 000 events, 100 000 notes (checked before any array is built), 10 000 tempo changes, 1024 chunks, 2^40 ticks a track, notes after 3 h left out, notes longer than 60 s cut to it, tempos outside 1-1000 bpm ignored (said), a VLQ of more than 4 bytes, a data byte past 127, running status with no status, a real-time status in a file, a meta/sysex/track length past what is there: that track stops there (said). Format 2 is refused with a reason ("save it as format 0 or 1"); format 7, 0 tracks, 0 division, SMPTE with another frame rate or 0 ticks a frame: refused. Never raises (any other exception is said as `error`) |
| SMF writer | `midi_math.write_smf` | format 1: a conductor track (title, 120 bpm, 4/4), then one named track per part; 960 ticks a quarter (1/1920 s); running status; note-offs before program changes before note-ons at one tick; a channel's program written before its first note and where it changes (never on channel 10); pitches rounded and clamped to 0-127, velocities to 1-127, a note at least one tick |
| Synthesizer | `midi_math.Synth` (a `time.narration.Source`) | 48 kHz; voices `sine`, `pluck` (24 harmonics, decaying, low notes ring longer), `organ` (drawbars), `bell` (inharmonic partials 1, 2.76, 5.4, 8.93, each dying faster), `pad` (a detuned second oscillator, slow attack), `lead` (saw-like); General MIDI program families map onto them (`timbre_of`), channel 10 is a drum kit (kick: a falling sine; snare: noise and a tone; hats and cymbals: differenced noise; toms: tuned sines; the rest a noise tick) whose noise is `rng(object_id, seed)`; ADSR (attack up from 0, exponential decay to sustain, a release that ends at 0); velocity as (v/127)^2 x 0.25; at most `MAX_VOICES` = 64 notes sounding at once (with their release; a note starting when 64 sound is dropped, counted, its time kept); harmonics above 0.45 x 48 kHz are left out (band-limited tables) |
| Determinism | `Synth.read` | every sample is a pure function of the notes, the seed and its index: integer phases (32-bit), fixed 4096-entry waveform and exp tables with linear interpolation (only +, -, x, /, floor and indexing on varying arrays), voices summed in one order in float64, cast to float32 once, made 2^16 samples at a time inside a read. Tested: any window equals that slice of the whole (every instrument, odd offsets), the same sha256 in another process, the mixer's window equals the slice of the whole mix on the resampled path |
| Mixing | `time/narration.py`, `engine.py` | the synthesizer is a `Clip.source` (M22's protocol, unchanged): the mixer reads, resamples, budgets (`MAX_SOUNDS`, `MAX_MIX_SAMPLES`) and cuts it exactly as a WAV clip. `Clip.marker` (new, default None): a clip that starts at a marker; `Engine._at_markers` resolves it before mixing (an unknown marker: left out and said in `problems`). TTS, Sound and M22 recording mixes are **bit-identical** to before (`tests/mix_baseline.py`: five sha256 values taken on the merged commit before any M23 change, integer-only test signals so they hold on any system) |
| `Notes` | `domains/midi.py` | `path` (a `.mid`/`.midi`/`.smf` in the scene's folder, through `core/files.py` (`stamp`, `read_bytes`: regular files only, never blocking on a FIFO), cached by stamp so it is read again when it changes) or `notes` (the script's `(pitch, start, length, velocity[, channel])`, at most 20 000, each checked; a fractional pitch plays its exact frequency), never both; `start` (seconds or a marker's name), `gain`, `transpose` (never drums), `tempo_scale`, `channels`, `repeat` (1-64, bounded by `MAX_NOTES`), `instrument`, `program`, `seed`; `show` with `pos`, `width`, `height`. The audio parameters are `Fixed` (a track on one is `bad_param`); `Life` none. Semantics: `kind`, `source`, `start`, `gain`, `count`, `instrument`, `drawn`, `span`, `pitch_range`, `channels`, `programs` (GM names, "Drum kit"), `voices`, `voices_dropped`, `out_of_range`, and for a file `file`, `missing`, `format`, `tracks`, `bpm`, `names`, `time_signature`, `unreadable`, `truncated`. Bounds: a point, or the roll's box |
| Piano roll | `Notes.draw` | with `show`: a frame and one bar per note (time across, pitch up) with `segment_batch`; with `start` in seconds, notes sounding at `t` take the `played` role and a playhead runs. Golden `tests/golden/software/notes_overlay.png` (looked at) |
| Export | `Engine.export_midi(path)`, `Scene.export_midi`, `Workspace.export_midi(out)` (MCP tool `export_midi`) | every `Notes` (the notes as the synthesizer plays them) at its times in the scene, one track per object named by its id, the scene's title on the conductor track; an object at a marker the scene lacks is left out. Returns `path`, `tracks`, `notes`, `seconds`. The workspace tool writes inside the workspace only, `.mid`/`.midi` only, default `<scene>.mid` in `out/` |
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

## Verification

Each suite on its own (`-m "not slow"`): test_midi 87 (+1 `gl`+`slow` golden, passed),
test_docs 5, test_kit + test_layers + test_contracts 28, test_lint + test_sound 72,
test_narration + test_narration_snap + test_narration_audit + test_audio_inputs +
test_style_freedom + test_pycode 160 (2 skipped: espeak-ng), test_mcp + test_pycode 44,
test_final `-k "not slow"` 475 passed, 8 skipped + the 2 it found (`notes` drew nothing in the type sweep: it now gets `show=True` there) fixed and rerun (7 passed). ruff check, ruff format --check and
lint-imports (2 contracts kept) clean. The full suite was not run (CI).

Checked against something other than phasor's own writer: the SMF 1.0 specification's
example file reads as the specification describes it (programs, notes, velocities, the
quarter-note timing at 120 bpm). No `mido` or `pretty_midi` is in the environment and none
was added, so no third-party reader has opened phasor's files; their structure is checked
chunk by chunk in the tests.

## Deviations

1. `time.narration.Clip` gained `marker: str | None = None` (the mix never reads it; the
   engine resolves it). Nothing else in the mixer changed; the bit-identity test pins it.
2. `Notes.start` is `float | str` (a marker's name), not a `MarkerRef`: a dict-shaped
   parameter would be read by the parameter compiler. There is no offset after the marker:
   place another marker.
3. The piano roll's playhead and lit notes need `start` in seconds: an object's `draw`
   sees only the state at `t`, which holds no markers (contract 3). With a marker `start`
   the roll is drawn still; the example reveals it at its marker.
4. The synthesizer runs at a fixed 48 kHz and the mixer resamples it like a WAV clip
   (linear interpolation, no low-pass). Under a narration at a low rate (the example's
   8 kHz recording) the upper harmonics alias; nobody has listened to how much.
5. `export_midi` writes every note, also those past the scene's end (the file is the
   score); lint `notes_limit` says which are not heard. Notes are written at 120 bpm, not
   the source file's tempo map (times in seconds are kept; bars are not).
6. Notes count towards `MAX_SOUNDS` (256 clips in one track) like `Sound`.
7. `inspect/lint.py` `_sound_rules`: the PyAV check skips non-`sound` objects (a `.mid` had
   no business there; it would have warned `missing_extra` where PyAV is absent).
8. No theme `[lint]` threshold: every `notes_limit` case is a hard limit of the mix.

## Deferred

1. Pitch bend, controllers (sustain pedal, volume, pan), aftertouch, key signature, lyrics
   and markers in a `.mid` are read past and not played.
2. Stereo: the track is mono, as for every sound in phasor.
3. Format 2 files (refused, with the reason).
4. The app's MIDI panel (Milestone 25), a timeline lane for notes.
5. A third-party check of written files (`mido` or a DAW) on the owner's PC.

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
- An independent audit and contract review by a model that did not write the code.
