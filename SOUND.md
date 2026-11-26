# Sound: a bed under the narration

Working doc, like `APP_PLAN.md`. Asked for by the owner on 2026-11-26: "a sound bed: sound
effects merged nicely with background music, most likely ambient, under the narration, and
all of it editable in the desktop app". Status: **approved by the owner 2026-11-26; step 1 built 2026-11-26** (see "Progress"). It comes after `BEAUTY.md`; the
owner's answers are at the end of this doc.

## What exists to build on

| Piece | Where | State |
| --- | --- | --- |
| Narration blocks with WAV clips, markers on words | `core/schema.py` `Narration`, `voice.py` | works |
| Narration mixed to one track | `time/narration.py` `mix_track`, `Engine.audio_track()` | works; **mono int16**, clips summed and clipped |
| Audio muxed into the video | `render/encode.py` `mux_audio`, `render/job.py` | works; mono only |
| Files an object reads, kept inside the scene's folder | `core/files.py`, lint `missing_file` | works |
| Verbs anchored to markers, timing from theme `[motion]` | `time/verbs.py` | works |
| Timeline panel | `gui/window.py`, `gui/controller.py` | markers and parameter lanes; **no audio lane, no playback sound** |
| Manim `add_sound` | `MANIM_PARITY.md` Milestone 20 | planned as "a `Sound` clip mixed in at encode" |

This plan is that `Sound` clip, grown into a small mixer. Milestone 20's `add_sound` then
maps onto it instead of building its own.

## The design

### Three kinds of sound, one place in the document

A new top-level `sound` section of the document (schema 8 to 9, migrated as "no sound"),
beside `narration`. Not objects: sound draws nothing, has no bounds and no semantics, so it
does not belong in the object contract.

- **Bed**: music or ambience under a stretch of the scene. A file, a start and an end (each
  a marker or the scene's start/end), loop on or off, gain, fade in/out. Several beds may
  overlap; a crossfade is two beds with overlapping fades.
- **Cue**: a one-shot effect. A file (or a built-in sound, below), a marker plus an offset,
  gain, pan. Like motions in narrated scenes, every cue is anchored to a marker, and
  `lint(strict=True)` checks it.
- **Voice**: the narration, as today. It is the lane everything else makes room for.

Script side (verbs, so the person, the AI and the app write the same thing):

```python
s.bed("audio/drift.ogg", start="intro", end="outro", gain=-18, fade=2.0)
s.cue("pole_appears", "chime", gain=-12)  # built-in sound
s.cue("zoom", "audio/whoosh.wav", offset=-0.15)  # lands just before the word
```

### Mixing nicely: what "merge" means in numbers

The mix is a pure function of (document, the files it names), like `state_at`: same
document, same samples, bit for bit. It runs in NumPy at encode and for the app's preview.

1. **Stereo float32 at 48 kHz** throughout; int16 only when written out. Mono clips are
   centred; other sample rates are resampled once (polyphase, cached).
2. **Loudness, not peaks.** Each source is measured in LUFS (ITU BS.1770 K-weighting, a few
   lines of NumPy) and set relative to the voice: by default voice at -16 LUFS, bed 18 dB
   under, cues 10 dB under (owner: effects a bit quieter). These live in a new theme table `[sound]`, so a look can be
   quieter or louder (taste knobs stay in themes).
3. **Ducking.** While a narration block speaks, beds dip by `[sound] duck` dB (default 8)
   with `attack`/`release` ramps. The envelope comes from the narration's word times, not
   from listening to the signal, so it is exact and deterministic and needs no sidechain.
   Cues are not ducked (they belong to the picture) but are kept off words where possible:
   a cue whose peak falls on a stressed word is reported (lint below).
4. **A gentle bus.** A soft-knee limiter at -1 dBTP on the sum, then a final loudness trim
   to the theme's target. Nothing else: no reverb on the bed by default (the acoustics
   domain can convolve a cue with a room's impulse response later, see "Later").

### Built-in effects, so a scene needs no downloads

Short sounds synthesized from (id, seed) through `rng(object_id, seed)`, cached as WAV:
`tick`, `chime` (tuned to the bed's key if it has one), `whoosh` (filtered noise sweep),
`pop`, `swell`. Licence-free by construction. No built-in beds: the owner imports their own
ambient sounds and handles their licences (owner, 2026-11-26). Imported files, beds and
cues alike, stay inside the scene's folder (`core/files.py`).

### Lint rules (`inspect/lint.py`, thresholds from `[sound]`)

| Rule | Fires when | Fix |
| --- | --- | --- |
| `missing_file` | a bed or cue names a file that is not there | (existing rule) |
| `bed_masks_voice` | a bed is within `[sound] masking` dB of the voice during speech | lower the bed (certain) |
| `cue_on_word` | a cue's peak lands within 80 ms of a stressed word | none (judgement) |
| `audio_clip` | the mix before the limiter exceeds 0 dBFS by more than 3 dB | none |
| `cue_unanchored` | a cue has no marker (strict mode) | none |

Each reports `times`, and is tested with `passes=None` (sound needs no GL at all).

### The desktop app

Built on the code-first window (`APP_PLAN.md` phase 2), so edits are tweaks in the script
(ADR 0004) and go through the `Workspace` under the shared lock.

- **Audio lanes in the timeline** under the parameter lanes: the voice waveform with its
  words (already wanted by SPEC's timeline panel), one lane per bed showing its gain with
  the ducking dips drawn in, and cues as diamonds on their markers.
- **Editing:** drag a cue along the timeline and it snaps to the nearest marker and stores
  an offset; drag a bed's ends to markers; drag its fade handles; a gain field per lane.
  Mute and solo per lane, for listening only (not saved).
- **Listening:** playback with sound, from the playhead, through Qt Multimedia (in
  PySide6-Essentials). The app plays the same mix the encoder writes, cached per document
  version so scrubbing does not re-mix.
- **A Sound tab** beside Problems / Inspector / Looks: the lanes' loudness (LUFS), the
  ducking depth, the sound lint issues (click to jump), and a sound library: built-ins plus
  the files in the scene's `audio/` folder, drag onto the timeline to add a cue.

### For the AI

The AI cannot hear, so it gets numbers and a picture: a `sound_report` tool on `Workspace`
(loudness per lane per narration block, ducking, lint) and the loudness lanes drawn under the
storyboard. Adding beds and cues needs no new tool: it is script editing (`script_edit`,
`APP_PLAN.md` phase 3). `TOOL_NAMES`, `docs/ai/MCP.md`, `docs/ai/API.md` and `llms.txt`
updated in the same commit, as CLAUDE.md requires.

## Contracts touched

- **2 (document is truth):** a new `sound` section, schema 9 with a migration. The contract
  itself does not change; flagged because it is the schema (strongest model, per CLAUDE.md).
- **3 (time is pure):** the picture does not change. The mix is pure in the same sense and
  reads only files named in the document, inside the scene's folder, re-read when they
  change, as `Files` already do. No ADR needed unless the owner wants one.
- **6 (layers):** mixing and loudness in `time/sound.py` (NumPy only, beside
  `time/narration.py`); decoding OGG/FLAC/MP3 needs `soundfile` (already in SPEC's stack,
  wheels for Windows), read where `read_wav` is read today; playback only in `gui/`.

## Steps, each ending with its test

1. **Stereo mix and loudness.** `time/sound.py`: stereo float bus, resampling, BS.1770
   meter, limiter; `mix_track` becomes its voice lane; `mux_audio` stereo. Exit: meter
   against the EBU test signals, a narration-only scene renders bit-identical audio to
   today's (apart from mono to stereo), determinism test.
2. **Beds, cues, ducking in the document.** Schema 9, migration, verbs `bed`/`cue`, theme
   `[sound]`, lint rules. Exit: tests first for ducking envelopes and cue anchoring (they
   follow markers when narration is re-recorded), lint in `MODES`, an `examples/` scene and
   a storyboard snapshot with loudness lanes.
3. **Built-in effects.** Synths and their cache. Exit: each is a pure function
   of (id, seed) (hash test), loudness within 0.5 LU of its nominal level.
4. **The app.** Audio lanes, drag to retime, playback, the Sound tab. Exit: controller tests
   without Qt (drag a cue, it snaps; undo restores), window tests on `offscreen`, and the
   owner listening on the PC (playback is untested anywhere without speakers).
5. **AI.** `sound_report`, docs, `llms.txt`. Exit: tool test without the protocol, docs
   blocks run in CI.

Steps 1-3 need nothing from the app; step 4 waits for `APP_PLAN.md` phase 2. None starts
before `BEAUTY.md` is done, and not without the owner's go-ahead.

## Progress

- **Step 1, done 2026-11-26.** `time/sound.py`: stereo float32 bus (`Placed`, `mix`), polyphase
  resampling, BS.1770-4 integrated loudness with R128 gating, 4x-oversampled true peak, the
  master limiter (`master`), `mixdown` for the engine. `Engine.audio_track()` returns stereo
  int16 and `render/encode.py` `mux_audio` writes stereo. `time/narration.py` `mix_track` is
  gone: the narration is now the voice lane. Reading: WAV of any width through SciPy;
  OGG/FLAC/MP3 through `soundfile`, a new `sound` extra (in CI, and in `all`).
  Tests: `tests/test_sound.py` (EBU Tech 3341 cases 1-4 at 48 and 44.1 kHz, the standard's
  K-weighting coefficients, true peak between samples, the limiter, placement, rates,
  determinism, reading, a stereo mux through PyAV).

### Deviations

- **Bus rate.** Planned as "48 kHz throughout". As built, the bus runs at the sources' common
  rate and falls back to 48 kHz only when they differ. Why: a narration-only scene then keeps
  its exact samples (espeak's 22.05 kHz is not resampled), which is the step's exit test.
- **The limiter.** Planned as "soft-knee at -1 dBTP". As built, a look-ahead limiter on
  sample peaks at -1 dBFS: the gain needed is held 50 ms either side of each sample, then
  averaged over the same span. It ramps smoothly and never lets a sample through. True peak
  is measured (`true_peak`) but not limited, so it can sit a fraction of a dB above -1. A mix
  under the ceiling passes through untouched.
- **Exit test.** A narration-only scene keeps its samples bit for bit in both channels
  *when the voice peaks under -1 dBFS*; a louder voice is now limited where before it was
  clipped at full scale.

## Later (not in this plan)

- Cues convolved with a `Room3D` impulse response (the "convolution-reverb audio export" the
  acoustics milestone left out), so a whoosh sounds like it is in the room on screen.
- Sonification: a cue or bed driven by a scene value (a Fourier partial you can hear).
- Music that follows the scene's beats (a bed cut to bar lines at markers).

## Owner's answers (2026-11-26)

1. **Scope:** all five steps.
2. **Order:** after `BEAUTY.md`. Do not start before the owner says so.
3. **Levels:** voice -16 LUFS, bed 18 dB under and ducking 8 dB are fine; effects quieter,
   so cues default to 10 dB under the voice (was 6).
4. **Sounds:** beds are the owner's own imported ambient files (licensing is theirs); no
   built-in beds. Effects are built in (synthesized).
5. **Stereo:** yes.
