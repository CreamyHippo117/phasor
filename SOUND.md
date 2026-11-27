# Sound: a bed under the narration

Working doc, like `APP_PLAN.md`. Asked for by the owner on 2026-11-26: "a sound bed: sound
effects merged nicely with background music, most likely ambient, under the narration, and
all of it editable in the desktop app". Status: **approved by the owner 2026-11-26; started on the
owner's word the same day. All five steps are built (2026-11-26): steps 1-3 and
`sound_report` first; then step 4 in today's editor window on the owner's ask, with
mastering, the owner's own voice and editable copies of script scenes (see "Step 4, as
built").** The owner's answers are at the end of this doc; what was built differently is under
"Deviations".

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

## Step 4, as built (2026-11-26)

The owner asked for it in today's window: "I have preview and all the related stuff and then
I can add the soundtrack I want and maybe the AI masters it for me. I add the narration
where I want, the extra sound effects or prompt AI to do so. Same MCP integration." Then:
"I want the narration to be my voice. I want to add in my sound tracks and my voice tracks",
and for scenes built by scripts: "make that allowed but as a new copy".

- **The Sound tab** (`gui/window.py` `SoundPanel`, logic in `gui/controller.py`): each lane's
  loudness and its level relative to the voice (typed in LU), mute and solo (listening
  only), Add voice, Marker at playhead, Add music (any file, copied into the scene's
  `audio/` folder), Add effect at playhead (tied to the nearest marker), Narrate
  (synthetic), Master, Show levels (the storyboard's loudness lanes), Play with sound.
- **The timeline** has a row per bed and one for the cues. Beds, cues and spoken sentences
  drag (`move_bed`, `move_cue` snapping to the nearest marker, `move_sentence` kept between
  its neighbours: `time/narration.move_block`).
- **Your own voice**: `Workspace.voice_add` (MCP `voice_add`) puts a recording into the
  narration at a time or marker (`time/narration.add_block`; overlapping another sentence is
  refused). A marker added inside a recording belongs to it (`mark_block`) and moves with
  it; recordings drag and can be taken out (`remove_block`). Word timings are not found
  automatically: place markers by ear with the playhead.
- **Mastering**: `Engine.master_sound()` (MCP `sound_master`), rounds of certain changes:
  the `bed_masks_voice` fix, every bed and cue turned down evenly when the mix would clip
  (`audio_clip`), a cue that peaks on a stressed word moved just before it (`cue_on_word`'s
  own hint). Each round is one undo step.
- **Script scenes**: the first sound, voice or narration edit makes an editable copy,
  `<name>_edit.phasor.json` beside the script (`Engine.as_document`, `Workspace.scene_copy`,
  MCP `scene_copy`); the script and its overrides file are never touched, so contract 2
  stands as written.
- **Playback** (`gui/audio.py`) needs no audio library: `winsound` on Windows, `afplay` on
  macOS, `paplay`/`aplay`/`ffplay` on Linux play a WAV of the mix from the playhead
  (`Editor.listen_file`: the encoder's own track; with lanes muted or soloed, the mix made
  again without them). The playhead follows the clock while it plays.
- **Also**: opening an example copies the examples' `assets/` folder; `core/pycode.py` writes
  the narration back as its own `set_narration` op, so narrated scenes survive the round
  trip (`sound_bed` is back in that test).
- **Tests**: `tests/test_gui_sound.py`, the sound tests in `tests/test_gui_window.py`.
  Untested: hearing it (no speakers in the build environment), Windows' `winsound`, macOS.
- **Built twice, kept once**: a second session built steps 1-3 and 5 in parallel
  (PR CreamyHippo117/phasor#27); the design merged first (this document's) was kept, and
  that PR's editor, mastering, voice and copy work was moved onto it. Its per-take voice
  levelling was not carried over: a narration is levelled as one track, as here.

## Deviations (as built, 2026-11-26)

- **One design.** The interim `s.note` / `s.sound` (the `sounds` list of notes and WAV files,
  built the day before and never on the default branch) is gone: a note is a cue of the
  built-in `tone` (`s.cue(marker, "tone", midi=69, voice="pluck", dur=0.5)`), a file is a cue
  naming the file. Its patch ops (`add_sound`, `set_sound`) and MCP tool (`sound`) became
  `add_bed`/`set_bed`/`add_cue`/`set_cue`/`remove_sound` and the tools `cue`, `bed`,
  `sound_report`. Schema 9 is this design's `sound` section; there is no schema 10.
- **Tools.** Adding beds and cues was to be script editing only (`script_edit`, APP_PLAN phase
  3), which does not exist yet, so the MCP server has two small tools, `cue` and `bed`, for
  scenes without a script. They can go when `script_edit` lands.
- **A narration-only scene is untouched** (step 1's exit): the bus, the trim and the limiter
  run only when there is a bed or a cue, and the narration then keeps its own sample rate.
  With beds or cues everything is at 48 kHz.
- **The trim** runs before the limiter (gain, then the ceiling), not after it, so the trim
  can never push a peak past -1 dBTP. While the narration has no audio yet (estimated timing)
  there is no trim: the lanes stay where they will sit under the voice, rather than the cues
  being raised to the voice's level.
- **A cue's level** is its loudest 400 ms (BS.1770 momentary loudness, sampled every 10 ms);
  a bed's is its integrated loudness. Short sounds are measured as themselves followed by
  silence.
- **Ducking**: spans closer than `attack + release` duck as one (no pumping between
  sentences); the ramps are smoothsteps in dB.
- **`cue_on_word`**: the narration has no stress marks, so a "stressed word" is a word of
  four letters or more, and the rule compares the cue's peak with that word's onset.
- **Lint thresholds** live in `[sound]` (`masking`, `word_gap`, `clip`), as this plan says,
  not in `[lint]` as CLAUDE.md's general rule says: they sit beside the levels they judge.
- **`bed_masks_voice`'s fix** (lower the bed's gain) is offered on scenes without a script; on
  a script's scene the hint names the gain to write in the script (a bed is not a tweak).
- **Built-ins are cached in memory** (per cue id, seed and knobs), not as WAV files: a pure
  function needs no files written into the scene's folder. `chime` takes `midi` instead of
  following a bed's key (beds carry no key).
- **The loudness lanes** are drawn under every storyboard of a scene with sound (step 5 asked
  for them; step 2's exit wanted a snapshot with them).
- **`soundfile`** is now a core dependency (it was in SPEC's stack but not in
  `pyproject.toml`); it decodes every bed and cue file. The narration's clips are still read
  by the 16-bit WAV reader, which keeps step 1's bit-identity.
- **Untested**: Windows and macOS (the `soundfile` wheel bundles libsndfile there; the
  desktop build lists it), and listening: every check is a number (loudness, true peak,
  hashes), none is an ear. The owner should listen to `examples/sound_bed.py` rendered with
  a real voice.
