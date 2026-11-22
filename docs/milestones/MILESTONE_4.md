# Milestone 4: Narration sync

Roadmap row: script markers, recording/TTS, WhisperX alignment, `TimeMap`, choreography
verbs. Exit demo: **a residue theorem short driven entirely by narration.**

Status: **built; all gates green on Linux.** The exit demo renders with a synthesized
voice (espeak-ng). WhisperX and Piper are written but **untested**: their models could not
be downloaded here (Hugging Face is blocked from this container).

## Built

| Area | Where | What |
| --- | --- | --- |
| Script format | `time/narration.py` | sentences are blocks; `[[marker]]` = when the next word starts; `[[pause s]]` |
| Timing | `time/narration.py` | blocks laid end to end (`lead`, `gap`); estimated words; `pace` re-spacing |
| Voices | `voice.py` | `estimate`, `espeak[:lang[:wpm]]` (tested), `piper:<model>`, recordings + `WhisperX` or `TimingsFile` |
| Exact markers | `voice.py` | synthesized piece by piece between markers; clips trimmed and cached by text |
| Schema v4 | `core/schema.py` | `narration` (blocks, words, markers), `Segment.verb`; `set_narration` op owns its markers |
| Verbs | `time/verbs.py`, `scene.At` | reveal, exit, trace, bloom_in, dim_others, undim, focus_pull, morph, settle; `then()` |
| Theme | `[narration]`, `[motion]` | speaking rate, gaps, pacing; per-verb duration and easing, stagger, dim level |
| Easings | `time/easing.py` | `back`, `spring` (overshoot, land exactly) |
| Objects | `ParamsBase.dim`, `Line.progress` | dimming separate from opacity; partial curves |
| Pacing | `Scene.pace()` | extra pause after a sentence = changed objects x `pause_per_change` (capped) |
| Audio | `Engine.audio_track`, `render/encode.mux_audio` | narration mixed and muxed into MP4/MOV |
| Lint | `inspect/lint.py` | `raw_motion` (strict); contrast judged at the intended opacity |

## Verification

- `uv run pytest`: **386 passed, 16 skipped** (all CUDA). Without OpenGL: every non-GL test
  passes. `ruff`, `lint-imports`: clean.
- **Exit demo** (`examples/residue_theorem.py`): 6 blocks, ~20 s, voice espeak-ng. Every
  segment is anchored to a narration marker and made by a verb; `lint(strict=True)` is clean;
  the story order is asserted marker by marker (nothing before the first word, loop drawn
  before it is discussed, the other pole dimmed, the result revealed at "pi").
- Rendered to `examples/out/residue_theorem.mp4` at 960x540 with the voice muxed in, and
  watched frame by frame through its storyboard.
- Exact markers: the marker inside a sentence equals the duration of its first piece spoken
  alone (within 50 ms).
- Re-narrating with a longer sentence moves the marker and everything anchored to it.

## Bugs caught while building

1. **dim_others and reveal fought over opacity**: an object dimmed while still unrevealed was
   visible from the start. Dimming now has its own `dim` factor, and a reveal drops dimming
   from before it.
2. **bloom_in faded opacity with an overshooting ease** (opacity above 1).
3. **Contrast flagged deliberately dimmed text**; it is now judged at the intended opacity.
4. Light labels on the domain plane failed contrast for real: the theme's phase wheel is darker.
5. An untraced curve still got a box in storyboard marks.
6. **Text lost its last glyph** at some sizes ("= pi" rendered "= p"; a Milestone 1 bug,
   found by watching the rendered video). Text is now drawn, then resampled to its exact
   width; a regression test covers 8 sizes.

## Deviations from SPEC (SPEC updated in the same commit)

1. **Markers for synthesized voices come from piece-by-piece synthesis, not forced
   alignment.** Exact and model-free; the cost is a tiny seam in the voice at each marker.
   WhisperX is used for recordings only.
2. **No custom easing functions**: a document must mean the same on every machine.
   Named easings only (`back` and `spring` added).
3. **`after("span")` is `then()`** on the chain that made the span.
4. **Auto-pause is explicit** (`Scene.pace()`), run after the verbs, because it needs them.
5. **`bloom_in` grows objects that have a radius**; others only fade. Glow is a style role and
   cannot be animated yet.
6. **orbit_reveal** waits for 3D (Milestone 6).
7. No cloud TTS provider; `Synth` is a small protocol to add one.

## Untested

- WhisperX alignment, Piper voices (need model downloads), Windows espeak-ng.
- CUDA, as before.

## Open

- Fresh-eyes boundary review not run (Milestones 2-4).
