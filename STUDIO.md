# Studio: preview render, audio and media inputs, MIDI, and a desktop app that works

Asked for by the owner on 2026-11-24: "add midi output, mp4/png input scenes and a real working
desktop app. The current one is not great." The basics the owner listed:

1. In the app, with a script open, a button for a **low-quality preview render**: no blur, no
   motion blur, no haze or bloom, 15 fps, "ridiculously efficient in other ways".
2. The normal **high-quality render**, with a choice of quality.
3. A **MIDI input** to overlay with the audio at certain points.
4. **mp4 / wav input** for the audio (narration), and an option for **scenes to snap to the
   narration smartly**.
5. **MIDI output**, and **mp4 / png input** as scenes.

This file holds the plan and the status, like `MANIM_PARITY.md`. Milestones 21-25.
`APP_PLAN.md` (ADR 0004, accepted) stays the design for the code-first window; Milestone 25
builds its phases 1 and 2.

## How the requests are read (the owner can redirect any of these)

- **Draft render** = a new quality tier below `preview`: 15 fps, a smaller picture, and the
  look stripped to what makes the picture readable (paint, strokes, text; no bloom, no lens, no
  haze, no motion blur). It is a render of the same document, so what you see is what the real
  render draws, only plainer. Efficiency: frames whose state is identical to the last are not
  drawn again; no layout re-bake; CPU kernels on every core; progress and cancel.
- **MIDI** = a notes track. A `Notes` object (or document-level track) holds notes (pitch, start,
  length, velocity, channel). They come from a `.mid` file in the scene folder, or from a script
  (any object may generate notes, e.g. the partials of a Fourier series). They are **played**
  (a small built-in synthesizer, deterministic, no new dependency) and mixed into the audio at a
  start time, exactly like `Sound`; and they can be **exported** to a `.mid` file. So "MIDI
  input to overlay with the audio at certain points" and "MIDI output" are the two ends of one
  type.
- **mp4 / wav input** = the narration audio comes from a file instead of the text-to-speech
  voice. mp4: the audio track is extracted. **Snap to narration** = given that audio, detect
  speech and pauses, and move the document's markers (scene starts) onto the sentence and
  phrase boundaries, using the script text for the words when there is one. It writes marker
  times through `Engine.apply`; it is an offered action with a preview of what moves, never
  silent.
- **mp4 / png input scenes** = a `Video` object (an mp4 in the scene folder: start, trim, speed,
  loop, fit to frame or placed like any object) beside the existing `Image` (png/jpg). Both can
  fill the frame as a background scene.
- **Desktop app** = the code-first window from `APP_PLAN.md`: the script on the left, the preview
  on the right, the timeline, Problems, Inspector. With the new buttons: Run, Draft preview,
  Render (a quality chooser), and panels to attach narration audio, MIDI and sound clips.

## Milestones

| # | What | Depends on | Status |
| --- | --- | --- | --- |
| 21 | Draft render: the `draft` preset, a draft path in the renderer, frame de-duplication, CLI/MCP/engine API | none | DONE (independently audited, blocker fixed; `MILESTONE_21.md`) |
| 22 | Audio inputs and narration snap: wav and mp4 audio as narration, speech/pause detection, `snap_to_narration` | none | planned |
| 23 | MIDI: a self-written SMF reader/writer, `Notes`, the synthesizer, mixing, export | 22 shares the mixer; do after or with it | built, awaiting an independent audit (`MILESTONE_23.md`); hearing it untested |
| 24 | Video and picture inputs: `Video` (mp4), `Image` extended, full-frame background scenes | none | DONE (independently audited, blockers fixed; `MILESTONE_24.md`) |
| 25 | The desktop app: APP_PLAN phases 1-2 plus the buttons and panels above | 21, 22, 23, 24 | planned |

Order: 21, 22 and 24 are independent and run at once. 23 follows 22 (same mixer and audio
track code). 25 comes last, since it puts all of them behind buttons.

## Rules every milestone follows

CLAUDE.md's seven contracts and conventions, and the lessons the independent audits of
Milestones 15-20 taught:

- Nothing that worked before fails to load, import or render because of a new limit: clamp and
  lint instead.
- Every parameter, file and input is bounded; hostile or malformed files (WAV, MIDI, mp4, png)
  are a lint issue and draw or play nothing, never a traceback, hang or memory blow-up.
  Measure time and memory at the bounds.
- Files are read only from inside the scene's folder, regular files only (`core/files.py`).
- Layers: `core`, `style`, `time`, `layout` import no GPU/GL/video library (so mp4 decoding
  lives above them, in `domains` or `render`); `lint-imports` enforces it.
- Windows CI has no OpenGL: new lint and examples are tested with `passes=None` and with OpenGL,
  and no POSIX-only calls in tests (no `utime(follow_symlinks=)`; a `0` time budget can read as
  not exceeded on Windows' coarse clock).
- Every shipped type: semantics, bounds, inspector output, lint, an `API.md` entry, an example,
  a storyboard snapshot test, a golden still (look at it before committing).
- Windows, macOS, CUDA, and hearing the audio are untested until they run there: say so.

## Status

Planned 2026-11-24. Nothing built yet.
