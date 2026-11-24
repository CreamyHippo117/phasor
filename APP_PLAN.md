# The code-first desktop app: plan

Working doc, like `GPU_SPEED.md`. Asked for by the owner on 2026-11-22: "one shared script
everyone uses". Status: ADR 0004 accepted; **phases 1 and 2 built in Milestone 25**
(`MILESTONE_25.md`, with the buttons and panels of `STUDIO.md`); phases 3 and 4 not started.

## What the owner wants

- One shared script that the person, the AI and the app all edit. Code can always be edited
  by hand, with or without an AI.
- Three ways in: type a new script, open a phasor script, or import a Manim scene's code.
- Every tool already there to edit, check and improve what renders.
- An AI that knows the library: in the app through an API when connected; without it,
  everything else still works.

## What exists to build on

| Piece | Where | State |
| --- | --- | --- |
| Code panel with Run (Ctrl+Enter) | `gui/window.py` `CodePanel`, `gui/controller.py` `Editor.code`/`run_code` | works; one panel among many |
| Preview, timeline, inspector, lint list, looks, controls | `gui/window.py` | works |
| Start screen, windows, export, settings, packaging | `gui/desktop.py`, `gui/app.py`, `packaging/` | works, installed on the owner's PC |
| Scene to Python | `core/pycode.py` `document_to_python` | works |
| Manim import | `importers/manim.py` | works; output is a phasor scene |
| AI tools (about 30) | `workspace/tools.py` `Workspace`, `phasor mcp` | work; patch-based |
| AI docs and skill | `docs/ai/`, `llms.txt`, `skills/phasor/` | say "you do not write rendering code": to rewrite |

The new app rearranges the existing window around the code panel. It is not a second app.

## The design

```
+------------------------------------------------------------------------------+
| File  Edit  View  Scene  AI                                                  |
+--------------------------------------+---------------------------------------+
|  scene.py                     [Run]  |                                       |
|  from phasor import ...              |            preview                    |
|  s = Scene("residues")               |         (click to select,             |
|  s.add(...)                          |          drag to tweak)               |
|  ...                                 |                                       |
|  # --- phasor tweaks ---             +---------------------------------------+
|  s.tweak("pole.label", ...)          |  timeline: markers, tracks, playhead  |
|  # --- end phasor tweaks ---         +---------------------------------------+
|                                      | Problems | Inspector | Looks | AI      |
|                                      |  lint list; click goes to the line    |
+--------------------------------------+---------------------------------------+
```

- **Left: the script.** Always editable. Run with Ctrl+Enter or on save. An error shows
  its line, and the preview keeps the last good scene.
- **Right: what it renders.** Preview, then timeline. Selecting an object in the preview
  shows its line in the script.
- **Bottom tabs:** Problems (lint, review), Inspector (what is on screen at the playhead),
  Looks, and AI (the chat, phase 4).
- **Hidden, not deleted:** sketching (pending the owner's answer), the controls panel and the
  activity dock move to View.

### One script, several editors (ADR 0004)

- A tweak from the inspector, the preview or an AI is applied in memory and written into the
  managed tweaks block at the end of the script. Structure and timing are code.
- The `Workspace` holds the script text with a version number. The editor and the AI tools
  edit it under the shared lock (CLAUDE.md: every edit goes through the `Workspace`).
- **Outside edits.** Claude Code over MCP, or any editor, changes the file on disk, so the
  app watches it. A clean buffer reloads and re-runs. If the buffer has unsaved changes,
  the app shows both versions and asks which to keep; it never merges silently.
- **One undo history, over the script text.** A tweak is a text edit to the block, so undo
  works the same whoever made the change.

### The AI

- **Without the app (works today):** Claude Code or Claude Desktop with `phasor mcp` and the
  phasor skill. Registered on the owner's PC on 2026-11-22 (Claude Code, this project,
  `--root Documents\phasor --allow-scripts`).
- **New tools** on `Workspace`, which serve both MCP and the app:
  - `script_read`: the text and its version.
  - `script_edit`: old string, new string and the expected version (refused if the file
    changed since it was read).
  - `script_run`: errors with line numbers, then a short review: lint, what is on screen at
    each marker, and the storyboard picture.
  
  Following CLAUDE.md, add them to `TOOL_NAMES`, `docs/ai/MCP.md` and `llms.txt` in the same
  commit, and rewrite `SKILL.md` and `MCP.md` for the code-first workflow: write the script,
  run, look, fix. A test keeps `.claude/skills/phasor/SKILL.md` equal to `skills/phasor/SKILL.md`.
- **In the app (phase 4):** an AI tab calling the Claude API with the owner's key, using the
  same tools, with the AI docs as a cached system prompt. Without a key the tab says how to
  add one, and nothing else changes.

### Ways in

- **New:** a template script (2D, 3D, complex function), as on today's start screen.
- **Open a phasor script:** as is.
- **Open a `.phasor.json` scene:** offer once to convert it to a script (`core/pycode.py`),
  which then becomes the source. No second generated file is kept beside it.
- **Import Manim:** a one-time conversion to a phasor script (importer, then document, then
  `pycode`), with the importer's warnings as comments at the top. The Manim file itself is
  left untouched.

### Trust

The app runs whatever the script contains, including code an AI wrote or an import brought
in. That is the point of a shared script, so it is said plainly in the app: the first run of
a script the person did not write asks once. The MCP server keeps `--allow-scripts` opt-in.
Related: `TESTING.md` C4 (the AI stays in its folder).

## Phases, each ending with its test

1. **The tweaks block** (ADR 0004): `Scene.tweak*`, writing and reading the block, folding
   in old overrides files. Exit: the invariant test (a fresh run of the script equals the live
   document after random mixes of edits and undo) and the migration test.
2. **The code-first window:** the layout above, Run on save, file watching with the conflict
   rule, errors at their line, preview to line and back. Exit: window tests on the offscreen
   Qt platform (an outside edit reloads; a conflict asks; an error keeps the last good
   scene), and the packaged app rebuilt.
3. **Script tools for any AI:** `script_read`/`script_edit`/`script_run`, the docs and skill
   rewrite. Exit: tool tests without the protocol, an MCP test, and the docs blocks running
   in CI. Then a real session: the owner asks Claude Code for a scene with the app open.
4. **The AI tab:** Claude API, key in settings (never in a scene or a log), prompt caching
   on the docs, cost shown per session. Exit: tests with a fake client; one real call
   from the packaged app.

## Questions for the owner

1. **ADR 0004:** tweaks in a block at the end of the script instead of `overrides.json`.
   Accept?
2. **Sketching:** hide it in the new layout (the code stays, with Milestone 11's tests), or
   remove it?
3. **Manim:** is a one-time conversion to a phasor script right, or should a Manim file stay
   the source and be re-imported on every change? Tweaks cannot be written into a Manim file.
4. **`.phasor.json` scenes:** convert to a script when opened, as proposed?
5. **AI tab:** your own Claude API key, paid per use. A working session should cost cents,
   since the docs are cached. Fine, or keep the AI outside the app (Claude Code) for now?
