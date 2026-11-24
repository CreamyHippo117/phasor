# Milestone 25: the desktop app that works

From `STUDIO.md` (owner, 2026-11-24: "a real working desktop app; the current one is not
great") and `APP_PLAN.md` phases 1 and 2 (ADR 0004, accepted). Exit condition: **phase 1's
invariant test (a fresh run of the script equals the live document after random mixes of
edits and undo) and its migration test; phase 2's window tests on the offscreen Qt platform
(an outside edit reloads; a conflict asks; an error keeps the last good scene); the buttons
and panels of `STUDIO.md` (Draft preview, Render with a quality, narration and snap, sound
and MIDI, footage) calling the shared `Workspace`; the packaged app rebuilt** (the last on
the owner's PC: `pc_tasks/13_desktop_app.md`).

Status: **built on Linux (llvmpipe OpenGL 4.5, no CUDA), Qt offscreen; not independently
audited.** Untested on Windows, macOS, a real GPU, a real screen, and as a packaged app.

## Built

| Area | Where | What |
| --- | --- | --- |
| Tweaks block (phase 1, ADR 0004) | `core/pycode.py` | `tweak_lines` (canonical: params by path, styles by `id.key`, rules in the order added), `with_tweaks` (replace the block in place, or add it before a trailing `__main__` guard, else at the end; code outside byte for byte, line endings kept), `find_tweaks`, `read_tweaks` (the block read with `ast.literal_eval`, nothing run), `insertion_offset`/`insert_code` (where the panels add code) |
| `Scene.tweak*` | `scene.py` | `tweak(path, value)`, `tweak_style(id, key, value)`, `tweak_rule(rule, priority=)`: through `Engine.apply(origin="tweak")` into the script engine's overrides, never its base; a tweak whose object is gone is kept (`Scene._orphans`) and the script runs (lint `orphan_override`); `Scene.tweaks`; `Scene.lines` (the script line that added each object, `_script_line`) |
| Running a script | `scene.py` `run_script`, `ScriptRun` | the engine (script base + block tweaks + an old overrides file, `core/patch.merge_overrides`), the Scene's variable name, the object lines; `load()` is `run_script(...).engine` |
| Workspace | `workspace/tools.py` | opening a script records `script_var`, `script_lines`, `script_disk` (text, stamp); an old `name.overrides.json` is folded into the block the first time and removed (`migrated`); `scene_save` on a script writes the block only (`_save_tweaks`), a `.phasor.json` path saves a copy as a document, another `.py` is refused; `diff("saved")` on a script reads the block (no code run); `read_text`/`write_text` (newline kept, atomic) and `file_stamp` |
| Render jobs | `workspace/jobs.py`, `Workspace.render_start/render_status/render_cancel`, `Engine.snapshot` | a render on a thread of a snapshot taken under the lock (with the live bake when there is one, ADR 0003's payload, else a fork); progress from `render_file(progress=)`, cancel through its `cancel` Event; writes `<stem>.partial<ext>` beside the target and `os.replace`s only on success (a cancel or failure removes the partial; an existing file is kept until then); an existing target only with `overwrite` (a draft's own default always); bad preset/time/fps/suffix refused before anything starts; at most 2 running; ids per workspace. MCP tools (in `TOOL_NAMES`, `docs/ai/MCP.md` with a `json calls` block, `llms.txt`) |
| Editor logic (phase 2) | `gui/controller.py` | `buffer` (the script in the editor), `is_script`, `set_buffer`, `run`/`run_code` (writes on success; an error keeps the last good scene, restores the file and records `error` {line, message}, line found in the traceback or its cause), `save` (Run on save when code outside the block changed, else write the buffer; no rerun for a tweak), text-based `dirty`, `_sync_block` (the live overrides written into the buffer's block after every edit, undo, redo and AI edit; a hand-edited block is left and flagged `block_edited`), `poll_file`/`resolve_conflict` (an outside edit reloads and runs when clean, else a `conflict`; nothing merged), `line_of`/`object_at_line`/`select_line`, `problems` (the script error, file conflict, block note, then lint with lines), `generation` (counts rebuilt scenes for caches) |
| Panels' logic | `gui/controller.py` | `add_footage` (video/picture, background or placed), `add_sequence` (a frame -> `%0Nd` pattern), `add_sound` (at a marker's time or seconds), `add_notes` (`.mid` or typed `parse_notes`: `C4 0 0.5, F#3 ...`; at a marker: they follow it), `export_midi`, `narration`, `set_narration_audio`, `speech_preview`, `snap_preview` (rows, kept, flags said plainly, `can_apply`), `snap_apply`; on a document each is one undoable `scene_edit`/`narrate`/snap, on a script a line of code (with its import) before the block, run, taken back if it fails; a file from outside the folder is copied into `media/` (`bring_file`); `render_plan`, `start_render`, `draft_preview`, `poll_job`, `cancel_job` |
| Window | `gui/window.py` | the code-first layout: `CodePanel` left (Run, error line in red, the selection's line in amber, a click on a line selects its object), viewport + timeline + tabs right (Problems, Inspector, Looks, Narration, Sound and MIDI, Footage, Sliders, AI stub, then `phasor.panels` entry points: Sketch now, Explorer later); toolbar Run, Draft preview (Ctrl+Shift+P), Render... (Ctrl+R), Play, Undo, Redo, Save; a progress bar and Cancel in the status bar; the timer polls the file (about twice a second) and the job; `ask_conflict`, `play_file` (the system player); "Who changed what" hidden under View; `RenderDialog` (video or picture, quality with size and fps, path, replace asked, progress, Cancel); without OpenGL the preview says so |
| Desktop | `gui/desktop.py`, `gui/app.py` | trust: `Settings.trusted`, `Desktop.confirm_run` asked once per script the app did not write (examples it copies and scripts it saves are trusted); Scene menu; `ExportDialog` is `RenderDialog` plus the web page (`ExportJob`); `ExportJob.cancel` now removes what the process wrote (the Milestone 21 flaw); closing a window cancels its render jobs; Save As keeps a script a `.py` |
| Docs | `docs/ai/API.md`, `MCP.md`, `PITFALLS.md`, `llms.txt`, `SPEC.md`, ADR 0004 (Accepted), `APP_PLAN.md`, `packaging/README.md`, `pc_tasks/13_desktop_app.md` | the tweaks block with a running block; the desktop app section rewritten with a running block; contract 2 text (SPEC line 63 and the "Scene document" paragraph) per ADR 0004; the GUI "As built (Milestone 25)" paragraph; the job tools |

## Verification (Linux, 4 cores, llvmpipe OpenGL 4.5, no CUDA; Qt offscreen)

| Step | Result |
| --- | --- |
| `ruff check .`, `ruff format --check .`, `lint-imports` | pass; 2 contracts kept |
| New tests | `test_tweaks_block.py` 6, `test_render_jobs.py` 6 (one real GL draft), `test_app_code_first.py` 18 (the invariant, 3 seeds x 14 random steps), `test_app_window.py` 11: all pass |
| Focused suites | `test_gui_controller` 20, `test_gui_window` 13, `test_desktop` 42, `test_m10_exit_demo` 1, `test_mcp` + `test_remote` 46, `test_pycode` + `test_engine` + `test_layers` + `test_contracts` 53, `test_draft -m "not slow"` 66, `test_docs` + `test_kit` 16: all pass. `test_final -k "not slow"`: see below |
| Full suite | not run (instructed: the box is shared and slow) |
| `phasor doctor` | tier B (llvmpipe, no CUDA) |
| Exit demo | phase 1 and 2 exit tests pass here; the packaged app is **not** rebuilt here (Windows): `pc_tasks/13_desktop_app.md` |
| Contract audit | by the authoring model only (no subagent available): every document change still goes through `Engine.apply` (tweaks via `origin="tweak"`, panels via `scene_edit`/`narrate`/snap or a script run); orphan tweaks reach the engine only through its constructor's overrides, like an overrides file; the job snapshot is built under the lock and never edited; GL only through the renderer on the GL thread; no new layer import (`lint-imports`). Not independent |

## Deviations

1. **Contract 2 text in `CLAUDE.md` still names `overrides.json`** (instructed to leave
   `CLAUDE.md` untouched). SPEC.md is updated per ADR 0004 (accepted by the owner); the
   CLAUDE.md summary needs the same edit by the owner.
2. **New from a template still makes a document**, not a template script (APP_PLAN "Ways in").
   The code view shows its generated code and Run makes it a script (`name.scene.py`), as
   before. The owner's answers to APP_PLAN's questions are not written down anywhere in the
   repo; this keeps today's behaviour until they are.
3. **A `.phasor.json` scene is not offered a conversion when opened** (APP_PLAN question 4):
   documents open as before (Explorer documents must stay documents).
4. **Run writes the file** (it did before; an error restores the file), so "unsaved" in a script
   means typed and not run, or a tweak not saved yet. The editor's own text undo is Qt's; the
   tweak undo is the engine's (one history per run, as before), not one undo over the text.
5. **A hand-edited tweaks block is not rewritten** until Run (`block_edited`, shown in
   Problems); a tweak made meanwhile is in the scene but not in the text until then.
6. **Blank lines and trailing spaces are not "code changes"** for Run on save (the blank
   line written before a new block must not force a rerun).
7. **Sound clips "at a marker" take the marker's time now** (`Sound.start` is seconds only);
   `Notes` at a marker follow it (its `start` takes a marker name).
8. **Render size is the preset's**; only frames a second can be changed in the dialog.
9. **No recording from a microphone**: the `gui` extra (PySide6-Essentials) has no
   QtMultimedia and no new dependency is allowed; the Record button is shown disabled with the
   reason. Drafts play in the system's player, not inside the window, for the same reason.
10. **No frames shown while a render runs** (`render_file` reports progress, not frames).
11. **A render job runs in the app's process** (a thread; GL on the GL thread) with one
    worker; the web page export stays a separate process.
12. **"Selecting an object shows its line"** uses the line of the `s.add` that made it (or its
    parent), else the first line naming its id in a string; objects made inside helper
    functions point at the line in the script that called into phasor.

## Deferred

- APP_PLAN phase 3 (`script_read`/`script_edit`/`script_run` tools, the skill and MCP.md
  rewrite for the code-first workflow) and phase 4 (the AI tab with the Claude API).
- Conversion of `.phasor.json` scenes and Manim imports to scripts on open (owner's call).
- Rewriting literals in place instead of the block (ADR 0004 alternative).

## Untested

- **Windows 11, the packaged app, a real screen**: everything in `pc_tasks/13_desktop_app.md`
  (draft preview button, cancel, final render quality, audio snap heard, MIDI heard, video
  background, an outside edit from Notepad, a conflict, an error line, the trust question,
  no console window).
- macOS; CUDA (the jobs use the same `render_file`); hearing the sound of drafts.
- The real `phasor mcp --attach` session driving the jobs from Claude Code.
