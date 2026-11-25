# Milestone 25: the desktop app that works

From `STUDIO.md` (owner, 2026-11-24: "a real working desktop app; the current one is not
great") and `APP_PLAN.md` phases 1 and 2 (ADR 0004, accepted). Exit condition: **phase 1's
invariant test (a fresh run of the script equals the live document after random mixes of
edits and undo) and its migration test; phase 2's window tests on the offscreen Qt platform
(an outside edit reloads; a conflict asks; an error keeps the last good scene); the buttons
and panels of `STUDIO.md` (Draft preview, Render with a quality, narration and snap, sound
and MIDI, footage) calling the shared `Workspace`; the packaged app rebuilt** (the last on
the owner's PC: `pc_tasks/13_desktop_app.md`).

Status: **built on Linux (llvmpipe OpenGL 4.5, no CUDA), Qt offscreen; audited
independently (2026-11-25): 3 blockers (plus a lost-update race) and 12 should-fix fixed,
with 9 of the nits; each with a regression test that fails on the code before the fixes
and passes after (see "Independent audit").** Untested on Windows, macOS, a real GPU, a
real screen, and as a packaged app.

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
| New tests | `test_tweaks_block.py` 6, `test_render_jobs.py` 6 (one real GL draft), `test_app_code_first.py` 19 (the invariant: an orphaned tweak first, then 3 seeds x 18 random steps of typed runs, tweaks, AI edits, undo, redo, saves, outside edits; a CRLF script), `test_app_window.py` 11: all pass |
| Focused suites | `test_gui_controller` 20, `test_gui_window` 13, `test_desktop` 42, `test_m10_exit_demo` 1, `test_mcp` + `test_remote` 46, `test_pycode` + `test_engine` + `test_layers` + `test_contracts` 53, `test_draft -m "not slow"` 66, `test_docs` + `test_kit` 16: all pass. `test_final -k "not slow"`: 477 passed, 8 skipped, 3 deselected (4 min 51 s). After the last fixes (CRLF, render decision moved into the Editor) the touched suites again: 56 + 48 passed |
| After the audit fixes (2026-11-25, on the merge with M28/M30) | `ruff check`, `ruff format --check`, `lint-imports` pass. One suite at a time: `test_tweaks_block` 6, `test_render_jobs` 6, `test_app_code_first` 19, `test_app_audit` 28, `test_app_window` 14, `test_gui_controller` 20, `test_gui_window` 13, `test_desktop` 42, `test_m10_exit_demo` 1, `test_mcp` 39, `test_remote` 7, `test_pycode` 5, `test_engine` 30, `test_layers` 9, `test_contracts` 9, `test_proc` 2, `test_docs` 5, `test_kit` 11, `test_draft -m "not slow"` 66, `test_final -k "not slow"` 507 passed, 8 skipped: all pass. The auditor's `fuzz.py`: no failures |
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
4. **Run writes the file** (it did before; an error restores the file, unless another
   editor changed it meanwhile), so "unsaved" in a script means typed and not run, or a tweak
   not saved yet. There is no single undo over the text: the text box keeps its own undo for
   typing (a tweak written into the block is one step of it), the scene's undo takes back
   tweaks, and then the app's own additions to a script (a panel's footage, sound, notes,
   narration or snap: one step each, the script runs again). A typed Run clears that last
   history (undoing past it would throw away the typed code).
5. **A hand-edited tweaks block is not rewritten** until Run (`block_edited`, shown in
   Problems); a tweak made meanwhile is in the scene but not in the text until then.
6. **Blank lines and trailing spaces are not "code changes"** for Run on save (the blank
   line written before a new block must not force a rerun).
7. **`Sound.start` takes a marker's name** (audit S9; as `Notes.start` did in Milestone 23):
   a sound at a marker follows it. Documents with seconds load as before; a numeric string
   ("1.5") stays seconds. A sound at a marker the scene lacks is left out of the render and
   said in its problems (no lint rule for it yet: deferred).
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
13. **Trust is per script file, not per content** (audit nit): a script the person let run
    once runs without asking again, also after another program (git pull, an AI) changed it.
    The question says so. `phasor gui X.py --root DIR` now asks too; a `name.scene.py` that
    Run made from a document, and a script saved with Save As, are trusted.
14. **A runaway script stuck inside C code cannot be stopped in-process** (audit S7): Run and
    reloads go on a worker thread; Cancel run (and the timeout, `PHASOR_RUN_TIMEOUT`, 600 s)
    raise `RunCancelled` in that thread (`PyThreadState_SetAsyncExc`), which stops a
    pure-Python loop at once and a long C call when it returns. One that never returns is let
    go: the editor keeps the last good scene and ignores its result; the thread may keep a
    core busy until the app quits. A child process was not chosen: a script's scene-local
    types and updaters must be built in this process.
15. **While a script runs, edits are refused** (here and from an attached AI:
    `Workspace.busy`), since the scene it builds replaces the open one.
16. **Template scripts**: New from a template still makes a `.phasor.json` scene (deviation 2);
    it needs the owner's answer to APP_PLAN's "Ways in".

## Independent audit (2026-11-25) and what was fixed

Every fix has a regression test (`tests/test_app_audit.py` unless named) that fails on the
code before the fixes (checked by running the new tests against a copy of the tree at the
merge commit 4bed8de: 26 of 27 failed or errored and the two runaway tests hung until
killed; the S4 test was corrected so it fails there too) and passes after.

| Finding | Before | Fix | After |
| --- | --- | --- | --- |
| B1 begin marker without its end | every save replaced the span from the orphan begin to a new end: the person's code after it was deleted (Editor and MCP) | `find_tweaks` reads real comment markers at column 0 with `tokenize` and raises `BlockError` (with the line) for a begin without an end, an end without a begin, an end first, two blocks, an indented marker, or code that does not tokenize; nothing is written, the tweaks stay in the scene (`block_problem`, `unwritten`, Problems), `scene_save` refuses | the file byte for byte unchanged; refused with the line |
| B2 outside edit during a Run | `script_disk` recorded after the run: a change written while the script ran looked seen; the next Save erased it | the text and stamp are read **before** the script runs (`Workspace._load_script`) | the change is noticed (reloaded or a conflict) and kept |
| B3a Save without re-checking the disk | an edit made between two polls was overwritten | `_check_disk` before every write (`_write_script`, Run, Save): a difference is `ScriptChanged`, the editor turns it into a conflict and asks | nothing written; the person chooses |
| B3b an AI's `scene_save` absorbed an outside edit | it read whatever was on disk, wrote block + outside code and marked it seen | `_save_tweaks` writes only over the text it last ran or wrote, else refuses (`ToolError`) | the outside edit stays; the editor sees it |
| S1 Render dialog confined to the folder | a path in ~/Videos was refused | `Workspace._render_job` (not a tool) for the person; `render_start` keeps the AI in its folder | renders anywhere for the person |
| S2 one `job` for the whole editor | the dialog's Cancel cancelled the draft; the final kept rendering | each caller keeps its own job id (`start_render` returns it; `job_status`, `cancel_render`); `ed.job` is only the Draft preview's | each Cancel stops its own render |
| S3 partial name `a.partial.mp4` | a person's file of that name was truncated then deleted | a hidden partial with a random part, created with `O_EXCL`, removed only if ours | the file untouched |
| S4 web export cancel | deleted a page that was there before | `ExportJob` writes to its own partial file and renames on success | the old page stays |
| S5 markers in a docstring; two blocks | block written inside the string; the second block kept stale values (live 0.1, fresh 0.9) | `tokenize` (strings are not markers); two blocks refused | fresh run equals live; docstring untouched |
| S6 file modes | 0444 became 0644, 0600 became 0644; a read-only file escaped the window's slot | `write_text` keeps the mode, refuses read-only (`ToolError`), the window catches `OSError` | refused, modes kept |
| S7 runs on the GUI thread | `while True` froze every window and the AI | `RunTask` on a worker thread, Cancel run, timeout; edits refused meanwhile | the window lives; cancel keeps the last scene (`test_app_window.py` too) |
| S8 undo | every tweak wiped the text box's undo; panel additions to a script were not undoable | the box is edited in place as one undo step (`CodePanel._replace_text`); the app's additions are undone one step each | `isUndoAvailable` kept; undo/redo of a sound |
| S9 sound at a marker frozen to seconds | a snap desynced it | `Sound.start` takes a marker name; the clip is placed at the marker by the engine | follows the marker |
| S10 two draft defaults | Editor beside the script, Workspace in `phasor_out/<title>` | `Workspace.default_render_path`: a draft beside the scene's file for both | one default |
| S11 render over an input | `render_start(out="assets/clip.mp4", overwrite=true)` replaced the footage | `_not_an_input`: every `Files` value, the narration's recording, the script | refused (`render` too) |
| S12 Windows: draft open in a player | `os.replace` would fail the job | `_move_into_place`: the next free `<scene> (2).draft.mp4`, said in `note` | done, and said (checked on the PC: `pc_tasks/13`) |
| nit mixed line ends | one CRLF made the whole file CRLF | `_with_line_ends`: each existing line keeps its own end (`difflib`) | kept |
| nit `{s}`/`{id}` in a file name | mangled by the code template | code built by a function, no template | kept |
| nit rule order; dirty on open (p8) | an orphan rule moved to the end; a hand-formatted block was rewritten at once | `Scene._tweaks` in call order; a block that already says what the scene holds is left as written | order kept, not dirty |
| nit copied file left in `media/` | stayed after a failed add | removed when the add fails (ours only) | gone |
| nit `_same_file` read both files whole | | `filecmp` (sizes, then chunks) | |
| nit Run on a document over an existing `name.scene.py` | overwritten | refused with a message | kept |
| nit Looks / marker drags on a script | bare `scripted_scene` | the message says where it belongs in the code | |
| nit `render_cancel` held the lock 5 s | | waits at most 0.5 s; the editor's cancel holds no lock | |
| nit `phasor gui X.py --root DIR` | ran without asking | asks once (`cli._may_run`) | asked (`test_app_window.py`) |
| nit tests | a tautology; a GL cancel test that asserted only when cancelled; the invariant compared only the document | assert; assert the outcome; compare document and overrides | |

The auditor's `fuzz.py` (moving_dot, video_input, notes_overlay; seeds 1-3; 30 random steps
of tweaks, styles, AI edits, undo, redo, saves, rules, looks, marker moves, zooms, removed
objects, outside edits, typed runs): no failures, document and overrides both compared.

Re-verified: SPEC's "re-running the script never destroys tweaks" holds after B1 (the
tweaks stay in the scene and the file is never rewritten while the block is unreadable).

## CLAUDE.md edits needed (for the lead; this branch does not touch CLAUDE.md)

- Contract 2, replace "`name.phasor.json` (sorted keys) + `name.overrides.json` (id + param
  path, plus added constraints; ADR 0002). Re-running a script never loses overrides; a
  dangling override is lint `orphan_override`." with: "`name.phasor.json` (sorted keys);
  tweaks (id + param path, plus added constraints; ADR 0002) live in the script's managed
  tweaks block for a script scene (ADR 0004) and in `name.overrides.json` for a scene with no
  script. Re-running a script never loses tweaks; a dangling one is lint `orphan_override`."
- "Desktop app" line: replace "ADR 0004 (tweaks inside the script, touches contract 2)
  accepted by the owner; phase 1 not started." with "ADR 0004 (tweaks inside the script,
  touches contract 2) accepted by the owner; phases 1 and 2 built in Milestone 25
  (`MILESTONE_25.md`, audited); phases 3 and 4 not started."
- Studio line: "25 the desktop app ... comes last" -> "25 the desktop app is built and
  audited (`MILESTONE_25.md`); the packaged app awaits `pc_tasks/13_desktop_app.md`."

## Deferred

- APP_PLAN phase 3 (`script_read`/`script_edit`/`script_run` tools, the skill and MCP.md
  rewrite for the code-first workflow) and phase 4 (the AI tab with the Claude API).
- Conversion of `.phasor.json` scenes and Manim imports to scripts on open (owner's call).
- Rewriting literals in place instead of the block (ADR 0004 alternative).
- Audit nits not done: a tweak naming an unknown parameter of an existing object still
  fails the script at the block's line instead of becoming lint `orphan_override` (this
  predates Milestone 25: an overrides file did the same); trust per content (deviation 13);
  a lint rule for a sound at a marker the scene lacks.

## Untested

- **Windows 11, the packaged app, a real screen**: everything in `pc_tasks/13_desktop_app.md`
  (draft preview button, cancel, final render quality, audio snap heard, MIDI heard, video
  background, an outside edit from Notepad, a conflict, an error line, the trust question,
  no console window).
- macOS; CUDA (the jobs use the same `render_file`); hearing the sound of drafts.
- The real `phasor mcp --attach` session driving the jobs from Claude Code.
