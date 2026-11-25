# Milestone 29: the Explorer panel

Plan: `EXPLORER.md`, row 29; design: `docs/decisions/explorer_advice_2026-11-24.md` section 3
(layout, direct manipulation, undo and save, send to scene), 7.4-7.5 (sliders, typing) and
8.5 (how AI edits appear). Built 2026-11-25 on the merged Milestones 21-28, 30 and 25.

Status: **built on Linux (llvmpipe OpenGL 4.5, no CUDA), Qt offscreen; audited
independently (2026-11-25): 2 blockers and 7 should-fix fixed, with most of the nits, each
with a regression test that fails on the code before the fixes and passes after (see
"Independent audit"); merged with the Milestone 25 second fix round; not gated.** The feel (typing, dragging, orbiting at 1e4-1e5 points) can only be
judged at a real screen: `pc_tasks/14_explorer_feel.md`. **Untested on Windows, macOS, a
real GPU, a real screen and as a packaged app.**

## Built

| Area | Where | What |
| --- | --- | --- |
| The logic (no Qt) | `phasor/gui/explorer.py` `Explorer` | a view of the one engine over an `Editor`: `rows()` is `explorer.rows_of(document)` with the session over it (the text being typed, its error and pending answer, `locked`, the kind badge, a slider's shown value, `playing`, `animated`), then the Explorer's objects that are not typed rows (sets, sections, equilibrium views); no second store. Every change goes through the shared `Workspace` under `Editor.lock` (`explorer_edit`, `explorer_preset`, `_explorer_typing/_commit/_leave/_preview`, or `Editor._edit` for a patch) and is logged in the history and "who changed what" with a readable name |
| Typing | same | each keystroke locks the row (`_explorer_typing`), reads it on a worker thread (`classify`, an injected executor) with a generation number (a late answer for older text is dropped, `dropped`), previews a clean row on a fork (`_explorer_preview`), and commits it as **one** patch when the debounce (0.2 s, injected `clock`) has passed, or 1 s when the commit would make a new variable (a pause mid-name never makes a slider for `sig`), or at once when the row is left (with an error too: the row is kept with its message, as `explorer_edit` keeps it). The draft follows its row to the new key after a commit (a new row, a kind change). Errors are plain words under the row; hostile text never raises. `settle()` commits a row typed inside the debounce (and a wheel zoom at rest) before Save, the desktop's close question, R2/R3 and convert, so it is never lost |
| Row actions | same | kind badge (a kind over the guess), visibility, colour (a style role), delete, reorder (`row`), each one patch; non-row objects: opacity, a style role, `remove_object` |
| Sliders | same | from the Variables and their Controls: drag on the view (a fork), commit on release (`rho = 24.74`; on a script, a tweak through `Editor.set_control/apply_controls`); a typed value; min/max/step edited in place (hostile text refused plainly); **play** (session: crosses the range and back) and **keep** (a track on `<id>.value`, linear, 4 s a crossing: it renders); a slider a track animates says so and is not dragged |
| Presets, + row, palette | same | `presets()`/`load_preset` (`explorer_preset`); `add_items()`/`add()`: tracer (a point row), disk or ball (`flow_set`), source (a 2D field term), section plane (`flow_section`), obstacle shown with why it is not there; `palette()` generated from the registry: every shipped type with an example in `examples/` that places with its defaults (72 types here), `place()` one patch (a flow object goes on the first system; `pos` at the point given) |
| Session state | same | playhead (flow time past the document's resting `tau`), play, pause, speed (`1x normal` readout), reverse (the playhead replays backwards and stops at 0), restart, step, scrub; the camera while the mouse is down, the wheel's zoom until it rests, a handle or a new disk mid-drag, the hover, the equilibrium card, the record marks. None of it is an op |
| The view and the picture | same | `view()`: a fork of the engine (or of the typing preview) with every system's `rate` 1 (so the scene time is the playhead) and the session's slider, camera and handle values, rebuilt only when one of them or the document changes (about 1 ms; the flow sets' checkpoints are a module-level cache, so scrubbing and orbiting never integrate again). `frame(size)`: half size with `quality="interactive"` while the person interacts and 150 ms after, then full (`render_plan` is the seam the tests use). `perf()`/`perf_text()`: frames a second, draw ms (moving, still) and input-to-picture ms, wall clock, shown in the transport line for the owner's check |
| Direct manipulation | same | `press/move/release/double_click/wheel/key` in viewport pixels: a click on empty space adds a tracer there (2D: z = 0; 3D: the plane through the orbit target facing the camera); shift+drag draws a disk or ball (centre, then radius); drag a set's centre or rim, or a tracer's dot (the tracer's row is retyped); click an equilibrium for its card (position, kind, eigenvalues, subspace dimensions from the `flow_equilibria` semantics), double-click or `zoom_in` flies the camera there (0.5 s, session) and commits one patch: the camera and its `EquilibriumView`; drag turns the 3D camera (ctrl or the middle button slides it) or slides the 2D one, committed once on release; the wheel commits once it has rested 0.4 s; space, left/right, `r`. Equilibria and handles are found by projecting their points (`Camera.to_screen`), so it works without OpenGL's id pass |
| R2 / R3 | same, `Editor.open_document` | two documents, two files (`<name>_r2` / `<name>_r3.phasor.json`; the file the person started from stays the one of its own space); the other is made from a preset the first time (Van der Pol, Lorenz); `needs_save` while unsaved; rows locked to the person are let go; each file keeps its own camera |
| Convert to script | same, window | `CONVERT_WARNING` asked at the moment of conversion; Run on the code generated from the document now writes `name.scene.py`; then rows are read-only (`CONVERTED`), sliders write tweaks, recording is refused |
| Record, send to scene | same, `marks_ops` | record, then **mark** keeps the playhead, the camera as shown and every variable's value; stopping turns the marks into markers (`mark_1`, ...) and, for each camera parameter and variable that differs, its value at the first mark and `smooth` segments between the markers (`MarkerRef` times), and `rate` 1 on systems that did not flow: one patch ("recorded 3 marks"). Only marks, never every mouse wiggle |
| Export | same, `Workspace._render_job(source=)` | `export_plan(tau0, tau1, seconds)` ("from tau 0 to 30 in 20 s"; hostile numbers, a zero duration, a rate over 1000 refused; a note when `MAX_STEPS` will clamp); `export_engine` a fork with each system's `tau` and `rate` and the duration; `export` a render job of that fork through the normal renderer, followed in the status bar. The document is not changed |
| Analysis and AI notes | same | `equilibria_table()` (with the window and grid searched), `lyapunov()` and `sweep()` on a fork, computed off the GUI thread (their own executor, so a sweep never delays typing), marked `stale` once the scene changes; `ai_notes()` and `history_names()` ("AI: rho = 24.74") |
| AI names | `workspace/explorer.py` `describe_edit`, `workspace/remote.py` | an attached AI's `explorer_edit` is named by its rows in "who changed what" (`rho = 24.74`, `added (1, 2, 3)`, `removed point`), `explorer_preset` by its preset; other tools by their name as before. `fresh_id` (ids by meaning, never a retired one) |
| The window | `phasor/gui/window.py` | `ExplorerPanel` (R2/R3, `+ row` with Insert, presets, Convert to script, the rows, the card and zoom in, the transport with its readout, record/mark/Export video..., tabs Analysis and AI notes), `RowWidget` (visibility, badge, text, colour, delete, the error under it; a variable's slider, value, play, keep, `>` range), `ExportFlowDialog`. The left pane is a `QStackedWidget`: the script or the rows (View > Explorer, Ctrl+Shift+E; an explorer document opens in it). In the Explorer the viewport forwards press, move, release, double-click, wheel and keys to the `Explorer`; Play/Space drives its transport; the 30 Hz tick polls it (answers, commits, the zoom flight, the full picture due) and advances it; an AI's edit refreshes the rows through `Editor.changed()`. The list is rebuilt only when its shape changes, so the row being typed in keeps its widget and cursor. Without OpenGL the preview says so and the rest works. Desktop: View > Explorer |
| Docs | `docs/ai/API.md` | "The Explorer panel (Milestone 29)" with a running python block (headless, no Qt, no GL) |
| PC check | `pc_tasks/14_explorer_feel.md` | exact steps (type Lorenz, drag rho, play, drop a ball, click C+, zoom in, orbit with 1e4 and 1e5 points, R2/R3, record and export, the AI while typing, convert) and an expected-vs-measured table |

## Tests (Linux, this box)

| Suite | What |
| --- | --- |
| `tests/test_explorer_panel.py` (67) | no Qt: the rows follow the document and an AI edit; `gui/explorer.py` imports no Qt (AST); fast typing commits once after the debounce; no prefix sliders, and a pause mid-name makes none; a late answer is dropped; a retyped row previews on a fork, then one named step; an error row is shown and kept only when left; 22 hostile texts never raise; Ctrl+Z while typing; the lock against the AI (`row_locked` for an edit, a new row rewriting it and a preset), then `row_changed` for an old version; an AI edit elsewhere drops a stale preview; AI edits named `AI: ...` through the real remote server; `describe_edit` never raises; badge, visibility, colour, order, delete one step each with their names; slider drag and release; 7 bad ranges refused; range edit and typed value; play and keep (the track's value half way at 2 s); **a minute of play at 2x, reverse, scrub, step, restart and bad speeds leave `Editor.version` and the undo depth unchanged**; the playhead is flow time past the resting `tau`; quality interactive while dragging and full after; orbit session then one step; 2D pan, the wheel commits once at rest; a click drops a tracer under the mouse (2D, and 3D on the plane through the target); shift+drag draws a disk, its handles move and resize it; dragging a tracer retypes its row; C+ card and zoom in as one step; the analysis tab (stale, hostile inputs); the add menu; every palette type places in one step; presets (hostile names); R2/R3 two files; record marks to markers and eased segments; `marks_ops` pure; export plan and engine (6 bad questions); convert keeps the rows and refuses new ones (sliders become tweaks); OpenGL: the picture is the view, half size while interacting, `perf`; the real reader thread while an AI thread edits |
| `tests/test_explorer_window.py` (19) | Qt offscreen: an explorer document opens in the Explorer and View toggles it; a script opens in the code view; typing in the new row commits after the debounce (the row keeps its widget); an error under the row; the row being typed is locked and an AI's tracer appears live; a slider drag previews and its release is one step; the transport plays without touching the document; a click drops a tracer and a drag pans (session until release); the add menu and palette; presets and the analysis tab; the export dialog checks the question; convert asks first; without OpenGL the preview says so; closing commits and lets go of a row being typed; Ctrl+S and closing the desktop window (its save question) keep a row typed inside the debounce; the export has its own file (`<scene>.explore.mp4`) and asks before replacing; OpenGL + av: the export renders the fork to an mp4 and the document is unchanged |
| `tests/test_explorer_panel_audit.py` (20) | the audit's regression tests (see "Independent audit") |
| `tests/test_app_window.py` | the left pane is now a stack (the script, or the rows): one assertion changed accordingly |

Results: see "Test runs" below.

## Deviations

1. **The playhead is flow time past the document's resting `tau`, through a view fork with
   `rate` 1 on every system.** The document's `tau` and `rate` are untouched (session state
   never reaches it). Because the view's scene time is the playhead, a kept slider track and
   recorded camera segments also play along the playhead, as they will in the video.
2. **The Analysis and AI notes tabs are inside the Explorer panel** (the left pane), not in
   the window's right-hand tab strip, so the window's tabs stay the code-first window's.
3. **R2/R3 do not keep the undo history across a switch** (`Editor.open_document` opens a
   new scene, as a reload does) and need the current file saved first. The files are
   `<name>_r2`/`<name>_r3.phasor.json`; the file the person started from is used for its own
   space.
4. **Non-row objects in the list** (sets, sections, equilibrium views) have no `visible`
   parameter: hiding sets `opacity` 0; their text is a read-only summary; delete is
   `remove_object`.
5. **The wheel commits its zoom once it has rested 0.4 s** (there is no mouse-down for a
   wheel); the camera while the mouse is down commits on release. A camera a track animates
   (after a recording) cannot be moved by hand: the view keeps the session camera and says
   why.
6. **Record:** a mark's scene time is the playhead's; a mark taken at the same time as the
   last is held 1 s later (`HOLD`); systems with `rate` 0 get `rate` 1 in the same patch,
   so the video's flow time follows the playhead as recorded.
7. **A clean row waits 1 s, not 0.2 s, when its commit would make a new variable**
   (`NAME_PAUSE`), so a pause inside `sigma` never makes a slider for `sig`. Leaving the
   row commits at once.
8. **"Keep" makes a fixed track**: min to max and back, 4 s each way, linear, from scene
   time 0.
9. **`+ row`**: a source is a term of a 2D field (`source(pos=..., strength=1,
   core=0.1)`), refused in 3D; an obstacle is listed with why it is not offered
   (collisions are out of v1, EXPLORER.md). The palette leaves out types with a required
   parameter or defaults that do not validate alone, and the Explorer's row types; only
   flow sets and tracers have drag handles (any type's handles are Milestone 32).
10. **Export renders a fork** through a private `source=` on `Workspace._render_job` (not a
    tool; the AI's `render_start` is unchanged).
11. **Equilibria and handles are hit by projecting their points**, not by the GL id pass,
    so clicks work without OpenGL (and on Windows CI).
12. **The equilibrium card names points `O`, `C+`, `C-`** (the origin; the sign of the
    first coordinate that is not zero; a number when several share one), not the owner's
    own names: a rule that does not move with a slider (audit S4).
13. **"Who changed what" for an attached AI's `explorer_edit` now says what it did**
    (`rho = 24.74`) instead of the tool's name; other tools are named as before.
14. **3D pan is ctrl+drag or the middle button** (shift+drag draws a ball).
15. **A frame-timing readout** (`perf`) was added to the transport line for the owner's
    feel check; it reads `time.monotonic` in the window only to report (contract 3's
    clock rule: nothing it measures reaches a scene's state; on Windows that clock is coarse,
    about 16 ms, so its numbers are rough there).
16. **While the person types in a row, an attached AI's tools that would change it are
    refused or taken back** (audit B2), not only `explorer_edit`. A tool without its own
    check (a verb, a look) is applied and then taken back if it touched the row; it leaves
    nothing to redo. The person's own edits are never refused.
17. **"Keep" and recording remove the slider of a value they animate** (audit B1), in the
    same patch; clearing the track brings a slider back (range from the variable, step
    0.01).
18. **A flow object from the palette is placed after it is computed** on the analysis
    thread (audit S6): `place` returns `pending` and the commit follows on a tick.

## Not built

- The advice's on-scene overlays as object params (the system in LaTeX top right, `phi^t`
  labels on tracers, the speed readout in the picture): the speed and frame times are in
  the panel's transport line only. A `Probe`'s `label` and a `System`'s `label` exist and
  can be set from the Inspector.
- The faint plane grid shown while hovering in 3D (where a click will drop a tracer); the
  hover position is kept (`Explorer.hover`) and the status bar shows the coordinates, but
  nothing is drawn.
- A box for a set's `count` (and other parameters of the Explorer's objects) in the panel:
  the Inspector (with the Explorer off, or the AI) sets them.
- Drag handles for every placed type (Milestone 32), arrangements (33).
- The unused variable's dimmed row with a remove button (advice 7.4): an unused variable
  shows as any other row; delete removes it.

## Independent audit (2026-11-25) and what was fixed

The auditor's probes are in the session scratchpad (`probes/`). Every fix has a regression
test in `tests/test_explorer_panel_audit.py` unless named. Checked by running the new tests
against a copy of the tree at the merge commit 0c04928 (before any fix): 21 of the 21 new
tests fail there, plus the two new window tests (`test_a_commit_keeps_the_cursor_and_the_line_edit`,
`test_closing_the_window_commits_a_wheel_zoom_at_rest`); all pass after.

| # | Finding | Before | Fix | Test |
| --- | --- | --- | --- | --- |
| B1 | "keep" and recording slider values left the slider's Control on an animated value | lint error `orphan_control`; Convert to script then failed (`bad_control`) | the same patch removes the Control (`remove_control`), so undo brings both back; clearing the track adds the slider back | `test_b1_keep_takes_the_slider_away...`, `test_b1_record_then_convert_runs_and_the_script_is_the_document` (record, convert, the script's document equals the explorer's), `test_b1_keep_then_convert` |
| B2 | an attached AI's raw `undo`, `redo`, `scene_edit` could remove or rewrite the row the person typed in; the typed text was dropped and the lock leaked | ok, the row gone, `ws._drafts` left locked; the person's commit "no row" | `scene_edit` checks its ops (`_locked_by`), `undo`/`redo`/`restore` the document they would switch to (`Engine.undo_document`/`redo_document`), `scene_new`/`scene_open` wait; after any other tool the remote server compares the locked rows (their object and the tracks on it) and takes the tool's steps back (`Engine.forget_redo`); all only for an AI (the person's own edits are theirs). In the panel: a row gone under a draft lets go of its lock and keeps the typed text in the new row (never dropped); locks nobody types in are let go each tick | `test_b2_an_ai_undo_redo_or_restore...`, `..._redo_cannot_change...`, `..._raw_scene_edit...`, `test_b2_other_ai_tools_are_taken_back...` (through the real remote server: a verb on the locked row's value), `test_b2_a_row_gone_under_the_draft_keeps_the_text...`, `..._noticed_at_the_next_tick` |
| S1 | a typed row with a kind chosen on the badge committed as 2 patches under 1 name; the draft stayed under a dead key | 2 undo steps, focus lost | one `explorer_edit` with text and kind | `test_s1_text_and_kind_commit_as_one_patch` |
| S2 | a kind override on a row that becomes a System did not read back (`field:0`, `ode`), so each leave committed again (new ids) | 2 undo steps per leave | the override is kept on one-row objects and read back (`object_kind`, `rows_of`); leave commits only a kind chosen and different | `test_s2_a_kind_override_on_a_field_row_reads_back_and_leave_is_quiet` |
| S3 | every debounce commit rebuilt the row's widget: the cursor jumped to the end, the line edit's undo was lost | cursor 9 for 7 | the row typed in keeps its widget when its kind is the same; otherwise the cursor and selection are carried | `test_explorer_window.py::test_a_commit_keeps_the_cursor_and_the_line_edit` |
| S4 | the equilibrium card went stale (kind and eigenvalues after a slider moved); names `E1..En` were positions | "E2 saddle" after rho = 0.5; zoom in named with the stale kind | `refresh_card` finds it again by position (within 5% of the window's diagonal) after every change: updated and said when its kind changed, closed when it is gone; zoom in uses fresh numbers; names `O`, `C+`, `C-` (`equilibrium_names`) | `test_s4_the_card_follows...`, `test_s4_names_do_not_move_with_a_slider` |
| S5 | the flow memos (`_PATHS`, `_TRAILS`, `_EQ`, `_MAN`) had no lock | `KeyError` from three threads | `flow._LOCK` and `_memo` (read and store under the lock, compute outside), as `flow_analysis` | `test_s5_the_flow_memos_are_safe_under_threads` |
| S6 | placing a limit cycle blocked the GUI thread 2-2.5 s | 2.8 s in `place` | a flow object is computed on a fork on the analysis thread (its memos warm), then committed (`placing()` says what is being computed) | `test_s6_a_slow_flow_object_is_computed_off_the_gui_thread` |
| S7 | CI never ran the window tests | `--extra gui` missing | done by the Milestone 25 round (`ci.yml` installs `--extra gui`, `QT_QPA_PLATFORM=offscreen`); the Explorer window tests need only `importorskip("PySide6")` and skip only their GL tests without OpenGL | none needed |
| nits | export took over the editor's job; marks unbounded and backwards marks silently retimed; the duration stretched unsaid; 9 palette types draw nothing alone; `rows()` read `_drafts` without the lock; private `Editor` methods; closing did not settle; two convert paths; overwrite decided in the window; EXPLORER.md row 30; "no sleeps"; a multi-line paste was one error row | | the export is its own job (`export_job`, followed in the status bar, cancelled on close); at most 200 marks; a backwards mark and the new duration are said; `PALETTE_HIDDEN` with reasons, manifolds placed near the window's centre; the lock; public `Editor.edit`, `camera_params`, `set_camera`, `render_engine`; closing settles; one convert path (`prepare_convert`); `needs_replace`; the table fixed; the wording fixed; a paste is a row each in one patch | `test_nit_*` (6), `test_closing_the_window_commits_a_wheel_zoom_at_rest` |

Deferred nits: marks are capped, not made O(n) (stopping 200 marks is one patch built in one
pass; the apply itself was not profiled); `bar_chart` and the eight content types are left
out of the palette rather than given placement defaults.

## Found, not fixed (outside this milestone's surface)

- `test_app_window.py`'s first test assumed the code panel is the splitter's first widget;
  it now checks the stack shows the script.

## What only the PC can judge (`pc_tasks/14_explorer_feel.md`)

Frames a second while orbiting with 1e4 and 1e5 points, input-to-picture latency, whether
the half-size picture while dragging looks acceptable and the full one arrives soon enough,
whether the 0.2 s / 1 s commit pauses feel right while typing, whether handles stay under
the mouse, the zoom-in flight, and everything that "feels wrong". On this box (llvmpipe)
a 640x360 Lorenz frame takes tens of milliseconds; no number here says anything about the
4070 Ti.

## Untested

Windows (no OpenGL in its CI; the Explorer code uses no POSIX-only call, reads and writes
files as UTF-8, and its tests use an injected clock and no timing asserts; the GL export
test waits for its render job with short sleeps and a generous bound, and the S5 thread test
lowers the switch interval),
macOS, a real GPU (CUDA flow kernels are Milestone 31), a real screen (only Qt offscreen),
the packaged app.

## Test runs here

Linux, 4 shared cores, llvmpipe OpenGL 4.5, no CUDA, Qt offscreen; one suite at a time.
First pass (all suites, on the code before the last fixes): test_explorer_panel 67,
test_explorer_window 14, test_gui_controller 20, test_gui_window 13, test_app_window 21,
test_app_code_first 19, test_app_audit 28, test_desktop 42, test_explorer_rows 60,
test_explorer_tools 13, test_explorer_audit 14, test_mcp 39, test_remote 7, test_flow 69,
test_batches 52, test_layers 9, test_contracts 8 + 1 failed (the frame timing read
`perf_counter`, which contract 3's clock rule forbids outside its list: it now reads
`time.monotonic`), test_kit 11, test_docs 5, test_render_jobs 6, test_final `-k "not slow"`
507 passed, 8 skipped. After the last fixes, the touched suites again: test_explorer_panel
67, test_explorer_window 17 (the GL export test ran), test_desktop 42, test_app_window 21,
test_gui_window 13, test_contracts 9, test_layers 9, test_docs `-k API` 1. `ruff check`,
`ruff format --check` and `lint-imports` (2 kept) clean. The whole suite was not run; the
`phasor-milestone-gate` skill was not run (not gated).

After the merge with the Milestone 25 second fix round (0f43079) and the audit fixes, every
suite again, one at a time: test_explorer_panel 67, test_explorer_panel_audit 20,
test_explorer_window 19 (its GL export test ran), test_explorer_rows 60, test_explorer_tools
13, test_explorer_audit 14, test_gui_controller 20, test_gui_window 13, test_app_window 36,
test_app_code_first 19, test_app_audit 42, test_desktop 42, test_tweaks_block 6,
test_render_jobs 6, test_open_contract 71, test_mcp 39, test_remote 7, test_flow 69,
test_flow_analysis 78, test_batches 52, test_layers 9, test_contracts 9, test_kit 11,
test_docs 5, test_final `-k "not slow"` 507 passed, 8 skipped. Right after the merge (before the
fixes) the Milestone 25 suites passed as well (test_desktop's two failures in that run came
from a half-written edit of `remote.py` made while it ran; rerun: passed). `ruff check`,
`ruff format --check` and `lint-imports` clean. Not run: the whole suite, the gate skill.
