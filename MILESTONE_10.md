# Milestone 10: GUI editor

Roadmap row: viewport, timeline, inspector, lint panel, MCP attach, variant picker. Exit
demo: edit a scene by hand and by AI in one session (the row named the reverb scene, which
Milestone 8 skipped by decision; the owner's prism scene stands in).

Status: **built; gates green on Linux (software OpenGL, Qt offscreen).** Untested on Windows,
on a real GPU, and with a real display (every window test runs on Qt's offscreen platform).

## Built

| Area | Where | What |
| --- | --- | --- |
| Toolkit decision | `pyproject.toml` (`gui` extra) | PySide6-Essentials: testable headless, mature on Windows, no web engine pulled in |
| Shared tools | `phasor/workspace/` (moved from `mcp/`) | `Workspace` and docs search, now below both clients (layer added to lint-imports) |
| One live scene, two editors | `workspace/remote.py` | localhost JSON-RPC, token in `<root>/.phasor/gui.json` (mode 600, removed on close), only public tools callable, calls serialized with the editor's lock; `RemoteWorkspace` has every tool with the same signature and docstring |
| `phasor mcp --attach` | `mcp/server.py`, `mcp/cli.py` | the same MCP server, built on the proxy |
| Editor logic | `gui/controller.py` (`Editor`, no Qt) | playhead and play, selection by hit test, drag in the object's plane, inspector (document values, animated ones at the playhead), typed edits as JSON or text, style roles, lint list and jump, apply a look (one undo step), variants and picks recorded as the person's, history of the person's edits, `changed()` for edits made by the AI |
| Timeline data | `Engine.timeline()` | markers and resolved spans per animated parameter |
| Window | `gui/window.py` | viewport (engine frames, click/drag, selection box), timeline (lanes, scrub), inspector/lint/looks tabs, toolbar (play, undo, redo, save), 30 Hz tick, dark styling |
| Command | `gui/cli.py`, `phasor gui [SCENE] [--root] [--no-ai]` | registered through `phasor.commands`; opening a `.py` scene is allowed to the person, never to the attached AI |
| Docs search | `workspace/docs.py` | ranks by word rarity, caps repeats (a long section repeating common words buried precise answers) |

## Verification

- `uv run pytest`: **710 passed, 24 skipped** (after regenerating llms.txt) (22 need CUDA; one empty parametrization now
  that no command is "planned"; one GPU doctor check); ruff, ruff format and lint-imports clean.
- **Exit demo** (`tests/test_m10_exit_demo.py`): the editor opens `prism_primes.py`; the
  person changes `truth.hue_along` in the inspector field and drags an equation; a real
  `phasor mcp --attach` subprocess (MCP over stdio) inspects the curve (sees 1.5), edits a
  style and a position and lints; the window's next tick shows the AI's edit in the
  inspector; saving writes both editors' tweaks to `prism_primes.overrides.json`; re-running
  the script keeps all of them (contract 2); closing withdraws the handoff file.
- Remote (`test_remote.py`): edits and reads go to the host's engine; refused patches report
  and change nothing; tool errors arrive as `ToolError`; the proxy's signatures and docstrings
  equal `Workspace`'s; an attached MCP server lists exactly the same tool schemas; wrong
  token, private names and `engine` are refused; the socket only binds locally; ten AI and
  ten GUI edits at once all land.
- Controller (`test_gui_controller.py`) and window (`test_gui_window.py`): every panel, typed
  edits (and refusals shown in the status bar), style menus, lint jump, scrub and play, click
  selection (GL), an attached AI's edit appearing, variants and a person's pick.
- Looked at the window with the prism scene open (offscreen screenshot) and fixed what
  showed: the timeline took half the height, lane names overlapped their bars, the inspector
  showed life's wiggle instead of the document's value, and Qt's light default clashed with
  the house look.

## Deviations from SPEC (SPEC updated)

1. Exit demo on the prism scene (no reverb scene exists).
2. The viewport shows engine frames as an image, not an embedded GL widget: what you see is
   the final look, at the cost of frame rate on software GL.
3. Not built: 3D gizmos and orbit/pan of the view (the camera is a document object, edited in
   the inspector), audio waveform and narration text in the timeline, dragging spans to
   retime, the AI chat dock (an MCP client attaches instead), the two-way code view, the
   sketch layer (Milestone 11 dropped by the owner).
4. One undo history for both editors: `undo` takes back the newest edit, whoever made it.
5. The tool logic moved from `mcp/tools.py` to `workspace/tools.py` (CLAUDE.md updated).

## Bugs caught while building

1. `phasor mcp --attach` crashed at startup: the CLI's placeholder `gui` command clashed with
   the real one registering itself (found by the exit demo's real subprocess).
2. A `float` life channel edit had silently not applied (the formatter had rewrapped the
   anchor); the prism tests caught it before this milestone.
3. Docs search returned the object-types table instead of `dim_others` once the docs grew.
4. The AI could not read a parameter's value at all (`inspect` reports what is on screen, not
   what was set), so it could not see a person's edit: `inspect(detail="full")` now gives each
   object's document `params`. Found by writing the exit demo's assertion.

## Untested

- Windows, a real display, a real GPU. Frame rate on software GL is a few frames per second
  for busy scenes.

## After the roadmap: the desktop app (owner's request, 2026-11-19)

`phasor-gui` became a program: start screen, menus, recent files, several windows, unsaved
prompts, Save As, export with progress (a separate process), settings, an icon, an
applications-menu shortcut, and a PyInstaller build (`packaging/README.md`). Built and run on
Linux (the packaged app renders, lints and opens its window); Windows and macOS builds are
untested. Tests: `tests/test_desktop.py` (settings, unsaved changes, exports with progress,
cancel and narration audio, the start screen, templates, examples opened as copies, closing
prompts, quit, shortcuts, the math setting, menus, the export dialog, launch commands) and
`tests/test_packaging.py` (the build itself is opt-in: `PHASOR_TEST_PACKAGE=1`; it builds the
app and runs every command, a render with typeset maths, and the window). Bug they caught: an
export to a folder that cannot be made left a hidden copy of the scene behind.
