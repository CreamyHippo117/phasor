# Milestone 5: MCP server + AI docs

Roadmap row: MCP server, detail levels, `docs/ai/`, self-review loop, skill package.
Exit demo: **an AI builds and fixes a scene from a one-line prompt.**

Status: **built; gates green on Linux.** Not tried with a real MCP client app or on Windows.

## Built

| Area | Where | What |
| --- | --- | --- |
| Tool logic | `mcp/tools.py` (`Workspace`) | every tool as plain Python: scene_new/open/save/edit, animate, narrate, undo/redo, checkpoint/restore/diff, inspect (brief/standard/full), lint (+fix), review, storyboard, hit_test, resolve, render, docs_search, object_types |
| Server | `mcp/server.py` | names the tools for MCP; errors come back as readable JSON; `review`/`storyboard` return the picture as an image |
| Command | `mcp/cli.py` | `phasor mcp --root DIR [--out DIR] [--allow-scripts]`, registered via the `phasor.commands` entry point |
| Diff | `core/diff.py` | structured document diff (objects, params, markers, tracks, constraints, meta) |
| Render job | `render/job.py` | the render loop, shared by `phasor render` and the MCP `render` tool |
| Docs for AI | `docs/ai/MCP.md`, `llms.txt`, `skills/phasor/SKILL.md` | the loop, every tool and patch op; `llms.txt` is generated and tested for staleness |
| Searching docs | `mcp/docs.py` | keyword search over docs/ai sections and examples; docs ship in the wheel |
| Exit demo | `examples/ai_session/` | `session.json` (the tool calls) + `replay.py` |

## Verification

- `uv run pytest`: **418 passed, 16 skipped** (all need CUDA) at the last full run, then 27
  in `tests/test_mcp.py` again after the dependency pin. ruff, ruff format, lint-imports clean.
- **Exit demo, honestly described:** the one-line prompt was "explain the residue of
  1/(z^2+1) at i". I (Claude) drove the real server over stdio, looked at the storyboard,
  and fixed what I saw (off-center title, hand-placed answer text, 24-sided loop, an overlay
  that lint flagged `offscreen`). The recorded calls replay in one session in
  `test_exit_demo_...`, which asserts a lint issue appears and is resolved, the diff against
  the checkpoint, a clean final review, and strict lint on the narrated scene.
- Docs run in CI: python blocks, plus ```json patch / calls blocks in MCP.md.
- Safety tests: paths cannot leave the root, scripts refused, failed patch changes nothing.

## Bugs caught while building

1. **`review` and `storyboard` crashed over the real protocol** (the SDK tried to serialize
   the image). Unit tests had called the methods directly; only driving the server found it.
   Now a protocol-level test checks both return text + PNG.
2. `phasor mcp` in `cli` broke the layer rule (sibling clients may not import each other):
   moved to an entry-point registration.
3. Lint output listed 35 sample times per issue; trimmed to 6 plus a total for clients.
4. Lint was clean on a draft that looked wrong (picture review caught it): documented in
   MCP.md and the skill ("look at the storyboard every round").

## Deviations from SPEC (SPEC updated in the same commit)

1. More tools than the table (animate, narrate, review, undo/redo, checkpoint/restore,
   object_types); `phasor mcp` registers through an entry point.
2. No attach to a running GUI (Milestone 10); it always hosts its own headless Engine.
3. `docs_search` is keyword scoring, not embeddings.
4. ~~`restore` rebuilds the engine~~ (fixed in the Milestone 6 boundary review: checkpoints
   live in the Engine and a restore is one undoable step).
5. The MCP SDK is v2 (`mcp>=2.0,<3`); the optional extra is `mcp`.

## Untested

- Claude Desktop / Claude Code / any real MCP client app; SSE or HTTP transports; Windows.
- `render` is synchronous: a long final render blocks the server and streams no progress.
- Everything CUDA, as before.

## Open

- Fresh-eyes boundary review not run for Milestones 2-5.
- `skills/phasor/SKILL.md` is in the repo but not installed anywhere.
