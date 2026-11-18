# phasor

GPU-rendered math animations that an AI can see, check and fix. Working name.

Design: [SPEC.md](SPEC.md). Rules for contributors and agents: [CLAUDE.md](CLAUDE.md).
Status: Milestone 5 (MCP server + AI docs). See [MILESTONE_5.md](MILESTONE_5.md).
API for humans and agents: [docs/ai/API.md](docs/ai/API.md). Driving it from an AI: [docs/ai/MCP.md](docs/ai/MCP.md).

```bash
uv sync --extra gl --extra encode --extra gpu   # drop --extra gpu without an NVIDIA GPU
uv run phasor doctor
uv run python examples/moving_dot.py
uv run phasor inspect examples/moving_dot.py --t 2 --detail brief
```

## Let an AI build scenes (MCP)

```bash
uv sync --extra gl --extra encode --extra mcp
uv run phasor mcp --root ./scenes          # an MCP server on stdio; scenes stay inside ./scenes
```

Point an MCP client at that command (for example `claude mcp add phasor -- uv run phasor mcp
--root ./scenes`). Tested with the MCP SDK's own stdio client on Linux; not yet tried with
Claude Desktop, Claude Code's client or on Windows. Give the AI `skills/phasor/SKILL.md`
(or `llms.txt`) and it knows the loop: build, `review`, fix, repeat. `.py` scenes are code,
so the server refuses them unless you add `--allow-scripts`.
