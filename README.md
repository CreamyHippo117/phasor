# phasor

GPU-rendered math animations that an AI can see, check and fix. Working name.

Design: [SPEC.md](SPEC.md). Rules for contributors and agents: [CLAUDE.md](CLAUDE.md).
Status: every milestone on the roadmap is built (see the `MILESTONE_*.md` files); the Windows and
NVIDIA/CUDA sides are not tested yet. License: MIT ([LICENSE](LICENSE)).
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

## The editor, sliders, and Manim scenes

```bash
uv sync --extra gl --extra encode --extra mcp --extra gui --extra notebook
uv run phasor gui examples/basics_3d.py          # the desktop editor (also: phasor-gui)
uv run phasor gui examples/julia_explorer.py     # Sliders tab: move c, watch the Julia set change
uv run phasor web examples/julia_explorer.py -o julia.html --steps 7   # one offline web page
uv run phasor manim examples/manim_reverb.py -o reverb.phasor.json     # a Manim scene, imported
```

The editor shows the scene's picture, timeline, inspector, lint, looks, sliders and its code
(Python, runnable). While it is open, `phasor mcp --attach --root <its folder>` lets an AI edit
the same live scene. Math text (`$...$`) is typeset by default with a built-in engine; set
`PHASOR_MATH=latex` to use a real TeX (`latex` and `dvipng` on the PATH).
