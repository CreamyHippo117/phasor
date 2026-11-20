# phasor

GPU-rendered math animations that an AI can see, check and fix. Working name.

Design: [SPEC.md](SPEC.md). Rules for contributors and agents: [CLAUDE.md](CLAUDE.md).
Status: every milestone on the roadmap is built (see the `MILESTONE_*.md` files); the Windows and
NVIDIA/CUDA sides are not tested yet. License: MIT ([LICENSE](LICENSE)).
Testing plan and what is untested: [TESTING.md](TESTING.md).
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

## The editor, sliders, sketching and Manim scenes

```bash
uv sync --extra gl --extra encode --extra mcp --extra gui --extra notebook
uv run phasor-gui                                # the desktop app: a start screen
uv run phasor gui examples/basics_3d.py          # ... or straight into a scene
uv run phasor-gui --install-shortcut             # add it to the applications menu
uv run phasor gui examples/julia_explorer.py     # Sliders tab: move c, watch the Julia set change
uv run phasor web examples/julia_explorer.py -o julia.html --steps 7   # one offline web page
uv run phasor manim examples/manim_reverb.py -o reverb.phasor.json     # a Manim scene, imported
uv run python examples/sketch_contour.py         # a drawn loop becomes a contour integral (pi)
```

A packaged app (no Python needed to run it) is built with PyInstaller: see
[packaging/README.md](packaging/README.md); built and run on Linux, untested on Windows
and macOS. The editor has a start screen, menus, recent files, and File > Export for
pictures, videos and web pages. It shows the scene's picture, timeline, inspector, lint, looks, sliders, its code
(Python, runnable) and a Sketch tab: draw on the picture and strokes become proposals (a loop
over a complex function becomes a traced contour with its integral; a scribble crosses
objects out; handwriting is shown to an AI to read). While it is open, `phasor mcp --attach --root <its folder>` lets an AI edit
the same live scene. Math text (`$...$`) is typeset by default with a built-in engine; set
`PHASOR_MATH=latex` to use a real TeX (`latex` and `dvipng` on the PATH). Write maths in
either form: `phasor.to_latex` and `phasor.to_mathtext` convert both ways, and each engine
gets the form it needs.
