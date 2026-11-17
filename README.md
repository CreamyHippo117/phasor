# phasor

GPU-rendered math animations that an AI can see, check and fix. Working name.

Design: [SPEC.md](SPEC.md). Rules for contributors and agents: [CLAUDE.md](CLAUDE.md).
Status: Milestone 2 (layout + linter). See [MILESTONE_2.md](MILESTONE_2.md).
API for humans and agents: [docs/ai/API.md](docs/ai/API.md).

```bash
uv sync --extra gl --extra encode --extra gpu   # drop --extra gpu without an NVIDIA GPU
uv run phasor doctor
uv run python examples/moving_dot.py
uv run phasor inspect examples/moving_dot.py --t 2 --detail brief
```
