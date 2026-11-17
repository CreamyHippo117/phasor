# phasor

GPU-rendered math animations that an AI can see, check and fix. Working name.

Design: [SPEC.md](SPEC.md). Rules for contributors and agents: [CLAUDE.md](CLAUDE.md).
Status: Milestone 0 (skeleton + `phasor doctor`). See [MILESTONE_0.md](MILESTONE_0.md).

```bash
uv sync --extra gl --extra encode --extra gpu   # drop --extra gpu without an NVIDIA GPU
uv run phasor doctor
```
