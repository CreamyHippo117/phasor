# phasor API (for AI agents)

One canonical example per public name. No deprecated aliases. Every ```python block
in this folder runs in CI (`tests/test_docs.py`); mark a block ```python no-run only
when it truly cannot run in CI (e.g. needs a GPU) and say why next to it.

Status: Milestone 0. The scene API (`Scene`, objects, timeline) arrives in Milestone 1.

## Command line

| Command | Status | Purpose |
| --- | --- | --- |
| `phasor doctor [--json] [--require A\|B\|C]` | M0 | Prove CUDA, NVRTC, OpenGL 4.3 and encoders work; print hardware tier |
| `phasor inspect` | M1 | JSON snapshot of a scene at time t |
| `phasor lint` / `phasor storyboard` | M2 | Linter and keyframe contact sheet |
| `phasor render` / `phasor bench` | M3 | Final render, kernel benchmarks |
| `phasor mcp` | M5 | MCP server |
| `phasor gui` | M10 | Editor |

Unimplemented commands exit with code 2 and name their milestone.

## `phasor.render.doctor`

```python
from phasor.render.doctor import run_doctor

report = run_doctor()  # never raises; each probe fails on its own
assert report.tier in {"A", "B", "C"}
opengl = report.get("opengl")  # Check(name, ok, detail, data)
print(report.tier, report.encoder, opengl.ok if opengl else None)
```

Tiers: **A** CUDA kernels and GL 4.3 both work. **B** GL 4.3 only (kernels fall back to
NumPy). **C** no usable GL: inspector (analytic only), lint and timeline still work.

## `phasor.render.glcontext`

The only way to create an OpenGL context. Do not call `moderngl.create_context` yourself.

```python no-run
# no-run: needs OpenGL 4.3 (the gl-marked tests cover it)
from phasor.render.glcontext import create_standalone_context

ctx = create_standalone_context()  # GL >= 4.3, headless; EGL on Linux without a display
```
