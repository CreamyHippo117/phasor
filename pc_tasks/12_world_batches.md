# 12. World-space batches (Milestone 28): frame rates on the RTX 4070 Ti

Milestone 28 moved the flow objects onto `ctx.points3d/lines3d/arrows3d`: the arrays go to
the card once and the vertex shader faces them to the camera. The cloud box (4 shared
cores, llvmpipe: OpenGL drawn by the CPU) measured the table below; **none of the PC column
has been run**. This task fills it in. Write results at the bottom; do not change code
unless a step below fails, and then report rather than fix.

Branch: the Milestone 28 branch (or `claude/eloquent-mayer-jiqxs9` once merged). Folder:
`C:\Users\Atlas\phasor-dev`. Close other heavy programs (games, browsers playing video):
the numbers are wall clock.

## A. Setup and the correctness checks (5 minutes)

```powershell
git pull
uv sync --extra gl --extra encode --extra gui --extra mcp --extra gpu --extra boolean --extra svg --extra code
uv run phasor doctor
uv run pytest -q tests/test_batches.py tests/test_flow.py
```

Record: pass/fail counts; the GL renderer `phasor doctor` names (it must be the NVIDIA card,
not "Microsoft Basic Render Driver" or llvmpipe). If any test in `test_batches.py` fails,
stop and paste the failure: in particular `test_the_vertex_shader_equals_the_cpu_expansion`
and `test_gl_and_cpu_expanded_pictures_agree` compare the NVIDIA shader with the NumPy
reference, which no machine but the cloud box has checked.

Then the goldens (they are kept per renderer class; the GPU class has no flow golden yet):

```powershell
uv run pytest -q tests/test_flow.py -k golden          # skips: no gpu golden yet
$env:PHASOR_UPDATE_GOLDEN = "1"; uv run pytest -q tests/test_flow.py -k golden; Remove-Item Env:PHASOR_UPDATE_GOLDEN
```

Open `tests/golden/gpu/lorenz_explorer.png` and look at it next to
`tests/golden/software/lorenz_explorer.png`: the same three frames (cones, red balls, the
ball of points folding onto the wings, the C+ close-up). Commit it only if it looks right.

## B. The frame-rate table (10-20 minutes)

```powershell
uv run phasor bench --explorer --width 640 --height 360 --count 60
uv run phasor bench --explorer --width 1920 --height 1080 --count 30
```

Each prints one line per case, way and motion, and saves
`benchmarks/<date>_explorer_view_<backend>_<w>x<h>.json`. Commit both JSON files.

What the columns mean:

- **shader, full**: the Milestone 28 path (batches expanded in the vertex shader).
- **cpu, full**: the same triangles built in NumPy every frame and uploaded (what strokes cost:
  the "before").
- **shader, interactive**: `quality="interactive"`, the hint the Explorer passes while the
  mouse is down (at most 4000 arrows, 5000 points of a set).
- **orbit / pan**: only the camera moves (expect `uploads/frame 0.0` everywhere).
- **flow**: the flow time advances (expect one upload per moving batch: 1 for the 10^5
  case, 2 for the Lorenz case with trails, 4 for the 2D case).

Every frame is the draft look (no bloom, lens or fog) read back as sRGB bytes, which is
what the Explorer's viewport will show (advice 1.4).

## C. Where the time goes (5 minutes)

```powershell
uv run phasor bench --frames lorenz_explorer --preset draft --count 60
uv run phasor bench --frames pendulum_flow --preset draft --count 60
```

Record the `draw` and `gpu draw` lines: `draw` is Python (the objects' `draw()`), `gpu draw`
the card. On the cloud box llvmpipe's rasterising dominates; on the RTX card we expect
Python to dominate, which says what Milestone 29/31 should cache next.

## D. Template: expected against measured

Expected numbers are the advice's estimates (`docs/decisions/explorer_advice_2026-11-24.md`
section 1.5), not measurements. Cloud numbers: this box on 2026-11-24, 640x360, 30 frames,
load average 3.1-3.9 on 4 cores (`benchmarks/2026-11-24_explorer_view_cpu_640x360.json`).

| Case (640x360) | Motion | Cloud, shader (llvmpipe) | Cloud, cpu-built | Cloud, interactive | Advice expects on PC | PC shader | PC cpu-built | PC interactive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D portrait, strokes (today's path) | pan | 13.9 fps | (same path) | - | 15-20 fps | | | |
| 2D field, 2500 arrows + 2000 points | pan | 23.4 fps | 19.7 fps | 20.2 fps | 60+ fps | | | |
| 2D field, 2500 arrows + 2000 points | flow | 19.3 fps | 19.2 fps | 19.2 fps | 60+ fps | | | |
| 3D Lorenz, 4913 cones + 10^4 tracers, 8-step trails | orbit | 4.2 fps | 4.2 fps | 7.7 fps | 60+ fps | | | |
| 3D Lorenz, 4913 cones + 10^4 tracers, 8-step trails | flow | 3.1 fps | 2.8 fps | 4.4 fps | 60+ fps | | | |
| 3D Lorenz, 343 cones + 10^5 tracers | orbit | 3.5 fps | 3.0 fps | 10.9 fps | 60 fps plausible | | | |
| 3D Lorenz, 343 cones + 10^5 tracers | flow | 2.9 fps | 2.4 fps | 6.2 fps | needs the CUDA flow kernel (M31) | | | |

Also record for 1920x1080 (the second command) in the same shape.

## Results

(Write here: date, commit, `phasor doctor`'s GPU and driver lines, A's test counts, the
table above filled in for 640x360 and 1920x1080, C's `draw` and `gpu draw` per scene, and
anything that looked wrong in the GPU golden.)
