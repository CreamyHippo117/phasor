# phasor patterns

Idioms for common tasks, one canonical way each. Filled in as milestones land:

- Label a pole — Milestone 2
- Animate a contour integral — Milestone 3
- Sync to narration — Milestone 4
- Orbit a 3D camera — Milestone 6

## Check what this machine can do before rendering

```python
from phasor.render.doctor import run_doctor

if run_doctor().tier == "C":
    print("No GPU rendering here: use inspect and lint only.")
```
