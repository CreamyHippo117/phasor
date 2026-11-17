# phasor patterns

Idioms for common tasks, one canonical way each. More arrive with their milestones:
label placement by constraint (M2), contour integrals (M3), narration sync (M4), 3D orbits (M6).

## Label something so the label can never be left behind

Bind the label's position to the object with an expression, then animate only the object.

```python
from phasor import Dot, Scene, Text, expr

s = Scene("label_follows")
s.add(Dot(id="pole", pos=(-2, 0), style={"color": "highlight"}, tags=["key"]))
s.add(Text(id="pole_label", text="pole", pos=expr("pole.pos + vec(0, 0.5, 0)")))
s.at(0).to("pole.pos", (2, 0), duration=2)

mid = s.engine.state_at(1.0)
assert mid["pole_label"].params.pos == (0.0, 0.5, 0.0)
```

## The self-review loop

After every edit: check the result, then look at what is on screen at each marker.

```python
engine = s.engine
result = engine.apply({"op": "set_param", "path": "pole.r", "value": 0.2})
assert result.ok, result.errors

for name, t in sorted(engine.document.timeline.markers.items()) or [("start", 0.0)]:
    for line in engine.inspect(t, detail="brief")["objects"]:
        print(name, line)

for obj in engine.inspect(1.0)["objects"]:
    hidden = obj["visible_fraction"] is not None and obj["visible_fraction"] < 0.5
    off = obj["on_screen_fraction"] < 1.0
    if hidden or off:
        print("fix:", obj["id"], obj["hidden_by"], obj["on_screen_fraction"])
```

## Screen-space titles

```python
s.add(Text(id="title", text="Residues", pos=(8 / 9, 0.1), layer="overlay", style={"size": "title"}))
assert s.engine.where("title", 0)["region"] == "upper-center"
```

Overlay coordinates are frame heights from the top-left: on a 16:9 frame x runs 0 to 16/9
and y runs 0 to 1. The overlay ignores the camera and always draws on top.

## Check what this machine can do before rendering

```python
from phasor.render.doctor import run_doctor

if run_doctor().tier == "C":
    print("No OpenGL here: inspect works in analytic mode, stills do not.")
```
