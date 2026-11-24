---
name: phasor
description: Write, check and fix math animations with the phasor Python library and its MCP tools. Use when asked to make a math animation, explain a concept visually (complex analysis, domain coloring, fractals, Fourier series, 3D surfaces), or fix a phasor scene that looks wrong.
---

# phasor

phasor is a Python library for math animations you can check. A scene is a Python script
(`scenes/NAME.py`) that builds one `Scene`; the script is the source that you and the
person both edit. The `phasor` MCP tools (when connected) let you see what it renders.

## Do this

1. Find the closest example (`kit/examples/INDEX.md` in a kit folder, `examples/`
   otherwise) and start from its structure. Look up names in the API docs (`docs_search`,
   or `kit/docs/API.md`); call `object_types` for exact parameters and style roles.
2. Write or edit the script. Name objects by meaning (`pole_i`, `loop.label`). Labels are
   `<owner>.label` with a `label(x) near x` constraint. Style takes roles; type a color or
   size only when the person asks for one (review flags it if it is off the theme).
   Positions are world coordinates (x = Re, y = Im).
3. For a narrated scene: narration with `[[markers]]`, and every motion anchored to a
   marker with a verb (`reveal`, `trace`, `dim_others`, `morph`, ...).
   One thing becoming another: `transform(a, b)` morphs any outline (path, shape, text,
   SVG) into another; between two equations drawn as pictures it glides equal parts and
   crossfades the rest. To rearrange a formula so each symbol travels to its new place,
   draw both equations `as_shapes=True` and use `transform` or `transform_matching`
   (matching pieces move, the rest fade; also for text into text).
4. Check: `scene_open` the script (again after every edit), then `review`: lint, what is on
   screen at each marker, and a storyboard picture. Without the tools: `phasor lint SCENE
   --aesthetic`, then `phasor storyboard SCENE --marks` and read the picture. Fix each item
   in the script and check again. At most 5 rounds, then report what is left.
5. The beauty pass: `review` with `aesthetic: true`, or `critique`. Look at the storyboard
   against `STYLE.md` and the person's `TASTE.md`.
6. Render a draft to check the motion quickly: `render` with `preset: "draft"` (or
   `phasor preview SCENE`; small, 15 fps, plain look). Then a preview with the full look:
   `render` with `preset: "preview"` (or `phasor render SCENE`).

## Never

- Type label positions. Use constraints. Prefer roles to typed colors and sizes, and never
  snap a typed value to a role (`polish` with `snap`) without asking.
- Animate by raw seconds in a narrated scene. Use markers and verbs.
- Claim it looks good without having looked at the storyboard.
- Reach for a rainbow everywhere: the full wheel is for where hue means phase (domain
  coloring); elsewhere vary the palette. Dark is the default theme; do not switch it unasked.
- Run a script someone else wrote without the person saying it is trusted: a script is code.
