---
name: phasor
description: Build, check and fix math animations with the phasor MCP server. Use when asked to make a math animation, explain a concept visually (complex analysis, domain coloring, fractals, poles and residues), or fix a phasor scene that looks wrong.
---

# phasor

phasor renders math animations and lets you see, check and fix them. You work through the
`phasor` MCP server (`phasor mcp`); you do not write rendering code.

## Do this

1. Read `docs/ai/MCP.md` (or call `docs_search`). Call `object_types` for exact type names,
   parameters and style roles.
2. `scene_new`, then `scene_edit` patches. Name objects by meaning (`pole_i`, `loop.label`).
   Labels are `<owner>.label` with a `label(x) near x` constraint. Style takes roles, never
   raw colors. Positions are world coordinates (x = Re, y = Im).
3. For a narrated scene: `narrate` a script with `[[markers]]`, then `animate` verbs
   (`reveal`, `trace`, `dim_others`, `morph`, ...) anchored to the markers.
4. `review` after every batch of edits. Fix each `todo`, review again. Look at the storyboard
   image: lint cannot tell you a layout is ugly. At most 5 rounds, then report what is left.
5. The beauty pass: `review` with `aesthetic: true`, or `critique` (score, issues, rubric).
   Look at the storyboard against `STYLE.md`, `polish` what is certain, decide the rest.
   Unsure? `variants` on a marker, then `choose`. Start scenes from a look
   (`scene_new` with `look: "calm"`), and use `preset` in patches.
6. `scene_save`, then `render` with `preset: "preview"`.

## Never

- Type label positions, raw colors or pixel sizes. Use constraints and roles.
- Animate by raw seconds in a narrated scene. Use markers and verbs.
- Reuse the id of an object you removed.
- Claim it looks good without having looked at the storyboard.
- Reach for a rainbow everywhere: the full wheel is for where hue means phase (domain
  coloring); elsewhere vary the palette. Dark is the default theme; do not switch it unasked.
- Open `.py` scenes unless the user says to trust them: a script is code.
