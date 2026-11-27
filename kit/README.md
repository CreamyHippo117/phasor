# The AI kit

A folder where an AI writes phasor scenes and knows how, set up by `phasor kit [FOLDER]`
(default: `Documents/phasor`, the desktop app's scenes folder). Start Claude Code there and
type `/scene-plan <your idea>` to plan a video (taste first, no code), or
`/scene-shots <your script>` when you already know what the video says and want only the
pictures planned; then `/scene-builder plans/NAME.md` to code the plan.

This directory is the kit's single source; `phasor kit` copies it, with `docs/ai/`,
`STYLE.md`, `skills/phasor/` and the sorted `examples/`, into the folder. Run it again after
changing phasor to update the folder: everything the kit owns is replaced, and what is
yours (`TASTE.md`, `WISHLIST.md`, `scenes/`, `plans/`, other MCP servers in `.mcp.json`) is kept.

| File | In the folder | What it is for |
| --- | --- | --- |
| `CLAUDE.md` | `CLAUDE.md` | read at the start of every session: the loop, the rules, the commands for this machine |
| `TASTE.md` | `TASTE.md` (yours) | the person's taste; imported into every session; the AI adds to it |
| `WISHLIST.md` | `WISHLIST.md` (yours) | what scenes needed that phasor could not do |
| `skills/scene-plan/SKILL.md` | `.claude/skills/scene-plan/` | the `/scene-plan` command: an idea becomes a plan for a beautiful teaching video (wide brainstorm with many subagents, pruned to one path); writes `plans/NAME.md`, no code |
| `skills/scene-shots/SKILL.md` | `.claude/skills/scene-shots/` | the `/scene-shots` command: the person brings the story (script, outline, notes, or a finished plan to reshoot); it designs how each beat is seen, as a cinematographer works from a finished screenplay (breakdown, one world from four treatments including a phasor expert's, each sketched as a still, four shot lists, continuity; `TASTE.md` in every brief); writes `plans/NAME.md` in the same shape, no code |
| `skills/scene-builder/SKILL.md` | `.claude/skills/scene-builder/` | the `/scene-builder` command: codes an approved plan as faithfully as phasor allows; script, checks, storyboard, preview; what phasor cannot do goes to `WISHLIST.md` |
| `examples.toml` | `kit/examples/<kind>/`, `kit/examples/INDEX.md` | every example sorted by kind and tier |

## Adding an example

Put the script in `examples/`, then give it an entry in `examples.toml` (`kind`, `tier`,
`teaches`); `tests/test_kit.py` fails until it has one. A `model` example must pass
`phasor lint --aesthetic --strict` with nothing left, and the test checks that too.

The current examples are placeholders (the milestone demos). The plan (owner, 2026-11-22):
beautification first, from the owner's reference images, then great examples replace them.

## Standard prompts

After a change to phasor or to the kit, run these in a fresh kit folder, one session each,
first `/scene-plan` with the prompt, then `/scene-builder` on the plan it wrote, and check
every result: the plan follows `TASTE.md`; lint clean with `--aesthetic` (or each warning
left on purpose is named), the storyboard looked at, a preview rendered, the report honest
about what is missing.

1. `/scene-plan the residue theorem: two poles, a contour around them, the integral as the sum of residues`
2. `/scene-plan a Fourier series building a square wave, term by term, with the Gibbs overshoot`
3. `/scene-plan the Riemann surface of the square root, with the camera turning once around it`
4. `/scene-plan a slow zoom into the Mandelbrot set, ending on a spiral`
5. `/scene-plan one idea, two objects: a dot moving along a circle and its shadow on the x-axis`
6. `/scene-shots introduce LTI systems: a clap in a room; the room unfolds into its mirror rooms, each mirror room an echo; so what you hear is a sum of delayed, scaled copies of the clap; that is linearity and time invariance, and the room is its impulse response` (check also that every beat of the source is on screen, in order, with nothing added)
