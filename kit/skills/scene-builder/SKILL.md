---
name: scene-builder
description: Build a phasor scene from a plan made by /scene-plan - code each animation of the plan as faithfully as phasor allows, check it, look at it, render a preview. Use when the person pastes a plan or points at one ("build this", "code this plan", "/scene-builder plans/NAME.md", a pasted block with Point A / Point B / Animations), or says a plan is approved. If there is no plan yet, run /scene-plan first.
argument-hint: "<plans/NAME.md, or the plan pasted in>"
---

# /scene-builder: from a plan to a finished scene

The plan: $ARGUMENTS

The thinking is already done. `/scene-plan` chose what to show, in what order, how long
each animation holds and where the voice is silent, and the person approved it. Your job
is to code that plan as well as phasor can, not to improve, shorten or re-plan it. If you
think the plan is wrong somewhere, build it as written and say so in the report.

The plan is either a file (`plans/NAME.md`) or pasted text. If pasted, save it to
`plans/NAME.md` first so the scene and its plan stay together. If what you were given is
a bare idea and not a plan (no path, no list of animations), stop and tell the person to
run `/scene-plan` on it.

`CLAUDE.md` in this folder has the commands and the rules; `TASTE.md` has the person's
taste, and it outranks everything else here.

## What "faithful" means

These come from the plan and are not yours to trade away for convenience:

- **One animation owns the stage**: full frame on midnight black, alone. Not a small
  figure in an empty frame, and nothing else sharing the frame.
- **The holds and the silences.** If the plan holds an animation for ninety seconds with a
  forty-second silent stretch, the scene does too. Long is intended.
- **How it changes while it holds**: the plan's iterations, cases, parameters and big
  moves, in its order.
- **No hard cuts**: each animation becomes the next the way the plan says.
- **What color encodes**, and the same meaning for a hue throughout.
- **Little text**: only the labels and equations the plan names.

## The steps

1. **Read the plan and list the work.** One line per animation: what fills the stage, how
   it changes, how long, how it becomes the next. Name the scene after the plan.
2. **Find the means.** For each animation, find in `kit/docs/API.md` (or `docs_search`)
   the object types, parameters, verbs and camera moves that make it; and in
   `kit/examples/INDEX.md` the closest example for its technique (copy the technique, not
   the look). Do not guess names. Sort each thing the plan asks for into:
   - **can do**;
   - **can do nearly** (say what differs);
   - **cannot do yet**.

   The plan's own "Wanted additions" section is a start, not the whole answer; phasor may
   have gained some of them since the plan was written.
3. **Say what will fall short, before building.** Show the person the nearly and cannot
   lists in a few lines, with the closest thing you will build for each. Add each cannot
   to `WISHLIST.md` (what was wanted, which scene, what was done instead). Then carry on;
   do not wait unless a missing piece removes the central picture itself.
4. **Write `scenes/NAME.py`, one animation at a time**, in the plan's order, on one
   timeline. Finish and check each before adding the next (steps 5 and 6), so a problem is
   found where it starts. For timing, take the plan's holds and silences; put a
   `[[marker]]` at each change and anchor each motion to its marker with a verb. Where the
   plan gives words to say, use its gist as placeholder narration: the person will record
   their own.
5. **Check:** lint with `--aesthetic` and `--strict`. Fix every error. Fix each warning,
   unless it warns about something the plan asks for on purpose (a frame filled edge to
   edge, the whole color wheel where hue is structure); then leave it and name it in the
   report.
6. **Look:** make the storyboard with `--marks` and look at the picture. For the
   animation you just added, answer:
   - Does it fill the stage, alone?
   - Does it hold as long as the plan says, and is something always changing a little?
   - Silent test: from the frames alone, would the viewer work out what the plan says
     they should?
   - Does it become the next animation without a cut?
   - Is every color doing what the plan says it encodes?

   Fix and go back to 5. At most 5 rounds per animation; then note what is left and move
   on.
7. **Preview:** render `--preset preview` to `out/NAME.mp4`.
8. **Report** in a few lines: each animation as built against the plan (as planned,
   nearly, or not possible), the files (plan, script, storyboard, video), what went to
   `WISHLIST.md`, warnings left on purpose, and anything in the plan you would question.
