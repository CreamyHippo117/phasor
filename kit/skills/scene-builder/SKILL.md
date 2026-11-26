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
- **A hold extends the idea; it never replays it.** See the next section.
- **No hard cuts**: each animation becomes the next the way the plan says.
- **What color encodes**, and the same meaning for a hue throughout.
- **Little prose, enough math.** No sentences on screen, but the mathematics being taught
  is shown: the plan's equations and labels are the least you put up, not the most. See
  "The math on screen" below.

## Filling a hold: extensions, not replays

A long hold is long because the idea has more to give, not so the same motion can run
again. Every stretch of a hold must show the viewer something they have not yet seen in
this video.

A replay is: the same motion looped; a parameter swept out and then back over the range
it just covered; a case run a second time; the same camera move again; the earlier
picture again with nothing but its color or speed changed. A replay can also hide behind a
symmetry: new parameter values whose picture is an old one mirrored, rotated or shifted (a
Julia set for c and for its conjugate are mirror images, so a drift across the real axis
replays itself upside down). None of these is allowed as a way to fill time. (Coming back to a picture *with a new case* is not a replay: the case
is the new thing.)

An extension stays inside the same idea and the same picture and takes it one step
further. In rough order of strength:

- **the next case**: a different starting point, shape, function or number through the
  same animation, chosen because it behaves differently from the ones already shown;
- **further along**: the parameter goes on past where it stopped, into a range where the
  picture does something new (it should never turn round and go home);
- **another number turns**: a second parameter of the same object, the first now fixed;
- **deeper**: more iterations, a finer level, the next order of the approximation;
- **another view of the same thing**: a different big move (a lift after a sweep, a
  descent into one part), which shows the part against the whole;
- **the edge**: the extreme, degenerate or breaking case, where the idea almost fails.

Before coding a hold, write its stretches as a list with one line each: what it shows
that the stretches before it did not. If two lines say the same thing, one of them is a
replay; replace it. If the plan's own "While it holds" list runs out before the hold
does, add extensions of these kinds, the plan's first, yours after, and name the ones
you added in the report. If the idea is truly used up, end the hold there and say so in
the report: a shorter hold is better than a replay.

The small motion that keeps a resting frame alive (life, a slow drift) is not a stretch
and does not count as content; it only keeps the picture from freezing between them.

## The math on screen

A beautiful picture with no mathematics beside it is a screensaver. The viewer should be
able to pause any frame and say which mathematical object they are looking at and what is
changing in it. Add what the plan leaves out for this, and name each addition in the
report; it is part of building faithfully, not re-planning.

- **The formula being drawn**, typeset (`Equation`, `Text` with `$...$`), for as long as
  the picture is that formula: `z^2 + c` beside a Julia set, `x' = Ax` beside a flow.
- **The number that turns, live.** When a stretch turns a parameter, show its value as it
  moves (`sweep(path, value, show=True)`, or a `Counter`), so the viewer can tie what
  they see to a value. When the stretch ends, the number stays as the case's name.
- **Labels on what the picture is about**: the zeros, the fixed point, the axis, the
  eigenvector, the case's name. A label is `x.label` with `label(x) near x`.
- **One color, one meaning, in the picture and in the math.** Give a term of the
  equation the color role of the thing it names (`Equation(colors=...)`), so the viewer
  sees which part of the formula is which part of the picture.
- **The step, not only the result.** When the animation shows why, show the line of
  mathematics that goes with it, and change it with `transform` as the picture changes,
  rather than replacing it with a new one.

Keep it legible and out of the way: few objects of math at once (a formula, a value, a
label or two), at the theme's sizes, placed by layout at the edge of the stage, never
over the part of the picture being studied. Math comes in after the viewer has seen the
thing it names (write it in, do not switch it on), and leaves when its object leaves. A
long silent stretch keeps its formula and its live number; they are what makes the
silence readable.

## The steps

1. **Read the plan and list the work.** One line per animation: what fills the stage, how
   it changes (the list of stretches from the section above), how long, how it becomes
   the next. Name the scene after the plan.
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
   - Replay test: the marker frames show only where each stretch starts, so make a second
     storyboard with `--t` at two or three times inside every stretch, and go along it in
     order. Does each stretch show something no earlier frame of the video did? Any two
     frames that look alike, even mirrored or turned, are a replay; turn the later stretch
     into an extension.
   - Silent test: from the frames alone, would the viewer work out what the plan says
     they should?
   - Math test: pick any frame. Can you read off it which formula is drawn, which number
     is changing and what its value is, and what the labelled features are? If not, add
     the formula, the readout or the label.
   - Does it become the next animation without a cut?
   - Is every color doing what the plan says it encodes?

   Fix and go back to 5. At most 5 rounds per animation; then note what is left and move
   on.
7. **Preview:** render `--preset preview` to `out/NAME.mp4`.
8. **Report** in a few lines: each animation as built against the plan (as planned,
   nearly, or not possible), the extensions you added to fill a hold and any hold you
   ended early, the math you put on screen beyond what the plan named, the files (plan, script, storyboard, video), what went to
   `WISHLIST.md`, warnings left on purpose, and anything in the plan you would question.
