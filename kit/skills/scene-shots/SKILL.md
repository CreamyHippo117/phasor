---
name: scene-shots
description: Plan the pictures for a story the person already knows - they bring what is said and in what order (a script, an outline, notes, the equations), and this skill designs how each beat is seen, the way a cinematographer works from a finished screenplay - no new ways to teach it, every beat made beautiful and true. Use when the person knows what they want discussed and asks for the visuals ("I know what I want to say, make it beautiful", "here is my script/outline, plan the animations", "a shot list for ...", "keep this plan's story, redo its pictures", "/scene-shots ..."). Writes plans/NAME.md in the /scene-plan format, so /scene-builder codes it. If the person has only a topic and wants to find how to teach it, use /scene-plan instead.
argument-hint: "<the script, outline or notes, pasted or as a path>"
---

# /scene-shots: from a story you already know to a shot plan

The source: $ARGUMENTS

The person has already decided what the video says. A cinematographer is handed a finished
screenplay and does not rewrite it. They work out how each scene is seen: where the camera
is, what fills the frame, how the light falls, how one shot becomes the next. This skill
does that job. The story, its order and its claims come from the person. The pictures come
from you. Nothing here brainstorms new ways to teach the idea (that is `/scene-plan`). The
brainstorming here is about how each beat looks.

An example of the kind of source this is for: *"Introduce LTI systems. A clap in a room;
the room unfolds into its mirror rooms, and each mirror room is an echo; so what you hear
is a sum of delayed, scaled copies of the clap. That is linearity and time invariance, and
the room is its impulse response."* `/scene-plan` would ask whether LTI is the right place
to end and would search twenty paths. `/scene-shots` keeps this order as given and asks
only how it looks. What does the clap look like? How does the room unfold? What does the
viewer see that makes "a delayed, scaled copy" plainly true? How does that picture turn
into the next one? The bar is the opening of the person's reverb A1 plan, the clap and the
room unfolding into its crystal of mirror rooms: it established LTI, which everything after
it stood on, and it was beautiful while doing it (`.claude/skills/scene-plan/EXEMPLAR.md`).

This skill writes no scene code. Its only code is the throwaway sketches of the look book
(step 4). It ends with a plan the person approves. `/scene-builder` codes it afterwards.

## What is fixed, and what is yours

**The person's, fixed:**

- What is said, in what order, and every claim. Where the video begins and ends.
- Their words, if they wrote narration. You may make light edits for timing only, and you
  list each one in the plan.
- Every equation and number they named is taught, and every showcase moment they named
  is kept.

**Yours:**

- **The evidence for each beat**: what the viewer sees that makes its sentence true. A beat
  that is only said has not been shot yet.
- **The pictures**: what fills the stage for each beat, how it moves and holds, and how it
  becomes the next one.
- **The world**: the one object or space most beats live in, what each colour means, the
  palette, and where written math sits, if any.
- **How their math is carried** (scene-plan's "How math is carried"): enacted by the
  picture where it can be, spoken where it cannot, written beside the picture only for a
  result to look at again, a callback, or the one line a stretch rests on. You may also
  enact one of their claims as its equation ("each mirror room is a delayed, scaled copy"
  becomes h(t) = Σ a_k δ(t − τ_k), one room per term). That shows a claim they already
  made, so it is allowed. Mark each such equation *from your words* in the inventory. A
  result the source does not state is new content and is not yours to add. If you carry
  an equation they named as spoken rather than written, say so under "Notes for you".
- **Narration from notes.** If the source is notes, not finished narration, you may draft
  the *say* lines from it, with no claim the notes do not make. The person records their
  own voice anyway. Draft them to the house rule "Introduce, then use" (`CASEBOOK.md` §1):
  the thing in plain words and in the picture first, then its name.
- **Pictures that introduce.** Make sure that a term the source names is on screen as a
  thing before the word lands (the sharp strike drawn before "impulse").
- Holds, predictions as pauses before a reveal, verbs, camera moves, transitions and sound
  cues.

**Not yours. Write it under "Notes for you" instead:**

- Adding a topic, case or derivation. Cutting a beat, merging beats or reordering them. If
  a beat looks weak or the story would be clearer in another order, say so and why, then
  plan it as given.
- A term or step used before the viewer has met it. If the person's own words name a
  thing before showing it, suggest the plain naming. If the story needs a bridge the
  source lacks (an FFT for a viewer who has never met the roots of unity), propose the
  bridge, what it starts from and where it would go, and say which beat the viewer will
  be lost at without it.
- A false claim in the source. Never fix it quietly. Give the correction and ask. If the
  person does not answer, plan it with the correction and say clearly that you did.

## The rules you work under

`TASTE.md` is the person's taste and outranks everything here. Read it first, then
`.claude/skills/scene-plan/EXEMPLAR.md` (the fixed point every plan is held against) and
`.claude/skills/scene-plan/CASEBOOK.md` (the detailed cases). Then read these parts of
`.claude/skills/scene-plan/SKILL.md`, which this skill shares rather than repeats: **What a
good video is here**, **House rules**, **How math is carried**, step **8. Cinematographer**,
step **9. Sound** and step **10. Checks**.

- **The visual rules apply in full.** These are: one animation owns the stage, midnight
  black, no hard cuts, hue is structure, draw things on, the big moves, math enacted where
  the picture can carry it and spoken where it cannot, the Cinematographer's frame veto,
  watched segments, never shrink an idea to fit the tool, and stay true.
- **The story rules belong to the person.** These are: the spine, open with a question,
  concrete before abstract, length, and which cases come back. If the source lacks one of
  these, suggest it under "Notes for you" and do not impose it. Two of them are partly
  yours. A pause before a reveal is a choice of timing, so place one wherever the source
  already has a reveal. A return the source makes is the person's choice, so keep it, but
  make the picture fresh (step 6).

## The steps

Work files go in `plans/NAME.work/`, and the plan is `plans/NAME.md`. Have subagents write
their output to files and return only a two-line summary, so your own context stays free
for judging. **Every brief carries `TASTE.md` in full**, plus the source and the
breakdown: a subagent knows only its brief, and taste it never saw cannot steer it. Every
judge, picker and editor scores against `TASTE.md` too, and where a choice goes against a
line of it, says which line and why. `EXEMPLAR.md` goes in every brief too. This is a
medium run, a few agents per stage, not the wide brainstorm of `/scene-plan`. Say so in one
line at the start. If you have no way to spawn subagents, play each role yourself, one at a
time, and write each role's file before you start the next.

### 1. Read

Read `TASTE.md`, the source and the scene-plan sections above. If the source was pasted,
save it word for word to `plans/NAME.source.md`, so the plan can always be checked against
it. If the source points at an earlier plan or scene ("like the opening of reverb A1"),
read it from `plans/` or `scenes/`. If it is not there, ask for it in step 3.

The source may also be a finished plan, from `/scene-plan` or an earlier `/scene-shots`,
whose story the person wants kept and whose pictures they want made again. Then its spine,
say lines, math inventory and claims are the source, and its pictures, look, verbs and
sound are not: plan them fresh. Write the new plan to a new name
(`plans/NAME_shots.md`), so the old one stays to compare against.

### 2. Break it down

This is the script breakdown a cinematographer does first. Split the source into beats, one
claim or one moment each. Write `breakdown.md` with one entry per beat:

- **Source**: the words, quoted or as a line reference.
- **Claim**: the sentence the viewer should believe after it.
- **Evidence**: what must be *seen* for the claim to be seen true. This is the brief for the
  shot, so it is never "a nice picture of X".
- **Math**: the equations and numbers the source names, plus any claim of theirs that has
  an equation (*from your words*).
- **Stands on it**: the later beats that use what this beat establishes. A beat many
  later beats stand on is load-bearing, like the A1 opening, and its picture deserves the
  most care.
- **Showcase**: *named* if the source asks for it; *proposed* if the beat is load-bearing
  and its picture could be breathtaking; otherwise no. A breathtaking picture that no later
  beat stands on is proposed only under "Notes for you", as a possible detour.
- **Carries over**: what this beat takes from the last one (an object, a picture, a colour).
- **Leans on**: each term, idea or step this beat uses, marked *known* (the viewer has it),
  *introduced* (an earlier beat gives it) or *gap* (nothing does). Mark a gap as *named
  early* when the source names it before showing it, or *needs a bridge* when the viewer
  lacks its ground.

For the example above, the first beats might read:

> **1. The clap.** *Source:* "A clap in a room." *Claim:* one short sound fills a room.
> *Evidence:* a single pulse leaves the clap and its wavefront spreads, reaches the walls
> and comes back. *Stands on it:* every later beat. *Showcase:* proposed, the start of
> the unfolding.
>
> **2. The unfolding.** *Source:* "the room unfolds into its mirror rooms, and each mirror
> room is an echo." *Claim:* an echo off a wall sounds like a clap from the mirror image of
> the room. *Evidence:* the wall turns into glass, the mirror room appears behind it with
> its own clap, and the straight line from that clap to the listener crosses the wall
> exactly where the bounced path does. Then the mirror rooms tile outward into a crystal.
> *Math:* τ_k = d_k / c and the sum h(t) = Σ a_k δ(t − τ_k), enacted, one room per term
> (*from your words*). *Stands on it:* linearity, time invariance, the impulse response,
> and everything after. *Showcase:* named. *Carries over:* the room, the clap and its
> colour.

### 3. Consult

Ask in one message, giving your suggested answer to each, and skip what the source already
answers:

1. **The breakdown**: one line per beat. Is this how you split it?
2. **Viewer**: who is it for, and what do they already know?
3. **Math**: only what you named, or also your claims enacted as equations? Any you
   want written even where the picture or the voice could carry it?
4. **Feel**: calm, eerie, playful or grand?
5. **Words**: is the source narration to keep, or notes to draft narration from?
6. **Gaps**: each gap the breakdown found, with your proposed fix: plain words before the
   name, or a bridge (what it starts from, and where it would go). Ask whether to add each
   one. If they say yes, the bridge becomes beats of the source, marked *bridge, added with
   your OK* in `plans/NAME.source.md`, and every agent after this treats it as the source.

If the person does not answer, use your suggestions and say so in the plan.

### 4. The world

Before planning single shots, find the one world the whole video lives in. This is the
cinematographer's look book. The biggest decision is whether one object can carry most of
the beats (the room carries the clap, the echoes, the crystal and the impulse response),
because the house rule "come back to the same picture" depends on it.

Spawn four treatment agents in parallel. Give three of them a different kind of world, so
they do not all find the same one:

- **the thing itself**: the physical or concrete object the source talks about;
- **the space it lives in**: the mathematical object behind it (a lattice, a phase plane, a
  function space);
- **an outsider's view**: the source seen by a field that seems to have nothing to do with
  it (a geometer sees the mirror rooms as a crystal, a topologist sees the unfolding as a
  covering).

The fourth is the **phasor expert**. It reads `kit/docs/API.md`, `kit/examples/INDEX.md`
and the model examples (`kit/examples/complex/prism_primes.py` is the reference look), and
proposes the world phasor already draws most beautifully for this source: the prism
look's glowing strokes and spectrum, a kernel-drawn field filling the frame, a 3D surface
under a turning camera. Its job is to bring what the tool is best at, not to set a limit:
the other three do not read the docs and do not ask whether phasor can draw their world.

Each brief carries `TASTE.md`, the breakdown, the consult answers, the scene-plan sections
above and these instructions. Cover every beat in the source's order, adding none and
dropping none. Write to `treatments/KIND.md`:

- the central picture, and how it fills the frame;
- what colour encodes, with a meaning for each hue that holds for the whole video, and the
  palette chosen from the feel;
- where written math sits, if any;
- three key frames, each described as a still the viewer could pause on with the sound off
  (what is lit, what is symbol and how much of the frame it takes, where the eye goes),
  the first being the load-bearing beat;
- for each beat, one line: how this world shows its evidence;
- how the world reaches the last beat, and what the opening picture looks like when it
  comes back understood;
- the music bed.

**The look book.** Words are a poor way to choose a look. When the four treatments are in,
one agent sketches each world as one still: the key frame that shows it best, written as
a throwaway script in `plans/NAME.work/sketches/KIND.py` and rendered as one frame with
the render command in `CLAUDE.md` (`--frame` at the chosen time, `--preset preview`). A
rough still is enough. It is not a scene: no checks beyond rendering, nothing reused in
the build. Look at each still yourself. Where phasor cannot draw a world yet, sketch the
closest thing it can and say on the still's line what is missing. A sketch that falls
short never counts against its world. If a sketch will not render after a few tries, give
that world its words only and say so.

A fresh judge, with the stills and the treatments, passes or fails each world on four
gates, and no strength on one buys a pass on another: **evidence** (is each beat's
evidence visible, its math enacted where it can be?), **beauty** (would each still be
worth looking at with the sound off?), **continuity** (one world, no cuts) and **truth**
(does the picture itself claim anything false?). It ranks the worlds that pass by
comparison with the A1 opening, as true, as load-bearing and as watched, and checks the
fit with `TASTE.md`. It does not score whether phasor can draw a world: one phasor lacks
goes under "Wanted additions", never out of the running. It recommends one, or one with a
graft from another.

Show the person the four stills, each with two lines and what its sketch is missing, and
the judge's pick and why. Ask which they want. This is the second and last question. If
they say "you choose", choose.

### 5. Shots

Now plan each beat in the chosen world. Spawn four shot designers in parallel. Each writes
the *whole* shot list, so each version runs as a film from the first beat to the last.
Each brief carries `TASTE.md`, the breakdown, the chosen world and the scene-plan sections
above.

- **cinematographer**: the unforgettable shot, and how each shot moves, holds and becomes
  the next;
- **2swap student**: what 2swap would do here, judged from the references in scene-plan;
- **the beat's home mathematician** (a geometer for mirror rooms, an analyst for spectra):
  what shape the thing really has, so the picture is true as well as beautiful;
- **phasor expert**: which of phasor's objects, looks, verbs and camera moves would make
  each shot most beautiful, including ones nobody would think to ask for. It reads the
  docs and examples as in step 4; the other three still do not.

For each beat, each shot says: what fills the stage; the evidence it shows (its picture
test: with the voice off, what is seen); how it changes while it holds (iterations, cases,
one number turning, a sweep, a lift, a passage); how long it holds and why; where the
source's words land; how each piece of math is carried (enacted by which change in the
picture, spoken, or placed and why); its key frames; and how it becomes the next shot.
Write to `shots/LENS.md`.

**Showcase segments get their own designer.** For each one, a separate agent writes it out
second by second: what reaches the frame when, what the camera does, where the music comes
forward, and the composed frame it ends on. A showcase has room to be watched, its math
enacted rather than laid beside it, and the beats right after it build on what it
established. Write to `shots/showcase_NAME.md`.

A fresh picker then chooses, beat by beat, the best of the four versions and stitches them
into `shots.md`. A shot that wins its own beat but breaks the next one's continuity loses
to one that runs on smoothly. A shot never wins because it is easier to build: the phasor
expert's version wins a beat only on beauty, evidence and continuity, as any other does.
A shot with its math enacted beats one with the same math placed. The picker says why
each beat went the way it did.

### 6. Continuity

One fresh agent reads `shots.md` from end to end as a film and fixes the following:

- a hard cut, or an object that switches on instead of being drawn on;
- a picture taken off the stage before its work is done, or a small figure in an empty
  frame;
- a colour whose meaning drifts;
- a written line that crowds a key frame or takes the eye from the subject: it becomes
  spoken or enacted (the frame veto), and is never dropped from what is taught;
- an equation band or ledger that moves for no reason;
- a hold that cannot fit its words. Narration runs at about 2.5 words a second, so count
  the *say* lines of each shot. If the source's words for a beat need longer than its
  picture holds, lengthen the hold with something the viewer gains (the next case, the
  camera's next move), never with a repeat, and never cut their words to fit;
- a return. When the source comes back to something, the content repeat is the person's
  choice, but the picture has to give the viewer something new: a new angle, a new case on
  the same object, the picture now heard. Write that in one sentence for each return.

### 7. Verbs, sound

Follow the **Verbs** bullet of scene-plan step 8 and step 9 **Sound**. Read `kit/docs/API.md` only now,
after the shots exist. Give each motion its verb and mark where it lands. A motion no verb
can make stays in the plan as imagined and goes under "Wanted additions".

### 8. Checks

Follow scene-plan step 10, claims, order and frames, with fresh agents that wrote none of
the plan. The claims check looks as hard at what each shot *shows* as at what it says: an
exact lattice of mirror rooms when the room is not a box, a wave drawn faster than the
sound it stands for. The order check reads the source's approved bridges as part of the
story. A "not yet" or a false claim in the person's own words goes under "Notes for you"
(with its correction, see "Not yours"); one in pictures or narration you drafted is
fixed.

### 9. Editor

A fresh agent reads the plan as the viewer from step 3, with the source beside it, and
answers each question below. Fix whatever fails.

1. **Faithful**: is every beat of the source on screen, in order? Is nothing added that
   teaches something the source does not, except bridges the person approved? (An
   approved bridge is part of the source: keep it in full and never count it as an
   addition.) Are the person's words intact, or each edit listed?
2. With the voice off, is each beat's evidence seen?
3. Would a still from each shot be worth looking at with the sound off? Which shot makes
   the weakest still? Improve it.
4. Does each showcase have room to be watched, and do the beats after it stand on what
   it established?
5. Is it one world, with no hard cuts, and does each colour keep its meaning?
6. Is any math placed that the picture could have enacted, or written that could have been
   said? Is every written equation built one at a time, and kept in the ledger while later
   steps need it?
7. Was any picture weakened to fit phasor? Restore it and list it under "Wanted additions".
8. Is any verb there only as decoration? Cut it.
9. Is anything used before it is introduced? Fix it in the pictures and in narration you
   drafted; for the person's own words and for a missing bridge, add it to "Notes for
   you".
10. Held beside the A1 opening, is the load-bearing beat as true, as load-bearing and as
    watched? Beside the 2swap clips, where does it fall short?
11. Does anything go against a line of `TASTE.md`? Fix it, or say which line and why the
    plan keeps it.

If the world itself fails, go back to step 4. Do not patch it.

### 10. Hand back

Write `plans/NAME.md` in the shape given in scene-plan step 12, so that `/scene-builder`
builds it unchanged. Change these parts:

- **Point A, Point B and the spine** are the source's. Write them from it and invent
  nothing. If the source has no spine sentences, use its main claims in its own words, and
  say so.
- **The path** becomes **The source**: the path to `plans/NAME.source.md`, then a table
  with one row per beat: the source words, the animation that shows them, and its
  evidence.
- **Runners-up** become the worlds not chosen, one line each, and why.
- Add **Narration edits**: each change to the person's words, and why. Write "none" if
  there were none.
- Keep scene-plan's **What the video stands on**, **Math inventory** (with the enacted,
  spoken and placed counts) and **Concept order**, filled from the source and the
  pictures.
- Add **Notes for you**: suggestions about the content (a term used before it is
  introduced, a missing bridge, a reorder, a weak beat, a missing prediction, a false
  claim and its correction). Say for each one whether the plan applies it. Only corrected
  false claims are applied.

Add each wanted addition to `WISHLIST.md` as its own line. Keep `breakdown.md`,
`treatments/`, `sketches/` and `shots/`. In your reply, give the world in a few sentences,
the list of animations by title, the math counts, the notes for you, the wanted additions
and the file. Then stop. Do not start `/scene-builder` until the person approves the
plan.
