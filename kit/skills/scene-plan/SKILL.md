---
name: scene-plan
description: Plan a beautiful teaching video before any code is written - brainstorm very widely with many subagents, prune to one path, and hand back a plan (the question, the central picture, an animation list with what to say and where to stay silent). Use whenever the person gives a topic or idea to teach or express ("a video about Lorenz attractors", "something on group theory", "how would you show X beautifully", "plan a scene/video on ...", "/scene-plan ..."), or asks how to make an idea beautiful, intuitive or well taught, even if they do not say "plan". Run it before /scene for anything bigger than a single small scene.
argument-hint: "<the idea or topic, in a sentence or a paragraph>"
---

# /scene-plan: from an idea to a plan for a beautiful video

The idea: $ARGUMENTS

This skill is about taste. It writes no scene code. It answers two questions, in this
order: how is this idea best taught, and how is that made beautiful? The result is a plan
the person approves; `/scene` builds from it afterwards.

`TASTE.md` is the person's taste and outranks everything here. Read it first. The house
rules below are a summary of it as of 2026-11-25; where they differ, `TASTE.md` is right.

## House rules

- **The aim is teaching things beautifully.** Beauty comes mostly from the mathematics
  shown well, not from decoration.
- **One animation owns the stage.** It fills the frame on midnight black, alone, and keeps
  the stage for as long as it needs: through iterations, new cases, a parameter turning.
- **The animation does the talking.** The person narrates, but briefly: a short setup, then
  a long silent stretch over ambient music, then more words only if needed. Half the video
  may have no voice.
- **Never rush.** A long hold is not a fault.
- **Quality, not quantity.** As many animations as the best story needs (ten or three). None
  added to fill time, none dropped to save it.
- **Come back to the same picture** and test it on new cases, instead of a new picture for
  each point.
- **Open with a question** or a picture the viewer wants explained; never a title or a
  definition. **Let the viewer predict** before each reveal.
- **Usually not concrete before abstract.** Ask which route the person wants (step 4).
- **Little text on screen. No hard cuts.**
- **Never shrink an idea to fit the tool.** If the best animation is one phasor cannot make
  yet, keep it in the plan, mark it, and add a line to `WISHLIST.md`. The person adds these
  to phasor; a plan weakened in advance hides what is missing.

The reference is the YouTube channel 2swap; see "The reference" at the end.

## The steps

Work files go in `plans/NAME.work/` (NAME is a short file name for the idea); the plan
itself is `plans/NAME.md`. Have subagents write their output to files and return only a
two-line summary, so your own context stays free for judging.

Step 2 and 3 are meant to be expensive: dozens of subagent runs. That is the point. A plan
can only be as good as the ideas it was chosen from, and the first ideas anyone has are the
ones every existing video already uses. Say in one line at the start that this is a long,
heavy run, then do it in full; do not shorten it to save effort. If you have no way to
spawn subagents, play each lens yourself, one at a time, writing each one's file before
starting the next and without rereading the earlier ones until the merge.

### 1. Consult

Ask in one message, each with your suggested answer; skip what the idea already answers:

1. **Viewer**: who is this for, and what do they already know?
2. **Point B**: what should the viewer understand or feel at the end that they did not at
   the start?
3. **Length and feel**: roughly how long; calm, eerie, playful or grand?

If the person does not answer, use your suggestions and say so in the plan.

### 2. Map

From point A (the opening question) to point B there are thousands of paths: different
questions, pictures, stories, animations and orders. See a great many before choosing.

**Cast about ten brainstormers** from the three families below: the idea's home fields,
the craft lenses, and at least three mathematical lenses that seem to have nothing to do
with the topic. The surprising picture usually comes from an outsider (billiards seen by a
hyperbolic geometer). Write the cast and why to `cast.md`.

Mathematical lenses: each asks "what is this, seen from my field?"

| Lens | Asks | Can split into |
| --- | --- | --- |
| Geometer | What shape or space is this? | Euclidean, hyperbolic, projective, differential |
| Algebraist | What is the structure; what is preserved? | groups and symmetry, representations, rings and polynomials |
| Linear algebraist | Which transformation is this; what are its special directions? | eigen-pictures, decompositions, high dimensions |
| Topologist | What survives stretching; what is glued to what? | surfaces, knots, coverings and unfoldings |
| Dynamicist | What happens when it repeats or flows? | orbits, phase space, chaos and bifurcation |
| Analyst | What happens in the limit, or at tiny scales? | approximation, Fourier, complex analysis |
| Combinatorialist | What is the space of all states, drawn as a graph? | counting, graphs, games and puzzles |
| Probabilist | What does a random one look like? | random walks, typical against extreme |
| Number theorist | Where do the integers or primes hide in this? | |
| Physicist | What machine does this; what is conserved? | mechanics, waves, optics |
| Computer scientist | What computes this? | |
| Duality hunter | What other object is secretly the same as this one? | |

Story lenses: skeptic (why would anyone doubt it), newcomer (what looks strange),
puzzle-setter (what game leads here), misconception collector (what people wrongly
believe), archivist (earlier ideas about the concept: how it was first pictured, older
approaches, forgotten constructions; a source of ideas, not a history lesson for the video).

Craft lenses: filmmaker (the unforgettable shot), animator (how each picture moves and
becomes the next), musician (what can be heard), phasor expert (what phasor already draws
beautifully; reads `kit/docs/API.md` and `kit/examples/INDEX.md`), 2swap student (what
would 2swap do with this).

**Round one.** Spawn the whole cast in one turn so they run in parallel and cannot see each
other. A brainstormer knows only its brief, so each brief carries: the idea, the viewer,
point B, the feel, its lens and question, the house rules above, and these instructions:

- Fill a mind map with these branches: opening questions, central pictures, ways of seeing
  the insight, cases to test it on, connections to other mathematics and to the physical
  world, things that can be heard, transitions, endings.
- Quantity, no judging. At least 8 opening questions, 8 central pictures and 4 in every
  other branch. Write the obvious ones down, then keep going past them.
- For every picture say how it fills the frame, how it moves, and what could change in it
  to keep it worth watching for a minute or more (which parameter turns, which cases run
  through it). A picture that is finished in five seconds cannot hold a stage.
- Do not ask whether phasor can draw it. (The phasor expert is the exception: it says what
  phasor draws well now.)
- Write to `plans/NAME.work/round1/LENS.md`.

**Merge and cross.** One agent reads all of round one and writes `map.md`: the branches
with repeats removed, each idea tagged with its lens, then a section of crossings (one
lens's motion on another's shape). It also names the two or three most fertile lenses.

**Round two.** Spawn, in parallel: the sub-lenses of the fertile lenses (third column of
the table), plus two fresh brainstormers. Each gets `map.md` and asks: what is missing,
what is the opposite of the obvious choice, what does this look like at another scale, in
another dimension, run backwards? Their additions go into `map.md`.

Aim, across both rounds, for at least 40 opening questions, 40 central pictures and 20 in
every other branch. If the map is short of that, run another round with new lenses.

**Paths.** Two or three agents each join branches of the map into whole paths from A to B,
a short paragraph each, deliberately unlike one another (different openings, different
central pictures, different orders). At least 20 in all, in `paths.md`.

### 3. Prune

Brainstormers never judge; use fresh agents. Four tests:

- **Beauty**: would a still from it be worth looking at with the sound off?
- **Intuition**: does the viewer see why, not only that? Would they follow it with the
  voice off?
- **Cohesiveness**: is it one picture and one thread from A to B, each beat growing out of
  the last?
- **Immersion**: does the viewer stay inside it, with nothing that breaks the spell (a
  hard cut, a crowded frame, a wall of symbols, a change of look)?

1. **20 to 5.** Four judges, one per test, plus a simulated viewer at the level from step
   1, each score every path and give one line on its weakest point. Keep the five best
   that are still unlike one another.
2. **Develop the five.** One agent per path writes it out animation by animation.
   Weaknesses hidden in a paragraph show here.

### 4. Ask the person which route

Show the five in two or three lines each, with your recommendation and why. Ask which
route they want, or which parts to mix. This is the consult on how to teach it: it comes
now, not earlier, so the choices are real ones for this idea. If they say "you choose",
choose.

### 5. Choose and attack

- **5 to 1.** Judge again with the person's answer in hand. A path strong on three tests
  and weak on one loses to a path good on all four. Parts of losing paths may be grafted
  on only if cohesiveness survives.
- **Attack the winner.** One agent argues as hard as it can that the winner is the wrong
  choice and proposes a better one. If it is right, go back to "20 to 5".

### 6. Teacher

Now make the chosen path teach as well as it can:

- **The central picture is the idea**, not a decoration of it.
- **Let the picture argue.** The viewer sees that something is true instead of being told.
  (To show two operations are the same, run them side by side on identical objects and let
  them move identically.)
- **Say, show, say.** For each stretch: a short setup in words (what to watch for), then
  the animation alone, long enough to be understood without help, then words again only if
  something still needs saying.
- **Write the silent test** for each animation: what should the viewer work out from it
  with no voice? If the answer is "nothing", the animation or the setup is wrong.
- **Name what is hard**: the one step where people get lost. Give it the most time.
- **Plan the returns**: cases that run through the same animation, each changing one thing.
  The last should surprise.
- **Place the predictions**: points where the picture holds before answering.
- **Earn every symbol.** A word or equation appears only after the viewer has seen the
  thing it names.
- **End on the picture, changed**: the opening picture again, now understood.
- **Stay true.** A simplification may leave things out; it may not say something false.
  Note what was left out.

### 7. Art director

The Art director may not change what is taught, only how it looks.

**Stage.** Midnight black. One animation, full frame: the mathematics reaches the edges,
not a small figure in the middle of a slide. It keeps the stage; do not cut away and come
back. Variation comes from inside it. Full, bright color is welcome when the mathematics
is that full (a domain coloring, a tiling).

**Color.** Hue is structure: assign color by something mathematical (phase, position in a
lattice, which tile, time along a path), and then the whole wheel is welcome. Where color
encodes no quantity, use two or three hues, each with a meaning that holds for the whole
video, and write the meanings down. Choose the palette from the feel of the subject.

**Motion.** Draw things on rather than switching them on; how a thing appears should show
how it is made. Slow is the default. During a hold something is always changing a little.
With the voice, each motion belongs to a word; without it, motion follows the music. One
animation becomes the next: an object carries over, morphs, or the camera travels through.

**The big moves** are how an animation renews itself while keeping the stage, mostly in
the silent stretches:

- **One number turns.** The camera is still, one anchor stays fixed, and a single
  parameter changes slowly so the whole picture reorganises around the anchor. The
  workhorse.
- **Sweep.** The camera travels steadily across or around a structure too large or too rich
  for one frame, so the viewer discovers it piece by piece. "Look how much is here."
- **Lift.** The camera pulls up and away from a detail the viewer has been studying until
  the whole is in view, and the detail turns out to be one part of a larger pattern. "What
  you just understood is everywhere." Its reverse, the descent, goes from the whole into
  one part.
- **Passage.** The camera goes into or through the object until its color fills the frame,
  and comes out in the next animation.

Each must show something the viewer could not see before it began, start and end at rest,
end on a composed frame that then holds, and bring into view only things already alive.

**Text.** Not much. One large plain label naming what is happening is often all a frame
needs. An equation only when earned; it leaves when its work is done.

### 8. Sound

Mark cues on the animation list; do not compose.

- **Music** carries the silent stretches, so it matters as much as the voice: one ambient
  bed, sitting back under the voice and coming forward when the voice stops.
- **Sounded mathematics**: where the subject has a natural sound, note it as MIDI (a
  frequency, a rhythm, a pitch that rises with a value, a note for each element or return).
- **Accents**: a soft sound on a reveal, a landing, a collision. Few, quiet, the same sound
  for the same kind of event.
- **Silence**: a short quiet before the payoff.

### 9. Editor

A fresh agent reads the plan as the viewer from step 1 and answers each; fix what fails:

1. Do the first ten seconds make me want the answer?
2. Do I reach point B, and feel it?
3. With the voice off, how much would I still understand? If little, the animations are
   illustrating the words instead of teaching.
4. Is the voice explaining something the animation already shows? Cut the words.
5. Does every animation earn its place and fit with the others? Cut any that fills time;
   restore any cut to save it.
6. Is any animation taken off the stage before it has finished its work?
7. Is any animation small, or sharing the frame?
8. Does anything break the spell?
9. Does each color still mean what it meant at the start?
10. Is anything said that is not true?
11. Held next to the reference clips below, where does this fall short?
12. Was any idea weakened to fit phasor? Restore it and put it on the wishlist.

If the Editor finds the path itself weak, go back to the map; do not patch.

### 10. Hand back

Write `plans/NAME.md` in this shape, then show it to the person and stop. Do not start
`/scene` until they approve.

```markdown
# NAME: the title

**Point A:** the opening question
**Point B:** what the viewer understands or feels at the end
**Viewer:** ...   **Length and feel:** ...

## The path
Three or four sentences.
Runners-up: the two best paths that lost, one line each, and why.

## Look
The central picture. What color encodes. The music bed.

## Animations
### 1. What fills the stage
- **While it holds:** how it changes, in order (iterations, cases, parameters, big moves)
- **Say before:** brief; the gist, for the person to put in their own words
- **Silent stretch:** how long, and what the viewer should work out alone
- **Say after:** if anything
- **Sound:** ...
- **Becomes the next by:** ...

## Left out
What was simplified away.

## Not possible in phasor yet
Each animation the plan needs that phasor lacks (also added to WISHLIST.md).
```

Add each "not possible yet" item to `WISHLIST.md` as its own line. Keep `map.md`; a later
video on a nearby idea can start from it. In your reply give the path in a few sentences,
the list of animations by title, anything not possible yet, and the file.

## The reference: 2swap

What the person admires, in their words: an animation takes the whole stage and keeps it;
he often does not talk for half the video; he explains briefly, lets the animation do the
talking, and sometimes explains more afterwards. Two clips they named (described from
their account and still frames):

- **"don't flip, algebra inside", 6:55 to 7:10.** The video opens on a full-frame color
  field before any explanation. Two identical rainbow lattices sit side by side, labelled
  only "×i ×j" and "×ij": the sameness is seen, not stated. Within seconds the lattice
  gives way to color filling the frame as the chapter changes.
- **"i'm so bad at billiards that i ended up in the hyperbolic plane", 5:00 to 5:40.** One
  green polygon stays fixed at the centre while the whole tiling around it changes shape
  continuously; every tile its own color, finer and finer detail in the gaps; no text; one
  picture, slowly changing, held for about 40 seconds.
