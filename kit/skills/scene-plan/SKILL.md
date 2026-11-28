---
name: scene-plan
description: Plan a teaching video, beautiful and full of real mathematics, before any code is written - find the concepts the video stands on and the breathtaking picture that establishes each one, brainstorm widely with many subagents, pass paths through gates (the question, teaching, the picture, can the viewer follow), judge rendered frames rather than prose, and hand back a plan (the question, the spine, an animation list with what to say and how each piece of math is carried). Use whenever the person gives a topic or idea to teach or express ("a video about Lorenz attractors", "something on group theory", "how would you show X beautifully", "plan a scene/video on ...", "/scene-plan ..."), or asks how to make an idea beautiful, intuitive or well taught, even if they do not say "plan". Run it before /scene-builder, which codes the plan. If the person already knows what the video says (a script, an outline, notes) and wants only the pictures planned, use /scene-shots instead.
argument-hint: "<the idea or topic, in a sentence or a paragraph>"
---

# /scene-plan: from an idea to a plan for a beautiful video

The idea: $ARGUMENTS

This skill writes no scene code, apart from throwaway sketches for stills (step 4). It
ends with a plan the person approves; `/scene-builder` codes it afterwards.

Read three files first:

- `TASTE.md` (in this folder), the person's taste. It outranks everything here.
- `.claude/skills/scene-plan/EXEMPLAR.md`, the opening of the person's reverb A1 plan, and
  why it worked. It is the one fixed point every plan is held against.
- `.claude/skills/scene-plan/CASEBOOK.md`, the detailed lessons, as cases. The steps below
  say when each part is needed.

Below, `EXEMPLAR.md` and `CASEBOOK.md` mean those two files.

## What a good video is here

The opening of A1 was a clap in a room, unfolding into a crystal of mirror rooms. It was
the exemplar because it established LTI, the concept everything later in the video stood
on, and it was beautiful while doing it. The picture *was* the derivation, so nothing had
to be laid beside it. Every part of this skill aims at that:

1. **Stay on the question.** The video opens with a question (point A) and every stretch
   answers it or moves toward the answer. This is the leash on beauty and on math alike.
2. **Find what the video stands on.** Two to four concepts carry the whole explanation (in
   A1: LTI). Each one gets the breathtaking picture that establishes it. Beauty goes into
   the foundation, not beside it.
3. **Let the picture prove the math.** The math of the video is first of all what its
   pictures prove. Math that no picture proves is said by the voice, not written, or cut.
4. **Judge what will be seen.** Beauty cannot be judged in prose. A plan says what is on
   screen at fixed times, and finalists are rendered as rough stills that the judges look
   at.

## House rules

- **The spine.** Two to four sentences the video exists to make obvious ("a room is a
  matrix"; "sines are its eigenvectors"; "so in the Fourier basis it is an EQ"). Every
  animation serves one, or makes one reachable (an introduction or a bridge).
- **One picture owns the stage.** It fills the frame on midnight black and keeps it while it
  has work to do. One animation becomes the next, with no hard cuts.
- **Explain.** The voice carries the reasoning and the picture carries the evidence. Every
  mathematical move gets its reason in one sentence before it happens.
- **Watched pictures get time.** A breathtaking picture that a later step stands on gets a
  segment of its own: a long hold, the music forward, the math right after. It is never
  cut to save time. A breathtaking picture that nothing later stands on is a detour,
  however true it is.
- **Never rush, never pad.** Hold as long as understanding needs and no longer. Length
  follows content. As many animations as the story needs, and none to fill time.
- **Come back to the same picture** with a new case, and only when the viewer gains
  something new from the return (`CASEBOOK.md` §4).
- **Open with a question. Let the viewer predict** before each reveal. Usually not concrete
  before abstract (ask which route, step 5).
- **Introduce, then use.** Nothing is used before the viewer has met it. The thing comes
  first, in plain words and in the picture, then its name. Where an idea needs ground the
  viewer lacks, build a bridge from what they know. The cases are in `CASEBOOK.md` §1.
- **Stay true, and check.** A simplification may leave things out; it may not say
  something false.
- **Never shrink an idea to fit the tool.** If phasor cannot make the best picture yet, keep
  it, list it under "Wanted additions" and add a line to `WISHLIST.md`.

## How math is carried

Every piece of math in the plan is carried in one of three ways, and the plan says which:

- **Enacted.** The picture performs the step, and the written terms come out of the objects
  as they act. *Example:* each mirror room of A1 appears and adds its term a_k δ(t − τ_k) to
  the sum. Without the picture, the line would mean nothing. This is the first choice.
- **Spoken.** The voice says it and nothing is written. This is for a step the viewer
  needs to hear but not to look at again.
- **Written beside the picture (placed).** It is written, but the picture on screen does
  not derive it. This is allowed for three things only: a result the viewer will need to
  look at again (it goes to the ledger), a callback written beside a new result so the
  sameness is seen, and the one line a whole stretch rests on (Euler, in an FFT video).
  Each placed line names which of the three it is.

The plan counts all three, enacted, spoken and placed, and reports the counts. The
person's working hypothesis (2026-11-28) is that the frames they found ugly were mostly
placed equations, so the count is kept so that it can be checked. More math means more
enacted and more spoken, not more written. The details of building written math (term by
term, the ledger, colour, numbers) are in `CASEBOOK.md` §2.

## The steps

Work files go in `plans/NAME.work/`, and the plan is `plans/NAME.md`. Have subagents write
to files and return a two-line summary. Every brief carries `TASTE.md` and `EXEMPLAR.md`
in full: a subagent knows only its brief. Steps 2 and 3 are meant to be expensive, with
dozens of subagent runs. Say so in one line at the start, then do them in full. Without
subagents, play each role yourself, one at a time, writing each file before starting the
next.

### 1. Consult

Ask in one message, with your suggested answer to each, and skip what the idea answers:

1. **Viewer**: who is it for, and what do they already know? Write it as a short list
   ("introductory ODEs: e^(rt), the integrating factor; introductory linear algebra:
   matrices, eigenvectors, change of basis"). Anything not on the list is introduced in
   the video.
2. **Depth**: named results, or derivations followed step by step?
3. **Feel**: calm, eerie, playful or grand?

Do not ask where the video ends, and do not decide it. Point B is what the brainstorm
finds. A destination the person names goes to the brainstormers as one possible ending
among others.

### 2. Map: harvest the pictures

Cast about ten brainstormers from `CASEBOOK.md` §5: the idea's home fields, the craft
lenses, and at least three mathematical lenses that seem to have nothing to do with the
topic (the surprising picture usually comes from an outsider). Write the cast to
`cast.md`.

**Round one.** Spawn the whole cast in parallel. Each brief carries the idea, the viewer's
list, the feel, its lens, this skill's first two sections and these instructions:

- Fill these branches: opening questions; destinations; **load-bearing concepts** (what an
  explanation of this idea stands on); **pictures**; cases to test them on; things to hear;
  connections; transitions; endings. Quantity, no judging: at least 8 opening questions,
  8 pictures, 5 destinations and 4 in every other branch. Write down the obvious ones, then
  keep going past them.
- For **every picture**, say: what fills the frame and how it moves; which concept it
  *establishes*; what later step could stand on it; which math it *enacts*, step by step;
  and why it would be watched with the sound off. A picture that establishes nothing is
  decoration, however long it can be watched.
- Do not ask whether phasor can draw it (the phasor expert excepted).
- Write to `round1/LENS.md`.

**Merge.** One agent writes `map.md`: the branches without repeats, each idea tagged with
its lens, the crossings (one lens's motion on another's shape), the two or three most
fertile lenses, and the fifteen strongest pictures marked.

**Round two.** The fertile lenses' sub-lenses and two fresh brainstormers each get
`map.md` and ask what is missing, what the opposite of the obvious choice would be, and
what it looks like at another scale, run backwards. Aim for at least 40 opening questions
and 40 pictures across both rounds; if short, run another round.

**Weave.** Three or four weavers, each pairing differently (across senses, across fields,
across scales), find two ideas carried by one object (`CASEBOOK.md` §4). "None worth it" is
an allowed answer. Write to `weaves.md`.

**Paths.** Two or three agents join the map into at least 20 whole paths in `paths.md`,
deliberately unlike one another. Each path is a short paragraph that names:

- its point A (the question) and point B;
- its spine;
- its load-bearing concepts, each with the picture that establishes it;
- its math, as what those pictures enact, plus what is spoken.

### 3. Gates: 20 to 5

Fresh agents judge; brainstormers never do. There are four gates, and a path must pass
every one. No gate counts double, and no strength on one buys a pass on another.

- **Question.** Does every stretch answer point A, or move toward the answer? The judge
  names any stretch that wanders.
- **Teaching.** The judge writes what the viewer can explain, derive or compute at the end,
  as sentences and equations they could reproduce: reasons, mechanisms and results, never
  impressions. Is each spine sentence on that list? Is each load-bearing concept
  established before it is used?
- **Picture.** Does each load-bearing concept have a picture that establishes it, enacts
  its math and would be watched with the sound off? A path whose math is mostly placed
  fails here.
- **Follow.** A simulated viewer who knows only the list from step 1 names every idea the
  path uses but never introduces, and every bridge it lacks. A path that needs more
  introductions than it has room for fails.

Each judge gives pass or fail with the line where the path fails. Among the paths that
pass, rank by comparison with the exemplar, not by points: for each pair, which is closer
to the A1 opening, as true, as load-bearing, as watched and as much on its question? Keep
the five best that are still unlike one another. If fewer than five pass, fix a path or go
back to the map. Never lower a gate.

### 4. Develop and sketch the five

One agent per path writes it out animation by animation: the spine sentence each serves,
its load-bearing picture if it has one, and how each piece of math is carried.

Each development also writes **key frames**: what is on screen at fixed times (the start
of each animation and every minute mark). For each one it gives what is lit, what is text
or symbol and roughly how much of the frame that takes, and where the eye goes.

Then **sketch the stills.** For each path, one agent writes throwaway scripts for two or
three of its key frames, starting with its load-bearing picture, in
`plans/NAME.work/sketches/`. Each is rendered as one frame with the render command in
`CLAUDE.md` (`--frame` at the chosen time, `--preset preview`). A rough still is enough:
no checks beyond rendering, nothing reused in the build. Where phasor cannot draw it yet,
sketch the closest thing and note what is missing; that never counts against the path.

The Picture gate is judged again, now **on the stills**, by a fresh judge who looks at
them beside the key-frame list. Prose that reads well does not pass a still that looks
crowded.

### 5. Ask the person which route

Show the five: each with its point A and point B in two lines, its stills, and your
recommendation and why. Ask which route they want, or which parts to mix. If they say
"you choose", choose.

### 6. Choose and attack

Choose with the person's answer in hand, by the same gates and the same comparison with
the exemplar. Parts of losing paths may be grafted on only if the path still passes. Then
one agent argues as hard as it can that the winner is the wrong choice. "It spends time on
a bridge" is not an attack; "it is less load-bearing than the A1 opening, here is why" is.
If the attack is right, go back to step 3.

### 7. Teacher: what is taught

The Teacher owns *what* the viewer learns, and in what order.

- **Say, show, say.** For each stretch: what to watch for and why it matters; the
  animation; then what was seen, why it is so, and what it means for the spine.
- **Two tests per animation.** With the voice off, which evidence does the picture show?
  With the picture off, does the narration still give the reason?
- **The takeaway** for each animation: the sentence or equation the viewer could
  reproduce, and the spine sentence it serves.
- **Introductions and reasons.** Write the concept order as you go (`CASEBOOK.md` §1): what
  each animation introduces, with the phrase, line or picture that does it, and what it
  uses. Give each move its reason in "Say before".
- **Name what is hard**, the step where people get lost, and give it the most time.
- **Predictions, returns, and the ending:** places where the picture holds before
  answering; cases that change one thing each, the last a surprise; and the opening picture
  again at the end, now understood.
- **The math** is listed as what must be understood, each item with a proposed way of
  carrying it (enacted, spoken, placed). The Teacher does not decide what is written.

### 8. Cinematographer: how it is carried and seen

The Cinematographer owns *how* each thing reaches the viewer: the stage, colour, motion,
camera, verbs, text and the way each piece of math is carried. It may not remove anything
the Teacher's list says must be understood. It decides how each piece is carried.

- **The carriage.** For each math item: enact it if the picture can; otherwise speak it;
  place it only for the three reasons in "How math is carried".
- **The frame veto.** If a written line makes a key frame worse (crowded, symbols taking the
  eye from the subject), the line becomes spoken or enacted. It is never forced onto the
  frame, and never dropped from what is taught.
- **The stage, colour, motion and big moves:** `CASEBOOK.md` §3. Watched segments get their
  hold and the music forward.
- **Verbs come last.** Only now, read phasor's verbs and camera moves in `kit/docs/API.md`
  (read it each time; it grows). Give each motion the verb that does it best, then read the
  list again for a verb that would show something nobody imagined. Every verb reveals
  something or keeps the picture alive; one that does neither is cut. Mark where each lands.
  A motion no verb can make stays as imagined and goes under "Wanted additions".
- Update the key frames to match.

### 9. Sound

Mark cues; do not compose. One ambient bed, back under the voice and forward in the
watched stretches. Where the subject has a natural sound, note it as MIDI. A few quiet
accents, the same sound for the same kind of event, each where a verb lands. A short quiet
before the payoff.

### 10. Checks

Three fresh agents that wrote none of the plan work in parallel. Each reads `CASEBOOK.md`.

- **Claims.** Every number, equation, derivation and demo is derived, computed (a short
  NumPy check is fine) or marked unverified, in `claims.md`. Look hardest at the edges of a
  model: an infinite case shown on a finite one, a limit taken silently, a demo whose
  physics or hearing does not work as said. Pictures make claims too: echoes drawn evenly
  spaced when they are not, every copy drawn equally loud. A false claim is fixed, never
  softened.
- **Order.** The viewer from step 1, knowing only their list, reads the plan in order and
  writes every word, symbol, unit, property, identity and construction each animation uses,
  with where it was introduced or "not yet", plus every move with no reason and every
  switch between two versions of a thing with no bridge, in `order.md`.
- **Frames.** It reads the key frames and renders any not yet sketched that look doubtful.
  At each one it writes what is lit, how much of the frame is symbols, and where the eye
  goes, and it marks every frame where symbols take the eye from the subject. It checks the
  enacted, spoken and placed count, and that each placed line names its reason. It writes
  `frames.md`.

Fix everything they find before the Editor reads the plan.

### 11. Editor

A fresh agent reads the plan as the viewer from step 1, with `EXEMPLAR.md` beside it, and
answers each question. Fix what fails.

1. Do the first ten seconds make me want the answer, and does every stretch stay on that
   question?
2. Do I reach point B, and could I then say each spine sentence and write each key result?
   Which animation taught each one?
3. For each load-bearing picture: is it as true, as load-bearing and as watched as the A1
   opening? What is missing?
4. With the voice off, do I see the evidence? With the picture off, do I get the reason?
5. Is any math placed that the picture could have enacted, or written that could have been
   said? Is any explanation written as text?
6. Did I meet everything before it was used, and was I told why at each move?
7. Is anything padded, or cut short before it finished its work? Is any introduction or
   bridge missing?
8. Did all three checks pass, with every unverified claim marked?
9. Was any picture weakened to fit phasor? Restore it under "Wanted additions".
10. Does anything go against a line of `TASTE.md`?

If the path itself is weak, go back to step 3. Do not patch it.

### 12. Hand back

Write `plans/NAME.md` in this shape, show it to the person, and stop. Do not start
`/scene-builder` until they approve.

```markdown
# NAME: the title

**Point A:** the opening question
**Point B:** what the viewer understands or feels at the end
**Viewer:** ... (what they know, as a list)   **Depth:** ...   **Feel:** ...   **Length:** ...
**Math carried:** N enacted, N spoken, N placed

## The spine
The two to four sentences, each with the animations that prove it.

## What the video stands on
Each load-bearing concept, the picture that establishes it, and the later animations that
use it.

## The path
Three or four sentences. Runners-up: the two best paths that lost, one line each, and why.

## Look
The central picture. What colour encodes. Where written math sits, if any. The music bed.
The stills: `plans/NAME.work/sketches/`.

## Animations
### 1. What fills the stage
- **Serves:** which spine sentence; **takeaway:** what the viewer keeps
- **Load-bearing:** the concept it establishes, if any, and what stands on it later
- **While it holds:** how it changes, in order, with the verb for each motion and where it
  lands
- **Introduces:** each new word, symbol, unit or property, and how
- **Say before:** what to watch for, why it matters, and the reason for any move
- **Math:** each item, how it is carried (enacted: by which change in the picture; spoken;
  placed: for which of the three reasons), and the words it lands on
- **Key frames:** at fixed times, what is lit, what is symbol, where the eye goes
- **Watched stretch:** if any, how long, and the one thing to watch
- **Say after:** why it is so, and what it means for the spine
- **Sound:** ...
- **Becomes the next by:** ...

## Math inventory
| Animation | Math | Carried (enacted / spoken / placed: reason) | Picture that earns it |

## Concept order
| Concept | Introduced in (how) | First used in |

## Claims
Each number, demo and derivation: checked (how), or unverified (what test, and what to say
if it fails).

## Left out
What was simplified away.

## Wanted additions
Each picture or verb phasor does not have yet: what it should do, where the plan uses it,
and the closest thing that exists today.
```

Add each wanted addition to `WISHLIST.md` as its own line. Keep `map.md`, `weaves.md` and
`sketches/`; a later video on a nearby idea can start from them. In your reply give the
path in a few sentences, the animations by title, the stills, the math counts, the wanted
additions and the file.
