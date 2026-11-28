# Casebook

The detailed lessons behind `SKILL.md`, kept as cases: what went wrong in a real plan, and
what fixed it. `SKILL.md` is short on purpose and says when to read each part of this
file. The checkers (the order check, the claims check, the frame check and the Editor)
check a plan against it. A generator's brief carries the part it needs, as examples, not
as a checklist to fill.

When a case here and `EXEMPLAR.md` seem to disagree, the exemplar wins.

## 1. Introduce, then use

The person's rules (2026-11-27, from a review of an LTI → convolution → FFT plan whose math
"just gets shoved at us without explanation", and 2026-11-28).

- **Read the viewer's level literally.** "Introductory ODEs and linear algebra" means a
  student who has just finished those courses, not a mathematician.
- **The concept order.** Every concept the plan uses (word, symbol, unit, property,
  operation, picture) is listed with the animation that introduces it and the first
  animation that uses it. The introduction comes first. If a later animation needs
  something, introduce it earlier or move the use later. Do not make the viewer wait.
- **Move a result to where its idea lives.** *Case:* a sinc spectrum justified a narrow
  clap in animation 3, but frequency arrived in animation 7. *Fix:* drop it from 3 and bring
  it back in 8 as "why 10 µs was enough".
- **Say the line that does the work.** When a whole stretch rests on one identity, the
  voice says it, and it is written because the viewer will look back at it. *Case:* working
  with complex tones rests on Euler's formula and on a real tone being two arrows turning
  opposite ways, cos ωt = (e^(iωt) + e^(−iωt))/2. That one line gives the reason for going
  complex and explains the negative frequencies that appear later.
- **Define before using, and use the smallest form that does the job.** A transform, an
  operator or a matrix is defined from a picture before any formula uses it. *Case:* once the
  discrete tones exist, x = (1/N) Σ X_k e_k is enough. The continuous integral with its 1/2π,
  which was never defined, stays out. Never move between two versions (continuous ω and
  discrete ω_k) without showing the bridge between them.
- **Every construction gets its reason.** A wrap, a substitution, a change of basis or a
  padding gets one sentence saying why, before it happens. *Case:* the circulant matrix came
  with no reason, so it read as a trick. *Fix:* treat a block of N samples as one turn of a
  loop, so whatever runs off the end comes back at the start. That is also exactly the
  spill the next animation shows, so the reason sets up a return.
- **Name a property where it first appears.** *Case:* "zero above ↔ causal" in animation 6,
  with causality never named. *Fix:* one line in animation 4: echoes only come after the
  clap, so h(t) = 0 for t < 0.
- **Show what can be shown, briefly.** *Case:* "a mirror image of the source" needs one wall,
  with the bounced path folded flat into a straight line: ten seconds, the A1 unfolding in
  miniature.
- **Say what is left out.** *Case:* the plan switched from h(t) to h[k] without a word.
  *Fix:* "h[k] is h sampled every 1/48 000 s; sampling theory stays out of this video".
- **The thing first, then its name.** *Case:* "the impulse response is ...". *Fix:* "strike the
  room once, as short and sharp as you can, and listen. A strike like that is what
  mathematicians call an impulse." The name labels something the viewer already holds.
- **Build a bridge from what they know.** *Case:* an FFT for a viewer who knows ODEs and
  linear algebra. *Fix:* the roots of unity first: e^(iωt) from their ODE course, sampled N
  times around one turn, lands on N points of the circle; the powers of one of them fill the
  DFT matrix, F_jk = ω^(jk); only then the trick that splits it in half. Build it on the
  central picture where it can go, here the same plane the video already lives in.
- **Small terms get one spoken phrase.** "decibels, a log scale where every 20 dB is ten
  times the pressure"; "sound pressure level, loudness measured against the quietest sound
  we hear"; "the same sound spreads over a sphere, so its pressure falls as 1/distance".

The cost of an introduction matches the size of the idea: a phrase for a unit, a line for
a property, about ten seconds of picture for a geometric fact, a whole animation for a
load-bearing concept. A plan with no room for its introductions has too many ideas. Cut an
idea before cutting the introduction of one it keeps. No agent cuts an introduction or a
bridge as padding. It goes only when the viewer's list already holds its ground, and the
agent that cuts it says so.

## 2. Carrying the math

From "A Room Is a Matrix" (2026-09-30), the reviews of 2026-11-27 and 2026-11-28.

- **Let the picture argue.** The viewer sees that something is true instead of being
  told. *Case:* to show two operations are the same, run them side by side on identical
  objects and let them move identically (2swap's "×i ×j" beside "×ij").
- **Enacted, not placed.** *Exemplar:* the mirror rooms of A1 become the image-source sum,
  one room per term. *Case of placed:* a convolution formula written in a band while the
  picture shows a room it does not come from. The frame got uglier and the viewer learned
  no more than the voice could have said. *Fix:* make the picture do the step (slide one
  copy of the clap along the other and let the overlaps add up), or say it.
- **The voice explains; nothing explanatory is written.** No sentences, captions, bullet
  points or definitions in words on screen. An equation that appears with nothing said
  about it is "shoved on screen". So is one the voice explains only after the viewer has
  stopped reading it.
- **Picture, then words, then symbols.** Each term is assembled from the object it names: a
  spike becomes δ, a diagonal becomes h[k], a slider becomes H(ω).
- **One equation growing at a time.** Older results stay readable, but only one is being
  built.
- **A derivation step needs a visual change.** Each written line pairs with a change in
  the picture. Pure algebra with nothing to see is spoken ("rearranging, ...").
- **The ledger.** A finished result the viewer will need again shrinks into a column at the
  edge of the frame and stays. The ending can reassemble it.
- **Numbers beside symbols.** 10 ms, 50 Hz, about 10,000 echoes a second: the viewer can
  check them.
- **Callbacks are written beside.** Where a result is one the viewer met before (the
  integrating factor, Laplace's shift theorem, guessing e^(rt)), it is written next to the
  new one, so the sameness is seen.
- **Colour ties symbol to object.** A term takes the colour of the thing it names, for the
  whole video.

## 3. The stage

- **Stage.** Midnight black. One animation, full frame: the mathematics reaches the edges,
  not a small figure in the middle of a slide. Variation comes from inside it.
- **Colour.** Hue is structure. Assign colour by something mathematical (phase, position in
  a lattice, which tile, time along a path), and then the whole wheel is welcome. Where
  colour encodes no quantity, use two or three hues, each with a meaning that holds for the
  whole video. Choose the palette from the feel.
- **Motion.** Draw things on rather than switching them on; how a thing appears shows how it
  is made. Slow is the default. During a hold something is always changing a little. One
  animation becomes the next: an object carries over, morphs, or the camera travels
  through.
- **The big moves**, how an animation renews itself while keeping the stage:
  - *One number turns.* The camera is still, one anchor stays fixed, and a single parameter
    changes slowly so the whole picture reorganises around it. The workhorse.
  - *Sweep.* The camera travels across a structure too rich for one frame. "Look how much
    is here."
  - *Lift.* The camera pulls up from a detail until the detail turns out to be one part of a
    larger pattern. Its reverse, the descent, goes from the whole into one part.
  - *Passage.* The camera goes into the object until its colour fills the frame, and comes
    out in the next animation.

  Each one shows something the viewer could not see before it began, starts and ends at
  rest, and ends on a composed frame that holds.
- **Equation places.** Where written math is kept, it has one place for the whole video:
  a band for the line being built, a column at one edge for the ledger, both clear of the
  picture's subject.

## 4. Weaves and returns

- **A weave is two ideas carried by one object**, where the second follows from the first
  inside it. *Case:* pendulums drawn to show the geometry, then heard: a stable pendulum
  beside an unstable one. The pendulum that is seen is the one that is heard. Two good ideas
  placed side by side are a staple, not a weave.
- **A return must add something:** a new question, a new sense (the picture now heard), or
  an outcome the viewer could not predict. Write in one sentence what the viewer gains from
  it. If that sentence restates an earlier point ("again, the hands line up"), drop the
  return. Of the three reverb plans, A1 was liked most because it repeats itself least.

## 5. The cast

Mathematical lenses, each asking "what is this, seen from my field?":

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
believe), archivist (how the concept was first pictured, older approaches, forgotten
constructions; a source of pictures, not a history lesson).

Craft lenses: cinematographer (the unforgettable shot, from imagination; it does not read
phasor's verbs), musician (what can be heard), phasor expert (what phasor already draws
beautifully; reads `kit/docs/API.md` and `kit/examples/INDEX.md`), 2swap student (what
2swap would do), textbook writer (which results carry the idea, which ones the viewer
already knows that it is secretly the same as, and which bridges a viewer at this level
needs).
