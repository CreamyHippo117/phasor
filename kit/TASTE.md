# Taste

What the person likes and dislikes, kept as fixed points that accumulate. Read it before
every scene; it outranks the examples.

Each entry is dated and is one of two kinds, with the case it came from:

- **Keep**: something that worked and must survive every later change.
- **Change**: something that failed, and what to do instead.

An entry is never deleted and never "replaces" another. When a Change seems to conflict
with an earlier Keep, it narrows it: write what still holds and what no longer does, and
keep both. A reaction to the last output tends to overshoot, so a Change says what went
wrong, not the opposite of the last rule.

## Keep

- Dark by default (`midnight_glow`); light themes only when asked.
- The reference style is the prism look (`kit/examples/complex/prism_primes.py`): glowing
  strokes, a spectrum along curves, calm pacing.
- Fewer rainbows is not no rainbows: colour that carries meaning (domain-colouring phase)
  keeps the full wheel.
- 2026-11-25: the reference for the look and the pacing of a picture is 2swap ("don't flip,
  algebra inside" 6:55-7:10; "i'm so bad at billiards that i ended up in the hyperbolic
  plane" 5:00-5:40).
- 2026-11-25: one animation owns the stage: full frame on midnight black, alone, for as
  long as it needs (iterations, new cases, a parameter turning). Full, bright colour is
  welcome when the mathematics is that full.
- 2026-11-25: a picture worth watching is given time to be watched, over ambient music,
  with one thing to watch. Never rush. (Narrowed 2026-11-27: this is not a substitute for
  explanation; see Change.)
- 2026-11-25: quality, not quantity: as many animations as the best story needs (ten or
  three), none added to fill time.
- 2026-11-25: open with a question, let the viewer predict, come back to the same picture
  with new cases; usually not concrete before abstract (ask which route).
- 2026-11-25: no hard cuts; one animation becomes the next. Not much text or equations
  laid on the screen. (Narrowed 2026-11-27 and 2026-11-28: what was missing was more
  explanation and more math that the picture enacts or the voice says, not more written
  on the screen.)
- 2026-11-27: no repetition unless it adds something: a picture, case or point comes back
  only with a new question, a new sense or an outcome the viewer could not predict. (Of the
  three reverb plans, A1 was liked most because it repeats itself least.)
- 2026-11-27: the model for explanation is the person's own "A Room Is a Matrix": a spine
  of a few sentences, each equation earned by a visual and built term by term, a ledger of
  results, worked numbers, earlier courses written beside new results. No padding; length
  follows content. Every claim is checked.
- 2026-11-27, sharpened 2026-11-28: **the opening of reverb A1 is the exemplar**, the clap
  in the room unfolding into its crystal of mirror rooms. It established LTI, the concept
  the explanation of reverb through ODEs, linear algebra and the FFT stands on, while being
  literally beautiful. The picture was the derivation (the method of images). Breathtaking
  pictures get segments of their own when a later step stands on them, as here. See
  `.claude/skills/scene-plan/EXEMPLAR.md`.

## Change

- 2026-11-27, from the three reverb plans: the plans had become "low quality content
  enshrined in glamour". Beauty drifted off the question. Do instead: more explanation,
  with the voice carrying the reasoning; silence only with one thing to watch; worked
  numbers; every stretch stays on the question.
- 2026-11-27, from the LTI → convolution → FFT plan: the math "just gets shoved at us
  without explanation". Do instead: introduce every piece of math before it is used and
  give every move its reason. "An audience familiar with ODEs and linear algebra at an
  introductory level" means just that, not Terence Tao. The cases: no spectrum before
  frequency exists; state Euler and cos ωt = (e^(iωt) + e^(−iωt))/2 before going complex;
  define X before using it, and stay discrete rather than jump to continuous integrals; say
  why a block is wrapped into a circle; name causality where it first appears; show an
  image source in ten seconds rather than assert it; say in one sentence that h[k] is h
  sampled; one phrase each for dB, SPL and the 1/distance loss.
- 2026-11-27: math had come with text explanations on screen. Do instead: the voice
  explains, and the math on screen is built as the voice talks about it. No explanatory
  sentences, captions or bullet points on screen.
- 2026-11-28: terms were named before they were shown. Do instead: a new term arrives as
  the thing, in plain words and in the picture, and only then gets its name ("a single
  sharp strike ... mathematicians call it an impulse"). When an idea stands on ground the
  viewer lacks, build a bridge from what they know first (the FFT for a viewer of ODEs and
  linear algebra: the roots of unity first). Bridges are never cut as padding.
- 2026-11-28, from the plans written after the math rules: "more math" had become more
  equations laid on the screen, and frames got ugly; beauty and math were scored as two
  dials, so each bought the other; judges read prose and never saw a frame. Do instead:
  the picture enacts the math where it can, the voice says the rest, and only results to
  look at again are written. Stay on the question. Beauty and teaching are gates, not
  weights. Judge rendered frames, not paragraphs. Hold every plan against the A1 opening.
