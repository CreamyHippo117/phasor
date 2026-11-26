---
name: idea-scout
description: Hunt for video ideas worth a full scene plan, without writing scenes or plans. A panel of specialists (geometer, algebraist, combinatorialist, ...) each proposes a mechanism and one picture for a topic, a separate attacker tries to kill each one, and only survivors reach the ledger in `ideas/`. Use when asked to "scout ideas", "find video ideas", "run the idea scout", or on an unattended scheduled run.
---

# idea scout

Purpose: raise the hit rate of the expensive step (a full scene plan, done by the owner) by
killing weak ideas cheaply first. A good run ends with a few survivors and many recorded
deaths. A run that kills everything is a valid result; say so and stop.

An idea is **a claim, the reason it is true, and the one picture that shows the reason**.
A topic ("aliasing", "the DFT and group theory") is not an idea and never reaches the ledger.

## Files (all in `ideas/`; write nowhere else)

- `ideas/VERDICTS.md`: the owner's verdicts on past candidates and why. Read it first, every
  run. Only the owner's words go in it; never add a verdict of your own.
- `ideas/LEDGER.md`: live candidates, newest first, plus the owner's "Topics to try" queue.
- `ideas/DEAD.md`: killed ideas and exhausted topics, with the lenses tried and the reason.
- `ideas/sketches/`: throwaway stills for step 6. Not scenes, not examples, no tests.

## Budget for one run (unless the person gives another)

- At most 6 topics.
- 4 lenses per topic, one card per lens.
- One attack, one repair, one second attack. Then the idea is in or out.
- At most 5 new survivors in the ledger, at most 2 stills.

## Steps

1. **Read the record.** `ideas/VERDICTS.md`, `ideas/LEDGER.md`, `ideas/DEAD.md`. Write down
   in your own words what the owner's verdicts say makes an idea hold up or fall over. If
   `VERDICTS.md` is empty, say in the run report that the scout is running without taste.

2. **Pick topics.** First from "Topics to try" in the ledger. Then your own choice: about
   two in three from signal processing and its neighbours (sampling, filtering, transforms,
   estimation, coding, waves, arrays), the rest from anywhere in maths. Skip a topic that
   `DEAD.md` marks exhausted unless you bring a lens it has not seen. Check `examples/` and
   `ANIMATIONS.md` so you do not propose a picture phasor already ships.

3. **Pick lenses.** Four per topic from the roster, at least two from far outside the
   topic's home field: the best ideas are a topic seen through the wrong specialist's eyes.
   Never reuse a (topic, lens) pair that `DEAD.md` lists.

   Roster: geometer, combinatorialist, linear algebraist, abstract algebraist, analyst,
   probabilist, topologist, number theorist, physicist, signal engineer, algorithmist,
   information theorist.

4. **Propose.** One agent per lens, run in parallel, each given the topic, its lens, the
   taste notes from step 1 and the dead ends for that topic. Proposers do not see each
   other's cards. Each returns exactly one card, or "nothing" (which is an honest answer):

   ```
   ### <slug_named_by_meaning>
   Topic / lens:
   Claim:   one sentence the viewer will believe afterwards and did not before.
   Because: the mechanism, in one or two sentences. Not "X is related to Y".
   Frame:   the one picture, in words: what is on screen, what moves, what the viewer notices.
   Needs:   what the viewer must already know (two things at most).
   Draws with: phasor objects that could draw it, or what is missing.
   ```

   Reject at the door, without an attack: a card with no mechanism, a frame that is a
   formula on screen, or a frame that needs narration to mean anything.

5. **Attack.** A separate agent that sees only the cards, never the proposers' reasoning.
   Its job is to kill. For each card it answers these, each with pass, fail or repairable
   and one sentence of why:

   1. **True?** Is the claim correct as stated? Check it with a few lines of NumPy
      whenever it can be checked; a claim the attacker could not check is marked
      "unchecked", not passed.
   2. **Is the picture the reason?** Would the viewer see *why* from the frame alone, or
      is the frame decoration next to an argument made in words?
   3. **Fresh?** Is this the explanation every textbook and well-known video already
      gives? A standard picture is a fail unless the lens changes what it shows.
   4. **Payoff before patience runs out?** Does it need more than the two things under
      "Needs" before the frame makes sense?
   5. **Against the record.** Does it repeat something `VERDICTS.md` says fell over?

   Any fail on 1 or 2 kills the card. Two fails among 3-5 kill it. A "repairable" goes back
   to its proposer once; the attacker judges the repaired card again with the same five
   questions and there is no third round.

   **Convergence.** If two lenses arrived at the same frame on their own, or one card's
   mechanism makes another card's hard fact obvious, merge them into one card, name both
   lenses, and mark it `converged`. Rank these above every single-lens survivor.

6. **Sketch (survivors only, at most 2 per run).** Following the `phasor` skill, make one
   rough still of the frame in `ideas/sketches/<slug>.py` and `<slug>.png`. No narration,
   no motion, no polish, no lint pass. If the frame cannot be made to show the mechanism in
   a still, that is the finding: move the card to `DEAD.md` with reason "no picture".
   If it needs an object phasor lacks, do not build it: record the gap on the card.

7. **Write the record.**
   - Survivors: the card at the top of `ideas/LEDGER.md`, with the date, `status: new`,
     the attacker's five answers, `converged` if so, and the still's path if one was made.
   - Deaths: one line each in `ideas/DEAD.md`: date, topic, lens, slug, the question that
     killed it, why. If every lens on a topic died, add the topic under "Exhausted" with
     the lenses tried.
   - Remove a topic from "Topics to try" once it has been run.

8. **Report.** A short run report: topics tried, cards proposed, killed at the door, killed
   by attack, survivors (slug and claim, converged ones first), stills made, phasor gaps
   found, and anything left "unchecked". On an unattended run, commit only `ideas/` to the
   session's branch and push.

## Never

- Write or edit a scene, an example, a plan, a test, or anything outside `ideas/`.
- Let a proposer attack its own card, or let the attacker see a proposer's reasoning.
- Keep a card alive because the run would otherwise have no survivors.
- Mark "True?" as passed without a check or a named, standard result.
- Write in `ideas/VERDICTS.md`, or change a card's status from `new`; both are the owner's.
- Raise the budget to rescue a topic. Record it as exhausted and move on.
