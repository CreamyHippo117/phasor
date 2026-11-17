---
name: phasor-milestone-gate
description: Run before declaring any phasor milestone (or a large chunk of one) done, before opening or merging a milestone PR, or when asked "is milestone N done?", "review this milestone", or "check against the spec". Verifies tests, layer rules, the exit demo, the seven core contracts and docs, and refuses to call the milestone done if anything fails.
---

# phasor milestone gate

Purpose: stop errors from compounding across milestones. A milestone is done only when
every step below passes. Report honestly; never mark an item passed that you did not run.

## Steps

1. **Identify the milestone.** Read the "Roadmap and milestones" table in `SPEC.md` and the
   current `MILESTONE_N.md`. Write down its scope and its exit demo.

2. **Run the mechanical gates** and paste the summary lines:
   ```bash
   uv run ruff check . && uv run ruff format --check .
   uv run lint-imports
   uv run pytest -q
   uv run phasor doctor
   ```
   Any failure means NOT DONE. Skipped `gl`/`gpu` tests are not failures, but list them:
   they are untested on this machine.

3. **Run the exit demo** exactly as the roadmap states it. If it needs hardware this machine
   lacks (Windows, CUDA), say so and mark it "not verified here", never "passed".

4. **Contract audit.** Prefer a fresh subagent that did not write the code. Give it
   `SPEC.md` "Core architecture contracts", `CLAUDE.md` and the branch diff
   (`git diff main...HEAD` or since the previous milestone tag) and ask:
   "List every deviation from the seven contracts, with file:line. For each, say whether it
   is recorded in MILESTONE_N.md under Deviations." Check in particular:
   - any write path other than `Engine.apply`;
   - any state that depends on something other than (document, t);
   - any lower-layer import of render/GPU code (even via protocol leakage);
   - any object type missing a `SceneObject` member;
   - any second way of doing something the API already does.

5. **Docs and artifacts.** Confirm, for this milestone:
   - `docs/ai/API.md` covers every new public name, one example each, and those blocks run;
   - `examples/` has the demo scene; golden frames exist where rendering exists;
   - `MILESTONE_N.md` lists Built / Deferred / Deviations / Untested-on-hardware;
   - any SPEC disagreement was fixed in SPEC in the same commit (contract changes need the owner).

6. **Verdict.** Output a table: step, result (pass / fail / not verified here), evidence.
   Then one line: `MILESTONE N: DONE` or `MILESTONE N: NOT DONE: <blocking items>`.
   If not done, fix and re-run the gate, at most 3 rounds, then report what is still open.
