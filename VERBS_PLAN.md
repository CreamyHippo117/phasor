# Verbs plan: the second set of choreography verbs, and what to continue

Written 2026-11-25 when the owner ran low on credits. Everything below is state as of that
moment; nothing in the unfinished groups has been reviewed or merged.

## Where it stands

27 new verbs in six groups. **Phase 0** (committed, pushed, on PR #18): every verb has its
`[motion.<verb>]` timing in the four base themes (`prism_vivid` and `chroma` inherit), an
entry in `theme.VERBS` and `workspace/tools.py` `VERBS`, a stub in `time/verbs.py` `Verbs`
and a wrapper in `scene.At`, each inside its own `# ===== group X =====` block. The animate
tool passes keyword options as a dict in `value`, ids as `targets`. A stub raises
`VerbError("<verb> is not built yet")`.

| Group | Verbs | State |
|---|---|---|
| F2a dynamics and signals | `iterate`, `decompose`, `recompose` | **Built, merged and pushed** (`dyn.Orbit`, `PartialSum.spread`); its targeted tests passed. |
| S scale | `grow`, `shrink`, `squish`, `flatten` | Stopped near the end. One WIP commit; patch `pending/verbs_wip/scale.patch`. |
| A algebra | `act`, `apply`, `generate`, `coset`, `quotient`, `permute` | Stopped while fixing its tests. Patch `pending/verbs_wip/algebra.patch`. |
| E equations and emphasis | `substitute`, `cancel`, `factor`, `expand`, `ignite`, `pulse` | Stopped about to run its checks. Patch `pending/verbs_wip/equations.patch`. |
| F1 camera and light | `follow`, `ripple`, `spotlight` | Stopped committing tests and docs. 3 commits. Patch `pending/verbs_wip/camera.patch`. |
| F2b 3D and points | `lift`, `slice`, `project`, `dissolve`, `assemble` | Stopped writing docs. Patch `pending/verbs_wip/space.patch`. |

The five WIP patches are **unreviewed model output**. Read each diff before merging; the
agents were stopped mid-task, so a patch may fail its own tests or leave a verb as a stub.
The "State" lines come from each agent's last message, not from a test run.

## To resume any WIP group

Run from the repository root on branch `claude/adoring-lovelace-5ddkue`:

```bash
git apply --3way pending/verbs_wip/<group>.patch
uv sync --extra gl --extra encode --extra mcp
uv run pytest -q tests/test_verbs_<group>.py tests/test_motion.py tests/test_docs.py
```

The test files are `test_verbs_scale.py`, `test_verbs_algebra.py`, `test_verbs_equations.py`,
`test_verbs_camera.py` and `test_verbs_space.py`. If a patch conflicts in `time/verbs.py` or
`scene.py`, keep both sides: each group only owns its own marked block. Then read the diff,
run `ruff check .`, `ruff format .` and `lint-imports`, and commit one group at a time.
Delete a patch from `pending/verbs_wip/` when its group is merged.

The brief every agent worked from (rules, files it may not touch, deliverables) is in the
section "The shared brief" at the end of this file.

## Per group: what to finish

**S: scale (`grow`, `shrink`, `squish`, `flatten`).** The one that touches core, so review it
first. Design (settled): a verb-owned `stretch: Vec3 = (1, 1, 1)` on `ParamsBase`
(`core/objects.py`), like `dim`; applied centrally to recorded geometry in both the recorder
(`core/record.py`) and the renderer's `_Collector` (`render/passes.py`) about a pivot (the
object's `pos`, else its bounds centre); types with their own `world_bounds`/`features` are
transformed once in `register()`. The patch touches `core/objects.py`, `render/passes.py`
and `tests/test_open_contract.py`; check that `core/record.py` is handled too, or the
inspector disagrees with the picture. Remaining: (1) confirm that with `stretch == (1,1,1)`
every existing golden is **pixel-identical** (run `uv run pytest -k golden`; do not
regenerate any golden); (2) check the bounds and inspector for a stretched Dot, Line,
Polygon, Equation, TransformGrid and DomainColoring; (3) round trip `squish(to=1.0)`;
(4) text cannot be squished through `ctx.text`: it moves and scales uniformly only, document
that; (5) add the SPEC.md "As built (motion verbs)" note (the patch does not touch SPEC.md).
Needs owner OK? It adds a universal param (precedent: `flash`, commit 62e1aa1, no schema
bump) but touches the object/render path every type uses, so confirm before merging.

**A: algebra (`act`, `apply`, `generate`, `coset`, `quotient`, `permute`).** Remaining: get
`tests/test_verbs_algebra.py` green (the agent was fixing tests that read the inspector's
`semantic` key), run the algebra goldens unchanged, check that `generate` on a Cayley graph
and `permute` on `Permutation` look right in the example's golden, and that colours with
meaning (generator, coset, cycle) are kept. `apply` must compose onto the matrix at the
anchor (`new = M @ current`) and stay order independent.

**E: equations and emphasis (`substitute`, `cancel`, `factor`, `expand`, `ignite`,
`pulse`).** Remaining: run its tests, the docs tests, `test_motion.py` and the equation
goldens (they must not change); look at the example golden for the strike-through of
`cancel`; check `factor`/`expand` reuse `transform` rather than copy it; `pulse` stays a
tagged, order independent repeat of `flash`.

**F1: camera and light (`follow`, `ripple`, `spotlight`).** The risk is `follow`: a verb sees
only the compiled document and the state at its anchor, but a feature (a zero, a tip) moves.
Read the report the agent would have given: check whether `follow` was built through the
camera goals in `layout/camera.py` (the patch touches `layout/bake.py`, `layout/camera.py`,
`layout/plan.py` and `core/pycode.py`) and whether that respects contract 6 (the solver runs
only inside the bake and sets only camera parameters the document leaves unset). If it
bends a contract, stop and ask the owner. `spotlight` is a new object type in
`domains/lighting.py`: it needs semantics, bounds, an `API.md` entry and a golden.
`ripple` makes a child `<id>.ripple` like `circumscribe` makes `<id>.ring`.

**F2b: 3D and points (`lift`, `slice`, `project`, `dissolve`, `assemble`).** Remaining:
write/finish `docs/ai/verbs_space.md`, run the geometry, pde and cloud goldens unchanged,
check that poles do not blow up the `lift` mesh, and that `dissolve` samples the object's
geometry as it is at the anchor and is a pure function of (document, t) (`rng(id, seed)`).
New types are in `domains/space.py`; `project` uses `linalg.Arrow` and `Line` as children
(`linalg.py` belongs to group A).

## Integration, once the groups are merged (do this once, at the end)

The shared files the agents were told not to touch:

1. `kit/examples.toml`: one entry per example (`verbs_signals` is done). The test
   `test_every_example_is_sorted_into_the_catalogue` fails without them. Kinds: basics, complex, fourier_linalg, fractals, motion, physics,
   three_d.
2. `docs/ai/API.md`: add `dyn.Orbit`, `PartialSum.spread`, and one entry for every other new
   object type or parameter (the per-group notes are in `docs/ai/verbs_<group>.md`).
3. `docs/ai/MCP.md` and `SPEC.md` ("As built (motion verbs)"): list the new verbs.
4. `llms.txt`: `uv run python -m phasor.mcp.llms` (needs `--extra mcp`). CI fails on a stale
   one (it did twice this session).
5. Run the whole suite once (about 30 minutes) and `ruff`, `lint-imports`; then push.

## Also waiting on the owner

- **The prism audit fixes** (no glow was widened; the owner said some things need not glow).
  Ranked: (1) route `Dot` and the `Tracer` head through `strokes.disc`, and give
  `strokes.polyline(s)` a per-point opacity option, so hue by place reaches them (Tracer
  trails, `Directivity` and the phased-array beam pass explicit colours only to fade);
  (2) a hue opt-out for colours that carry meaning (Cayley generators, cosets, cycles,
  vector-field magnitude, FixedPoints rays, the `Convolution` f/g pair), before widening
  `[strokes] only` in the prism theme; (3) types that ignore their glow role (subgroup
  lattice, Galois diagram, Homomorphism, Cayley nodes, the `TransformGrid` basis arrows and
  cube edges); (4) nodes: `ComplexMap` (the closest to the lattice), `Network` (`node_size`
  0 by default), Epicycles joints, 3D room corners; (5) solid fills for `Polygon`,
  `RiemannSum`, the `Convolution` product, room walls. Also: lines that blend colour between
  two points need a vertex per crossing (`EigenLines`); hue by place barely changes across a
  small object (scale the cycles to the object), and the image frame and spectrogram
  playhead have a fixed glow of 2.5.
- **Chroma vs prism.** Chroma is an option beside prism. Suggested use: prism for analysis,
  signals and fields, chroma for algebra and structure, switching at section boundaries.
- **GPU goldens** for `chroma_algebra`, `chroma_domain` and the new verbs' examples, and
  Windows/CUDA/macOS runs, belong on the owner's PC (software goldens only so far).
- **Not covered from the references:** the stained-glass hyperbolic tiling (a white ground
  and filled tiles): phasor has no tiling object.

## The PR

[PR #18](https://github.com/CreamyHippo117/phasor/pull/18) (draft) carries the chroma look,
white-hot domain coloring, hue by place, prism linear algebra, verbs phase 0 and group F2a.
CI was green on `cfe136a`; later pushes restarted it. A check-in was scheduled for about
06:20 UTC; with the session over, look at the PR's checks by hand. A stale `llms.txt` or a
missing `kit/examples.toml` entry for `verbs_signals` are the likely failures.

## The shared brief (what each agent was told)

Add the verbs in your own marked block in `time/verbs.py` and `scene.py`; use `_segment`,
`_span` and `_instant` so segments carry the verb tag (strict lint reports `raw_motion`);
timing comes from `self.motion["<verb>"]`; the result must not depend on the order verbs are
written in; everything is pure (randomness only through `rng(object_id, seed)`); no new GPU
kernels; a new object type needs Params, `draw`, bounds/semantics, an `API.md` entry, an
example and a golden; do not widen glow on existing types; stop and report if a contract
(CLAUDE.md, the seven) would change. Do not touch themes, `theme.py`, `workspace/tools.py`,
`llms.txt`, `SPEC.md`, `kit/examples.toml`, `API.md`, `MCP.md` or another group's block.
Deliver: the code, `tests/test_verbs_<group>.py`, `docs/ai/verbs_<group>.md` with runnable
python blocks, `examples/verbs_<group>.py` with a software golden, and a report.
