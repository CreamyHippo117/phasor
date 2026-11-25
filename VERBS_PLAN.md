# Verbs, second set: status and what is left

27 new choreography verbs in six groups, **all built and merged** on
`claude/adoring-lovelace-5ddkue` (PR #18), 2026-11-25: scale (`grow`, `shrink`, `squish`,
`flatten`), algebra (`act`, `apply`, `generate`, `coset`, `quotient`, `permute`), equations and
emphasis (`substitute`, `cancel`, `factor`, `expand`, `ignite`, `pulse`), camera and light
(`follow`, `ripple`, `spotlight`), signals (`iterate`, `decompose`, `recompose`) and space
(`lift`, `slice`, `project`, `dissolve`, `assemble`). Each has its page in
`docs/ai/verbs_<group>.md`, its timing in every theme, an example in `examples/verbs_<group>.py`
with a software golden, and tests in `tests/test_verbs_<group>.py`. The full suite passed
(2008 passed, 162 skipped, on Linux); Windows, CUDA and macOS are untested.

## Waiting on the owner

- **Confirm `ParamsBase.stretch`** (the scale verbs): a universal, verb-owned per-axis scale
  of what an object draws, applied centrally in the renderer's collector and in the
  `world_bounds`/`features` that `register` wraps. A defaulted parameter, like `flash`, so no
  schema bump, but it sits on every type's path; the SPEC note asks for confirmation
  (contract 5). With the identity stretch every existing golden is unchanged.
- **The prism audit fixes** (no glow was widened; some things need not glow). Ranked:
  (1) route `Dot` and the `Tracer` head through `strokes.disc`, and give `strokes.polyline(s)`
  a per-point opacity option, so hue by place reaches them (Tracer trails, `Directivity` and
  the phased-array beam pass explicit colours only to fade); (2) a hue opt-out for colours
  that carry meaning (Cayley generators, cosets, cycles, vector-field magnitude, FixedPoints
  rays, the `Convolution` f/g pair) before widening `[strokes] only` in the prism theme;
  (3) types that ignore their glow role (subgroup lattice, Galois diagram, Homomorphism,
  Cayley nodes, `TransformGrid` basis arrows and cube edges); (4) nodes for `ComplexMap`,
  `Network` (`node_size` 0 by default), Epicycles joints, 3D room corners; (5) solid fills for
  `Polygon`, `RiemannSum`, the `Convolution` product, room walls. Also: two-point lines blend
  colour between their ends (`EigenLines` needs a vertex per crossing); hue by place barely
  changes across a small object (scale the cycles to it); the image frame and spectrogram
  playhead glow at a fixed 2.5.
- **Chroma vs prism**: chroma is an option beside prism; suggested use is prism for analysis,
  signals and fields, chroma for algebra and structure, switching at section boundaries.
- **GPU goldens** for `chroma_algebra`, `chroma_domain` and the verbs examples, and Windows,
  CUDA and macOS runs, belong on the owner's PC (software goldens only so far).
- Not covered from the references: the stained-glass hyperbolic tiling (white ground, filled
  tiles): phasor has no tiling object.

## Known limits of what was built

- `decompose` works on `fourier.PartialSum` only (not epicycles) and draws the first 8
  harmonics as rows.
- Text cannot be squished through `ctx.text`: it scales as a picture.
- Two verbs on one object's track must be written in time order (the engine's `add_segment`
  rule); verbs on different objects may come in any order.
