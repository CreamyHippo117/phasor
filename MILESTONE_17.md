# Milestone 17: text as shapes, morphing anything, animated style

Plan: `MANIM_PARITY.md`, "Milestone 17". Exit demo: **text morphs into a shape, a formula
rearranges with `transform_matching`, a color animates between roles, and every new type and
verb has inspector output, lint, docs, an example and a golden; every count is bounded and
fuzzed** (`examples/text_shapes.py`). The independent audit (2026-11-23) found that two
were not (cached plans, the in-between's fill): see "Independent audit".

Status: **built on Linux (llvmpipe OpenGL, no CUDA); focused suites green (see
Verification); independent audit done 2026-11-23, its findings fixed.** Untested on
Windows, macOS and CUDA. The full suite was not run (over an hour here; CI runs it).

## Built

| Area | Where | What |
| --- | --- | --- |
| Style transitions | `time/compile.py` (`StyleTrack`, `StyleEnd`, `ANIMATED_STYLE`), `core/schema.py` (`Document.style_track`), `core/ids.py` (`split_style_path`), `core/patch.py`, `core/state.py` (`ObjectState.restyling`) | a track on `<id>.style.<key>` moves an `emphasis` (color), `fill`, `stroke` or `text` key between roles or explicit values (Milestone 18's `style/explicit.py` resolves each end). Colors mix in linear RGBA; an explicit world size is turned into a share of the frame height at t through the camera; easings that overshoot never make a channel negative or an opacity above 1. The document stores a role or a value at the key and in each segment (a hex target canonical, upper case); `state.roles[key]` is the typed value nearer in time; glow, surface, colormap and dash keys are refused (`bad_style_track`). `remove_object` drops the object's style tracks. A real child object named `<id>.style` keeps its own parameter paths (existence first) |
| Style transitions in tools | `time/verbs.py` (`morph`), `inspect/lint.py` (`raw_motion`), `inspect/aesthetic.py`, `inspect/inspector.py` | the `morph` verb takes a style path (theme timing, strict-lint clean); `raw_motion` names the object, not `<id>.style`; `off_palette`/`off_scale` judge an explicit target where a transition reaches it, and offer no snap for it (the snap would set the base, not the target); the inspector shows `restyling: {key: {from, to, progress}}` |
| Morph geometry | `domains/morph_math.py` | pure NumPy: dense sampling of cubics (bounded per subpath), canonical order, counts matched by splitting the longest piece, optimal pairing by travel (scipy, up to `MAX_ASSIGN`), arc-length resampling, alignment (rings keep the target's winding and roll to least travel by FFT cross-correlation; open pieces run the shorter way; ring/open open where the points move least), growth from and collapse to centres, identity for identical paths; `rings` (pieces joined as either side's subpaths, for fills); `signature` (a shape at any place and size) and `digest` (cache keys); the `Shape`/`Group`/`Look` data an outline is described by. Bounds: `MAX_SUBPATHS` 512, `MAX_POINTS` 512, `MIN_POINTS` 8, `MAX_TOTAL` 40 000, `MAX_ASSIGN` 256, `SAMPLE_CUBICS` 2048, `SAMPLES_PER_CUBIC` 64 |
| Morph drawing | `domains/morph.py` | `shape_of` (any object with `morph_shape`, or a composite whose parts have one: an SVG), `frame` (whole or matching), `pair_groups` (equations: keys first, then equal TeX in reading order; anything else: by glyph key or path signature in reading order), arcs for pairs carried past others (`basic.crossing`), `paint` (fill then stroke; the fill crossfades from the source's layout, even-odd, to the target's, its rule, so both ends are exact), `facts` (semantics, lint). Plans cached by the content digest of both sides, in their own byte-bounded pool (`morph.PLAN_BYTES`, after the audit) |
| Paths morph | `domains/path.py` (`MorphParams`, `PathBase.morph_shape`, draw, features, semantics) | every path and shape takes `morph_from`, `morph`, `morph_kind`; mid-morph it draws the in-between; `region`/`curve` features are left out while morphing; `morph*` are not geometry (the cache is not re-keyed every frame) |
| Glyph outlines | `domains/glyph_math.py` | `outline(text)`: plain text in Aileron (the bytes Pillow bundles, read by matplotlib's FT2Font), kerned and stretched to the picture's width; math through mathtext (`MathTextParser("path")`, Computer Modern) with fraction bars and radicals; TrueType quadratics converted to equal cubics; glyphs named by meaning (`cap_h`, `two`, `plus`, `pi`, `bar`) and keyed by font glyph; `MAX_GLYPHS` 400; cached per text; a lock around matplotlib's shared fonts; a character the font lacks is skipped |
| Text and Equation as shapes | `domains/basic.py` | `as_shapes`, `fill_opacity`, `stroke_opacity` and the `MorphParams` on both, the glyph outline `OUTLINE` of the line height wide; `build` gives one `Path` part per glyph (names via `svg.part_names`, clear of `state.reserved` through `svg.reserved_under`), none while the text is becoming something (it draws the in-between itself); an equation's `colors` color its parts' glyphs; `write` draws each glyph's outline then fills it, glyph after glyph (`written`, `WRITE_SPAN`); a style transition on the text reaches its glyphs (an explicit linear color or world size at t); bounds cover what a morph draws; semantics `as_shapes`, `glyphs`, `glyph_limit`, `morphing_from`, `morph`, `morph_kind`, `morph_missing`, `morph_limit` |
| Verbs | `time/verbs.py`, `scene.py` | `transform(a, b)`: two equations drawn as pictures keep the parts transform of before (same ops); two equations with the target `as_shapes` use matching; anything else with an outline morphs whole. `transform_matching(a, b)` (new). Both show the target, hide the source, switch `morph_from`/`morph_kind` at the instant (tracks: an object can become itself several times) and run `morph` 0 to 1 in the theme's `transform` timing; they refuse a source with no outline, a target that cannot be morphed into, and the same object twice |
| Lint | `inspect/lint.py` | `glyph_limit`, `morph_missing`, `morph_limit` (warnings, with `times`, no `fix`; no threshold: module constants), tested in both lint MODES |
| Docs | `docs/ai/API.md`, `llms.txt` | the verbs table, the lint table, "Style transitions" (running block), "Text as shapes and morphing" (running block, every bound named) |
| Example | `examples/text_shapes.py`, `kit/examples.toml` | prism look; strict lint clean; golden `tests/golden/software/text_shapes.png` (looked at) |

## Verification (Linux, 4 cores, llvmpipe OpenGL 4.5, no CUDA)

Tests written first or with the code:

- `tests/test_style_motion.py` (25): roles and explicit values mixing in linear light, fill /
  stroke / text sizes and world sizes through a zoom, order-free and pure, overshooting
  easings clamped, refused tracks (glow, bad roles, unknown keys, unknown objects), NaN /
  inf / huge ints refused cleanly, `remove_object`, the `<id>.style` child, inspector,
  `morph` verb strict-clean, `raw_motion` naming, `off_palette` on a target in both MODES,
  save/load.
- `tests/test_morph.py` (15): the plan's identity, least travel, splitting, pairing,
  winding of holes, reversed rings, closed/open, open lines, growth and collapse, zero
  length, order independence, bounds and time at the limits (812 subpaths, a 200 000-cubic
  subpath, growth from nothing), signatures and digests.
- `tests/test_glyphs.py` (37): outlines valid, closed, inside the box, named and keyed,
  covering the picture's ink, quadratic-to-cubic exact, bounded; `as_shapes` parts,
  inspector `part_of`, text without `as_shapes` unchanged (one `text` command, the same
  semantics), parts inside the box in 2D and orbit, part names clear of document ids, write
  outline-then-fill left to right, write order-free, equation colors, a style transition
  reaching glyphs, `glyph_limit` in both MODES, bad opacities and `morph_from` refused,
  a fuzz over awkward texts (spaces, Unicode, long, big sums) through state, inspect, lint.
- `tests/test_transform.py` (28): text into a circle with exact ends, a shape into text then
  its parts, identity, fill and color mixing, five awkward pairs (hole, open/closed, zero
  length, a fraction into a ring), a 60 000-point scribble in bounded time, a moving source
  followed purely, order-free documents, becoming twice, verb refusals, the equation
  picture transform unchanged, the formula by TeX with the arc, keyed parts morphing,
  glyphs by shape, `morph_missing` and `morph_limit` in both MODES, save/load, an SVG
  source, clamped easings, the example and its golden.
- Changed: `tests/test_motion.py::test_transform_refuses_what_is_not_an_equation` encoded
  the old equations-only rule; it now checks what has no outline is refused.

Suite results: see "Gate" (counts above are from the build; test_glyphs grew to 51 and test_transform to 28 with the review fixes).

## Deviations

1. **Two ways to rearrange a formula.** `transform` between two equations drawn as pictures
   keeps its exact behaviour (parts pair by keys then by longest common subsequence;
   changed text crossfades) so saved scenes do not change. `transform_matching`, and
   `transform` when the target is `as_shapes`, draw outlines and pair equal TeX in any order
   (a part moved across the `=` glides without a key); a keyed part whose text changes
   morphs its outline. The crossfade M17 replaces is that changed-part crossfade, under
   `as_shapes`. The Manim importer's `Transform` crossfade and its `TransformMatching*`
   aliases are left to Milestone 20.
2. **The morph is drawn by the target, from the source at t**, not baked into the document
   as `become` bakes points. Nothing stale when the source moves, and no big arrays in saved
   files; the cost is a plan per (source, target) geometry, cached by content digest.
3. **Counts are matched by splitting**, not duplicating; fills crossfade from the source's
   layout of the in-between points (even-odd) to the target's (its rule), so both ends are
   exact and holes stay holes. Mid-way the two fill layers blend (a little lighter where
   they overlap).
4. **Glyph parts are absent while a text is becoming something**: the text draws the
   in-between; its parts return when the morph ends. Dashes and arrow tips are not drawn
   in between.
5. **`become` is unchanged** (its ops are in saved documents); `transform` is the general
   morph.
6. **`transform_matching` uses the theme's `transform` timing** (no new `[motion]` entry, so
   the theme files are untouched).
7. **No new style key on Text or Equation.** A glyph outline is `basic.OUTLINE` (0.04) of the
   line height wide (an explicit world width on the glyph parts), not a stroke role: a new
   style key changed the inspector output of every text, which
   `tests/golden/contract/ported_inspector.json` froze (tried, caught, reverted). So the
   inspector output of a text that is not drawn as shapes is byte-identical.
8. **Outlines come from mathtext even with the `latex` engine**; the plain-text outline is
   stretched to Pillow's width like the picture. Math outlines sit within a few percent of a
   line height of the picture.
9. **Aesthetic `off_palette`/`off_scale` skip glyph parts** (tag `glyph`): a glyph's color is
   its text's and its outline an explicit world width (`OUTLINE` of the text's height), so
   the text is judged once instead of every glyph. (As built it skipped *every* composite
   part, with the reason "a part's values come from its composite"; that is not true of a
   part a `build` gives its own explicit values. Fixed after the independent audit.)
10. **Non-overshooting easings are clamped to 0..1** (`time/easing.py`): `smoother` just below
    u = 1 gave 1 + 2e-16, which a parameter bounded by 1 (`morph`, `progress`) refused at a
    frame. No visible change.
11. **`core/values.normalize`** turns an int too large for a float into a `ValueError`
    (was an `OverflowError`) for every parameter.
12. **Style transition semantics chosen:** the typed value in `roles` switches half-way; the
    `<id>.style` path loses to a real child object of that name.
13. `svg.reserved_under` was factored out of `SVG._parts` so glyph parts share it.
14. **`glyph_limit`, `morph_missing` and `morph_limit` take no theme threshold**: they report
    a fixed bound of the code (`MAX_GLYPHS`, `MAX_COORD`, `MAX_SUBPATHS`) or a missing
    source, not a matter of taste, like `dash_limit` (CLAUDE.md asks lint rules to take
    their threshold from the theme's `[lint]` table).
15. **Plans are made in the world, not in each outline's own frame.** Pairing is by least
    travel in the world and the start of a ring by least travel too, so a plan depends on
    where the two outlines are relative to each other: a source that moves, a camera that
    zooms (a text's size is a share of the frame) or orbits makes a new plan each frame
    (measured: about 0.2-0.3 s for 400 glyphs into a circle; about 25 ms for a 34-character text). Planning
    in normalised frames would pair by shape and change still frames, so it was not done;
    plans are bounded by bytes instead (`morph.PLAN_BYTES`).
16. **The in-between's fill thins long rings**: a ring of more than `morph.MAX_FILL_RING`
    (4096) points (only the in-between makes them: the pieces of one subpath joined, up to
    40 000 points, crossing itself mid-way) keeps every k-th point before earcut. Strokes
    keep every point and the end (`morph` 1, drawn as itself) is exact; the first frame's
    fill is thinned too when one source subpath is split into many pieces (a circle into
    a long text), within a fraction of a pixel of its outline for smooth curves.
17. **A text drawn as a picture pops a little at `m = 0`**: the morph draws its outline
    from the first frame (the picture's anti-aliased glyphs become filled outlines at the
    same place). Small; recorded, not changed.

## Deferred

- Morphing *into* an SVG (an SVG can be a source only) and into `Line`/`Arrow`.
- A style key for the glyph outline width (it is `OUTLINE` of the line height); per-glyph styling.
- The importer's `Transform` / `TransformMatchingTex` / `TransformMatchingShapes` mapped onto
  the new verbs (Milestone 20).
- A live check that a source changing type between transforms stays morphable (it is
  re-read each frame; a source that loses its outline is `morph_missing`).
- Matching by TeX *inside* one math string (substrings of a single `Text`): formulas match by
  `Equation` parts.
- Reusing one plan while a source moves or the camera zooms (Deviation 15).
- `become` (`verbs._matched_shapes`/`_resample`) keeps its own resample-and-align: unifying
  it with `morph_math` would change the points a re-run script bakes into its document.
- Caching a still `as_shapes` text's glyph parts between frames: 400 `Path` parts are
  built and validated each `state_at` (about 0.28 s for 400 glyphs).

## Untested

Windows, macOS, CUDA (no GPU golden for `text_shapes`). Outlines need matplotlib's FT2Font
to read a font from bytes. As built they used API new in 3.10 (`LoadFlags`, `Kerning`, six
entries per mathtext glyph), so on 3.8 and 3.9 (allowed by `matplotlib>=3.8`) every outline
failed; the claim "3.8+" was false until the audit fix. Now `tests/test_glyphs.py` and
`tests/test_transform.py` pass on matplotlib 3.8.4, 3.9.4 (throwaway venvs) and 3.11.2.

## Advisor review fixes (before the gate)

A review by a stronger model of this session (not an independent audit) found, and the
commit after it fixes, with tests first in `tests/test_glyphs.py`:

- the outline cache read the picture's math depth (engine-dependent) under a text-only key:
  outlines now take the depth from their own mathtext parse (`test_outlines_do_not_depend_on_the_math_engine`);
- glyph parts were judged as objects: a full stop was `tiny`, a highlighted text was one
  focal point per glyph. Glyph parts carry the tag `glyph`; `tiny` and `focal_point` judge
  the text, not its glyphs (both lint MODES);
- work past `MAX_GLYPHS` was unbounded (every character loaded, every equation part
  outlined): it stops at the cap; `glyph_limit` then reports a lower bound (a 200 000-char
  text, a 600-part equation, timed);
- `Equation.arc` had no upper bound: `MAX_ARC` (10 line heights);
- morph parameters fuzzed at their bounds (`morph` 0, 1e-300, 1; `morph_from` at
  `MAX_ID` characters and pointing nowhere; both kinds; `arc` at its maximum);
- `transform_matching` was missing from the MCP `animate` tool (`workspace/tools.py`
  `VERBS`, `docs/ai/MCP.md`).

## Gate (phasor-milestone-gate, 2026-11-23, by the authoring model)

| Step | Result | Evidence |
| --- | --- | --- |
| ruff check, ruff format --check, lint-imports | pass | clean; 2 import contracts kept |
| full `uv run pytest -q` | not run | over an hour here; left to CI (as for M15, M16, M18) |
| focused suites | pass | test_final `-k "not slow"`, test_lint, test_aesthetic, test_beauty, test_mcp, test_kit, test_docs, test_open_contract, test_style_freedom, test_engine, test_time, test_layers, test_manim_import, test_shapes, test_motion, test_transitions, test_pycode, test_contracts, test_style_motion, test_morph: **888 passed, 3 skipped** (Value draws nothing x2, no PySide6), 13 deselected (slow). test_glyphs, test_transform: **79 passed** (gl tests ran on llvmpipe) |
| `phasor doctor` | tier B | llvmpipe OpenGL 4.5, CPU kernels (g++), extras installed, no CUDA |
| exit demo | pass (Linux) | `examples/text_shapes.py`: the word "circle" morphs into a circle, `a^2+b^2=c^2` rearranges into `a^2=c^2-b^2` by TeX (b^2 arcs over the `=`), the disc's fill moves from `highlight` to `danger`; `lint(strict=True)` empty; golden `tests/golden/software/text_shapes.png` looked at |
| contract audit | done by the authoring model, **not independently** (no subagent tool in this session) | 1: every change goes through `Engine.apply` (verbs return ops). 2: a role or value at each style key and in each segment; part ids by meaning, clear of `retired_ids`. 3: the in-between reads only the state at t; caches keyed by content digest (plans) or text (outlines, fonts fixed in a process). 4: colors mix in linear RGBA. 5: no new types; Text/Equation keep every member and gain `build`. 6: lint-imports kept (`time` and `inspect` import no `domains`). 7: layout untouched. Second ways: Deviations 1, 5 |
| docs and artifacts | pass | API.md (verbs, lint table, two running blocks), MCP.md, llms.txt regenerated, SPEC "As built (Milestone 17)" (non-contract text), example, golden, this file |

Independent audit: done 2026-11-23 (a separate model session); findings fixed below.

**MILESTONE 17: DONE on Linux for the focused gates and the independent audit; the full
suite, Windows, macOS and CUDA remain.**

## Independent audit (2026-11-23)

Repro scripts were turned into tests; each test fails without its fix (most were written
first; for #1 and #4 the fix came first and the test was then run against the old code).

| # | Finding | Fix | Test |
| --- | --- | --- | --- |
| 1 (blocker) | The plan cache had no memory bound: `path._nbytes` counted a `Plan` (a dataclass of arrays, up to ~2 MB) as ~320 B, so the 256 MB budget never evicted plans, and a moving source or zooming camera makes a plan a frame: 600 frames of a 400-glyph text moving into a circle held 912 MB of plans (RSS 1.4 GB) | `core/cache.py`: one size function (arrays inside tuples, lists, dicts and dataclasses) and one byte-bounded LRU (`ByteCache`), used by the path geometry cache, a separate plan pool (`morph.PLAN_BYTES` 64 MB: about 30 of the largest plans, so many morphs at once keep theirs, and moving morphs never push a drawing's flattened curves out) and the `draw` recordings (`record.CACHE_BYTES` 256 MB besides `CACHE_SIZE` 4096: the same bug class, a morph's recording is up to megabytes a frame). A plan in each outline's own frame is not feasible without changing outputs (Deviation 15). After: the same 600 frames hold 58 MB of plans (32 plans, at the 64 MB pool), RSS 318 MB | `test_transform.py::test_a_moving_morph_keeps_plan_memory_bounded` (real array bytes summed, not the counter), `::test_a_still_morph_makes_one_plan_and_the_same_frames`, `::test_recordings_are_bounded_by_their_bytes_too` |
| 2 | The in-between's fill cost 8.5-11.6 s a frame for a 400-glyph text into a filled circle (and 211 s to lint a transform from a 200 000-character text): `path_math.region` tested 512 rings pairwise in Python (261 632 `inside` calls, 6.7 s), and earcut took 1.3-5.9 s on the target's ring of ~40 000 points crossing itself | `region` tests all probes against a ring at once, in chunks (`INSIDE_BLOCK`), with the same arithmetic (exactly the old regions); `morph.MAX_FILL_RING` thins longer rings (Deviation 16). After: 0.4-1.0 s a frame including the plan, the 512-ring layer 0.25 s | `test_morph.py::test_the_vectorised_fill_region_is_exactly_the_ring_by_ring_one` (against the old loop, both rules, 12 awkward sets), `::test_a_ring_of_very_many_points_is_tested_in_bounded_memory`, `::test_a_400_glyph_text_fills_into_a_circle_quickly`, `::test_the_in_between_fill_thins_only_rings_past_its_limit` |
| 3 | `off_palette`/`off_scale` were hidden on every composite part (Deviation 9's reason was wrong) | only parts tagged `glyph` are skipped | `test_glyphs.py::test_a_composite_part_with_explicit_values_is_judged_but_a_glyph_is_not` (both MODES) |
| 4 | The matplotlib floor was stale (see Untested) | `glyph_math._ft_flags` feature-detects the enums, `_math` reads both glyph layouts; the floor stays 3.8 (tested) | `test_glyphs.py::test_outlines_read_matplotlib_before_and_after_3_10`; the two files run on 3.8.4 and 3.9.4 |
| 5 | With the `latex` engine a text TeX typesets but mathtext cannot read made `inspect`/`lint` raise from its outline | the parse is guarded: `Outline.error`; `morph_shape` is None, so a morph from it is `morph_missing`, a morph into it draws its picture, and as shapes it is refused when added, saying why | `test_glyphs.py::test_a_text_mathtext_cannot_outline_is_morph_missing_and_never_raises` |
| 6 | A source id over 200 characters was accepted by `transform`, then the document failed to evaluate (`bad_param`) | the limit is `core.ids.MAX_REF`; the verbs refuse such a source | `test_transform.py::test_a_source_whose_id_is_too_long_to_name_is_refused_by_the_verb` |
| 7 | An `as_shapes` text past +-1e9 made `state_at` raise (a track, a camera), or was refused with a message about part 'f' | glyphs past `MAX_COORD` are left out of the parts; `glyph_limit` says so (semantics `glyphs_far`) | `test_glyphs.py::test_glyphs_past_the_coordinate_limit_are_left_out_with_a_lint_not_an_error` (both MODES) |
| notes | repeat-dropping written three times in `morph_math`; the skill did not mention the transforms; lint rules without theme thresholds; static as_shapes cost; the `m = 0` pop | `dense` and `at` call `_clean`; the phasor skill (both copies) says when to use which transform; Deviations 14, 17; Deferred | `test_morph.py`, `test_kit.py` |
