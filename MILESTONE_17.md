# Milestone 17: text as shapes, morphing anything, animated style

Plan: `MANIM_PARITY.md`, "Milestone 17". Exit demo: **text morphs into a shape, a formula
rearranges with `transform_matching`, a color animates between roles, and every new type and
verb has inspector output, lint, docs, an example and a golden; every count is bounded and
fuzzed** (`examples/text_shapes.py`).

Status: **built on Linux (llvmpipe OpenGL, no CUDA); focused suites green (see
Verification); contract audit by the authoring model, not independent.** Untested on
Windows, macOS and CUDA. The full suite was not run (over an hour here; CI runs it).

## Built

| Area | Where | What |
| --- | --- | --- |
| Style transitions | `time/compile.py` (`StyleTrack`, `StyleEnd`, `ANIMATED_STYLE`), `core/schema.py` (`Document.style_track`), `core/ids.py` (`split_style_path`), `core/patch.py`, `core/state.py` (`ObjectState.restyling`) | a track on `<id>.style.<key>` moves an `emphasis` (color), `fill`, `stroke` or `text` key between roles or explicit values (Milestone 18's `style/explicit.py` resolves each end). Colors mix in linear RGBA; an explicit world size is turned into a share of the frame height at t through the camera; easings that overshoot never make a channel negative or an opacity above 1. The document stores a role or a value at the key and in each segment (a hex target canonical, upper case); `state.roles[key]` is the typed value nearer in time; glow, surface, colormap and dash keys are refused (`bad_style_track`). `remove_object` drops the object's style tracks. A real child object named `<id>.style` keeps its own parameter paths (existence first) |
| Style transitions in tools | `time/verbs.py` (`morph`), `inspect/lint.py` (`raw_motion`), `inspect/aesthetic.py`, `inspect/inspector.py` | the `morph` verb takes a style path (theme timing, strict-lint clean); `raw_motion` names the object, not `<id>.style`; `off_palette`/`off_scale` judge an explicit target where a transition reaches it, and offer no snap for it (the snap would set the base, not the target); the inspector shows `restyling: {key: {from, to, progress}}` |
| Morph geometry | `domains/morph_math.py` | pure NumPy: dense sampling of cubics (bounded per subpath), canonical order, counts matched by splitting the longest piece, optimal pairing by travel (scipy, up to `MAX_ASSIGN`), arc-length resampling, alignment (rings keep the target's winding and roll to least travel by FFT cross-correlation; open pieces run the shorter way; ring/open open where the points move least), growth from and collapse to centres, identity for identical paths; `rings` (pieces joined as either side's subpaths, for fills); `signature` (a shape at any place and size) and `digest` (cache keys); the `Shape`/`Group`/`Look` data an outline is described by. Bounds: `MAX_SUBPATHS` 512, `MAX_POINTS` 512, `MIN_POINTS` 8, `MAX_TOTAL` 40 000, `MAX_ASSIGN` 256, `SAMPLE_CUBICS` 2048, `SAMPLES_PER_CUBIC` 64 |
| Morph drawing | `domains/morph.py` | `shape_of` (any object with `morph_shape`, or a composite whose parts have one: an SVG), `frame` (whole or matching), `pair_groups` (equations: keys first, then equal TeX in reading order; anything else: by glyph key or path signature in reading order), arcs for pairs carried past others (`basic.crossing`), `paint` (fill then stroke; the fill crossfades from the source's layout, even-odd, to the target's, its rule, so both ends are exact), `facts` (semantics, lint). Plans cached in `path._cached` keyed by the content digest of both sides |
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

Suite results: see "Gate".

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
9. **Aesthetic `off_palette`/`off_scale` skip composite parts**: a part's values come from its
   composite, which is judged (a glyph would otherwise repeat its text's warning per glyph).
10. **Non-overshooting easings are clamped to 0..1** (`time/easing.py`): `smoother` just below
    u = 1 gave 1 + 2e-16, which a parameter bounded by 1 (`morph`, `progress`) refused at a
    frame. No visible change.
11. **`core/values.normalize`** turns an int too large for a float into a `ValueError`
    (was an `OverflowError`) for every parameter.
12. **Style transition semantics chosen:** the typed value in `roles` switches half-way; the
    `<id>.style` path loses to a real child object of that name.
13. `svg.reserved_under` was factored out of `SVG._parts` so glyph parts share it.

## Deferred

- Morphing *into* an SVG (an SVG can be a source only) and into `Line`/`Arrow`.
- A style key for the glyph outline width (it is `OUTLINE` of the line height); per-glyph styling.
- The importer's `Transform` / `TransformMatchingTex` / `TransformMatchingShapes` mapped onto
  the new verbs (Milestone 20).
- A live check that a source changing type between transforms stays morphable (it is
  re-read each frame; a source that loses its outline is `morph_missing`).
- Matching by TeX *inside* one math string (substrings of a single `Text`): formulas match by
  `Equation` parts.

## Untested

Windows, macOS, CUDA (no GPU golden for `text_shapes`). The plain-text outline needs
matplotlib's FT2Font to read a font from bytes (true on matplotlib 3.8+; checked on 3.11).
