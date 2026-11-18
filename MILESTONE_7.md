# Milestone 7: Beauty system

Roadmap row: aesthetic lint rules, gallery compare, art-director pass, variants,
`choices.jsonl`, two more themes (dark stays the default), living motion, softened edges,
varied palettes, look and object presets. Exit demo: **the critic pass measurably improves a
rough scene in 3 rounds.**

Status: **built; gates green on Linux (software OpenGL).** Untested on Windows and a GPU.
**Boundary review not run**: CLAUDE.md asks for the strongest model; this milestone was
built on Sonnet, so the review is open (see below).

## Owner requests of 2026-11-18, and where they landed

| Request | Built |
| --- | --- |
| An option to remove staticness: glow grows and fades, points wiggle, fractals drift | `life` (`off`/`subtle`/`lively`), `core/life.py`, theme `[life]`; deterministic in (id, seed, t); `exact` tag; off by default |
| Dark default, always | `midnight_glow` is the default theme; looks and themes are opt-in; SPEC and STYLE.md say so |
| Barriers clash | contour bands feathered (`domain` kernel `soft`, object `softness`) |
| Fewer rainbows, but not none; domain coloring keeps its rainbow | palettes `soft_phase`, `ember`, `ocean`; `color_restraint` exempts phase coloring; `phase` stays the default for domain coloring |
| Presets for making these | looks `calm`/`glow`/`chalk`; object presets on dot, line, domain coloring, fractal, Riemann surface; MCP `preset` and `scene_new(look=)` |
| Fewer rigid things / animation smoothness | life; motion blur for finals (`frame_blurred`, `--motion-blur`) |

## Built

| Area | Where |
| --- | --- |
| Themes `aurora` (dark) and `paper` (light, opt-in); roles `soft_phase`, `ember`, `ocean`; `[life]`, aesthetic `[lint]` thresholds | `style/themes/*.toml`, `style/theme.py` |
| Looks | `style/looks.py`, `style/looks/*.toml`, `Scene(look=, life=, seed=)` |
| Object presets | `ObjectBase.Presets`, `preset=` |
| Living motion | `core/life.py`, applied in `time/compile.py` |
| Schema v5 | `meta.life`, `meta.seed`, `meta.look`; migration keeps old scenes still |
| Soft contours | `render/gpu/kernels/domain_coloring.py` (CUDA source, NumPy reference, CPU-compiled test) |
| Aesthetic rules | `inspect/aesthetic.py`: focal_point, color_restraint, glow_clip, density, motion_rhythm, margin, stroke_mix |
| Art director | `inspect/critic.py`, `Engine.critique/polish`, MCP `critique`/`polish` |
| Gallery compare | `render/gallery.py`, `Engine.storyboard(compare=)`, MCP `storyboard(compare=)` |
| Variants, your taste as data | `inspect/variants.py`, `Engine.variants`, `style/choices.py`, MCP `variants`/`choose`, `phasor style tune` |
| Motion blur | `Engine.frame_blurred`, `render/job.py`, `phasor render --motion-blur` |
| Docs | `docs/ai/` (API, MCP, PATTERNS, PITFALLS), `STYLE.md` (your words so far), skill |
| Exit demo | `examples/rough_beauty.py`, `tests/test_critic.py` |

## Verification

- `uv run pytest`: **528 passed, 17 skipped** (all need CUDA); ruff, ruff format, lint-imports clean; `phasor doctor` tier B.
- Exit demo: the rough scene scores 6.5; the critic's polish lowers it to 3.17 in one round
  (nothing certain is left; at most 3 rounds are allowed); asserted: every round improves,
  at least 45% lower, margin and stroke issues gone, exactly one highlight left and it is a
  pole, not the probe; the judgement it will not make (empty frame, colliding motions) is
  still reported. Looked at the before/after storyboards.
- Looked at all three looks, the soft-vs-classic domain coloring, the variants sheet.
- Tests added: `test_life.py`, `test_aesthetic.py`, `test_critic.py`, `test_gallery.py`,
  `test_choices.py`, plus kernel, render (blur), MCP and docs tests.

## Deviations from SPEC (SPEC updated in the same commit)

1. `life` is **off by default** (plain scenes and migrated documents); looks turn it on.
   The owner wanted an option to remove staticness, which this gives; flipping the default
   to `subtle` is one word in `core/schema.py` plus regenerating goldens.
2. Backdrops do not pan: a field's geometry must stay true (a pole has a place), so fields
   turn their *colors* instead (a `hue` parameter). Strokes shimmer uniformly along their
   length, not per point.
3. `stroke_mix` fires above two weights (the theme has three, so three could never fire).
4. `glow_clip` measures light over three times plain paint before bloom, not after tone mapping.
5. The vision critic is the AI reading `STYLE.md` and the storyboard through the tools; the
   built-in critic is the deterministic half (score, fixes, polish).
6. `phasor style tune` reports preferences; it does not yet edit a theme.
7. The gallery is empty: frames you love are yours to add (`gallery/name.png` + `name.txt`).
8. Variants are as-is, calmer, bolder; different verbs and camera alternatives are not offered.

## Bugs caught while building

1. My palette test assumed `soft_phase` spans fewer hues than the full wheel; it does not
   in absolute terms (it is lighter), so the real invariant is "at most three hues": every
   theme is now tested for that (aurora's was four; fixed).
2. The first `glow_clip` counted any over-range pixel, so a legitimate soft glow always
   failed and no fix could clear it; it now means "beyond what the tone map can hold".
3. The margin rule compared against fractions of width, the layout margin is in frame
   heights; they disagreed after an autofix.
4. The critic first proposed highlighting a decoy because the real hero was not on screen
   yet; it no longer invents a hero when something is highlighted elsewhere.

## Untested

- Windows, NVIDIA GPU (the new `soft` kernel argument is tested against the CUDA source
  compiled for the CPU; 16 tests need a real GPU).
- `aurora` and `paper` themes only on single stills (no full-scene review of `paper`).
- A real vision model as the art director.

## Open

- **Boundary review of Milestone 7** (and the M6 review's follow-ups) to be run on the
  strongest model.
- Fill in `STYLE.md` (negative space, motion rhythm, typography) and `gallery/`.
