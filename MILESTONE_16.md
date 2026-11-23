# Milestone 16: vector paths, fills and shapes

From `MANIM_PARITY.md` ("Milestone 16"), on the open object contract of Milestone 15 (ADR
0005). Exit demo: **a gallery of shapes with fills, dashes, booleans and an SVG renders
correctly, and every type has inspector output, lint, docs, an example and a golden**
(`examples/shapes.py`, `examples/vector_toolkit.py`).

**Gate (2026-11-23): MILESTONE 16: DONE** on Linux, after an independent audit (4 blockers and 5 should-fix items found, all fixed with tests). Windows CI (4 known no-OpenGL/ACL failures only, 1376 passed on the combined Milestone 16+18 commit) passed; the full suite on Ubuntu CI is the final check. Untested: macOS, CUDA, GPU goldens, FIFO/symlink hostile-file tests on Windows.

Status: **built on Linux (llvmpipe OpenGL, no CUDA); independent audit done 2026-11-23,
its findings fixed (see "Independent audit"); a few items deferred.** Untested on Windows,
macOS and CUDA.

## Built

| Area | Where | What |
| --- | --- | --- |
| Geometry | `phasor/domains/path_math.py` | pure NumPy: subpaths of cubic Béziers (an anchor, then handle, handle, anchor per cubic); builders (`polyline`, `arc`, `circle`, `ellipse`, `rectangle` with corner radius, `sector`, `regular_vertices`, `brace`); adaptive flattening (Wang's formula per cubic, tolerance in world units; zero-length steps dropped so strokes always miter); `region` (rings to outer edges and holes under `nonzero` or `evenodd`, by nesting and winding) and `fill` (earcut); `trim`, `piece`, `dashes`, `dots` by arc length; `crossings`; `boolean` (shapely, lazy); `read_svg` (svgelements, lazy, cached by `files.stamp`, 8 MB cap); `MissingExtra` |
| `Path` | `phasor/domains/path.py` | `PathBase` draws, measures and describes any path: a fill (earcut triangles through `ctx.triangles`, always `glow=1.0`: plain paint), then a stroke through `strokes.polyline` (per-point width, mitered; a whole closed loop starts mid-way along its first segment so its seam is mitered too), dashes or round dots by arc length, arrow `tips` on open ends. Flattened within `FLATNESS` (2.5e-4 frame heights) of the true curve. `progress` traces the stroke and fades the fill in. Geometry cached by the geometric parameters only (a fade does not re-triangulate). Semantics: `kind`, `subpaths`, `closed`, `length`, `area` (closed or filled), `rule`, `fill`, `dash`, `tips`, `drawn`, `fill_crossing`, plus each shape's facts. Features: the box's compass points (derived from the recording), `start`, `end`, and `region` (a filled one-ring path) or `curve` (an unfilled one-subpath path), plus `center`, `vertices`, `tip` where a shape knows them. Bounds are derived (ADR 0005). `path_points`: a `Tracer` rides any path. `Path.through(id, points, closed=)` |
| Shapes | `phasor/domains/shapes.py` | `Circle`, `Ellipse`, `Arc` (with `tips`), `Sector` (pie or ring slice), `Annulus`, `Polygon` (type `polygon_path`), `RegularPolygon`, `Rectangle` (corner radius, rotation), `Brace` (feature `tip`): each a `PathBase` with its own params and a `bezier` method, and exact facts (radius, area) in its semantics |
| Fill and dash style groups | `phasor/style/theme.py`, the four base themes | `fill` (`none soft solid`: the fill's opacity, 0 / 0.12 / 1) and `dash` (`solid dashed dotted`: the period as a share of the frame height, 0 / 0.03 / 0.014) |
| Boolean operations | `shapes.union`, `intersection`, `difference`, `exclusion` | two shapes or paths in, a new `Path` of straight segments out (outer rings counterclockwise, holes clockwise, so they are holes under either rule), measured once when called; `a`'s style unless given one |
| `SVG` | `phasor/domains/svg.py` | a composite (ADR 0005) with a `Files` param: each shape of the file becomes a `Path` part `<svg>.<name>`, named from its `id` (sanitized, unique, stable); y flipped; the view box centred on `pos`, `height` world units tall; transforms applied (svgelements); fill rule kept; filled shapes filled and stroked ones stroked in the SVG object's roles. Text, images, external references ignored |
| Contract 3 for composites that read files | `phasor/engine.py` (`_Snapshot.refresh`, `file_stamps`) | every state, inspector snapshot, lint and bake an engine caches is forgotten when a file the document names changes (path, mtime, size). Needed because an `SVG` builds its parts inside `state_at` |
| Lint | `phasor/inspect/lint.py` | `fill_crossing` (warning: a filled path whose edges cross; earcut assumes they do not; no fix) and `missing_extra` (warning: a type's `Requires` package is not installed; the fix hint names the extra). `tiny` waits until a traced curve is drawn (it flagged every `trace` at its first instant) |
| Layout without OpenGL | `phasor/layout/geometry.py` | `footprint` fills the `region` outline of a filled path, so a label near a disc is judged by the disc, not its box, in both lint MODES |
| Shared helpers | `domains/strokes.py`, `domains/algebra.py` | `strokes.arrow_head` (the head `arrow` always drew; `Arrow` byte-identical, tested) is the path tip; `algebra.Polygon` takes its vertices from `path_math.regular_vertices` (identical floats, checked for n = 3..12) |
| Doctor | `phasor/render/doctor.py` | an `extras` line: which optional object packages are missing and the `--extra` to add (never a failure) |
| Dependencies | `pyproject.toml`, `uv.lock` | base `mapbox-earcut>=1.0` (2.1.0 locked; ISC); extras `boolean = ["shapely>=2.0"]` (BSD), `svg = ["svgelements>=1.9"]` (MIT), both in `all`. CI (`.github/workflows/ci.yml`) and the packaging spec install both extras; `mapbox_earcut` is a PyInstaller hidden import |
| Docs | `docs/ai/API.md` ("Paths, fills and shapes"), roles line, lint table; `llms.txt` regenerated | running examples (shapes, even-odd hole, booleans); the SVG block is `no-run` (needs a file; the example runs it) |
| Examples | `examples/shapes.py` (reference), `examples/vector_toolkit.py` (technique), `examples/assets/emblem.svg`, `kit/examples.toml` | prism look; both pass `lint(strict=True)` with nothing |

## Verification (Linux, 4 cores, llvmpipe OpenGL 4.5, no CUDA)

- `tests/test_shapes.py` (33 tests): geometry against known maths (circle, rounded
  rectangle and sector areas, lengths, flattening error and adaptivity, holes under both
  rules, an island in a hole, a tilted plane, dashes that do not crawl when traced or
  lengthened, dots, trim, crossings, the four booleans' areas by set algebra, determinism,
  missing extras); every shape's semantics and features; subpath validation; the fill is
  `glow 1` at the role's opacity and fades with `progress`; stroke-only and fill-only;
  dashes stable under `progress`; tips; `Arrow` byte-identical; booleans as objects;
  `fill_crossing`, `missing_file` and `missing_extra` in both lint MODES; a label beside a
  filled disc judged alike in both MODES; SVG parts, names, y flip, transforms, even-odd,
  roles; the SVG re-read after its file changes; paths outside the folder refused; a solid
  `muted` fill renders exactly its sRGB value (GL); the examples cover every new type and
  lint clean (strict); storyboard goldens.
- Goldens (looked at): `tests/golden/software/shapes.png`, `tests/golden/software/vector_toolkit.png`.
  One storyboard per example; a test asserts that every new type appears in one of the
  two examples, so a golden covers each type.
- `tests/test_final.py` covers the 11 new types like every other (2D and orbit: inspect,
  draw, lint), with `REQUIRED` entries for `path` and `svg`.
- Focused suites and gates: see the gate section at the end.

## Deviations

1. **`shapes.Polygon` has the type name `polygon_path`.** `polygon` is `algebra.Polygon` (a
   regular n-gon with numbered vertices and group semantics); it is kept as it was
   (renaming it would need a migration of saved documents). Python names differ by module
   (`shapes.Polygon`, `algebra.Polygon`). The owner may prefer another name.
2. **Boolean operations are functions, not live objects.** They measure their inputs once
   and return a plain `Path`; moving an operand later does not change the result. A live
   `Boolean` type (reading two objects through the state) is deferred.
3. **An SVG's own colors and stroke widths are not used**: roles only (the rule until
   Milestone 18 allows explicit values). Fill vs. no fill and stroke vs. no stroke are kept.
4. **Engine change (contract 3, no contract text changed):** `_Snapshot.refresh` drops the
   engine's cached states, snapshots, lint and bake when a named file changes. Before, a
   changed file was re-read only by `draw` (Room3D, Image); inspector output cached for the
   same document version could be stale. Now it is not, for every file type.
5. **Fills assume rings that do not cross** (earcut). A crossing filled path is lint
   `fill_crossing`, not repaired; shapely is not used in the fill path, so pictures do not
   depend on whether an extra is installed.
6. **SVG parts without an `id` are named by their kind and a counter** (`circle`,
   `circle_2`): there is no meaning to name them by. Ids that start with a digit get the
   kind in front (`rect_1st`).
7. **A missing or unreadable SVG is also `orphan`** (it draws nothing, so it never
   appears), next to `missing_file`, `bad_file`, `svg_too_complex` or `missing_extra`.
8. **`tiny` no longer fires while a curve is being traced** (`drawn` < 1). It affects `Line`
   and every type with `progress`; it removed a warning at the first instant of every
   `trace`.
9. **`Brace` is a stroked curve**, not Manim's filled tapered glyph.
10. **Theme values chosen by eye:** fill `soft` = 0.12 (fills blend in linear light, so
    0.22 looked grey-white on dark themes), dash periods 0.03 and 0.014 of the frame height,
    dashes 55% on.
11. **`Line` and `Arrow` stay as they were**, beside `Path`. A straight open `Path` and a
    `Line` draw alike, but `Line` has `hue_along` and the `polyline` semantics layout and
    lint already rely on; `Arrow` is a vector (`components`). Making them paths would change
    saved scenes' pictures and semantics; they share the stroke and head helpers instead.
    `Path` is the one way to draw a filled or dashed outline.
12. `core/pycode.py` (the Code view) reaches the new types as `shapes.<Name>`. A script
    regenerated from a document whose verb staggered four objects differed in the last
    float digit of a track's end (`0.36 + 1.4` vs the stored `1.76`). Fixed in the audit: a
    segment whose end is not exactly start + duration is written with both ends
    (`s.segment`), so the round trip is exact; stored values are not rounded.
13. `Engine.layout`, `state_at`, the inspector and lint all call `_Snapshot.refresh` first.
14. Shapes lie in the plane z = `pos.z`; a `Path` may lie in any plane (fills are
    triangulated in the plane of its points).
15. **Audit: new bounds that can refuse a saved document.** `Arc.angle` is -360..360 (not
    0), and every number of a path or shape is within ±1e9 (`path.MAX_COORD`). Checked: no
    `shapes.Arc` in the repository (examples, docs, tests) has a larger angle, and the two
    saved `*.phasor.json` documents in the repository load; gallery and showcase scripts
    were not all re-run (they predate the shape types). A full-turn `Arc` stays
    open (tips kept, no area unless filled); a `Brace` is always open.
16. **Audit: `SceneState.reserved`** (new field, `compare=False`): the document's object and
    camera ids and its retired ids, so a composite whose part names come from a file avoids
    them. An `SVG` part that would take one gets the next suffix (`label_2`).
17. **Audit: only regular files are read.** `core/files.py`: a folder, FIFO, device or a
    link that loops is "not there" for every `Files` type (`missing_file`, saying why);
    `resolve` turns a looping link into `ValueError`; `regular_file` and `read_bytes`
    (non-blocking open, `fstat` check, size cap) are the readers' way in. `Image`, `Room3D`
    and `Network` read through `regular_file`.
18. **Audit: a render worker refuses a changed file** (ADR 0003). `worker_payload` carries the
    file stamps the bake was made from; `from_payload` refuses a payload whose files changed
    since, and a worker engine never forgets its applied bake: a file that changes during the
    render stops it with "changed while the scene was being rendered; render it again".
19. **Audit: dashes per path are capped** at `MAX_DASHES` (2000); past it the period grows so
    the whole path is still dashed (lint `dash_limit`). Theme `[dash]` periods must be 0 or
    0.002..1 frame heights, `[fill]` 0..1 (`ThemeError`).
20. **Audit: geometry cache bounded by bytes** (`path.CACHE_BYTES`, 256 MB) instead of 1024
    entries. `Camera.project`/`to_screen` build their arrays without `np.c_` (same values).
21. **Audit: `core/extras.py`** is the one "can this optional package be imported" check and
    module -> extra map (was in `path_math`, `lint.file_rules` and `doctor`).
22. **Audit: boolean operations** keep only the parts with an area before the overlay, snap
    and retry on a GEOS error, and otherwise raise a `ValueError` naming the operation.

## Deferred

- A live boolean type; boolean ops with self-crossing inputs beyond what shapely's
  `make_valid` gives.
- SVG: colors, stroke widths, opacity, gradients, text, `<use>` of external files, CSS
  classes beyond what svgelements resolves.
- Prism-look defaults for the new types (`style/looks/prism.toml`): left out to stay clear
  of Milestone 18's style work; the examples set glow and fill themselves.
- Dash offset as an animatable parameter (marching ants); line caps and joins other than
  mitered butt ends.
- Path morphing (Milestone 17).

- Frame cost of a large SVG (audit item 8): the file is parsed and placed once per change
  and the geometry is no longer flattened or measured again each frame, but `state_at` still
  validates every part's parameters at each new t (in `time/compile.py` `_expand`, generic
  for composites) and the inspector's per-object work (recording, screen boxes) is the
  same for a part as for any object: about 0.2 s per 1000 parts per new t for `state_at`
  and 2.5 s per 1000 parts for a new inspector snapshot (numbers below). Caching validated
  parts across t in `_expand` would need a key cheaper than the parameters themselves. Lint
  of a 5000-part drawing still takes about 94 s and 2.1 GB (it keeps an inspector snapshot
  of every part at each sampled time).
- `progress < 1` objects never get `tiny` (Deviation 8): a curve left part-drawn at the end
  of a scene is not judged; its `drawn` value stays in its semantics.
- `Image` and `Room3D` check the file is regular and then open it by path (Pillow, the OBJ
  reader); a file swapped for a FIFO between the check and the read could still block.
  Only the SVG reader opens non-blocking and checks the open file.
- SVG part names follow file order: an unnamed shape added early renumbers later unnamed
  shapes of its kind (documented; give shapes ids).
- A `Files` parameter driven by an expression is not watched (expressions give numbers);
  Const and track values are.

## Untested

Windows, macOS and CUDA (no GPU goldens in `tests/golden/gpu/`). mapbox-earcut, shapely and
svgelements publish Windows wheels; not installed on Windows here.

## Setup lines

README, TESTING.md and the Commands block in `CLAUDE.md` (c62e033) install `--extra boolean
--extra svg`; CI and packaging too. Without the two extras the suite skips the tests that
need them (named in the skip reason) instead of failing.

## Merge notes (Milestone 18 runs in parallel)

Touched in the style layer: `phasor/style/theme.py` (two entries in `ROLE_GROUPS`, two in
`_SECTION`, nothing else) and the four base theme TOMLs (a `[fill]` and a `[dash]` table
before `[colormap]`). `inspect/aesthetic.py`, `core/schema.py`, looks and `polish` are
untouched. `inspect/lint.py`: `file_rules` (missing_extra), the per-frame loop (`tiny`,
`fill_crossing`), the `make` messages and the module table.

## Gate (phasor-milestone-gate, 2026-11-23, by the authoring model)

| Step | Result | Evidence |
| --- | --- | --- |
| ruff check, ruff format --check, lint-imports | pass | all clean; 2 import contracts kept |
| full `uv run pytest -q` | not run | over an hour on this box; the focused set below instead |
| focused suites | pass | test_final, test_lint, test_beauty, test_aesthetic, test_open_contract, test_layers, test_kit, test_contracts, test_algebra, test_linalg, test_strokes, test_shapes, test_engine, test_inspect, test_layout, test_doctor, test_pycode, test_docs, test_mcp, test_workers with `-k "not slow or shapes or vector_toolkit"`: 838 passed, 4 skipped (CUDA, an empty parametrization, `value` draws nothing). test_remote, test_manim_import, test_time: 109 passed, 1 skipped (no PySide6). Without the extras, the tests that need them skip and name the extra |
| `phasor doctor` | tier B | llvmpipe OpenGL 4.5, CPU kernels (g++), extras installed |
| exit demo | pass (Linux) | `examples/shapes.py`, `examples/vector_toolkit.py`: strict lint empty; both goldens looked at |
| contract audit | done by the authoring model, not independently | no write path besides `Engine.apply`; state depends on the document, t and named files (SVG through `Files`; the engine's caches are dropped when a named file changes); layers kept (lint-imports); every type has the five members after registration; deviations above |
| docs and artifacts | pass | API.md section with running blocks, examples, goldens, this file |
| independent audit (2026-11-23) | findings fixed; deferred items listed | after the fixes: ruff, ruff format, lint-imports clean; test_shapes, test_lint, test_docs, test_layers, test_open_contract, test_beauty, test_engine, test_workers, test_style_freedom, test_pycode, test_doctor: 323 passed, 4 skipped; test_final `-k "not slow"`: 374 passed, 2 skipped. Full suite left to CI |

## Independent audit (2026-11-23) and fixes

An independent audit found four blockers, five should-fix items and notes; each is fixed
with a test written first in `tests/test_shapes.py` (sections 6-10), hostile files made in
`tmp_path`:

| Finding | Fix | Test |
| --- | --- | --- |
| B1a SVG ids colliding with document/retired ids raised `bad_part` | suffix past `SceneState.reserved` | `test_svg_part_names_avoid_the_documents_ids_and_retired_ids` |
| B1b `scale(1e308)` made NaN points (validator raised) | shapes with non-finite or absurd points skipped, view box sanitized; `bad_file` | `test_svg_shapes_with_coordinates_that_are_not_finite_are_dropped`, `test_a_hostile_view_box_still_draws_finite_parts` |
| B1c a looping link raised `RuntimeError` from every read | `files.resolve` -> `ValueError`; missing | `test_a_link_that_loops_is_a_missing_file_for_every_type_that_reads_files` |
| B2 a FIFO hung forever | regular files only; non-blocking read with a cap | `test_a_fifo_is_never_read` |
| B3 nested `<use>` expanded 10^6 times | expansion counted on the tree first (elements, depth, cycles), shape and segment limits; `svg_too_complex` | `test_nested_use_that_fans_out_is_svg_too_complex_and_fast`, `test_use_cycles_and_deep_nesting_never_raise`, `test_a_drawing_past_the_segment_limit_is_truncated_with_a_lint_note` |
| B4 `Arc.angle=1e8` exhausted memory | `|angle| <= 360`; every path number within ±1e9 | `test_shape_params_at_their_bounds_never_exhaust_memory`, `test_arc_angle_is_bounded_and_a_full_turn_stays_open` |
| 5 dashes unbounded; theme dashes unchecked | `MAX_DASHES` per path, `dash_limit`; dashes counted from `start`; theme validated | `test_dashes_are_capped_per_path_and_counted_from_where_they_start`, `test_a_huge_dashed_circle_draws_a_bounded_number_of_dashes_and_lint_says_so`, `test_a_theme_dash_too_small_to_draw_is_a_theme_error` |
| 6 a worker re-baked after a file change | stamps in the payload, frozen worker snapshot | `test_a_worker_never_bakes_and_refuses_a_file_changed_mid_render` |
| 7 a `Files` param on a track was never re-read | `compile.file_values` for the engine and lint | `test_a_tracked_file_path_is_read_again_when_any_of_its_files_changes` |
| 8 SVG frame cost | byte-bounded cache; placement cached; see numbers | `test_a_drawing_of_many_parts_is_not_flattened_again_at_a_new_frame` |
| 9 raw `GEOSException` | areas only, snap and retry, else `ValueError` | `test_a_boolean_on_degenerate_input_never_raises_a_raw_geos_error` |
| notes | `bad_file` rule; malformed files cached as bad; part names capped (40) and documented; `fill_unchecked`; `core/extras.py`; full-turn Arc and degenerate Brace open; pycode round trip exact; setup lines | `test_a_malformed_svg_is_bad_file_and_parsed_once`, `test_svg_part_names_are_capped_in_length`, `test_a_fill_too_detailed_to_check_for_crossings_says_so`, `test_a_brace_with_no_width_is_open`, `test_a_staggered_trace_round_trips_through_the_code_view_exactly` |

Frame cost (Linux, 4 cores, no OpenGL: `state_at`, every part's `draw` into a context that
records nothing, an inspector snapshot at a new t, then `lint()`; synthetic files of 1000
and 5000 filled paths, scratch benchmark, wall clock):

| Parts | | `state_at`, new t | draw all parts, first t / later t | inspector, new t | `lint()` | peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | before | 0.15-0.19 s | 0.53 s / 0.42-0.45 s | 3.8 s | 33.2 s | 542 MB |
| 1000 | after | 0.16-0.21 s | 0.47 s / 0.03 s | 2.6 s | 14.8 s | 528 MB |
| 5000 | before | 0.87-1.04 s | 2.5 s / 2.4-2.7 s | 28.9 s | 314 s | 2.2 GB |
| 5000 | after | 0.89-1.01 s | 2.5 s / 0.14-0.16 s | 15.8 s | 94 s | 2.1 GB |

Drawing a later frame is 15-17 times faster and lint 2.2-3.3 times; `state_at` and lint's
memory are unchanged and deferred (above): lint keeps an inspector snapshot of all 5000
parts per sampled time, and `state_at` validates every part at each new t.


**MILESTONE 16: independent audit findings fixed on Linux (2026-11-23); focused suites
pass, the full suite is left to CI; deferred items above.** Untested on Windows, macOS and CUDA.
