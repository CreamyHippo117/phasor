# Milestone 16: vector paths, fills and shapes

From `MANIM_PARITY.md` ("Milestone 16"), on the open object contract of Milestone 15 (ADR
0005). Exit demo: **a gallery of shapes with fills, dashes, booleans and an SVG renders
correctly, and every type has inspector output, lint, docs, an example and a golden**
(`examples/shapes.py`, `examples/vector_toolkit.py`).

Status: **built on Linux (llvmpipe OpenGL, no CUDA); gate run by the authoring model, not
independently (see Verification).** Untested on Windows, macOS and CUDA.

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
   appears), next to `missing_file` / `missing_extra`.
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
12. `core/pycode.py` (the Code view) reaches the new types as `shapes.<Name>`. Found on the
    way, not fixed (it is older than this milestone): a script regenerated from a document
    whose verb staggers four objects can differ in the last float digit of a track's end
    (`0.36 + 1.4` vs the stored `1.76`), which `tests/test_final.py`'s round trip flags;
    `examples/vector_toolkit.py` traces two at a time so it does not hit it.
13. `Engine.layout`, `state_at`, the inspector and lint all call `_Snapshot.refresh` first.
14. Shapes lie in the plane z = `pos.z`; a `Path` may lie in any plane (fills are
    triangulated in the plane of its points).

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

- An `SVG` builds and validates its parts at every `state_at` (the file itself is parsed
  once per change); the frame cost of a large drawing is not measured.

## Untested

Windows, macOS and CUDA (no GPU goldens in `tests/golden/gpu/`). mapbox-earcut, shapely and
svgelements publish Windows wheels; not installed on Windows here.

## Setup lines

README and TESTING.md install `--extra boolean --extra svg`; CI and packaging too. The
Commands block in `CLAUDE.md` was left for the owner (it says `uv sync --extra gl --extra
encode`); without the two extras the suite skips the tests that need them (named in the
skip reason) instead of failing.

## Merge notes (Milestone 18 runs in parallel)

Touched in the style layer: `phasor/style/theme.py` (two entries in `ROLE_GROUPS`, two in
`_SECTION`, nothing else) and the four base theme TOMLs (a `[fill]` and a `[dash]` table
before `[colormap]`). `inspect/aesthetic.py`, `core/schema.py`, looks and `polish` are
untouched. `inspect/lint.py`: `file_rules` (missing_extra), the per-frame loop (`tiny`,
`fill_crossing`), the `make` messages and the module table.
