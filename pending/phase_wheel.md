# Phase wheel: "about six hues" in PhasorField stills (2026-11-26)

**Verdict: not a bug, and no change proposed.** The prism `phase` colormap is a continuous
OKLCH hue circle, and the finished still keeps it continuous. No patch, so no golden changes
(`prism_primes` untouched).

## The report

The agent that built `filt.PhasorField` saw "about six distinct hues" in the stills of
`examples/reverb_modes.py`. Each hand takes the hue of its angle from the theme's
`colormap.phase` role.

## How phase becomes colour

- `phasor/style/themes/prism.toml`: `phase = "oklch 0.52 0.1"`. `style/theme.py::_colormap`
  turns this into `core/color.py::phase_lut(256, 0.52, 0.1)`: 256 entries spaced evenly
  around the OKLCH hue circle at constant lightness and chroma. `midnight_glow`, and so
  `look="none"`, uses the same spec.
- `domains/filters_math.py::phase_hue` maps an angle to a share of a turn, `(angle/2pi + hue) % 1`.
- `domains/strokes.py::lut_colors` blends linearly between the two nearest of the 256 entries
  and wraps at 1. It does not snap to the nearest entry.
- The draw multiplies by glow `strong` (2.6) in the shader, then the bloom, then the shoulder
  tone map (identity below the knee 0.9), then sRGB.

## Evidence (`uv run python pending/phase_wheel_probe.py`)

| check | result |
|---|---|
| distinct LUT entries | 256 of 256 |
| colormap, 4096 samples through `lut_colors` | OKLab L 0.520-0.525, C 0.089-0.100; ΔE per step max/median **1.01**; hue per step 0.068-0.099° (even: 0.088°) |
| wheel closes at 2π | ΔE(last, first) = 0.00015, the same as any other step |
| sRGB clipping in `phase_lut` | 38 of 256 entries touch a channel bound. They only lower chroma slightly (C ≥ 0.089), with no flat zones |
| tone map | the hands come out at L ≈ 0.72. Linear lut × 2.6 stays under the knee, so the per-channel shoulder does not pull hues toward the RGB primaries |
| end to end: 144 still hands at 2.5° apart, read back from the finished prism still (`phase_wheel_probe.png`) | L 0.706-0.743, C 0.126-0.145. The hue rises with the angle, median **2.45° per 2.5°** of angle and never backwards. The 1.2-4.4° spread is 8-bit and antialiasing noise. ΔE×100 between neighbours is 0.4-1.5 |
| `reverb_modes` at t=0 (`phase_wheel_reverb_t0.png`) | hue histogram of the coloured pixels, 36 bins of 10°: **0 empty bins**. The hand angles at t=0 are spread evenly too (largest gap 7.7° against 5.6° for even spacing) |

## Why it can read as "six hues"

The 64 hands each show one point of a continuous wheel. A viewer names them by hue category:
orange, olive-yellow, green, teal, blue, violet, pink. A constant-lightness wheel at modest
chroma (L 0.52, C 0.1 before glow) also has no bright yellow or deep blue landmarks, so
neighbouring categories look alike. That is the point of the design: equal lightness means
no phase looks louder than another (`phase_lut` docstring, and the theme comment "the
perceptually even phase wheel"). The probe picture shows the wheel as a smooth sweep when
the hands are dense.

## Options, not proposed

- More hue variety in PhasorField alone: set `[defaults.phasor_field.style] colormap =
  "spectrum"` (`oklch 0.7 0.16`) in `looks/prism.toml`. This would affect only
  `phasor_field`, not `prism_primes` or domain coloring. But `spectrum` is the "hue along a
  curve" role, and here the hue carries phase, so the `phase` role is the right one.
- Changing `phase` itself would change every domain-coloring picture, `prism_primes`
  included. The owner ruled that out.

Files: `pending/phase_wheel.md`, `pending/phase_wheel_probe.py`, `pending/phase_wheel_probe.png`
(144 hands, prism look, as rendered today), `pending/phase_wheel_reverb_t0.png`
(`examples/reverb_modes.py` at t=0, 960×540).
