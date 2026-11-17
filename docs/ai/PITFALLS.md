# phasor pitfalls

Known traps and how phasor reports them.

## A wrong edit changes nothing, and says why

`Engine.apply` is all-or-nothing. Always check `result.ok`; on failure read
`result.errors[i].code` and `fix_hint`. Common codes: `unknown_id`, `unknown_param`,
`bad_param`, `bad_role` (you passed a raw color or size instead of a role), `bad_expr`,
`unknown_ref`, `cycle`, `track_on_expr`, `bad_segment` (overlapping or backwards),
`unknown_marker`, `scripted_scene`.

## `visible_fraction` is `null`

Either there is no OpenGL on this machine (`"occlusion": "analytic"`), or the object is
thinner than one inspector pixel (`note` says so; hairlines at 480x270 are). Do not treat
`null` as visible or hidden. Geometry (`bbox`, `center`, `region`) is always exact.

## Translucent objects do not hide things

An object with `opacity` below 0.5 does not count as an occluder, so objects behind it
report `visible_fraction` 1.0. Its own visibility is still measured.

## Tweaks to a scripted scene

`set_param` and `set_style` on a scene built by a `.py` script go to `name.overrides.json`.
If the script later drops that object, lint reports `orphan_override` (an error). Fix the
script or delete the entry; phasor never discards a tweak silently.

## GL contexts

GL calls go to whichever context is current. Never call `moderngl.create_context`; use
`phasor.render.passes.default_renderer()` (or `Engine`), which shares one context and makes
it current for every render.

## Codec lists lie

PyAV wheels list `h264_nvenc` and `hevc_nvenc` even on machines with no NVIDIA GPU.
`phasor doctor` only reports an encoder as working after encoding real frames with it.
Never pick an encoder from `av.codecs_available`.

## Headless OpenGL on Linux

There is no X display in CI or cloud containers. `create_standalone_context()` falls back
to EGL (Mesa llvmpipe gives GL 4.5 in software). Install `libegl1 libgl1-mesa-dri` there.
Software GL is fine for the inspector's 480×270 passes, too slow for final renders.

## `phasor.time` and `phasor.inspect` share names with the standard library

Always import them by their full name (`from phasor.time import ...`). Never run a
script from inside the `phasor/` folder: Python would put that folder on `sys.path`
and the stdlib `time`/`inspect` modules would be shadowed.

## Importing GPU libraries from a lower layer

`phasor.core`, `phasor.style`, `phasor.time` and `phasor.layout` must not import CuPy,
ModernGL or PyAV, even lazily. `lint-imports` catches static imports and
`tests/test_layers.py` catches runtime ones.
