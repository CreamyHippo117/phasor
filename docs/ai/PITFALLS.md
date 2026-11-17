# phasor pitfalls

Known traps and how phasor reports them.

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
