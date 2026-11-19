# Building the phasor desktop app

The app is the editor (`phasor-gui`) frozen with PyInstaller into a folder you can run
without installing Python: `phasor` (or `phasor.exe`, or `phasor.app` on macOS). It opens
a start screen; File > Export makes pictures, videos and web pages; the command line is
inside it too (`phasor --cli render scene.phasor.json`).

Build on each system for that system (PyInstaller does not cross-compile):

```bash
uv sync --extra gl --extra encode --extra gui --extra mcp --group package
uv run python packaging/make_icon.py          # only if you changed the icon drawing
uv run pyinstaller packaging/phasor.spec --noconfirm
```

The result is `dist/phasor/` (about 480 MB on Linux, most of it Qt, SciPy and SymPy). Zip
that folder to share it, or run `dist/phasor/phasor` directly.

| System | State |
| --- | --- |
| Linux (x86-64) | built and run here: the command line inside the app renders, lints and runs `doctor`; the window opens |
| Windows | **untested**: the recipe uses `icon.ico` and hides the console; build it on Windows and try it |
| macOS | **untested**: the recipe also makes `dist/phasor.app` |
| NVIDIA / CUDA | **untested**: CuPy is left out of the app (it is large and needs a matching driver); kernels run on NumPy. Add `--extra gpu` and drop `"cupy"` from `excludes` in the spec to try |

To check a build: `PHASOR_TEST_PACKAGE=1 uv run pytest tests/test_packaging.py` builds the
app and runs every command, a render with typeset maths, and the window.

Without building, a shortcut in the applications menu works too:

```bash
uv run phasor-gui --install-shortcut
```

On Linux that writes `~/.local/share/applications/phasor.desktop`; on Windows a Start-menu
entry (untested); on macOS build the `.app` instead.

Settings (recent scenes, the math typesetting choice) live in `%APPDATA%\phasor`,
`~/Library/Application Support/phasor` or `~/.config/phasor`. New scenes and copies of
the examples go to `Documents/phasor`.
