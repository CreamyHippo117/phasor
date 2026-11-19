"""The packaged app's entry point. ``phasor`` (the app) opens the desktop editor;
``phasor --cli ARGS`` is the command line (the app runs its own exports this way, since a
packaged app has no separate Python)."""

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()
    import os

    for name in ("stdout", "stderr"):  # a windowed app has none: printing must not crash
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from phasor.cli import main as cli

        return cli(sys.argv[2:])
    from phasor.gui.cli import main as gui

    return gui(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
