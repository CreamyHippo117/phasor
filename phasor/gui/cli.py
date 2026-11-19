"""``phasor gui``: registers itself with the CLI through the ``phasor.commands`` entry point."""

from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    g = sub.add_parser("gui", help="the editor: viewport, timeline, inspector, lint, looks")
    g.add_argument("scene", nargs="?", help="a .phasor.json or a .py builder script")
    g.add_argument("--root", help="the folder an attached AI may read and write (default: "
                   "the scene's folder)")  # fmt: skip
    g.add_argument("--no-ai", action="store_true",
                   help="do not let `phasor mcp --attach` edit this scene")  # fmt: skip
    g.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    try:
        from phasor.gui.window import open_editor
    except ModuleNotFoundError as e:
        print(f"phasor gui: {e}. Install it with: uv sync --extra gui", file=sys.stderr)
        return 1
    app, _window, server = open_editor(
        args.scene, args.root, serve_ai=not args.no_ai
    )  # keep a reference: Qt closes a collected window
    try:
        return app.exec()
    finally:
        if server is not None:
            server.stop()
