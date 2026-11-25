"""``phasor gui``: registers itself with the CLI through the ``phasor.commands`` entry point."""

from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    g = sub.add_parser("gui", help="the desktop app: a start screen, or the editor on SCENE")
    g.add_argument("scene", nargs="?", help="a .phasor.json or a .py builder script")
    g.add_argument("--root", help="the folder an attached AI may read and write (default: "
                   "the scene's folder)")  # fmt: skip
    g.add_argument("--no-ai", action="store_true",
                   help="do not let `phasor mcp --attach` edit this scene")  # fmt: skip
    g.add_argument("--install-shortcut", action="store_true",
                   help="add phasor to the desktop's applications menu, then exit")  # fmt: skip
    g.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    if getattr(args, "install_shortcut", False):
        from phasor.gui.app import install_shortcut

        r = install_shortcut()
        print(f"added {r['path']}" if r.get("ok") else f"not added: {r.get('error')}")
        return 0 if r.get("ok") else 1
    try:
        from phasor.gui.desktop import run
    except ModuleNotFoundError as e:
        print(f"phasor gui: {e}. Install it with: uv sync --extra gui", file=sys.stderr)
        return 1
    if args.root:  # one scene in a chosen workspace folder (the AI may use that folder)
        from pathlib import Path

        from phasor.gui.window import open_editor

        if args.scene and Path(args.scene).suffix == ".py" and not _may_run(Path(args.scene)):
            return 1

        app, _window, server = open_editor(args.scene, args.root, serve_ai=not args.no_ai)
        try:
            return app.exec()
        finally:
            if server is not None:
                server.stop()
    return run([args.scene] if args.scene else [], serve_ai=not args.no_ai)


def _may_run(path) -> bool:
    """A script runs when it opens: the first time, the person says so (as the desktop app
    asks, ``desktop.confirm_run``), once per script."""
    from phasor.gui.app import Settings
    from phasor.gui.desktop import confirm_run
    from phasor.gui.window import qt_app

    settings = Settings.load()
    if settings.is_trusted(path):
        return True
    qt_app()
    if not confirm_run(path.resolve()):
        return False
    settings.trust(path)
    settings.save()
    return True


def main(argv: list[str] | None = None) -> int:
    """``phasor-gui [SCENE]``: the desktop app (a start screen when no scene is given),
    for a Start-menu or desktop shortcut, without the rest of the command line."""
    p = argparse.ArgumentParser(prog="phasor-gui", description="the phasor desktop app")
    p.add_argument("scene", nargs="?", help="a .phasor.json or a .py builder script")
    p.add_argument("--root", help="the folder an attached AI may read and write")
    p.add_argument("--no-ai", action="store_true", help="do not let `phasor mcp --attach` in")
    p.add_argument("--install-shortcut", action="store_true",
                   help="add phasor to the desktop's applications menu, then exit")  # fmt: skip
    return _run(p.parse_args(argv))
