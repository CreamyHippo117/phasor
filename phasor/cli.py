"""phasor command line: doctor | render | storyboard | lint | inspect | bench | mcp | gui.

A thin client (SPEC contract 1): it holds no scene state of its own. Commands from
later milestones are registered now so the surface is stable, and exit with code 2
and the milestone that implements them.
"""

from __future__ import annotations

import argparse
import json
import sys

# command -> (milestone, help)
PLANNED = {
    "inspect": (1, "JSON snapshot of a scene at time t"),
    "lint": (2, "run the linter at sampled times and every marker"),
    "storyboard": (2, "keyframe stills + contact sheet + inspector JSON"),
    "render": (3, "final render with a preset"),
    "bench": (3, "ms per frame per kernel and peak VRAM"),
    "mcp": (5, "MCP server for AI clients"),
    "gui": (10, "editor app"),
}

EXIT_NOT_IMPLEMENTED = 2
EXIT_TIER_TOO_LOW = 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    from phasor.render.doctor import format_report, run_doctor

    report = run_doctor()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    if args.require and report.tier > args.require:  # "A" < "B" < "C"
        print(
            f"tier {report.tier} is below the required tier {args.require}",
            file=sys.stderr,
        )
        return EXIT_TIER_TOO_LOW
    return 0


def _planned(name: str, milestone: int):
    def run(_: argparse.Namespace) -> int:
        print(f"phasor {name}: not implemented yet (Milestone {milestone})", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    return run


def build_parser() -> argparse.ArgumentParser:
    from phasor import __version__

    p = argparse.ArgumentParser(prog="phasor", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"phasor {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    d = sub.add_parser("doctor", help="check CUDA, NVRTC, OpenGL 4.3 and encoders; print tier")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.add_argument(
        "--require",
        choices=["A", "B", "C"],
        help="exit 1 if the machine's tier is below this one",
    )
    d.set_defaults(func=_cmd_doctor)

    for name, (milestone, help_) in PLANNED.items():
        sp = sub.add_parser(name, help=f"{help_} (Milestone {milestone})")
        sp.add_argument("args", nargs=argparse.REMAINDER)
        sp.set_defaults(func=_planned(name, milestone))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
