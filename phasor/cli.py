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
    "mcp": (5, "MCP server for AI clients"),
    "gui": (10, "editor app"),
}

EXIT_NOT_IMPLEMENTED = 2
EXIT_TIER_TOO_LOW = 1
EXIT_LINT_FAILED = 1


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


def _cmd_inspect(args: argparse.Namespace) -> int:
    from phasor.core.issues import DocumentError
    from phasor.scene import load

    try:
        engine = load(args.scene)
        out = engine.inspect(args.t, id=args.id, detail=args.detail)
    except (DocumentError, KeyError, OSError, ValueError) as e:
        print(f"phasor inspect: {e}", file=sys.stderr)
        return 1
    if args.detail == "brief":
        print(f"t={out['t']} occlusion={out['occlusion']}", *out["objects"], sep="\n")
    else:
        print(json.dumps(out, indent=2))
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    from phasor.core.issues import DocumentError
    from phasor.scene import load

    try:
        engine = load(args.scene)
        fixed = engine.autofix()["applied"] if args.fix else []
        issues = engine.lint(step=args.step)
        if fixed:
            # A script keeps its fixes in name.overrides.json; a document is rewritten.
            engine.save(_save_target(args.scene))
    except (DocumentError, OSError, ValueError) as e:
        print(f"phasor lint: {e}", file=sys.stderr)
        return EXIT_LINT_FAILED
    errors = [i for i in issues if i.severity == "error"]
    if args.json:
        print(json.dumps({"fixed": fixed, "issues": [i.to_dict() for i in issues]}, indent=2))
    else:
        for op in fixed:
            print(f"fixed: {op['op']} {op.get('rule', '')}".rstrip())
        for i in issues:
            print(f"{i.severity}: {i}")
            if i.fix_hint:
                print(f"    hint: {i.fix_hint}")
        print(f"{len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
    return EXIT_LINT_FAILED if errors or (args.strict and issues) else 0


def _save_target(scene: str):
    """Where ``lint --fix`` saves: the document itself, or a script's document beside it
    (which writes only the overrides file's sibling; the script is never edited)."""
    from pathlib import Path

    from phasor.core import io

    p = Path(scene)
    return p if p.suffix != ".py" else p.with_name(p.stem + io.DOC_SUFFIX)


def _cmd_storyboard(args: argparse.Namespace) -> int:
    from pathlib import Path

    from phasor.core.issues import DocumentError
    from phasor.scene import load

    try:
        engine = load(args.scene)
        board = engine.storyboard(args.t or None, marks=args.marks, columns=args.columns)
        from phasor.render.output import save_png

        out = Path(args.out or Path(args.scene).name.split(".")[0] + ".storyboard.png")
        save_png(out, board["image"])
    except (DocumentError, OSError, ValueError, RuntimeError) as e:
        print(f"phasor storyboard: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"image": str(out), "frames": board["frames"]}, indent=2))
    else:
        print(f"wrote {out}")
        for fr in board["frames"]:
            name = f" [{fr['marker']}]" if fr["marker"] else ""
            print(f"t={fr['t']:g}{name}", *fr["objects"], sep="\n  ")
    return 0


PRESETS = {  # name -> (width, height, fps); None means "the scene's own"
    "preview": (960, 540, 30.0),
    "final": (None, None, None),
    "4k": (3840, 2160, None),
}


def _cmd_render(args: argparse.Namespace) -> int:
    from pathlib import Path

    from phasor.core.issues import DocumentError
    from phasor.render.doctor import run_doctor
    from phasor.render.encode import EncodeError, open_writer
    from phasor.render.output import encode_srgb8, save_png
    from phasor.scene import load

    try:
        engine = load(args.scene)
        meta = engine.document.meta
        w, h, fps = PRESETS[args.preset]
        size = (w or meta.width, h or meta.height)
        fps = args.fps or fps or meta.fps
        tier = run_doctor().tier
        if tier == "C":
            print(
                "phasor render: no OpenGL here (tier C); nothing can be rendered", file=sys.stderr
            )
            return 1
        if tier == "B":
            print("warning: no CUDA device (tier B): kernels run on NumPy, so this is slow",
                  file=sys.stderr)  # fmt: skip
        if args.frame is not None:
            out = Path(args.out or Path(args.scene).name.split(".")[0] + ".png")
            save_png(out, encode_srgb8(engine.frame(args.frame, size)))
            print(f"wrote {out} ({size[0]}x{size[1]}, t = {args.frame:g} s)")
            return 0
        start = args.start or 0.0
        end = engine.duration if args.end is None else args.end
        count = max(1, round((end - start) * fps))
        out = Path(args.out or Path(args.scene).name.split(".")[0] + ".mp4")
        writer = open_writer(out, *size, fps)
        for k in range(count):
            writer.write(encode_srgb8(engine.frame(start + k / fps, size)))
            if not args.quiet:
                print(f"\rframe {k + 1}/{count}", end="", file=sys.stderr, flush=True)
        writer.close()
    except (DocumentError, OSError, ValueError, RuntimeError, EncodeError) as e:
        print(f"\nphasor render: {e}", file=sys.stderr)
        return 1
    print(f"\nwrote {out}: {count} frames, {size[0]}x{size[1]} at {fps:g} fps, {writer.encoder}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from phasor.render.bench import run_bench, save

    result = run_bench(args.width, args.height, args.repeats)
    path = save(result, args.dir)
    print(f"backend {result['backend']}, {args.width}x{args.height}, median of {args.repeats}")
    for name, r in sorted(result["kernels"].items()):
        print(f"  {name:8s} {r['ms']:10.2f} ms")
    if result["peak_vram_bytes"] is not None:
        print(f"  peak VRAM {result['peak_vram_bytes'] / 2**20:.0f} MB")
    print(f"saved {path}")
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

    i = sub.add_parser("inspect", help="what is on screen at time t, as JSON")
    i.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    i.add_argument("--t", type=float, default=0.0, help="time in seconds (default 0)")
    i.add_argument("--id", help="only this object")
    i.add_argument("--detail", choices=["brief", "standard", "full"], default="standard")
    i.set_defaults(func=_cmd_inspect)

    li = sub.add_parser("lint", help="find visual problems at every marker and sampled times")
    li.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    li.add_argument("--fix", action="store_true", help="apply the fixes the linter suggests")
    li.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    li.add_argument("--json", action="store_true", help="machine-readable output")
    li.add_argument("--step", type=float, default=0.5, help="seconds between samples")
    li.set_defaults(func=_cmd_lint)

    sb = sub.add_parser("storyboard", help="keyframe stills on one contact sheet + inspector")
    sb.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    sb.add_argument("-o", "--out", help="output PNG (default: <scene>.storyboard.png)")
    sb.add_argument("--t", type=float, action="append", help="a time to show (repeatable)")
    sb.add_argument("--marks", action="store_true", help="draw each object's box and id")
    sb.add_argument("--columns", type=int, default=3)
    sb.add_argument("--json", action="store_true", help="machine-readable output")
    sb.set_defaults(func=_cmd_storyboard)

    r = sub.add_parser("render", help="render a video (or one frame) with the final look")
    r.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    r.add_argument("-o", "--out", help="output: .mp4, .mov (ProRes), a folder or a %%05d.png")
    r.add_argument("--preset", choices=sorted(PRESETS), default="preview")
    r.add_argument("--fps", type=float, help="override the preset's frame rate")
    r.add_argument("--start", type=float, help="first time in seconds (default 0)")
    r.add_argument("--end", type=float, help="last time in seconds (default: the end)")
    r.add_argument("--frame", type=float, help="render only the frame at this time, as PNG")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_render)

    b = sub.add_parser("bench", help="ms per frame per kernel and peak VRAM")
    b.add_argument("--width", type=int, default=1920)
    b.add_argument("--height", type=int, default=1080)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--dir", default="benchmarks", help="where results are saved")
    b.set_defaults(func=_cmd_bench)

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
