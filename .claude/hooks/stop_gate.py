"""Claude Code Stop hook: block "done" while the fast gates are red.

Runs only when Python files or pyproject.toml changed in the working tree, so chat-only turns
cost nothing. Exit code 2 sends stderr back to Claude and makes it keep working.
Cross-platform (Windows and Linux); needs `uv` on PATH, otherwise it steps aside.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATES = [
    ["uv", "run", "--quiet", "ruff", "check", "."],
    ["uv", "run", "--quiet", "lint-imports"],
    ["uv", "run", "--quiet", "pytest", "-q", "-x", "-m", "not slow"],
]


def changed_files() -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    if payload.get("stop_hook_active"):
        return 0  # already continued once because of this hook; don't loop forever
    if shutil.which("uv") is None:
        return 0
    if not any(f.endswith((".py", "pyproject.toml")) for f in changed_files()):
        return 0
    for cmd in GATES:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stdout + r.stderr).strip().splitlines()[-30:]
            print(
                f"Gate failed: {' '.join(cmd[3:])}\n"
                + "\n".join(tail)
                + "\nFix this before finishing, or say explicitly why it is expected.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
