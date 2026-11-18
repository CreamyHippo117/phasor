"""Search the docs written for AI (``docs/ai``) and the example scenes.

Plain keyword scoring over headed sections: no index to build, nothing to go stale. The
docs live in the repository (``docs/``) or, in an installed package, in ``phasor/_docs``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent.parent


def docs_root() -> Path | None:
    for c in (_HERE / "_docs", _HERE.parent):
        if (c / "docs" / "ai").is_dir() or (c / "ai").is_dir():
            return c
    return None


def _ai_dir(root: Path) -> Path:
    return root / "docs" / "ai" if (root / "docs" / "ai").is_dir() else root / "ai"


def _examples_dir(root: Path) -> Path:
    return root / "examples"


def sections() -> list[dict[str, str]]:
    """Every ``##``/``###`` section of the AI docs, and one entry per example scene."""
    root = docs_root()
    if root is None:
        return []
    out: list[dict[str, str]] = []
    for f in sorted(_ai_dir(root).glob("*.md")):
        title, body = f.stem, []
        for line in [*f.read_text().splitlines(), "## "]:
            if line.startswith(("## ", "### ")):
                if "".join(body).strip():
                    out.append(
                        {"source": f.name, "heading": title, "text": "\n".join(body).strip()}
                    )
                title, body = line.lstrip("# ").strip() or f.stem, []
            else:
                body.append(line)
    ex = _examples_dir(root)
    if ex.is_dir():
        for f in sorted(ex.glob("*.py")):
            text = f.read_text()
            out.append({"source": f"examples/{f.name}", "heading": f.stem, "text": text})
    return out


def _words(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 1]


def search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    q = set(_words(query))
    if not q:
        return []
    hits = []
    for sec in sections():
        head, body = set(_words(sec["heading"])), _words(sec["text"])
        score = 3.0 * len(q & head) + sum(1 for w in body if w in q) ** 0.5 * 1.0
        score += 2.0 * len(q & set(body)) / max(len(q), 1)
        if score > 0:
            hits.append((score, sec))
    hits.sort(key=lambda h: -h[0])
    return [
        {
            "source": s["source"],
            "heading": s["heading"],
            "score": round(sc, 2),
            "text": s["text"][:1500],
        }
        for sc, s in hits[:limit]
    ]


def read(name: str) -> str:
    """One doc file by name (``API``, ``PATTERNS``, ``PITFALLS``) for the client to read whole."""
    root = docs_root()
    if root is None:
        raise FileNotFoundError("the AI docs are not installed with this build")
    f = _ai_dir(root) / f"{name}.md"
    if not f.exists():
        raise KeyError(f"no doc {name!r}")
    return f.read_text()
