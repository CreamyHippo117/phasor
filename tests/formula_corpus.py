"""Every formula-shaped string in the examples, showcase, gallery and the AI docs, and what the
whitelist parser made of it (EXPLORER.md rules; advice section 5 item 21).

``tests/golden/contract/formulas.json`` holds the answers the parser gave *before* the
Milestone 26 hardening; ``test_flow_parser.py`` checks that every string that parsed then
still parses to the same expression. Regenerate (only on purpose, and before a parser change):

    uv run python -m tests.formula_corpus
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "golden" / "contract" / "formulas.json"
TREES = ("examples", "showcase", "gallery", "docs/ai")
_CONSTS = {"pi", "e", "i", "I", "j"}
_OK = (ast.Expression, ast.Constant, ast.Name, ast.UnaryOp, ast.BinOp, ast.Call, ast.Load,
       ast.operator, ast.unaryop)  # fmt: skip


def _strings_in_python(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str)]  # fmt: skip


def _strings_in_json(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings_in_json(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings_in_json(v)]
    return []


def _raw_strings() -> set[str]:
    out: set[str] = set()
    for tree in TREES:
        for path in sorted((ROOT / tree).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".py":
                out.update(_strings_in_python(path.read_text(encoding="utf-8")))
            elif path.suffix == ".json":
                with contextlib.suppress(ValueError):
                    out.update(_strings_in_json(json.loads(path.read_text(encoding="utf-8"))))
            elif path.suffix == ".md":
                text = path.read_text(encoding="utf-8")
                for block in re.findall(r"```python\n(.*?)```", text, flags=re.S):
                    out.update(_strings_in_python(block))
                for block in re.findall(r"```json[^\n]*\n(.*?)```", text, flags=re.S):
                    with contextlib.suppress(ValueError):
                        out.update(_strings_in_json(json.loads(block)))
    return out


def candidates() -> list[tuple[str, tuple[str, ...]]]:
    """(string, the free names in it) for each string that is an expression of the parser's
    shapes, sorted."""
    found = []
    for s in _raw_strings():
        s2 = s.strip()
        if not s2 or len(s2) > 4000:
            continue
        try:
            tree = ast.parse(s2, mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        nodes = list(ast.walk(tree))
        if not all(isinstance(n, _OK) for n in nodes):
            continue
        if not any(isinstance(n, ast.Name | ast.BinOp | ast.Call) for n in nodes):
            continue  # a bare number
        called = {n.func.id for n in nodes if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name)}  # fmt: skip
        names = {n.id for n in nodes if isinstance(n, ast.Name)} - called - _CONSTS
        found.append((s2, tuple(sorted(names))))
    return sorted(set(found))


def answers() -> list[dict]:
    """What the parser makes of each candidate: ``real`` (``parse_expression``) and, for a
    string in ``z`` (and ``a``) only, ``complex`` (``ComplexFunction.parse``)."""
    import sympy as sp

    from phasor.render.gpu.codegen import ComplexFunction, parse_expression

    out = []
    for s, names in candidates():
        row: dict = {"s": s, "names": list(names)}
        try:
            row["real"] = sp.srepr(parse_expression(s, names))
        except Exception as e:
            row["real"] = f"error: {type(e).__name__}"
        if set(names) <= {"z", "a"}:
            try:
                row["complex"] = sp.srepr(ComplexFunction.parse(s, constant=True).expr)
            except Exception as e:
                row["complex"] = f"error: {type(e).__name__}"
        out.append(row)
    return out


if __name__ == "__main__":
    rows = answers()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    ok = sum(not r["real"].startswith("error") for r in rows)
    print(f"{len(rows)} strings, {ok} parse as real formulas -> {FIXTURE}")
