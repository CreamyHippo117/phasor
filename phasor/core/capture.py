"""Updaters defined interactively, captured as source so a saved copy can name a file.

An updater (``obj.always(fn)``) must live in a ``.py`` file in the scene's folder, because the
document names that file (SPEC contract 3: the state depends only on the document, ``t`` and
the files it names). A function defined in a notebook cell or at a prompt has no such file.
``capture(fn)`` reads its source (``inspect.getsource``, which works wherever the shell keeps
cell text in ``linecache``: IPython, Jupyter, the new Python 3.13 prompt) and works out what
it needs from the rest of the session; ``module_text`` writes the captured functions as one
module. ``Scene.save_copy`` does the writing and names the module in the copy's document
(ADR 0006). Nothing here touches a document or a file.

The supported subset is small and checked, never guessed. A function is captured when it

- is a plain ``def`` at the top level of its cell (no lambda, decorator, method, nested
  function, closure, ``global`` or ``nonlocal``), and
- uses, besides its arguments and Python's builtins, only names that the saved module can
  define again: a module (``np``), a function or class importable from a module that is not
  ``__main__`` (``from numpy.linalg import norm``), or a constant of plain data (a number,
  string, ``None``, or a tuple or frozenset of those). Its value at the moment of capture is
  what the file keeps.

Anything else (a list or array from the notebook, another function defined in the notebook,
an instance of a class from ``__main__``) is refused, naming the thing and the fix.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import inspect
import os
import symtable
import sys
import textwrap
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

HEADER = "# Updaters saved by phasor from interactive code (Scene.save_copy)."
_PLAIN = (int, float, complex, bool, str, bytes, type(None))


class UpdaterCaptureError(ValueError):
    """An interactive updater cannot be written to a file; the message says why and how to
    fix it."""


class HeldUpdaterWarning(UserWarning):
    """``Scene.add`` kept an interactive updater for ``save_copy``: the open scene does not
    run it."""


@dataclass(frozen=True)
class Captured:
    """One updater, ready to be written: its source and what the file must define first."""

    name: str
    source: str  # the dedented ``def``, ending with a newline
    imports: tuple[tuple[str, str], ...] = ()  # (name it defines, import line)
    constants: tuple[tuple[str, str], ...] = ()  # (name, Python literal)


def is_interactive(fn: Any) -> bool:
    """Whether ``fn`` is a function that was defined where no file holds it: a notebook cell,
    a prompt, ``exec`` (its ``co_filename`` names no file on disk). A function from a file,
    wherever it is, is not interactive: the old rule about files in the scene's folder
    applies to it."""
    code = getattr(fn, "__code__", None)
    return isinstance(fn, types.FunctionType) and not os.path.isfile(code.co_filename)


def _fail(fn: Any, why: str, fix: str) -> UpdaterCaptureError:
    name = getattr(fn, "__name__", repr(fn))
    return UpdaterCaptureError(f"cannot save the updater {name!r} into the scene's folder: "
                               f"{why}. {fix}")  # fmt: skip


def capture(fn: Any) -> Captured:
    """The source of ``fn`` and its prelude, or ``UpdaterCaptureError``."""
    if not isinstance(fn, types.FunctionType):
        raise _fail(fn, f"it is a {type(fn).__name__}, not a function",
                    "write it as a plain def in a file in the scene's folder")  # fmt: skip
    name = fn.__name__
    if name == "<lambda>":
        raise _fail(fn, "a lambda has no source of its own", "give it a name with def")
    if fn.__qualname__ != name:
        raise _fail(fn, f"it is defined inside {fn.__qualname__.rpartition('.')[0]!r}",
                    "define it at the top level of the cell")  # fmt: skip
    if fn.__code__.co_freevars:
        raise _fail(fn, "it uses variables of an enclosing function "
                    f"({', '.join(fn.__code__.co_freevars)})",
                    "pass them as defaults or write them inside the function")  # fmt: skip
    try:
        raw = inspect.getsource(fn)
    except (OSError, TypeError):
        raise _fail(fn, "this shell kept no source for it (a plain Python prompt does not)",
                    "define it in an IPython or Jupyter cell, or write it in a .py file in "
                    "the scene's folder") from None  # fmt: skip
    source = textwrap.dedent(raw)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise _fail(fn, f"its source does not parse ({e.msg})",
                    "define it at the top level of one cell") from None  # fmt: skip
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise _fail(fn, "its source is not a single def", "use a plain def, not async")
    node = tree.body[0]
    if node.decorator_list:
        raise _fail(fn, "it has a decorator, whose effect a file cannot see",
                    "remove the decorator, or write the function in a .py file")  # fmt: skip
    for sub in ast.walk(node):
        if isinstance(sub, ast.Global | ast.Nonlocal):
            raise _fail(fn, "it declares global or nonlocal names (an updater is pure)",
                        "return the values instead")  # fmt: skip
    imports: dict[str, str] = {}
    constants: dict[str, str] = {}
    for used in _global_names(source, name):
        if used == name:  # a recursive call: the saved def is itself
            if fn.__globals__.get(name) is not fn:
                raise _fail(fn, f"{name!r} was defined again after it", "rename one of them")
            continue
        if used in fn.__globals__:
            kind, text = _resolve(fn, used, fn.__globals__[used])
            (imports if kind == "import" else constants)[used] = text
        elif not hasattr(builtins, used):
            raise _fail(fn, f"it uses {used!r}, which is not defined", "define or import it")
    return Captured(name, source if source.endswith("\n") else source + "\n",
                    tuple(sorted(imports.items())), tuple(sorted(constants.items())))  # fmt: skip


def _global_names(source: str, own_name: str) -> list[str]:
    """Names the function reads from outside itself (what the saved file must provide):
    implicit globals in the function and everything nested in it, and names its defaults and
    annotations read at module level."""
    top = symtable.symtable(source, "<updater>", "exec")
    found = {s.get_name() for s in top.get_symbols() if s.is_referenced()}
    stack = list(top.get_children())
    while stack:
        table = stack.pop()
        stack.extend(table.get_children())
        found |= {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
    return sorted(found)


def _resolve(fn: Any, name: str, value: Any) -> tuple[str, str]:
    """How the saved file defines ``name``: ("import", line) or ("constant", literal)."""
    if isinstance(value, types.ModuleType):
        mod = value.__name__
        if mod == "__main__" or sys.modules.get(mod) is not value or _unfindable(mod):
            raise _fail(fn, f"{name!r} is a module that cannot be imported by name",
                        "import it with an ordinary import statement")  # fmt: skip
        return "import", f"import {mod}" + ("" if mod == name else f" as {name}")
    if isinstance(value, types.FunctionType | type | types.BuiltinFunctionType):
        mod, qual = getattr(value, "__module__", None), getattr(value, "__qualname__", "")
        home = sys.modules.get(mod or "")
        findable = home is not None and "." not in qual and "<" not in qual
        if (not mod or mod == "__main__" or mod.startswith("<") or not findable
                or getattr(home, qual, None) is not value):  # fmt: skip
            kind = "a class" if isinstance(value, type) else "a function"
            raise _fail(fn, f"{name!r} is {kind} defined in the session, not in an importable "
                        "module, and the file could not find it again",
                        f"put {name} in a .py file in the scene's folder and import it there, "
                        "or write it inside the updater")  # fmt: skip
        return "import", f"from {mod} import {qual}" + ("" if qual == name else f" as {name}")
    if _is_plain(value):
        return "constant", repr(value)
    raise _fail(fn, f"{name!r} is a {type(value).__name__} from the session; only numbers, "
                "strings, None and tuples of them are kept",
                f"make it a tuple or number, or write {name} inside the updater")  # fmt: skip


def _unfindable(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except (ImportError, ValueError):
        return True


def _is_plain(value: Any) -> bool:
    if isinstance(value, tuple | frozenset):
        return all(_is_plain(v) for v in value) and _round_trips(value)
    return isinstance(value, _PLAIN) and _round_trips(value)


def _round_trips(value: Any) -> bool:
    try:
        return bool(ast.literal_eval(repr(value)) == value)
    except (ValueError, SyntaxError):
        return False  # nan, inf, a frozenset(), ...


def module_text(items: Sequence[Captured]) -> str:
    """One module that defines every captured function (same name and source: once). Names the
    functions need come first; two things under one name that differ are refused."""
    imports: dict[str, str] = {}
    constants: dict[str, str] = {}
    defs: dict[str, str] = {}
    for c in items:
        for table, entries, what in ((imports, c.imports, "import"),
                                      (constants, c.constants, "constant")):  # fmt: skip
            for n, text in entries:
                if table.setdefault(n, text) != text:
                    raise UpdaterCaptureError(
                        f"the updaters use the name {n!r} for two different {what}s "
                        f"({table[n]!r} and {text!r}); rename one of them")  # fmt: skip
        if defs.setdefault(c.name, c.source) != c.source:
            raise UpdaterCaptureError(
                f"two different functions are named {c.name!r}; rename one of them"
            )
    clash = sorted((set(imports) | set(constants)) & set(defs) | set(imports) & set(constants))
    if clash:
        raise UpdaterCaptureError(f"{clash} is used both as an updater and as another name")
    blocks = [
        HEADER + "\n# Pure functions of (t, look); edit them freely, the document names this file."
    ]
    if imports:
        blocks.append("\n".join(imports[n] for n in sorted(imports, key=lambda n: imports[n])))
    if constants:
        blocks.append("\n".join(f"{n} = {constants[n]}" for n in sorted(constants)))
    text = "\n\n".join(blocks) + "\n\n\n" + "\n\n".join(defs.values())
    verify(text, sorted(defs))
    return text.rstrip("\n") + "\n"


def verify(text: str, names: Sequence[str]) -> None:
    """The module parses, holds only imports, constants and defs (loading it runs nothing
    else) and defines each of ``names``. It is never executed here (SPEC: no ``exec``)."""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise UpdaterCaptureError(f"the saved module does not parse ({e.msg})") from e
    kinds = (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.Expr)
    odd = [type(n).__name__ for n in tree.body if not isinstance(n, kinds)]
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if odd or set(names) - defined:
        raise UpdaterCaptureError(f"the saved module is not what was captured ({odd}, "
                                  f"missing {sorted(set(names) - defined)})")  # fmt: skip
