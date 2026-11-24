"""Explorer rows (EXPLORER.md, Milestone 27 part A): one typed line of maths, read the way a
person writes it, classified into what it draws. Pure text and SymPy: no drawing, no GL.

    row = normalize_row("x' = σ(y − x)")
    row.kind       -> "ode"
    row.text       -> "x' = sigma*(y - x)"
    row.parts      -> ("sigma*(y - x)",)
    row.latex      -> "\\dot{x} = \\sigma \\left(- x + y\\right)"
    row.variables  -> ("sigma",)

Notation (``docs/decisions/explorer_advice_2026-11-24.md`` section 7.2). A small tokenizer
rewrites a line into the grammar of ``render.gpu.codegen.parse_expression`` and every part
then goes through that parser, so the safety argument (a whitelist, never evaluated as
Python) stays one argument:

- ``^`` is ``**``; superscripts (``x²``, ``x⁻¹``) are powers.
- Implicit products: ``2x``, ``2(x+1)``, ``x y``, ``(a)(b)``, ``3sin(x)``. A name followed by
  ``(`` is a call only when the parser knows it as a function (asked of the parser itself,
  so functions it learns later need no change here); otherwise it is a product (``a(b)``).
  A function name without brackets (``sin x``) is an error with a hint.
- A name is a whole run of letters, digits and ``_`` starting with a letter: ``rho``,
  ``sigma``, ``k1`` are one name each, and so is ``xy`` (a warning says to write ``x y``).
  Products need a space, ``*``, a digit or a bracket.
- Unicode: ``·`` ``⋅`` ``×`` ``∗`` are ``*``, ``−`` ``–`` are ``-``, ``÷`` is ``/``, Greek
  letters are their names (``π`` -> ``pi``, ``θ`` -> ``theta``, ``λ`` -> ``lamda``, SymPy's
  spelling, since ``lambda`` is a Python word; typed ``lambda`` becomes ``lamda`` too). A Greek
  letter is a name of its own (``kα`` is ``k*alpha``). Full-width forms and subscripts
  (``k₁``) fold to ASCII; invisible format characters are spaces.
- Derivatives only as a left-hand side: ``x'``, ``x′``, ``ẋ``, ``dx/dt`` (``d x / d t``) for
  ``x``, ``y``, ``z``. A prime anywhere else, a second derivative, or a primed name that is
  not a coordinate is an error.

Kinds (section 7.3, decided by the shape of the statement, in this order): ``variable``
(``name = number``), ``ode`` (``x' = ...``), ``point`` (a tuple with no coordinates),
``vector_field`` (``(P, Q[, R])`` in x, y, z, or ``F(x, y) = (P, Q)``), ``parametric_curve``
(a tuple in ``t``), ``parametric_surface`` (three parts in ``u, v``), ``graph`` (``y = f(x)``
or a bare expression in ``x``), ``sideways_graph`` (``x = g(y)``), ``polar``
(``r = f(theta)``), ``surface`` (``z = f(x, y)``), ``scalar_field`` (a bare expression in
``x, y``), ``implicit_curve`` (``lhs = rhs`` in x, y, not solved for one of them), and
``error``. ``normalize_row(text, kind=...)`` overrides the guess (a bare ``x*y`` as a
``surface``); an override the row's shape does not fit is an error row that says why.

Reserved names: ``x y z t u v r theta`` (coordinates) and ``pi e i`` (constants). Any other
free name is a variable; creating its ``Variable`` is the caller's job, at commit time only.

Hostile input is an error row, never an exception or a hang: the text is capped
(``MAX_CHARS``, ``MAX_TOKENS``, ``MAX_DEPTH``, ``MAX_DIGITS``) and constant powers whose
result would pass ``10**MAX_MAGNITUDE`` are refused before SymPy builds them. Each cap is the
stricter of the rows layer's and the hardened parser's (``codegen.MAX_SOURCE`` and the rest;
Milestone 27 Deviation 1), so a row meets the same limits whichever reads it first.
"""

from __future__ import annotations

import ast
import keyword
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from phasor.render.gpu import codegen
from phasor.render.gpu.codegen import parse_expression

# The rows layer's caps, each the stricter of its own and the hardened parser's (Milestone 26
# put length, node, depth and power limits into ``parse_expression``; a row is read by both,
# so the stricter one is what a person meets first, in the rows layer's words).
MAX_CHARS = min(2000, codegen.MAX_SOURCE)
MAX_TOKENS = min(400, codegen.MAX_NODES)
MAX_DEPTH = min(30, codegen.MAX_DEPTH)
MAX_DIGITS = 50
# log10 of the largest constant a row may build (the parser's own guard: e**MAX_POWER)
MAX_MAGNITUDE = min(300, int(codegen.MAX_POWER / math.log(10)))

COORDINATES = ("x", "y", "z", "t", "u", "v", "r", "theta")
CONSTANTS = ("pi", "e", "i")
RESERVED = frozenset(COORDINATES + CONSTANTS)

KINDS = (
    "variable", "ode", "point", "vector_field", "parametric_curve", "parametric_surface",
    "graph", "sideways_graph", "polar", "surface", "scalar_field", "implicit_curve", "error",
)  # fmt: skip

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPER_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")
_SYMBOLS = {
    "·": "*", "⋅": "*", "×": "*", "∗": "*", "−": "-", "–": "-", "÷": "/",
    "′": "'", "’": "'", "̇": "'", "̈": "''",
}  # fmt: skip
_TOKEN = re.compile(
    r"(?P<ws>\s+)"
    r"|(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z][A-Za-z0-9_]*)"
    r"|(?P<op>\*\*|[-+*/^])"
    r"|(?P<lp>\()|(?P<rp>\))|(?P<comma>,)|(?P<eq>=)|(?P<prime>')"
)
_HINTS = {
    "<": "inequalities are not supported yet",
    ">": "inequalities are not supported yet",
    "≤": "inequalities are not supported yet",
    "≥": "inequalities are not supported yet",
    "≠": "inequalities are not supported yet",
    "|": "write abs(x) for |x|",
    "[": "lists are not supported yet; use ( ) for brackets",
    "]": "lists are not supported yet; use ( ) for brackets",
    "{": "pieces are not supported yet; use ( ) for brackets",
    "}": "pieces are not supported yet; use ( ) for brackets",
    "!": "factorials are not supported",
}
_DERIVATIVE = re.compile(r"d([xyz])/dt")


@dataclass(frozen=True)
class Row:
    """One row, read. ``text`` is the normalised line (``normalize_row(row.text)`` gives the
    same row again); ``parts`` are its expressions in ``parse_expression``'s grammar, in the
    order the kind names them (``ode``: the right side; ``vector_field``/``point``/curves:
    the components; ``variable``: the value; ``implicit_curve``: left, right)."""

    source: str
    kind: str
    text: str
    parts: tuple[str, ...] = ()
    latex: str = ""
    error: str | None = None
    name: str | None = None  # variable's name, ODE coordinate, named field (F)
    value: float | None = None  # a variable's value
    variables: tuple[str, ...] = ()  # free names, sorted
    coordinates: tuple[str, ...] = ()  # reserved coordinates used, sorted
    guess: str = "error"  # the kind without an override
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class OdeSystem:
    """The ODE rows of a list taken together (``group_ode_rows``)."""

    dimension: int  # the number of primed coordinates
    coordinates: tuple[str, ...]  # primed coordinates, in x, y, z order
    components: tuple[tuple[str, str], ...]  # (coordinate, right side)
    variables: tuple[str, ...]  # free names the components use, sorted
    autonomous: bool  # no component uses t
    rows: tuple[int, ...]  # indices of the ODE rows in the list given
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class _Refuse(Exception):
    """A row that cannot be read; the message is for people."""


@dataclass(frozen=True)
class _Tok:
    kind: str  # num name func op lp rp comma eq prime
    text: str


@dataclass(frozen=True)
class _Side:
    parts: tuple[str, ...]
    exprs: tuple[sp.Expr, ...]
    is_tuple: bool
    names: frozenset[str]  # every name used, reserved or free

    @property
    def coords(self) -> frozenset[str]:
        return self.names & frozenset(COORDINATES)

    @property
    def free(self) -> frozenset[str]:
        return self.names - RESERVED


@dataclass(frozen=True)
class _Stmt:
    rhs: _Side
    lhs: _Side | None = None  # an expression left of "="
    derivative: str | None = None  # "x" for x' = ...
    field_name: str | None = None  # "F" for F(x, y) = ...
    field_args: tuple[str, ...] = ()
    text: str = ""
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------- reading


@lru_cache(maxsize=4096)
def _is_function(name: str, arity: int = 1) -> bool:
    """Whether the parser reads ``name(...)`` as a function call: asked of the parser, so
    this list never drifts from it."""
    args = ", ".join(["x"] * arity)
    try:
        parse_expression(f"{name}({args})", ("x",))
    except Exception:  # any refusal means "not a function"
        return False
    return True


def _greek(ch: str) -> str | None:
    n = unicodedata.name(ch, "")
    for prefix, case in (("GREEK SMALL LETTER ", str.lower), ("GREEK CAPITAL LETTER ", str.title)):
        if n.startswith(prefix):
            word = n[len(prefix) :]
            if word == "FINAL SIGMA":
                word = "SIGMA"
            if word.isalpha():
                return case(word)
    return None


def _fold(text: str) -> str:
    """Unicode to the ASCII the tokenizer reads."""
    text = _SUPER_RUN.sub(lambda m: "^" + m.group().translate(_SUPERSCRIPTS), text)
    text = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    if len(text) > MAX_CHARS:
        raise _Refuse(f"the row is too long (at most {MAX_CHARS} characters)")
    out = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
        elif ch in _SYMBOLS:
            out.append(_SYMBOLS[ch])
        elif (g := _greek(ch)) is not None:
            out.append(f" {g} ")
        elif ch.isspace() or unicodedata.category(ch) == "Cf":
            out.append(" ")
        elif ch in _HINTS:
            raise _Refuse(_HINTS[ch])
        else:
            code = f"U+{ord(ch):04X}"
            shown = f"{ch!r} ({code})" if ch.isprintable() else code
            raise _Refuse(f"cannot read the character {shown}")
    return "".join(out)


def _lex(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            ch = text[pos]
            raise _Refuse(_HINTS.get(ch, f"cannot read {ch!r}"))
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        tok = m.group()
        if kind == "num" and sum(c.isdigit() for c in tok) > MAX_DIGITS:
            raise _Refuse(f"a number has more than {MAX_DIGITS} digits")
        if kind == "op" and tok == "^":
            tok = "**"
        if kind == "name":
            if tok == "lambda":
                tok = "lamda"
            elif keyword.iskeyword(tok):
                raise _Refuse(f"{tok!r} is a reserved word; choose another name")
        toks.append(_Tok(kind, tok))
        if len(toks) > MAX_TOKENS:
            raise _Refuse(f"the row is too long (at most {MAX_TOKENS} symbols)")
    return toks


def _calls(toks: list[_Tok]) -> list[_Tok]:
    """Mark names that are calls (``sin(``) as ``func``; refuse a function without
    brackets."""
    out = []
    for k, tok in enumerate(toks):
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if tok.kind == "name" and tok.text not in RESERVED:
            followed = nxt is not None and nxt.kind == "lp"
            arity = _arity(toks, k + 1) if followed else 1
            if followed and _is_function(tok.text, arity):
                tok = _Tok("func", tok.text)
            elif followed and any(_is_function(tok.text, n) for n in (1, 2) if n != arity):
                raise _Refuse(f"{tok.text}(...) does not take {arity} arguments")
            elif not followed and _is_function(tok.text):
                raise _Refuse(f"write {tok.text}(...) with brackets, as in {tok.text}(x)")
        out.append(tok)
    return out


def _arity(toks: list[_Tok], open_at: int) -> int:
    depth, commas = 0, 0
    for tok in toks[open_at:]:
        if tok.kind == "lp":
            depth += 1
        elif tok.kind == "rp":
            depth -= 1
            if depth == 0:
                break
        elif tok.kind == "comma" and depth == 1:
            commas += 1
    return commas + 1


_ENDS = {"num", "name", "rp"}
_STARTS = {"num", "name", "func", "lp"}


def _products(toks: list[_Tok]) -> list[_Tok]:
    """Insert the ``*`` of implicit products; refuse two operators or two numbers in a row."""
    out: list[_Tok] = []
    for tok in toks:
        prev = out[-1] if out else None
        if prev is not None and prev.kind in _ENDS and tok.kind in _STARTS:
            if prev.kind == "num" and tok.kind == "num":
                raise _Refuse(f"two numbers in a row ({prev.text} {tok.text}); add an operator")
            out.append(_Tok("op", "*"))
        elif prev is not None and prev.kind == "op" and tok.kind == "op" and tok.text not in "+-":
            raise _Refuse(f"two operators in a row ({prev.text} {tok.text})")
        out.append(tok)
    return out


def _join(toks: list[_Tok]) -> str:
    out = []
    prev: _Tok | None = None
    for tok in toks:
        if tok.kind == "op" and tok.text in "+-":
            unary = prev is None or prev.kind in ("op", "lp", "comma", "eq")
            out.append(tok.text if unary else f" {tok.text} ")
        elif tok.kind == "comma":
            out.append(", ")
        elif tok.kind == "eq":
            out.append(" = ")
        else:
            out.append(tok.text)
        prev = tok
    return "".join(out)


def _magnitude(node: ast.AST) -> float | None:
    """log10 of a constant subtree's size (an upper estimate), None if it is not constant;
    refuses a constant that would pass ``10**MAX_MAGNITUDE`` before SymPy builds it."""
    match node:
        case ast.Constant(value=bool()):
            return None
        case ast.Constant(value=int() | float() as v):
            m = math.log10(abs(v)) if v else -math.inf
        case ast.Name(id="pi" | "e"):
            m = 0.5
        case ast.Name(id="i"):
            m = 0.0
        case ast.UnaryOp(operand=operand):
            return _magnitude(operand)
        case ast.BinOp(left=left, op=op, right=right):
            a, b = _magnitude(left), _magnitude(right)
            if a is None or b is None:
                return None
            if isinstance(op, ast.Add | ast.Sub):
                m = max(a, b) + 0.302
            elif isinstance(op, ast.Mult):
                m = a + b
            elif isinstance(op, ast.Div):
                if b == -math.inf:
                    return None  # a division by zero: the parser says "not finite"
                m = a - b
            elif isinstance(op, ast.Pow):
                big = abs(a) * (10.0**b if b < 300 else math.inf)
                m = a if a in (0.0, -math.inf) else big  # 0**n and 1**n stay small
            else:
                return None
        case ast.AST():
            for child in ast.iter_child_nodes(node):
                _magnitude(child)
            return None
    if m > MAX_MAGNITUDE + 1e-9:  # slack for 10**300 itself
        raise _Refuse(f"a number in this row is too large (above 1e{MAX_MAGNITUDE})")
    return m


def _parse_part(text: str, names: frozenset[str]) -> sp.Expr:
    if not text:
        raise _Refuse("an empty part")
    try:
        _magnitude(ast.parse(text, mode="eval"))
    except SyntaxError:
        pass  # parse_expression gives the message
    except (RecursionError, MemoryError) as e:
        raise _Refuse("the row is too deeply nested") from e
    variables = tuple(sorted(names - set(CONSTANTS)))
    try:
        return parse_expression(text, variables)
    except (RecursionError, MemoryError) as e:
        raise _Refuse("the row is too long or too deeply nested to read") from e
    except Exception as e:  # FunctionError, and anything SymPy raises
        raise _Refuse(str(e) or type(e).__name__) from e


def _split(toks: list[_Tok], what: str) -> tuple[list[list[_Tok]], bool]:
    """One side of a statement: its parts and whether it is a tuple ``(a, b[, c])``."""
    if not toks:
        raise _Refuse(f"nothing on the {what}")
    opened: list[int] = []  # indices of the open brackets
    closes: dict[int, int] = {}
    for k, tok in enumerate(toks):
        if tok.kind == "lp":
            opened.append(k)
            if len(opened) > MAX_DEPTH:
                raise _Refuse(f"brackets nest too deep (at most {MAX_DEPTH})")
        elif tok.kind == "rp":
            if not opened:
                raise _Refuse("a ')' without its '('")
            closes[opened.pop()] = k
    if opened:
        raise _Refuse("a '(' without its ')'")
    spans_all = toks[0].kind == "lp" and closes[0] == len(toks) - 1
    parts: list[list[_Tok]] = [[]]
    for k, tok in enumerate(toks):
        if tok.kind == "lp":
            opened.append(k)
        elif tok.kind == "rp":
            opened.pop()
        if tok.kind != "comma":
            parts[-1].append(tok)
        elif not opened:
            raise _Refuse("put the components in brackets: (P, Q)")
        elif opened[-1] > 0 and toks[opened[-1] - 1].kind == "func":
            parts[-1].append(tok)  # a call's arguments
        elif spans_all and opened == [0]:
            parts.append([])
        else:
            raise _Refuse("a component cannot itself be a list")
    if len(parts) == 1:
        return [toks], False
    parts[0] = parts[0][1:]  # the outer brackets
    parts[-1] = parts[-1][:-1]
    if any(not p for p in parts):
        raise _Refuse("an empty component")
    if len(parts) > 3:
        raise _Refuse(f"at most three components ({len(parts)} given)")
    return parts, True


def _side(toks: list[_Tok], what: str) -> _Side:
    groups, is_tuple = _split(toks, what)
    parts, exprs, names = [], [], set()
    for g in groups:
        text = _join(g)
        used = frozenset(t.text for t in g if t.kind == "name")
        parts.append(text)
        exprs.append(_parse_part(text, used))
        names |= used
    return _Side(tuple(parts), tuple(exprs), is_tuple, frozenset(names))


def _split_words(name: str) -> bool:
    """Whether a name of two or more letters is made only of reserved names (``xy``)."""
    if len(name) < 2 or name in RESERVED or _greek_name(name):
        return False
    ok = [True] + [False] * len(name)
    for k in range(len(name)):
        if ok[k]:
            for r in RESERVED:
                if name.startswith(r, k):
                    ok[k + len(r)] = True
    return ok[-1]


def _greek_name(name: str) -> bool:
    try:
        unicodedata.lookup(f"GREEK SMALL LETTER {name.upper()}")
    except KeyError:
        return False
    return True


def _read(source: str) -> _Stmt:
    if not isinstance(source, str):
        raise _Refuse("a row is text")
    if len(source) > MAX_CHARS:
        raise _Refuse(f"the row is too long (at most {MAX_CHARS} characters)")
    raw = _lex(_fold(source))
    if not raw:
        raise _Refuse("the row is empty")
    eqs = [k for k, t in enumerate(raw) if t.kind == "eq"]
    if len(eqs) > 1:
        raise _Refuse("use one '=' in a row")
    lhs_raw, rhs_raw = (raw[: eqs[0]], raw[eqs[0] + 1 :]) if eqs else (None, raw)
    if any(t.kind == "prime" for t in rhs_raw):
        raise _Refuse("a prime (x') goes only on the left of '='")
    warnings = sorted(
        {
            f"{t.text!r} is one name; write {' '.join(_words(t.text))} for a product"
            for t in raw
            if t.kind == "name" and _split_words(t.text)
        }
    )
    rhs_toks = _products(_calls(rhs_raw))
    if lhs_raw is not None and not lhs_raw:
        raise _Refuse("nothing on the left of '='")
    derivative = field_name = None
    field_args: tuple[str, ...] = ()
    lhs = None
    if lhs_raw is not None:
        flat = "".join(t.text for t in lhs_raw)
        primes = sum(t.kind == "prime" for t in lhs_raw)
        if (m := _DERIVATIVE.fullmatch(flat)) is not None:
            derivative = m.group(1)
        elif primes:
            head = lhs_raw[0]
            if head.kind != "name" or any(t.kind != "prime" for t in lhs_raw[1:]):
                raise _Refuse("a prime (x') goes only right after x, y or z, left of '='")
            if primes > 1:
                raise _Refuse(
                    "second derivatives are not supported; write a system (x' = v, v' = ...)"
                )
            if head.text not in ("x", "y", "z"):
                raise _Refuse(f"only x', y' and z' can be derivatives (not {head.text}')")
            derivative = head.text
        elif _is_definition(lhs_raw):
            field_name = lhs_raw[0].text
            field_args = tuple(t.text for t in lhs_raw[2:-1:2])
        else:
            lhs = _side(_products(_calls(lhs_raw)), "left of '='")
    rhs = _side(rhs_toks, "right of '='" if lhs_raw is not None else "row")
    if derivative is not None:
        text = f"{derivative}' = {_text(rhs)}"
    elif field_name is not None:
        text = f"{field_name}({', '.join(field_args)}) = {_text(rhs)}"
    elif lhs is not None:
        text = f"{_text(lhs)} = {_text(rhs)}"
    else:
        text = _text(rhs)
    return _Stmt(rhs, lhs, derivative, field_name, field_args, text, tuple(warnings))


def _words(name: str) -> list[str]:
    out, k = [], 0
    while k < len(name):
        for r in sorted(RESERVED, key=len, reverse=True):
            if name.startswith(r, k):
                out.append(r)
                k += len(r)
                break
        else:
            return [name]
    return out


def _is_definition(toks: list[_Tok]) -> bool:
    """``F(x, y)``: a name that is not a function, then bracketed names."""
    if len(toks) < 4 or toks[0].kind != "name" or toks[0].text in RESERVED:
        return False
    if toks[1].kind != "lp" or toks[-1].kind != "rp" or _is_function(toks[0].text):
        return False
    inner = toks[2:-1]
    return all(t.kind == ("name" if k % 2 == 0 else "comma") for k, t in enumerate(inner)) and (
        len(inner) % 2 == 1
    )


def _text(side: _Side) -> str:
    return f"({', '.join(side.parts)})" if side.is_tuple else side.parts[0]


# ---------------------------------------------------------------- kinds


def _only(coords: frozenset[str], allowed: str, kind: str) -> None:
    extra = sorted(coords - set(allowed.split()))
    if extra:
        raise _Refuse(f"{kind} cannot use {', '.join(extra)}")


def _expr(side: _Side | None, kind: str) -> _Side:
    if side is None or side.is_tuple:
        raise _Refuse(f"{kind} is one expression, not a list")
    return side


def _lhs_name(s: _Stmt) -> str | None:
    if s.lhs is None or s.lhs.is_tuple:
        return None
    text = s.lhs.parts[0]
    return text if text.isidentifier() else None


def _plain(s: _Stmt) -> bool:
    return s.lhs is None and s.derivative is None and s.field_name is None


def _k_variable(s: _Stmt) -> dict:
    name = _lhs_name(s)
    if name is None or s.rhs.is_tuple:
        raise _Refuse("a variable is written name = number")
    if name in RESERVED:
        raise _Refuse(f"{name} is a reserved name ({' '.join(COORDINATES + CONSTANTS)})")
    if s.rhs.names - set(CONSTANTS):
        raise _Refuse(
            f"a variable is set to a number; {name} = {s.rhs.parts[0]} uses "
            f"{', '.join(sorted(s.rhs.names - set(CONSTANTS)))}"
        )
    try:
        value = complex(s.rhs.exprs[0])
    except (TypeError, ValueError, OverflowError) as e:
        raise _Refuse(f"{name} must be a real number") from e
    if value.imag != 0:
        raise _Refuse(f"{name} must be a real number")
    if not math.isfinite(value.real):
        raise _Refuse(f"{name} must be a finite number")
    return {"name": name, "value": value.real}


def _k_ode(s: _Stmt) -> dict:
    if s.derivative is None:
        raise _Refuse("an ODE row is written x' = ...")
    _expr(s.rhs, "the right side of an ODE")
    _only(s.rhs.coords, "x y z t", "an ODE")
    return {"name": s.derivative}


def _k_point(s: _Stmt) -> dict:
    if not _plain(s) or not s.rhs.is_tuple:
        raise _Refuse("a point is written (a, b) or (a, b, c)")
    if s.rhs.coords:
        raise _Refuse(f"a point cannot use {', '.join(sorted(s.rhs.coords))}")
    return {}


def _k_vector_field(s: _Stmt) -> dict:
    if not s.rhs.is_tuple or not (_plain(s) or s.field_name is not None):
        raise _Refuse("a vector field is written (P, Q) or F(x, y) = (P, Q)")
    n = len(s.rhs.parts)
    allowed = "x y" if n == 2 else "x y z"
    if s.field_name is not None and s.field_args != tuple(allowed.split()):
        raise _Refuse(f"a field of {n} components is {s.field_name}({', '.join(allowed.split())})")
    _only(s.rhs.coords, allowed, f"a vector field of {n} components")
    return {"name": s.field_name}


def _k_parametric_curve(s: _Stmt) -> dict:
    if not _plain(s) or not s.rhs.is_tuple:
        raise _Refuse("a parametric curve is written (f(t), g(t))")
    _only(s.rhs.coords, "t", "a parametric curve")
    return {}


def _k_parametric_surface(s: _Stmt) -> dict:
    if not _plain(s) or not s.rhs.is_tuple or len(s.rhs.parts) != 3:
        raise _Refuse("a parametric surface is written (f(u, v), g(u, v), h(u, v))")
    _only(s.rhs.coords, "u v", "a parametric surface")
    return {}


def _solved(s: _Stmt, lhs: str, allowed: str, kind: str, bare: bool) -> dict:
    if (_plain(s) and bare) or _lhs_name(s) == lhs:
        _expr(s.rhs, kind)
    else:
        raise _Refuse(f"{kind} is written {lhs} = ...")
    _only(s.rhs.coords, allowed, kind)
    return {}


def _k_graph(s: _Stmt) -> dict:
    return _solved(s, "y", "x", "a graph", bare=True)


def _k_sideways_graph(s: _Stmt) -> dict:
    return _solved(s, "x", "y", "a sideways graph", bare=False)


def _k_polar(s: _Stmt) -> dict:
    return _solved(s, "r", "theta", "a polar curve", bare=False)


def _k_surface(s: _Stmt) -> dict:
    return _solved(s, "z", "x y", "a surface", bare=True)


def _k_scalar_field(s: _Stmt) -> dict:
    if not _plain(s):
        raise _Refuse("a scalar field is a bare expression in x, y")
    _expr(s.rhs, "a scalar field")
    _only(s.rhs.coords, "x y", "a scalar field")
    return {}


def _k_implicit_curve(s: _Stmt) -> dict:
    if s.lhs is None or s.lhs.is_tuple or s.rhs.is_tuple:
        raise _Refuse("an implicit curve is written lhs = rhs in x and y")
    coords = s.lhs.coords | s.rhs.coords
    if "z" in coords:
        raise _Refuse("implicit surfaces are not supported yet")
    _only(coords, "x y", "an implicit curve")
    if not coords:
        raise _Refuse("an implicit curve uses x or y")
    return {}


_ACCEPT = {
    "variable": _k_variable, "ode": _k_ode, "point": _k_point,
    "vector_field": _k_vector_field, "parametric_curve": _k_parametric_curve,
    "parametric_surface": _k_parametric_surface, "graph": _k_graph,
    "sideways_graph": _k_sideways_graph, "polar": _k_polar, "surface": _k_surface,
    "scalar_field": _k_scalar_field, "implicit_curve": _k_implicit_curve,
}  # fmt: skip
# the guess takes the first that fits: a bare expression in x is a graph, in x, y a scalar
# field (a surface only by override)
_GUESS = (
    "variable", "ode", "point", "vector_field", "parametric_curve", "parametric_surface",
    "graph", "sideways_graph", "polar", "scalar_field", "surface", "implicit_curve",
)  # fmt: skip


def _no_kind(s: _Stmt) -> str:
    """Why no kind fits, in words."""
    if s.field_name is not None and not s.rhs.is_tuple:
        return (
            f"user-defined functions ({s.field_name}(...) = ...) are not supported yet; "
            "write y = ... or a field F(x, y) = (P, Q)"
        )
    name = _lhs_name(s)
    if name is not None and name not in RESERVED and not s.rhs.is_tuple:
        try:
            _k_variable(s)
        except _Refuse as e:
            return f"{e} (not supported yet)" if "uses" in str(e) else str(e)
    if name in RESERVED - {"x", "y", "z", "r"}:
        return f"{name} is a reserved name; it cannot be set"
    coords = s.rhs.coords | (s.lhs.coords if s.lhs is not None else frozenset())
    if s.lhs is not None and "z" in coords:
        return "implicit surfaces are not supported yet"
    if s.lhs is not None and not coords:
        return "an equation of numbers draws nothing"
    if not s.rhs.is_tuple and s.lhs is None:
        if coords == {"t"}:
            return "write (f(t), g(t)) for a curve in t"
        if coords == {"theta"}:
            return "write r = f(theta) for a polar curve"
        if coords <= {"x", "y", "z"}:
            return "a scalar field of x, y, z is not supported yet"
    return f"no kind of row uses {', '.join(sorted(coords))} together like this"


def _latex_side(side: _Side) -> str:
    parts = [sp.latex(e) for e in side.exprs]
    return f"\\left({', '.join(parts)}\\right)" if side.is_tuple else parts[0]


def _latex(s: _Stmt) -> str:
    try:
        rhs = _latex_side(s.rhs)
        if s.derivative is not None:
            return f"\\dot{{{s.derivative}}} = {rhs}"
        if s.field_name is not None:
            head = sp.latex(sp.Symbol(s.field_name))
            args = ", ".join(sp.latex(sp.Symbol(a)) for a in s.field_args)
            return f"{head}\\left({args}\\right) = {rhs}"
        if s.lhs is not None:
            return f"{_latex_side(s.lhs)} = {rhs}"
        return rhs
    except Exception:  # a label is never worth an exception
        return ""


def _classify(s: _Stmt, source: str, kind: str | None) -> Row:
    guess, found, why = "error", {}, None
    for k in _GUESS:
        try:
            found = _ACCEPT[k](s)
        except _Refuse:
            continue
        guess = k
        break
    else:
        why = _no_kind(s)
    chosen = guess
    if kind is not None and kind != guess:
        if kind not in _ACCEPT:
            why = f"unknown row kind {kind!r}; one of {', '.join(_GUESS)}"
        else:
            try:
                found, why = _ACCEPT[kind](s), None
            except _Refuse as e:
                why = f"this row cannot be a {kind.replace('_', ' ')}: {e}"
        chosen = kind
    names = s.rhs.names | (s.lhs.names if s.lhs is not None else frozenset())
    common = {
        "source": source,
        "text": s.text,
        "variables": tuple(sorted(names - RESERVED - {found.get("name")} - {None})),
        "coordinates": tuple(c for c in COORDINATES if c in names),
        "guess": guess,
        "warnings": s.warnings,
    }
    if why is not None:
        return Row(kind="error", error=why, **common)
    parts = s.rhs.parts
    if chosen == "implicit_curve":
        parts = (s.lhs.parts[0], s.rhs.parts[0])
    if chosen == "variable":
        common["variables"] = ()
    return Row(
        kind=chosen,
        parts=parts,
        latex=_latex(s),
        name=found.get("name"),
        value=found.get("value"),
        **common,
    )


# ---------------------------------------------------------------- public


def normalize_row(text: str, kind: str | None = None) -> Row:
    """Read one typed line: its kind (or ``kind``, an override), normalised text, parts,
    LaTeX and, when it cannot be read, a plain-words ``error`` (``kind == "error"``).
    Never raises on any text; bounded in time by the module's caps."""
    try:
        stmt = _read(text)
    except _Refuse as e:
        return Row(source=text, kind="error", text=_safe_text(text), error=str(e))
    except (RecursionError, MemoryError):
        return Row(source=text, kind="error", text=_safe_text(text), error="the row is too long")
    return _classify(stmt, text, kind)


def classify_row(row: Row | str, kind: str | None = None) -> Row:
    """The row's kind by the shape of its statement (section 7.3), or ``kind`` when given
    and the shape fits it (an error row saying why when it does not). Re-reads the row's
    source, so it costs what ``normalize_row`` costs."""
    return normalize_row(row.source if isinstance(row, Row) else row, kind)


def group_ode_rows(rows: Iterable[Row]) -> OdeSystem:
    """The ``ode`` rows of a list as one system: its dimension is the number of primed
    coordinates (2 or 3, x and y always among them); a component may use only primed
    coordinates, ``t`` and variables. Other kinds of row are ignored."""
    comps: dict[str, str] = {}
    used: set[str] = set()
    free: set[str] = set()
    index: list[int] = []
    errors: list[str] = []
    for k, row in enumerate(rows):
        if row.kind != "ode" or row.name is None:
            continue
        index.append(k)
        if row.name in comps:
            errors.append(f"two rows define {row.name}'")
            continue
        comps[row.name] = row.parts[0]
        used |= set(row.coordinates)
        free |= set(row.variables)
    coords = tuple(c for c in ("x", "y", "z") if c in comps)
    if not coords:
        errors.append("no rows of the form x' = ...")
    elif len(coords) == 1:
        errors.append(f"a system needs at least two rows ({coords[0]}' alone)")
    elif coords == ("x", "z"):
        errors.append("z' needs y' as well")
    for c in sorted(used & {"x", "y", "z"} - set(coords)):
        if coords:
            errors.append(f"the system uses {c} but no row defines {c}'")
    return OdeSystem(
        dimension=len(coords),
        coordinates=coords,
        components=tuple((c, comps[c]) for c in coords),
        variables=tuple(sorted(free)),
        autonomous="t" not in used,
        rows=tuple(index),
        errors=tuple(errors),
    )


def _safe_text(text: object) -> str:
    """A refused row keeps its whole text, so reading it again refuses it again."""
    return text.strip() if isinstance(text, str) else ""
