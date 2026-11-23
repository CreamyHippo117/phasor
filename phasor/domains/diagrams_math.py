"""Text and numbers behind the diagram kit (``domains/diagrams.py``): pure, no drawing.

Every count a diagram draws is bounded here, by the constants below, and checked when its
parameters are validated (or, for a file, when it is read: a file past a limit is cut short
and lint says so as ``truncated``). Tick values are in ``core/ticks.py``.

Code is highlighted by pygments (the ``code`` extra) when it is installed; without it, or for
a language pygments does not know, every character is plain and the code is still drawn.
"""

from __future__ import annotations

import csv
import functools
import io

from phasor.core import extras
from phasor.core import text as text_metrics

# Tables
MAX_ROWS = 40
MAX_COLS = 12
MAX_CELLS = 300  # rows times columns
MAX_CELL = 60  # characters in one cell (or one bar's name, one matrix entry)
# Bar charts
MAX_BARS = 40
# Matrices
MAX_MATRIX = 10  # rows, and columns
# Code
MAX_LINES = 120
MAX_LINE = 120  # characters per line, after tabs become spaces
MAX_CODE = 12_000  # characters in all
TAB = 4
MAX_LANGUAGE = 32
# Files (a table's .csv, a code block's source): read at most this many bytes
MAX_FILE_BYTES = 256 * 1024

# Character classes of highlighted code, one letter per character.
CLASSES = {
    "p": "plain",
    "k": "keyword",
    "f": "name",
    "s": "string",
    "n": "number",
    "c": "comment",
    "g": "gutter",  # line numbers
}


def column_name(j: int) -> str:
    """Spreadsheet column letters: 0 -> ``a``, 25 -> ``z``, 26 -> ``aa``."""
    out = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        out = chr(ord("a") + r) + out
    return out


def check_text(s: str, what: str, limit: int = MAX_CELL) -> str:
    """A cell, name or entry: at most ``limit`` characters, one line, typesettable math."""
    if not isinstance(s, str):
        raise ValueError(f"{what} must be text, got {s!r}")
    if len(s) > limit:
        raise ValueError(f"{what} is longer than {limit} characters")
    if "\n" in s or "\r" in s:
        raise ValueError(f"{what} must be one line")
    text_metrics.check_math(s)
    return s


def check_grid(rows, what: str, max_rows: int, max_cols: int, max_cells: int = MAX_CELLS):
    """Rows of text: at most ``max_rows`` of at most ``max_cols``, ``max_cells`` in all, each
    cell checked (``check_text``). Rows may be shorter than the longest (padded with "")."""
    if len(rows) > max_rows:
        raise ValueError(f"{what}: at most {max_rows} rows, got {len(rows)}")
    widest = max((len(r) for r in rows), default=0)
    if widest > max_cols:
        raise ValueError(f"{what}: at most {max_cols} columns, got {widest}")
    if len(rows) * widest > max_cells:
        raise ValueError(f"{what}: at most {max_cells} cells, got {len(rows) * widest}")
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            check_text(cell, f"{what}[{i}][{j}]")
    return rows


def padded(rows) -> list[list[str]]:
    widest = max((len(r) for r in rows), default=0)
    return [list(r) + [""] * (widest - len(r)) for r in rows]


def parse_csv(data: bytes) -> tuple[list[list[str]], str | None, str | None]:
    """(rows, problem, cut) of a CSV file: ``problem`` when it cannot be read at all (not
    UTF-8, not CSV, a cell that cannot be typeset), ``cut`` when it was longer or wider than a
    table may be (only the first ``MAX_ROWS`` rows and ``MAX_COLS`` columns, cells cut to
    ``MAX_CELL`` characters, are kept)."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        return [], f"not UTF-8 text ({e.reason} at byte {e.start})", None
    rows, cut = [], []
    try:
        for i, row in enumerate(csv.reader(io.StringIO(text))):
            if i >= MAX_ROWS:
                cut.append(f"more than {MAX_ROWS} rows")
                break
            if len(row) > MAX_COLS:
                cut.append(f"more than {MAX_COLS} columns")
                row = row[:MAX_COLS]
            if any(len(c) > MAX_CELL for c in row):
                cut.append(f"cells longer than {MAX_CELL} characters")
                row = [c[:MAX_CELL] for c in row]
            rows.append([c.replace("\r", " ").replace("\n", " ").strip() for c in row])
    except csv.Error as e:
        return [], f"not CSV ({e})", None
    while len(rows) * max((len(r) for r in rows), default=0) > MAX_CELLS:
        rows.pop()
        cut.append(f"more than {MAX_CELLS} cells")
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            try:
                text_metrics.check_math(cell)
            except ValueError as e:
                return [], f"cell {column_name(j)}{i + 1}: {e}", None
    return rows, None, "; ".join(dict.fromkeys(cut)) or None


def code_lines(source: str) -> list[str]:
    """The lines of a piece of code, tabs as spaces, no line endings."""
    return source.replace("\r\n", "\n").replace("\r", "\n").expandtabs(TAB).split("\n")


def check_code(source: str) -> str:
    """Code a ``Code`` object may hold: at most ``MAX_CODE`` characters, ``MAX_LINES``
    lines of at most ``MAX_LINE`` characters, nothing but printable text."""
    if len(source) > MAX_CODE:
        raise ValueError(f"code is longer than {MAX_CODE} characters")
    lines = code_lines(source)
    if len(lines) > MAX_LINES:
        raise ValueError(f"code has more than {MAX_LINES} lines")
    for i, line in enumerate(lines):
        if len(line) > MAX_LINE:
            raise ValueError(f"line {i + 1} is longer than {MAX_LINE} characters")
        bad = [ch for ch in line if not ch.isprintable()]
        if bad:
            raise ValueError(f"line {i + 1} holds a control character {bad[0]!r}")
    return source


def read_code(data: bytes) -> tuple[str, str | None, str | None]:
    """(source, problem, cut) of a code file: cut to the limits of ``check_code`` (and
    control characters shown as ``?``) instead of refused."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        return "", f"not UTF-8 text ({e.reason} at byte {e.start})", None
    lines, cut = code_lines(text), []
    if lines and lines[-1] == "":
        lines.pop()  # the file's last line ending
    if len(lines) > MAX_LINES:
        lines, cut = lines[:MAX_LINES], [f"more than {MAX_LINES} lines"]
    if any(len(x) > MAX_LINE for x in lines):
        lines = [x[:MAX_LINE] for x in lines]
        cut.append(f"lines longer than {MAX_LINE} characters")
    clean = ["".join(ch if ch.isprintable() else "?" for ch in x) for x in lines]
    return "\n".join(clean)[:MAX_CODE], None, "; ".join(cut) or None


def _class_of(token) -> str:
    from pygments import token as T

    if token in T.Comment:
        return "c"
    if token in T.Literal.String:
        return "s"
    if token in T.Literal.Number:
        return "n"
    if token in T.Keyword or token in T.Operator.Word:
        return "k"
    if (
        token in T.Name.Function
        or token in T.Name.Class
        or token in T.Name.Builtin
        or (token in T.Name.Decorator)
    ):
        return "f"
    return "p"


def highlight(source: str, language: str) -> tuple[tuple[str, ...], str]:
    """One class letter per character of each line of ``source`` (``CLASSES``), and how
    they were found: ``pygments``, ``plain`` (pygments is not installed) or ``unknown``
    (pygments does not know ``language``). Cached by the source, the language and whether
    pygments is installed."""
    return _highlight(source, language, extras.available("pygments"))


@functools.lru_cache(maxsize=64)
def _highlight(source: str, language: str, have: bool) -> tuple[tuple[str, ...], str]:
    lines = code_lines(source)
    plain = tuple("p" * len(x) for x in lines)
    if not have:
        return plain, "plain"
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound

    try:
        lexer = get_lexer_by_name(language, stripnl=False, stripall=False, ensurenl=False)
    except ClassNotFound:
        return plain, "unknown"
    text = "\n".join(lines)
    classes = []
    for token, value in lexer.get_tokens(text):
        classes.append(_class_of(token) * len(value))
    flat = "".join(classes)
    if len(flat) != len(text):  # a lexer that changed the text: keep it plain
        return plain, "plain"
    out, k = [], 0
    for x in lines:
        out.append(flat[k : k + len(x)])
        k += len(x) + 1
    return tuple(out), "pygments"
