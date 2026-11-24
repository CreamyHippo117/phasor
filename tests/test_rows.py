"""Explorer rows (Milestone 27 part A): notation, classification, ODE grouping, hostile input."""

from __future__ import annotations

import random
import time

import pytest
import sympy as sp

from phasor.domains.rows import (
    KINDS,
    MAX_CHARS,
    Row,
    classify_row,
    group_ode_rows,
    normalize_row,
)
from phasor.render.gpu.codegen import parse_expression

# typed line -> (kind, normalised text, a piece of the error or None)
TABLE = [
    # graphs, implicit products, ^
    ("y = x^2", "graph", "y = x**2", None),
    ("x^2", "graph", "x**2", None),
    ("2x", "graph", "2*x", None),
    ("2(x+1)", "graph", "2*(x + 1)", None),
    ("(x)(x+1)", "graph", "(x)*(x + 1)", None),
    ("3sin(x)", "graph", "3*sin(x)", None),
    ("3 sin (x)", "graph", "3*sin(x)", None),
    ("y = 2x^2 + 3x - 1", "graph", "y = 2*x**2 + 3*x - 1", None),
    ("y=-x", "graph", "y = -x", None),
    ("y = x^-1", "graph", "y = x**-1", None),
    ("y = x**2", "graph", "y = x**2", None),
    ("e^x", "graph", "e**x", None),
    ("2e3x", "graph", "2e3*x", None),
    ("y = 1e-3x", "graph", "y = 1e-3*x", None),
    ("y = x^2 + q", "graph", "y = x**2 + q", None),
    ("y = a(x - 1)", "graph", "y = a*(x - 1)", None),
    ("y = 3", "graph", "y = 3", None),
    ("y = sqrt(x)", "graph", "y = sqrt(x)", None),
    ("rho", "graph", "rho", None),
    # unicode
    ("y = x·2", "graph", "y = x*2", None),
    ("y = 2×x", "graph", "y = 2*x", None),
    ("y = x − 1", "graph", "y = x - 1", None),
    ("y = πx", "graph", "y = pi*x", None),
    ("y = 2π", "graph", "y = 2*pi", None),
    ("x²", "graph", "x**2", None),
    ("x³ + y²", "scalar_field", "x**3 + y**2", None),
    ("x⁻¹", "graph", "x**-1", None),
    ("ｘ + 1", "graph", "x + 1", None),
    ("y = kα x", "graph", "y = k*alpha*x", None),
    ("y = x​ + 1", "graph", "y = x + 1", None),
    # other solved forms
    ("x = y^2", "sideways_graph", "x = y**2", None),
    ("x = 3", "sideways_graph", "x = 3", None),
    ("r = 1 + cos(θ)", "polar", "r = 1 + cos(theta)", None),
    ("r = theta", "polar", "r = theta", None),
    ("z = x^2 - y^2", "surface", "z = x**2 - y**2", None),
    ("z = 1", "surface", "z = 1", None),
    # scalar fields and implicit curves
    ("x y", "scalar_field", "x*y", None),
    ("1/(x-y)", "scalar_field", "1/(x - y)", None),
    ("x^2 + y^2 = 1", "implicit_curve", "x**2 + y**2 = 1", None),
    ("y = x y", "implicit_curve", "y = x*y", None),
    ("sin(x) = cos(y)", "implicit_curve", "sin(x) = cos(y)", None),
    ("x^2 = 1", "implicit_curve", "x**2 = 1", None),
    # variables
    ("sigma = 10", "variable", "sigma = 10", None),
    ("ρ = 28", "variable", "rho = 28", None),
    ("β = 8/3", "variable", "beta = 8/3", None),
    ("a = -2.5", "variable", "a = -2.5", None),
    ("k1 = 2pi", "variable", "k1 = 2*pi", None),
    ("k₁ = 3", "variable", "k1 = 3", None),
    ("λ = 2", "variable", "lamda = 2", None),
    ("lambda = 2", "variable", "lamda = 2", None),
    ("Ω = 1", "variable", "Omega = 1", None),
    # ODE components
    ("x' = y", "ode", "x' = y", None),
    ("ẋ = -x", "ode", "x' = -x", None),
    ("x′ = y", "ode", "x' = y", None),
    ("dx/dt = sigma(y - x)", "ode", "x' = sigma*(y - x)", None),
    ("d y / d t = x", "ode", "y' = x", None),
    ("z' = x y - beta z", "ode", "z' = x*y - beta*z", None),
    ("y' = -sin(x) + cos(t)", "ode", "y' = -sin(x) + cos(t)", None),
    # tuples
    ("(y, -x)", "vector_field", "(y, -x)", None),
    ("(-y, x, z)", "vector_field", "(-y, x, z)", None),
    ("F(x, y) = (y, -x)", "vector_field", "F(x, y) = (y, -x)", None),
    ("(cos(t), sin(t))", "parametric_curve", "(cos(t), sin(t))", None),
    ("(cos(t), sin(t), t/5)", "parametric_curve", "(cos(t), sin(t), t/5)", None),
    ("(u, v, u v)", "parametric_surface", "(u, v, u*v)", None),
    ("(1, 2)", "point", "(1, 2)", None),
    ("(1, 2, 3)", "point", "(1, 2, 3)", None),
    ("(a, 2b)", "point", "(a, 2*b)", None),
    ("(sin(x)/x, 1)", "vector_field", "(sin(x)/x, 1)", None),
    # errors
    ("", "error", "", "empty"),
    ("   ", "error", "", "empty"),
    ("sin x", "error", "sin x", "brackets"),
    ("(cos t, sin t)", "error", "(cos t, sin t)", "brackets"),
    ("x^2 + y^2 + z^2 = 1", "error", "x**2 + y**2 + z**2 = 1", "implicit surfaces"),
    ("w' = x", "error", "w' = x", "only x'"),
    ("x'' = -x", "error", "x'' = -x", "second derivatives"),
    ("y = x'", "error", "y = x'", "left of"),
    ("f(x) = x^2", "error", "f(x) = x**2", "user-defined functions"),
    ("(x, t)", "error", "(x, t)", "together"),
    ("(1, 2, 3, 4)", "error", "(1, 2, 3, 4)", "at most three"),
    ("(1,)", "error", "(1,)", "empty component"),
    ("((1, 2), 3)", "error", "((1, 2), 3)", "list"),
    ("x, y", "error", "x, y", "brackets"),
    ("y = ", "error", "y =", "nothing on the right"),
    ("= 3", "error", "= 3", "nothing on the left"),
    ("y == x", "error", "y == x", "one '='"),
    ("y = x = 2", "error", "y = x = 2", "one '='"),
    ("y < x", "error", "y < x", "inequalit"),
    ("y ≤ x", "error", "y ≤ x", "inequalit"),
    ("|x|", "error", "|x|", "abs"),
    ("[1, 2]", "error", "[1, 2]", "lists"),
    ("2 3", "error", "2 3", "two numbers"),
    ("x * * 2", "error", "x * * 2", "two operators"),
    ("t = 3", "error", "t = 3", "reserved"),
    ("if = 2", "error", "if = 2", "reserved word"),
    ("a = 2b", "error", "a = 2*b", "set to a number"),
    ("a = i", "error", "a = i", "real number"),
    ("sin(x, y)", "error", "sin(x, y)", "does not take 2 arguments"),
    ("2 = 3", "error", "2 = 3", "numbers"),
    ("x y z", "error", "x*y*z", "x, y, z"),
    ("sin(t)", "error", "sin(t)", "curve in t"),
    ("cos(theta)", "error", "cos(theta)", "polar"),
    ("x + ", "error", "x +", "cannot read"),
    ("(x", "error", "(x", "without its ')'"),
    ("x)", "error", "x)", "without its '('"),
    ("y = 1/0", "error", "y = 1/0", "not finite"),
    ("é", "error", "é", "cannot read"),
]


@pytest.mark.parametrize(
    ("typed", "kind", "text", "error"), TABLE, ids=[t[0] or "<empty>" for t in TABLE]
)
def test_table(typed, kind, text, error):
    row = normalize_row(typed)
    assert row.kind == kind, row.error
    assert row.text == text
    if error is None:
        assert row.ok and row.error is None
        assert row.latex
        assert row.guess == kind
    else:
        assert not row.ok and error in row.error, row.error
        assert row.parts == () and row.latex == ""


def test_parts_go_through_the_parser():
    """Every part of a clean row is in the parser's own grammar."""
    for typed, *_ in TABLE:
        row = normalize_row(typed)
        if row.ok:
            names = tuple(sorted({*row.variables, *row.coordinates} | ({row.name} - {None})))
            for part in row.parts:
                parse_expression(part, names)


def test_details():
    row = normalize_row("x' = σ(y − x)")
    assert (row.kind, row.name, row.parts, row.variables) == (
        "ode", "x", ("sigma*(y - x)",), ("sigma",)
    )  # fmt: skip
    assert row.latex.startswith("\\dot{x} = ") and "\\sigma" in row.latex
    assert row.coordinates == ("x", "y")
    v = normalize_row("β = 8/3")
    assert (v.name, v.parts, v.variables) == ("beta", ("8/3",), ())
    assert v.value == pytest.approx(8 / 3)
    assert normalize_row("k = 2π").value == pytest.approx(2 * 3.141592653589793)
    imp = normalize_row("x^2 + y^2 = r0^2")
    assert imp.parts == ("x**2 + y**2", "r0**2") and imp.variables == ("r0",)
    f = normalize_row("F(x, y, z) = (y, -x, 1)")
    assert (f.kind, f.name, f.parts) == ("vector_field", "F", ("y", "-x", "1"))
    assert normalize_row("F(x) = (y, -x)").kind == "error"
    assert normalize_row("(cos(t), sin(t))").latex == (
        "\\left(\\cos{\\left(t \\right)}, \\sin{\\left(t \\right)}\\right)"
    )


def test_one_name_per_letter_run():
    row = normalize_row("xy")
    assert row.kind == "graph" and row.variables == ("xy",)
    assert row.warnings and "x y" in row.warnings[0]
    assert normalize_row("x y").variables == () and not normalize_row("x y").warnings
    assert normalize_row("rho").warnings == ()  # not made of reserved names
    assert normalize_row("xi = 1").warnings == ()  # a Greek letter's name


def test_function_names_come_from_the_parser():
    """A name followed by ( is a call only when the parser knows the function."""
    assert normalize_row("y = a(x)").text == "y = a*(x)"
    assert normalize_row("y = exp(x)").text == "y = exp(x)"
    assert normalize_row("y = abs(x)").text == "y = abs(x)"
    assert normalize_row("y = conj (x)").text == "y = conj(x)"
    assert "brackets" in normalize_row("y = exp").error


def test_override_kind():
    s = normalize_row("x y", kind="surface")
    assert (s.kind, s.guess, s.parts) == ("surface", "scalar_field", ("x*y",))
    g = normalize_row("x y", kind="graph")
    assert g.kind == "error" and "cannot be a graph" in g.error and g.guess == "scalar_field"
    assert normalize_row("y = x^2", kind="implicit_curve").parts == ("y", "x**2")
    assert normalize_row("(1, 2)", kind="vector_field").kind == "vector_field"
    assert normalize_row("(1, 2)", kind="parametric_curve").kind == "parametric_curve"
    bad = normalize_row("x", kind="banana")
    assert bad.kind == "error" and "unknown row kind" in bad.error
    row = normalize_row("x y")
    assert classify_row(row, "surface") == s
    assert classify_row(row) == row
    assert classify_row("x y", "surface") == s
    for kind in KINDS[:-1]:
        assert isinstance(normalize_row("x y", kind=kind), Row)


def test_lorenz_is_one_3d_system():
    rows = [
        normalize_row(t)
        for t in (
            "x' = σ(y − x)",
            "y' = x(ρ − z) − y",
            "z' = x y − β z",
            "σ = 10",
            "ρ = 28",
            "β = 8/3",
            "y = x^2",  # another kind: ignored
        )
    ]
    assert all(r.ok for r in rows), [r.error for r in rows]
    system = group_ode_rows(rows)
    assert system.ok, system.errors
    assert system.dimension == 3
    assert system.coordinates == ("x", "y", "z")
    assert system.components == (
        ("x", "sigma*(y - x)"),
        ("y", "x*(rho - z) - y"),
        ("z", "x*y - beta*z"),
    )
    assert system.variables == ("beta", "rho", "sigma")
    assert system.autonomous and system.rows == (0, 1, 2)
    defined = {r.name: r.value for r in rows if r.kind == "variable"}
    assert set(defined) == set(system.variables)


def test_ode_groups():
    two = group_ode_rows([normalize_row("x' = y"), normalize_row("y' = -x + cos(t)")])
    assert two.ok and two.dimension == 2 and not two.autonomous and two.variables == ()
    xy = group_ode_rows([normalize_row("x' = xy"), normalize_row("y' = 1")])
    assert xy.ok and xy.variables == ("xy",)  # one name, as the warning said
    assert "no rows" in group_ode_rows([normalize_row("y = x")]).errors[0]
    assert "alone" in group_ode_rows([normalize_row("x' = 1")]).errors[0]
    dup = group_ode_rows(
        [normalize_row("x' = y"), normalize_row("x' = 1"), normalize_row("y' = 0")]
    )
    assert "two rows define x'" in dup.errors
    assert dup.dimension == 2 and dup.rows == (0, 1, 2)
    missing = group_ode_rows([normalize_row("x' = z"), normalize_row("y' = 1")])
    assert missing.errors == ("the system uses z but no row defines z'",)
    gap = group_ode_rows([normalize_row("x' = 1"), normalize_row("z' = 1")])
    assert "y'" in gap.errors[0]


# ---------------------------------------------------------------- properties


def _same(row: Row, again: Row) -> None:
    assert (again.text, again.kind, again.error, again.parts) == (
        row.text, row.kind, row.error, row.parts
    ), row.source  # fmt: skip


def test_normalise_is_idempotent_on_the_table():
    for typed, *_ in TABLE:
        row = normalize_row(typed)
        _same(row, normalize_row(row.text))


_ATOMS = ["x", "y", "z", "t", "2", "3.5", "1e-2", "pi", "e", "sigma", "rho", "a", "k1", "π",
          "θ", "x²", "(x+1)", "sin(x)", "exp(-y)", "σ", "−x", "u", "v"]  # fmt: skip
_JOIN = [" + ", " - ", "*", "/", "^", " ", "", "·", " × ", " − "]


def _random_expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 2 or rng.random() < 0.3:
        return rng.choice(_ATOMS)
    n = rng.randint(2, 4)
    out = rng.choice(_ATOMS)
    for _ in range(n - 1):
        out += rng.choice(_JOIN) + _random_expr(rng, depth + 1)
    return f"({out})" if rng.random() < 0.3 else out


def _random_row(rng: random.Random) -> str:
    shape = rng.randrange(6)
    e = lambda: _random_expr(rng)  # noqa: E731
    if shape == 0:
        return e()
    if shape == 1:
        return f"{rng.choice(['x', 'y', 'z', 'r', 'sigma'])} = {e()}"
    if shape == 2:
        mark = rng.choice(["'", "′", "\u0307"])
        return f"{rng.choice(['x', 'y', 'z'])}{mark} = {e()}"
    if shape == 3:
        return "(" + ", ".join(e() for _ in range(rng.randint(2, 3))) + ")"
    if shape == 4:
        return f"{e()} = {e()}"
    return "".join(rng.choice("xy+-*/^()=,' 2πσ") for _ in range(rng.randint(0, 12)))


def test_normalise_is_idempotent_on_random_rows():
    rng = random.Random(27)
    kinds = set()
    for _ in range(600):
        typed = _random_row(rng)
        row = normalize_row(typed)
        kinds.add(row.kind)
        _same(row, normalize_row(row.text))
        if row.ok:
            assert row.kind in KINDS and row.latex
    assert {"graph", "ode", "scalar_field", "error", "implicit_curve"} <= kinds


MEANING = ["sin(x)/x", "x**2 - 1", "1e-3*x", "2**-1", "-x**2", "x**y**2", "2*x - -3",
           "exp(-x**2 - y**2)", "(x + 1)*(x - 1)", "1/2*x", "x/2/y", "+x", "3.5e2*x - .5",
           "sqrt(abs(x))", "x**-1/2"]  # fmt: skip


@pytest.mark.parametrize("source", MEANING)
def test_normalising_keeps_the_meaning(source):
    """Text already in the parser's grammar means the same after normalising."""
    row = normalize_row(source)
    assert row.ok, row.error
    names = ("x", "y")
    assert sp.simplify(parse_expression(row.text, names) - parse_expression(source, names)) == 0


# ---------------------------------------------------------------- hostile input

HOSTILE = [
    "x" * 5000,
    "x+" * 1000 + "x",
    "x*" * 199 + "x",
    "x+" * 199 + "x",
    "(" * 1000 + "x" + ")" * 1000,
    "(" * 25 + "x" + ")" * 25,
    "sin(" * 29 + "x" + ")" * 29,
    "-" * 399 + "x",
    "x**" * 150 + "x",
    "9^9^9",
    "9**9**9",
    "x^9^9^9",
    "x^(-9^9^9)",
    "a = 9^(9^9)",
    "2^2^2^2^2",
    "a = 10^300 * 10^300",
    "a = 2^100000",
    "a = 1e400",
    "1e-400 x",
    "9" * 5000,
    "9" * 60,
    "a = exp(exp(exp(1000)))",
    "a = exp(1000)",
    "a = 0^-1",
    "y = x^1e9",
    "exp(exp(exp(x)))",
    "sin(10^300)",
    "\x00",
    "\ud800",
    "😀 = 1",
    "x‮ + 1",
    "x" + "̇" * 1000,
    "x" + "́" * 3,
    "ﷺ" * 300,
    "'" * 500,
    "=" * 500,
    "," * 500,
    "(" + "1," * 500 + "1)",
    "__import__('os').system('true')",
    "x.__class__",
    "lambda: 0",
    "[x for x in ()]",
    "True",
    "None = 1",
    "\n\t x \r\n",
    "y = x;",
    "y = x # comment",
    "y = 'x'",
]


@pytest.mark.parametrize("typed", HOSTILE, ids=[repr(h[:20]) for h in HOSTILE])
def test_hostile_input_is_an_error_value_and_quick(typed):
    start = time.perf_counter()
    row = normalize_row(typed)
    again = normalize_row(row.text)
    elapsed = time.perf_counter() - start
    assert isinstance(row, Row) and isinstance(again, Row)
    assert row.kind in KINDS
    assert row.ok == (row.error is None)
    assert len(row.text) <= MAX_CHARS
    assert elapsed < 2.0  # each read is meant to take under 0.1 s; generous for slow clocks


def test_hostile_rows_that_must_be_errors():
    for typed in ("9^9^9", "x^9^9^9", "a = 2^100000", "a = 1e400", "9" * 5000, "x" * 5000,
                  "(" * 1000 + "x" + ")" * 1000, "\x00", "😀 = 1", "x.__class__",
                  "__import__('os').system('true')", "lambda: 0", "True", "y = 'x'",
                  "a = exp(exp(exp(1000)))", "a = exp(1000)"):  # fmt: skip
        row = normalize_row(typed)
        assert row.kind == "error" and row.error, typed


def test_typical_rows_are_quick():
    """The budget in EXPLORER.md is 0.1 s a row while typing; assert it loosely over many."""
    rows = [t for t, *_ in TABLE] * 3
    start = time.perf_counter()
    for typed in rows:
        normalize_row(typed)
    assert (time.perf_counter() - start) / len(rows) < 0.1


def test_not_text():
    row = normalize_row(None)  # type: ignore[arg-type]
    assert row.kind == "error" and "text" in row.error
