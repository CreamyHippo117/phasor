"""The hardened whitelist parser (Milestone 26; EXPLORER.md, advice section 5 items 1-6):
hostile text is refused quickly with a ``FunctionError``, never a hang or another exception,
and every formula the repo's examples and docs use still reads exactly as before."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import sympy as sp

from phasor.domains import geometry_math as gm
from phasor.render.gpu import codegen
from phasor.render.gpu.codegen import ComplexFunction, FunctionError, parse_expression
from tests.formula_corpus import FIXTURE, candidates

TOWERS = ["9**9**9", "2**-1000000", "(1/2)**1000000", "2**2**2**2**2", "10**10**10",
          "(1+i)**(10**9)", "9.0**9**9", "pi**pi**pi**pi", "x + 7**7**7", "-(3**3**3**3)"]  # fmt: skip


@pytest.mark.parametrize("source", TOWERS)
def test_towers_of_powers_are_refused_at_once(source):
    start = time.perf_counter()  # perf_counter: fine-grained on Windows too
    with pytest.raises(FunctionError, match="too large"):
        parse_expression(source, ("x",))
    assert time.perf_counter() - start < 0.1
    with pytest.raises(FunctionError):
        ComplexFunction.parse(source.replace("x", "z"))


@pytest.mark.parametrize(
    "source",
    [
        "+".join(["x"] * 20000),  # long: RecursionError before
        "x" * 3000,
        "-" * 1500 + "x",  # deep
        "sin(" * 120 + "x" + ")" * 120,
        "x**" * 300 + "x",
        "(" * 400 + "x" + ")" * 400,  # Python's own parser gives up
        "1" * 5000,  # past the int digits limit
        "x +",
        "",
    ],
)
def test_long_deep_and_broken_text_is_a_function_error(source):
    with pytest.raises(FunctionError):
        parse_expression(source, ("x",))
    with pytest.raises(FunctionError):
        ComplexFunction.parse(source.replace("x", "z"))


def test_a_long_legal_sum_reads_and_keeps_its_order():
    long = " + ".join(f"{k / 10:g}*x**{k % 4}*y" for k in range(1, 150))
    assert 1500 < len(long) <= codegen.MAX_SOURCE
    e = parse_expression(long, ("x", "y"))
    assert e.free_symbols == {sp.Symbol("x", real=True), sp.Symbol("y", real=True)}
    terms = "+".join(["x"] * 999)
    assert parse_expression(terms, ("x",)) == 999 * sp.Symbol("x", real=True)
    assert parse_expression("x - y - x", ("x", "y")) == -sp.Symbol("y", real=True)
    assert parse_expression("8 / 2 / 2", ()) == 2  # left to right, as Python reads it


def test_hostile_but_legal_formulas_parse():
    for s in ("x**1e9", "exp(exp(exp(x)))", "1/(x-y)", "log(x)", "1e6*x"):
        parse_expression(s, ("x", "y"))


def test_real_only_extras_and_the_complex_whitelist():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = np.ones(5)
    assert np.array_equal(gm.compile_fn("sign(x)", ("x",))(x), np.sign(x))
    assert np.array_equal(gm.compile_fn("floor(x)", ("x",))(x), np.floor(x))
    assert np.array_equal(gm.compile_fn("max(x, 0)", ("x",))(x), np.maximum(x, 0))
    assert np.array_equal(gm.compile_fn("min(x, y, 0)", ("x", "y"))(x, y), np.minimum(x, 0))
    assert np.allclose(gm.compile_fn("atan2(y, x)", ("x", "y"))(x, y), np.arctan2(y, x))
    with pytest.raises(FunctionError, match="arguments"):
        parse_expression("min(x)", ("x",))
    with pytest.raises(FunctionError, match="arguments"):
        parse_expression("atan2(x)", ("x",))
    for s in ("sign(z)", "floor(z)", "max(z, 1)", "atan2(z, 1)"):  # no kernel code for these
        with pytest.raises(FunctionError):
            ComplexFunction.parse(s)


def test_python_is_still_never_run():
    for s in ("__import__('os')", "x.real", "[x]", "x if x else 1", "lambda: 1", "f(x)(x)"):
        with pytest.raises(FunctionError):
            parse_expression(s, ("x",))


def test_simplify_is_bounded_for_long_fields():
    long = " + ".join(f"{k}*x**{k % 5}*y*z" for k in range(1, 120))
    start = time.perf_counter()
    d = gm.divergence(long, "y", "z")
    assert d.free_symbols  # not simplified (too large), still the right derivative
    assert time.perf_counter() - start < 5.0  # generous: a guard against minutes, not a timer
    assert gm.divergence("x", "y", "z") == 3  # small ones are still simplified


def test_every_formula_in_the_repo_reads_as_it_did_before_the_hardening():
    """Advice section 5 item 21: the fixture was written by the parser before Milestone 26."""
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(rows) > 100
    changed = []
    for r in rows:
        if not r["real"].startswith("error"):
            got = sp.srepr(parse_expression(r["s"], tuple(r["names"])))
            if got != r["real"]:
                changed.append(r["s"])
        if "complex" in r and not r["complex"].startswith("error"):
            got = sp.srepr(ComplexFunction.parse(r["s"], constant=True).expr)
            if got != r["complex"]:
                changed.append(r["s"])
    assert not changed, changed[:5]
    # The fixture covers what is in the trees today: a new formula is read by the new parser.
    known = {r["s"] for r in rows}
    for s, names in candidates():
        if s not in known:
            try:
                parse_expression(s, names)
            except FunctionError:
                pass  # a string that is not a formula; it must still fail cleanly
