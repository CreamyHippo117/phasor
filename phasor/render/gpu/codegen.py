"""One expression, three outputs (SPEC "GPU and CUDA plan"): CUDA C for the kernel, a NumPy
function for the reference, and LaTeX for the label.

    f = ComplexFunction.parse("1/(z**2 + 1)")
    f.c()       -> "c_powi(c_add(c_powi(z, 2), c_make(R(1.0), R(0.0))), -1)"
    f.numpy()   -> callable on complex arrays
    f.latex()   -> "\\frac{1}{z^{2} + 1}"

The text is read with a whitelist (names, numbers, + - * / **, known functions) and built
into a SymPy expression node by node; it is never evaluated as Python.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

Z = sp.Symbol("z")
A = sp.Symbol("a")  # a complex constant the kernel takes as an argument, so it can animate
_NAMES = {"z": Z, "i": sp.I, "I": sp.I, "j": sp.I, "pi": sp.pi, "e": sp.E}
_FUNCS = {
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos,
    "tan": sp.tan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "conj": sp.conjugate, "abs": sp.Abs,
}  # fmt: skip
_C_FUNCS = {
    sp.exp: "c_exp", sp.log: "c_log", sp.sin: "c_sin", sp.cos: "c_cos", sp.tan: "c_tan",
    sp.sinh: "c_sinh", sp.cosh: "c_cosh", sp.tanh: "c_tanh", sp.conjugate: "c_conj",
}  # fmt: skip
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}


class FunctionError(ValueError):
    pass


# Limits on what the parser reads (EXPLORER.md, advice section 5 items 1-2): a box that
# re-parses while someone types must never hang or crash on what they type.
MAX_SOURCE = 2000  # characters
MAX_NODES = 2000  # expression nodes
MAX_DEPTH = 100  # nesting (a run of + and - or of * and / counts once)
MAX_POWER = 700.0  # a power of two numbers whose size would pass e**700 is refused
# Real-only functions for ``parse_expression`` (formulas of x, y, ...): not for
# ``ComplexFunction``, whose kernels have no code for them. name -> (SymPy, argument counts).
_REAL_FUNCS = {
    "sign": (sp.sign, (1,)), "floor": (sp.floor, (1,)), "atan2": (sp.atan2, (2,)),
    "min": (sp.Min, (2, 3, 4)), "max": (sp.Max, (2, 3, 4)),
}  # fmt: skip
_FAMILY = {ast.Add: 0, ast.Sub: 0, ast.Mult: 1, ast.Div: 1}


def _short(source: str) -> str:
    return repr(source if len(source) <= 80 else source[:60] + "...")


def _tree(source: str) -> ast.Expression:
    """The syntax tree of ``source``, refused (``FunctionError``) when it is too long, too
    big or too deep to read safely."""
    if not isinstance(source, str):
        raise FunctionError(f"a formula is text, not {type(source).__name__}")
    text = source.strip()
    if len(text) > MAX_SOURCE:
        raise FunctionError(f"{_short(source)} is {len(text)} characters long; a formula may "
                            f"have at most {MAX_SOURCE}")  # fmt: skip
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise FunctionError(f"cannot read {_short(source)}: {e.msg}") from e
    except (RecursionError, MemoryError, ValueError) as e:
        raise FunctionError(f"cannot read {_short(source)}: nested too deeply") from e
    count, stack = 0, [(tree.body, 1)]
    while stack:  # iterative: the check itself must not recurse
        node, depth = stack.pop()
        count += 1
        if count > MAX_NODES:
            raise FunctionError(f"{_short(source)} is too big: at most {MAX_NODES} parts")
        if depth > MAX_DEPTH:
            raise FunctionError(f"{_short(source)} is nested too deeply (at most {MAX_DEPTH})")
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.expr):
                continue
            same = (isinstance(node, ast.BinOp) and child is node.left
                    and isinstance(child, ast.BinOp) and type(node.op) in _FAMILY
                    and _FAMILY.get(type(child.op)) == _FAMILY[type(node.op)])  # fmt: skip
            stack.append((child, depth if same else depth + 1))
    return tree


def _power(base: sp.Expr, exponent: sp.Expr, source: str) -> sp.Expr:
    """``base ** exponent``, refused when both are numbers and the answer would be huge or
    tiny past any float: SymPy computes such a power exactly (``9**9**9`` never returns)."""
    if base.is_number and exponent.is_number:
        try:
            b = abs(complex(base.evalf(15)))
            x = abs(complex(exponent.evalf(15)))
        except (TypeError, ValueError, OverflowError):
            b, x = math.inf, math.inf
        if b != 1.0 and b != 0.0 and x * abs(math.log(b)) > MAX_POWER:
            raise FunctionError(f"{_short(source)} has a power of numbers too large to compute")
    return base**exponent


def _build(node: ast.AST, source: str, names: dict | None = None,
           funcs: dict | None = None) -> sp.Expr:  # fmt: skip
    match node:
        case ast.Expression(body=body):
            return _build(body, source, names, funcs)
        case ast.Constant(value=int() | float() as v) if not isinstance(v, bool):
            return sp.Integer(v) if isinstance(v, int) else sp.Float(v)
        case ast.Constant(value=complex() as v):
            return sp.Float(v.imag) * sp.I
        case ast.Name(id=name):
            table = _NAMES if names is None else names
            if name not in table:
                known = sorted(_FUNCS) + sorted(funcs or {})
                raise FunctionError(
                    f"unknown name {name!r} in {_short(source)}; use "
                    f"{', '.join(sorted(table))} and {known}"
                )
            return table[name]
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_build(operand, source, names, funcs)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _build(operand, source, names, funcs)
        case ast.BinOp(op=op) if type(op) in _FAMILY:
            # a run of + and - (or * and /) is folded left to right without recursing, so a
            # long sum neither overflows the stack nor changes the order SymPy sees
            chain = []
            while (isinstance(node, ast.BinOp) and type(node.op) in _FAMILY
                   and _FAMILY[type(node.op)] == _FAMILY[type(op)]):  # fmt: skip
                chain.append((node.op, node.right))
                node = node.left
            out = _build(node, source, names, funcs)
            for o, right in reversed(chain):
                out = _BINOPS[type(o)](out, _build(right, source, names, funcs))
            return out
        case ast.BinOp(left=left, op=ast.Pow(), right=right):
            return _power(_build(left, source, names, funcs),
                          _build(right, source, names, funcs), source)  # fmt: skip
        case ast.Call(func=ast.Name(id=name), args=[arg], keywords=[]) if name in _FUNCS:
            return _FUNCS[name](_build(arg, source, names, funcs))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in (funcs or {}):
            fn, counts = funcs[name]
            if len(args) not in counts:
                raise FunctionError(f"{name} takes {' or '.join(map(str, counts))} arguments "
                                    f"in {_short(source)}")  # fmt: skip
            return fn(*[_build(a, source, names, funcs) for a in args])
    what = "a function of z" if funcs is None else "a formula"
    raise FunctionError(f"cannot use {ast.unparse(node)!r} in {what} ({_short(source)})")


def _read(source: str, names: dict | None, funcs: dict | None) -> sp.Expr:
    """The one way in: size limits, then the whitelist build; whatever goes wrong inside
    SymPy or Python is a ``FunctionError`` naming the text, never another exception."""
    tree = _tree(source)
    try:
        expr = sp.sympify(_build(tree, source, names, funcs))
    except FunctionError:
        raise
    except (RecursionError, MemoryError, OverflowError, ValueError, TypeError,
            ZeroDivisionError, AttributeError) as e:  # fmt: skip
        raise FunctionError(f"cannot read {_short(source)}: {type(e).__name__}") from e
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise FunctionError(f"{_short(source)} is not finite anywhere")
    return expr


def _real(value: sp.Expr) -> str:
    return f"R({float(value)!r})"


def _emit(e: sp.Expr) -> str:
    if e == Z:
        return "z"
    if e == A:
        return "a"
    if e.is_number and e.is_real and e.is_finite:
        return f"c_make({_real(e)}, R(0.0))"
    if e == sp.I:
        return "c_make(R(0.0), R(1.0))"
    if isinstance(e, sp.Add | sp.Mul):
        fn = "c_add" if isinstance(e, sp.Add) else "c_mul"
        terms = [_emit(a) for a in e.args]
        out = terms[0]
        for term in terms[1:]:
            out = f"{fn}({out}, {term})"
        return out
    if isinstance(e, sp.Pow):
        base, exponent = e.args
        if exponent.is_Integer:
            return f"c_powi({_emit(base)}, {int(exponent)})"
        if exponent == sp.Rational(1, 2):
            return f"c_sqrt({_emit(base)})"
        if exponent == sp.Rational(-1, 2):
            return f"c_powi(c_sqrt({_emit(base)}), -1)"
        return f"c_pow({_emit(base)}, {_emit(exponent)})"
    if isinstance(e, sp.Abs):
        return f"c_make(c_abs({_emit(e.args[0])}), R(0.0))"
    if type(e) in _C_FUNCS:
        return f"{_C_FUNCS[type(e)]}({_emit(e.args[0])})"
    raise FunctionError(f"no kernel code for {e!r}")


def parse_expression(source: str, variables: tuple[str, ...]) -> sp.Expr:
    """A SymPy expression in the named ``variables`` (plus ``pi``, ``e``, ``i`` and the known
    functions), read with the same whitelist as ``ComplexFunction`` and never evaluated as
    Python. For domains whose functions are not of ``z`` (calculus: ``x``, ``x, y``).
    Real formulas also know ``sign``, ``floor``, ``atan2(y, x)``, ``min`` and ``max`` (2 to
    4 arguments); ``ComplexFunction`` does not. Text past ``MAX_SOURCE`` characters,
    ``MAX_NODES`` parts or ``MAX_DEPTH`` nesting, and powers of numbers too large to
    compute (``9**9**9``), are refused with a ``FunctionError``."""
    names = {**{k: v for k, v in _NAMES.items() if k not in ("z",)},
             **{v: sp.Symbol(v, real=True) for v in variables}}  # fmt: skip
    return _read(source, names, _REAL_FUNCS)


@dataclass(frozen=True)
class ComplexFunction:
    """A function of one complex variable ``z``."""

    source: str
    expr: sp.Expr

    @classmethod
    def parse(cls, source: str, *, constant: bool = False) -> ComplexFunction:
        """``constant=True`` also allows the name ``a``: a complex constant given at run
        time (``c()`` then reads a ``cplx a`` in scope, ``numpy()`` takes ``a=``)."""
        expr = _read(source, {**_NAMES, "a": A} if constant else None, None)
        return cls(source, expr)

    def c(self) -> str:
        """A C expression in ``z`` (type ``cplx``) using ``complex.cuh``."""
        return _emit(self.expr)

    def numpy(self):
        """``f(z)`` for complex NumPy arrays; poles give inf or nan, never a warning. A
        function of the constant ``a`` takes it as ``f(z, a=...)`` (default 0)."""
        fn = sp.lambdify((Z, A), self.expr, "numpy")

        def call(z: np.ndarray, a: complex = 0j) -> np.ndarray:
            with np.errstate(all="ignore"):
                w = fn(np.asarray(z, dtype=np.complex128), complex(a))
            return np.broadcast_to(np.asarray(w, dtype=np.complex128), np.shape(z))

        return call

    def latex(self) -> str:
        return sp.latex(self.expr)
