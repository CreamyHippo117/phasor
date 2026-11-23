"""Tick values and the numbers written at them (Milestone 19): pure and deterministic.

Shared by the axes, number lines, number planes and bar charts that draw ticks, and by lint
(``crowded_ticks``), which sits below the domain layer. A range is ``(min, max, step)``;
``step`` 0 asks for a "nice" step (1, 2 or 5 times a power of ten) that depends only on the
range, never on the camera or the text size, so the same document always gets the same
ticks.

Every range of a new diagram type is checked before anything is made (``check_range``):
its numbers are finite and within ``MAX_VALUE``, ``min < max``, the span is not lost in the
numbers' own rounding (``RESOLUTION``), and a range never asks for more than ``MAX_TICKS``
ticks. ``geo.Axes`` predates these limits, so its ranges are only checked loosely
(``check_loose``: what documents saved before them already were); whatever a range asks,
``step_of`` and ``values`` never make more than ``MAX_TICKS`` + 1 ticks (a wider nice step;
none when the span is lost in rounding) and ``clamp_note`` says why (lint ``clamped``).
Huge integers are refused as a ``ValueError``, never an ``OverflowError``.
"""

from __future__ import annotations

import math

MAX_TICKS = 100  # steps per axis: past it a new type's range is refused, an Axes' widened
MAX_VALUE = 1e9  # no end of a range may be larger in size
RESOLUTION = 1e-9  # a span must be at least this share of its largest end (else rounding)
TARGET = 8  # about how many steps a nice step makes of a range
NICE = (1.0, 2.0, 5.0, 10.0)
MAX_DECIMALS = 10  # numbers written at ticks have at most this many decimals


def number(v, name: str = "value") -> float:
    """``v`` as a finite float, or ``ValueError`` (NaN, infinities, ints too large for a
    float, booleans, text)."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"{name} must be a number, got {v!r}")
    try:
        f = float(v)
    except OverflowError:
        raise ValueError(f"{name} is too large: {v!r}") from None
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return f


def nice_step(span: float, target: int = TARGET) -> float:
    """The smallest of 1, 2, 5 (times a power of ten) that cuts ``span`` into at most
    ``target`` steps."""
    if not (math.isfinite(span) and span > 0):
        raise ValueError(f"a span must be positive and finite, got {span!r}")
    raw = span / max(int(target), 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for f in NICE:
        if f * mag >= raw * (1 - 1e-9):
            return float(f"{f * mag:.12g}")
    return float(f"{10 * mag:.12g}")  # unreachable: 10 * mag >= raw


def nice_above(step: float) -> float:
    """The smallest nice step (1, 2 or 5 times a power of ten) at least ``step``."""
    return nice_step(step, 1)


def check_range(lo, hi, step, name: str = "range", max_ticks: int = MAX_TICKS) -> None:
    """Raise ``ValueError`` naming the problem unless ``(lo, hi, step)`` is a range ticks
    can be made from (see the module's docstring)."""
    lo, hi = number(lo, f"{name} min"), number(hi, f"{name} max")
    step = number(step, f"{name} step")
    if max(abs(lo), abs(hi)) > MAX_VALUE:
        raise ValueError(f"{name}: numbers must be within +-{MAX_VALUE:g}")
    if not hi > lo:
        raise ValueError(f"{name} must be (min, max, step) with min < max")
    if hi - lo < RESOLUTION * max(1.0, abs(lo), abs(hi)):
        raise ValueError(f"{name}: the span {hi - lo:g} is too small for numbers this size")
    if step < 0:
        raise ValueError(f"{name}: the step must be positive (or 0 for a nice one)")
    if step > 0 and (hi - lo) / step > max_ticks:
        raise ValueError(f"{name}: a step of {step:g} makes more than {max_ticks} ticks; "
                         f"use a larger step (or 0 for a nice one)")  # fmt: skip


def check_loose(lo, hi, step, name: str = "range") -> None:
    """What a range of a type that existed before these limits (``geo.Axes``) must be:
    finite numbers, ``min < max``, a step of 0 or more. Past the other limits of
    ``check_range`` it still loads (documents saved before them): ``step_of`` and ``values``
    draw it bounded and ``clamp_note`` says how (lint ``clamped``)."""
    lo, hi = number(lo, f"{name} min"), number(hi, f"{name} max")
    if not hi > lo:
        raise ValueError(f"{name} must be (min, max, step) with min < max")
    if number(step, f"{name} step") < 0:
        raise ValueError(f"{name}: the step must be positive (or 0 for a nice one)")


def _lost(lo: float, hi: float) -> bool:
    """Whether the span of ``lo``..``hi`` is lost in the numbers' own rounding (or overflows)."""
    span = hi - lo
    return not math.isfinite(span) or span < RESOLUTION * max(1.0, abs(lo), abs(hi))


def step_of(lo: float, hi: float, step: float) -> float:
    """The step a range is drawn at: its own, or a nice one when it is 0; widened to the
    smallest nice step that makes at most ``MAX_TICKS`` ticks when its own would make more
    (a range that loaded before the limit; ``clamp_note`` says so)."""
    span = hi - lo
    if not (math.isfinite(span) and span > 0):
        return float(step) if step > 0 else 1.0  # no ticks are drawn (``values``)
    if step <= 0:
        return nice_step(span)
    if span / step > MAX_TICKS * (1 + 1e-9):
        return nice_above(span / MAX_TICKS)
    return float(step)


def clamp_note(lo: float, hi: float, step: float, name: str = "range") -> str | None:
    """Why the ticks of a range are not drawn as asked (lint ``clamped``), or None."""
    if _lost(lo, hi):
        return (f"{name}: the span {hi - lo:g} is lost in the rounding of numbers this size "
                "(ticks.RESOLUTION); no ticks are drawn")  # fmt: skip
    drawn = step_of(lo, hi, step)
    if step > 0 and drawn != step:
        return (f"{name}: a step of {step:g} makes {(hi - lo) / step:.0f} ticks, past "
                f"the limit of {MAX_TICKS} (ticks.MAX_TICKS); drawn at a step of "
                f"{drawn:g}")  # fmt: skip
    return None


def decimals(step: float) -> int:
    """How many decimals the numbers at multiples of ``step`` need (at most
    ``MAX_DECIMALS``, enough for any step ``check_range`` allows): 0 for 1 or 50, 1 for
    0.5, 2 for 0.25."""
    for d in range(MAX_DECIMALS + 1):
        scaled = step * 10**d
        if round(scaled) and abs(scaled - round(scaled)) <= 1e-6 * scaled:
            return d
    return MAX_DECIMALS


def values(lo: float, hi: float, step: float) -> list[float]:
    """The multiples of ``step`` in [lo, hi] (zero included), in order. ``step`` 0: a nice
    one; a step that would make more than ``MAX_TICKS`` is widened (``step_of``); a span lost
    in rounding has none."""
    if _lost(lo, hi):
        return []
    step = step_of(lo, hi, step)
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    if k1 - k0 + 1 > MAX_TICKS + 1:
        raise ValueError(f"more than {MAX_TICKS} ticks in ({lo:g}, {hi:g}) at step {step:g}")
    # 15 significant digits: 0.3, not 0.30000000000000004; + 0.0: no "-0.0"
    return [float(f"{k * step:.15g}") + 0.0 for k in range(k0, k1 + 1)]


def nice_range(lo: float, hi: float) -> tuple[float, float, float]:
    """A range with a nice step around the data ``lo``..``hi`` (min == max is widened by
    one each way, or by a tenth of its size): (min, max, step), min and max on ticks."""
    lo, hi = number(lo), number(hi)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < RESOLUTION * max(1.0, abs(lo), abs(hi)):
        pad = max(1.0, abs(lo) * 0.1)
        lo, hi = lo - pad, hi + pad
    step = nice_step(hi - lo)
    return (math.floor(lo / step + 1e-9) * step + 0.0, math.ceil(hi / step - 1e-9) * step + 0.0,
            step)  # fmt: skip


def text(v: float, digits: int, *, group: bool = False, sign: bool = False) -> str:
    """``v`` written with ``digits`` decimals (no ``-0``): ``group`` puts a comma between
    thousands, ``sign`` writes ``+`` before a positive number. Not finite: ``nan``, ``inf``;
    a number of 1e15 or more in size is written in scientific notation (``1.5e+20``) so the
    text stays short."""
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("-inf" if v < 0 else "inf")
    if abs(v) >= 1e15:
        out = f"{v:.{min(digits, MAX_DECIMALS)}e}"
    else:
        out = f"{v:,.{digits}f}" if group else f"{v:.{digits}f}"
        if float(out.replace(",", "")) == 0:
            out = out.lstrip("-")
    if sign and not out.startswith("-") and v > 0:
        out = "+" + out
    return out


def as_math(s: str) -> str:
    """A number's text as math (a true minus sign; a comma as a thousands separator)."""
    return "$" + s.replace(",", "{,}") + "$"


def name(prefix: str, s: str) -> str:
    """A part name for the number written ``s``: ``x_2``, ``x_m2`` (minus), ``x_2p5``
    (point); commas left out."""
    body = s.replace(",", "").replace("-", "m").replace("+", "").replace(".", "p")
    return f"{prefix}_{body}"


def clear_step(step: float, spacing: float, need: float, height: float, gap: float) -> float:
    """The smallest nice step from which numbers no longer crowd: at ``step`` neighbours are
    ``spacing`` apart and need ``need`` of it (widths), plus ``gap`` text ``height``s."""
    if spacing <= 0:
        return nice_above(step)
    return nice_above(step * (need + gap * height) / spacing)
