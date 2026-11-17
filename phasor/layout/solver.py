"""Stage 1 of layout: linear relations with kiwisolver (SPEC contract 7).

Every movable object keeps its size and gets a center (x, y) in screen heights. Relations
become linear equalities or inequalities with the rule's priority. A weak "stay where the
document put you" keeps everything else still. Whatever does not hold after solving is
returned, never dropped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import kiwisolver as kiwi

from phasor.core.constraints import Clause, Guard
from phasor.layout.geometry import Box

STRENGTH = {
    "required": kiwi.strength.required,
    "strong": kiwi.strength.strong,
    "medium": kiwi.strength.medium,
    "weak": kiwi.strength.weak,
}
# The side rule of a relation ("above" also centers horizontally) is one notch weaker,
# so an explicit align can override it.
_SECONDARY = {
    "required": kiwi.strength.strong,
    "strong": kiwi.strength.medium,
    "medium": kiwi.strength.weak,
    "weak": kiwi.strength.create(0, 0, 0.03),
}
_STAY = kiwi.strength.create(0, 0, 0.001)
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Item:
    index: int  # position of the rule in document.constraints
    priority: str
    clause: Clause
    value: float | None  # gap or margin in screen heights (already defaulted)


@dataclass(frozen=True)
class Unsolved:
    index: int
    clause: Clause
    message: str


def solve(
    items: Sequence[Item], boxes: Mapping[str, Box], movable: frozenset[str], frame: Box
) -> tuple[dict[str, tuple[float, float]], list[Unsolved]]:
    """Returns (new center per movable id, constraints that do not hold)."""
    solver = kiwi.Solver()
    vx = {i: kiwi.Variable(f"{i}.x") for i in sorted(movable)}
    vy = {i: kiwi.Variable(f"{i}.y") for i in sorted(movable)}
    for i in sorted(movable):
        cx, cy = boxes[i].center
        solver.addConstraint((vx[i] == cx) | _STAY)
        solver.addConstraint((vy[i] == cy) | _STAY)

    def x(i):
        return vx[i] if i in vx else boxes[i].center[0]

    def y(i):
        return vy[i] if i in vy else boxes[i].center[1]

    def equations(item: Item):
        """(primary constraints, secondary constraints) as kiwi expressions."""
        c = item.clause
        g = item.value or 0.0
        if isinstance(c, Guard):  # on_screen
            b = boxes[c.ids[0]]
            i = c.ids[0]
            return [
                x(i) - b.w / 2 >= frame.x0 + g,
                x(i) + b.w / 2 <= frame.x1 - g,
                y(i) - b.h / 2 >= frame.y0 + g,
                y(i) + b.h / 2 <= frame.y1 - g,
            ], []
        s, t = c.subject, c.target
        bs, bt = boxes[s], boxes[t]
        match c.kind:
            case "above":
                return [y(s) - bs.h / 2 == y(t) + bt.h / 2 + g], [x(s) == x(t)]
            case "below":
                return [y(s) + bs.h / 2 == y(t) - bt.h / 2 - g], [x(s) == x(t)]
            case "left_of":
                return [x(s) + bs.w / 2 == x(t) - bt.w / 2 - g], [y(s) == y(t)]
            case "right_of":
                return [x(s) - bs.w / 2 == x(t) + bt.w / 2 + g], [y(s) == y(t)]
            case "align_x":
                return [x(s) == x(t)], []
            case "align_y":
                return [y(s) == y(t)], []
            case "inside":
                return [
                    x(s) - bs.w / 2 >= x(t) - bt.w / 2,
                    x(s) + bs.w / 2 <= x(t) + bt.w / 2,
                    y(s) - bs.h / 2 >= y(t) - bt.h / 2,
                    y(s) + bs.h / 2 <= y(t) + bt.h / 2,
                ], []
        raise ValueError(f"stage 1 cannot solve {c.kind!r}")  # pragma: no cover

    unsolved: list[Unsolved] = []
    added: list[tuple[Item, list]] = []
    for item in items:
        primary, secondary = equations(item)
        done = []
        try:
            for eq in primary:
                constraint = eq | STRENGTH[item.priority]
                solver.addConstraint(constraint)
                done.append(constraint)
        except kiwi.UnsatisfiableConstraint:
            for constraint in done:  # all or nothing: no half-applied rule
                solver.removeConstraint(constraint)
            message = "is required but conflicts with another required rule"
            unsolved.append(Unsolved(item.index, item.clause, message))
            continue
        for eq in secondary:
            solver.addConstraint(eq | _SECONDARY[item.priority])
        added.append((item, primary))
    solver.updateVariables()

    for item, primary in added:
        worst = max(_violation(eq) for eq in primary)
        if worst > TOLERANCE:
            unsolved.append(
                Unsolved(
                    item.index,
                    item.clause,
                    f"cannot hold; it is off by {worst:.3g} screen heights "
                    "because a rule of equal or higher priority wins",
                )
            )
    return {i: (vx[i].value(), vy[i].value()) for i in sorted(movable)}, unsolved


def _violation(constraint: kiwi.Constraint) -> float:
    value = constraint.expression().value()
    match constraint.op():
        case "==":
            return abs(value)
        case ">=":
            return max(0.0, -value)
        case _:
            return max(0.0, value)
