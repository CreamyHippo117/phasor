"""Give the document's constraints meaning: who moves, which stage solves what.

``compile_plan`` checks every rule against the compiled document and raises
``DocumentError`` for rules that cannot work (unknown ids, objects that cannot move).
Rules that are valid but cannot be satisfied at some time are found later, by the bake,
and reported as ``unsolved`` lint issues.
"""

from __future__ import annotations

from dataclasses import dataclass

from phasor.core.constraints import Guard, Relation, parse_rule
from phasor.core.issues import DocumentError, Issue
from phasor.core.schema import ConstraintSpec
from phasor.time.compile import Compiled, CompiledExpr

MOVE_PARAM = "pos"  # layout moves an object by writing this parameter
LEADER_PARAMS = ("leader", "leader_on")


@dataclass(frozen=True)
class Rule:
    index: int
    spec: ConstraintSpec
    clauses: tuple[Relation | Guard, ...]

    @property
    def path(self) -> str:
        return f"constraints[{self.spec.rule}]"


@dataclass(frozen=True)
class Plan:
    rules: tuple[Rule, ...] = ()
    solved: frozenset[str] = frozenset()  # ids moved by stage 1 (the linear solver)
    placed: tuple[str, ...] = ()  # ids moved by stage 2 (label placer), in rule order
    leaders: frozenset[str] = frozenset()  # placed ids that can draw a leader line

    @property
    def moving(self) -> frozenset[str]:
        return self.solved | frozenset(self.placed)

    @property
    def empty(self) -> bool:
        return not self.moving

    def relations(self, kinds=None):
        for r in self.rules:
            for c in r.clauses:
                if isinstance(c, Relation) and (kinds is None or c.kind in kinds):
                    yield r, c

    def guards(self, kind: str | None = None):
        for r in self.rules:
            for c in r.clauses:
                if isinstance(c, Guard) and (kind is None or c.kind == kind):
                    yield r, c

    def has_guard(self, kind: str, object_id: str) -> bool:
        return any(object_id in c.ids for _, c in self.guards(kind))


def compile_plan(compiled: Compiled) -> Plan:
    issues: list[Issue] = []
    objects = {o.id: o for o in compiled.objects}
    rules: list[Rule] = []
    solved: set[str] = set()
    placed: list[str] = []

    def movable(rule: Rule, object_id: str) -> bool:
        obj = objects[object_id]
        path = f"{object_id}.{MOVE_PARAM}"
        if MOVE_PARAM not in obj.params_model.model_fields:
            issues.append(Issue(
                "not_movable", f"{object_id!r} ({obj.type}) has no '{MOVE_PARAM}' parameter, so "
                "layout cannot move it", ids=(object_id,), path=rule.path,
                fix_hint="put it on the right-hand side of the rule, or constrain something else",
            ))  # fmt: skip
            return False
        if isinstance(compiled.nodes.get(path), CompiledExpr):
            issues.append(Issue(
                "not_movable", f"{path} is an expression, so layout cannot move {object_id!r}",
                ids=(object_id,), path=rule.path,
                fix_hint=f"set {path} to a plain value and let the constraint place it",
            ))  # fmt: skip
            return False
        return True

    for index, spec in enumerate(compiled.doc.constraints):
        rule = Rule(index, spec, parse_rule(spec.rule))
        rules.append(rule)
        for clause in rule.clauses:
            missing = [i for i in clause.ids if i not in objects]
            if missing:
                issues.append(Issue(
                    "unknown_id", f"constraint {spec.rule!r} names unknown object(s) {missing}",
                    ids=tuple(missing), path=rule.path,
                    fix_hint="objects are " + ", ".join(sorted(objects)),
                ))  # fmt: skip
                continue
            if isinstance(clause, Relation):
                if not movable(rule, clause.subject):
                    continue
                (placed.append if clause.kind == "near" else solved.add)(clause.subject)
            elif clause.kind == "on_screen":
                obj = objects[clause.ids[0]]
                path = f"{obj.id}.{MOVE_PARAM}"
                if MOVE_PARAM in obj.params_model.model_fields and not isinstance(
                    compiled.nodes.get(path), CompiledExpr
                ):
                    solved.add(obj.id)  # otherwise it is only checked, by the linter

    for object_id in sorted(set(placed)):
        if placed.count(object_id) > 1 or object_id in solved:
            issues.append(Issue(
                "conflicting_constraints", f"{object_id!r} has a 'near' rule and another rule "
                "that also places it", ids=(object_id,),
                fix_hint="keep one: 'near' lets the placer choose; above/below/... fix the side",
            ))  # fmt: skip
    if issues:
        raise DocumentError(issues)
    leaders = frozenset(
        i for i in placed
        if all(p in objects[i].params_model.model_fields for p in LEADER_PARAMS)
    )  # fmt: skip
    return Plan(tuple(rules), frozenset(solved), tuple(placed), leaders)
