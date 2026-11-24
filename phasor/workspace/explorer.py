"""Explorer rows on the document (Milestone 27 part B, EXPLORER.md). Pure: no GUI, no MCP.

The Explorer's list of typed rows is a *view* of the document: every row is an ordinary
object (or one line of a ``flow.System``) carrying the text as typed (``source``), its place
(``row``), the kind the person chose over the guess (``kind``) and ``visible``. So undo,
save, lint, the AI tools and the renderer all see the same rows, and the list is rebuilt
from the document alone (``rows_of``).

What each row becomes (``docs/decisions/explorer_advice_2026-11-24.md`` section 7.3):

- ``name = number``: a ``flow.Variable`` and a document ``Control`` on ``<id>.value`` (the
  slider the player, the notebook and the web export already read);
- ``x' = ...`` rows together: one ``flow.System`` (one line of its ``source`` per row) with
  its field (``<id>.field``) and equilibria (``<id>.eq``);
- ``(P, Q)`` / ``F(x, y) = (P, Q)``: a ``flow.System`` with its field (``<id>.field``);
- ``(a, b)``: a ``flow.Probe`` seeded there, on the system of its dimension;
- anything else: a ``flow.Note`` keeping the text and, for a row that does not read, the
  error in plain words. Graphs, curves, surfaces, heat maps and implicit curves are read and
  classified but not drawn yet (Milestone 27 Deviation 5).

``plan`` turns a list of edits into **one** patch: a row whose kind changes is removed and
added in the same patch, so undo is one step. Names a row uses that no variable defines get a
``Variable`` (value 1, range -10..10) at commit time only: ``plan`` runs on a commit, never
while typing. Ids are named by meaning (``rho``, ``system``, ``point``), never by the text,
and never reuse a retired id (``unique_names`` against every id the document has had).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from phasor.core.ids import name_from, unique_names
from phasor.core.schema import Document, ObjectSpec
from phasor.domains import flow
from phasor.domains import flow_math as fm
from phasor.domains.rows import Row, group_ode_rows, normalize_row
from phasor.render.gpu.codegen import parse_expression

ROW_TYPES = ("variable", "flow_system", "flow_probe", "flow_note")
MAX_EDITS = 200  # edits in one call
NEW_VARIABLE = {"value": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}
WINDOW_2D = ((-4.0, 4.0), (-3.0, 3.0))  # flow.System's default
WINDOW_3D = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))
# Ids a new row's object is named by (its meaning; the next free suffix makes it unique).
BASE = {"variable": "var", "ode": "system", "vector_field": "field", "point": "point",
        "error": "row", "graph": "graph", "sideways_graph": "graph", "polar": "polar",
        "parametric_curve": "curve", "parametric_surface": "surface", "surface": "surface",
        "scalar_field": "heat", "implicit_curve": "curve"}  # fmt: skip
NOT_DRAWN = ("read and classified, not drawn by the Explorer yet (Milestone 27 Deviation 5); "
             "the text is kept")  # fmt: skip


class EditError(ValueError):
    """An edit the planner cannot make; the message is for the caller to act on."""


@functools.lru_cache(maxsize=4096)
def read(text: str, kind: str | None = None) -> Row:
    """``rows.normalize_row``, remembered (a row's text is read on every look at the list)."""
    return normalize_row(text, kind)


def _num(v: float) -> str:
    return format(float(v), ".12g")


def _p(spec: ObjectSpec, name: str, default: Any = None) -> Any:
    v = spec.params.get(name, default)
    return default if isinstance(v, dict) else v  # an expression: not a row's constant


def version_of(spec: ObjectSpec) -> str:
    """A row's version: a hash of its object as the document stores it (changes when the row
    changes, not when other rows do)."""
    blob = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def is_row(spec: ObjectSpec) -> bool:
    r = _p(spec, "row", -1)
    return (
        spec.type in ROW_TYPES and isinstance(r, int | float) and not isinstance(r, bool) and r >= 0
    )


def _system_lines(spec: ObjectSpec) -> list[str]:
    src = _p(spec, "source", "") or ""
    return [ln for ln in src.split("\n") if ln.strip()] or [""]


def object_kind(spec: ObjectSpec) -> str:
    """The row kind an object stands for."""
    if spec.type == "variable":
        return "variable"
    if spec.type == "flow_probe":
        return "point"
    if spec.type == "flow_system":
        lines = _system_lines(spec)
        one = len(lines) == 1 and read(lines[0]).kind == "vector_field"
        return "vector_field" if one else "ode"
    if _p(spec, "error", ""):  # a note that holds an error stays one until it reads
        return "error"
    over = _p(spec, "kind", "") or None
    return read(_p(spec, "source", "") or "", over).kind


# ---------------------------------------------------------------- the list, read


def rows_of(doc: Document, controls: bool = True) -> list[dict[str, Any]]:
    """The Explorer's rows, in order: one entry per row with its ``id`` (the key edits name:
    the object's id, or ``<system>:<coordinate>`` for an ODE line), ``object``, ``kind``,
    ``source`` (as typed), ``text`` (normalised), ``latex``, ``error``, ``visible``,
    ``row`` and ``version``."""
    ctl = {c.path: c for c in doc.controls} if controls else {}
    out: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for spec in doc.objects:
        if not is_row(spec):
            continue
        base = {"object": spec.id, "visible": bool(_p(spec, "visible", True)),
                "row": int(_p(spec, "row", -1)), "version": version_of(spec),
                "kind_override": _p(spec, "kind", "") or None}  # fmt: skip
        if spec.type == "flow_system":
            kind = object_kind(spec)
            for k, line in enumerate(_system_lines(spec)):
                r = read(line)
                key = spec.id if kind == "vector_field" else f"{spec.id}:{r.name or k}"
                out.append(((base["row"], k), {**base, "id": key, "kind": kind, "source": line,
                            "text": r.text, "latex": r.latex, "error": None,
                            "variables": list(r.variables), "drawn": True}))  # fmt: skip
            continue
        src = _p(spec, "source", "") or ""
        r = read(src, base["kind_override"])
        entry = {**base, "id": spec.id, "source": src, "text": r.text, "latex": r.latex,
                 "variables": list(r.variables)}  # fmt: skip
        if spec.type == "variable":
            name, value = _p(spec, "name", "a"), float(_p(spec, "value", 1.0))
            c = ctl.get(f"{spec.id}.value")
            entry.update(kind="variable", name=name, value=value, text=f"{name} = {_num(value)}",
                         min=float(_p(spec, "min", -10.0)), max=float(_p(spec, "max", 10.0)),
                         step=c.step if c else None, slider=c is not None, error=None,
                         drawn=False)  # fmt: skip
        elif spec.type == "flow_probe":
            entry.update(kind="point", error=None, drawn=True,
                         system=_p(spec, "system", "") or None)  # fmt: skip
        else:
            err = _p(spec, "error", "") or None
            kind = "error" if err else object_kind(spec)
            entry.update(kind=kind, error=err, drawn=False,
                         why=None if err else NOT_DRAWN)  # fmt: skip
        out.append(((base["row"], 0), entry))
    out.sort(key=lambda e: e[0])
    return [e for _, e in out]


# ---------------------------------------------------------------- planning a commit


@dataclass
class _Want:
    """One row as it will be: where it came from (``origin``: its object, ``line``: its
    coordinate in an ODE system) and what the edits made of it."""

    source: str
    origin: str | None = None
    kind: str | None = None  # an override
    visible: bool | None = None
    color: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)  # value/min/max/step/window/dt/...
    base: str | None = None  # the id a new object is named by
    key: str | None = None  # the row's id before the edit
    changed: bool = False
    row: Row | None = None


@dataclass
class _Unit:
    kind: str  # variable, ode, vector_field, point, note
    type: str
    rows: list[_Want]
    params: dict[str, Any]
    origin: str | None = None
    children: tuple[tuple[str, str, dict[str, Any]], ...] = ()  # (suffix, type, params)
    id: str | None = None
    name: str | None = None  # a variable's
    error: str | None = None


SETTINGS = {
    "variable": ("value", "min", "max", "step"),
    "system": ("window", "dt", "tau", "rate"),
}


def _number(part: str) -> float | None:
    try:
        v = complex(parse_expression(part, ()))
    except Exception:  # anything the parser refuses
        return None
    return v.real if v.imag == 0 and math.isfinite(v.real) else None


def _defaults(type_name: str) -> dict[str, Any]:
    from phasor.core.objects import get_type

    cls = get_type(type_name)
    return {k: f.get_default(call_default_factory=True) for k, f in cls.Params.model_fields.items()
            if not f.is_required()}  # fmt: skip


def _same(a: Any, b: Any) -> bool:
    """Equal as the document stores them (ints are floats, lists are tuples)."""
    from phasor.core.values import normalize

    try:
        return normalize(a) == normalize(b)
    except ValueError:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True,
                                                                        default=str)  # fmt: skip


LIMITS = {"dt": (1e-5, 1.0), "tau": (-flow.MAX_TAU, flow.MAX_TAU), "rate": (-1e3, 1e3)}


def _settings(ed: dict[str, Any], i: int) -> None:
    """Settings an edit names are checked here, so a bad one is refused with a reason and
    never turns a row into an error note or reaches the validator as a traceback."""

    def num(k: str) -> float:
        v = ed[k]
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise EditError(f"edit {i}: {k} must be a finite number, not {v!r}")
        return float(v)

    for k in ("remove", "visible"):
        if k in ed and not isinstance(ed[k], bool):
            raise EditError(f"edit {i}: {k} is true or false, not {ed[k]!r}")
    if "id" in ed and not isinstance(ed["id"], str):
        raise EditError(f"edit {i}: id is a row's id (text), not {ed['id']!r}")
    if "at" in ed and (isinstance(ed["at"], bool) or not isinstance(ed["at"], int)):
        raise EditError(f"edit {i}: at is a whole number (a place in the list)")
    for k in ("color", "kind", "base"):
        if k in ed and not isinstance(ed[k], str):
            raise EditError(f"edit {i}: {k} must be text")
    for k in ("value", "min", "max", "step", "dt", "tau", "rate"):
        if k in ed:
            num(k)
    if "step" in ed and ed["step"] <= 0:
        raise EditError(f"edit {i}: step must be above 0")
    if "min" in ed and "max" in ed and not ed["min"] < ed["max"]:
        raise EditError(f"edit {i}: min must be below max")
    for k, (lo, hi) in LIMITS.items():
        if k in ed and not lo <= ed[k] <= hi:
            raise EditError(f"edit {i}: {k} must be between {lo:g} and {hi:g}")
    if "window" in ed:
        w = ed["window"]
        ok = isinstance(w, list | tuple) and len(w) in (2, 3) and all(
            isinstance(p, list | tuple) and len(p) == 2
            and all(isinstance(a, int | float) and not isinstance(a, bool) and math.isfinite(a)
                    for a in p) and p[0] < p[1] for p in w)  # fmt: skip
        if not ok:
            raise EditError(f"edit {i}: window is a (lo, hi) pair per coordinate, lo < hi")


def _check(type_name: str, params: dict[str, Any]) -> str | None:
    """The validator's message for these parameters, or None when they are valid."""
    from pydantic import ValidationError

    from phasor.core.objects import get_type

    try:
        get_type(type_name).Params(**params)
    except ValidationError as e:
        msg = e.errors()[0].get("msg", str(e))
        return msg.removeprefix("Value error, ")
    except (ValueError, TypeError) as e:
        return str(e)
    return None


def _note(w: _Want, error: str | None) -> _Unit:
    return _Unit("note", "flow_note", [w], {"source": w.source, "kind": w.kind or "",
                                           "error": error or ""}, origin=w.origin,
                 error=error)  # fmt: skip


def _one_line(text: Any, i: int) -> str:
    """A row is one line (a system keeps its rows' lines in one ``source``)."""
    if not isinstance(text, str):
        raise EditError(f"edit {i}: text must be a string")
    if len(text) > flow.MAX_ROW_SOURCE // 2:
        raise EditError(f"edit {i}: a row is at most {flow.MAX_ROW_SOURCE // 2} characters")
    return " ".join(text.splitlines()) if ("\n" in text or "\r" in text) else text


def plan(doc: Document, edits: list[dict[str, Any]],
         camera_kind: str = "ortho") -> tuple[list[dict[str, Any]], dict[str, Any]]:  # fmt: skip
    """The ops of one commit and a report of what they do. ``edits``, each a dict:

    - ``{"text": "x' = y"}``: a new row (``"at": n`` puts it at place n);
    - ``{"id": key, "text": ...}``: retype a row; ``"remove": true`` deletes it;
      ``"kind"``: a kind over the guess (``""``: the guess); ``"visible"``; ``"color"`` (a
      role); a variable's ``"value"``, ``"min"``, ``"max"``, ``"step"``; a system's
      ``"window"``, ``"dt"``, ``"tau"``, ``"rate"`` (an ODE line's settings are its system's).

    Raises ``EditError`` for an edit that names no row or a setting the row does not have."""
    if not isinstance(edits, list | tuple):
        raise EditError("edits are a list of rows, e.g. [{'text': \"x' = y\"}]")
    if len(edits) > MAX_EDITS:
        raise EditError(f"at most {MAX_EDITS} edits in one call")
    current = rows_of(doc)
    specs = {o.id: o for o in doc.objects}
    want: list[_Want] = []
    by_key: dict[str, _Want] = {}
    for e in current:
        w = _Want(e["source"], origin=e["object"], kind=e.get("kind_override"), key=e["id"])
        want.append(w)
        by_key[e["id"]] = w
    by_object: dict[str, list[_Want]] = {}
    for w in want:
        by_object.setdefault(w.origin or "", []).append(w)
    removed_keys: list[str] = []
    for i, ed in enumerate(edits):
        if not isinstance(ed, dict):
            raise EditError(f"edit {i} is not an object: {ed!r}")
        unknown = set(ed) - {"id", "text", "remove", "kind", "visible", "color", "at", "version",
                             "base", *SETTINGS["variable"], *SETTINGS["system"]}  # fmt: skip
        if unknown:
            raise EditError(f"edit {i}: unknown fields {sorted(unknown)}")
        _settings(ed, i)
        if "text" in ed:
            ed = {**ed, "text": _one_line(ed["text"], i)}
        key = ed.get("id")
        if key is None:
            if not isinstance(ed.get("text"), str):
                raise EditError(f"edit {i}: a new row needs 'text'")
            w = _Want(ed["text"], changed=True, base=ed.get("base"))
            at = ed.get("at")
            want.insert(len(want) if at is None else max(0, min(int(at), len(want))), w)
            targets = [w]
        else:
            if key in by_key:
                targets = [by_key[key]]
            elif key in by_object:
                targets = by_object[key]  # a whole system: its settings, visibility, colour
                if "text" in ed and len(targets) > 1:
                    raise EditError(f"edit {i}: {key!r} has {len(targets)} rows; name one, e.g. "
                                    f"{targets[0].key!r}")  # fmt: skip
            else:
                raise EditError(f"edit {i}: no row {key!r}; rows are "
                                f"{[w.key for w in want if w.key]}")  # fmt: skip
            if ed.get("remove"):
                for w in targets:
                    if w in want:
                        want.remove(w)
                        removed_keys.append(w.key or "")
                continue
        for w in targets:
            if "text" in ed:
                if not isinstance(ed["text"], str):
                    raise EditError(f"edit {i}: text must be a string")
                w.source, w.changed = ed["text"], True
            if "kind" in ed:
                if ed["kind"] and ed["kind"] not in flow.ROW_KINDS[:-1]:
                    raise EditError(f"edit {i}: kind is one of {', '.join(flow.ROW_KINDS[:-1])} "
                                    "(or empty for the guess)")  # fmt: skip
                w.kind, w.changed = (ed["kind"] or None), True
            if "visible" in ed:
                w.visible, w.changed = bool(ed["visible"]), True
            if "color" in ed:
                w.color, w.changed = str(ed["color"]), True
            for s in (*SETTINGS["variable"], *SETTINGS["system"]):
                if s in ed:
                    w.settings[s], w.changed = ed[s], True
            if key is not None and "at" in ed and w in want:
                want.remove(w)
                want.insert(max(0, min(int(ed["at"]), len(want))), w)
    for w in want:
        w.row = read(w.source, w.kind)
    return _build(doc, specs, want, removed_keys, camera_kind)


def _build(doc: Document, specs: dict[str, ObjectSpec], want: list[_Want], removed_keys: list[str],
           camera_kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # fmt: skip
    units: list[_Unit] = []
    # ---- variables (rows), each name once
    var_units: dict[str, _Unit] = {}
    for w in list(want):
        r = w.row
        if r.kind != "variable":
            continue
        if r.name in var_units:
            first = var_units[r.name].rows[0]
            if w.origin is None:  # a new row naming a variable that has one: that row, retyped
                first.source, first.changed, first.row = w.source, True, w.row
                first.settings = {**first.settings, **w.settings}
                want.remove(w)
            else:
                units.append(_note(w, f"two rows set {r.name}; this one is not used"))
            continue
        u = _Unit("variable", "variable", [w], {}, origin=w.origin, name=r.name)
        var_units[r.name] = u
        units.append(u)
    # ---- the ODE rows: one system
    odes = [w for w in want if w.row.kind == "ode"]
    if odes:
        group = group_ode_rows([w.row for w in odes])
        errors = list(group.errors)
        if not errors and not group.autonomous:
            errors.append("a system that uses t (not autonomous) is not supported yet")
        bad = sorted(set(group.variables) & fm.RESERVED)
        if not errors and bad:
            errors.append(f"{', '.join(bad)} cannot be variables here (reserved by the flow "
                          "core); use another name")  # fmt: skip
        if errors:
            units += [_note(w, "; ".join(errors)) for w in odes]
        else:
            comps = dict(group.components)
            eqs = tuple(comps[c] for c in group.coordinates)
            origin = next((w.origin for w in odes if w.origin
                           and specs[w.origin].type == "flow_system"
                           and object_kind(specs[w.origin]) == "ode"), None)  # fmt: skip
            units.append(_Unit("ode", "flow_system", odes,
                               {"equations": eqs, "source": "\n".join(w.source for w in odes)},
                               origin=origin, name=",".join(group.variables),
                               children=(("field", "flow_arrows", {}),
                                         ("eq", "flow_equilibria", {}))))  # fmt: skip
    # ---- one row each
    for w in want:
        r = w.row
        if r.kind in ("variable", "ode"):
            continue
        if r.kind == "vector_field":
            bad = sorted(set(r.variables) & fm.RESERVED)
            if bad:
                units.append(_note(w, f"{', '.join(bad)} cannot be variables here (reserved by "
                                      "the flow core); use another name"))  # fmt: skip
                continue
            units.append(_Unit("vector_field", "flow_system", [w],
                               {"equations": tuple(r.parts), "source": w.source},
                               origin=w.origin, name=",".join(r.variables),
                               children=(("field", "flow_arrows", {}),)))  # fmt: skip
        elif r.kind == "point":
            x0 = [_number(p) for p in r.parts]
            if any(v is None for v in x0):
                units.append(_note(w, "a point of numbers only (a point that uses variables "
                                      "is not supported yet)"))  # fmt: skip
                continue
            units.append(_Unit("point", "flow_probe", [w], {"x0": tuple(x0), "source": w.source},
                               origin=w.origin))  # fmt: skip
        else:
            units.append(_note(w, r.error if r.kind == "error" else None))
    # ---- names used that no variable defines: new variables (commit time only)
    have = {**{n: u for n, u in var_units.items()}}
    # Variables that are not rows (a script's) count; a row variable removed here does not
    existing_vars = {_p(s, "name", "a"): s for s in specs.values()
                     if s.type == "variable" and not is_row(s)}  # fmt: skip
    used: list[str] = []
    for u in units:
        if u.type == "flow_system":
            for n in (u.name or "").split(","):
                if n and n not in used:
                    used.append(n)
    new_vars: list[str] = []
    for n in used:
        if n not in have and n not in existing_vars:
            w = _Want(f"{n} = 1", changed=True)
            w.row = read(w.source)
            u = _Unit("variable", "variable", [w], {}, name=n)
            have[n] = u
            units.append(u)
            want.append(w)
            new_vars.append(n)
    # ---- variable values (a System lists every name it uses in ``constants``)
    values: dict[str, float] = {}
    for n, s in existing_vars.items():
        v = _p(s, "value", 1.0)
        if isinstance(v, int | float):
            values[n] = float(v)
    for u in units:
        if u.kind != "variable":
            continue
        w = u.rows[0]
        origin = specs.get(w.origin) if w.origin else None
        if origin is None or origin.type != "variable":
            origin = existing_vars.get(u.name)
            u.origin = origin.id if origin is not None else None
        old = dict(NEW_VARIABLE)
        if origin is not None:
            old = {"value": float(_p(origin, "value", 1.0)), "min": float(_p(origin, "min", -10.0)),
                   "max": float(_p(origin, "max", 10.0))}  # fmt: skip
        typed = w.row.value if (w.changed and "value" not in w.settings) or origin is None else None
        value = float(w.settings.get("value", typed if typed is not None else old["value"]))
        lo = float(w.settings.get("min", old["min"]))
        hi = float(w.settings.get("max", old["max"]))
        if "min" not in w.settings and value < lo:
            lo = value - max(abs(value), 1.0)
        if "max" not in w.settings and value > hi:
            hi = value + max(abs(value), 1.0)
        if not lo < hi:
            raise EditError(f"variable {u.name}: min ({lo:g}) must be below max ({hi:g})")
        u.params = {"name": u.name, "value": value, "min": lo, "max": hi, "source": w.source}
        values[u.name] = value
    # ---- systems: constants, window, settings
    for u in units:
        if u.type != "flow_system":
            continue
        names = [n for n in (u.name or "").split(",") if n]
        u.params["constants"] = ", ".join(f"{n}={_num(values.get(n, 1.0))}" for n in names)
        dim = len(u.params["equations"])
        origin = specs.get(u.origin) if u.origin else None
        win = _p(origin, "window", None) if origin is not None else None
        if win is None or len(win) != dim:
            win = WINDOW_2D if dim == 2 else WINDOW_3D
        u.params["window"] = tuple(tuple(float(a) for a in w) for w in win)
        for w in u.rows:
            for s in SETTINGS["system"]:
                if s in w.settings:
                    v = w.settings[s]
                    if s == "window" and len(v) != dim:
                        raise EditError(f"the system has {dim} coordinates: its window needs "
                                        f"{dim} (lo, hi) pairs")  # fmt: skip
                    u.params[s] = tuple(tuple(float(a) for a in x) for x in v) \
                        if s == "window" else float(v)  # fmt: skip
    # ---- points: the system of their dimension
    systems = [u for u in units if u.type == "flow_system"]
    for u in units:
        if u.kind == "point":
            dim = len(u.params["x0"])
            match = next((s for s in systems if s.kind == "ode" and
                          len(s.params["equations"]) == dim), None) or next(
                (s for s in systems if len(s.params["equations"]) == dim), None)  # fmt: skip
            u.params["system"] = match  # resolved to an id below
    # ---- row order, visibility, kind override; every unit validated before it is planned
    order = {id(w): k for k, w in enumerate(want)}
    units.sort(key=lambda u: min(order[id(w)] for w in u.rows))
    final: list[_Unit] = []
    for k, u in enumerate(units):
        w0 = u.rows[0]
        vis = [w.visible for w in u.rows if w.visible is not None]
        origin = specs.get(u.origin) if u.origin else None
        u.params["row"] = k
        u.params["kind"] = (w0.kind or "") if u.type == "flow_note" else ""
        u.params["visible"] = vis[-1] if vis else bool(_p(origin, "visible", True)) \
            if origin is not None else True  # fmt: skip
        probe = {**u.params, "system": "s"} if u.kind == "point" else u.params
        err = _check(u.type, probe)
        if err is not None and any(w.settings for w in u.rows):
            raise EditError(f"row {w0.key or w0.source!r}: {err}")  # a setting, not the text
        if err is not None and u.type != "flow_note":
            note = _note(w0, err)
            note.rows = u.rows
            note.params.update(row=k, visible=u.params["visible"],
                               source="\n".join(w.source for w in u.rows))  # fmt: skip
            u = note
        final.append(u)
    units = final
    # ---- ids: keep an object whose kind is unchanged; name new ones by meaning
    taken: set[str] = {o.id for o in doc.objects} | set(doc.retired_ids)
    claimed: set[str] = set()
    for u in units:
        o = specs.get(u.origin) if u.origin else None
        same = (o is not None and o.type == u.type and u.origin not in claimed
                and (_p(o, "name") == u.name if u.type == "variable"
                     else object_kind(o) == _unit_kind(u)))  # fmt: skip
        if same:
            u.id = u.origin
            claimed.add(u.id)
    for u in units:
        if u.id is not None:
            continue
        hint = next((w.base for w in u.rows if w.base), None)
        base = hint or (name_from(u.name, "var") if u.kind == "variable" else
                        BASE.get(_unit_kind(u), "row"))  # fmt: skip
        u.id = _fresh(base, taken)
        taken.add(u.id)
    for u in units:
        if u.kind == "point":
            s = u.params["system"]
            u.params["system"] = s.id if isinstance(s, _Unit) and s.type == "flow_system" else ""
    # ---- ops
    ops: list[dict[str, Any]] = []
    keep = {u.id for u in units}
    gone = sorted(s.id for s in specs.values() if is_row(s) and s.id not in keep)
    controls = {c.path: c for c in doc.controls}
    for oid in gone:
        if f"{oid}.value" in controls:
            ops.append({"op": "remove_control", "path": f"{oid}.value"})
        for child in sorted((i for i in specs if i.startswith(f"{oid}.")),
                            key=lambda i: -i.count(".")):  # fmt: skip
            ops.append({"op": "remove_object", "id": child})
        ops.append({"op": "remove_object", "id": oid})
    report_rows: list[dict[str, Any]] = []
    added_3d = False
    for u in units:
        o = specs.get(u.id)
        if o is None or u.id in gone:
            ops.append({"op": "add_object", "object": {"id": u.id, "type": u.type,
                                                       "params": _plain(u.params)}})  # fmt: skip
            for suffix, t, params in u.children:
                ops.append({"op": "add_object", "object": {"id": f"{u.id}.{suffix}", "type": t,
                                                           "params": params}})  # fmt: skip
            if u.type == "flow_system" and len(u.params["equations"]) == 3:
                added_3d = True
            fresh = True
        else:
            defaults = _defaults(u.type)
            for k, v in u.params.items():
                if not _same(o.params.get(k, defaults.get(k)), v):
                    ops.append({"op": "set_param", "path": f"{u.id}.{k}", "value": _plain(v)})
            fresh = False
        if u.kind == "variable":
            path = f"{u.id}.value"
            c = controls.get(path)
            step = float(u.rows[0].settings.get("step", c.step if c else NEW_VARIABLE["step"]))
            ctl = {"path": path, "min": u.params["min"], "max": u.params["max"], "step": step,
                   "label": u.name}  # fmt: skip
            if c is None or fresh or (c.min, c.max, c.step, c.label) != (
                    ctl["min"], ctl["max"], ctl["step"], ctl["label"]):  # fmt: skip
                ops.append({"op": "add_control", "control": ctl})
        color = next((w.color for w in u.rows if w.color), None)
        if color is not None:
            ops.append(_color_op(u, color))
        if any(w.changed for w in u.rows) or fresh:
            report_rows.append({"id": u.id, "kind": _unit_kind(u), "type": u.type,
                                "error": u.error, "new": fresh,
                                "text": [w.row.text for w in u.rows]})  # fmt: skip
    if added_3d and camera_kind == "ortho":
        view = {"azimuth": -50.0, "elevation": 20.0}
        ops.append({"op": "set_camera", "kind": "orbit", "params": view})
    report = {"rows": report_rows, "removed": gone, "new_variables": new_vars,
              "removed_rows": removed_keys}  # fmt: skip
    return ops, report


def _unit_kind(u: _Unit) -> str:
    if u.type == "flow_note":
        return "error" if u.error else (u.rows[0].row.kind if u.rows[0].row else "error")
    return u.kind


def _fresh(base: str, taken: set[str]) -> str:
    """The first free id from ``base`` (``system``, ``system_2``): not used, not retired, and
    no id (a retired child) already under it."""
    base = name_from(base, "row")
    reserved = set(taken) | {i.split(".")[0] for i in taken if "." in i}
    return unique_names([base], reserved)[0]


def _plain(v: Any) -> Any:
    if isinstance(v, tuple | list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _color_op(u: _Unit, color: str) -> dict[str, Any]:
    from phasor.core.objects import get_type

    keys = [k for k, (g, _) in getattr(get_type(u.type), "Style", {}).items() if g == "emphasis"]
    if not keys:
        raise EditError(f"row {u.id!r} draws nothing, so it has no colour")
    return {"op": "set_style", "id": u.id, "key": keys[0], "role": color}


# ---------------------------------------------------------------- presets


def preset_edits(name: str, current: list[dict[str, Any]], replace: bool = True) -> list[dict]:
    """The edits that load ``flow_math.PRESETS[name]``: its ODE rows (the system named after
    the preset), a row per variable with its range, the window and ``dt``; with ``replace``,
    every current row is removed first (in the same patch)."""
    if name not in fm.PRESETS:
        raise EditError(f"no preset {name!r}; presets are {sorted(fm.PRESETS)}")
    p = fm.PRESETS[name]
    edits: list[dict[str, Any]] = []
    keep: dict[str, str] = {}
    keep = {e["name"]: e["id"] for e in current if e["kind"] == "variable"}
    if replace:  # every row goes, except the variables the preset sets again
        edits += [{"id": e["id"], "remove": True} for e in current
                  if not (e["kind"] == "variable" and e["name"] in p.variables)]  # fmt: skip
    for c, eq in zip(fm.COORDS, p.equations, strict=False):
        edits.append({"text": f"{c}' = {eq}", "base": name, "window": [list(w) for w in p.window],
                      "dt": p.dt})  # fmt: skip
    for v, (value, lo, hi) in p.variables.items():
        span = {"min": lo, "max": hi, "value": value}
        if v in keep:
            edits.append({"id": keep[v], **span})
        else:
            edits.append({"text": f"{v} = {_num(value)}", **span})
    return edits


# ---------------------------------------------------------------- analysis (the AI tools)


def system_ids(doc: Document) -> list[str]:
    """The systems, an ODE row system first."""
    ids = [o for o in doc.objects if o.type == "flow_system"]
    ids.sort(key=lambda o: (not (is_row(o) and object_kind(o) == "ode"), not is_row(o)))
    return [o.id for o in ids]


def _finite(v: Any, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise EditError(f"{what} must be a number, not {v!r}") from None
    if not math.isfinite(x):
        raise EditError(f"{what} must be a finite number")
    return x


def resolve(engine, system: str | None = None, t: float = 0.0,
            variables: dict[str, float] | None = None):  # fmt: skip
    """``(id, flow.Resolved)`` of a system at scene time ``t``. ``variables`` (name -> value)
    are applied on a fork of the engine (a Variable's value), or to the system's constants
    for a name no Variable holds; the open scene never changes."""
    doc = engine.document
    ids = system_ids(doc)
    if not ids:
        raise EditError("there is no system: type rows like x' = y, y' = -x, or load a preset "
                        f"(explorer_preset; one of {', '.join(sorted(fm.PRESETS))})")  # fmt: skip
    sid = system or ids[0]
    if sid not in ids:
        raise EditError(f"no system {sid!r}; systems are {ids}")
    t = _finite(t, "t")
    over: dict[str, float] = {}
    eng = engine
    if variables:
        if not isinstance(variables, dict):
            raise EditError("variables is a table of name -> value, e.g. {'rho': 24}")
        firsts: dict[str, str] = {}
        for o in doc.objects:
            if o.type == "variable":
                firsts.setdefault(_p(o, "name", "a"), o.id)
        ops = []
        for name, v in variables.items():
            x = _finite(v, f"the value of {name}")
            if name in firsts:
                ops.append({"op": "set_param", "path": f"{firsts[name]}.value", "value": x})
            over[name] = x
        if ops:
            eng = engine.fork()
            r = eng.apply(ops, origin="script")
            if not r.ok:
                raise EditError("; ".join(i.message for i in r.errors))
    st = eng.state_at(t)
    r = st[sid].impl.resolve(st)
    unknown = sorted(set(over) - set(r.values))
    if unknown:
        raise EditError(f"the system {sid!r} does not use {unknown}; it uses "
                        f"{sorted(r.values) or 'no variables'}")  # fmt: skip
    if over:
        r.values = {**r.values, **over}
        r.vals = r.sys.values_of(r.values)
    return sid, r


def _window(r, window) -> tuple[list[float], list[float]]:
    if window is None:
        return [float(a) for a in r.lo], [float(b) for b in r.hi]
    try:
        pairs = [(float(a), float(b)) for a, b in window]
    except (TypeError, ValueError):
        raise EditError("window is a (lo, hi) pair per coordinate") from None
    if len(pairs) != r.dim or not all(math.isfinite(a) and math.isfinite(b) and a < b
                                      for a, b in pairs):  # fmt: skip
        raise EditError(f"window needs {r.dim} finite (lo, hi) pairs with lo < hi")
    return [a for a, _ in pairs], [b for _, b in pairs]


def _r6(v: Any) -> Any:
    import numpy as np

    if isinstance(v, dict):
        return {k: _r6(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_r6(x) for x in v]
    if isinstance(v, np.ndarray):
        return _r6(v.tolist())
    if isinstance(v, bool | np.bool_):
        return bool(v)
    if isinstance(v, int | np.integer):
        return int(v)
    if isinstance(v, float | np.floating):
        x = float(v)
        return round(x, 6) if math.isfinite(x) else None
    return v


def _provenance(sid: str, r, lo, hi, **more: Any) -> dict[str, Any]:
    return {"system": sid, "window": [[a, b] for a, b in zip(lo, hi, strict=True)],
            "dt": r.dt, "variables": dict(r.values),
            "escape_radius": r.escape[1], **more}  # fmt: skip


def find_equilibria(engine, window=None, grid: int = 9, variables=None, system=None,
                    t: float = 0.0) -> dict[str, Any]:  # fmt: skip
    sid, r = resolve(engine, system, t, variables)
    lo, hi = _window(r, window)
    grid = int(min(max(int(_finite(grid, "grid")), 2), 64))
    found = fm.classified_equilibria(r.sys, r.values, lo, hi, grid)
    pts = [{"at": c["at"], "kind": c["kind"], "stable": c["stable"],
            "eigenvalues": [list(e) for e in c["eigenvalues"]], "dims": c["dims"],
            "trace": c["trace"], "det": c["det"], "hyperbolic": c["dims"]["c"] == 0,
            "subspaces": {k: c[k] for k in ("Es", "Eu", "Ec")}, "tolerance": c["tolerance"]}
           for c in found["points"]]  # fmt: skip
    return _r6({
        **_provenance(sid, r, lo, hi, grid=found["grid"], starts=found["starts"], T=None),
        "count": len(pts), "points": pts, "non_isolated": found["non_isolated"],
        "converged": True,
        "tolerance": {"newton_residual": 1e-9, "newton_step": fm.XTOL,
                      "merge_share_of_window": fm.MERGE,
                      "classify": "1e-7 x max|J| per point (its 'tolerance')"},
        "notes": ["searched only inside the window, from a grid of starts: none found there "
                  "is not none exist", "only starts where Newton converged are reported",
                  "a centre or a weak focus cannot be told apart from the linear part: "
                  "such a point is 'non-hyperbolic' or 'centre'"],
    })  # fmt: skip


def flow_probe(engine, x0, T: float = 10.0, samples: int = 16, variables=None, system=None,
               t: float = 0.0) -> dict[str, Any]:  # fmt: skip
    sid, r = resolve(engine, system, t, variables)
    T = _finite(T, "T")
    if abs(T) < r.dt:
        raise EditError(f"T must be at least one step (dt = {r.dt:g})")
    try:
        import numpy as np

        seeds = np.atleast_2d(np.asarray(x0, dtype=np.float64))
    except (TypeError, ValueError):
        raise EditError("x0 is a point (a list of numbers) or a list of points") from None
    if seeds.ndim != 2 or seeds.shape[1] != r.dim:
        raise EditError(f"x0 needs {r.dim} coordinates per point")
    samples = int(_finite(samples, "samples"))
    if not 2 <= samples <= fm.MAX_SAMPLES:
        raise EditError(f"samples is 2 to {fm.MAX_SAMPLES}")
    try:
        out = fm.probe(r.sys, r.values, seeds, T, r.dt, samples=samples, size=r.diag,
                       escape=r.escape, max_step=r.max_step)  # fmt: skip
    except fm.FunctionError as e:
        raise EditError(str(e)) from None
    agree = all(s["dt_check"]["agree"] for s in out["seeds"])
    return _r6({
        **_provenance(sid, r, list(r.lo), list(r.hi), grid=None), **out, "converged": agree,
        "notes": ["positions are given only up to chaotic_beyond (where an error of 1e-12 of "
                  "the window has grown to 1e-3 at the largest Lyapunov exponent); after it, "
                  "statistics", "the Lyapunov estimate is converged only when 'converged' "
                  "says so; quote its spread", "converged here means dt and dt/2 agreed"],
    })  # fmt: skip


MEASURES = ("equilibria",)


def sweep_parameter(engine, name: str, lo: float, hi: float, n: int = 17,
                    measure: str = "equilibria", window=None, grid: int = 9, variables=None,
                    system=None, t: float = 0.0) -> dict[str, Any]:  # fmt: skip
    if measure not in MEASURES:
        raise EditError(f"measure is one of {MEASURES} (a Lyapunov or extent sweep is "
                        "Milestone 30)")  # fmt: skip
    sid, r = resolve(engine, system, t, variables)
    lo, hi = _finite(lo, "lo"), _finite(hi, "hi")
    n = int(_finite(n, "n"))
    if not 2 <= n <= fm.MAX_SWEEP:
        raise EditError(f"n is 2 to {fm.MAX_SWEEP}")
    wlo, whi = _window(r, window)
    grid = int(min(max(int(_finite(grid, "grid")), 2), 64))
    try:
        out = fm.sweep(r.sys, r.values, name, lo, hi, n, wlo, whi, grid)
    except fm.FunctionError as e:
        raise EditError(str(e)) from None
    held = {k: v for k, v in r.values.items() if k != name}
    res = _r6({
        **_provenance(sid, r, wlo, whi, T=None), **out, "variables": held,
        "converged": all(c["converged"] for c in out["candidates"]),
        "tolerance": {"bisection_width": out["bisection"]["width"], "newton_residual": 1e-9},
        "notes": ["candidates are named by the local test that found them; a count change "
                  "or a crossing between two values of the sweep is all they say",
                  "global bifurcations are not detected: " + out["not_detected"]],
    })  # fmt: skip
    # a bisected value keeps its digits (rounding to 6 would undo the bisection)
    for c, raw in zip(res["candidates"], out["candidates"], strict=True):
        c["value"], c["bracket"] = raw["value"], raw["bracket"]
    res["bisection"]["width"] = out["bisection"]["width"]
    res["tolerance"]["bisection_width"] = out["bisection"]["width"]
    return res


def state(engine, t: float = 0.0, locked: dict[str, str] | None = None) -> dict[str, Any]:
    """``explorer_state``: the rows, the systems with their equilibria, the variables, the
    camera and lint, each with where its numbers came from."""
    doc = engine.document
    t = _finite(t, "t")
    rows = rows_of(doc)
    locked = locked or {}
    for e in rows:
        e["locked"] = e["id"] in locked
    st = engine.state_at(t)
    systems = []
    space = "R3" if doc.cameras and doc.cameras[0].kind == "orbit" else "R2"
    for sid in system_ids(doc):
        sem = st[sid].impl.semantics(st)
        r = st[sid].impl.resolve(st)
        if r.dim == 3:
            space = "R3"
        try:
            eq = find_equilibria(engine, system=sid, t=t)
            eq = {k: eq[k] for k in ("window", "grid", "starts", "count", "points", "dt", "T",
                                     "non_isolated", "converged", "tolerance")}  # fmt: skip
        except EditError as e:  # pragma: no cover - a system always resolves here
            eq = {"error": str(e)}
        systems.append({"id": sid, **{k: sem[k] for k in ("dim", "equations", "latex",
                        "variables", "from_variables", "divergence", "linear", "window", "tau",
                        "dt", "rate")}, "equilibria": eq})  # fmt: skip
    variables = {e["name"]: {k: e[k] for k in ("id", "value", "min", "max", "step", "slider")}
                 for e in rows if e["kind"] == "variable"}  # fmt: skip
    for o in doc.objects:
        if o.type == "variable" and not is_row(o):
            variables.setdefault(_p(o, "name", "a"), {"id": o.id, "value": _p(o, "value", 1.0),
                                                      "row": False})  # fmt: skip
    issues = engine.lint()
    return _r6({
        "version": engine.version, "t": t, "space": space,
        "camera": doc.cameras[0].kind if doc.cameras else "ortho",
        "rows": rows, "systems": systems, "variables": variables,
        "locked": sorted(locked),
        "lint": [{"code": i.code, "severity": i.severity, "ids": list(i.ids),
                  "message": i.message} for i in issues[:30]],
        "notes": ["t is the scene time asked for: the playhead lives in the editor, not in "
                  "the document", "equilibria are searched inside each system's window only",
                  "for how paths behave call flow_probe; for parameters, sweep_parameter",
                  "rows of kind graph, curve, surface, heat map and implicit curve are read "
                  "but not drawn yet"],
    })  # fmt: skip
