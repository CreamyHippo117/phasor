"""A document as a Python builder script (the editor's code view, SPEC "GUI editor").

``document_to_python(doc)`` writes a script that, run, builds the same scene: objects with
their parameters, style roles and tags, the camera, markers, animations (as ``at(...).to``),
constraints and controls. An object of a type defined in the scene's folder, or with
updaters, is written as the document's own ``add_object`` op (ADR 0005). Narration blocks
are not reproduced (their audio belongs to the voice module): a comment says so; a narration
read from a recording (M22) is, as ``s.narrate(script, audio=...)``. Running the
result and diffing it with the document gives no differences except narration
(``tests/test_pycode.py``).
"""

from __future__ import annotations

import functools
from typing import Any

from phasor.core.objects import is_builtin, registered_types
from phasor.core.schema import Document, ExprSpec, MarkerRef

# class module -> how the generated script reaches it
# from phasor import Dot, Image, Video, ...
_TOP = {
    "phasor.domains.basic",
    "phasor.domains.media",
    "phasor.domains.video",
    "phasor.domains.midi",
}
_ALIAS = {
    "phasor.domains.cx": "cx",
    "phasor.domains.acoustics": "ac",
    "phasor.domains.fourier": "fourier",
    "phasor.domains.linalg": "linalg",
    "phasor.domains.pde": "pde",
    "phasor.domains.algebra": "algebra",
    "phasor.domains.calc": "calc",
    "phasor.domains.geometry": "geo",
    "phasor.domains.cloud": "cloud",
    "phasor.domains.network": "nw",
    "phasor.domains.dynamics": "dyn",
    "phasor.domains.flow": "flow",
    "phasor.domains.flow_analysis": "flow",  # flow.Section, ... (Milestone 30)
    "phasor.domains.shapes": "shapes",
    "phasor.domains.path": "shapes",  # shapes.Path
    "phasor.domains.svg": "shapes",  # shapes.SVG
    "phasor.domains.diagrams": "dg",
}


def _lit(v: Any) -> str:
    if isinstance(v, ExprSpec):
        return f"expr({v.expr!r})"
    if isinstance(v, tuple | list):
        inner = ", ".join(_lit(x) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_lit(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return repr(v)
    return repr(v)


def _class_ref(type_name: str, uses: set[str]) -> str:
    cls = registered_types().get(type_name)
    if cls is None:
        raise ValueError(f"unknown object type {type_name!r}")
    if cls.__module__ in _TOP:
        uses.add(cls.__name__)
        return cls.__name__
    alias = _ALIAS.get(cls.__module__)
    if alias is None:
        raise ValueError(f"no import for {cls.__module__}")
    uses.add(alias)
    return f"{alias}.{cls.__name__}"


def _op(name: str, **fields: Any) -> str:
    """A patch op as a Python literal (from its JSON form: lists, True, None)."""
    return repr({"op": name, **fields})


def _time(t: float | MarkerRef) -> tuple[str, float]:
    """(when, offset) arguments for ``s.at``."""
    if isinstance(t, MarkerRef):
        return repr(t.marker), float(t.offset)
    return repr(float(t)), 0.0


def _ref(t: float | MarkerRef) -> str:
    if isinstance(t, MarkerRef):
        return f"({t.marker!r}, {float(t.offset)!r})"
    return repr(float(t))


def document_to_python(doc: Document) -> str:
    uses: set[str] = {"Scene"}
    m = doc.meta
    lines: list[str] = []
    args = [repr(m.title), f"theme={m.theme!r}", f"life={m.life!r}", f"seed={m.seed}",
            f"width={m.width}", f"height={m.height}", f"fps={m.fps!r}"]  # fmt: skip
    if m.look:
        args.append(f"look={m.look!r}")
    if m.palette:
        args.append(f"palette={m.palette!r}")
    if m.background is not None:
        args.append(f"background={_lit(m.background)}")
    lines.append(f"s = Scene({', '.join(args)})")
    cam = doc.cameras[0] if doc.cameras else None
    if cam is not None and cam.kind == "orbit":
        kw = ", ".join(f"{k}={_lit(v)}" for k, v in cam.params.items())
        lines.append(f"s.use_camera('orbit'{', ' + kw if kw else ''})")
    elif cam is not None:
        for k, v in cam.params.items():
            lines.append(f"s.set('{cam.id}.{k}', {_lit(v)})")
    lines.append("")
    if m.modules:  # code in the scene's folder (ADR 0005): its types and updaters
        modules = _op("set_meta", field="modules", value=list(m.modules))
        lines.append(f"s.engine.apply({modules}, origin='script')")
    if m.duration is not None:  # set by the scene (Scene.from_video: as long as its clip)
        duration = _op("set_meta", field="duration", value=m.duration)
        lines.append(f"s.engine.apply({duration}, origin='script')")
    for o in doc.objects:
        cls = registered_types().get(o.type)
        if o.always or cls is None or not is_builtin(cls):
            # A scene-local type or an updater has no import: the document's own op says it.
            obj = o.model_dump(mode="json")
            lines.append(f"s.engine.apply({_op('add_object', object=obj)}, origin='script')")
            continue
        ref = _class_ref(o.type, uses)
        parts = [f"id={o.id!r}"] + [f"{k}={_lit(v)}" for k, v in o.params.items()]
        if o.style:
            parts.append(f"style={_lit(dict(o.style))}")
        if o.tags:
            parts.append(f"tags={_lit(list(o.tags))}")
        if o.layer != "world":
            parts.append(f"layer={o.layer!r}")
        lines.append(f"s.add({ref}({', '.join(parts)}))")
    if doc.objects:
        lines.append("")
    owned = set(doc.narration.markers()) if doc.narration else set()
    for name, t in doc.timeline.markers.items():
        if name not in owned:
            lines.append(f"s.marker({name!r}, {float(t)!r})")
    if doc.narration is not None and doc.narration.voice.startswith("file:"):
        # a recording (M22): narrating from the same file finds the same sentences again
        audio = doc.narration.voice[len("file:") :]
        lines.append(f"s.narrate({doc.narration.script!r}, audio={audio!r})")
    elif doc.narration is not None:
        lines.append("# (narration is not reproduced here: record it with s.narrate(...))")
    for path, segs in doc.timeline.tracks.items():
        for seg in segs:
            same_anchor = (
                isinstance(seg.t0, MarkerRef) == isinstance(seg.t1, MarkerRef)
                and (not isinstance(seg.t0, MarkerRef) or seg.t0.marker == seg.t1.marker)
            )  # fmt: skip
            end = seg.t1.offset if isinstance(seg.t1, MarkerRef) else seg.t1
            start = seg.t0.offset if isinstance(seg.t0, MarkerRef) else seg.t0
            # ``at(...).to(duration=)`` only when start + duration gives the stored end back
            # exactly (0.36 + (1.76 - 0.36) is not 1.76); else both ends, as stored.
            if same_anchor and float(start) + float(end - start) == float(end):
                when, off = _time(seg.t0)
                at = f"s.at({when}" + (f", {off!r}" if off else "") + ")"
                lines.append(f"{at}.to({path!r}, {_lit(seg.to)}, duration={float(end - start)!r}, "
                             f"ease={seg.ease!r})")  # fmt: skip
            else:
                lines.append(f"s.segment({path!r}, {_ref(seg.t0)}, {_ref(seg.t1)}, {_lit(seg.to)}, "
                             f"ease={seg.ease!r})")  # fmt: skip
    for c in doc.constraints:
        lines.append(f"s.constrain({c.rule!r}, priority={c.priority!r})")
    for c in doc.controls:
        extra = f", component={c.component}" if c.component is not None else ""
        lines.append(f"s.control({c.path!r}, min={c.min!r}, max={c.max!r}, step={c.step!r}, "
                     f"label={c.label!r}{extra})")  # fmt: skip
    top = sorted(u for u in uses if u[0].isupper())
    mods = sorted(u for u in uses if u[0].islower())
    head = [
        "from phasor import "
        + ", ".join([*top, *(["expr"] if "expr(" in "\n".join(lines) else []), *mods])
    ]
    return "\n".join([*head, "", *lines]) + "\n"


# ---------------------------------------------------------------- the tweaks block (ADR 0004)

# A script scene keeps the editor's and the AI's tweaks (``Overrides``) in a managed block at
# its end, as ordinary code: one line per tweak, in a canonical order, so rewriting the block
# never reorders anything and the code outside the markers stays byte for byte as it was.
TWEAKS_BEGIN = "# --- phasor tweaks:"  # the first line starts with this
TWEAKS_HEADER = (
    "# --- phasor tweaks: phasor rewrites this block when you or an AI tweak the scene; "
    "put your own code above it ---"
)
TWEAKS_END = "# --- end phasor tweaks ---"
TWEAK_CALLS = ("tweak", "tweak_style", "tweak_rule")


def tweak_lines(overrides: Any, var: str = "s") -> list[str]:
    """The block's lines for ``overrides`` (an ``Overrides``): parameters sorted by path, then
    styles sorted by ``id.key``, then layout rules in the order they were added (their
    order is the order they are laid over the script's own)."""
    data = overrides.model_dump(mode="json")
    lines = [f"{var}.tweak({path!r}, {value!r})" for path, value in sorted(data["params"].items())]
    for path, value in sorted(data["styles"].items()):
        target, _, key = path.rpartition(".")
        lines.append(f"{var}.tweak_style({target!r}, {key!r}, {value!r})")
    for c in data["constraints"]:
        extra = "" if c["priority"] == "strong" else f", priority={c['priority']!r}"
        lines.append(f"{var}.tweak_rule({c['rule']!r}{extra})")
    return lines


def _split(text: str) -> list[str]:
    """Lines with their ends, split on "\\n" only (as ``tokenize`` and ``ast`` count them; a
    form feed or a Unicode line separator inside a line is not a new line)."""
    lines = [x + "\n" for x in text.split("\n")]
    lines[-1] = lines[-1][:-1]
    return [x for x in lines if x]


def _newline(text: str) -> str:
    """The line end most lines of ``text`` use (new lines are written with it)."""
    crlf = text.count("\r\n")
    return "\r\n" if crlf and crlf >= text.count("\n") - crlf else "\n"


class BlockError(ValueError):
    """The tweaks block's markers cannot be read for certain (a begin without an end, two
    blocks, an end before its begin, a marker indented or the code not readable yet):
    nothing is written into the script until the person fixes it. ``line`` is where."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message + (f" (line {line})" if line else ""))
        self.line = line


MAX_SCAN = 64 * 1024 * 1024  # characters: a script longer than this is not scanned (refused)


@functools.lru_cache(maxsize=16)
def _scan(text: str) -> tuple[tuple[int, ...], tuple[int, ...], str | None, int | None]:
    """The 1-based lines of the begin and end markers in ``text`` that are real markers, and
    a reason (with its line) when the markers cannot be read for certain.

    A marker is a comment line at column 0 that lies between the module's top-level
    statements (``ast``: not inside a string or docstring, a bracket, an f-string or a
    compound statement's body). Only lines holding the marker text are looked at, and the
    parse is the C parser's (``tokenize`` is quadratic in line length on Python 3.12: a
    pasted data line would freeze the window), cached per text."""
    import ast
    import bisect

    if len(text) > MAX_SCAN:
        return (), (), "the script is too long to look for its tweaks block", None
    candidates = TWEAKS_BEGIN in text or TWEAKS_END in text
    if "\r" in text.replace("\r\n", ""):
        return (
            (),
            (),
            (
                "the script ends lines with a lone CR (old Mac line ends): save it with "
                "LF or CRLF line ends so its tweaks block can be found for certain"
            ),
            None,
        )
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as e:
        return (), (), ("the script cannot be read yet (" + (getattr(e, "msg", None) or str(e))
                        + "), so its tweaks block cannot be found for certain"), getattr(
                            e, "lineno", None)  # fmt: skip
    if not candidates:
        return (), (), None, None
    starts, ends_ = [], []
    for node in tree.body:
        first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        starts.append(first)
        ends_.append(node.end_lineno or node.lineno)
    begins: list[int] = []
    ends: list[int] = []
    for k, line in enumerate(text.split("\n"), 1):
        if "# --- " not in line:
            continue
        bare = line.strip()
        begin, end = bare.startswith(TWEAKS_BEGIN), bare == TWEAKS_END
        if not (begin or end):
            continue
        i = bisect.bisect_right(starts, k) - 1
        if i >= 0 and starts[i] <= k <= ends_[i]:
            continue  # inside a statement: a string, a bracket, a body
        if line[:1] in (" ", "\t"):
            return (), (), ("a tweaks marker is indented: markers start at the beginning of "
                            "their line"), k  # fmt: skip
        (begins if begin else ends).append(k)
    return tuple(begins), tuple(ends), None, None


def find_tweaks(text: str, *, strict: bool = True) -> tuple[int, int] | None:
    """Where the block is in ``text``: (offset of its begin line, offset just past its end
    line and that line's newline), or None when the script has none. Only real marker
    comments at column 0 between statements count (``_scan``). A begin without an end, an
    end without a begin, an end before its begin, two blocks, an indented marker, code that
    cannot be read yet, or lone-CR line ends: ``BlockError`` (nothing is guessed; phasor
    writes nothing until the person fixes it). ``strict=False`` (reading only, never
    writing) says None instead of raising."""
    begins, ends, problem, where = _scan(text)
    try:
        if problem is not None:
            raise BlockError(problem, where)
        if not begins and not ends:
            return None
        if len(begins) > 1 or len(ends) > 1:
            where = (begins[1:] or ends[1:])[0]
            raise BlockError("the script has more than one tweaks block: keep one (move the "
                             "tweaks into it)", where)  # fmt: skip
        if not ends:
            raise BlockError("the tweaks block has a begin marker but no end marker "
                             f"({TWEAKS_END!r})", begins[0])  # fmt: skip
        if not begins:
            raise BlockError("the tweaks block has an end marker but no begin marker", ends[0])
        if ends[0] < begins[0]:
            raise BlockError("the tweaks block's end marker comes before its begin marker",
                             ends[0])  # fmt: skip
    except BlockError:
        if strict:
            raise
        return None
    starts = [0]
    for line in _split(text):
        starts.append(starts[-1] + len(line))
    return starts[begins[0] - 1], starts[ends[0]]


@functools.lru_cache(maxsize=8)
def _guard_line(text: str) -> int | None:
    """The 0-based line of a trailing ``if __name__ == "__main__":`` block, or None: new code
    goes before it, so ``python scene.py`` sees it too."""
    import ast

    try:
        body = ast.parse(text).body
    except (SyntaxError, ValueError):
        return None
    if not body:
        return None
    last = body[-1]
    test = getattr(last, "test", None)
    if (isinstance(last, ast.If) and isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name) and test.left.id == "__name__"):  # fmt: skip
        return last.lineno - 1
    return None


def insertion_offset(text: str) -> int:
    """Where code added to a script goes (the app's panels add objects, narration, a snap):
    before the tweaks block, else before a trailing ``__main__`` guard, else at the end."""
    found = find_tweaks(text)
    if found is not None:
        return found[0]
    guard = _guard_line(text)
    if guard is not None:
        lines = _split(text)
        return sum(len(x) for x in lines[:guard])
    return len(text)


def insert_code(text: str, code: list[str]) -> str:
    """``text`` with the lines ``code`` added at ``insertion_offset``; nothing else changes."""
    nl = _newline(text)
    at = insertion_offset(text)
    head, tail = text[:at], text[at:]
    if head and not head.endswith(("\n", "\r")):
        head += nl
    return head + "".join(line + nl for line in code) + (nl if tail.strip() else "") + tail


def with_tweaks(text: str, overrides: Any, var: str = "s") -> str:
    """``text`` with its tweaks block holding ``overrides``: the block replaced in place, or
    added (before a trailing ``__main__`` guard, else at the end) when there are tweaks and no
    block yet. An existing block stays, emptied, when there are none. Everything outside the
    block is kept byte for byte (a missing final newline is added when the block goes last)."""
    nl = _newline(text)
    lines = tweak_lines(overrides, var)
    if not lines and TWEAKS_BEGIN not in text and TWEAKS_END not in text:
        return text  # nothing to write and no block: nothing to look for
    block = TWEAKS_HEADER + nl + "".join(line + nl for line in lines) + TWEAKS_END + nl
    found = find_tweaks(text)
    if found is not None:
        a, b = found
        return text[:a] + block + text[b:]
    if not lines:
        return text
    guard = _guard_line(text)
    if guard is not None:
        src = _split(text)
        at = sum(len(x) for x in src[:guard])
        return text[:at] + block + nl + text[at:]
    if text and not text.endswith(("\n", "\r")):
        text += nl
    return text + (nl if text.strip() else "") + block


def read_tweaks(text: str) -> Any:
    """The ``Overrides`` the block in ``text`` holds, read without running anything (its
    lines are calls with literal arguments; any other line is skipped). Empty when there is
    no block. What "saved" means for a script scene's tweaks."""
    import ast

    from phasor.core.schema import ConstraintSpec, Overrides

    found = find_tweaks(text, strict=False)
    params: dict[str, Any] = {}
    styles: dict[str, Any] = {}
    rules: list[ConstraintSpec] = []
    if found is None:
        return Overrides()
    for line in text[found[0] : found[1]].splitlines():
        try:
            node = ast.parse(line.strip(), mode="eval").body
        except SyntaxError:
            continue
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in TWEAK_CALLS):  # fmt: skip
            continue
        try:
            args = [ast.literal_eval(a) for a in node.args]
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
        except ValueError:
            continue
        name = node.func.attr
        if name == "tweak" and len(args) == 2:
            params[args[0]] = args[1]
        elif name == "tweak_style" and len(args) == 3:
            styles[f"{args[0]}.{args[1]}"] = args[2]
        elif name == "tweak_rule" and len(args) >= 1:
            prio = kw.get("priority", args[1] if len(args) > 1 else "strong")
            if all(r.rule != " ".join(str(args[0]).split()) for r in rules):
                rules.append(ConstraintSpec(rule=args[0], priority=prio))
    return Overrides(params=params, styles=styles, constraints=tuple(rules))
