"""A cue sheet: every marker and every motion of a scene, one row each, for lining up music
and sound design with the transitions in an audio editor (``phasor cues``).

Built on ``Engine.timeline()`` (the markers) and the document's tracks (which verb made each
transition). One verb call writes several transitions (a staggered ``reveal`` of three
objects, a ``flash`` that swells and falls, a ``transform`` that hides one equation and
moves the next); they are one cue: the segments a verb anchored at the same marker whose
spans touch or overlap. Two calls of a verb at different markers stay two cues even when
one ends where the next begins. A transition written with raw ``to()`` is a ``raw`` cue.
The document does not record verb calls, so two calls of one verb at one marker whose spans
touch are one cue too (``at("m").reveal("a").then().reveal("b")``; ``.then()`` with a
different verb, as the examples write it, stays two cues), as are touching segments of one
verb anchored at plain seconds.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from phasor.core.schema import MarkerRef
from phasor.engine import Engine
from phasor.time.tracks import TimeMap

COLUMNS = ("time", "timecode", "kind", "name", "start", "end", "duration")
EPS = 1e-6


def _ms(t: float) -> int:
    return round(t * 1000)


def seconds(t: float) -> str:
    """Seconds to the millisecond, from whole milliseconds (``59.9996`` is ``60.000``)."""
    ms = _ms(t)
    sign, ms = ("-", -ms) if ms < 0 else ("", ms)
    return f"{sign}{ms // 1000}.{ms % 1000:03d}"


def timecode(t: float) -> str:
    """``HH:MM:SS.mmm``, from whole milliseconds, so it never reads ``00:00:60.000``."""
    ms = max(_ms(t), 0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _gestures(engine: Engine, time_map: TimeMap) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None], list[tuple[float, float, str]]] = {}
    for path, segs in engine.document.timeline.tracks.items():
        owner = path.rpartition(".")[0]
        for seg in segs:
            anchor = seg.t0.marker if isinstance(seg.t0, MarkerRef) else None
            span = (time_map.resolve(seg.t0), time_map.resolve(seg.t1), owner)
            groups.setdefault((seg.verb or "raw", anchor), []).append(span)
    out = []
    for (verb, _), spans in groups.items():
        cluster: list[tuple[float, float, str]] = []
        for span in sorted(spans):
            if cluster and span[0] > max(s[1] for s in cluster) + EPS:
                out.append(_gesture(verb, cluster))
                cluster = []
            cluster.append(span)
        out.append(_gesture(verb, cluster))
    return out


def _gesture(verb: str, spans: list[tuple[float, float, str]]) -> dict[str, Any]:
    start, end = min(s[0] for s in spans), max(s[1] for s in spans)
    # In the order they start moving, then finish: a staggered reveal lists its objects in
    # turn, and a transform its source (hidden at once) before its target (which moves on).
    first = {}
    for t0, t1, owner in spans:
        a, b = first.get(owner, (t0, t1))
        first[owner] = (min(a, t0), max(b, t1))
    ids = sorted(first, key=lambda i: (_ms(first[i][0]), _ms(first[i][1]), i))
    return {"verb": verb, "ids": ids, "start": start, "end": end}


def cue_rows(engine: Engine) -> list[dict[str, Any]]:
    """The scene's cues in time order (a marker before a motion that starts with it).

    Each row: ``time`` (seconds), ``kind`` (``"marker"`` or ``"motion"``), ``name`` (the
    marker's name, or the verb and the ids it moves), and for a motion ``verb``, ``ids``,
    ``start``, ``end`` and ``duration`` (``None`` for a marker). The ids are named in the
    order they start moving, then finish (a transform: source, then target)."""
    markers = engine.timeline()["markers"]
    rows: list[dict[str, Any]] = [
        {"time": t, "kind": "marker", "name": name, "start": None, "end": None,
         "duration": None}
        for name, t in markers.items()
    ]  # fmt: skip
    for g in _gestures(engine, TimeMap(markers)):
        rows.append({"time": g["start"], "kind": "motion", "name": " ".join([g["verb"], *g["ids"]]),
                     "verb": g["verb"], "ids": g["ids"], "start": g["start"], "end": g["end"],
                     "duration": (_ms(g["end"]) - _ms(g["start"])) / 1000})  # fmt: skip
    rows.sort(key=lambda r: (_ms(r["time"]), r["kind"] != "marker", r["name"],
                             _ms(r["end"] or 0.0)))  # fmt: skip
    return rows


def cues_csv(rows: list[dict[str, Any]]) -> str:
    """The rows as CSV text with a header (``COLUMNS``), ``\\n`` line ends, times to the
    millisecond. Write it with ``newline=""`` so Windows keeps the line ends."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        start, end, duration = ("" if r[k] is None else seconds(r[k])
                                for k in ("start", "end", "duration"))  # fmt: skip
        w.writerow([seconds(r["time"]), timecode(r["time"]), r["kind"], r["name"],
                    start, end, duration])  # fmt: skip
    return buf.getvalue()
