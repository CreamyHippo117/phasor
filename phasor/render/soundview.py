"""A picture of the scene's sound (SOUND.md step 5), for an AI that cannot hear it.

One lane per source: the voice and each bed as their momentary loudness over time (LUFS,
400 ms windows every 100 ms), cues as diamonds where they land, markers as faint lines. The
theme's voice level is a dashed line, so "the bed sits well under the voice" is something
you can see. Drawn with matplotlib's Agg canvas (no window, the same on every system).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from phasor.time import sound as snd

FLOOR = -60.0  # LUFS shown at the bottom of a lane
TOP = -6.0
INK = "#e8e6f0"
FAINT = "#5c5a70"
COLORS = {"voice": "#ffd27a", "bed": "#7ad7ff", "cue": "#ff8fd0"}


def curve(samples: np.ndarray, rate: int, start: float) -> tuple[np.ndarray, np.ndarray]:
    """(times, momentary loudness in LUFS) of one lane: 400 ms windows every 100 ms, centred."""
    z = snd.block_powers(samples, rate)
    with np.errstate(divide="ignore"):
        lufs = -0.691 + 10.0 * np.log10(np.maximum(z, 1e-30))
    t = start + 0.2 + 0.1 * np.arange(len(z))
    return t, np.maximum(lufs, FLOOR)


def picture(m: snd.Mix, markers: Mapping[str, float], voice_level: float, path: str | Path,
            title: str = "") -> Path:  # fmt: skip
    """Draw ``m`` (a ``time/sound.Mix``) to the PNG ``path`` and return the path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    voice = [x for x in m.lanes if x.kind == "voice"]
    beds = [x for x in m.lanes if x.kind == "bed"]
    cues = [x for x in m.lanes if x.kind == "cue"]
    rows: list[tuple[str, str, list[Any]]] = []
    if voice:
        rows.append(("voice", "voice", voice))
    rows += [(b.id, "bed", [b]) for b in beds]
    if cues:
        rows.append(("cues", "cue", cues))
    rows = rows or [("silence", "voice", [])]
    fig = Figure(figsize=(16, 1.2 + 1.4 * len(rows)), dpi=100, facecolor="#0b0a12")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(rows), 1, sharex=True, squeeze=False)[:, 0]
    end = max(m.duration, max((x.start + len(x.samples) / m.rate for x in m.lanes), default=0))
    for ax, (name, kind, lanes) in zip(axes, rows, strict=True):
        ax.set_facecolor("#0b0a12")
        for side in ax.spines.values():
            side.set_color(FAINT)
        ax.tick_params(colors=INK, labelsize=9)
        ax.set_ylim(FLOOR, TOP)
        ax.set_xlim(0, max(end, 0.1))
        ax.set_ylabel(name, color=COLORS[kind], fontsize=11, rotation=0, ha="right", va="center")
        for t in markers.values():
            ax.axvline(t, color=FAINT, lw=0.6, alpha=0.6)
        ax.axhline(voice_level, color=INK, lw=0.8, ls="--", alpha=0.35)
        if kind == "cue":
            for x in lanes:
                level = snd.momentary_max(x.samples, m.rate)
                y = level if np.isfinite(level) else FLOOR
                ax.plot([x.start], [y], marker="D", color=COLORS[kind], ms=8)
                ax.annotate(x.id, (x.start, y), xytext=(4, 6), textcoords="offset points",
                            color=INK, fontsize=8)  # fmt: skip
            continue
        if kind == "voice" and lanes:
            bus = snd.mix([snd.Placed(x.start, x.samples, m.rate) for x in lanes], m.duration)
            t, y = curve(bus[0], m.rate, 0.0)
        elif lanes:
            t, y = curve(lanes[0].samples, m.rate, lanes[0].start)
        else:
            t, y = np.zeros(0), np.zeros(0)
        ax.fill_between(t, FLOOR, y, color=COLORS[kind], alpha=0.25, lw=0)
        ax.plot(t, y, color=COLORS[kind], lw=1.2)
        for a, b in m.speech if kind == "bed" else ():
            ax.axvspan(a, b, color=COLORS["voice"], alpha=0.06, lw=0)
    top = axes[0]
    for name, t in markers.items():
        top.annotate(name, (t, TOP), xytext=(2, -10), textcoords="offset points", color=FAINT,
                     fontsize=8, rotation=0)  # fmt: skip
    axes[-1].set_xlabel("seconds", color=INK)
    fig.suptitle(title or "sound (LUFS; dashed: the voice level)", color=INK, fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.12, hspace=0.25)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    return path
