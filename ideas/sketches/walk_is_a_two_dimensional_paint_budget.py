"""Throwaway still for the idea card `walk_is_a_two_dimensional_paint_budget` (idea-scout).

Top: a random walk on the plane, started at the centre and stopped when it first leaves a
disc. Each dot is a grid site; its brightness is the expected number of visits to it. The
disc is shown at three sizes (radius 4, 8, 16), drawn the same size on screen: the picture
looks the same every time, and the marked half-way ring keeps the same shade (about 0.44).
Bottom: the visits home gained each time the radius doubles, as bricks. On a line they double,
on the plane they are equal for ever, in space they halve and the pile stops under a ceiling.

Run:  uv run python ideas/sketches/walk_is_a_two_dimensional_paint_budget.py
"""

from pathlib import Path

import numpy as np
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import spsolve

from phasor import Dot, Line, Scene, Text


def visits_2d(radius: int) -> dict[tuple[int, int], float]:
    """Expected visits to each site before a walk from the origin first leaves the disc."""
    sites = [
        (x, y)
        for x in range(-radius, radius + 1)
        for y in range(-radius, radius + 1)
        if x * x + y * y < radius * radius
    ]
    index = {p: i for i, p in enumerate(sites)}
    step = lil_matrix((len(sites), len(sites)))
    for (x, y), i in index.items():
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if q in index:
                step[i, index[q]] = 0.25
    start = np.zeros(len(sites))
    start[index[(0, 0)]] = 1.0
    g = spsolve((identity(len(sites)) - step.T).tocsc(), start)
    return {p: float(g[i]) for p, i in index.items()}


s = Scene("walk_is_a_two_dimensional_paint_budget", look="none")
s.set("camera.height", 9)

HOME = {}  # disc radius -> expected visits home
for k, radius in enumerate((4, 8, 16)):
    g = visits_2d(radius)
    HOME[radius] = g[(0, 0)]
    cx, cy, half = -4.6 + 4.6 * k, 2.2, 1.9  # every disc drawn the same size
    u = half / radius
    for (x, y), v in g.items():
        s.add(
            Dot(
                id=f"site_{radius}_{x + radius}_{y + radius}",
                pos=(cx + u * x, cy + u * y),
                r=0.42 * u,
                opacity=float(np.clip(v / 1.6, 0.04, 1.0)),
                style={"color": "highlight"},
            )
        )
    ring = [
        (cx + half / 2 * np.cos(a), cy + half / 2 * np.sin(a))
        for a in np.linspace(0, 2 * np.pi, 73)
    ]
    s.add(
        Line(
            id=f"halfway_ring_{radius}",
            points=ring,
            style={"color": "secondary", "stroke": "hairline"},
        )
    )
    mid = np.mean([v for (x, y), v in g.items() if abs(np.hypot(x, y) - radius / 2) < 0.5])
    s.add(
        Text(
            id=f"ring_shade_{radius}",
            text=f"radius {radius}: ring shade {mid:.2f}",
            pos=(cx, cy - half - 0.35),
            style={"size": "annotation"},
        )
    )

# Bricks: visits home gained per doubling of the radius. The plane's come from the discs
# above; the line's are exactly R; space's are the attacker's computed values (R = 2, 4, 8).
BRICKS = {
    "line": [2.0, 4.0, 8.0],
    "plane": [HOME[8] - HOME[4], HOME[16] - HOME[8]] * 2,  # two computed, drawn twice
    "space": [0.110, 0.055, 0.028],
}
BASE, UNIT, WIDTH = -3.7, 1.0, 0.7  # one visit = UNIT world units
for k, (name, bricks) in enumerate(BRICKS.items()):
    x0, y = -4.6 + 4.6 * k - WIDTH / 2, BASE
    for i, b in enumerate(bricks):
        top = min(y + UNIT * b, BASE + 2.8)  # the line's pile runs off the top
        s.add(
            Line(
                id=f"{name}_brick_{i}",
                style={"color": "highlight" if name == "plane" else "muted"},
                points=[(x0, y), (x0 + WIDTH, y), (x0 + WIDTH, top), (x0, top), (x0, y)],
            )
        )
        y = top
        if y >= BASE + 2.8:
            break
    s.add(
        Text(
            id=f"{name}_label",
            text=name,
            pos=(x0 + WIDTH / 2, BASE - 0.3),
            style={"size": "annotation"},
        )
    )
ceiling = BASE + UNIT * 0.22  # space: about 0.22 more visits in all beyond radius 2
s.add(
    Line(
        id="space_ceiling",
        points=[(4.6 - 0.9, ceiling), (4.6 + 0.9, ceiling)],
        style={"color": "secondary"},
    )
)
s.add(
    Text(
        id="bricks_title",
        text="visits home gained per doubling of the radius",
        pos=(0, BASE + 3.15),
        style={"size": "annotation"},
    )
)

if __name__ == "__main__":
    from phasor.render.output import save_png

    print("visits home:", {r: round(v, 4) for r, v in HOME.items()})
    print("plane bricks:", [round(b, 4) for b in BRICKS["plane"][:2]])
    save_png(Path(__file__).with_suffix(".png"), s.engine.still(0.0, (1600, 900)))
