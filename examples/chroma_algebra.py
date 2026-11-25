"""The chroma look on algebra: a lattice turned by a quaternion.

A cubic lattice of points (-2..2 on each axis) is drawn as a glowing cage, every edge and node
coloured by where it sits, so one rainbow runs across the whole. Multiplying every point v by
a unit quaternion, v -> q v q*, turns space: first by q = cos(pi/6) + sin(pi/6) i (a turn of
60 degrees about the i axis), then by q = cos(pi/6) + sin(pi/6) j on top. The cage keeps its
colours as it moves: hue is fixed to the lattice point, so you can follow each one. The
camera swings slowly round. The look is ``chroma``: deep indigo, candy hues, white type.

Run:  uv run python examples/chroma_algebra.py      (prints the matrix of each turn)
"""

import math

from phasor import Equation, Scene, linalg


def turn(axis: int, degrees: float):
    """The 3x3 matrix of v -> q v q* for q = cos(a/2) + sin(a/2) e_axis (a rotation by a)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    m = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    a, b = [k for k in range(3) if k != axis]
    m[a][a], m[a][b], m[b][a], m[b][b] = c, -s, s, c
    return tuple(tuple(round(x, 12) for x in row) for row in m)


def times(m, n):
    return tuple(tuple(sum(m[i][k] * n[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))  # fmt: skip


I3 = turn(0, 0.0)
QI = turn(0, 60.0)  # q = cos 30 + sin 30 i
QJI = times(turn(1, 60.0), QI)  # then q = cos 30 + sin 30 j

s = Scene("chroma_algebra", look="chroma")
s.use_camera("orbit")
s.set("camera.target", (0.0, 0.0, 0.0))
s.set("camera.distance", 14.0)
s.set("camera.elevation", 20.0)
s.set("camera.azimuth", -35.0)
s.add(linalg.TransformGrid(id="cage", matrix=I3, extent=2, lattice=True, square=0.0,
                           basis=0.0))  # fmt: skip
s.add(Equation(id="by_q", parts=(r"$v \;\mapsto\; q\,v\,\bar q$",), pos=(8 / 9, 0.92),
               layer="overlay", style={"size": "title"}))  # fmt: skip

s.marker("appear", 0.3)
s.marker("turn_i", 2.0)
s.marker("turn_j", 4.5)
s.marker("end", 8.0)
s.at("appear").reveal("cage")
s.at("appear", 0.6).write("by_q")
s.at("turn_i").morph("cage.matrix", QI)
s.at("turn_j").morph("cage.matrix", QJI)
s.at("appear").orbit(40.0, until="end")

if __name__ == "__main__":
    for t in (1.0, 4.0, 7.5):
        sem = s.engine.inspect(t, id="cage")["objects"][0]["semantic"]
        print(f"t = {t}: det {sem['determinant']}, matrix {sem['matrix']}")
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
