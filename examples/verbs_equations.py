"""Equation and emphasis verbs, one beat each: a value is plugged into a formula (substitute),
two terms cross out and the gap closes (cancel), an expression is factored and expanded
again, a stroke is lit like a fuse (ignite) and a point pulses.

Run:  uv run python examples/verbs_equations.py      (prints what the inspector reads)
"""

import math

from phasor import Dot, Equation, Line, Scene, Text

s = Scene("verbs_equations", look="prism_everywhere")
s.set("camera.height", 10.0)

# Top left: plug a = 3 into the formula.
s.add(
    Equation(
        id="formula",
        parts=("$f$", "$=a$", "$x$", "$+b$"),
        keys=("f", "a", "x", "b"),
        colors=("primary", "secondary", "primary", "highlight"),
        pos=(-4.5, 3.4, 0),
        style={"size": "title"},
    )
)
s.add(Text(id="given", text="a = 3", pos=(-4.5, 1.4, 0), style={"size": "title"}))

# Top right: cancel the two fives.
s.add(
    Equation(
        id="sum",
        parts=("$x$", "$+5$", "$-5$", "$=7$"),
        keys=("x", "up", "down", "rhs"),
        colors=("primary", "highlight", "highlight", "secondary"),
        pos=(4.5, 3.4, 0),
        style={"size": "title"},
    )
)

# Middle: factor, then expand again (each is a new equation the last one becomes).
s.add(
    Equation(
        id="spread",
        parts=("$ab$", "$+ac$"),
        keys=("ab", "ac"),
        colors=("primary", "secondary"),
        pos=(0, -0.2, 0),
        style={"size": "title"},
    )
)
s.add(
    Equation(
        id="factored",
        parts=("$a$", "$(b+c)$"),
        keys=("ab", "ac"),
        colors=("primary", "secondary"),
        pos=(0, -0.2, 0),
        style={"size": "title"},
    )
)
s.add(
    Equation(
        id="spread_again",
        parts=("$ab$", "$+ac$"),
        keys=("ab", "ac"),
        colors=("primary", "secondary"),
        pos=(0, -0.2, 0),
        style={"size": "title"},
    )
)

# Bottom: a stroke to light and a point that pulses.
wave = [(-6.0 + 0.1 * k, -3.2 + 0.8 * math.sin(0.35 * k), 0.0) for k in range(100)]
s.add(Line(id="wave", points=wave))
s.add(Dot(id="beat", pos=(7.2, -3.2, 0), r=0.2))

for name, t in (
    ("start", 0.0),
    ("plug", 0.5),
    ("cross", 3.0),
    ("factor", 6.0),
    ("expand", 9.0),
    ("fuse", 12.0),
    ("beat", 15.0),
    ("end", 17.5),
):
    s.marker(name, t)
s.at("plug").substitute("formula", "a", "$=3$", source="given")
s.at("cross").cancel("sum", "up", "down")
s.at("factor").factor("spread", "factored")
s.at("expand").expand("factored", "spread_again")
s.at("fuse").ignite("wave")
s.at("beat").pulse("beat", n=3)

if __name__ == "__main__":
    e = s.engine
    for oid, t in (("formula", 3.0), ("sum", 5.0), ("spread_again", 12.0), ("wave.fuse", 12.8)):
        print(oid, e.inspect(t, id=oid)["objects"][0]["semantic"])
