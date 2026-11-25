# Equation and emphasis verbs

Six verbs for algebra on screen and for "look here". Each is a choreography verb: timing and
easing come from the theme's `[motion.<verb>]`, every segment it writes is tagged with the
verb's name (so `lint(strict=True)` calls it choreographed, not `raw_motion`), and it reads the
scene at its anchor time. Verbs that touch one path (say two verbs on the same `flash`) must
be written in time order, like any track.

An `Equation` is typeset in parts; the verbs that point at a part address it by its **key**:
`Equation(parts=("$y$", "$=a$", "$+b$"), keys=("y", "eq", "plus"))`.

```python
from phasor import Counter, Dot, Equation, Line, Scene, Text

s = Scene("equations_demo", passes=None)
s.add(
    Equation(
        id="f",
        parts=("$y$", "$=a$", "$+b$", "$-b$"),
        keys=("y", "eq", "plus", "minus"),
        pos=(0, 3, 0),
    )
)
s.add(Text(id="given", text="a = 3", pos=(-5, 1, 0)))
for name, t in (("plug", 1.0), ("cross", 4.0)):
    s.marker(name, t)
```

## substitute

`substitute(eq_id, key, value, source=None)` replaces one part of an equation by `value`:
the old text leaves where it stands, the new one flies in from the object `source` (an
equation, counter or text id) and the other parts slide to make room. With no `source` the new
part fades in where it stands. `value` is mathtext (`"$=3$"`) or a number (`3` becomes `$3$`).
An equation takes one substitution; to go on, `transform` to a new equation.
`f.parts` already reads the new text from the start of the document: until the verb runs the
equation still draws (and the inspector still reports) the old one, through the parameters
`swap_key`, `swap_old` and `swap`.

```python
s.at("plug").substitute("f", "eq", "$=3$", source="given")
```

MCP: `animate(verb="substitute", targets=["f"], value={"key": "eq", "value": "$=3$", "source": "given"}, at="plug")`.

## cancel

`cancel(eq_id, key_a, key_b, ...)` strikes through two (or more) terms, then fades them and
closes the gap: the remaining parts glide together. The equation's `struck` parameter lists the
keys, `strike` and `close` animate the stroke and the fading. Once closed, the inspector no
longer lists the cancelled parts. An equation takes one `cancel`.

```python
s.add(
    Equation(
        id="sum",
        parts=("$x$", "$+5$", "$-5$", "$=7$"),
        keys=("x", "up", "down", "rhs"),
        pos=(0, -1, 0),
    )
)
s.at("cross").cancel("sum", "up", "down")
```

MCP: the keys follow the equation in `targets`: `animate(verb="cancel", targets=["sum", "up", "down"], at="cross")`.

## factor and expand

`factor(source_eq, target_eq)` and `expand(source_eq, target_eq)` are `transform` with their
own tag and duration and their own arc. Parts are matched as `transform` matches them (same key,
or same text in order). `factor` carries parts under the others and draws them together
beneath a bracket; `expand` carries them over the others and spreads them apart. The target is
hidden until the verb, the source disappears as it starts, as with `transform`.

```python
s.add(Equation(id="spread", parts=("$ab$", "$+ac$"), keys=("ab", "ac"), pos=(0, -3, 0)))
s.add(Equation(id="factored", parts=("$a$", "$(b+c)$"), keys=("ab", "ac"), pos=(0, -3, 0)))
s.add(Equation(id="spread_again", parts=("$ab$", "$+ac$"), keys=("ab", "ac"), pos=(0, -3, 0)))
s.marker("fac", 7.0)
s.marker("exp", 10.0)
s.at("fac").factor("spread", "factored")
s.at("exp").expand("factored", "spread_again")
```

MCP: `animate(verb="factor", targets=["spread", "factored"], at="fac")`.

## ignite

`ignite(id..., settle=0.3)` lights a stroke like a fuse: a bright head runs along a line or
any curve (anything that offers `path_points`, as `travel` does) leaving the stroke lit
behind it, then the whole stroke settles brighter (`settle` is the share of a `flash` it keeps;
0 puts it back). The head is a `Tracer` child, `<id>.fuse`, made the first time and reused.

```python
s.add(Line(id="rail", points=[(-4, -5, 0), (0, -4, 0), (4, -5, 0)]))
s.marker("fuse", 13.0)
s.at("fuse").ignite("rail")
```

MCP: `animate(verb="ignite", targets=["rail"], value={"settle": 0.3}, at="fuse")`.

## pulse

`pulse(id..., n=2)` is `flash` repeated `n` times, quick and even: each beat is half the theme's
`pulse` duration, so the default two beats take one `pulse` duration. Several ids stagger as
`flash` does.

```python
s.add(Dot(id="beat", pos=(5, -4.5, 0)))
s.marker("beat", 16.0)
s.at("beat").pulse("beat", n=3)
assert not [i for i in s.engine.lint(strict=True) if i.code == "raw_motion"]
```

MCP: `animate(verb="pulse", targets=["beat"], value={"n": 3}, at="beat")`.
