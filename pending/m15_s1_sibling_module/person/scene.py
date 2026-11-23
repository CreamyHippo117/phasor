"""Re-audit S1 (owner decision, MILESTONE_15.md Deferred). Run: python person/scene.py
from pending/m15_s1_sibling_module/. The person's scene imports gemlib beside it; later in
the same process a downloaded document that names gemlib.py (not in its folder) and type
gem2 is opened with allow_scripts=False. Expected for contract 3: missing_type and no draw.
Today: the process's Gem2 is used and its draw runs."""

import pathlib

from gemlib import MARK, Gem2

from phasor import Scene
from phasor.engine import Engine

s = Scene("s1", look="none", passes=None)
s.add(Gem2(id="g"))

MARK.unlink(missing_ok=True)
doc = pathlib.Path(__file__).parents[1] / "downloaded" / "b2.phasor.json"
e = Engine.open(doc, passes=None, allow_scripts=False)
st = e.state_at(0.0)
st["evil"].impl.world_bounds(st)
print("impl from:", type(st["evil"].impl).__module__)
print("lint:", [i.code for i in e.lint()])
print("gemlib draw ran:", MARK.exists())
MARK.unlink(missing_ok=True)
