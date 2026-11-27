"""ac.ModeSum: the room's impulse response summed over modes against summed over echoes."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac, expr
from phasor.domains import ac_math
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
SIZE, SRC, MIC = (3.0, 2.5, 2.0), (0.7, 0.9, 0.5), (2.2, 1.6, 1.3)
SIGMA = 1.0e-3
TIMES = np.linspace(0.0, 0.025, 200)


def test_modes_are_sorted_by_frequency_and_complete():
    n, omega = ac_math.room_modes(SIZE, 500)
    assert tuple(n[0]) == (0, 0, 0) and omega[0] == 0
    assert np.all(np.diff(omega) >= 0)
    # nothing below the 500th mode was left out: count every index triple under its frequency
    top = omega[-1]
    ks = [np.arange(int(top / ac_math.C * L / math.pi) + 2) * math.pi / L for L in SIZE]
    k2 = ks[0][:, None, None] ** 2 + ks[1][None, :, None] ** 2 + ks[2][None, None, :] ** 2
    assert int((ac_math.C * np.sqrt(k2) <= top).sum()) in (500, 501, 502, 503, 504, 505, 506)
    assert ac_math.room_modes(SIZE, 500)[1] == pytest.approx(omega, abs=0)  # deterministic
    with pytest.raises(ValueError, match="modes"):
        ac_math.room_modes(SIZE, 0)


def test_the_first_modes_are_the_axial_standing_waves():
    n, omega = ac_math.room_modes((4.0, 3.0, 2.0), 3)
    assert [tuple(r) for r in n] == [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    assert omega[1] / (2 * math.pi) == pytest.approx(343.0 / (2 * 4.0))  # c / 2L


def test_every_partial_sum_is_a_lookup_equal_to_summing_those_modes():
    tab = ac_math.ModeSumTable(SIZE, SRC, MIC, 1000, SIGMA, TIMES)
    n, omega = ac_math.room_modes(SIZE, 1000)
    for count in (0, 1, 7, 255, 256, 257, 600, 1000):
        direct = np.zeros(len(TIMES))
        for i in range(count):
            amp = np.prod(np.where(n[i] == 0, 1.0, 2.0)) / np.prod(SIZE)
            for a in range(3):
                q = n[i, a] * math.pi / SIZE[a]
                amp *= math.cos(q * SRC[a]) * math.cos(q * MIC[a])
            w = omega[i]
            wave = TIMES if w == 0 else np.sin(w * TIMES) / w
            direct += amp * math.exp(-0.5 * (w * SIGMA) ** 2) * wave
        assert tab.at(count) == pytest.approx(direct, rel=1e-9, abs=1e-15), count
    assert tab.at(10**9) == pytest.approx(tab.at(1000), abs=0)  # past the end: all of them


def test_the_mode_sum_converges_on_the_image_sum():
    """The same smoothed impulse response two ways: the gap closes as notes are added."""
    images = ac_math.image_trace(SIZE, SRC, MIC, SIGMA, TIMES)
    peak = np.abs(images).max()
    tab = ac_math.ModeSumTable(SIZE, SRC, MIC, 6000, SIGMA, TIMES)
    gaps = [np.abs(tab.at(n) - images).max() / peak for n in (10, 100, 1000, 6000)]
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3]
    assert gaps[0] > 0.1 and gaps[3] < 1e-6, gaps
    # the bound the object reports tracks what is left over
    assert gaps[3] <= 1e-12 + 100 * math.exp(-0.5 * (tab.omega[-1] * SIGMA) ** 2)


def test_the_image_trace_puts_the_direct_sound_at_distance_over_c():
    times = np.linspace(0.0, 0.02, 4001)
    sigma = 5e-5
    trace = ac_math.image_trace(SIZE, SRC, MIC, sigma, times)
    d = float(np.linalg.norm(np.subtract(SRC, MIC)))
    first = int(np.argmax(trace > 0.5 * trace[np.argmin(np.abs(times - d / ac_math.C))]))
    near = slice(max(first - 40, 0), first + 40)
    assert times[near][np.argmax(trace[near])] == pytest.approx(d / ac_math.C, abs=2e-6)
    peak = 1 / (4 * math.pi * ac_math.C**2 * d) / (sigma * math.sqrt(2 * math.pi))
    assert trace[np.argmin(np.abs(times - d / ac_math.C))] == pytest.approx(peak, rel=1e-3)
    assert np.abs(trace[times < d / ac_math.C - 6 * sigma]).max() < 1e-6 * peak  # silent before


def test_image_trace_refuses_a_span_it_cannot_cover():
    with pytest.raises(ValueError, match="order"):
        ac_math.image_trace((0.2, 0.2, 0.2), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), 1e-3,
                            np.linspace(0, 5.0, 4))  # fmt: skip


def scene(**kw) -> Scene:
    s = Scene("ms_test", passes=None)
    params = dict(size=SIZE, source=SRC, listener=MIC, smoothing=SIGMA, samples=200,
                  max_notes=6000, notes=1.0, layer="overlay")  # fmt: skip
    s.add(ac.ModeSum(id="ms", **(params | kw)))
    return s


def sem(s: Scene, t: float) -> dict:
    return s.engine.inspect(t, id="ms")["objects"][0]["semantic"]


def test_morphing_the_notes_draws_a_trace_for_every_whole_number_between():
    s = scene()
    s.marker("go", 1.0)
    s.marker("done", 5.0)
    s.at("go").morph("ms.notes", 6000.0, until="done")
    e = s.engine
    seen, gaps = [], []
    for t in np.linspace(0.0, 6.0, 25):
        r = sem(s, float(t))
        seen.append(r["notes"])
        gaps.append(r["max_rel_diff"])
        want = round(e.state_at(float(t))["ms"].params.notes)
        assert r["notes"] == want  # the count shown is the count drawn
    assert seen[0] == 1 and seen[-1] == 6000
    assert seen == sorted(seen) and len(set(seen)) > 15  # no stage jumps
    assert gaps[-1] < 1e-6 and sem(s, 6.0)["converged"]
    assert gaps[0] > 0.1 and sem(s, 6.0)["truncation_bound"] < 1e-5


def test_state_is_pure_in_the_document_and_time():
    a, b = scene(notes=expr("t * 100")), scene(notes=expr("t * 100"))
    ma, mb = a.engine.state_at(3.3)["ms"].impl, b.engine.state_at(3.3)["ms"].impl
    ta, tb = a.engine.state_at(3.3), b.engine.state_at(3.3)
    assert np.array_equal(ma.modes(ta), mb.modes(tb))
    assert np.array_equal(ma.modes(ta), ma.modes(a.engine.state_at(3.3)))
    assert sem(a, 3.3) == sem(b, 3.3)


def test_semantics_bounds_and_features():
    s = scene(notes=500.0, pos=(0.1, 0.2, 0.0), extent=(0.6, 0.3))
    r = sem(s, 0.0)
    assert r["kind"] == "mode_sum" and r["notes"] == 500 and r["notes_total"] == 6000
    assert 0 < r["highest_mode_hz"] < r["last_mode_hz"] and not r["converged"]
    assert not r["notes_clipped"]
    assert sem(scene(notes=9000.0), 0.0)["notes_clipped"]
    assert sem(scene(notes=9000.0), 0.0)["notes"] == 6000
    st = s.engine.state_at(0.0)
    feats = st["ms"].impl.features(st)
    assert {"mode_trace", "image_trace", "mode_end", "image_end"} <= set(feats)
    assert "image_trace" not in scene(compare=False).engine.state_at(0.0)["ms"].impl.features(
        scene(compare=False).engine.state_at(0.0))  # fmt: skip


def test_bad_parameters_are_refused():
    with pytest.raises(Exception, match="t_range"):
        scene(t_range=(0.02, 0.01))
    with pytest.raises(Exception, match="inside the room"):
        scene(listener=(9.0, 1.0, 1.0))
    with pytest.raises(Exception, match="smoothing"):
        scene(smoothing=0.0)


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.slow
def test_mode_sum_golden_storyboard():
    from tests.test_beauty import _golden

    scene_ = runpy.run_path(str(ROOT / "examples" / "mode_sum.py"))["s"]
    engine = scene_.engine
    assert engine.lint(strict=True) == []
    sheet = engine.storyboard([0.5, 2.0, 3.0, 4.0, 7.0], size=(320, 180))["image"]
    _golden("mode_sum", sheet, engine.pass_runner().renderer_class())
