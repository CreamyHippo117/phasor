"""Fourier analysis objects (SPEC "Domain modules: fourier").

    from phasor import fourier

    s.add(fourier.Epicycles(id="heart", points=curve, terms=24, phase=expr("t / 10")))
    s.add(fourier.PartialSum(id="wave", kind="square", terms=9))
    s.add(fourier.Spectrum(id="spec", frequencies=(100.5,), amplitudes=(1.0,), window="hann"))
    s.add(fourier.Spectrogram(id="sg", f0=200, f1=1200))

Numerics are in ``fourier_math``; these draw them. Everything is a pure function of the
parameters.
"""

from __future__ import annotations

import functools
import itertools
import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.color import ramp
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import fourier_math as fm
from phasor.domains.heatmap import heat_pixels, lut_colour
from phasor.domains.strokes import circle, disc, hue_colors, polyline, with_alpha


def _ydir(me, state: SceneState) -> float:
    """+1 in a world layer, -1 on the overlay (y runs down): plots grow upward either way."""
    return 1.0 if state.camera_for(me).up[1] >= 0 else -1.0


# ---------------------------------------------------------------- Epicycles


@functools.lru_cache(maxsize=16)
def _coeffs(points: tuple, terms: int):
    return fm.epicycle_coefficients(np.asarray(points, dtype=np.float64), terms)


@register
class Epicycles(ObjectBase):
    """A closed curve drawn by a chain of rotating circles: each circle is one Fourier
    coefficient. ``phase`` (0-1) is how far around the curve the tip is; ``trace`` (0-1)
    keeps the path the tip has drawn so far visible."""

    type: ClassVar[str] = "epicycles"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {
        "glowing": {"style": {"glow": "soft"}},
    }

    class Params(ParamsBase):
        points: tuple[Vec3, ...] = Field(min_length=4)  # the closed curve (x, y)
        terms: int = Field(12, ge=1, le=400)
        phase: float = Field(0.0, ge=0, le=1)
        trace: float = Field(1.0, ge=0, le=1)  # share of the path drawn behind the tip

        @field_validator("phase")
        @classmethod
        def _wrap(cls, v):
            return v

    def _co(self, state: SceneState):
        p = self.me(state).params
        return _coeffs(tuple(p.points), p.terms)

    def _chain(self, state: SceneState) -> np.ndarray:
        p = self.me(state).params
        return fm.epicycle_chain(self._co(state), p.phase)

    def world_bounds(self, state: SceneState) -> Bounds:
        co = self._co(state)
        r = sum(abs(c) for k, c in co if k != 0)
        c0 = next((c for k, c in co if k == 0), 0j)
        z = self.me(state).params.points[0][2]
        return Bounds((c0.real - r, c0.imag - r, z), (c0.real + r, c0.imag + r, z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        chain = self._chain(state)
        z = self.me(state).params.points[0][2]
        pts = np.stack([chain.real, chain.imag, np.full(len(chain), z)], axis=1)
        return {"tip": pts[-1], "centers": pts[:-1]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        co = self._co(state)
        target = np.asarray(p.points)
        approx = fm.epicycle_curve(co, len(target))
        err = float(np.abs(approx - (target[:, 0] + 1j * target[:, 1])).max())
        radii = [round(abs(c), 4) for _, c in co[:8]]
        return {
            "kind": "epicycles",
            "terms": len(co),
            "phase": p.phase,
            "max_error": round(err, 6),
            "radii": radii,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        z = p.points[0][2]
        co = self._co(state)
        base = with_alpha(me.style["color"], p.opacity)
        accent = with_alpha(me.style["accent"], p.opacity)
        chain = self._chain(state)

        def xyz(zs):
            return np.stack([zs.real, zs.imag, np.full(len(zs), z)], axis=-1)

        w = me.style["stroke"] * float(cam.units_per_height())
        if p.trace > 0:  # the path drawn so far: the curve the full chain traces up to phase
            n = max(int(160 * p.trace), 2)
            s = np.linspace(max(p.phase - p.trace, 0.0), p.phase, n) if p.phase > 0 else np.zeros(2)
            path = np.array([fm.epicycle_chain(co, v)[-1] for v in s])
            polyline(ctx, cam, xyz(path), w * 3, color=accent, glow=me.style["glow"])
        for (k, c), centre in zip(co, chain[:-1], strict=True):
            if k == 0 or abs(c) * 1.0 < 1e-9:
                continue
            circle(ctx, cam, xyz(np.array([centre]))[0], abs(c), w, color=with_alpha(base, 0.45))
        polyline(ctx, cam, xyz(chain), w * 2, color=base)
        disc(ctx, cam, xyz(chain[-1:])[0], 3.0 * w, color=accent, glow=me.style["glow"])


# ---------------------------------------------------------------- PartialSum


@register
class PartialSum(ObjectBase):
    """A Fourier series partial sum over the wave it converges to, with the Gibbs overshoot
    at the jumps. ``pos`` is the origin of the plot, ``extent`` its (width, height) in world
    units: x spans one period, y spans -1.2 to 1.2."""

    type: ClassVar[str] = "partial_sum"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "target": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        kind: Literal["square", "sawtooth", "triangle"] = "square"
        terms: int = Field(5, ge=1, le=500)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the plot
        extent: tuple[float, float] = (6.0, 2.4)
        reveal: float = Field(1.0, ge=0, le=1)  # share of the period drawn

    def _xs(self, p):
        return np.linspace(-math.pi, math.pi, 600)

    def _curve(self, state: SceneState, p, y):
        xs = self._xs(p)
        px = p.pos[0] + xs / (2 * math.pi) * p.extent[0]
        py = p.pos[1] + y / 1.2 * p.extent[1] / 2
        n = max(int(len(xs) * p.reveal), 2)
        return np.stack([px, py, np.full_like(px, p.pos[2])], axis=1)[:n]

    def path_points(self, state: SceneState) -> np.ndarray:
        """The partial sum's curve (what a ``Tracer`` travels along)."""
        return self.features(state)["curve"]

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x - p.extent[0] / 2, y - p.extent[1] / 2, z),
                      (x + p.extent[0] / 2, y + p.extent[1] / 2, z))  # fmt: skip

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        y = fm.series_partial_sum(p.kind, p.terms, self._xs(p))
        return {"curve": self._curve(state, p, y)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        xs = self._xs(p)
        err = np.abs(fm.series_partial_sum(p.kind, p.terms, xs) - fm.series_target(p.kind, xs))
        away = np.abs(np.sin(xs)) > 0.3  # not next to a jump
        return {"kind": "partial_sum", "series": p.kind, "terms": p.terms,
                "gibbs_overshoot": round(fm.gibbs_overshoot(p.kind, p.terms), 5),
                "max_error_away_from_jumps": round(float(err[away].max()), 5)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        xs = self._xs(p)
        target = fm.series_target(p.kind, xs)
        # the target as separate strokes so the jump is not drawn as a vertical line
        jump = np.abs(np.diff(target)) > 0.5
        pts = self._curve(state, p, target)
        starts = [0, *(np.nonzero(jump)[0] + 1), len(pts)]
        for a, b in itertools.pairwise(starts):
            polyline(
                ctx, cam, pts[a:b], w * 0.6, color=with_alpha(me.style["target"], p.opacity * 0.7)
            )
        y = fm.series_partial_sum(p.kind, p.terms, xs)
        color = with_alpha(me.style["color"], p.opacity)
        polyline(ctx, cam, self._curve(state, p, y), w, color=color, glow=me.style["glow"])


# ---------------------------------------------------------------- Spectrum


@register
class Spectrum(ObjectBase):
    """The magnitude spectrum of a few sinusoids seen through a window: bars per frequency
    bin. Shows leakage (a tone between bins smears) and what a window does about it."""

    type: ClassVar[str] = "spectrum"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        frequencies: tuple[float, ...] = (100.0,)
        amplitudes: tuple[float, ...] = (1.0,)
        window: Literal["rect", "hann", "hamming", "blackman"] = "rect"
        samples: int = Field(256, ge=16, le=8192)
        fs: float = Field(256.0, gt=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # lower-left corner
        extent: tuple[float, float] = (6.0, 2.5)
        bins: int = Field(0, ge=0)  # bins shown from DC (0: up to the highest tone + margin)

        @field_validator("amplitudes")
        @classmethod
        def _same(cls, v, info):
            f = info.data.get("frequencies", ())
            if len(v) != len(f):
                raise ValueError("amplitudes needs one value per frequency")
            return v

    def _spec(self, p):
        sp = fm.spectrum(p.frequencies, p.amplitudes, p.window, p.samples, p.fs)
        shown = p.bins or min(
            len(sp["mag"]), int(max(p.frequencies, default=1) / (p.fs / p.samples)) + 12
        )
        return sp["freq"][:shown], sp["mag"][:shown]

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        y2 = y + _ydir(self.me(state), state) * p.extent[1]
        return Bounds((x, min(y, y2), z), (x + p.extent[0], max(y, y2), z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        _, m = self._spec(p)
        x, y, z = p.pos
        h = p.extent[1] * _ydir(self.me(state), state)
        top = np.max(m) if len(m) else 1.0
        peak = int(np.argmax(m))
        return {
            "peak": np.array([x + (peak + 0.5) / len(m) * p.extent[0], y + m[peak] / top * h, z])
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f, m = self._spec(p)
        leak = fm.leakage(p.frequencies, p.amplitudes, p.window, p.samples, p.fs)
        return {
            "kind": "spectrum",
            "window": p.window,
            "bin_width_hz": round(p.fs / p.samples, 6),
            "peak_hz": round(float(f[int(np.argmax(m))]), 4),
            "leakage": round(leak, 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        _, m = self._spec(p)
        x, y, z = p.pos
        h = p.extent[1] * _ydir(me, state)
        top = max(float(np.max(m)), 1e-12)
        n = len(m)
        base = with_alpha(me.style["color"], p.opacity)
        accent = with_alpha(me.style["accent"], p.opacity)
        w = me.style["stroke"] * float(cam.units_per_height())
        polyline(ctx, cam, [(x, y, z), (x + p.extent[0], y, z)], w, color=with_alpha(base, 0.6))
        df = p.fs / p.samples
        tones = {round(t / df) for t in p.frequencies}
        bar = max(p.extent[0] / n * 0.7, w)
        for k in range(n):
            cx_ = x + (k + 0.5) / n * p.extent[0]
            height = m[k] / top * h
            if abs(height) < 1e-6:
                continue
            polyline(ctx, cam, [(cx_, y, z), (cx_, y + height, z)], bar,
                     color=accent if k in tones else base, glow=me.style["glow"])  # fmt: skip


# ---------------------------------------------------------------- Winding


@functools.lru_cache(maxsize=8)
def _winding_grid(frequencies, amplitudes, samples, fs, limit):
    signal = fm.sinusoids(frequencies, amplitudes, samples, fs)
    # Sample at least eight times per DFT bin so the connecting stroke resolves each peak.
    count = max(128, math.ceil(8 * limit * samples / fs))
    grid = np.linspace(0, limit, count + 1)
    # Keep the temporary frequency-by-time matrix bounded for long signals.
    mean = np.concatenate(
        [fm.winding_mean(signal, block, fs) for block in np.array_split(grid, count // 64 + 1)]
    )
    return signal, grid, np.abs(mean)


@register
class Winding(ObjectBase):
    """A signal wound clockwise, its centroid, and the spectrum up to the test frequency."""

    type: ClassVar[str] = "winding"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "accent": ("emphasis", "highlight"),
        "guide": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "phase"),
    }

    class Params(ParamsBase):
        frequencies: tuple[float, ...] = (3.0, 6.0)
        amplitudes: tuple[float, ...] = (1.0, 0.55)
        fs: float = Field(32.0, gt=0, allow_inf_nan=False)
        samples: int = Field(128, ge=16, le=2048)
        freq: float = Field(0.0, ge=0, allow_inf_nan=False)
        max_freq: float = Field(8.0, gt=0, allow_inf_nan=False)
        progress: float = Field(1.0, ge=0, le=1)
        size: float = Field(2.0, gt=0, allow_inf_nan=False)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the winding diagram
        hue_along: float = Field(0.0, ge=0, allow_inf_nan=False)

        @model_validator(mode="after")
        def _signal(self):
            if not self.frequencies or len(self.frequencies) != len(self.amplitudes):
                raise ValueError("provide one amplitude per tone")
            if any(not 0 < f < self.fs / 2 for f in self.frequencies):
                raise ValueError("tone frequencies must be between DC and Nyquist")
            if not math.isfinite(sum(map(abs, self.amplitudes))):
                raise ValueError("amplitudes must have a finite sum of magnitudes")
            if not self.freq <= self.max_freq <= self.fs / 2:
                raise ValueError("freq must be at most max_freq, which cannot exceed Nyquist")
            return self

    def _data(self, p):
        signal, grid, magnitude = _winding_grid(
            p.frequencies, p.amplitudes, p.samples, p.fs, p.max_freq
        )
        wound = fm.wound_signal(signal, p.freq, p.fs)
        return wound, wound.mean(), grid, magnitude

    def _winding_points(self, p, ydir, values):
        x, y, z = p.pos
        scale = p.size / (sum(map(abs, p.amplitudes)) or 1.0)
        return np.stack(
            np.broadcast_arrays(x + values.real * scale, y + ydir * values.imag * scale, z),
            axis=-1,
        )

    def _spectrum_points(self, p, ydir, freq, magnitude):
        x, y, z = p.pos
        scale = p.size / (sum(map(abs, p.amplitudes)) or 1.0)
        return np.stack(
            np.broadcast_arrays(
                x + p.size * (1.5 + 3 * freq / p.max_freq),
                y + ydir * (-p.size + 2 * magnitude * scale),
                z,
            ),
            axis=-1,
        )

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x - p.size, y - p.size, z), (x + 4.5 * p.size, y + p.size, z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        me = self.me(state)
        p = me.params
        _, mean, _, _ = self._data(p)
        ydir = _ydir(me, state)
        return {
            "centroid": self._winding_points(p, ydir, mean),
            "spectrum_tip": self._spectrum_points(p, ydir, p.freq, abs(mean)),
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        _, mean, _, _ = self._data(p)
        return {
            "kind": "winding",
            "frequency_hz": p.freq,
            "centroid": (float(mean.real), float(mean.imag)),
            "magnitude": float(abs(mean)),
            "bin_width_hz": p.fs / p.samples,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        ydir = _ydir(me, state)
        wound, mean, grid, magnitude = self._data(p)
        style = me.style
        guide = with_alpha(style["guide"], p.opacity)
        color = with_alpha(style["color"], p.opacity)
        accent = with_alpha(style["accent"], p.opacity)
        width = style["stroke"] * float(cam.units_per_height())
        glow = style["glow"]

        circle(ctx, cam, p.pos, p.size, width, color=guide)
        x, y, z = p.pos
        origin = self._spectrum_points(p, ydir, 0.0, 0.0)
        end = self._spectrum_points(p, ydir, p.max_freq, 0.0)
        polyline(ctx, cam, [end, origin, (origin[0], y + ydir * p.size, z)], width, color=guide)
        disc(ctx, cam, (x, y, z), width, color=guide)

        if p.progress > 0:
            points = self._winding_points(p, ydir, wound)
            index = p.progress * (len(points) - 1)
            whole = int(index)
            visible = points[: whole + 1]
            if index > whole:
                endpoint = points[whole] + (index - whole) * (points[whole + 1] - points[whole])
                visible = np.vstack((visible, endpoint))
            colors = None
            if p.hue_along and pass_ == "color":
                length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                colors = hue_colors(visible, style["colormap"], p.hue_along, total=length)
                colors[:, 3] *= p.opacity
            polyline(ctx, cam, visible, width, color=color, colors=colors, glow=glow)

        stop = np.searchsorted(grid, p.freq)
        traced = self._spectrum_points(
            p, ydir, np.r_[grid[:stop], p.freq], np.r_[magnitude[:stop], abs(mean)]
        )
        polyline(ctx, cam, traced, width, color=color, glow=glow)
        disc(ctx, cam, traced[-1], 2 * width, color=color, glow=glow)
        disc(ctx, cam, self._winding_points(p, ydir, mean), 3 * width, color=accent, glow=glow)


# ---------------------------------------------------------------- Spectrogram


@functools.lru_cache(maxsize=8)
def _sg(f0, f1, duration, fs, nwin):
    return fm.spectrogram(fm.chirp(f0, f1, duration, fs), fs, nwin)


@register
class Spectrogram(ObjectBase):
    """A spectrogram of a chirp (a tone sliding from ``f0`` to ``f1`` Hz): time across,
    frequency up, brightness for strength. A backdrop: labels may sit on it."""

    type: ClassVar[str] = "spectrogram"
    Style: ClassVar = {"colormap": ("colormap", "ocean"), "glow": ("glow", "none")}
    Presets: ClassVar = {
        # the prism look: smooth, glowing where strong, contour lines of light
        "prism": {"params": {"smooth": True, "isolines": 8}, "style": {"glow": "soft"}},
    }

    class Params(ParamsBase):
        f0: float = Field(200.0, gt=0)
        f1: float = Field(1200.0, gt=0)
        duration: float = Field(1.0, gt=0)
        fs: float = Field(4000.0, gt=0)
        nwin: int = Field(256, ge=16, le=4096)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        size: tuple[float, float] = (6.0, 3.0)
        smooth: bool = False  # bicubic between grid values (the prism look); else as before
        isolines: int = Field(0, ge=0, le=64)  # contour levels drawn as thin light

    def _rect(self, p):
        c = np.asarray(p.pos, dtype=np.float64) - np.array([p.size[0] / 2, p.size[1] / 2, 0.0])
        return c, np.array([p.size[0], 0.0, 0.0]), np.array([0.0, p.size[1], 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([c, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "spectrogram", "backdrop": True, "f0": p.f0, "f1": p.f1,
                "duration": p.duration, "max_frequency": p.fs / 2}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        sg = _sg(p.f0, p.f1, p.duration, p.fs, p.nwin)
        mag = sg["mag"] / max(float(sg["mag"].max()), 1e-12)
        lut = ramp(me.style["colormap"], np.float32)  # strongest = brightest (ocean is cyclic)
        corner, right, up = self._rect(p)
        glow = float(me.style.get("glow", 1.0))
        paint = lut_colour(lut, "linear" if p.smooth else "nearest")
        pixels = heat_pixels(
            mag.T, p.size, lambda v: paint(np.sqrt(np.clip(v, 0, 1))), cells=True,
            how="smooth" if p.smooth else "nearest", isolines=p.isolines,
            level=lambda v: np.sqrt(np.clip(v, 0, 1)),
            weight=(lambda v: np.sqrt(np.clip(v, 0, 1))) if glow > 1 else None,
            background=state.background,
        )  # fmt: skip
        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity, glow=glow)
