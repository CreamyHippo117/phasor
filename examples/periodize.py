"""Poisson summation in one object: copies, their sum, the pieces folded back, the spectrum.

A bump f(x) = exp(-(x - 1.3)^2 / 9.68) is copied every P = 10. The copies add up to the periodic
function g (bright). Cut f at the cell edges and slide the pieces into one cell (fold = 0) and
they add up to g there too; as `fold` goes 0 -> 1 they slide back out until they draw f exactly.
Below, |fhat| with a stem at every k/P: each stem is P times the size of g's k-th Fourier
coefficient c_k, so the coefficients of the periodized function are samples of f's spectrum.

Run:  uv run python examples/periodize.py      (prints c_k next to fhat(k/P)/P)
"""

from phasor import Scene, calc, expr

s = Scene("periodize", look="calm")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 8)
s.add(calc.Periodize(id="poisson", f="exp(-(x-1.3)^2/9.68)", period=10.0, copies=(-2, 3),
                     fold=expr("min(t / 4, 1)"), harmonics=4, extent=(13.0, 7.2)))  # fmt: skip
s.marker("end", 4.0)

if __name__ == "__main__":
    info = s.engine.inspect(4.0, id="poisson")["objects"][0]["semantic"]
    for k, c, h in zip(info["k"], info["c_k"], info["fhat_over_P"], strict=True):
        print(f"k={k:+d}  c_k={complex(*c):.6f}  fhat(k/P)/P={complex(*h):.6f}")
    print("max mismatch:", info["max_mismatch"])
