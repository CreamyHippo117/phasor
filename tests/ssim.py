"""Structural similarity for golden-frame tests (mean SSIM over 8x8 windows, luma)."""

from __future__ import annotations

import numpy as np


def ssim(a: np.ndarray, b: np.ndarray, window: int = 8, busy: float | None = None) -> float:
    """``busy``: average only over windows where either picture varies (luma variance above
    ``busy``), so a mostly empty dark frame is judged by its content, not its background.
    When no window is busy (two flat pictures) it is the plain mean over every window."""

    def luma(x):
        x = x.astype(np.float64)
        return x @ [0.2126, 0.7152, 0.0722] if x.ndim == 3 else x

    a, b = luma(a), luma(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    h, w = (s - s % window for s in a.shape)

    def blocks(x):
        return x[:h, :w].reshape(h // window, window, w // window, window).swapaxes(1, 2)

    A, B = blocks(a), blocks(b)
    mu_a, mu_b = A.mean((2, 3)), B.mean((2, 3))
    var_a, var_b = A.var((2, 3)), B.var((2, 3))
    cov = ((A - mu_a[..., None, None]) * (B - mu_b[..., None, None])).mean((2, 3))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    if busy is not None:
        mask = (var_a > busy) | (var_b > busy)
        if mask.any():  # no busy window: both pictures flat, judged over every window
            s = s[mask]
    return float(s.mean())
