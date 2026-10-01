"""Uncertainty estimates. Chosen per metric:

- accuracy (a proportion)        -> Wilson score interval (well-behaved near 0%/100% and for small n)
- difference between two modes   -> PAIRED bootstrap over questions (same questions in both arms) and
                                    McNemar's exact test on the discordant pairs
- median latency                 -> percentile bootstrap
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float], stat: Callable[[np.ndarray], float] = np.median, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(arr[rng.integers(0, arr.size, arr.size)]) for _ in range(n_boot)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def paired_diff(a: Sequence[bool], b: Sequence[bool], n_boot: int = 5000, seed: int = 0) -> dict:
    """Accuracy(b) - accuracy(a) on the SAME items. Returns diff, 95% CI and McNemar exact p."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired_diff needs aligned arrays")
    n = x.size
    if n == 0:
        return {"n": 0, "diff": float("nan"), "ci": (float("nan"), float("nan")), "p_mcnemar": float("nan")}
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    only_a = int(((x == 1) & (y == 0)).sum())  # a right, b wrong
    only_b = int(((x == 0) & (y == 1)).sum())
    return {
        "n": n,
        "diff": float(d.mean()),
        "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "p_mcnemar": mcnemar_exact(only_a, only_b),
        "only_a": only_a,
        "only_b": only_b,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test = binomial test on discordant pairs with p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def significant(result: dict, alpha: float = 0.05) -> bool:
    """Conservative: the CI must exclude 0 AND McNemar must agree."""
    lo, hi = result["ci"]
    return (lo > 0 or hi < 0) and result["p_mcnemar"] < alpha
