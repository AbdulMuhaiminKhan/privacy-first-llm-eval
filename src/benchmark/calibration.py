"""Does the model's self-reported `confidence` mean anything?

Metrics:
- reliability table: observed accuracy per confidence bin (fixed edges; small LLMs cluster at 0.8-1.0,
  so bins are finer at the top)
- ECE: expected calibration error = sum_b (n_b/N) * |acc_b - conf_b|   (0 = perfect)
- Brier score: mean (conf - correct)^2                                  (lower is better; 0.25 = coin flip at 0.5)
- overconfidence gap: mean confidence - accuracy                        (>0 overconfident)
- AUROC: P(conf of a random correct answer > conf of a random wrong one); 0.5 = confidence carries no signal
- confidence collapse: share of answers using the single most common confidence value
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .stats import bootstrap_ci

BIN_EDGES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0000001]


def reliability_table(conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:], strict=True):
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        rows.append({
            "bin": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "lo": lo,
            "hi": min(hi, 1.0),
            "n": n,
            "mean_confidence": float(conf[mask].mean()) if n else None,
            "accuracy": float(correct[mask].mean()) if n else None,
        })  # fmt: skip
    return rows


def ece(conf: np.ndarray, correct: np.ndarray) -> float:
    total = conf.size
    if total == 0:
        return float("nan")
    err = 0.0
    for row in reliability_table(conf, correct):
        if row["n"]:
            err += row["n"] / total * abs(row["accuracy"] - row["mean_confidence"])
    return float(err)


def auroc(conf: np.ndarray, correct: np.ndarray) -> float | None:
    """Mann-Whitney formulation; ties count half. None when only one class is present."""
    pos, neg = conf[correct == 1], conf[correct == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def calibration_summary(conf_values, correct_values, seed: int = 0) -> dict:
    conf = np.asarray([c for c in conf_values], dtype=float)
    correct = np.asarray(correct_values, dtype=float)
    keep = ~np.isnan(conf)
    conf, correct = conf[keep], correct[keep]
    n = int(conf.size)
    if n == 0:
        return {"n": 0}
    pairs = np.column_stack([conf, correct])

    def _ece(sample: np.ndarray) -> float:
        return ece(sample[:, 0], sample[:, 1])

    rng = np.random.default_rng(seed)
    boots = [_ece(pairs[rng.integers(0, n, n)]) for _ in range(2000)]
    common_value, common_count = Counter(np.round(conf, 3)).most_common(1)[0]
    return {
        "n": n,
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
        "overconfidence": float(conf.mean() - correct.mean()),
        "ece": ece(conf, correct),
        "ece_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "brier": float(np.mean((conf - correct) ** 2)),
        "auroc": auroc(conf, correct),
        "most_common_confidence": float(common_value),
        "most_common_share": common_count / n,
        "high_conf_n": int((conf >= 0.9).sum()),
        "high_conf_accuracy": float(correct[conf >= 0.9].mean()) if (conf >= 0.9).any() else None,
        "reliability": reliability_table(conf, correct),
    }


# re-exported for reports
__all__ = ["BIN_EDGES", "auroc", "bootstrap_ci", "calibration_summary", "ece", "reliability_table"]
