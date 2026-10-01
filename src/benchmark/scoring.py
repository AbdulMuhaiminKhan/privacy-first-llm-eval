"""Overall score = 10 * (0.6 * accuracy + 0.2 * speed_score + 0.2 * ram_score), all inputs in [0, 1].

Normalisation (how seconds / GB become a 0-1 score) is the part people get wrong:

* absolute (default): fixed bounds, e.g. latency 1 s -> 1.0 and 8 s -> 0.0. Scores are comparable
  across runs and machines, and adding a 4th model doesn't change the first three's scores.
* relative: min-max across the models in this run. Best model gets 1.0, worst gets 0.0. Exaggerates
  small differences and makes the worst model score 0 even if it's perfectly usable.
"""

from __future__ import annotations

from dataclasses import dataclass

WEIGHTS = {"accuracy": 0.6, "speed": 0.2, "ram": 0.2}


@dataclass(frozen=True)
class Bounds:
    best: float
    worst: float

    def score(self, value: float) -> float:
        """Linear 1.0 at `best`, 0.0 at `worst`, clamped. Lower raw value is better."""
        if self.worst == self.best:
            return 1.0
        return max(0.0, min(1.0, (self.worst - value) / (self.worst - self.best)))


DEFAULT_LATENCY_BOUNDS = Bounds(best=1.0, worst=8.0)  # seconds; ~8 s is where a chat UX feels broken
DEFAULT_RAM_BOUNDS = Bounds(best=2.0, worst=10.0)  # GB; 10 GB is the ceiling on a 16 GB laptop


def overall_score(accuracy: float, speed_score: float, ram_score: float) -> float:
    for name, v in (("accuracy", accuracy), ("speed_score", speed_score), ("ram_score", ram_score)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    raw = WEIGHTS["accuracy"] * accuracy + WEIGHTS["speed"] * speed_score + WEIGHTS["ram"] * ram_score
    return round(10 * raw, 2)


def relative_bounds(values: list[float]) -> Bounds:
    return Bounds(best=min(values), worst=max(values))


def speed_labels(latencies: dict[str, float]) -> dict[str, str]:
    """Rank-based labels: fastest = Fast, slowest = Slow, the rest Medium."""
    ordered = sorted(latencies, key=latencies.get)
    labels = {m: "Medium" for m in ordered}
    if ordered:
        labels[ordered[0]] = "Fast"
    if len(ordered) > 1:
        labels[ordered[-1]] = "Slow"
    return labels
