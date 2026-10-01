"""Turn a (manually reviewed) raw results CSV into the portfolio summary table.

python -m benchmark.summarize results/raw_results_20260930_101500.csv
python -m benchmark.summarize results/raw_results_*.csv --normalization relative --min-accuracy 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .grading import parse_manual
from .scoring import (
    DEFAULT_LATENCY_BOUNDS,
    DEFAULT_RAM_BOUNDS,
    Bounds,
    overall_score,
    relative_bounds,
    speed_labels,
)


def _final_correct(row: pd.Series) -> int:
    manual = parse_manual(row.get("manual_correct"))
    return int(manual) if manual is not None else int(row["auto_correct"])


def summarize(
    df: pd.DataFrame,
    normalization: str = "absolute",
    latency_bounds: Bounds = DEFAULT_LATENCY_BOUNDS,
    ram_bounds: Bounds = DEFAULT_RAM_BOUNDS,
    ram_source: str = "psutil",
) -> pd.DataFrame:
    df = df.copy()
    df["correct"] = df.apply(_final_correct, axis=1)
    ram_col = "model_ram_gb" if ram_source == "psutil" else "ollama_ps_size_gb"

    g = df.groupby("model", sort=False)
    out = pd.DataFrame(
        {
            "questions": g.size(),
            "accuracy": g["correct"].mean(),
            "page_accuracy": g["page_correct"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean()),
            "latency_median_s": g["latency_s"].median(),
            "latency_p95_s": g["latency_s"].quantile(0.95),
            "tokens_per_s": g["tokens_per_s"].apply(lambda s: pd.to_numeric(s, errors="coerce").median()),
            "ram_gb": g[ram_col].max(),
            "ollama_ps_gb": g["ollama_ps_size_gb"].max(),
            "vram_gb": g["ollama_ps_vram_gb"].max(),
            "json_first_try": g["first_attempt_valid"].mean(),
            "json_after_retry": g["final_valid"].mean(),
            "avg_attempts": g["attempts"].mean(),
        }
    )

    if normalization == "relative":
        latency_bounds = relative_bounds(out["latency_median_s"].tolist())
        ram_bounds = relative_bounds(out["ram_gb"].tolist())
    out["speed_score"] = out["latency_median_s"].map(latency_bounds.score)
    out["ram_score"] = out["ram_gb"].map(ram_bounds.score)
    out["overall"] = [
        overall_score(a, s, r) for a, s, r in zip(out["accuracy"], out["speed_score"], out["ram_score"], strict=True)
    ]
    out["speed_label"] = pd.Series(speed_labels(out["latency_median_s"].to_dict()))
    return out


def recommend(summary: pd.DataFrame, min_accuracy: float) -> tuple[str | None, str]:
    """Accuracy is a gate, not just a weight: a fast model that is wrong 30% of the time is unusable
    for document Q&A. Among models meeting the floor, pick the highest overall score."""
    eligible = summary[summary["accuracy"] >= min_accuracy]
    if eligible.empty:
        return None, f"No model reached the {min_accuracy:.0%} accuracy floor."
    best = eligible["overall"].idxmax()
    return best, (
        f"{best}: highest overall score ({eligible.loc[best, 'overall']:.1f}/10) among models with "
        f"accuracy >= {min_accuracy:.0%}. Top unconstrained score: {summary['overall'].idxmax()}."
    )


def to_markdown(summary: pd.DataFrame) -> str:
    table = pd.DataFrame(
        {
            "Model": summary.index,
            "RAM Usage (Approx.)": [f"~{v:.1f} GB" for v in summary["ram_gb"]],
            "Speed (Response Time)": [
                f"{lbl} (~{s:.1f} sec)"
                for lbl, s in zip(summary["speed_label"], summary["latency_median_s"], strict=True)
            ],
            f"Accuracy ({int(summary['questions'].max())} Qs)": [f"{a:.0%}" for a in summary["accuracy"]],
            "Overall Score": [f"{o:.1f} / 10" for o in summary["overall"]],
        }
    )
    detail = pd.DataFrame(
        {
            "Model": summary.index,
            "p95 latency": [f"{v:.2f} s" for v in summary["latency_p95_s"]],
            "tok/s": [f"{v:.1f}" for v in summary["tokens_per_s"]],
            "Page citation acc.": [f"{v:.0%}" for v in summary["page_accuracy"]],
            "Valid JSON (1st try)": [f"{v:.0%}" for v in summary["json_first_try"]],
            "Valid JSON (after retry)": [f"{v:.0%}" for v in summary["json_after_retry"]],
            "ollama ps size": [f"{v:.1f} GB" for v in summary["ollama_ps_gb"]],
            "VRAM": [f"{v:.1f} GB" for v in summary["vram_gb"]],
        }
    )
    return table.to_markdown(index=False) + "\n\n**Details**\n\n" + detail.to_markdown(index=False)


def summarize_file(
    csv_path: Path, normalization: str = "absolute", min_accuracy: float = 0.8, ram_source: str = "psutil"
) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        return "No results."
    summary = summarize(df, normalization=normalization, ram_source=ram_source)
    best, reason = recommend(summary, min_accuracy)
    md = to_markdown(summary) + f"\n\n**Recommendation:** {reason}\n"
    stem = csv_path.stem.replace("raw_results", "summary")
    summary.to_csv(csv_path.with_name(f"{stem}.csv"))
    csv_path.with_name(f"{stem}.md").write_text(md, encoding="utf-8")
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize benchmark results")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--normalization", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--min-accuracy", type=float, default=0.8)
    ap.add_argument(
        "--ram-source",
        choices=["psutil", "ollama_ps"],
        default="psutil",
        help="psutil = peak RSS delta over idle (CPU/Apple Silicon); ollama_ps = allocated size (GPU)",
    )
    args = ap.parse_args(argv)
    print(summarize_file(args.csv, args.normalization, args.min_accuracy, args.ram_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
