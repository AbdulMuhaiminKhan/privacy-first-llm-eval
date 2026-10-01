"""Static charts (PNG) for the README and report. Every chart answers one question.

Colour = output mode, everywhere (text / json / schema use categorical slots 1-3 of a palette validated
for colour-vision deficiency). Models are always on an axis or in a panel, never a colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MODE_COLORS = {"text": "#2a78d6", "json": "#eb6834", "schema": "#1baf7a"}
MODE_LABELS = {"text": "Plain text", "json": "JSON mode", "schema": "Schema-constrained"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 150, "axes.axisbelow": True,
    })  # fmt: skip


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _grouped(ax, models, modes, values, errs=None, fmt="{:.0%}"):
    width = 0.8 / len(modes)
    x = np.arange(len(models))
    for j, mode in enumerate(modes):
        v = np.array([values[(m, mode)] for m in models], dtype=float)
        pos = x - 0.4 + width * (j + 0.5)
        yerr = None
        if errs is not None:
            lo = np.array([errs[(m, mode)][0] for m in models])
            hi = np.array([errs[(m, mode)][1] for m in models])
            yerr = np.vstack([v - lo, hi - v])
        ax.bar(pos, v, width * 0.92, color=MODE_COLORS.get(mode), label=MODE_LABELS.get(mode, mode),
               yerr=yerr, error_kw={"elinewidth": 1, "ecolor": INK_2, "capsize": 2})  # fmt: skip
        for p, val in zip(pos, v, strict=True):
            ax.text(p, 0.01 if np.isnan(val) else val / 2, "" if np.isnan(val) else fmt.format(val),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")  # fmt: skip
    ax.set_xticks(x, models)
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=len(modes), loc="upper left", bbox_to_anchor=(0, -0.08))


def chart_mode_effect(a: dict, path: Path) -> Path | None:
    """HEADLINE: how much does each structured mode change accuracy vs plain text (paired, with 95% CI)?"""
    eff = [e for e in a["mode_effects"] if e["metric"] == "accuracy"]
    if not eff:
        return None
    _style()
    scopes = ["ALL", *a["models"]]
    modes = [m for m in a["modes"] if m != "text"]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.55 * len(scopes) * len(modes)))
    y, labels, ys = 0.0, [], []
    for scope in scopes:
        for mode in modes:
            e = next((e for e in eff if e["scope"] == scope and e["mode"] == mode), None)
            if e is None:
                continue
            lo, hi = e["ci"]
            c = MODE_COLORS[mode]
            ax.plot([lo * 100, hi * 100], [y, y], color=c, lw=2, solid_capstyle="round")
            ax.plot(e["diff"] * 100, y, "o", ms=8, color=c, mec=SURFACE, mew=2)
            ax.text(max(hi * 100, e["diff"] * 100) + 1, y, f"{e['diff'] * 100:+.1f} pp" +
                    ("  *" if e["significant"] else ""), va="center", fontsize=8, color=INK_2)  # fmt: skip
            labels.append(f"{'All models' if scope == 'ALL' else scope} · {MODE_LABELS[mode]}")
            ys.append(y)
            y += 1
        y += 0.4
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ task accuracy vs plain text (percentage points, paired 95% CI)")
    ax.set_title("Does forcing structured output change accuracy?")
    fig.text(0.01, -0.02, "* = CI excludes 0 and McNemar p < 0.05", fontsize=8, color=INK_2)
    return _save(fig, path)


def chart_accuracy(a: dict, path: Path) -> Path:
    _style()
    cells = {(c["model"], c["mode"]): c for c in a["cells"]}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _grouped(ax, a["models"], a["modes"], {k: v["accuracy"] for k, v in cells.items()},
             {k: v["accuracy_ci"] for k, v in cells.items()})  # fmt: skip
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Task accuracy (Wilson 95% CI)")
    ax.set_title("Accuracy by model and output mode")
    return _save(fig, path)


def chart_latency(a: dict, path: Path) -> Path:
    _style()
    cells = {(c["model"], c["mode"]): c for c in a["cells"]}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _grouped(ax, a["models"], a["modes"], {k: v["latency_median_ms"] / 1000 for k, v in cells.items()},
             {k: (v["latency_median_ci"][0] / 1000, v["latency_median_ci"][1] / 1000) for k, v in cells.items()},
             fmt="{:.1f}s")  # fmt: skip
    ax.set_ylabel("Median response time, s (incl. retries; bootstrap 95% CI)")
    ax.set_title("What does structure cost in latency?")
    return _save(fig, path)


def chart_reliability(a: dict, path: Path) -> Path:
    _style()
    cells = {(c["model"], c["mode"]): c for c in a["cells"]}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _grouped(ax, a["models"], a["modes"], {k: v["parse_first_attempt"] for k, v in cells.items()})
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Valid on first attempt (same Pydantic contract)")
    ax.set_title("How often is the raw output machine-readable?")
    return _save(fig, path)


def chart_calibration(a: dict, path: Path) -> Path | None:
    """Reliability diagram per mode (pooled over models) + confidence histogram.

    Markers only (no connecting lines): real small-model confidence is often bimodal (0.0 / 1.0), and lines
    through 1-answer bins draw shapes that are not in the data. Bins with fewer than 5 answers are hollow.
    """
    cal = a["calibration"]
    modes = [m for m in a["modes"] if cal.get(f"mode:{m}", {}).get("n")]
    if not modes:
        return None
    _style()
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.5, 6.8), height_ratios=[3, 1.2], sharex=True)
    ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1, label="Perfect calibration")
    offsets = np.linspace(-0.03, 0.03, len(modes))
    for off, mode in zip(offsets, modes, strict=True):
        c = MODE_COLORS[mode]
        rows = [r for r in cal[f"mode:{mode}"]["reliability"] if r["n"]]
        for r in rows:
            solid = r["n"] >= 5
            ax.scatter(r["mean_confidence"] + off, r["accuracy"], s=30 + 4 * min(r["n"], 150),
                       color=c if solid else SURFACE, edgecolor=c, linewidth=2, zorder=3)  # fmt: skip
            if not solid:
                ax.annotate(f"n={r['n']}", (r["mean_confidence"] + off, r["accuracy"]), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=7, color=INK_2)  # fmt: skip
        ax.scatter([], [], s=60, color=c, label=f"{MODE_LABELS[mode]} (ECE {cal[f'mode:{mode}']['ece']:.2f})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Does stated confidence match reality?")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncols=2, fontsize=8)
    bins = np.linspace(0, 1, 21)
    for mode in modes:
        confs = [r["confidence"] for r in a["records"] if r["output_mode"] == mode and r["confidence"] is not None]
        axh.hist(confs, bins=bins, histtype="step", lw=2, color=MODE_COLORS[mode])
    axh.set_xlabel("Stated confidence (marker area = answers; hollow = fewer than 5)")
    axh.set_ylabel("Answers")
    return _save(fig, path)


def chart_tradeoff(a: dict, path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c in a["cells"]:
        ax.scatter(c["latency_median_ms"] / 1000, c["strict_accuracy"], s=80, color=MODE_COLORS.get(c["mode"]),
                   edgecolor=SURFACE, linewidth=2, zorder=3)  # fmt: skip
        ax.annotate(c["model"], (c["latency_median_ms"] / 1000, c["strict_accuracy"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip
    for mode in a["modes"]:
        ax.scatter([], [], s=80, color=MODE_COLORS.get(mode), label=MODE_LABELS.get(mode, mode))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Median response time (s)")
    ax.set_ylabel("End-to-end accuracy (correct AND valid format)")
    ax.set_title("Accuracy-latency trade-off (up and left is better)")
    ax.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, -0.14))
    return _save(fig, path)


def make_all(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = {
        "mode_effect": chart_mode_effect(a, out_dir / "mode_effect.png"),
        "accuracy": chart_accuracy(a, out_dir / "accuracy_by_mode.png"),
        "latency": chart_latency(a, out_dir / "latency_by_mode.png"),
        "reliability": chart_reliability(a, out_dir / "format_reliability.png"),
        "calibration": chart_calibration(a, out_dir / "calibration.png"),
        "tradeoff": chart_tradeoff(a, out_dir / "tradeoff.png"),
    }
    return {k: v for k, v in made.items() if v is not None}
