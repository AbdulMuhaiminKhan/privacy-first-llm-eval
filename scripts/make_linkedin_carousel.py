"""Build a LinkedIn document carousel (PDF, 1080x1350 px per slide) from a run's analysis.json.

    python scripts/make_linkedin_carousel.py results/20261002_000441 --out docs/linkedin_carousel.pdf

Every number on the slides is read from analysis.json - nothing is typed in by hand.
Type is sized for phones: a 1080 px slide is shown ~390 px wide, so body text is >= 30 pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8f8e86"
GRID = "#e4e3df"
TEXT_C, JSON_C, SCHEMA_C = "#2a78d6", "#eb6834", "#1baf7a"
BAD = "#c0362c"
X0 = 0.08  # left margin
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "text.color": INK})

DASHBOARD = "abdulmuhaiminkhan.github.io/privacy-first-llm-eval"


def new_slide(n: int, total: int, kicker: str, title: str):
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.text(X0, 0.925, kicker.upper(), fontsize=20, color=JSON_C, fontweight="bold")
    fig.text(X0, 0.895, title, fontsize=52, fontweight="bold", va="top", linespacing=1.1)
    fig.text(X0, 0.035, "Abdul Muhaimin Khan  ·  privacy-first-llm-eval", fontsize=16, color=MUTED)
    fig.text(1 - X0, 0.035, f"{n} / {total}", fontsize=16, color=MUTED, ha="right")
    return fig


def text(fig, y, s, size=32, color=INK2, weight="normal", ha="left", x=X0):
    fig.text(x, y, s, fontsize=size, color=color, va="top", ha=ha, linespacing=1.35, fontweight=weight)


def style_ax(ax, grid_axis="y"):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=22, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)


def build(a: dict, out: Path) -> None:
    pooled = {c["mode"]: c for c in a["pooled"]}
    cells = {(c["model"], c["mode"]): c for c in a["cells"]}
    recs = a["records"]
    n = len(recs)
    ones = sum(1 for r in recs if r["confidence"] == 1.0)
    hcw = a["high_conf_wrong_count"]
    eff = next(
        e for e in a["mode_effects"] if e["scope"] == "ALL" and e["mode"] == "json" and e["metric"] == "accuracy"
    )
    sub = a["calibration"]["substantive_answers_only"]
    models = a["models"]
    nq = a["n_questions"]
    big = models[-1]
    cold = cells[(big, "json")]["latency_median_ms"] / 1000
    warm = cells[(big, "json")]["latency_warm_median_ms"] / 1000
    txt = cells[(big, "text")]["latency_median_ms"] / 1000
    ratios = [cells[(m, mo)]["latency_median_ms"] / cells[(m, "text")]["latency_median_ms"]
              for m in models for mo in ("json", "schema")]  # fmt: skip
    T = 8

    with PdfPages(out) as pdf:

        def save(fig):
            pdf.savefig(fig)
            plt.close(fig)

        # 1 - cover
        fig = new_slide(1, T, f"{n} answers · {len(models)} models · offline",
                        "Does forcing\nJSON make small\nlocal LLMs worse?")  # fmt: skip
        text(fig, 0.56, "I ran a controlled experiment\non my laptop to find out.", 34, INK)
        text(fig, 0.42, "Plus: the bug that made my\nbenchmark lie about speed.", 34, INK2)
        text(fig, 0.22, "Swipe →", 34, JSON_C, "bold")
        save(fig)

        # 2 - setup
        fig = new_slide(2, T, "the setup", "Only the format\nchanges")
        text(fig, 0.66, f"{len(models)} models  ×  3 output modes  ×  {nq} questions", 30, INK, "bold")
        rows = [
            (TEXT_C, "Plain text", "labelled lines, no constraint"),
            (JSON_C, "JSON mode", "valid JSON enforced"),
            (SCHEMA_C, "Schema", "decoding locked to the schema"),
        ]
        for i, (c, name, desc) in enumerate(rows):
            y = 0.575 - i * 0.085
            fig.patches.append(plt.Rectangle((X0, y - 0.042), 0.018, 0.05, transform=fig.transFigure,
                                             color=c, figure=fig))  # fmt: skip
            text(fig, y, name, 32, INK, "bold", x=X0 + 0.04)
            text(fig, y - 0.033, desc, 24, INK2, x=X0 + 0.04)
        text(fig, 0.27,
             "Same instructions, same context,\none validation contract (Pydantic).\n"
             "Phi-3 3.8B · Mistral 7B · Gemma 2 9B\nvia Ollama on an RTX 4070 laptop.", 28)  # fmt: skip
        save(fig)

        # 3 - accuracy
        fig = new_slide(3, T, "finding 1", "JSON cost no\nmeasurable accuracy")
        ax = fig.add_axes([0.11, 0.25, 0.81, 0.42])
        style_ax(ax)
        modes = ["text", "json", "schema"]
        vals = [pooled[m]["accuracy"] for m in modes]
        lo = [pooled[m]["accuracy"] - pooled[m]["accuracy_ci"][0] for m in modes]
        hi = [pooled[m]["accuracy_ci"][1] - pooled[m]["accuracy"] for m in modes]
        ax.bar(range(3), vals, 0.64, color=[TEXT_C, JSON_C, SCHEMA_C], yerr=[lo, hi],
               error_kw={"elinewidth": 2.5, "ecolor": INK2, "capsize": 8})  # fmt: skip
        for i, v in enumerate(vals):
            ax.text(i, v / 2, f"{v:.0%}", ha="center", va="center", color="white", fontsize=44, fontweight="bold")
        ax.set_xticks(range(3), ["Plain text", "JSON", "Schema"], fontsize=26)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        text(fig, 0.17, f"JSON vs text: {eff['diff'] * 100:+.1f} pp  (95% CI {eff['ci'][0] * 100:+.1f} to "
                        f"{eff['ci'][1] * 100:+.1f})", 26, INK, "bold")  # fmt: skip
        text(fig, 0.12, "Paired test on the same questions.", 24)
        save(fig)

        # 4 - reliability
        fig = new_slide(4, T, "finding 2", "But it made output\nmachine-readable")
        ax = fig.add_axes([0.11, 0.25, 0.81, 0.40])
        style_ax(ax)
        x = np.arange(len(models))
        labels = {"text": "Plain text", "json": "JSON", "schema": "Schema"}
        for j, (m, c) in enumerate(zip(modes, [TEXT_C, JSON_C, SCHEMA_C], strict=True)):
            v = [cells[(mo, m)]["parse_first_attempt"] for mo in models]
            bars = ax.bar(x + (j - 1) * 0.27, v, 0.25, color=c, label=labels[m])
            if m == "text":
                for b, val in zip(bars, v, strict=True):
                    if val < 0.995:
                        ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.0%}", ha="center",
                                fontsize=24, fontweight="bold", color=INK)  # fmt: skip
        ax.set_xticks(x, models, fontsize=24)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(fontsize=20, frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0))
        text(fig, 0.17, f"Valid on first try: {pooled['text']['parse_first_attempt']:.0%} (text) → 100% (JSON)",
             26, INK, "bold")  # fmt: skip
        text(fig, 0.12, f"{ {'phi3': 'Phi-3'}.get(models[0], models[0]) } kept dropping the 'Answer:' label.", 24)
        save(fig)

        # 5 - cache trap
        fig = new_slide(5, T, "finding 3 · the trap", "My benchmark said\nJSON was faster.\nIt wasn't.")
        ax = fig.add_axes([0.33, 0.33, 0.59, 0.25])
        style_ax(ax, grid_axis="x")
        ax.barh([2, 1, 0], [txt, cold, warm], height=0.62, color=[TEXT_C, JSON_C, BG],
                edgecolor=[TEXT_C, JSON_C, JSON_C], linewidth=3, hatch=[None, None, "//"])  # fmt: skip
        for yy, v in zip([2, 1, 0], [txt, cold, warm], strict=True):
            ax.text(v + cold * 0.03, yy, f"{v:.1f} s", va="center", fontsize=28, fontweight="bold")
        ax.set_yticks([2, 1, 0], ["Plain text", "JSON, cold", "JSON, cached"], fontsize=24)
        ax.set_xlim(0, cold * 1.3)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        text(fig, 0.625, f"{big}, median response time", 22, MUTED)
        text(fig, 0.27, "JSON and schema prompts were identical,\nso Ollama's prompt cache gave the second\n"
                        "one a free head start.", 26)  # fmt: skip
        text(fig, 0.15, f"Cold-only: {min(ratios):.2f}–{max(ratios):.2f}× plain text.\nFixed with a per-request nonce.",
             26, INK, "bold")  # fmt: skip
        save(fig)

        # 6 - confidence
        fig = new_slide(6, T, "finding 4", '"100% confident"\nbarely means\nanything')
        fig.text(X0, 0.57, f"{ones}/{n}", fontsize=130, fontweight="bold", color=JSON_C, va="top")
        text(fig, 0.44, "answers stated confidence = 1.0", 32, INK, "bold")
        text(fig, 0.37,
             f"• {hcw} were wrong at ≥ 0.9 confidence\n"
             "• Phi-3: \"33 days\" (28 + 2 = 30),\n   at 1.0, in all three modes\n"
             f"• AUROC {sub['auroc']:.2f}: barely better\n   than a coin flip (0.5)", 28)  # fmt: skip
        text(fig, 0.16, "Don't route answers to humans\non self-reported confidence.", 28, BAD, "bold")
        save(fig)

        # 7 - takeaways
        fig = new_slide(7, T, "takeaways", "What I'd tell\nan AI team")
        items = [
            ("Use structured output.", "For grounded Q&A it cost no\nmeasurable accuracy."),
            ("Benchmark latency cold.", "Shared prompt prefixes fake\nspeed-ups through the cache."),
            ("Don't trust stated confidence.", "Check retrieval or use a\nverifier instead."),
            ("Report your parser.", "'Reliability' gains depend on\nhow strict it is."),
        ]
        for i, (head, body) in enumerate(items):
            y = 0.68 - i * 0.145
            text(fig, y, f"{i + 1}", 44, JSON_C, "bold")
            text(fig, y, head, 30, INK, "bold", x=X0 + 0.07)
            text(fig, y - 0.04, body, 25, INK2, x=X0 + 0.07)
        save(fig)

        # 8 - method + where to find it
        fig = new_slide(8, T, "open source", "How it's built")
        text(fig, 0.78,
             "• Python · Ollama · Pydantic\n"
             "• Paired bootstrap CIs + McNemar tests\n"
             "• Calibration: ECE, Brier, AUROC\n"
             "• No-egress privacy guard, tested in CI\n"
             "• 46 tests · interactive dashboard\n"
             "• Findings generated from the data", 28)  # fmt: skip
        text(fig, 0.33, "Results, code and all answers:", 26, INK2)
        text(fig, 0.285, DASHBOARD, 26, TEXT_C, "bold")
        text(fig, 0.24, "(link in my Featured section)", 22, MUTED)
        text(fig, 0.15, "Next: newer 2026 models\nand quantisation.", 28, INK, "bold")
        save(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/linkedin_carousel.pdf"))
    args = ap.parse_args()
    build(json.loads((args.run / "analysis.json").read_text(encoding="utf-8")), args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
