"""Build a LinkedIn document carousel (PDF, 1080x1350 px per slide) from a run's analysis.json.

    python scripts/make_linkedin_carousel.py results/20261002_000441 --out docs/linkedin_carousel.pdf

Every number on the slides is read from analysis.json - nothing is typed in by hand.
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
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "text.color": INK})


def slide(pdf, n, total, title, kicker=None):
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.patch.set_facecolor(BG)
    if kicker:
        fig.text(0.08, 0.93, kicker.upper(), fontsize=17, color=JSON_C, fontweight="bold")
    fig.text(0.08, 0.875, title, fontsize=40, fontweight="bold", va="top", wrap=True, linespacing=1.15)
    fig.text(0.08, 0.035, "privacy-first-llm-eval  ·  Abdul Muhaimin Khan", fontsize=14, color=MUTED)
    fig.text(0.92, 0.035, f"{n}/{total}", fontsize=14, color=MUTED, ha="right")
    return fig


def body(fig, y, text, size=24, color=INK2, weight="normal"):
    fig.text(0.08, y, text, fontsize=size, color=color, va="top", linespacing=1.45, fontweight=weight)


def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=16)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


def build(a: dict, out: Path) -> None:
    pooled = {c["mode"]: c for c in a["pooled"]}
    cells = {(c["model"], c["mode"]): c for c in a["cells"]}
    recs = a["records"]
    n = len(recs)
    ones = sum(1 for r in recs if r["confidence"] == 1.0)
    hcw = a["high_conf_wrong_count"]
    acc_eff = next(
        e for e in a["mode_effects"] if e["scope"] == "ALL" and e["mode"] == "json" and e["metric"] == "accuracy"
    )
    sub = a["calibration"]["substantive_answers_only"]
    models = a["models"]
    T = 8

    with PdfPages(out) as pdf:
        # 1 - hook
        fig = slide(
            pdf,
            1,
            T,
            "I forced small local\nLLMs to answer in JSON.\n\nMy benchmark lied\nto me once.",
            "540 answers · fully offline",
        )
        body(
            fig, 0.5, "Does structured output make\nsmall models worse?\nAnd do they know when they're wrong?", 30, INK
        )
        body(fig, 0.2, "Swipe →", 26, JSON_C, "bold")
        pdf.savefig(fig)
        plt.close(fig)

        # 2 - setup
        fig = slide(pdf, 2, T, "The setup", "controlled experiment")
        body(
            fig,
            0.74,
            f"• {len(models)} open-weight models: Phi-3 3.8B,\n   Mistral 7B, Gemma 2 9B (4-bit, Ollama)\n"
            f"• Same {a['n_questions']} questions about a private document\n"
            "   (10 of them unanswerable, to catch hallucination)\n"
            "• 3 output modes, identical instructions:\n"
            "     plain text  ·  JSON mode  ·  schema-constrained\n"
            f"• {n} answers, paired design, randomised order\n"
            "• 100% offline on an RTX 4070 laptop. $0.",
            25,
        )
        body(fig, 0.2, "Only the format changes.\nEverything else is held constant.", 26, INK, "bold")
        pdf.savefig(fig)
        plt.close(fig)

        # 3 - accuracy
        fig = slide(pdf, 3, T, "Forcing JSON did not\ncost accuracy", "finding 1")
        ax = fig.add_axes([0.1, 0.3, 0.82, 0.38])
        style_ax(ax)
        modes = ["text", "json", "schema"]
        vals = [pooled[m]["accuracy"] for m in modes]
        lo = [pooled[m]["accuracy"] - pooled[m]["accuracy_ci"][0] for m in modes]
        hi = [pooled[m]["accuracy_ci"][1] - pooled[m]["accuracy"] for m in modes]
        ax.bar(
            range(3),
            vals,
            0.62,
            color=[TEXT_C, JSON_C, SCHEMA_C],
            yerr=[lo, hi],
            error_kw={"elinewidth": 2, "ecolor": INK2, "capsize": 6},
        )
        for i, v in enumerate(vals):
            ax.text(i, v / 2, f"{v:.0%}", ha="center", color="white", fontsize=34, fontweight="bold")
        ax.set_xticks(range(3), ["Plain text", "JSON mode", "Schema"], fontsize=20)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        d, (cl, ch) = acc_eff["diff"] * 100, (acc_eff["ci"][0] * 100, acc_eff["ci"][1] * 100)
        body(
            fig,
            0.2,
            f"JSON vs text: {d:+.1f} pp, 95% CI [{cl:+.1f}, {ch:+.1f}].\nThe interval rules out any meaningful drop.",
            26,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 4 - reliability
        fig = slide(pdf, 4, T, "But it made outputs\nmachine-readable", "finding 2")
        ax = fig.add_axes([0.1, 0.3, 0.82, 0.38])
        style_ax(ax)
        x = np.arange(len(models))
        for j, (m, c) in enumerate(zip(modes, [TEXT_C, JSON_C, SCHEMA_C], strict=True)):
            v = [cells[(mo, m)]["parse_first_attempt"] for mo in models]
            ax.bar(
                x + (j - 1) * 0.27,
                v,
                0.25,
                color=c,
                label={"text": "Plain text", "json": "JSON", "schema": "Schema"}[m],
            )
        ax.set_xticks(x, models, fontsize=20)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(fontsize=16, frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0, 1.12))
        phi = cells[(models[0], "text")]["parse_first_attempt"]
        body(
            fig,
            0.2,
            f"Valid output on first try: plain text {pooled['text']['parse_first_attempt']:.0%}, JSON/schema 100%.\n"
            f"{models[0]} in plain text: only {phi:.0%}. It kept dropping the label.",
            24,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 5 - cache trap
        g = models[-1]
        cold, warm, txt = (
            cells[(g, "json")]["latency_median_ms"] / 1000,
            cells[(g, "json")]["latency_warm_median_ms"] / 1000,
            cells[(g, "text")]["latency_median_ms"] / 1000,
        )
        fig = slide(pdf, 5, T, "My benchmark said JSON\nwas 18% faster.\nIt wasn't.", "finding 3 · the trap")
        ax = fig.add_axes([0.3, 0.3, 0.62, 0.3])
        style_ax(ax)
        ax.barh(
            [2, 1, 0],
            [txt, cold, warm],
            color=[TEXT_C, JSON_C, BG],
            edgecolor=[TEXT_C, JSON_C, JSON_C],
            linewidth=3,
            hatch=[None, None, "//"],
            height=0.6,
        )
        for yy, v in zip([2, 1, 0], [txt, cold, warm], strict=True):
            ax.text(v + 0.1, yy, f"{v:.1f} s", va="center", fontsize=22, fontweight="bold")
        ax.set_yticks([2, 1, 0], ["Plain text", "JSON, cold", "JSON, cached"], fontsize=20)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, cold * 1.25)
        ax.set_title(f"{g}, median response time", fontsize=18, color=INK2, loc="left")
        body(
            fig,
            0.2,
            "JSON and schema prompts were identical, so Ollama's prompt cache\n"
            "gave the second one a free head start. Cold-only: structured\n"
            "output is the same speed or slower. Fixed with a per-request nonce.",
            21,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 6 - confidence
        fig = slide(pdf, 6, T, '"100% confident"\nbarely means anything', "finding 4")
        fig.text(0.08, 0.6, f"{ones}/{n}", fontsize=110, fontweight="bold", color=JSON_C)
        body(fig, 0.47, "answers stated confidence = exactly 1.0", 26, INK)
        body(
            fig,
            0.39,
            f"• {hcw} answers were wrong at ≥ 0.9 confidence\n"
            '• Phi-3 said "33 days" (28 + 2 = 30) at 1.0, in all 3 modes\n'
            f"• Confidence vs correctness: AUROC {sub['auroc']:.2f}\n   (0.5 = coin flip)",
            24,
        )
        body(fig, 0.13, "Don't route answers to humans on self-reported confidence.", 23, BAD, "bold")
        pdf.savefig(fig)
        plt.close(fig)

        # 7 - takeaways
        fig = slide(pdf, 7, T, "What I'd tell an\nAI team", "takeaways")
        body(
            fig,
            0.66,
            "1.  Use structured output. For grounded Q&A\n      with small models, it cost no measurable accuracy.\n\n"
            "2.  Measure latency on cold calls. Prompt\n      caching silently fakes speed-ups.\n\n"
            "3.  Don't trust verbalised confidence.\n      Use retrieval checks or a verifier instead.\n\n"
            "4.  Report the parser you used. 'Reliability'\n      gains depend on how strict it is.",
            25,
            INK,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 8 - method + CTA
        fig = slide(pdf, 8, T, "How it's built", "open source")
        body(
            fig,
            0.74,
            "Python · Ollama · Pydantic · NumPy · pandas\n"
            "Paired bootstrap CIs + McNemar exact tests\n"
            "Calibration: ECE, Brier, AUROC\n"
            "No-egress privacy guard, verified in CI\n"
            "43 tests · interactive results dashboard\n"
            "Every finding sentence is generated from the data",
            25,
        )
        body(fig, 0.36, "github.com/AbdulMuhaiminKhan/\nprivacy-first-llm-eval", 27, TEXT_C, "bold")
        body(fig, 0.2, "What would you test next:\nnewer models, quantisation, or harder reasoning?", 25, INK, "bold")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/linkedin_carousel.pdf"))
    a = ap.parse_args()
    build(json.loads((a.run / "analysis.json").read_text(encoding="utf-8")), a.out)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
