"""Turn graded records into metrics, uncertainty estimates and data-driven findings.

Every sentence in `findings` is generated from the numbers with fixed rules: a difference is called a
difference only if its paired 95% CI excludes zero AND McNemar's exact test agrees (p < 0.05);
otherwise it is reported as "no detectable difference at this sample size".
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from assistant.textparse import parse_labelled

from .calibration import calibration_summary
from .grading import is_refusal
from .scoring import DEFAULT_LATENCY_BOUNDS, DEFAULT_RAM_BOUNDS, overall_score, speed_labels
from .stats import bootstrap_ci, paired_diff, significant, wilson_ci

MODE_ORDER = ["text", "json", "schema"]
BASELINE_MODE = "text"
HIGH_CONF = 0.9


def _ordered_modes(df: pd.DataFrame) -> list[str]:
    present = list(df["output_mode"].unique())
    return [m for m in MODE_ORDER if m in present] + [m for m in present if m not in MODE_ORDER]


def _cell_metrics(g: pd.DataFrame) -> dict:
    n = len(g)
    k = int(g["correct"].sum())
    ks = int(g["correct_strict"].sum())
    ans = g[g["answerable"]]
    unans = g[~g["answerable"]]
    cold = g[g["prompt_cache_cold"]]
    lat = (cold if len(cold) else g)["latency_ms"].to_numpy(dtype=float)
    warm = g[~g["prompt_cache_cold"]]["latency_ms"].to_numpy(dtype=float)
    conf = g["confidence"].dropna()
    return {
        "n": n,
        "accuracy": k / n,
        "accuracy_ci": wilson_ci(k, n),
        "strict_accuracy": ks / n,
        "strict_accuracy_ci": wilson_ci(ks, n),
        "answerable_accuracy": float(ans["correct"].mean()) if len(ans) else None,
        "unanswerable_n": int(len(unans)),
        "hallucination_rate": float(1 - unans["correct"].mean()) if len(unans) else None,
        "parse_first_attempt": float(g["first_attempt_parse_success"].mean()),
        "parse_final": float(g["parse_success"].mean()),
        "retry_rate": float((g["retry_count"] > 0).mean()),
        "mean_retries": float(g["retry_count"].mean()),
        "latency_median_ms": float(np.median(lat)),
        "latency_median_ci": bootstrap_ci(lat),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_cold_n": int(len(cold)),
        "latency_warm_median_ms": float(np.median(warm)) if warm.size else None,
        "parse_lenient": float(g["parse_success_lenient"].mean()),
        "strict_accuracy_lenient": float(g["correct_strict_lenient"].mean()),
        "output_tokens_median": _median_or_none(g["output_tokens"]),
        "answer_chars_median": _median_or_none(g["answer_chars"]),
        "page_accuracy": _mean_or_none(g["page_correct"]),
        "confidence_coverage": len(conf) / n,
        "errors": int(g["error"].notna().sum()),
    }


def _memory(info: dict, g: pd.DataFrame) -> dict:
    """On a GPU the weights live in VRAM and process RSS reads ~0, so use Ollama's allocated size there."""
    vram = info.get("ollama_ps_vram_gb") or 0
    if vram > 0 and info.get("ollama_ps_size_gb"):
        return {"ram_gb": float(info["ollama_ps_size_gb"]), "ram_source": "ollama_ps", "vram_gb": float(vram)}
    return {"ram_gb": _max_or_none(g["model_ram_gb"]), "ram_source": "psutil_rss_delta", "vram_gb": 0.0}


def _median_or_none(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def _max_or_none(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.max()) if len(s) else None


def _mean_or_none(s: pd.Series):
    s = s.dropna().astype(float)
    return float(s.mean()) if len(s) else None


def _paired(df: pd.DataFrame, mode_a: str, mode_b: str, col: str) -> dict:
    a = df[df.output_mode == mode_a].set_index(["model", "question_id"])[col]
    b = df[df.output_mode == mode_b].set_index(["model", "question_id"])[col]
    common = a.index.intersection(b.index)
    res = paired_diff(a.loc[common].astype(bool).tolist(), b.loc[common].astype(bool).tolist())
    res["significant"] = significant(res) if res["n"] else False
    return res


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """prompt_cache_cold: Ollama reuses the processed prompt prefix between calls. json and schema prompts
    are identical, so whichever runs second for a question gets a warm cache (found in run 20261002_000441:
    Gemma 2 json 6.2 s cold vs 2.2 s warm). Latency is therefore compared on cold calls only, unless the
    run used a per-request cache-busting nonce.

    parse_success_lenient: text-mode validity if an unlabelled answer line is accepted (sensitivity analysis
    for parser strictness; json/schema are unchanged)."""
    df = df.copy()
    busted = df.get("prompt_cache_bust", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    cold = pd.Series(True, index=df.index)
    if "order_index" not in df:
        df["order_index"] = range(len(df))
    order = df.set_index(["model", "question_id", "output_mode"])["order_index"].to_dict()
    for i, r in df.iterrows():
        twin = {"json": "schema", "schema": "json"}.get(r["output_mode"])
        if twin and not busted[i]:
            t = order.get((r["model"], r["question_id"], twin))
            cold[i] = t is None or t > r["order_index"]
    df["prompt_cache_cold"] = cold
    lenient = df["parse_success"].astype(bool).copy()
    for i, r in df[df["output_mode"] == "text"].iterrows():
        lenient[i] = parse_labelled(r.get("raw_output") or "", lenient=True).valid
    df["parse_success_lenient"] = lenient
    df["correct_strict_lenient"] = df["correct"].astype(bool) & lenient
    return df


def analyze(records: list[dict], meta: dict) -> dict:
    df = pd.DataFrame(records)
    df["answerable"] = df["answerable"].astype(bool)
    df = add_derived_columns(df)
    modes = _ordered_modes(df)
    models = list(dict.fromkeys(df["model"]))

    info = meta.get("model_info") or {}
    cells = [
        {"model": model, "mode": mode, **_cell_metrics(g), **_memory(info.get(model) or {}, g)}
        for model in models
        for mode in modes
        if len(g := df[(df.model == model) & (df.output_mode == mode)])
    ]
    pooled_cells = [{"model": "ALL", "mode": m, **_cell_metrics(df[df.output_mode == m])} for m in modes]

    effects = []
    for other in [m for m in modes if m != BASELINE_MODE]:
        if BASELINE_MODE not in modes:
            break
        for scope in ["ALL", *models]:
            sub = df if scope == "ALL" else df[df.model == scope]
            for metric, col in (("accuracy", "correct"), ("strict_accuracy", "correct_strict"),
                                ("strict_accuracy_lenient_parser", "correct_strict_lenient")):  # fmt: skip
                r = _paired(sub, BASELINE_MODE, other, col)
                effects.append({"scope": scope, "comparison": f"{other} vs {BASELINE_MODE}", "mode": other,
                                "metric": metric, **r})  # fmt: skip

    calib = {"pooled": calibration_summary(df["confidence"], df["correct"])}
    # Refusals ("Not found") are ambiguous for confidence: the pilot showed some models report 0.0 there
    # (confidence that an answer exists?) rather than confidence that the refusal is correct.
    substantive = df[~df["predicted_answer"].fillna("").map(is_refusal)]
    calib["substantive_answers_only"] = calibration_summary(substantive["confidence"], substantive["correct"])
    for mode in modes:
        sub = df[df.output_mode == mode]
        calib[f"mode:{mode}"] = calibration_summary(sub["confidence"], sub["correct"])
    for model in models:
        sub = df[df.model == model]
        calib[f"model:{model}"] = calibration_summary(sub["confidence"], sub["correct"])

    hcw = df[(df["confidence"] >= HIGH_CONF) & (~df["correct"])].sort_values("confidence", ascending=False)
    high_conf_wrong = hcw[
        ["model", "output_mode", "question_id", "question", "expected_answer", "predicted_answer", "confidence"]
    ].to_dict("records")

    analysis = {
        "meta": {**meta, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "models": models,
        "modes": modes,
        "n_questions": int(df["question_id"].nunique()),
        "cells": cells,
        "pooled": pooled_cells,
        "mode_effects": effects,
        "calibration": calib,
        "high_conf_wrong": high_conf_wrong,
        "high_conf_wrong_count": len(high_conf_wrong),
        "overall_scores": _overall_scores(cells, prefer_mode="json" if "json" in modes else modes[0]),
        "records": _slim(df),
    }
    analysis["findings"] = findings(analysis)
    return analysis


def _overall_scores(cells: list[dict], prefer_mode: str) -> list[dict]:
    """The original portfolio table: overall = 10 x (0.6 acc + 0.2 speed + 0.2 RAM), for one mode."""
    rows = [c for c in cells if c["mode"] == prefer_mode]
    if not rows:
        return []
    labels = speed_labels({c["model"]: c["latency_median_ms"] for c in rows})
    out = []
    for c in rows:
        speed = DEFAULT_LATENCY_BOUNDS.score(c["latency_median_ms"] / 1000)
        ram = DEFAULT_RAM_BOUNDS.score(c["ram_gb"]) if c["ram_gb"] is not None else None
        out.append({
            "model": c["model"], "mode": prefer_mode, "ram_gb": c["ram_gb"],
            "latency_s": c["latency_median_ms"] / 1000, "speed_label": labels[c["model"]],
            "accuracy": c["accuracy"],
            "overall": overall_score(c["accuracy"], speed, ram) if ram is not None else None,
        })  # fmt: skip
    return out


def _slim(df: pd.DataFrame) -> list[dict]:
    cols = ["model", "output_mode", "question_id", "answerable", "question", "expected_answer",
            "predicted_answer", "correct", "correct_strict", "confidence", "latency_ms", "retry_count",
            "parse_success", "first_attempt_parse_success", "prompt_cache_cold"]  # fmt: skip
    out = df[cols].copy()
    out["latency_ms"] = out["latency_ms"].round(0)
    return out.replace({np.nan: None}).to_dict("records")


# ---- findings ---------------------------------------------------------------------------------------
def _pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def _effect_sentence(e: dict) -> str:
    lo, hi = e["ci"]
    stats = f"{_pp(e['diff'])}, 95% CI [{_pp(lo)}, {_pp(hi)}], McNemar p = {e['p_mcnemar']:.3f}, n = {e['n']} pairs"
    if e["significant"]:
        direction = "higher" if e["diff"] > 0 else "lower"
        return f"`{e['mode']}` mode {e['metric'].replace('_', ' ')} is **{direction}** than plain text ({stats})."
    return (
        f"`{e['mode']}` vs plain text {e['metric'].replace('_', ' ')}: **no detectable difference** at this "
        f"sample size ({stats})."
    )


def findings(a: dict) -> list[str]:
    out: list[str] = []
    pooled = {c["mode"]: c for c in a["pooled"]}
    eff = [e for e in a["mode_effects"] if e["scope"] == "ALL"]

    for e in eff:
        if e["metric"] == "accuracy":
            out.append("**Task accuracy (all models pooled):** " + _effect_sentence(e))
    for e in eff:
        if e["metric"] == "strict_accuracy":
            out.append("**End-to-end accuracy (wrong format counts as failure):** " + _effect_sentence(e))

    parse = ", ".join(
        f"`{m}` {pooled[m]['parse_first_attempt']:.0%} first try → {pooled[m]['parse_final']:.0%} final"
        for m in a["modes"]
    )
    out.append(f"**Format reliability (valid output under the same Pydantic contract):** {parse}.")

    if "text" in pooled:
        cells = {(c["model"], c["mode"]): c for c in a["cells"]}
        parts = []
        for model in a["models"]:
            base = cells.get((model, "text"))
            if not base:
                continue
            ratios = ", ".join(
                f"`{m}` {cells[(model, m)]['latency_median_ms'] / base['latency_median_ms']:.2f}×"
                for m in a["modes"]
                if m != "text" and (model, m) in cells
            )
            parts.append(f"{model} {ratios}")
        out.append(
            "**Latency vs plain text (median, cold prompt cache only):** " + "; ".join(parts) + ". Warm-cache "
            "calls are excluded: json and schema prompts are identical, so the second one reuses Ollama's prompt "
            "cache and looks up to 2.8× faster than it is."
        )

    if "text" in pooled:
        len_eff = [e for e in eff if e["metric"] == "strict_accuracy_lenient_parser"]
        detail = "; ".join(
            f"`{e['mode']}` vs text {_pp(e['diff'])} [{_pp(e['ci'][0])}, {_pp(e['ci'][1])}]"
            + (" (difference)" if e["significant"] else " (no detectable difference)")
            for e in len_eff
        )
        out.append(
            f"**Parser-strictness check:** accepting an unlabelled answer line raises plain-text validity from "
            f"{pooled['text']['parse_first_attempt']:.0%} to {pooled['text']['parse_lenient']:.0%}; end-to-end "
            f"accuracy then: {detail}."
        )

    cal = a["calibration"]["pooled"]
    if cal.get("n"):
        au = cal["auroc"]
        signal = (
            "carries little signal about correctness" if au is not None and au < 0.65
            else "separates right from wrong answers to some degree" if au is not None
            else "cannot be assessed (only one outcome class)"
        )  # fmt: skip
        out.append(
            f"**Confidence calibration (n = {cal['n']}):** mean stated confidence {cal['mean_confidence']:.0%} vs "
            f"actual accuracy {cal['accuracy']:.0%} (gap {_pp(cal['overconfidence'])}); ECE {cal['ece']:.3f} "
            f"[{cal['ece_ci'][0]:.3f}, {cal['ece_ci'][1]:.3f}]; AUROC {au if au is None else round(au, 2)} - "
            f"stated confidence {signal}."
        )
        sub = a["calibration"].get("substantive_answers_only", {})
        if sub.get("n") and sub["n"] < cal["n"]:
            sau = "–" if sub["auroc"] is None else f"{sub['auroc']:.2f}"
            out.append(
                f"**Calibration excluding refusals (n = {sub['n']}):** confidence {sub['mean_confidence']:.0%} vs "
                f"accuracy {sub['accuracy']:.0%}; ECE {sub['ece']:.3f}; AUROC {sau}."
            )
        if cal["most_common_share"] >= 0.5:
            out.append(
                f"**Confidence collapse:** {cal['most_common_share']:.0%} of answers report the same confidence "
                f"value ({cal['most_common_confidence']:g})."
            )
        if cal.get("high_conf_accuracy") is not None:
            out.append(
                f"**High-confidence answers (≥ {HIGH_CONF}):** {cal['high_conf_n']} answers, "
                f"{cal['high_conf_accuracy']:.0%} correct; {a['high_conf_wrong_count']} were confidently wrong."
            )

    hall = [c for c in a["pooled"] if c["hallucination_rate"] is not None]
    if hall:
        txt = ", ".join(f"`{c['mode']}` {c['hallucination_rate']:.0%}" for c in hall)
        out.append(
            f"**Hallucination on unanswerable questions (n = {hall[0]['unanswerable_n']} per mode, "
            f"models pooled):** {txt}."
        )

    n = a["n_questions"]
    half = np.mean([(c["accuracy_ci"][1] - c["accuracy_ci"][0]) / 2 for c in a["cells"]])
    out.append(
        f"**Sample-size caveat:** {n} questions per model and mode; per-cell accuracy 95% CIs are about "
        f"±{half * 100:.0f} pp wide. Mode comparisons are paired (same questions), which is more sensitive, "
        f"but gaps whose CI includes zero are unresolved, not evidence of equality."
    )
    return out
