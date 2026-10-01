"""analyze: records -> analysis.json + report.md + charts.   publish: run -> docs/ + README block."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from . import charts
from . import records as rec_io
from .analysis import analyze
from .grading import grade

README = Path("README.md")
DOCS = Path("docs")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
SIM_BANNER = (
    "> ⚠️ **SIMULATED DATA.** This run used the test server in `tests/fake_ollama.py`. The numbers are "
    "meaningless and exist only to exercise the pipeline."
)


def _json_default(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


def _clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def regrade(records: list[dict], env: dict) -> list[dict]:
    """Re-apply the CURRENT auto-grader, so grader fixes apply to old runs too. The count of changed grades
    is reported in the meta block."""
    qpath = Path("data") / env.get("questions_file", "questions.jsonl")
    if not qpath.exists():
        return records
    qs = {q["id"]: q for q in rec_io.load_questions(qpath)}
    out = []
    for r in records:
        q = qs.get(r["question_id"])
        r = dict(r)
        if q is not None:
            new = grade(q, r.get("predicted_answer") or "", r.get("predicted_page"))
            r["regraded"] = new != bool(r["correct_auto"])
            r["correct_auto"] = new
        out.append(r)
    return out


def analyze_run(run_dir: Path, make_charts: bool = True) -> dict:
    env = json.loads((run_dir / "environment.json").read_text())
    raw = rec_io.load(run_dir / "records.jsonl")
    if not raw:
        raise SystemExit(f"{run_dir}/records.jsonl is empty")
    raw = regrade(raw, env)
    grades = rec_io.load_manual_grades(run_dir / "grading.csv")
    graded = rec_io.apply_manual_grades(raw, grades)
    meta = {
        "run_id": env.get("run_id"),
        "simulated": (env.get("environment", {}).get("ollama_version") == "SIMULATED"),
        "environment": env.get("environment"),
        "settings": env.get("settings"),
        "model_info": env.get("models"),
        "document": env.get("document"),
        "manual_grades": sum(v is not None for v in grades.values()),
        "manual_overrides": sum(
            1 for r in graded if r["manual_correct"] is not None and r["manual_correct"] != bool(r["correct_auto"])
        ),  # fmt: skip
        "regraded": sum(1 for r in raw if r.get("regraded")),
        "started_at": env.get("started_at"),
        "finished_at": env.get("finished_at"),
    }
    a = _clean(analyze(graded, meta))
    out_json = run_dir / "analysis.json"
    out_json.write_text(json.dumps(a, indent=2, default=_json_default), encoding="utf-8", newline="\n")
    made = charts.make_all(a, run_dir / "charts") if make_charts else {}
    report = run_dir / "report.md"
    report.write_text(
        render_report(a, {k: f"charts/{p.name}" for k, p in made.items()}), encoding="utf-8", newline="\n"
    )
    return {"analysis": a, "analysis_json": out_json, "report_md": report, "charts": made}


# ---- markdown -----------------------------------------------------------------------------------------
def _pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def _ci(ci) -> str:
    return "–" if not ci or ci[0] is None else f"[{ci[0]:.0%}, {ci[1]:.0%}]"


def matrix_table(a: dict) -> str:
    rows = ["| Model | Mode | Accuracy (95% CI) | End-to-end acc. | Valid 1st try | Retry rate | Median latency | "
            "Hallucination* |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for c in a["cells"]:
        rows.append(
            f"| {c['model']} | `{c['mode']}` | **{_pct(c['accuracy'])}** {_ci(c['accuracy_ci'])} | "
            f"{_pct(c['strict_accuracy'])} | {_pct(c['parse_first_attempt'])} | {_pct(c['retry_rate'])} | "
            f"{c['latency_median_ms'] / 1000:.2f} s | {_pct(c['hallucination_rate'])} |"
        )
    rows.append("\n\\* share of unanswerable questions answered anyway (lower is better).")
    return "\n".join(rows)


def effects_table(a: dict) -> str:
    rows = [
        "| Scope | Comparison | Metric | Δ accuracy | 95% CI | McNemar p | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in a["mode_effects"]:
        verdict = "**difference**" if e["significant"] else "no detectable difference"
        rows.append(
            f"| {e['scope']} | {e['comparison']} | {e['metric'].replace('_', ' ')} | {e['diff'] * 100:+.1f} pp | "
            f"[{e['ci'][0] * 100:+.1f}, {e['ci'][1] * 100:+.1f}] | {e['p_mcnemar']:.3f} | {verdict} |"
        )
    return "\n".join(rows)


def calibration_table(a: dict) -> str:
    rows = ["| Slice | n | Mean confidence | Accuracy | Gap | ECE (95% CI) | Brier | AUROC | Most common value |",
            "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for key, c in a["calibration"].items():
        if not c.get("n"):
            continue
        au = "–" if c["auroc"] is None else f"{c['auroc']:.2f}"
        rows.append(
            f"| {key} | {c['n']} | {c['mean_confidence']:.0%} | {c['accuracy']:.0%} | "
            f"{c['overconfidence'] * 100:+.0f} pp | {c['ece']:.3f} [{c['ece_ci'][0]:.3f}, {c['ece_ci'][1]:.3f}] | "
            f"{c['brier']:.3f} | {au} | {c['most_common_confidence']:g} ({c['most_common_share']:.0%}) |"
        )
    return "\n".join(rows)


def hcw_table(a: dict, limit: int = 10) -> str:
    if not a["high_conf_wrong"]:
        return "_None._"
    rows = ["| Model | Mode | Question | Expected | Model said | Confidence |", "|---|---|---|---|---|---|"]
    for r in a["high_conf_wrong"][:limit]:
        said = (r["predicted_answer"] or "").replace("|", "/").replace("\n", " ")[:90]
        rows.append(f"| {r['model']} | `{r['output_mode']}` | {r['question']} | {r['expected_answer']} | {said} | "
                    f"{r['confidence']:.2f} |")  # fmt: skip
    return "\n".join(rows)


def scores_table(a: dict) -> str:
    if not a["overall_scores"]:
        return ""
    rows = [
        "| Model | RAM Usage (Approx.) | Speed (Response Time) | Accuracy | Overall Score |",
        "|---|---|---|---|---|",
    ]
    for s in a["overall_scores"]:
        ram = "–" if s["ram_gb"] is None else f"~{s['ram_gb']:.1f} GB"
        ov = "–" if s["overall"] is None else f"{s['overall']:.1f} / 10"
        rows.append(f"| {s['model']} | {ram} | {s['speed_label']} (~{s['latency_s']:.1f} s) | {_pct(s['accuracy'])} | "
                    f"{ov} |")  # fmt: skip
    src = {c["model"]: c.get("ram_source") for c in a["cells"]}
    if "ollama_ps" in src.values():
        rows.append("\nMemory: `ollama ps` allocated size for models running on the GPU (process RSS reads ~0 there).")
    return "\n".join(rows)


def models_table(a: dict) -> str:
    info = a["meta"].get("model_info") or {}
    rows = ["| Model tag | Parameters | Quantization | Size on disk | Digest |", "|---|---|---|---|---|"]
    for tag, m in info.items():
        rows.append(f"| `{m.get('tag', tag)}` | {m.get('parameter_size') or '–'} | {m.get('quantization') or '–'} | "
                    f"{m.get('size_gb', '–')} GB | `{(m.get('digest') or '')[:12]}` |")  # fmt: skip
    return "\n".join(rows)


def render_report(a: dict, chart_paths: dict[str, str]) -> str:
    env = a["meta"].get("environment") or {}
    parts = [f"# Experiment report: run `{a['meta']['run_id']}`", ""]
    if a["meta"]["simulated"]:
        parts += [SIM_BANNER, ""]
    parts += ["## Findings", "", *[f"- {f}" for f in a["findings"]], ""]
    if "mode_effect" in chart_paths:
        parts += [f"![Mode effect]({chart_paths['mode_effect']})", ""]
    parts += ["## Results matrix", "", matrix_table(a), ""]
    for key in ("accuracy", "reliability", "latency", "tradeoff"):
        if key in chart_paths:
            parts += [f"![{key}]({chart_paths[key]})", ""]
    parts += ["## Structured output vs plain text (paired tests)", "", effects_table(a), ""]
    parts += ["## Confidence calibration", "", calibration_table(a), ""]
    if "calibration" in chart_paths:
        parts += [f"![Calibration]({chart_paths['calibration']})", ""]
    parts += ["### Confidently wrong (confidence ≥ 0.9, incorrect)", "", hcw_table(a), ""]
    parts += ["## Original portfolio table (JSON mode)", "", scores_table(a), "",
              "Overall = 10 × (0.6 acc + 0.2 speed + 0.2 RAM); bounds in README.", ""]  # fmt: skip
    parts += ["## Models", "", models_table(a), ""]
    parts += [
        "## Environment", "",
        f"- OS: {env.get('os')} · CPU: {env.get('cpu')} · GPU: {env.get('gpu') or 'none detected'} · "
        f"cores: {env.get('physical_cores')}/{env.get('logical_cores')}"
        f" · RAM: {env.get('total_ram_gb')} GB · Ollama: {env.get('ollama_version')}",
        f"- Settings: {json.dumps(a['meta'].get('settings'))}",
        f"- Manual grades applied: {a['meta']['manual_grades']} (overrides of the auto-grade: "
        f"{a['meta']['manual_overrides']}); auto-grades changed by the current grader: {a['meta'].get('regraded', 0)}",
        f"- Run: {a['meta'].get('started_at')} → {a['meta'].get('finished_at')}",
    ]  # fmt: skip
    return "\n".join(parts) + "\n"


# ---- publish ------------------------------------------------------------------------------------------
def readme_block(a: dict) -> str:
    lines = []
    if a["meta"]["simulated"]:
        lines += [SIM_BANNER, ""]
    env = a["meta"].get("environment") or {}
    lines += [
        f"_Run `{a['meta']['run_id']}` · {len(a['models'])} models × {len(a['modes'])} output modes × "
        f"{a['n_questions']} questions · {env.get('os')}, {env.get('total_ram_gb')} GB RAM, "
        f"Ollama {env.get('ollama_version')} · full report: [`docs/REPORT.md`](docs/REPORT.md)_",
        "",
        *[f"- {f}" for f in a["findings"]],
        "",
        "![Does forcing structured output change accuracy?](docs/assets/mode_effect.png)",
        "",
        matrix_table(a),
        "",
        "<p><img src=\"docs/assets/calibration.png\" width=\"49%\"> "
        "<img src=\"docs/assets/tradeoff.png\" width=\"49%\"></p>",
    ]  # fmt: skip
    return "\n".join(lines)


def update_readme(block: str, readme: Path = README) -> None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README is missing the {START} / {END} markers")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)
    readme.write_text(new, encoding="utf-8", newline="\n")


def publish_run(run_dir: Path, allow_simulated: bool = False, docs: Path = DOCS, readme: Path = README) -> None:
    out = analyze_run(run_dir)
    a = out["analysis"]
    if a["meta"]["simulated"] and not allow_simulated:
        raise SystemExit("Refusing to publish a SIMULATED run. Run the benchmark against real models first.")
    assets = docs / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for p in out["charts"].values():
        shutil.copy2(p, assets / p.name)
    (docs / "data").mkdir(parents=True, exist_ok=True)
    payload = json.dumps(a, default=_json_default, ensure_ascii=False)
    (docs / "data" / "results.js").write_text(
        f"window.BENCHMARK_RESULTS = {payload};\n", encoding="utf-8", newline="\n"
    )
    report = render_report(a, {k: f"assets/{p.name}" for k, p in out["charts"].items()})
    (docs / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    update_readme(readme_block(a), readme)
