"""Run the standardized question set against several local models.

    python -m benchmark.run_benchmark                                  # phi3, mistral, gemma2:9b
    python -m benchmark.run_benchmark --models phi3 mistral --limit 5  # quick smoke run

Per model: unload everything -> measure idle baseline RAM -> warm-up (load time recorded separately)
-> ask every question through the structured (Pydantic + retry) path while sampling RAM -> unload.
Rows are appended to the CSV as they complete, so a crash never loses finished work.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
import urllib.request
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import psutil

from assistant import DocumentAssistant, OllamaLLM, Settings, StructuredOutputError

from .grading import keyword_grade
from .memory import GB, PeakMemorySampler, wait_for_memory_to_settle
from .summarize import summarize_file

ROOT = Path.cwd()  # run from the repo root
DEFAULT_MODELS = ["phi3", "mistral", "gemma2:9b"]
FIELDS = [
    "model",
    "qid",
    "difficulty",
    "question",
    "expected_answer",
    "expected_page",
    "answer",
    "source_page",
    "confidence",
    "auto_correct",
    "page_correct",
    "manual_correct",
    "latency_s",
    "first_call_s",
    "attempts",
    "first_attempt_valid",
    "final_valid",
    "output_tokens",
    "tokens_per_s",
    "peak_rss_gb",
    "baseline_rss_gb",
    "model_ram_gb",
    "ollama_ps_size_gb",
    "ollama_ps_vram_gb",
    "load_s",
    "error",
]
log = logging.getLogger("benchmark")


def load_questions(path: Path) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids")
    return qs


def environment_info(llm: OllamaLLM) -> dict:
    try:
        with urllib.request.urlopen(f"{llm.settings.ollama_host.rstrip('/')}/api/version", timeout=5) as r:
            version = json.load(r).get("version", "unknown")
    except Exception:
        version = "unknown"
    vm = psutil.virtual_memory()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / GB, 1),
        "python": platform.python_version(),
        "ollama_version": version,
    }


def run_model(
    model: str, assistant: DocumentAssistant, llm: OllamaLLM, questions: list[dict], writer: csv.DictWriter, fh
) -> None:
    llm.ensure_model(model)
    llm.unload_all()
    baseline = wait_for_memory_to_settle()
    load_s = llm.warm_up(model)
    ps = next((m for m in llm.loaded_models() if m["model"].startswith(model)), {"size": 0, "size_vram": 0})
    log.info(
        "%s loaded in %.1fs | ollama ps size=%.2f GB (vram %.2f GB) | baseline RSS %.2f GB",
        model,
        load_s,
        ps["size"] / GB,
        ps["size_vram"] / GB,
        baseline / GB,
    )

    for i, q in enumerate(questions, 1):
        row = {k: "" for k in FIELDS}
        row.update(
            model=model,
            qid=q["id"],
            difficulty=q.get("difficulty", ""),
            question=q["question"],
            expected_answer=q["expected_answer"],
            expected_page=q.get("expected_page", ""),
            baseline_rss_gb=round(baseline / GB, 3),
            ollama_ps_size_gb=round(ps["size"] / GB, 3),
            ollama_ps_vram_gb=round(ps["size_vram"] / GB, 3),
            load_s=round(load_s, 2),
        )
        t0 = time.perf_counter()
        with PeakMemorySampler() as mem:
            try:
                res = assistant.ask_structured(q["question"], model=model)
                row.update(
                    answer=res.answer.answer,
                    source_page=res.answer.source_page,
                    confidence=res.answer.confidence,
                    attempts=res.attempts,
                    first_attempt_valid=int(res.first_attempt_valid),
                    final_valid=1,
                    first_call_s=round(res.stats[0].wall_s, 3),
                    output_tokens=sum(s.output_tokens for s in res.stats),
                    tokens_per_s=round(res.stats[-1].tokens_per_s, 1),
                )
            except StructuredOutputError as exc:
                row.update(
                    attempts=exc.attempts,
                    first_attempt_valid=0,
                    final_valid=0,
                    first_call_s=round(exc.stats[0].wall_s, 3) if exc.stats else "",
                    error=str(exc),
                    answer=(exc.raw_outputs[-1] if exc.raw_outputs else "")[:500],
                )
        row["latency_s"] = round(time.perf_counter() - t0, 3)
        row["peak_rss_gb"] = round(mem.peak_bytes / GB, 3)
        row["model_ram_gb"] = round(max(mem.peak_bytes - baseline, 0) / GB, 3)
        row["auto_correct"] = int(bool(row["final_valid"]) and keyword_grade(row["answer"], q["expected_keywords"]))
        if row["source_page"] != "" and q.get("expected_page"):
            row["page_correct"] = int(int(row["source_page"]) == int(q["expected_page"]))
        writer.writerow(row)
        fh.flush()
        log.info(
            "[%s %2d/%d] %s %.2fs attempts=%s auto=%s",
            model,
            i,
            len(questions),
            q["id"],
            row["latency_s"],
            row["attempts"],
            row["auto_correct"],
        )

    llm.unload(model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--doc", type=Path, default=ROOT / "data" / "sample" / "handbook.pdf")
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "questions.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    ap.add_argument("--format-mode", choices=["json", "schema"], default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    settings = Settings()
    if args.format_mode:
        settings = replace(settings, format_mode=args.format_mode)
    llm = OllamaLLM(settings)
    assistant = DocumentAssistant(args.doc, llm=llm, settings=settings)
    questions = load_questions(args.questions)[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"raw_results_{stamp}.csv"
    env = environment_info(llm) | {
        "models": args.models,
        "questions": len(questions),
        "document": args.doc.name,
        "format_mode": settings.format_mode,
        "num_ctx": settings.num_ctx,
        "temperature": settings.temperature,
    }
    (args.out_dir / f"environment_{stamp}.json").write_text(json.dumps(env, indent=2))
    log.info("Environment: %s", env)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for model in args.models:
            try:
                run_model(model, assistant, llm, questions, writer, fh)
            except Exception as exc:  # one broken model must not kill the whole run
                log.error("Model %s failed: %s", model, exc)

    log.info("Raw results: %s", csv_path)
    print(summarize_file(csv_path))
    print(
        f"\nNext: review {csv_path.name}, fill `manual_correct` (1/0) where the auto grade is wrong, "
        f"then run:  python -m benchmark.summarize {csv_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
