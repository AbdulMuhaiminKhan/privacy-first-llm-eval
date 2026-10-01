"""Experiment runner: models x output modes x questions.

Design choices that matter for validity:
- Per model: unload everything, measure idle RAM baseline, warm up (load time excluded from latency).
- Within a model, questions are shuffled and the 3 modes are run in a random order PER QUESTION
  (seeded). Time-dependent drift (thermal throttling, background load) therefore can't line up with a
  mode and masquerade as a mode effect.
- Every result is appended to records.jsonl immediately; `--resume` skips completed cells.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import subprocess
import time
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import psutil

from assistant import DocumentAssistant, OllamaLLM, Settings
from assistant.assistant import OutputMode

from . import records as rec_io
from .grading import grade
from .memory import GB, PeakMemorySampler, wait_for_memory_to_settle

log = logging.getLogger("benchmark")
DEFAULT_MODES = [m.value for m in OutputMode]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_info(llm: OllamaLLM) -> dict:
    try:
        with urllib.request.urlopen(f"{llm.settings.ollama_host.rstrip('/')}/api/version", timeout=5) as r:
            version = json.load(r).get("version")
    except Exception:
        version = None
    vm = psutil.virtual_memory()
    gpu = None
    try:  # NVIDIA only; absent tool -> None (never guessed)
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)  # fmt: skip
        gpu = out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {
        "gpu": gpu,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": platform.processor() or None,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / GB, 1),
        "python": platform.python_version(),
        "ollama_version": version,
    }


REGISTRY = Path("configs/models.json")


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"groups": {}}


def registry_overrides(path: Path = REGISTRY) -> dict[str, dict]:
    """Request overrides declared in the registry (e.g. think=false for reasoning models)."""
    out: dict[str, dict] = {}
    for group in load_registry(path)["groups"].values():
        for m in group["models"]:
            if "think" in m:
                out[m["tag"]] = {"think": m["think"]}
    return out


def run_experiment(
    models: list[str],
    modes: list[str],
    doc: Path,
    questions_path: Path,
    out_root: Path,
    run_id: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    settings: Settings | None = None,
) -> Path:
    settings = settings or Settings()
    llm = OllamaLLM(settings)
    llm.model_overrides.update(registry_overrides())
    assistant = DocumentAssistant(doc, llm=llm, settings=settings)
    questions = rec_io.load_questions(questions_path)[:limit]
    modes = [OutputMode(m).value for m in modes]

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.jsonl"
    done = {(r["model"], r["output_mode"], r["question_id"]) for r in rec_io.load(rec_path)}
    if done:
        log.info("Resuming run %s: %d records already present", run_id, len(done))

    env_path = run_dir / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    env.update(
        run_id=run_id,
        started_at=env.get("started_at", _now()),
        environment=environment_info(llm),
        document=doc.name,
        questions_file=questions_path.name,
        n_questions=len(questions),
        modes=modes,
        seed=seed,
        settings={
            "temperature": settings.temperature,
            "seed": settings.seed,
            "num_ctx": settings.num_ctx,
            "max_answer_tokens": settings.max_answer_tokens,
            "top_k_chunks": settings.top_k_chunks,
            "max_retries": settings.max_retries,
        },
        models=env.get("models", {}),
        model_overrides=llm.model_overrides,
    )

    order = len(done)
    for model in models:
        try:
            info = llm.model_info(model)
        except Exception as exc:  # one missing model must not kill the run
            log.error("Skipping %s: %s", model, exc)
            env.setdefault("skipped_models", {})[model] = str(exc)
            continue
        llm.unload_all()
        baseline = wait_for_memory_to_settle()
        info["load_s"] = round(llm.warm_up(model), 2)
        ps = next((m for m in llm.loaded_models() if m["model"].startswith(model)), None)
        info["ollama_ps_size_gb"] = round(ps["size"] / GB, 2) if ps else None
        info["ollama_ps_vram_gb"] = round(ps["size_vram"] / GB, 2) if ps else None
        info["baseline_rss_gb"] = round(baseline / GB, 2)
        env["models"][model] = info
        env_path.write_text(json.dumps(env, indent=2))
        log.info("%s ready: %s", model, info)

        rng = random.Random(f"{seed}:{model}")
        qs = questions[:]
        rng.shuffle(qs)
        for i, q in enumerate(qs, 1):
            q_modes = modes[:]
            rng.shuffle(q_modes)
            for mode in q_modes:
                if (model, mode, q["id"]) in done:
                    continue
                order += 1
                t0 = time.perf_counter()
                with PeakMemorySampler() as mem:
                    try:
                        nonce = hashlib.sha256(f"{seed}:{model}:{mode}:{q['id']}".encode()).hexdigest()[:8]
                        res = assistant.answer(q["question"], model=model, mode=mode, nonce=nonce)
                        err = res.error
                    except Exception as exc:  # transport failure: record it, never drop it
                        res, err = None, f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - t0) * 1000
                record = _to_record(run_id, model, info, mode, q, res, err, latency_ms, mem.peak_bytes, baseline, order)
                rec_io.append(rec_path, record)
                log.info(
                    "[%s %2d/%d %-6s] %s %.0fms parse=%s retries=%s correct=%s",
                    model, i, len(qs), mode, q["id"], latency_ms, record.parse_success,
                    record.retry_count, record.correct_auto,
                )  # fmt: skip
        llm.unload(model)

    env["finished_at"] = _now()
    env_path.write_text(json.dumps(env, indent=2))
    rec_io.export_grading(rec_io.load(rec_path), run_dir / "grading.csv")
    return run_dir


def _to_record(run_id, model, info, mode, q, res, err, latency_ms, peak, baseline, order) -> rec_io.Record:
    if res is None:
        predicted, conf, page, parse_ok, first_ok, retries, raw, stats = "", None, None, False, False, 0, "", []
    else:
        predicted, conf, page = res.predicted_answer, res.confidence, res.source_page
        parse_ok, first_ok, retries = res.parse_success, res.first_attempt_parse_success, res.retry_count
        raw, stats = res.raw_output, res.stats
    correct = grade(q, predicted, page) if res is not None else False
    exp_page = q.get("expected_page")
    return rec_io.Record(
        run_id=run_id,
        timestamp=_now(),
        model=model,
        model_digest=info.get("digest"),
        output_mode=mode,
        question_id=q["id"],
        answerable=q["answerable"],
        difficulty=q.get("difficulty"),
        question=q["question"],
        expected_answer=q["expected_answer"],
        expected_page=exp_page,
        predicted_answer=predicted,
        predicted_page=page,
        raw_output=raw[:2000],
        correct_auto=correct,
        correct_strict=correct and parse_ok,
        page_correct=(page == exp_page) if (page is not None and exp_page is not None) else None,
        confidence=conf,
        parse_success=parse_ok,
        first_attempt_parse_success=first_ok,
        retry_count=retries,
        latency_ms=round(latency_ms, 1),
        first_call_latency_ms=round(stats[0].wall_s * 1000, 1) if stats else None,
        input_tokens=sum(s.prompt_tokens for s in stats) if stats else None,
        output_tokens=sum(s.output_tokens for s in stats) if stats else None,
        tokens_per_s=round(stats[-1].tokens_per_s, 1) if stats and stats[-1].eval_s > 0 else None,
        answer_chars=len(predicted or ""),
        peak_rss_gb=round(peak / GB, 3) if peak else None,
        model_ram_gb=round(max(peak - baseline, 0) / GB, 3) if peak else None,
        order_index=order,
        error=err,
        prompt_cache_bust=True,
    )


def settings_with(settings: Settings, **kw) -> Settings:
    return replace(settings, **kw)
