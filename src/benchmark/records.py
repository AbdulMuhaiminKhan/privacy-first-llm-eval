"""Result records: one JSON line per (model, output_mode, question). Missing values are null."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .grading import parse_manual


@dataclass
class Record:
    run_id: str
    timestamp: str
    model: str
    model_digest: str | None
    output_mode: str
    question_id: str
    answerable: bool
    difficulty: str | None
    question: str
    expected_answer: str
    expected_page: int | None
    predicted_answer: str
    predicted_page: int | None
    raw_output: str
    correct_auto: bool  # lenient: content graded even if the format contract failed
    correct_strict: bool  # end-to-end: correct AND parse_success (what a downstream system gets)
    page_correct: bool | None
    confidence: float | None
    parse_success: bool
    first_attempt_parse_success: bool
    retry_count: int
    latency_ms: float  # wall clock incl. retries - what the user waits
    first_call_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    tokens_per_s: float | None
    answer_chars: int
    peak_rss_gb: float | None
    model_ram_gb: float | None
    order_index: int  # position in the run, for drift checks
    error: str | None = None
    prompt_cache_bust: bool = False  # True when a per-request nonce prevented prompt-cache reuse


FIELDS = [f.name for f in fields(Record)]


def append(path: Path, rec: Record) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_questions(path: Path) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids")
    for q in qs:
        q.setdefault("answerable", True)
        q.setdefault("expected_keywords", [])
    return qs


# ---- human review loop -----------------------------------------------------------------------------
GRADING_COLUMNS = [
    "model", "output_mode", "question_id", "answerable", "question", "expected_answer",
    "predicted_answer", "correct_auto", "manual_correct",
]  # fmt: skip


def export_grading(records: list[dict], path: Path) -> None:
    """Write a spreadsheet-friendly review sheet. Existing manual grades are preserved."""
    existing = load_manual_grades(path)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=GRADING_COLUMNS)
        w.writeheader()
        for r in records:
            key = (r["model"], r["output_mode"], r["question_id"])
            row = {c: r.get(c, "") for c in GRADING_COLUMNS}
            row["correct_auto"] = int(r["correct_auto"])
            row["manual_correct"] = "" if existing.get(key) is None else int(existing[key])
            w.writerow(row)


def load_manual_grades(path: Path) -> dict[tuple[str, str, str], bool | None]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as fh:
        return {
            (row["model"], row["output_mode"], row["question_id"]): parse_manual(row.get("manual_correct"))
            for row in csv.DictReader(fh)
        }


def apply_manual_grades(records: list[dict], grades: dict) -> list[dict]:
    """Final correctness = human grade when given, else auto grade. Strict = correct AND parsed."""
    out = []
    for r in records:
        r = dict(r)
        manual = grades.get((r["model"], r["output_mode"], r["question_id"]))
        r["manual_correct"] = manual
        r["correct"] = bool(manual) if manual is not None else bool(r["correct_auto"])
        r["correct_strict"] = r["correct"] and bool(r["parse_success"])
        out.append(r)
    return out
