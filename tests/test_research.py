"""Correctness of the research machinery: prompts, parsing, grading, statistics, calibration, findings."""

from __future__ import annotations

import numpy as np
import pytest

from assistant.assistant import DocumentAssistant, OutputMode
from assistant.config import Settings
from assistant.textparse import parse_labelled
from benchmark.analysis import analyze
from benchmark.calibration import auroc, calibration_summary, ece
from benchmark.grading import grade, is_refusal
from benchmark.stats import mcnemar_exact, paired_diff, significant, wilson_ci

PDF = "data/sample/handbook.pdf"


class NoLLM:
    def chat(self, *a, **k):  # pragma: no cover - never called
        raise AssertionError


# ---- RQ1 control: modes differ ONLY in the format block ----------------------------------------
def test_modes_share_everything_but_the_format_block():
    asst = DocumentAssistant(PDF, llm=NoLLM(), settings=Settings())
    q = "What is the hotel limit in Zurich?"
    prompts = {m: asst.build_messages(q, m)[0]["content"] for m in OutputMode}
    head = {m: p.split("Output format")[0] for m, p in prompts.items()}
    tail = {m: p.split("Context:")[1] for m, p in prompts.items()}
    assert len(set(head.values())) == 1 and len(set(tail.values())) == 1  # same rules, context, question
    for p in prompts.values():
        assert "confidence" in p.lower() and "page" in p.lower()  # all modes request the same fields
    assert prompts[OutputMode.JSON] == prompts[OutputMode.SCHEMA]  # schema differs only in decoding


# ---- text-mode parsing uses the same contract ----------------------------------------------------
def test_parse_labelled_valid():
    p = parse_labelled("Answer: 220 euros per night.\nConfidence: 0.85\nSource page: 3", max_page=6)
    assert p.valid and p.answer == "220 euros per night." and p.confidence == 0.85 and p.source_page == 3


@pytest.mark.parametrize(
    "raw",
    [
        "Answer: 220 euros.\nSource page: 3",  # missing confidence
        "Answer: 220 euros.\nConfidence: 85\nSource page: 3",  # percent, same rule as JSON
        "Answer: 220 euros.\nConfidence: 0.9\nSource page: 42",  # page does not exist
        "The limit is 220 euros.",  # no structure at all
    ],
)
def test_parse_labelled_invalid_never_coerced(raw):
    p = parse_labelled(raw, max_page=6)
    assert not p.valid
    assert p.confidence is None or 0 <= p.confidence <= 1
    assert "220" in p.answer  # content is still gradable (lenient accuracy)


def test_parse_labelled_markdown_bold_labels():
    p = parse_labelled("**Answer:** No.\n**Confidence:** 0.7\n**Source page:** 5", max_page=6)
    assert p.valid and p.answer == "No."


# ---- grading -------------------------------------------------------------------------------------
def test_grade_unanswerable_and_refusals():
    unans = {"answerable": False, "expected_keywords": []}
    assert grade(unans, "Not found in the document.", 0)
    assert grade(unans, "The document does not mention the CEO.", 0)
    assert not grade(unans, "The CEO is Mirela Hofstad.", 1)  # hallucination
    ans = {"answerable": True, "expected_keywords": ["220"]}
    assert grade(ans, "220 euros", 3) and not grade(ans, "Not found in the document.", 0)
    assert is_refusal("This is not specified in the context")


# ---- statistics ----------------------------------------------------------------------------------
def test_wilson_known_values():
    lo, hi = wilson_ci(7, 10)
    assert lo == pytest.approx(0.3968, abs=1e-3) and hi == pytest.approx(0.8922, abs=1e-3)
    assert wilson_ci(0, 10)[0] == pytest.approx(0.0) and wilson_ci(10, 10)[1] == pytest.approx(1.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)  # 2 * 0.5**5
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_paired_diff_and_significance_rule():
    a = [1] * 30 + [0] * 30
    b = [1] * 30 + [1] * 15 + [0] * 15  # b fixes 15 of a's errors, breaks none
    r = paired_diff(a, b)
    assert r["diff"] == pytest.approx(0.25) and r["ci"][0] > 0 and r["p_mcnemar"] < 0.001
    assert significant(r)
    same = paired_diff(a, a)
    assert same["diff"] == 0 and not significant(same)


# ---- calibration ---------------------------------------------------------------------------------
def test_perfectly_calibrated_has_zero_ece():
    conf = np.array([0.6] * 10 + [0.95] * 20)
    correct = np.array([1] * 6 + [0] * 4 + [1] * 19 + [0] * 1)
    assert ece(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_constant_confidence_has_no_signal():
    conf = np.full(20, 0.9)
    correct = np.array([1] * 10 + [0] * 10)
    assert auroc(conf, correct) == 0.5
    s = calibration_summary(conf, correct)
    assert s["overconfidence"] == pytest.approx(0.4) and s["most_common_share"] == 1.0


def test_calibration_ignores_missing_confidence():
    s = calibration_summary([0.9, None, float("nan"), 0.8], [1, 0, 0, 1])
    assert s["n"] == 2


# ---- findings are rule-generated, never overstated ------------------------------------------------
def _records(model_acc: dict[str, list[int]]) -> list[dict]:
    out = []
    for mode, flags in model_acc.items():
        for i, c in enumerate(flags):
            out.append({
                "model": "m", "output_mode": mode, "question_id": f"q{i}", "answerable": True, "question": "q",
                "expected_answer": "e", "predicted_answer": "p", "correct": bool(c), "correct_auto": bool(c),
                "correct_strict": bool(c), "confidence": 0.9, "latency_ms": 1000.0, "retry_count": 0,
                "parse_success": True, "first_attempt_parse_success": True, "output_tokens": 10,
                "answer_chars": 5, "model_ram_gb": None, "page_correct": None, "error": None,
            })  # fmt: skip
    return out


def test_small_gap_is_reported_as_unresolved():
    flags = [1] * 20 + [0] * 10
    worse = flags[:]
    worse[0] = 0  # one question flips: -3.3 pp
    a = analyze(_records({"text": flags, "json": worse}), meta={"run_id": "t", "simulated": True})
    acc = next(f for f in a["findings"] if f.startswith("**Task accuracy"))
    assert "no detectable difference" in acc


def test_large_gap_is_reported_as_difference():
    text = [1] * 40
    json_ = [1] * 20 + [0] * 20
    a = analyze(_records({"text": text, "json": json_}), meta={"run_id": "t", "simulated": True})
    acc = next(f for f in a["findings"] if f.startswith("**Task accuracy"))
    assert "**lower**" in acc and "-50.0 pp" in acc


# ---- regressions found in the real run 20261002_000441 -------------------------------------------
def test_unanswerable_page_zero_is_not_a_refusal():
    unans = {"answerable": False, "expected_keywords": []}
    assert not grade(unans, "Business casual.", 0)  # Phi-3 hallucination that the old shortcut accepted


def test_lenient_parser_accepts_unlabelled_answer_only():
    raw = "Premium economy\n\nConfidence: 1.0\n\nSource page: 0"
    assert not parse_labelled(raw, 6).valid
    lenient = parse_labelled(raw, 6, lenient=True)
    assert lenient.valid and lenient.answer == "Premium economy"
    assert not parse_labelled("Not found in the document.", 6, lenient=True).valid  # fields still required


def test_prompt_cache_cold_flag():
    """json and schema share an identical prompt, so the later of the two is warm; text is always cold."""
    import pandas as pd

    from benchmark.analysis import add_derived_columns

    rows = [
        {"model": "m", "question_id": "q1", "output_mode": "schema", "order_index": 1},
        {"model": "m", "question_id": "q1", "output_mode": "text", "order_index": 2},
        {"model": "m", "question_id": "q1", "output_mode": "json", "order_index": 3},
    ]
    for r in rows:
        r.update(parse_success=True, correct=True, raw_output="")
    df = add_derived_columns(pd.DataFrame(rows)).set_index("output_mode")
    assert df.loc["schema", "prompt_cache_cold"] and df.loc["text", "prompt_cache_cold"]
    assert not df.loc["json", "prompt_cache_cold"]
    busted = add_derived_columns(pd.DataFrame([{**r, "prompt_cache_bust": True} for r in rows]))
    assert busted["prompt_cache_cold"].all()


def test_benchmark_prompts_get_unique_nonce():
    asst = DocumentAssistant(PDF, llm=NoLLM(), settings=Settings())
    a = asst.build_messages("q?", "json", nonce="aaaa1111")[0]["content"]
    b = asst.build_messages("q?", "schema", nonce="bbbb2222")[0]["content"]
    assert a.startswith("[request aaaa1111]") and a.splitlines()[0] != b.splitlines()[0]
