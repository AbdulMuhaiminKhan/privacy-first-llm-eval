from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from assistant.config import Settings
from assistant.documents import Page, chunk_pages, load_document
from assistant.llm import LLMResponse, LLMStats
from assistant.retrieval import BM25Index
from assistant.schemas import Answer
from assistant.structured import StructuredOutputError, ask_structured
from benchmark.grading import keyword_grade, parse_manual
from benchmark.scoring import Bounds, overall_score, speed_labels

PDF = "data/sample/handbook.pdf"
STATS = LLMStats(wall_s=0.1, total_s=0.1, load_s=0, prompt_tokens=10, output_tokens=5, eval_s=0.05)


class ScriptedLLM:
    """Returns canned replies in order and records every message history it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, model, messages, fmt=None):
        self.calls.append(list(messages))
        return LLMResponse(content=self.replies.pop(0), stats=STATS)


VALID = json.dumps({"answer": "220 euros", "confidence": 0.9, "source_page": 3})


# ---- schema -------------------------------------------------------------------------------------
def test_answer_valid():
    a = Answer.model_validate_json(VALID, context={"max_page": 6})
    assert a.source_page == 3 and a.confidence == 0.9


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer": "x", "confidence": 85, "source_page": 3}',  # percent, not 0-1
        '{"answer": "x", "confidence": 0.5}',  # missing field
        '{"answer": "x", "confidence": 0.5, "source_page": 3, "reasoning": "..."}',  # extra key
        '{"answer": "", "confidence": 0.5, "source_page": 3}',  # empty answer
        "Sure! Here is the JSON: {...}",  # not JSON at all
        '{"answer": "x", "confidence": 0.5, "source_page": 42}',  # hallucinated page
    ],
)
def test_answer_invalid(raw):
    with pytest.raises(ValidationError):
        Answer.model_validate_json(raw, context={"max_page": 6})


# ---- retry loop ---------------------------------------------------------------------------------
def test_retry_recovers_and_feeds_back_errors():
    llm = ScriptedLLM(['{"answer": "220 euros", "confidence": 90, "source_page": 3}', VALID])
    res = ask_structured(llm, "m", [{"role": "user", "content": "q"}], max_page=6)
    assert res.attempts == 2 and not res.first_attempt_valid
    repair_turn = llm.calls[1][-1]["content"]
    assert "confidence" in repair_turn and "less than or equal to 1" in repair_turn
    assert llm.calls[1][-2]["role"] == "assistant"  # model sees its own bad output


def test_retry_gives_up_after_max_retries():
    llm = ScriptedLLM(["not json"] * 4)
    with pytest.raises(StructuredOutputError) as ei:
        ask_structured(llm, "m", [{"role": "user", "content": "q"}], max_retries=3)
    assert ei.value.attempts == 4 and len(llm.calls) == 4


def test_first_try_valid():
    res = ask_structured(ScriptedLLM([VALID]), "m", [{"role": "user", "content": "q"}])
    assert res.first_attempt_valid


# ---- documents & retrieval ----------------------------------------------------------------------
def test_pdf_loads_all_pages():
    pages = load_document(PDF)
    assert [p.number for p in pages] == [1, 2, 3, 4, 5, 6]


def test_chunks_never_cross_pages():
    pages = [Page(1, " ".join(f"a{i}" for i in range(500)), "x"), Page(2, "b c d", "x")]
    chunks = chunk_pages(pages, size_words=100, overlap_words=20)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert "a499" in chunks[[c.page for c in chunks].index(2) - 1].text  # tail of page 1 covered


def test_bm25_recall_at_k_on_eval_set():
    """The benchmark measures the LLM, so retrieval must not be the bottleneck:
    the gold page has to be inside the top-k chunks the model sees."""
    idx = BM25Index(chunk_pages(load_document(PDF)))
    with open("data/questions.jsonl", encoding="utf-8") as fh:
        qs = [q for line in fh if (q := json.loads(line)).get("answerable", True)]
    k = Settings().top_k_chunks
    hits = [q["expected_page"] in {c.page for c, _ in idx.search(q["question"], k=k)} for q in qs]
    assert sum(hits) / len(qs) == 1.0


# ---- grading & scoring --------------------------------------------------------------------------
def test_keyword_grade():
    assert keyword_grade("It was founded in 2017 in Leipzig.", ["2017", "Leipzig"])
    assert keyword_grade("Twenty-one days", ["21|twenty-one"])
    assert not keyword_grade("It lasts 210 days", ["21"])  # whole-number match only
    assert not keyword_grade("Not found in the document.", ["no|not"])


def test_parse_manual():
    assert parse_manual("1") is True and parse_manual("0") is False and parse_manual("") is None


def test_overall_score_formula():
    # 0.6*0.83 + 0.2*0.5 + 0.2*0.5 = 0.698 -> 6.98
    assert overall_score(0.83, 0.5, 0.5) == 6.98
    with pytest.raises(ValueError):
        overall_score(83, 0.5, 0.5)


def test_bounds_and_labels():
    b = Bounds(best=1.0, worst=8.0)
    assert b.score(1.0) == 1.0 and b.score(8.0) == 0.0 and b.score(20) == 0.0 and b.score(4.5) == 0.5
    assert speed_labels({"a": 1.2, "b": 2.1, "c": 2.8}) == {"a": "Fast", "b": "Medium", "c": "Slow"}


def test_offline_guard(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://api.example.com")
    with pytest.raises(RuntimeError, match="offline-only"):
        Settings().assert_offline()
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    Settings().assert_offline()
