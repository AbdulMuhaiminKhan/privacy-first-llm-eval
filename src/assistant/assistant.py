"""Document Q&A orchestration: load -> chunk -> retrieve -> prompt -> (validate)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import Settings
from .documents import Chunk, chunk_pages, load_document
from .llm import LLMResponse, LLMStats, OllamaLLM
from .retrieval import BM25Index
from .structured import ChatModel, StructuredOutputError, StructuredResult, ask_structured
from .textparse import parse_labelled

NOT_FOUND = "Not found in the document."


class OutputMode(str, Enum):
    """Independent variable of the structured-output experiment.

    All three modes request the SAME three fields with the SAME instructions; only the encoding and
    the decoding constraint differ:
      text   - labelled plain-text lines, no decoding constraint, no retries
      json   - JSON object, Ollama JSON mode (format="json"), Pydantic validation + retries
      schema - JSON object, decoding constrained to the Pydantic JSON schema, validation + retries
    """

    TEXT = "text"
    JSON = "json"
    SCHEMA = "schema"


# Instructions + context go in ONE user turn: Gemma 2's chat template has no system role, and using the
# same prompt shape for every model keeps the benchmark fair.
_PROMPT = """You are a precise assistant answering questions about a private document.
Rules:
- Use ONLY the context below. Do not use outside knowledge.
- If the answer is not in the context, say "{not_found}".
- Be concise: one or two sentences.
- Give your confidence that the answer is correct (0.0 = guessing, 1.0 = certain).
- Give the [Page N] number the answer came from, or 0 if not found.
{format_block}
Context:
{context}

Question: {question}"""

_FORMAT_BLOCKS = {
    OutputMode.TEXT: """Output format - exactly three lines:
Answer: <your answer>
Confidence: <number between 0.0 and 1.0>
Source page: <integer>""",
    OutputMode.JSON: """Output format - ONLY a JSON object with exactly these keys:
{{"answer": "<your answer>", "confidence": <number between 0.0 and 1.0>, "source_page": <integer>}}""",
}
_FORMAT_BLOCKS[OutputMode.SCHEMA] = _FORMAT_BLOCKS[OutputMode.JSON]


@dataclass
class ModeResult:
    """Uniform result for every output mode. Fields that could not be obtained are None, never guessed."""

    mode: OutputMode
    predicted_answer: str
    confidence: float | None
    source_page: int | None
    parse_success: bool  # final output satisfies the Answer contract
    first_attempt_parse_success: bool
    retry_count: int
    raw_output: str
    stats: list[LLMStats] = field(default_factory=list)
    error: str | None = None


@dataclass
class Retrieved:
    chunks: list[Chunk]
    scores: list[float]


class DocumentAssistant:
    def __init__(self, doc_path: str | Path, llm: ChatModel | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = llm or OllamaLLM(self.settings)
        self.pages = load_document(doc_path)
        self.max_page = max(p.number for p in self.pages)
        self.index = BM25Index(
            chunk_pages(self.pages, self.settings.chunk_size_words, self.settings.chunk_overlap_words)
        )

    def retrieve(self, question: str) -> Retrieved:
        hits = self.index.search(question, k=self.settings.top_k_chunks)
        return Retrieved(chunks=[c for c, _ in hits], scores=[s for _, s in hits])

    def build_messages(
        self, question: str, mode: OutputMode | str = OutputMode.TEXT, nonce: str | None = None
    ) -> list[dict[str, str]]:
        """`nonce` (benchmark only) is prepended so no two requests share a prompt prefix: otherwise Ollama's
        prompt cache makes whichever identical prompt runs second look faster."""
        mode = OutputMode(mode)
        retrieved = self.retrieve(question)
        context = "\n\n".join(f"[Page {c.page}] {c.text}" for c in retrieved.chunks) or "(no relevant context)"
        prompt = _PROMPT.format(
            not_found=NOT_FOUND,
            format_block=_FORMAT_BLOCKS[mode],
            context=context,
            question=question.strip(),
        )
        if nonce:
            prompt = f"[request {nonce}]\n{prompt}"
        return [{"role": "user", "content": prompt}]

    # Plain-text answer (labelled lines), as a human would read it
    def ask(self, question: str, model: str | None = None) -> LLMResponse:
        return self.llm.chat(model or self.settings.default_model, self.build_messages(question, OutputMode.TEXT))

    # Validated JSON answer with retry
    def ask_structured(self, question: str, model: str | None = None) -> StructuredResult:
        mode = OutputMode(self.settings.format_mode)
        return ask_structured(
            self.llm,
            model or self.settings.default_model,
            self.build_messages(question, mode),
            max_page=self.max_page,
            max_retries=self.settings.max_retries,
            format_mode=mode.value,
        )

    # Experiment entry point: one question, one model, one output mode
    def answer(self, question: str, model: str, mode: OutputMode | str, nonce: str | None = None) -> ModeResult:
        mode = OutputMode(mode)
        messages = self.build_messages(question, mode, nonce=nonce)
        if mode is OutputMode.TEXT:
            resp = self.llm.chat(model, messages)
            parsed = parse_labelled(resp.content, max_page=self.max_page)
            return ModeResult(
                mode=mode,
                predicted_answer=parsed.answer,
                confidence=parsed.confidence,
                source_page=parsed.source_page,
                parse_success=parsed.valid,
                first_attempt_parse_success=parsed.valid,
                retry_count=0,
                raw_output=resp.content,
                stats=[resp.stats],
            )
        try:
            res = ask_structured(
                self.llm,
                model,
                messages,
                max_page=self.max_page,
                max_retries=self.settings.max_retries,
                format_mode=mode.value,
            )
            return ModeResult(
                mode=mode,
                predicted_answer=res.answer.answer,
                confidence=res.answer.confidence,
                source_page=res.answer.source_page,
                parse_success=True,
                first_attempt_parse_success=res.first_attempt_valid,
                retry_count=res.attempts - 1,
                raw_output=res.raw_outputs[-1],
                stats=res.stats,
            )
        except StructuredOutputError as exc:
            raw = exc.raw_outputs[-1] if exc.raw_outputs else ""
            return ModeResult(
                mode=mode,
                predicted_answer=_best_effort_answer(raw),
                confidence=None,
                source_page=None,
                parse_success=False,
                first_attempt_parse_success=False,
                retry_count=exc.attempts - 1,
                raw_output=raw,
                stats=exc.stats,
                error=str(exc),
            )


def _best_effort_answer(raw: str) -> str:
    """For lenient grading of an output that failed the contract: use its "answer" value if any."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            return obj["answer"]
    except (ValueError, TypeError):
        pass
    return raw
