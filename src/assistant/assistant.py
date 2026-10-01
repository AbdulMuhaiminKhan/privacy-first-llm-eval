"""Document Q&A orchestration: load -> chunk -> retrieve -> prompt -> (validate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .documents import Chunk, chunk_pages, load_document
from .llm import LLMResponse, OllamaLLM
from .retrieval import BM25Index
from .structured import ChatModel, StructuredResult, ask_structured

NOT_FOUND = "Not found in the document."

# Instructions + context go in ONE user turn: Gemma 2's chat template has no system role, and using the
# same prompt shape for every model keeps the benchmark fair.
_PROMPT = """You are a precise assistant answering questions about a private document.
Rules:
- Use ONLY the context below. Do not use outside knowledge.
- If the answer is not in the context, say "{not_found}".
- Be concise: one or two sentences.
{format_rules}
Context:
{context}

Question: {question}"""

_JSON_RULES = """- Reply with ONLY a JSON object:
  {{"answer": string, "confidence": number 0.0-1.0, "source_page": integer}}
- source_page is the [Page N] number the answer came from, or 0 if not found."""


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

    def build_messages(self, question: str, structured: bool) -> tuple[list[dict[str, str]], Retrieved]:
        retrieved = self.retrieve(question)
        context = "\n\n".join(f"[Page {c.page}] {c.text}" for c in retrieved.chunks) or "(no relevant context)"
        prompt = _PROMPT.format(
            not_found=NOT_FOUND,
            format_rules=_JSON_RULES if structured else "",
            context=context,
            question=question.strip(),
        )
        return [{"role": "user", "content": prompt}], retrieved

    # Section 2: plain-text answer
    def ask(self, question: str, model: str | None = None) -> LLMResponse:
        messages, _ = self.build_messages(question, structured=False)
        return self.llm.chat(model or self.settings.default_model, messages)

    # Section 4: validated JSON answer with retry
    def ask_structured(self, question: str, model: str | None = None) -> StructuredResult:
        messages, _ = self.build_messages(question, structured=True)
        return ask_structured(
            self.llm,
            model or self.settings.default_model,
            messages,
            max_page=self.max_page,
            max_retries=self.settings.max_retries,
            format_mode=self.settings.format_mode,
        )
