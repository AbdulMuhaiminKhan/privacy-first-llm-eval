"""Structured (JSON) answers with Pydantic validation and self-repair retries."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from .llm import LLMResponse, LLMStats
from .schemas import SCHEMA_HINT, Answer

logger = logging.getLogger(__name__)


class ChatModel(Protocol):
    def chat(self, model: str, messages: list[dict[str, str]], fmt: str | dict | None = None) -> LLMResponse: ...


class StructuredOutputError(RuntimeError):
    def __init__(
        self, message: str, attempts: int, errors: list[str], raw_outputs: list[str], stats: list[LLMStats]
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors
        self.raw_outputs = raw_outputs
        self.stats = stats


@dataclass
class StructuredResult:
    answer: Answer
    attempts: int  # 1 = valid on the first try
    errors: list[str] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)
    stats: list[LLMStats] = field(default_factory=list)

    @property
    def first_attempt_valid(self) -> bool:
        return self.attempts == 1

    @property
    def total_wall_s(self) -> float:
        return sum(s.wall_s for s in self.stats)


def format_validation_error(exc: ValidationError) -> str:
    """Compact, model-readable error list: `field: problem (got: value)`."""
    lines = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        got = err.get("input")
        got_repr = json.dumps(got)[:80] if not isinstance(got, dict) else "<object>"
        lines.append(f"- {loc}: {err['msg']} (got: {got_repr})")
    return "\n".join(lines)


def repair_prompt(error_text: str) -> str:
    return (
        "Your previous reply did not match the required JSON schema.\n"
        f"Validation errors:\n{error_text}\n\n"
        f"Reply again with ONLY a JSON object of exactly this shape and no other keys:\n{SCHEMA_HINT}\n"
        "Keep the same answer content; only fix the format."
    )


def resolve_format(format_mode: str) -> str | dict:
    if format_mode == "json":
        return "json"
    if format_mode == "schema":
        return Answer.model_json_schema()
    raise ValueError(f"format_mode must be 'json' or 'schema', got {format_mode!r}")


def ask_structured(
    llm: ChatModel,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_page: int | None = None,
    max_retries: int = 3,
    format_mode: str = "json",
) -> StructuredResult:
    """Call the model, validate against `Answer`, and on failure feed the validation errors back
    as a follow-up turn — up to `max_retries` times (so at most max_retries + 1 calls)."""
    fmt = resolve_format(format_mode)
    history = list(messages)
    errors: list[str] = []
    raws: list[str] = []
    stats: list[LLMStats] = []
    context = {"max_page": max_page} if max_page is not None else None

    for attempt in range(1, max_retries + 2):
        resp = llm.chat(model, history, fmt=fmt)
        raws.append(resp.content)
        stats.append(resp.stats)
        try:
            answer = Answer.model_validate_json(resp.content, context=context)
            if attempt > 1:
                logger.info("%s: valid JSON after %d attempts", model, attempt)
            return StructuredResult(answer=answer, attempts=attempt, errors=errors, raw_outputs=raws, stats=stats)
        except ValidationError as exc:
            err_text = format_validation_error(exc)
            errors.append(err_text)
            logger.warning("%s attempt %d/%d invalid:\n%s", model, attempt, max_retries + 1, err_text)
            history += [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": repair_prompt(err_text)},
            ]

    raise StructuredOutputError(
        f"{model}: no valid output after {max_retries + 1} attempts",
        attempts=max_retries + 1,
        errors=errors,
        raw_outputs=raws,
        stats=stats,
    )
