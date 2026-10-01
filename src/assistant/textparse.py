"""Parse the labelled plain-text format (text mode) and validate it with the SAME Pydantic contract
that JSON modes must satisfy, so `parse_success` means the same thing in every mode."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from .schemas import Answer


def _label(name: str) -> str:
    """`Answer:`, `**Answer:**`, `**Answer**:` ... (models often add markdown bold)."""
    return rf"^[ \t]*\**[ \t]*{name}[ \t]*\**[ \t]*:[ \t]*\**[ \t]*"


_ANSWER = re.compile(
    _label("answer") + r"(.+?)(?=" + _label("confidence") + "|" + _label(r"source\s*page") + r"|\Z)", re.I | re.M | re.S
)
_CONF = re.compile(_label("confidence") + r"([-+]?\d*\.?\d+)", re.I | re.M)
_PAGE = re.compile(_label(r"source\s*page") + r"\[?(?:page\s*)?(\d+)", re.I | re.M)


@dataclass(frozen=True)
class ParsedText:
    answer: str  # the Answer line if present, else the whole reply (so content can still be graded)
    confidence: float | None  # None when missing or outside [0, 1] - never coerced
    source_page: int | None
    valid: bool  # all three fields present and valid under the Answer schema


def parse_labelled(raw: str, max_page: int | None = None, lenient: bool = False) -> ParsedText:
    """Strict (default): all three labelled lines required. Lenient: an unlabelled answer before the
    Confidence/Source lines is accepted - used only for the parser-strictness sensitivity analysis."""
    a, c, p = _ANSWER.search(raw), _CONF.search(raw), _PAGE.search(raw)
    if a is None and lenient and c is not None:
        head = raw[: c.start()].strip()
        if head:
            a = re.match(r"(?s)(.+)", head)
    answer = a.group(1).strip() if a else raw.strip()
    fields = {
        "answer": answer,
        "confidence": float(c.group(1)) if c else None,
        "source_page": int(p.group(1)) if p else None,
    }
    try:
        Answer.model_validate(fields, context={"max_page": max_page} if max_page else None)
        valid = bool(a and c and p)
    except ValidationError:
        valid = False
    conf = fields["confidence"]
    page = fields["source_page"]
    return ParsedText(
        answer=answer,
        confidence=conf if conf is not None and 0.0 <= conf <= 1.0 else None,
        source_page=page if page is not None and (max_page is None or page <= max_page) else None,
        valid=valid,
    )
