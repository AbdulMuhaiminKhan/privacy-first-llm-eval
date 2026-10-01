"""Output contract for structured answers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class Answer(BaseModel):
    """{"answer": str, "confidence": float, "source_page": int}

    source_page = 0 means "not found in the document".
    Validate with context={"max_page": N} to also reject page numbers that don't exist
    (a hallucinated citation is a failure, not a formatting nit).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)
    source_page: int = Field(ge=0)

    @field_validator("source_page")
    @classmethod
    def page_exists(cls, v: int, info: ValidationInfo) -> int:
        max_page = (info.context or {}).get("max_page")
        if max_page is not None and v > max_page:
            raise ValueError(f"source_page {v} does not exist; the document has pages 1..{max_page} (0 = not found)")
        return v


SCHEMA_HINT = '{"answer": "<string>", "confidence": <float between 0.0 and 1.0>, "source_page": <integer>}'
