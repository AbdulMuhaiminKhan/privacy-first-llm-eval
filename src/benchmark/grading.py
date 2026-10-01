"""Automatic first-pass grading. A human reviews the CSV and overrides via `manual_correct`."""

from __future__ import annotations

import re

TRUE_VALUES = {"1", "true", "yes", "y", "correct", "x"}
FALSE_VALUES = {"0", "false", "no", "n", "wrong"}


def _matches(alt: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(alt.strip().lower())}(?!\w)", text) is not None


def keyword_grade(answer: str, expected_keywords: list[str]) -> bool:
    """True if every keyword group matches as a whole word/phrase. `a|b` means either a or b."""
    text = (answer or "").lower()
    if "not found in the document" in text:  # every question in the set is answerable
        return False
    return all(any(_matches(alt, text) for alt in group.split("|")) for group in expected_keywords)


def parse_manual(value: object) -> bool | None:
    """Interpret the human `manual_correct` column; blank -> None (fall back to auto grade)."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s in TRUE_VALUES:
        return True
    if s in FALSE_VALUES:
        return False
    raise ValueError(f"Unrecognised manual_correct value {value!r}; use 1/0")
