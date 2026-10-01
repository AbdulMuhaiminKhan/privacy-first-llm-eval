"""Automatic first-pass grading. A human reviews `grading.csv` and overrides via `manual_correct`.

Rules (identical for every output mode):
- answerable question: every keyword group must appear as a whole word/phrase in the predicted answer
  (`a|b` = either), and the answer must not be a refusal.
- unanswerable question: correct iff the answer contains a refusal phrase. Answering anyway = hallucination.
  (source_page == 0 is NOT used: in run 20261002_000441 Phi-3 answered "Business casual." with page 0.)
"""

from __future__ import annotations

import re

TRUE_VALUES = {"1", "true", "yes", "y", "correct", "x"}
FALSE_VALUES = {"0", "false", "no", "n", "wrong"}

_REFUSAL = re.compile(
    r"not found in the document|not (?:mentioned|specified|stated|provided|included|covered|available|present)"
    r"|(?:does|do) not (?:say|mention|specify|state|contain|include|provide)"
    r"|doesn't (?:say|mention|specify|state|contain|include|provide)"
    r"|no (?:information|mention|details?)|isn't (?:mentioned|specified|stated)|cannot (?:be )?(?:found|determined)"
    r"|can't (?:be )?(?:found|determined)|unable to (?:find|determine)",
    re.I,
)


def _matches(alt: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(alt.strip().lower())}(?!\w)", text) is not None


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer or ""))


def keyword_grade(answer: str, expected_keywords: list[str]) -> bool:
    """True if every keyword group matches as a whole word/phrase and the answer is not a refusal."""
    text = (answer or "").lower()
    if is_refusal(text):
        return False
    return all(any(_matches(alt, text) for alt in group.split("|")) for group in expected_keywords)


def grade(question: dict, answer: str, source_page: int | None = None) -> bool:
    if question.get("answerable", True):
        return keyword_grade(answer, question["expected_keywords"])
    return is_refusal(answer)


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
