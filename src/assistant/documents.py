"""Document loading (PDF / text) and page-aware chunking."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
_WS = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a human would cite it
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str
    source: str


class DocumentError(ValueError):
    pass


def _clean(text: str) -> str:
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_document(path: str | Path) -> list[Page]:
    """Load a PDF or text file into pages.

    Text files are split into pages on form-feed characters (\\f); without them the file is one page.
    """
    path = Path(path)
    if not path.is_file():
        raise DocumentError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"Unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        raw_pages = [(page.extract_text() or "") for page in reader.pages]
    else:
        raw_pages = path.read_text(encoding="utf-8", errors="replace").split("\f")

    pages = [Page(number=i, text=_clean(t), source=path.name) for i, t in enumerate(raw_pages, start=1)]
    pages = [p for p in pages if p.text]
    if not pages:
        raise DocumentError(
            f"No extractable text in {path.name}. If it is a scanned PDF, run OCR first (e.g. `ocrmypdf`)."
        )
    logger.info("Loaded %s: %d pages with text", path.name, len(pages))
    return pages


def chunk_pages(pages: list[Page], size_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    """Split pages into overlapping word windows. Chunks never cross page boundaries,
    so every chunk maps to exactly one page and `source_page` citations are verifiable."""
    if overlap_words >= size_words:
        raise ValueError("overlap_words must be smaller than size_words")
    step = size_words - overlap_words
    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        for idx, start in enumerate(range(0, max(len(words) - overlap_words, 1), step)):
            window = words[start : start + size_words]
            if window:
                chunks.append(
                    Chunk(
                        chunk_id=f"p{page.number}-c{idx}", page=page.number, text=" ".join(window), source=page.source
                    )
                )
    return chunks
