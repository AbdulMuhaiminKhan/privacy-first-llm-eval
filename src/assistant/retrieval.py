"""Dependency-free BM25 retrieval.

Why BM25 instead of embeddings: zero extra model in RAM (the benchmark measures the LLM, not an
embedding model), deterministic, and strong on exact terms (names, numbers, policy keywords) which
dominate handbook-style Q&A. Swap in `ollama.embed` + cosine similarity if you need semantic recall.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .documents import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,:][0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its many much of on or "
    "per should the their there this to was what when where which who why will with within".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("Cannot build an index over zero chunks")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(c.text)) for c in chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = sum(self._lens) / len(self._lens)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def _score(self, query_terms: list[str], i: int) -> float:
        tf, length = self._tfs[i], self._lens[i]
        score = 0.0
        for term in query_terms:
            f = tf.get(term)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * length / self._avg_len)
            score += self._idf[term] * f * (self.k1 + 1) / denom
        return score

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = [(self.chunks[i], self._score(terms, i)) for i in range(len(self.chunks))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:k]
