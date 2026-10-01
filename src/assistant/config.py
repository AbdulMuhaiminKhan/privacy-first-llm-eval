"""Runtime settings. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    default_model: str = field(default_factory=lambda: _env("ASSISTANT_MODEL", "mistral"))
    # Ollama's default context window is small; retrieved context + instructions need ~2-3k tokens.
    num_ctx: int = field(default_factory=lambda: int(_env("ASSISTANT_NUM_CTX", "4096")))
    temperature: float = 0.0  # deterministic answers -> reproducible benchmark
    seed: int = 42
    max_answer_tokens: int = 256
    top_k_chunks: int = 4
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    max_retries: int = 3  # retries AFTER the first attempt -> up to 4 calls total
    # "json" = Ollama JSON mode (valid JSON, any shape). "schema" = grammar-constrained to the Pydantic schema.
    format_mode: str = field(default_factory=lambda: _env("ASSISTANT_FORMAT_MODE", "json"))
    request_timeout_s: float = 300.0
    keep_alive: str = "10m"
    allow_remote: bool = field(default_factory=lambda: _env("ASSISTANT_ALLOW_REMOTE", "0") == "1")

    def llm_options(self) -> dict:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.max_answer_tokens,
        }

    def assert_offline(self) -> None:
        """Privacy guard: refuse to send document text anywhere but this machine."""
        host = urlparse(self.ollama_host if "://" in self.ollama_host else f"http://{self.ollama_host}").hostname
        if host not in LOOPBACK_HOSTS and not self.allow_remote:
            raise RuntimeError(
                f"OLLAMA_HOST={self.ollama_host!r} is not a loopback address. This assistant is offline-only; "
                "set ASSISTANT_ALLOW_REMOTE=1 if you deliberately run Ollama on another trusted machine."
            )
