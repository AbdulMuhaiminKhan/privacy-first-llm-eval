"""Thin, typed wrapper around the local Ollama server."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import ollama

from .config import Settings

logger = logging.getLogger(__name__)
_NS = 1e9
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMStats:
    wall_s: float  # client-side, what the user actually waits
    total_s: float  # server-side total_duration
    load_s: float  # model load time (≈0 once warm)
    prompt_tokens: int
    output_tokens: int
    eval_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.eval_s if self.eval_s > 0 else 0.0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    stats: LLMStats


class OllamaLLM:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.assert_offline()
        self.client = ollama.Client(host=self.settings.ollama_host, timeout=self.settings.request_timeout_s)
        # per-model request overrides, e.g. {"qwen3:8b": {"think": False}} for reasoning models
        self.model_overrides: dict[str, dict[str, Any]] = {}

    # ---- health / model management -------------------------------------------------------------
    def installed_models(self) -> set[str]:
        try:
            resp = self.client.list()
        except Exception as exc:  # httpx.ConnectError etc.
            raise LLMError(
                f"Cannot reach Ollama at {self.settings.ollama_host}. Is it running? (`ollama serve`)"
            ) from exc
        names: set[str] = set()
        for m in resp.models:
            name = m.model or ""
            names.add(name)
            if name.endswith(":latest"):
                names.add(name.removesuffix(":latest"))
        return names

    def model_info(self, model: str) -> dict[str, Any]:
        """Exact identity of a pulled model: digest, size and (if reported) parameter count / quant."""
        target = model if ":" in model else f"{model}:latest"
        for m in self.client.list().models:
            if m.model == target:
                d = m.details
                return {
                    "tag": m.model,
                    "digest": m.digest,
                    "size_gb": round((m.size or 0) / 1024**3, 2),
                    "family": getattr(d, "family", None) if d else None,
                    "parameter_size": getattr(d, "parameter_size", None) if d else None,
                    "quantization": getattr(d, "quantization_level", None) if d else None,
                    "format": getattr(d, "format", None) if d else None,
                }
        raise LLMError(f"Model {model!r} is not pulled. Run: ollama pull {model}")

    def ensure_model(self, model: str) -> None:
        if model not in self.installed_models():
            raise LLMError(f"Model {model!r} is not pulled. Run: ollama pull {model}")

    def loaded_models(self) -> list[dict[str, Any]]:
        """What `ollama ps` shows: size = total bytes allocated, size_vram = bytes on GPU."""
        return [{"model": m.model, "size": m.size or 0, "size_vram": m.size_vram or 0} for m in self.client.ps().models]

    def unload(self, model: str) -> None:
        self.client.generate(model=model, prompt="", keep_alive=0)

    def unload_all(self) -> None:
        for m in self.loaded_models():
            self.unload(m["model"])

    def warm_up(self, model: str) -> float:
        """Load the model into memory; returns load seconds. Excluded from latency measurements."""
        t0 = time.perf_counter()
        self.client.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            options={**self.settings.llm_options(), "num_predict": 2},
            keep_alive=self.settings.keep_alive,
            **self.model_overrides.get(model, {}),
        )
        return time.perf_counter() - t0

    # ---- inference -----------------------------------------------------------------------------
    def chat(self, model: str, messages: list[dict[str, str]], fmt: str | dict | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            resp = self.client.chat(
                model=model,
                messages=messages,
                format=fmt,
                options=self.settings.llm_options(),
                keep_alive=self.settings.keep_alive,
                **self.model_overrides.get(model, {}),
            )
        except ollama.ResponseError as exc:
            raise LLMError(f"Ollama error for {model}: {exc.error}") from exc
        except Exception as exc:
            raise LLMError(f"Request to Ollama failed for {model}: {exc}") from exc
        wall = time.perf_counter() - t0
        stats = LLMStats(
            wall_s=wall,
            total_s=(resp.total_duration or 0) / _NS,
            load_s=(resp.load_duration or 0) / _NS,
            prompt_tokens=resp.prompt_eval_count or 0,
            output_tokens=resp.eval_count or 0,
            eval_s=(resp.eval_duration or 0) / _NS,
        )
        # Defensive: strip reasoning traces if a thinking model emits them inline
        content = _THINK.sub("", resp.message.content or "").strip()
        return LLMResponse(content=content, stats=stats)
