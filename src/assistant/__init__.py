"""Offline document Q&A assistant backed by local Ollama models."""

from .assistant import DocumentAssistant
from .config import Settings
from .llm import OllamaLLM
from .schemas import Answer
from .structured import StructuredOutputError, StructuredResult, ask_structured

__all__ = [
    "Answer",
    "DocumentAssistant",
    "OllamaLLM",
    "Settings",
    "StructuredOutputError",
    "StructuredResult",
    "ask_structured",
]
