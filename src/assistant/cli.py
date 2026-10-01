"""Command-line interface.

python -m assistant.cli --doc data/sample/handbook.pdf "What is the hotel limit in Zurich?"
python -m assistant.cli --doc notes.pdf --model phi3 --json          # interactive, JSON answers
"""

from __future__ import annotations

import argparse
import logging
import sys

from .assistant import DocumentAssistant
from .config import Settings
from .llm import LLMError, OllamaLLM
from .structured import StructuredOutputError


def _answer(assistant: DocumentAssistant, question: str, model: str, as_json: bool) -> None:
    if as_json:
        result = assistant.ask_structured(question, model=model)
        print(result.answer.model_dump_json(indent=2))
        print(f"  [{model} | {result.total_wall_s:.2f}s | attempts={result.attempts}]", file=sys.stderr)
    else:
        resp = assistant.ask(question, model=model)
        print(resp.content.strip())
        print(f"  [{model} | {resp.stats.wall_s:.2f}s | {resp.stats.tokens_per_s:.1f} tok/s]", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline document Q&A (Ollama)")
    parser.add_argument("question", nargs="?", help="Question; omit for interactive mode")
    parser.add_argument("--doc", required=True, help="PDF / .txt / .md file")
    parser.add_argument("--model", default=None, help="Ollama model tag (default: $ASSISTANT_MODEL or mistral)")
    parser.add_argument("--json", action="store_true", help="Return validated JSON (Pydantic + retries)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    settings = Settings()
    model = args.model or settings.default_model
    try:
        llm = OllamaLLM(settings)
        llm.ensure_model(model)
        assistant = DocumentAssistant(args.doc, llm=llm, settings=settings)
    except (LLMError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    questions = [args.question] if args.question else None
    try:
        if questions:
            _answer(assistant, questions[0], model, args.json)
            return 0
        print(f"Loaded {args.doc} ({assistant.max_page} pages). Model: {model}. Ctrl+C to exit.")
        while True:
            q = input("\n> ").strip()
            if q:
                try:
                    _answer(assistant, q, model, args.json)
                except StructuredOutputError as exc:
                    print(f"error: {exc}", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        return 0
    except (LLMError, StructuredOutputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
