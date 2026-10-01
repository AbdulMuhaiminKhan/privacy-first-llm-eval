"""Minimal stand-in for the Ollama HTTP API so the full pipeline runs in CI without models.

Behaviour:
- "fake-good":  always returns schema-valid JSON.
- "fake-flaky": first reply per question has confidence=85 (percent instead of 0-1) -> forces a retry.
Answers are looked up from data/questions.jsonl; this tests plumbing, not model quality.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

QUESTIONS = {
    q["question"]: q
    for q in (json.loads(line) for line in (Path(__file__).resolve().parents[1] / "data" / "questions.jsonl").open())
}
MODELS = ["fake-good:latest", "fake-flaky:latest"]
_loaded: set[str] = set()


def _reply(model: str, messages: list[dict], fmt) -> str:
    first = messages[0]["content"]
    m = re.search(r"Question: (.+)$", first, re.S)
    q = QUESTIONS.get(m.group(1).strip() if m else "")
    answer = q["expected_answer"] if q else "Not found in the document."
    page = q["expected_page"] if q else 0
    if fmt is None:
        return answer
    is_retry = len(messages) > 1
    confidence = 85 if (model.startswith("fake-flaky") and not is_retry) else 0.9
    return json.dumps({"answer": answer, "confidence": confidence, "source_page": page})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": m, "model": m, "size": 1} for m in MODELS]})
        elif self.path == "/api/ps":
            self._send({"models": [{"name": m, "model": m, "size": 2 * 1024**3, "size_vram": 0} for m in _loaded]})
        elif self.path == "/api/version":
            self._send({"version": "fake-0.0"})
        else:
            self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = payload["model"]
        full = model if ":" in model else f"{model}:latest"
        if full not in MODELS:
            self.send_response(404)
            body = json.dumps({"error": f"model '{model}' not found"}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/generate":  # used for unload (keep_alive=0)
            if payload.get("keep_alive") == 0:
                _loaded.discard(full)
            self._send({"model": model, "response": "", "done": True})
        elif self.path == "/api/chat":
            _loaded.add(full)
            time.sleep(0.01)
            content = _reply(model, payload["messages"], payload.get("format"))
            self._send(
                {
                    "model": model,
                    "created_at": "2026-01-01T00:00:00Z",
                    "done": True,
                    "message": {"role": "assistant", "content": content},
                    "total_duration": 20_000_000,
                    "load_duration": 1_000_000,
                    "prompt_eval_count": 300,
                    "eval_count": 30,
                    "eval_duration": 15_000_000,
                }
            )
        else:
            self.send_error(404)


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


if __name__ == "__main__":
    srv, url = start(11500)
    print(f"Fake Ollama on {url}")
    srv.serve_forever()
