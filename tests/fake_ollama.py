"""Stand-in for the Ollama HTTP API so the whole pipeline runs in CI without models.

!!! Everything this server returns is SIMULATED. It exists to test plumbing, statistics and charts.
!!! `ollama_version` reports "SIMULATED" and `python -m benchmark publish` refuses such runs.

Models:
- fake-good:  always correct, always valid.
- fake-flaky: correct; first JSON reply uses confidence=85 (percent) -> forces exactly one retry.
- fake-sim-small / fake-sim-mid / fake-sim-large: stochastic but deterministic per (model, mode, question),
  with different accuracy levels, occasional format errors and over-confident answers - so charts and
  statistics have realistic-looking (but meaningless) input.
"""

from __future__ import annotations

import hashlib
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
SIM = {"fake-sim-small": 0.62, "fake-sim-mid": 0.78, "fake-sim-large": 0.86}
MODELS = ["fake-good:latest", "fake-flaky:latest", *(f"{m}:latest" for m in SIM)]
_loaded: set[str] = set()


def _u(*parts: object) -> float:
    """Deterministic uniform [0, 1) from the inputs."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


def _mode(fmt) -> str:
    return "text" if fmt in (None, "") else "json" if fmt == "json" else "schema"


def _render(mode: str, answer: str, conf, page, extra_key: bool = False) -> str:
    if mode == "text":
        lines = [f"Answer: {answer}"]
        if conf is not None:
            lines.append(f"Confidence: {conf}")
        lines.append(f"Source page: {page}")
        return "\n".join(lines)
    obj = {"answer": answer, "confidence": conf, "source_page": page}
    if extra_key:
        obj["reasoning"] = "Found in the context."
    return json.dumps(obj)


def _reply(model: str, messages: list[dict], fmt) -> str:
    m = re.search(r"Question: (.+)$", messages[0]["content"], re.S)
    q = QUESTIONS.get(m.group(1).strip() if m else "")
    mode = _mode(fmt)
    retry = len(messages) > 1
    base = model.split(":")[0]
    if q is None:
        return _render(mode, "Not found in the document.", 0.5, 0)

    if base in ("fake-good", "fake-flaky"):
        ans = q["expected_answer"] if q.get("answerable", True) else "Not found in the document."
        page = q.get("expected_page") or 0
        conf = 85 if (base == "fake-flaky" and mode != "text" and not retry) else 0.9
        return _render(mode, ans, conf, page)

    p_correct = SIM[base] - {"text": 0.0, "json": 0.04, "schema": 0.02}[mode]
    correct = _u(base, mode, q["id"], "c") < p_correct
    if q.get("answerable", True):
        ans = q["expected_answer"] if correct else "It is 99 days according to the policy."
        page = (q.get("expected_page") or 0) if correct else 2
    else:
        ans = "Not found in the document." if correct else "It is 12 days per year."
        page = 0 if correct else 3
    conf = round(min(0.99, 0.9 + 0.08 * _u(base, q["id"], "conf")) if _u(base, q["id"], "k") < 0.7
                 else 0.5 + 0.4 * _u(base, q["id"], "conf2") + (0.1 if correct else 0), 2)  # fmt: skip
    if retry:
        return _render(mode, ans, conf, page)
    r = _u(base, mode, q["id"], "fmt")
    if mode == "text" and r < 0.08:
        return _render(mode, ans, None, page)  # forgot the confidence line
    if mode == "json" and r < 0.12:
        return _render(mode, ans, int(conf * 100), page)  # percent instead of 0-1
    if mode == "json" and r < 0.17:
        return _render(mode, ans, conf, page, extra_key=True)
    return _render(mode, ans, conf, page)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{
                "name": m, "model": m, "size": 2 * 1024**3, "digest": hashlib.sha256(m.encode()).hexdigest(),
                "details": {"family": "simulated", "parameter_size": "0B", "quantization_level": "SIM",
                            "format": "gguf"},
            } for m in MODELS]})  # fmt: skip
        elif self.path == "/api/ps":
            self._send({"models": [{"name": m, "model": m, "size": 2 * 1024**3, "size_vram": 0} for m in _loaded]})
        elif self.path == "/api/version":
            self._send({"version": "SIMULATED"})
        else:
            self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = payload["model"]
        full = model if ":" in model else f"{model}:latest"
        if full not in MODELS:
            self._send({"error": f"model '{model}' not found"}, status=404)
            return
        if self.path == "/api/generate":
            if payload.get("keep_alive") == 0:
                _loaded.discard(full)
            self._send({"model": model, "response": "", "done": True})
        elif self.path == "/api/chat":
            _loaded.add(full)
            mode = _mode(payload.get("format"))
            time.sleep(0.002 * (1 + {"text": 0, "json": 0.3, "schema": 0.5}[mode]))
            content = _reply(model, payload["messages"], payload.get("format"))
            self._send({
                "model": model, "created_at": "2026-01-01T00:00:00Z", "done": True,
                "message": {"role": "assistant", "content": content},
                "total_duration": 20_000_000, "load_duration": 1_000_000,
                "prompt_eval_count": 300, "eval_count": len(content) // 4, "eval_duration": 15_000_000,
            })  # fmt: skip
        else:
            self.send_error(404)


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


if __name__ == "__main__":
    srv, url = start(11500)
    print(f"SIMULATED Ollama on {url}")
    srv.serve_forever()
