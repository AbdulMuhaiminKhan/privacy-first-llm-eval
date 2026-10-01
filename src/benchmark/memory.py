"""RAM measurement for the Ollama server + model runner processes (psutil).

Ollama runs the model in a child "runner" process (`ollama runner ...`, `ollama_llama_server` on
older builds, `ollama.exe` on Windows). We sum RSS over every process whose name contains "ollama"
and sample it on a background thread, because peak usage happens *during* generation.

Caveats worth stating in the README:
- With a GPU, weights live in VRAM and RSS under-reports; use `ollama ps` (size_vram) alongside.
- Weights are memory-mapped; RSS counts pages actually touched, which after a full answer is ~all.
"""

from __future__ import annotations

import threading
import time

import psutil

GB = 1024**3


def ollama_processes() -> list[psutil.Process]:
    procs = []
    for p in psutil.process_iter(["name"]):
        name = (p.info.get("name") or "").lower()
        if "ollama" in name:
            procs.append(p)
    return procs


def ollama_rss_bytes() -> int:
    total = 0
    for p in ollama_processes():
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


class PeakMemorySampler:
    """Context manager that records peak summed RSS of Ollama processes.

    with PeakMemorySampler() as mem:
        run_inference()
    mem.peak_bytes
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, ollama_rss_bytes())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakMemorySampler:
        self.peak_bytes = ollama_rss_bytes()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak_bytes = max(self.peak_bytes, ollama_rss_bytes())


def wait_for_memory_to_settle(timeout_s: float = 15.0, tolerance_bytes: int = 50 * 1024**2) -> int:
    """After unloading a model, wait until Ollama RSS stops dropping; returns the settled value."""
    deadline = time.monotonic() + timeout_s
    prev = ollama_rss_bytes()
    while time.monotonic() < deadline:
        time.sleep(1.0)
        cur = ollama_rss_bytes()
        if abs(cur - prev) < tolerance_bytes:
            return cur
        prev = cur
    return prev
