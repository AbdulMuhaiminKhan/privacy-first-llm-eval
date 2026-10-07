"""Robust launcher for the Streamlit app (used by run_app.bat).

- installs missing packages into the current interpreter
- picks the first free port from a list
- starts Streamlit, waits until it answers, then opens the browser
- logs everything to app_log.txt in the project folder
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "app_log.txt"
PORTS = [8765, 8899, 9123, 8502, 8601, 7861]


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def free_port() -> int:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        log(f"port {port} is busy, trying the next one")
    raise SystemExit("No free port found")


def ensure_packages() -> None:
    missing = [m for m in ("streamlit", "ollama", "pydantic", "pypdf", "pandas", "matplotlib")
               if importlib.util.find_spec(m) is None]  # fmt: skip
    if not missing:
        log("all packages present")
        return
    log(f"installing missing packages: {missing} (first time can take a few minutes)")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
                       capture_output=True, text=True)  # fmt: skip
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(r.stdout[-4000:] + r.stderr[-4000:])
    if r.returncode != 0:
        raise SystemExit("pip install failed - see app_log.txt")
    log("install finished")


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    log(f"python: {sys.executable} ({sys.version.split()[0]})")
    ensure_packages()
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log(f"starting Streamlit on {url}")
    out = LOG.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(port),
         "--server.address", "127.0.0.1", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
    )  # fmt: skip
    for _ in range(90):
        if proc.poll() is not None:
            log(f"Streamlit exited early with code {proc.returncode} - see the lines above in app_log.txt")
            raise SystemExit(1)
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        log("Streamlit did not answer within 90 s")
    log(f"READY - opening {url}")
    webbrowser.open(url)
    print(f"\n  App running at {url}\n  Keep this window open. Close it (or press Ctrl+C) to stop.\n", flush=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
