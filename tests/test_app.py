"""Streamlit app smoke tests against the SIMULATED Ollama server (no models needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from benchmark import egress  # noqa: E402
from tests import fake_ollama  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def fake_host(monkeypatch):
    server, url = fake_ollama.start()
    monkeypatch.setenv("OLLAMA_HOST", url)
    yield url
    server.shutdown()
    egress.uninstall()


def test_app_answers_a_question(fake_host):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Ollama connected" in at.sidebar.success[0].value
    at.sidebar.selectbox[0].select("fake-good").run()
    at.text_input(key="question").input("What is the minimum password length?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "14 characters" in page
    assert any(m.label == "Source page" and m.value == "4" for m in at.metric)
    assert any(m.label == "Valid format" and m.value == "✅" for m in at.metric)


def test_app_without_ollama_shows_help(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(APP, default_timeout=60).run()
    egress.uninstall()
    assert not at.exception
    assert "Ollama not reachable" in at.sidebar.error[0].value


def test_results_tab_renders_real_run(fake_host):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any(m.label == "Task accuracy" for m in at.metric)
