"""End-to-end: real HTTP client -> fake Ollama server -> benchmark CSV -> summary table."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmark import run_benchmark
from tests import fake_ollama


@pytest.fixture()
def fake_host(monkeypatch):
    server, url = fake_ollama.start()
    monkeypatch.setenv("OLLAMA_HOST", url)
    yield url
    server.shutdown()


def test_cli_plain_and_json(fake_host, capsys):
    from assistant.cli import main

    assert (
        main(["--doc", "data/sample/handbook.pdf", "--model", "fake-good", "What is the minimum password length?"]) == 0
    )
    assert "14 characters" in capsys.readouterr().out
    assert (
        main(
            [
                "--doc",
                "data/sample/handbook.pdf",
                "--model",
                "fake-flaky",
                "--json",
                "How much is the weekly on-call allowance?",
            ]
        )
        == 0
    )
    out = capsys.readouterr()
    assert '"source_page": 5' in out.out and "attempts=2" in out.err


def test_benchmark_pipeline(fake_host, tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark, "wait_for_memory_to_settle", lambda: 0)
    run_benchmark.main(["--models", "fake-good", "fake-flaky", "missing-model", "--out-dir", str(tmp_path)])

    raw = pd.read_csv(next(tmp_path.glob("raw_results_*.csv")))
    assert len(raw) == 60  # 30 questions x 2 working models; missing model skipped, run continues
    assert raw.groupby("model")["auto_correct"].mean().eq(1.0).all()
    flaky = raw[raw.model == "fake-flaky"]
    assert flaky.first_attempt_valid.sum() == 0 and flaky.final_valid.sum() == 30

    summary_md = next(tmp_path.glob("summary_*.md")).read_text()
    assert "Overall Score" in summary_md and "fake-good" in summary_md
    assert (tmp_path / next(tmp_path.glob("environment_*.json")).name).exists()
