"""End-to-end against the SIMULATED Ollama server: CLI, experiment matrix, analysis, publish, privacy guard."""

from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path

import pytest

from benchmark import egress
from benchmark import records as rec_io
from benchmark.__main__ import main as bench_main
from benchmark.report import START, analyze_run, publish_run
from tests import fake_ollama

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def fake_host(monkeypatch):
    server, url = fake_ollama.start()
    monkeypatch.setenv("OLLAMA_HOST", url)
    yield url
    server.shutdown()


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """A scratch copy of the repo layout, so runs never touch the real results/ docs/ README."""
    for d in ("data", "configs", "docs"):
        shutil.copytree(ROOT / d, tmp_path / d)
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("benchmark.runner.wait_for_memory_to_settle", lambda: 0)
    return tmp_path


def test_cli_plain_and_json(fake_host, capsys):
    from assistant.cli import main

    doc = str(ROOT / "data/sample/handbook.pdf")
    assert main(["--doc", doc, "--model", "fake-good", "What is the minimum password length?"]) == 0
    assert "14 characters" in capsys.readouterr().out
    assert main(["--doc", doc, "--model", "fake-flaky", "--json", "How much is the weekly on-call allowance?"]) == 0
    out = capsys.readouterr()
    assert '"source_page": 5' in out.out and "attempts=2" in out.err


def test_experiment_matrix_and_resume(fake_host, workdir):
    egress.uninstall()
    rc = bench_main(["run", "--models", "fake-good", "fake-flaky", "missing-model", "--run-id", "t1", "--limit", "8"])
    egress.uninstall()
    assert rc == 0
    run = workdir / "results/t1"
    recs = rec_io.load(run / "records.jsonl")
    assert len(recs) == 2 * 3 * 8  # 2 working models x 3 modes x 8 questions; missing model skipped, not fatal
    env = json.loads((run / "environment.json").read_text())
    assert "missing-model" in env["skipped_models"]
    assert env["models"]["fake-good"]["digest"]

    flaky_json = [r for r in recs if r["model"] == "fake-flaky" and r["output_mode"] == "json"]
    assert all(
        r["retry_count"] == 1 and r["parse_success"] and not r["first_attempt_parse_success"] for r in flaky_json
    )
    flaky_text = [r for r in recs if r["model"] == "fake-flaky" and r["output_mode"] == "text"]
    assert all(r["retry_count"] == 0 for r in flaky_text)  # text mode never retries
    assert all(r["correct_auto"] for r in recs if r["model"] == "fake-good")

    # modes are interleaved per question, not run in blocks
    seq = [r["output_mode"] for r in sorted(recs, key=lambda r: r["order_index"]) if r["model"] == "fake-good"]
    assert seq[:3] != ["text", "text", "text"] and len(set(seq[:3])) == 3

    # resume: nothing is re-run
    bench_main(["run", "--models", "fake-good", "--resume", "t1", "--limit", "8"])
    egress.uninstall()
    assert len(rec_io.load(run / "records.jsonl")) == len(recs)


def test_manual_grades_override_auto(fake_host, workdir):
    bench_main(["run", "--models", "fake-good", "--run-id", "t2", "--limit", "3", "--modes", "text"])
    egress.uninstall()
    run = workdir / "results/t2"
    lines = (run / "grading.csv").read_text(encoding="utf-8-sig").splitlines()
    header, first = lines[0], lines[1].rsplit(",", 1)[0] + ",0"  # human marks the first answer wrong
    (run / "grading.csv").write_text("\n".join([header, first, *lines[2:]]) + "\n", encoding="utf-8")
    a = analyze_run(run, make_charts=False)["analysis"]
    assert a["meta"]["manual_overrides"] == 1
    assert a["cells"][0]["accuracy"] == pytest.approx(2 / 3)


def test_analysis_charts_and_publish_guard(fake_host, workdir):
    bench_main(["run", "--models", "fake-sim-small", "fake-sim-large", "--run-id", "t3"])
    egress.uninstall()
    run = workdir / "results/t3"
    out = analyze_run(run)
    a = out["analysis"]
    assert a["meta"]["simulated"] is True
    assert {c["mode"] for c in a["cells"]} == {"text", "json", "schema"}
    assert len(out["charts"]) == 8 and all(p.stat().st_size > 10_000 for p in out["charts"].values())
    assert any("Hallucination" in f for f in a["findings"])
    assert a["calibration"]["pooled"]["n"] > 0

    with pytest.raises(SystemExit, match="SIMULATED"):
        publish_run(run)  # simulated data can never reach the README / dashboard by accident
    readme_before = (workdir / "README.md").read_text(encoding="utf-8")
    assert START in readme_before

    publish_run(run, allow_simulated=True)  # test-only escape hatch: output must carry the banner
    readme = (workdir / "README.md").read_text(encoding="utf-8")
    assert "SIMULATED DATA" in readme and "docs/assets/mode_effect.png" in readme
    assert (workdir / "docs/data/results.js").read_text().startswith("window.BENCHMARK_RESULTS = ")


def test_privacy_no_egress_during_full_run(fake_host, workdir):
    """With the guard on, the whole pipeline completes against a loopback server, while any attempt
    to reach a non-loopback address fails."""
    with egress.no_egress():
        with pytest.raises(egress.EgressBlocked):
            socket.create_connection(("93.184.216.34", 80), timeout=1)
        with pytest.raises(egress.EgressBlocked):
            socket.socket().connect(("8.8.8.8", 53))
        from benchmark.runner import run_experiment

        run = run_experiment(
            ["fake-good"], ["text", "json", "schema"], Path("data/sample/handbook.pdf"),
            Path("data/questions.jsonl"), Path("results"), run_id="priv", limit=5,
        )  # fmt: skip
    assert len(rec_io.load(run / "records.jsonl")) == 15
