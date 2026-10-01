"""Benchmark CLI.

python -m benchmark run                       # models x {text, json, schema} x questions
python -m benchmark run --models phi3 --limit 3 --run-id smoke
python -m benchmark run --resume 20261001_101500
python -m benchmark analyze results/<run_id>  # stats, calibration, charts, report.md
python -m benchmark publish results/<run_id>  # charts -> docs/, dashboard data, README results block
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import egress
from .runner import DEFAULT_MODES, run_experiment

DEFAULT_MODELS = ["phi3", "mistral", "gemma2:9b"]
log = logging.getLogger("benchmark")


def _latest(results: Path) -> Path:
    runs = sorted(p for p in results.iterdir() if (p / "records.jsonl").exists())
    if not runs:
        raise SystemExit(f"No runs in {results}/ - run `python -m benchmark run` first")
    return runs[-1]


def _resolve_run(arg: str | None) -> Path:
    if arg in (None, "latest"):
        return _latest(Path("results"))
    p = Path(arg)
    return p if p.exists() else Path("results") / arg


def cmd_run(a: argparse.Namespace) -> int:
    models = a.models
    if a.model_group:
        groups = json.loads(Path("configs/models.json").read_text())["groups"]
        models = [m["tag"] for m in groups[a.model_group]["models"]]
    if not a.allow_egress:
        egress.install()
        log.info("Egress guard ON: this process can only connect to loopback addresses")
    run_dir = run_experiment(
        models=models,
        modes=a.modes,
        doc=Path(a.doc),
        questions_path=Path(a.questions),
        out_root=Path("results"),
        run_id=a.resume or a.run_id,
        limit=a.limit,
        seed=a.seed,
    )
    print(f"\nRun complete: {run_dir}")
    print(f"1) Review {run_dir / 'grading.csv'} - fill manual_correct (1/0) where the auto-grade is wrong")
    print(f"2) python -m benchmark analyze {run_dir}")
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    from assistant import OllamaLLM

    from .runner import load_registry

    try:
        pulled = OllamaLLM().installed_models()
    except Exception as exc:
        pulled = set()
        print(f"(Ollama not reachable: {exc})")
    for name, group in load_registry()["groups"].items():
        print(f"\n{name}: {group['description']}")
        for m in group["models"]:
            mark = "pulled " if m["tag"] in pulled else "MISSING"
            print(f"  [{mark}] {m['tag']:<36} {m.get('params', ''):>6}  ~{m.get('download_gb', '?')} GB")
        missing = [m["tag"] for m in group["models"] if m["tag"] not in pulled]
        if missing:
            print("  pull with: " + " ; ".join(f"ollama pull {t}" for t in missing))
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    from .report import analyze_run

    run_dir = _resolve_run(a.run)
    out = analyze_run(run_dir, make_charts=not a.no_charts)
    print(out["report_md"].read_text(encoding="utf-8"))
    print(f"\nWrote {out['analysis_json']}, {out['report_md']} and charts in {run_dir / 'charts'}")
    return 0


def cmd_publish(a: argparse.Namespace) -> int:
    from .report import publish_run

    run_dir = _resolve_run(a.run)
    publish_run(run_dir, allow_simulated=a.allow_simulated)
    print("Published: docs/assets/*.png, docs/data/results.js, README results block updated")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the experiment matrix")
    r.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    r.add_argument("--model-group", help="use a group from configs/models.json (e.g. baseline_2024, current_2026)")
    r.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=DEFAULT_MODES)
    r.add_argument("--doc", default="data/sample/handbook.pdf")
    r.add_argument("--questions", default="data/questions.jsonl")
    r.add_argument("--limit", type=int, help="only the first N questions (smoke tests)")
    r.add_argument("--run-id")
    r.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    r.add_argument("--seed", type=int, default=42, help="controls question/mode order")
    r.add_argument("--allow-egress", action="store_true", help="disable the loopback-only network guard")
    r.set_defaults(func=cmd_run)

    md = sub.add_parser("models", help="show the model registry and which models are pulled")
    md.set_defaults(func=cmd_models)

    an = sub.add_parser("analyze", help="statistics, calibration, charts and report for a run")
    an.add_argument("run", nargs="?", default="latest")
    an.add_argument("--no-charts", action="store_true")
    an.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("publish", help="copy a run's results into docs/ and the README")
    pb.add_argument("run", nargs="?", default="latest")
    pb.add_argument("--allow-simulated", action="store_true", help=argparse.SUPPRESS)
    pb.set_defaults(func=cmd_publish)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
