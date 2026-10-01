# Resume bullets

Every number comes from run `20261001_075634` (`results/summary_20261001_075634.*`), so each claim can be backed up in an interview.

- Built a **fully offline document Q&A assistant** (Python, Ollama, Pydantic, BM25 retrieval) that answers questions over private PDFs with **zero external API calls and $0 inference cost**, reaching **90% accuracy** at **~1.1 s median latency** on a 30-question grounded eval set.
  *(Source: Accuracy and Speed columns, Phi-3 row)*

- Benchmarked **3 open-weight LLMs** (Phi-3 3.8B, Mistral 7B, Gemma 2 9B; 4-bit GGUF) on accuracy, median/p95 latency, memory and citation accuracy, and designed an accuracy-gated weighted score. Selected **Phi-3**: within **3 pp** of the most accurate model at **46% less memory (3.5 vs 6.6 GB)** and **2.5× lower median latency**. I documented the trade-off against the larger model's higher citation accuracy (93% vs 77%).
  *(Source: summary table plus Decision D0 in docs/DECISIONS.md)*

- Enforced a typed JSON contract with **Pydantic validation and an error-feedback retry loop** (extra keys, out-of-range confidence and nonexistent page citations are all rejected and fed back to the model), achieving **100% schema-valid output across 90 benchmark queries**. Shipped with **20 automated tests** and CI that runs the full pipeline end to end against a mock Ollama server.
  *(Source: Valid JSON 1st try / after retry in summary_*.csv; tests/)*

- Diagnosed a shared failure mode: all three models answered "not found" on policy questions that need inference from omission (for example, a city missing from the hotel table means the standard limit applies), rather than hallucinating. Retrieval was ruled out by checking it in CI.
  *(Source: raw_results_*.csv, q14 and q27)*

Rule: if a number didn't come out of your own run, leave it out. Interviewers ask "how did you measure that?", and the answer has to be this repo.

---

## Controlled experiment (v2): run `20261002_000441`, numbers from `docs/REPORT.md`

- Designed and ran a **controlled offline experiment** (3 open-weight LLMs × 3 output modes × 60 questions = **540 answers**; paired design, bootstrap CIs, McNemar tests) on whether forcing JSON degrades small-model accuracy. Found **no measurable cost: +1.1 pp, 95% CI [−1.1, +3.3]**, which rules out a pooled drop larger than about 1 pp.
  *(Source: Findings → Task accuracy)*

- Quantified the reliability side: structured modes raised end-to-end accuracy by **+11.7 pp** (strict parser) and **+4.4 pp** (lenient parser). I also caught a **prompt-cache artifact** that made JSON look 18% faster; on cold calls it is **0.93–1.34× slower**, and I fixed the benchmark so no run can reuse another's prompt cache.
  *(Source: Findings → End-to-end accuracy, Parser-strictness check, Latency; docs/DECISIONS.md E9)*

- Showed that LLM **self-reported confidence works like a switch, not a probability**: **475/540** answers stated exactly 1.0, and confidence barely separated right from wrong answers (**AUROC 0.58**, excluding refusals). **18 answers were wrong at ≥ 0.9 confidence.** Shipped an interactive dashboard, **43 automated tests** and a CI-verified **no-egress privacy guard**.
  *(Source: Findings → Confidence calibration, Confidence collapse, High-confidence answers)*
