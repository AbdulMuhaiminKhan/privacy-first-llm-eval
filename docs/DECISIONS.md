# Decision log

Each entry: what was chosen, what was rejected, and why.

## Model choice (pilot)

### D0: Pilot recommendation: Phi-3

*From pilot run `20261001_075634` (JSON mode, 30 questions, RTX 4070 Laptop GPU). The controlled experiment may revise it.*

**Decision:** ship Phi-3 as the default model.

**Why accuracy is a gate and not just a weight:** a weighted sum alone rewards small models heavily. A model that gets 1 in 4 answers wrong could still top the table if it's fast and small. So the rule is: **highest overall score among models with ≥ 80% accuracy**.

**Going in, I expected Mistral 7B to win, and the data said otherwise.** Phi-3 clears the gate at 90%:
- **Accuracy:** 90%, only **3 pp** (one question out of 30) behind Gemma 2 9B, well inside the noise of a 30-question set. It beat Mistral by **10 pp**.
- **Memory:** **3.5 GB**, **46% less** than Gemma's 6.6 GB.
- **Latency:** **~1.1 s** median, **2.5× faster** than Gemma (2.6 s); **68 tok/s** vs 20.
- **Structured output:** 100% valid JSON on the first try, the same as the larger models.

**The cost of choosing Phi-3:**
- **Weaker citations:** **77%** correct source pages vs 93% for Gemma.
- **Worse tail latency:** p95 **6.4 s** vs 4.0 s.
- **The only arithmetic error** in the run: 28 + 2 Focus Days → "33 days".

**When I would choose differently:** choose Gemma 2 9B when citations must be verifiable (legal, medical, audit) or response time must be consistent. On this evidence, there is no case for Mistral 7B.

## Experiment design

**E1: Output mode is the only thing that changes between conditions.** All three modes request the same three fields with identical instructions, context and question; a test (`test_modes_share_everything_but_the_format_block`) enforces this. *Rejected:* comparing a bare "just answer" prompt to a JSON prompt. That changes the task (asking for confidence and a citation) and the format at the same time, so any effect would be uninterpretable.

**E2: Paired design plus paired statistics.** Every model answers every question in every mode. Comparisons use a paired bootstrap and McNemar's exact test on discordant pairs. *Rejected:* comparing two independent accuracy percentages. With 60 questions, that is far less sensitive, because between-question difficulty variance swamps the mode effect.

**E3: Seeded interleaving of modes per question.** Running all `text`, then all `json`, then all `schema` would line up thermal throttling or background load with mode and create fake latency effects.

**E4: Two accuracy metrics.** *Task accuracy* grades content even when the format broke, which isolates "did structure make the model worse?". *End-to-end accuracy* also requires valid output, which is "what does the application receive?". Reporting only one of them would hide either the reasoning effect or the reliability effect.

**E5: The same validation contract in every mode.** Text mode is parsed and then validated with the same Pydantic `Answer` model, so "valid" means the same thing everywhere. Text mode gets no retries, because a free-text pipeline has nothing to validate against; that asymmetry is the point of the comparison.

**E6: Fixed findings rules.** A difference is reported only if the 95% CI excludes 0 and McNemar p < 0.05. Everything else is "no detectable difference at this sample size". The text is generated from the numbers, so no one (including me) can over-claim in the README.

**E7: Fictional corpus plus unanswerable questions.** Invented facts mean pretraining knowledge can't help; unanswerable questions measure hallucination, which accuracy alone hides.

**E8: Simulated data can't be published.** The CI test server reports `ollama_version = "SIMULATED"`, and `publish` refuses such runs. The pipeline is fully tested without a single fabricated number reaching the README or dashboard.

**E9: Latency is compared on cold prompt-cache calls (found in the real run).** Ollama reuses a processed prompt prefix. The json and schema prompts are identical, so the second of the two for each question reused the first one's work: Gemma 2 took 6.2 s cold vs 2.2 s warm. Naively, JSON looked 18% *faster* than text; cold-only, it is 0.93–1.34× slower. The analysis now flags warm calls from run order, and new runs prepend a per-request ID so no two prompts share a prefix.

**E10: Parser strictness is reported, not hidden (found in the real run).** 15 of the 21 plain-text format failures only lacked the `Answer:` label. Reporting both a strict and a lenient parser stops the reliability result from depending on an arbitrary parsing choice.

**E11: Re-grading with the current grader.** `analyze` re-applies the current auto-grader to old runs, so grader fixes apply retroactively, and it reports how many grades changed. The real run exposed a shortcut that treated `source_page == 0` as a refusal: Phi-3's "Business casual." was graded correct. The shortcut is removed and has a regression test.

## Engineering

**D1: Ollama over llama.cpp / vLLM / HF Transformers.** Ollama gives one binary, a REST API, model management, GGUF quantisation, `format=json` and schema-constrained decoding. vLLM targets GPU throughput serving. Transformers in fp16 needs about 4× the memory of Q4.

**D2: 4-bit quantisation by default.** About 4× less memory than fp16 for a small quality loss. The `quantization_phi3` group measures that loss directly (Q2_K / Q4_K_M / Q8_0).

**D3: BM25 retrieval instead of embeddings.** It keeps a second model out of RAM, is deterministic, and is strong on exact terms such as names, numbers and policy keywords. CI asserts the gold page is in the top-4 for 100% of answerable questions, so retrieval is not a confounder.

**D4: Pydantic validation with error-feedback retries (max 3).** The model sees its own invalid output plus the exact validation errors. It catches wrong shapes, percent-style confidence (`85`), extra keys and **hallucinated page numbers** (`source_page` > page count).

**D5: Single user turn, temperature 0, fixed seed, explicit `num_ctx`.** Gemma 2 has no system role, so every model gets the same single-turn prompt. Determinism makes runs reproducible. An explicit context window prevents silent truncation of the retrieved context.

**D6: Reasoning models run with `think=false`.** Qwen3 and Gemma 4 can emit hidden reasoning, which would give them a different output budget and break parsing. Disabling it keeps the comparison fair; `<think>` blocks are also stripped defensively.

**D7: JSONL records plus resumable runs.** One line per answer, appended immediately. A crash loses at most one answer, and `--resume` continues. Missing values are `null`, never estimated.

**D8: Accuracy as a gate in the original portfolio score.** The brief's weighted score (60% accuracy, 20% speed, 20% RAM) rewards small models heavily. With the brief's own reference numbers, Phi-3 (73% accuracy) outscores Mistral (83%), so the recommendation also needs an accuracy floor.
