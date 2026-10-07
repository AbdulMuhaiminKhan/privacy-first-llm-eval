"""Streamlit app: ask questions about a private document, fully offline, and explore the benchmark results.

    streamlit run app.py

Privacy: the egress guard blocks every outbound connection from this process except loopback (Ollama),
and .streamlit/config.toml turns off Streamlit's usage statistics.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from assistant import DocumentAssistant, OllamaLLM, Settings  # noqa: E402
from assistant.assistant import OutputMode  # noqa: E402
from assistant.llm import LLMError  # noqa: E402
from benchmark import egress  # noqa: E402

egress.install()  # loopback only, for the whole app process

SAMPLE_DOC = ROOT / "data" / "sample" / "handbook.pdf"
RESULTS = ROOT / "results"
MODE_LABELS = {"text": "Plain text", "json": "JSON mode", "schema": "Schema-constrained"}
SAMPLE_QUESTIONS = [
    "What is the hotel limit per night in Zurich?",
    "How many days off does a full-time employee get in total?",
    "Is a written postmortem required for a SEV3 incident?",
    "What is the name of the company's CEO?",
]

st.set_page_config(page_title="Private Document Q&A", page_icon="🔒", layout="wide")


# ---- cached resources ---------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm() -> OllamaLLM:
    return OllamaLLM(Settings())


@st.cache_resource(show_spinner="Indexing document…")
def get_assistant(doc_bytes: bytes, suffix: str) -> DocumentAssistant:
    digest = hashlib.sha256(doc_bytes).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"pfle_{digest}{suffix}"
    path.write_bytes(doc_bytes)
    return DocumentAssistant(path, llm=get_llm(), settings=Settings())


def list_models() -> tuple[list[str], str | None]:
    try:
        names = get_llm().installed_models()
    except LLMError as exc:
        return [], str(exc)
    tags = sorted({n.removesuffix(":latest") for n in names})
    return tags, None


def list_runs() -> list[Path]:
    if not RESULTS.exists():
        return []
    return sorted((p for p in RESULTS.iterdir() if (p / "analysis.json").exists()), reverse=True)


# ---- sidebar ------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("🔒 Private Document Q&A")
    st.caption("Runs 100% on this machine via Ollama. No API keys, no cloud.")

    models, model_error = list_models()
    if model_error:
        st.error(f"Ollama not reachable.\n\n{model_error}")
        st.code("ollama serve", language="bash")
    else:
        st.success(f"Ollama connected · {len(models)} model(s)")

    preferred = next((m for m in ("phi3", "mistral", "gemma2:9b") if m in models), models[0] if models else None)
    model = st.selectbox("Model", models, index=models.index(preferred) if preferred else 0, disabled=not models)
    mode = st.radio(
        "Output mode",
        [m.value for m in OutputMode],
        index=1,
        format_func=MODE_LABELS.get,
        help="Same instructions in every mode; only the format and decoding constraint change.",
    )

    st.divider()
    source = st.radio("Document", ["Sample handbook (fictional)", "Upload my own"], index=0)
    uploaded = None
    if source == "Upload my own":
        uploaded = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"])
        st.caption("The file stays on this computer.")

    st.divider()
    st.caption("🛡️ Egress guard: on · Usage stats: off")


if uploaded is not None:
    doc_bytes, suffix, doc_name = uploaded.getvalue(), Path(uploaded.name).suffix.lower(), uploaded.name
else:
    doc_bytes, suffix, doc_name = SAMPLE_DOC.read_bytes(), ".pdf", SAMPLE_DOC.name

tab_ask, tab_results, tab_about = st.tabs(["💬 Ask the document", "📊 Benchmark results", "ℹ️ How it works"])

# ---- tab 1: ask -----------------------------------------------------------------------------------------
with tab_ask:
    try:
        assistant = get_assistant(doc_bytes, suffix)
    except Exception as exc:  # unreadable / scanned PDF etc.
        st.error(f"Could not read {doc_name}: {exc}")
        st.stop()

    st.subheader(f"Ask about `{doc_name}`")
    st.caption(f"{assistant.max_page} pages · {len(assistant.index.chunks)} chunks indexed with BM25")

    if uploaded is None:
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, q in zip(cols, SAMPLE_QUESTIONS, strict=True):
            if col.button(q, width="stretch"):
                st.session_state["question"] = q

    with st.form("ask", clear_on_submit=False):
        question = st.text_input("Question", key="question", placeholder="e.g. What is the hotel limit in Zurich?")
        submitted = st.form_submit_button("Ask", type="primary", disabled=not models)

    if submitted and question.strip():
        with st.spinner(f"{model} is reading the relevant pages…"):
            retrieved = assistant.retrieve(question)
            try:
                res = assistant.answer(question, model=model, mode=mode)
            except LLMError as exc:
                st.error(str(exc))
                st.stop()
        latency = sum(s.wall_s for s in res.stats)
        st.session_state.setdefault("history", []).insert(
            0,
            {
                "question": question,
                "model": model,
                "mode": mode,
                "res": res,
                "latency": latency,
                "retrieved": retrieved,
            },  # fmt: skip
        )

    for i, item in enumerate(st.session_state.get("history", [])):
        res = item["res"]
        with st.container(border=True):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"### {res.predicted_answer}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Source page", res.source_page if res.source_page is not None else "–")
            c2.metric(
                "Stated confidence",
                "–" if res.confidence is None else f"{res.confidence:.2f}",
                help="Self-reported by the model. In the benchmark this barely separated right from wrong "
                "answers (AUROC 0.58), so don't rely on it.",
            )
            c3.metric("Latency", f"{item['latency']:.2f} s")
            c4.metric("Valid format", "✅" if res.parse_success else "❌",
                      help=f"Retries used: {res.retry_count}")  # fmt: skip
            st.caption(f"{item['model']} · {MODE_LABELS[item['mode']]} · retries: {res.retry_count}")
            with st.expander("Retrieved context (what the model saw)"):
                for chunk, score in zip(item["retrieved"].chunks, item["retrieved"].scores, strict=True):
                    st.markdown(f"**Page {chunk.page}** · BM25 score {score:.2f}")
                    st.write(chunk.text)
            with st.expander("Raw model output"):
                st.code(res.raw_output or "(empty)", language="json" if item["mode"] != "text" else None)
            if res.error:
                st.warning(res.error)
        if i >= 9:
            break

# ---- tab 2: results -------------------------------------------------------------------------------------
with tab_results:
    runs = list_runs()
    if not runs:
        st.info("No analysed runs yet. Run `python -m benchmark run` then `python -m benchmark analyze`.")
    else:
        run_dir = st.selectbox("Run", runs, format_func=lambda p: p.name)
        a = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
        if a["meta"].get("simulated"):
            st.warning("SIMULATED data from the test server: the numbers mean nothing.")
        env = a["meta"].get("environment") or {}
        st.caption(
            f"{len(a['models'])} models × {len(a['modes'])} modes × {a['n_questions']} questions · "
            f"{env.get('gpu') or env.get('cpu')} · Ollama {env.get('ollama_version')}"
        )

        pooled = {c["mode"]: c for c in a["pooled"]}
        cols = st.columns(len(a["modes"]))
        for col, m in zip(cols, a["modes"], strict=True):
            c = pooled[m]
            col.markdown(f"**{MODE_LABELS.get(m, m)}**")
            col.metric("Task accuracy", f"{c['accuracy']:.0%}")
            col.metric("Valid on first try", f"{c['parse_first_attempt']:.0%}")
            col.metric("Median latency (cold)", f"{c['latency_median_ms'] / 1000:.2f} s")

        hero = run_dir / "charts" / "hero.png"
        if hero.exists():
            st.image(str(hero), width="stretch")

        with st.expander("Key findings (generated from the data)", expanded=True):
            for f in a["findings"]:
                st.markdown(f"- {f}")

        st.subheader("Results matrix")
        matrix = pd.DataFrame(a["cells"])[
            ["model", "mode", "accuracy", "strict_accuracy", "parse_first_attempt", "retry_rate",
             "latency_median_ms", "hallucination_rate"]
        ]  # fmt: skip
        matrix["latency_median_ms"] = (matrix["latency_median_ms"] / 1000).round(2)
        st.dataframe(
            matrix.rename(
                columns={
                    "accuracy": "Task acc.",
                    "strict_accuracy": "End-to-end acc.",
                    "parse_first_attempt": "Valid 1st try",
                    "retry_rate": "Retry rate",
                    "latency_median_ms": "Median latency (s)",
                    "hallucination_rate": "Hallucination",
                }
            ),  # fmt: skip
            hide_index=True,
            width="stretch",
            column_config={
                k: st.column_config.NumberColumn(format="percent")
                for k in ["Task acc.", "End-to-end acc.", "Valid 1st try", "Retry rate", "Hallucination"]
            },  # fmt: skip
        )

        charts = run_dir / "charts"
        c1, c2 = st.columns(2)
        for col, name in ((c1, "mode_effect.png"), (c2, "calibration.png")):
            if (charts / name).exists():
                col.image(str(charts / name), width="stretch")

        st.subheader("Every answer")
        recs = pd.DataFrame(a["records"])
        f1, f2, f3 = st.columns(3)
        mf = f1.multiselect("Model", a["models"], default=a["models"])
        modef = f2.multiselect("Mode", a["modes"], default=a["modes"], format_func=MODE_LABELS.get)
        only = f3.selectbox("Show", ["All", "Wrong only", "Confidently wrong (≥ 0.9)", "Invalid format"])
        view = recs[recs.model.isin(mf) & recs.output_mode.isin(modef)]
        if only == "Wrong only":
            view = view[~view.correct]
        elif only.startswith("Confidently"):
            view = view[(~view.correct) & (view.confidence >= 0.9)]
        elif only == "Invalid format":
            view = view[~view.parse_success]
        st.caption(f"{len(view)} of {len(recs)} answers")
        st.dataframe(
            view[
                [
                    "model",
                    "output_mode",
                    "question",
                    "expected_answer",
                    "predicted_answer",
                    "correct",
                    "confidence",
                    "latency_ms",
                    "retry_count",
                ]
            ],  # fmt: skip
            hide_index=True,
            width="stretch",
        )

# ---- tab 3: about ---------------------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### What happens when you ask a question
1. **Load & chunk:** the document is split into ~180-word chunks that never cross page boundaries.
2. **Retrieve:** BM25 picks the 4 most relevant chunks (shown under *Retrieved context*).
3. **Prompt:** the same instructions in every mode; only the output format differs.
4. **Generate:** a local model served by **Ollama** on `127.0.0.1`.
5. **Validate:** **Pydantic** checks the answer (fields, 0 ≤ confidence ≤ 1, page exists). In JSON/schema
   mode, invalid output is sent back to the model with the errors, up to 3 times.

### Output modes
| Mode | Model produces | Constraint |
|---|---|---|
| Plain text | labelled lines | none |
| JSON mode | a JSON object | valid JSON enforced by Ollama |
| Schema-constrained | a JSON object | decoding restricted to the exact schema |

### Privacy
- This app installs an **egress guard**: any connection to a non-loopback address raises an error.
- Streamlit usage statistics are disabled in `.streamlit/config.toml`.
- Uploaded files are written only to your local temp folder.

### What the benchmark found
Forcing JSON cost no measurable accuracy, made outputs machine-readable, and was not faster once prompt
caching was controlled for. The models' self-reported confidence was 1.0 for almost every answer, so treat
the *Stated confidence* number above with suspicion. See the **Benchmark results** tab.
"""
    )
