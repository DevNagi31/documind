"""DocuMind Streamlit dashboard: chat, source citations, eval scores, chunk explorer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make the project root importable when running `streamlit run dashboard/app.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from documind.evaluation.faithfulness import score_faithfulness
from documind.evaluation.hallucination import score_hallucination
from documind.evaluation.precision import score_context_precision
from documind.evaluation.relevance import score_answer_relevance
from documind.generation.llm import GroqClient, generate_answer
from documind.ingestion.embedder import Embedder
from documind.retrieval.retriever import Retriever
from documind.retrieval.vector_store import VectorStore

st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")


@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_llm():
    return GroqClient()


@st.cache_resource
def get_store(collection: str, persist_dir: str):
    return VectorStore(collection=collection, persist_dir=persist_dir)


def list_collections(persist_dir: str) -> list[str]:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir)
        return [c.name for c in client.list_collections()]
    except Exception:
        return []


def render_score(label: str, value: float, lower_is_better: bool = False):
    display = value
    color = "#0a7c2f" if (value >= 0.8 if not lower_is_better else value <= 0.2) else "#b07b00"
    if (value < 0.5 and not lower_is_better) or (value > 0.5 and lower_is_better):
        color = "#a8322a"
    st.markdown(
        f"<div style='display:flex;justify-content:space-between'>"
        f"<span><b>{label}</b></span>"
        f"<span style='color:{color}'><b>{display:.2f}</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, max(0.0, value)))


def main():
    st.title("📚 DocuMind")
    st.caption("RAG knowledge assistant with built-in evaluation")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", str(ROOT / ".chroma"))
    collections = list_collections(persist_dir)

    with st.sidebar:
        st.header("Configuration")
        if collections:
            collection = st.selectbox("Collection", collections)
        else:
            st.warning(
                "No collections found. Ingest documents first:\n\n"
                "`python -m documind.ingest --path ./docs --collection my_docs`"
            )
            collection = st.text_input("Collection name", value="my_docs")

        k = st.slider("Chunks to retrieve (k)", 1, 15, 5)
        use_mmr = st.checkbox("Use MMR (diversity)", value=True)
        mmr_lambda = st.slider("MMR λ (relevance ↔ diversity)", 0.0, 1.0, 0.5)
        run_eval = st.checkbox("Evaluate answer quality", value=True)
        st.divider()
        st.caption(f"Persist dir: `{persist_dir}`")

    tab_chat, tab_chunks, tab_about = st.tabs(["💬 Chat", "🔍 Chunk Explorer", "ℹ️ About"])

    with tab_chat:
        question = st.text_input(
            "Ask a question grounded in your documents:",
            placeholder="How does Python's GIL affect threading?",
        )
        ask = st.button("Ask", type="primary", disabled=not question.strip())

        if ask and question.strip():
            if not collections and collection not in list_collections(persist_dir):
                st.error(f"Collection '{collection}' does not exist. Ingest first.")
                return

            with st.spinner("Retrieving..."):
                store = get_store(collection, persist_dir)
                retriever = Retriever(store, get_embedder())
                chunks = retriever.retrieve(question, k=k, use_mmr=use_mmr, mmr_lambda=mmr_lambda)

            if not chunks:
                st.warning("No chunks retrieved. Is the collection empty?")
                return

            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(question, chunks, client=get_llm())
                except RuntimeError as e:
                    st.error(str(e))
                    return

            left, right = st.columns([3, 1])
            with left:
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources")
                for c in chunks:
                    with st.expander(f"{c.citation()} — score {c.score:.3f}"):
                        st.code(c.text, language=c.metadata.get("doc_type", "text"))

            with right:
                if run_eval:
                    st.subheader("Evaluation")
                    with st.spinner("Scoring..."):
                        try:
                            faith = score_faithfulness(answer, chunks, client=get_llm())
                            rel = score_answer_relevance(question, answer, embedder=get_embedder())
                            prec = score_context_precision(question, chunks, client=get_llm())
                            hall = score_hallucination(answer, chunks, client=get_llm())
                        except RuntimeError as e:
                            st.error(str(e))
                            return
                    render_score("Faithfulness", faith)
                    render_score("Answer Relevance", rel)
                    render_score("Context Precision", prec)
                    render_score("Hallucination", hall, lower_is_better=True)

    with tab_chunks:
        st.subheader("Chunk Explorer")
        if not collections and not collection:
            st.info("Select a collection in the sidebar.")
        else:
            try:
                store = get_store(collection, persist_dir)
                st.metric("Chunks in collection", store.count())
            except Exception as e:
                st.error(f"Failed to open collection: {e}")
                return
            sample_q = st.text_input("Inspect retrieval for a probe query:", key="probe")
            if sample_q:
                retriever = Retriever(store, get_embedder())
                chunks = retriever.retrieve(sample_q, k=k, use_mmr=use_mmr, mmr_lambda=mmr_lambda)
                for c in chunks:
                    with st.expander(f"{c.citation()} — score {c.score:.3f}"):
                        st.json({"metadata": c.metadata})
                        st.code(c.text, language=c.metadata.get("doc_type", "text"))

    with tab_about:
        st.markdown(
            "**DocuMind** is a local-first RAG system: documents are chunked, embedded with "
            "`sentence-transformers/all-MiniLM-L6-v2`, stored in ChromaDB, and answered by "
            "Groq's free-tier Llama 3.3 70B.\n\n"
            "Every answer is scored on four dimensions:\n"
            "- **Faithfulness** — claims supported by context\n"
            "- **Answer Relevance** — addresses the question\n"
            "- **Context Precision** — retrieved chunks were relevant\n"
            "- **Hallucination** — fraction of unsupported claims (lower = better)"
        )


if __name__ == "__main__":
    main()
