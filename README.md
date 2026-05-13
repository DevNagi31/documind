# DocuMind — RAG Knowledge Assistant with Evaluation Suite

A retrieval-augmented generation (RAG) system that ingests documents (PDFs, Markdown, code), chunks and embeds them locally, stores in a vector database, and answers questions grounded in your data — with a built-in evaluation framework measuring faithfulness, relevance, and hallucination rate.

## Why This Project

- **RAG is top-3 most in-demand AI skill** — appears in 70%+ of AI engineer job postings (2026)
- Shows: embeddings, vector databases, chunking strategies, prompt engineering, evaluation
- Every company building AI products needs RAG — it's the bridge between generic LLMs and domain-specific intelligence
- Built-in evaluation suite shows you understand production AI quality, not just demos

## The Problem

LLMs hallucinate. Companies need AI that answers from their own data, not training data. RAG solves this, but most implementations are naive — bad chunking, no evaluation, no way to know if answers are actually grounded. DocuMind solves the full pipeline including quality measurement.

## What It Does

```
┌─────────────────────────────────────────────────────┐
│ DocuMind — Knowledge Assistant                       │
│                                                       │
│ 📁 Knowledge Base: 3 collections loaded              │
│    Python Docs (847 chunks) | RFC Specs (234 chunks) │
│    Internal Wiki (1,203 chunks)                       │
│                                                       │
│ 💬 Ask: "How does Python's GIL affect threading?"    │
│                                                       │
│ Answer: Python's GIL (Global Interpreter Lock)       │
│ prevents multiple threads from executing Python       │
│ bytecode simultaneously. For CPU-bound tasks, use    │
│ multiprocessing instead. For I/O-bound tasks,        │
│ threading still helps because the GIL is released    │
│ during I/O operations.                                │
│                                                       │
│ Sources: [python-docs/threading.md:42-68]            │
│          [python-docs/multiprocessing.md:15-30]      │
│                                                       │
│ ┌─────────────────────────────────────────┐          │
│ │ Evaluation Scores                       │          │
│ │ Faithfulness:      0.94 ████████████░   │          │
│ │ Answer Relevance:  0.91 ███████████░░   │          │
│ │ Context Precision: 0.87 ██████████░░░   │          │
│ │ Hallucination:     0.06 █░░░░░░░░░░░░   │          │
│ └─────────────────────────────────────────┘          │
│                                                       │
│ Eval Dashboard    Chunk Explorer    Settings          │
└─────────────────────────────────────────────────────┘
```

### Features

- **Multi-format ingestion** — PDFs, Markdown, plain text, Python/JS code files
- **Smart chunking** — recursive character splitting with overlap, code-aware chunking
- **Local embeddings** — HuggingFace sentence-transformers (no API calls)
- **Vector search** — ChromaDB with metadata filtering and MMR diversity
- **LLM generation** — Groq (Llama 3.3 70B, free) for answer synthesis
- **Evaluation suite** — faithfulness, relevance, context precision, hallucination rate
- **Chunk explorer** — visualize what chunks were retrieved and why
- **A/B testing** — compare chunking strategies and embedding models side-by-side

### Evaluation Metrics (What Sets This Apart)

Most RAG demos just retrieve and generate. DocuMind measures quality:

| Metric | What It Measures | How |
|---|---|---|
| **Faithfulness** | Is the answer supported by retrieved context? | LLM-as-judge compares answer claims to source chunks |
| **Answer Relevance** | Does the answer address the question? | Embedding similarity between question and answer |
| **Context Precision** | Are the retrieved chunks actually relevant? | LLM-as-judge scores each chunk's relevance |
| **Hallucination Rate** | Does the answer contain unsupported claims? | Cross-reference answer statements with sources |

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ Documents│───▶│  Chunking    │───▶│  Embedding   │───▶│ ChromaDB │
│ (PDF,MD) │    │  Engine      │    │  (HuggingFace│    │ (Vector  │
│          │    │              │    │   local)     │    │  Store)  │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘
                                                              │
┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│ Streamlit│◀───│  Groq LLM    │◀───│  Retriever   │◀─────────┘
│ Dashboard│    │  (free)      │    │  (similarity  │
│          │    │              │    │   + MMR)      │
└──────────┘    └──────────────┘    └──────────────┘
      │
      ▼
┌──────────────┐
│  Evaluation  │
│  Suite       │
│  (RAGAS-style│
│   metrics)   │
└──────────────┘
```

## Tech Stack (All Free)

| Component | Tool | Cost |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 80MB) | $0 |
| Vector DB | ChromaDB (local, open source) | $0 |
| LLM | Groq (Llama 3.3 70B, free tier: 6K tokens/min) | $0 |
| Chunking | LangChain text splitters (open source) | $0 |
| PDF Parsing | PyMuPDF / pdfplumber (open source) | $0 |
| Evaluation | Custom RAGAS-inspired metrics (no API needed for embedding-based) | $0 |
| Dashboard | Streamlit on Streamlit Community Cloud | $0 |
| CI/CD | GitHub Actions | $0 |

## Project Structure

```
documind/
├── ingestion/
│   ├── loader.py              # Multi-format document loader
│   ├── chunker.py             # Chunking strategies (recursive, code-aware)
│   └── embedder.py            # HuggingFace embedding wrapper
├── retrieval/
│   ├── vector_store.py        # ChromaDB interface
│   ├── retriever.py           # Similarity + MMR retrieval
│   └── reranker.py            # Cross-encoder reranking (optional)
├── generation/
│   ├── prompts.py             # Prompt templates with source citation
│   └── llm.py                 # Groq client wrapper
├── evaluation/
│   ├── faithfulness.py        # Answer grounded in context?
│   ├── relevance.py           # Answer addresses the question?
│   ├── precision.py           # Retrieved chunks relevant?
│   ├── hallucination.py       # Unsupported claims detection
│   └── benchmark.py           # Run eval suite on test questions
├── dashboard/
│   └── app.py                 # Streamlit chat + eval dashboard
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_evaluation.py
└── README.md
```

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest documents
python -m documind.ingest --path ./docs --collection my_docs

# Launch dashboard
cd dashboard && streamlit run app.py

# Run evaluation
python -m documind.evaluate --collection my_docs --test-set eval_questions.json
```

## Sample Evaluation Output

```
Evaluation Results (50 test questions)
──────────────────────────────────────
Faithfulness:       0.91 (±0.04)
Answer Relevance:   0.88 (±0.06)
Context Precision:  0.85 (±0.05)
Hallucination Rate: 0.09

Chunking Strategy Comparison:
┌─────────────────┬─────────────┬──────────────┐
│ Strategy        │ Faithfulness│ Retrieval F1 │
├─────────────────┼─────────────┼──────────────┤
│ Fixed 512 tokens│ 0.82        │ 0.71         │
│ Recursive 500   │ 0.91        │ 0.85         │
│ Semantic chunks │ 0.89        │ 0.83         │
└─────────────────┴─────────────┴──────────────┘
```

This is the project that shows you can build production-grade AI, not just call an API.
