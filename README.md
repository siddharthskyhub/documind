# 🧠 DocuMind — Hybrid Search RAG Engine

> **Enterprise Technical Documentation RAG with Reciprocal Rank Fusion (RRF) & Anti-Hallucination Guardrails.**  
> Powered by Local **Ollama** (`llama3.2` & `nomic-embed-text`), **FastAPI**, **BM25**, and **Vector Embeddings**.

---

## 🌟 Why DocuMind?

Standard RAG implementations rely on naive vector search over chunks. While effective for semantic concepts, **vector search consistently fails on exact technical queries** — such as specific error codes (`ERR_AUTH_EXPIRED`), function names, part IDs, and architectural constants.

**DocuMind** solves this by unifying:
1. **Dense Semantic Retrieval**: Dense vector embeddings (`nomic-embed-text`) capturing high-level conceptual meaning.
2. **Sparse Lexical Retrieval**: Best Matching 25 (**BM25**) with Robertson-Spärck Jones IDF scoring for exact keyword matches.
3. **Reciprocal Rank Fusion (RRF)**: Mathematical rank aggregation merging dense and sparse scores with zero score scale calibration needed.
4. **Anti-Hallucination Verification Guardrail**: Enforces strict sentence-level citation mapping and refutes claims unsupported by retrieved context chunks.

---

## 🏗️ Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │               User Query                     │
                       └──────────────────────┬───────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │   Dense Vector Search   │                       │    Sparse BM25 Search   │
        │    (nomic-embed-text)   │                       │     (Okapi BM25 k1/b)   │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     │  Ranked Dense Chunks       Ranked Sparse Chunks │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Reciprocal Rank Fusion (RRF)    │
                             │  RRF(d) = α/(k+r_d) + (1-α)/(k+r_s)
                             └────────────────┬────────────────┘
                                              │ Top-K Scored Chunks
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Strict Grounding Prompt Builder │
                             │   (Chunk IDs + Refusal Policy)  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │      Ollama (Llama 3.2)         │
                             │    Context-Grounded Answer      │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Anti-Hallucination Guardrail   │
                             │  - Verify [Ref: ID] Citations   │
                             │  - Sentence Faithfulness Score  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Interactive Web Dashboard    │
                             └─────────────────────────────────┘
```

---

## 🚀 Quick Start (One-Click)

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) running locally with:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch App
```bash
python run.py
```
*DocuMind automatically checks your local Ollama instance, pre-loads the built-in system architecture documents, and opens `http://127.0.0.1:8000` in your default browser!*

---

## 🧪 Run Automated Tests
```bash
pytest tests/test_retrieval.py -v
```

All 4 test suites validate:
- Sliding-window paragraph chunking with metadata retention
- Exact token BM25 retrieval for error codes
- RRF rank calculation across dense and sparse engines
- Anti-hallucination citation verification and invalid citation rejection

---

## 📡 REST API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload `.pdf`, `.docx`, `.txt`, `.md` and auto-index into vector store + BM25 |
| `POST` | `/api/query` | Run hybrid search with customizable `top_k`, `alpha`, and `temperature` |
| `GET` | `/api/stats` | View indexed documents, chunk counts, and active models |
| `DELETE` | `/api/documents/{doc_id}` | Remove a document and its chunks from all indexes |
| `POST` | `/api/load-samples` | 1-click loading of sample technical specifications |
| `GET` | `/api/health` | Check engine status, Ollama connection, and model readiness |

---

## 👤 Author
**Siddharth Kumar**  
- GitHub: [@siddharthskyhub](https://github.com/siddharthskyhub)  
- Prior Experience: Generative AI Intern at DRDO–CEMILAC
