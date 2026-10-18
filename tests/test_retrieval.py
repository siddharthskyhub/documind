import sys
import pytest
from pathlib import Path

# Ensure documind root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.document_processor import DocumentChunk, chunk_text_sliding_window
from backend.sparse_engine import SparseBM25Engine
from backend.dense_engine import DenseEngine
from backend.hybrid_retriever import HybridRetriever, ScoredChunk
from backend.hallucination_guard import HallucinationGuard

def test_document_chunking():
    sample_text = (
        "Section 1: Distributed Locking.\n\n"
        "Redlock provides distributed locks across Redis clusters with a 3000ms TTL.\n\n"
        "Section 2: Database Model.\n\n"
        "All transactions are isolated using SERIALIZABLE isolation in PostgreSQL."
    )
    chunks = chunk_text_sliding_window(
        text=sample_text,
        doc_id="test_doc",
        doc_name="test.txt",
        chunk_size=120,
        chunk_overlap=30
    )
    assert len(chunks) >= 2
    assert all(c.doc_id == "test_doc" for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)

def test_sparse_bm25_exact_match():
    c1 = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_name="auth.txt",
        page=1,
        chunk_index=0,
        text="The server returns error ERR_AUTH_EXPIRED when OAuth token timestamp has passed.",
        char_count=80
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        doc_id="d1",
        doc_name="auth.txt",
        page=1,
        chunk_index=1,
        text="Rate limiting uses Redis Token Bucket and returns HTTP 429 when capacity is exceeded.",
        char_count=86
    )

    sparse = SparseBM25Engine()
    # Reset internal memory for isolated test
    sparse.corpus_size = 0
    sparse.chunk_ids = []
    sparse.doc_lengths = []
    sparse.doc_freqs = {}
    sparse.term_freqs = []
    sparse.metadata_lookup = {}

    sparse.add_chunks([c1, c2])

    results = sparse.search("ERR_AUTH_EXPIRED", top_k=2)
    assert len(results) >= 1
    assert results[0][0] == "c1"  # c1 must be top rank for exact error code

def test_hybrid_rrf_scoring():
    c1 = DocumentChunk(chunk_id="c1", doc_id="d1", doc_name="doc.txt", page=1, chunk_index=0, text="Redis Redlock distributed lock with 3000ms TTL.", char_count=48)
    c2 = DocumentChunk(chunk_id="c2", doc_id="d1", doc_name="doc.txt", page=1, chunk_index=1, text="PostgreSQL serializable transaction isolation.", char_count=46)

    dense = DenseEngine()
    sparse = SparseBM25Engine()
    
    dense.chunk_ids = []
    dense.embeddings_matrix = None
    dense.metadata_lookup = {}

    sparse.corpus_size = 0
    sparse.chunk_ids = []
    sparse.doc_lengths = []
    sparse.doc_freqs = {}
    sparse.term_freqs = []
    sparse.metadata_lookup = {}

    retriever = HybridRetriever(dense, sparse)
    retriever.index_chunks([c1, c2])

    results = retriever.retrieve(query="Redlock Redis distributed lock", mode="hybrid", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"
    assert results[0].rrf_score > 0.0

def test_hallucination_guard():
    retrieved = [
        ScoredChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_name="arch.md",
            page=1,
            chunk_index=0,
            text="Redlock acquisition timeout is 250 milliseconds and lease duration is 3000 milliseconds.",
            rrf_score=0.03
        )
    ]

    # Valid grounded answer with citation
    valid_answer = "The Redlock lease duration is 3000 milliseconds with a 250ms timeout [Ref: c1]."
    report = HallucinationGuard.verify_grounding(valid_answer, retrieved)
    assert report.is_grounded is True
    assert len(report.verified_citations) == 1
    assert len(report.invalid_citations) == 0

    # Hallucinated answer with non-existent citation
    hallucinated_answer = "The system uses DynamoDB with 50000 transactions per second [Ref: c999]."
    bad_report = HallucinationGuard.verify_grounding(hallucinated_answer, retrieved)
    assert bad_report.is_grounded is False
    assert "c999" in bad_report.invalid_citations
