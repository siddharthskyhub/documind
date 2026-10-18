import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .dense_engine import DenseEngine
from .sparse_engine import SparseBM25Engine, tokenize
from .document_processor import DocumentChunk
from .config import DEFAULT_TOP_K, DEFAULT_RRF_K, DEFAULT_HYBRID_ALPHA

class ScoredChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    chunk_index: int
    text: str
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    final_rank: int = 1
    matched_keywords: List[str] = []

class HybridRetriever:
    def __init__(self, dense_engine: DenseEngine, sparse_engine: SparseBM25Engine):
        self.dense = dense_engine
        self.sparse = sparse_engine

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Index chunks into both dense and sparse engines."""
        self.dense.add_chunks(chunks)
        self.sparse.add_chunks(chunks)

    def remove_document(self, doc_id: str):
        """Remove document chunks from both engines."""
        self.dense.remove_document(doc_id)
        self.sparse.remove_document(doc_id)

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        rrf_k: int = DEFAULT_RRF_K
    ) -> List[ScoredChunk]:
        """
        Execute retrieval across dense, sparse, or hybrid (RRF) pipeline.
        Modes: 'hybrid', 'dense_only', 'sparse_only'.
        """
        # Fetch candidate pool (retrieve 2x top_k to fuse effectively)
        candidate_pool_size = max(top_k * 2, 20)
        
        dense_results = self.dense.search(query, top_k=candidate_pool_size) if mode in ("hybrid", "dense_only") else []
        sparse_results = self.sparse.search(query, top_k=candidate_pool_size) if mode in ("hybrid", "sparse_only") else []

        # Maps for quick rank/score lookup
        dense_map = {cid: (score, rank) for cid, score, rank in dense_results}
        sparse_map = {cid: (score, rank) for cid, score, rank in sparse_results}

        query_tokens = set(tokenize(query))
        all_chunk_ids = set(dense_map.keys()) | set(sparse_map.keys())

        scored_chunks: List[ScoredChunk] = []

        for cid in all_chunk_ids:
            chunk_obj = self.dense.metadata_lookup.get(cid) or self.sparse.metadata_lookup.get(cid)
            if not chunk_obj:
                continue

            d_score, d_rank = dense_map.get(cid, (None, None))
            s_score, s_rank = sparse_map.get(cid, (None, None))

            # Compute RRF score
            if mode == "hybrid":
                rrf_dense = (alpha / (rrf_k + d_rank)) if d_rank is not None else 0.0
                rrf_sparse = ((1.0 - alpha) / (rrf_k + s_rank)) if s_rank is not None else 0.0
                final_score = rrf_dense + rrf_sparse
            elif mode == "dense_only":
                final_score = d_score if d_score is not None else 0.0
            else:  # sparse_only
                final_score = s_score if s_score is not None else 0.0

            # Find matching keywords in chunk
            chunk_tokens = set(tokenize(chunk_obj.text))
            matched = sorted(list(query_tokens & chunk_tokens))

            scored_chunks.append(ScoredChunk(
                chunk_id=chunk_obj.chunk_id,
                doc_id=chunk_obj.doc_id,
                doc_name=chunk_obj.doc_name,
                page=chunk_obj.page,
                chunk_index=chunk_obj.chunk_index,
                text=chunk_obj.text,
                dense_score=d_score,
                dense_rank=d_rank,
                sparse_score=s_score,
                sparse_rank=s_rank,
                rrf_score=round(final_score, 6),
                final_rank=1,
                matched_keywords=matched
            ))

        # Sort descending by final score
        scored_chunks.sort(key=lambda x: x.rrf_score, reverse=True)

        # Assign final 1-indexed ranks and slice top_k
        top_results = scored_chunks[:top_k]
        for rank, item in enumerate(top_results, start=1):
            item.final_rank = rank

        return top_results
