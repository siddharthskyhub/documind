import re
import json
import math
import logging
from typing import List, Dict, Tuple, Set
from pathlib import Path
from .config import INDEX_DIR
from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words and identifiers."""
    # Keeps words, numbers, and technical terms with underscores/hyphens
    tokens = re.findall(r'[a-zA-Z0-9_\-\.]+', text.lower())
    # Filter out pure punctuation or single characters
    return [t for t in tokens if len(t) > 1 or t.isalnum()]

class SparseBM25Engine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids: List[str] = []
        self.corpus_size: int = 0
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # doc_idx -> {term: count}
        self.metadata_lookup: Dict[str, DocumentChunk] = {}
        
        self.storage_file = INDEX_DIR / "sparse_index.json"
        self.load_index()

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Index a list of chunks into the BM25 corpus."""
        if not chunks:
            return

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self.chunk_ids.append(chunk.chunk_id)
            self.metadata_lookup[chunk.chunk_id] = chunk
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)

            # Frequency count for this document
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            # Update document frequencies
            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.corpus_size = len(self.chunk_ids)
        self.avg_doc_len = sum(self.doc_lengths) / (self.corpus_size + 1e-10)
        self.save_index()

    def _idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones IDF with smoothing."""
        df = self.doc_freqs.get(term, 0)
        # Standard Lucene / BM25 smoothed IDF
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        """
        Search by BM25 relevance score.
        Returns list of (chunk_id, bm25_score, rank_1_indexed).
        """
        if self.corpus_size == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            idf = self._idf(term)
            for doc_idx, tf_dict in enumerate(self.term_freqs):
                f = tf_dict.get(term, 0)
                if f > 0:
                    numerator = f * (self.k1 + 1.0)
                    denominator = f + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[doc_idx] / (self.avg_doc_len + 1e-10)))
                    scores[doc_idx] += idf * (numerator / denominator)

        # Sort descending
        ranked_indices = sorted(
            range(self.corpus_size),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] > 0:
                results.append((self.chunk_ids[idx], float(scores[idx]), rank))
        return results

    def remove_document(self, doc_id: str):
        """Remove chunks of a document and rebuild doc frequencies."""
        remaining_chunks = [
            chunk for cid, chunk in self.metadata_lookup.items()
            if not cid.startswith(f"{doc_id}_")
        ]
        self.chunk_ids = []
        self.corpus_size = 0
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        self.doc_freqs = {}
        self.term_freqs = []
        self.metadata_lookup = {}
        
        self.add_chunks(remaining_chunks)

    def save_index(self):
        """Persist BM25 index state to disk."""
        data = {
            "chunk_ids": self.chunk_ids,
            "corpus_size": self.corpus_size,
            "doc_lengths": self.doc_lengths,
            "avg_doc_len": self.avg_doc_len,
            "doc_freqs": self.doc_freqs,
            "term_freqs": self.term_freqs,
            "metadata": {cid: chunk.model_dump() for cid, chunk in self.metadata_lookup.items()}
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_index(self):
        """Load BM25 index state from disk."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunk_ids = data.get("chunk_ids", [])
                self.corpus_size = data.get("corpus_size", 0)
                self.doc_lengths = data.get("doc_lengths", [])
                self.avg_doc_len = data.get("avg_doc_len", 0.0)
                self.doc_freqs = data.get("doc_freqs", {})
                self.term_freqs = data.get("term_freqs", [])
                meta_json = data.get("metadata", {})
                self.metadata_lookup = {
                    cid: DocumentChunk(**val) for cid, val in meta_json.items()
                }
            except Exception as e:
                logger.error(f"Failed to load sparse index: {e}")
