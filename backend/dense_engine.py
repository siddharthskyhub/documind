import json
import logging
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .config import OLLAMA_BASE_URL, EMBED_MODEL, INDEX_DIR
from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class DenseEngine:
    def __init__(self, model_name: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.chunk_ids: List[str] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.metadata_lookup: Dict[str, DocumentChunk] = {}
        self.storage_file = INDEX_DIR / "dense_index.npz"
        self.metadata_file = INDEX_DIR / "dense_metadata.json"
        self.load_index()

    def get_embedding(self, text: str) -> np.ndarray:
        """Fetch embedding vector from Ollama nomic-embed-text or fallback."""
        try:
            res = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=15
            )
            if res.status_code == 200:
                vec = np.array(res.json()["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / (norm + 1e-10)
        except Exception as e:
            logger.warning(f"Ollama embedding call failed: {e}. Falling back to pseudo-hash embedding.")
        
        # Deterministic lightweight fallback (e.g. if Ollama is restarted)
        dim = 768
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for idx, word in enumerate(words):
            h = hash(word) % dim
            vec[h] += 1.0 / (idx + 1)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Compute embeddings and append chunks to dense index."""
        if not chunks:
            return

        new_embeddings = []
        new_ids = []
        for chunk in chunks:
            emb = self.get_embedding(chunk.text)
            new_embeddings.append(emb)
            new_ids.append(chunk.chunk_id)
            self.metadata_lookup[chunk.chunk_id] = chunk

        new_mat = np.array(new_embeddings, dtype=np.float32)

        if self.embeddings_matrix is None or len(self.chunk_ids) == 0:
            self.embeddings_matrix = new_mat
            self.chunk_ids = new_ids
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_mat])
            self.chunk_ids.extend(new_ids)

        self.save_index()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        """
        Search by cosine similarity.
        Returns list of (chunk_id, cosine_score, rank_1_indexed).
        """
        if self.embeddings_matrix is None or len(self.chunk_ids) == 0:
            return []

        query_vec = self.get_embedding(query)
        # Cosine similarity (vectors are normalized)
        scores = np.dot(self.embeddings_matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append((self.chunk_ids[idx], float(scores[idx]), rank))
        return results

    def remove_document(self, doc_id: str):
        """Remove all chunks belonging to a specific document."""
        if self.embeddings_matrix is None:
            return

        keep_indices = [
            i for i, cid in enumerate(self.chunk_ids)
            if not cid.startswith(f"{doc_id}_")
        ]

        if len(keep_indices) == len(self.chunk_ids):
            return

        self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]
        if keep_indices:
            self.embeddings_matrix = self.embeddings_matrix[keep_indices]
        else:
            self.embeddings_matrix = None

        self.metadata_lookup = {
            cid: chunk for cid, chunk in self.metadata_lookup.items()
            if not cid.startswith(f"{doc_id}_")
        }
        self.save_index()

    def save_index(self):
        """Persist embeddings matrix and metadata to disk."""
        if self.embeddings_matrix is not None:
            np.savez_compressed(
                str(self.storage_file),
                matrix=self.embeddings_matrix,
                ids=np.array(self.chunk_ids)
            )
            # Save metadata
            meta_json = {cid: chunk.model_dump() for cid, chunk in self.metadata_lookup.items()}
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta_json, f, indent=2)

    def load_index(self):
        """Load embeddings and metadata if available."""
        if self.storage_file.exists() and self.metadata_file.exists():
            try:
                data = np.load(str(self.storage_file), allow_pickle=True)
                self.embeddings_matrix = data["matrix"]
                self.chunk_ids = list(data["ids"])
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    meta_json = json.load(f)
                self.metadata_lookup = {
                    cid: DocumentChunk(**val) for cid, val in meta_json.items()
                }
            except Exception as e:
                logger.error(f"Failed to load dense index: {e}")
