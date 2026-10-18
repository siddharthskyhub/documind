import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"

# Ensure directories exist
for path in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, SAMPLE_DOCS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# Retrieval & Chunking Settings
DEFAULT_CHUNK_SIZE = 500       # Target characters per chunk
DEFAULT_CHUNK_OVERLAP = 100    # Character overlap between chunks
DEFAULT_TOP_K = 4              # Default number of retrieved chunks
DEFAULT_RRF_K = 60             # Cormack RRF smoothing constant
DEFAULT_HYBRID_ALPHA = 0.5     # 0.0 = Pure Sparse (BM25), 1.0 = Pure Dense (Embeddings)

# Server Settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
