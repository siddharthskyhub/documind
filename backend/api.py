import time
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import UPLOAD_DIR, SAMPLE_DOCS_DIR, EMBED_MODEL, LLM_MODEL, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA
from .document_processor import extract_chunks_from_file
from .dense_engine import DenseEngine
from .sparse_engine import SparseBM25Engine
from .hybrid_retriever import HybridRetriever, ScoredChunk
from .hallucination_guard import HallucinationGuard, GroundingReport
from .llm_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("documind")

# Initialize app
app = FastAPI(
    title="DocuMind - Hybrid Search RAG Engine",
    description="Enterprise Documentation RAG with Reciprocal Rank Fusion & Anti-Hallucination Guardrails",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engines
dense_engine = DenseEngine()
sparse_engine = SparseBM25Engine()
retriever = HybridRetriever(dense_engine, sparse_engine)
llm_client = OllamaClient()

# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # "hybrid" | "dense_only" | "sparse_only"
    top_k: int = DEFAULT_TOP_K
    alpha: float = DEFAULT_HYBRID_ALPHA
    temperature: float = 0.1

class QueryResponse(BaseModel):
    query: str
    answer: str
    mode: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    sources: List[ScoredChunk]
    grounding: GroundingReport

# --- API Endpoints ---

@app.get("/api/health")
def health_check():
    ollama_status = llm_client.check_health()
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "models": {
            "embeddings": EMBED_MODEL,
            "llm": LLM_MODEL
        },
        "indexed_chunks": len(dense_engine.chunk_ids)
    }

@app.get("/api/stats")
def get_stats():
    doc_ids = set(c.doc_id for c in dense_engine.metadata_lookup.values())
    docs_summary = {}
    for c in dense_engine.metadata_lookup.values():
        if c.doc_id not in docs_summary:
            docs_summary[c.doc_id] = {
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "chunks": 0,
                "pages": set()
            }
        docs_summary[c.doc_id]["chunks"] += 1
        docs_summary[c.doc_id]["pages"].add(c.page)

    documents_list = [
        {
            "doc_id": v["doc_id"],
            "doc_name": v["doc_name"],
            "chunks": v["chunks"],
            "page_count": len(v["pages"])
        }
        for v in docs_summary.values()
    ]

    return {
        "total_documents": len(documents_list),
        "total_chunks": len(dense_engine.chunk_ids),
        "documents": documents_list,
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, DOCX, TXT, MD)."""
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process and extract chunks
    try:
        chunks = extract_chunks_from_file(save_path, doc_name=filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text extracted from document.")

        retriever.index_chunks(chunks)
        doc_id = chunks[0].doc_id
        pages = max([c.page for c in chunks])

        return {
            "success": True,
            "message": f"Successfully indexed '{filename}'",
            "doc_id": doc_id,
            "chunks_count": len(chunks),
            "pages_count": pages
        }
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and its chunks from all indexes."""
    retriever.remove_document(doc_id)
    return {"success": True, "message": f"Document '{doc_id}' deleted."}

@app.post("/api/query", response_model=QueryResponse)
def query_knowledge_base(req: QueryRequest):
    """Perform hybrid retrieval, LLM grounded answer synthesis, and citation verification."""
    t0 = time.time()

    # 1. Hybrid Retrieval
    t_ret_start = time.time()
    retrieved_chunks = retriever.retrieve(
        query=req.query,
        mode=req.mode,
        top_k=req.top_k,
        alpha=req.alpha
    )
    retrieval_ms = round((time.time() - t_ret_start) * 1000, 2)

    if not retrieved_chunks:
        return QueryResponse(
            query=req.query,
            answer="No relevant documents or chunks were found in the knowledge base. Please upload documents first.",
            mode=req.mode,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=0.0,
            total_time_ms=round((time.time() - t0) * 1000, 2),
            sources=[],
            grounding=GroundingReport(
                faithfulness_score=1.0,
                is_grounded=True,
                verified_citations=[],
                invalid_citations=[],
                ungrounded_sentences=[],
                warnings=["Knowledge base is empty or no matches found."]
            )
        )

    # 2. Build strictly grounded prompt
    prompt = HallucinationGuard.build_grounded_prompt(req.query, retrieved_chunks)

    # 3. Call LLM
    t_gen_start = time.time()
    raw_answer = llm_client.generate(prompt=prompt, temperature=req.temperature)
    generation_ms = round((time.time() - t_gen_start) * 1000, 2)

    # 4. Anti-Hallucination & Citation Verification
    grounding_report = HallucinationGuard.verify_grounding(raw_answer, retrieved_chunks)

    total_ms = round((time.time() - t0) * 1000, 2)

    return QueryResponse(
        query=req.query,
        answer=raw_answer,
        mode=req.mode,
        retrieval_time_ms=retrieval_ms,
        generation_time_ms=generation_ms,
        total_time_ms=total_ms,
        sources=retrieved_chunks,
        grounding=grounding_report
    )

@app.post("/api/load-samples")
def load_sample_documents():
    """Load built-in technical specs and manuals for instant testing."""
    sample_files = list(SAMPLE_DOCS_DIR.glob("*.*"))
    indexed = []
    for sf in sample_files:
        chunks = extract_chunks_from_file(sf, doc_name=sf.name)
        if chunks:
            retriever.index_chunks(chunks)
            indexed.append({"file": sf.name, "chunks": len(chunks)})
    return {
        "success": True,
        "message": f"Loaded {len(indexed)} sample documents into knowledge base.",
        "documents": indexed
    }

# Mount Frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
