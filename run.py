import sys
import time
import webbrowser
import requests
import uvicorn
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import OLLAMA_BASE_URL, EMBED_MODEL, LLM_MODEL, HOST, PORT, SAMPLE_DOCS_DIR
from backend.document_processor import extract_chunks_from_file
from backend.api import retriever, dense_engine

def verify_ollama():
    print("=" * 60)
    print("🧠 DocuMind — Hybrid Search RAG Engine")
    print("=" * 60)
    print("Checking Ollama status at:", OLLAMA_BASE_URL)
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            print("✅ Ollama is running.")
            print(f"📦 Installed models: {', '.join(models)}")

            has_embed = any(EMBED_MODEL in m for m in models)
            has_llm = any(LLM_MODEL in m for m in models)

            if not has_embed:
                print(f"⚠️ Warning: '{EMBED_MODEL}' model not found in Ollama. Run: `ollama pull {EMBED_MODEL}`")
            if not has_llm:
                print(f"⚠️ Warning: '{LLM_MODEL}' model not found in Ollama. Run: `ollama pull {LLM_MODEL}`")
        else:
            print("⚠️ Ollama returned non-200 status. Engine will run with deterministic fallback.")
    except Exception as e:
        print(f"⚠️ Could not reach Ollama ({e}). Engine will run with deterministic fallback.")

def preload_samples_if_empty():
    if len(dense_engine.chunk_ids) == 0:
        print("📂 Knowledge base is empty. Pre-loading sample architecture documents...")
        sample_files = list(SAMPLE_DOCS_DIR.glob("*.*"))
        for sf in sample_files:
            chunks = extract_chunks_from_file(sf, doc_name=sf.name)
            if chunks:
                retriever.index_chunks(chunks)
        print(f"✅ Pre-loaded {len(dense_engine.chunk_ids)} chunks into index.")
    else:
        print(f"✅ Loaded existing index with {len(dense_engine.chunk_ids)} chunks.")

def main():
    verify_ollama()
    preload_samples_if_empty()
    url = f"http://{HOST}:{PORT}"
    print("=" * 60)
    print(f"🚀 Starting DocuMind Web App on: {url}")
    print("=" * 60)

    # Delay browser opening slightly until server is listening
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("backend.api:app", host=HOST, port=PORT, reload=False, log_level="info")

if __name__ == "__main__":
    main()
