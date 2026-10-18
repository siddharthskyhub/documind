import re
import uuid
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from pypdf import PdfReader
import docx

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    chunk_index: int
    text: str
    char_count: int

def clean_text(text: str) -> str:
    """Normalize whitespace and strip unnecessary control characters."""
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text_sliding_window(
    text: str,
    doc_id: str,
    doc_name: str,
    page: int = 1,
    start_index: int = 0,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[DocumentChunk]:
    """Chunk text into overlapping windows while respecting paragraph boundaries where possible."""
    chunks: List[DocumentChunk] = []
    text = clean_text(text)
    if not text:
        return chunks

    # Split into logical paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_chunk = ""
    chunk_idx = start_index

    for p in paragraphs:
        if not current_chunk:
            current_chunk = p
        elif len(current_chunk) + len(p) + 2 <= chunk_size:
            current_chunk += "\n\n" + p
        else:
            # Current chunk has reached target size
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                doc_id=doc_id,
                doc_name=doc_name,
                page=page,
                chunk_index=chunk_idx,
                text=current_chunk.strip(),
                char_count=len(current_chunk.strip())
            ))
            chunk_idx += 1
            # Add overlap from the tail of current_chunk
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
            current_chunk = (overlap_text + " " + p).strip()

    # Flush remaining chunk
    if current_chunk:
        chunks.append(DocumentChunk(
            chunk_id=f"{doc_id}_c{chunk_idx}",
            doc_id=doc_id,
            doc_name=doc_name,
            page=page,
            chunk_index=chunk_idx,
            text=current_chunk.strip(),
            char_count=len(current_chunk.strip())
        ))

    return chunks

def extract_chunks_from_file(file_path: Path, doc_name: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[DocumentChunk]:
    """Extract and chunk content from PDF, DOCX, TXT, or Markdown files with page metadata."""
    suffix = file_path.suffix.lower()
    doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_name.rsplit('.', 1)[0]) + "_" + uuid.uuid4().hex[:6]
    all_chunks: List[DocumentChunk] = []
    chunk_counter = 0

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_chunks = chunk_text_sliding_window(
                text=page_text,
                doc_id=doc_id,
                doc_name=doc_name,
                page=page_num,
                start_index=chunk_counter,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            all_chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

    elif suffix == ".docx":
        doc = docx.Document(str(file_path))
        full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        all_chunks = chunk_text_sliding_window(
            text=full_text,
            doc_id=doc_id,
            doc_name=doc_name,
            page=1,
            start_index=0,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    else:
        # Default to plain text / markdown
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            full_text = f.read()
        all_chunks = chunk_text_sliding_window(
            text=full_text,
            doc_id=doc_id,
            doc_name=doc_name,
            page=1,
            start_index=0,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    return all_chunks
