import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from .hybrid_retriever import ScoredChunk

class CitationCheck(BaseModel):
    citation_tag: str
    chunk_id: str
    is_valid: bool
    doc_name: str
    page: int

class GroundingReport(BaseModel):
    faithfulness_score: float  # 0.0 - 1.0
    is_grounded: bool
    verified_citations: List[CitationCheck]
    invalid_citations: List[str]
    ungrounded_sentences: List[str]
    warnings: List[str]

class HallucinationGuard:
    @staticmethod
    def build_grounded_prompt(query: str, retrieved_chunks: List[ScoredChunk]) -> str:
        """Construct a strictly constrained system + context prompt for the LLM."""
        context_blocks = []
        for c in retrieved_chunks:
            block = (
                f"--- BEGIN CHUNK [{c.chunk_id}] ---\n"
                f"Source: {c.doc_name} (Page {c.page})\n"
                f"Content:\n{c.text}\n"
                f"--- END CHUNK [{c.chunk_id}] ---"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        prompt = (
            f"You are DocuMind, an enterprise technical documentation AI assistant. "
            f"Your mission is to provide strictly factual, zero-hallucination answers.\n\n"
            f"STRICT RULES:\n"
            f"1. Base your answer ONLY on the provided context chunks below.\n"
            f"2. Every factual claim MUST include an inline citation tag pointing to the chunk ID, e.g., [Ref: {retrieved_chunks[0].chunk_id if retrieved_chunks else 'chunk_id'}].\n"
            f"3. Do NOT extrapolate, speculate, or introduce external knowledge.\n"
            f"4. If the context does not contain enough information to answer the question, state: "
            f"\"I cannot answer this based on the provided documents.\"\n\n"
            f"CONTEXT:\n"
            f"{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"GROUNDED ANSWER (with citations):"
        )
        return prompt

    @staticmethod
    def verify_grounding(
        response_text: str,
        retrieved_chunks: List[ScoredChunk]
    ) -> GroundingReport:
        """
        Post-generation verification:
        1. Extract and validate all [Ref: chunk_id] citations.
        2. Sentence-by-sentence grounding check against context.
        """
        valid_chunk_map = {c.chunk_id: c for c in retrieved_chunks}
        
        # Extract citations like [Ref: chunk_id] or [chunk_id]
        raw_citations = re.findall(r'\[(?:Ref:\s*)?([a-zA-Z0-9_\-]+)\]', response_text)
        
        verified_citations: List[CitationCheck] = []
        invalid_citations: List[str] = []
        warnings: List[str] = []

        for cit in raw_citations:
            if cit in valid_chunk_map:
                chunk = valid_chunk_map[cit]
                verified_citations.append(CitationCheck(
                    citation_tag=f"[Ref: {cit}]",
                    chunk_id=cit,
                    is_valid=True,
                    doc_name=chunk.doc_name,
                    page=chunk.page
                ))
            else:
                invalid_citations.append(cit)

        if invalid_citations:
            warnings.append(f"Model cited non-existent or unretrieved chunks: {', '.join(invalid_citations)}")

        # Sentence-level check
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response_text) if len(s.strip()) > 15]
        ungrounded_sentences: List[str] = []

        all_context_text = " ".join([c.text.lower() for c in retrieved_chunks])
        context_words = set(re.findall(r'[a-zA-Z0-9_]+', all_context_text))

        if sentences:
            supported_count = 0
            for sent in sentences:
                sent_words = [w for w in re.findall(r'[a-zA-Z0-9_]+', sent.lower()) if len(w) > 3]
                if not sent_words:
                    supported_count += 1
                    continue
                # Overlap ratio with context
                overlap = sum(1 for w in sent_words if w in context_words) / len(sent_words)
                if overlap >= 0.40:
                    supported_count += 1
                else:
                    ungrounded_sentences.append(sent)

            faithfulness = round(supported_count / len(sentences), 2)
        else:
            faithfulness = 1.0

        if len(verified_citations) == 0 and len(sentences) > 0 and "cannot answer" not in response_text.lower():
            warnings.append("Answer lacks explicit inline citations to source documents.")
            faithfulness = min(faithfulness, 0.6)

        is_grounded = faithfulness >= 0.70 and len(invalid_citations) == 0

        return GroundingReport(
            faithfulness_score=faithfulness,
            is_grounded=is_grounded,
            verified_citations=verified_citations,
            invalid_citations=invalid_citations,
            ungrounded_sentences=ungrounded_sentences,
            warnings=warnings
        )
