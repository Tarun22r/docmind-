"""
Lightweight, honest evaluation of a single RAG answer.

No made-up confidence percentages. Everything returned here is
computed directly from the retrieval result — real numbers or
nothing at all.
"""

from typing import Dict, List


def evaluate_result(result: Dict) -> Dict:
    """
    result: the dict returned by RAGPipeline.answer().
    """
    chunks: List[Dict] = result.get("retrieved_chunks", [])
    num_chunks = len(chunks)
    distinct_documents = len({c["document_id"] for c in chunks}) if chunks else 0
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    avg_similarity = round(sum(scores) / len(scores), 4) if scores else None
    top_similarity = round(max(scores), 4) if scores else None

    return {
        "num_chunks_retrieved": num_chunks,
        "source_document_coverage": distinct_documents,
        "average_similarity": avg_similarity,
        "top_similarity": top_similarity,
        "response_latency_seconds": result.get("latency"),
        "answered_from_documents": num_chunks > 0,
        "used_llm": not result.get("is_demo", False),
        "provider": result.get("provider"),
    }
