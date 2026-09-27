"""
Retriever: turns a question into ranked, document-grounded chunks.

This is the piece that sits between "raw vector search" and the RAG
pipeline — it resolves FAISS chunk ids back into real chunk text and
filenames via the database, and applies the minimum-similarity cutoff
so obviously irrelevant chunks don't get passed to the LLM.
"""

from typing import Dict, List, Optional

from models.database import Document, DocumentChunk
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        min_similarity: float = 0.15,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.min_similarity = min_similarity

    def retrieve(
        self,
        session,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        query_embedding = self.embedding_service.embed_query(query)
        raw_hits = self.vector_store.search(query_embedding, top_k=top_k, document_ids=document_ids)

        results = []
        for hit in raw_hits:
            if hit["score"] < self.min_similarity:
                continue
            chunk = session.get(DocumentChunk, hit["chunk_id"])
            if chunk is None:
                continue
            document = session.get(Document, chunk.document_id)
            results.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": document.filename if document else "Unknown document",
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                    "score": round(hit["score"], 4),
                }
            )
        return results
