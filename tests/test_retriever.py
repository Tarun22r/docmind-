"""
Tests for services/retriever.py.

The embedding model and FAISS index are replaced with lightweight
fakes so these tests run fast and don't need the real ML dependencies
installed — they exercise the retriever's own logic (DB lookup,
similarity filtering, result shape) in isolation.
"""

import numpy as np
import pytest

from models.database import Document, DocumentChunk, init_db, get_session
from services.retriever import Retriever


class FakeEmbeddingService:
    def embed_query(self, text):
        return np.zeros(4, dtype="float32")


class FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits  # list of {"chunk_id": int, "score": float}

    def search(self, query_embedding, top_k=5, document_ids=None):
        hits = self._hits
        if document_ids:
            # Filtering is normally done by metadata inside the real store;
            # the fake just returns everything and lets the caller ignore
            # ids that don't belong to a real chunk in the test DB.
            hits = hits
        return hits[:top_k]


@pytest.fixture()
def session():
    init_db("sqlite://")
    s = get_session()
    doc = Document(filename="handbook.pdf", file_type="pdf", file_size=1234, status="ready")
    s.add(doc)
    s.commit()

    chunk1 = DocumentChunk(document_id=doc.id, chunk_index=0, page_number=1, content="Annual leave is 18 days.")
    chunk2 = DocumentChunk(document_id=doc.id, chunk_index=1, page_number=2, content="Sick leave is 10 days.")
    s.add_all([chunk1, chunk2])
    s.commit()

    yield s, doc, chunk1, chunk2
    s.close()


def test_retrieve_returns_chunk_text_and_metadata(session):
    s, doc, chunk1, chunk2 = session
    fake_store = FakeVectorStore([{"chunk_id": chunk1.id, "score": 0.9}])
    retriever = Retriever(FakeEmbeddingService(), fake_store, min_similarity=0.15)

    results = retriever.retrieve(s, "How many annual leave days?", top_k=5)

    assert len(results) == 1
    assert results[0]["content"] == "Annual leave is 18 days."
    assert results[0]["document_name"] == "handbook.pdf"
    assert results[0]["page_number"] == 1
    assert results[0]["score"] == 0.9


def test_retrieve_filters_below_min_similarity(session):
    s, doc, chunk1, chunk2 = session
    fake_store = FakeVectorStore([{"chunk_id": chunk1.id, "score": 0.05}])
    retriever = Retriever(FakeEmbeddingService(), fake_store, min_similarity=0.15)

    results = retriever.retrieve(s, "irrelevant question", top_k=5)

    assert results == []


def test_retrieve_skips_missing_chunks_gracefully(session):
    s, doc, chunk1, chunk2 = session
    fake_store = FakeVectorStore([{"chunk_id": 999999, "score": 0.9}])
    retriever = Retriever(FakeEmbeddingService(), fake_store, min_similarity=0.15)

    results = retriever.retrieve(s, "question", top_k=5)

    assert results == []


def test_retrieve_returns_multiple_ranked_results(session):
    s, doc, chunk1, chunk2 = session
    fake_store = FakeVectorStore(
        [
            {"chunk_id": chunk2.id, "score": 0.8},
            {"chunk_id": chunk1.id, "score": 0.7},
        ]
    )
    retriever = Retriever(FakeEmbeddingService(), fake_store, min_similarity=0.15)

    results = retriever.retrieve(s, "leave policy", top_k=5)

    assert [r["chunk_id"] for r in results] == [chunk2.id, chunk1.id]
