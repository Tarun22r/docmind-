"""
Route-level tests using Flask's test client.

The embedding model, FAISS index and LLM call are replaced with fakes
after app creation so these tests exercise routing, validation and
error handling without needing model downloads or an API key.
"""

import io

import pytest

from app import create_app
from config import Config


class FakeEmbeddingService:
    def embed_texts(self, texts):
        import numpy as np

        return np.zeros((len(texts), 4), dtype="float32")

    def embed_query(self, text):
        import numpy as np

        return np.zeros(4, dtype="float32")


class FakeVectorStore:
    def __init__(self):
        self.added = []

    def add_chunks(self, chunk_ids, document_id, embeddings):
        self.added.append((document_id, list(chunk_ids)))

    def remove_document(self, document_id):
        self.added = [a for a in self.added if a[0] != document_id]

    def search(self, query_embedding, top_k=5, document_ids=None):
        return []


@pytest.fixture()
def app(tmp_path):
    class TestConfig(Config):
        TESTING = True
        DEBUG = False
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        VECTOR_STORE_FOLDER = str(tmp_path / "vector_store")
        DATABASE_PATH = str(tmp_path / "test.db")
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'test.db'}"
        FORCE_DEMO_MODE = True

    flask_app = create_app(TestConfig)
    # Swap the real (network/model-dependent) services for fakes so
    # these tests run offline and fast.
    flask_app.config["EMBEDDING_SERVICE"] = FakeEmbeddingService()
    flask_app.config["VECTOR_STORE"] = FakeVectorStore()
    flask_app.config["RETRIEVER"].embedding_service = flask_app.config["EMBEDDING_SERVICE"]
    flask_app.config["RETRIEVER"].vector_store = flask_app.config["VECTOR_STORE"]
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["demo_mode"] is True


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Your documents, searchable." in response.data


def test_documents_page_loads_with_no_documents(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert b"No documents yet." in response.data


def test_upload_rejects_missing_file(client):
    response = client.post("/api/documents/upload", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_upload_rejects_unsupported_extension(client):
    data = {"file": (io.BytesIO(b"hello world"), "notes.exe")}
    response = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


def test_upload_rejects_empty_file(client):
    data = {"file": (io.BytesIO(b""), "notes.txt")}
    response = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "empty" in response.get_json()["error"].lower()


def test_upload_txt_succeeds_and_is_ready(client):
    content = b"This is a small test document about annual leave policy. " * 5
    data = {"file": (io.BytesIO(content), "policy.txt")}
    response = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    doc = response.get_json()["document"]
    assert doc["status"] == "ready"
    assert doc["chunk_count"] >= 1


def test_get_unknown_document_returns_404(client):
    response = client.get("/api/documents/99999")
    assert response.status_code == 404


def test_chat_rejects_empty_query(client):
    response = client.post("/api/chat", json={"query": ""})
    assert response.status_code == 400


def test_chat_with_no_documents_reports_not_found(client):
    response = client.post("/api/chat", json={"query": "What is the leave policy?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "couldn't find enough information" in data["answer"]


def test_unknown_page_returns_404(client):
    response = client.get("/this-page-does-not-exist")
    assert response.status_code == 404
