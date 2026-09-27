"""
FAISS-backed vector store.

Design:
    - A single flat index holds chunks from every document.
    - Each vector is tagged with its DocumentChunk.id (so the mapping
      back to SQLite is a plain primary-key lookup), via IndexIDMap.
    - Embeddings are pre-normalized (see embeddings.py), so we use
      an inner-product index — inner product of normalized vectors
      equals cosine similarity.
    - The index is persisted to disk so re-uploading isn't needed
      after a restart. If a document is already indexed, we skip
      re-embedding it (see routes/documents.py).

This keeps things simple: one file-backed index rather than a
per-document index, with document-scoped search done by filtering
candidate results by chunk metadata after the FAISS search.
"""

import json
import logging
import os
import threading
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(self, storage_dir: str, dim: int):
        self.storage_dir = storage_dir
        self.dim = dim
        self.index_path = os.path.join(storage_dir, "index.faiss")
        self.meta_path = os.path.join(storage_dir, "metadata.json")
        self._index = None
        self._meta: Dict[int, Dict] = {}  # chunk_id -> {"document_id": int}
        os.makedirs(storage_dir, exist_ok=True)
        self._load()

    # --- persistence -----------------------------------------------------

    def _new_index(self):
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "faiss is not installed. Run `pip install faiss-cpu` (see requirements.txt)."
            ) from exc
        flat = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap(flat)

    def _load(self):
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "faiss is not installed. Run `pip install faiss-cpu` (see requirements.txt)."
            ) from exc

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._meta = {int(k): v for k, v in raw.items()}
        else:
            self._index = self._new_index()
            self._meta = {}

    def _save(self):
        import faiss

        faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh)

    # --- mutation -----------------------------------------------------

    def add_chunks(self, chunk_ids: List[int], document_id: int, embeddings: np.ndarray):
        if len(chunk_ids) != embeddings.shape[0]:
            raise VectorStoreError("chunk_ids and embeddings must have the same length.")
        if embeddings.shape[0] == 0:
            return
        with _lock:
            ids = np.array(chunk_ids, dtype="int64")
            self._index.add_with_ids(embeddings.astype("float32"), ids)
            for cid in chunk_ids:
                self._meta[cid] = {"document_id": document_id}
            self._save()

    def remove_document(self, document_id: int):
        with _lock:
            ids_to_remove = [cid for cid, meta in self._meta.items() if meta["document_id"] == document_id]
            if not ids_to_remove:
                return

            id_array = np.array(ids_to_remove, dtype="int64")
            self._index.remove_ids(id_array)
            for cid in ids_to_remove:
                self._meta.pop(cid, None)
            self._save()

    # --- search -----------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Returns a list of {"chunk_id": int, "score": float}, best first.
        If document_ids is given, only chunks from those documents are
        returned (we over-fetch from FAISS to compensate for filtering).
        """
        if self._index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype("float32")
        fetch_k = top_k if not document_ids else min(self._index.ntotal, max(top_k * 10, 50))

        scores, ids = self._index.search(query, fetch_k)
        results = []
        for score, cid in zip(scores[0], ids[0]):
            if cid == -1:
                continue
            meta = self._meta.get(int(cid))
            if meta is None:
                continue
            if document_ids and meta["document_id"] not in document_ids:
                continue
            results.append({"chunk_id": int(cid), "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    @property
    def total_vectors(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0


_store: Optional[VectorStore] = None


def get_vector_store(storage_dir: str, dim: int) -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(storage_dir, dim)
    return _store
