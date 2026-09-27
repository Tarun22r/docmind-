"""
Embedding generation using Sentence Transformers.

The model is loaded once per process (it's a few hundred MB and takes
a second or two to load) and reused for every request.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_name = None


class EmbeddingError(Exception):
    pass


def _get_model(model_name: str):
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is not installed. "
            "Run `pip install sentence-transformers` (see requirements.txt)."
        ) from exc

    logger.info("Loading embedding model '%s'...", model_name)
    _model = SentenceTransformer(model_name)
    _model_name = model_name
    return _model


class EmbeddingService:
    """Thin wrapper so callers don't need to know about the underlying model."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        model = _get_model(self.model_name)
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine similarity == dot product
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
