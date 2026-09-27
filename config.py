"""
Central configuration for DocMind AI.

All tunable values live here so the RAG pipeline can be adjusted
without hunting through the codebase. Values are read from
environment variables (see .env.example) with sensible defaults.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _bool(env_value, default=False):
    if env_value is None:
        return default
    return env_value.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # --- Flask -----------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    DEBUG = _bool(os.environ.get("FLASK_DEBUG"), default=True)

    # --- Storage -----------------------------------------------------
    UPLOAD_FOLDER = str(BASE_DIR / "uploads")
    VECTOR_STORE_FOLDER = str(BASE_DIR / "vector_store")
    DATABASE_PATH = str(BASE_DIR / "docmind.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"

    # --- Upload rules --------------------------------------------------
    ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

    # --- Chunking ------------------------------------------------------
    # Character-based chunking. Overlap keeps context from being cut
    # in half at a chunk boundary.
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 900))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))

    # --- Embeddings ------------------------------------------------------
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 384))  # matches MiniLM-L6-v2

    # --- Retrieval -----------------------------------------------------
    TOP_K = int(os.environ.get("TOP_K", 5))
    MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", 0.15))

    # --- LLM -----------------------------------------------------------
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "demo")  # openai|gemini|groq|ollama|demo
    MODEL_NAME = os.environ.get("MODEL_NAME", "")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", 30))

    # If true (or if no provider is configured), the app answers using
    # a deterministic local sample instead of calling a real LLM.
    FORCE_DEMO_MODE = _bool(os.environ.get("FORCE_DEMO_MODE"), default=False)


def _cfg_get(cfg, key):
    """
    Reads `key` from either a plain class (dot access, e.g. the Config
    class itself) or a dict-like mapping (e.g. Flask's app.config).
    """
    if isinstance(cfg, type) or not hasattr(cfg, "get"):
        return getattr(cfg, key, None)
    return cfg.get(key)


def is_demo_mode(cfg=None) -> bool:
    """
    Demo mode is active when the developer forces it, or when no
    provider/key is configured for the selected LLM_PROVIDER.
    The app must always be usable without an API key.

    `cfg` may be the Config class itself, or a Flask app.config
    (dict-like) instance. If omitted, the current app's config is used
    when available, falling back to the static Config class.
    """
    if cfg is None:
        try:
            from flask import current_app

            cfg = current_app.config
        except RuntimeError:
            cfg = Config

    provider = _cfg_get(cfg, "LLM_PROVIDER")
    if _cfg_get(cfg, "FORCE_DEMO_MODE") or provider == "demo":
        return True
    if provider == "openai" and not _cfg_get(cfg, "OPENAI_API_KEY"):
        return True
    if provider == "gemini" and not _cfg_get(cfg, "GEMINI_API_KEY"):
        return True
    if provider == "groq" and not _cfg_get(cfg, "GROQ_API_KEY"):
        return True
    # Ollama runs locally without a key, so it is never forced into demo
    # mode purely for lacking a key.
    return False
