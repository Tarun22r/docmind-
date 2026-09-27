"""
DocMind AI — application entry point.

Run with:
    python app.py
"""

import logging
import os

from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from models.database import init_db
from services.embeddings import EmbeddingService
from services.llm_service import LLMService
from services.rag_pipeline import RAGPipeline
from services.retriever import Retriever
from services.vector_store import get_vector_store


def create_app(config_object: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)

    logging.basicConfig(
        level=logging.DEBUG if app.config["DEBUG"] else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["VECTOR_STORE_FOLDER"], exist_ok=True)

    # --- database -----------------------------------------------------
    init_db(app.config["SQLALCHEMY_DATABASE_URI"])

    # --- RAG services ---------------------------------------------------
    # Built once per process and stashed on app.config so routes can
    # reach them via `current_app.config[...]` without global singletons
    # scattered across modules.
    embedding_service = EmbeddingService(app.config["EMBEDDING_MODEL"])
    vector_store = get_vector_store(app.config["VECTOR_STORE_FOLDER"], app.config["EMBEDDING_DIM"])
    retriever = Retriever(embedding_service, vector_store, min_similarity=app.config["MIN_SIMILARITY"])
    llm_service = LLMService(app.config)
    rag_pipeline = RAGPipeline(app.config, retriever, llm_service)

    app.config["EMBEDDING_SERVICE"] = embedding_service
    app.config["VECTOR_STORE"] = vector_store
    app.config["RETRIEVER"] = retriever
    app.config["LLM_SERVICE"] = llm_service
    app.config["RAG_PIPELINE"] = rag_pipeline

    # --- blueprints -----------------------------------------------------
    from routes.chat import chat_bp
    from routes.documents import documents_bp
    from routes.pages import pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(chat_bp)

    # --- error handlers -----------------------------------------------------
    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_exc):
        max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": f"File too large. Maximum size is {max_mb} MB."}), 413

    @app.errorhandler(404)
    def handle_not_found(_exc):
        if _wants_json():
            return jsonify({"error": "Not found."}), 404
        from flask import render_template

        return render_template("404.html"), 404

    @app.errorhandler(500)
    def handle_server_error(exc):
        app.logger.exception("Unhandled server error: %s", exc)
        return jsonify({"error": "An unexpected server error occurred."}), 500

    return app


def _wants_json() -> bool:
    from flask import request

    return request.path.startswith("/api/")


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=app.config["DEBUG"])
