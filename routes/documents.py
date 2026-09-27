"""
Document API routes: upload, list, fetch, delete.

Upload triggers the full processing pipeline synchronously (extract ->
clean -> chunk -> embed -> index -> mark ready). For a portfolio-sized
project this is fine; a production system would push this to a background
worker so the request doesn't block on embedding generation.
"""

import logging
import os
import shutil

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from models.database import Document, DocumentChunk, get_session
from services import chunker, document_loader, text_cleaner
from services.document_loader import DocumentLoadError
from services.embeddings import EmbeddingError
from services.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")


def _allowed_file(filename: str, allowed_extensions) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower()


def _process_document(document_id: int):
    """
    Runs the extract -> clean -> chunk -> embed -> index pipeline for an
    already-saved file. Any failure marks the document as failed with a
    human-readable message instead of raising past this function.
    """
    cfg = current_app.config
    session = get_session()
    try:
        doc = session.get(Document, document_id)
        if doc is None:
            return

        doc.status = "processing"
        session.commit()

        file_path = os.path.join(cfg["UPLOAD_FOLDER"], f"{doc.id}_{doc.filename}")

        try:
            pages = document_loader.load_document(file_path, doc.file_type)
        except DocumentLoadError as exc:
            doc.status = "failed"
            doc.error_message = str(exc)
            session.commit()
            return

        cleaned_pages = [(page_num, text_cleaner.clean_text(text)) for page_num, text in pages]
        doc.page_count = len([p for p in cleaned_pages if p[0] is not None]) or len(cleaned_pages)

        chunks = chunker.chunk_document(
            cleaned_pages,
            chunk_size=cfg["CHUNK_SIZE"],
            overlap=cfg["CHUNK_OVERLAP"],
        )

        if not chunks:
            doc.status = "failed"
            doc.error_message = "No extractable text was found in this document."
            session.commit()
            return

        doc.status = "indexing"
        session.commit()

        # Persist chunks first so we have real DocumentChunk.id values to
        # use as the FAISS vector ids.
        chunk_rows = []
        for c in chunks:
            row = DocumentChunk(
                document_id=doc.id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                content=c.content,
            )
            session.add(row)
            chunk_rows.append(row)
        session.commit()

        try:
            embedding_service = cfg["EMBEDDING_SERVICE"]
            vector_store = cfg["VECTOR_STORE"]
            texts = [row.content for row in chunk_rows]
            embeddings = embedding_service.embed_texts(texts)
            chunk_ids = [row.id for row in chunk_rows]
            vector_store.add_chunks(chunk_ids, doc.id, embeddings)
        except (EmbeddingError, VectorStoreError) as exc:
            doc.status = "failed"
            doc.error_message = f"Indexing failed: {exc}"
            session.commit()
            return

        doc.chunk_count = len(chunk_rows)
        doc.status = "ready"
        doc.error_message = None
        session.commit()
    except Exception as exc:  # last-resort safety net; never leave a doc stuck mid-status
        logger.exception("Unexpected error processing document %s", document_id)
        doc = session.get(Document, document_id)
        if doc is not None:
            doc.status = "failed"
            doc.error_message = "An unexpected error occurred while processing this document."
            session.commit()
    finally:
        session.close()


@documents_bp.route("/upload", methods=["POST"])
def upload_document():
    cfg = current_app.config

    if "file" not in request.files:
        return jsonify({"error": "No file was included in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    if not _allowed_file(filename, cfg["ALLOWED_EXTENSIONS"]):
        allowed = ", ".join(sorted(cfg["ALLOWED_EXTENSIONS"]))
        return jsonify({"error": f"Unsupported file type. Allowed types: {allowed}."}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size == 0:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if file_size > cfg["MAX_CONTENT_LENGTH"]:
        max_mb = cfg["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": f"File too large. Maximum size is {max_mb} MB."}), 400

    session = get_session()
    try:
        doc = Document(
            filename=filename,
            file_type=_extension(filename),
            file_size=file_size,
            status="uploading",
        )
        session.add(doc)
        session.commit()

        # Prefix with the document id to avoid filename collisions while
        # keeping the original name readable on disk.
        dest_path = os.path.join(cfg["UPLOAD_FOLDER"], f"{doc.id}_{filename}")
        try:
            file.save(dest_path)
        except OSError as exc:
            doc.status = "failed"
            doc.error_message = f"Could not save the uploaded file: {exc}"
            session.commit()
            return jsonify({"error": doc.error_message, "document": doc.to_dict()}), 500

        doc_id = doc.id
    finally:
        session.close()

    _process_document(doc_id)

    session = get_session()
    try:
        doc = session.get(Document, doc_id)
        status_code = 201 if doc.status == "ready" else 422
        return jsonify({"document": doc.to_dict()}), status_code
    finally:
        session.close()


@documents_bp.route("/load-sample", methods=["POST"])
def load_sample_document():
    """Copies the bundled sample document in so reviewers can try the app
    without needing files of their own."""
    cfg = current_app.config
    sample_path = os.path.join(current_app.root_path, "sample_data", "sample_policy.txt")
    if not os.path.exists(sample_path):
        return jsonify({"error": "Sample document is not available."}), 404

    filename = "sample_leave_policy.txt"
    session = get_session()
    try:
        file_size = os.path.getsize(sample_path)
        doc = Document(filename=filename, file_type="txt", file_size=file_size, status="uploading")
        session.add(doc)
        session.commit()
        dest_path = os.path.join(cfg["UPLOAD_FOLDER"], f"{doc.id}_{filename}")
        shutil.copyfile(sample_path, dest_path)
        doc_id = doc.id
    finally:
        session.close()

    _process_document(doc_id)

    session = get_session()
    try:
        doc = session.get(Document, doc_id)
        status_code = 201 if doc.status == "ready" else 422
        return jsonify({"document": doc.to_dict()}), status_code
    finally:
        session.close()


@documents_bp.route("", methods=["GET"])
def list_documents():
    session = get_session()
    try:
        documents = session.query(Document).order_by(Document.upload_time.desc()).all()
        return jsonify({"documents": [d.to_dict() for d in documents]})
    finally:
        session.close()


@documents_bp.route("/<int:document_id>", methods=["GET"])
def get_document(document_id):
    session = get_session()
    try:
        doc = session.get(Document, document_id)
        if doc is None:
            return jsonify({"error": "Document not found."}), 404
        return jsonify({"document": doc.to_dict()})
    finally:
        session.close()


@documents_bp.route("/<int:document_id>", methods=["DELETE"])
def delete_document(document_id):
    cfg = current_app.config
    session = get_session()
    try:
        doc = session.get(Document, document_id)
        if doc is None:
            return jsonify({"error": "Document not found."}), 404

        try:
            cfg["VECTOR_STORE"].remove_document(document_id)
        except VectorStoreError as exc:
            logger.warning("Could not remove vectors for document %s: %s", document_id, exc)

        file_path = os.path.join(cfg["UPLOAD_FOLDER"], f"{doc.id}_{doc.filename}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as exc:
                logger.warning("Could not remove file %s: %s", file_path, exc)

        session.delete(doc)  # cascades to chunks
        session.commit()
        return jsonify({"message": "Document deleted."})
    finally:
        session.close()
