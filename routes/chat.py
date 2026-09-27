"""
Chat, search, evaluation and health API routes.
"""

import json
import logging

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from config import is_demo_mode
from models.database import Conversation, Message, QueryLog, get_session
from models.schemas import ChatRequest, EvaluateRequest, SearchRequest
from services import evaluator
from services.embeddings import EmbeddingError
from services.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__, url_prefix="/api")


def _title_from_query(query: str) -> str:
    title = query.strip().split("\n")[0]
    return title[:60] + ("..." if len(title) > 60 else "")


@chat_bp.route("/chat", methods=["POST"])
def chat():
    try:
        payload = ChatRequest.model_validate(request.get_json(force=True, silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": "Invalid request.", "details": exc.errors()}), 400

    cfg = current_app.config
    session = get_session()
    try:
        if payload.conversation_id:
            conversation = session.get(Conversation, payload.conversation_id)
            if conversation is None:
                return jsonify({"error": "Conversation not found."}), 404
        else:
            conversation = Conversation(
                title=_title_from_query(payload.query),
                document_ids=",".join(str(i) for i in payload.document_ids),
            )
            session.add(conversation)
            session.commit()

        user_message = Message(conversation_id=conversation.id, role="user", content=payload.query)
        session.add(user_message)
        session.commit()

        try:
            result = cfg["RAG_PIPELINE"].answer(
                session,
                payload.query,
                document_ids=payload.document_ids or None,
            )
        except (EmbeddingError, VectorStoreError) as exc:
            return jsonify({"error": f"Search index error: {exc}"}), 503

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
            sources_json=json.dumps(result["sources"]),
        )
        session.add(assistant_message)

        session.add(
            QueryLog(
                query=payload.query,
                response=result["answer"],
                retrieved_chunks=len(result["retrieved_chunks"]),
                latency=result["latency"],
            )
        )
        session.commit()

        return jsonify(
            {
                "conversation_id": conversation.id,
                "answer": result["answer"],
                "sources": result["sources"],
                "retrieved_chunks": result["retrieved_chunks"],
                "is_demo": result["is_demo"],
                "provider": result["provider"],
                "latency": result["latency"],
            }
        )
    finally:
        session.close()


@chat_bp.route("/conversations", methods=["GET"])
def list_conversations():
    session = get_session()
    try:
        conversations = session.query(Conversation).order_by(Conversation.created_at.desc()).all()
        return jsonify({"conversations": [c.to_dict() for c in conversations]})
    finally:
        session.close()


@chat_bp.route("/conversations/<int:conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    session = get_session()
    try:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return jsonify({"error": "Conversation not found."}), 404
        messages = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return jsonify(
            {
                "conversation": conversation.to_dict(),
                "messages": [m.to_dict() for m in messages],
            }
        )
    finally:
        session.close()


@chat_bp.route("/search", methods=["POST"])
def search():
    try:
        payload = SearchRequest.model_validate(request.get_json(force=True, silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": "Invalid request.", "details": exc.errors()}), 400

    cfg = current_app.config
    session = get_session()
    try:
        top_k = payload.top_k or cfg["TOP_K"]
        try:
            chunks = cfg["RETRIEVER"].retrieve(
                session,
                payload.query,
                document_ids=payload.document_ids or None,
                top_k=top_k,
            )
        except (EmbeddingError, VectorStoreError) as exc:
            return jsonify({"error": f"Search index error: {exc}"}), 503
        return jsonify({"results": chunks})
    finally:
        session.close()


@chat_bp.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        payload = EvaluateRequest.model_validate(request.get_json(force=True, silent=True) or {})
    except ValidationError as exc:
        return jsonify({"error": "Invalid request.", "details": exc.errors()}), 400

    cfg = current_app.config
    session = get_session()
    try:
        try:
            result = cfg["RAG_PIPELINE"].answer(
                session,
                payload.query,
                document_ids=payload.document_ids or None,
            )
        except (EmbeddingError, VectorStoreError) as exc:
            return jsonify({"error": f"Search index error: {exc}"}), 503

        metrics = evaluator.evaluate_result(result)
        return jsonify({"answer": result["answer"], "metrics": metrics})
    finally:
        session.close()


@chat_bp.route("/health", methods=["GET"])
def health():
    cfg = current_app.config
    return jsonify(
        {
            "status": "ok",
            "demo_mode": is_demo_mode(),
            "llm_provider": cfg.get("LLM_PROVIDER"),
            "embedding_model": cfg.get("EMBEDDING_MODEL"),
        }
    )
