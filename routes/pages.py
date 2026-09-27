"""HTML page routes, rendered with Jinja2. No JSON here."""

from flask import Blueprint, render_template

from config import is_demo_mode
from models.database import Conversation, Document, get_session

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    session = get_session()
    try:
        documents = session.query(Document).order_by(Document.upload_time.desc()).limit(5).all()
        return render_template("index.html", documents=documents, active_page="home")
    finally:
        session.close()


@pages_bp.route("/documents")
def documents_page():
    session = get_session()
    try:
        documents = session.query(Document).order_by(Document.upload_time.desc()).all()
        return render_template("documents.html", documents=documents, active_page="documents")
    finally:
        session.close()


@pages_bp.route("/chat")
def chat_page():
    session = get_session()
    try:
        documents = (
            session.query(Document)
            .filter(Document.status == "ready")
            .order_by(Document.upload_time.desc())
            .all()
        )
        return render_template(
            "chat.html",
            documents=documents,
            active_page="chat",
            demo_mode=is_demo_mode(),
        )
    finally:
        session.close()


@pages_bp.route("/history")
def history_page():
    session = get_session()
    try:
        conversations = session.query(Conversation).order_by(Conversation.created_at.desc()).all()
        return render_template("history.html", conversations=conversations, active_page="history")
    finally:
        session.close()


@pages_bp.route("/settings")
def settings_page():
    from flask import current_app

    cfg = current_app.config
    settings = {
        "LLM_PROVIDER": cfg.get("LLM_PROVIDER"),
        "MODEL_NAME": cfg.get("MODEL_NAME") or "(default for provider)",
        "EMBEDDING_MODEL": cfg.get("EMBEDDING_MODEL"),
        "CHUNK_SIZE": cfg.get("CHUNK_SIZE"),
        "CHUNK_OVERLAP": cfg.get("CHUNK_OVERLAP"),
        "TOP_K": cfg.get("TOP_K"),
        "demo_mode": is_demo_mode(),
    }
    return render_template("settings.html", settings=settings, active_page="settings")
