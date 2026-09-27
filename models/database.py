"""
SQLAlchemy models and database session helpers.

Kept deliberately simple: five tables, no migrations framework.
For a portfolio project, `db.create_all()` on startup is enough.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    upload_time = Column(DateTime, default=datetime.utcnow)
    chunk_count = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    status = Column(String(20), default="uploading")  # uploading|processing|indexing|ready|failed
    error_message = Column(Text, nullable=True)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "upload_time": self.upload_time.isoformat() if self.upload_time else None,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "status": self.status,
            "error_message": self.error_message,
        }


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)  # None when unavailable (e.g. TXT)
    content = Column(Text, nullable=False)

    document = relationship("Document", back_populates="chunks")

    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "content": self.content,
        }


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), default="New conversation")
    document_ids = Column(String(255), default="")  # comma-separated document ids in scope
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def to_dict(self):
        doc_ids = [int(x) for x in self.document_ids.split(",") if x]
        return {
            "id": self.id,
            "title": self.title,
            "document_ids": doc_ids,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user|assistant
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # serialized list of source dicts
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self):
        import json

        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "sources": json.loads(self.sources_json) if self.sources_json else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    retrieved_chunks = Column(Integer, default=0)
    latency = Column(Float, default=0.0)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response,
            "retrieved_chunks": self.retrieved_chunks,
            "latency": self.latency,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# --- engine / session setup ---------------------------------------------

_engine = None
SessionLocal = None


def init_db(database_uri: str):
    """Create the engine, tables, and session factory. Call once at startup."""
    global _engine, SessionLocal
    _engine = create_engine(database_uri, connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session():
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return SessionLocal()
