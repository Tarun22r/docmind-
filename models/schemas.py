"""
Pydantic schemas used to validate incoming API request bodies.

These are intentionally minimal — this project uses them where they
add real value (validating JSON payloads), not everywhere just to
say "Pydantic is used".
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[int] = None
    document_ids: List[int] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: List[int] = Field(default_factory=list)
    top_k: Optional[int] = None


class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: List[int] = Field(default_factory=list)
