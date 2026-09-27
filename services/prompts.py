"""
The RAG system prompt, kept in one place so it's easy to find, read,
and tune without touching pipeline logic.
"""

SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer the user's question using ONLY the provided context extracted from
their uploaded documents. Do not use outside knowledge and do not invent
facts that are not present in the context.

Rules:
- If the context does not contain enough information to answer, say clearly
  that the information was not found in the uploaded documents. Do not guess.
- When possible, mention which document and page number the answer came from.
- Be concise and factual. Do not pad the answer with generic advice.
- If different chunks disagree, point out the discrepancy rather than
  picking one silently.
"""


def build_user_prompt(question: str, chunks: list) -> str:
    """
    chunks: list of dicts with document_name, page_number, content (as
    produced by services/retriever.py).
    """
    if not chunks:
        context_block = "(No relevant context was retrieved from the uploaded documents.)"
    else:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            page = f"Page {chunk['page_number']}" if chunk.get("page_number") else "Page unavailable"
            parts.append(
                f"[Source {i}] {chunk['document_name']} — {page}\n{chunk['content']}"
            )
        context_block = "\n\n".join(parts)

    return (
        f"Context from uploaded documents:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
