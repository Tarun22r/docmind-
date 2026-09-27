"""
Chunking strategy.

We split text into overlapping, fixed-size character windows rather
than splitting on sentences/tokens — it's simple, predictable, and
easy to explain in an interview. Each page is chunked independently
so that page numbers stay accurate (a chunk never spans two pages).

Defaults (see config.py):
    CHUNK_SIZE    ~900 characters
    CHUNK_OVERLAP ~150 characters

A chunk boundary that would land mid-word is nudged to the next
whitespace where possible, so chunks don't end on a severed word.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Chunk:
    chunk_index: int
    page_number: Optional[int]
    content: str


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Nudge the boundary to the next whitespace so we don't cut a
        # word in half, as long as that doesn't shrink the chunk a lot.
        if end < length:
            next_space = text.find(" ", end)
            if next_space != -1 and next_space - end < 40:
                end = next_space

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = end - overlap

    return chunks


def chunk_document(
    pages: List[Tuple[Optional[int], str]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[Chunk]:
    """
    pages: list of (page_number, page_text) as returned by document_loader.
    Returns a flat, globally-indexed list of Chunk objects.
    """
    chunks: List[Chunk] = []
    index = 0
    for page_number, page_text in pages:
        for piece in _split_text(page_text, chunk_size, overlap):
            chunks.append(Chunk(chunk_index=index, page_number=page_number, content=piece))
            index += 1
    return chunks
