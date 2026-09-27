"""Tests for services/chunker.py — no external dependencies required."""

from services.chunker import chunk_document, _split_text


def test_split_text_respects_chunk_size_roughly():
    text = "word " * 500  # 2500 characters
    pieces = _split_text(text, chunk_size=900, overlap=150)
    assert len(pieces) > 1
    for piece in pieces[:-1]:
        # Allow some slack for the whitespace-boundary nudge.
        assert len(piece) <= 940


def test_split_text_empty_input():
    assert _split_text("", chunk_size=900, overlap=150) == []
    assert _split_text("   ", chunk_size=900, overlap=150) == []


def test_split_text_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        _split_text("hello world", chunk_size=100, overlap=150)


def test_chunk_document_preserves_page_numbers():
    pages = [(1, "a" * 1000), (2, "b" * 500)]
    chunks = chunk_document(pages, chunk_size=900, overlap=150)

    page_1_chunks = [c for c in chunks if c.page_number == 1]
    page_2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page_1_chunks) >= 1
    assert len(page_2_chunks) >= 1
    # No chunk should mix content from two pages.
    assert all("b" not in c.content for c in page_1_chunks)
    assert all("a" not in c.content for c in page_2_chunks)


def test_chunk_document_global_index_is_sequential():
    pages = [(1, "a" * 2000), (2, "b" * 2000)]
    chunks = chunk_document(pages, chunk_size=900, overlap=150)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_document_handles_none_page_number():
    pages = [(None, "some text " * 200)]
    chunks = chunk_document(pages, chunk_size=900, overlap=150)
    assert all(c.page_number is None for c in chunks)
    assert len(chunks) >= 1
