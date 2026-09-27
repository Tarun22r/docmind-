"""
Light text normalization applied after extraction and before chunking.

Deliberately conservative — we normalize whitespace but do not rewrite
or reinterpret the source content, since the RAG pipeline depends on
the text matching what is actually in the document.
"""

import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK_LINE.sub("\n\n", text)
    return text.strip()
