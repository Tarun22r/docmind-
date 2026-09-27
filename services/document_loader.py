"""
Text extraction for PDF, DOCX and TXT files.

Every loader returns the same shape: a list of (page_number, text)
tuples. `page_number` is 1-indexed where the format supports pages
(PDF), and None where it does not (DOCX, TXT) — we never fabricate
page numbers.
"""

from typing import List, Optional, Tuple

Page = Tuple[Optional[int], str]


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or extraction fails."""


def load_pdf(file_path: str) -> List[Page]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise DocumentLoadError(
            "PDF support requires PyMuPDF. Install it with `pip install pymupdf`."
        ) from exc

    pages: List[Page] = []
    try:
        with fitz.open(file_path) as doc:
            for index, page in enumerate(doc):
                text = page.get_text("text")
                pages.append((index + 1, text))
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise DocumentLoadError(f"Could not read PDF file: {exc}") from exc

    if not pages:
        raise DocumentLoadError("The PDF has no pages.")
    return pages


def load_docx(file_path: str) -> List[Page]:
    try:
        import docx
    except ImportError as exc:
        raise DocumentLoadError(
            "DOCX support requires python-docx. Install it with `pip install python-docx`."
        ) from exc

    try:
        document = docx.Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    except Exception as exc:
        raise DocumentLoadError(f"Could not read DOCX file: {exc}") from exc

    if not paragraphs:
        raise DocumentLoadError("The DOCX file contains no readable text.")

    # DOCX has no reliable page boundaries without rendering the file,
    # so we treat the whole document as a single "page" of text.
    return [(None, "\n".join(paragraphs))]


def load_txt(file_path: str) -> List[Page]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read TXT file: {exc}") from exc

    if not text or not text.strip():
        raise DocumentLoadError("The TXT file is empty.")
    return [(None, text)]


LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "txt": load_txt,
}


def load_document(file_path: str, file_type: str) -> List[Page]:
    loader = LOADERS.get(file_type.lower())
    if loader is None:
        raise DocumentLoadError(f"Unsupported file type: {file_type}")
    return loader(file_path)
