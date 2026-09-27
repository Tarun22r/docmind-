# DocMind AI

### Document Intelligence & Retrieval-Augmented Generation Assistant

## Overview

DocMind AI is a small, self-contained web application that lets you upload
documents (PDF, DOCX, TXT) and ask questions about them in plain English.
Answers are generated using **Retrieval-Augmented Generation (RAG)**: the
app retrieves the most relevant passages from your own documents and asks
a language model to answer *using only that context*, citing the source
document and page number.

This is a portfolio project built to demonstrate practical, hands-on
understanding of the RAG stack — chunking, embeddings, vector search,
prompt design, and grounded/source-cited answers — using a simple,
readable Flask architecture rather than a heavyweight framework.

## Problem

General-purpose LLMs answer from their training data, which means they:

- can't answer questions about your private/internal documents at all,
- can confidently hallucinate when they don't actually know something,
- give no way to verify *where* an answer came from.

## Solution

DocMind AI grounds every answer in your uploaded documents:

1. Documents are split into overlapping chunks and embedded into vectors.
2. A question is embedded the same way and matched against those chunks
   using similarity search (FAISS).
3. Only the top-matching chunks are sent to the LLM as context, with an
   instruction to answer *only* from that context.
4. The answer is displayed alongside the exact source documents, page
   numbers, and the retrieved passages themselves — so you can verify it.

If nothing relevant is found, the app says so explicitly instead of
guessing from general knowledge.

## Features

- Upload PDF, DOCX, and TXT documents (drag-and-drop or file picker)
- Automatic text extraction, cleaning, chunking, embedding, and indexing
- Ask questions scoped to one document, several, or all documents
- Source-grounded answers with document name + page number citations
- A "View retrieved context" panel showing the exact chunks used, with
  similarity scores — full pipeline transparency, nothing hidden
- Conversation history, stored and browsable later
- Lightweight, honest evaluation metrics per answer (no invented accuracy
  numbers)
- **Demo Mode**: the entire app — upload, retrieval, chat, sources — works
  with zero API keys configured
- Swappable LLM backend: OpenAI, Gemini, Groq, or a local Ollama model,
  via one environment variable

## Architecture

```
 Browser (HTML / CSS / vanilla JS)
        |
        v
 Flask routes  ─────────────┬─ pages.py      (server-rendered HTML views)
        |                   ├─ documents.py  (upload / list / delete API)
        |                   └─ chat.py       (chat / search / evaluate API)
        v
 Services layer ────────────┬─ document_loader.py  (PDF/DOCX/TXT extraction)
        |                   ├─ text_cleaner.py
        |                   ├─ chunker.py          (sliding-window chunking)
        |                   ├─ embeddings.py       (Sentence Transformers)
        |                   ├─ vector_store.py     (FAISS index + metadata)
        |                   ├─ retriever.py        (search -> DB lookup)
        |                   ├─ prompts.py          (RAG system prompt)
        |                   ├─ llm_service.py      (OpenAI/Gemini/Groq/Ollama)
        |                   ├─ rag_pipeline.py     (orchestrates all of the above)
        |                   └─ evaluator.py        (retrieval/latency metrics)
        v
 SQLite (via SQLAlchemy) ── documents, chunks, conversations, messages, logs
 FAISS index (on disk)   ── vector embeddings + chunk-id metadata
```

## RAG Pipeline

```
Upload document
   -> Extract text (PyMuPDF / python-docx / plain read)
   -> Clean text (normalize whitespace)
   -> Split into overlapping chunks (page numbers preserved)
   -> Embed each chunk (Sentence Transformers)
   -> Store vectors in FAISS + chunk rows in SQLite
   -> Mark document "ready"

User asks a question
   -> Embed the question
   -> FAISS similarity search (optionally scoped to selected documents)
   -> Drop chunks below a minimum similarity threshold
   -> Build a prompt: system instructions + retrieved chunks + question
   -> Send to the configured LLM (or the demo fallback)
   -> Return the answer + deduplicated sources + full retrieved context
```

## Tech Stack

| Layer              | Technology                                   |
|--------------------|-----------------------------------------------|
| Backend            | Python, Flask, Flask-CORS                    |
| Data / validation  | SQLAlchemy (SQLite), Pydantic                |
| Document parsing   | PyMuPDF (PDF), python-docx (DOCX), stdlib (TXT) |
| Embeddings         | Sentence Transformers (`all-MiniLM-L6-v2`)   |
| Vector search      | FAISS (`IndexFlatIP` + `IndexIDMap`)         |
| LLM                | OpenAI-compatible HTTP API abstraction (OpenAI, Groq, Gemini, Ollama) |
| Frontend           | Jinja2 templates, vanilla CSS, vanilla JS    |
| Testing            | pytest                                        |

## Project Structure

```
docmind-ai/
├── app.py                   # Flask application factory / entry point
├── config.py                # Environment-driven settings
├── requirements.txt
├── .env.example
├── models/
│   ├── database.py          # SQLAlchemy models + session helpers
│   └── schemas.py           # Pydantic request schemas
├── routes/
│   ├── pages.py             # HTML page routes
│   ├── documents.py         # Upload / list / delete API + processing pipeline
│   └── chat.py              # Chat / search / evaluate / health API
├── services/
│   ├── document_loader.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── llm_service.py
│   ├── rag_pipeline.py
│   └── evaluator.py
├── templates/                # Jinja2 HTML
├── static/css/style.css
├── static/js/{main,upload,chat}.js
├── sample_data/sample_policy.txt   # bundled sample doc for Demo Mode
├── uploads/                  # uploaded files land here (gitignored)
├── vector_store/             # FAISS index + metadata (gitignored)
└── tests/
```

## Installation

```bash
git clone <this-repo>
cd docmind-ai

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

See `.env.example` for the full list. Nothing is required to run the app —
if `LLM_PROVIDER` is left as `demo` (or the chosen provider has no key),
the app runs in **Demo Mode** automatically.

Key variables:

| Variable         | Purpose                                             |
|------------------|------------------------------------------------------|
| `LLM_PROVIDER`   | `openai` \| `gemini` \| `groq` \| `ollama` \| `demo` |
| `MODEL_NAME`     | Model name for the chosen provider                   |
| `*_API_KEY`      | API key for OpenAI / Gemini / Groq                   |
| `OLLAMA_BASE_URL`| Base URL for a local Ollama server                   |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters (characters)      |
| `TOP_K`          | Number of chunks retrieved per question              |
| `MIN_SIMILARITY` | Minimum cosine similarity to keep a retrieved chunk  |

## Running the Backend

```bash
python app.py
```

The app starts on `http://localhost:5000` and serves both the API and the
HTML frontend (there is no separate frontend server or build step).

## Using the Application

1. Go to **Documents** and upload a PDF, DOCX, or TXT file (or click
   **Load sample document** to try it instantly with a bundled sample
   leave policy).
2. Wait for the status to change from *processing* → *indexing* → *ready*.
3. Go to **Chat**, choose which document(s) to search (or leave all
   selected), and ask a question.
4. Expand **View retrieved context** under any answer to see exactly which
   passages were used and their similarity scores.
5. Past conversations are listed under **History**.

## Demo Mode

DocMind AI is fully usable with **no API key**. In Demo Mode:

- Document upload, extraction, chunking, embedding, and FAISS retrieval
  all run exactly as they would in production — nothing about the RAG
  pipeline itself is faked.
- Only the final "ask an LLM to write an answer" step is replaced. Instead,
  the app returns the single most relevant retrieved passage directly,
  clearly labeled `[Demo Mode — no LLM configured, ...]`.
- If a real provider is configured but the request fails (bad key, timeout,
  network error), the app **falls back to Demo Mode for that answer**
  rather than crashing, and still logs a human-readable error.

This means the app never pretends a canned or extractive answer came from
a real model, and grading/interview reviewers can exercise the whole
pipeline without needing any credentials.

## How RAG Works

RAG (Retrieval-Augmented Generation) separates "knowing where information
is" from "writing a fluent answer". The LLM is not asked to *recall* facts
from its training data — it's handed the actual source text and asked to
summarize/answer strictly from that. This is what makes the system:

- **Grounded**: answers are traceable to specific documents and pages.
- **Current**: it works on documents the model has never seen before.
- **Honest**: when the documents don't contain the answer, the app says so
  instead of guessing.

## Chunking Strategy

Text is split into fixed-size, overlapping character windows:

- **Chunk size**: ~900 characters (configurable via `CHUNK_SIZE`)
- **Overlap**: ~150 characters (configurable via `CHUNK_OVERLAP`)
- Chunk boundaries are nudged to the nearest whitespace so words aren't
  split in half.
- **PDFs are chunked per page**, so a chunk never spans two pages and page
  citations stay accurate. DOCX/TXT don't have a native page concept, so
  their single block of text is chunked as one continuous document
  (citations show "Page unavailable" for these).

This is a simple, predictable strategy (as opposed to sentence- or
token-aware splitting) chosen deliberately so it's easy to reason about
and explain.

## Embedding Model

[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) —
a small (~80MB), fast, 384-dimensional sentence embedding model. It's not
the most powerful embedding model available, but it runs comfortably on
CPU with no GPU required, which matters for a project meant to run
locally. Embeddings are normalized so that FAISS inner-product search is
equivalent to cosine similarity.

## Vector Search

A single FAISS `IndexFlatIP` (wrapped in `IndexIDMap` so each vector keeps
its real database chunk ID) holds every chunk from every document. This
keeps the implementation simple — one index, one file on disk — while
still supporting per-document search scope by filtering results against
chunk metadata after the similarity search. For the corpus sizes this
project targets (a handful of documents), a flat index is fast enough;
an IVF/HNSW index would be the natural next step at much larger scale.

## Prompt Design

The system prompt (in `services/prompts.py`) explicitly instructs the
model to:

- answer **only** from the provided context,
- say clearly when the context is insufficient rather than guessing,
- cite the document/page where possible,
- flag disagreements between sources rather than silently picking one.

Keeping the prompt in its own module (rather than inlined in the pipeline
logic) makes it easy to find, review, and iterate on independently of the
retrieval and API-calling code.

## Limitations

- **Not production-scale.** Document processing runs synchronously in the
  request; a real deployment would move this to a background worker/queue.
- The FAISS index is a single flat file — fine for a handful of documents,
  not built for a large corpus.
- DOCX and TXT files have no native page numbers, so citations for these
  fall back to "Page unavailable" rather than inventing a page number.
- The bundled "evaluation" is retrieval/latency statistics, not answer
  correctness — judging whether an answer is actually *right* would need
  a labeled test set and/or an LLM-as-judge setup, which is out of scope
  here.
- Single-user, single-process design — no authentication, no multi-tenant
  isolation.

## Future Improvements

- Background job queue for document processing (e.g. Celery/RQ) with
  progress streamed to the UI instead of a blocking request
- Hybrid search (keyword + vector) for exact-term queries
- Streaming LLM responses in the chat UI
- Reranking retrieved chunks with a cross-encoder before prompting
- Per-user accounts and document access control
- An LLM-based answer-quality evaluator with a small labeled test set

## Testing

```bash
pytest
```

Tests cover chunking logic, the retriever (using fake embedding/vector
store implementations so no model download or network access is
required), and the Flask routes (upload validation, chat, health check,
404 handling). The app is designed to still start and serve pages/APIs
even if the configured LLM provider is unreachable — it simply falls back
to Demo Mode for that response.

Troubleshooting


Problem: Upload shows "Failed to fetch"

"Failed to fetch" is a browser-level network error — it means the request never received any response
at all, not that the server rejected something. When this happens right after selecting a file on the
Documents page, it almost always means the upload request is hanging on the server rather than
failing quickly.

The most common cause: the very first time a document is uploaded, the app needs to load the
Sentence Transformers embedding model (all-MiniLM-L6-v2) to generate embeddings. If that model is
not already cached locally, Python downloads it from Hugging Face the first time it is used. If the
machine running the server has no outbound internet access (or it is very slow/blocked), that download
call hangs indefinitely inside the upload request. The browser eventually gives up waiting and reports
"Failed to fetch" instead of a real error message, because the connection was dropped rather than
answered.

The fix: 
run an isolated download test
Before troubleshooting anything else, isolate this one step from the rest of the app. Stop the server
(Ctrl+C in the terminal running app.py), then run this command by itself:

****python -c "from sentence_transformers import SentenceTransformer;
SentenceTransformer('all-MiniLM-L6-v2')"****

This loads only the embedding model, with no Flask, no file upload, and no FAISS involved — so it tells
you definitively whether the model download is the problem, and if it succeeds, it fixes the issue as a
side effect (the model is cached to disk and never needs to download again).

l If it downloads and finishes cleanly — the model is now cached. Restart the server with
python app.py and try uploading again; it should now be instant.

l If it hangs for a long time or fails with a connection error — the machine running the server
has no usable outbound internet access to Hugging Face. See the two options below.

If there is no internet access at all

Option A — bring the model in from another machine. On any machine that does have internet
access, run the same command above, then locate the cached model folder, typically at:
~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
Copy that folder to the same relative path on the machine running the server, then add this to .env so it
never tries to reach the network again:

HF_HUB_OFFLINE=1

Option B — ask for a fully offline embedding fallback. The embedding step can be swapped for a
simple offline method (such as a hashing-based vectorizer) that needs no model download and no
internet access at all, at some cost to retrieval quality compared to real semantic embeddings. This is a
deliberate trade-off worth discussing before making, since it changes how the RAG pipeline behaves.

Quick checklist before assuming it's a code bug

l Is the terminal running app.py still alive, or did it crash/exit?

l Does the isolated download test above hang, or does it fail instantly with an import error (missing
package) versus a network error (no internet)?

l Check whether the model is already cached: ls ~/.cache/huggingface/ (or
~/.cache/torch/sentence_transformers/ on older versions).

l If running FLASK_DEBUG=true, try setting it to false and restarting — the auto-reloader can
occasionally drop an in-progress request if it restarts mid-upload
