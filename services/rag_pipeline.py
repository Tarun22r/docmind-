"""
Orchestrates a single question -> grounded answer request.

    question
      -> retriever.retrieve()          (embed + FAISS search + DB lookup)
      -> prompts.build_user_prompt()   (context + question)
      -> llm_service.generate()        (or a demo/extractive fallback)
      -> sources + retrieved_chunks returned alongside the answer

If no LLM provider/key is configured, or the configured provider
fails, the pipeline falls back to a deterministic, clearly-labelled
"demo" answer built directly from the retrieved text — the app never
pretends a canned or extractive answer came from a real model.
"""

import logging
import time
from typing import Dict, List, Optional

from config import is_demo_mode
from services import prompts
from services.llm_service import LLMError, LLMService
from services.retriever import Retriever

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find enough information in the uploaded documents to answer that."


def _demo_answer(chunks: List[Dict]) -> str:
    """
    A deterministic, non-LLM answer: it does not paraphrase or reason
    about the content, it simply surfaces the most relevant retrieved
    passage so the RAG pipeline is still visibly working end-to-end.
    """
    if not chunks:
        return NOT_FOUND_MESSAGE

    top = chunks[0]
    page = f"page {top['page_number']}" if top.get("page_number") else "an unspecified page"
    snippet = top["content"].strip()
    if len(snippet) > 500:
        snippet = snippet[:500].rsplit(" ", 1)[0] + "..."

    return (
        f"[Demo Mode — no LLM configured, showing the most relevant passage retrieved]\n\n"
        f'From "{top["document_name"]}" ({page}):\n\n{snippet}'
    )


class RAGPipeline:
    def __init__(self, config, retriever: Retriever, llm_service: LLMService):
        self.config = config
        self.retriever = retriever
        self.llm_service = llm_service

    def answer(
        self,
        session,
        question: str,
        document_ids: Optional[List[int]] = None,
    ) -> Dict:
        start = time.time()

        chunks = self.retriever.retrieve(
            session,
            question,
            document_ids=document_ids,
            top_k=self.config["TOP_K"],
        )

        demo = is_demo_mode(self.config)
        provider_used = "demo"
        error_note = None

        if not chunks:
            answer_text = NOT_FOUND_MESSAGE
        elif demo:
            answer_text = _demo_answer(chunks)
        else:
            system_prompt = prompts.SYSTEM_PROMPT
            user_prompt = prompts.build_user_prompt(question, chunks)
            try:
                answer_text = self.llm_service.generate(system_prompt, user_prompt)
                provider_used = self.config["LLM_PROVIDER"]
            except LLMError as exc:
                logger.warning("LLM call failed, falling back to demo answer: %s", exc)
                answer_text = _demo_answer(chunks)
                error_note = str(exc)
                demo = True

        # De-duplicated, ordered source list for display.
        seen = set()
        sources = []
        for chunk in chunks:
            key = (chunk["document_name"], chunk["page_number"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({"document_name": chunk["document_name"], "page_number": chunk["page_number"]})

        latency = round(time.time() - start, 3)

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": chunks,
            "latency": latency,
            "is_demo": demo,
            "provider": provider_used,
            "llm_error": error_note,
        }
