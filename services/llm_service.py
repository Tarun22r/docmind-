"""
LLM provider abstraction.

The rest of the app only calls `LLMService.generate(system_prompt, user_prompt)`.
Swapping providers is a one-line config change (LLM_PROVIDER) — nothing
else in the codebase needs to know which backend answered the question.

Supported providers:
    - openai  : any OpenAI-compatible /chat/completions endpoint
    - groq    : OpenAI-compatible endpoint, different base URL
    - gemini  : Google's generateContent REST endpoint
    - ollama  : local Ollama server
    - demo    : no network call, deterministic canned response

Demo mode (config.is_demo_mode()) is handled one layer up, in
rag_pipeline.py, so that "no API key configured" never crashes the
app — it just answers using the demo pipeline instead of calling here.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the configured provider fails to produce an answer."""


class LLMService:
    def __init__(self, config):
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        provider = self.config["LLM_PROVIDER"]
        try:
            if provider == "openai":
                return self._call_openai(system_prompt, user_prompt)
            if provider == "groq":
                return self._call_groq(system_prompt, user_prompt)
            if provider == "gemini":
                return self._call_gemini(system_prompt, user_prompt)
            if provider == "ollama":
                return self._call_ollama(system_prompt, user_prompt)
            raise LLMError(f"Unknown LLM_PROVIDER: {provider}")
        except requests.exceptions.Timeout as exc:
            raise LLMError("The language model timed out. Please try again.") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Could not reach the language model provider: {exc}") from exc

    # --- providers -----------------------------------------------------

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        return self._call_openai_compatible(
            base_url=self.config["OPENAI_BASE_URL"],
            api_key=self.config["OPENAI_API_KEY"],
            model=self.config["MODEL_NAME"] or "gpt-4o-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        return self._call_openai_compatible(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.config["GROQ_API_KEY"],
            model=self.config["MODEL_NAME"] or "llama-3.1-8b-instant",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _call_openai_compatible(self, base_url, api_key, model, system_prompt, user_prompt) -> str:
        if not api_key:
            raise LLMError("No API key configured for this provider.")
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=self.config["LLM_TIMEOUT_SECONDS"],
        )
        if response.status_code == 401:
            raise LLMError("The configured API key was rejected (invalid API key).")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if not self.config["GEMINI_API_KEY"]:
            raise LLMError("No API key configured for Gemini.")
        model = self.config["MODEL_NAME"] or "gemini-1.5-flash"
        api_key = self.config["GEMINI_API_KEY"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        response = requests.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            },
            timeout=self.config["LLM_TIMEOUT_SECONDS"],
        )
        if response.status_code == 401 or response.status_code == 403:
            raise LLMError("The configured API key was rejected (invalid API key).")
        response.raise_for_status()
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError("Gemini returned an unexpected response shape.") from exc

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        model = self.config["MODEL_NAME"] or "llama3"
        base_url = self.config["OLLAMA_BASE_URL"].rstrip("/")
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=self.config["LLM_TIMEOUT_SECONDS"],
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
