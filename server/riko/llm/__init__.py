"""LLM providers. Select with ``llm.provider`` in config.yaml."""
from __future__ import annotations

from typing import Optional

import httpx

from ..core.config import LLMConfig, get_secret
from .base import LLMError, LLMProvider, Message
from .ollama import OllamaLLM
from .openai_compat import OpenAICompatibleLLM

__all__ = ["LLMError", "LLMProvider", "Message", "create_llm", "OllamaLLM", "OpenAICompatibleLLM"]


def create_llm(cfg: LLMConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> LLMProvider:
    if cfg.provider == "ollama":
        return OllamaLLM(
            base_url=cfg.base_url or "http://127.0.0.1:11434", model=cfg.model, temperature=cfg.temperature,
            top_p=cfg.top_p, max_tokens=cfg.max_tokens, keep_alive=cfg.ollama_keep_alive,
            num_ctx=cfg.ollama_num_ctx, timeout_s=cfg.timeout_s, extra_options=cfg.extra_body,
            transport=transport,
        )
    api_key = get_secret(cfg.api_key_env)
    if cfg.provider in ("openai", "openrouter", "groq") and not api_key:
        raise LLMError(f"llm.provider '{cfg.provider}' needs an API key: set {cfg.api_key_env} in .env")
    return OpenAICompatibleLLM(
        name=cfg.provider, base_url=cfg.base_url or "", model=cfg.model, api_key=api_key,
        temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens, timeout_s=cfg.timeout_s,
        extra_body=cfg.extra_body, transport=transport,
    )
