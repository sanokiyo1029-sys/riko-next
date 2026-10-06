"""LLM providers. Select with ``llm.provider`` in config.yaml; route roles with ``models:``."""
from __future__ import annotations

import json
from typing import Optional

import httpx

from ..core.config import LLMConfig, RikoConfig, get_secret
from .base import LLMError, LLMEvent, LLMProvider, Message, ToolCall
from .ollama import OllamaLLM
from .openai_compat import OpenAICompatibleLLM

__all__ = ["LLMError", "LLMEvent", "LLMProvider", "Message", "ToolCall", "create_llm", "ModelRouter",
           "OllamaLLM", "OpenAICompatibleLLM"]


def create_llm(cfg: LLMConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> LLMProvider:
    if cfg.provider == "ollama":
        return OllamaLLM(
            base_url=cfg.base_url or "http://127.0.0.1:11434", model=cfg.model, temperature=cfg.temperature,
            top_p=cfg.top_p, max_tokens=cfg.max_tokens, keep_alive=cfg.ollama_keep_alive, num_ctx=cfg.num_ctx,
            reasoning_effort=cfg.reasoning_effort, seed=cfg.seed, timeout_s=cfg.timeout_s,
            extra_options=cfg.extra_body, transport=transport,
        )
    api_key = get_secret(cfg.api_key_env)
    if cfg.provider in ("openai", "openrouter", "groq") and not api_key:
        raise LLMError(f"llm.provider '{cfg.provider}' needs an API key: set {cfg.api_key_env} in .env")
    return OpenAICompatibleLLM(
        name=cfg.provider, base_url=cfg.base_url or "", model=cfg.model, api_key=api_key,
        temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
        reasoning_effort=cfg.reasoning_effort, seed=cfg.seed, timeout_s=cfg.timeout_s,
        extra_body=cfg.extra_body, transport=transport,
    )


class ModelRouter:
    """Role -> provider. Roles with identical settings share one client (and one loaded model)."""

    def __init__(self, cfg: RikoConfig, primary: Optional[LLMProvider] = None) -> None:
        self.cfg = cfg
        self._by_key: dict[str, LLMProvider] = {}
        self._roles: dict[str, Optional[LLMProvider]] = {}
        if primary is not None:  # injected (tests): every role without an override uses it
            self._by_key[self._key(cfg.llm)] = primary

    @staticmethod
    def _key(c: LLMConfig) -> str:
        return json.dumps(c.model_dump(), sort_keys=True, default=str)

    def get(self, role: str) -> Optional[LLMProvider]:
        if role not in self._roles:
            c = self.cfg.model_for(role)
            if c is None:
                self._roles[role] = None
            else:
                key = self._key(c)
                if key not in self._by_key:
                    self._by_key[key] = create_llm(c)
                self._roles[role] = self._by_key[key]
        return self._roles[role]

    def set(self, role: str, provider: LLMProvider) -> None:
        self._roles[role] = provider

    def config_for(self, role: str) -> Optional[LLMConfig]:
        return self.cfg.model_for(role)

    def providers(self) -> list[LLMProvider]:
        return list({id(p): p for p in [*self._by_key.values(), *(p for p in self._roles.values() if p)]}.values())

    async def aclose(self) -> None:
        for p in self.providers():
            try:
                await p.aclose()
            except Exception:  # noqa: BLE001
                pass
