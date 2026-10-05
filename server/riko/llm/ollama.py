"""Ollama native ``/api/chat`` (NDJSON streaming).

The native API is used instead of Ollama's OpenAI shim because it exposes
``keep_alive`` (keeps the model resident in VRAM -> no multi-second cold
start between turns) and ``num_ctx``.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from .base import LLMError, LLMProvider, Message, http_error_message


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_tokens: int = 400,
        keep_alive: str = "30m",
        num_ctx: Optional[int] = None,
        timeout_s: float = 60.0,
        extra_options: Optional[dict[str, Any]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(model)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.extra_options = dict(extra_options or {})
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=5.0), transport=transport
        )

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "num_predict": max_tokens or self.max_tokens,
            **self.extra_options,
        }
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        body = {"model": self.model, "messages": messages, "stream": True,
                "keep_alive": self.keep_alive, "options": options}
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    hint = ""
                    if resp.status_code == 404:
                        hint = f" Run: ollama pull {self.model}"
                    raise LLMError(http_error_message("ollama", resp.status_code, text) + hint,
                                   status=resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in data:
                        raise LLMError(f"ollama error: {data['error']}")
                    content = (data.get("message") or {}).get("content")
                    if content:
                        yield content
                    if data.get("done"):
                        break
        except httpx.ConnectError as e:
            raise LLMError(f"ollama: cannot connect to {self._client.base_url} - start it with 'ollama serve'",
                           retryable=True) from e
        except httpx.TimeoutException as e:
            raise LLMError("ollama: request timed out (model may still be loading)", retryable=True) from e

    async def list_models(self) -> list[str]:
        try:
            r = await self._client.get("/api/tags")
        except httpx.HTTPError as e:
            raise LLMError(f"ollama: cannot reach {self._client.base_url} ({e.__class__.__name__})") from e
        if r.status_code != 200:
            raise LLMError(http_error_message("ollama", r.status_code, r.text), status=r.status_code)
        return sorted(m["name"] for m in r.json().get("models", []) if "name" in m)

    async def health(self) -> tuple[bool, str]:
        try:
            models = await self.list_models()
        except LLMError as e:
            return False, str(e)
        # Ollama accepts "llama3.1" for "llama3.1:latest"
        if self.model in models or f"{self.model}:latest" in models:
            return True, "ok"
        return False, f"model '{self.model}' is not pulled. Run: ollama pull {self.model}"

    async def aclose(self) -> None:
        await self._client.aclose()
