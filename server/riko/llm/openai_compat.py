"""OpenAI-compatible Chat Completions with SSE streaming.

Works with OpenAI, LM Studio (``http://127.0.0.1:1234/v1``), OpenRouter, Groq,
llama.cpp ``llama-server``, vLLM and others. Implemented on ``httpx`` directly
rather than the ``openai`` SDK to keep dependencies small and to control
cancellation precisely (closing the stream aborts generation server-side).
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from ..core.logging import get_logger
from .base import LLMError, LLMProvider, Message, http_error_message

log = get_logger("llm")


class OpenAICompatibleLLM(LLMProvider):
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_tokens: int = 400,
        timeout_s: float = 60.0,
        extra_body: Optional[dict[str, Any]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(model)
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            **self.extra_body,
        }
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    raise LLMError(http_error_message(self.name, resp.status_code, text),
                                   status=resp.status_code, retryable=resp.status_code in (429, 500, 502, 503))
                async for line in resp.aiter_lines():
                    delta = self._parse_sse_line(line)
                    if delta is None:
                        continue
                    if delta == "\x00DONE":
                        break
                    yield delta
        except httpx.ConnectError as e:
            raise LLMError(f"{self.name}: cannot connect to {self._client.base_url} - is the server running?",
                           retryable=True) from e
        except httpx.TimeoutException as e:
            raise LLMError(f"{self.name}: request timed out", retryable=True) from e

    @staticmethod
    def _parse_sse_line(line: str) -> Optional[str]:
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            return None
        payload = line[5:].strip()
        if payload == "[DONE]":
            return "\x00DONE"
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.debug("ignoring malformed SSE line", extra={"line": payload[:120]})
            return None
        if "error" in data:
            err = data["error"]
            raise LLMError(f"provider error: {err.get('message', err) if isinstance(err, dict) else err}")
        choices = data.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) and content else None

    async def list_models(self) -> list[str]:
        try:
            r = await self._client.get("/models")
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name}: cannot reach {self._client.base_url} ({e.__class__.__name__})") from e
        if r.status_code != 200:
            raise LLMError(http_error_message(self.name, r.status_code, r.text), status=r.status_code)
        data = r.json().get("data", [])
        return sorted(str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id"))

    async def aclose(self) -> None:
        await self._client.aclose()
