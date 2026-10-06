"""Provider-agnostic LLM interface.

Providers normalise their wire format into ``LLMEvent``s:

* ``thinking``   - the model is reasoning. **No text is carried** (invariant I01: reasoning never leaves the
                   provider layer, so it cannot reach the UI, TTS, history, memory or logs).
* ``content``    - a delta of the user-visible answer.
* ``tool_calls`` - complete tool calls (arguments already parsed into a dict).
* ``done``       - final usage counters and finish reason (missing counters stay ``None``, never 0).

Messages use one canonical shape; each provider converts it to its own wire format:

    {"role": "system"|"user"|"assistant", "content": str, "images"?: [data URL]}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional

Message = dict[str, Any]


class LLMError(Exception):
    """A provider failure with a user-presentable message (never includes secrets)."""

    def __init__(self, message: str, *, status: Optional[int] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:10])


@dataclass
class LLMEvent:
    kind: Literal["thinking", "content", "tool_calls", "done"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None


class LLMProvider(ABC):
    """Subclasses implement ``stream_events`` (preferred) or ``stream_chat`` (text only); each defaults
    to the other, so simple providers and test doubles stay small."""

    name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    async def stream_events(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[LLMEvent]:
        if type(self).stream_chat is LLMProvider.stream_chat:
            raise NotImplementedError("implement stream_events or stream_chat")
        async for delta in self.stream_chat(messages, temperature=temperature, max_tokens=max_tokens):
            yield LLMEvent("content", delta)
        yield LLMEvent("done", finish_reason="stop")

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield answer text deltas only. Cancelling the consumer must close the HTTP stream."""
        async for ev in self.stream_events(messages, temperature=temperature, max_tokens=max_tokens):
            if ev.kind == "content" and ev.text:
                yield ev.text

    async def complete(self, messages: list[Message], *, json_schema: Optional[dict[str, Any]] = None,
                       **kw: Any) -> str:
        parts: list[str] = []
        async for ev in self.stream_events(messages, json_schema=json_schema, **kw):
            if ev.kind == "content":
                parts.append(ev.text)
        return "".join(parts)

    async def list_models(self) -> list[str]:
        return [self.model]

    async def capabilities(self) -> dict[str, Any]:
        """What the provider/model can do: supported / unsupported / unknown (None)."""
        return {"thinking": None, "think_levels": None, "tools": None, "vision": None, "json_schema": None,
                "context_length": None, "cache_counters": None}

    async def warmup(self) -> dict[str, Any]:
        return {"loaded": None}

    async def health(self) -> tuple[bool, str]:
        try:
            models = await self.list_models()
        except Exception as e:  # noqa: BLE001 - surface any connectivity problem
            return False, str(e)
        if models and self.model not in models:
            return False, f"model '{self.model}' not found (available: {', '.join(models[:8])})"
        return True, "ok"

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


def http_error_message(provider: str, status: int, body: str) -> str:
    hints = {
        400: "bad request - the model may not support a requested feature (tools, schema, images)",
        401: "authentication failed - check the API key environment variable",
        403: "access denied by the provider",
        404: "endpoint or model not found - check base_url and model name",
        429: "rate limited by the provider",
    }
    snippet = body.strip().replace("\n", " ")[:300]
    return f"{provider} HTTP {status}: {hints.get(status, 'request failed')} ({snippet})"


def split_data_url(url: str) -> tuple[str, str]:
    """'data:image/png;base64,AAA' -> ('image/png', 'AAA')."""
    head, _, b64 = url.partition(",")
    mime = head[5:].split(";")[0] if head.startswith("data:") else "image/png"
    return mime or "image/png", b64
