"""OpenAI-compatible Chat Completions with SSE streaming.

Works with OpenAI, LM Studio (``http://127.0.0.1:1234/v1``), OpenRouter, Groq, llama.cpp ``llama-server``,
vLLM and others - including gpt-oss served by any of them. Implemented on ``httpx`` directly rather than
the ``openai`` SDK to keep dependencies small and to control cancellation precisely (closing the stream
aborts generation server-side).

Reasoning arrives as ``delta.reasoning_content`` (llama.cpp, vLLM, DeepSeek) or ``delta.reasoning``
(OpenRouter, LM Studio); both become text-less ``thinking`` events. Tool-call arguments arrive as JSON
string fragments keyed by ``index`` and are assembled before being emitted. An OpenAI-shaped API does not
imply identical behaviour, so every optional field is only sent when configured.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from ..core.logging import get_logger
from .base import LLMError, LLMEvent, LLMProvider, Message, ToolCall, http_error_message

log = get_logger("llm")


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": str(m.get("content", ""))})
        elif role == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                       for c in m["tool_calls"]]})
        elif m.get("images"):
            parts: list[dict[str, Any]] = [{"type": "text", "text": m.get("content") or ""}]
            parts += [{"type": "image_url", "image_url": {"url": u}} for u in m["images"]]
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


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
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
        timeout_s: float = 60.0,
        extra_body: Optional[dict[str, Any]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(model)
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.seed = seed
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

    async def stream_events(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[LLMEvent]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "stream": True,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            **self.extra_body,
        }
        if tools:
            body["tools"] = tools
        if json_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "result", "schema": json_schema}}
        if self.reasoning_effort and self.reasoning_effort != "off":
            body["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            body["seed"] = self.seed
        calls: dict[int, dict[str, Any]] = {}
        finish: Optional[str] = None
        usage: dict[str, Any] = {}
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    raise LLMError(http_error_message(self.name, resp.status_code, text),
                                   status=resp.status_code, retryable=resp.status_code in (429, 500, 502, 503))
                async for line in resp.aiter_lines():
                    data = self._parse_sse_line(line)
                    if data is None:
                        continue
                    if data == "DONE":
                        break
                    if isinstance(data.get("usage"), dict):
                        u = data["usage"]
                        usage = {"prompt_tokens": u.get("prompt_tokens"),
                                 "completion_tokens": u.get("completion_tokens"),
                                 "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content") or delta.get("reasoning"):
                            yield LLMEvent("thinking")
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            yield LLMEvent("content", content)
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(int(tc.get("index", 0)), {"id": None, "name": "", "args": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["args"] += fn.get("arguments") or ""
                        finish = choice.get("finish_reason") or finish
        except httpx.ConnectError as e:
            raise LLMError(f"{self.name}: cannot connect to {self._client.base_url} - is the server running?",
                           retryable=True) from e
        except httpx.TimeoutException as e:
            raise LLMError(f"{self.name}: request timed out", retryable=True) from e
        if calls:
            yield LLMEvent("tool_calls", tool_calls=[_assemble(calls[i]) for i in sorted(calls)])
        yield LLMEvent("done", usage=usage, finish_reason=finish)

    @staticmethod
    def _parse_sse_line(line: str) -> Any:
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            return None
        payload = line[5:].strip()
        if payload == "[DONE]":
            return "DONE"
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.debug("ignoring malformed SSE line", extra={"chars": len(payload)})
            return None
        if not isinstance(data, dict):
            return None
        if "error" in data:
            err = data["error"]
            raise LLMError(f"provider error: {err.get('message', err) if isinstance(err, dict) else err}")
        return data

    async def capabilities(self) -> dict[str, Any]:
        # An OpenAI-shaped API advertises nothing reliable; everything stays unknown until configured/tested.
        return {"thinking": None, "think_levels": None, "tools": None, "vision": None, "json_schema": None,
                "context_length": None, "cache_counters": None}

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


def _assemble(slot: dict[str, Any]) -> ToolCall:
    try:
        args = json.loads(slot["args"]) if slot["args"].strip() else {}
    except json.JSONDecodeError:
        args = {"_invalid": slot["args"][:200]}
    call = ToolCall(slot["name"], args if isinstance(args, dict) else {"_invalid": str(args)[:200]})
    if slot["id"]:
        call.id = slot["id"]
    return call
