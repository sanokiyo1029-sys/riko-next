"""Ollama native ``/api/chat`` (NDJSON streaming).

The native API is used instead of Ollama's OpenAI shim because it exposes ``keep_alive``, ``num_ctx``,
``think`` (reasoning effort), structured ``format`` and cache counters. Wire facts verified against
Ollama 0.35.1 + gpt-oss:20b (fixtures in tests/fixtures/ollama_gpt_oss/):

* reasoning streams in ``message.thinking``, answer in ``message.content``;
* ``think: false`` is ignored by gpt-oss, which only accepts low|medium|high (``/api/show`` lists them);
* tool calls arrive whole: ``message.tool_calls[].function.{name, arguments: object}``; results go back
  as ``{"role": "tool", "tool_name": ..., "content": ...}``;
* the final chunk carries prompt_eval_count, prompt_eval_cached_count, eval_count, durations, done_reason;
* every request must use the same num_ctx, or the model is reloaded.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from ..core.logging import get_logger
from .base import LLMError, LLMEvent, LLMProvider, Message, ToolCall, http_error_message, split_data_url

log = get_logger("llm")


def to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({"role": "tool", "tool_name": m.get("name", ""), "content": str(m.get("content", ""))})
        elif role == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                       for c in m["tool_calls"]]})
        else:
            msg: dict[str, Any] = {"role": role, "content": m.get("content") or ""}
            if m.get("images"):
                msg["images"] = [split_data_url(u)[1] for u in m["images"]]
            out.append(msg)
    return out


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        keep_alive: str = "30m",
        num_ctx: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
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
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.extra_options = dict(extra_options or {})
        self._caps: dict[str, dict[str, Any]] = {}  # per model name, probed once
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=5.0), transport=transport
        )

    # ------------------------------------------------------------------ capabilities
    async def capabilities(self) -> dict[str, Any]:
        if self.model in self._caps:
            return self._caps[self.model]
        caps: dict[str, Any] = {"thinking": None, "think_levels": None, "tools": None, "vision": None,
                                "json_schema": True, "context_length": None, "cache_counters": None,
                                "digest": None}
        try:
            r = await self._client.post("/api/show", json={"model": self.model}, timeout=10.0)
            if r.status_code == 200:
                d = r.json()
                listed = d.get("capabilities")
                if isinstance(listed, list):
                    caps.update(thinking="thinking" in listed, tools="tools" in listed, vision="vision" in listed)
                think = d.get("thinking")
                if isinstance(think, dict) and isinstance(think.get("values"), list):
                    caps["think_levels"] = [str(v) for v in think["values"]]
                for k, v in (d.get("model_info") or {}).items():
                    if k.endswith(".context_length"):
                        caps["context_length"] = v
                self._caps[self.model] = caps  # only cache a real answer
        except (httpx.HTTPError, ValueError) as e:
            log.debug("ollama /api/show failed", extra={"error": e.__class__.__name__})
        return caps

    async def _think_param(self) -> Any:
        """Translate the configured effort into what this model accepts (never sent blindly)."""
        if not self.reasoning_effort:
            return None
        caps = await self.capabilities()
        levels = caps.get("think_levels")
        if levels:  # effort-level models; gpt-oss ignores false, so "off" sends nothing
            return self.reasoning_effort if self.reasoning_effort in levels else None
        if caps.get("thinking"):  # boolean-only thinking models (e.g. qwen3): no effort levels
            return self.reasoning_effort != "off"
        return None

    # ------------------------------------------------------------------ chat
    async def stream_events(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[LLMEvent]:
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "num_predict": max_tokens or self.max_tokens,
            **self.extra_options,
        }
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if self.seed is not None:
            options["seed"] = self.seed
        body: dict[str, Any] = {"model": self.model, "messages": to_ollama_messages(messages), "stream": True,
                                "keep_alive": self.keep_alive, "options": options}
        think = await self._think_param()
        if think is not None:
            body["think"] = think
        if tools:
            body["tools"] = tools
        if json_schema:
            body["format"] = json_schema
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    hint = f" Run: ollama pull {self.model}" if resp.status_code == 404 else ""
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
                        raise LLMError(f"ollama error: {str(data['error'])[:300]}")
                    msg = data.get("message") or {}
                    # A chunk may carry thinking and content together: handle each independently.
                    if msg.get("thinking"):
                        yield LLMEvent("thinking")
                    if msg.get("content"):
                        yield LLMEvent("content", msg["content"])
                    if msg.get("tool_calls"):
                        yield LLMEvent("tool_calls", tool_calls=[_parse_call(c) for c in msg["tool_calls"]])
                    if data.get("done"):
                        yield LLMEvent("done", usage=_usage(data), finish_reason=data.get("done_reason"))
                        break
        except httpx.ConnectError as e:
            raise LLMError(f"ollama: cannot connect to {self._client.base_url} - start it with 'ollama serve'",
                           retryable=True) from e
        except httpx.TimeoutException as e:
            raise LLMError("ollama: request timed out (model may still be loading)", retryable=True) from e

    async def warmup(self) -> dict[str, Any]:
        """Documented preload: an empty chat with keep_alive and the *same* num_ctx as real requests."""
        body: dict[str, Any] = {"model": self.model, "messages": [], "keep_alive": self.keep_alive}
        if self.num_ctx:
            body["options"] = {"num_ctx": self.num_ctx}
        try:
            r = await self._client.post("/api/chat", json=body, timeout=300.0)
        except httpx.HTTPError as e:
            return {"loaded": False, "error": e.__class__.__name__}
        if r.status_code != 200:
            return {"loaded": False, "error": f"HTTP {r.status_code}"}
        d = r.json()
        return {"loaded": True, "done_reason": d.get("done_reason"),
                "load_ms": round((d.get("load_duration") or 0) / 1e6) or None}

    async def running(self) -> list[dict[str, Any]]:
        """Model residency from /api/ps (model-related fields only, not per-device free memory)."""
        try:
            r = await self._client.get("/api/ps", timeout=5.0)
            return r.json().get("models", []) if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            return []

    async def version(self) -> Optional[str]:
        try:
            r = await self._client.get("/api/version", timeout=5.0)
            return r.json().get("version") if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

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


def _parse_call(c: dict[str, Any]) -> ToolCall:
    fn = c.get("function") or {}
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_invalid": args[:200]}
    call = ToolCall(str(fn.get("name", "")), args if isinstance(args, dict) else {"_invalid": str(args)[:200]})
    if c.get("id"):
        call.id = str(c["id"])
    return call


def _usage(d: dict[str, Any]) -> dict[str, Any]:
    ms = lambda k: round(d[k] / 1e6) if isinstance(d.get(k), (int, float)) else None  # noqa: E731
    return {"prompt_tokens": d.get("prompt_eval_count"), "cached_tokens": d.get("prompt_eval_cached_count"),
            "completion_tokens": d.get("eval_count"), "load_ms": ms("load_duration"),
            "prompt_ms": ms("prompt_eval_duration"), "eval_ms": ms("eval_duration"), "total_ms": ms("total_duration")}
