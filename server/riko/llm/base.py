"""Provider-agnostic LLM interface. All providers stream text deltas."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(Exception):
    """A provider failure with a user-presentable message (never includes secrets)."""

    def __init__(self, message: str, *, status: Optional[int] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def stream_chat(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Cancelling the consumer must close the HTTP stream."""

    async def complete(self, messages: list[Message], **kw: object) -> str:
        parts: list[str] = []
        async for delta in self.stream_chat(messages, **kw):  # type: ignore[arg-type]
            parts.append(delta)
        return "".join(parts)

    @abstractmethod
    async def list_models(self) -> list[str]: ...

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
        401: "authentication failed - check the API key environment variable",
        403: "access denied by the provider",
        404: "endpoint or model not found - check base_url and model name",
        429: "rate limited by the provider",
    }
    snippet = body.strip().replace("\n", " ")[:300]
    return f"{provider} HTTP {status}: {hints.get(status, 'request failed')} ({snippet})"
